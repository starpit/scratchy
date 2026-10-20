# SPDX-License-Identifier: Apache-2.0
"""The Spyre-native batched paged-decode attention kernel.

torch-spyre's `batched_decode_head_major` algorithm (`spyre_inference/v1/
attention/ops/batched_decode_head_major.py`), stated in Triton over the access
forms this ladder lowers: descriptor windows and index-table page gathers.
NO masked pointer loads anywhere.

The contract (the module doc of `attn.py` states the shared laws; this file
follows them):

  * ONE program, grid `[1, 1]` — every sequence in one kernel. All descriptor
    corners are compile-time constants (both loops are `tl.static_range`,
    unrolled), so nothing materializes per grid position.
  * PAGES are gathered WHOLE: one K gather + one V gather per (block-slot,
    chunk), the index vector `[NUM_SEQS]` loaded from the memory-resident id
    table (ids are never computed in-register).
  * The J block-slots of torch's entry order are UNROLLED PER-J STATE: each
    `j` walk keeps its own running softmax (`_entry_local_update`) over its
    own pages, and the cross-slot merge (`_merge_entry_local`) runs ONLINE in
    the same j loop — max, rescale both sides, combine — as elementwise ops
    and trailing `[:, :, None]` sprays only. No leading-axis reduce, no
    leading-axis broadcast: the ladder's flat-frame Col form states every
    shape here.
  * The two dots carry their batch dims explicitly: operands are rank-3
    `[batch, M, K]` x `[batch, K, N]` with batch = NUM_SEQS*NKVH — the KV
    heads and the GQA group ride the dot's batch and M axes, never new
    descriptors.
  * The scale rides log2(e) as an immediate multiply on the dot's result, so
    the softmax is `exp2` end to end (the flash fixtures' own law).
  * The mask is staged at the SCORE TILE'S OWN ROWS — pre-replicated xGQA on
    the host (attn.py's own law) — so the add is a pure flat reshape and an
    elementwise add. No in-kernel broadcast.

Staged by the host, per step:

  * `desc_q`    — Q PRE-REPEATED in block-slot order (torch's `rep_row_ids`
                  `index_select`): rows `(j*NUM_SEQS*NKVH + s*NKVH + kvh)*GQA
                  + g` = `Q[s, kvh*GQA + g, :]`.
  * `desc_kt`   — K TRANSPOSED units, row = one WHOLE page for ALL KV heads:
                  `[PAGES_TOTAL, NKVH*HD*PAGE_SLOTS]`, each page's content
                  row-major `[NKVH, HD, PAGE_SLOTS]` (K^T per head).
  * `desc_v`    — V units, row = one whole page: content row-major
                  `[NKVH, PAGE_SLOTS, HD]`.
  * `desc_ids`  — the page-id table, `[J*NUM_CHUNKS*NUM_SEQS]` i32, ordered
                  (j, chunk, seq): entry `(j*NUM_CHUNKS + c)*NUM_SEQS + s` is
                  page `c*J + j` of sequence `s` when that page index is in
                  range, else a WRITTEN pad page (masked, never an unwritten
                  row).
  * `desc_mask` — additive fp16 mask `[J*NUM_CHUNKS*NUM_SEQS*NKVH*GQA,
                  PAGE_SLOTS]`, pre-replicated xGQA to the score tile's own
                  rows: 0 where the slot is valid, -65504 where it is past
                  the sequence's length or a padded page. The finite sentinel
                  is torch's own clamp trick: a slot masked in every chunk
                  still merges with weight exp2(-65504 - m) == 0, never NaN.
  * `desc_o`    — output, head-major `[NUM_SEQS*NQH, HD]`.
"""


@triton.jit
def batched_decode_hm(desc_q, desc_o,  #
                      desc_kt, desc_v, desc_ids, desc_mask,  #
                      NUM_SEQS: tl.constexpr, NKVH: tl.constexpr,  #
                      GQA: tl.constexpr, HD: tl.constexpr,  #
                      PAGE_SLOTS: tl.constexpr, PAGES: tl.constexpr,  #
                      BLOCKS_PER_CHUNK: tl.constexpr,  #
                      NUM_CHUNKS: tl.constexpr, SCALE: tl.constexpr):
    """One decode step for every sequence: per-block-slot chunked whole-page
    gathers, a per-slot online softmax, an online cross-slot merge, one
    store."""
    # torch's chunking law at this batch: blocks_per_chunk = 32 // num_seqs.
    # Each block-slot's walk covers its own pages: page c*J + j of each
    # sequence, NUM_CHUNKS walks over the rung's page count.
    ROWS: tl.constexpr = NUM_SEQS * NKVH
    LOG2E: tl.constexpr = 1.44269504

    # Q, pre-repeated in block-slot order by the host: one [ROWS*GQA, HD]
    # window per block-slot j.
    q_desc = tl.make_tensor_descriptor(
        desc_q, shape=[BLOCKS_PER_CHUNK * ROWS * GQA, HD],
        strides=[HD, 1], block_shape=[ROWS * GQA, HD])
    o_desc = tl.make_tensor_descriptor(
        desc_o, shape=[ROWS * GQA, HD], strides=[HD, 1],
        block_shape=[ROWS * GQA, HD])
    # The whole-page K^T and V units: ONE row per page, the row the gather's
    # index names. Block [1, C] is the "gather, don't tile" form.
    kt_desc = tl.make_tensor_descriptor(
        desc_kt, shape=[PAGES, NKVH * HD * PAGE_SLOTS],
        strides=[NKVH * HD * PAGE_SLOTS, 1],
        block_shape=[1, NKVH * HD * PAGE_SLOTS])
    v_desc = tl.make_tensor_descriptor(
        desc_v, shape=[PAGES, NKVH * PAGE_SLOTS * HD],
        strides=[NKVH * PAGE_SLOTS * HD, 1],
        block_shape=[1, NKVH * PAGE_SLOTS * HD])
    # The memory-resident page-id table and the pre-replicated additive mask,
    # both ordered (block-slot, chunk, row).
    ids_desc = tl.make_tensor_descriptor(
        desc_ids, shape=[BLOCKS_PER_CHUNK * NUM_CHUNKS * NUM_SEQS], strides=[1],
        block_shape=[NUM_SEQS])
    mask_desc = tl.make_tensor_descriptor(
        desc_mask, shape=[BLOCKS_PER_CHUNK * NUM_CHUNKS * ROWS * GQA,
                          PAGE_SLOTS],
        strides=[PAGE_SLOTS, 1], block_shape=[ROWS * GQA, PAGE_SLOTS])

    # The cross-slot merge state, torch's `_merge_entry_local` run online:
    # per (sequence, kv-head, group) across the J block-slots.
    mm = tl.full([ROWS, GQA], -65504.0, tl.float32)
    ml = tl.zeros([ROWS, GQA], tl.float16)
    macc = tl.zeros([ROWS, GQA, HD], tl.float16)

    for j in tl.static_range(BLOCKS_PER_CHUNK):
        # This block-slot's walk: torch's `_entry_local_update` carry, per
        # (sequence, kv-head, group).
        q = tl.reshape(q_desc.load([j * ROWS * GQA, 0]), [ROWS, GQA, HD])
        m = tl.full([ROWS, GQA], -65504.0, tl.float32)
        l = tl.zeros([ROWS, GQA], tl.float16)
        acc = tl.zeros([ROWS, GQA, HD], tl.float16)
        for c in tl.static_range(NUM_CHUNKS):
            ids = ids_desc.load([(j * NUM_CHUNKS + c) * NUM_SEQS])
            # ONE K gather + ONE V gather per (block-slot, chunk), whole
            # pages, one entry per sequence.
            kt = tl.reshape(kt_desc.gather(ids, 0), [ROWS, HD, PAGE_SLOTS])
            v = tl.reshape(v_desc.gather(ids, 0), [ROWS, PAGE_SLOTS, HD])
            # The score dot, heads and GQA as the batch and M axes.
            qk = tl.dot(q, kt, out_dtype=tl.float16)            # [ROWS,GQA,PS]
            qk = qk * (SCALE * LOG2E)
            # The mask, staged at the score tile's own rows: a pure flat
            # reshape, an elementwise add. No broadcast.
            qk = qk + tl.reshape(
                mask_desc.load([(j * NUM_CHUNKS + c) * ROWS * GQA, 0]),
                [ROWS, GQA, PAGE_SLOTS])
            # The online update: running max, rescaled sum, rescaled
            # accumulator.
            m_new = tl.maximum(m, tl.max(qk, 2))                # [ROWS, GQA]
            rescale = tl.math.exp2(m - m_new).to(tl.float16)    # [ROWS, GQA]
            p = tl.math.exp2(qk - m_new[:, :, None]).to(tl.float16)
            l = l * rescale + tl.sum(p, 2)
            acc = acc * rescale[:, :, None] + tl.dot(p, v, out_dtype=tl.float16)
            m = m_new
        # The online cross-slot merge: rescale BOTH sides to the new joint
        # max and combine — elementwise and trailing sprays only.
        new_m = tl.maximum(mm, m)
        w_old = tl.math.exp2(mm - new_m).to(tl.float16)
        w_new = tl.math.exp2(m - new_m).to(tl.float16)
        ml = ml * w_old + l * w_new
        macc = macc * w_old[:, :, None] + acc * w_new[:, :, None]
        mm = new_m

    o = macc / ml[:, :, None]                                   # [ROWS,GQA,HD]
    o_desc.store([0, 0], tl.reshape(o, [ROWS * GQA, HD]))
