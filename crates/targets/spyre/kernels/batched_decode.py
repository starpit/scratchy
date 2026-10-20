# SPDX-License-Identifier: Apache-2.0
"""The Spyre-native batched paged-decode attention kernel.

torch-spyre's `batched_decode_head_major` algorithm (`spyre_inference/v1/
attention/ops/batched_decode_head_major.py`), stated in Triton over the access
forms this ladder lowers: descriptor windows and index-table page gathers.
NO masked pointer loads anywhere.

The contract (the module doc of `attn.py` states the shared laws; this file
follows them):

  * ONE program, grid `[1, 1]` — every sequence in one kernel. All descriptor
    corners are compile-time constants (the chunk loop is `tl.static_range`,
    unrolled), so nothing materializes per grid position.
  * PAGES are gathered WHOLE: one K gather + one V gather per chunk, the index
    vector `[ENTRIES]` loaded from the memory-resident id table (ids are never
    computed in-register). ENTRIES = NUM_SEQS * BLOCKS_PER_CHUNK <= 32 — one
    index stick, one gather leg, one entry per core.
  * Entry order is torch's own: entry `j*B + s` is block slot `j` of sequence
    `s`, so a sequence's J chunk-slots keep SEPARATE running softmax state and
    merge once at the end (`_entry_local_update` / `_merge_entry_local`).
  * The two dots carry their batch dims explicitly: operands are rank-3
    `[batch, M, K]` x `[batch, K, N]` with batch = ENTRIES*NKVH — the KV heads
    and the GQA group ride the dot's batch and M axes, never new descriptors.
  * The scale rides log2(e) as an immediate multiply on the dot's result, so
    the softmax is `exp2` end to end (the flash fixtures' own law).

Staged by the host, per step:

  * `desc_q`    — Q PRE-REPEATED in entry order (torch's `rep_row_ids`
                  `index_select`): rows `(j*B + s)*NKVH*GQA + kvh*GQA + g`.
  * `desc_kt`   — K TRANSPOSED units, row = one WHOLE page for ALL KV heads:
                  `[PAGES_TOTAL, NKVHD*PAGE_SLOTS]`, each page's content
                  row-major `[NKVH, HD, PAGE_SLOTS]` (K^T per head).
  * `desc_v`    — V units, row = one whole page: content row-major
                  `[NKVH, PAGE_SLOTS, HD]`.
  * `desc_ids`  — the page-id table, `[NUM_CHUNKS*ENTRIES]` i32, chunk-major.
                  Pad ids name a WRITTEN page and are masked (never read
                  unwritten bytes).
  * `desc_mask` — additive fp16 mask `[NUM_CHUNKS*ENTRIES*NKVH, PAGE_SLOTS]`,
                  expanded over the folded (entry, kv-head) batch axis: 0 where
                  the slot is valid, -65504 where it is past the sequence's
                  length or a padded page. The finite sentinel is torch's own
                  clamp trick: a slot masked in every chunk still merges with
                  weight exp2(-65504 - m) == 0, never NaN.
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
    """One decode step for every sequence: chunked whole-page gathers, an
    online softmax carried per entry-slot, one merge, one store."""
    # torch's chunking law at this batch: blocks_per_chunk = 32 // num_seqs,
    # so the chunk's entries fill the 32-entry index stick exactly.
    ENTRIES: tl.constexpr = NUM_SEQS * BLOCKS_PER_CHUNK
    BATCH: tl.constexpr = ENTRIES * NKVH
    LOG2E: tl.constexpr = 1.44269504

    # Q, pre-repeated in entry order by the host: [BATCH, GQA, HD] as one
    # window at base.
    q_desc = tl.make_tensor_descriptor(desc_q, shape=[BATCH * GQA, HD],
                                       strides=[HD, 1],
                                       block_shape=[BATCH * GQA, HD])
    o_desc = tl.make_tensor_descriptor(desc_o, shape=[NUM_SEQS * NKVH * GQA, HD],
                                       strides=[HD, 1],
                                       block_shape=[NUM_SEQS * NKVH * GQA, HD])
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
    # The memory-resident page-id table and the additive mask, both chunk-major.
    ids_desc = tl.make_tensor_descriptor(
        desc_ids, shape=[NUM_CHUNKS * ENTRIES], strides=[1],
        block_shape=[ENTRIES])
    mask_desc = tl.make_tensor_descriptor(
        desc_mask, shape=[NUM_CHUNKS * BATCH, PAGE_SLOTS],
        strides=[PAGE_SLOTS, 1], block_shape=[BATCH, PAGE_SLOTS])

    # The running softmax, per entry-slot: torch's `_entry_local_update` carry.
    q = tl.reshape(q_desc.load([0, 0]), [BATCH, GQA, HD])
    m = tl.full([BATCH, GQA], -65504.0, tl.float32)
    l = tl.zeros([BATCH, GQA], tl.float16)
    acc = tl.zeros([BATCH, GQA, HD], tl.float16)

    for c in tl.static_range(NUM_CHUNKS):
        ids = ids_desc.load([c * ENTRIES])                      # [ENTRIES] i32
        # ONE K gather + ONE V gather per chunk, whole pages, <= 32 entries.
        kt = tl.reshape(kt_desc.gather(ids, 0), [BATCH, HD, PAGE_SLOTS])
        v = tl.reshape(v_desc.gather(ids, 0), [BATCH, PAGE_SLOTS, HD])
        # The score dot, heads and GQA as the batch and M axes.
        qk = tl.dot(q, kt, out_dtype=tl.float16)                # [BATCH, GQA, PS]
        qk = qk * (SCALE * LOG2E)
        qk = qk + tl.reshape(mask_desc.load([c * BATCH, 0]), [BATCH, 1, PAGE_SLOTS])
        # The online update: running max, rescaled sum, rescaled accumulator.
        m_new = tl.maximum(m, tl.max(qk, 2))                    # [BATCH, GQA] f32
        rescale = tl.math.exp2(m - m_new).to(tl.float16)        # [BATCH, GQA]
        p = tl.math.exp2(qk - m_new[:, :, None]).to(tl.float16)
        l = l * rescale + tl.sum(p, 2)
        acc = acc * rescale[:, :, None] + tl.dot(p, v, out_dtype=tl.float16)
        m = m_new

    # The slot merge, torch's `_merge_entry_local`: entry order is j*B + s, so
    # the batch axis reshapes [J, B*NKVH] and the J slots of one sequence merge
    # against their common maximum, then normalize once.
    mm = tl.reshape(m, [BLOCKS_PER_CHUNK, NUM_SEQS * NKVH, GQA])
    lm = tl.reshape(l, [BLOCKS_PER_CHUNK, NUM_SEQS * NKVH, GQA])
    am = tl.reshape(acc, [BLOCKS_PER_CHUNK, NUM_SEQS * NKVH, GQA, HD])
    merged = tl.max(mm, 0)                                      # [B*NKVH, GQA]
    w = tl.math.exp2(mm - tl.reshape(merged, [1, NUM_SEQS * NKVH, GQA])).to(tl.float16)
    l = tl.sum(w * lm, 0)
    acc = tl.sum(w[:, :, :, None] * am, 0)                      # [B*NKVH, GQA, HD]
    o = acc / l[:, :, None]
    o_desc.store([0, 0], tl.reshape(o, [NUM_SEQS * NKVH * GQA, HD]))
