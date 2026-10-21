# SPDX-License-Identifier: Apache-2.0
"""The Spyre-native batched prefill attention kernel.

torch-spyre's chunked-prefill attention (`spyre_inference/v1/attention/ops/
page_attn_head_major_prefill.py`), stated in Triton over the access forms this
ladder lowers: descriptor windows and index-table page gathers. NO masked
pointer loads anywhere.

torch's algorithm, cited:
  * q gathered by row index and laid out `[kvh, g, qlen, hd]` —
    `page_attn_head_major_prefill.py:59-64`;
  * one page per walk trip, `matmul(q, k_page^T) * scale` — `:76-80`;
  * the mask added at the score tile — `:85`;
  * the online softmax carry (max, rescale, sum, output) — `:97-105`;
  * one normalize, then the head-major store — `:123-129`.

The contract (the module doc of `attn.py` states the shared laws; this file
follows them):

  * ONE program, grid `[1, 1]` — the whole chunk in one kernel. All
    descriptor corners are compile-time constants (the page walk is
    `tl.static_range`, unrolled), so nothing materializes per grid position.
  * The dots carry their batch dims explicitly: operands are rank-3
    `[batch, M, K]` x `[batch, K, N]` with batch = NKVH*GQA (the KV heads AND
    the GQA group, torch's `[kvh, g, qlen, hd]` q layout with `(kvh, g)`
    flattened), M = the chunk's query rows, K = HD, N = PAGE_SLOTS.
  * PAGES are gathered WHOLE, one page per trip, as ONE full 32-entry stick:
    the K/V unit rows are (page, kv head, group) triples — a page's 32 rows,
    its whole g-replicated content — named by one 32-entry index stick, every
    entry consumed, stick-aligned by construction. This is the batched-decode
    kernel's own gather shape, and it is forced twice over: Triton refuses a
    gather with fewer than 8 indices, and the door's work division spends its
    cores on the entry axis (`mb`) alone — a gather with fewer entries than
    cores splits `out`, which dxp's `skip_addr` cannot address.
  * K and V ride ONE ids table — the two-pair, one-descriptor-each gather.
  * q is loaded ONCE as one window and reused by every page's dot — the
    one-program form pays the page transfer over every query row, which is
    torch's own stated trade (`page_attn_head_major_prefill.py:17-20`).
  * The scale rides log2(e) as an immediate multiply on the dot's result, so
    the softmax is `exp2` end to end (the flash fixtures' own law).
  * The causal mask is host-staged at a compile-time-constant shape
    (`[PAGES*MQ, PAGE_SLOTS]`, torch's `mask_stack` law at `:51`) — one row
    per (page, query row), UNAMPLIFIED over the batch — and added at the
    score tile as a leading stride-0 broadcast — the same operand mode the
    J-slot merge rides.

Staged by the host, per chunk:

  * `desc_q`    — the chunk's q in torch's own layout: row
                  `(kvh*GQA + g)*MQ + t` = `Q[q_start + t, kvh*GQA + g, :]`,
                  padded rows repeated from row 0 and masked.
  * `desc_kt`   — K TRANSPOSED units, one row per (page, kv head, group):
                  `[PAGES*NKVH*GQA, HD*PAGE_SLOTS]`, row
                  `p*NKVH*GQA + kvh*GQA + g` = `K[p*PAGE_SLOTS + slot, kvh, :]`
                  laid out `[d, slot]` (K^T per head, the group axis
                  pre-replicated).
  * `desc_v`    — V units, one row per (page, kv head, group):
                  `[PAGES*NKVH*GQA, PAGE_SLOTS*HD]`, row
                  `p*NKVH*GQA + kvh*GQA + g` =
                  `V[p*PAGE_SLOTS + slot, kvh, :]` laid out `[slot, d]`.
  * `desc_ids`  — the page-id table, `[PAGES*32]` i32: page s's stick is its
                  own 32 row-ids `s*NKVH*GQA + [0..32)`, stick s of the table.
  * `desc_mask` — additive fp16 mask `[PAGES*MQ, PAGE_SLOTS]`: row
                  `s*MQ + t`, `0.0` where the slot is visible to that query
                  row (within the sequence and causally), `-65504.0` where it
                  is not, or where the query row is padding. The finite
                  sentinel is torch's own clamp trick: a masked-everywhere
                  row still normalizes with weight 0, never NaN.
  * `desc_o`    — output, the chunk's attention at the same row law as q.
"""


@triton.jit
def batched_prefill_hm(desc_q, desc_o,  #
                       desc_kt, desc_v, desc_ids, desc_mask,  #
                       MQ: tl.constexpr, NKVH: tl.constexpr,  #
                       GQA: tl.constexpr, HD: tl.constexpr,  #
                       PAGE_SLOTS: tl.constexpr, PAGES: tl.constexpr,  #
                       SCALE: tl.constexpr):
    """One prefill chunk for the whole sequence: a per-page walk over the
    cached pages, an online softmax carried across pages, one store."""
    # torch's own batch: the KV heads AND the GQA group ride the dot's batch
    # axis; M is the chunk's query rows.
    BATCH: tl.constexpr = NKVH * GQA
    LOG2E: tl.constexpr = 1.44269504
    # The id stick's width is the door's own (one SenUint32 stick = 32
    # entries), so one page's rows must fill it exactly. A model class with
    # NQH != 32 refuses HERE, at compile time, instead of silently reading
    # into the next page's stick; the pending PAGES_PER_TRIP generalization
    # lifts this.
    ID_STICK: tl.constexpr = 32
    tl.static_assert(
        BATCH == ID_STICK,
        "batched_prefill_hm: one full id stick per page requires NQH == NKVH*GQA == 32; NQH != 32 needs PAGES_PER_TRIP (the pending generalization)")

    # The chunk's q, in torch's [kvh, g, qlen, hd] layout, ONE window at base
    # reused by every page's dot.
    q_desc = tl.make_tensor_descriptor(
        desc_q, shape=[BATCH * MQ, HD], strides=[HD, 1],
        block_shape=[BATCH * MQ, HD])
    o_desc = tl.make_tensor_descriptor(
        desc_o, shape=[BATCH * MQ, HD], strides=[HD, 1],
        block_shape=[BATCH * MQ, HD])
    # The K^T and V units, one row per (page, kv head, group) — a page's 32
    # rows gathered as ONE full stick, every entry consumed. Block [1, C] is
    # the "gather, don't tile" form.
    kt_desc = tl.make_tensor_descriptor(
        desc_kt, shape=[PAGES * BATCH, HD * PAGE_SLOTS],
        strides=[HD * PAGE_SLOTS, 1],
        block_shape=[1, HD * PAGE_SLOTS])
    v_desc = tl.make_tensor_descriptor(
        desc_v, shape=[PAGES * BATCH, PAGE_SLOTS * HD],
        strides=[PAGE_SLOTS * HD, 1],
        block_shape=[1, PAGE_SLOTS * HD])
    # The page-id table: page s's 32 row-ids at stick s, one whole stick per
    # trip (ID_STICK — asserted equal to BATCH above). The mask, page-major,
    # one row per (page, query row).
    ids_desc = tl.make_tensor_descriptor(
        desc_ids, shape=[PAGES * ID_STICK], strides=[1], block_shape=[BATCH])
    mask_desc = tl.make_tensor_descriptor(
        desc_mask, shape=[PAGES * MQ, PAGE_SLOTS],
        strides=[PAGE_SLOTS, 1], block_shape=[MQ, PAGE_SLOTS])

    # The running softmax across pages, torch's carry: per (kv head, group,
    # query row).
    q = tl.reshape(q_desc.load([0, 0]), [BATCH, MQ, HD])
    m = tl.full([BATCH, MQ], -65504.0, tl.float32)
    l = tl.zeros([BATCH, MQ], tl.float16)
    acc = tl.zeros([BATCH, MQ, HD], tl.float16)

    for s in tl.static_range(PAGES):
        ids = ids_desc.load([s * ID_STICK])                    # [BATCH] i32
        # ONE K gather + ONE V gather per page, whole pages, one stick.
        kt = tl.reshape(kt_desc.gather(ids, 0), [BATCH, HD, PAGE_SLOTS])
        v = tl.reshape(v_desc.gather(ids, 0), [BATCH, PAGE_SLOTS, HD])
        # The score dot: batch = (kv-head, group), M = query rows.
        qk = tl.dot(q, kt, out_dtype=tl.float16)               # [BATCH,MQ,PS]
        qk = qk * (SCALE * LOG2E)
        # The causal mask, host-staged at a constant shape, added at the
        # score tile as a leading stride-0 broadcast.
        qk = qk + tl.reshape(mask_desc.load([s * MQ, 0]),
                             [1, MQ, PAGE_SLOTS])
        # The online update: running max, rescaled sum, rescaled accumulator.
        m_new = tl.maximum(m, tl.max(qk, 2))                   # [BATCH, MQ]
        rescale = tl.math.exp2(m - m_new).to(tl.float16)       # [BATCH, MQ]
        p = tl.math.exp2(qk - m_new[:, :, None]).to(tl.float16)
        l = l * rescale + tl.sum(p, 2)
        acc = acc * rescale[:, :, None] + tl.dot(p, v, out_dtype=tl.float16)
        m = m_new

    o = acc / l[:, :, None]                                    # [BATCH,MQ,HD]
    o_desc.store([0, 0], tl.reshape(o, [BATCH * MQ, HD]))
