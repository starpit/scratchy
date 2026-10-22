# SPDX-License-Identifier: Apache-2.0
"""The Spyre-native paged fold: the per-page attention body the runtime
re-launches once per resident page.

`batched_prefill_hm` walks every page in ONE launch (`tl.static_range` over
`PAGES`), so its context is a BAKED extent — the #438 cliff: past the widest
chunk nothing serves. This file is the other shape: ONE page per launch, the
online-softmax carry (max, sum, accumulator) in DEVICE MEMORY instead of
registers, so context is a RUNTIME page count. Three kernels, one contract:

  * `paged_fold_seed_hm` — writes the carry's seed values (max = the finite
    -65504 sentinel, sum = 0, accumulator = 0). Runs ONCE per step, BEFORE
    the first pass. A seeding PASS would reset the running state
    mid-accumulation, so the seed is a separate program, not a flag.
  * `paged_fold_hm` — ONE page of the walk: loads the carry, gathers this
    launch's page through the pool's index unit, folds the page into the
    carry, stores the carry. NO finalize, NO output: the launch is pure
    accumulation, and repeating it IS the loop.
  * `paged_fold_finalize_hm` — `out = acc / l`, ONCE, after the last pass.

The runtime contract this family states (and `fold_plan` enforces — the
launch-side arithmetic is single-sourced there):

  * A GATHERED FOLD TAKES NO KV SHIFT. The index entries are GLOBAL
    stick-block numbers (`addr = idx * skip_addr + base`), so the pool's
    segment base never moves; the page a pass reads is named by the entries
    the host staged for it.
  * THE INDEX TABLES AND THE MASK RIDE THE ACTIVATION SEGMENT, PAGE-MAJOR.
    Each pass shifts that segment's base by one page's worth of bytes —
    512 at this geometry (`W4 * KTB * 4` for a table, `PAGE_SLOTS * 2` for
    the mask row; the same number, which is why one delta moves all three
    tensors correctly). The descriptors below state the WHOLE region and
    load at THIS page's window, so the shifted base lands inside the
    region for every pass.
  * THE CARRY RIDES THE INTERMEDIATE SEGMENT, UNSHIFTED. A request's pages
    all fold into the SAME rows — that is what makes the online softmax a
    running state — so nothing about the carry may move per pass.

The per-trip laws are `batched_prefill_hm`'s own (one full id stick per
page-window trip, the V block contiguous only at HD 64, the exp2 softmax),
restated at ONE page per launch. `MQ == 1` is this family's own law: the
mask's per-page block and a table's per-page block step by the same bytes
only at one query row, and the fold's mask row is one validity row per
page shared by the whole (single) request.

⛔ THE CARRY'S MEMORY IMAGE IS f16 — the SAME grading the one-shot kernel's
LEGALIZED program carries, so the fold is numerically IDENTICAL to the
body it replaces, page for page. `batched_prefill_hm` SPELLS its max/sum
at f32 in Triton, but `LegalizeTypes` collapses that island to f16 before
the door ever sees it (the widening casts around the f16 score tile are
removed and the chain retypes), so an f16 carry here is parity, not a
degradation. An f32-GRADED carry in device memory is a different campaign
— it needs a transfer-boundary anchor in `LegalizeTypes` (a genuine-f32
value persisted between f32 descriptor views must not collapse) plus the
door's fp32 allowlist extended past the merge chain — and is NOT this
family's ask. The -65504 sentinel is f16-exact (the format's own max),
so the seed loses nothing by the narrowing.
"""


@triton.jit
def paged_fold_seed_hm(desc_m, desc_l, desc_acc,  #
                       MQ: tl.constexpr, NKVH: tl.constexpr,  #
                       GQA: tl.constexpr, HD: tl.constexpr):
    """Seed the fold's carry: max at the finite sentinel, sum and
    accumulator at zero. Once per step, before the first pass. The carry's
    memory image is f16 (the module doc's law) and the sentinel is f16-exact."""
    BATCH: tl.constexpr = NKVH * GQA
    # The finite sentinel, not -inf: a masked-everywhere row still
    # normalizes with weight 0, never NaN (torch's own clamp trick).
    m_desc = tl.make_tensor_descriptor(
        desc_m, shape=[BATCH * MQ], strides=[1],
        block_shape=[BATCH * MQ])
    l_desc = tl.make_tensor_descriptor(
        desc_l, shape=[BATCH * MQ], strides=[1],
        block_shape=[BATCH * MQ])
    acc_desc = tl.make_tensor_descriptor(
        desc_acc, shape=[BATCH * MQ, HD], strides=[HD, 1],
        block_shape=[BATCH * MQ, HD])
    m_desc.store([0], tl.full([BATCH * MQ], -65504.0, tl.float16))
    l_desc.store([0], tl.zeros([BATCH * MQ], tl.float16))
    acc_desc.store([0, 0], tl.zeros([BATCH * MQ, HD], tl.float16))


@triton.jit
def paged_fold_hm(desc_q, desc_m, desc_l, desc_acc,  #
                  desc_kt, desc_v, desc_ids_kt, desc_ids_v, desc_mask,  #
                  MQ: tl.constexpr, NKVH: tl.constexpr,  #
                  GQA: tl.constexpr, HD: tl.constexpr,  #
                  PAGE_SLOTS: tl.constexpr, REGION_PAGES: tl.constexpr,  #
                  SCALE: tl.constexpr):
    """ONE page of the walk, folded into the carried online-softmax state.
    This launch's page is the one the host's per-pass segment shift lands
    on; the loads below read that page's own window."""
    BATCH: tl.constexpr = NKVH * GQA
    W4: tl.constexpr = PAGE_SLOTS // 64
    LOG2E: tl.constexpr = 1.44269504
    ID_STICK: tl.constexpr = 32
    tl.static_assert(
        NKVH * GQA == ID_STICK,
        "paged_fold_hm: one full id stick per page-window trip requires NKVH * GQA == 32; other geometries need the PAGES_PER_TRIP generalization (the pending worklist item)")
    tl.static_assert(HD == 64, "paged_fold_hm: the V plane's [64, HD] block is one contiguous stick block only at HD 64; a wider head needs the per-(window, slab) V gather this kernel does not state yet")
    tl.static_assert(PAGE_SLOTS % 64 == 0, "paged_fold_hm: a page must be a whole number of 64-slot windows")
    tl.static_assert(MQ == 1, "paged_fold_hm: the fold's mask row is one validity row per page shared by the whole request — the per-page block strides of the mask and the index tables agree only at one query row")
    # ⛔ THE REGION, NOT THE PAGE: every table below states the WHOLE
    # page-major region the host stages, and every load reads THIS launch's
    # page-window at its own offsets — the runtime's per-pass segment shift
    # is what moves the base from page to page. A view stated at one page
    # would place a 512 B buffer the shift walks right off.
    tl.static_assert(REGION_PAGES >= 1, "paged_fold_hm: the table region holds at least one page")

    # The chunk's q, in torch's [kvh, g, qlen, hd] layout — reloaded every
    # pass; the carry is what persists, not the query.
    q_desc = tl.make_tensor_descriptor(
        desc_q, shape=[BATCH * MQ, HD], strides=[HD, 1],
        block_shape=[BATCH * MQ, HD])
    # The carried online-softmax state, per (kv head, group, query row):
    # torch's carry, at the f16 memory image the module doc's law states
    # (the one-shot's own legalized grading — parity, not degradation).
    m_desc = tl.make_tensor_descriptor(
        desc_m, shape=[BATCH * MQ], strides=[1],
        block_shape=[BATCH * MQ])
    l_desc = tl.make_tensor_descriptor(
        desc_l, shape=[BATCH * MQ], strides=[1],
        block_shape=[BATCH * MQ])
    acc_desc = tl.make_tensor_descriptor(
        desc_acc, shape=[BATCH * MQ, HD], strides=[HD, 1],
        block_shape=[BATCH * MQ, HD])
    # The pool planes at BLOCK granularity over the WHOLE region: the
    # entries are global stick-block numbers, so a pass may name any page
    # the pool holds.
    kt_blk_desc = tl.make_tensor_descriptor(
        desc_kt, shape=[REGION_PAGES * NKVH * W4, HD * 64],
        strides=[HD * 64, 1], block_shape=[1, HD * 64])
    v_blk_desc = tl.make_tensor_descriptor(
        desc_v, shape=[REGION_PAGES * NKVH * W4, 64 * HD],
        strides=[64 * HD, 1], block_shape=[1, 64 * HD])
    # The block-entry tables, page-major over the whole region: this
    # launch's page is the base's own, so the per-window loads read
    # [w * KTB] — never a page term, which the shift already added.
    KTB: tl.constexpr = NKVH * GQA
    kt_ids_desc = tl.make_tensor_descriptor(
        desc_ids_kt, shape=[REGION_PAGES * W4 * KTB], strides=[1],
        block_shape=[KTB])
    v_ids_desc = tl.make_tensor_descriptor(
        desc_ids_v, shape=[REGION_PAGES * W4 * KTB], strides=[1],
        block_shape=[KTB])
    # The additive validity mask, window-major within each page, one row
    # per page over the whole region — the per-pass shift walks it.
    mask_desc = tl.make_tensor_descriptor(
        desc_mask, shape=[REGION_PAGES * W4 * MQ, 64],
        strides=[64, 1], block_shape=[MQ, 64])

    q = tl.reshape(q_desc.load([0, 0]), [BATCH, MQ, HD])
    m = tl.reshape(m_desc.load([0]), [BATCH, MQ])
    l = tl.reshape(l_desc.load([0]), [BATCH, MQ])
    acc = tl.reshape(acc_desc.load([0, 0]), [BATCH, MQ, HD])

    for w in tl.static_range(W4):
        # The block gathers: one stick block per index entry, both planes,
        # this window's own g-replicated id stick, the ids loaded from the
        # memory-resident tables (never computed in-register).
        kt_ids = kt_ids_desc.load([w * KTB])             # [KTB] i32
        v_ids = v_ids_desc.load([w * KTB])               # [KTB] i32
        kt_blk = tl.reshape(kt_blk_desc.gather(kt_ids, 0),
                            [BATCH, HD, 64])
        v_blk = tl.reshape(v_blk_desc.gather(v_ids, 0),
                           [BATCH, 64, HD])
        qk = tl.dot(q, kt_blk, out_dtype=tl.float16)     # [B, MQ, 64]
        qk = qk * (SCALE * LOG2E)
        # The per-page validity row, added at the score tile as a leading
        # stride-0 broadcast.
        qk = qk + tl.reshape(mask_desc.load([w * MQ, 0]),
                             [1, MQ, 64])
        # The online update, folded per trip onto the CARRIED state.
        m_new = tl.maximum(m, tl.max(qk, 2))             # [B, MQ]
        rescale = tl.math.exp2(m - m_new).to(tl.float16)
        p = tl.math.exp2(qk - m_new[:, :, None]).to(tl.float16)
        l = l * rescale + tl.sum(p, 2)
        pv = tl.dot(p, v_blk, out_dtype=tl.float16)      # [B, MQ, HD]
        acc = acc * rescale[:, :, None] + pv
        m = m_new

    m_desc.store([0], tl.reshape(m, [BATCH * MQ]))
    l_desc.store([0], tl.reshape(l, [BATCH * MQ]))
    acc_desc.store([0, 0], tl.reshape(acc, [BATCH * MQ, HD]))


@triton.jit
def paged_fold_finalize_hm(desc_o, desc_l, desc_acc,  #
                           MQ: tl.constexpr, NKVH: tl.constexpr,  #
                           GQA: tl.constexpr, HD: tl.constexpr):
    """The one normalize: `out = acc / l`, ONCE, after the last pass, at the
    carry's own f16 grading. The max is spent by the last pass — nothing
    reads it here."""
    BATCH: tl.constexpr = NKVH * GQA
    o_desc = tl.make_tensor_descriptor(
        desc_o, shape=[BATCH * MQ, HD], strides=[HD, 1],
        block_shape=[BATCH * MQ, HD])
    l_desc = tl.make_tensor_descriptor(
        desc_l, shape=[BATCH * MQ], strides=[1],
        block_shape=[BATCH * MQ])
    acc_desc = tl.make_tensor_descriptor(
        desc_acc, shape=[BATCH * MQ, HD], strides=[HD, 1],
        block_shape=[BATCH * MQ, HD])
    l = tl.reshape(l_desc.load([0]), [BATCH, MQ])        # [B, MQ]
    acc = tl.reshape(acc_desc.load([0, 0]), [BATCH, MQ, HD])
    o = (acc / l[:, :, None]).to(tl.float16)             # [B, MQ, HD]
    o_desc.store([0, 0], tl.reshape(o, [BATCH * MQ, HD]))
