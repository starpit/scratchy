# SPDX-License-Identifier: Apache-2.0
"""Scratchy's Triton mixture-of-experts kernel — gemma-4's whole sparse block, router
included, as ONE program.

    out[m] = Σ_j  w[m, j] · down_e( act_e( x_e[m] ) )        e = ids[m, j]

THE COMPUTATION, IN PROGRAM ORDER:

1. ROUTER NORM — `rmsnorm(x_r, router.scale)`, gemma's plain-gain convention (no `+1`),
   over the UN-normed residual (the DSL's `gemma_moe(hidden_states, h2, …)` routes off its
   first argument).
2. ROUTER LOGITS — `xn @ router.proj.T`, the dense `[E, H]` projection, f16.
3. TOP-K — K rounds over the whole `[M, E]` row: the row maximum, the highest expert id
   among the lanes equal to it, then that lane masked out. The selection `topk.ddl`'s
   ripple computes: descending by value, and among equal values the later scan position
   first. Slot 0 the maximum.
4. ROUTE WEIGHTS — the K selected logits × `H^-0.5`, a softmax over the K, then × each
   selected expert's `router.per_expert_scale`, read BY ITS ID.
5. EXPERTS — per routed pair, the gate/up/down slabs selected BY ID out of the stacked
   `[E, ·, ·]` fp8 banks (`W.load([e, 0, 0])`, the `_p_matmul.py:423` slab gather), at
   the card's W8A8 numerics, `gelu_tanh(gate) · up` between.
6. COMBINE — `Σ_j w[m, j] · down_j`, accumulated in f16 (the card's combine legs).

THE NUMERICS ARE THE CARD'S, SPELLED HERE. A W8A8 projection on the card is
`fp8_quantize_descriptors` + `matmulfp8` + two dequant legs
(`ktir_matmul_fp8.rs`):

    s     = min(max(amax(|a|) · 1/448, 1/448²), 448)    one scale per token row
    q     = fp8(min(max(a · (1/s), -448), 448))         the activation's codes
    raw   = q @ W_codes.T                                f16
    out   = (raw · s) · w_scale                          two f16 roundings

and the scale's ROW is the unit the projection's activation actually has:

* gate/up read the TOKEN's input, so the token row `[1, H]` takes one scale, shared by all
  K pairs (quantized once — every pair's two gemms read the same codes);
* down reads each PAIR's own activation, so each `(token, slot)` row `[1, I]` takes its
  own scale — the `[K, I]` activation matrix quantized in one `quantscalepertokenfp8` op,
  one scale per row.

Per pair, and not one scale over all K activations, because a prompt's experts are swept
densely and its down activations are natively per `(token, expert)` rows; the decode
kernel quantizes at the same grain so the same token computes the same numbers on either
path (a prefix cached by one must be reusable by the other). Writing the quantize in the
program is what makes the emulator compute what the card computes; before this kernel the
emulator ran these projections W8A16.

THE SPLICE'S OWN CONTRACT:

* PARAMETERS, IN ORDER: the router input `desc_xr` `[M, H]`, its gain `desc_rg` `[H]`, its
  projection `desc_rw` `[E, H]`, its per-expert scale `desc_pes` `[E]`; the expert input
  `desc_xe` `[M, H]`; the three banks and their per-channel scales `desc_wg`/`desc_sg`
  `[E, I, H]`/`[E, I]`, `desc_wu`/`desc_su` the same, `desc_wd`/`desc_sd` `[E, H, I]`/`[E, H]`;
  the output `desc_o` `[M, H]`; then the SCRATCH — `desc_ids` `[M, K]` the routed ids as
  i32 and `desc_rs` `[M, K]` the route weights, both round-tripped through memory, and
  `desc_act` `[M, K·I]`, part of the signature the splice states, which the program does
  not read.
* CONSTEXPRS: `M`, `H`, `I`, `E`, `K`; `EPS`, `INV_H` (the router norm); `TEMP` (`H^-0.5`);
  `INV_QMAX` (`1/448`) and `SCALE_FLOOR` (`1/448²`), the per-token scale's `mulConst` and
  `clipMin` (`OpFunc::op_consts`).
* GRID: `[1]`.

⛔ WHY THE SCRATCH IS MEMORY. The slab gather reads its index through a POINTER (the
indirect tile's index view is a kernel argument — `convert_ttir_to_ktdp::slab_gather`), so
the routed ids are stored as i32 before any slab is read, and read back as the gather's
index; the route weights ride with them.

THE EXPERTS ARE THREE BATCHED CONTRACTIONS. The banks are read as `[E, rows·cols]`, so the
routed experts' slabs are one gather of the ids (row `m·K + j` is token m's slot-j expert).
All three batch over the M·K pairs (`[M·K, 1, H] × [M·K, H, I]` for gate/up,
`[M·K, 1, I] × [M·K, I, H]` for down), so each reads its slabs through the gather: no copy
of the weights. A pair's gate/up input is its token's row, repeated across the token's K
slots before quantizing (one scale per token, as above).

⛔ WHY THE TOP-K IS ROUND-BASED, NOT THE `topk.ddl` RIPPLE. The ripple reads the logits
one `[M, 1]` column at a time, and a descriptor block must carry at least 16 bytes in its
last dimension (upstream's own check — real Triton refuses the 2-byte column). Rounds over
the whole row select the same experts in the same slot order with no column access.

⛔ EVERY LOOP IS `tl.static_range` — the ladder's law: every
offset but the routed expert id is a trace-time constant, and the id is the ONLY data-
dependent address in the program.
"""

import triton
import triton.language as tl


@triton.jit
def moe_fwd(desc_xr, desc_rg, desc_rw, desc_pes,  #
            desc_xe, desc_wg, desc_sg, desc_wu, desc_su, desc_wd, desc_sd,  #
            desc_o,  #
            desc_ids, desc_rs, desc_act,  #
            M: tl.constexpr, H: tl.constexpr, I: tl.constexpr,  #
            E: tl.constexpr, K: tl.constexpr,  #
            EPS: tl.constexpr, INV_H: tl.constexpr, TEMP: tl.constexpr,  #
            INV_QMAX: tl.constexpr, SCALE_FLOOR: tl.constexpr):
    # ── the router ────────────────────────────────────────────────────────────
    xr_desc = tl.make_tensor_descriptor(desc_xr, shape=[M, H], strides=[H, 1],
                                        block_shape=[M, H])
    rg_desc = tl.make_tensor_descriptor(desc_rg, shape=[H], strides=[1], block_shape=[H])
    rw_desc = tl.make_tensor_descriptor(desc_rw, shape=[E, H], strides=[H, 1],
                                        block_shape=[E, H])
    pes_desc = tl.make_tensor_descriptor(desc_pes, shape=[E], strides=[1], block_shape=[E])
    xr = xr_desc.load([0, 0])
    ms = tl.sum(xr * xr, 1) * INV_H
    r = tl.rsqrt((ms + EPS).to(tl.float32)).to(tl.float16)
    xn = xr * r[:, None] * rg_desc.load([0])[None, :]
    lg = tl.dot(xn, rw_desc.load([0, 0]).T, tl.zeros([M, E], dtype=tl.float16))

    # THE TOP-K, K rounds over the whole `[M, E]` row: each round takes the row maximum
    # and the HIGHEST expert id among the lanes equal to it, then masks that lane out.
    # This is `topk.ddl`'s ripple order exactly — descending by value, and among equal
    # values the later scan position first (the ripple's strict `>` lets the newcomer take
    # the slot). The lane ids ride f16 (exact to 2048; the real router's `E = 128`).
    offs = tl.arange(0, E).to(tl.float16)
    koffs = tl.arange(0, K).to(tl.float16)
    pes = pes_desc.load([0])
    vals = tl.zeros([M, K], dtype=tl.float16)
    idsf = tl.zeros([M, K], dtype=tl.float16)
    psel = tl.zeros([M, K], dtype=tl.float16)
    # Every select is tile-on-tile f16 (the select law): a bare Python float in a
    # `tl.where` becomes an f32 tensor and widens the whole select.
    neg_inf = tl.full([M, E], float("-inf"), tl.float16)
    no_lane = tl.full([M, E], -1.0, tl.float16)
    zero_e = tl.zeros([M, E], dtype=tl.float16)
    one_k = tl.full([K], 1.0, tl.float16)
    zero_k = tl.zeros([K], dtype=tl.float16)
    for j in tl.static_range(0, K):
        # `tl.max` reduces an f16 tile in f32 (upstream's own promotion); each reduce is
        # narrowed straight back, the widening island LegalizeTypes collapses.
        mx = tl.max(lg, 1).to(tl.float16)
        lanes = tl.where(lg == mx[:, None], offs[None, :] + zero_e, no_lane)
        idx = tl.max(lanes, 1).to(tl.float16)
        sel = offs[None, :] == idx[:, None]
        # The selected expert's per-expert scale, by its id (a one-hot over the E lanes).
        pj = tl.sum(tl.where(sel, pes[None, :] + zero_e, zero_e), 1)
        # Slot j of the three `[M, K]` results: each slot is written exactly once, so
        # accumulating a one-hot column into zeros places it exactly. The one-hot is a
        # select of f16 tiles, not a bool cast (`.to` of an i1 is `arith.uitofp`).
        oh = tl.where(koffs == j, one_k, zero_k)[None, :]
        vals = vals + mx[:, None] * oh
        idsf = idsf + idx[:, None] * oh
        psel = psel + pj[:, None] * oh
        lg = tl.where(sel, neg_inf, lg)

    # The route weights: the K selected logits at the `H^-0.5` temperature, a softmax over
    # the K (`routesoftmax.py`'s chain), × each selected expert's scale. The ids land as i32:
    # the slab gather reads them back through a pointer (its index view).
    ids_desc = tl.make_tensor_descriptor(desc_ids, shape=[M, K], strides=[K, 1],
                                         block_shape=[M, K])
    rs_desc = tl.make_tensor_descriptor(desc_rs, shape=[M, K], strides=[K, 1],
                                        block_shape=[M, K])
    ids_desc.store([0, 0], idsf.to(tl.int32))
    sv = (vals * TEMP).to(tl.float32)
    ex = tl.exp(sv - tl.max(sv, 1)[:, None])
    rs_desc.store([0, 0], tl.fdiv(ex, tl.sum(ex, 1)[:, None]).to(tl.float16) * psel)

    # ── the experts: THREE BATCHED CONTRACTIONS over the M·K pairs ────────────
    # The banks are read as `[E, rows·cols]`: an expert's slab is one row, so the routed
    # experts' slabs are one gather of the ids (pair `p = m·K + j` is token m's slot-j
    # expert), and reshaping a gathered `[M·K, rows·cols]` to `[M·K, rows, cols]` splits only
    # the slab's own axis, so the contraction reads the slabs through the gather, never a
    # copy of the weights. Each pair's gate/up input is its token's row: the token input is
    # repeated across its K slots before it is quantized (an activation copy, K·H elements a
    # token), and a token's K rows quantize to the token's one scale.
    MK: tl.constexpr = M * K
    xe_desc = tl.make_tensor_descriptor(desc_xe, shape=[M, H], strides=[H, 1],
                                        block_shape=[M, H])
    wg_desc = tl.make_tensor_descriptor(desc_wg, shape=[E, I * H], strides=[I * H, 1],
                                        block_shape=[1, I * H])
    wu_desc = tl.make_tensor_descriptor(desc_wu, shape=[E, I * H], strides=[I * H, 1],
                                        block_shape=[1, I * H])
    wd_desc = tl.make_tensor_descriptor(desc_wd, shape=[E, H * I], strides=[H * I, 1],
                                        block_shape=[1, H * I])
    sg_desc = tl.make_tensor_descriptor(desc_sg, shape=[E, I], strides=[I, 1],
                                        block_shape=[1, I])
    su_desc = tl.make_tensor_descriptor(desc_su, shape=[E, I], strides=[I, 1],
                                        block_shape=[1, I])
    sd_desc = tl.make_tensor_descriptor(desc_sd, shape=[E, H], strides=[H, 1],
                                        block_shape=[1, H])
    pairs_desc = tl.make_tensor_descriptor(desc_ids, shape=[MK], strides=[1], block_shape=[MK])
    rw_desc = tl.make_tensor_descriptor(desc_rs, shape=[M, K], strides=[K, 1],
                                        block_shape=[M, K])
    o_desc = tl.make_tensor_descriptor(desc_o, shape=[M, H], strides=[H, 1],
                                       block_shape=[M, H])
    pairs = pairs_desc.load([0])
    # The gate/up activation: each pair's row is its token's input, at its token's scale.
    xe = tl.reshape(xe_desc.load([0, 0])[:, None, :] + tl.zeros([M, K, H], dtype=tl.float16),
                    [MK, H])
    sx = tl.minimum(tl.maximum(tl.max(tl.abs(xe), 1).to(tl.float16) * INV_QMAX,
                               tl.full([MK], SCALE_FLOOR, tl.float16)),
                    tl.full([MK], 448.0, tl.float16))
    ix = tl.fdiv(tl.full([MK], 1.0, tl.float16), sx)
    qx = tl.minimum(tl.maximum(xe * ix[:, None], tl.full([MK, H], -448.0, tl.float16)),
                    tl.full([MK, H], 448.0, tl.float16))
    qx = tl.reshape(qx.to(tl.float8e4nv).to(tl.float16), [MK, 1, H])
    wg = tl.reshape(wg_desc.gather(pairs, 0), [MK, I, H]).to(tl.float16)
    wu = tl.reshape(wu_desc.gather(pairs, 0), [MK, I, H]).to(tl.float16)
    gate = tl.reshape(tl.dot(qx, tl.trans(wg, (0, 2, 1)), out_dtype=tl.float16), [MK, I])
    up = tl.reshape(tl.dot(qx, tl.trans(wu, (0, 2, 1)), out_dtype=tl.float16), [MK, I])
    gate = (gate * sx[:, None]) * sg_desc.gather(pairs, 0)
    up = (up * sx[:, None]) * su_desc.gather(pairs, 0)
    # gelu_tanh(gate) · up. The source states the cubic in f32 (an f16 x³ overflows at
    # |x| ≥ 41); the front end's type legalization lowers the island to f16, where with
    # 0.044715 multiplied first the cube overflows only past |x| ≈ 114, to the right limit.
    g32 = gate.to(tl.float32)
    v = 1.5957691216057308 * (g32 + 0.044715 * g32 * g32 * g32)
    e = tl.exp(-v).to(tl.float16)
    act = tl.fdiv(gate, tl.full([MK, I], 1.0, tl.float16) + e) * up
    # The down activation: each pair's own row, one scale per pair.
    sa = tl.minimum(tl.maximum(tl.max(tl.abs(act), 1).to(tl.float16) * INV_QMAX,
                               tl.full([MK], SCALE_FLOOR, tl.float16)),
                    tl.full([MK], 448.0, tl.float16))
    ia = tl.fdiv(tl.full([MK], 1.0, tl.float16), sa)
    qa = tl.minimum(tl.maximum(act * ia[:, None], tl.full([MK, I], -448.0, tl.float16)),
                    tl.full([MK, I], 448.0, tl.float16))
    qa = tl.reshape(qa.to(tl.float8e4nv).to(tl.float16), [MK, 1, I])
    wd = tl.reshape(wd_desc.gather(pairs, 0), [MK, H, I]).to(tl.float16)
    down = tl.reshape(tl.dot(qa, tl.trans(wd, (0, 2, 1)), out_dtype=tl.float16), [MK, H])
    down = tl.reshape((down * sa[:, None]) * sd_desc.gather(pairs, 0), [M, K, H])
    # The combine: Σ_j w[m, j] · down[m, j], in f16.
    w = rw_desc.load([0, 0])
    o_desc.store([0, 0], tl.sum(down * w[:, :, None], 1))
