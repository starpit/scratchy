# SPDX-License-Identifier: Apache-2.0
"""Scratchy's Triton mixture-of-experts kernel for a PROMPT — gemma-4's whole sparse block
over T tokens at once, router included, every expert swept densely.

    out[t] = Σ_e  w[t, e] · down_e( act_e( x_e[t] ) )        w[t, e] = 0 off the top-K

THE PREFILL TWIN OF `moe.py`. The decode kernel (`moe.py`) reads one token's K routed
experts by id — a slab gather per pair, the address depending on data. A prompt routes T
tokens to up to T·K distinct experts, so this kernel does what IBM spyre-inference's
`_moe_persistent` does: it builds a DENSE route `[T, E]` (each token's K route weights at
their expert lanes, zero elsewhere) and runs ALL E experts over the whole `[T, H]` token
tile, scaled per token by its dense weight. No address depends on data: the banks are read
whole.

THE COMPUTATION, IN PROGRAM ORDER:

1. ROUTER — `moe.py`'s router verbatim over T rows: rmsnorm (plain gain), the f16 logits
   `[T, E]`, the round-form top-K (descending, ties to the higher id), a softmax over the K
   at `H^-0.5`, × each selected expert's `per_expert_scale`.
2. DENSE ROUTE — the K route weights scattered to their expert lanes: round j adds its weight
   to lane `ids[:, j]` by a one-hot select. Lanes off the top-K stay exactly 0.
3. EXPERTS — three contractions over all E experts: gate/up W8A8 each ONE contraction of the
   token codes against every expert's rows stacked (`[T, H] × [H, E·I]`), gelu-tanh(gate)·up
   `[T, E, I]`, down W8A8 batched over the experts, each (token, expert) row at its own scale
   (`[E, T, I] × [E, I, H]`), × the dense weight, summed over the experts in f16.

THE NUMERICS ARE `moe.py`'s, AT THE SAME GRAIN, so a token computes the same numbers on
either path (a prefix cached by one is reusable by the other):

* gate/up: one fp8 scale per TOKEN row `[1, H]` — the input is the token's own, quantized once
  and read by every expert's two gemms;
* down: one scale per (token, expert) row `[1, I]` — `moe.py`'s per-pair grain (#639): at a
  routed expert the row is the pair's own activation.

For a token NOT routed to e, its row's product is multiplied by an exact 0 and adds nothing.
⛔ THE f16 ACCUMULATION ORDER IS EXPERT-ID ORDER here, slot order in `moe.py`; the two are the
same sum in a different association, so they agree to f16 rounding, not bit for bit.

THE SPLICE'S OWN CONTRACT:

* PARAMETERS, IN ORDER: `moe.py`'s first twelve (router input, gain, projection, per-expert
  scale; expert input; the three fp8 banks and their scales; the output), then the SCRATCH:
  `desc_dense` `[T, E]` the dense route.
* CONSTEXPRS: `T` (the prompt's token rows), `H`, `I`, `E`, `K`; `EPS`, `INV_H`, `TEMP`;
  `INV_QMAX`, `SCALE_FLOOR` — `moe.py`'s, with `T` for `M`.
* GRID: `[1]`.

The router's top-K and dense-route rounds are `tl.static_range` over the K slots (K is the
model's, not the prompt's); the experts are no loop at all, each projection one contraction
over all E.
"""

import triton
import triton.language as tl


@triton.jit
def moe_prefill_fwd(desc_xr, desc_rg, desc_rw, desc_pes,  #
                    desc_xe, desc_wg, desc_sg, desc_wu, desc_su, desc_wd, desc_sd,  #
                    desc_o,  #
                    desc_dense,  #
                    T: tl.constexpr, H: tl.constexpr, I: tl.constexpr,  #
                    E: tl.constexpr, K: tl.constexpr,  #
                    EPS: tl.constexpr, INV_H: tl.constexpr, TEMP: tl.constexpr,  #
                    INV_QMAX: tl.constexpr, SCALE_FLOOR: tl.constexpr):
    # ── the router: `moe.py`'s, over T rows ───────────────────────────────────
    xr_desc = tl.make_tensor_descriptor(desc_xr, shape=[T, H], strides=[H, 1],
                                        block_shape=[T, H])
    rg_desc = tl.make_tensor_descriptor(desc_rg, shape=[H], strides=[1], block_shape=[H])
    rw_desc = tl.make_tensor_descriptor(desc_rw, shape=[E, H], strides=[H, 1],
                                        block_shape=[E, H])
    pes_desc = tl.make_tensor_descriptor(desc_pes, shape=[E], strides=[1], block_shape=[E])
    xr = xr_desc.load([0, 0])
    ms = tl.sum(xr * xr, 1) * INV_H
    r = tl.rsqrt((ms + EPS).to(tl.float32)).to(tl.float16)
    xn = xr * r[:, None] * rg_desc.load([0])[None, :]
    lg = tl.dot(xn, rw_desc.load([0, 0]).T, tl.zeros([T, E], dtype=tl.float16))

    offs = tl.arange(0, E).to(tl.float16)
    koffs = tl.arange(0, K).to(tl.float16)
    pes = pes_desc.load([0])
    vals = tl.zeros([T, K], dtype=tl.float16)
    idsf = tl.zeros([T, K], dtype=tl.float16)
    psel = tl.zeros([T, K], dtype=tl.float16)
    neg_inf = tl.full([T, E], float("-inf"), tl.float16)
    no_lane = tl.full([T, E], -1.0, tl.float16)
    zero_e = tl.zeros([T, E], dtype=tl.float16)
    # A per-row or per-lane value enters a `[T, E]` select as itself: `tl.where` broadcasts
    # both sides to the condition's shape. Not `x + zero_e` (`moe.py`'s decode spelling,
    # where every tile is one row): at T rows that add of a zero splat is the one shape
    # upstream's `reorder-broadcast` rewrites, and staying out of that pass's reach keeps our
    # make_ttir and theirs one form.
    one_k = tl.full([K], 1.0, tl.float16)
    zero_k = tl.zeros([K], dtype=tl.float16)
    for j in tl.static_range(0, K):
        mx = tl.max(lg, 1).to(tl.float16)
        lanes = tl.where(lg == mx[:, None], offs[None, :], no_lane)
        idx = tl.max(lanes, 1).to(tl.float16)
        sel = offs[None, :] == idx[:, None]
        pj = tl.sum(tl.where(sel, pes[None, :], zero_e), 1)
        oh = tl.where(koffs == j, one_k, zero_k)[None, :]
        vals = vals + mx[:, None] * oh
        idsf = idsf + idx[:, None] * oh
        psel = psel + pj[:, None] * oh
        lg = tl.where(sel, neg_inf, lg)

    # The route weights, `moe.py`'s chain: softmax over the K at `H^-0.5`, × the scales.
    sv = (vals * TEMP).to(tl.float32)
    ex = tl.exp(sv - tl.max(sv, 1)[:, None])
    rw = tl.fdiv(ex, tl.sum(ex, 1)[:, None]).to(tl.float16) * psel

    # ── the dense route `[T, E]`: slot j's weight at lane ids[:, j] ──────────
    # Each lane is hit at most once per row (the top-K ids are distinct), so adding a
    # one-hot column into zeros places every weight exactly and leaves every other lane 0.
    dense = tl.zeros([T, E], dtype=tl.float16)
    zero_tk = tl.zeros([T, K], dtype=tl.float16)
    for j in tl.static_range(0, K):
        idj = tl.sum(tl.where(koffs[None, :] == j, idsf, zero_tk), 1)
        wj = tl.sum(tl.where(koffs[None, :] == j, rw, zero_tk), 1)
        dense = dense + tl.where(offs[None, :] == idj[:, None], wj[:, None], zero_e)
    dense_desc = tl.make_tensor_descriptor(desc_dense, shape=[T, E], strides=[E, 1],
                                           block_shape=[T, E])
    dense_desc.store([0, 0], dense)

    # ── the experts, every one, over the whole token tile ─────────────────────
    # ── the experts: THREE CONTRACTIONS over all E experts ───────────────────
    # Every expert runs every token, weighted by the dense route weight (zero where the
    # token did not route to it). gate/up read the same token codes for every expert, so
    # each is ONE contraction against all experts' rows stacked (`[T, H] × [H, E·I]`); down
    # reads each (expert, token) row's own codes, so it batches over the experts
    # (`[E, T, I] × [E, I, H]`); the combine is a sum over the experts.
    xe_desc = tl.make_tensor_descriptor(desc_xe, shape=[T, H], strides=[H, 1],
                                        block_shape=[T, H])
    wg_desc = tl.make_tensor_descriptor(desc_wg, shape=[E * I, H], strides=[H, 1],
                                        block_shape=[E * I, H])
    wu_desc = tl.make_tensor_descriptor(desc_wu, shape=[E * I, H], strides=[H, 1],
                                        block_shape=[E * I, H])
    wd_desc = tl.make_tensor_descriptor(desc_wd, shape=[E, H, I], strides=[H * I, I, 1],
                                        block_shape=[E, H, I])
    sg_desc = tl.make_tensor_descriptor(desc_sg, shape=[E * I], strides=[1],
                                        block_shape=[E * I])
    su_desc = tl.make_tensor_descriptor(desc_su, shape=[E * I], strides=[1],
                                        block_shape=[E * I])
    sd_desc = tl.make_tensor_descriptor(desc_sd, shape=[E, H], strides=[H, 1],
                                        block_shape=[E, H])
    o_desc = tl.make_tensor_descriptor(desc_o, shape=[T, H], strides=[H, 1],
                                       block_shape=[T, H])
    # The gate/up activation: each token's input, one scale per token for every expert.
    xe = xe_desc.load([0, 0])
    sx = tl.minimum(tl.maximum(tl.max(tl.abs(xe), 1).to(tl.float16) * INV_QMAX,
                               tl.full([T], SCALE_FLOOR, tl.float16)),
                    tl.full([T], 448.0, tl.float16))
    ix = tl.fdiv(tl.full([T], 1.0, tl.float16), sx)
    qx = tl.minimum(tl.maximum(xe * ix[:, None], tl.full([T, H], -448.0, tl.float16)),
                    tl.full([T, H], 448.0, tl.float16))
    qx = qx.to(tl.float8e4nv).to(tl.float16)
    wg = wg_desc.load([0, 0]).to(tl.float16)
    wu = wu_desc.load([0, 0]).to(tl.float16)
    gate = tl.dot(qx, wg.T, out_dtype=tl.float16)
    up = tl.dot(qx, wu.T, out_dtype=tl.float16)
    gate = tl.reshape((gate * sx[:, None]) * sg_desc.load([0])[None, :], [T, E, I])
    up = tl.reshape((up * sx[:, None]) * su_desc.load([0])[None, :], [T, E, I])
    # gelu_tanh(gate) · up. The source states the cubic in f32 (an f16 x³ overflows at
    # |x| ≥ 41); the front end's type legalization lowers the island to f16, where with
    # 0.044715 multiplied first the cube overflows only past |x| ≈ 114, to the right limit.
    g32 = gate.to(tl.float32)
    v = 1.5957691216057308 * (g32 + 0.044715 * g32 * g32 * g32)
    eg = tl.exp(-v).to(tl.float16)
    act = tl.fdiv(gate, tl.full([T, E, I], 1.0, tl.float16) + eg) * up
    # The down activation: each (token, expert) row's own scale.
    sa = tl.minimum(tl.maximum(tl.max(tl.abs(act), 2).to(tl.float16) * INV_QMAX,
                               tl.full([T, E], SCALE_FLOOR, tl.float16)),
                    tl.full([T, E], 448.0, tl.float16))
    ia = tl.fdiv(tl.full([T, E], 1.0, tl.float16), sa)
    qa = tl.minimum(tl.maximum(act * ia[:, :, None], tl.full([T, E, I], -448.0, tl.float16)),
                    tl.full([T, E, I], 448.0, tl.float16))
    qa = tl.trans(qa.to(tl.float8e4nv).to(tl.float16), (1, 0, 2))
    wd = wd_desc.load([0, 0, 0]).to(tl.float16)
    down = tl.dot(qa, tl.trans(wd, (0, 2, 1)), out_dtype=tl.float16)
    down = (down * tl.trans(sa, (1, 0))[:, :, None]) * sd_desc.load([0, 0])[:, None, :]
    # The combine: Σ_e dense[t, e] · down[e, t], in f16.
    o_desc.store([0, 0], tl.sum(down * tl.trans(dense, (1, 0))[:, :, None], 0))
