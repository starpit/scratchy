// SPDX-License-Identifier: Apache-2.0
//! THE PROVEN fp8 W8A8 MATMUL DESCRIPTORS, driven from the KTIR.
//!
//! ⭐⭐⭐ THE BODY IS `ibm/main`'s `lower_matmul_node` arity-3 branch, UNCHANGED — the whole W8A8 chain:
//! per-token activation quantize (`abs` → `amax` → scale → clamp → `qfp8ch`), `matmulfp8`
//! (fp8×fp8→fp16), then dequant by `a_scale`(per row) · `w_scale`(per channel). Only its DOOR changed:
//! it takes [`Fp8Facts`] recovered from the KTIR program instead of a `&SubtileNode`.
//!
//! ⛔ AND THAT CHAIN IS WHY fp8 NEEDS THIS. The builder's `KtirFunc::matmul_fp8` computed **fp16 activation × fp8
//! weight** — its own doc says outright that it does not quantize the activation — while this device's
//! fp8 is fp8×fp8. Lowering the KTIR op-for-op therefore cannot produce `matmulfp8` at all, and it
//! also addressed the weight as fp16: a `[2048, 2048]` fp8 weight is placed 4 MB and was being read
//! 8 MB. fp8 W8A8 runs on hardware today through this code, so the card keeps using it — reached
//! through the KTIR, from facts the KTIR itself states.

use super::{
    EmittedOp, In, PointwiseFacts, assemble_convert, assemble_convert_col,
    assemble_pointwise_broadcast_off, bmm_site, emit_sdsc_tiled, op_func_from_str, pw1, pw2, rb,
    rb_df, rbo,
};
use crate::addr::{DevOff, Idx, Nest, Row, Slabs};
use crate::ir::bridge::tiled_op_sdsc_op::reduce::{
    assemble_reduce_off, assemble_reduce_seeded, reduce_opspec, win_reduce_accum, win_reduce_data,
};
use crate::place::{PlaceId, SynthRole};
use crate::placement::{BundleLayout, syn};
use crate::reserved_tids::{FP8_INV448_TID, FP8_NEG448_TID, FP8_POS448_TID};
use crate::sdsc_abstract::{
    BlockCols, Fp8NarrowStaging, Fp8TileStaging, KernelTag, Lanes, OperandPlacement, RowCount, Stk,
};
use crate::superdsc_error::SuperDscError;
use crate::superdsc_opspec::{DataFormat, Df, Fp16, Role, SdscFoldSet};
use std::collections::BTreeMap;

/// What the KTIR states about one fp8 W8A8 contraction, recovered from its shapes and parameters.
///
/// ⛔ EVERY FIELD IS READ OUT OF THE KTIR. The fp8-ness itself is too: the builder's
/// `KtirFunc::matmul_fp8` built
/// its weight view through `view_fp8`, which writes `Dtype = Fp8E4m3` on the
/// `ktdp.construct_memory_view` — so the recogniser asks the view what its element type is rather than
/// guessing from arity.
pub(crate) struct Fp8Facts {
    /// The activation's tensor id — the quant chain's tensors are keyed off it so two matmuls on one
    /// activation share a single quantize.
    pub a_tid: u32,
    pub a_name: String,
    /// The fp8 weight's buffer name.
    pub w_name: String,
    /// The per-output-channel `w_scale`.
    pub ws_name: String,
    /// The output tensor id.
    pub out_tid: u32,
}

/// ⭐ THE QUANT CHAIN'S SOURCE — the activation whole (the dense law) or ONE target-ABI row
/// tile's fresh clone (the row-tiled law, [`crate::work::Fp8GemmRowTiles`]). The chain is a
/// pure function of its source's `[rows, k]` window, so ONE body serves both laws; what
/// changes is WHERE it reads and WHAT its tensors are named — a tile's quant tensors carry
/// their tile index ([`PlaceId::synth_tiled`]), keyed off the ACTIVATION's id either way
/// because a tile's staging is a fact of the activation that tiled, and its ops carry the
/// index in their names. The `quantized` sharing key is the source's own name, so a second
/// GEMM over the same activation at the same plan reuses one staging instead of
/// re-quantizing per matmul.
pub(crate) struct Fp8QuantSource {
    /// The tensor the chain reads — the activation, or the tile's fp16 clone.
    pub name: String,
    /// The id every chain synth derives from — the ACTIVATION's, either way.
    pub id: PlaceId,
    /// The rows the source carries — `m`, or the tile's ABI row class.
    pub rows: u32,
    /// The tile index when this source is one tile of the row-tiled law (`None` = dense).
    pub tile: Option<u32>,
    /// The output tensor's id — the op-name namespace, shared by both laws.
    pub out_tid: u32,
}

impl Fp8QuantSource {
    /// The DENSE law's source — the activation whole, at its own row count.
    pub(crate) fn activation(facts: &Fp8Facts, m: u32) -> Self {
        Self {
            name: facts.a_name.clone(),
            id: PlaceId::Act(facts.a_tid),
            rows: m,
            tile: None,
            out_tid: facts.out_tid,
        }
    }
}

/// ⭐⭐⭐ THE SHARED W8A8 ACTIVATION QUANTIZE — the first half of every fp8 matmul door
/// (`matmul_fp8_descriptors`'s own body, extracted): `abs → amax → floor → scale → clamp →
/// zero-pad → qfp8ch`, over the source's `[rows, k]` window, ONE scale per token row. The EXPERT
/// door calls this DIRECTLY (its per-pair matmuls and flat dequant legs replace the dense body),
/// and so does the row-tiled arm — once per tile over the tile's fresh clone (the reference
/// quantizes AFTER the clone, `fp8_linear_kernel.py:351` then `:33`) — so every door shares one
/// chain and no op is emitted only to be spliced out — the dense body's
/// own `matmulfp8` binds its W as a whole-tensor `[k_pad, n]` view, which against the expert
/// door's chunk scratch the synth footprint guard refuses at bake.
///
/// ⭐⭐ THE TAIL PADS PER PAIR. `pairs` is the K axis's pair count (dense = 1; the expert door
/// passes its top-k): the activation is `pairs` windows of `inn = k/pairs` lanes, and the tail
/// tensors (`cl`, `afp8`) are laid out per-pair PADDED — pair `p`'s window at `p·inn_pad` of a
/// `[m, pairs·inn_pad]` buffer. The reason is the fp8 STICK LAW: the matmul's A operand start
/// must land on a whole 128-elem fp8 stick (the HBM walk is stick-granular — the senulator
/// loads whole 128-B sticks by stick index), and at `inn ≡ 64 (mod 128)` (the 26b DOWN door's
/// 704) the LOGICAL pair stride is a HALF stick, so the padded layout is the only one whose
/// every window start is stick-aligned. The per-pair form fires ONLY at `pairs > 1 ∧
/// inn_pad ≠ inn`: at an aligned `inn` the padded layout IS the logical one and the whole-tensor
/// convert stands (byte-identical to the pre-pair emission), and at `pairs == 1` this is the
/// dense law verbatim (`cl_reach = k_pad = fp8_k_pad(k)`, one tail window, one convert).
pub(crate) fn fp8_quantize_descriptors(
    src: &Fp8QuantSource,
    k: u32,
    pairs: u32,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
    quantized: &mut std::collections::HashSet<String>,
) -> Result<Vec<EmittedOp>, SuperDscError> {
    let m = src.rows;
    // fp8 W8A8 PREFILL (m>1). torch-spyre's `quantize_fp8_with_scale`
    // (decompositions.py:966) takes `scale` as an input computed by the CALLER via native `torch.abs` +
    // `torch.ops.aten.amax` (confirmed via grep — no L2-norm substitute anywhere in torch-spyre); the
    // `#33` belief that reduce-MAX "seeds 0 for rows>1" was disproved by the stable-softmax `block_max`
    // (attn.rs), itself a multi-row MAX reduce. ONE formula for any row count (m==1 decode or m>1
    // prefill), matching torch-spyre exactly — mirrors rmsnorm's unification (no scratchy-only approx,
    // no diagnostic-sweep scaffolding left in the live path).
    if pairs == 0 || !k.is_multiple_of(pairs) {
        return Err(SuperDscError(format!(
            "fp8 quantize t{}: the K axis is {k} lanes over {pairs} pairs — the per-pair tail law \
             needs a whole lane count per pair",
            src.id.tid()
        )));
    }
    let inn = k / pairs;
    let inn_pad = crate::work::fp8_k_pad(inn);
    // ⭐ THE HALF-STICK TAIL, m-GENERIC: at an ALIGNED `inn` (`inn_pad == inn` — the expert
    // door's gate/up at 2816, every dense caller at `pairs == 1`) the padded layout IS the
    // logical one and the whole-tensor convert stands at ANY row count (the dense prefill's
    // own card-proven form). At the half-stick tail (`inn_pad != inn`, the 26b DOWN's 704)
    // the per-pair arms below now split on `m`: m=1 keeps the FLAT stick-axis slice (the
    // decode form, byte-identical), m>1 takes [`assemble_convert_col`] at [`crate::addr::col_of`]
    // group corners — the per-slot rank-2 stick-major arms of design §4.5, landed.
    let stk = Fp16::ELEMS_PER_STICK; // 64
    // ⭐ THE 26b K-PAD — the contraction width rounded to a whole 128-elem SEN143_FP8 stick
    // (`work::fp8_k_pad`, the K-axis twin of `DeviceWidth`'s N law). gemma-4-26b's dense GeGLU
    // DOWN gemm contracts k=2112 = 16.5 fp8 sticks — the FIRST fp8 model with a non-legal K
    // (granite/g8b/g12b are all 128-multiples) — and `qfp8ch`'s convert guard refuses a
    // sub-stick tile outright. The fp16 READS of this chain stay at the LOGICAL `[m, k]` (the
    // activation's own reservation is k-wide; reading past it is out-of-footprint); only the
    // chain's TAIL widens — per pair at `pairs > 1`, one window at `pairs == 1`.
    // (`inn`/`inn_pad` are computed ABOVE, stated once for the whole tail law — the arms
    // below and the reservations all read them.)
    let k_pad = pairs * inn_pad; // the tail tensors' width (the per-pair padded layout)
    // `cl`'s own reach: the LAST pair's convert reads `[p·inn, p·inn + inn_pad)` — its pad
    // lanes overhang the logical width by exactly `inn_pad − inn` (one fp16 stick at the
    // door's whole-stick `inn`), so `cl` reserves `k + (inn_pad − inn)` and the fill below
    // covers the overhang. At `pairs == 1` this is `k_pad` itself (the dense law).
    let cl_reach = (pairs - 1) * inn + inn_pad;
    let ops = {
        let m_rows = crate::sdsc_abstract::RowCount::of_token_rows(m);
        let k_cols = crate::sdsc_abstract::BlockCols::of_feature_cols(k);
        let scale_cols =
            crate::sdsc_abstract::BlockCols::of_one_stick(crate::sdsc_abstract::Lanes::FP16);
        // ⭐ THE NAMES COME FROM THE KTIR'S OWN PARAMETERS — `args[i]` → `wiring::act_name(tid)`, the
        // same spelling `ds_name` produced for the SubtileIR path.
        let a_name = src.name.clone();
        let out = crate::place::act_name(src.out_tid);
        use SynthRole as R;
        // SHARED quant tensors are a pure function of the ACTIVATION (abs→amax→scale→clamp→fp8) — derive
        // them from the ACTIVATION's id so a second matmul on the same activation resolves the IDENTICAL
        // fp8 tensor + a_scale (the layout `synth` is idempotent per id → one reservation). PER-MATMUL
        // tensors (the matmul output `raw` and its dequant `dqa`) derive from the OUTPUT's id — those
        // belong to the dense body, not this half.
        // ⭐ THE DOOR: the ids come from the KTIR-recovered facts, not from a `&SubtileNode`. They are
        // the same tensor ids either way — `Fp8QuantSource::id` is recorded off the program's
        // own views — so the placements these mint are the ones the proven path minted.
        let a_id = src.id;
        // ⭐ THE TILE ID MAP — a tiled source wraps EVERY role's id in its tile index
        // (`synth_tiled`), so a tile's staging tensors are named apart from the activation's
        // own and from every other tile's; the dense source maps every role to itself
        // (byte-identical names).
        let st = |r: R| match src.tile {
            Some(t) => a_id.synth_tiled(r, t),
            None => a_id.synth(r),
        };
        let qn = |r: R| syn(layout, st(r));
        // OP names — a DIFFERENT namespace from tensor names (a program's ops are not placed), so
        // they stay strings and take no `PlaceId`. A tiled source carries its tile index here too,
        // BEFORE the trailing `_op` — the tiled arm's one naming law (`fq_tz{t}_op`,
        // `fq_mm_t{t}_op`) — so a tile's ops are named apart from every other tile's.
        let opn = |s: &str| match src.tile {
            Some(t) => format!("{out}_{}_t{t}_op", s.strip_suffix("_op").unwrap_or(s)),
            None => format!("{out}_{s}"),
        };
        let (absx, amax, amaxfl, ascale, invs, sc, chi, cl) = (
            qn(R::FqAbsX),
            qn(R::FqAmax),
            qn(R::FqAmaxFl),
            qn(R::FqAscale),
            qn(R::FqInvS),
            qn(R::FqSc),
            qn(R::FqChi),
            qn(R::FqCl),
        );
        let afp8 = qn(R::FqAfp8); // quantized fp8 activation — sized/typed fp8 via synth_df + the convert
        if let Some(l) = layout {
            // Footprints are `m` query rows (decode m=1 ⇒ byte-identical to the old `[1,·]`; prefill m>1
            // reserves the real multi-row extent so a per-row write can't clobber a neighbor row).
            for r in [R::FqAbsX, R::FqSc, R::FqChi] {
                l.synth(st(r), &[m, k]);
            }
            // The chain's tail widens to the PAD: `cl` feeds the convert (whose 128-elem stick law
            // is the reason the pad exists) and `afp8` is the fp8 contraction operand — both are
            // reserved at the PER-PAIR PADDED width, the one layout the convert, the per-pair
            // matmul A windows, and the packed weight's K rows all share (`cl` only to its last
            // convert's REACH — the one-stick overhang the fill below covers).
            l.synth(st(R::FqCl), &[m, cl_reach]);
            l.synth_df(st(R::FqAfp8), &[m, k_pad], Df::Fp8); // 1-byte / 128-stick residency (½ fp16)
            for r in [R::FqAmax, R::FqAmaxFl, R::FqAscale, R::FqInvS] {
                l.synth(st(r), &[m, stk]);
            }
        }
        let pos448 = rbo(&crate::place::act_name(FP8_POS448_TID));
        let neg448 = rbo(&crate::place::act_name(FP8_NEG448_TID));
        let inv448 = rbo(&crate::place::act_name(FP8_INV448_TID));
        let mut ops: Vec<EmittedOp> = Vec::new();
        // SHARE the activation quantize across matmuls that read the SAME activation. `insert` returns false
        // when this activation (`a_name`) was already quantized earlier in the bundle (e.g. K/V after Q, or
        // Up after Gate — the expert door's gate/up pair is the same law) — in that case skip re-emitting
        // the whole abs→amax→scale→clamp→qfp8ch chain and reuse its `afp8`/`ascale` (both named by
        // `a_name`); only the per-matmul legs stay per-matmul. (The chain is a launch/op cost paid ONCE
        // per distinct activation instead of once per matmul.)
        let already_quantized = !quantized.insert(a_name.clone());
        if !already_quantized {
            // `absx`'s ONLY consumer is the MAX-reduce right below (never a matmul) — same bug class as
            // rmsnorm's `sq16` (see rmsnorm.rs): a shape-driven RowBlocked handle would make it stickmajor-
            // eligible purely from shape (rows>1 && cols>64, true here for any real hidden/intermediate `k`),
            // and the native reduce cannot correctly walk a stick-major activation along `cols` at rows>1 (it
            // sums/maxes across the wrong rows). Emitted via the TYPED `pw1`/`Stk<FlatTag>` path (not a raw
            // `&str` one) so the producer and this reduce's consumer are the SAME compile-time-checked
            // handle — `fl()` forces both ends flat; a kind mismatch would be a `cargo build` type error.
            // `absx` is RowBlocked, NOT Flat (2026-07-28) -- same fix, same reason, as rmsnorm's
            // `sq16` (see the long note in ir/bridge/tiled_op_sdsc_op/rmsnorm.rs). A Flat OUTPUT
            // handle forces head_major=true for the WHOLE op, which per pointwise.rs "stays rank-3"
            // and therefore also reads the INPUT activation rank-3 flat. But `a_name` is the
            // stick-major residual stream, so at m=mq=31 with k=2048 that read is (pointwise.rs's own
            // words) "scrambled at rows>1 AND cols>64": the per-row amax was taken over a scrambled
            // element set, giving every row the wrong fp8 activation scale -- on EVERY fp8 matmul
            // (qkv, o_proj, gate/up, down), every layer, for batched prefill only. RowBlocked puts
            // both this op and the amax reduce on the stick-major rank-2 path, matching how the
            // activation was actually written. At m==1 (decode) `stickmajor` is false in both, so the
            // emission is byte-identical to the previous Flat form.
            // ⭐⭐⭐⭐⭐ ONE HARDWARE OP FOR THE FIRST FOUR STEPS — see [`OpFunc::QuantScalePerTokenFp8`].
            //
            // `abs → max-reduce → floor → ·(1/448)` IS `quantscalepertokenfp8`, whose DDL body is the
            // FABSMAX tree (`quant_scale_per_token.ddl:178-232`) with its partials in
            // `ddl.internal_tensor`, i.e. on-chip, and whose tail is exactly
            // `FMIN(FMAX(absmax·mulConst, clipMin), clipMax)` (`:259/:283/:287`) — the three constants
            // [`OpFunc::op_consts`] supplies are picked to make that identical to `fq_amaxfl` +
            // `fq_ascale`, floor included. Registered for dd2 (`multi_sentient_dd2_sysconfig.json:87`).
            //
            // ⛔ IT IS SHAPED EXACTLY LIKE THE REDUCE IT REPLACES, which is why it goes through the
            // reduce assembler and not a new one: input `[m, k]`, output `[m, stick]` — the DDL's
            // `%global_layout_output` drops the reduce dim its `%global_layout_input` carries, so the
            // output IS the per-token scale. `invs` below still reciprocates it (the DDL keeps its own
            // reciprocal INTERNAL, so it is not an operand we can read), and `sc/chi/cl/afp8` still
            // build the `[m,k]` fp8 tensor. 4 ops → 1, ×4 activations a layer = 12 of the 51 glue ops
            // that are 67% of a bs=1 decode layer.
            //
            // ⛔ IT IS AN UNSEEDED REDUCE, so it does NOT take the accumulator as a second input the way
            // `max`/`sum`/`mean` do — `OpFunc::ddl_input_arity` carries that fact and the emitter makes
            // a violation a build error. Passing the seed is not a diagnosable error in the compiler: it
            // is a silent template non-match (MEASURED — the same descriptor with two inputs is
            // `[DDC] DDL found but not suitable for op quantscalepertokenfp8`, with one input it is
            // accepted).
            //
            // ⛔ `absx`/`amax`/`amaxfl` are no longer materialised. Their `synth` reservations are left
            // in place deliberately for this first measurement so the bundle's ADDRESS ASSIGNMENT does
            // not move — `BundleLayout::synth` is a bump allocator, so dropping entries would shift
            // every later tensor and confound a coherence comparison with the baseline.
            let ascale_h = rb(
                &ascale,
                RowCount::of_token_rows(m),
                BlockCols::of_one_stick(Lanes::FP16),
            );
            let a_h = rb(
                &a_name,
                RowCount::of_token_rows(m),
                BlockCols::of_feature_cols(k),
            );
            // ⛔ ASK THE TILER FIRST — the SAME `reduce_opspec` the emission below builds, never a
            // second LX formula. A time-tiled reduce is UNEMITTABLE for this op class: the DDL's
            // accumulator has OVERWRITE semantics while `rewrite_op_for_time_tile`'s tiled predicate
            // exempts the `[Active, RedStick]` output from per-trip strides — every trip writes the
            // SAME `[m,64]` scale bytes, so the surviving scale would be the LAST slice's, not the
            // global one. The bake's `tiled_trips_alias` guard correctly refuses it (12b fp8 prefill
            // rung 71: at the true DDL arity the residency is `1·71·15360·2 = 2,181,120 B` vs the
            // 1,677,721-B budget — a PRIME rung, so `core_split` finds no divisor and the reduce
            // keeps the whole row count per core). So when the tiler would time-tile, emit the
            // remedy `reduce.rs`'s own doc names — "tile the reduction ... and combine the
            // partials": one fused reduce per k/time-slice into a DISJOINT `[m,64]` slot, then
            // fold them with `maximum`.
            let single = reduce_opspec(
                op_func_from_str("quantscalepertokenfp8"),
                m,
                k,
                &a_name,
                &ascale,
                false,
            )
            .map_err(|e| SuperDscError(format!("fq_qspt_op: {e}")))?;
            let qspt_time = single.time();
            if qspt_time == 1 || single.iter.split_of("out") > 1 {
                // Non-tiled (every existing geometry: decode m=1, granite fp8 prefill, the 12b's
                // split rungs) — the single fused reduce, byte-identical to the pre-partials
                // emission. (A reduction-core split is left to this path too: the partials+combine
                // remedy below is derived for the pure time-tiling case.)
                ops.push(assemble_reduce_seeded(
                    &opn("fq_qspt_op"),
                    "quantscalepertokenfp8",
                    m,
                    k,
                    &a_h,
                    &ascale_h,
                    sym_id_base,
                    layout,
                ));
            } else {
                // PARTIALS: `time` single-shot fused reduces, each over a `[m, k/time]` column slice
                // of the RowBlocked activation. Slice j starts at column `j·k/time` — a stick
                // boundary by the TimeTile invariant (`k/time` is a whole 64-stick) — whose device
                // element offset `col_of` renders through the RowBlocked law: stick-group g of every
                // row sits at `g·m·64`, so a whole-slice shift is exactly `j` blocks of the reduce's
                // own `[m, k/time]` slab (block-aligned, which is what keeps `reduce_opspec_off` on
                // the stick-major rank-2 path — a flat rank-3 read of a RowBlocked buffer scrambles
                // at rows>1 && cols>64). Partial 0 seeds `ascale` itself (attn's bmax→run_m
                // precedent: no copy op for the only-ever-first slot); partials 1..time each land in
                // their OWN `[m,64]` synth, so no two reduces share an output byte — the aliasing
                // the bake guard refuses is unrepresentable here, not merely avoided. Each partial
                // is itself single-shot BY CONSTRUCTION: its residency is exactly the per-trip
                // residency the tiler validated when it chose `time`.
                //
                // ⛔ THE COMBINE MAY FOLD SCALES, NOT AMAXES, AND STILL BE EXACT: the fused op's
                // tail `FMIN(FMAX(absmax·mulConst, clipMin), clipMax)` is MONOTONE NON-DECREASING
                // in the slice's amax (a positive scale, then a clamp), and a monotone
                // non-decreasing g commutes with max — `max_j g(amax_j) = g(max_j amax_j)` — so
                // `maximum` over the per-slice SCALES is bit-exact the single-shot scale, floor and
                // clip included. All-zero padded rows included: every slice's g(0) is the clipMin
                // floor, and the max of identical floors is that floor.
                let k_part = k / qspt_time;
                let partials: Vec<String> = (1..qspt_time)
                    .map(|j| syn(layout, st(R::FqAmaxP(j))))
                    .collect();
                if let Some(l) = layout {
                    for j in 1..qspt_time {
                        l.synth(st(R::FqAmaxP(j)), &[m, stk]);
                    }
                }
                for j in 0..qspt_time {
                    let accum_h = if j == 0 {
                        rb(
                            &ascale,
                            RowCount::of_token_rows(m),
                            BlockCols::of_one_stick(Lanes::FP16),
                        )
                    } else {
                        rb(
                            &partials[(j - 1) as usize],
                            RowCount::of_token_rows(m),
                            BlockCols::of_one_stick(Lanes::FP16),
                        )
                    };
                    let op_name = if j == 0 {
                        opn("fq_qspt_op")
                    } else {
                        opn(&format!("fq_qspt_p{j}_op"))
                    };
                    ops.push(assemble_reduce_off(
                        &op_name,
                        "quantscalepertokenfp8",
                        m_rows,
                        crate::sdsc_abstract::BlockCols::of_feature_cols(k_part),
                        // The data window is the j-th `[m, k/time]` slab — `col_of`'s whole-block
                        // shift (k/time is a whole 64-stick, the TimeTile invariant), minted
                        // against the activation's own `[m, k]` handle.
                        win_reduce_data(
                            layout,
                            &a_h,
                            crate::addr::col_of(m, k, j * k_part, Df::Fp16),
                            m_rows,
                            crate::sdsc_abstract::BlockCols::of_feature_cols(k_part),
                            &format!("{op_name}: activation"),
                        )
                        .map_err(SuperDscError)?,
                        win_reduce_accum(
                            layout,
                            &accum_h,
                            crate::addr::DevOff::ZERO,
                            m_rows,
                            &format!("{op_name}: scale"),
                        )
                        .map_err(SuperDscError)?,
                        sym_id_base,
                        layout,
                    ));
                }
                // COMBINE: ascale = maximum(ascale, partial_j) — running, in place on `ascale` (the
                // old amax chain's own law). Lands in the SAME `ascale` name every downstream op
                // (`fq_invs_op` on) already reads, so nothing below changes.
                for (i, p) in partials.iter().enumerate() {
                    let p_h = rb(
                        p,
                        RowCount::of_token_rows(m),
                        BlockCols::of_one_stick(Lanes::FP16),
                    );
                    ops.push(pw2(
                        &opn(&format!("fq_qspt_c{}_op", i + 1)),
                        "maximum",
                        m_rows,
                        scale_cols,
                        In::full(&ascale_h),
                        In::full(&p_h),
                        &ascale_h,
                        sym_id_base,
                        layout,
                    ));
                }
            }
            // The three tensors the two fused ops subsume, plus the `1/448` and ±448 they used to take as
            // scalar tensor operands — the constants now reach their own `mulConst`/`clipMin`/`clipMax`
            // through `ddl.get_external_constant` BY NAME out of the descriptor's `constantInfo_`.
            // (`chi` joins them: materialised by nothing, its reservation held for allocator stability.)
            let _ = (&absx, &amax, &amaxfl, &chi, &inv448, &pos448, &neg448);
            // REVERTED (2026-07-28): a multi-stage blocked-reduce experiment lived here (three
            // iterations, all confirmed on real hardware to make ZERO difference to the actual bug —
            // the K-cache inf this was meant to fix turned out to be caused by cachewr's matmul
            // mechanism instead, see the cachewr plain-copy fix). Unlike the reduce-MAX-at-rows>1
            // theory (which was checked against the OLD proven code and definitively disproven — that
            // code batches rows=nqh fine), this width-blocking theory was never confirmed against any
            // proven reference at all, was pure speculation, and never demonstrated any real effect.
            // Keeping ~150 lines of never-validated-on-hardware blocking logic around after its own
            // premise (fixing K-cache inf) is already fixed elsewhere is unjustified risk with no
            // upside. Reverted to the simple, single unblocked reduce — decode already proves this form
            // works (m=1 always takes this path); prefill (m>1) now takes it too.
            let invs_h = rb(
                &invs,
                RowCount::of_token_rows(m),
                BlockCols::of_one_stick(Lanes::FP16),
            );
            ops.push(pw1(
                &opn("fq_invs_op"),
                "reciprocal",
                m_rows,
                scale_cols,
                In::full(&ascale_h),
                &invs_h,
                sym_id_base,
                layout,
            ));
            // scaled = act·invs; clamp to [-448,448] (torch-spyre's `quantize_fp8_with_scale` ALWAYS clamps).
            let sc_h = rb(
                &sc,
                RowCount::of_token_rows(m),
                BlockCols::of_feature_cols(k),
            );
            ops.push(pw2(
                &opn("fq_sc_op"),
                "mul",
                m_rows,
                k_cols,
                In::full(&rb(
                    &a_name,
                    RowCount::of_token_rows(m),
                    BlockCols::of_feature_cols(k),
                )),
                In::col(&invs_h),
                &sc_h,
                sym_id_base,
                layout,
            ));
            // ⭐⭐⭐⭐⭐ THE CLAMP IS THE DEVICE'S OWN `clip` OP — ONE input, computing
            // `FMIN(FMAX(x, clipMin), clipMax)` (`unary_pipeline.ddl:370-377` PE stage, `:472-479` SFP
            // stage) — the same min/max pair the two ops it replaces encoded, so the swap is
            // value-preserving by construction. The ±448 E4M3 bounds arrive as the external constants
            // `clipMin`/`clipMax` out of the descriptor's `constantInfo_` ([`OpFunc::op_consts`]),
            // NOT as the `[1, stick]` H2D tensor operands `fq_chi`/`fq_cl` read through
            // `In::scalar(&pos448)`/`In::scalar(&neg448)` — the constants dxp folds before the
            // constant tables, per the same contract as `quantscalepertokenfp8`'s three. 2 ops → 1,
            // ×4 activations ×40 layers = 40 more ops off a granite-2b fp8 decode step. `FP8_POS448`
            // keeps exactly ONE reader — the pad window's zero fill below (`sub(448,448)`, a
            // broadcast zero); `FP8_NEG448`'s bind loses its last reader.
            //
            // ⛔ `chi` IS NO LONGER MATERIALISED. Its `synth` reservation stays for the same reason
            // `absx`/`amax`/`amaxfl`'s do above: `BundleLayout::synth` is a bump allocator, so
            // dropping entries would shift every later tensor and confound a coherence comparison
            // with the baseline.
            let cl_h = rb(
                &cl,
                RowCount::of_token_rows(m),
                BlockCols::of_feature_cols(cl_reach),
            );
            ops.push(pw1(
                &opn("fq_clip_op"),
                "clip",
                m_rows,
                k_cols,
                In::full(&sc_h),
                &cl_h,
                sym_id_base,
                layout,
            ));
            // ⭐ THE PAD WINDOW'S ZERO FILL. `cl`'s logical `[m, k]` window is written by the
            // clamp above; the `[k, cl_reach)` window — the LAST pair's convert overhang,
            // `inn_pad − inn` lanes — is written HERE, unconditionally, every step — a `sub`
            // of the 448 const with itself is a broadcast ZERO (no new const: the fp8 consts
            // are width-independent `[1,64]` scalars). `In::scalar` broadcasts both axes, so
            // the op's reads stay in the const's own footprint while the write covers `[m, 64]`
            // at `col_of(m, cl_reach, k, Fp16)` — the RowBlocked stick-group law. The door's
            // own 64-stick input guard keeps `inn` — hence `k = pairs·inn` — a whole fp16
            // stick, so the window starts at a stick boundary BY ARITHMETIC, and the overhang
            // is one stick exactly (`fp8_k_pad` adds 0 or 64 at a whole-64 `inn`), which is
            // why ONE one-stick op covers it. The NON-last pairs' pad lanes never read this
            // fill — their converts' `[p·inn, p·inn + inn_pad)` windows borrow the NEXT pair's
            // clamped lanes (finite, and inert × the packed weight's host-zero-filled pad K
            // rows); only the LAST pair's window reaches past `k` into the fill.
            if cl_reach != k {
                let zf_h = rb(
                    &cl,
                    RowCount::of_token_rows(m),
                    BlockCols::of_feature_cols(cl_reach),
                );
                ops.push(assemble_pointwise_broadcast_off(
                    PointwiseFacts {
                        op_name: &opn("fq_zfpad_op"),
                        op_func: "sub",
                        rows: m_rows,
                        // ⭐ ONE STICK WIDE (the pad width, not `cl_reach`): the write is aimed at the
                        // pad WINDOW via the output offset, so the op's own width is the window's
                        // 64 elements — a `[m, cl_reach]` op would rewrite the logical window too
                        // (reading `sc`'s unpad-ded bytes at the pad position).
                        cols: crate::sdsc_abstract::BlockCols::of_one_stick(
                            crate::sdsc_abstract::Lanes::FP16,
                        ),
                        inputs: &[In::scalar(&pos448).ew(), In::scalar(&pos448).ew()],
                        o: &zf_h,
                        // The RowBlocked stick-group law aims the write at the pad window: stick-
                        // group `k/64` of every row (whole — see the fill comment above).
                        out_offset: crate::addr::col_of(m, cl_reach, k, Df::Fp16).into(),
                    },
                    sym_id_base,
                    layout,
                ));
            }
            // qfp8ch: f16 clamped act → SEN143_FP8. A dtype CONVERT (input 64-stick, output 128-stick), so it
            // goes through `assemble_convert` (two `primaryDsInfo_`), NOT a same-stick pointwise. The output's
            // `Df::Fp8` is intrinsic to the op (`convert_dtypes`). ⭐ AT THE PADDED WIDTH: the convert's own
            // `assert_df_stick_multiple` (128-elem SEN143_FP8 law) is the guard that refused 2112 — the
            // pad exists so THIS op sees a whole stick count — and its input reads the zero-filled `cl`.
            if pairs == 1 || inn_pad == inn {
                // ONE whole-tensor convert — the padded layout IS the logical one: an aligned
                // `inn` (every existing model's geometry, and the expert door's gate/up at
                // inn=2816) or the dense `pairs == 1` law (`cl_reach == k_pad`). Byte-identical
                // to the pre-pair emission: same op, same name, same `[m, k_pad]` windows.
                ops.push(assemble_convert(
                    &opn("fq_afp8_op"),
                    "qfp8ch",
                    m,
                    k_pad,
                    &rb(
                        &cl,
                        RowCount::of_token_rows(m),
                        BlockCols::of_feature_cols(k_pad),
                    ),
                    &rb_df(
                        &afp8,
                        RowCount::of_token_rows(m),
                        BlockCols::of_feature_cols(k_pad),
                        Df::Fp8,
                    ),
                    sym_id_base,
                    layout,
                ));
            } else {
                // PER SLOT — the half-stick `inn` case (the 26b DOWN door's 704): slot `s`'s
                // convert reads `cl` at column `s·inn` (an fp16 stick boundary — `inn` is a
                // whole 64-stick) and writes `afp8` at column `s·inn_pad` (an fp8 stick
                // boundary — the whole reason the padded layout exists). ⭐ THE ARMS SPLIT ON
                // `m` (design §4.5): at m=1 the offsets are FLAT stick-axis slices — the
                // decode form, byte-identical, because `col_of` degenerates to the flat law
                // at one row; at m>1 the SAME corners go through [`assemble_convert_col`] as
                // [`crate::addr::col_of`] DevOffs, which carry the `(c/lanes)·(m·lanes)`
                // group-corner term of the stick-major layout — exactly where the expert
                // door's per-(slot, chunk) matmul A-windows read `afp8`
                // (`col_of(rows, k·inn_pad, s·inn_pad)` — one law at both ends of the seam).
                // The pad lanes stay inert by the pair law: a non-last slot's overhang
                // borrows the NEXT slot's clamped lanes (finite, × the packed weight's
                // host-zero-filled pad K rows), and the LAST slot's overhang reads the
                // `[k, cl_reach)` zero fill. Slot 0 keeps the shipped `fq_afp8_op` name so
                // an already-quantized activation's splice-free reuse (and every pinned
                // test) is unchanged.
                for s in 0..pairs {
                    let name = if s == 0 {
                        opn("fq_afp8_op")
                    } else {
                        opn(&format!("fq_afp8_p{s}_op"))
                    };
                    // ⭐ ONE DOOR FOR EVERY `m`. The m=1 arm used to route to a raw-`u32` flat-offset
                    // twin (`assemble_convert_off`, now deleted); at one row a `col_of` corner IS the
                    // flat stick-axis slice exactly — the `(c/lanes)·(m·lanes)` group-corner term is
                    // `m·c` and `m == 1` — so the decode arm emits the SAME offsets through the
                    // DevOff twin and its mint. THE HANDLES ARE TRUED TO THE FULL BUFFERS (`cl_reach`,
                    // `k_pad`) and the fp8 side carries its OWN df (`rb_df`): the mint's placement
                    // door prices bytes at the handle's word length, so the fp16 annotation the
                    // piece-wise handles used to carry would overstate every fp8 window 2× and refuse
                    // the last slot — whose window ends exactly at `cl_reach = (pairs−1)·inn +
                    // inn_pad` / `k_pad = pairs·inn_pad` BY CONSTRUCTION (the zero fill the last
                    // slot's overhang reads is the `[k, cl_reach)` fill the clamp stage wrote).
                    ops.push(assemble_convert_col(
                        &name,
                        "qfp8ch",
                        m,
                        inn_pad,
                        &rb(
                            &cl,
                            RowCount::of_token_rows(m),
                            BlockCols::of_feature_cols(cl_reach),
                        ),
                        crate::addr::col_of(m, cl_reach, s * inn, Df::Fp16),
                        &rb_df(
                            &afp8,
                            RowCount::of_token_rows(m),
                            BlockCols::of_feature_cols(k_pad),
                            Df::Fp8,
                        ),
                        crate::addr::col_of(m, k_pad, s * inn_pad, Df::Fp8),
                        sym_id_base,
                        layout,
                    ));
                }
            }
        } // end `if !already_quantized` — the shared activation-quantize chain (per distinct activation).
        ops
    };
    Ok(ops)
}

/// ⭐⭐⭐ THE PROVEN fp8 W8A8 MATMUL DESCRIPTORS, dense form — the quantize chain (via
/// [`fp8_quantize_descriptors`] at `pairs == 1`, the dense law verbatim) plus the dense
/// body: `matmulfp8` binding the WHOLE packed weight as one `[k_pad, n]` view, then the
/// two dequant legs. This is the single-tile arm of [`matmul_fp8_descriptors`] — the plan's
/// `tile_count() == 1 ∧ total_rows() == m`, i.e. every decode (m=1) and every narrow
/// single-tile prefill — byte-identical to the pre-split emission (same ops, same names,
/// same reservations).
pub(crate) fn matmul_fp8_dense_descriptors(
    facts: &Fp8Facts,
    m: u32,
    k: u32,
    n: u32,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
    quantized: &mut std::collections::HashSet<String>,
) -> Result<Vec<EmittedOp>, SuperDscError> {
    let mut ops = fp8_quantize_descriptors(
        &Fp8QuantSource::activation(facts, m),
        k,
        1,
        sym_id_base,
        layout,
        quantized,
    )?;
    // ⭐ THE 26b K-PAD (dense form) — the contraction width the packed weight was staged at.
    let k_pad = crate::work::fp8_k_pad(k);
    let m_rows = crate::sdsc_abstract::RowCount::of_token_rows(m);
    let n_cols = crate::sdsc_abstract::BlockCols::of_feature_cols(n);
    let out = crate::place::act_name(facts.out_tid);
    use SynthRole as R;
    let a_id = PlaceId::Act(facts.a_tid);
    let out_id = PlaceId::Act(facts.out_tid);
    let w_name = facts.w_name.clone(); // fp8 weight `t{id}` — declared fp8 on the matmul operand (set_df), not the name
    let ws_name = facts.ws_name.clone(); // w_scale, read [1,n] (byte-identical to the loaded [n,1])
    // OP names — a DIFFERENT namespace from tensor names (a program's ops are not placed).
    let opn = |s: &str| format!("{out}_{s}");
    // `afp8`/`ascale` resolve the SAME synth names the chain reserved (`syn` is idempotent per
    // id) — the already-quantized reuse path reads them back by name, exactly as before the
    // split; PER-MATMUL tensors (`raw`, `dqa`) derive from the OUTPUT's id.
    let afp8 = syn(layout, a_id.synth(R::FqAfp8));
    let ascale = syn(layout, a_id.synth(R::FqAscale));
    let (raw, dqa) = (
        syn(layout, out_id.synth(R::FqRaw)),
        syn(layout, out_id.synth(R::FqDqA)),
    );
    if let Some(l) = layout {
        for r in [R::FqRaw, R::FqDqA] {
            l.synth(out_id.synth(r), &[m, n]);
        }
    }
    // matmulfp8: raw = act_fp8 @ W_fp8 → fp16. Build the plain matmul opspec, then declare both operands
    // (A = quantized act, W = kernel) fp8 via `set_df` — that typed marker is what selects the `matmulfp8`
    // opFunc + the 128-stick / 1-byte operand residency. The output `raw` stays fp16 (the DDL `%ptsum_fp`).
    ops.push({
        // fp8 matmul GEOMETRY = fp16 (`::<Fp16>`): the OUTPUT is fp16 and the KERNEL's N-stick is 64,
        // so the `out` (N) work-division splits on the 64-stick — fp8-ness does NOT change the N-split.
        // (128 is ONLY the fp8 ACTIVATION's K-stick, which is the reduction axis — never split.) Then
        // stamp the activation + weight operands `Df::Fp8` (the ½-HBM 1-byte residency); `emit_sdsc`
        // detects the fp8 matmul and emits the 2-D PACKED KERNEL stick `[in:2, out:64]` + the
        // SEN143_FP8 compute marker (the DDL `batchmatmulfp8` operand contract). The OUTPUT stays fp16.
        // `Df::Fp8` twice, deliberately: `operand_df` is what A/W COST in LX and is the only one
        // available early enough to reach the work-division and time-tiling decision (both run
        // inside the opspec builder); the `set_df` loop below is what they ARE on the wire.
        let mut spec = crate::ir::bridge::tiled_op_sdsc_op::matmul_opspec_off_operands::<Fp16>(
            crate::sdsc_abstract::MatM::of_token_rows(m),
            crate::sdsc_abstract::MatN::of_out_features(n),
            // ⭐ `k_pad`, NOT `k`: the packed fp8 weight is staged with `fp8_k_pad` K rows
            // (the retile + the worker's staging both take it from the same rule), so the
            // contraction's K must match the staged weight or the tile walks past the
            // buffer. The activation operand is `[m, k_pad]` — the quantize chain's own
            // widened tail. No-op at every 128-aligned k.
            crate::sdsc_abstract::MatK::of_in_features(k_pad),
            crate::sdsc_abstract::MatY::unbatched(),
            crate::ir::bridge::tiled_op_sdsc_op::SharedKernelBmmForm::batch_inner_proven(
                bmm_site::TapeLoweringSite::witness(),
            ),
            &afp8,
            &w_name,
            &raw,
            0,
            0,
            0,
            Df::Fp8,
        )
        .map_err(SuperDscError)?;
        for arg in &mut spec.args {
            if !matches!(arg.view().role, Role::Output) {
                arg.set_df(Df::Fp8);
            }
        }
        let folds = SdscFoldSet::new(spec.iter.cores_used());
        emit_sdsc_tiled(&opn("fq_mm"), &spec, &folds, sym_id_base, layout)?
    });
    // dequant: out = raw · a_scale (per-row) · w_scale[n] (per-col). `ascale` is [m,stk] (one scale per
    // query row) read via `col` (out-broadcast over N, row r→row r) — per-row-correct for m>1,
    // byte-identical at decode m=1. `w_scale` is [1,n] per-CHANNEL; for m>1 it broadcasts over the m
    // query rows (`mb`, mirroring rmsnorm's gamma multiply) — a plain `full` would demand m distinct
    // rows from the 1-row w_scale.
    let ascale_h = rb(
        &ascale,
        RowCount::of_token_rows(m),
        BlockCols::of_one_stick(Lanes::FP16),
    );
    let raw_h = rb(&raw, m_rows, n_cols);
    let dqa_h = rb(&dqa, m_rows, n_cols);
    ops.push(pw2(
        &opn("fq_dqa_op"),
        "mul",
        m_rows,
        n_cols,
        In::full(&raw_h),
        In::col(&ascale_h),
        &dqa_h,
        sym_id_base,
        layout,
    ));
    let out_h = rb(&out, m_rows, n_cols);
    let ws_h = rb(&ws_name, RowCount::of_broadcast_row(), n_cols);
    let w_scale_in = if m > 1 {
        In::mb(&ws_h)
    } else {
        In::full(&ws_h)
    };
    ops.push(pw2(
        &opn("fq_dqw_op"),
        "mul",
        m_rows,
        n_cols,
        In::full(&dqa_h),
        w_scale_in,
        &out_h,
        sym_id_base,
        layout,
    ));
    Ok(ops)
}

/// ⭐⭐⭐ THE fp8 MATMUL DOOR — ONE entry, TWO laws, and the PLAN picks. `work::Fp8GemmRowTiles`
/// is the copied `_m_tiles` law (`fp8_linear_kernel.py:70`): a WIDE geometry (max(k,n) ≥ 4096 or
/// an N the target-ABI set decomposes into more than one tile) rows-tiles the prefill into
/// target-ABI tiles PADDED UP with zero rows, M=1 stays whole at decode, and a narrow geometry
/// runs its rows whole. This door is the ONLY spelling of that choice — the dense arm
/// ([`matmul_fp8_dense_descriptors`], byte-identical to the pre-split emission at every
/// single-tile geometry) and the row-tiled arm ([`matmul_fp8_row_tiled_descriptors`], the
/// reference's clone-then-quantize-per-tile) are both behind it, so no caller can reach one
/// by a second route and the two can never disagree about which law a geometry takes.
///
/// ⛔ THE TILED ARM IS THE DEFECT-3 FIX. The dense arm's shared quant staging writes the
/// activation's fp8 at one walk ({mb:21} — the whole tensor) while its mm read the SAME bytes
/// at another ({in:4|8, mb:1, out:8|4} — the packed-kernel walk), and at m>1 those two
/// spellings of the mb axis tear the rows. The tiled arm never shares that staging: each tile
/// quantizes its OWN fresh `[ROWS, k]` clone, and the mm reads that clone through
/// [`Fp8TileStaging::<ROWS>`] — the row class is a const generic on the type, so a
/// writer/reader disagreement is a `cargo build` type error, never a torn read on card.
pub(crate) fn matmul_fp8_descriptors(
    facts: &Fp8Facts,
    m: u32,
    k: u32,
    n: u32,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
    quantized: &mut std::collections::HashSet<String>,
) -> Result<Vec<EmittedOp>, SuperDscError> {
    let plan = crate::work::Fp8GemmRowTiles::of(
        RowCount::of_token_rows(m),
        BlockCols::of_feature_cols(k),
        BlockCols::of_feature_cols(n),
    )
    .map_err(SuperDscError)?;
    if plan.tile_count() == crate::work::Fp8GemmTileCount::new(1)
        && plan.total_rows() == RowCount::of_token_rows(m)
        && (m == 1 || !plan.is_wide())
    {
        return matmul_fp8_dense_descriptors(facts, m, k, n, sym_id_base, layout, quantized);
    }
    // ⭐ THE DISPATCH IS EXHAUSTIVE OVER THE CLASS — exactly two tiled row classes exist,
    // the enum says so, and a third is a COMPILE ERROR at this match (no catch-all arm, no
    // dead-looking runtime refusal). Each arm states its own row extent: the wide tile's
    // [`FP8_GEMM_WIDE_M`] is our whole-tile pick over the reference's `_m_tiles` 4
    // (fp8_linear_kernel.py:76-77 — see the const's doc for the trip-bill law), and the
    // narrow part's padded 128-multiple is a fact of the input carried as a `RowCount`
    // (S3, :343-346 — one part at ANY multiple, no refusal on M).
    match plan.row_class() {
        crate::work::Fp8RowClass::Wide => Fp8TileEmit::<{ crate::work::FP8_GEMM_WIDE_M }>::new(
            facts,
            &plan,
            sym_id_base,
            layout,
            quantized,
        )
        .emit(),
        crate::work::Fp8RowClass::Narrow(padded) => {
            Fp8NarrowEmit::new(facts, &plan, padded, sym_id_base, layout, quantized).emit()
        }
    }
}

/// One stick-group's byte advance in a `[rows, groups, 64]` fp16 nest — the group law
/// (`g·rows·64` elems) as bytes: the per-trip stride a time-folded fan op carries, minted
/// from the same footprint law the geometry guard compares against (`synth_footprint_bytes`).
fn nest_group_stride_bytes(rows: u32, stk: u32) -> i64 {
    crate::placement::synth_footprint_bytes(&[rows, stk], Df::Fp16) as i64
}

/// ⭐ ONE TILE'S `matmulfp8` — the dense body's own mm leg at the tile's row class: the SAME
/// `matmul_opspec_off_operands::<Fp16>` + fp8 `set_df` stamp + `emit_sdsc_tiled`, differing only
/// in that the A operand name comes from the TYPED staging (so the mm reads exactly the buffer
/// the tile's quant wrote, at the row class the type states) and the output lands in the tile's
/// own `[ROWS, n]` raw synth. `ROWS`-wide on purpose: the pad rows quantize to zero (the clone's
/// zero fill), so their raw rows are written zero — deterministic — and the out-splice copies
/// only the real rows anyway.
fn fp8_tile_mm_op<const ROWS: u32>(
    staging: &Fp8TileStaging<ROWS>,
    facts: &Fp8Facts,
    n: u32,
    k_pad: u32,
    t: u32,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> Result<EmittedOp, SuperDscError> {
    let out = crate::place::act_name(facts.out_tid);
    let out_id = PlaceId::Act(facts.out_tid);
    let raw = syn(layout, out_id.synth_tiled(SynthRole::FqRaw, t));
    if let Some(l) = layout {
        l.synth(out_id.synth_tiled(SynthRole::FqRaw, t), &[ROWS, n]);
    }
    let mut spec = crate::ir::bridge::tiled_op_sdsc_op::matmul_opspec_off_operands::<Fp16>(
        // ⭐ M FROM THE TYPE — `Fp8TileStaging::<ROWS>::ROWS`, never a second spelling of the
        // row count: the staging's class and the mm's M are one fact because they are one type.
        crate::sdsc_abstract::MatM::of_token_rows(Fp8TileStaging::<ROWS>::ROWS),
        crate::sdsc_abstract::MatN::of_out_features(n),
        crate::sdsc_abstract::MatK::of_in_features(k_pad),
        crate::sdsc_abstract::MatY::unbatched(),
        crate::ir::bridge::tiled_op_sdsc_op::SharedKernelBmmForm::batch_inner_proven(
            bmm_site::TapeLoweringSite::witness(),
        ),
        staging.afp8().name(),
        &facts.w_name,
        &raw,
        0,
        0,
        0,
        Df::Fp8,
    )
    .map_err(SuperDscError)?;
    for arg in &mut spec.args {
        if !matches!(arg.view().role, Role::Output) {
            arg.set_df(Df::Fp8);
        }
    }
    let folds = SdscFoldSet::new(spec.iter.cores_used());
    emit_sdsc_tiled(
        &format!("{out}_fq_mm_t{t}_op"),
        &spec,
        &folds,
        sym_id_base,
        layout,
    )
}

/// ⭐ THE NARROW PART'S `matmulfp8` — `fp8_tile_mm_op`'s runtime-rows twin: the SAME
/// `matmul_opspec_off_operands::<Fp16>` + fp8 `set_df` stamp + `emit_sdsc_tiled`, with
/// the A operand name from the narrow staging and M from the staging's OWN extent — the
/// padded 128-multiple is a fact of the input (S3, `fp8_linear_kernel.py:343-346`), a
/// `RowCount` on the staging, never a const generic and never re-derived here.
fn fp8_narrow_mm_op(
    staging: &Fp8NarrowStaging,
    facts: &Fp8Facts,
    n: u32,
    k_pad: u32,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> Result<EmittedOp, SuperDscError> {
    let rows = staging.rows().get();
    let out = crate::place::act_name(facts.out_tid);
    let out_id = PlaceId::Act(facts.out_tid);
    let raw = syn(layout, out_id.synth_tiled(SynthRole::FqRaw, 0));
    if let Some(l) = layout {
        l.synth(out_id.synth_tiled(SynthRole::FqRaw, 0), &[rows, n]);
    }
    let mut spec = crate::ir::bridge::tiled_op_sdsc_op::matmul_opspec_off_operands::<Fp16>(
        // ⭐ M FROM THE STAGING'S OWN EXTENT — the narrow law's padded 128-multiple,
        // carried by the staging type so the mm's M and the quant's row count are one
        // fact (the wide twin's `Fp8TileStaging::<ROWS>::ROWS`, at a runtime extent).
        crate::sdsc_abstract::MatM::of_token_rows(rows),
        crate::sdsc_abstract::MatN::of_out_features(n),
        crate::sdsc_abstract::MatK::of_in_features(k_pad),
        crate::sdsc_abstract::MatY::unbatched(),
        crate::ir::bridge::tiled_op_sdsc_op::SharedKernelBmmForm::batch_inner_proven(
            bmm_site::TapeLoweringSite::witness(),
        ),
        staging.afp8().name(),
        &facts.w_name,
        &raw,
        0,
        0,
        0,
        Df::Fp8,
    )
    .map_err(SuperDscError)?;
    for arg in &mut spec.args {
        if !matches!(arg.view().role, Role::Output) {
            arg.set_df(Df::Fp8);
        }
    }
    let folds = SdscFoldSet::new(spec.iter.cores_used());
    emit_sdsc_tiled(
        &format!("{out}_fq_mm_narrow_op"),
        &spec,
        &folds,
        sym_id_base,
        layout,
    )
}

/// ⭐⭐⭐ THE ROW-TILED LAW'S EMISSION CONTEXT — every fact the tiled arm reads, in ONE
/// typed argument, with the row class as the const generic: the mm's M, the staging's row
/// count, and the tile buffers' row extent are ONE fact by type (`Fp8TileStaging::<ROWS>`
/// refuses any other). Built by the door at the dispatch arm from the PLAN itself (m/k/n
/// are the plan's own facts; `k_pad` is minted HERE, once, as typed feature-cols — the
/// single spelling the tile mm and the staging handles consume), so no shape is passed
/// twice or spelled twice. What the context carries, in order: the KTIR-recovered
/// contraction facts; the row-tiling plan (the one source of tile windows and stick-group
/// extents); the activation's real row count; the contraction and output widths; the
/// contraction width at the packed-kernel pad; and the three emission threads.
///
/// ⭐ THE CLONES ARE IDENTITY MATMULS — the card-proven KV-cache-write form
/// (`assemble_matmul_placed`, `at_head` + `MatY::unbatched()`): one op per stick group,
/// `N = K = one stick` so the walk never crosses a group boundary, reading the
/// `[64,64]` identity table the tape's placement pass registers (`identity_tid_of_class`).
/// Pointwise outputs are column-windowed only (`OutCorner`), which is why the out-splice is
/// this form too: a matmul output is placed by a LAW (`OperandPlacement`), so the row window
/// a tile's real rows occupy is expressible here and nowhere else.
///
/// ⭐ SHARING: the tile clone/zero/quant legs are keyed by the TILE's own name in `quantized`,
/// so a second GEMM over the same activation at the same plan (K/V after Q, Up after Gate)
/// reuses one staging instead of re-quantizing per matmul — the dense law's own sharing,
/// at tile grain. The reservation carries a GEOMETRY GUARD: `synth` is idempotent per name,
/// so a second GEMM that re-tiles the same activation at a DIFFERENT row class would
/// silently reuse the wrong-shaped buffer — the guard compares the reserved footprint
/// against this plan's and refuses loudly instead.
struct Fp8TileEmit<'a, const ROWS: u32> {
    /// The KTIR-recovered contraction facts.
    facts: &'a Fp8Facts,
    /// The row-tiling plan — the one source of tile windows and stick-group extents.
    plan: &'a crate::work::Fp8GemmRowTiles,
    /// The activation's real row count (the source nest's rows).
    m: RowCount,
    /// The contraction width.
    k: BlockCols,
    /// The output width.
    n: BlockCols,
    /// The contraction width at the packed-kernel pad — minted ONCE, here.
    k_pad: BlockCols,
    sym_id_base: &'a mut i64,
    layout: Option<&'a BundleLayout>,
    quantized: &'a mut std::collections::HashSet<String>,
}

impl<'a, const ROWS: u32> Fp8TileEmit<'a, ROWS> {
    /// The door's arm mints the context from the plan itself — no shape argument exists to
    /// disagree with the plan, and the row class is the const generic on the type.
    fn new(
        facts: &'a Fp8Facts,
        plan: &'a crate::work::Fp8GemmRowTiles,
        sym_id_base: &'a mut i64,
        layout: Option<&'a BundleLayout>,
        quantized: &'a mut std::collections::HashSet<String>,
    ) -> Self {
        Self {
            facts,
            plan,
            m: plan.real_rows(),
            k: plan.contraction_width(),
            n: plan.out_width(),
            k_pad: BlockCols::of_feature_cols(crate::work::fp8_k_pad(
                plan.contraction_width().get(),
            )),
            sym_id_base,
            layout,
            quantized,
        }
    }

    /// The whole tiled arm — the reference's wide-GEMM form, copied not invented
    /// (`fp8_linear_kernel.py`): per tile, zero-fill the pad rows (:32), `clone()` the real
    /// rows into the fresh `[ROWS, k]` buffer (:351), quantize THE CLONE (:33), contract at
    /// the tile's row class, dequant into a tile-local landing, splice the REAL rows back.
    fn emit(self) -> Result<Vec<EmittedOp>, SuperDscError> {
        use SynthRole as R;
        let Self {
            facts,
            plan,
            m,
            k,
            n,
            k_pad,
            sym_id_base,
            layout,
            quantized,
        } = self;
        // ⚠ F2/F3/F4 ROUND re-types these seams (source struct carries the minted tile;
        // MatM takes RowCount; tiles()/stick-group windows iterate minted values).
        let (m, k, n, k_pad) = (m.get(), k.get(), n.get(), k_pad.get());
        let stk = Fp16::ELEMS_PER_STICK; // 64
        let out = crate::place::act_name(facts.out_tid);
        let a_id = PlaceId::Act(facts.a_tid);
        let out_id = PlaceId::Act(facts.out_tid);
        let r_rows = RowCount::of_token_rows(ROWS);
        let k_cols = BlockCols::of_feature_cols(k);
        let n_cols = BlockCols::of_feature_cols(n);
        let slabs = Slabs::new(1);
        // ⭐ THE IDENTITY TABLE — the same [`stk`,`stk`] registry entry the card-proven KV-cache
        // write's clone reads. A tape that registers no hd-`stk` class cannot tile rows: the
        // refusal names the table, never a fallback copy form (there is none — pointwise outputs
        // are column-windowed, so the identity matmul is the only placed-row copy this compiler
        // has).
        let ident = layout
            .and_then(|l| l.identity_tid_of_class(stk))
            .ok_or_else(|| {
                SuperDscError(format!(
                    "fp8 matmul {out}: the row-tiled law's clone reads the [{stk},{stk}] identity \
                 table, and this tape registers no hd-{stk} class — a tape without the table \
                 cannot tile rows"
                ))
            })?;
        let ident_h = Stk::<KernelTag>::one_stick_identity(crate::place::act_name(ident));
        let pos448 = rbo(&crate::place::act_name(FP8_POS448_TID));
        // ⭐ THE NESTS SPELL THE ROWBLOCKED STICK-GROUP LAW — stick group `g` of every row of a
        // `[rows, cols]` buffer sits at `g·rows·64`, rows 64 apart within the group. A rank-3
        // `[row, head, feat]` nest over `[rows, groups, 64]` reproduces it exactly
        // (`off([r,g,f]) = g·(rows·64) + r·64 + f`), which is what lets ONE `at_head` placement
        // aim a clone at (row, group) on both sides without any raw address arithmetic.
        let k_groups = plan.k_stick_groups().get();
        let n_groups = plan.n_stick_groups().get();
        let src_nest = Nest::new(&["row", "head", "feat"], &[m, k_groups, stk], Df::Fp16);
        let tile_nest = Nest::new(&["row", "head", "feat"], &[ROWS, k_groups, stk], Df::Fp16);
        let landing_nest = Nest::new(&["row", "head", "feat"], &[ROWS, n_groups, stk], Df::Fp16);
        let out_nest = Nest::new(&["row", "head", "feat"], &[m, n_groups, stk], Df::Fp16);
        let a_h = rb(&facts.a_name, RowCount::of_token_rows(m), k_cols);
        let out_h = rb(&out, RowCount::of_token_rows(m), n_cols);
        let ws_h = rb(&facts.ws_name, RowCount::of_broadcast_row(), n_cols);
        let mut ops: Vec<EmittedOp> = Vec::new();
        for t in 0..plan.tile_count().get() {
            let tile = plan.tile(crate::work::Fp8GemmTileIdx::new(t));
            let real = tile.real_rows.get();
            let tile_id = a_id.synth_tiled(R::FqTileRow, t);
            let tile_name = syn(layout, tile_id);
            // ⭐ THE GEOMETRY GUARD — `synth` is idempotent per name, so a second GEMM over the
            // same activation re-resolves the SAME tile names; if it does so at a different row
            // class (a different plan for the same activation), the footprints disagree and the
            // reuse is refused here, at bake, rather than silently reading the wrong-shaped clone.
            // ⭐ THE ARM IS THE DISCRIMINATOR, not only the bytes: at `FP8_GEMM_WIDE_M` == the
            // narrow arm's 128 alignment a wide tile and a narrow part reserve the SAME
            // footprint, so the byte-compare alone goes blind exactly there — converged
            // footprints do not make the staging walks agree, they hide the disagreement
            // (defect 3's class). The arm stamp (the same reservation call, one door) refuses
            // it first.
            if let Some(l) = layout {
                let want = crate::placement::synth_footprint_bytes(&[ROWS, k], Df::Fp16);
                if let Some(have) = l.synth.borrow().sizes.get(&tile_name).copied()
                    && have != want
                {
                    return Err(SuperDscError(format!(
                        "fp8 matmul {out}: tile {t} of activation t{} was already reserved at {have} bytes but \
                         this plan needs {want} — two GEMMs over one activation at different row \
                         classes is a plan disagreement; it is not a resize",
                        facts.a_tid
                    )));
                }
                if l.synth.borrow().fp8_row_arms.get(&tile_name).copied() == Some(false) {
                    return Err(SuperDscError(format!(
                        "fp8 matmul {out}: tile {t} of activation t{} was already reserved by the \
                         NARROW row class — this plan takes the WIDE tile at the same 128-row \
                         footprint, and converged footprints do not make the staging walks agree \
                         (defect 3's class); it is a plan disagreement, not a resize",
                        facts.a_tid
                    )));
                }
                l.synth_fp8_row_tile(tile_id, &[ROWS, k], true);
            }
            let tile_h = rb(&tile_name, r_rows, k_cols);
            if !quantized.contains(&tile_name) {
                // (1) THE PAD ROWS' ZERO FILL — a broadcast zero (`sub(448,448)`, the quant chain's
                // own pad-window law) over the WHOLE tile buffer, so the rows the clone below does
                // not reach (the pad rows, `ROWS − real` of them on the last tile) are written zero
                // exactly as the reference's zero-padded staging writes them — never-written bytes
                // are card nondeterminism, and a quantized zero row is the clipMin-floor scale over
                // a zero activation, the value the reference's own pad rows carry.
                ops.push(assemble_pointwise_broadcast_off(
                    PointwiseFacts {
                        op_name: &format!("{out}_fq_tz{t}_op"),
                        op_func: "sub",
                        rows: r_rows,
                        cols: k_cols,
                        inputs: &[In::scalar(&pos448).ew(), In::scalar(&pos448).ew()],
                        o: &tile_h,
                        out_offset: crate::addr::OutCorner::ZERO,
                    },
                    sym_id_base,
                    layout,
                ));
                // (2) THE CLONE — ONE identity matmul, TIME-FOLDED over the K stick groups: the
                // g=0 op at full shape, `time = k_groups`, each fan tensor stepping by its nest's
                // head stride (`g·rows·64` elems — the group law the nests spell), the identity
                // table FIXED across trips (empty stride map — `concrete_trips`' no-bump path).
                // `concrete_trips` lands each trip at exactly the address the per-g ops
                // carried, so the render's trips are the same ops at the same windows: the fan
                // is the fold, not new work. (`M = real`; the source has no rows past `m`, so a
                // full-`ROWS` read would sweep out of footprint. The pad rows keep the zero
                // fill.)
                let mut tc = crate::ir::bridge::tiled_op_sdsc_op::assemble_matmul_placed(
                    &format!("{out}_fq_tc{t}_op"),
                    crate::sdsc_abstract::MatM::of_token_rows(real),
                    crate::sdsc_abstract::MatN::one_stick(Lanes::FP16),
                    crate::sdsc_abstract::MatK::one_stick(Lanes::FP16),
                    crate::sdsc_abstract::MatY::unbatched(),
                    crate::ir::bridge::tiled_op_sdsc_op::SharedKernelBmmForm::batch_inner_proven(
                        bmm_site::TapeLoweringSite::witness(),
                    ),
                    &a_h,
                    OperandPlacement::at_head(&src_nest, tile.row, 0, 0, slabs),
                    &ident_h,
                    DevOff::ZERO,
                    &tile_h,
                    OperandPlacement::at_head(&tile_nest, Idx::<Row>::n(0), 0, 0, slabs),
                    sym_id_base,
                    layout,
                );
                tc.time = k_groups;
                tc.affine_strides = vec![
                    BTreeMap::from([("out", nest_group_stride_bytes(m, stk))]),
                    BTreeMap::new(),
                    BTreeMap::from([("out", nest_group_stride_bytes(ROWS, stk))]),
                ];
                ops.push(tc);
                // (3) THE QUANT, OVER THE CLONE — the reference quantizes AFTER the clone
                // (:351 then :33), and so does the law: the chain's source is the tile's fresh
                // buffer, its tensors carry the tile index, and its ops carry it in their names.
                ops.extend(fp8_quantize_descriptors(
                    &Fp8QuantSource {
                        name: tile_name.clone(),
                        id: a_id,
                        rows: ROWS,
                        tile: Some(t),
                        out_tid: facts.out_tid,
                    },
                    k,
                    1,
                    sym_id_base,
                    layout,
                    quantized,
                )?);
            }
            // (4) THE TILE'S mm — A comes from the TYPED staging: the row class is on the type, so
            // the mm's M and the quant's row count are one fact, and a handle at another class is
            // refused at `staged`.
            let staging = Fp8TileStaging::<ROWS>::staged(
                rb_df(
                    &syn(layout, a_id.synth_tiled(R::FqAfp8, t)),
                    r_rows,
                    BlockCols::of_feature_cols(k_pad),
                    Df::Fp8,
                ),
                rb(
                    &syn(layout, a_id.synth_tiled(R::FqAscale, t)),
                    r_rows,
                    BlockCols::of_one_stick(Lanes::FP16),
                ),
            )
            .map_err(SuperDscError)?;
            ops.push(fp8_tile_mm_op::<ROWS>(
                &staging,
                facts,
                n,
                k_pad,
                t,
                sym_id_base,
                layout,
            )?);
            // (5) THE DEQUANT LEGS — into a TILE-LOCAL landing buffer, not the output: pointwise
            // outputs are column-windowed only, so a tile's `[ROWS, n]` rows cannot land directly
            // at `tile.row` of the output; they land in their own buffer and the out-splice below
            // copies the real rows over. `ROWS`-wide (the mm wrote every row; the pad rows are
            // zero through the zero scale path), and `In::mb` for the w_scale broadcast — `ROWS >
            // 1` by this arm's own law (the plan only tiles at multi-row prefill).
            let raw = syn(layout, out_id.synth_tiled(R::FqRaw, t));
            let dqa = syn(layout, out_id.synth_tiled(R::FqDqA, t));
            let landing = syn(layout, out_id.synth_tiled(R::FqTileRow, t));
            if let Some(l) = layout {
                l.synth(out_id.synth_tiled(R::FqDqA, t), &[ROWS, n]);
                l.synth(out_id.synth_tiled(R::FqTileRow, t), &[ROWS, n]);
            }
            let raw_h = rb(&raw, r_rows, n_cols);
            let dqa_h = rb(&dqa, r_rows, n_cols);
            let landing_h = rb(&landing, r_rows, n_cols);
            ops.push(pw2(
                &format!("{out}_fq_dqa_t{t}_op"),
                "mul",
                r_rows,
                n_cols,
                In::full(&raw_h),
                In::col(staging.ascale()),
                &dqa_h,
                sym_id_base,
                layout,
            ));
            ops.push(pw2(
                &format!("{out}_fq_dqw_t{t}_op"),
                "mul",
                r_rows,
                n_cols,
                In::full(&dqa_h),
                In::mb(&ws_h),
                &landing_h,
                sym_id_base,
                layout,
            ));
            // (6) THE OUT-SPLICE — the clone form again, N-side, TIME-FOLDED over the output
            // stick groups: the c=0 op, `time = n_groups`, the landing side stepping by its
            // ROWS-row group stride and the output side by its m-row one — the tile's real rows
            // of each output stick group land at `tile.row` of the output. The union of the
            // tiles' real rows covers `[0, m)` exactly (the plan's own law), so every output
            // byte is written — one (tile, stick) trip at a time, as before.
            let mut to = crate::ir::bridge::tiled_op_sdsc_op::assemble_matmul_placed(
                &format!("{out}_fq_to{t}_op"),
                crate::sdsc_abstract::MatM::of_token_rows(real),
                crate::sdsc_abstract::MatN::one_stick(Lanes::FP16),
                crate::sdsc_abstract::MatK::one_stick(Lanes::FP16),
                crate::sdsc_abstract::MatY::unbatched(),
                crate::ir::bridge::tiled_op_sdsc_op::SharedKernelBmmForm::batch_inner_proven(
                    bmm_site::TapeLoweringSite::witness(),
                ),
                &landing_h,
                OperandPlacement::at_head(&landing_nest, Idx::<Row>::n(0), 0, 0, slabs),
                &ident_h,
                DevOff::ZERO,
                &out_h,
                OperandPlacement::at_head(&out_nest, tile.row, 0, 0, slabs),
                sym_id_base,
                layout,
            );
            to.time = n_groups;
            to.affine_strides = vec![
                BTreeMap::from([("out", nest_group_stride_bytes(ROWS, stk))]),
                BTreeMap::new(),
                BTreeMap::from([("out", nest_group_stride_bytes(m, stk))]),
            ];
            ops.push(to);
        }
        Ok(ops)
    }
}

/// ⭐⭐⭐ THE NARROW ARM'S EMISSION CONTEXT — `Fp8TileEmit`'s runtime-rows twin, for the
/// reference's narrow law (S3, `fp8_linear_kernel.py:343-346`): ONE part at the FULL
/// padded 128-multiple, ANY multiple (21→128, 192→256), no upper bound. The extent is a
/// runtime `RowCount` — a fact of the input, not of the ABI — so the context carries it
/// where the wide context carries a const generic (`Fp8TileStaging::<ROWS>` cannot state
/// it). Every leg is the wide context's own law at the part's extent: zero-fill the pad
/// rows, clone the real rows, quantize the clone, contract at the staged extent, dequant
/// into a local landing, splice the REAL rows back — S5's `out[:orig_m]` (`:362-367`).
/// Sharing and the geometry guard are the wide context's, unchanged: keyed by the part's
/// own name in `quantized`, footprint-compared at `synth`.
struct Fp8NarrowEmit<'a> {
    /// The KTIR-recovered contraction facts.
    facts: &'a Fp8Facts,
    /// The row-tiling plan — the one source of stick-group extents.
    plan: &'a crate::work::Fp8GemmRowTiles,
    /// The ONE part's row extent — the padded 128-multiple (S3).
    rows: RowCount,
    /// The activation's real row count (the source nest's rows, the splice's extent).
    m: RowCount,
    /// The contraction width.
    k: BlockCols,
    /// The output width.
    n: BlockCols,
    /// The contraction width at the packed-kernel pad — minted ONCE, here.
    k_pad: BlockCols,
    sym_id_base: &'a mut i64,
    layout: Option<&'a BundleLayout>,
    quantized: &'a mut std::collections::HashSet<String>,
}

impl<'a> Fp8NarrowEmit<'a> {
    /// The door's arm mints the context from the plan and the class's own extent — no
    /// shape argument exists to disagree with the plan.
    fn new(
        facts: &'a Fp8Facts,
        plan: &'a crate::work::Fp8GemmRowTiles,
        rows: RowCount,
        sym_id_base: &'a mut i64,
        layout: Option<&'a BundleLayout>,
        quantized: &'a mut std::collections::HashSet<String>,
    ) -> Self {
        Self {
            facts,
            plan,
            rows,
            m: plan.real_rows(),
            k: plan.contraction_width(),
            n: plan.out_width(),
            k_pad: BlockCols::of_feature_cols(crate::work::fp8_k_pad(
                plan.contraction_width().get(),
            )),
            sym_id_base,
            layout,
            quantized,
        }
    }

    /// The whole narrow arm — the reference's narrow form, copied not invented: pad the
    /// rows UP to the 128-multiple with zeros (`:344-345` + `_pad_m`'s materialized zero
    /// rows, `:266-271`), ONE part (`:346`), quantize the part, contract at its extent,
    /// dequant, splice the real rows back (`:362-367`).
    fn emit(self) -> Result<Vec<EmittedOp>, SuperDscError> {
        use SynthRole as R;
        let Self {
            facts,
            plan,
            m,
            rows,
            k,
            n,
            k_pad,
            sym_id_base,
            layout,
            quantized,
        } = self;
        let (m, rows, k, n, k_pad) = (m.get(), rows.get(), k.get(), n.get(), k_pad.get());
        let stk = Fp16::ELEMS_PER_STICK; // 64
        let out = crate::place::act_name(facts.out_tid);
        let a_id = PlaceId::Act(facts.a_tid);
        let out_id = PlaceId::Act(facts.out_tid);
        let r_rows = RowCount::of_token_rows(rows);
        let k_cols = BlockCols::of_feature_cols(k);
        let n_cols = BlockCols::of_feature_cols(n);
        let slabs = Slabs::new(1);
        // ⭐ THE IDENTITY TABLE — the same [`stk`,`stk`] registry entry the card-proven KV-cache
        // write's clone reads. A tape that registers no hd-`stk` class cannot tile rows: the
        // refusal names the table, never a fallback copy form (there is none — pointwise outputs
        // are column-windowed, so the identity matmul is the only placed-row copy this compiler
        // has).
        let ident = layout
            .and_then(|l| l.identity_tid_of_class(stk))
            .ok_or_else(|| {
                SuperDscError(format!(
                    "fp8 matmul {out}: the narrow law's clone reads the [{stk},{stk}] identity \
                 table, and this tape registers no hd-{stk} class — a tape without the table \
                 cannot run a padded part"
                ))
            })?;
        let ident_h = Stk::<KernelTag>::one_stick_identity(crate::place::act_name(ident));
        let pos448 = rbo(&crate::place::act_name(FP8_POS448_TID));
        // ⭐ THE NESTS SPELL THE ROWBLOCKED STICK-GROUP LAW — stick group `g` of every row of a
        // `[rows, cols]` buffer sits at `g·rows·64`, rows 64 apart within the group. A rank-3
        // `[row, head, feat]` nest over `[rows, groups, 64]` reproduces it exactly
        // (`off([r,g,f]) = g·(rows·64) + r·64 + f`), which is what lets ONE `at_head` placement
        // aim a clone at (row, group) on both sides without any raw address arithmetic.
        let k_groups = plan.k_stick_groups().get();
        let n_groups = plan.n_stick_groups().get();
        let src_nest = Nest::new(&["row", "head", "feat"], &[m, k_groups, stk], Df::Fp16);
        let part_nest = Nest::new(&["row", "head", "feat"], &[rows, k_groups, stk], Df::Fp16);
        let landing_nest = Nest::new(&["row", "head", "feat"], &[rows, n_groups, stk], Df::Fp16);
        let out_nest = Nest::new(&["row", "head", "feat"], &[m, n_groups, stk], Df::Fp16);
        let a_h = rb(&facts.a_name, RowCount::of_token_rows(m), k_cols);
        let out_h = rb(&out, RowCount::of_token_rows(m), n_cols);
        let ws_h = rb(&facts.ws_name, RowCount::of_broadcast_row(), n_cols);
        let mut ops: Vec<EmittedOp> = Vec::new();
        // THE ONE PART — tile index 0: the narrow law has exactly one row part (`:346`,
        // `m_parts = [padded]`), so the staging names carry t0 and the geometry guard pins
        // the part's footprint exactly as a wide tile's.
        let t = 0u32;
        let tile_id = a_id.synth_tiled(R::FqTileRow, t);
        let tile_name = syn(layout, tile_id);
        // ⭐ THE GEOMETRY GUARD — `synth` is idempotent per name, so a second GEMM over the
        // same activation re-resolves the SAME part names; if it does so at a different row
        // class (a different plan for the same activation), the footprints disagree and the
        // reuse is refused here, at bake, rather than silently reading the wrong-shaped part.
        // ⭐ THE ARM IS THE DISCRIMINATOR, not only the bytes: at `FP8_GEMM_WIDE_M` == the
        // narrow arm's 128 alignment a wide tile and a narrow part reserve the SAME
        // footprint, so the byte-compare alone goes blind exactly there — converged
        // footprints do not make the staging walks agree, they hide the disagreement
        // (defect 3's class). The arm stamp (the same reservation call, one door) refuses
        // it first.
        if let Some(l) = layout {
            let want = crate::placement::synth_footprint_bytes(&[rows, k], Df::Fp16);
            if let Some(have) = l.synth.borrow().sizes.get(&tile_name).copied()
                && have != want
            {
                return Err(SuperDscError(format!(
                    "fp8 matmul {out}: part t{t} of activation t{} was already reserved at {have} bytes but \
                     this plan needs {want} — two GEMMs over one activation at different row \
                     classes is a plan disagreement; it is not a resize",
                    facts.a_tid
                )));
            }
            if l.synth.borrow().fp8_row_arms.get(&tile_name).copied() == Some(true) {
                return Err(SuperDscError(format!(
                    "fp8 matmul {out}: part t{t} of activation t{} was already reserved by the \
                     WIDE row class — this plan takes the NARROW part at the same 128-row \
                     footprint, and converged footprints do not make the staging walks agree \
                     (defect 3's class); it is a plan disagreement, not a resize",
                    facts.a_tid
                )));
            }
            l.synth_fp8_row_tile(tile_id, &[rows, k], false);
        }
        let tile_h = rb(&tile_name, r_rows, k_cols);
        if !quantized.contains(&tile_name) {
            // (1) THE PAD ROWS' ZERO FILL — a broadcast zero (`sub(448,448)`, the quant chain's
            // own pad-window law) over the WHOLE part buffer, so the rows the clone below does
            // not reach (the pad rows, `rows − m` of them) are written zero exactly as the
            // reference's `_pad_m` materializes them (`:266-271`, `:344-345`) — never-written
            // bytes are card nondeterminism, and a quantized zero row is the clipMin-floor
            // scale over a zero activation, the value the reference's own pad rows carry.
            ops.push(assemble_pointwise_broadcast_off(
                PointwiseFacts {
                    op_name: &format!("{out}_fq_tz{t}_op"),
                    op_func: "sub",
                    rows: r_rows,
                    cols: k_cols,
                    inputs: &[In::scalar(&pos448).ew(), In::scalar(&pos448).ew()],
                    o: &tile_h,
                    out_offset: crate::addr::OutCorner::ZERO,
                },
                sym_id_base,
                layout,
            ));
            // (2) THE CLONE — one identity matmul per stick group, copying the part's REAL rows
            // (`M = m`; the source has no rows past `m`, so a full-`rows` read would sweep out
            // of footprint). The pad rows keep the zero fill.
            for g in 0..k_groups {
                ops.push(crate::ir::bridge::tiled_op_sdsc_op::assemble_matmul_placed(
                    &format!("{out}_fq_tc{t}g{g}_op"),
                    crate::sdsc_abstract::MatM::of_token_rows(m),
                    crate::sdsc_abstract::MatN::one_stick(Lanes::FP16),
                    crate::sdsc_abstract::MatK::one_stick(Lanes::FP16),
                    crate::sdsc_abstract::MatY::unbatched(),
                    crate::ir::bridge::tiled_op_sdsc_op::SharedKernelBmmForm::batch_inner_proven(
                        bmm_site::TapeLoweringSite::witness(),
                    ),
                    &a_h,
                    OperandPlacement::at_head(&src_nest, Idx::<Row>::n(0), g, 0, slabs),
                    &ident_h,
                    DevOff::ZERO,
                    &tile_h,
                    OperandPlacement::at_head(&part_nest, Idx::<Row>::n(0), g, 0, slabs),
                    sym_id_base,
                    layout,
                ));
            }
            // (3) THE QUANT, OVER THE PART — the reference quantizes the padded part whole
            // (`xi = x2d[i:i+mt].clone()` at `mt` = the padded extent, `:350-351`), pad rows
            // included: they quantize to zero, exactly the wide path's own pad law.
            ops.extend(fp8_quantize_descriptors(
                &Fp8QuantSource {
                    name: tile_name.clone(),
                    id: a_id,
                    rows,
                    tile: Some(t),
                    out_tid: facts.out_tid,
                },
                k,
                1,
                sym_id_base,
                layout,
                quantized,
            )?);
        }
        // (4) THE PART'S mm — A comes from the TYPED staging: the extent is on the staging,
        // so the mm's M and the quant's row count are one fact, and a handle at another
        // extent is refused at `staged`.
        let staging = Fp8NarrowStaging::staged(
            r_rows,
            rb_df(
                &syn(layout, a_id.synth_tiled(R::FqAfp8, t)),
                r_rows,
                BlockCols::of_feature_cols(k_pad),
                Df::Fp8,
            ),
            rb(
                &syn(layout, a_id.synth_tiled(R::FqAscale, t)),
                r_rows,
                BlockCols::of_one_stick(Lanes::FP16),
            ),
        )
        .map_err(SuperDscError)?;
        ops.push(fp8_narrow_mm_op(
            &staging,
            facts,
            n,
            k_pad,
            sym_id_base,
            layout,
        )?);
        // (5) THE DEQUANT LEGS — into a PART-LOCAL landing buffer, not the output: pointwise
        // outputs are column-windowed only, so the part's `[rows, n]` rows cannot land
        // directly in the output; they land in their own buffer and the out-splice below
        // copies the real rows over. `rows`-wide (the mm wrote every row; the pad rows are
        // zero through the zero scale path), and `In::mb` for the w_scale broadcast.
        let raw = syn(layout, out_id.synth_tiled(R::FqRaw, t));
        let dqa = syn(layout, out_id.synth_tiled(R::FqDqA, t));
        let landing = syn(layout, out_id.synth_tiled(R::FqTileRow, t));
        if let Some(l) = layout {
            l.synth(out_id.synth_tiled(R::FqDqA, t), &[rows, n]);
            l.synth(out_id.synth_tiled(R::FqTileRow, t), &[rows, n]);
        }
        let raw_h = rb(&raw, r_rows, n_cols);
        let dqa_h = rb(&dqa, r_rows, n_cols);
        let landing_h = rb(&landing, r_rows, n_cols);
        ops.push(pw2(
            &format!("{out}_fq_dqa_t{t}_op"),
            "mul",
            r_rows,
            n_cols,
            In::full(&raw_h),
            In::col(staging.ascale()),
            &dqa_h,
            sym_id_base,
            layout,
        ));
        ops.push(pw2(
            &format!("{out}_fq_dqw_t{t}_op"),
            "mul",
            r_rows,
            n_cols,
            In::full(&dqa_h),
            In::mb(&ws_h),
            &landing_h,
            sym_id_base,
            layout,
        ));
        // (6) THE OUT-SPLICE — the clone form again, N-side: the part's REAL rows (`m` of
        // them — S5's `out[:orig_m].clone()`, `:362-367`) of each output stick group land
        // at row 0 of the output. The part starts at row 0 (`m_parts = [padded]`, one part),
        // and exactly `[0, m)` is written — the pad rows never reach the output.
        for c in 0..n_groups {
            ops.push(crate::ir::bridge::tiled_op_sdsc_op::assemble_matmul_placed(
                &format!("{out}_fq_to{t}c{c}_op"),
                crate::sdsc_abstract::MatM::of_token_rows(m),
                crate::sdsc_abstract::MatN::one_stick(Lanes::FP16),
                crate::sdsc_abstract::MatK::one_stick(Lanes::FP16),
                crate::sdsc_abstract::MatY::unbatched(),
                crate::ir::bridge::tiled_op_sdsc_op::SharedKernelBmmForm::batch_inner_proven(
                    bmm_site::TapeLoweringSite::witness(),
                ),
                &landing_h,
                OperandPlacement::at_head(&landing_nest, Idx::<Row>::n(0), c, 0, slabs),
                &ident_h,
                DevOff::ZERO,
                &out_h,
                OperandPlacement::at_head(&out_nest, Idx::<Row>::n(0), c, 0, slabs),
                sym_id_base,
                layout,
            ));
        }
        Ok(ops)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::place::act_name;
    use crate::placement::{SegRole, TensorPlacement};

    /// A layout that places every tensor the fp8 chain names — the activation, the weight, the
    /// per-channel w_scale, the output, and the three worker-bound E4M3 consts — the same facts
    /// `lower_graph_to_superdsc`'s layout carries, stated here directly so the descriptors can be
    /// asked for in-crate. Ids are MINTED (`ids` map), because `resolve_seg_base` recovers a
    /// tensor's identity by lookup, never by parsing its spelling.
    fn fp8_layout(m: u32, k: u32, n: u32) -> BundleLayout {
        let mut l = BundleLayout::default();
        let place = |tid: u32, rows: u32, cols: u32, role: SegRole, l: &mut BundleLayout| {
            l.ids.borrow_mut().insert(act_name(tid), PlaceId::Act(tid));
            let seg = role.segment();
            let off = l.segment_bytes[seg];
            let size = crate::placement::synth_footprint_bytes(&[rows, cols], Df::Fp16);
            l.placements.insert(
                tid,
                TensorPlacement {
                    tid,
                    role,
                    segment: seg,
                    bank: 0,
                    offset: off,
                    size,
                    cover: crate::placement::BindCover::Full,
                },
            );
            l.segment_bytes[seg] = crate::placement::align128(off + size);
        };
        place(7, m, k, SegRole::Activation, &mut l); // a_tid: the activation
        place(8, n, k, SegRole::Weight, &mut l); // the fp8 weight (fp16-sized reservation is fine)
        place(9, 1, n, SegRole::Activation, &mut l); // w_scale
        place(10, m, n, SegRole::Activation, &mut l); // out_tid
        place(11, m, n, SegRole::Activation, &mut l); // a second GEMM's out_tid (the sharing tests)
        for tid in [FP8_POS448_TID, FP8_NEG448_TID, FP8_INV448_TID] {
            place(tid, 1, Fp16::ELEMS_PER_STICK, SegRole::Activation, &mut l);
        }
        // ⭐ THE IDENTITY TABLE — the row-tiled law's clone reads the same `[64,64]` registry
        // entry the card-proven KV-cache write's clone does: one attention class at hd=64,
        // class 0 = `IDENTITY_TID` (the widest). Placed LAST so no earlier tensor's absolute
        // offset moves — the per-pair tail tests read address DELTAS, and those must stay
        // pinned to the tail law alone.
        l.attn_class_hds = vec![Fp16::ELEMS_PER_STICK];
        place(
            crate::reserved_tids::IDENTITY_TID,
            Fp16::ELEMS_PER_STICK,
            Fp16::ELEMS_PER_STICK,
            SegRole::Activation,
            &mut l,
        );
        l
    }

    fn facts() -> Fp8Facts {
        Fp8Facts {
            a_tid: 7,
            a_name: act_name(7),
            w_name: act_name(8),
            ws_name: act_name(9),
            out_tid: 10,
        }
    }

    /// ⭐⭐⭐⭐⭐ THE QUANT-CHAIN REDUCE AT AN LX-OVERFLOWING GEOMETRY STAYS SINGLE-SHOT.
    ///
    /// This is the gemma-4-12b fp8 prefill blocker's own shape class: `m=1024, k=16384`. The OLD
    /// two-operand amax reduce sat at `2·(1024/32)·16384·2 = 2,097,152 B` of per-core LX — past
    /// the 1,677,721-B budget — so it time-tiled (`time=2`), which the bake's `tiled_trips_alias`
    /// guard refuses (every trip writes the same `[m,64]` accum bytes; with the vendor DDL's
    /// overwrite semantics the surviving amax would be the LAST slice's max, not the global max).
    /// The fused `quantscalepertokenfp8` op is UNSEEDED — one `[m,k]` input at
    /// `(1024/32)·16384·2 = 1,048,576 B` per core, HALF the old residency — so it fits the budget
    /// single-shot and the blocker's geometry no longer tiles the reduce at all. (The wide
    /// POINTWISE ops over the same `[m,k]` legitimately time-tile: their outputs are `Active` over
    /// `out`, so `rewrite_op_for_time_tile` strides them per trip and the bake accepts them. A
    /// REDUCE accum is `RedStick` — exempt from the stride — which is the defect class, so the
    /// reduce is what this pins.)
    /// The reduce law belongs to the CHAIN, so this pins it with a direct chain call — the
    /// door itself rows-tiles this geometry now (`work::Fp8GemmRowTiles`: k ≥ 4096 is wide),
    /// and the tile's own quant runs at the ABI row class where the budget is never in
    /// question. The chain still serves callers at whole-tensor row counts (the expert
    /// door's per-chunk activations), so the single-shot law is pinned where it lives.
    #[test]
    fn the_quant_reduce_at_an_lx_overflowing_geometry_stays_single_shot() {
        let (m, k, n) = (1024u32, 16384, 128);
        let l = fp8_layout(m, k, n);
        let mut quantized = std::collections::HashSet::new();
        let ops = fp8_quantize_descriptors(
            &Fp8QuantSource::activation(&facts(), m),
            k,
            1,
            &mut 0,
            Some(&l),
            &mut quantized,
        )
        .expect("the wide-geometry fp8 chain emits");
        // The chain's ONE reduce — the fused quantscalepertokenfp8 — must be single-shot.
        let tiled: Vec<&str> = ops
            .iter()
            .filter(|o| o.time > 1 && o.op_name.ends_with("fq_qspt_op"))
            .map(|o| o.op_name.as_str())
            .collect();
        assert!(
            tiled.is_empty(),
            "the fused quant reduce must be single-shot (time=1); time-tiled: {tiled:?}"
        );
        // The explicit amax family is GONE — fused into the one op — and with it the partial
        // slots (nothing mints `FqAmaxP` while the fused op fits the budget).
        assert_eq!(
            ops.iter()
                .filter(|o| o.op_name.ends_with("fq_qspt_op"))
                .count(),
            1,
            "the amax chain is ONE fused quantscalepertokenfp8 op"
        );
        assert!(
            !ops.iter().any(|o| o.op_name.contains("fq_amax")),
            "the explicit amax family is fused away; a stray amax op is a partial re-introduction"
        );
    }

    /// ⭐⭐⭐⭐⭐ THE FUSED QUANT REDUCE AT ITS OWN OVERFLOWING GEOMETRY SPLITS INTO PARTIALS.
    ///
    /// The 12b fp8 prefill's rung-71 wall: `m=71, k=15360` — a PRIME rung, so `core_split` finds
    /// no divisor ≤ 32 and the reduce keeps the whole row count per core. Even at the true DDL
    /// arity of ONE operand the residency is `1·71·15360·2 = 2,181,120 B` vs the 1,677,721-B
    /// budget, so the tiler legitimately answers `time=2` — and a time-tiled reduce is
    /// unemittable for this op class (the DDL's accumulator has overwrite semantics; every trip
    /// would write the same `[m,64]` scale bytes — the bake's `tiled_trips_alias` refusal). The
    /// emission must therefore never hand the tiler's `time>1` answer to the assembler: it
    /// splits into per-k-slice PARTIALS + a `maximum` COMBINE, every one of them single-shot.
    #[test]
    fn the_quant_reduce_at_a_fused_overflowing_geometry_emits_partials() {
        let (m, k, n) = (71u32, 15360, 3840);
        let l = fp8_layout(m, k, n);
        let mut quantized = std::collections::HashSet::new();
        let ops = fp8_quantize_descriptors(
            &Fp8QuantSource::activation(&facts(), m),
            k,
            1,
            &mut 0,
            Some(&l),
            &mut quantized,
        )
        .expect("the fused-overflowing fp8 quant chain emits");
        assert!(
            ops.iter()
                .all(|o| !o.op_name.contains("fq_qspt") || o.time == 1),
            "every fused-quant op (base, partials, combines) is single-shot; tiled: {:?}",
            ops.iter()
                .filter(|o| o.op_name.contains("fq_qspt") && o.time > 1)
                .map(|o| o.op_name.as_str())
                .collect::<Vec<_>>()
        );
        // The partial + combine family exists, folded into the same `ascale` the tail reads.
        assert!(
            ops.iter().any(|o| o.op_name.ends_with("fq_qspt_p1_op")),
            "the first partial is emitted"
        );
        assert!(
            ops.iter().any(|o| o.op_name.ends_with("fq_qspt_c1_op")),
            "the first combine is emitted"
        );
        assert!(
            l.synth
                .borrow()
                .map
                .contains_key(&PlaceId::Act(7).synth(SynthRole::FqAmaxP(1)).to_string()),
            "the first partial synth is declared"
        );
    }

    /// ⭐ THE FITTING CASE IS ONE FUSED OP: at a geometry the tiler passes (decode m=1, and
    /// every granite fp8 prefill), the quant chain's amax is the ONE `fq_qspt_op` and NO
    /// explicit amax-family op exists. Pins that no partial path can fire where the fused op
    /// was correct. (m=4 keeps the door on its DENSE arm — `Fp8GemmRowTiles`'s small-M law
    /// runs rows ≤ 4 whole — so this pins the chain the shipped bundles run.)
    #[test]
    fn the_quant_chain_at_a_fitting_geometry_is_one_fused_op() {
        let (m, k, n) = (4u32, 2048, 128);
        let l = fp8_layout(m, k, n);
        let mut quantized = std::collections::HashSet::new();
        let ops = matmul_fp8_descriptors(&facts(), m, k, n, &mut 0, Some(&l), &mut quantized)
            .expect("the fitting-geometry fp8 chain emits");
        assert!(
            ops.iter().all(|o| o.time == 1),
            "a fitting geometry has no time-tiled op at all"
        );
        assert_eq!(
            ops.iter()
                .filter(|o| o.op_name.ends_with("fq_qspt_op"))
                .count(),
            1,
            "exactly one fused quant reduce"
        );
        assert!(
            !ops.iter()
                .any(|o| o.op_name.contains("fq_amax_p") || o.op_name.contains("fq_amax_c")),
            "no partial or combine op may exist at a fitting geometry"
        );
        assert!(
            !l.synth
                .borrow()
                .map
                .contains_key(&PlaceId::Act(7).synth(SynthRole::FqAmaxP(1)).to_string()),
            "no partial synth may be declared at a fitting geometry"
        );
    }

    /// ⭐⭐⭐⭐⭐ THE 26b K-PAD (2112 → 2176): gemma-4-26b's dense GeGLU DOWN gemm is the FIRST fp8
    /// model whose contraction k is not a whole 128-elem SEN143_FP8 stick (2112 = 16.5 sticks —
    /// the panic that stopped the 26b phase-2 bake at `matmul_s20`). The chain must:
    ///   · emit at all (the `qfp8ch` convert's `assert_df_stick_multiple` refused the sub-stick
    ///     width — this test passing IS the wall gone);
    ///   · reserve `cl` and `afp8` at `[m, k_pad]` (the widened tail the convert and the
    ///     contraction share);
    ///   · ZERO-FILL the pad window (one `sub(448,448)` op) — stale pad lanes are last step's
    ///     activation, and a NaN there passes `maximum`/`minimum` unchanged into the product;
    ///   · keep the amax at the LOGICAL width (the pad never reaches the activation scale).
    #[test]
    fn the_26b_k2112_pads_to_a_whole_fp8_stick() {
        let (m, k, n) = (1u32, 2112, 2816);
        let k_pad = crate::work::fp8_k_pad(k);
        assert_eq!(
            k_pad, 2176,
            "2112 rounds up to the next whole 128-elem stick"
        );
        let l = fp8_layout(m, k, n);
        let mut quantized = std::collections::HashSet::new();
        let ops = matmul_fp8_descriptors(&facts(), m, k, n, &mut 0, Some(&l), &mut quantized)
            .expect("the 2112-wide fp8 chain emits (the 26b wall is gone)");
        // The zero-fill exists (the `sub(448,448)` broadcast), and only ONE such op.
        assert_eq!(
            ops.iter()
                .filter(|o| o.op_name.ends_with("fq_zfpad_op"))
                .count(),
            1,
            "exactly one pad-window zero fill"
        );
        // The synth reservations for the widened tail: `cl` at [m, k_pad] fp16, `afp8` at
        // [m, k_pad] fp8 (1 byte/elem — the ½-fp16 residency).
        let s = l.synth.borrow();
        let cl_sz = s
            .sizes
            .get(&PlaceId::Act(7).synth(SynthRole::FqCl).to_string())
            .expect("cl is placed");
        assert_eq!(
            *cl_sz,
            m as u64 * k_pad as u64 * 2,
            "cl owns [m, k_pad] fp16"
        );
        let afp8_sz = s
            .sizes
            .get(&PlaceId::Act(7).synth(SynthRole::FqAfp8).to_string())
            .expect("afp8 is placed");
        assert_eq!(
            *afp8_sz,
            m as u64 * k_pad as u64,
            "afp8 owns [m, k_pad] fp8 (1 B/elem)"
        );
        // The amax family is untouched by the pad: `absx` stays at the LOGICAL [m, k].
        let absx_sz = s
            .sizes
            .get(&PlaceId::Act(7).synth(SynthRole::FqAbsX).to_string())
            .expect("absx is placed");
        assert_eq!(*absx_sz, m as u64 * k as u64 * 2, "absx owns [m, k] fp16");
    }

    /// ⭐ THE ALIGNED CASE IS BYTE-IDENTICAL: at a 128-aligned k (every earlier fp8 model), no
    /// zero-fill op exists, no widened reservation exists, and the chain is exactly the ops the
    /// pre-pad emitter produced. Pins that the pad cannot fire where the width was already legal.
    /// (m=128 keeps the door on its DENSE arm — `Fp8GemmRowTiles`'s narrow law runs a
    /// 128-aligned row count whole.)
    #[test]
    fn an_aligned_k_emits_no_pad_op_and_no_widened_reservation() {
        let (m, k, n) = (128u32, 2048, 128);
        let l = fp8_layout(m, k, n);
        let mut quantized = std::collections::HashSet::new();
        let ops = matmul_fp8_descriptors(&facts(), m, k, n, &mut 0, Some(&l), &mut quantized)
            .expect("the aligned-geometry fp8 chain emits");
        assert!(
            !ops.iter().any(|o| o.op_name.ends_with("fq_zfpad_op")),
            "no zero-fill op may exist at an aligned k"
        );
        let s = l.synth.borrow();
        let cl_sz = s
            .sizes
            .get(&PlaceId::Act(7).synth(SynthRole::FqCl).to_string())
            .expect("cl is placed");
        assert_eq!(
            *cl_sz,
            m as u64 * k as u64 * 2,
            "cl owns [m, k] fp16 — no widening"
        );
    }

    /// An operand's own HBM start (core 0's allocate-node address for the binding whose buffer
    /// matches `needle`), in BYTES — `resolve_seg_base` folds each arg's `offset_elems` into the
    /// segment base, so the DELTAS between ops reading the same buffer at different offsets are
    /// the offsets themselves. The per-pair tail's laws read those deltas.
    fn operand_start_bytes(op: &EmittedOp, needle: &str, want_input: bool) -> u64 {
        let dsc = op.dsc();
        for dsc_map in dsc.dscs_.iter() {
            for dsc in dsc_map.values() {
                for node in dsc.scheduleTree_.iter() {
                    let binding = &op.arg_bindings[node.ldsIdx_ as usize];
                    if binding.is_input == want_input && binding.buffer.contains(needle) {
                        let v = node
                            .startAddressCoreCorelet_
                            .data_
                            .values()
                            .next()
                            .expect("a start address");
                        return v.parse().expect("addresses are numeric");
                    }
                }
            }
        }
        panic!("{}: no {} AllocNode", op.op_name, needle);
    }

    /// ⭐⭐⭐⭐⭐ THE 26b DOWN TAIL PADS PER PAIR (704 → 768 × 8 pairs): the expert door's DOWN
    /// projection contracts inn=704 = 5.5 fp8 sticks, so the LOGICAL pair stride is a HALF fp8
    /// stick and the quantized activation is laid out per-pair PADDED — pair `p`'s window at
    /// `p·inn_pad` of a `[m, pairs·inn_pad]` buffer. This pins the whole tail law at the door's
    /// own geometry (m=1, pairs=8, k=5632):
    ///   · EIGHT `qfp8ch` converts — one per pair, pair 0 keeping the shipped `fq_afp8_op`
    ///     name and pairs 1..7 carrying `fq_afp8_p{p}_op`;
    ///   · the OUTPUT operand starts advance by the PADDED stride (768 B/pair) and every one
    ///     lands on a whole 128-B stick — the mid-stick start this layout exists to exclude;
    ///   · the INPUT (fp16 `cl`) starts advance by the LOGICAL stride (1408 B/pair);
    ///   · ONE zero-fill, aimed exactly one fp16 stick past the logical width (the last
    ///     pair's overhang `[k, cl_reach)`);
    ///   · the reservations: `cl` at `[m, cl_reach]` (5696·2 B), `afp8` at `[m, k_pad]`
    ///     (6144 B fp8), `absx` still at the LOGICAL `[m, k]`.
    #[test]
    fn the_26b_down_tail_pads_per_pair() {
        let (m, pairs, inn, n) = (1u32, 8u32, 704u32, 128u32);
        let k = pairs * inn; // 5632 — itself 128-aligned, so only the PER-PAIR law pads here
        let inn_pad = crate::work::fp8_k_pad(inn);
        assert_eq!(
            inn_pad, 768,
            "704 rounds up to the next whole 128-elem stick"
        );
        let (k_pad, cl_reach) = (pairs * inn_pad, (pairs - 1) * inn + inn_pad);
        let l = fp8_layout(m, k, n);
        let mut quantized = std::collections::HashSet::new();
        let ops = fp8_quantize_descriptors(
            &Fp8QuantSource::activation(&facts(), m),
            k,
            pairs,
            &mut 0,
            Some(&l),
            &mut quantized,
        )
        .expect("the per-pair tail emits at the 26b DOWN geometry");
        // EIGHT converts, named per pair.
        let first = ops
            .iter()
            .filter(|o| o.op_name.ends_with("fq_afp8_op"))
            .count();
        let rest = ops
            .iter()
            .filter(|o| o.op_name.contains("fq_afp8_p"))
            .count();
        assert_eq!(
            (first, rest),
            (1, 7),
            "pair 0 keeps the shipped name; pairs 1..7 carry fq_afp8_p{{p}}_op"
        );
        // The OUTPUT windows advance by the PADDED stride and stay whole-stick — the
        // fp8 A operand starts the per-pair layout exists to keep legal.
        let afp8_starts: Vec<u64> = ops
            .iter()
            .filter(|o| o.op_name.ends_with("fq_afp8_op") || o.op_name.contains("fq_afp8_p"))
            .map(|o| operand_start_bytes(o, "fq_afp8", false))
            .collect();
        assert_eq!(afp8_starts.len(), 8);
        for s in &afp8_starts {
            assert_eq!(
                s % 128,
                0,
                "every afp8 window starts on a whole 128-B stick"
            );
        }
        assert_eq!(
            afp8_starts[7] - afp8_starts[0],
            7 * inn_pad as u64,
            "the fp8 windows advance by the PADDED stride ({} B/pair), not the logical half-stick one",
            inn_pad
        );
        // The INPUT (fp16 cl) windows advance by the LOGICAL stride.
        let cl_starts: Vec<u64> = ops
            .iter()
            .filter(|o| o.op_name.ends_with("fq_afp8_op") || o.op_name.contains("fq_afp8_p"))
            .map(|o| operand_start_bytes(o, "fq_cl", true))
            .collect();
        assert_eq!(
            cl_starts[7] - cl_starts[0],
            7 * inn as u64 * 2,
            "the fp16 cl windows advance by the LOGICAL stride ({} B/pair)",
            inn * 2
        );
        // ONE zero-fill, aimed exactly one fp16 stick past the logical width.
        let zf = ops
            .iter()
            .find(|o| o.op_name.ends_with("fq_zfpad_op"))
            .expect("the per-pair tail zero-fills the last pair's overhang");
        assert_eq!(
            ops.iter()
                .filter(|o| o.op_name.ends_with("fq_zfpad_op"))
                .count(),
            1,
            "exactly one pad-window zero fill"
        );
        assert_eq!(
            operand_start_bytes(zf, "fq_cl", false) - cl_starts[0],
            k as u64 * 2,
            "the fill writes [k, cl_reach) — one fp16 stick past the logical width"
        );
        // The reservations: cl to its REACH, afp8 at the PADDED width, absx at the LOGICAL one.
        let s = l.synth.borrow();
        let a7 = PlaceId::Act(7);
        let cl_sz = s
            .sizes
            .get(&a7.synth(SynthRole::FqCl).to_string())
            .expect("cl is placed");
        assert_eq!(
            *cl_sz,
            m as u64 * cl_reach as u64 * 2,
            "cl owns [m, cl_reach] = [1, {cl_reach}] fp16"
        );
        let afp8_sz = s
            .sizes
            .get(&a7.synth(SynthRole::FqAfp8).to_string())
            .expect("afp8 is placed");
        assert_eq!(
            *afp8_sz,
            m as u64 * k_pad as u64,
            "afp8 owns [m, k_pad] = [1, {k_pad}] fp8 (1 B/elem)"
        );
        let absx_sz = s
            .sizes
            .get(&a7.synth(SynthRole::FqAbsX).to_string())
            .expect("absx is placed");
        assert_eq!(*absx_sz, m as u64 * k as u64 * 2, "absx owns [m, k] fp16");
    }

    /// ⭐ THE ALIGNED-`inn` CONTROL: at the expert door's gate/up geometry (inn=2816, itself a
    /// whole fp8 stick), the padded layout IS the logical one — ONE whole-tensor convert, no
    /// per-pair naming, no zero-fill, no widened `cl`. Byte-identical to the pre-pair emission;
    /// pins that the per-pair law cannot fire where the logical stride was already legal.
    #[test]
    fn an_aligned_inn_keeps_the_whole_tensor_convert() {
        let (m, pairs, inn, n) = (1u32, 8u32, 2816u32, 128u32);
        let k = pairs * inn;
        let l = fp8_layout(m, k, n);
        let mut quantized = std::collections::HashSet::new();
        let ops = fp8_quantize_descriptors(
            &Fp8QuantSource::activation(&facts(), m),
            k,
            pairs,
            &mut 0,
            Some(&l),
            &mut quantized,
        )
        .expect("the aligned-inn pair chain emits");
        assert_eq!(
            ops.iter()
                .filter(|o| o.op_name.ends_with("fq_afp8_op"))
                .count(),
            1,
            "one whole-tensor convert"
        );
        assert!(
            !ops.iter().any(|o| o.op_name.contains("fq_afp8_p")),
            "no per-pair convert at an aligned inn"
        );
        assert!(
            !ops.iter().any(|o| o.op_name.ends_with("fq_zfpad_op")),
            "no zero-fill at an aligned inn (cl_reach == k)"
        );
        let s = l.synth.borrow();
        let cl_sz = s
            .sizes
            .get(&PlaceId::Act(7).synth(SynthRole::FqCl).to_string())
            .expect("cl is placed");
        assert_eq!(
            *cl_sz,
            m as u64 * k as u64 * 2,
            "cl owns [m, k] fp16 — no widening"
        );
    }

    /// ⛔ THE PAIR LAW'S OWN REFUSAL: a K axis that does not divide into whole per-pair lane
    /// counts is a stated design error (typed `Err`, never a sub-stick bake) — the one refusal
    /// left in this fn. ⭐ THE HALF-STICK TAIL AT m>1 NOW EMITS (design §4.5 landed): the
    /// per-slot arms split on `m` — m=1 keeps the FLAT stick-axis slices (the decode form,
    /// byte-identical), m>1 takes [`assemble_convert_col`] at [`crate::addr::col_of`] group
    /// corners, whose `(c/lanes)·(m·lanes)` term carries the row count. This pins the m>1
    /// corner law at (m=2, pairs=2, inn=704): output windows advance by the PADDED
    /// group-corner stride `m·inn_pad` B (the `m` is the law — the flat form would say
    /// `inn_pad`), input windows by the LOGICAL group-corner stride `m·inn` fp16 elems, the
    /// zero-fill still aims one fp16 group past the logical width at the m=2 corner, and the
    /// reservations widen to the m-generic `[m, cl_reach]`/`[m, k_pad]`.
    #[test]
    fn the_per_pair_tail_refuses_a_non_whole_lane_count_and_pads_per_slot_at_prefill() {
        let l = fp8_layout(1, 9, 128);
        let mut quantized = std::collections::HashSet::new();
        let e = fp8_quantize_descriptors(
            &Fp8QuantSource::activation(&facts(), 1),
            9,
            2,
            &mut 0,
            Some(&l),
            &mut quantized,
        );
        assert!(
            e.is_err(),
            "k=9 over pairs=2 has no whole lane count per pair — a typed refusal, not a bake"
        );
        // The 26b DOWN's own tail at prefill: inn=704 pads to 768, and the per-slot arms take
        // the GROUP-CORNER offsets — m=2 EMITS now, two rank-2 stick-major converts.
        let (m, pairs, inn) = (2u32, 2u32, 704u32);
        let k = pairs * inn; // 1408
        let inn_pad = crate::work::fp8_k_pad(inn);
        assert_eq!(
            inn_pad, 768,
            "704 rounds up to the next whole 128-elem stick"
        );
        let (k_pad, cl_reach) = (pairs * inn_pad, (pairs - 1) * inn + inn_pad);
        let l2 = fp8_layout(m, k, 128);
        let mut quantized2 = std::collections::HashSet::new();
        let ops2 = fp8_quantize_descriptors(
            &Fp8QuantSource::activation(&facts(), m),
            k,
            pairs,
            &mut 0,
            Some(&l2),
            &mut quantized2,
        )
        .expect("the half-stick tail emits at m>1 — the per-slot §4.5 arms");
        assert_eq!(
            ops2.iter()
                .filter(|o| o.op_name.ends_with("fq_afp8_op"))
                .count(),
            1,
            "slot 0 keeps the shipped fq_afp8_op name"
        );
        assert_eq!(
            ops2.iter()
                .filter(|o| o.op_name.contains("fq_afp8_p"))
                .count(),
            (pairs - 1) as usize,
            "slots 1.. carry fq_afp8_p{{s}}_op"
        );
        let arms: Vec<&EmittedOp> = ops2
            .iter()
            .filter(|o| o.op_name.ends_with("fq_afp8_op") || o.op_name.contains("fq_afp8_p"))
            .collect();
        assert_eq!(arms.len(), pairs as usize, "one arm per slot");
        let afp8_starts: Vec<u64> = arms
            .iter()
            .map(|o| operand_start_bytes(o, "fq_afp8", false))
            .collect();
        let cl_starts: Vec<u64> = arms
            .iter()
            .map(|o| operand_start_bytes(o, "fq_cl", true))
            .collect();
        for s in &afp8_starts {
            assert_eq!(
                s % 128,
                0,
                "every afp8 window starts on a whole 128-B stick at m>1 too"
            );
        }
        assert_eq!(
            afp8_starts[1] - afp8_starts[0],
            m as u64 * inn_pad as u64,
            "the fp8 windows advance by the PADDED GROUP-CORNER stride m·inn_pad = {} B — \
             the m term is the §4.5 law (the flat m=1 form would say {inn_pad})",
            m * inn_pad
        );
        assert_eq!(
            cl_starts[1] - cl_starts[0],
            m as u64 * inn as u64 * 2,
            "the fp16 cl windows advance by the LOGICAL GROUP-CORNER stride m·inn fp16 \
             elems = {} B (the flat m=1 form would say {})",
            m * inn * 2,
            inn * 2
        );
        // ONE zero-fill, aimed at the m=2 group corner one fp16 group past the logical width:
        // `col_of(2, cl_reach, k)` = (k/64)·(2·64) elems = m·k fp16 elems.
        let zf = ops2
            .iter()
            .find(|o| o.op_name.ends_with("fq_zfpad_op"))
            .expect("the per-slot tail zero-fills the last slot's overhang");
        assert_eq!(
            ops2.iter()
                .filter(|o| o.op_name.ends_with("fq_zfpad_op"))
                .count(),
            1,
            "exactly one pad-window zero fill at m>1"
        );
        assert_eq!(
            operand_start_bytes(zf, "fq_cl", false) - cl_starts[0],
            m as u64 * k as u64 * 2,
            "the fill writes [k, cl_reach) at the m=2 group corner — m·k fp16 elems past base"
        );
        // The reservations, m-generic: cl [m, cl_reach], afp8 [m, k_pad], absx [m, k].
        let s2 = l2.synth.borrow();
        let a7 = PlaceId::Act(7);
        let cl_sz = s2
            .sizes
            .get(&a7.synth(SynthRole::FqCl).to_string())
            .expect("cl is placed");
        assert_eq!(
            *cl_sz,
            m as u64 * cl_reach as u64 * 2,
            "cl owns [m, cl_reach] = [{m}, {cl_reach}] fp16"
        );
        let afp8_sz = s2
            .sizes
            .get(&a7.synth(SynthRole::FqAfp8).to_string())
            .expect("afp8 is placed");
        assert_eq!(
            *afp8_sz,
            m as u64 * k_pad as u64,
            "afp8 owns [m, k_pad] = [{m}, {k_pad}] fp8 (1 B/elem)"
        );
        let absx_sz = s2
            .sizes
            .get(&a7.synth(SynthRole::FqAbsX).to_string())
            .expect("absx is placed");
        assert_eq!(*absx_sz, m as u64 * k as u64 * 2, "absx owns [m, k] fp16");
        // And the positive control: the same m=2 at an ALIGNED inn (1024, a whole fp8 stick)
        // EMITS — one whole-tensor convert, the m-generic form the prefill door consumes.
        let l3 = fp8_layout(2, 2048, 128);
        let mut quantized3 = std::collections::HashSet::new();
        let ops = fp8_quantize_descriptors(
            &Fp8QuantSource::activation(&facts(), 2),
            2048,
            2,
            &mut 0,
            Some(&l3),
            &mut quantized3,
        )
        .expect("the aligned-inn pair chain emits at m=2");
        assert_eq!(
            ops.iter()
                .filter(|o| o.op_name.ends_with("fq_afp8_op"))
                .count(),
            1,
            "one whole-tensor convert at m=2 — no per-pair arms at an aligned inn"
        );
        assert!(
            !ops.iter().any(|o| o.op_name.contains("fq_afp8_p")),
            "no per-pair convert at an aligned inn, prefill included"
        );
    }

    /// ⭐⭐⭐⭐⭐ THE WIDE PREFILL TILES ITS ROWS AND SPLICES THE REAL ONES BACK — the defect-3
    /// geometry itself (granite-2b q/k/v at m=21, k=2048, n=3072: 3×1024 output tiles ⇒ wide).
    /// The door must emit the row-tiled law, whole: per tile a zero-fill (`fq_tz`), one clone
    /// per K stick group (`fq_tc`), the quant chain over the CLONE (its ops carrying the tile
    /// index — never the dense staging's shared `fq_afp8_op`, the torn walk this law replaces),
    /// one mm, the two dequant legs, and one TIME-FOLDED out-copy per tile (`fq_to`, the N
    /// stick groups its trips). The counts are the plan's own: 1 tile of 128 rows (21 padded
    /// to 128), 32 K groups, 48 N groups.
    #[test]
    fn the_wide_prefill_tiles_its_rows_and_splices_the_real_ones_back() {
        let (m, k, n) = (21u32, 2048, 3072);
        let plan = crate::work::Fp8GemmRowTiles::of(
            RowCount::of_token_rows(m),
            BlockCols::of_feature_cols(k),
            BlockCols::of_feature_cols(n),
        )
        .expect("the granite-2b qkv geometry plans");
        assert_eq!(
            (plan.tile_rows(), plan.tile_count(), plan.total_rows()),
            (
                RowCount::of_token_rows(128),
                crate::work::Fp8GemmTileCount::new(1),
                RowCount::of_token_rows(128)
            ),
            "21 rows pad UP to the one 128-row tile — never a trailing short tile"
        );
        let l = fp8_layout(m, k, n);
        let mut quantized = std::collections::HashSet::new();
        let ops = matmul_fp8_descriptors(&facts(), m, k, n, &mut 0, Some(&l), &mut quantized)
            .expect("the wide prefill emits the row-tiled law");
        let n_named = |pfx: &str| {
            ops.iter()
                .filter(|o| o.op_name.starts_with(pfx) && o.op_name.ends_with("_op"))
                .count()
        };
        assert_eq!(n_named("t10_fq_tz"), 1, "one pad zero-fill per tile");
        assert_eq!(
            n_named("t10_fq_tc"),
            1,
            "one TIME-FOLDED clone per tile — the K stick groups are its trips (1×32)"
        );
        assert_eq!(n_named("t10_fq_qspt"), 1, "the quant chain runs per tile");
        assert_eq!(n_named("t10_fq_mm_t"), 1, "one tile mm per tile");
        assert_eq!(
            n_named("t10_fq_dqa_t"),
            1,
            "one a-scale dequant leg per tile"
        );
        assert_eq!(
            n_named("t10_fq_dqw_t"),
            1,
            "one w-scale dequant leg per tile"
        );
        assert_eq!(
            n_named("t10_fq_to"),
            1,
            "one TIME-FOLDED out-copy per tile — the N stick groups are its trips (1×48)"
        );
        // ⭐ THE FANS ARE FOLDS — one op per tile, the stick groups its trips: `time` carries
        // the count, `concrete_trips` materializes the per-group windows, each fan tensor
        // steps by its nest's own group stride, and the identity table stays FIXED (empty
        // stride map — the no-bump path).
        for o in ops.iter().filter(|o| o.op_name.starts_with("t10_fq_tc")) {
            assert_eq!(
                o.time, 32,
                "{}: the clone folds the 32 K stick groups",
                o.op_name
            );
            assert_eq!(
                o.concrete_trips().len(),
                32,
                "{}: the clone's trips are the 32 K stick-group windows",
                o.op_name
            );
            assert_eq!(
                o.affine_strides[0].get("out"),
                Some(&nest_group_stride_bytes(m, 64)),
                "{}: the A side steps by the source nest's group stride",
                o.op_name
            );
            assert_eq!(
                o.affine_strides[2].get("out"),
                Some(&nest_group_stride_bytes(128, 64)),
                "{}: the C side steps by the tile nest's group stride",
                o.op_name
            );
            assert!(
                o.affine_strides[1].is_empty(),
                "{}: the identity table is fixed across trips",
                o.op_name
            );
        }
        for o in ops.iter().filter(|o| o.op_name.starts_with("t10_fq_to")) {
            assert_eq!(
                o.time, 48,
                "{}: the splice folds the 48 N stick groups",
                o.op_name
            );
            assert_eq!(
                o.concrete_trips().len(),
                48,
                "{}: the splice's trips are the 48 N stick-group windows",
                o.op_name
            );
            assert_eq!(
                o.affine_strides[0].get("out"),
                Some(&nest_group_stride_bytes(128, 64)),
                "{}: the landing side steps by the tile nest's group stride",
                o.op_name
            );
            assert_eq!(
                o.affine_strides[2].get("out"),
                Some(&nest_group_stride_bytes(m, 64)),
                "{}: the output side steps by the output nest's group stride",
                o.op_name
            );
            assert!(
                o.affine_strides[1].is_empty(),
                "{}: the identity table is fixed across trips",
                o.op_name
            );
        }
        // ⛔ THE TORN STAGING IS GONE: no op carries the DENSE law's shared convert name —
        // every quant tensor and op is tile-indexed, so the quant walk and the mm's A walk
        // spell ONE buffer's layout (the one-mint law).
        assert!(
            !ops.iter().any(|o| o.op_name.ends_with("fq_afp8_op")),
            "the dense law's shared fq_afp8_op must not exist at a tiled geometry"
        );
        // Every tile mm reads ITS OWN staged afp8 (the typed staging's name, tile-indexed).
        for o in ops.iter().filter(|o| o.op_name.starts_with("t10_fq_mm_t")) {
            assert!(
                o.arg_bindings
                    .iter()
                    .any(|b| b.is_input && b.buffer.contains("fq_afp8t")),
                "{}: the tile mm binds the tile's own staged afp8",
                o.op_name
            );
        }
        // The staging reservations, at the TILE's row class: the clone `[128, k]` fp16 and the
        // quantized `[128, k_pad]` fp8 (k is 128-aligned here, so k_pad == k).
        let s = l.synth.borrow();
        let a7 = PlaceId::Act(7);
        let clone_sz = s
            .sizes
            .get(&a7.synth_tiled(SynthRole::FqTileRow, 0).to_string())
            .expect("the tile clone is placed");
        assert_eq!(
            *clone_sz,
            128u64 * k as u64 * 2,
            "tile 0's clone owns [128, k] fp16"
        );
        let afp8_sz = s
            .sizes
            .get(&a7.synth_tiled(SynthRole::FqAfp8, 0).to_string())
            .expect("the tile afp8 is placed");
        assert_eq!(
            *afp8_sz,
            128u64 * k as u64,
            "tile 0's afp8 owns [128, k_pad] fp8 (1 B/elem)"
        );
    }

    /// ⭐ THE SECOND GEMM AT THE SAME PLAN REUSES THE TILE STAGING — the dense law's own
    /// sharing (K/V after Q, Up after Gate), at tile grain: the clone, the zero-fill, and the
    /// quant chain are keyed by the TILE's name in `quantized`, so only the per-matmul legs
    /// (mm, dequant, out-splice) are emitted again, under the second output's namespace.
    #[test]
    fn a_second_gemm_at_the_same_plan_reuses_the_tile_staging() {
        let (m, k, n) = (21u32, 2048, 3072);
        let l = fp8_layout(m, k, n);
        let mut quantized = std::collections::HashSet::new();
        let f2 = Fp8Facts {
            a_tid: 7,
            a_name: act_name(7),
            w_name: act_name(8),
            ws_name: act_name(9),
            out_tid: 11,
        };
        let _ = matmul_fp8_descriptors(&facts(), m, k, n, &mut 0, Some(&l), &mut quantized)
            .expect("the first GEMM emits");
        let ops2 = matmul_fp8_descriptors(&f2, m, k, n, &mut 0, Some(&l), &mut quantized)
            .expect("the second GEMM over the same activation emits");
        let n_named = |pfx: &str| {
            ops2.iter()
                .filter(|o| o.op_name.starts_with(pfx) && o.op_name.ends_with("_op"))
                .count()
        };
        assert_eq!(n_named("t11_fq_tz"), 0, "no second zero-fill");
        assert_eq!(n_named("t11_fq_tc"), 0, "no second clone");
        assert_eq!(n_named("t11_fq_qspt"), 0, "no second quant chain");
        assert_eq!(
            n_named("t11_fq_mm_t"),
            1,
            "the per-matmul mm legs still run per tile"
        );
    }

    /// ⛔ THE PLAN DISAGREEMENT IS A REFUSAL, NOT A RESIZE: a second GEMM over the SAME
    /// activation whose own geometry takes the NARROW row class (n=128: one ABI tile, so the
    /// rows run whole at the 128 alignment) would re-resolve the SAME tile names at a
    /// different row class. `synth` is idempotent per name, so without the geometry guard
    /// that reuse would be silent — and at this convergence the footprint compare is BLIND
    /// (the narrow part's [128, k] is the wide tile's own bytes), so the refusal is the ARM
    /// guard's: the stamp `fp8_row_arms` minted by the first (wide) call is the discriminator.
    #[test]
    fn a_second_gemm_at_a_different_row_class_refuses() {
        let (m, k) = (21u32, 2048);
        let l = fp8_layout(m, k, 3072);
        let mut quantized = std::collections::HashSet::new();
        let _ = matmul_fp8_descriptors(&facts(), m, k, 3072, &mut 0, Some(&l), &mut quantized)
            .expect("the wide GEMM emits and reserves its tiles");
        let e = matmul_fp8_descriptors(&facts(), m, k, 128, &mut 0, Some(&l), &mut quantized)
            .err()
            .expect(
                "a second GEMM re-tiling the same activation at another row class is a \
                 plan disagreement — refused at bake, never a silent resize",
            );
        let msg = e.to_string();
        assert!(
            msg.contains("WIDE row class"),
            "the refusal must be the ARM guard's own text (wide-then-narrow at converged \
             footprints), not some earlier door refusal: {msg}"
        );
    }
}
