PROPOSED EXACT CODE — ROUND 3: F6+F7+F8 (the test round). Stacks on: hunk 1 → F1/F5 → F2/F3/F4 (all pending your ruling; the chain-call shapes below assume F2's signatures). Nothing is written until you rule. This AMENDS hunk 1's Fp8GemmGeometry sketch (it now carries the k_pad mints the chain call needs) and the pair-tail geometry from my stop-state message.

════════ F6a — work.rs, the two test-geometry types (COMPLETE; supersedes hunk 1's item) ════════

/// ⭐ A NAMED fp8 GEMM GEOMETRY — how a TEST states a shape: as a TYPE, never a bare
/// triple. No value constructor exists (a geometry is named, never built); the aliases
/// below are the geometries the suite actually states. The plan door goes through
/// [`Self::plan`], and the emission calls read the consts — so a test cannot state a
/// shape the plan was not asked about, and no shape is spelled twice.
pub struct Fp8GemmGeometry<const M: u32, const K: u32, const N: u32>;

impl<const M: u32, const K: u32, const N: u32> Fp8GemmGeometry<M, K, N> {
    /// The activation's token rows.
    pub const ROWS: u32 = M;
    /// The contraction width.
    pub const FEATURES: u32 = K;
    /// The output width.
    pub const OUT: u32 = N;
    pub fn rows() -> crate::sdsc_abstract::RowCount {
        crate::sdsc_abstract::RowCount::of_token_rows(M)
    }
    pub fn features() -> crate::sdsc_abstract::BlockCols {
        crate::sdsc_abstract::BlockCols::of_feature_cols(K)
    }
    pub fn out_features() -> crate::sdsc_abstract::BlockCols {
        crate::sdsc_abstract::BlockCols::of_feature_cols(N)
    }
    /// The tail tensors' padded width at `pairs == 1` — the chain's `k_pad` mint.
    pub fn k_pad() -> u32 {
        fp8_k_pad(K)
    }
    pub fn k_pad_cols() -> crate::sdsc_abstract::BlockCols {
        crate::sdsc_abstract::BlockCols::of_feature_cols(Self::k_pad())
    }
    pub fn plan() -> Result<Fp8GemmRowTiles, String> {
        Fp8GemmRowTiles::of(
            crate::sdsc_abstract::RowCount::of_token_rows(M),
            crate::sdsc_abstract::BlockCols::of_feature_cols(K),
            crate::sdsc_abstract::BlockCols::of_feature_cols(N),
        )
    }
}

/// ⭐ A NAMED PAIR-TAIL GEOMETRY — the per-pair quant law, stated as a type: `PAIRS`
/// windows of `INN` lanes, K DERIVED (one law, no second spelling), and the tail-law
/// derivations (`inn_pad`, `k_pad`, `cl_reach`) as METHODS — so no test re-derives the
/// pair arithmetic it is asserting against.
pub struct Fp8PairTailGeometry<const M: u32, const PAIRS: u32, const INN: u32, const N: u32 = 128>;

impl<const M: u32, const P: u32, const INN: u32, const N: u32>
    Fp8PairTailGeometry<M, P, INN, N>
{
    pub const ROWS: u32 = M;
    pub const PAIRS: u32 = P;
    pub const INN: u32 = INN;
    pub const OUT: u32 = N;
    /// The contraction width — `PAIRS · INN`, ONE law.
    pub const FEATURES: u32 = P * INN;
    pub fn inn_pad() -> u32 {
        fp8_k_pad(INN)
    }
    /// The tail tensors' padded width — the chain's `k_pad` mint, by the pair law.
    pub fn k_pad() -> u32 {
        P * Self::inn_pad()
    }
    /// `cl`'s reach: the last pair's convert overhang past the logical width.
    pub fn cl_reach() -> u32 {
        (P - 1) * INN + Self::inn_pad()
    }
    pub fn rows() -> crate::sdsc_abstract::RowCount {
        crate::sdsc_abstract::RowCount::of_token_rows(M)
    }
    pub fn features() -> crate::sdsc_abstract::BlockCols {
        crate::sdsc_abstract::BlockCols::of_feature_cols(Self::FEATURES)
    }
    pub fn k_pad_cols() -> crate::sdsc_abstract::BlockCols {
        crate::sdsc_abstract::BlockCols::of_feature_cols(Self::k_pad())
    }
}

The alias block (the subset the EMIT tests state; work.rs's own tests keep hunk 1's set —
same aliases, one home):

/// The granite-2b QKV prefill — the defect-3 shape itself.
pub type Granite2bQkvPrefill = Fp8GemmGeometry<21, 2048, 3072>;
/// The gemma-4-12b LX-overflowing prefill.
pub type Gemma12bLxOverflow = Fp8GemmGeometry<1024, 16384, 128>;
/// The gemma-4-12b rung-71 prefill.
pub type Gemma12bRung71 = Fp8GemmGeometry<71, 15360, 3840>;
/// The 26b DOWN decode (k=2112: the non-legal K).
pub type Gemma26bDownDecode = Fp8GemmGeometry<1, 2112, 2816>;
/// The 26b gate decode (n=2112: the packed-kernel out-stick tail).
pub type Gemma26bGateDecode = Fp8GemmGeometry<1, 2816, 2112>;
/// A narrow small-M control.
pub type NarrowSmallMHidden = Fp8GemmGeometry<4, 2048, 128>;
/// A narrow 128-aligned control.
pub type NarrowAlignedHidden = Fp8GemmGeometry<128, 2048, 128>;
/// A narrow single-tile-N geometry — the plan-disagreement test's second GEMM.
pub type NarrowSingleTileN = Fp8GemmGeometry<21, 2048, 64>;
/// A narrow PADDED row class — one 128-row tile over 21 real rows (F8).
pub type NarrowPadded = Fp8GemmGeometry<21, 128, 128>;
/// The 26b DOWN's per-pair tail, decode.
pub type DownDecodePairs = Fp8PairTailGeometry<1, 8, 704>;
/// The 26b gate/up aligned-inn control.
pub type GateUpAlignedPairs = Fp8PairTailGeometry<1, 8, 2816>;
/// The half-stick tail at prefill (m>1, the §4.5 per-slot arms).
pub type PrefillHalfStickPairs = Fp8PairTailGeometry<2, 2, 704>;

F8 NOTE: your finding named "32 real rows"; the alias states 21 (the granite-2b prefill's
own row count at a narrow geometry, and the number hunk 1's work.rs test already pins).
Same class either way — say the word if you want 32.

════════ F7 — the fp8_layout helper's identity mint (COMPLETE seam; doc block above it unchanged) ════════

        // ⭐ THE IDENTITY TABLE — the row-tiled law's clone reads the same `[64,64]` registry
        // entry the card-proven KV-cache write's clone does: one attention class at hd=64,
        // class 0 = `IDENTITY_TID` (the widest). Placed LAST so no earlier tensor's absolute
        // offset moves — the per-pair tail tests read address DELTAS, and those must stay
        // pinned to the tail law alone. The tid is MINTED through the same door production
        // reads (`identity_tid_of_class`) — never a hand-spelled registry constant, so the
        // test's table and the emission's lookup cannot drift apart.
        l.attn_class_hds = vec![Fp16::ELEMS_PER_STICK];
        let ident = l
            .identity_tid_of_class(Fp16::ELEMS_PER_STICK)
            .expect("the hd-64 attention class mints its identity table's tid");
        place(
            ident,
            Fp16::ELEMS_PER_STICK,
            Fp16::ELEMS_PER_STICK,
            SegRole::Activation,
            &mut l,
        );

════════ F6b — the facts door for the second GEMM (minted together with the layout it reads) ════════

    fn facts_out(out_tid: u32) -> Fp8Facts {
        Fp8Facts {
            a_tid: 7,
            a_name: act_name(7),
            w_name: act_name(8),
            ws_name: act_name(9),
            out_tid,
        }
    }

    fn facts() -> Fp8Facts {
        facts_out(10)
    }

(fp8_layout already places tid 11 at [m, n] — the sharing test's `f2` literal is replaced
by `facts_out(11)`, minted against the same layout.)

════════ F6c — the geometry-fed call pattern (every test, no bare triples anywhere) ════════
DOOR tests (four; complete exemplar — the fitting test; the other three are the same seam):

    #[test]
    fn the_quant_chain_at_a_fitting_geometry_is_one_fused_op() {
        use crate::work::NarrowSmallMHidden as G;
        let l = fp8_layout(G::ROWS, G::FEATURES, G::OUT);
        let mut quantized = std::collections::HashSet::new();
        let ops = matmul_fp8_descriptors(
            &facts(),
            G::ROWS,
            G::FEATURES,
            G::OUT,
            &mut 0,
            Some(&l),
            &mut quantized,
        )
        [body otherwise unchanged]

DIRECT-CHAIN tests (three; exemplar — the LX-overflow test):

        use crate::work::Gemma12bLxOverflow as G;
        let l = fp8_layout(G::ROWS, G::FEATURES, G::OUT);
        let ops = fp8_quantize_descriptors(
            &Fp8QuantSource::activation(&facts(), G::rows()),
            G::features(),
            G::k_pad_cols(),
            1,
            &mut 0,
            Some(&l),
            &mut quantized,
        )
        [body otherwise unchanged]

PAIR-TAIL tests (three; exemplar — the 26b DOWN tail; the derivation locals all become
G-methods, the stride asserts keep their WIRE arithmetic but read `G::inn_pad()` /
`G::INN` / `G::cl_reach()` instead of locals):

        use crate::work::DownDecodePairs as G;
        assert_eq!(
            G::inn_pad(),
            768,
            "704 rounds up to the next whole 128-elem stick"
        );
        let l = fp8_layout(G::ROWS, G::FEATURES, G::OUT);
        let ops = fp8_quantize_descriptors(
            &Fp8QuantSource::activation(&facts(), G::rows()),
            G::features(),
            G::k_pad_cols(),
            G::PAIRS,
            &mut 0,
            Some(&l),
            &mut quantized,
        )
        [stride/reservation asserts otherwise unchanged, locals → G::inn_pad(), G::INN,
         G::cl_reach(), G::k_pad()]

THE REFUSAL FIXTURE (k=9 over pairs=2): the illegal width is the refusal's own INPUT —
typing it as a legal-looking geometry would be a lie. It stays a number, but a NAMED one
with the doc saying what it is:

    /// The pair law's own refusal input — a width that does not divide into whole per-pair
    /// lane counts. Stated as the fact it is (an illegal width), never as a geometry.
    const NON_WHOLE_LANE_K: u32 = 9;

        let e = fp8_quantize_descriptors(
            &Fp8QuantSource::activation(&facts(), RowCount::of_token_rows(1)),
            BlockCols::of_feature_cols(NON_WHOLE_LANE_K),
            BlockCols::of_feature_cols(2 * crate::work::fp8_k_pad(NON_WHOLE_LANE_K / 2)),
            2,
            &mut 0,
            Some(&l),
            &mut quantized,
        );

════════ F6d — the wide test, COMPLETE (typed plan facts, synth-minted names, footprint-API sizes) ════════

    #[test]
    fn the_wide_prefill_tiles_its_rows_and_splices_the_real_ones_back() {
        use crate::work::Granite2bQkvPrefill as G;
        let plan = G::plan().expect("the granite-2b qkv geometry plans");
        let tiles = plan.tile_count().get() as usize;
        assert_eq!(
            plan.tile_rows().get(),
            crate::work::FP8_GEMM_WIDE_M,
            "the wide class tiles by the ABI tile"
        );
        assert_eq!(
            plan.total_rows().get(),
            G::ROWS.next_multiple_of(crate::work::FP8_GEMM_WIDE_M),
            "21 rows tile by four, padded UP — never a trailing short tile"
        );
        assert!(!plan.rows_run_whole());
        let l = fp8_layout(G::ROWS, G::FEATURES, G::OUT);
        let mut quantized = std::collections::HashSet::new();
        let ops = matmul_fp8_descriptors(
            &facts(),
            G::ROWS,
            G::FEATURES,
            G::OUT,
            &mut 0,
            Some(&l),
            &mut quantized,
        )
        .expect("the wide prefill emits the row-tiled law");
        let n_named = |pfx: &str| {
            ops.iter()
                .filter(|o| o.op_name.starts_with(pfx) && o.op_name.ends_with("_op"))
                .count()
        };
        assert_eq!(n_named("t10_fq_tz"), tiles, "one pad zero-fill per tile");
        assert_eq!(
            n_named("t10_fq_tc"),
            tiles * plan.k_stick_groups().get() as usize,
            "one clone per (tile, K stick group) — 6×32"
        );
        assert_eq!(n_named("t10_fq_qspt"), tiles, "the quant chain runs per tile");
        assert_eq!(n_named("t10_fq_mm_t"), tiles, "one tile mm per tile");
        assert_eq!(n_named("t10_fq_dqa_t"), tiles, "one a-scale dequant leg per tile");
        assert_eq!(n_named("t10_fq_dqw_t"), tiles, "one w-scale dequant leg per tile");
        assert_eq!(
            n_named("t10_fq_to"),
            tiles * plan.n_stick_groups().get() as usize,
            "one out-copy per (tile, N stick group) — 6×48"
        );
        // ⛔ THE TORN STAGING IS GONE: no op carries the DENSE law's shared convert name —
        // every quant tensor and op is tile-indexed, so the quant walk and the mm's A walk
        // spell ONE buffer's layout (the one-mint law).
        assert!(
            !ops.iter().any(|o| o.op_name.ends_with("fq_afp8_op")),
            "the dense law's shared fq_afp8_op must not exist at a tiled geometry"
        );
        // Every tile mm reads ITS OWN staged afp8 — the expected name MINTED through the same
        // PlaceId door the emission used, not a substring guess.
        let t0 = crate::work::Fp8GemmTileIdx::new(0);
        let afp8_t0 = PlaceId::Act(7)
            .synth(t0.fq_role(SynthRole::FqAfp8))
            .to_string();
        for o in ops.iter().filter(|o| o.op_name.starts_with("t10_fq_mm_t")) {
            assert!(
                o.arg_bindings
                    .iter()
                    .any(|b| b.is_input && b.buffer.contains(&afp8_t0)),
                "{}: the tile mm binds the tile's own staged afp8 ({afp8_t0})",
                o.op_name
            );
        }
        // The staging reservations, at the TILE's row class — the class and the widths come
        // from the PLAN and the GEOMETRY through the layout's own footprint API; no hand
        // arithmetic. (k is 128-aligned here, so k_pad == k.)
        let s = l.synth.borrow();
        let a7 = PlaceId::Act(7);
        let staged0 = plan.tile(t0).staged_rows.get();
        let clone_sz = s
            .sizes
            .get(&a7.synth(t0.fq_role(SynthRole::FqTileRow)).to_string())
            .expect("the tile clone is placed");
        assert_eq!(
            *clone_sz,
            crate::placement::synth_footprint_bytes(&[staged0, G::FEATURES], Df::Fp16),
            "tile 0's clone owns [staged_rows, k] fp16"
        );
        let afp8_sz = s
            .sizes
            .get(&a7.synth(t0.fq_role(SynthRole::FqAfp8)).to_string())
            .expect("the tile afp8 is placed");
        assert_eq!(
            *afp8_sz,
            crate::placement::synth_footprint_bytes(&[staged0, G::FEATURES], Df::Fp8),
            "tile 0's afp8 owns [staged_rows, k_pad] fp8 (1 B/elem)"
        );
    }

════════ F6e — the two sharing tests, COMPLETE ════════

    #[test]
    fn a_second_gemm_at_the_same_plan_reuses_the_tile_staging() {
        use crate::work::Granite2bQkvPrefill as G;
        let tiles = G::plan()
            .expect("the granite-2b qkv geometry plans")
            .tile_count()
            .get() as usize;
        let l = fp8_layout(G::ROWS, G::FEATURES, G::OUT);
        let mut quantized = std::collections::HashSet::new();
        let _ = matmul_fp8_descriptors(
            &facts(),
            G::ROWS,
            G::FEATURES,
            G::OUT,
            &mut 0,
            Some(&l),
            &mut quantized,
        )
        .expect("the first GEMM emits");
        let ops2 = matmul_fp8_descriptors(
            &facts_out(11),
            G::ROWS,
            G::FEATURES,
            G::OUT,
            &mut 0,
            Some(&l),
            &mut quantized,
        )
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
            tiles,
            "the per-matmul mm legs still run per tile"
        );
    }

    #[test]
    fn a_second_gemm_at_a_different_row_class_refuses() {
        use crate::work::{Granite2bQkvPrefill as Wide, NarrowSingleTileN as Narrow};
        let l = fp8_layout(Wide::ROWS, Wide::FEATURES, Wide::OUT);
        let mut quantized = std::collections::HashSet::new();
        let _ = matmul_fp8_descriptors(
            &facts(),
            Wide::ROWS,
            Wide::FEATURES,
            Wide::OUT,
            &mut 0,
            Some(&l),
            &mut quantized,
        )
        .expect("the wide GEMM emits and reserves its tiles");
        let e = matmul_fp8_descriptors(
            &facts(),
            Narrow::ROWS,
            Narrow::FEATURES,
            Narrow::OUT,
            &mut 0,
            Some(&l),
            &mut quantized,
        );
        assert!(
            e.is_err(),
            "a second GEMM re-tiling the same activation at another row class is a plan \
             disagreement — refused at bake, never a silent resize"
        );
    }

════════ F8 — the narrow-PADDED row-class test, COMPLETE (the behavior change vs main, NAMED) ════════

    /// ⭐⭐ THE NARROW-PADDED ROW CLASS — a behavior CHANGE vs main, named here: at a narrow
    /// geometry whose rows are neither a small M nor 128-aligned (the reference's `_M_ALIGN`,
    /// `fp8_linear_kernel.py:67`), main's dense arm ran the activation's rows WHOLE at m;
    /// this law pads them UP to the 128 alignment and runs the row-tiled arm — ONE tile of
    /// 128 rows. The clone covers only the REAL rows, the zero-fill covers the pad, and the
    /// out-splice copies only the real rows back: the pad NEVER reaches the `[m, n]` output.
    #[test]
    fn the_narrow_padded_row_class_runs_one_128_row_tile_and_splices_the_real_rows_back() {
        use crate::work::NarrowPadded as G;
        let plan = G::plan().expect("the narrow-padded geometry plans");
        assert!(
            !plan.rows_run_whole(),
            "rows neither small nor aligned do not run whole at a narrow geometry"
        );
        assert_eq!(
            plan.tile_count().get(),
            1,
            "the narrow class pads to ONE whole-alignment tile — never tiles of 4"
        );
        assert_eq!(
            plan.total_rows().get(),
            crate::work::FP8_GEMM_NARROW_M_ALIGN,
            "the rows pad UP to the 128 alignment"
        );
        let l = fp8_layout(G::ROWS, G::FEATURES, G::OUT);
        let mut quantized = std::collections::HashSet::new();
        let ops = matmul_fp8_descriptors(
            &facts(),
            G::ROWS,
            G::FEATURES,
            G::OUT,
            &mut 0,
            Some(&l),
            &mut quantized,
        )
        .expect("the narrow-padded geometry emits the row-tiled law");
        let n_named = |pfx: &str| {
            ops.iter()
                .filter(|o| o.op_name.starts_with(pfx) && o.op_name.ends_with("_op"))
                .count()
        };
        assert_eq!(
            n_named("t10_fq_tz"),
            1,
            "one zero-fill over the whole 128-row tile"
        );
        assert_eq!(
            n_named("t10_fq_tc"),
            plan.k_stick_groups().get() as usize,
            "one clone per K stick group, M = the tile's REAL rows"
        );
        assert_eq!(
            n_named("t10_fq_mm_t"),
            1,
            "one tile mm at the 128-row class"
        );
        assert_eq!(
            n_named("t10_fq_to"),
            plan.n_stick_groups().get() as usize,
            "one out-copy per N stick group, M = the REAL rows"
        );
        // ⛔ THE PAD NEVER REACHES THE OUTPUT: the landing buffer owns the tile's full
        /// 128-row extent (pad rows and all), and the OUTPUT keeps the geometry's own `[m, n]`
        /// placement, byte-for-byte — both stated through the layout's footprint API, no
        /// hand arithmetic.
        let s = l.synth.borrow();
        let landing_sz = s
            .sizes
            .get(
                &PlaceId::Act(10)
                    .synth(crate::work::Fp8GemmTileIdx::new(0).fq_role(SynthRole::FqTileRow))
                    .to_string(),
            )
            .expect("the tile landing is placed");
        assert_eq!(
            *landing_sz,
            crate::placement::synth_footprint_bytes(&[plan.total_rows().get(), G::OUT], Df::Fp16),
            "the landing owns the tile's full [128, n] — pad rows and all"
        );
        assert_eq!(
            l.placements.get(&10).expect("the output is placed").size,
            crate::placement::synth_footprint_bytes(&[G::ROWS, G::OUT], Df::Fp16),
            "the output stays [m, n] — the pad rows never reach it"
        );
    }

════════ CONVERSION TABLE (the remaining tests — same seam as F6c's exemplars; verbatim on request) ════════
· the_quant_reduce_at_a_fused_overflowing_geometry_emits_partials → Gemma12bRung71, direct-chain pattern
· the_26b_k2112_pads_to_a_whole_fp8_stick → Gemma26bDownDecode, door pattern
· an_aligned_k_emits_no_pad_op_and_no_widened_reservation → NarrowAlignedHidden, door pattern
· the_26b_n2112_packed_kernel_out_stick → Gemma26bGateDecode, door pattern
· the_26b_down_tail_pads_per_pair → DownDecodePairs, pair pattern (locals → G-methods)
· an_aligned_inn_keeps_the_whole_tensor_convert → GateUpAlignedPairs, pair pattern
· the_per_pair_tail_refuses_a_non_whole_lane_count_and_pads_per_slot_at_prefill →
  NON_WHOLE_LANE_K fixture + PrefillHalfStickPairs, pair pattern
Reservation-size asserts in the pair tests (`m as u64 * cl_reach as u64 * 2` etc.) become
`synth_footprint_bytes(&[G::ROWS, G::cl_reach()], Df::Fp16)` / `(&[G::ROWS, G::k_pad()], Df::Fp8)`
/ `(&[G::ROWS, G::FEATURES], Df::Fp16)` — the footprint API your F6 prescribed. The
address-DELTA asserts keep their arithmetic (they pin the wire law) but read G-methods.

REMAINING .get()s in tests, named: `plan.tile_count().get()`, `staged_rows.get()`,
`total_rows().get()` feeding usize counts and the footprint API — both u32-shaped. If you
want counts as usize at the source (`Fp8GemmTileCount::get` returning usize is a lie about
units), say so and I will add a `usize()` exit to the counts — I did not unasked.

ORDER: this is the last F-round. After your rulings land and I apply, I tell you, you rsync
+ run the standing legs, then S1-S5 per the S-follows choice, then the order of record
continues (judgment chat vs torch-spyre tokens on card).
