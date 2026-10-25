PROPOSED EXACT CODE — work.rs findings 1-3 (nothing written; this is the hunk set for your approval). HUNK 1 is work.rs, complete. HUNK 2 is the ktir_matmul_fp8.rs call-site ripple HUNK 1 forces (the door + tiled arm only — that file's test typing rides your audit of it, per the standing raw-scalars finding).

════════ HUNK 1a — work.rs, struct doc citation (FINDING 3) ════════
OLD:
/// and N a single ABI width) runs its rows whole, padded to a 128-multiple when not already a
/// small M (`fp8_linear_kernel.py:70-77`, `_SMALL_M`/`_M_ALIGN`). The row count the quantized
NEW:
/// and N a single ABI width) runs its rows whole, padded to a 128-multiple when not already a
/// small M (`fp8_linear_kernel.py:338-346` — `_SMALL_M`/`_M_ALIGN` at their use in
/// `apply_weights`). The row count the quantized

════════ HUNK 1b — work.rs, new typed vocabulary, inserted after `FP8_GEMM_NARROW_M_ALIGN` (before the expert-slab tombstone) ════════
/// ⭐ ONE fp8 GEMM'S TILE COUNT — the number of ABI row tiles the padded row extent splits
/// into. A newtype in the `counts!` idiom, local to this door's vocabulary: a bare `u32` at
/// the seam is the raw-scalar disease, and the count has its own law (`rows × tiles` is the
/// padded total, uniform — never a trailing short tile).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fp8GemmTileCount(u32);

impl Fp8GemmTileCount {
    pub const fn new(v: u32) -> Self {
        Self(v)
    }
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// ⭐ ONE TILE'S INDEX in the row tiling — the tile the clone/quant/mm/out-splice legs name
/// (`SynthRole::FqTiled`'s index takes its value). Minted from the plan's own tile count,
/// never a bare loop counter at a seam.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fp8GemmTileIdx(u32);

impl Fp8GemmTileIdx {
    pub const fn new(v: u32) -> Self {
        Self(v)
    }
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// ⭐ A `[rows, cols]` BUFFER'S fp16 STICK-GROUP COUNT — the clone/out-splice loop extent.
/// Minted by the plan (which refuses a sub-stick width at `of`), never re-derived at a site.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fp8GemmStickGroups(u32);

impl Fp8GemmStickGroups {
    pub const fn new(v: u32) -> Self {
        Self(v)
    }
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// ⭐ ONE fp8 GEMM'S SHAPE FACTS, AS A NAMED TYPE — the only way a TEST states a geometry:
/// `M`/`K`/`N` are const-generic parameters of a NAMED instance (a fact about a model's own
/// GEMM), never bare `u32`s flowing through untyped parameters. The production call states
/// its shapes as the KTIR's own facts, typed at the door's signature (`RowCount`/
/// `BlockCols`); a test has no KTIR, so its geometry is a type — and `plan()` is the same
/// single door either way.
///
/// ⛔ NO VALUE CONSTRUCTOR, EVER: a geometry is named, not built. A runtime-built geometry
/// would be a second spelling of a model fact.
#[derive(Clone, Copy, Debug)]
pub struct Fp8GemmGeometry<const M: u32, const K: u32, const N: u32>;

impl<const M: u32, const K: u32, const N: u32> Fp8GemmGeometry<M, K, N> {
    /// The GEMM's real token rows — the source of every per-tile window.
    pub const fn rows() -> crate::sdsc_abstract::RowCount {
        crate::sdsc_abstract::RowCount::of_token_rows(M)
    }
    /// The contraction width.
    pub fn features() -> crate::sdsc_abstract::BlockCols {
        crate::sdsc_abstract::BlockCols::of_feature_cols(K)
    }
    /// The output width.
    pub fn out_features() -> crate::sdsc_abstract::BlockCols {
        crate::sdsc_abstract::BlockCols::of_feature_cols(N)
    }
    /// The row-tiling plan for this geometry — the ONE door, same as production.
    pub fn plan() -> Result<Fp8GemmRowTiles, String> {
        Fp8GemmRowTiles::of(Self::rows(), Self::features(), Self::out_features())
    }
}

// The named instances — every geometry the tests state, and nothing else:
/// granite-3.1-2b's QKV prefill — THE defect-3 geometry (21 tokens, hidden 2048, fused
/// q+k+v 3072 = 3·1024 ⇒ wide via the split N).
pub type Granite2bQkvPrefill = Fp8GemmGeometry<21, 2048, 3072>;
/// granite-3.1-2b's QKV decode row.
pub type Granite2bQkvDecode = Fp8GemmGeometry<1, 2048, 3072>;
/// granite-3.1-2b's o_proj prefill (2048 = 2·1024 ⇒ wide below 4096).
pub type Granite2bOProjPrefill = Fp8GemmGeometry<21, 2048, 2048>;
/// A narrow small-M control (rows ≤ 4 run whole).
pub type NarrowSmallM = Fp8GemmGeometry<3, 128, 128>;
/// A narrow 128-align control (21 pads up to 128).
pub type NarrowPadded = Fp8GemmGeometry<21, 128, 128>;
/// A sub-granularity N tail (100) — the reference's own `_n_tiles` refusal width.
pub type InexpressibleNTail = Fp8GemmGeometry<21, 2048, 100>;
/// A contraction width that is not a whole fp16 stick (2044 = 31·64 + 60) — the stick-law
/// refusal's control.
pub type SubStickContraction = Fp8GemmGeometry<21, 2044, 128>;
/// gemma-4-26b's gate prefill (n=2112 = 2·1024 + a 64-aligned tail ⇒ one more tile).
pub type Gemma26bGatePrefill = Fp8GemmGeometry<21, 2816, 2112>;
/// gemma-4-26b's gate decode row.
pub type Gemma26bGateDecode = Fp8GemmGeometry<1, 2816, 2112>;
/// gemma-4-26b's DOWN decode row (k=2112, the K-pad law's own width).
pub type Gemma26bDownDecode = Fp8GemmGeometry<1, 2112, 2816>;
/// gemma-4-12b's LX-overflowing reduce geometry (the fused-quant single-shot pin).
pub type Gemma12bLxOverflow = Fp8GemmGeometry<1024, 16384, 128>;
/// gemma-4-12b's rung-71 fused-overflow geometry (the partials+combine pin).
pub type Gemma12bRung71 = Fp8GemmGeometry<71, 15360, 3840>;
/// A narrow small-M control at granite's hidden width (the dense arm's fitting geometry).
pub type NarrowSmallMHidden = Fp8GemmGeometry<4, 2048, 128>;
/// A narrow 128-aligned control at granite's hidden width.
pub type NarrowAlignedHidden = Fp8GemmGeometry<128, 2048, 128>;
/// A narrow single-tile-N control over the SAME activation as the wide QKV plan (the
/// plan-disagreement refusal's geometry).
pub type NarrowSingleTileN = Fp8GemmGeometry<21, 2048, 64>;

════════ HUNK 1c — work.rs, Fp8GemmRowTiles: typed fields, typed `of`, stick refusal (FINDINGS 1+2) ════════
The struct, `of`, and accessors replaced WHOLE (Fp8GemmRowTile unchanged — already typed):

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fp8GemmRowTiles {
    /// The caller's real row count — the source of every per-tile window.
    m: crate::sdsc_abstract::RowCount,
    /// Rows per tile — the A operand's M: a target-ABI row count (1, 4, a small M, or a
    /// 128-multiple), never the caller's raw row count at wide geometries.
    rows: crate::sdsc_abstract::RowCount,
    /// How many such tiles the row extent splits into.
    tiles: Fp8GemmTileCount,
    /// The activation's contraction width — the clone loop's stick-group count is derived
    /// from it here, so no site re-derives it.
    k: crate::sdsc_abstract::BlockCols,
    /// The GEMM's output width — the out-splice loop's stick-group count is derived from it.
    n: crate::sdsc_abstract::BlockCols,
}

impl Fp8GemmRowTiles {
    /// The row tiling for an fp8 GEMM of `m` rows, `k` contraction features, `n` outputs —
    /// the copied `_m_tiles` predicate, keyed on GEOMETRY alone, with the shape facts
    /// arriving as the door's own types (`RowCount`/`BlockCols`: the KTIR's stated facts at
    /// the production call, a named `Fp8GemmGeometry` in a test — never bare `u32`s at a
    /// seam). `wide` is the reference's own test (`fp8_linear_kernel.py:338`): `max(k, n) ≥
    /// 4096` OR the N decomposes over [`FP8_GEMM_N_SET`] into more than one tile. Wide ⇒
    /// `m == 1` stays 1, anything else pads UP to a multiple of 4 and tiles by 4. Narrow ⇒
    /// the rows run whole (`:338-346`, `_SMALL_M`/`_M_ALIGN` at their use in
    /// `apply_weights`): a small M (≤ 4) or a 128-multiple as-is, otherwise padded up to a
    /// 128-multiple. An N tail the ABI set cannot express splits by our own packed-kernel
    /// law, not the reference's refusal — a 64-aligned tail counts as one more tile, a
    /// sub-64 tail is a refusal naming the width (see [`Self::n_tile_count`], the 26b
    /// N-wall's own granularity).
    ///
    /// ⛔ FAIL-CLOSED ON THE STICK LAW: a `k` or `n` that is not a whole fp16 stick is
    /// refused HERE — the clone and out-splice loops walk whole stick groups, and an
    /// unchecked width division would silently truncate a sub-stick tail (the silent-tail
    /// defect class). The refusal is the door's, so the accessors below divide only over a
    /// width the door already proved stick-aligned.
    pub fn of(
        m: crate::sdsc_abstract::RowCount,
        k: crate::sdsc_abstract::BlockCols,
        n: crate::sdsc_abstract::BlockCols,
    ) -> Result<Self, String> {
        let (m, k, n) = (m.get(), k.get(), n.get());
        let stick = crate::superdsc_opspec::Df::Fp16.elems_per_stick();
        for (what, w) in [("k", k), ("n", n)] {
            if w % stick != 0 {
                return Err(format!(
                    "fp8 GEMM {what}={w}: not a whole fp16 stick ({stick} elems) — the row \
                     tiling walks whole stick groups and a sub-stick tail would divide \
                     silently. Refused at the door, never truncated"
                ));
            }
        }
        let n_tiles = Self::n_tile_count(n)?;
        let wide = k.max(n) >= FP8_GEMM_N_SET[0] || n_tiles > 1;
        let (rows, tiles) = if !wide {
            if m <= FP8_GEMM_SMALL_M || m % FP8_GEMM_NARROW_M_ALIGN == 0 {
                (m, 1)
            } else {
                (m.next_multiple_of(FP8_GEMM_NARROW_M_ALIGN), 1)
            }
        } else if m == 1 {
            (1, 1)
        } else {
            let padded = m.next_multiple_of(FP8_GEMM_WIDE_M);
            (FP8_GEMM_WIDE_M, padded / FP8_GEMM_WIDE_M)
        };
        Ok(Self {
            m: crate::sdsc_abstract::RowCount::of_token_rows(m),
            rows: crate::sdsc_abstract::RowCount::of_token_rows(rows),
            tiles: Fp8GemmTileCount::new(tiles),
            k: crate::sdsc_abstract::BlockCols::of_feature_cols(k),
            n: crate::sdsc_abstract::BlockCols::of_feature_cols(n),
        })
    }

    // `n_tile_count` UNCHANGED (already fail-closed).

    /// The A operand's M — one tile's row count, a target-ABI fact by construction.
    pub fn tile_rows(&self) -> crate::sdsc_abstract::RowCount {
        self.rows
    }

    /// How many row tiles the extent covers.
    pub fn tile_count(&self) -> Fp8GemmTileCount {
        self.tiles
    }

    /// The PADDED row total — `rows × tiles`. Rows `[m, total)` are the zero-filled pad.
    pub fn total_rows(&self) -> crate::sdsc_abstract::RowCount {
        crate::sdsc_abstract::RowCount::of_token_rows(self.rows.get() * self.tiles.get())
    }

    /// Tile `t`'s placement facts — the ONE site that computes the tile's row window, so no
    /// clone/quant/mm/out site re-derives it: the first row as the placement vocabulary's own
    /// typed index, and the rows the tile covers in the ACTIVATION (fewer on the last tile —
    /// its clone never reads past the footprint).
    pub fn tile(&self, t: Fp8GemmTileIdx) -> Fp8GemmRowTile {
        let start = t.get() * self.rows.get();
        Fp8GemmRowTile {
            row: crate::addr::Idx::<crate::addr::Row>::n(start),
            real_rows: crate::sdsc_abstract::RowCount::of_token_rows(
                self.rows.get().min(self.m.get().saturating_sub(start)),
            ),
        }
    }

    /// The activation's fp16 stick-group count — the clone loop's extent, minted here (the
    /// door refused a sub-stick `k` at `of`; a site re-deriving `k/64` is the
    /// second-spelling disease).
    pub fn k_stick_groups(&self) -> Fp8GemmStickGroups {
        Fp8GemmStickGroups::new(self.k.get() / crate::superdsc_opspec::Df::Fp16.elems_per_stick())
    }

    /// The output's fp16 stick-group count — the out-splice loop's extent, minted here.
    pub fn n_stick_groups(&self) -> Fp8GemmStickGroups {
        Fp8GemmStickGroups::new(self.n.get() / crate::superdsc_opspec::Df::Fp16.elems_per_stick())
    }
}

════════ HUNK 1d — work.rs tests, converted to named geometries (raw-scalars finding, this file) ════════
Every `Fp8GemmRowTiles::of(..u32..)` call replaced; assertions compare the typed values. Complete replacements:

    #[test]
    fn wide_prefill_rows_tile_by_four_padded_up() {
        let t = Granite2bQkvPrefill::plan().expect("granite QKV prefill geometry");
        assert_eq!(
            t.tile_rows(),
            crate::sdsc_abstract::RowCount::of_token_rows(4),
            "the A operand's M is the ABI's 4, not the chunk's 21"
        );
        assert_eq!(t.tile_count(), Fp8GemmTileCount::new(6));
        assert_eq!(
            t.total_rows(),
            crate::sdsc_abstract::RowCount::of_token_rows(24),
            "rows [21,24) are the zero-filled pad"
        );
    }

    #[test]
    fn decode_is_one_untiled_row() {
        let t = Granite2bQkvDecode::plan().expect("granite QKV decode geometry");
        assert_eq!(
            (
                t.tile_rows(),
                t.tile_count(),
                t.total_rows()
            ),
            (
                crate::sdsc_abstract::RowCount::of_token_rows(1),
                Fp8GemmTileCount::new(1),
                crate::sdsc_abstract::RowCount::of_token_rows(1)
            )
        );
    }

    #[test]
    fn a_split_n_is_wide_below_4096() {
        let t = Granite2bOProjPrefill::plan().expect("o_proj geometry");
        assert_eq!(
            (t.tile_rows(), t.tile_count()),
            (
                crate::sdsc_abstract::RowCount::of_token_rows(4),
                Fp8GemmTileCount::new(6)
            )
        );
    }

    #[test]
    fn narrow_rows_run_whole_at_the_128_alignment() {
        let small = NarrowSmallM::plan().expect("narrow small-M geometry");
        assert_eq!(
            (small.tile_rows(), small.tile_count()),
            (
                crate::sdsc_abstract::RowCount::of_token_rows(3),
                Fp8GemmTileCount::new(1)
            )
        );
        let padded = NarrowPadded::plan().expect("narrow padded geometry");
        assert_eq!(
            (padded.tile_rows(), padded.tile_count(), padded.total_rows()),
            (
                crate::sdsc_abstract::RowCount::of_token_rows(128),
                Fp8GemmTileCount::new(1),
                crate::sdsc_abstract::RowCount::of_token_rows(128)
            )
        );
    }

    #[test]
    fn an_inexpressible_n_refuses() {
        assert!(InexpressibleNTail::plan().is_err());
    }

    /// ⛔ A CONTRACTION WIDTH THAT IS NOT A WHOLE fp16 STICK IS REFUSED — the clone loop
    /// walks whole stick groups, and an unchecked `k/64` would silently truncate a sub-stick
    /// tail (the silent-tail class). The refusal is at the DOOR, so no accessor ever divides
    /// an unchecked width.
    #[test]
    fn a_sub_stick_contraction_width_is_refused() {
        let Err(m) = SubStickContraction::plan() else {
            panic!("k=2044 is not a whole fp16 stick — refused, never truncated");
        };
        assert!(m.contains("not a whole fp16 stick"), "names the stick law: {m}");
    }

    #[test]
    fn the_26b_gate_n_tail_counts_as_one_more_tile() {
        let prefill = Gemma26bGatePrefill::plan().expect("the 26b gate geometry");
        assert_eq!(
            (prefill.tile_rows(), prefill.tile_count()),
            (
                crate::sdsc_abstract::RowCount::of_token_rows(4),
                Fp8GemmTileCount::new(6)
            ),
            "2112 = 2·1024 + one 64-wide tail tile: wide, rows of 4 over a 24-row pad"
        );
        let decode = Gemma26bGateDecode::plan().expect("the 26b gate decode geometry");
        assert_eq!(
            (decode.tile_rows(), decode.tile_count(), decode.total_rows()),
            (
                crate::sdsc_abstract::RowCount::of_token_rows(1),
                Fp8GemmTileCount::new(1),
                crate::sdsc_abstract::RowCount::of_token_rows(1)
            )
        );
    }

    #[test]
    fn tile_windows_and_stick_groups_are_minted_by_the_door() {
        let t = Granite2bQkvPrefill::plan().expect("granite QKV prefill geometry");
        let first = t.tile(Fp8GemmTileIdx::new(0));
        assert_eq!(first.row, crate::addr::Idx::<crate::addr::Row>::n(0));
        assert_eq!(
            first.real_rows,
            crate::sdsc_abstract::RowCount::of_token_rows(4)
        );
        let last = t.tile(Fp8GemmTileIdx::new(5));
        assert_eq!(last.row, crate::addr::Idx::<crate::addr::Row>::n(20));
        assert_eq!(
            last.real_rows,
            crate::sdsc_abstract::RowCount::of_token_rows(1),
            "the last tile covers only the real row — never past the footprint"
        );
        assert_eq!(
            t.k_stick_groups(),
            Fp8GemmStickGroups::new(32),
            "2048 fp16 = 32 stick groups"
        );
        assert_eq!(
            t.n_stick_groups(),
            Fp8GemmStickGroups::new(48),
            "3072 fp16 = 48 stick groups"
        );
    }

════════ HUNK 2 — ktir_matmul_fp8.rs, the call-site ripple HUNK 1 forces (door + tiled arm ONLY; that file's tests wait for your audit of it) ════════
In `matmul_fp8_descriptors` (the door):
OLD:
    let plan = crate::work::Fp8GemmRowTiles::of(m, k, n).map_err(SuperDscError)?;
    if plan.tile_count() == 1 && plan.total_rows() == m {
NEW:
    let plan = crate::work::Fp8GemmRowTiles::of(
        RowCount::of_token_rows(m),
        BlockCols::of_feature_cols(k),
        BlockCols::of_feature_cols(n),
    )
    .map_err(SuperDscError)?;
    if plan.tile_count().get() == 1 && plan.total_rows() == RowCount::of_token_rows(m) {
and the match head:
OLD:
    match plan.tile_rows() {
NEW:
    match plan.tile_rows().get() {

In `matmul_fp8_row_tiled_descriptors`:
OLD:
    let k_groups = plan.k_stick_groups();
    let n_groups = plan.n_stick_groups();
NEW:
    let k_groups = plan.k_stick_groups().get();
    let n_groups = plan.n_stick_groups().get();
OLD:
    for t in 0..plan.tile_count() {
        let tile = plan.tile(t);
NEW:
    for t in 0..plan.tile_count().get() {
        let tile = plan.tile(crate::work::Fp8GemmTileIdx::new(t));

(The `m`/`k`/`n` params of both fns stay `u32` — they are the KTIR-stated facts, typed AT the door call; the door is the boundary.)

Open questions before I write anything:
1. HUNK 1b's named-instance list — right home (work.rs, pub) and right names? `Fp8GemmGeometry` has no value constructor; is the const-generic form what you meant by "typed constant or const-generic instance"?
2. Does HUNK 2 land WITH hunk 1 (compilation requires it), or hold it for the ktir_matmul_fp8.rs audit's own pass?
3. The `FP8_GEMM_SMALL_M` doc cites `fp8_linear_kernel.py:66` and `FP8_GEMM_NARROW_M_ALIGN` cites `:67` (the constants' definitions). I read those as correct-at-definition; say the word if you want them re-pointed at :338-346 too.

Nothing is written until you approve.
