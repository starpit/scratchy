# 11 — D1-D3: reference-fidelity hunks against the pinned spec (S1-S7)

Per the pinned spec and the redirect ("implement precisely what torch-spyre does"). I
re-verified every citation in `spyre-inference/spyre_inference/custom_ops/fp8_linear_kernel.py`
myself before drafting: S1 :338, S2 :70-77/:266-271/:340-341, S3 :343-346, S4 :80-91
(remainder `RuntimeError` at :90, no other arm), S5 :348-367, S6 :244-263, S7 :99-101.
Stacks on the frozen tree (01+01a, 02+02a applied; 03/04 not applied). Nothing lands until
your line-level ruling, as before.

Fidelity check on the models in scope (greedy decomposition over {4096, 1024, 128}):
granite-2b (3072/2048/512/11008), granite-8b (4096/12800), gemma-3-12b (8192/15360/3840)
— all decompose with ZERO remainder, S4-unchanged; the judgment model is untouched.
TWO 26b widths refuse under S4 and under their `:90`: the gate's 2112 (= 2·1024 + 64)
and the EXPERT w1/w3 output width 704 (= 5·128 + 64) — both leave remainder 64.
Both are widths torch-spyre never runs through this kernel (the 26b is bf16 there), so
both refusals are fidelity, not regression. The experts' door membership is unresolved
either way: if the MoE lowers through the open fused/fold path rather than this door,
n=704 never reaches `n_tile_count` — the refusal is then dormant for it, not absent.

════════ D1 — work.rs, `n_tile_count`: the packed-gran arm DELETED, S4 is the whole law ════════

Doc + body replaced whole:

OLD:
```rust
    /// How many ABI tiles `n` decomposes into over [`FP8_GEMM_N_SET`], largest first — the
    /// reference's `_n_tiles` (`fp8_linear_kernel.py:80`). A remainder the reference refuses
    /// (`:90`) splits TWO ways here, because our packed kernel is wider than their N staging:
    /// a [`FP8_PACKED_KERNEL_OUT_GRAN`]-aligned tail (the 26b gate's 2112 = 2·1024 + 64) is a
    /// width our retile law already accepts — the 26b N-wall's own law — so it counts as ONE
    /// MORE tile (wide; the rows tile by 4) instead of a refusal; a sub-granularity tail
    /// (n=100) is a width the packed kernel itself cannot stage, and THAT is the refusal.
    fn n_tile_count(n: u32) -> Result<u32, String> {
        let mut left = n;
        let mut tiles = 0;
        for &size in FP8_GEMM_N_SET.iter() {
            tiles += left / size;
            left %= size;
        }
        if left == 0 {
            return Ok(tiles);
        }
        if left % FP8_PACKED_KERNEL_OUT_GRAN == 0 {
            return Ok(tiles + 1);
        }
        Err(format!(
            "fp8 GEMM N={n} leaves remainder {left}: not expressible over the \
             target-ABI N set {FP8_GEMM_N_SET:?} nor as a {FP8_PACKED_KERNEL_OUT_GRAN}-aligned \
             packed-kernel tail (the reference's _n_tiles refuses the same width, \
             fp8_linear_kernel.py:90)"
        ))
    }
```
NEW:
```rust
    /// How many ABI tiles `n` decomposes into over [`FP8_GEMM_N_SET`], largest first —
    /// the reference's `_n_tiles` (`fp8_linear_kernel.py:80-91`) and the WHOLE law: `n`
    /// already in the set is one tile, anything else decomposes greedily over
    /// (4096, 1024, 128), and a remainder is the reference's own `RuntimeError` (`:90` —
    /// "N={n} remainder {left} is not in _WIDE_N"). There is no other acceptance arm:
    /// where our packed kernel could stage a width theirs refuses, the defect is in our
    /// lowering or kernel, never a width law of ours (user arch — fixes in the lowering,
    /// never invention at the ABI).
    fn n_tile_count(n: u32) -> Result<u32, String> {
        let mut left = n;
        let mut tiles = 0;
        for &size in FP8_GEMM_N_SET.iter() {
            tiles += left / size;
            left %= size;
        }
        if left == 0 {
            return Ok(tiles);
        }
        Err(format!(
            "fp8 GEMM N={n} remainder {left} is not in {FP8_GEMM_N_SET:?} — the reference's \
             own refusal (fp8_linear_kernel.py:90). There is no other acceptance arm; where \
             our packed kernel could stage this width, the defect is in our lowering or \
             kernel, never a width law of ours"
        ))
    }
```

`of()`'s doc, tail sentence REPLACED (the only other statement of the dead arm):
OLD:
```rust
    /// 128-multiple. An N tail the ABI set cannot express splits by our own packed-kernel
    /// law, not the reference's refusal — a 64-aligned tail counts as one more tile, a
    /// sub-64 tail is a refusal naming the width (see [`Self::n_tile_count`], the 26b
    /// N-wall's own granularity).
```
NEW:
```rust
    /// 128-multiple. An N the ABI set cannot express is the reference's own refusal
    /// (`fp8_linear_kernel.py:90`), carried whole — see [`Self::n_tile_count`]: there is
    /// no other acceptance arm, ours or theirs.
```

[`FP8_PACKED_KERNEL_OUT_GRAN`] itself SURVIVES unchanged — its remaining consumers are the
retile guard and `emit/mod.rs`'s per-core validation (the kernel's own staging granularity,
verified by grep: work.rs:263 def, emit/mod.rs:307-313 use; n_tile_count was its only
work.rs use). No dead-code risk.

════════ D2a — work.rs, `Fp8RowClass::Narrow` carries the extent ════════

OLD:
```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fp8RowClass {
    /// Wide: tiles of `FP8_GEMM_WIDE_M` rows (prefill rows padded up to a multiple of 4).
    Wide,
    /// Narrow: the whole row extent padded up to the 128 alignment, ONE tile.
    Narrow128,
}
```
NEW:
```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fp8RowClass {
    /// Wide: tiles of `FP8_GEMM_WIDE_M` rows (prefill rows padded up to a multiple of 4) —
    /// a true const of the reference's ABI (`_m_tiles`, `fp8_linear_kernel.py:76-77`).
    Wide,
    /// Narrow: the whole row extent as ONE part at the FULL padded 128-multiple — ANY
    /// multiple, no upper bound (S3, `fp8_linear_kernel.py:343-346`: 21→128, 192→256,
    /// 256→256 unpadded). The extent is a runtime `RowCount`: it is a fact of the input,
    /// not of the ABI, so the class carries it instead of a const generic —
    /// `Fp8TileStaging::<ROWS>` cannot state it, and the reference refuses no M anywhere
    /// in the file.
    Narrow(crate::sdsc_abstract::RowCount),
}
```

════════ D2b + D3 — work.rs, `row_class`: branch 1 DEAD, the census doc dies with it ════════

OLD (the whole method incl. doc):
```rust
    /// ⭐ THE TILED PLAN'S ROW CLASS — the emission's dispatch fact. A dense plan (rows run
    /// whole at `m`) never reaches this door: the emission dispatches it first, and a plan
    /// whose rows are neither ABI class arriving here is a plan invariant break — refused
    /// BY NAME, fail-closed, exactly as `n_tile_count` refuses a width. The refusal is not
    /// a dispatch arm; the dispatch itself is exhaustive over the enum.
    ///
    /// ⛔ THE NARROW ARM EXISTS AT EXTENT 128 ONLY — narrow padded extents above 128 rows
    /// are not emitted. Census (02a, `crates/models/arch/configs/`): no in-scope GEMM is
    /// confirmed narrow at all (granite 2b/8b and gemma-3-12b: every projection wide by
    /// split-N or ≥4096); gemma-4-26b has two UNRESOLVED candidates (global-kv n=1024,
    /// MoE router n=128) whose door membership rides the open gemma-4-attention and MoE
    /// lowerings — if either lands, the 192 prefill rung pads to 256 and needs the class
    /// generalized in `work` first. A model that does needs exactly that: the class
    /// generalized in `work`, never a second spelling at the emission.
    pub fn row_class(&self) -> Result<Fp8RowClass, String> {
        match self.rows.get() {
            FP8_GEMM_WIDE_M => Ok(Fp8RowClass::Wide),
            FP8_GEMM_NARROW_M_ALIGN => Ok(Fp8RowClass::Narrow128),
            rows => Err(format!(
                "fp8 GEMM row class {rows}: not a tiled ABI class — whole-row plans dispatch \
                 to the dense arm before the tiling door, and narrow padded extents above 128 \
                 rows are not emitted (no in-scope GEMM produces one; see row_class's doc for \
                 the census and the 26b candidates). Generalize the class in `work` first, \
                 never a second spelling at the emission"
            )),
        }
    }
```
NEW:
```rust
    /// ⭐ THE TILED PLAN'S ROW CLASS — the emission's dispatch fact. A dense plan (rows run
    /// whole at `m` — the small-M and 128-aligned inputs the reference runs unpadded,
    /// `fp8_linear_kernel.py:343-346`) never reaches this door: the emission dispatches it
    /// first. What remains is exactly the reference's two tiled forms, and the reference
    /// refuses NO M anywhere in the file: the wide tile of 4 (`_m_tiles`, `:76-77`) and the
    /// narrow single part at the padded extent (`:344-345` — pad UP to the next
    /// 128-multiple, ONE part at ANY multiple).
    pub fn row_class(&self) -> Fp8RowClass {
        match self.rows.get() {
            FP8_GEMM_WIDE_M => Fp8RowClass::Wide,
            _ => Fp8RowClass::Narrow(self.rows),
        }
    }
```
(Infallible now — S3 has no M refusal to carry. Reachability, verified against `of()`:
wide m≥2 ⇒ rows=4; narrow unaligned m>4 ⇒ rows = a 128-multiple ≥ 128; the dense arm
intercepts m=1, small-M, and already-aligned m before this door, so the `_` arm only ever
sees a narrow extent.)

════════ D2c — work.rs, alias docs + the 192-rung control (REPLACES 04a's T5 refusal pin) ════════

The two 26b gate aliases, docs REPLACED (the run pins die with the arm):
OLD:
```rust
/// gemma-4-26b's gate prefill (n=2112 = 2·1024 + a 64-aligned tail ⇒ one more tile).
pub type Gemma26bGatePrefill = Fp8GemmGeometry<21, 2816, 2112>;
/// gemma-4-26b's gate decode row.
pub type Gemma26bGateDecode = Fp8GemmGeometry<1, 2816, 2112>;
```
NEW:
```rust
/// The 26b gate prefill — the reference's own `_n_tiles` REFUSAL control (n=2112 =
/// 2·1024 + 64; `fp8_linear_kernel.py:90`). torch-spyre never runs this width through
/// this kernel (the 26b is bf16 there), so the refusal is fidelity, not a regression.
pub type Gemma26bGatePrefill = Fp8GemmGeometry<21, 2816, 2112>;
/// The 26b gate decode row — the same refusal control at m=1.
pub type Gemma26bGateDecode = Fp8GemmGeometry<1, 2816, 2112>;
```
NEW ALIAS (immediately after `NarrowPadded`):
```rust
/// The 192-rung narrow control — m=192 pads UP to 256 and RUNS (S3,
/// `fp8_linear_kernel.py:343-346`: one part at ANY 128-multiple, no upper bound, no
/// refusal on M). Replaces 04a's T5 refusal pin, whose premise the redirect overturned.
pub type NarrowHighUnaligned = Fp8GemmGeometry<192, 128, 128>;
```

════════ D2d — work.rs tests ════════

`the_26b_gate_n_tail_counts_as_one_more_tile` REPLACED WHOLE:
```rust
    /// ⛔ THE 26b GATE WIDTH (2112 = 2·1024 + 64) IS THE REFERENCE'S OWN REFUSAL —
    /// `_n_tiles` raises at the remainder (`fp8_linear_kernel.py:90`), and there is no
    /// other acceptance arm: where our packed kernel could stage a width theirs refuses,
    /// the defect is in our lowering or kernel, never a width law of ours. torch-spyre
    /// never runs this width through this kernel (the 26b is bf16 there) — the refusal is
    /// fidelity, not a regression.
    #[test]
    fn the_26b_gate_n_tail_is_the_references_own_refusal() {
        let Err(m) = Gemma26bGatePrefill::plan() else {
            panic!("n=2112 leaves remainder 64 — the reference refuses it (:90), and so do we");
        };
        assert!(m.contains("fp8_linear_kernel.py:90"), "cites the reference's refusal: {m}");
        let Err(m) = Gemma26bGateDecode::plan() else {
            panic!("n=2112 leaves remainder 64 — the reference refuses it (:90), and so do we");
        };
        assert!(m.contains("fp8_linear_kernel.py:90"), "cites the reference's refusal: {m}");
    }
```
NEW TEST (immediately after `narrow_rows_run_whole_at_the_128_alignment`):
```rust
    /// ⭐ THE NARROW LAW (S3, `fp8_linear_kernel.py:343-346`): ONE part at the FULL padded
    /// 128-multiple — ANY multiple, no upper bound, no refusal on M anywhere in the file.
    /// 21 pads to 128; 192 pads to 256 — exactly the extent the overturned branch-1
    /// refusal once rejected, now run as the reference runs it.
    #[test]
    fn narrow_rows_pad_to_any_128_multiple_and_run() {
        let t = NarrowPadded::plan().expect("narrow padded geometry");
        assert_eq!(
            t.row_class(),
            Fp8RowClass::Narrow(crate::sdsc_abstract::RowCount::of_token_rows(128))
        );
        let t = NarrowHighUnaligned::plan().expect("the 192-rung geometry");
        assert_eq!(
            t.total_rows(),
            crate::sdsc_abstract::RowCount::of_token_rows(256),
            "192 pads UP to 256 — any 128-multiple, no upper bound"
        );
        assert_eq!(
            t.row_class(),
            Fp8RowClass::Narrow(crate::sdsc_abstract::RowCount::of_token_rows(256))
        );
    }
```
(`narrow_rows_run_whole_at_the_128_alignment` and `an_inexpressible_n_refuses` stand
unchanged — small-M/aligned dense dispatch and the `is_err` pin both still hold.)

════════ D2e — sdsc_abstract.rs, `Fp8NarrowStaging` (NEW, immediately after `Fp8TileStaging`'s impl) ════════

```rust
/// ⭐ THE NARROW fp8 STAGING — [`Fp8TileStaging`]'s runtime-rows twin, for the narrow
/// law's ONE part at the padded 128-multiple (S3, `fp8_linear_kernel.py:343-346`): the
/// extent is a fact of the INPUT (21→128, 192→256 — any 128-multiple), so it is carried
/// as a [`RowCount`] the constructor pins, not a const generic. Same seam law as the tile
/// staging: the handles are the quantize chain's own mints, and a handle whose row count
/// is not the stated extent is exactly the writer/reader drift the type refuses.
#[derive(Clone, Debug)]
pub struct Fp8NarrowStaging {
    rows: RowCount,
    afp8: Stk<RowBlockedTag>,
    ascale: Stk<RowBlockedTag>,
}

impl Fp8NarrowStaging {
    /// The staging's row extent — the padded 128-multiple the quant wrote and every
    /// consumer must read.
    pub fn rows(&self) -> RowCount {
        self.rows
    }

    /// The quantized activation handle — the GEMM's A operand, minted by the quantize that
    /// wrote it (`row_blocked_df(rows, k_pad, Fp8)`).
    pub fn afp8(&self) -> &Stk<RowBlockedTag> {
        &self.afp8
    }

    /// The per-token scale handle — the dequant's row input (`[rows, one stick]`).
    pub fn ascale(&self) -> &Stk<RowBlockedTag> {
        &self.ascale
    }

    /// The one constructor — the quantize emission's output mint, at the part's stated
    /// extent. REFUSES a handle whose row count is not that extent, for the same reason
    /// [`Fp8TileStaging::staged`] refuses one at another class.
    pub(crate) fn staged(
        rows: RowCount,
        afp8: Stk<RowBlockedTag>,
        ascale: Stk<RowBlockedTag>,
    ) -> Result<Self, String> {
        for (what, h) in [("afp8", &afp8), ("ascale", &ascale)] {
            if h.layout().rows as u32 != rows.get() {
                return Err(format!(
                    "Fp8NarrowStaging::staged: the {what} handle carries {} rows — the \
                     staging's extent is {}, and a handle at another extent is exactly the \
                     writer/reader drift this type refuses",
                    h.layout().rows,
                    rows.get()
                ));
            }
        }
        Ok(Self { rows, afp8, ascale })
    }
}
```

════════ D2f — ktir_matmul_fp8.rs, the door's narrow arm ════════

Import line gains `Fp8NarrowStaging`:
OLD: `use crate::sdsc_abstract::{`
     `    BlockCols, Fp8TileStaging, KernelTag, Lanes, OperandPlacement, RowCount, Stk,`
     `};`
NEW: `use crate::sdsc_abstract::{`
     `    BlockCols, Fp8NarrowStaging, Fp8TileStaging, KernelTag, Lanes, OperandPlacement,`
     `    RowCount, Stk,`
     `};`

The dispatch (the comment's extent sentence restated for the runtime arm):
OLD:
```rust
    // ⭐ THE DISPATCH IS EXHAUSTIVE OVER THE CLASS — exactly two tiled row classes exist,
    // the enum says so, and a third is a COMPILE ERROR at this match (no catch-all arm, no
    // dead-looking runtime refusal). Each arm's const-generic `ROWS` is the class's own row
    // count: the emission context's row extent and the plan's class are one fact.
    match plan.row_class().map_err(SuperDscError)? {
        crate::work::Fp8RowClass::Wide => {
            Fp8TileEmit::<crate::work::FP8_GEMM_WIDE_M>::new(facts, &plan, sym_id_base, layout, quantized)?.emit()
        }
        crate::work::Fp8RowClass::Narrow128 => {
            Fp8TileEmit::<crate::work::FP8_GEMM_NARROW_M_ALIGN>::new(facts, &plan, sym_id_base, layout, quantized)?.emit()
        }
    }
```
NEW:
```rust
    // ⭐ THE DISPATCH IS EXHAUSTIVE OVER THE CLASS — exactly two tiled row classes exist,
    // the enum says so, and a third is a COMPILE ERROR at this match (no catch-all arm, no
    // dead-looking runtime refusal). Each arm states its own row extent: the wide tile's 4
    // is a const of the reference's ABI (`_m_tiles`, fp8_linear_kernel.py:76-77), and the
    // narrow part's padded 128-multiple is a fact of the input carried as a `RowCount`
    // (S3, :343-346 — one part at ANY multiple, no refusal on M).
    match plan.row_class() {
        crate::work::Fp8RowClass::Wide => {
            Fp8TileEmit::<crate::work::FP8_GEMM_WIDE_M>::new(facts, &plan, sym_id_base, layout, quantized)?.emit()
        }
        crate::work::Fp8RowClass::Narrow(padded) => {
            Fp8NarrowEmit::new(facts, &plan, padded, sym_id_base, layout, quantized)?.emit()
        }
    }
```

════════ D2g — ktir_matmul_fp8.rs, `fp8_narrow_mm_op` (NEW, immediately after `fp8_tile_mm_op`) ════════

```rust
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
    let raw = syn(layout, out_id.synth(SynthRole::FqTiled(SynthRole::FqRaw, 0)));
    if let Some(l) = layout {
        l.synth(out_id.synth(SynthRole::FqTiled(SynthRole::FqRaw, 0)), &[rows, n]);
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
    Ok(emit_sdsc_tiled(
        &format!("{out}_fq_mm_narrow_op"),
        &spec,
        &folds,
        sym_id_base,
        layout,
    )?)
}
```

════════ D2h — ktir_matmul_fp8.rs, `Fp8NarrowEmit` (NEW, immediately after `Fp8TileEmit`'s impl) ════════

```rust
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
        let Self { facts, plan, m, rows, k, n, k_pad, sym_id_base, layout, quantized } = self;
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
        let ident = layout.and_then(|l| l.identity_tid_of_class(stk)).ok_or_else(|| {
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
        let tile_id = a_id.synth(R::FqTiled(R::FqTileRow, t));
        let tile_name = syn(layout, tile_id);
        // ⭐ THE GEOMETRY GUARD — `synth` is idempotent per name, so a second GEMM over the
        // same activation re-resolves the SAME part names; if it does so at a different row
        // class (a different plan for the same activation), the footprints disagree and the
        // reuse is refused here, at bake, rather than silently reading the wrong-shaped part.
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
            l.synth(tile_id, &[rows, k]);
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
                &syn(layout, a_id.synth(R::FqTiled(R::FqAfp8, t))),
                r_rows,
                BlockCols::of_feature_cols(k_pad),
                Df::Fp8,
            ),
            rb(
                &syn(layout, a_id.synth(R::FqTiled(R::FqAscale, t))),
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
        let raw = syn(layout, out_id.synth(R::FqTiled(R::FqRaw, t)));
        let dqa = syn(layout, out_id.synth(R::FqTiled(R::FqDqA, t)));
        let landing = syn(layout, out_id.synth(R::FqTiled(R::FqTileRow, t)));
        if let Some(l) = layout {
            l.synth(out_id.synth(R::FqTiled(R::FqDqA, t)), &[rows, n]);
            l.synth(out_id.synth(R::FqTiled(R::FqTileRow, t)), &[rows, n]);
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
```

## 04a T5 amendment (paper, rides this ruling)

04a's T5 (`NarrowHighUnaligned` pinning `row_class()`'s Err) is REPLACED by D2c/D2d above:
the same alias now pins the RUN (192→256, `Fp8RowClass::Narrow(256)`). 04's application
takes D2's forms; no separate T5 hunk lands.

## Standing-question answers for every hunk above

- D1: `_n_tiles` `:80-91` — remainder → RuntimeError `:90`; no other arm (their code, read).
- D2a/b: `_m_tiles` `:70-77` (wide tile of 4 is their const) + `:343-346` (narrow pads to
  any 128-multiple, one part, no M refusal anywhere in the file).
- D2e-h: S5's per-part structure `:348-367` (clone at the part extent, mm at the part
  extent, `out[:orig_m]` splice) at S3's padded extent; the staging/mm/context TYPES are
  our ladder's mechanism wrapping those laws (your ruling: mechanism is the lowering's,
  laws are the spec's).

Nothing lands until your line-level ruling.
