PROPOSED EXACT CODE — F1+F5 (the context struct + the typed class dispatch). Nothing written; this stacks ON TOP of my pending work.rs hunk set (hunk 1 = typed plan; its door-ripple "hunk 2" is SUPERSEDED by the F5 door below). Stacking order when approved: work.rs hunk 1 → this proposal. The emit body's seam lines that F2/F3/F4 will re-type are marked ⚠ so you can see exactly what each round touches.

════════ F5 — work.rs, additions to the pending hunk 1 (after `n_stick_groups`, inside `impl Fp8GemmRowTiles`) ════════
Plus the enum, placed right after the `Fp8GemmGeometry` type-alias block:

/// ⭐ THE ROW CLASS A TILED PLAN TILES ITS ROWS INTO — exactly two variants, because the
/// target ABI has exactly two tiled row classes: the wide GEMM's tile of
/// [`FP8_GEMM_WIDE_M`] rows and the narrow GEMM's [`FP8_GEMM_NARROW_M_ALIGN`]-aligned
/// whole rows. The emission's dispatch is EXHAUSTIVE over this enum with no catch-all:
/// a third class is a compile error at the match, not a runtime refusal arm — and each
/// arm's const-generic `ROWS` is the class's own row count, so the class and the emission
/// context's row extent are one fact.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fp8RowClass {
    /// Wide: tiles of `FP8_GEMM_WIDE_M` rows (prefill rows padded up to a multiple of 4).
    Wide,
    /// Narrow: the whole row extent padded up to the 128 alignment, ONE tile.
    Narrow128,
}

Inside `impl Fp8GemmRowTiles` (after `n_stick_groups`):

    /// The caller's real row count — the source nest's rows and the out-splice's
    /// destination extent. (The plan's own fact: no site re-derives it from the tiling.)
    pub fn real_rows(&self) -> crate::sdsc_abstract::RowCount {
        self.m
    }

    /// The contraction width, as the door's own typed width.
    pub fn contraction_width(&self) -> crate::sdsc_abstract::BlockCols {
        self.k
    }

    /// The output width, as the door's own typed width.
    pub fn out_width(&self) -> crate::sdsc_abstract::BlockCols {
        self.n
    }

    /// ⭐ THE TILED PLAN'S ROW CLASS — the emission's dispatch fact. A dense plan (rows run
    /// whole at `m`) never reaches this door: the emission dispatches it first, and a plan
    /// whose rows are neither ABI class arriving here is a plan invariant break — refused
    /// BY NAME, fail-closed, exactly as `n_tile_count` refuses a width. The refusal is not
    /// a dispatch arm; the dispatch itself is exhaustive over the enum.
    pub fn row_class(&self) -> Result<Fp8RowClass, String> {
        match self.rows.get() {
            FP8_GEMM_WIDE_M => Ok(Fp8RowClass::Wide),
            FP8_GEMM_NARROW_M_ALIGN => Ok(Fp8RowClass::Narrow128),
            rows => Err(format!(
                "fp8 GEMM row class {rows}: not a tiled ABI class — whole-row plans dispatch \
                 to the dense arm before the tiling door, so a third tiled class here is a \
                 plan invariant break. Add the class in `work` first, never a second spelling \
                 at the emission"
            )),
        }
    }

════════ F1+F5 — ktir_matmul_fp8.rs, the door REPLACED WHOLE (supersedes pending hunk 2's door part) ════════
The dense-arm byte-identity test is untouched (your CONFIRMED GOOD). The dispatch:

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
    if plan.tile_count().get() == 1 && plan.total_rows() == RowCount::of_token_rows(m) {
        return matmul_fp8_dense_descriptors(facts, m, k, n, sym_id_base, layout, quantized);
    }
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
}

════════ F1 — ktir_matmul_fp8.rs, `matmul_fp8_row_tiled_descriptors` REPLACED WHOLE by the context ════════
(The `#[allow(clippy::too_many_arguments)]` is gone with the fn — one struct, one method. The doc block moves to the struct. Body is the written tiled body verbatim except: params → destructured fields; `k_pad` → the context's single mint; the ⚠ lines are the seams F2/F3/F4 re-type.)

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
        let Self { facts, plan, m, k, n, k_pad, sym_id_base, layout, quantized } = self;
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
        let ident = layout.and_then(|l| l.identity_tid_of_class(stk)).ok_or_else(|| {
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
            let tile_id = a_id.synth(R::FqTiled(R::FqTileRow, t));
            let tile_name = syn(layout, tile_id);
            // ⭐ THE GEOMETRY GUARD — `synth` is idempotent per name, so a second GEMM over the
            // same activation re-resolves the SAME tile names; if it does so at a different row
            // class (a different plan for the same activation), the footprints disagree and the
            // reuse is refused here, at bake, rather than silently reading the wrong-shaped clone.
            if let Some(l) = layout {
                let want = crate::placement::synth_footprint_bytes(&[ROWS, k], Df::Fp16);
                if let Some(have) = l.synth.borrow().sizes.get(&tile_name).copied()
                    && have != want
                {
                    return Err(SuperDscError(format!(
                        "fp8 matmul {out}: tile {t} of t{} was already reserved at {have} bytes but \
                         this plan needs {want} — two GEMMs over one activation at different row \
                         classes is a plan disagreement; it is not a resize",
                        facts.a_tid
                    )));
                }
                l.synth(tile_id, &[ROWS, k]);
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
                // (2) THE CLONE — one identity matmul per stick group, copying the tile's REAL rows
                // (`M = real`; the source has no rows past `m`, so a full-`ROWS` read would sweep
                // out of footprint). The pad rows keep the zero fill.
                for g in 0..k_groups {
                    ops.push(crate::ir::bridge::tiled_op_sdsc_op::assemble_matmul_placed(
                        &format!("{out}_fq_tc{t}g{g}_op"),
                        crate::sdsc_abstract::MatM::of_token_rows(real),
                        crate::sdsc_abstract::MatN::one_stick(Lanes::FP16),
                        crate::sdsc_abstract::MatK::one_stick(Lanes::FP16),
                        crate::sdsc_abstract::MatY::unbatched(),
                        crate::ir::bridge::tiled_op_sdsc_op::SharedKernelBmmForm::batch_inner_proven(
                            bmm_site::TapeLoweringSite::witness(),
                        ),
                        &a_h,
                        OperandPlacement::at_head(&src_nest, tile.row, g, 0, slabs),
                        &ident_h,
                        DevOff::ZERO,
                        &tile_h,
                        OperandPlacement::at_head(&tile_nest, Idx::<Row>::n(0), g, 0, slabs),
                        sym_id_base,
                        layout,
                    ));
                }
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
            let raw = syn(layout, out_id.synth(R::FqTiled(R::FqRaw, t)));
            let dqa = syn(layout, out_id.synth(R::FqTiled(R::FqDqA, t)));
            let landing = syn(layout, out_id.synth(R::FqTiled(R::FqTileRow, t)));
            if let Some(l) = layout {
                l.synth(out_id.synth(R::FqTiled(R::FqDqA, t)), &[ROWS, n]);
                l.synth(out_id.synth(R::FqTiled(R::FqTileRow, t)), &[ROWS, n]);
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
            // (6) THE OUT-SPLICE — the clone form again, N-side: the tile's real rows of each
            // output stick group land at `tile.row` of the output. The union of the tiles' real
            // rows covers `[0, m)` exactly (the plan's own law), so every output byte is written.
            for c in 0..n_groups {
                ops.push(crate::ir::bridge::tiled_op_sdsc_op::assemble_matmul_placed(
                    &format!("{out}_fq_to{t}c{c}_op"),
                    crate::sdsc_abstract::MatM::of_token_rows(real),
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
                    OperandPlacement::at_head(&out_nest, tile.row, c, 0, slabs),
                    sym_id_base,
                    layout,
                ));
            }
        }
        Ok(ops)
    }
}

NOTE ON F2's k_pad ITEM, so the round-2 proposal is shaped right: the quant chain's k_pad is PAIR-AWARE (`pairs · fp8_k_pad(k/pairs)` — at the 26b DOWN's pairs=8 that is NOT `fp8_k_pad(k)`), and the chain is also called at pairs>1 by the tail tests. My round-2 plan: `fp8_quantize_descriptors` TAKES the minted `k_pad` (the caller mints by the pair law; at pairs==1 — the dense door and this context — that is exactly the context's single mint), so the tiled path has ONE spelling and the pair law has ONE home (the mint site the caller documents). Say if you want a different cut.

ON THE BUILD LEGS: acknowledged — I write nothing until hunks land, and I will tell you the moment the approved set is applied so you can rsync and run the legs at that state. If rustc's 14:02/14:04 errors name my WIP, they are sites on your list (most likely the two `RowCount`/`BlockCols` spellings the pending hunks already re-type) — send the text if you want them folded into the round-2 exact code.

Order after your ruling on this: F2+F3+F4 (with the work.rs tiles()/stick-group iterator law), then F6+F7+F8.
