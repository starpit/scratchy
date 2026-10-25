PROPOSED EXACT CODE — ROUND 2: F2+F3+F4 (typed plan surfaces). Stacks on: pending work.rs hunk 1 → pending F1/F5 → THIS. Where an item below also exists in hunk 1 or F1/F5, this text SUPERSEDES. Nothing is written until you rule; per-item rulings welcome (each is independent except the emit body, which needs them all).

════════ A0 — addr/mod.rs, ONE addition (a NINTH file — needs your line audit; it is 6 lines) ════════
Idx has no Display, and F4's typed group windows must render into op names (the wire-format
literals your S1 ruling lets stay). Rendered index = the number, nothing more:

impl<A: Axis> std::fmt::Display for Idx<A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

(Field `0` is private-in-module — the impl sits in addr/mod.rs beside the type. `Idx<A>` is
`Copy + Debug` already; no other trait changes.)

════════ A1 — work.rs, Fp8GemmTileIdx (SUPERSEDES hunk 1's item; now with Display + the role mint) ════════

/// ONE TILE'S ORDINAL — the [`SynthRole::FqTiled`] payload and the op-name rendering. Its
/// only exits are TYPED: the role mint ([`Self::fq_role`]) and [`Display`] for op names,
/// so no emission site counts or spells a raw tile index.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Fp8GemmTileIdx(u32);

impl Fp8GemmTileIdx {
    pub const fn new(t: u32) -> Self {
        Self(t)
    }
    pub const fn get(self) -> u32 {
        self.0
    }
    /// The tile's view of a quant-chain role — [`SynthRole::FqTiled`]-wrapped, this ordinal
    /// the payload. ONE site wraps; no emission site spells the payload integer, so the tile
    /// index and the role can never disagree.
    pub fn fq_role(self, r: crate::place::SynthRole) -> crate::place::SynthRole {
        crate::place::SynthRole::FqTiled(r, self.get())
    }
}

impl std::fmt::Display for Fp8GemmTileIdx {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

════════ A2 — work.rs, Fp8GemmRowTile (SUPERSEDES hunk 1's item: + idx, + staged_rows) ════════

/// ONE TILE'S PLACEMENT FACTS — minted by [`Fp8GemmRowTiles::tile`]/[`Fp8GemmRowTiles::tiles`],
/// never re-derived at a site. Carries EVERY fact the emission reads off a tile: its ordinal
/// (the role payload and op-name rendering), its first row as the placement vocabulary's own
/// `Idx<Row>`, the rows it actually covers in the activation (the last tile's are fewer — its
/// clone never reads past the footprint), and the rows its STAGING carries (the ABI row class —
/// the count the quant chain and the tile mm sweep, the clone writing only the real rows and
/// the zero fill the rest).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fp8GemmRowTile {
    /// The tile's ordinal.
    pub idx: Fp8GemmTileIdx,
    /// The tile's first row in the activation — for `OperandPlacement`'s row coordinate.
    pub row: crate::addr::Idx<crate::addr::Row>,
    /// The rows this tile covers IN THE ACTIVATION (the clone and out-splice ops' M).
    pub real_rows: crate::sdsc_abstract::RowCount,
    /// The rows the tile's STAGING carries — the ABI row class, `rows × tiles` padded total.
    pub staged_rows: crate::sdsc_abstract::RowCount,
}

════════ A3 — work.rs, inside `impl Fp8GemmRowTiles` (SUPERSEDES hunk 1's `tile`; + tiles, + windows, + rows_run_whole) ════════

    /// Tile `t`'s placement facts — the ONE site that computes the tile's row window, so no
    /// clone/quant/mm/out site re-derives it.
    pub fn tile(&self, t: Fp8GemmTileIdx) -> Fp8GemmRowTile {
        let start = t.get() * self.rows;
        Fp8GemmRowTile {
            idx: t,
            row: crate::addr::Idx::<crate::addr::Row>::n(start),
            real_rows: crate::sdsc_abstract::RowCount::of_token_rows(
                self.rows.min(self.m.saturating_sub(start)),
            ),
            staged_rows: crate::sdsc_abstract::RowCount::of_token_rows(self.rows),
        }
    }

    /// The plan's tiles, as minted values — the emission's ONLY tile loop: no site counts
    /// tile indices, the iterator hands each tile its own facts.
    pub fn tiles(&self) -> impl Iterator<Item = Fp8GemmRowTile> + '_ {
        let tiles = self.tiles;
        (0..tiles).map(Fp8GemmTileIdx::new).map(|t| self.tile(t))
    }

    /// The contraction's stick groups as the nests' own head coordinates — the clone walks'
    /// `g`, minted: no emission loop counts a raw group index. (The count door
    /// [`Self::k_stick_groups`] stays for NEST EXTENTS, which are u32-shaped.)
    pub fn k_stick_group_windows(&self) -> impl Iterator<Item = crate::addr::Idx<crate::addr::Head>> {
        (0..self.k_stick_groups().get()).map(crate::addr::Idx::<crate::addr::Head>::n)
    }

    /// The output's stick groups as the nests' own head coordinates — the out-splice walks'
    /// `c`, minted.
    pub fn n_stick_group_windows(&self) -> impl Iterator<Item = crate::addr::Idx<crate::addr::Head>> {
        (0..self.n_stick_groups().get()).map(crate::addr::Idx::<crate::addr::Head>::n)
    }

    /// Whether the rows run WHOLE at the caller's own count — the dense arm's own test,
    /// minted here as a named fact so the door does not re-derive tiling arithmetic. Exactly
    /// `tile_count() == 1 ∧ total_rows() == m` (your CONFIRMED GOOD dense-arm byte-identity
    /// test, same predicate, one spelling).
    pub fn rows_run_whole(&self) -> bool {
        self.tiles == 1 && self.rows == self.m
    }

════════ B1 — sdsc_abstract.rs, MatM::of_token_rows RETYPED (F3; complete item) ════════
Caller inventory (complete, grep-verified): matmul/opspec.rs:36, ktir_matmul_fp8.rs :704/:850/:1004/:1109. No others.

    /// A node's token rows — the dense projections, the lm-head, the fp8 W8A8 chain. Takes
    /// the typed count, so a padded row count, a shared-buffer extent or a lane count cannot
    /// fill this row slot.
    pub const fn of_token_rows(rows: RowCount) -> MatM {
        MatM(rows.get())
    }

════════ B2 — sdsc_abstract.rs, Fp8TileStaging gains its typed row count (one associated const) ════════
(Inside the existing `impl<const R: u32> Fp8TileStaging<R>` — beside `ROWS`.)

    /// `ROWS` as the TYPED count — the tile mm's M mint, so the mm's M and the staging's
    /// row class are one fact end to end (the mm already reads A from this staging type,
    /// whose `staged` refuses any handle at another class).
    pub const STAGED_ROWS: RowCount = RowCount::of_token_rows(R);

════════ B3 — sdsc_abstract.rs, OperandPlacement::at_head's head coordinate RETYPED (F4's typed windows) ════════
Caller inventory (complete, grep-verified): ktir_matmul_fp8.rs ×4 (mine), lower_ktir_to_superdsc.rs:5994 (ONE legacy site — the cachewr precedent; the mint below is a one-line change in the SpraysFirst WIP file, so it rides your S-coordination ruling on who lands that file).

    /// [`Self::of_plane_walk_by_head`] AT ONE HEAD, for a copy that cannot put the heads on `y`.
    ///
    /// The head is a COORDINATE of the offset here, so the emitted op needs no batch axis
    /// ([`MatY::unbatched`]) and `head_stride` is never read. Required where one op spans several ROWS:
    /// the plane-walk pitch describes head-outermost planes of `rows` rows, which only agrees with a
    /// row-outermost source (`row*(nkvh*hd) + head*hd`) when an op covers a single row.
    pub fn at_head(
        nest: &crate::addr::Nest,
        row: crate::addr::Idx<crate::addr::Row>,
        head: crate::addr::Idx<crate::addr::Head>,
        s: u32,
        slabs: crate::addr::Slabs,
    ) -> OperandPlacement {
        OperandPlacement {
            off: nest.view().at(row).at(head).slab(s).dev(),
            pitch: PhysM(nest.rows() * slabs.get()),
            head_stride: nest.span(
                crate::addr::Idx::<crate::addr::Head>::n(0),
                crate::addr::Idx::<crate::addr::Head>::n(1),
            ),
        }
    }

The ONE legacy caller's mint (lower_ktir_to_superdsc.rs:5994-6000, the `else` arm only):

                        crate::sdsc_abstract::OperandPlacement::at_head(
                            &src_nest,
                            crate::addr::Idx::<crate::addr::Row>::n(row0),
                            crate::addr::Idx::<crate::addr::Head>::n(h),
                            s,
                            slabs,
                        )

════════ D1 — ktir_matmul_fp8.rs, Fp8QuantSource RETYPED (F2; complete replacement of struct + impl) ════════
Doc block keeps its ⭐⭐⭐ content; the changed sentences are the source-shape ones:

/// ⭐ THE QUANT CHAIN'S SOURCE — the activation whole (the dense law) or ONE minted row
/// tile's fresh clone (the row-tiled law, [`crate::work::Fp8GemmRowTiles`]). The chain is a
/// pure function of its source's `[rows, k]` window, so ONE body serves both laws; what
/// changes is WHERE it reads and WHAT its tensors are named — a tile's quant tensors carry
/// their tile index ([`SynthRole::FqTiled`]), keyed off the ACTIVATION's id either way
/// because a tile's staging is a fact of the activation that tiled, and its ops carry the
/// index in their names. The `quantized` sharing key is the source's own name, so a second
/// GEMM over the same activation at the same plan reuses one staging instead of
/// re-quantizing per matmul. The row extent is DERIVED from the extent below, never stated
/// beside it: a tile's extent is the tile's own staged class, and no caller can name a
/// count the buffer does not have.
pub(crate) struct Fp8QuantSource {
    /// The tensor the chain reads — the activation, or the tile's fp16 clone.
    pub name: String,
    /// The id every chain synth derives from — the ACTIVATION's, either way.
    pub id: PlaceId,
    /// WHAT the source is — the activation whole, or one minted tile.
    extent: Fp8QuantExtent,
    /// The output tensor's id — the op-name namespace, shared by both laws.
    pub out_tid: u32,
}

/// The source's KIND — every fact the chain varies by (row extent, role wrapping, op-name
/// ordinal) reads from here, so the two laws cannot drift apart in any of them.
enum Fp8QuantExtent {
    /// The activation, whole, at its own row count — the dense law.
    Activation { rows: RowCount },
    /// One minted row tile — the extent, the role wrapping, and the op-name ordinal are the
    /// tile's own, minted by the plan.
    Tile(crate::work::Fp8GemmRowTile),
}

impl Fp8QuantSource {
    /// The DENSE law's source — the activation whole, at its own row count.
    pub(crate) fn activation(facts: &Fp8Facts, m: RowCount) -> Self {
        Self {
            name: facts.a_name.clone(),
            id: PlaceId::Act(facts.a_tid),
            extent: Fp8QuantExtent::Activation { rows: m },
            out_tid: facts.out_tid,
        }
    }

    /// The ROW-TILED law's source — one tile's fresh clone. The tile comes WHOLE (minted by
    /// the plan), so the extent cannot disagree with the tile's placement facts.
    pub(crate) fn tile(facts: &Fp8Facts, tile: crate::work::Fp8GemmRowTile, name: String) -> Self {
        Self {
            name,
            id: PlaceId::Act(facts.a_tid),
            extent: Fp8QuantExtent::Tile(tile),
            out_tid: facts.out_tid,
        }
    }

    /// The rows the source carries — the activation's own, or the tile's staged class.
    pub(crate) fn rows(&self) -> RowCount {
        match self.extent {
            Fp8QuantExtent::Activation { rows } => rows,
            Fp8QuantExtent::Tile(tile) => tile.staged_rows,
        }
    }
}

════════ D2 — ktir_matmul_fp8.rs, fp8_quantize_descriptors SIGNATURE + the four seams it retypes ════════
k_pad THREADED, per the cut you rule on below: the chain TAKES the minted tail width; the
pair law has ONE home (the mint site each caller documents). Signature:

pub(crate) fn fp8_quantize_descriptors(
    src: &Fp8QuantSource,
    k: BlockCols,
    k_pad: BlockCols,
    pairs: u32,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
    quantized: &mut std::collections::HashSet<String>,
) -> Result<Vec<EmittedOp>, SuperDscError> {

Doc line ADDED to the ⭐⭐ THE TAIL PADS PER PAIR block (after "…`[m, pairs·inn_pad]` buffer."):

/// `k_pad` is the CALLER's single mint of that padded width (`pairs · fp8_k_pad(k/pairs)`;
/// at `pairs == 1`, exactly `fp8_k_pad(k)` — what the dense door and the row-tiled context
/// mint), so the tail width is spelled once per bundle, at the door that owns the pair law.

Body seams (the ONLY changes inside; everything else verbatim):
- `let m = src.rows;` → `let m_rows = src.rows();` and the `let m_rows = RowCount::of_token_rows(m);` at old :159 DELETED (one spelling, the derived one). Reservation sizes and the `k`/`pairs` arithmetic read `.get()` off the typed params:

    let k_u = k.get();
    let k_pad_u = k_pad.get();
    let inn = k_u / pairs;
    let inn_pad = k_pad_u / pairs; // exact: k_pad IS pairs·fp8_k_pad(inn)

- `let m = src.rows;` (old :118) gone; the refusal at old :126 keeps its numbers (`{k_u}`, `{pairs}`).
- `let k_cols = BlockCols::of_feature_cols(k);` (old :160) → `let k_cols = k;`
- `let k_pad = pairs * inn_pad;` (old :152) DELETED — `k_pad_u` above is the single spelling; `cl_reach` unchanged, reading `inn`/`inn_pad`.
- Reservations `&[m, k]` → `&[m_rows.get(), k_u]`, `&[m, cl_reach]` → `&[m_rows.get(), cl_reach]`, `&[m, k_pad]` → `&[m_rows.get(), k_pad_u]`, `&[m, stk]` → `&[m_rows.get(), stk]` (the layout's size table is a u32-shaped API — the one boundary `.get()`s, listed in the inventory below).
- `tr` closure → derives the wrapping from the extent:

        let tr = |r: R| match src.extent {
            Fp8QuantExtent::Tile(tile) => tile.idx.fq_role(r),
            Fp8QuantExtent::Activation { .. } => r,
        };

- `opn` closure → same shape:

        let opn = |s: &str| match src.extent {
            Fp8QuantExtent::Tile(tile) => format!("{out}_{s}_t{}", tile.idx),
            Fp8QuantExtent::Activation { .. } => format!("{out}_{s}"),
        };

════════ D3 — ktir_matmul_fp8.rs, the dense door's front (complete; k_pad minted ONCE) ════════

    let k_pad = BlockCols::of_feature_cols(crate::work::fp8_k_pad(k));
    let mut ops = fp8_quantize_descriptors(
        &Fp8QuantSource::activation(facts, RowCount::of_token_rows(m)),
        BlockCols::of_feature_cols(k),
        k_pad,
        1,
        sym_id_base,
        layout,
        quantized,
    )?;
    // ⭐ THE 26b K-PAD (dense form) — the contraction width the packed weight was staged at,
    // minted ONCE above as typed feature-cols: the one spelling the quant chain's tail and
    // the mm's K both consume.
    let m_rows = crate::sdsc_abstract::RowCount::of_token_rows(m);
    let n_cols = crate::sdsc_abstract::BlockCols::of_feature_cols(n);

…and the mm's `MatM::of_token_rows(m)` / `MatK::of_in_features(k_pad)` become
`MatM::of_token_rows(m_rows)` / `MatK::of_in_features(k_pad.get())` (MatN/MatK note below).

════════ D4 — ktir_matmul_fp8.rs, fp8_tile_mm_op (complete replacement) ════════

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
    n: BlockCols,
    k_pad: BlockCols,
    idx: crate::work::Fp8GemmTileIdx,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> Result<EmittedOp, SuperDscError> {
    let out = crate::place::act_name(facts.out_tid);
    let out_id = PlaceId::Act(facts.out_tid);
    let raw = syn(layout, out_id.synth(idx.fq_role(SynthRole::FqRaw)));
    if let Some(l) = layout {
        l.synth(out_id.synth(idx.fq_role(SynthRole::FqRaw)), &[ROWS, n.get()]);
    }
    let mut spec = crate::ir::bridge::tiled_op_sdsc_op::matmul_opspec_off_operands::<Fp16>(
        // ⭐ M FROM THE TYPE — `Fp8TileStaging::<ROWS>::STAGED_ROWS`, never a second spelling of
        // the row count: the staging's class and the mm's M are one fact because they are one type.
        crate::sdsc_abstract::MatM::of_token_rows(Fp8TileStaging::<ROWS>::STAGED_ROWS),
        crate::sdsc_abstract::MatN::of_out_features(n.get()),
        crate::sdsc_abstract::MatK::of_in_features(k_pad.get()),
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
        &format!("{out}_fq_mm_t{idx}"),
        &spec,
        &folds,
        sym_id_base,
        layout,
    )?)
}

════════ D5 — ktir_matmul_fp8.rs, Fp8TileEmit::emit, FINAL FORM (complete; supersedes the F1/F5 draft's ⚠ bridge and loops) ════════
The struct and `new` are UNCHANGED from the F1/F5 proposal (fields m/k/n/k_pad typed). The
door amendment: the dense test becomes `if plan.rows_run_whole() { return matmul_fp8_dense_descriptors(...); }`
(A3's named fact — same predicate you confirmed, one spelling). Emit:

    fn emit(self) -> Result<Vec<EmittedOp>, SuperDscError> {
        use SynthRole as R;
        let Self { facts, plan, m, k, n, k_pad, sym_id_base, layout, quantized } = self;
        // The u32 BOUNDARY feeds — `Nest::new` extents and the layout's `synth` size table are
        // u32-shaped APIs; every one is fed from the typed mint the context carries, never from
        // a second spelling of a shape.
        let (m_u, k_u, n_u) = (m.get(), k.get(), n.get());
        let stk = Fp16::ELEMS_PER_STICK; // 64
        let out = crate::place::act_name(facts.out_tid);
        let a_id = PlaceId::Act(facts.a_tid);
        let out_id = PlaceId::Act(facts.out_tid);
        let k_cols = k;
        let n_cols = n;
        let slabs = Slabs::new(1);
        // ⭐ THE IDENTITY TABLE — [unchanged from the written body, verbatim]
        let ident = layout.and_then(|l| l.identity_tid_of_class(stk)).ok_or_else(|| {
            SuperDscError(format!(
                "fp8 matmul {out}: the row-tiled law's clone reads the [{stk},{stk}] identity table, \
                 and this tape registers no hd-{stk} class — a tape without the table cannot tile rows"
            ))
        })?;
        let ident_h = Stk::<KernelTag>::one_stick_identity(crate::place::act_name(ident));
        let pos448 = rbo(&crate::place::act_name(FP8_POS448_TID));
        // ⭐ THE NESTS SPELL THE ROWBLOCKED STICK-GROUP LAW — [doc unchanged, verbatim]
        let k_groups = plan.k_stick_groups().get();
        let n_groups = plan.n_stick_groups().get();
        let src_nest = Nest::new(&["row", "head", "feat"], &[m_u, k_groups, stk], Df::Fp16);
        let tile_nest = Nest::new(&["row", "head", "feat"], &[ROWS, k_groups, stk], Df::Fp16);
        let landing_nest = Nest::new(&["row", "head", "feat"], &[ROWS, n_groups, stk], Df::Fp16);
        let out_nest = Nest::new(&["row", "head", "feat"], &[m_u, n_groups, stk], Df::Fp16);
        let a_h = rb(&facts.a_name, m, k_cols);
        let out_h = rb(&out, m, n_cols);
        let ws_h = rb(&facts.ws_name, RowCount::of_broadcast_row(), n_cols);
        let mut ops: Vec<EmittedOp> = Vec::new();
        for tile in plan.tiles() {
            let tile_id = a_id.synth(tile.idx.fq_role(R::FqTileRow));
            let tile_name = syn(layout, tile_id);
            // ⭐ THE GEOMETRY GUARD — [doc unchanged, verbatim]
            if let Some(l) = layout {
                let want = crate::placement::synth_footprint_bytes(&[ROWS, k_u], Df::Fp16);
                if let Some(have) = l.synth.borrow().sizes.get(&tile_name).copied()
                    && have != want
                {
                    return Err(SuperDscError(format!(
                        "fp8 matmul {out}: tile {} of t{} was already reserved at {have} bytes but \
                         this plan needs {want} — two GEMMs over one activation at different row \
                         classes is a plan disagreement; it is not a resize",
                        tile.idx, facts.a_tid
                    )));
                }
                l.synth(tile_id, &[ROWS, k_u]);
            }
            let tile_h = rb(&tile_name, tile.staged_rows, k_cols);
            if !quantized.contains(&tile_name) {
                // (1) THE PAD ROWS' ZERO FILL — [doc unchanged, verbatim]
                ops.push(assemble_pointwise_broadcast_off(
                    PointwiseFacts {
                        op_name: &format!("{out}_fq_tz{}_op", tile.idx),
                        op_func: "sub",
                        rows: tile.staged_rows,
                        cols: k_cols,
                        inputs: &[In::scalar(&pos448).ew(), In::scalar(&pos448).ew()],
                        o: &tile_h,
                        out_offset: crate::addr::OutCorner::ZERO,
                    },
                    sym_id_base,
                    layout,
                ));
                // (2) THE CLONE — [doc unchanged, verbatim] M is the tile's OWN real-row count,
                // the typed field; the group is the plan's minted head window.
                for g in plan.k_stick_group_windows() {
                    ops.push(crate::ir::bridge::tiled_op_sdsc_op::assemble_matmul_placed(
                        &format!("{out}_fq_tc{}g{g}_op", tile.idx),
                        crate::sdsc_abstract::MatM::of_token_rows(tile.real_rows),
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
                // (3) THE QUANT, OVER THE CLONE — [doc unchanged, verbatim] The source is the
                // MINTED TILE: its extent, role wrapping, and op-name ordinal are the tile's own.
                ops.extend(fp8_quantize_descriptors(
                    &Fp8QuantSource::tile(facts, tile, tile_name.clone()),
                    k,
                    k_pad,
                    1,
                    sym_id_base,
                    layout,
                    quantized,
                )?);
            }
            // (4) THE TILE'S mm — [doc unchanged, verbatim]
            let staging = Fp8TileStaging::<ROWS>::staged(
                rb_df(
                    &syn(layout, a_id.synth(tile.idx.fq_role(R::FqAfp8))),
                    tile.staged_rows,
                    BlockCols::of_feature_cols(k_pad.get()),
                    Df::Fp8,
                ),
                rb(
                    &syn(layout, a_id.synth(tile.idx.fq_role(R::FqAscale))),
                    tile.staged_rows,
                    BlockCols::of_one_stick(Lanes::FP16),
                ),
            )
            .map_err(SuperDscError)?;
            ops.push(fp8_tile_mm_op::<ROWS>(
                &staging,
                facts,
                n,
                k_pad,
                tile.idx,
                sym_id_base,
                layout,
            )?);
            // (5) THE DEQUANT LEGS — [doc unchanged, verbatim]
            let raw = syn(layout, out_id.synth(tile.idx.fq_role(R::FqRaw)));
            let dqa = syn(layout, out_id.synth(tile.idx.fq_role(R::FqDqA)));
            let landing = syn(layout, out_id.synth(tile.idx.fq_role(R::FqTileRow)));
            if let Some(l) = layout {
                l.synth(out_id.synth(tile.idx.fq_role(R::FqDqA)), &[ROWS, n_u]);
                l.synth(out_id.synth(tile.idx.fq_role(R::FqTileRow)), &[ROWS, n_u]);
            }
            let raw_h = rb(&raw, tile.staged_rows, n_cols);
            let dqa_h = rb(&dqa, tile.staged_rows, n_cols);
            let landing_h = rb(&landing, tile.staged_rows, n_cols);
            ops.push(pw2(
                &format!("{out}_fq_dqa_t{}_op", tile.idx),
                "mul",
                tile.staged_rows,
                n_cols,
                In::full(&raw_h),
                In::col(staging.ascale()),
                &dqa_h,
                sym_id_base,
                layout,
            ));
            ops.push(pw2(
                &format!("{out}_fq_dqw_t{}_op", tile.idx),
                "mul",
                tile.staged_rows,
                n_cols,
                In::full(&dqa_h),
                In::mb(&ws_h),
                &landing_h,
                sym_id_base,
                layout,
            ));
            // (6) THE OUT-SPLICE — [doc unchanged, verbatim] the group is the plan's minted
            // head window, the destination row the tile's own.
            for c in plan.n_stick_group_windows() {
                ops.push(crate::ir::bridge::tiled_op_sdsc_op::assemble_matmul_placed(
                    &format!("{out}_fq_to{}c{c}_op", tile.idx),
                    crate::sdsc_abstract::MatM::of_token_rows(tile.real_rows),
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

NET EFFECT on the body: the class row count has ONE value-source in the loop (the tile's own
`staged_rows`, cross-checked by `Fp8TileStaging::staged` against the type's ROWS); the `real`
local and its `.get()` are GONE (MatM takes the tile's typed field); every `{t}` op-name
rendering and every `FqTiled` payload comes from the minted `tile.idx`; the `g`/`c` loops are
gone (typed windows); the quant source is constructed by a door that cannot misstate rows.

════════ E — forced test-call updates (compile ripple only; F6 does the real retype) ════════
Seven direct chain calls in the test module (:1220, :1269, :1501, :1612, :1663, :1687, :1793).
Mechanical mint, same shape at every site:

    fp8_quantize_descriptors(
        &Fp8QuantSource::activation(&facts(), RowCount::of_token_rows(m)),
        BlockCols::of_feature_cols(k),
        BlockCols::of_feature_cols(crate::work::fp8_k_pad(k)),   // pairs == 1 sites
        1,
        sym_id_base, layout, quantized,
    )

At the pair-tail sites (pairs = P > 1), the third argument mints the pair law:
`BlockCols::of_feature_cols(P * crate::work::fp8_k_pad(k / P))`. Say the word and I will send
all seven as exact one-line hunks; they are superseded by F6's typed-geometry rewrite anyway,
which lands before your leg run.

════════ THE HONEST .get() INVENTORY (every remaining one in my diff, post-this-round) ════════
1. `MatN::of_out_features(n.get())` / `MatK::of_in_features(k_pad.get())` — the two un-retyped
   Mat* doors (D3, D4). SAME LAW, and I did NOT retype them unasked: the family-wide cut adds
   matmul/opspec.rs:36-38 + :816-817, emit/batched_gathered.rs ×3 (`STK` consts), and
   lower_ktir:6735 (WIP file). Caller list attached above in the greps; rule on it and I will
   propose it as its own hunk set.
2. `Nest::new` extents + the layout's `synth` size table (D5, D2) — u32-shaped APIs, each fed
   from the context's typed mint. Retyping `Nest::new`/`synth` is a crate-wide ripple I did
   not price unasked.
3. The counts' own bodies (work.rs, Fp8GemmTileIdx) — the types' internals, where get lives.

ORDER REMINDER: after your rulings land (hunk 1 → F1/F5 → this), F6+F7+F8 (tests: typed
geometry ctor only, identity placement minted through identity_tid_of_class, the narrow-PADDED
row-class test), then S1-S5 per the S-follows choice.
