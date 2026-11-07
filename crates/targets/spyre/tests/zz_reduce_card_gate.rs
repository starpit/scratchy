// SPDX-License-Identifier: Apache-2.0

//! ⭐ THE #1113 CARD GATE — the long-reduce split's device paths, dumped for an on-card
//! `bundle_run_dev` run. Test tooling: nothing here ships, and the only shipped code it
//! calls is the splice and the doors the bake calls.
//!
//! ## What this dump states
//! * `rmsnorm` — THE TALL WIDE-HIDDEN RUNG: a prefill-width rmsnorm at hidden 16384 whose
//!   mean(x²) bills past the per-core LX budget (2048 rows × 32 KiB/row), so the chain's
//!   own reduce goes through the LONG-REDUCE SPLIT — the same program
//!   `the_tall_wide_hidden_rmsnorm_mean_splits_through_the_door` pins at the door, run here
//!   on the card. The dump asserts the split's own slice marker (`_p1_op`) is present, so
//!   the cell cannot silently degrade to a fitting rung.
//! * `rmsnorm` (the control sweep) — THE SAME OP, UNSPLIT, at FOUR hidden widths
//!   ([`CONTROL_WIDTHS`], all at [`CONTROL_M`] rows, all under the budget so none can
//!   split — no `_p{j}_op` marker anywhere, the dump asserts the negative at every
//!   width). The moe-lead's ruled control experiment (2026-11-07), widened per the
//!   auditor (four points, not one fitted): every control is seeded with the tall rung's
//!   own seed — the same rows, the same gain values, prefix-consistent across widths —
//!   so the controls differ from the tall rung in the split alone and from each other in
//!   N alone, and the runner's per-row deterministic level-B bound column (the filed
//!   issue's resolution path) has four points to be checked against, never fitted.
//! * `reduce_sum` / `reduce_max` — the BARE-REDUCE DOOR TWINS, WIDE MULTI-ROW, at the
//!   split's own fire point (`[4, 1048576]`, 16,384 sticks): the reduce door
//!   (`lower_ktir_to_superdsc::reduce`) with `Sum` and `Max`, driven the way production
//!   drives it — with a REAL `BundleLayout` from `compute_bundle_layout` over the twins'
//!   two-tensor placement manifest (see [`reduce_twin_ir`]), never the door's no-layout
//!   arm. Per row the swept width bills 2 MiB past the 1,677,721-B usable-LX budget, so
//!   the door's own probe-and-split answers in trips: the dump asserts the split's slice
//!   and combine markers are present, so the cell cannot silently degrade to a fitting
//!   rung.
//!
//! ## The bar (checked by the runner, not here)
//! Inputs are staged and outputs read back through the launcher's IEEE↔SEN codec
//! (`t<id>=<file> stick:t<id>=<rows>` both ways — `raw:` binds IEEE bytes verbatim, but the
//! card's activation format is SEN fp16), so the read-back file is host row-major IEEE and
//! the runner decodes it as the IDENTITY: the stick-column law is applied exactly once, by
//! the launcher, never again below. The rmsnorm gain row is staged the same way with
//! `stick:t1=1`, which IS the `[1, C]` row's identity map (`bundle_run.rs`: "stick:tN=1 is
//! the identity, and saying it out loud is the point"), from the cell's own gain tensor.
//! Over SEEDED RANDOM fills — never constants, so a wrong lane cannot hide — against a HOST
//! reference in fp32 (rmsnorm: `x/√(mean(x²)+1e-5)·gain`; sum: Σx per row; max: the STAGE
//! IMAGE of max x per row — the codec tie law, stated with its card measurement in the
//! runner header). The reduce readback decodes the door's OWN out placement, never a guessed stride:
//! the door mints the caller's accum as ONE STICK PER ROW row-blocked (the reduce door's Fp16
//! arm), a row-blocked tensor's device order interleaves the rows inside each stick-group
//! (`StickLayout::group_stride`), and the placement reserves `row_footprint_bytes(1, Fp16)`
//! per row — so the launcher's whole-placement readback at `stick:t=m` resticks to host
//! `[m, size/2/m]` and row r's value is host `(r, 0)`, lane 0 of its stick. The per-op
//! bars (tail law; rmsnorm 8 ulp at the SHIPPED widths with the wide cells REPORTED
//! pending the filed issue's ruled bar, alongside its per-row deterministic level-B
//! bound column; the exact reduce_max law; and the reduce_sum split-facts bar — each
//! with its reason) are stated in the runner header, before the run. The sum bar's
//! inputs — the split's trip count T, each slice's swept width, and the accum's device
//! format — are written into `cell.json` by THIS dump off the door's own emitted ops
//! ([`SplitFacts`]), never re-derived from geometry by the runner.
//!
//! ```bash
//! REDUCE_CARD_GATE_ROOT=<dir> REDUCE_CARD_GATE_SHA=<sha> \
//!   cargo test --release -p scratchy-target-spyre \
//!   --test zz_reduce_card_gate -- --ignored --nocapture
//! ```
//!
//! Then on a pod:
//! ```bash
//! tests/ew_card_gate_run.sh <dir> <warm-tree-at-this-sha> <verdict-log>
//! ```

use std::collections::HashSet;

use ktir_superdsc::emit::ktir_matmul_fp8::QuantizedActivations;
use ktir_superdsc::emit::lower_ktir_to_superdsc::{Region, reduce};
use ktir_superdsc::ktir_node::ReduceKind;
use ktir_superdsc::placement::{BundleLayout, SegRole};
use ktir_superdsc::superdsc_opspec::Df;
use scratchy_subtile::subtile_ir::{
    GainConvention, SubOp, SubtileIR, SubtileNode, TensorId, TensorRegion, TensorShape,
};
use scratchy_target_spyre::ktir_superdsc_door::lower as bundle_lower;
use scratchy_target_spyre::lower_subtile_tape_to_superdsc as superdsc;
use scratchy_target_spyre::lower_subtile_tape_to_superdsc::compute_bundle_layout;
use scratchy_triton_splice::PagedRows;

/// One program through the production door as a bundle of its own — the fp8 quantize set
/// starts empty, which is what a one-program bundle's is.
fn door_lower(
    k: &ktir_superdsc::ktir_node::KtirNode,
    sym: &mut i64,
    layout: Option<&ktir_superdsc::placement::BundleLayout>,
) -> Result<Vec<ktir_superdsc::emit::EmittedOp>, ktir_superdsc::emit::lower_ktir_to_superdsc::Error>
{
    bundle_lower(k, sym, layout, None, &mut QuantizedActivations::default())
}

/// The tall rung's own geometry — `the_tall_wide_hidden_rmsnorm_mean_splits_through_the_door`'s.
const TALL: (u32, u32) = (2048, 16384);

/// ⭐ THE CONTROL SWEEP — the same rmsnorm chain, UNSPLIT, at FOUR hidden widths (the
/// filed issue's resolution path: four points, not one fitted). m = 512 at every width:
/// the probe bills the exx2 data window per core as `ceil(m/32)` rows × width × 2 B, so
/// the widest (16384) bills 16 × 32,768 B = 524,288 B — 3.2× under the 1,677,721-B
/// usable-LX budget — and every narrower width bills less, so no cell can split (the
/// tall rung at m = 2048, c = 16384 bills 2 MiB and splits, T = 2 measured at
/// cg1113r-1a0882da). The four cells differ in N ALONE. Every one is seeded with the
/// TALL RUNG'S OWN SEED, so all four run the tall rung's first 512 rows and gain values
/// (prefix-consistent: a row's first k staged values are the same values at every width
/// — the reduce just sweeps fewer of them); the c = 16384 cell is the tall rung's own
/// control and also carries the same-rows reference (the only width its output is
/// elementwise comparable with the tall rung's at).
const CONTROL_M: u32 = 512;
const CONTROL_WIDTHS: [u32; 4] = [2048, 4096, 8192, 16384];

/// The cell's fill seed — the runner derives its fills from `rng(seed + input_index)`, so
/// two cells with the same seed and the same input order draw the SAME values, and a cell
/// with a smaller row count draws the PREFIX of a taller cell's rows (row-major draws).
fn seed_of((m, c): (u32, u32)) -> u64 {
    u64::from(m) * 1_000_003 + u64::from(c)
}

/// The bare-reduce twins' shape — the historical wide multi-row cell's family (`ROWS` in
/// `zz_the_ktir_door_lowers_the_wide_multi_row_max.rs`), scaled to the split's own fire
/// point: 4 rows × 1,048,576 (16,384 sticks, whole-stick aligned). Per row the swept width
/// bills 2 MiB past the 1,677,721-B usable-LX budget and half of it fits, so the door's own
/// probe-and-split answers in trips — the same geometry class the door fixture
/// `the_past_the_budget_reduce_splits_through_the_door` pins at one row.
const TWIN: (u32, u32) = (4, 1_048_576);

/// The twins' placement-manifest tids: t0 the `[rows, cols]` data source, t1 the `[rows, 1]`
/// result — the two tensors [`reduce_twin_ir`] declares.
const REDUCE_DATA_TID: u32 = 0;
const REDUCE_OUT_TID: u32 = 1;

/// The cell's own manifest row — everything the runner's fills and decoder need, stated
/// once. The same schema `zz_ew_card_gate.rs` writes; the runner is shared.
#[derive(serde::Serialize)]
struct Cell {
    /// The op tag — the runner's host-model key.
    op: String,
    m: u32,
    c: u32,
    /// The seed the runner's fills derive from (one per input, offset by the input's index).
    seed: u64,
    /// The tids the runner stages, in order.
    in_tids: Vec<u32>,
    /// The buffer the runner reads back.
    out_tid: u32,
    /// The commit this dump ran at — required at dump time (a dump refuses to run unstamped).
    sha: String,
    /// ⭐ THE SAME-ROWS PROOF (moe-lead 2026-11-07): names the cell whose FIRST m rows this
    /// cell runs (this dump seeds it so — see [`seed_of`]), so the runner can compare this
    /// cell's output elementwise with that cell's output over exactly those rows: the
    /// direct measure of whether the split computes the same function as the one-pass
    /// reduce. Absent on every cell that is not the control.
    #[serde(skip_serializing_if = "Option::is_none")]
    same_rows_ref: Option<String>,
    /// ⭐ THE DOOR'S OWN SPLIT FACTS (moe-lead ruling 2026-11-07) — the inputs of the
    /// runner's reduce bars, written by the DUMP off what the door EMITTED, never
    /// re-derived from geometry by the runner: a bar calibrated on the run it judges
    /// proves nothing. See [`SplitFacts`].
    split: SplitFacts,
}

/// ⭐ THE DOOR'S OWN SPLIT FACTS — every field is read off the split ops the door emitted
/// (`{op}_p{j}_op` slices, the spelling `assemble_reduce_split` itself mints), never
/// recomputed from the cell geometry:
/// * `time` — the split's trip count T: the number of `_p{j}_op` slice ops, `j` a
///   contiguous run from 0 (the door emits one slice per trip, in trip order);
/// * `slice_lanes` — each slice's swept reduce width, in lanes: the `out` extent the
///   slice's OWN descriptor states (`N_.out_` — the stick-padded width the reduce walks;
///   its pad lanes are masked to the reduce's identity, so they contribute nothing);
/// * `accum_df` — the accumulator's device format, off the slice's OWN descriptor: the
///   `wordLength` of its OUTPUT `labeledDs_` row (2 = fp16, 4 = fp32). The runner's sum
///   bar takes its ε from this alone (2⁻¹¹ fp16, 2⁻²⁴ fp32).
#[derive(serde::Serialize)]
struct SplitFacts {
    time: u32,
    slice_lanes: Vec<u32>,
    accum_df: String,
}

/// The `_p{j}_op` slice markers the split itself mints (`assemble_reduce_split`'s own
/// spelling), `j` → op. Empty when the door answered in one pass — the control's own
/// negative marker.
fn slice_markers<'a>(
    ops: &'a [ktir_superdsc::emit::EmittedOp],
) -> std::collections::BTreeMap<u32, &'a ktir_superdsc::emit::EmittedOp> {
    let mut slices: std::collections::BTreeMap<u32, &ktir_superdsc::emit::EmittedOp> =
        std::collections::BTreeMap::new();
    for op in ops {
        let Some(j) = op
            .op_name
            .rsplit_once("_p")
            .and_then(|(_, rest)| rest.strip_suffix("_op"))
            .and_then(|j| j.parse::<u32>().ok())
        else {
            continue;
        };
        assert!(
            slices.insert(j, op).is_none(),
            "two slice ops spell _p{j}_op"
        );
    }
    slices
}

/// One reduce op's OWN swept width and accum df, read off its descriptor — the law the
/// split's `_p{j}_op` slices and the single-shot reduce state identically (the sweep is
/// the `out` extent `N_.out_` states, the format the OUTPUT `labeledDs_` wordLength).
fn op_reduce_facts(
    op: &ktir_superdsc::emit::EmittedOp,
    what: &str,
    name: &str,
) -> (u32, &'static str) {
    assert!(
        op.time == 1,
        "{what}: {name} is time-tiled (time={}) — its N_ would state a per-trip extent, \
         and this dump refuses to guess the sweep",
        op.time
    );
    let sdsc = op
        .op
        .as_ref()
        .unwrap_or_else(|| panic!("{what}: {name} carries no descriptor"));
    let dscs: Vec<&ktir_superdsc::wire::Dsc> = sdsc.dscs_.iter().flat_map(|m| m.values()).collect();
    assert!(!dscs.is_empty(), "{what}: {name} states no dsc");
    let out_extents: HashSet<i64> = dscs.iter().map(|d| d.N_.out_).collect();
    assert!(
        out_extents.len() == 1 && *out_extents.iter().next().unwrap() > 0,
        "{what}: {name} does not state ONE positive out extent: {out_extents:?}"
    );
    let dfs: HashSet<u32> = dscs
        .iter()
        .flat_map(|d| d.labeledDs_.iter().filter(|ds| ds.dsType_ == "OUTPUT"))
        .map(|ds| ds.wordLength)
        .collect();
    let df = match dfs.iter().copied().min() {
        Some(2) if dfs.len() == 1 => "fp16",
        Some(4) if dfs.len() == 1 => "fp32",
        _ => panic!("{what}: {name} does not state ONE accum df: wordLengths {dfs:?}"),
    };
    (*out_extents.iter().next().unwrap() as u32, df)
}

/// Derive [`SplitFacts`] from the ops the door emitted. Every assertion is a refusal
/// naming the cell — a fact this fn cannot state off the door own emission is a dump
/// bug, never a guessed number. `unsplit_prefix` names the reduce's own single-shot op
/// spelling (the `assemble_reduce_seeded` name the caller states) for the one-pass case:
/// the control's facts then come off that op's own descriptor — T = 1 by the absence of
/// every `_p{j}_op` marker, the lane width and the accum df by the same descriptor law.
fn split_facts(
    ops: &[ktir_superdsc::emit::EmittedOp],
    what: &str,
    unsplit_prefix: &str,
) -> SplitFacts {
    let slices = slice_markers(ops);
    if slices.is_empty() {
        let hits: Vec<&ktir_superdsc::emit::EmittedOp> = ops
            .iter()
            .filter(|e| e.op_name.starts_with(unsplit_prefix))
            .collect();
        assert!(
            hits.len() == 1,
            "{what}: the door answered in one pass, so exactly one {unsplit_prefix}* op \
             must state the reduce — found {}",
            hits.len()
        );
        let (lanes, df) = op_reduce_facts(hits[0], what, &hits[0].op_name);
        return SplitFacts {
            time: 1,
            slice_lanes: vec![lanes],
            accum_df: df.to_string(),
        };
    }
    let time = slices.len() as u32;
    assert!(
        time > 0 && slices.keys().copied().eq(0..time),
        "{what}: the door emitted no contiguous 0.._p{{j}}_op slice run"
    );
    let mut slice_lanes = Vec::with_capacity(time as usize);
    let mut accum_df: Option<String> = None;
    for (j, op) in &slices {
        let name = format!("_p{j}_op");
        let (lanes, df) = op_reduce_facts(op, what, &name);
        slice_lanes.push(lanes);
        match accum_df.as_deref() {
            None => accum_df = Some(df.to_string()),
            Some(prev) => assert!(
                prev == df,
                "{what}: the slices disagree on the accum df: {prev} vs _p{j}_op {df}"
            ),
        }
    }
    SplitFacts {
        time,
        slice_lanes,
        accum_df: accum_df.expect("time > 0 holds a df"),
    }
}

/// Dump every cell. One directory per cell: `group_0/` ([`superdsc::write_dxp_input`]),
/// `placements.json` (the launch map) and `cell.json`.
#[test]
#[ignore = "needs a spyre card: dumps cells for ew_card_gate_run.sh"]
fn dump_reduce_card_gate() {
    let Some(root) = std::env::var_os("REDUCE_CARD_GATE_ROOT") else {
        panic!("REDUCE_CARD_GATE_ROOT=<dir> names where the cells are written");
    };
    let sha = std::env::var("REDUCE_CARD_GATE_SHA")
        .unwrap_or_else(|_| panic!("REDUCE_CARD_GATE_SHA=<sha> stamps the commit the dump runs at — a dump refuses to run unstamped"));
    let root = std::path::Path::new(&root);
    std::fs::create_dir_all(root).unwrap();

    // ── THE RMSNORM CELLS — the tall rung (SPLIT) and the four-width control sweep (one
    //    LX pass, UNSPLIT, same rows at every width: every control is seeded with the
    //    tall rung's own seed, so they run the tall rung's first 512 rows and the same
    //    gain values): the same program, the same fill law, the same runner — the controls
    //    differ from the tall rung in the split alone, and from each other in N alone
    //    (moe-lead's ruled control, 2026-11-07; four-width amendment after
    //    cg1113r-bcd507ca, auditor: four points, not one fitted) ──
    rmsnorm_cell(root, &sha, TALL, "the tall rung", true, seed_of(TALL), None);
    for c in CONTROL_WIDTHS {
        let same_rows = (c == TALL.1).then(|| format!("rmsnorm_m{}_c{}", TALL.0, TALL.1));
        rmsnorm_cell(
            root,
            &sha,
            (CONTROL_M, c),
            "the control rung",
            false,
            seed_of(TALL),
            same_rows,
        );
    }

    // ── THE BARE-REDUCE DOOR TWINS — the door itself, wide multi-row, through the real
    //    placement (compute_bundle_layout), with the split FIRING ──
    for kind in [ReduceKind::Sum, ReduceKind::Max] {
        let (m, c) = TWIN;
        let (ops, layout) = reduce_twin(kind, m, c);
        // ⛔ THE CELL IS THE SPLIT — the same law the tall rung's marker states: a bundle
        // whose reduce came back SINGLE-SHOT (the bare `{kind}_o{tid}` op name) is a fitting
        // rung, not this cell, and the run would prove nothing about the split path.
        let bare = format!("{}_o{}", kind.op_func(), REDUCE_OUT_TID);
        assert!(
            !ops.iter().any(|e| e.op_name == bare),
            "{kind:?} [{m}, {c}]: the wide reduce must SPLIT, not fit one LX pass — \
             {} ops, none named {bare}",
            ops.len()
        );
        assert!(
            ops.iter().any(|e| e.op_name.contains("_p1_op")),
            "{kind:?} [{m}, {c}]: no `_p1_op` slice among {} emitted ops",
            ops.len()
        );
        assert!(
            ops.iter().any(|e| e.op_name.contains("_c1_op")),
            "{kind:?} [{m}, {c}]: no `_c1_op` combine among {} emitted ops",
            ops.len()
        );
        let dir = root.join(format!("reduce_{}_m{m}_c{c}", kind.op_func()));
        std::fs::create_dir_all(&dir).unwrap();
        write_layout_cell(&dir, &layout, Some(&ops), REDUCE_OUT_TID);
        let split = split_facts(&ops, &format!("{kind:?} [{m}, {c}]"), &bare);
        write_cell_json(
            &dir,
            &format!("reduce_{}", kind.op_func()),
            m,
            c,
            seed_of((m, c)),
            vec![REDUCE_DATA_TID],
            REDUCE_OUT_TID,
            &sha,
            split,
            None,
        );
    }
}

/// One rmsnorm chain cell — the tall rung and its control differ only in their row count
/// and their split expectation. `expect_split` states the cell's own marker law: the tall
/// rung IS the split (a bundle without `_p1_op` is a fitting rung, and the run would prove
/// nothing about the split path), the control IS the one-pass answer (no `_p{j}_op` marker
/// at all — a bundle that split is the tall rung, not this cell). `seed` is the fill seed
/// and `same_rows_ref` the same-rows reference: the control is seeded with the TALL RUNG'S
/// OWN SEED, so it draws the tall rung's first m rows and the same gain row — the pair
/// then differs in the split alone, over identical inputs, and the runner's elementwise
/// same-rows report is the direct proof of whether the split computes the same function.
fn rmsnorm_cell(
    root: &std::path::Path,
    sha: &str,
    (m, c): (u32, u32),
    what: &str,
    expect_split: bool,
    seed: u64,
    same_rows_ref: Option<String>,
) {
    let ir = rmsnorm_ir(m, c, GainConvention::Scale);
    // ⛔ EXACTLY THE GAIN tid is a weight here: t0 (the source x) is an ACTIVATION — staged
    // through the per-step codec path — and t1 (the gain row) is the one real weight, staged
    // through the weight loop's IEEE→sen re-encode with `stick:t1=1`, the [1, C] row's
    // identity map (`stage_weights` refuses a raw id by name: bundle_run.rs:742-752).
    // Handing the layout {0, 1} made x itself a Weight, and every raw bind a refusal on the
    // card.
    let weight_ids: HashSet<u32> = [1u32].into_iter().collect();
    let layout = compute_bundle_layout(&ir, &weight_ids, PagedRows::Sequence, &[])
        .unwrap_or_else(|e| panic!("{what}: layout minted m={m} c={c}: {e}"));
    let node = &ir.nodes[0];
    let spliced = scratchy_triton_splice::lower(node, &ir, false)
        .unwrap_or_else(|e| panic!("{what}: splice compiled m={m} c={c}: {e}"));
    let mut sym = 0i64;
    let ops = door_lower(
        spliced
            .ktir
            .as_ref()
            .expect("spliced op carries its program"),
        &mut sym,
        Some(&layout),
    )
    .unwrap_or_else(|e| panic!("{what}: spliced program lowered m={m} c={c}: {}", e.message));
    let markers = slice_markers(&ops);
    if expect_split {
        // ⛔ THE CELL IS THE SPLIT — the same marker the door test pins: a bundle without
        // it is a fitting rung, not this cell.
        assert!(
            markers.contains_key(&1),
            "{what}: the mean(x²) splits: no `_p1_op` slice among {} emitted ops",
            ops.len()
        );
    } else {
        // ⛔ THE CELL IS THE ONE-PASS ANSWER — the control's own negative marker: any slice
        // op means the probe split a row count the budget arithmetic said fits, and the cell
        // would silently be a second tall rung.
        assert!(
            markers.is_empty(),
            "{what}: the control must run UNSPLIT — {} `_p{{j}}_op` slice markers among {} \
             emitted ops",
            markers.len(),
            ops.len()
        );
    }
    let dir = root.join(format!("rmsnorm_m{m}_c{c}"));
    std::fs::create_dir_all(&dir).unwrap();
    write_layout_cell(&dir, &layout, Some(&ops), 2);
    let split = split_facts(&ops, what, "rmmean_");
    write_cell_json(
        &dir,
        "rmsnorm",
        m,
        c,
        seed,
        vec![0, 1],
        2,
        sha,
        split,
        same_rows_ref,
    );
}

/// `triton_splice_golden.rs`'s own rmsnorm builder, verbatim: t0 = source, t1 = the gain
/// row, t2 = result; eps is granite's `1e-5`; the gain convention is the Scale row's.
fn rmsnorm_ir(m: u32, c: u32, gain: GainConvention) -> SubtileIR {
    let tensors = vec![
        TensorShape { rows: m, cols: c },
        TensorShape { rows: 1, cols: c },
        TensorShape { rows: m, cols: c },
    ];
    let whole = |t: usize| TensorRegion {
        tensor: TensorId::from_index(t),
        region: tensors[t].whole(),
    };
    // The node is built BEFORE the struct moves `tensors` — the golden builder's own order:
    // `whole` borrows `tensors`, so a node built inside the struct literal after the
    // `tensors` field is a move-out-while-borrowed (E0505).
    let node = SubtileNode {
        id: scratchy_subtile::subtile_ir::SubtileId::from_index(0),
        op: SubOp::RmsNorm { eps: 1e-5, gain },
        inputs: vec![whole(0), whole(1)],
        output: whole(2),
    };
    SubtileIR {
        tensors,
        num_sources: 2,
        nodes: vec![node],
        result: TensorId::from_index(2),
        op_output: Vec::new(),
    }
}

/// The twins' PLACEMENT MANIFEST — the two-tensor `SubtileIR` the walk's own layout law
/// places: t0 the `[rows, cols]` source (an ACTIVATION, staged per step through the codec)
/// and t1 the `[rows, 1]` result (LOGITS, the read-back tensor). No `SubOp` splices to a
/// bare column reduce (`SumReduce` is unconstructed; the rmsnorm reduce rides the chain
/// arm), so the op itself is the door's `reduce` — the SAME door `whole_function`'s
/// `Lowering::Reduce` arm drives with a real layout in production — and this manifest is
/// what hands that door the placement: `compute_bundle_layout` sizes and places both
/// tensors by its own role law (source → seg0, result → seg4), never a hand-written
/// placements row. The split's partial accumulators mint through the layout's own synth
/// map, which `write_layout_cell` writes out with the bundle.
fn reduce_twin_ir(rows: u32, cols: u32) -> SubtileIR {
    SubtileIR {
        tensors: vec![TensorShape { rows, cols }, TensorShape { rows, cols: 1 }],
        num_sources: 1,
        nodes: Vec::new(),
        result: TensorId::from_index(1),
        op_output: Vec::new(),
    }
}

/// One bare-reduce door twin: the door with `Sum`/`Max` at `[rows, cols]`, `Df::Fp16` (the
/// reduce's own proven basis), driven with the REAL placement — the layout
/// `compute_bundle_layout` mints over [`reduce_twin_ir`], the same law every production
/// bundle's placement comes from. Returns the emitted ops together with the layout they
/// were placed through (the cell writer needs both).
fn reduce_twin(
    kind: ReduceKind,
    rows: u32,
    cols: u32,
) -> (Vec<ktir_superdsc::emit::EmittedOp>, BundleLayout) {
    let ir = reduce_twin_ir(rows, cols);
    let weight_ids: HashSet<u32> = HashSet::new();
    let layout = compute_bundle_layout(&ir, &weight_ids, PagedRows::Sequence, &[])
        .unwrap_or_else(|e| panic!("{kind:?} [{rows}, {cols}]: the layout must mint: {e}"));
    let data = region(REDUCE_DATA_TID, rows, cols, false);
    let out = region(REDUCE_OUT_TID, rows, 1, true);
    let mut sym = 0i64;
    let ops = reduce(
        "attn_bmax_g0",
        kind,
        Df::Fp16,
        &[data, out],
        &mut sym,
        Some(&layout),
    )
    .unwrap_or_else(|e| {
        panic!(
            "{kind:?} [{rows}, {cols}]: the door must lower it through the real placement: {}",
            e.message
        )
    });
    (ops, layout)
}

/// The door test's own region spelling.
fn region(tid: u32, rows: u32, cols: u32, is_out: bool) -> Region {
    Region {
        tid,
        v_rows: rows,
        v_cols: cols,
        r_start: 0,
        c_start: 0,
        r_len: rows,
        c_len: cols,
        r_cover: (0, rows),
        c_cover: (0, cols),
        is_out,
        is_fp8: false,
        band: None,
    }
}

/// Write the cell's manifest. `seed` is the fill seed ([`seed_of`] — the control passes
/// the tall rung's own seed, so it runs the tall rung's first rows); `same_rows_ref` names
/// the cell those rows come from, for the runner's elementwise same-rows report.
fn write_cell_json(
    dir: &std::path::Path,
    op: &str,
    m: u32,
    c: u32,
    seed: u64,
    in_tids: Vec<u32>,
    out_tid: u32,
    sha: &str,
    split: SplitFacts,
    same_rows_ref: Option<String>,
) {
    let cell = Cell {
        op: op.to_string(),
        m,
        c,
        seed,
        in_tids,
        out_tid,
        sha: sha.to_string(),
        split,
        same_rows_ref,
    };
    std::fs::write(
        dir.join("cell.json"),
        serde_json::to_string_pretty(&cell).unwrap(),
    )
    .unwrap();
}

/// Write the bundle (`group_0/` + `placements.json`) for a cell whose placement comes from
/// the walk's own layout — the same writer `zz_ew_card_gate.rs` uses.
fn write_layout_cell(
    dir: &std::path::Path,
    l: &ktir_superdsc::placement::BundleLayout,
    ops: Option<&Vec<ktir_superdsc::emit::EmittedOp>>,
    readback_tid: u32,
) {
    if let Some(ops) = ops.filter(|o| !o.is_empty()) {
        superdsc::write_dxp_input(dir, ops, superdsc::FoldGrouping::Split)
            .unwrap_or_else(|e| panic!("{}: write_dxp_input: {e}", dir.display()));
    }
    let synth = l.synth.borrow();
    let mut seg_bytes = l.segment_bytes;
    let intermediate = SegRole::Intermediate.segment();
    seg_bytes[intermediate] = seg_bytes[intermediate].max(synth.next);
    let mut s = String::new();
    s.push_str("{\n  \"segment_bytes\": [");
    for (i, b) in seg_bytes.iter().enumerate() {
        s.push_str(&format!("{}{}", if i == 0 { "" } else { ", " }, b));
    }
    s.push_str("],\n  \"places\": [\n");
    let mut rows: Vec<String> = l
        .placements
        .iter()
        .map(|(tid, p)| {
            format!(
                "    {{\"tid\": {}, \"role\": \"{:?}\", \"segment\": {}, \"bank\": {}, \
                 \"offset\": {}, \"size\": {}}}",
                tid, p.role, p.segment, p.bank, p.offset, p.size
            )
        })
        .collect();
    // A synth NAME is either a plain `t<id>` (an intermediate placed as its own tensor — it
    // gets a `places` row) or a NAMED intermediate the walk mints from another tensor
    // (`<t>_mean`, …) — those are stated BY NAME in the `synth` list alone, with no `places`
    // row: their placement is the runtime's, read off the synth entry, never a bare tid.
    for (name, off) in synth.map.iter() {
        let Some(tid) = name
            .strip_prefix('t')
            .and_then(|rest| rest.parse::<u32>().ok())
        else {
            continue;
        };
        rows.push(format!(
            "    {{\"tid\": {}, \"role\": \"{}\", \"segment\": {}, \"bank\": 0, \"offset\": {}, \
             \"size\": {}}}",
            tid,
            if tid == readback_tid {
                "Logits"
            } else {
                "Activation"
            },
            intermediate,
            off,
            synth.sizes.get(name).copied().unwrap_or(0),
        ));
    }
    s.push_str(&rows.join(",\n"));
    s.push_str("\n  ],\n  \"synth\": [\n");
    for (i, (name, off)) in synth.map.iter().enumerate() {
        s.push_str(&format!(
            "    {{\"name\": \"{}\", \"segment\": {}, \"offset\": {}, \"size\": {}}}{}\n",
            name,
            intermediate,
            off,
            synth.sizes.get(name).copied().unwrap_or(0),
            if i + 1 == synth.map.len() { "" } else { "," },
        ));
    }
    // ⭐⭐ THE EVERY-BOUND-TENSOR LAW (moe-lead, 2026-11-06; dc17da7dcc's dump half, ported to
    // this line's own writer — the EW gate's dump test is card-tooling's file, not carried
    // here): the launch map states its OWN worker-bound consts — the layout's scale registry
    // as (tid, value) rows, index i at `scalarmul_scale_tid(i)` (the address law the
    // placement pass and the worker's wiring both state, so the dump cannot disagree with
    // either). The zero that taught this law was measured on card (cgpx-4f71242b): a
    // never-staged registry scale made a launch report success over ALL-ZEROS numerics,
    // because `bundle_run` zeroes every segment holding no staged weight. The READ SET the
    // runner stages them against is stated below, off the emitted ops' own operand bindings.
    s.push_str("  ],\n  \"scalarmul_scales\": [\n");
    for (i, v) in l.scalarmul_scales.iter().enumerate() {
        s.push_str(&format!(
            "    {{\"tid\": {}, \"value\": {:?}}}{}\n",
            superdsc::scalarmul_scale_tid(i),
            v,
            if i + 1 == l.scalarmul_scales.len() {
                ""
            } else {
                ","
            },
        ));
    }
    // ⭐⭐ THE EVERY-BOUND-TENSOR LAW's READ SET (read-set amendment 2026-11-07) — the same
    // rows `zz_ew_card_gate.rs`'s writer states, so the SHARED runner stages this dump by the
    // same law: every RESERVED tid an op of this bundle READS, taken from the emitted ops'
    // own operand bindings (`EmittedOp`'s `arg_bindings`, the same walk `write_dxp_input`
    // states), never the placement list. A placed-but-UNREAD reserved tensor
    // (RMS_HALF/RMS_INVCOLS, placed for every RmsNorm node) is a layout fact, not a
    // correctness one — measured at cg1113-ca8a1115, where the placement-list expectation
    // made the tall rung NOT-RUN on the two tids nothing reads — so the placement list is
    // not the expected set. The protected invariant is NO OP READS A SOURCE NOBODY WROTE.
    let mut reads: Vec<u32> = ops
        .into_iter()
        .flatten()
        .flat_map(|op| op.arg_bindings.iter().filter(|b| b.is_input))
        .filter_map(|b| {
            b.buffer
                .strip_prefix('t')
                .and_then(|rest| rest.parse::<u32>().ok())
                .filter(|&tid| ktir_superdsc::reserved_tids::is_reserved(tid))
        })
        .collect();
    reads.sort_unstable();
    reads.dedup();
    s.push_str("  ],\n  \"reads\": [\n");
    for (i, tid) in reads.iter().enumerate() {
        s.push_str(&format!(
            "    {}{}\n",
            tid,
            if i + 1 == reads.len() { "" } else { "," },
        ));
    }
    s.push_str("  ]\n}\n");
    std::fs::write(dir.join("placements.json"), s).unwrap();
}
