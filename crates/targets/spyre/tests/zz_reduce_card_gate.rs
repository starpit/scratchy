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
//! reference in fp32 (rmsnorm: `x/√(mean(x²)+1e-5)·gain`; sum: Σx per row; max: max x per
//! row). The reduce readback decodes the door's OWN out placement, never a guessed stride:
//! the door mints the caller's accum as ONE STICK PER ROW row-blocked (the reduce door's Fp16
//! arm), a row-blocked tensor's device order interleaves the rows inside each stick-group
//! (`StickLayout::group_stride`), and the placement reserves `row_footprint_bytes(1, Fp16)`
//! per row — so the launcher's whole-placement readback at `stick:t=m` resticks to host
//! `[m, size/2/m]` and row r's value is host `(r, 0)`, lane 0 of its stick. The per-op bars
//! (tail law, rmsnorm ulp ≤ 8, reduce_max ulp ≤ 8, and the reduce_sum walk bar — each with
//! its reason) are stated in the runner header, before the run.
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

    // ── THE TALL RUNG — the full production path, exactly the door test's own program ──
    let (m, c) = TALL;
    let ir = rmsnorm_ir(m, c, GainConvention::Scale);
    // ⛔ EXACTLY THE GAIN tid is a weight here: t0 (the source x) is an ACTIVATION — staged
    // through the per-step codec path — and t1 (the gain row) is the one real weight, staged
    // through the weight loop's IEEE→sen re-encode with `stick:t1=1`, the [1, C] row's
    // identity map (`stage_weights` refuses a raw id by name: bundle_run.rs:742-752).
    // Handing the layout {0, 1} made x itself a Weight, and every raw bind a refusal on the
    // card.
    let weight_ids: HashSet<u32> = [1u32].into_iter().collect();
    let layout = compute_bundle_layout(&ir, &weight_ids, PagedRows::Sequence, &[])
        .unwrap_or_else(|e| panic!("layout minted m={m} c={c}: {e}"));
    let node = &ir.nodes[0];
    let spliced = scratchy_triton_splice::lower(node, &ir, false)
        .unwrap_or_else(|e| panic!("splice compiled m={m} c={c}: {e}"));
    let mut sym = 0i64;
    let ops = door_lower(
        spliced
            .ktir
            .as_ref()
            .expect("spliced op carries its program"),
        &mut sym,
        Some(&layout),
    )
    .unwrap_or_else(|e| panic!("spliced program lowered m={m} c={c}: {}", e.message));
    // ⛔ THE CELL IS THE SPLIT — the same marker the door test pins: a bundle without it is
    // a fitting rung, not this cell, and the run would prove nothing about the split.
    assert!(
        ops.iter().any(|e| e.op_name.contains("_p1_op")),
        "the tall rung's mean(x²) splits: no `_p1_op` slice among {} emitted ops",
        ops.len()
    );
    let dir = root.join(format!("rmsnorm_m{m}_c{c}"));
    std::fs::create_dir_all(&dir).unwrap();
    write_layout_cell(&dir, &layout, Some(&ops), 2);
    write_cell_json(&dir, "rmsnorm", m, c, vec![0, 1], 2, &sha);

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
        write_cell_json(
            &dir,
            &format!("reduce_{}", kind.op_func()),
            m,
            c,
            vec![REDUCE_DATA_TID],
            REDUCE_OUT_TID,
            &sha,
        );
    }
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

/// Write the cell's manifest.
fn write_cell_json(
    dir: &std::path::Path,
    op: &str,
    m: u32,
    c: u32,
    in_tids: Vec<u32>,
    out_tid: u32,
    sha: &str,
) {
    let cell = Cell {
        op: op.to_string(),
        m,
        c,
        seed: u64::from(m) * 1_000_003 + u64::from(c),
        in_tids,
        out_tid,
        sha: sha.to_string(),
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
