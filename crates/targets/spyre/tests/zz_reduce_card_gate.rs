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
//! * `reduce_sum` / `reduce_max` — the BARE-REDUCE DOOR TWINS at the historical wide
//!   multi-row cell's shape family (`[4, 512]`, 8 sticks): the reduce door
//!   (`lower_ktir_to_superdsc::reduce`) with `Sum` and `Max`, single-shot (the shape fits
//!   one LX pass — the split path is the tall rung's cell above), laid out by hand: the
//!   data at its stick-padded width, the accum at one stick per row, both what a placement
//!   reserves in production.
//!
//! ## The bar (checked by the runner, not here)
//! The read-back buffer holds the card image of the op over SEEDED RANDOM fills — never
//! constants, so a wrong lane cannot hide — against a HOST reference computed in fp32 from
//! the formula (rmsnorm: `x/√(mean(x²)+1e-5)·gain`; sum: Σx per row; max: max x per row).
//! The rmsnorm read-back decodes by the rank-2 stick-column law
//! (`D = (c/64)·(m·64) + t·64 + (c%64)`); the reduce accum is one stick per row, its value
//! at lane 0. The verdict line states the program, the shape, the max abs error, the max
//! fp16 ulp, and the artifact path.
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
use ktir_superdsc::placement::SegRole;
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

/// The bare-reduce twins' shape — the historical wide multi-row cell's family (`ROWS`,
/// `WIDE` in `zz_the_ktir_door_lowers_the_wide_multi_row_max.rs`): 4 rows × 512 (8 sticks).
const TWIN: (u32, u32) = (4, 512);

/// The reduce door's own operand tids (the door test's own spelling).
const REDUCE_DATA_TID: u32 = 7;
const REDUCE_OUT_TID: u32 = 8;

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
#[ignore]
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
    let weight_ids: HashSet<u32> = [0u32, 1u32].into_iter().collect();
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

    // ── THE BARE-REDUCE DOOR TWINS — the door itself, at the historical cell's shape ──
    for kind in [ReduceKind::Sum, ReduceKind::Max] {
        let (m, c) = TWIN;
        let ops = reduce_twin(kind, m, c);
        let dir = root.join(format!("reduce_{}_m{m}_c{c}", kind.op_func()));
        std::fs::create_dir_all(&dir).unwrap();
        write_reduce_cell(&dir, &ops, m, c);
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
    SubtileIR {
        tensors,
        num_sources: 2,
        nodes: vec![SubtileNode {
            id: scratchy_subtile::subtile_ir::SubtileId::from_index(0),
            op: SubOp::RmsNorm { eps: 1e-5, gain },
            inputs: vec![whole(0), whole(1)],
            output: whole(2),
        }],
        result: TensorId::from_index(2),
        op_output: Vec::new(),
    }
}

/// One bare-reduce door twin: the door with `Sum`/`Max` at `[rows, cols]`, single-shot
/// (the shape fits one LX pass), `Df::Fp16` — the reduce's own proven basis.
fn reduce_twin(kind: ReduceKind, rows: u32, cols: u32) -> Vec<ktir_superdsc::emit::EmittedOp> {
    let data = region(REDUCE_DATA_TID, rows, cols, false);
    let out = region(REDUCE_OUT_TID, rows, 1, true);
    let mut sym = 0i64;
    reduce("attn_bmax_g0", kind, Df::Fp16, &[data, out], &mut sym, None).unwrap_or_else(|e| {
        panic!(
            "{kind:?} [{rows}, {cols}]: the door must lower it: {}",
            e.message
        )
    })
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
    for (name, off) in synth.map.iter() {
        let tid: u32 = name
            .trim_start_matches('t')
            .parse()
            .unwrap_or_else(|_| panic!("synth intermediate {name} is not a t<tid>"));
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
    s.push_str("  ],\n  \"scalarmul_scales\": [\n  ]\n}\n");
    std::fs::write(dir.join("placements.json"), s).unwrap();
}

/// Write the bundle for a bare-reduce door twin: the door's own two placements, laid out
/// BY HAND exactly as a production placement reserves them — the data at its stick-padded
/// width, the accum at one stick per row — in the intermediate segment, back to back, the
/// accum labeled `Logits` (the role the runtime's launcher demands of the read-back
/// tensor). No synthetics: the twin is single-shot.
fn write_reduce_cell(
    dir: &std::path::Path,
    ops: &Vec<ktir_superdsc::emit::EmittedOp>,
    rows: u32,
    cols: u32,
) {
    superdsc::write_dxp_input(dir, ops, superdsc::FoldGrouping::Split)
        .unwrap_or_else(|e| panic!("{}: write_dxp_input: {e}", dir.display()));
    let sticks = cols.div_ceil(64) as u64;
    let data_bytes = rows as u64 * sticks * 64 * 2;
    let accum_bytes = rows as u64 * 64 * 2;
    let intermediate = SegRole::Intermediate.segment();
    let mut seg_bytes = [0u64; scratchy_target_spyre::bundle_code::NUM_SEGMENTS];
    seg_bytes[intermediate] = data_bytes + accum_bytes;
    let s = format!(
        "{{\n  \"segment_bytes\": [{}],\n  \"places\": [\n\
         \x20   {{\"tid\": {REDUCE_DATA_TID}, \"role\": \"Activation\", \"segment\": {intermediate}, \
         \"bank\": 0, \"offset\": 0, \"size\": {data_bytes}}},\n\
         \x20   {{\"tid\": {REDUCE_OUT_TID}, \"role\": \"Logits\", \"segment\": {intermediate}, \
         \"bank\": 0, \"offset\": {data_bytes}, \"size\": {accum_bytes}}}\n\
         \x20 ],\n  \"synth\": [\n  ],\n  \"scalarmul_scales\": [\n  ]\n}}\n",
        seg_bytes
            .iter()
            .map(|b| b.to_string())
            .collect::<Vec<_>>()
            .join(", "),
    );
    std::fs::write(dir.join("placements.json"), s).unwrap();
}
