// SPDX-License-Identifier: Apache-2.0

//! Drive one fixture through `ktir-superdsc`'s lowering.
//!
//! ```bash
//! cargo run --offline --example drive -- <config> <grid-csv> <program>
//! ```
//!
//! `<program>` is stated, not inferred — see the crate docs on why the node kind is an argument
//! during bring-up. Accepted: `rmsnorm`, `matmul`, `silumul`, `scalarmul`, `lmlast`, `transpose`,
//! `ew:add|mul|sub|silu|gelu|exp|rsqrt|sqrt|abs|reciprocal|sigmoid|tanh|mish|realdiv|maximum|minimum`,
//! and `rope`/`attn`, both of which now REFUSE here by name: their hand-rolled geometry doors are
//! deleted, and each lowers from its compiled ops through the whole-function door instead —
//! `examples/bake_py` with `KTIR_WHOLE=1` is that path. Staying accepted keeps the refusal honest
//! (a named answer, not `no case for rope`).
//!
//! Environment:
//! * `KTIR_DUMP=1` — the handed-off `IRFunction`, one op per line.
//! * `KTIR_LAYOUT_DUMP=1` — the derived `BundleLayout` (placements + scale registry).
//! * `KTIR_ADDR_DUMP=1` — each emitted descriptor's per-core HBM start addresses.
//! * `KTIR_NO_LAYOUT=1` — drive with `layout: None`, `ktir-superdsc`'s unit-test addressing. The
//!   CONTROL for "did the layout change this verdict?".

use std::path::PathBuf;

use ktir_superdsc::ktir_node::{Elementwise, Program};

fn program_from(s: &str) -> Option<Program> {
    Some(match s {
        "rmsnorm" => Program::RmsNorm,
        "matmul" => Program::Matmul,
        "silumul" => Program::SiluMul,
        "scalarmul" => Program::ScalarMul,
        "lmlast" => Program::LmLast,
        "transpose" => Program::Transpose,
        "rope" => Program::Rope,
        "attn" => Program::Attn,
        other => {
            let kind = other.strip_prefix("ew:")?;
            Program::Elementwise(match kind {
                "add" => Elementwise::Add,
                "mul" => Elementwise::Mul,
                "sub" => Elementwise::Sub,
                "silu" => Elementwise::Silu,
                "gelu" => Elementwise::Gelu,
                "exp" => Elementwise::Exp,
                "rsqrt" => Elementwise::Rsqrt,
                "sqrt" => Elementwise::Sqrt,
                "abs" => Elementwise::Abs,
                "reciprocal" => Elementwise::Reciprocal,
                "sigmoid" => Elementwise::Sigmoid,
                "tanh" => Elementwise::Tanh,
                "mish" => Elementwise::Mish,
                "realdiv" => Elementwise::RealDiv,
                "maximum" => Elementwise::Maximum,
                "minimum" => Elementwise::Minimum,
                _ => return None,
            })
        }
    })
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 3 {
        eprintln!("use: drive <config> <grid-csv> <program>");
        std::process::exit(2);
    }
    let (config, program_s) = (&args[0], &args[2]);
    let grid: Vec<i64> = args[1]
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| {
            s.trim()
                .parse()
                .expect("grid is a comma-separated integer list")
        })
        .collect();
    let Some(program) = program_from(program_s) else {
        eprintln!("unknown program `{program_s}`");
        std::process::exit(2);
    };

    let ttir_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crate sits one level under crates/triton")
        .join("test-experiment1/ktir")
        .join(format!("{config}.ttir.mlir"));
    let ttir = std::fs::read_to_string(&ttir_path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", ttir_path.display()));

    macro_rules! refused {
        ($stage:expr, $err:expr) => {{
            println!(
                "REFUSED  {config:32} {program_s:10} {:<12} {}",
                $stage, $err
            );
            return;
        }};
    }

    let mut m = match triton_ktir::text::parse::parse(&ttir) {
        Ok(m) => m,
        Err(e) => refused!("ttir-parse", e),
    };
    if let Err(e) = triton_ktir::make_ktir(&mut m, &grid) {
        refused!("make_ktir", e);
    }
    if let Err(e) = triton_ktir::passes::to_ktir::run(&mut m, &grid) {
        refused!("to_ktir", e);
    }

    let node = match triton_ktir_superdsc::node_for(&m, program) {
        Ok(n) => n,
        Err(e) => refused!(e.stage, e.message),
    };
    // `KTIR_DUMP=1` prints the handed-off function as their structural readings see it. The
    // instrument, not the path — every refusal below is about the SHAPE of this list.
    if std::env::var("KTIR_DUMP").is_ok() {
        print!("{}", triton_ktir_superdsc::dump(&node));
    }
    // THE LAYOUT IS DERIVED FROM THE PROGRAM (`triton_ktir_superdsc::layout`), because two bodies
    // cannot emit without one: `rmsnorm`/`scalarmul` resolve their constant to a slot in
    // `BundleLayout::scalarmul_scales` through `scale_slot`, which is `layout.and_then(..)` and so
    // always answers "absent" for `None`.
    //
    // `KTIR_NO_LAYOUT=1` selects the old `None` arm — `ktir-superdsc`'s unit-test addressing
    // (per-op `segment_base(arg_index)`, no registry). It is the CONTROL: it makes "did the layout
    // change this fixture's verdict?" a question this one binary answers.
    let layout = if std::env::var("KTIR_NO_LAYOUT").is_ok() {
        None
    } else {
        match triton_ktir_superdsc::layout::for_node(&node) {
            Ok(l) => Some(l),
            Err(e) => refused!(e.stage, e.message),
        }
    };
    // `KTIR_LAYOUT_DUMP=1` prints the plan the addresses come out of: every placement's
    // `(segment, offset, size)` and the scale registry in slot order. The instrument for
    // "which segment did t1 get, and is the epsilon in slot 0" — the two facts a wrong
    // address here would come from.
    if std::env::var("KTIR_LAYOUT_DUMP").is_ok() {
        match &layout {
            None => println!("layout: none (KTIR_NO_LAYOUT)"),
            Some(l) => {
                for (tid, p) in &l.placements {
                    println!(
                        "place t{tid:<12} seg{} off={:<10} size={:<10} role={:?}",
                        p.segment, p.offset, p.size, p.role
                    );
                }
                for (i, s) in l.scalarmul_scales.iter().enumerate() {
                    println!("scale[{i}] = {s:e}  -> const t{}", u32::MAX - 20 - i as u32);
                }
                println!("synth.next = {}", l.synth.borrow().next);
            }
        }
    }
    // ⛔ NO GEOMETRY DOOR HERE, ON PURPOSE. The old `Program::Attn` arm routed through
    // `drive_attn` → `attn_at::<NQH, NKVH, HD>`, and the `Program::Rope` arm through
    // `drive_rope` → `rope_at::<HD>` — scratchy's prebuilt fragment-assembly lowerings
    // selected by constexpr-geometry match, pattern matching instead of lowering the given
    // code. Both removed; both programs refuse by name in `emit_node_with` and lower from
    // their compiled ops through the whole-function door (`bake_py` with `KTIR_WHOLE=1`
    // is the from-source path that runs on card).
    let emitted = triton_ktir_superdsc::emit_node_with(&node, layout.as_ref());
    match emitted {
        Err(e) => refused!(e.stage, e.message),
        Ok(ops) => {
            // `KTIR_ADDR_DUMP=1` prints each descriptor's per-core START ADDRESSES
            // (`startAddressCoreCorelet_`), which is the ONE observable that proves the layout
            // reached the emission: every one must be `SEGMENT_OFFSETS[seg] + intra` for the
            // segment the placement above states. Reading the plan and reading the addresses are
            // two different checks — a plan can be right and still not be the one used.
            if std::env::var("KTIR_ADDR_DUMP").is_ok() {
                for e in &ops {
                    let Some(op) = &e.op else { continue };
                    for m in &op.dscs_ {
                        for dsc in m.values() {
                            for n in &dsc.scheduleTree_ {
                                let a: Vec<&str> = n
                                    .startAddressCoreCorelet_
                                    .data_
                                    .values()
                                    .map(|s| s.as_str())
                                    .collect();
                                println!(
                                    "addr {:<22} {:<28} {:<4} {a:?}",
                                    e.op_name, n.name_, n.component_
                                );
                            }
                        }
                    }
                }
            }
            println!(
                "EMITTED  {config:32} {program_s:10} {:<12} ops={}",
                "",
                ops.len()
            )
        }
    }
}
