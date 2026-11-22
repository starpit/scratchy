// SPDX-License-Identifier: Apache-2.0
//! The grid is one outer parallel axis, never one copy per position.
//!
//! The sampler is the production kernel with a grid (`[B]`, one program per request row). Lifted
//! at B = 1, 2 and 32, it must be the same program once the outer grid axis is taken off every
//! node: a front end or lift that copied the body per position, or cloned parameters per position,
//! fails here.

use std::collections::HashMap;

use ktir_core::arena::Arena;
use ktir_superdsc::lift::lift;
use ktir_superdsc::program::{Gather, Node, Operand, Program, Segment};
use ktir_superdsc::qty::Elems;
use triton_frontend::codegen::{ArgSpec, KernelSpec};
use triton_frontend::semantic::Val;
use triton_frontend::target::Target;

const SAMPLE_ARGS: [(&str, &str); 27] = [
    ("logits_ptr", "*fp16"),
    ("slo_ptr", "*i32"),
    ("shi_ptr", "*i32"),
    ("stp_ptr", "*i32"),
    ("sgate_ptr", "*i32"),
    ("invt_ptr", "*fp32"),
    ("bramp_ptr", "*fp32"),
    ("lramp_ptr", "*fp32"),
    ("binc_ptr", "*fp32"),
    ("linc_ptr", "*fp32"),
    ("ones_ptr", "*fp32"),
    ("sixtyfour_ptr", "*fp32"),
    ("padoff_ptr", "*fp32"),
    ("gramp_ptr", "*fp32"),
    ("ginc_ptr", "*fp32"),
    ("hc1_ptr", "*i32"),
    ("hc2_ptr", "*i32"),
    ("hc3_ptr", "*i32"),
    ("c256_ptr", "*i32"),
    ("lane_ptr", "*fp32"),
    ("fm8_ptr", "*fp32"),
    ("ftop_ptr", "*fp32"),
    ("fzero_ptr", "*fp32"),
    ("fhalf_ptr", "*fp32"),
    ("fm22_ptr", "*fp32"),
    ("cmin_ptr", "*fp16"),
    ("idx_ptr", "*i32"),
];

fn sampler(b: i64) -> Program {
    let path = format!("{}/../../../kernels/sample_fwd.py", env!("CARGO_MANIFEST_DIR"));
    let src = std::fs::read_to_string(&path).expect("read kernel");
    let mut signature: HashMap<String, ArgSpec> =
        SAMPLE_ARGS.iter().map(|(n, t)| (n.to_string(), ArgSpec::parse(t).expect("arg"))).collect();
    let constexprs: HashMap<String, Val> = [
        ("V", 49155i128),
        ("PAD_V", 65536),
        ("NB", 1024),
        ("BLOCK", 64),
        ("B", i128::from(b)),
    ]
    .into_iter()
    .map(|(n, v)| (n.to_string(), Val::Int(v)))
    .collect();
    for n in constexprs.keys() {
        signature.insert(n.clone(), ArgSpec::Constexpr);
    }
    let spec = KernelSpec {
        kernel: "sample_fwd".to_string(),
        signature,
        constexprs,
        file: path,
    };
    let grid = [b];
    let mut tt = triton_frontend::codegen::compile(&src, &spec, Target::spyre()).expect("codegen");
    triton_frontend::opt::make_ttir(&mut tt).expect("make_ttir");
    let mut m = triton_ktir::from_ttir::convert(&tt).expect("from_ttir");
    triton_ktir::make_ktir(&mut m, &grid).expect("make_ktir");
    triton_ktir::passes::to_ktir::run(&mut m, &grid).expect("to_ktir");
    let f = triton_ktir::passes::to_ktir_emit::lower(&m, Arena::global()).expect("handoff");
    lift(&f).expect("lift")
}

fn without_grid_axis<O: Clone>(o: &Operand<O>, grid: bool) -> Operand<O> {
    let mut o = o.clone();
    if grid {
        o.view.axes.remove(0);
    }
    o.gather = o.gather.map(|g| match g {
        Gather::Rows { axis, index, rows } => Gather::Rows {
            axis: if grid { axis - 1 } else { axis },
            index: Box::new(without_grid_axis(&index, grid)),
            rows,
        },
        Gather::Offset { index, step, bound } => Gather::Offset {
            index: Box::new(without_grid_axis(&index, grid)),
            step,
            bound,
        },
    });
    o
}

fn node_without_grid<O: Clone>(n: &Node<O>, grid: bool) -> Node<O> {
    let mut n = n.clone();
    if grid {
        n.space.remove(0);
    }
    n.inputs = n.inputs.iter().map(|o| without_grid_axis(o, grid)).collect();
    n.output = without_grid_axis(&n.output, grid);
    n
}

/// The program's segments with the outer grid axis taken off every node and operand.
fn per_position(p: &Program, b: i64) -> Vec<Segment> {
    let grid = b > 1;
    p.segments
        .iter()
        .map(|s| match s {
            Segment::Straight(nodes) => Segment::Straight(nodes.iter().map(|n| node_without_grid(n, grid)).collect()),
            Segment::Walk(w) => {
                let mut w = w.clone();
                w.body = w.body.iter().map(|n| node_without_grid(n, grid)).collect();
                Segment::Walk(w)
            }
        })
        .collect()
}

#[test]
fn the_sampler_is_one_program_at_every_batch_with_the_grid_as_its_outer_axis() {
    let one = sampler(1);
    for b in [2, 32] {
        let p = sampler(b);
        // Every node of the batched program runs over the grid as its outer axis.
        for s in &p.segments {
            if let Segment::Straight(nodes) = s {
                for n in nodes {
                    assert_eq!(n.space[0].get(), b as u64, "every node's outer axis is the grid");
                }
            }
        }
        // The same parameters (no per-position clones), the same intermediates once per
        // position, and the same nodes once the grid axis is taken off.
        assert_eq!(p.params, one.params, "B = {b}: the parameters are the kernel's own");
        assert_eq!(p.locals.len(), one.locals.len(), "B = {b}: the intermediates are the same set");
        for (lb, l1) in p.locals.iter().zip(&one.locals) {
            assert_eq!(lb.elements, Elems::new(l1.elements.get() * b as u64), "B = {b}: held once per position");
        }
        assert_eq!(per_position(&p, b), per_position(&one, 1), "B = {b}: the same program per position");
    }
}
