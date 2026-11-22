// SPDX-License-Identifier: Apache-2.0
//! A loop is a parallel axis only when its trips are independent. The same loop, storing each trip's
//! row to its own row, is a parallel axis; storing every trip to one fixed row (the last trip wins)
//! is a walk.

use std::collections::HashMap;

use ktir_core::arena::Arena;
use ktir_superdsc::lift::lift;
use ktir_superdsc::program::{Program, Segment};
use triton_frontend::codegen::{ArgSpec, KernelSpec};
use triton_frontend::semantic::Val;
use triton_frontend::target::Target;

const SRC: &str = r#"
import triton
import triton.language as tl


@triton.jit
def copy_rows(x, o, N: tl.constexpr, T: tl.constexpr, FIXED: tl.constexpr):
    xd = tl.make_tensor_descriptor(x, shape=[T, N], strides=[N, 1], block_shape=[1, N])
    od = tl.make_tensor_descriptor(o, shape=[T, N], strides=[N, 1], block_shape=[1, N])
    for i in tl.range(0, T, 1):
        if FIXED:
            od.store([0, 0], xd.load([i, 0]))
        else:
            od.store([i, 0], xd.load([i, 0]))
"#;

fn lifted(fixed: bool) -> Program {
    let signature: HashMap<String, ArgSpec> = [
        ("x", "*fp16"),
        ("o", "*fp16"),
        ("N", "constexpr"),
        ("T", "constexpr"),
        ("FIXED", "constexpr"),
    ]
    .iter()
    .map(|(n, t)| (n.to_string(), ArgSpec::parse(t).expect("arg")))
    .collect();
    let constexprs: HashMap<String, Val> = [("N", Val::Int(64)), ("T", Val::Int(4)), ("FIXED", Val::Int(i128::from(fixed)))]
        .into_iter()
        .map(|(n, v)| (n.to_string(), v))
        .collect();
    let spec = KernelSpec {
        kernel: "copy_rows".to_string(),
        signature,
        constexprs,
        file: "copy_rows.py".to_string(),
    };
    let grid = [1i64];
    let mut tt = triton_frontend::codegen::compile(SRC, &spec, Target::spyre()).expect("codegen");
    triton_frontend::opt::make_ttir(&mut tt).expect("make_ttir");
    let mut m = triton_ktir::from_ttir::convert(&tt).expect("from_ttir");
    triton_ktir::make_ktir(&mut m, &grid).expect("make_ktir");
    triton_ktir::passes::to_ktir::run(&mut m, &grid).expect("to_ktir");
    let f = triton_ktir::passes::to_ktir_emit::lower(&m, Arena::global()).expect("handoff");
    lift(&f).expect("lift")
}

fn walks(p: &Program) -> Vec<u32> {
    p.segments
        .iter()
        .filter_map(|s| match s {
            Segment::Walk(w) => Some(w.trips.get()),
            Segment::Straight(_) => None,
        })
        .collect()
}

#[test]
fn a_loop_writing_each_trip_to_its_own_row_is_a_parallel_axis() {
    assert_eq!(walks(&lifted(false)), Vec::<u32>::new());
}

#[test]
fn a_loop_writing_every_trip_to_one_row_walks() {
    let p = lifted(true);
    assert_eq!(walks(&p), vec![4], "{:#?}", p.segments);
}
