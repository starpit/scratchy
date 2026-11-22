// SPDX-License-Identifier: Apache-2.0
//! Constants are held as their format holds them: an int32 constant above 2^24 stays exact, and a
//! `tl.arange` of any width is a ramp stated by its form, never an element-by-element table.

use std::collections::HashMap;

use ktir_core::arena::Arena;
use ktir_superdsc::eval::run;
use ktir_superdsc::lift::lift;
use ktir_superdsc::program::{Program, TableData};
use triton_frontend::codegen::{ArgSpec, KernelSpec};
use triton_frontend::semantic::Val;
use triton_frontend::target::Target;

fn compile(src: &str, kernel: &str, args: &[(&str, &str)], constexprs: &[(&str, i128)]) -> Program {
    let mut signature: HashMap<String, ArgSpec> =
        args.iter().map(|(n, t)| (n.to_string(), ArgSpec::parse(t).expect("arg"))).collect();
    for (n, _) in constexprs {
        signature.insert(n.to_string(), ArgSpec::Constexpr);
    }
    let spec = KernelSpec {
        kernel: kernel.to_string(),
        signature,
        constexprs: constexprs.iter().map(|(n, v)| (n.to_string(), Val::Int(*v))).collect(),
        file: format!("{kernel}.py"),
    };
    let grid = [1i64];
    let mut tt = triton_frontend::codegen::compile(src, &spec, Target::spyre()).expect("codegen");
    triton_frontend::opt::make_ttir(&mut tt).expect("make_ttir");
    let mut m = triton_ktir::from_ttir::convert(&tt).expect("from_ttir");
    triton_ktir::make_ktir(&mut m, &grid).expect("make_ktir");
    triton_ktir::passes::to_ktir::run(&mut m, &grid).expect("to_ktir");
    let f = triton_ktir::passes::to_ktir_emit::lower(&m, Arena::global()).expect("handoff");
    lift(&f).expect("lift")
}

const HASH: &str = r#"
import triton
import triton.language as tl


@triton.jit
def hash_row(x, o, N: tl.constexpr):
    xd = tl.make_tensor_descriptor(x, shape=[1, N], strides=[N, 1], block_shape=[1, N])
    od = tl.make_tensor_descriptor(o, shape=[1, N], strides=[N, 1], block_shape=[1, N])
    od.store([0, 0], xd.load([0, 0]) * -1640531535 + 16777217)
"#;

#[test]
fn an_int32_constant_above_two_to_the_24_stays_exact() {
    let n = 64usize;
    let p = compile(HASH, "hash_row", &[("x", "*i32"), ("o", "*i32")], &[("N", n as i128)]);
    let x: Vec<i32> = (0..n as i32).map(|i| i * 7919 - 100_000).collect();
    let out = run(&p, vec![x.iter().map(|&v| f64::from(v)).collect(), vec![0.0; n]]).expect("runs");
    let want: Vec<f64> = x
        .iter()
        .map(|&v| f64::from(v.wrapping_mul(-1_640_531_535).wrapping_add(16_777_217)))
        .collect();
    assert_eq!(out[1], want, "the constants are the kernel's to the last bit");
}

const RAMP: &str = r#"
import triton
import triton.language as tl


@triton.jit
def ramp_row(o, N: tl.constexpr):
    od = tl.make_tensor_descriptor(o, shape=[1, N], strides=[N, 1], block_shape=[1, N])
    od.store([0, 0], (tl.arange(0, N) * 3 + 5)[None, :])
"#;

#[test]
fn a_full_width_arange_is_a_ramp() {
    let n = 65536usize;
    let p = compile(RAMP, "ramp_row", &[("o", "*i32")], &[("N", n as i128)]);
    assert!(
        p.tables.iter().all(|t| matches!(t.data, TableData::Ramp { .. })),
        "every table is a ramp, stated by its form"
    );
    assert!(
        p.tables.iter().any(|t| t.shape.iter().map(|e| e.get()).product::<u64>() == n as u64),
        "the arange is the full row"
    );
    let out = run(&p, vec![vec![0.0; n]]).expect("runs");
    let want: Vec<f64> = (0..n as i64).map(|i| (i * 3 + 5) as f64).collect();
    assert_eq!(out[0], want);
}
