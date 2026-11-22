// SPDX-License-Identifier: Apache-2.0
//! A kernel's page walk lifts to ONE walk whose body is one trip's ops, at any walk length.
//!
//! Each paged-attention kernel is compiled from its Triton source at two walk lengths. The lift
//! must give exactly one `Segment::Walk` with the walk's trip count, and the same program apart
//! from that count: a lowering that wrote trips out would grow with the walk and fail here.

use std::collections::HashMap;

use ktir_core::arena::Arena;
use ktir_superdsc::lift::lift;
use ktir_superdsc::program::{Program, Segment};
use triton_frontend::codegen::{ArgSpec, KernelSpec};
use triton_frontend::semantic::Val;
use triton_frontend::target::Target;

fn compile(file: &str, kernel: &str, args: &[(&str, &str)], constexprs: &[(&str, Val)]) -> Program {
    let path = format!("{}/../../../kernels/{file}", env!("CARGO_MANIFEST_DIR"));
    let src = std::fs::read_to_string(&path).expect("read kernel");
    let mut signature: HashMap<String, ArgSpec> =
        args.iter().map(|(n, t)| (n.to_string(), ArgSpec::parse(t).expect("arg type"))).collect();
    for (n, _) in constexprs {
        signature.insert(n.to_string(), ArgSpec::parse("constexpr").expect("constexpr"));
    }
    let spec = KernelSpec {
        kernel: kernel.to_string(),
        signature,
        constexprs: constexprs.iter().map(|(n, v)| (n.to_string(), v.clone())).collect(),
        file: path,
    };
    let grid = [1i64];
    let mut tt = triton_frontend::codegen::compile(&src, &spec, Target::spyre()).expect("codegen");
    triton_frontend::opt::make_ttir(&mut tt).expect("make_ttir");
    let mut m = triton_ktir::from_ttir::convert(&tt).expect("from_ttir");
    triton_ktir::make_ktir(&mut m, &grid).expect("make_ktir");
    triton_ktir::passes::to_ktir::run(&mut m, &grid).expect("to_ktir");
    let f = triton_ktir::passes::to_ktir_emit::lower(&m, Arena::global()).expect("handoff");
    lift(&f).expect("lift")
}

/// The program with every walk's trip count set to one: what must not depend on the walk length.
fn shape_of(p: &Program) -> (Vec<Segment>, Vec<u32>) {
    let mut trips = Vec::new();
    let segments = p
        .segments
        .iter()
        .map(|s| match s {
            Segment::Walk(w) => {
                trips.push(w.trips.get());
                let mut w = w.clone();
                w.trips = std::num::NonZeroU32::MIN;
                Segment::Walk(w)
            }
            s => s.clone(),
        })
        .collect();
    (segments, trips)
}

const ATTN_ARGS: [(&str, &str); 8] = [
    ("query", "*fp16"),
    ("query_row_index", "*i32"),
    ("k_pages", "*fp16"),
    ("v_pages", "*fp16"),
    ("page_index_table", "*i32"),
    ("kv_row_pool", "*i32"),
    ("mask_stack", "*fp16"),
    ("out", "*fp16"),
];

fn decode(blocks: i128) -> Program {
    compile(
        "page_attn_head_major_decode.py",
        "page_attn_head_major_decode_kernel",
        &ATTN_ARGS,
        &[
            ("NUM_TOKENS", Val::Int(1)),
            ("NUM_HEADS", Val::Int(4)),
            ("NUM_KV_HEADS", Val::Int(2)),
            ("HEAD_SIZE", Val::Int(64)),
            ("BLOCK_SIZE", Val::Int(64)),
            ("NUM_PAGES", Val::Int(16)),
            ("NUM_BLOCKS", Val::Int(blocks)),
            ("SCALE", Val::Float(0.125)),
        ],
    )
}

#[test]
fn the_decode_page_walk_is_one_walk_at_any_length() {
    let (short, short_trips) = shape_of(&decode(4));
    let (long, long_trips) = shape_of(&decode(8));
    assert_eq!(short_trips, vec![4]);
    assert_eq!(long_trips, vec![8]);
    assert_eq!(short, long, "only the trip count depends on the walk length");
}
