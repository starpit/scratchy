// SPDX-License-Identifier: Apache-2.0
//! PAGED ATTENTION, WALKED: the decode kernel's page walk, its running softmax carried trip to
//! trip, computes what the host computes.
//!
//! `kernels/page_attn_head_major_decode.py` is lifted to one walk and run by the evaluator over a
//! page table that visits the pool's pages out of order, with the last block partly masked. Its
//! output must equal an independent host computation of the kernel's own math (the front end's
//! type legalization states its exps and reductions in f16, and the reference does too), at
//! eight KV heads (the pages gathered through the kv row pool) and at two (the page's rows read at
//! `page · NUM_KV_HEADS`).

use std::collections::HashMap;

use half::f16;
use ktir_core::arena::Arena;
use ktir_superdsc::eval::run;
use ktir_superdsc::lift::lift;
use ktir_superdsc::program::{OpFunc, Program, Segment, Store};
use triton_frontend::codegen::{ArgSpec, KernelSpec};
use triton_frontend::semantic::Val;
use triton_frontend::target::Target;

const ARGS: [(&str, &str); 8] = [
    ("query", "*fp16"),
    ("query_row_index", "*i32"),
    ("k_pages", "*fp16"),
    ("v_pages", "*fp16"),
    ("page_index_table", "*i32"),
    ("kv_row_pool", "*i32"),
    ("mask_stack", "*fp16"),
    ("out", "*fp16"),
];

#[derive(Clone, Copy)]
struct Shape {
    kv_heads: usize,
    group: usize,
    head: usize,
    block: usize,
    pages: usize,
    blocks: usize,
}

const SCALE: f64 = 0.125;
const ROWS_PER_TABLE_ENTRY: usize = 32;

fn decode(s: Shape) -> Program {
    let path = format!("{}/../../../kernels/page_attn_head_major_decode.py", env!("CARGO_MANIFEST_DIR"));
    let src = std::fs::read_to_string(&path).expect("read kernel");
    let mut signature: HashMap<String, ArgSpec> =
        ARGS.iter().map(|(n, t)| (n.to_string(), ArgSpec::parse(t).expect("arg"))).collect();
    let constexprs: HashMap<String, Val> = [
        ("NUM_TOKENS", Val::Int(1)),
        ("NUM_HEADS", Val::Int((s.kv_heads * s.group) as i128)),
        ("NUM_KV_HEADS", Val::Int(s.kv_heads as i128)),
        ("HEAD_SIZE", Val::Int(s.head as i128)),
        ("BLOCK_SIZE", Val::Int(s.block as i128)),
        ("NUM_PAGES", Val::Int(s.pages as i128)),
        ("NUM_BLOCKS", Val::Int(s.blocks as i128)),
        ("SCALE", Val::Float(SCALE)),
    ]
    .into_iter()
    .map(|(n, v)| (n.to_string(), v))
    .collect();
    for n in constexprs.keys() {
        signature.insert(n.clone(), ArgSpec::Constexpr);
    }
    let spec = KernelSpec {
        kernel: "page_attn_head_major_decode_kernel".to_string(),
        signature,
        constexprs,
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

fn h(x: f32) -> f32 {
    f16::from_f32(x).to_f32()
}

struct Inputs {
    query: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    table: Vec<usize>,
    /// The kv row pool: the cache row page `p`'s kv head `kv` is read at, `pool[p·NKV + kv]`.
    pool: Vec<usize>,
    mask: Vec<f32>,
}

/// Whether the kernel reads a page's rows through the pool (eight KV heads or more: the row
/// gather) rather than at `page · NUM_KV_HEADS` (fewer: the page offset).
fn through_pool(s: Shape) -> bool {
    s.kv_heads >= 8
}

fn inputs(s: Shape) -> Inputs {
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut uniform = |lo: f32, hi: f32| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        h(lo + (hi - lo) * ((state >> 40) as f32 / (1u64 << 24) as f32))
    };
    let heads = s.kv_heads * s.group;
    let cache = s.pages * s.kv_heads * s.block * s.head;
    let query = (0..heads * s.head).map(|_| uniform(-1.0, 1.0)).collect();
    let k = (0..cache).map(|_| uniform(-1.0, 1.0)).collect();
    let v = (0..cache).map(|_| uniform(-1.0, 1.0)).collect();
    // The walk visits the pool's pages out of order.
    let table: Vec<usize> = (0..s.blocks).map(|i| (i * 5 + 3) % s.pages).collect();
    // The last block's second half is masked out.
    let mask = (0..s.blocks * s.block)
        .map(|j| if j >= s.blocks * s.block - s.block / 2 { -30000.0 } else { 0.0 })
        .collect();
    // A permuted pool, so a device that read the rows at the page instead of through the pool
    // would read different rows (the production pool is the identity).
    let rows = s.pages * s.kv_heads;
    let pool = (0..rows).map(|r| (r * 7 + 3) % rows).collect();
    Inputs { query, k, v, table, pool, mask }
}

/// THE HOST REFERENCE, written from the kernel's statement: per (kv head, group) row, a
/// running softmax over the walked pages, every op in f16 as the KTIR states it.
fn host(s: Shape, x: &Inputs) -> Vec<f32> {
    host_paged(s, x, 0)
}

/// The host reference with block `i` read at the table's page `i + shift`: shift 0 is the
/// reference, any other a reference the device must not match.
fn host_paged(s: Shape, x: &Inputs, shift: usize) -> Vec<f32> {
    let heads = s.kv_heads * s.group;
    let mut out = vec![0f32; heads * s.head];
    for kv in 0..s.kv_heads {
        for g in 0..s.group {
            let q = &x.query[(kv * s.group + g) * s.head..][..s.head];
            let (mut tmax, mut tsum) = (f32::NEG_INFINITY, 0f32);
            let mut tout = vec![0f32; s.head];
            for i in 0..s.blocks {
                let page = x.table[(i + shift) % s.blocks];
                let row = if through_pool(s) { x.pool[page * s.kv_heads + kv] } else { page * s.kv_heads + kv };
                let kp = &x.k[row * s.block * s.head..][..s.block * s.head];
                let vp = &x.v[row * s.block * s.head..][..s.block * s.head];
                let scores: Vec<f32> = (0..s.block)
                    .map(|b| {
                        let dot = h((0..s.head).fold(0f32, |a, d| a + q[d] * kp[b * s.head + d]));
                        h(h(dot * h(SCALE as f32)) + x.mask[i * s.block + b])
                    })
                    .collect();
                let smax = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let rescale = h((-(h(smax - tmax).max(0.0))).exp());
                let new_max = tmax.max(smax);
                let probs: Vec<f32> = scores.iter().map(|&x| h(h(x - new_max).exp())).collect();
                tsum = h(h(tsum * rescale) + h(probs.iter().fold(0f32, |a, &p| a + p)));
                for d in 0..s.head {
                    let pv = h((0..s.block).fold(0f32, |a, b| a + probs[b] * vp[b * s.head + d]));
                    tout[d] = h(h(tout[d] * rescale) + pv);
                }
                tmax = new_max;
            }
            for d in 0..s.head {
                out[(kv * s.group + g) * s.head + d] = h(tout[d] / tsum);
            }
        }
    }
    out
}

fn check(s: Shape) {
    let p = decode(s);
    let walks = p.segments.iter().filter(|seg| matches!(seg, Segment::Walk(_))).count();
    assert_eq!(walks, 1, "the page walk is one walk");
    let x = inputs(s);
    let f = |v: &[f32]| v.iter().map(|&a| f64::from(a)).collect::<Vec<f64>>();
    let mut table = vec![0f64; s.blocks * ROWS_PER_TABLE_ENTRY];
    for (i, &page) in x.table.iter().enumerate() {
        table[i * ROWS_PER_TABLE_ENTRY] = page as f64;
    }
    let pool: Vec<f64> = x.pool.iter().map(|&r| r as f64).collect();
    let heads = s.kv_heads * s.group;
    let params = vec![
        f(&x.query),
        vec![0.0],
        f(&x.k),
        f(&x.v),
        table,
        pool,
        f(&x.mask),
        vec![0.0; heads * s.head],
    ];
    // No copy reads a K or V page (parameters 2 and 3): the walk reads each page where the
    // cache holds it.
    for seg in &p.segments {
        let copies_kv = match seg {
            Segment::Walk(w) => w.body.iter().any(|n| {
                n.func == OpFunc::Copy && n.inputs.iter().any(|o| matches!(o.store, Store::Param(q) if q.get() == 2 || q.get() == 3))
            }),
            Segment::Straight(ns) => ns.iter().any(|n| {
                n.func == OpFunc::Copy && n.inputs.iter().any(|o| matches!(o.store, Store::Param(q) if q.get() == 2 || q.get() == 3))
            }),
        };
        assert!(!copies_kv, "a copy reads a K or V page");
    }
    let out = run(&p, params).expect("the program runs");
    let want = host(s, &x);
    assert!(want.iter().all(|a| a.is_finite()) && want.iter().any(|&a| a != 0.0), "a finite, nonzero output");
    assert_eq!(out[7], f(&want), "the attention output");
    // NEGATIVE CONTROL: a reference reading each block's neighbouring page must fail.
    assert_ne!(out[7], f(&host_paged(s, &x, 1)), "a reference on the wrong pages is not the device");
}

#[test]
fn the_decode_walk_computes_the_host_attention_through_the_row_gather() {
    check(Shape { kv_heads: 8, group: 2, head: 64, block: 64, pages: 8, blocks: 4 });
}

#[test]
fn the_decode_walk_computes_the_host_attention_through_the_page_offset() {
    check(Shape { kv_heads: 2, group: 4, head: 64, block: 64, pages: 8, blocks: 4 });
}
