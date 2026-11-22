// SPDX-License-Identifier: Apache-2.0
//! THE MIXTURE OF EXPERTS, AS ONE GATHERED BATCHED CONTRACTION PER PROJECTION.
//!
//! `kernels/moe.py` routes each token to K of E experts and runs the routed experts' gate, up
//! and down projections as three batched contractions, the experts' fp8 slabs gathered by id.
//! Lifted at different token and expert counts it is always the router's contraction plus those
//! three, never one per (token, expert). And under the evaluator's numeric model its routed ids,
//! route weights and output equal an independent host computation of the kernel's math.

use std::collections::HashMap;

use half::f16;
use ktir_core::arena::Arena;
use ktir_superdsc::eval::run;
use ktir_superdsc::lift::lift;
use ktir_superdsc::program::{Gather, Node, OpFunc, ParamId, Program, Segment, Store};
use triton_frontend::codegen::{ArgSpec, KernelSpec};
use triton_frontend::semantic::Val;
use triton_frontend::target::Target;

const ARGS: [(&str, &str); 15] = [
    ("desc_xr", "*fp16"),
    ("desc_rg", "*fp16"),
    ("desc_rw", "*fp16"),
    ("desc_pes", "*fp16"),
    ("desc_xe", "*fp16"),
    ("desc_wg", "*fp8e4nv"),
    ("desc_sg", "*fp16"),
    ("desc_wu", "*fp8e4nv"),
    ("desc_su", "*fp16"),
    ("desc_wd", "*fp8e4nv"),
    ("desc_sd", "*fp16"),
    ("desc_o", "*fp16"),
    ("desc_ids", "*i32"),
    ("desc_rs", "*fp16"),
    ("desc_act", "*fp16"),
];

#[derive(Clone, Copy)]
struct Shape {
    m: usize,
    h: usize,
    i: usize,
    e: usize,
    k: usize,
}

const EPS: f64 = 1e-6;
const INV_QMAX: f64 = 1.0 / 448.0;
const SCALE_FLOOR: f64 = 1.0 / 448.0 / 448.0;

fn temp(s: Shape) -> f64 {
    (s.h as f64).powf(-0.5)
}

/// Which MoE kernel: the decode form (`moe.py`, its rows `M`) or the prefill form
/// (`moe_prefill.py`, its rows `T`, the dense route as its last parameter).
#[derive(Clone, Copy, PartialEq)]
enum Form {
    Decode,
    Prefill,
}

fn moe(s: Shape) -> Program {
    compile(Form::Decode, s)
}

fn compile(form: Form, s: Shape) -> Program {
    let (file, kernel, rows) = match form {
        Form::Decode => ("moe.py", "moe_fwd", "M"),
        Form::Prefill => ("moe_prefill.py", "moe_prefill_fwd", "T"),
    };
    let path = format!("{}/../../../kernels/{file}", env!("CARGO_MANIFEST_DIR"));
    let src = std::fs::read_to_string(&path).expect("read kernel");
    let args: Vec<(&str, &str)> = match form {
        Form::Decode => ARGS.to_vec(),
        Form::Prefill => ARGS[..12].iter().copied().chain([("desc_dense", "*fp16")]).collect(),
    };
    let mut signature: HashMap<String, ArgSpec> =
        args.iter().map(|(n, t)| (n.to_string(), ArgSpec::parse(t).expect("arg"))).collect();
    let constexprs: HashMap<String, Val> = [
        (rows, Val::Int(s.m as i128)),
        ("H", Val::Int(s.h as i128)),
        ("I", Val::Int(s.i as i128)),
        ("E", Val::Int(s.e as i128)),
        ("K", Val::Int(s.k as i128)),
        ("EPS", Val::Float(EPS)),
        ("INV_H", Val::Float(1.0 / s.h as f64)),
        ("TEMP", Val::Float(temp(s))),
        ("INV_QMAX", Val::Float(INV_QMAX)),
        ("SCALE_FLOOR", Val::Float(SCALE_FLOOR)),
    ]
    .into_iter()
    .map(|(n, v)| (n.to_string(), v))
    .collect();
    for n in constexprs.keys() {
        signature.insert(n.clone(), ArgSpec::Constexpr);
    }
    let spec = KernelSpec {
        kernel: kernel.to_string(),
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

fn nodes(p: &Program) -> Vec<&Node<ktir_superdsc::qty::Elems>> {
    p.segments
        .iter()
        .flat_map(|s| match s {
            Segment::Straight(n) => n.iter().collect::<Vec<_>>(),
            Segment::Walk(_) => panic!("the MoE kernel does not walk"),
        })
        .collect()
}

fn contractions(p: &Program) -> usize {
    nodes(p).iter().filter(|n| n.func == OpFunc::Contraction).count()
}

/// The three fp8 banks: gate, up and down (parameters 5, 7 and 9).
const BANKS: [u32; 3] = [5, 7, 9];

/// No copy reads a weight bank: every slab reaches its contraction where it is held.
fn no_bank_copy(p: &Program) {
    for n in nodes(p).iter().filter(|n| n.func == OpFunc::Copy) {
        for o in &n.inputs {
            if let Store::Param(q) = o.store {
                assert!(!BANKS.contains(&q.get()), "a copy reads weight bank {}", q.get());
            }
        }
    }
}

/// Every expert contraction reads its bank through a gather, never a copy of it.
fn banks_gathered(p: &Program) {
    let reads: Vec<_> = nodes(p)
        .iter()
        .filter(|n| n.func == OpFunc::Contraction)
        .flat_map(|n| n.inputs.iter())
        .filter(|o| BANKS.iter().any(|&b| o.store == Store::Param(ParamId::new(b))))
        .collect();
    assert_eq!(reads.len(), 3, "each bank is read by one contraction");
    assert!(
        reads.iter().all(|o| matches!(o.gather, Some(Gather::Rows { .. }))),
        "each bank is read through the routed ids' gather"
    );
}

#[test]
fn the_experts_are_three_contractions_at_any_token_and_expert_count() {
    let one = moe(Shape { m: 1, h: 128, i: 64, e: 64, k: 8 });
    let two = moe(Shape { m: 2, h: 128, i: 64, e: 64, k: 8 });
    let four = moe(Shape { m: 4, h: 128, i: 64, e: 64, k: 8 });
    let wide = moe(Shape { m: 1, h: 128, i: 64, e: 64, k: 16 });
    let many = moe(Shape { m: 1, h: 128, i: 64, e: 128, k: 8 });
    for (what, p) in [("M = 1", &one), ("M = 2", &two), ("M = 4", &four), ("K = 16", &wide), ("E = 128", &many)] {
        assert_eq!(contractions(p), 4, "{what}: the router and the three expert projections");
        no_bank_copy(p);
        banks_gathered(p);
    }
    // The tokens and the experts are axes and gathers, never more nodes. A pair's gate/up input
    // is its token's row repeated across the K slots: at one token that is a view, past one it
    // is one copy of the M·K·H activations (merging the tokens with the stride-0 slots), so the
    // count is one higher from M = 2 on and constant after.
    assert_eq!(nodes(&two).len(), nodes(&four).len(), "the node count does not grow with the tokens");
    assert_eq!(nodes(&one).len() + 1, nodes(&two).len(), "one token's repeated row is a view");
    assert_eq!(nodes(&one).len(), nodes(&many).len(), "the node count does not grow with the experts");
}

// ── THE HOST REFERENCE, written from moe.py's statement, op by op, under the evaluator's
// model: every f16 op computes in f32 and rounds to f16, sums fold in f32 in index order
// and round once, contractions accumulate in f32 over k ascending and round once. The
// kernel's f32 islands (the router's rsqrt, the gelu cubic and exp, the route softmax) are
// stated in f16 by the front end's type legalization (`legalize_types`: the card binds those
// ops at fp16), so the reference computes them in f16 too: the KTIR is what is lowered. ──

fn h(x: f32) -> f32 {
    f16::from_f32(x).to_f32()
}

fn e4m3(x: f32) -> f32 {
    if x == 0.0 {
        return 0.0;
    }
    let a = f64::from(x.abs()).min(448.0);
    let quantum = if a >= 2f64.powi(-6) { 2f64.powi(a.log2().floor() as i32 - 3) } else { 2f64.powi(-9) };
    let q = a / quantum;
    let r = q.floor();
    let up = q - r > 0.5 || (q - r == 0.5 && r % 2.0 != 0.0);
    (((if up { r + 1.0 } else { r }) * quantum).min(448.0) as f32).copysign(x)
}

struct Weights {
    xr: Vec<f32>,
    rg: Vec<f32>,
    rw: Vec<f32>,
    pes: Vec<f32>,
    xe: Vec<f32>,
    wg: Vec<f32>,
    sg: Vec<f32>,
    wu: Vec<f32>,
    su: Vec<f32>,
    wd: Vec<f32>,
    sd: Vec<f32>,
}

struct Out {
    ids: Vec<i32>,
    rs: Vec<f32>,
    o: Vec<f32>,
}

fn quantize(row: &[f32]) -> (f32, Vec<f32>) {
    let amax = row.iter().fold(0f32, |a, &x| a.max(x.abs()));
    let s = h(h(amax) * h(INV_QMAX as f32)).max(h(SCALE_FLOOR as f32)).min(448.0);
    let inv = h(1.0 / s);
    (s, row.iter().map(|&x| e4m3(h(x * inv).clamp(-448.0, 448.0))).collect())
}

fn gelu_tanh_times(gate: f32, up: f32) -> f32 {
    let g = gate;
    let cube = h(h(h(g * h(0.044715)) * g) * g);
    let v = h(h(g + cube) * h(1.595_769_121_605_730_8_f64 as f32));
    let e = h((-v).exp());
    h(h(gate / h(e + 1.0)) * up)
}

fn host(s: Shape, w: &Weights) -> Out {
    host_misrouted(s, w, 0)
}

/// The host reference with every routed expert read `shift` experts further on: shift 0 is
/// the reference, any other shift a reference the device must not match.
fn host_misrouted(s: Shape, w: &Weights, shift: usize) -> Out {
    let Shape { m, h: hd, i: id, e: ne, k } = s;
    let mut ids = vec![0i32; m * k];
    let mut rs = vec![0f32; m * k];
    let mut o = vec![0f32; m * hd];
    for t in 0..m {
        // The router norm and logits.
        let xr = &w.xr[t * hd..(t + 1) * hd];
        let ms = h(h(xr.iter().fold(0f32, |a, &x| a + h(x * x))) * h(1.0 / hd as f32));
        let r = h(1.0 / h(ms + h(EPS as f32)).sqrt());
        let xn: Vec<f32> = (0..hd).map(|c| h(h(xr[c] * r) * w.rg[c])).collect();
        let mut lg: Vec<f32> = (0..ne)
            .map(|e| h((0..hd).fold(0f32, |a, c| a + xn[c] * w.rw[e * hd + c])))
            .collect();
        // Top K: the row maximum, the highest lane equal to it, that lane masked out.
        let (mut vals, mut psel) = (vec![0f32; k], vec![0f32; k]);
        for j in 0..k {
            let mx = lg.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let idx = (0..ne).rev().find(|&e| lg[e] == mx).expect("a lane at the maximum");
            vals[j] = mx;
            ids[t * k + j] = idx as i32;
            psel[j] = w.pes[idx];
            lg[idx] = f32::NEG_INFINITY;
        }
        // The route weights: softmax of the selected logits × H^-0.5, × each expert's scale.
        let sv: Vec<f32> = vals.iter().map(|&v| h(v * h(temp(s) as f32))).collect();
        let mx = sv.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let ex: Vec<f32> = sv.iter().map(|&v| h(h(v - mx).exp())).collect();
        let sum = h(ex.iter().fold(0f32, |a, &x| a + x));
        for j in 0..k {
            rs[t * k + j] = h(h(ex[j] / sum) * psel[j]);
        }
        // The experts.
        let (sx, qx) = quantize(&w.xe[t * hd..(t + 1) * hd]);
        let mut acc = vec![0f32; hd];
        let mut downs = Vec::with_capacity(k);
        for j in 0..k {
            let e = (ids[t * k + j] as usize + shift) % ne;
            let proj = |bank: &[f32], scale: &[f32], c: usize| -> f32 {
                let raw = h((0..hd).fold(0f32, |a, x| a + qx[x] * bank[(e * id + c) * hd + x]));
                h(h(raw * sx) * scale[e * id + c])
            };
            let act: Vec<f32> = (0..id).map(|c| gelu_tanh_times(proj(&w.wg, &w.sg, c), proj(&w.wu, &w.su, c))).collect();
            let (sa, qa) = quantize(&act);
            let down: Vec<f32> = (0..hd)
                .map(|c| {
                    let raw = h((0..id).fold(0f32, |a, x| a + qa[x] * w.wd[(e * hd + c) * id + x]));
                    h(h(raw * sa) * w.sd[e * hd + c])
                })
                .collect();
            downs.push(down);
        }
        for (c, a) in acc.iter_mut().enumerate() {
            *a = h((0..k).fold(0f32, |s, j| s + h(downs[j][c] * rs[t * k + j])));
        }
        o[t * hd..(t + 1) * hd].copy_from_slice(&acc);
    }
    Out { ids, rs, o }
}

fn weights(s: Shape) -> Weights {
    let mut state = 0x2545_f491_4f6c_dd1d_u64;
    let mut uniform = |lo: f32, hi: f32| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        lo + (hi - lo) * ((state >> 40) as f32 / (1u64 << 24) as f32)
    };
    let Shape { m, h: hd, i: id, e: ne, .. } = s;
    let mut f16s = |n: usize, lo: f32, hi: f32| (0..n).map(|_| h(uniform(lo, hi))).collect::<Vec<_>>();
    let xr = f16s(m * hd, -2.0, 2.0);
    let rg = f16s(hd, 0.5, 1.5);
    let rw = f16s(ne * hd, -0.2, 0.2);
    let pes = f16s(ne, 0.5, 1.5);
    let xe = f16s(m * hd, -2.0, 2.0);
    let sg = f16s(ne * id, 0.001, 0.01);
    let su = f16s(ne * id, 0.001, 0.01);
    let sd = f16s(ne * hd, 0.001, 0.01);
    let mut fp8s = |n: usize| (0..n).map(|_| e4m3(uniform(-4.0, 4.0))).collect::<Vec<_>>();
    let wg = fp8s(ne * id * hd);
    let wu = fp8s(ne * id * hd);
    let wd = fp8s(ne * hd * id);
    Weights { xr, rg, rw, pes, xe, wg, sg, wu, su, wd, sd }
}

#[test]
fn the_lifted_moe_computes_the_host_moe() {
    let s = Shape { m: 2, h: 128, i: 64, e: 16, k: 8 };
    let w = weights(s);
    let p = moe(s);
    let f = |v: &[f32]| v.iter().map(|&x| f64::from(x)).collect::<Vec<f64>>();
    let params = vec![
        f(&w.xr),
        f(&w.rg),
        f(&w.rw),
        f(&w.pes),
        f(&w.xe),
        f(&w.wg),
        f(&w.sg),
        f(&w.wu),
        f(&w.su),
        f(&w.wd),
        f(&w.sd),
        vec![0.0; s.m * s.h],
        vec![0.0; s.m * s.k],
        vec![0.0; s.m * s.k],
        vec![0.0; s.m * s.k * s.i],
    ];
    let out = run(&p, params).expect("the program runs");
    let want = host(s, &w);
    let ids: Vec<i32> = out[12].iter().map(|&x| x as i32).collect();
    assert_eq!(ids, want.ids, "the routed experts");
    assert_eq!(out[13], f(&want.rs), "the route weights");
    assert!(want.o.iter().all(|x| x.is_finite()) && want.o.iter().any(|&x| x != 0.0), "a finite, nonzero output");
    assert_eq!(out[11], f(&want.o), "the output");
    // NEGATIVE CONTROL: the same check against a reference that reads each pair's neighbouring
    // expert must fail, so the check sees which expert the device read.
    assert_ne!(out[11], f(&host_misrouted(s, &w, 1).o), "a misrouted reference is not the device");
}

// ── THE PREFILL FORM: every expert over every token, weighted by the dense route. ──

#[test]
fn the_prefill_experts_are_three_contractions_at_any_prompt_and_expert_count() {
    let short = compile(Form::Prefill, Shape { m: 16, h: 128, i: 64, e: 16, k: 8 });
    let long = compile(Form::Prefill, Shape { m: 32, h: 128, i: 64, e: 16, k: 8 });
    let wide = compile(Form::Prefill, Shape { m: 16, h: 128, i: 64, e: 32, k: 8 });
    for (what, p) in [("T = 16", &short), ("T = 32", &long), ("E = 32", &wide)] {
        assert_eq!(contractions(p), 4, "{what}: the router and the three expert projections");
        no_bank_copy(p);
    }
    assert_eq!(nodes(&short).len(), nodes(&long).len(), "the node count does not grow with the prompt");
    assert_eq!(nodes(&short).len(), nodes(&wide).len(), "the node count does not grow with the experts");
}

/// The prefill host reference: the router as decode's, the dense route, then every expert over
/// every token, summed over the experts in expert order.
fn host_prefill(s: Shape, w: &Weights) -> (Vec<f32>, Vec<f32>) {
    host_prefill_misrouted(s, w, 0)
}

/// The prefill reference with each route weight placed `shift` lanes from its expert.
fn host_prefill_misrouted(s: Shape, w: &Weights, shift: usize) -> (Vec<f32>, Vec<f32>) {
    let Shape { m: rows, h: hd, i: id, e: ne, k } = s;
    let decode = host(s, w);
    let mut dense = vec![0f32; rows * ne];
    for t in 0..rows {
        for j in 0..k {
            dense[t * ne + (decode.ids[t * k + j] as usize + shift) % ne] = decode.rs[t * k + j];
        }
    }
    let mut o = vec![0f32; rows * hd];
    for t in 0..rows {
        let (sx, qx) = quantize(&w.xe[t * hd..(t + 1) * hd]);
        let downs: Vec<Vec<f32>> = (0..ne)
            .map(|e| {
                let proj = |bank: &[f32], scale: &[f32], c: usize| -> f32 {
                    let raw = h((0..hd).fold(0f32, |a, x| a + qx[x] * bank[(e * id + c) * hd + x]));
                    h(h(raw * sx) * scale[e * id + c])
                };
                let act: Vec<f32> =
                    (0..id).map(|c| gelu_tanh_times(proj(&w.wg, &w.sg, c), proj(&w.wu, &w.su, c))).collect();
                let (sa, qa) = quantize(&act);
                (0..hd)
                    .map(|c| {
                        let raw = h((0..id).fold(0f32, |a, x| a + qa[x] * w.wd[(e * hd + c) * id + x]));
                        h(h(raw * sa) * w.sd[e * hd + c])
                    })
                    .collect()
            })
            .collect();
        for c in 0..hd {
            o[t * hd + c] = h((0..ne).fold(0f32, |a, e| a + h(downs[e][c] * dense[t * ne + e])));
        }
    }
    (dense, o)
}

#[test]
fn the_lifted_prefill_moe_computes_the_host_moe() {
    let s = Shape { m: 16, h: 128, i: 64, e: 16, k: 8 };
    let w = weights(s);
    let p = compile(Form::Prefill, s);
    let f = |v: &[f32]| v.iter().map(|&x| f64::from(x)).collect::<Vec<f64>>();
    let params = vec![
        f(&w.xr),
        f(&w.rg),
        f(&w.rw),
        f(&w.pes),
        f(&w.xe),
        f(&w.wg),
        f(&w.sg),
        f(&w.wu),
        f(&w.su),
        f(&w.wd),
        f(&w.sd),
        vec![0.0; s.m * s.h],
        vec![0.0; s.m * s.e],
    ];
    let out = run(&p, params).expect("the program runs");
    let (dense, o) = host_prefill(s, &w);
    assert_eq!(out[12], f(&dense), "the dense route");
    assert!(o.iter().all(|x| x.is_finite()) && o.iter().any(|&x| x != 0.0), "a finite, nonzero output");
    assert_eq!(out[11], f(&o), "the output");
    // NEGATIVE CONTROL: a reference with each route weight one lane off must fail.
    assert_ne!(out[11], f(&host_prefill_misrouted(s, &w, 1).1), "a misrouted reference is not the device");
}
