// SPDX-License-Identifier: Apache-2.0
//! THE TOKEN DRAW, RANDOM AND GREEDY, CHECKED. The on-card sampler (`kernels/sample_fwd.py`) is
//! lifted and run by the Program evaluator on staged tables built as production stages them,
//! and every row's drawn index must equal an independent host computation of the kernel's own
//! math: the int32 counter hash over (seed_lo, seed_hi, step, lane), the wrap-fold to `u`, the two
//! fp16 logs to the Gumbel noise, `score = x·inv_t + g·gate`, the pad law, and the first maximum.
//!
//! Both draws are checked: temperature > 0 (gate 1, the noise live) and temperature 0 (gate 0,
//! `inv_t` 1: exactly the first maximum of the logits). A request draws the same token in any
//! slot of a batched launch, and different seeds draw different tokens.

use std::collections::HashMap;

use half::f16;
use ktir_core::arena::Arena;
use ktir_superdsc::eval::run;
use ktir_superdsc::lift::lift;
use ktir_superdsc::program::Program;
use triton_frontend::codegen::{ArgSpec, KernelSpec};
use triton_frontend::semantic::Val;
use triton_frontend::target::Target;

const BLOCK: usize = 64;
const GROUP_BLOCKS: usize = 64;
const HC1: i32 = 0x9E37_79B1_u32 as i32;
const HC2: i32 = 0x85EB_CA77_u32 as i32;
const HC3: i32 = 0xC2B2_AE3D_u32 as i32;

const ARGS: [(&str, &str); 27] = [
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

fn sampler(vocab: usize, pad: usize, b: usize) -> Program {
    let path = format!("{}/../../../kernels/sample_fwd.py", env!("CARGO_MANIFEST_DIR"));
    let src = std::fs::read_to_string(&path).expect("read kernel");
    let mut signature: HashMap<String, ArgSpec> =
        ARGS.iter().map(|(n, t)| (n.to_string(), ArgSpec::parse(t).expect("arg"))).collect();
    let constexprs: HashMap<String, Val> = [
        ("V", vocab as i128),
        ("PAD_V", pad as i128),
        ("NB", (pad / BLOCK) as i128),
        ("BLOCK", BLOCK as i128),
        ("B", b as i128),
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
    let grid = [b as i64];
    let mut tt = triton_frontend::codegen::compile(&src, &spec, Target::spyre()).expect("codegen");
    triton_frontend::opt::make_ttir(&mut tt).expect("make_ttir");
    let mut m = triton_ktir::from_ttir::convert(&tt).expect("from_ttir");
    triton_ktir::make_ktir(&mut m, &grid).expect("make_ktir");
    triton_ktir::passes::to_ktir::run(&mut m, &grid).expect("to_ktir");
    let f = triton_ktir::passes::to_ktir_emit::lower(&m, Arena::global()).expect("handoff");
    lift(&f).expect("lift")
}

/// One request's row: its logits (fp16 values) and its seed quadruple and temperature.
#[derive(Clone)]
struct Request {
    logits: Vec<f32>,
    seed_lo: i32,
    seed_hi: i32,
    step: i32,
    /// `None` is the greedy draw (gate 0, `inv_t` 1).
    temperature: Option<f32>,
}

impl Request {
    fn gate(&self) -> i32 {
        i32::from(self.temperature.is_some())
    }
    fn inv_t(&self) -> f32 {
        self.temperature.map_or(1.0, |t| 1.0 / t)
    }
}

fn h16(x: f32) -> f32 {
    f16::from_f32(x).to_f32()
}

/// THE HOST REFERENCE, written from the kernel's statement, not from the lowering.
fn host_draw(r: &Request, vocab: usize, pad: usize) -> usize {
    let cmin = h16(2f32.powi(-14));
    let score = |i: usize| -> f32 {
        let lane = i as i32;
        let (slo, shi, stp) = (r.seed_lo, r.seed_hi, r.step);
        let mut h = lane.wrapping_mul(HC1).wrapping_add(slo);
        h = h.wrapping_mul(HC2).wrapping_add(shi).wrapping_add(stp);
        h = h.wrapping_mul(h);
        h = h.wrapping_mul(HC1).wrapping_add(slo);
        h = h.wrapping_mul(h);
        h = h.wrapping_mul(HC3).wrapping_add(shi).wrapping_add(stp);
        h = h.wrapping_mul(h);
        h = h.wrapping_mul(HC1).wrapping_add(lane.wrapping_mul(HC2)).wrapping_add(slo);
        let rr = h.wrapping_mul(256);
        let n = (rr as f32).abs() * 2f32.powi(-8);
        let np = n.min(8_388_607.0 - n).max(0.0);
        let u = (np + 0.5) * 2f32.powi(-22);
        let c1 = h16(u).max(cmin);
        let v = -h16(c1.ln());
        let c2 = v.max(cmin);
        let g = -h16(c2.ln());
        let x = r.logits[i] * r.inv_t() + g * r.gate() as f32;
        if i < vocab { x } else { x + f32::MIN }
    };
    let scores: Vec<f32> = (0..pad).map(score).collect();
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    scores.iter().position(|&s| s == max).expect("a maximum")
}

/// The staged tables, in the kernel's operand order, as production stages them (the definitions
/// `ktir-emulator`'s `sampler_on_emulator` pins to production's staging by hash), for `b` rows.
fn buffers(reqs: &[Request], vocab: usize, pad: usize) -> Vec<Vec<f64>> {
    let b = reqs.len();
    let groups = pad / (BLOCK * GROUP_BLOCKS);
    let row = |real: &dyn Fn(usize) -> usize, pad_lane: usize| -> Vec<f64> {
        (0..pad).map(|i| if i < vocab { real(i) } else { pad_lane } as f64).collect()
    };
    let per_row = |f: &dyn Fn(&Request) -> f64| -> Vec<f64> { reqs.iter().map(f).collect() };
    let words = |w: f64| -> Vec<f64> { vec![w; b] };
    vec![
        reqs.iter().flat_map(|r| r.logits.iter().map(|&x| f64::from(x))).collect(),
        per_row(&|r| f64::from(r.seed_lo)),
        per_row(&|r| f64::from(r.seed_hi)),
        per_row(&|r| f64::from(r.step)),
        per_row(&|r| f64::from(r.gate())),
        per_row(&|r| f64::from(r.inv_t())),
        row(&|i| GROUP_BLOCKS - (i / BLOCK) % GROUP_BLOCKS, 0),
        row(&|i| BLOCK - i % BLOCK, 0),
        row(&|i| (i / BLOCK) % GROUP_BLOCKS, GROUP_BLOCKS),
        row(&|i| i % BLOCK, BLOCK),
        vec![1.0; BLOCK],
        vec![64.0; BLOCK],
        (0..pad).map(|i| if i < vocab { 0.0 } else { f64::from(f32::MIN) }).collect(),
        row(&|i| groups - i / (BLOCK * GROUP_BLOCKS), 0),
        row(&|i| i / (BLOCK * GROUP_BLOCKS), groups),
        words(f64::from(HC1)),
        words(f64::from(HC2)),
        words(f64::from(HC3)),
        words(256.0),
        (0..pad).map(|i| i as f64).collect(),
        words(f64::from(2f32.powi(-8))),
        words(8_388_607.0),
        words(0.0),
        words(0.5),
        words(f64::from(2f32.powi(-22))),
        words(f64::from(h16(2f32.powi(-14)))),
        vec![0.0; b * BLOCK],
    ]
}

/// Every row's drawn index, as the lifted program computes it (all 64 lanes of a row's idx stick
/// must agree).
fn device_draws(p: &Program, reqs: &[Request], vocab: usize, pad: usize) -> Vec<usize> {
    let out = run(p, buffers(reqs, vocab, pad)).expect("the program runs");
    let idx = &out[26];
    (0..reqs.len())
        .map(|r| {
            let stick = &idx[r * BLOCK..(r + 1) * BLOCK];
            assert!(stick.iter().all(|&x| x == stick[0]), "row {r}: every lane of the idx stick agrees");
            stick[0] as usize
        })
        .collect()
}

/// Deterministic logits in fp16 with ties planted: the maximum appears twice, so the first-max law
/// is tested, not just a max.
fn logits(seed: u64, vocab: usize, pad: usize) -> Vec<f32> {
    let mut s = seed;
    let mut v: Vec<f32> = (0..pad)
        .map(|i| {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            if i < vocab { h16(((s >> 40) as f32 / (1u64 << 24) as f32) * 8.0 - 4.0) } else { 0.0 }
        })
        .collect();
    let (a, b) = ((seed as usize * 7919) % vocab, (seed as usize * 104_729) % vocab);
    let (lo, hi) = (a.min(b), a.max(b));
    v[lo] = 6.0;
    v[hi] = 6.0;
    v
}

fn request(seed: u64, temperature: Option<f32>, vocab: usize, pad: usize) -> Request {
    Request {
        logits: logits(seed, vocab, pad),
        seed_lo: (seed as i32).wrapping_mul(0x27d4_eb2d_u32 as i32),
        seed_hi: (seed as i32).wrapping_add(0x1656_67b1),
        step: (seed % 97) as i32,
        temperature,
    }
}

/// Two groups of the two-level ramp, a vocab short of the padded row and not a multiple of 64:
/// the last real block holds pad lanes, as production's 49155 (49155 mod 64 = 3) does.
const VOCAB: usize = 8003;
const PAD: usize = 8192;

#[test]
fn every_draw_is_the_host_draw_random_and_greedy() {
    let reqs: Vec<Request> = vec![
        request(1, Some(1.0), VOCAB, PAD),
        request(2, None, VOCAB, PAD),
        request(3, Some(0.7), VOCAB, PAD),
        request(4, None, VOCAB, PAD),
    ];
    let p = sampler(VOCAB, PAD, reqs.len());
    let got = device_draws(&p, &reqs, VOCAB, PAD);
    let want: Vec<usize> = reqs.iter().map(|r| host_draw(r, VOCAB, PAD)).collect();
    assert_eq!(got, want, "every row's draw is the host's");
    // The greedy rows are exactly the first maximum of the logits (the planted tie's first).
    for r in reqs.iter().filter(|r| r.temperature.is_none()) {
        let max = r.logits[..VOCAB].iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let first = r.logits.iter().position(|&x| x == max).expect("max");
        assert_eq!(host_draw(r, VOCAB, PAD), first);
    }
}

#[test]
fn the_random_draw_is_not_the_greedy_draw() {
    // Flat logits: greedy answers lane 0 every time; the noise must spread the draws.
    let flat = |seed: u64, t: Option<f32>| Request {
        logits: vec![0.0; PAD],
        ..request(seed, t, VOCAB, PAD)
    };
    let reqs: Vec<Request> = (10..14).map(|s| flat(s, Some(1.0))).chain([flat(20, None)]).collect();
    let p = sampler(VOCAB, PAD, reqs.len());
    let got = device_draws(&p, &reqs, VOCAB, PAD);
    assert_eq!(got[4], 0, "greedy on a flat row is its first lane");
    let random: std::collections::HashSet<usize> = got[..4].iter().copied().collect();
    assert!(random.len() > 1, "different seeds draw different tokens: {got:?}");
    assert!(got[..4].iter().all(|&i| i < VOCAB), "a pad lane never wins: {got:?}");
    let want: Vec<usize> = reqs.iter().map(|r| host_draw(r, VOCAB, PAD)).collect();
    assert_eq!(got, want);
}

#[test]
fn a_request_draws_the_same_token_in_any_slot() {
    let a = request(31, Some(0.9), VOCAB, PAD);
    let b = request(32, Some(1.3), VOCAB, PAD);
    let p1 = sampler(VOCAB, PAD, 1);
    let p2 = sampler(VOCAB, PAD, 2);
    let solo = device_draws(&p1, std::slice::from_ref(&a), VOCAB, PAD)[0];
    let ab = device_draws(&p2, &[a.clone(), b.clone()], VOCAB, PAD);
    let ba = device_draws(&p2, &[b, a], VOCAB, PAD);
    assert_eq!(ab[0], solo, "slot 0 of a pair draws the solo token");
    assert_eq!(ba[1], solo, "slot 1 of a pair draws the solo token");
    assert_eq!(ab[1], ba[0]);
}

#[test]
fn the_widest_batch_draws_every_row_as_the_host_does() {
    // The widest batch rung: 32 rows, every third one greedy.
    let reqs: Vec<Request> = (0..32u64)
        .map(|s| request(100 + s, (s % 3 != 0).then_some(0.5 + s as f32 / 16.0), VOCAB, PAD))
        .collect();
    let p = sampler(VOCAB, PAD, reqs.len());
    let got = device_draws(&p, &reqs, VOCAB, PAD);
    let want: Vec<usize> = reqs.iter().map(|r| host_draw(r, VOCAB, PAD)).collect();
    assert_eq!(got, want, "every row's draw is the host's");
}
