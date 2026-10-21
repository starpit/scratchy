// SPDX-License-Identifier: Apache-2.0
//! PROBE (#799, never merged): the one-row 4-bit matvec at Qwen3.6-35B-A3B's decode shapes, per
//! kernel variant: today's `Fast` (4-row tiles), MLX's 8-row tile (`FastFold`), the generic kernel,
//! and `affine_qmv_fast_sk{2,4,8}` (the `Fast` tile with K split across simdgroup pairs). Each cell
//! is µs a dispatch over one command buffer of `ITERS` dispatches a barrier apart, the weights
//! rotated past the system cache, best of 5, and its GB/s over every byte read. Each variant's
//! output is checked against `Fast` on the same weights.
//!
//! Run: cargo test -p scratchy-target-metal --release --test zz_local_qmv_shape_probe -- \
//!     --ignored --nocapture

mod common;

use half::bf16;
use objc2_metal::MTLSize;
use scratchy_target_metal::aot::baked_build;
use scratchy_target_metal::detect_device;
use scratchy_target_metal::mtl4_dispatch::{
    Mtl4DispatchBatch, read_slice, shared_slice, shared_zeroed,
};
use scratchy_target_metal::specialized_pipeline_cache::{
    ConstantValue, PipelineKey, SpecializedPipelineCache,
};
use scratchy_target_metal::tape::ids::{KDimI32, NDimI32};
use scratchy_target_metal::tape::kernel_constants::{AffineCodes, AffineQmvConstants};
use scratchy_target_metal::tape::quantized::{
    DequantDtype, QmvKernel, ScaleDtype, qmv_dispatch_shape, qmv_kernel_static_name,
};

const ITERS: usize = 64;
const ROTATED_BYTES: usize = 128 << 20;
const GROUP: usize = 64;

const SHAPES: &[(&str, u32, u32, u32)] = &[
    ("out_proj/o_proj", 2048, 4096, 40),
    ("gdn in_proj", 12288, 2048, 30),
    ("q+gate", 8192, 2048, 10),
    ("k / v", 512, 2048, 20),
    ("shared gate+up", 1024, 2048, 40),
    ("shared down", 2048, 512, 40),
];

#[derive(Clone, Copy)]
enum Variant {
    Plain(QmvKernel),
    SplitK(u32),
}

fn size((width, height, depth): (u32, u32, u32)) -> MTLSize {
    MTLSize {
        width: width as usize,
        height: height as usize,
        depth: depth as usize,
    }
}

fn bf16_row(len: usize, v: f32) -> Vec<u16> {
    (0..len)
        .map(|i| bf16::from_f32(v * (1.0 + 0.001 * (i % 97) as f32)).to_bits())
        .collect()
}

#[test]
#[ignore = "measurement probe: run with --ignored --nocapture"]
fn qmv_one_row_shape_variants() {
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let device = di.device.clone();
    let cache = SpecializedPipelineCache::new(device.clone(), &[]).expect("pipeline cache");
    let variants = [
        ("fast4", Variant::Plain(QmvKernel::Fast)),
        ("mlx8", Variant::Plain(QmvKernel::FastFold)),
        ("generic", Variant::Plain(QmvKernel::Generic)),
        ("sk2", Variant::SplitK(2)),
        ("sk4", Variant::SplitK(4)),
        ("sk8", Variant::SplitK(8)),
    ];
    println!("one row, µs a dispatch (GB/s): {:?}", variants.map(|v| v.0));
    for &(name, n, k, per_token) in SHAPES {
        let (nu, ku) = (n as usize, k as usize);
        let code_bytes = nu * ku / 2;
        let groups = nu * ku / GROUP;
        let all_bytes = code_bytes + 2 * groups * 2;
        let copies = ROTATED_BYTES / all_bytes + 1;
        let weights: Vec<_> = (0..copies)
            .map(|c| {
                let codes: Vec<u32> = (0..code_bytes as u32 / 4)
                    .map(|i| i.wrapping_mul(2654435761).wrapping_add(c as u32))
                    .collect();
                (
                    shared_slice(&device, &codes),
                    shared_slice(&device, &bf16_row(groups, 0.002)),
                    shared_slice(&device, &bf16_row(groups, -0.015)),
                )
            })
            .collect();
        let x = shared_slice(&device, &bf16_row(ku, 0.5));
        let gain = shared_slice(&device, &bf16_row(ku, 0.1));
        let mut line = format!("{name:<16} {n:>6}x{k:<5} x{per_token:<3}");
        let mut reference: Option<Vec<u16>> = None;
        let mut best_fast = 0.0;
        for (tag, v) in variants {
            if let Variant::SplitK(sk) = v
                && (k / sk) % 512 != 0
            {
                line += &format!("  {tag} {:>13}", "-");
                continue;
            }
            let (fname, grid, threads) = match v {
                Variant::Plain(kernel) => {
                    let (g, t) = qmv_dispatch_shape(kernel, 1, n, 1);
                    let f = qmv_kernel_static_name(
                        kernel,
                        DequantDtype::Bf16,
                        ScaleDtype::Bf16,
                        4,
                        GROUP as u32,
                    );
                    (f.to_string(), g, t)
                }
                Variant::SplitK(sk) => (
                    format!("affine_qmv_fast_sk{sk}_bf16_s_bf16_gs_64_b_4_batch_0"),
                    (1, n / 4, 1),
                    (32, 2 * sk, 1),
                ),
            };
            let constants: Vec<ConstantValue> = AffineQmvConstants {
                k: KDimI32(k as i32),
                n: NDimI32(n as i32),
                codes: AffineCodes::AsWritten,
            }
            .into();
            let fname: &'static str = Box::leak(fname.into_boxed_str());
            let pso = baked_build(&cache, &PipelineKey::new("quantized_qmv", fname, constants))
                .expect(fname);
            let y = shared_zeroed(&device, nu * 2);
            let run = |iters: usize| {
                let mut batch = Mtl4DispatchBatch::begin(&device).expect("mtl4 batch");
                for i in 0..iters {
                    let (w, scales, biases) = &weights[i % copies];
                    let binds = [
                        (w, 0),
                        (scales, 1),
                        (biases, 2),
                        (&x, 3),
                        (&y, 4),
                        (&gain, 15),
                    ];
                    batch.encode(&pso, &binds, &[], &[], &[], size(grid), size(threads));
                    batch.barrier();
                }
                let t0 = std::time::Instant::now();
                batch.commit(true);
                t0.elapsed().as_secs_f64() * 1e6 / iters as f64
            };
            run(1);
            let out: Vec<u16> = read_slice(&y, nu);
            let diff = match &reference {
                None => {
                    reference = Some(out);
                    0.0f32
                }
                Some(r) => r
                    .iter()
                    .zip(&out)
                    .map(|(a, b)| {
                        (bf16::from_bits(*a).to_f32() - bf16::from_bits(*b).to_f32()).abs()
                            / bf16::from_bits(*a).to_f32().abs().max(1e-3)
                    })
                    .fold(0.0, f32::max),
            };
            run(ITERS);
            let best = (0..5).map(|_| run(ITERS)).fold(f64::MAX, f64::min);
            if tag == "fast4" {
                best_fast = best;
            }
            let gbs = all_bytes as f64 / best / 1e3;
            let flag = if diff > 0.02 { "!" } else { "" };
            line += &format!("  {tag} {best:6.1} ({gbs:3.0}){flag}");
        }
        println!("{line}   [{:.2} MB; fast4 x{per_token} = {:.2} ms a token]", all_bytes as f64 / 1e6,
            best_fast * per_token as f64 / 1e3);
    }
}

/// The best a dispatch of each size can stream, a barrier apart: `probe_stream_read_u4` over
/// rotated copies past the system cache, the best of several grid sizes.
#[test]
#[ignore = "measurement probe: run with --ignored --nocapture"]
fn stream_read_ceiling_by_dispatch_size() {
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let device = di.device.clone();
    let cache = SpecializedPipelineCache::new(device.clone(), &[]).expect("pipeline cache");
    let out = shared_zeroed(&device, 16);
    println!("stream read, a barrier apart: µs a dispatch (GB/s) by threadgroups of 256 threads");
    for mb in [0.59f64, 1.18, 4.72, 9.44, 14.16, 64.0, 286.0] {
        let bytes = ((mb * 1e6) as usize / 16) * 16;
        let words = bytes / 16;
        let copies = (ROTATED_BYTES / bytes + 1).min(64);
        let bufs: Vec<_> = (0..copies)
            .map(|c| shared_slice(&device, &vec![c as u32 + 1; bytes / 4]))
            .collect();
        let constants: Vec<ConstantValue> = AffineQmvConstants {
            k: KDimI32(words as i32),
            n: NDimI32(1),
            codes: AffineCodes::AsWritten,
        }
        .into();
        let pso = baked_build(
            &cache,
            &PipelineKey::new("quantized_qmv", "probe_stream_read_u4", constants),
        )
        .expect("probe_stream_read_u4");
        let mut line = format!("{mb:>7.2} MB");
        let mut best_all = f64::MAX;
        for tgs in [40u32, 160, 640, 2560] {
            let iters = if bytes > (64 << 20) { 8 } else { ITERS };
            let run = |iters: usize| {
                let mut batch = Mtl4DispatchBatch::begin(&device).expect("mtl4 batch");
                for i in 0..iters {
                    let binds = [(&bufs[i % copies], 0), (&out, 4)];
                    batch.encode(&pso, &binds, &[], &[], &[], size((tgs, 1, 1)), size((256, 1, 1)));
                    batch.barrier();
                }
                let t0 = std::time::Instant::now();
                batch.commit(true);
                t0.elapsed().as_secs_f64() * 1e6 / iters as f64
            };
            run(iters);
            let best = (0..5).map(|_| run(iters)).fold(f64::MAX, f64::min);
            best_all = best_all.min(best);
            line += &format!("  tg{tgs} {best:7.1} ({:3.0})", bytes as f64 / best / 1e3);
        }
        println!("{line}   best {:.0} GB/s", bytes as f64 / best_all / 1e3);
    }
}
