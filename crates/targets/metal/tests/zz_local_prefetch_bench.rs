// PROBE BRANCH ONLY, not for merge: does a 4-bit matvec run faster when its weights were just read by
// another dispatch? Per rep: evict (stream ~290 MB), then a "prefetch" matvec over W (warm) or over
// a same-shape W2 (cold control), a barrier, then the timed matvec over W.

use objc2_foundation::NSRange;
use objc2_metal::{
    MTL4ComputeCommandEncoder as _, MTL4CounterHeap as _, MTL4CounterHeapDescriptor,
    MTL4CounterHeapType, MTL4TimestampGranularity, MTLBuffer, MTLDevice, MTLResourceOptions,
    MTLSize,
};
use scratchy_target_metal::aot::baked_build;
use scratchy_target_metal::detect_device;
use scratchy_target_metal::mtl4_dispatch::Mtl4DispatchBatch;
use scratchy_target_metal::specialized_pipeline_cache::{PipelineKey, SpecializedPipelineCache};
use scratchy_target_metal::tape::ids::{KDimI32, NDimI32};
use scratchy_target_metal::tape::kernel_constants::{AffineCodes, AffineQmvConstants};

type Device = objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLDevice>>;
type Buffer = objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLBuffer>>;
type Pso = scratchy_target_metal::specialized_pipeline_cache::ComputePipelineState;

fn buf(device: &Device, bytes: usize, fill: u8) -> Buffer {
    let b = device
        .newBufferWithLength_options(
            bytes,
            MTLResourceOptions::StorageModeShared | MTLResourceOptions::HazardTrackingModeUntracked,
        )
        .unwrap();
    unsafe { std::ptr::write_bytes(b.contents().as_ptr() as *mut u8, fill, bytes) };
    b
}

struct W {
    w: Buffer,
    s: Buffer,
    b: Buffer,
    n: usize,
    k: usize,
}

fn weights(device: &Device, n: usize, k: usize, l: u8) -> W {
    W {
        w: buf(device, n * k / 2, 0x35 ^ l),
        s: buf(device, n * k / 64 * 2, 0x10),
        b: buf(device, n * k / 64 * 2, 0x20),
        n,
        k,
    }
}

fn nb(batch: &Mtl4DispatchBatch) {
    use objc2_metal::{MTL4CommandEncoder as _, MTL4VisibilityOptions, MTLStages};
    batch.encoder().barrierAfterEncoderStages_beforeEncoderStages_visibilityOptions(
        MTLStages::Dispatch,
        MTLStages::Dispatch,
        MTL4VisibilityOptions::None,
    );
}

/// Median µs of the timed matvec over `n x k` weights: (cold, warm).
fn prefetch_bench(n: usize, k: usize, reps: usize) -> (f64, f64) {
    let device = detect_device().unwrap().device.clone();
    let cache = SpecializedPipelineCache::new(device.clone(), &[]).unwrap();
    let pso = |n: usize, k: usize| {
        let c = AffineQmvConstants {
            k: KDimI32(k as i32),
            n: NDimI32(n as i32),
            codes: AffineCodes::AsWritten,
        };
        baked_build(
            &cache,
            &PipelineKey::new("quantized_qmv", "affine_qmv_fast_bf16_s_bf16_gs_64_b_4_batch_0", c.into()),
        )
        .unwrap()
    };
    let (p, p_evict) = (pso(n, k), pso(16384, 8192));
    let (w, w2) = (weights(&device, n, k, 1), weights(&device, n, k, 2));
    let evict: Vec<W> = (0..4).map(|l| weights(&device, 16384, 8192, 10 + l)).collect();
    let x = buf(&device, 16384 * 2, 0x3c);
    let (y, dummy, ye) = (buf(&device, n * 2, 0), buf(&device, n * 2, 0), buf(&device, 16384 * 2, 0));
    let desc = MTL4CounterHeapDescriptor::new();
    desc.setType(MTL4CounterHeapType::Timestamp);
    unsafe { desc.setCount(4 * reps) };
    let heap = device.newCounterHeapWithDescriptor_error(&desc).unwrap();
    let ns_per_tick = 1e9 / device.queryTimestampFrequency() as f64;
    let tg = |a: usize, b: usize| MTLSize { width: a, height: b, depth: 1 };
    let mv = |batch: &mut Mtl4DispatchBatch, p, wt: &W, out: &Buffer| {
        batch.encode(p, &[(&wt.w, 0), (&wt.s, 1), (&wt.b, 2), (&x, 3), (out, 4)], &[], &[], &[], tg(1, wt.n / 8), tg(32, 2));
    };
    let mut batch = Mtl4DispatchBatch::begin(&device).unwrap();
    for r in 0..2 * reps {
        let warm = r % 2 == 1;
        for e in &evict {
            mv(&mut batch, &p_evict, e, &ye);
        }
        nb(&batch);
        mv(&mut batch, &p, if warm { &w } else { &w2 }, &dummy);
        nb(&batch);
        unsafe {
            batch.encoder().writeTimestampWithGranularity_intoHeap_atIndex(
                MTL4TimestampGranularity::Precise,
                &heap,
                2 * r,
            )
        };
        mv(&mut batch, &p, &w, &y);
        unsafe {
            batch.encoder().writeTimestampWithGranularity_intoHeap_atIndex(
                MTL4TimestampGranularity::Precise,
                &heap,
                2 * r + 1,
            )
        };
        nb(&batch);
    }
    batch.commit(true);
    let data = unsafe { heap.resolveCounterRange(NSRange::new(0, 4 * reps)) }.unwrap();
    let ts: Vec<u64> = data
        .to_vec()
        .chunks_exact(8)
        .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let mut cold = Vec::new();
    let mut warm = Vec::new();
    for r in 0..2 * reps {
        let us = ts[2 * r + 1].saturating_sub(ts[2 * r]) as f64 * ns_per_tick / 1e3;
        if r % 2 == 1 { warm.push(us) } else { cold.push(us) }
    }
    let med = |v: &mut Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    let _ = w.k;
    (med(&mut cold), med(&mut warm))
}

#[test]
fn prefetch_into_cache() {
    for (name, n, k) in [
        ("k or v 1024x3072", 1024usize, 3072usize),
        ("o 3072x3072", 3072, 3072),
        ("down 3072x8192", 3072, 8192),
        ("gate 8192x3072", 8192, 3072),
        ("gate+up 16384x3072", 16384, 3072),
        ("2x gate+up 32768x3072", 32768, 3072),
    ] {
        let mb = (n * k / 2 + n * k / 64 * 4) as f64 / 1e6;
        let (cold, warm) = prefetch_bench(n, k, 20);
        eprintln!(
            "PREFETCH {name:20} {mb:5.1} MB: cold {cold:7.1} us ({:5.1} GB/s), warm {warm:7.1} us ({:5.1} GB/s)",
            mb * 1e3 / cold,
            mb * 1e3 / warm
        );
    }
}


/// The 28-layer Llama 3.2 3B matvec chain (q/k/v | o | gate/up | down, each layer its own
/// weights), ms per pass. Each step's dispatches are followed (no barrier between them) by reads
/// of the next step's weights into a dummy output: none (`head_mb` 0), the first rows of each of
/// its matvecs worth `head_mb` together, or all of them (`f64::INFINITY`).
fn chain(head_mb: f64) -> f64 {
    let device = detect_device().unwrap().device.clone();
    let cache = SpecializedPipelineCache::new(device.clone(), &[]).unwrap();
    let (h, kv, ff) = (3072usize, 1024usize, 8192usize);
    let pso = |n: usize, k: usize| {
        let c = AffineQmvConstants { k: KDimI32(k as i32), n: NDimI32(n as i32), codes: AffineCodes::AsWritten };
        baked_build(&cache, &PipelineKey::new("quantized_qmv", "affine_qmv_fast_bf16_s_bf16_gs_64_b_4_batch_0", c.into()))
            .unwrap()
    };
    let (p_hh, p_kv, p_ff, p_down) = (pso(h, h), pso(kv, h), pso(ff, h), pso(h, ff));
    let layers: Vec<[W; 7]> = (0..28u8)
        .map(|l| {
            [
                weights(&device, h, h, l),
                weights(&device, kv, h, l),
                weights(&device, kv, h, l),
                weights(&device, h, h, l),
                weights(&device, ff, h, l),
                weights(&device, ff, h, l),
                weights(&device, h, ff, l),
            ]
        })
        .collect();
    let act = |n: usize| buf(&device, n * 2, 0x3c);
    let (x, q, k, v, o, g, u) = (act(h), act(h), act(kv), act(kv), act(h), act(ff), act(ff));
    let (px, pd) = (act(ff), act(ff));
    let tg = |a: usize, b: usize| MTLSize { width: a, height: b, depth: 1 };
    let mv = |batch: &mut Mtl4DispatchBatch, p, wt: &W, i: &Buffer, out: &Buffer| {
        batch.encode(p, &[(&wt.w, 0), (&wt.s, 1), (&wt.b, 2), (i, 3), (out, 4)], &[], &[], &[], tg(1, wt.n / 8), tg(32, 2));
    };
    let ahead = |batch: &mut Mtl4DispatchBatch, next: &[(&Pso, &W, usize)]| {
        if head_mb <= 0.0 {
            return;
        }
        let bytes: f64 = next.iter().map(|&(_, w, k)| (w.n * k) as f64 * 0.5625).sum();
        let f = (head_mb * 1e6 / bytes).min(1.0);
        for &(p, w, _) in next {
            let rows = ((w.n as f64 * f) as usize / 8).max(1);
            batch.encode(p, &[(&w.w, 0), (&w.s, 1), (&w.b, 2), (&px, 3), (&pd, 4)], &[], &[], &[], tg(1, rows), tg(32, 2));
        }
    };
    let passes = 5;
    let mut best = f64::MAX;
    for _ in 0..4 {
        let mut batch = Mtl4DispatchBatch::begin(&device).unwrap();
        for _ in 0..passes {
            for (i, lw) in layers.iter().enumerate() {
                let nx = &layers[(i + 1) % layers.len()];
                mv(&mut batch, &p_hh, &lw[0], &x, &q);
                mv(&mut batch, &p_kv, &lw[1], &x, &k);
                mv(&mut batch, &p_kv, &lw[2], &x, &v);
                ahead(&mut batch, &[(&p_hh, &lw[3], h)]);
                nb(&batch);
                mv(&mut batch, &p_hh, &lw[3], &q, &o);
                ahead(&mut batch, &[(&p_ff, &lw[4], h), (&p_ff, &lw[5], h)]);
                nb(&batch);
                mv(&mut batch, &p_ff, &lw[4], &o, &g);
                mv(&mut batch, &p_ff, &lw[5], &o, &u);
                ahead(&mut batch, &[(&p_down, &lw[6], ff)]);
                nb(&batch);
                mv(&mut batch, &p_down, &lw[6], &g, &x);
                ahead(&mut batch, &[(&p_hh, &nx[0], h), (&p_kv, &nx[1], h), (&p_kv, &nx[2], h)]);
                nb(&batch);
            }
        }
        let t = std::time::Instant::now();
        batch.commit(true);
        best = best.min(t.elapsed().as_secs_f64());
    }
    best / passes as f64 * 1e3
}

#[test]
fn chain_read_ahead() {
    let heads = [0.0, 2.0, 4.0, 8.0, 16.0, f64::INFINITY];
    let mut best = [f64::MAX; 6];
    for _ in 0..3 {
        for (i, &mb) in heads.iter().enumerate() {
            best[i] = best[i].min(chain(mb));
        }
    }
    for (i, &mb) in heads.iter().enumerate() {
        let what = if mb.is_infinite() { "all of the next step".to_string() } else { format!("{mb} MB of the next step") };
        eprintln!("CHAIN read-ahead {what:22}: {:.3} ms per 28-layer pass ({:.0} GB/s)", best[i], 1.58e9 / best[i] / 1e6);
    }
}
