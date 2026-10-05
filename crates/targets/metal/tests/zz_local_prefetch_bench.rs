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

