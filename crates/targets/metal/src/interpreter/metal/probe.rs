//! LOCAL DIAGNOSTIC, NEVER COMMITTED: per-dispatch GPU timestamps of decode steps
//! (`SCRATCHY_PROBE`). Precise timestamps serialize dispatches, so the per-kernel times are each
//! dispatch run alone; the shares are what to read.
use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSRange;
use objc2_metal::{
    MTL4ComputeCommandEncoder, MTL4CounterHeap, MTL4CounterHeapDescriptor, MTL4CounterHeapType,
    MTL4TimestampGranularity, MTLDevice, MTLSize,
};

const SLOTS: usize = 4096;
const WARM: u64 = 8;
const EVERY: u64 = 100;

struct Probe {
    heap: Retained<ProtocolObject<dyn MTL4CounterHeap>>,
    ns_per_tick: f64,
    state: Mutex<State>,
}

#[derive(Default)]
struct State {
    labels: Vec<String>,
    totals: BTreeMap<String, (f64, u64)>,
    steps: u64,
    gpu_ns: f64,
    /// Per dispatch position: summed duration, its label and grid.
    per_index: Vec<(f64, String)>,
}

unsafe impl Send for Probe {}
unsafe impl Sync for Probe {}

fn probe() -> Option<&'static Probe> {
    static P: OnceLock<Option<Probe>> = OnceLock::new();
    P.get_or_init(|| {
        std::env::var_os("SCRATCHY_PROBE")?;
        let device = objc2_metal::MTLCreateSystemDefaultDevice()?;
        let desc = MTL4CounterHeapDescriptor::new();
        desc.setType(MTL4CounterHeapType::Timestamp);
        unsafe { desc.setCount(SLOTS) };
        let heap = device.newCounterHeapWithDescriptor_error(&desc).ok()?;
        let freq = device.queryTimestampFrequency();
        Some(Probe {
            heap,
            ns_per_tick: 1e9 / freq as f64,
            state: Mutex::default(),
        })
    })
    .as_ref()
}

fn granularity() -> MTL4TimestampGranularity {
    match std::env::var("SCRATCHY_PROBE").as_deref() {
        Ok("precise") => MTL4TimestampGranularity::Precise,
        _ => MTL4TimestampGranularity::Relaxed,
    }
}

fn stamp(enc: &ProtocolObject<dyn MTL4ComputeCommandEncoder>, p: &Probe, i: usize) {
    unsafe {
        enc.writeTimestampWithGranularity_intoHeap_atIndex(
            granularity(),
            &p.heap,
            i,
        )
    };
}

/// Before a forward's first dispatch.
pub fn begin(enc: &ProtocolObject<dyn MTL4ComputeCommandEncoder>) {
    let Some(p) = probe() else { return };
    p.state.lock().unwrap().labels.clear();
    stamp(enc, p, 0);
}

/// After each dispatch.
pub fn mark(
    enc: &ProtocolObject<dyn MTL4ComputeCommandEncoder>,
    kernel: impl std::fmt::Debug,
    tg: MTLSize,
    tpt: MTLSize,
    barrier: bool,
) {
    let Some(p) = probe() else { return };
    let mut s = p.state.lock().unwrap();
    if s.labels.len() + 1 >= SLOTS {
        return;
    }
    if s.steps == WARM && s.labels.len() < 4000 {
        let b = if barrier { "BARRIER" } else { "       " };
        eprintln!(
            "[probe-seq] {:3} {b} {kernel:?} grid {}x{}x{}",
            s.labels.len(),
            tg.width,
            tg.height,
            tg.depth
        );
    }
    let _ = tpt;
    s.labels.push(format!("{kernel:?} {}x{}x{}{}", tg.width, tg.height, tg.depth, if barrier { " B" } else { "" }));
    let i = s.labels.len();
    stamp(enc, p, i);
}

/// After the forward's GPU work drained.
pub fn report(num_tokens: usize) {
    let Some(p) = probe() else { return };
    let mut s = p.state.lock().unwrap();
    let n = s.labels.len();
    if n == 0 || num_tokens != 1 {
        return;
    }
    let Some(data) = (unsafe { p.heap.resolveCounterRange(NSRange::new(0, n + 1)) }) else {
        return;
    };
    let bytes = data.to_vec();
    let ts: Vec<u64> = bytes
        .chunks_exact(8)
        .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    if ts.len() < n + 1 {
        return;
    }
    s.steps += 1;
    if s.steps <= WARM {
        return;
    }
    let labels = std::mem::take(&mut s.labels);
    if s.per_index.len() < labels.len() {
        s.per_index.resize(labels.len(), (0.0, String::new()));
    }
    for (i, label) in labels.iter().enumerate() {
        let d = ts[i + 1].saturating_sub(ts[i]) as f64 * p.ns_per_tick;
        s.per_index[i].0 += d;
        s.per_index[i].1 = label.clone();
        let e = s.totals.entry(label.clone()).or_default();
        e.0 += d;
        e.1 += 1;
    }
    s.gpu_ns += ts[n].saturating_sub(ts[0]) as f64 * p.ns_per_tick;
    s.labels = labels;
    let steps = s.steps - WARM;
    if steps % EVERY != 0 {
        return;
    }
    let per = |ns: f64| ns / steps as f64 / 1000.0;
    let total = per(s.gpu_ns);
    let mut rows: Vec<_> = s.totals.iter().collect();
    rows.sort_by(|a, b| b.1.0.total_cmp(&a.1.0));
    eprintln!("[probe] {steps} decode steps: {total:.1} us/step serialized, {n} dispatches/step");
    // What bounds each dispatch: its kind and how many 8-row blocks it runs.
    let mut kinds: BTreeMap<&str, (f64, u64)> = BTreeMap::new();
    for (ns, label) in &s.per_index {
        let kind = label.split(' ').next().unwrap_or("");
        let rows = (label.split(' ').nth(1).and_then(|g| g.split('x').nth(1)))
            .and_then(|y| y.parse::<u64>().ok())
            .unwrap_or(0);
        let class = match kind {
            "AffineQmvFast" if rows >= 256 => "big matvec (>=2048 rows)",
            "AffineQmvFast" | "AffineQmv" | "AffineQmvGated" => "small matvec (<=512 rows)",
            k if k.starts_with("Moe") || k == "Gemm" => "MoE router + experts",
            "GatedDeltaNet" => "GatedDeltaNet",
            k if k.contains("Attention") || k.starts_with("RopeAppend") => "attention",
            _ => "tiny elementwise / norm / embed",
        };
        let e = kinds.entry(class).or_default();
        e.0 += ns / steps as f64 / 1000.0;
        e.1 += 1;
    }
    for (class, (us, count)) in &kinds {
        eprintln!("[probe-class] {class:32} {:8.1} us/step  x{count}", us);
    }
    for (i, (ns, label)) in s.per_index.iter().enumerate() {
        eprintln!("[probe-at] {i:4} {:8.2} us  {label}", ns / steps as f64 / 1000.0);
    }
    for (label, (ns, count)) in rows {
        eprintln!(
            "[probe] {:9.1} us/step {:5.1}%  x{:<4} {:7.2} us each  {label}",
            per(*ns),
            per(*ns) / total * 100.0,
            count / steps,
            *ns / *count as f64 / 1000.0
        );
    }
}
