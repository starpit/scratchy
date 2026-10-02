// SPDX-License-Identifier: Apache-2.0
//! MEASUREMENT PROBE (local branch `mk-probe`, never merged): per-dispatch GPU time of a forward.
//!
//! Armed by a test, the worker writes one precise MTL4 timestamp before its first dispatch and one
//! after each dispatch it encodes, tagged with the dispatch's flat index in the bucket's step list
//! (so a dispatch-path index is the expanded command's index). After the command buffer completes
//! the pool resolves the heap; the test takes each dispatch's (index, end - previous end) in ns.
//! Precise timestamps serialize the dispatches they separate, so overlap between independent
//! dispatches is not seen.

use std::sync::Mutex;

use objc2_foundation::NSRange;
use objc2_metal::{MTL4CounterHeap, MTL4CounterHeapDescriptor, MTL4CounterHeapType, MTLDevice};

use super::__re::{Device, MTL4ComputeCommandEncoder, MTL4TimestampGranularity, Mtl4CounterHeap};

struct Armed {
    heap: Mtl4CounterHeap,
    capacity: usize,
    /// Flat index of the dispatch each written timestamp (after the first) follows.
    tags: Vec<u32>,
    /// ns per tick.
    ns_per_tick: f64,
    /// Resolved forwards: per dispatch, (flat index, ns since the previous timestamp).
    done: Vec<Vec<(u32, f64)>>,
}

// SAFETY: the heap is only touched under the mutex.
unsafe impl Send for Armed {}

static PROBE: Mutex<Option<Armed>> = Mutex::new(None);

/// Arm the probe for forwards of up to `capacity - 1` dispatches.
pub fn arm(device: &Device, capacity: usize) {
    let desc = MTL4CounterHeapDescriptor::new();
    desc.setType(MTL4CounterHeapType::Timestamp);
    unsafe { desc.setCount(capacity) };
    let heap = device
        .newCounterHeapWithDescriptor_error(&desc)
        .expect("counter heap");
    let freq = device.queryTimestampFrequency();
    *PROBE.lock().unwrap() = Some(Armed {
        heap,
        capacity,
        tags: Vec::new(),
        ns_per_tick: 1e9 / freq as f64,
        done: Vec::new(),
    });
}

/// Disarm, returning every resolved forward.
pub fn disarm() -> Vec<Vec<(u32, f64)>> {
    PROBE
        .lock()
        .unwrap()
        .take()
        .map(|a| a.done)
        .unwrap_or_default()
}

/// Take the forwards resolved so far, staying armed.
pub fn take() -> Vec<Vec<(u32, f64)>> {
    PROBE
        .lock()
        .unwrap()
        .as_mut()
        .map(|a| std::mem::take(&mut a.done))
        .unwrap_or_default()
}

/// Before a forward's first dispatch.
pub(crate) fn begin(enc: &objc2::runtime::ProtocolObject<dyn MTL4ComputeCommandEncoder>) {
    let mut g = PROBE.lock().unwrap();
    let Some(a) = g.as_mut() else { return };
    a.tags.clear();
    unsafe {
        enc.writeTimestampWithGranularity_intoHeap_atIndex(
            MTL4TimestampGranularity::Precise,
            &a.heap,
            0,
        )
    };
}

/// After the dispatch at flat index `flat`.
pub(crate) fn after(
    enc: &objc2::runtime::ProtocolObject<dyn MTL4ComputeCommandEncoder>,
    flat: u32,
) {
    let mut g = PROBE.lock().unwrap();
    let Some(a) = g.as_mut() else { return };
    let at = a.tags.len() + 1;
    assert!(at < a.capacity, "probe heap too small");
    unsafe {
        enc.writeTimestampWithGranularity_intoHeap_atIndex(
            MTL4TimestampGranularity::Precise,
            &a.heap,
            at,
        )
    };
    a.tags.push(flat);
}

/// After the forward's command buffer completed.
pub(crate) fn resolve() {
    let mut g = PROBE.lock().unwrap();
    let Some(a) = g.as_mut() else { return };
    let n = a.tags.len() + 1;
    if n < 2 {
        return;
    }
    let data = unsafe { a.heap.resolveCounterRange(NSRange::new(0, n)) }.expect("resolve");
    let bytes = data.to_vec();
    let ticks: Vec<u64> = (bytes.as_chunks::<8>().0.iter())
        .map(|c| u64::from_le_bytes(*c))
        .collect();
    let out = a
        .tags
        .iter()
        .enumerate()
        .map(|(i, &flat)| {
            (
                flat,
                ticks[i + 1].saturating_sub(ticks[i]) as f64 * a.ns_per_tick,
            )
        })
        .collect();
    a.done.push(out);
    a.tags.clear();
}

/// Armed?
pub(crate) fn armed() -> bool {
    PROBE.lock().unwrap().is_some()
}
