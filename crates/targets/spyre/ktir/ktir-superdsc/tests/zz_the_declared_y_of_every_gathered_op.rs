// SPDX-License-Identifier: Apache-2.0
//! ⭐⭐⭐⭐⭐ WHAT THE GATHERED DECODE BUNDLE **DECLARES**, OP BY OP, AT THREE WIDTHS — the emitter half
//! of the `job_bin_ptr + numCoresUsed_*128` launch refusal, obtained on a Mac in under a second, and the
//! gate on the gathered fold's request-batched legs.
//!
//! ## The card refusal this file localized
//! ```text
//! superdsc decode batch run_step (mq=8, start=97): compute launch rc=-1
//! CB status=Error locator=0x2  QGI addr=0x1c00000400 (939524104 flits)
//!     syndrome=0xc00 cases=[PrepZeroFlitCnt,PrepSwVer]
//! [segaddr] op[0] job_bin_ptr=0x1c00000000  PROG_OFFSET_BASE=0x1c00000000
//! ```
//! The offset is `mq * 128` — `mq` FLITS — measured at two widths (mq=8 → 0x400, mq=2 → 0x100), and
//! `0xc00` is syndrome bits 10|11, [`PrepZeroFlitCnt`] + [`PrepSwVer`]: the Prep unit read something at
//! that address that is neither a valid job header nor a non-zero flit count.
//!
//! ## ⭐⭐⭐⭐⭐ MEASURED ON `f8390f3bc`: THE FAULT IS IN THE **PROGRAM** SEGMENT, AND `bootstrap == 0`
//! The `[segaddr]` line above describes op[0] of whichever list launched FIRST — a PREFILL group, not the
//! op that faults — so it settled nothing about the fault. Re-measured per FAULTING op (the dump now runs
//! on every `rc != 0`, no knob), both rungs, `scr batch` on granite-3.1-2b fp8:
//! ```text
//! mq=2  FAULT op[2]  05f19f853f28683b/group_2   seg7/PROG region=128 offset=101508480 size=6016   (binary 6016 B)
//!       job_bin_ptr=0x1c00000000 bootstrap=0x0  prog_extent=6016 B = 47 flit(s)     QGI 0x1c00000100 = flit 2
//! mq=8  FAULT op[19] 6701a74bf9435e88/group_19  seg7/PROG region=128 offset=102826368 size=265728 (binary 265728 B)
//!       job_bin_ptr=0x1c00000000 bootstrap=0x0  prog_extent=265728 B = 2076 flit(s) QGI 0x1c00000400 = flit 8
//! ```
//! A QGI address is a DMVA and a DMVA's top bits ARE the segment (`SEGMENT_SIZE_BITS = 34`,
//! `PROG_SEGMENT = 7`), so `0x1c0000_0100 >> 34 == 7`: the PROGRAM segment. And no data segment can alias
//! it — at the faulting launch seg0..seg6 resolve to regions `17179869312`/`34359738496` at multi-GB
//! offsets while seg7 is region `128`. ⇒ **The "operand base resolved against the wrong segment" reading
//! is DEAD**, and with it the mask/index pitch (seg3's shift is 0 at the faulting launch), the two
//! synthetic scratch placements (seg0/4/5/6 shift 0) and the zeroed KV shift.
//!
//! ⛔ AND SO IS "ONE PAST THE PER-CORE PATCH TABLE". `bootstrap` is ZERO, so Prep started at flit 0 of the
//! group's own binary and walked `mq` flits IN before failing — flit 2 of 47 and flit 8 of 2076 are both
//! deep INSIDE the binary, not past anything. What survives: the job-header chain the Prep unit walks from
//! flit 0 is wrong at flit `mq` (it read a ZERO flit count there and a SW version that does not validate),
//! in a group whose op count is what `mq` scales.
//!
//! ## ⛔⛔⛔ THE BAKED JOB-HEADER CHAIN IS **NOT** THE DEFECT — 28 PROGRAMS PARSED, ALL WELL-FORMED
//! The header format is `deeptools/senulator/qg.h`'s `struct QGHeader` and the WRITER is
//! `deeptools/dip/dip.cpp:77-91`, which is the authority for the convention: one 128-B flit, `u8[0] =
//! SW_VER = 0xdd`, flit count in bits 21:8, terminal in bit 22, and — decisively — `myflits =
//! totalFlits_ - 1`, so a header's count EXCLUDES its own flit. Walking both faulting bundles' images
//! byte by byte under exactly that rule:
//! ```text
//! 05f19f853f28683b (mq=2)  g0 78 jobs  g1  90  g2  1  g3 2  g4 2  g5 8  g6 8  g7 51
//! 6701a74bf9435e88 (mq=8)  g0 78 jobs  g1 282  g2  1  g3..g10 2 each  g11..g18 8 each  g19 51
//! ```
//! EVERY header in all 28 programs carries `0xdd`, every declared count is exact, exactly the last job
//! of each program is terminal, and every chain ends at `1 + flits` == the file's own flit count with
//! nothing left over. ⇒ **The emission does not mis-declare a job count anywhere**, so "a header at flit
//! `mq` whose flit-count field is zero" is not something the bake wrote.
//!
//! What IS at flit `mq` is nothing at all: bytes 0..2 of flit 2 (mq=2) and flit 8 (mq=8) are `00 00 00`,
//! i.e. SW version `0x00` (≠ `0xdd`) and flit count 0 — **the `0xc00` syndrome verbatim**
//! (`prep_sw_ver` = bit 11, `prep_zero_flit_cnt` = bit 10, `app_data_sbf.hpp:360-361`). And
//! `qgi_address_flits` is `bootstrap.value() + mq` exactly (`0x38000000 + mq`, and `0x38000000` IS
//! `VirtualAddressSbf::new(7, 0).value()`), so the device ADVANCED `mq` flits from a correct bootstrap.
//! From flit 0 the baked header says the next job begins at flit 47 (mq=2) / 41 (mq=8) — never at `mq`.
//! ⇒ **Whatever told Prep the first job was `mq` flits long, it was not the header we baked.** The
//! remaining class is the DEVICE's copy of those bytes, not the bytes.
//!
//! ## ⛔ THE "44× SIZE CURVE" IS AN ARTEFACT OF COMPARING TWO DIFFERENT OPS
//! `6,016 B at mq=2 → 265,728 B at mq=8` is `group_2` measured against `group_19`, and those are not the
//! same op: `group_2` is a 1-job/47-flit program and `group_19` a 51-job/2076-flit one. `group_2` is
//! **6,016 B at BOTH rungs**, and the same-role 51-job group goes `g7` 1911 flits → `g19` 2076 flits —
//! **1.086×**, not 44×. Nothing scales super-linearly and there is no header-chain overrun to explain.
//!
//! What DOES scale is the GROUP COUNT: `4 + 2*mq` groups (8 at mq=2, 20 at mq=8), because the collapse
//! emits each per-request op as its OWN GROUP — `mq` copies of the 2-job group and `mq` of the 8-job
//! group — rather than `mq` ops inside one group. Only `group_1` grows as ops-in-a-group at all, by
//! exactly `32` jobs per request (`26 + 32*mq`: 90 at mq=2, 282 at mq=8, both under
//! `SCRATCHY_SUPERDSC_GROUP_SIZE=512`). So a request costs a PROGRAM and a LAUNCH, not a job.
//!
//! ## ⛔⛔⛔ A SEPARATE, PROVEN GAP THE BYTES EXPOSED: `ComputeOnHost`/`DataTransfer` IS UNIMPLEMENTED
//! A gather program's dxp job plan is THREE exec steps, not one. Measured over every retained
//! `spyrecode.json` on the pod — 24 of 85 carry a `ComputeOnHost`, and they are the gather probes, all
//! with the identical layout:
//! ```text
//! JobExecPlan:        ComputeOnHost  size=4224 (33 flits)  ohandle=progCorr  dsName_=ProgCorrectionFlit
//!                     DataTransfer   size=4224  dev_ptr=PROG_OFFSET_BASE + 1*128    (flits 1..33)
//!                     ComputeOnDevice           job_bin_ptr=PROG_OFFSET_BASE + 34*128
//! JobPreparationPlan: Allocate       size=133504 (1043 flits)
//!                     InitTransfer   size=125568 (981 flits)  dev_ptr=BASE + 34*128
//! ```
//! So the allocation is `[flit 0 gap][flits 1..33 correction][flits 34.. program]`, and execution starts
//! at flit 34. [`scratchy_spyre_bundle::correction`] types `DataTransfer` as
//! `JobCommand::Other` — "any step this port does not act on" — so the correction's DESTINATION is
//! discarded at parse time; `GroupCode::correction` then reaches the binary as const bytes with **no
//! runtime consumer anywhere** (`superdsc_exec.rs` never names it), and `Allocate`/`InitTransfer` are
//! never parsed at all, so the launch allocates `init_binary.len()` and H2Ds it at offset 0 regardless.
//! A correction region left ZERO is precisely a flit that reads back as SW version `0x00` and flit count
//! 0. ⛔ This is NOT the measured fault — these plans put the bootstrap at flit 34 and the faulting op
//! measured `bootstrap == 0` — but it is a real unimplemented step that fires the moment a shipped
//! group's dxp output carries a `ComputeOnHost`, and it fails with exactly this syndrome.
//!
//! ## ⛔ WHY A DIFF AND NOT A READING
//! `zz_diff_the_rung_descriptors` records what a hand-built score leg cost once: it measured
//! `MatY::of_requests` while the shipped bundle carried `MatY::of_gqa_group`, so the harness described a
//! different op than the card ran. Everything here calls `assemble_attn` itself, twice — with and
//! without the gather index — and reports only what DIFFERS. A shape present in both is, by
//! construction, a shape the card runs today.
//!
//! ## ⛔⛔⛔ THE FORM THAT FAULTED
//! The first collapsed fold carried the REQUESTS on `y`, which needs a per-batch 3-D `[y,in,out]` KERNEL.
//! Projected here, that op declared `y_=mq`, `mb_=1`, `numWkSlicesPerDim_ {y:mq, mb:1}` and
//! `numCoresUsed_ == mq` (`mb` is 1 and `out` is 64, which `work::matmul_cost_split`'s stick clause
//! forbids splitting, so `y` was the only splittable dim) — and its kernel showed **`mq` DISTINCT
//! per-core starts** where every shipped kernel shows ONE.
//!
//! It faulted at `job_bin_ptr + mq*128` at rungs 2, 4 and 8 — syndrome `0xc00`, locator `0x2`, cases
//! `[PrepZeroFlitCnt,PrepSwVer]` bit-identical, only the index moving. At mq=4 that op is `{mb:1, y:4}` on
//! FOUR cores, `matmul/dims.rs`'s on-hardware-proven solo-decode split, so the `y`-split VALUE is
//! exonerated. The reading "one flit per core, one past the program's per-core patch table" is refuted by
//! the per-fault `[segaddr]` above (`bootstrap == 0`, the fault flit inside the binary). ⛔ **THE CAUSE IS
//! NOT ESTABLISHED.** The unapplied `ComputeOnHost` correction above produces exactly this syndrome; the
//! request-batched legs' dxp plan for granite-3.1-2b fp8 carries NO `ComputeOnHost` (measured 2026-10-01
//! with a local `dxp_standalone`: `ComputeOnDevice`, `Allocate`, `InitTransfer` only).
//!
//! ## ⭐⭐⭐⭐⭐ WHAT THIS FILE GATES: THE REQUESTS ON `y`, ONE OP PER (QUERY HEAD, SLAB)
//! Emitting one leg op per kv head AND request — the request a baked offset — grew a gathered fold group
//! as `66 * width + 44` ops: 2,156 at width 32, in ONE dxp program the trip cap cannot cut, 128 s of dxp
//! alone; at hd=128 the 8b build spent ~40 minutes on it. The legs now carry the requests on `y`:
//!
//! | | shipped (`gather=false`) | gathered (`gather=true`) |
//! |---|---|---|
//! | prefix score/value op | `attn_pNsc_g{0..7}` — one per **kv head** | `attn_pNsc_q{0..31}` — one per **query head** |
//! | `N_` | `y=4` (the GQA group), `mb=mq` | **`y=mq`**, `mb=1` |
//! | `numWkSlicesPerDim_` | `{y:4, mb:mq}` | `{y:mq, mb:1}` |
//! | `numCoresUsed_` | 32 at mq=8, 8 at mq=2 | `mq` |
//! | kernel operand | `[in,out]`, **1** distinct per-core start | `[y,in,out]`, **`mq`** starts — each request's own page |
//! | fused epilogue | YES — 4 operands, actively `y`/`mb`-addressed | yes — per request, cloned from the output |
//!
//! ⭐ [`the_gathered_kernel_starts_are_each_requests_own_page`] is the oracle: every kernel start is
//! `PageScratch::coord_off` of that request — the pool's own address law — so a wrong `y` step fails it at
//! every rung, and the activation and output must step exactly one row per request.
//!
//! ⛔ AND THIS IS NOT THE PAIRING `attn.rs` FORBIDS. Its note says *"DO NOT PAIR `(gqa, request)` ONTO
//! ONE `y`"* — GARBAGE on the 8b at hd=128, 10/10 runs. That is a PACKED PAIR on a single axis
//! (`y = gqa*mq`) whose two components share one differenced step. Here `y` carries the requests ALONE;
//! the GQA group is not on an axis — one op per query head.
//!
//! ⛔ WHAT IS STILL UNMEASURED: a LAUNCH of this form. Every verdict in this file is the emitter's own
//! declaration read locally. The card gate is `scr batch` against a bs=1 solo oracle per row (the model's
//! own answers, never the textbook ones — an expected-answer detector reports ~9 phantom failures at
//! bs=1).

use ktir_superdsc::ir::bridge::tiled_op_sdsc_op::assemble_attn;
use ktir_superdsc::sdsc_abstract::{
    AttnGeometry, FeatIdx, KvCoord, KvHead, KvPlane, POOL_STICK, PageScratch, PagedKvPool,
    QueryRowCount, SlotCount, SlotWindow, attn_bundle_rows,
};
use ktir_superdsc::superdsc_opspec::{DataFormat, Fp16};

/// granite-3.1-2b — the model every card number is from.
const NQH: u32 = 32;
const NKVH: u32 = 8;
const HD: u32 = 64;
const CAP: u32 = PagedKvPool::PAGE_SLOTS as u32;
const ACTIVE_CAP: u32 = PagedKvPool::PAGE_SLOTS as u32;

/// One op's whole declaration, projected.
#[derive(Debug, Clone)]
struct OpPicture {
    name: String,
    /// `N_` extents the op uses, as sorted `(dim, size)`.
    iter: Vec<(String, i64)>,
    /// `numWkSlicesPerDim_`, sorted — how many slices each dim is cut into.
    split: Vec<(String, i64)>,
    /// The op-level `numCoresUsed_`.
    cores: i64,
    /// Per-operand `(layoutDimOrder_, maxDimSizes_, start-map entries, DISTINCT starts)`.
    operands: Vec<(Vec<String>, Vec<i64>, usize, usize)>,
    /// Per-operand DISTINCT per-core start addresses, ascending — the values the count above counts.
    starts: Vec<Vec<i64>>,
    /// `(label, factor)` of the fold attributes on operand 0 (core / corelet / time).
    fold: Vec<(String, i64)>,
}

impl OpPicture {
    fn y(&self) -> i64 {
        self.iter
            .iter()
            .find(|(k, _)| k == "y_")
            .map_or(-1, |(_, v)| *v)
    }
    /// The op's name with the per-head / per-pass suffixes replaced by `*`, so the 32 clones of one
    /// shape collapse to a single row. Without this the report is 300 identical lines.
    fn stem(&self) -> String {
        self.name
            .split('_')
            .map(|s| {
                let numbered = |p: char| {
                    s.len() > 1 && s.starts_with(p) && s[1..].chars().all(|c| c.is_ascii_digit())
                };
                if numbered('q') {
                    "q*".to_string()
                } else if numbered('p') {
                    "p*".to_string()
                } else if numbered('g') {
                    "g*".to_string()
                } else if numbered('r') {
                    // The collapsed fold's REQUEST index — one op per (kv head, request), so without
                    // this the report is `nkvh * mq` identical lines per pass instead of one.
                    "r*".to_string()
                } else {
                    s.to_string()
                }
            })
            .collect::<Vec<_>>()
            .join("_")
    }
    /// Everything but the name — the shape identity two bundles are compared on.
    fn shape(&self) -> String {
        let j = |v: &[(String, i64)], sep: &str| {
            v.iter()
                .map(|(k, n)| format!("{k}{sep}{n}"))
                .collect::<Vec<_>>()
                .join(",")
        };
        let ops = self
            .operands
            .iter()
            .map(|(l, d, tot, dis)| format!("[{}]{d:?} st{tot}({dis})", l.join(",")))
            .collect::<Vec<_>>()
            .join(" ");
        format!(
            "N_{{{}}} wk{{{}}} cores={} fold{{{}}} | {ops}",
            j(&self.iter, "="),
            j(&self.split, "/"),
            self.cores,
            j(&self.fold, "x"),
        )
    }
}

/// Emit the whole attention body at one decode width, with or without a gather index, and project every
/// op. `gather=false` is byte-for-byte what ships.
fn emit_at(mq: u32, gather: bool) -> Vec<OpPicture> {
    let geom = AttnGeometry::<NQH, NKVH, HD>::minted();
    let bundle_rows =
        attn_bundle_rows(geom, mq, true).unwrap_or_else(|| panic!("mq={mq} is not a baked rung"));
    let mut sym = 0i64;
    let ops = assemble_attn(
        0,
        geom,
        bundle_rows,
        CAP,
        ACTIVE_CAP,
        "t_qs",
        "t_new_k",
        "t_new_v",
        "t_kct",
        "t_vc",
        "t_pmask",
        "t_cmask",
        gather.then_some("t_kv_idx"),
        ktir_superdsc::place::PlaceId::Act(900),
        true,
        &mut sym,
        None,
    )
    .unwrap_or_else(|e| panic!("mq={mq} gather={gather}: assemble_attn refused: {}", e.0));

    ops.iter()
        .map(|e| {
            let v = serde_json::to_value(&e.op).unwrap();
            let (name, body) = v["dscs_"][0]
                .as_object()
                .and_then(|m| m.iter().next())
                .map(|(k, b)| (k.clone(), b.clone()))
                .expect("one named dsc per emitted op");
            let sorted_map = |x: &serde_json::Value| {
                let mut m: Vec<(String, i64)> = x
                    .as_object()
                    .map(|m| {
                        m.iter()
                            .filter_map(|(k, n)| n.as_i64().map(|i| (k.clone(), i)))
                            .collect()
                    })
                    .unwrap_or_default();
                m.sort();
                m
            };
            let operands = body["scheduleTree_"]
                .as_array()
                .map(|nodes| {
                    nodes
                        .iter()
                        .map(|n| {
                            let layout = n["layoutDimOrder_"]
                                .as_array()
                                .map(|a| {
                                    a.iter()
                                        .map(|d| d.as_str().unwrap_or("?").to_string())
                                        .collect()
                                })
                                .unwrap_or_default();
                            let maxd = n["maxDimSizes_"]
                                .as_array()
                                .map(|a| a.iter().map(|d| d.as_i64().unwrap_or(0)).collect())
                                .unwrap_or_default();
                            let tot = n["startAddressCoreCorelet_"]["data_"]
                                .as_object()
                                .map_or(0, |m| m.len());
                            (layout, maxd, tot, distinct_starts(n).len())
                        })
                        .collect()
                })
                .unwrap_or_default();
            let starts = body["scheduleTree_"]
                .as_array()
                .map(|nodes| nodes.iter().map(distinct_starts).collect())
                .unwrap_or_default();
            let fold = body["scheduleTree_"][0]["startAddressCoreCorelet_"]["dim_prop_attr"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .map(|d| {
                            (
                                d["label_"].as_str().unwrap_or("?").to_string(),
                                d["factor_"].as_i64().unwrap_or(0),
                            )
                        })
                        .collect()
                })
                .unwrap_or_default();
            OpPicture {
                name,
                iter: sorted_map(&body["N_"]),
                // ⛔ `numWkSlicesPerDim_` IS AN **OP-LEVEL** FIELD, not a `dscs_` one. Reading it off the
                // dsc body answers `{}` for every op in the bundle, which reads as "nothing is split".
                split: sorted_map(&v["numWkSlicesPerDim_"]),
                cores: v["numCoresUsed_"].as_i64().unwrap_or(-1),
                operands,
                starts,
                fold,
            }
        })
        .collect()
}

/// One `scheduleTree_` operand's DISTINCT per-core start addresses, ascending.
///
/// ⛔ THE START VALUES ARE STRINGS (`{"[0, 0, 0]": "0"}`). Reading them with `as_i64` alone reports ZERO
/// distinct starts for every operand — a broken extractor reading as a finding, paid for once already in
/// `zz_diff_the_rung_descriptors`.
fn distinct_starts(node: &serde_json::Value) -> Vec<i64> {
    node["startAddressCoreCorelet_"]["data_"]
        .as_object()
        .map(|m| {
            m.values()
                .filter_map(|x| {
                    x.as_str()
                        .and_then(|s| s.trim().parse::<i64>().ok())
                        .or_else(|| x.as_i64())
                })
                .collect::<std::collections::BTreeSet<_>>()
                .into_iter()
                .collect()
        })
        .unwrap_or_default()
}

/// The ops whose (stem, shape) pair appears in the gathered emission and NOT in the shipped one — what
/// the gather introduces, and the only population any of the assertions below may indict.
fn only_gathered(mq: u32) -> Vec<OpPicture> {
    let plain: std::collections::BTreeSet<(String, String)> = emit_at(mq, false)
        .iter()
        .map(|o| (o.stem(), o.shape()))
        .collect();
    emit_at(mq, true)
        .into_iter()
        .filter(|o| !plain.contains(&(o.stem(), o.shape())))
        .collect()
}

/// The report. `-- --nocapture` to read it.
#[test]
fn what_the_gather_changes() {
    for mq in [2u32, 8] {
        let mut seen: std::collections::BTreeSet<String> = Default::default();
        eprintln!("─── mq={mq}: shapes ONLY in the GATHERED emission ───");
        for o in only_gathered(mq) {
            let line = format!("{:<24} {}", o.stem(), o.shape());
            if seen.insert(line.clone()) {
                eprintln!("  + {line}");
            }
        }
        let gathered: std::collections::BTreeSet<(String, String)> = emit_at(mq, true)
            .iter()
            .map(|o| (o.stem(), o.shape()))
            .collect();
        eprintln!("─── mq={mq}: shapes ONLY in the SHIPPED emission ───");
        for o in emit_at(mq, false) {
            if gathered.contains(&(o.stem(), o.shape())) {
                continue;
            }
            let line = format!("{:<24} {}", o.stem(), o.shape());
            if seen.insert(line.clone()) {
                eprintln!("  - {line}");
            }
        }
    }
}

const GQA: u32 = NQH / NKVH;

/// A collapsed-fold prefix leg: one op per (query head, slab), named `attn_p{window}{sc|ov}_q{qh}…`.
fn is_gathered_leg(o: &OpPicture) -> bool {
    o.name.contains("sc_q") || o.name.contains("ov_q")
}

/// `N_`'s swept `mb_`.
fn mb(o: &OpPicture) -> Option<i64> {
    o.iter.iter().find(|(k, _)| k == "mb_").map(|(_, v)| *v)
}

/// How many slices `numWkSlicesPerDim_` cuts `dim` into — 1 when the dim is not split.
fn split_of(o: &OpPicture, dim: &str) -> i64 {
    o.split
        .iter()
        .find(|(k, _)| k == dim)
        .map_or(1, |(_, v)| *v)
}

/// ⭐ FACT 1 — EVERY GATHERED SCORE/VALUE LEG CARRIES THE BATCH'S REQUESTS ON `y`, AT ONE ROW.
///
/// The gather lays every request's page into a scratch of one page plane per request, and every other
/// operand of the two legs is request-MINOR, so the requests are a uniform batch axis: `y_ == mq` and
/// `mb_ == 1` at every rung. A leg whose `y_` is the GQA group is the per-request emission, whose fold
/// group grows as `66 * width + 44` ops — 2,156 at width 32, in ONE dxp program.
#[test]
fn every_gathered_leg_carries_the_requests_on_y_at_one_row() {
    for mq in [2u32, 4, 8] {
        let legs: Vec<OpPicture> = emit_at(mq, true)
            .into_iter()
            .filter(is_gathered_leg)
            .collect();
        assert!(
            !legs.is_empty(),
            "mq={mq}: the gathered emission has no collapsed-fold leg — the filter is broken, not the \
             emission"
        );
        for o in &legs {
            assert_eq!(
                (o.y(), mb(o)),
                (i64::from(mq), Some(1)),
                "mq={mq}: {} declares N_{:?} — a collapsed-fold leg sweeps ONE row per request with the \
                 {mq} requests on `y`",
                o.name,
                o.iter,
            );
        }
    }
}

/// ⭐ FACT 2 — THE GATHERED LEGS SPLIT ONLY THE REQUEST AXIS.
///
/// `mb` is one row and `out` one stick, which `work::matmul_cost_split`'s stick clause forbids splitting,
/// so `y` is the only split and each request lands on its own core: `numCoresUsed_ == mq`. The shipped
/// prefix score leg is checked alongside — `4 * mq` cores, the GQA group times the `mq` rows it sweeps —
/// so a change to the shared cost split shows up here as well as in the gathered form.
#[test]
fn the_gathered_legs_split_only_the_request_axis() {
    for (mq, shipped) in [(2u32, 8i64), (4, 16), (8, 32)] {
        let gathered = emit_at(mq, true);
        let legs: Vec<&OpPicture> = gathered.iter().filter(|o| is_gathered_leg(o)).collect();
        assert!(
            !legs.is_empty(),
            "mq={mq}: no collapsed-fold leg in the gathered emission"
        );
        for o in &legs {
            assert_eq!(
                (o.cores, split_of(o, "y"), split_of(o, "mb")),
                (i64::from(mq), i64::from(mq), 1),
                "mq={mq}: {} declares numCoresUsed_={} with wk={:?} — expected one request per core",
                o.name,
                o.cores,
                o.split,
            );
        }
        let ship = emit_at(mq, false);
        let g0 = ship
            .iter()
            .find(|o| o.name.starts_with("attn_p0sc_g0"))
            .expect("the shipped bundle has a prefix-pass-0 score leg per kv head");
        assert_eq!(
            g0.cores, shipped,
            "the shipped prefix score leg at mq={mq} should use {shipped} cores ({:?})",
            g0.split
        );
    }
}

/// ⛔⛔⛔ FACT 3 — THE FUSED EPILOGUE ON A `y`-BATCHED MATMUL **ALREADY SHIPS**, so splitting it into its
/// own op cannot be the fault.
///
/// The shipped prefix score and value legs are `y=4` batched matmuls with FOUR `scheduleTree_` operands
/// (`a`, the shared 2-D kernel, the output, and the fused epilogue source). This refutes the standing
/// "second cut" without a card run, which is the entire reason to project the emission locally.
#[test]
fn the_fused_epilogue_on_a_y_batched_matmul_already_ships() {
    for mq in [2u32, 8] {
        for stem in ["attn_p0sc_g0", "attn_p0ov_g0"] {
            let ship = emit_at(mq, false);
            let o = ship
                .iter()
                .find(|o| o.name.starts_with(stem))
                .unwrap_or_else(|| panic!("mq={mq}: the shipped bundle has no {stem}"));
            assert!(
                o.y() > 1,
                "mq={mq}: {stem} is meant to be `y`-batched, but N_={:?}",
                o.iter
            );
            assert_eq!(
                o.operands.len(),
                4,
                "mq={mq}: {stem} should carry a FUSED EPILOGUE as its 4th operand — if it does not, the \
                 refutation of the epilogue cut is void. operands={:?}",
                o.operands
                    .iter()
                    .map(|(l, _, _, _)| l.join(","))
                    .collect::<Vec<_>>(),
            );
            // ⛔ AND IT MUST BE ACTIVELY `y`-ADDRESSED, OR THE REFUTATION IS VOID. `attn.rs` documents a
            // BROADCAST pmask arm whose fused operand's address never advances per `y` or per `mb`; that
            // arm would establish nothing about the gathered form, whose epilogue advances on both. A
            // broadcast operand shows ONE distinct per-core start; this one must show as many as the
            // output does.
            let (epi, out) = (&o.operands[3], &o.operands[2]);
            assert_eq!(
                (epi.0.as_slice(), epi.3),
                (out.0.as_slice(), out.3),
                "mq={mq}: {stem}'s fused epilogue operand {:?} has {} distinct per-core start(s) against \
                 the output's {:?}/{} — a BROADCAST epilogue does not refute the epilogue cut.",
                epi.0,
                epi.3,
                out.0,
                out.3,
            );
        }
    }
}

/// A per-core start is TAGGED `operand slot << 34 | byte offset` — the convention `attn.rs`'s
/// `per_core_blocks` masks off — so the BYTE offset is the low 34 bits.
const START_TAG_SHIFT: u32 = 34;

/// One operand's distinct per-core starts as BYTE offsets, the slot tag masked off.
fn byte_starts(starts: &[i64]) -> Vec<i64> {
    starts
        .iter()
        .map(|s| s & ((1i64 << START_TAG_SHIFT) - 1))
        .collect()
}

/// Bytes per fp16 element — the attention operands' own format.
const ELEM_BYTES: i64 = <Fp16 as DataFormat>::WORD_LENGTH as i64;

/// The kernel's expected per-core starts, in BYTES, for one gathered leg: request `r`'s copy of the page
/// in the scratch, at this leg's (plane, kv head, window, slab) — [`PageScratch::coord_off`], the pool's
/// own address law plus a request row. The test does not pick the stride; the law does.
fn expected_kernel_starts(mq: u32, plane: KvPlane, kvh: KvHead, window: SlotWindow) -> Vec<i64> {
    let pool = PagedKvPool::new(NKVH as usize, HD as usize);
    let scratch = PageScratch::of_pass(pool, QueryRowCount::of_mq(mq))
        .unwrap_or_else(|| panic!("mq={mq}: the pool admits no page scratch"));
    let coord = KvCoord::block(plane, kvh)
        .at_slot(window.first_slot())
        .at_feat(FeatIdx::of_slab(0));
    (0..mq)
        .map(|r| {
            let off = scratch
                .coord_off(r, &pool, coord)
                .unwrap_or_else(|| panic!("mq={mq}: request {r} has no scratch row"));
            i64::try_from(off).expect("a scratch offset fits i64") * ELEM_BYTES
        })
        .collect()
}

/// ⭐⭐⭐⭐⭐ FACT 4 — ONLY THE GATHERED LEGS DECLARE A PER-BATCH 3-D `[y,in,out]` KERNEL, AND ITS PER-CORE
/// STARTS ARE EACH REQUEST'S OWN PAGE.
///
/// The oracle is [`PageScratch::coord_off`], not a stride chosen here: on its `mq` cores the kernel of
/// query head `qh`'s window-`w` score leg must start at exactly `{ coord_off(r, Kᵗ of kv head qh/gqa,
/// window w, slab 0) : r < mq }`, and its value leg at the same set on the V plane. A wrong `in` device
/// extent moves every start but request 0's, so a set compare catches it at every rung. The activation and
/// output, request-minor, must step exactly ONE ROW (one 64-element stick) per request. And no op of the
/// SHIPPED bundle may declare a 3-D kernel at all.
#[test]
fn the_gathered_kernel_starts_are_each_requests_own_page() {
    let nkvh = std::num::NonZeroU32::new(NKVH).expect("granite has kv heads");
    for mq in [2u32, 4, 8] {
        for o in emit_at(mq, false) {
            for (l, d, _, dis) in &o.operands {
                assert!(
                    !(l.len() == 3 && l[0] == "y" && l[1] == "in" && l[2] == "out"),
                    "mq={mq}: the SHIPPED {} declares a per-batch 3-D kernel {l:?}{d:?} ({dis} \
                     start(s)) — only the gathered fold's legs carry one",
                    o.name,
                );
            }
        }
        let gathered = emit_at(mq, true);
        let windows: Vec<SlotWindow> = SlotWindow::sweep(SlotCount::new(ACTIVE_CAP)).collect();
        for (wi, window) in windows.iter().enumerate() {
            for kvh in KvHead::all(nkvh) {
                for g in 0..GQA {
                    let qh = kvh.get() * GQA + g;
                    for (leg, plane) in [("sc", KvPlane::Kt), ("ov", KvPlane::V)] {
                        let name = format!("attn_p{wi}{leg}_q{qh}_o");
                        let o = gathered
                            .iter()
                            .find(|o| o.name.starts_with(&name))
                            .unwrap_or_else(|| panic!("mq={mq}: no op named {name}*"));
                        // Operand order is `[a, w, o, epi?]`, so the kernel is operand 1.
                        let (layout, _, _, _) = &o.operands[1];
                        assert_eq!(
                            layout.join(","),
                            "y,in,out",
                            "mq={mq}: {}'s kernel must be the per-request 3-D `[y,in,out]`",
                            o.name
                        );
                        assert_eq!(
                            byte_starts(&o.starts[1]),
                            expected_kernel_starts(mq, plane, kvh, *window),
                            "mq={mq}: {}'s kernel starts are not each request's own page in the \
                             gathered scratch — a request reading another request's keys",
                            o.name
                        );
                        for (i, what) in [(0usize, "activation"), (2, "output")] {
                            let s = byte_starts(&o.starts[i]);
                            let rows: Vec<i64> = s.iter().map(|v| v - s[0]).collect();
                            let one_row_each: Vec<i64> = (0..i64::from(mq))
                                .map(|r| r * i64::from(POOL_STICK) * ELEM_BYTES)
                                .collect();
                            assert_eq!(
                                rows, one_row_each,
                                "mq={mq}: {}'s {what} must step ONE ROW per request (it is \
                                 request-minor); starts {s:?}",
                                o.name
                            );
                        }
                    }
                }
            }
        }
    }
}

/// ⭐ FACT 5 — A GATHERED LEG CONTRACTS EXACTLY WHAT THE SHIPPED LEG CONTRACTS.
///
/// The collapse changes which rows an op computes and where its kernel comes from — never the matmul
/// itself. So every gathered leg's `in_`/`out_` equal the shipped pass-0 leg's of the same kind: the score
/// leg contracts one head-dim stick into one 64-slot window, the value leg one window into one stick.
#[test]
fn a_gathered_leg_contracts_what_the_shipped_leg_contracts() {
    let in_out = |o: &OpPicture| -> Vec<(String, i64)> {
        o.iter
            .iter()
            .filter(|(k, _)| k == "in_" || k == "out_")
            .cloned()
            .collect()
    };
    for mq in [2u32, 4, 8] {
        let (ship, gath) = (emit_at(mq, false), emit_at(mq, true));
        for leg in ["sc", "ov"] {
            let shipped = ship
                .iter()
                .find(|o| o.name.starts_with(&format!("attn_p0{leg}_g0_o")))
                .unwrap_or_else(|| panic!("mq={mq}: the shipped bundle has no attn_p0{leg}_g0"));
            let legs: Vec<&OpPicture> = gath
                .iter()
                .filter(|o| is_gathered_leg(o) && o.name.contains(&format!("{leg}_q")))
                .collect();
            assert!(!legs.is_empty(), "mq={mq}: no gathered {leg} leg");
            for o in legs {
                assert_eq!(
                    in_out(o),
                    in_out(shipped),
                    "mq={mq}: {} contracts {:?} where the shipped {} contracts {:?}",
                    o.name,
                    in_out(o),
                    shipped.name,
                    in_out(shipped),
                );
            }
        }
    }
}

/// ⭐ FACT 6 — EVERY `y` SPLIT IS THE GQA GROUP OR ONE, EXCEPT THE GATHERED LEGS', WHICH IS THE WIDTH.
///
/// Every shipped attention op splits `y` by 1 or the GQA group. The gathered legs split it by the
/// request count (FACT 2); nothing else in the gathered bundle may.
#[test]
fn every_y_split_is_the_gqa_group_or_one_but_the_gathered_legs() {
    for mq in [2u32, 4, 8] {
        for gather in [false, true] {
            for o in emit_at(mq, gather) {
                let ys = split_of(&o, "y");
                let expected_ok = if gather && is_gathered_leg(&o) {
                    ys == i64::from(mq)
                } else {
                    ys == 1 || ys == i64::from(GQA)
                };
                assert!(
                    expected_ok,
                    "mq={mq} gather={gather}: {} splits `y` {ys} ways. wk={:?} cores={}",
                    o.name, o.split, o.cores,
                );
            }
        }
    }
}

/// ⭐⭐⭐ FACT 7 — THE GATHERED PREFIX LEGS ARE ONE OP PER QUERY HEAD, AT EVERY WIDTH.
///
/// The count that made the fold group grow with the batch: per window, the shipped bundle has `nkvh` legs
/// of each kind, and the gathered one must have `nqh` — the SAME at every width, because the width rides
/// on `y`. `nkvh * mq` here is the per-request emission back.
#[test]
fn the_gathered_prefix_legs_are_one_op_per_query_head_at_every_width() {
    for mq in [2u32, 4, 8] {
        let count = |gather: bool, needle: &str| {
            emit_at(mq, gather)
                .iter()
                .filter(|o| o.name.starts_with(needle))
                .count()
        };
        for leg in ["attn_p0sc_", "attn_p0ov_"] {
            let (shipped, gathered) = (count(false, leg), count(true, leg));
            assert_eq!(
                (shipped, gathered),
                (NKVH as usize, NQH as usize),
                "mq={mq}: prefix-pass-0 {leg} legs — shipped {shipped}, gathered {gathered}. Expected \
                 `nkvh` and `nqh`: one op per query head serving every request."
            );
        }
    }
}
