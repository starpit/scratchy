// SPDX-License-Identifier: Apache-2.0
//! ⭐⭐⭐⭐⭐ WHAT THE GATHERED DECODE BUNDLE **DECLARES**, OP BY OP, AT THREE WIDTHS — the emitter half
//! of the `job_bin_ptr + numCoresUsed_*128` launch refusal, obtained on a Mac in under a second, and now
//! the gate on the form that replaced the refused one.
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
//! ## ⛔⛔⛔ THE FORM THAT FAULTED, AND HOW THIS FILE CONVICTED IT
//! The first collapsed fold carried the REQUESTS on `y`, which needs a per-batch 3-D `[y,in,out]` KERNEL.
//! Projected here, that op declared `y_=mq`, `mb_=1`, `numWkSlicesPerDim_ {y:mq, mb:1}` and
//! `numCoresUsed_ == mq` (`mb` is 1 and `out` is 64, which `work::matmul_cost_split`'s stick clause
//! forbids splitting, so `y` was the only splittable dim) — and its kernel showed **`mq` DISTINCT
//! per-core starts** where every shipped kernel shows ONE.
//!
//! So `job_bin_ptr + mq*128` was read as `job_bin_ptr + numCoresUsed_*128`: one flit per CORE, at index
//! `cores`, one past the last one the program holds (`init_binary.bin = 1920 + 128*cores`; see
//! `one-block-per-core-is-an-11x-oversized-init` — dxp's 1-op/32-core group is 1 header + 8 base + 31
//! patches). ⛔ THAT WHOLE PARAGRAPH IS REFUTED by the per-fault `[segaddr]` above — `bootstrap == 0` and
//! the fault flit is inside the binary — and `numCoresUsed_` was refuted separately on card. It is kept
//! because the *declarations* it records are still what this file gates. **RUNG 4 closed it**: at mq=4 that op is `{mb:1, y:4}` on FOUR cores, which is
//! `matmul/dims.rs`'s on-hardware-proven solo-decode split, and it faulted at `+0x200` exactly like 2 at
//! `+0x100` and 8 at `+0x400` — syndrome `0xc00`, locator `0x2`, cases `[PrepZeroFlitCnt,PrepSwVer]`
//! bit-identical, only the index moving. ⇒ The `y`-split VALUE is exonerated and the KERNEL RANK is what
//! the address is made of.
//!
//! ## ⭐⭐⭐⭐⭐ WHAT THE FIX DECLARES INSTEAD: THE SHIPPED OP WITH A **REQUEST AXIS** INSERTED
//! ⛔ THE `_r{R}`-PER-REQUEST FORM ABOVE IS ALSO GONE, AND NOT BECAUSE IT FAULTED. It emitted `mq` ops in
//! one pass, each a copy of the shipped op at one row, reaching request `r`'s gathered block by a BAKED
//! kernel offset. That costs `mq` PROGRAMS and `mq` LAUNCHES (§"the 44× size curve" above measured
//! exactly that: `4 + 2*mq` groups, a request buying a program, not a job) — which is the cost curve the
//! collapse exists to remove. The form that ships now puts the requests on a real iteration dim, `x`, so
//! ONE op computes all `mq` rows: **one program, one launch, `nkvh` legs at every rung.**
//!
//! ⭐ AND `x` IS THE ONLY AXIS THAT CAN CARRY THEM. The gathered kernel is a per-request PAGE PLANE, so
//! the kernel operand is genuinely 3-D (`[x,in,out]`) and each request's plane sits
//! `PageScratch::cols()` elements — `nkvh*hd*PAGE_SLOTS` = 131072, i.e. **262144 bytes** — past the last.
//! No walk channel can declare a step that large (`maxDimSizes_` only ever shrinks; both gap channels are
//! dropped for a folded op — the arithmetic is written out in `matmul/dims.rs`), so the pitch must come
//! from the PER-CORE START ADDRESS, and `per_core_addr` only moves a start for a dim the work division
//! actually SPLIT. Hence `x` is core-split `mq` ways, one request per core slice, and
//! `matmul_opspec_fold_requests` makes any plan that failed to split it a **build error**.
//!
//! ⛔ THIS IS ALSO WHY `x` IS SAFE WHERE THE REFUTED FORM'S `y` WAS NOT. `bmm.ddl` classes `y` as `%wrd`
//! (weight-reuse) — the kernel's global layout does not carry it, so a per-core kernel coordinate along
//! `y` has no layout to resolve against. `x` is `%nrd` AND is a kernel layout dim here, so a per-core
//! kernel start along `x` resolves by construction. The distinction is the dim CLASS, not the splitting.
//!
//! | | shipped (`gather=false`) | gathered (`gather=true`) |
//! |---|---|---|
//! | prefix score/value op | `attn_pNsc_g{0..7}_o*` — one per **kv head** | **same: one per kv head**, `nkvh` legs, no `_r` suffix |
//! | `N_` | `y=4` (the GQA group), `mb=mq` | `y=4`, **`mb=1`**, **`x=mq`** |
//! | `numWkSlicesPerDim_` | `{y:4, mb:mq}` | `{y:4, **x:mq**}`, `mb` unsplit |
//! | `numCoresUsed_` | 8 / 16 / 32 at mq 2 / 4 / 8 | **the same 8 / 16 / 32** — the width moved axis, not budget |
//! | activation / output | `[y,mb,in]` `[4,mq,64]` | `[y,x,mb,in]` `[4,mq,1,64]` — `mb`'s extent moved to `x` |
//! | kernel operand | `[in,out]`, **1** distinct per-core start | **`[x,in,out]` `[mq,2048,64]`, `mq` starts, Δ262144 B** |
//! | fused epilogue | YES — 4 operands, actively `y`/`mb`-addressed | yes — 4 operands |
//!
//! ⭐ [`the_collapsed_op_is_the_shipped_op_with_a_request_axis_inserted`] is the sharp version: delete `x`
//! from every gathered declaration and restore the extent it took from `mb`, and what is left is the
//! shipped op byte for byte. At `mq == 1` no deletion is even needed — the two ARE the same op, the one
//! running at 41 tok/s, because `collapses()` is deliberately false there (a size-1 `x` is a phantom dim
//! that breaks dxp's contraction inference).
//!
//! ⭐ AND THE TRADE IS THE ONE TO WANT: `nkvh` ops in ONE pass against `nkvh` ops in each of `mq` passes —
//! the SAME op count over the step at `mq`× fewer launches — and each op computes `mq` DISTINCT rows
//! where the shipped one computes `mq` and masks `mq-1` away. Measured: ~28 µs per fold pass removed.
//!
//! ⛔ AND THIS IS NOT THE PAIRING `attn.rs` FORBIDS. Its note says *"DO NOT PAIR `(gqa, request)` ONTO
//! ONE `y`"* — GARBAGE on the 8b at hd=128, 10/10 runs. That is a PACKED PAIR on a single axis
//! (`y = gqa*mq`) whose two components share one differenced step. Here `y` carries the group ALONE,
//! exactly as it ships, and the request has an axis of its own.
//!
//! ⛔ WHAT IS STILL UNMEASURED: a LAUNCH of this form, and hd=128. Every verdict in this file is the
//! emitter's own declaration read locally. The card gate is `scr batch` against a bs=1 solo oracle per
//! row (the model's own answers, never the textbook ones — an expected-answer detector reports ~9 phantom
//! failures at bs=1). ⛔ AND THE PREVIOUS FORM'S 1.19× AT w8 DOES NOT TRANSFER: it was measured on the
//! wrong-values emission with a different work division (4 cores/leg, not `gqa*mq`), so the timing is
//! unowned until re-measured.

use ktir_superdsc::ir::bridge::tiled_op_sdsc_op::assemble_attn;
use ktir_superdsc::sdsc_abstract::{AttnGeometry, PagedKvPool, attn_bundle_rows};

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
    /// Per-operand SORTED DISTINCT per-core start addresses, in BYTES — the same set `operands.3`
    /// counts, kept whole because a COUNT cannot say whether consecutive starts are one page plane
    /// apart or one 64-element block apart, and that difference is the entire collapsed-fold defect
    /// (see [`the_collapsed_kernel_steps_one_whole_page_plane_per_request`]).
    start_set: Vec<Vec<i64>>,
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
                            // ⛔ THE START VALUES ARE STRINGS (`{"[0, 0, 0]": "0"}`). Reading them with
                            // `as_i64` alone reports ZERO distinct starts for every operand — a broken
                            // extractor reading as a finding, paid for once already in
                            // `zz_diff_the_rung_descriptors`.
                            let m = n["startAddressCoreCorelet_"]["data_"].as_object();
                            let tot = m.map_or(0, |m| m.len());
                            let set: std::collections::BTreeSet<i64> = m
                                .map(|m| {
                                    m.values()
                                        .filter_map(|x| {
                                            x.as_str()
                                                .and_then(|s| s.trim().parse::<i64>().ok())
                                                .or_else(|| x.as_i64())
                                        })
                                        .collect()
                                })
                                .unwrap_or_default();
                            (layout, maxd, tot, set)
                        })
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            let start_set: Vec<Vec<i64>> = operands
                .iter()
                .map(
                    |(_, _, _, s): &(
                        Vec<String>,
                        Vec<i64>,
                        usize,
                        std::collections::BTreeSet<i64>,
                    )| { s.iter().copied().collect() },
                )
                .collect();
            let operands: Vec<(Vec<String>, Vec<i64>, usize, usize)> = operands
                .into_iter()
                .map(|(l, d, tot, s)| (l, d, tot, s.len()))
                .collect();
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
                start_set,
                fold,
            }
        })
        .collect()
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

/// ⭐ FACT 1 — THE DECLARED `y` IS THE **GQA GROUP**, AT EVERY RUNG, AND NEVER THE RUNG WIDTH.
///
/// This is the assertion that inverts. The refused form declared `y_ == mq` (2 at rung 2, 8 at rung 8);
/// the collapsed fold declares `y_ == gqa` — the SAME batch axis the shipped bundle carries — with the
/// requests on no axis at all. A `y` that tracks the rung width again means the per-batch kernel is back,
/// and with it `numCoresUsed_ == mq` and the fault address.
///
/// Compared against the SHIPPED emission rather than against 1, because `attn_newkt*` legitimately
/// carries a width-invariant `y_=64` that has nothing to do with the batch.
#[test]
fn every_y_the_gather_introduces_is_the_gqa_group_and_never_the_rung_width() {
    const GQA: i64 = (NQH / NKVH) as i64;
    for mq in [2u32, 4, 8] {
        for o in only_gathered(mq) {
            let y = o.y();
            assert!(
                y <= 1 || y == GQA,
                "mq={mq}: {} declares y_={y}, which is neither 1 nor the GQA group ({GQA}). If it is the \
                 rung width the requests are back on `y`, which needs the per-batch 3-D kernel the card \
                 REFUSED at every rung — N_{:?}",
                o.name,
                o.iter,
            );
        }
    }
}

/// ⭐⭐⭐⭐⭐ FACT 2 — THE COLLAPSED FOLD USES **EXACTLY THE CORES THE SHIPPED FOLD USES** (`gqa*mq`), with
/// the width moved off `mb` and onto `x`: `{y: gqa, x: mq}` where the shipped leg carries `{y: gqa, mb: mq}`.
///
/// ⛔ THE REQUEST AXIS IS CORE-SPLIT, AND THAT IS A CORRECTNESS REQUIREMENT, NOT A WORK PREFERENCE. A
/// walked `x` bakes and launches and computes the WRONG VALUES — the kernel is a gathered PAGE PLANE and
/// only the per-core START address can carry its request pitch, which `per_core_addr` moves for a split
/// dim and for no other (see [`the_collapsed_kernel_steps_one_whole_page_plane_per_request`] for the
/// pitch itself, and `matmul/dims.rs`'s law for why no walk channel can declare it). So `x` splitting
/// `mq` ways is the thing being pinned, not a number that happened to fall out.
///
/// ⭐ AND THE CORE COUNT IS NOT A NEW NUMBER ON THE CARD. `gqa*mq` — 8/16/32 at rungs 2/4/8 — is what the
/// shipped fold already runs at every one of those rungs; only WHICH axis spends them moves. The
/// shipped leg spends them sweeping `mq` rows and masking `mq-1` away; this one spends them on `mq`
/// independent requests.
///
/// ⛔ THE SWEEP OVER THREE RUNGS IS WHAT MAKES THIS SHARP. At rung 4 the group and the width are the SAME
/// NUMBER — which is exactly why rung 4 was the card's discriminator — so no single-width assertion can
/// tell `y`-is-the-group from `y`-is-the-width. Checked at 2, 4 and 8 together, a constant `y=4` beside
/// an `x` that tracks the rung can only be the group and the width respectively.
///
/// ⛔ AND THE CEILING IS REAL: `gqa*mq <= MAX_CORES` caps this at mq=8, where it saturates exactly.
/// A wider rung is a `cargo build` error out of `matmul_opspec_fold_requests`, not a silent walk.
#[test]
fn the_collapsed_folds_core_split_is_the_group_on_y_and_the_width_on_x() {
    const GQA: i64 = (NQH / NKVH) as i64;
    for mq in [2u32, 4, 8] {
        let want = i64::from(mq) * GQA;
        let gathered = only_gathered(mq);
        let legs: Vec<&OpPicture> = gathered
            .iter()
            .filter(|o| o.name.contains("sc_g") || o.name.contains("ov_g"))
            .collect();
        assert!(
            !legs.is_empty(),
            "mq={mq}: the gathered emission introduced no collapsed score/value leg at all — the diff \
             is broken, not the emission"
        );
        for o in &legs {
            let sp = |d: &str| o.split.iter().find(|(k, _)| k == d).map_or(1, |(_, v)| *v);
            assert_eq!(
                (sp("y"), sp("x"), sp("mb"), o.cores),
                (GQA, i64::from(mq), 1, want),
                "mq={mq}: {} splits `y` {} / `x` {} / `mb` {} on {} cores, wanted the GQA group ({GQA}) \
                 on `y`, the whole width ({mq}) on `x`, `mb` unsplit and {want} cores. An `x` split \
                 below {mq} leaves some request without a core slice, and its kernel start then aliases \
                 another request's page plane — the form that baked, launched and computed garbage. \
                 wk={:?} N_={:?}",
                o.name,
                sp("y"),
                sp("x"),
                sp("mb"),
                o.cores,
                o.split,
                o.iter,
            );
        }
        // ⭐ THE SHIPPED LEG AT THE SAME RUNG, WHICH IS THE POINT: same core count, same `y`, and the
        // width on `mb` instead. Without this half the assertion above could not tell "the collapse
        // keeps the shipped core budget" from "the collapse happens to want gqa*mq cores".
        let ship = emit_at(mq, false);
        let g0 = ship
            .iter()
            .find(|o| o.name.starts_with("attn_p0sc_g0"))
            .expect("the shipped bundle has a prefix-pass-0 score leg per kv head");
        let sp = |d: &str| g0.split.iter().find(|(k, _)| k == d).map_or(1, |(_, v)| *v);
        assert_eq!(
            (g0.cores, sp("y"), sp("mb"), sp("x")),
            (want, GQA, i64::from(mq), 1),
            "the shipped prefix score leg at mq={mq} should spend {want} cores as {{y:{GQA}, mb:{mq}}} \
             with no `x` at all — if this moved, the collapsed form's core budget is being compared \
             against something other than what ships. wk={:?}",
            g0.split,
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

/// ⭐⭐⭐⭐⭐ FACT 4 — **THE COLLAPSED KERNEL IS `[x,in,out]`, AND CONSECUTIVE PER-CORE STARTS ARE EXACTLY
/// ONE WHOLE GATHERED PAGE PLANE APART.** This is the assertion the fix exists to satisfy, and it is the
/// one that separates the emission that computes RIGHT values from the one that baked, launched, and
/// computed garbage.
///
/// ⛔ A RANK AND A START **COUNT** CANNOT TELL THOSE TWO APART. The wrong-values form declared the same
/// rank-3 `[x,in,out]` kernel and the same `mq` "starts" — the difference was entirely in the STEP: the
/// card walked `x` by `in*out = 4096` elements (the block the op sweeps) where a request's page plane is
/// `PageScratch::cols()` = `nkvh*hd*PAGE_SLOTS` = 131072 elements. So this test asserts on the DELTA
/// between consecutive starts, in bytes, which is why `OpPicture` keeps the whole start set and not a
/// count.
///
/// ⭐ AND IT PINS THE DELTA TWO WAYS ON PURPOSE:
/// - against the op's OWN declared `maxDimSizes_` (`in_dev * out_dev * 2`), so a change to the declared
///   `in_dev` that silently rescales the pitch is caught; and
/// - against the geometry constant derived from first principles (`NKVH*HD*PAGE_SLOTS*2`), so a change to
///   the FOLD that keeps the two declarations consistent with each other but no longer steps a page is
///   caught too.
///
/// A single one of those checks is a tautology against the other half of the same emission.
///
/// ⛔ THE CARD-REFUTED `[y,in,out]` NEGATIVE IS KEPT, ACROSS BOTH BUNDLES AND ALL RUNGS. That form faulted
/// at `job_bin_ptr + numCoresUsed_*128` at rungs 2, 4 and 8 alike, and the reason is the dim CLASS: `y` is
/// `%wrd` in `bmm.ddl` and the kernel's global layout does not carry it, so a per-core kernel coordinate
/// along `y` has no layout to resolve against. `x` is `%nrd` and IS a kernel layout dim, so the shape
/// below is not the refuted one wearing a different letter.
///
/// ⛔ IT ASSERTS ON THE EMITTED DESCRIPTOR, NOT ON THE BUILDER. A builder that no longer has a `y`-kernel
/// door is not evidence: the rank is a property of `layoutDimOrder_`, and that is what this reads.
#[test]
fn the_collapsed_kernel_steps_one_whole_page_plane_per_request() {
    /// One gathered page's Kᵗ plane, in fp16 BYTES — derived here from the geometry rather than read
    /// off the emission, so this is an independent witness of `PageScratch::cols() * 2`.
    const PLANE_BYTES: i64 = (NKVH as i64) * (HD as i64) * (PagedKvPool::PAGE_SLOTS as i64) * 2;
    for mq in [2u32, 4, 8] {
        for gather in [false, true] {
            for o in emit_at(mq, gather) {
                for (l, d, _, dis) in &o.operands {
                    assert!(
                        !(l.len() == 3 && l[0] == "y" && l[1] == "in" && l[2] == "out"),
                        "mq={mq} gather={gather}: {} declares a per-batch 3-D kernel {l:?}{d:?} with \
                         {dis} distinct start(s). That form is REFUTED ON CARD — it faulted at \
                         `job_bin_ptr + numCoresUsed_*128` at rungs 2, 4 and 8 alike, because `y` is a \
                         `%wrd` dim the kernel's global layout does not carry.",
                        o.name,
                    );
                }
            }
        }
        // ⭐ AND POSITIVELY: every collapsed-fold leg's KERNEL is `[x,in,out]` with one start per request,
        // each one page plane on from the last. Operand order is `[a, w, o, epi?]`, so it is operand 1.
        let legs: Vec<OpPicture> = only_gathered(mq)
            .into_iter()
            .filter(|o| o.name.contains("sc_g") || o.name.contains("ov_g"))
            .collect();
        assert!(
            !legs.is_empty(),
            "mq={mq}: no collapsed-fold leg in the gathered emission — the diff is broken"
        );
        for o in &legs {
            let k = &o.operands[1];
            assert_eq!(
                (k.0.join(","), k.1.len(), k.3),
                ("x,in,out".to_string(), 3, mq as usize),
                "mq={mq}: {}'s kernel declares {:?}{:?} with {} distinct per-core start(s). The gathered \
                 kernel is a per-request PAGE PLANE, so it must be rank-3 on `x` with one start per \
                 request — a single start means every request would read request 0's plane.",
                o.name,
                k.0,
                k.1,
                k.3,
            );
            assert_eq!(
                k.1,
                vec![
                    i64::from(mq),
                    PLANE_BYTES / 2 / i64::from(HD),
                    i64::from(HD)
                ],
                "mq={mq}: {}'s kernel maxDimSizes_ is {:?}, wanted `[mq, in_dev, out_dev]` where \
                 `in_dev * out_dev` is one whole page plane ({} elements). The per-core start pitch is \
                 COMPUTED from these two numbers, so they are the pitch's only source.",
                o.name,
                k.1,
                PLANE_BYTES / 2,
            );
            // ⭐ THE PITCH ITSELF. `k.1[1] * k.1[2] * 2` is what the emission's own declaration implies;
            // `PLANE_BYTES` is what the geometry demands. Both, or the check is circular.
            let want = k.1[1] * k.1[2] * 2;
            assert_eq!(
                want,
                PLANE_BYTES,
                "mq={mq}: {}'s kernel declares a {want}-byte block per request, but a gathered page \
                 plane is {PLANE_BYTES} B (nkvh {NKVH} * hd {HD} * slots {} * 2). The declaration and \
                 the geometry have come apart.",
                o.name,
                PagedKvPool::PAGE_SLOTS,
            );
            let starts = &o.start_set[1];
            let deltas: Vec<i64> = starts.windows(2).map(|w| w[1] - w[0]).collect();
            assert_eq!(
                deltas,
                vec![PLANE_BYTES; mq as usize - 1],
                "mq={mq}: {}'s kernel per-core starts are {starts:?} — deltas {deltas:?}, wanted every \
                 consecutive pair exactly {PLANE_BYTES} B apart. ⛔ THIS IS THE WRONG-VALUES DEFECT \
                 VERBATIM: the form that baked and launched and computed garbage advanced by the `in x \
                 out` BLOCK THE OP SWEEPS (4096 elements, 8192 B) instead of one whole page plane, \
                 because a WALKED `x` can only step by what the walk declares. A start set that is right \
                 in COUNT and wrong in STRIDE reads a slice of request 0's plane for every request.",
                o.name,
            );
        }
        // ⭐ AND THE SHIPPED KERNEL, for the contrast: rank-2 and core-invariant. Without this half, an
        // emission that gave EVERY kernel a request axis would pass the positive check above.
        let ship = emit_at(mq, false);
        let g0 = ship
            .iter()
            .find(|o| o.name.starts_with("attn_p0sc_g0"))
            .expect("the shipped bundle has a prefix-pass-0 score leg per kv head");
        let k = &g0.operands[1];
        assert_eq!(
            (k.0.join(","), k.3),
            ("in,out".to_string(), 1),
            "the shipped prefix score leg's kernel should stay `[in,out]` read from ONE address — it is \
             the shared natural-K plane every core reads. Got {:?} with {} distinct start(s).",
            k.0,
            k.3,
        );
    }
}

/// ⭐⭐⭐⭐⭐ FACT 4b — **THE COLLAPSED OP *IS* THE SHIPPED FOLD OP WITH A REQUEST AXIS INSERTED.** The
/// sharpest statement of the fix, and the reason it is worth a card run: delete `x` from every gathered
/// declaration and give `mb` back the extent `x` took from it, and what is left is the shipped prefix
/// leg's declaration — same layouts, same contraction, same output width, same fused epilogue.
///
/// So the shape is not new on the card. At `mq == 1` no deletion is needed at all: the collapse is
/// deliberately OFF there (a size-1 `x` is a phantom dim that breaks dxp's contraction inference), so the
/// gathered leg is byte-identical to the 41 tok/s solo-decode op and still carries its `_r0` name.
///
/// ⛔ THE KERNEL IS EXCLUDED FROM THE MAXDIM COMPARISON, AND THAT IS NOT A WEAKENING. The gathered kernel
/// legitimately declares `[mq, in_dev, out_dev]` where the shipped one declares an unwalked `[-1,-1]` —
/// it is a different operand kind (a gathered page plane against a shared weight), and it is pinned
/// exactly, both rank and pitch, by [`the_collapsed_kernel_steps_one_whole_page_plane_per_request`].
/// Comparing it here would only force this test to restate that one loosely.
#[test]
fn the_collapsed_op_is_the_shipped_op_with_a_request_axis_inserted() {
    for mq in [2u32, 4, 8] {
        let (ship, gath) = (emit_at(mq, false), emit_at(mq, true));
        for stem in ["attn_p0sc_g0_o", "attn_p0ov_g0_o"] {
            let find = |v: &[OpPicture]| {
                v.iter()
                    .find(|o| o.name.starts_with(stem))
                    .unwrap_or_else(|| panic!("mq={mq}: no op named {stem}*"))
                    .clone()
            };
            let (s, g) = (find(&ship), find(&gath));
            // ⭐ OPERAND DECLARATIONS WITH `x` DELETED. For the activation/output/epilogue operands the
            // request axis TOOK its extent from `mb` (`[y,mb,in][4,mq,64]` became
            // `[y,x,mb,in][4,mq,1,64]`), so dropping `x` and restoring `mb`'s extent must reproduce the
            // shipped declaration exactly. The kernel (operand 1) is FACT 4's job — see the doc above.
            let decl = |o: &OpPicture, restore_mb: bool| -> Vec<(Vec<String>, Vec<i64>)> {
                o.operands
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != 1)
                    .map(|(_, (l, d, _, _))| {
                        let mut l2 = Vec::new();
                        let mut d2 = Vec::new();
                        let mut took = 1i64;
                        for (i, name) in l.iter().enumerate() {
                            let ext = d.get(i).copied().unwrap_or(-1);
                            if name == "x" {
                                took = ext;
                                continue;
                            }
                            l2.push(name.clone());
                            d2.push(if restore_mb && name == "mb" && ext == 1 {
                                took
                            } else {
                                ext
                            });
                        }
                        (l2, d2)
                    })
                    .collect()
            };
            assert_eq!(
                decl(&s, false),
                decl(&g, true),
                "mq={mq}: with `x` removed and its extent handed back to `mb`, the gathered {stem}*'s \
                 operand declarations must be the shipped {stem}*'s exactly. A difference here is a NEW \
                 shape on the card, not a request axis — and needs its own justification.\n  shipped: \
                 {:?}\n  gathered: {:?}",
                s.operands,
                g.operands,
            );
            // `N_`: identical once `x_` is dropped, except `mb_`, which the request axis emptied.
            let iter_but = |o: &OpPicture| -> Vec<(String, i64)> {
                o.iter
                    .iter()
                    .filter(|(k, _)| k != "mb_" && k != "x_")
                    .cloned()
                    .collect()
            };
            assert_eq!(
                iter_but(&s),
                iter_but(&g),
                "mq={mq}: {stem}* gathered and shipped must agree on every iteration extent but `mb_` \
                 and the inserted `x_`"
            );
            let ext = |o: &OpPicture, k: &str| o.iter.iter().find(|(n, _)| n == k).map(|(_, v)| *v);
            assert_eq!(
                (ext(&s, "mb_"), ext(&s, "x_"), ext(&g, "mb_"), ext(&g, "x_")),
                (Some(i64::from(mq)), None, Some(1), Some(i64::from(mq))),
                "mq={mq}: the shipped leg sweeps `mq` rows on `mb` with no `x` at all; the collapsed one \
                 must hold `mb_=1` and carry the whole width on `x_`. That MOVE is the fix — the shipped \
                 leg computes `mq` rows and masks `mq-1` away, this one computes `mq` distinct requests."
            );
        }
        // ⛔ AND `mq == 1` IS NOT THIS SHAPE AT ALL, BY DESIGN. `GatheredFold::collapses()` is false at one
        // request because a size-1 `x` is a PHANTOM dim that breaks dxp's contraction inference, and mq=1
        // is the hardware-proven 41 tok/s bundle. So the gathered leg there still carries `_r0` and has no
        // `x`. Asserted so that "the collapse turned itself on at 1" is a build failure, not a card one.
        let solo = emit_at(1, true);
        let r0 = solo
            .iter()
            .find(|o| o.name.starts_with("attn_p0sc_g0_r0_o"))
            .unwrap_or_else(|| {
                panic!(
                    "mq=1: the gathered bundle must keep the per-request `_r0` solo-decode leg — names: \
                     {:?}",
                    solo.iter().map(|o| &o.name).collect::<Vec<_>>()
                )
            });
        assert_eq!(
            (
                r0.iter.iter().find(|(k, _)| k == "x_"),
                r0.operands[1].0.join(",")
            ),
            (None, "in,out".to_string()),
            "mq=1: the solo-decode gathered leg must declare NO `x` and a rank-2 kernel — it is the 41 \
             tok/s op. A size-1 `x` here is the phantom dim that breaks dxp's contraction inference. \
             N_={:?}",
            r0.iter,
        );
    }
}

/// ⭐⭐⭐ FACT 6 — **EVERY `y` SPLIT IN EVERY ATTENTION OP, GATHERED OR NOT, IS 4 (THE GQA GROUP) OR 1.**
///
/// This was the assertion that convicted the refused form: its ops split `y` by the rung width — 2, 4 and
/// 8 — and with `mb` pinned to 1 and a one-stick `out` that split IS `numCoresUsed_`, which IS the fault's
/// flit index. Nothing that has ever launched splits `y` by anything but the group or 1.
///
/// It now covers BOTH emissions, which is the strongest form of the pin: the collapsed fold does not
/// merely avoid the refused number, it lands on the same `y` split the shipped bundle has always carried.
///
/// ⛔ AND `be3ca5355`'s CONTROL STILL DOES NOT REACH, which is why the number is asserted rather than
/// argued. That commit inferred the rank was harmless because "the `mq`-entry one-flit table is in the
/// program of BOTH kernel ranks, and the 2-D one is what ships and runs today". Its premise is the
/// ISOLATION bundle — `/work/iso-gate/superdsc_gate/sweep_bmm2d_y8/group_0/sdsc_0.json` carries
/// `numCoresUsed_:8, numWkSlicesPerDim_:{in:1,mb:1,out:1,y:8}`, measured — and the shipped 2-D form is
/// `{y:4, mb:mq}` on 32 cores, so `{mb:1, y:8}` had never run on card in either rank.
#[test]
fn every_y_split_in_every_attention_op_is_the_gqa_group_or_one() {
    const GQA: i64 = (NQH / NKVH) as i64;
    for mq in [2u32, 4, 8] {
        for gather in [false, true] {
            for o in emit_at(mq, gather) {
                let ys = o
                    .split
                    .iter()
                    .find(|(k, _)| k == "y")
                    .map_or(1, |(_, v)| *v);
                assert!(
                    ys == 1 || ys == GQA,
                    "mq={mq} gather={gather}: {} splits `y` {ys} ways. Every attention op that has ever \
                     launched splits `y` by 1 or the GQA group ({GQA}); a split at the rung width with \
                     `mb` pinned to 1 IS `numCoresUsed_`, which IS the flit index the refused form \
                     faulted at. wk={:?} cores={}",
                    o.name,
                    o.split,
                    o.cores,
                );
            }
        }
    }
}

/// ⭐⭐⭐⭐ FACT 5 — THE GATHER EMITS **EXACTLY THE SHIPPED OP COUNT**, `nkvh` LEGS, AT EVERY RUNG — AND
/// RUNS THEM IN ONE PASS WHERE THE SHIPPED FOLD TAKES `mq`.
///
/// That is the whole trade in one number: same ops, `mq`× fewer LAUNCHES. Each op now carries the width on
/// its `x` axis instead of being cloned per request.
///
/// ⛔ AND THE `nkvh*mq` COUNT IS THE ABANDONED FORM'S, NOT A LOOSER VERSION OF THIS ONE. The `_r{R}` form
/// emitted one op per (kv head, request), reaching request `r`'s block by a baked offset, and §"the 44×
/// size curve" in this file's header measured what that cost: `4 + 2*mq` GROUPS, because each per-request
/// op became its own program and therefore its own launch. So `nkvh*mq` legs here would mean the launch
/// curve the collapse exists to flatten is back, which is why the count is asserted EQUAL across the two
/// emissions rather than merely bounded.
///
/// ⛔ AND IT IS NOT `nqh` EITHER. That was the card-refused form's count (one op per QUERY head, `y`
/// carrying the requests), priced from the card at 1.42 µs/op.
#[test]
fn the_gathered_prefix_legs_are_one_op_per_kv_head_at_every_rung() {
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
                (NKVH as usize, NKVH as usize),
                "mq={mq}: prefix-pass-0 {leg} legs — shipped {shipped}, gathered {gathered}. Both must be \
                 `nkvh` ({NKVH}): the collapse puts the width on an `x` axis INSIDE one op, so the op \
                 count cannot move. `nkvh*mq` would be the abandoned per-request `_r{{R}}` form, whose \
                 measured cost was `4 + 2*mq` programs and launches; `nqh` would be the card-refused \
                 per-query-head form."
            );
        }
    }
}
