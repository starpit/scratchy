//! #603 PR2 — one launch's compute control blocks, as DATA (stage 2a).
//!
//! The submit walk (`superdsc_exec::launch_ops_inner`) re-derives, every step,
//! a launch sequence whose only per-step inputs are a handful of scalars: the
//! fold page count, the write slot, the write page's base. This module is that
//! sequence as data — a [`CbPlan`] (one entry per (op, fold pass), carrying
//! exactly `build_compute_cb`'s inputs) plus a render that re-applies the
//! per-step scalars through the SAME `fold_plan` vocabulary the walk uses.
//! There is no second analysis to drift: the enumeration and the per-(op, rep)
//! arithmetic are single-sourced here ([`op_reps`], [`launch_site`]) and called
//! by BOTH the submit walk and the plan build, and the gate below proves the
//! data twin against the walk anyway.
//!
//! ⭐ THE PARITY GATE IS A PREPARE-TIME REFUSAL, NOT A RUNTIME CHECK. A step
//! never pays for it: `Executor::prepare` renders every allocated launch list
//! two ways — the walk ([`walk_items`]) and the plan
//! ([`build_cb_plan`] + [`CbPlan::render`]) — through ONE render door
//! (`crate::runtime::Stream::render_compute_batch`, the same
//! `build_compute_cb` the submit path runs) and REFUSES TO LOAD on the first
//! differing byte. A lazily-allocated split ladder is checked at its allocation
//! (`ensure_split_bodies`), the one event that makes it exist — never per step.
//!
//! The images this feeds (2c's player) are keyed (list × launch row ×
//! fold-page count × fold shape): the fold shape — live request count, declared
//! `fold_pages`, the installed page maps — changes what `reps` enumerates and
//! what `seg_deltas` computes, so an image is valid only under the shape it was
//! rendered for. 2a renders nothing cached; it only proves the machinery, and
//! the gate's synthetic session (in `superdsc_exec`, next to the hook)
//! deliberately exercises the batched fold (reps > 1), the slot write, the
//! non-fold page resolution and the per-request intermediate shift.
//!
//! ⭐ 2b — THE PATCH TABLE. Within one image key the ONLY per-step inputs are
//! the write position's SLOT IN ITS PAGE and its SLAB, and every address term
//! they move is affine in those two scalars ([`StepScalars`]) — the fold
//! terms are per-rep constants baked into the image, and the write PAGE never
//! moves inside an image's replayed span: every step between two page
//! boundaries writes the page its canvas was rendered on, and the boundary
//! step itself (slot 0 of the next page) is not replayed at all — it takes
//! the walk (see `try_replay_forward`; a moved write base is not affine in
//! any per-byte scalar, because a shifted write target's extent is END-PINNED
//! — `DevAddr::shifted` keeps the allocation's end and gives back the bytes
//! remaining, so the XLAT word's paddr and length fields move together and no
//! per-byte coefficient can express the pair). A replayed step is therefore
//! canvas + patches: copy the image, add each moving word's affine addend
//! ([`PatchPlan::apply`]). The coefficients are not read off the wire layout
//! by hand; they are MEASURED at prepare by finite differences through the
//! same render door ([`build_patch_plan`]) and then proven off-sample
//! ([`patch_check`]): at steps neither the canvas nor the differentials
//! visited, the patched canvas must equal the walk's fresh render byte for
//! byte, or the load refuses. A bundle whose movement is not affine in the
//! two scalars refuses there — loudly, at prepare, never as a wrong token.

use anyhow::{Result, anyhow, bail};
use flex_rs::control_block_wire::{CB_NUM_BYTES, ControlBlockWire};

use crate::bundle_code::NUM_SEGMENTS;
use crate::fold_plan::{self, OpKv, SessionKv, SlotPos, UniformPages};
use crate::runtime::{BatchedCompute, ByteSize, DevAddr, Stream, prog_offset_base};
use crate::superdsc_exec::{
    LaunchWhere, Ops, SEG_INTERMEDIATE, SEG_KV, SEG_MASK, SEG_WEIGHT, SeqPos, step_parity,
};

/// The fold pass count for one op — `fold_plan::reps` plus the walk's refusal,
/// single-sourced so the submit walk and the plan build cannot disagree about
/// how many passes an op takes (a disagreement is a different CB COUNT, which
/// the parity gate would catch — but refusing at the source is louder).
pub(crate) fn op_reps(kv: &OpKv, fold: &SessionKv, n_fold_pages: i64, oi: usize) -> Result<i64> {
    let reps = fold_plan::reps(kv, fold, n_fold_pages);
    if reps < 0 {
        // A NEGATIVE COUNT IS A REFUSAL, not a loop bound: the fold would take more passes
        // than the mask has blocks, and the bytes past the last block are unstaged zeros
        // that an additive mask reads as VALID. Abandon rather than fold in whatever the
        // pool holds — loud and incomplete beats quiet and wrong.
        // ⛔ NAME WHICH REFUSAL FIRED. `reps` returns -1 for TWO unrelated reasons and the
        // walk's message used to describe only the second, so a page-count DISAGREEMENT read
        // as a capacity shortfall and sent the reader off to enlarge a mask that was big
        // enough.
        let (declared_pages, blocks) = (fold.fold_pages, fold.mask_blocks);
        let why = if declared_pages > 0 && n_fold_pages != declared_pages as i64 {
            format!(
                "the launch walks {n_fold_pages} page(s) but the host STAGED the mask for \
                 {declared_pages} — a disagreement, not a shortfall; the mask is fine and \
                 one of the two counts is wrong"
            )
        } else {
            format!(
                "the staged mask has only {blocks} block(s) — a genuine capacity shortfall; \
                 mask bytes go as pages*width^2*nqh*512, so declare more blocks or fold \
                 fewer pages"
            )
        };
        bail!(
            "fold: {n_fold_pages} pass(es) REFUSED — {why}. ABANDONING the launch at op \
             {oi}. A pass past the last block reads bytes the host never staged; they are \
             zero, and zero is VALID in an additive mask, so the extra passes would fold in \
             whatever the pool holds."
        );
    }
    Ok(reps)
}

/// The base one segment's tensors resolve against for this launch — the weight
/// BANK's region for the weight segment (bank 0 IS `seg_addr[WEIGHT]`; bank
/// `b ≥ 1` is `banks[b - 1]`), the segment's own region otherwise. Shared by
/// the walk and the plan build so the bank selection cannot drift between
/// them. Refusals are the walk's own, verbatim — except that a bank slot which
/// exists but holds no region yet, which the walk's old code only reached as
/// the later "seg1 has no region", is named HERE as the bank it is (louder,
/// same unreachable-at-launch corner: prepare refuses a bank with no region
/// before any launch can name it).
fn segment_base<'a>(
    seg: &'a [Option<DevAddr>; NUM_SEGMENTS],
    banks: &'a [Option<DevAddr>],
    weight_bank: usize,
    i: usize,
) -> Result<&'a DevAddr> {
    if i == SEG_WEIGHT.get() {
        match weight_bank.checked_sub(1) {
            None => seg[i]
                .as_ref()
                .ok_or_else(|| anyhow!("seg{i} has no region")),
            Some(b) => banks.get(b).and_then(|a| a.as_ref()).ok_or_else(|| {
                anyhow!(
                    "launch: weight bank {weight_bank} has no region ({} bank(s))",
                    banks.len() + 1
                )
            }),
        }
    } else {
        seg[i]
            .as_ref()
            .ok_or_else(|| anyhow!("seg{i} has no region"))
    }
}

/// ONE (op, fold pass)'s launch arithmetic — the pure half of the submit walk,
/// factored out so the walk, the diag printers it feeds, and this module's
/// parity gate all read ONE computation. Every input arrives as a parameter —
/// the fold state and the segment bases included — so the prepare-time gate
/// can drive it with a synthetic session without touching live state, and so
/// the borrow of `Executor`'s fields lasts exactly one call (the walk's loop
/// interleaves `&mut self` diag hooks with this arithmetic).
pub(crate) struct LaunchSite {
    /// The per-segment byte offsets after the fold pass's deltas land on the
    /// launch row's static shifts.
    pub(crate) off: [u64; NUM_SEGMENTS],
    /// POSITIONAL per segment, owned: the resolved tensor bases with the
    /// combined shift applied — exactly what `compute`/`compute_batch` take.
    pub(crate) ta: Vec<DevAddr>,
    /// The program allocation (segment 7) this launch runs.
    pub(crate) prog: DevAddr,
    /// Offset within the program allocation where kernel execution begins.
    pub(crate) bootstrap: u64,
    /// The HOST fence flag — see [`CbPlanEntry::barrier`]; never a wire bit.
    pub(crate) barrier: bool,
}

impl LaunchSite {
    /// The staged-batch item this site becomes — ONE constructor shared by the
    /// submit walk's collect branch and [`walk_items`], so the item's shape
    /// cannot drift between the real walk and the parity gate's reference
    /// side (a field added or dropped in one would be a compile error in the
    /// other, not a silent divergence the gate cannot see).
    pub(crate) fn batched_compute(&self) -> BatchedCompute {
        BatchedCompute {
            prog: self.prog.clone(),
            tensor_allocs: self.ta.clone(),
            bootstrap_offset: self.bootstrap,
        }
    }
}

/// `Ok(None)` = THIS (op, pass) DOES NOT LAUNCH — a page part ([`OpKv::slot_offset`]) whose
/// page the request does not hold, a short last chunk's padding rows past the request's pages
/// (#620). Its rows are never read, and the only base left to aim it at is the cursor's own
/// page, which is part 0's keys. Every caller — the submit walk, `walk_items`, the render —
/// skips the entry, so the plan twin and the walk can never disagree about the enumeration.
pub(crate) fn launch_site(
    ops: &Ops,
    oi: usize,
    rep: i64,
    fold: &SessionKv,
    seg: &[Option<DevAddr>; NUM_SEGMENTS],
    banks: &[Option<DevAddr>],
    where_: &LaunchWhere,
) -> Result<Option<LaunchSite>> {
    let op = &ops[oi];
    let mut off = where_.byte_off;
    // ⭐ THE WITNESS IS MINTED HERE (was: inside the walk's rep loop).
    // `UniformPages::sweeping` checks the sweep against the pool that is
    // actually installed; a disagreement is a REFUSAL, never an address — the
    // alternative is a pass naming a page its row does not own.
    let Some(per) = UniformPages::sweeping(where_.n_fold_pages, fold) else {
        bail!(
            "fold_plan refused op {oi} (rep {rep}): the installed page maps do not \
             admit a uniform {}-page sweep",
            where_.n_fold_pages
        );
    };
    // WHERE THIS OP READS AND WRITES KV — resolved in `fold_plan`. It composes, in this
    // order: the fold pass's page and validity row, the write slot within its page, the
    // page a non-fold op works in, and the current slab.
    let aim = fold_plan::seg_deltas(
        &op.kv,
        fold,
        rep,
        per,
        SlotPos::of_launch(where_.slot_pos.0),
        where_.kv_page_base,
    );
    // ⛔ A PAGE PART WITH NO PAGE DOES NOT LAUNCH — a short last chunk's padding past the
    // request's pages. Its rows are never read, and the only base left to aim it at is the
    // cursor's own page, which is part 0's keys.
    let Some(delta) = aim.deltas() else {
        return Ok(None);
    };
    off[SEG_KV.get()] += delta.kv.0;
    // The mask has its own segment precisely so a fold can shift it without disturbing
    // the running softmax state or any other activation.
    off[SEG_MASK.get()] += delta.mask.0;
    // AND THE PASS'S OWN ROW BLOCK. A fold pass emitted for ONE request's rows has to
    // be aimed at them, and `qs`, the score block, the whole online-softmax state and
    // `out` are all Intermediates, so one shift of seg0 moves every operand it reads
    // and writes. Zero for every bundle whose fold spans the batch.
    off[SEG_INTERMEDIATE.get()] += delta.intermediate.0;

    // PER-LAYER ADDRESS ADVANCE (re-roll weight/KV threading). dxp does NOT apply
    // `ComputeParams::tensor_byte_offsets` to per-op compute addresses (MEASURED: every
    // layer read layer-0's weights). So bake the per-segment offset into a SHIFTED,
    // non-owning base per segment: all of that segment's tensors then resolve to
    // seg_base + v·stride = layer v's copy. Layer 0 (off=0) is unshifted.
    // ⭐ WHICH REGION BACKS THE WEIGHT SLOT: this launch's BANK. Positional slot 1 is
    // "the weights" to the program, and a `DevAddr` carries no segment identity, so
    // handing a different region here is the whole mechanism that lets one segment's
    // worth of addresses span several regions. Bank 0 (`weight_bank` = 0, every unbanked
    // bundle) is `seg_addr[SEG_WEIGHT]`, exactly as before.
    let wbase = segment_base(seg, banks, where_.weight_bank, SEG_WEIGHT.get())?;
    let mut ta = Vec::with_capacity(NUM_SEGMENTS);
    for (i, &sh) in off.iter().enumerate() {
        let base = if i == SEG_WEIGHT.get() {
            wbase
        } else {
            segment_base(seg, banks, where_.weight_bank, i)?
        };
        ta.push(
            if sh > 0
                && base.is_single_chunk()
                && let Some(s) = base.shifted(ByteSize(sh))
            {
                s
            } else {
                base.clone()
            },
        );
    }
    // ⭐ THE ROPE SLOT OVERRIDE (#603 PR3b E): a parity-routed group's rope descriptors
    // address their cos/sin rows through the route's LOGICAL slot, so that slot's word must
    // carry the Activation base at this step's parity — `ta[backing]`'s own shift plus one
    // more `parity·stride`. The parity comes from `step_parity(where_.slot_pos)`, the same
    // single law the staging half consults, so the slot the canvas reads is always the slot
    // this step's upload filled. ⛔ REFUSED, never fallen back: the loop above can silently
    // drop a shift (a multi-chunk base degrades to its unshifted self — safe there, the
    // shift is a fold convenience), but THIS shift is the ping-pong itself — a dropped one
    // hands the canvas the OTHER parity's bytes, which is the original torn read wearing a
    // patch-table costume.
    if let Some(route) = op.kv.parity {
        let sh = off[route.backing as usize] + step_parity(where_.slot_pos) * route.stride;
        let base = segment_base(seg, banks, where_.weight_bank, route.backing as usize)?;
        let Some(s) = base.shifted(ByteSize(sh)) else {
            bail!(
                "launch: op {oi}'s rope parity shift {sh} leaves the Activation region — the \
                 parity-1 slot does not fit, so the rope would read the other slot's bytes. The \
                 layout's footprint guard should have refused this bundle at bake; refusing the \
                 launch rather than tearing the read."
            );
        };
        ta[route.slot as usize] = s;
    }

    let prog = op
        .addr
        .clone()
        .ok_or_else(|| anyhow!("op {oi} has no program allocation"))?;
    // A BARRIER AFTER EVERY GROUP IS WHAT MAKES A 2 KB COPY COST A LAUNCH. It is a
    // device-wide settle, so consecutive groups cannot overlap however small they are —
    // and a batched decode layer is B per-request cache writes in a row, each moving
    // 2 KB into a DIFFERENT request's page. They are disjoint by construction, so
    // ordering them against EACH OTHER buys nothing; only the last has to settle before
    // the Kᵀ restickify and the next layer read what they wrote. Drop the barrier ONLY
    // between consecutive slot-writes; STRICT_ORDERING still holds the stream in order.
    // (The flag itself is a HOST fence call — `ComputeParams::pipeline_barrier` =
    // `issue_barrier` — never a wire bit: compute CBs are always built with the CTRL
    // barrier bit clear, pod-flex @ed96487 audit. The player ORs these into its one
    // doorbell.)
    // ⛔ AND ONLY WHEN THE NEXT ONE LAUNCHES (#620). A page part skipped for want of a
    // page leaves THIS write as the last before the readers, so it must settle like any
    // other last one.
    let next_is_slot_write = op.kv.slot_write
        && ops.get(oi + 1).is_some_and(|n| {
            n.kv.slot_write
                && fold_plan::seg_deltas(
                    &n.kv,
                    fold,
                    0,
                    per,
                    SlotPos::of_launch(where_.slot_pos.0),
                    where_.kv_page_base,
                )
                .deltas()
                .is_some()
        });
    let barrier = !next_is_slot_write;
    // #2814: the bootstrap argument is the OFFSET, not the full VA (flex bounds the
    // seg7 translation to the program allocation's size).
    // ⛔ THE SAME VALUE `alloc_ops_at` UPLOADED AT, not a second subtraction. This line was
    // `op.code.job_bin_ptr - prog_offset_base()` — bare, so an underflow wrapped here while
    // the allocation path refused it, and the two could only agree by both being written
    // correctly. `ProgImage` is now the one door: see `LaunchGroup::prog_image`.
    let bootstrap = op
        .code
        .prog_image(prog_offset_base())
        .ok_or_else(|| {
            anyhow!(
                "op {oi}: job_bin_ptr {:#x} is below PROG_OFFSET_BASE {:#x} — it names no \
                 bootstrap offset at all",
                op.code.job_bin_ptr,
                prog_offset_base()
            )
        })?
        .bootstrap();
    Ok(Some(LaunchSite {
        off,
        ta,
        prog,
        bootstrap,
        barrier,
    }))
}

/// The walk's staged batch for one launch call — the OLD way, built from the
/// same `op_reps`/`launch_site` the submit walk reads. This is the parity
/// gate's reference side. The submit walk keeps its own loop (its diag knobs
/// interleave with the arithmetic), which is exactly why the gate exists: if
/// the two loops ever disagree in enumeration or addressing, the rendered CBs
/// differ and prepare refuses. Returns the items and the walk's host-fence OR.
pub(crate) fn walk_items(
    ops: &Ops,
    fold: &SessionKv,
    seg: &[Option<DevAddr>; NUM_SEGMENTS],
    banks: &[Option<DevAddr>],
    where_: &LaunchWhere,
) -> Result<(Vec<BatchedCompute>, bool)> {
    let mut items = Vec::new();
    let mut barrier = false;
    for oi in 0..ops.len() {
        let reps = op_reps(&ops[oi].kv, fold, where_.n_fold_pages, oi)?;
        for rep in 0..reps {
            // A page part with no page contributes neither an item nor a barrier — the
            // walk's own `continue`, in the twin.
            let Some(site) = launch_site(ops, oi, rep, fold, seg, banks, where_)? else {
                continue;
            };
            barrier |= site.barrier;
            items.push(site.batched_compute());
        }
    }
    Ok((items, barrier))
}

/// ONE (op, fold pass) of a launch sequence — `build_compute_cb`'s inputs as
/// data. Everything here is a constant of the (list, launch row, fold shape)
/// the plan was built for; the per-step scalars enter only at
/// [`CbPlan::render`].
pub(crate) struct CbPlanEntry {
    /// Diagnostic identity — the label the walk's own traces print plus the
    /// pass — so a parity refusal names its CB.
    label: String,
    prog: DevAddr,
    bootstrap_offset: u64,
    /// The op's own KV descriptor, carried whole: the render calls the SAME
    /// `fold_plan::seg_deltas` the walk calls, never a re-derivation.
    kv: OpKv,
    rep: i64,
    /// The HOST fence flag for this entry (`!next_is_slot_write`, computed at
    /// build against the enumeration's next op). A host fence call, never a
    /// wire bit — compute CBs are built with the CTRL barrier bit clear
    /// (`CtrlSection::new(tag, false)`; pod-flex @ed96487 audit). The player
    /// submits the OR of these with the image's one doorbell.
    barrier: bool,
    /// POSITIONAL, one per segment: the base the walk would resolve (the
    /// weight BANK's region for the weight segment, the segment's own
    /// otherwise) and the launch row's static shift — stored SEPARATELY, not
    /// pre-folded, so the render's one combined `shifted(static + delta)` is
    /// the walk's own arithmetic to the byte (a pre-shifted base would clamp
    /// differently at the region's end).
    bases: Vec<(DevAddr, u64)>,
}

/// One launch's CB constructor list — the walk's sequence as data, keyed by
/// (list, launch row, fold shape) at build time.
pub(crate) struct CbPlan {
    entries: Vec<CbPlanEntry>,
}

impl CbPlan {
    /// The staged batch the plan renders to, with the walk's host-fence OR.
    ///
    /// The fold state arrives per RENDER, not per plan — it is the per-step
    /// input: the same `seg_deltas` the walk calls, applied to the entry's
    /// recorded (base, static shift) with ONE combined `shifted` per segment,
    /// exactly as the walk computes it.
    pub(crate) fn render(
        &self,
        fold: &SessionKv,
        where_: &LaunchWhere,
    ) -> Result<(Vec<BatchedCompute>, bool)> {
        let Some(per) = UniformPages::sweeping(where_.n_fold_pages, fold) else {
            bail!(
                "cb_plan render: the page maps do not admit a uniform {}-page sweep",
                where_.n_fold_pages
            );
        };
        let mut items = Vec::with_capacity(self.entries.len());
        let mut barrier = false;
        for e in &self.entries {
            let aim = fold_plan::seg_deltas(
                &e.kv,
                fold,
                e.rep,
                per,
                SlotPos::of_launch(where_.slot_pos.0),
                where_.kv_page_base,
            );
            // The walk skips an entry that does not launch at this `where_` (a page part
            // with no page); the render skips the SAME entry, so the twin's enumeration
            // tracks the walk's at every step the gate probes.
            let Some(delta) = aim.deltas() else {
                continue;
            };
            let mut dyn_off = [0u64; NUM_SEGMENTS];
            dyn_off[SEG_KV.get()] += delta.kv.0;
            dyn_off[SEG_MASK.get()] += delta.mask.0;
            dyn_off[SEG_INTERMEDIATE.get()] += delta.intermediate.0;
            let mut tensor_allocs = Vec::with_capacity(NUM_SEGMENTS);
            for (i, (base, static_off)) in e.bases.iter().enumerate() {
                let sh = *static_off + dyn_off[i];
                tensor_allocs.push(
                    if sh > 0
                        && base.is_single_chunk()
                        && let Some(s) = base.shifted(ByteSize(sh))
                    {
                        s
                    } else {
                        base.clone()
                    },
                );
            }
            // ⭐ THE ROPE SLOT OVERRIDE — the render's twin of `launch_site`'s, verbatim in
            // law: the routed slot's word carries the Activation base at the step's parity.
            // The parity gate and patch_check compare this render against the walk, so the
            // two overrides must move together or prepare refuses — which is the point.
            if let Some(route) = e.kv.parity {
                let (base, static_off) = &e.bases[route.backing as usize];
                let sh = *static_off
                    + dyn_off[route.backing as usize]
                    + step_parity(where_.slot_pos) * route.stride;
                let Some(s) = base.shifted(ByteSize(sh)) else {
                    bail!(
                        "cb_plan render: '{}'s rope parity shift {sh} leaves the Activation \
                         region — the parity-1 slot does not fit. The layout's footprint guard \
                         should have refused this bundle at bake; refusing the render rather \
                         than tearing the read.",
                        e.label
                    );
                };
                tensor_allocs[route.slot as usize] = s;
            }
            barrier |= e.barrier;
            items.push(BatchedCompute {
                prog: e.prog.clone(),
                tensor_allocs,
                bootstrap_offset: e.bootstrap_offset,
            });
        }
        Ok((items, barrier))
    }
}

/// Assemble the plan for one launch call: the (op, fold pass) enumeration and
/// every per-CB input that does not vary with the step's scalars. Built from
/// the same `op_reps` the walk calls; the bases come from the same
/// `segment_base` resolution the walk's `launch_site` performs.
pub(crate) fn build_cb_plan(
    ops: &Ops,
    fold: &SessionKv,
    seg: &[Option<DevAddr>; NUM_SEGMENTS],
    banks: &[Option<DevAddr>],
    where_: &LaunchWhere,
) -> Result<CbPlan> {
    // The weight base is per-launch, not per-entry — resolve it once, the way
    // the walk resolves `wbase` before its segment loop.
    let wbase = segment_base(seg, banks, where_.weight_bank, SEG_WEIGHT.get())?;
    // The sweep witness, once, for the barrier probe below — the same
    // `UniformPages::sweeping` the walk's `launch_site` mints per pass; a fold
    // that cannot sweep refuses here exactly as the walk would refuse there.
    let Some(per) = UniformPages::sweeping(where_.n_fold_pages, fold) else {
        bail!(
            "cb_plan build: the page maps do not admit a uniform {}-page sweep",
            where_.n_fold_pages
        );
    };
    let mut entries = Vec::new();
    for oi in 0..ops.len() {
        let op = &ops[oi];
        let reps = op_reps(&op.kv, fold, where_.n_fold_pages, oi)?;
        // ⛔ THE NEXT SLOT-WRITE MUST ALSO LAUNCH (#620) for this one to drop its
        // barrier — a page part skipped for want of a page leaves THIS write as the
        // last before the readers. The probe is the walk's own predicate, verbatim,
        // so the twin's host-fence OR is the walk's at any `where_` the gate probes.
        let next_is_slot_write = op.kv.slot_write
            && ops.get(oi + 1).is_some_and(|n| {
                n.kv.slot_write
                    && fold_plan::seg_deltas(
                        &n.kv,
                        fold,
                        0,
                        per,
                        SlotPos::of_launch(where_.slot_pos.0),
                        where_.kv_page_base,
                    )
                    .deltas()
                    .is_some()
            });
        for rep in 0..reps {
            let mut bases = Vec::with_capacity(NUM_SEGMENTS);
            for i in 0..NUM_SEGMENTS {
                let base = if i == SEG_WEIGHT.get() {
                    wbase
                } else {
                    segment_base(seg, banks, where_.weight_bank, i)?
                };
                bases.push((base.clone(), where_.byte_off[i]));
            }
            entries.push(CbPlanEntry {
                label: format!("{}/rep{rep}", op.label()),
                prog: op
                    .addr
                    .clone()
                    .ok_or_else(|| anyhow!("op {oi} has no program allocation"))?,
                bootstrap_offset: op
                    .code
                    .prog_image(prog_offset_base())
                    .ok_or_else(|| {
                        anyhow!(
                            "op {oi}: job_bin_ptr {:#x} is below PROG_OFFSET_BASE {:#x} — it \
                             names no bootstrap offset at all",
                            op.code.job_bin_ptr,
                            prog_offset_base()
                        )
                    })?
                    .bootstrap(),
                kv: op.kv,
                rep,
                barrier: !next_is_slot_write,
                bases,
            });
        }
    }
    Ok(CbPlan { entries })
}

/// ⭐ THE 2a GATE. Render one launch both ways — the walk's staged batch and
/// the CbPlan's — through one render door, and REFUSE on the first differing
/// byte. Called only from `Executor::prepare` (and the split ladder's lazy
/// allocation): a mismatch means the plan's data twin of the walk is wrong,
/// and the player that will replay these images must not load. Returns the
/// number of CBs proven byte-identical.
pub(crate) fn parity_check(
    ops: &Ops,
    fold: &SessionKv,
    seg: &[Option<DevAddr>; NUM_SEGMENTS],
    banks: &[Option<DevAddr>],
    where_: &LaunchWhere,
    label: &str,
    stream: &Stream,
) -> Result<usize> {
    let (walked, walked_barrier) = walk_items(ops, fold, seg, banks, where_)?;
    let plan = build_cb_plan(ops, fold, seg, banks, where_)?;
    let (rendered, rendered_barrier) = plan.render(fold, where_)?;
    if walked.len() != rendered.len() {
        bail!(
            "[cbplan] PARITY REFUSED at {label}: the walk stages {} launch(es) but the plan \
             renders {} — the (op, fold-pass) ENUMERATION itself disagrees. The plan is the \
             walk's data twin; refusing to load rather than replay a wrong image.",
            walked.len(),
            rendered.len()
        );
    }
    if walked_barrier != rendered_barrier {
        bail!(
            "[cbplan] PARITY REFUSED at {label}: the walk's host-fence OR is \
             {walked_barrier} but the plan's is {rendered_barrier}"
        );
    }
    // Both sides through the SAME door at the SAME tags: the tag field is a
    // patch site (2b stamps pool-leased tags), so it cancels in the compare
    // and what remains — tensor XLAT paddrs and lengths, program, bootstrap —
    // is everything the device would be handed.
    let old = stream.render_compute_batch(&walked, 1)?;
    let new = stream.render_compute_batch(&rendered, 1)?;
    for (i, (a, b)) in old.iter().zip(new.iter()).enumerate() {
        if a != b {
            let (mut at, mut wb, mut pb) = (0, 0u8, 0u8);
            for (j, (x, y)) in a.bytes().iter().zip(b.bytes().iter()).enumerate() {
                if x != y {
                    (at, wb, pb) = (j, *x, *y);
                    break;
                }
            }
            bail!(
                "[cbplan] PARITY REFUSED at {label}: CB {i} of {} ({}) differs — first \
                 differing wire byte at +{at}: walk {wb:#04x} vs plan {pb:#04x}. The plan's \
                 data twin of the walk disagrees; refusing to load rather than replay a \
                 wrong image.",
                old.len(),
                plan.entries[i].label
            );
        }
    }
    Ok(old.len())
}

// ─────────────────────────────────────────────────────────────────────────────
// 2b — the patch table: what moves between two steps of one image
// ─────────────────────────────────────────────────────────────────────────────

/// The per-step scalars — the WHOLE of what varies between two steps that
/// share an image key. Every address term the step moves is affine in these
/// three and nothing else:
///
/// * `slot_in_page` — `slot_write_delta`'s term: the write cursor's slot
///   within its page (`pos % page_slots`; the ABSOLUTE position for an
///   unpaged bundle, whose ops carry `page_slots == 0` and never wrap);
/// * `slab` — `slab_delta`'s index, `pos / SLAB_SLOTS`;
/// * `parity` — the rope ping-pong's slot (#603 PR3b E): `pos % 2`, the
///   activation slot a parity-routed group's rope rows live in. NOT affine in
///   `slot_in_page` (a slot wrap moves the position by `per−1` at either
///   parity) nor in `slab` (`SLAB_SLOTS` is even, so a slab step never flips
///   it) — it is a scalar of its own, and the patch model measures its
///   coefficient with the parity-flipping differential (the +1 step) after
///   the +2 step has isolated the slot term.
///
/// The write PAGE is deliberately not one of them. Every step an image
/// replays writes the page its canvas was rendered on; the one step that
/// would move it — `pos == n·per`, slot 0 of the next page — is a page
/// BOUNDARY, and the player routes it to the walk, one step per page,
/// because a moved write base is not affine in any per-byte scalar: a
/// shifted write target's extent is END-PINNED (`DevAddr::shifted` returns
/// the bytes remaining past the shift), so its XLAT word's paddr and length
/// fields move together and no single per-byte addend expresses the pair —
/// measured on the first real bundle, where the page differential's render
/// moved paddr `+stride/2` flits and length `-stride/2` flits for a
/// one-stride base move and the solve refused.
///
/// The fold terms — per-pass page bases from the block tables, the mask and
/// intermediate rep strides — are per-rep CONSTANTS of the image: baked into
/// the canvas, never patched. The block tables are append-only within a row's
/// life, so a page base baked into an image stays valid for that row.
#[derive(Clone, Copy)]
pub(crate) struct StepScalars {
    pub(crate) slot_in_page: u64,
    pub(crate) slab: u64,
    pub(crate) parity: u64,
}

/// The scalar rule the player runs per step — ONE implementation shared by
/// the gate and 2c's player, so the coefficients derived here and the scalars
/// applied there cannot disagree about what "slot" and "slab" mean.
///
/// The slot term divides by the SESSION's page count here and by the OP's in
/// `slot_write_delta`; they agree in every bundle, and [`build_patch_plan`]
/// checks that DIRECTLY — every slot-writing op's `page_slots` against the
/// session's — because an op that wrapped at a different modulus would put a
/// wrap inside an image's replayed span, where the patch model would replay
/// linearly across it. Refused by name, never probed for.
pub(crate) fn step_scalars(paged: bool, per: i64, pos: i64) -> StepScalars {
    StepScalars {
        slot_in_page: (if paged {
            pos.rem_euclid(per.max(1))
        } else {
            pos
        }) as u64,
        slab: (pos / fold_plan::SLAB_SLOTS) as u64,
        // The rope ping-pong's slot — the same ONE law (`step_parity`) the staging
        // half consults, so the player's patched canvas and the window's upload can
        // never disagree about which slot a step's rope rows live in.
        parity: step_parity(SeqPos(pos)),
    }
}

/// The canvas position's slot-within-page — ONE recipe shared by the
/// prepare gate's synthetic session and 2c's player, so the image the gate
/// proved and the image the player replays are cut the same way. It must
/// leave room for the +1 slot differential (≤ `per − 2` when paged, so the
/// step never crosses a page wrap) and the +1 slab differential (≤ 62, so
/// it never crosses a slab boundary), and it must not be slot 0 of a page
/// (a page boundary is the one step of the span the player does NOT replay —
/// it walks; see [`StepScalars`]). 5 — the gate's historical pick — whenever
/// it fits; `None` when no slot of the page can host a canvas (a page of
/// ≤ 2 slots), which the caller treats
/// as "no image for this shape", never as a wrong-coefficient derivation.
pub(crate) fn canvas_slot(per: i64, paged: bool) -> Option<i64> {
    let hi = if paged { per - 2 } else { 62 }.min(62);
    if hi >= 5 {
        Some(5)
    } else if hi >= 1 {
        Some(1)
    } else {
        None
    }
}

/// Whether a canvas at slot `r` leaves the OFF-SAMPLE ROOM the patch gate
/// needs: the `slot+65` sample — the one that crosses a slab boundary, and
/// so the only off-sample proof of the slab coefficient — must land strictly
/// inside the image's page span (the open interval `(n−1)·per < p < n·per`
/// the player replays), so `r + 66 ≤ per`. An unpaged session has no page
/// bound (its one image covers every position). ONE law, shared by the
/// prepare gate (without it, its patch leg skips loudly) and the player
/// (without it, every step walks): a page too narrow to host the proof
/// degrades to the walk, never to an unverified coefficient. Every shipped
/// bundle's page is 2048 slots, so this never fires in practice.
pub(crate) fn canvas_room(per: i64, paged: bool, r: i64) -> bool {
    !paged || r + 66 <= per
}

/// One 8-byte word of one CB image that moves with the step: where it sits,
/// and the addend per unit of each scalar — MEASURED by finite differences
/// through the render door, never read off the wire layout by hand. The XLAT
/// word packs paddr, mode and length in disjoint bit ranges, so a step's
/// movement is an additive delta within the word; these are that delta per
/// scalar unit, and [`PatchPlan::apply`] adds them with two's-complement
/// wrap (an image's early steps sit at slots BELOW the canvas's, so slot
/// deltas go NEGATIVE).
pub(crate) struct Patch {
    /// Which CB of the image — an index into the rendered batch.
    cb: usize,
    /// Byte offset of the 8-byte word within the CB. Every section is
    /// 8-aligned, so a moving field never straddles words.
    dst: usize,
    a_slot: i64,
    b_slab: i64,
    /// The rope ping-pong's term: the addend per FLIP of the step parity
    /// (`d_parity` is −1, 0 or +1 — the word moves by `c_parity` per flip, so
    /// the sign of `d_parity` carries the direction). Zero for every word the
    /// parity override does not touch.
    c_parity: i64,
}

/// The patches that turn one image into any step's CBs: the scalars the
/// canvas was rendered at, every moving word's affine addend, and the
/// launch's host-fence OR — the one `pipeline_barrier` the player submits
/// the whole image with (the walk's per-op ORs collapse to it; see
/// `Stream::compute_batch`'s flush). Built by [`build_patch_plan`] — at
/// prepare for the gate, at an image's first use for 2c's player — and
/// applied by copying the canvas and calling [`PatchPlan::apply`].
pub(crate) struct PatchPlan {
    reference: StepScalars,
    patches: Vec<Patch>,
    /// The OR of every entry's host-fence flag — a constant of the plan
    /// (entry-keyed, position-independent), so one value serves every step
    /// this image replays.
    pub(crate) barrier: bool,
}

impl PatchPlan {
    /// How many moving words the plan patches — the player's per-step patch
    /// workload, reported by its timers.
    pub(crate) fn sites(&self) -> usize {
        self.patches.len()
    }

    /// Patch a copied canvas in place for one step's scalars. The addend is
    /// computed in i128 and wrapped to u64: an image's early steps sit below
    /// the canvas slot, so slot deltas go negative, and the wire's arithmetic
    /// is two's-complement addition.
    pub(crate) fn apply(&self, image: &mut [ControlBlockWire], s: &StepScalars) {
        let d_slot = s.slot_in_page as i128 - self.reference.slot_in_page as i128;
        let d_slab = s.slab as i128 - self.reference.slab as i128;
        let d_parity = s.parity as i128 - self.reference.parity as i128;
        for p in &self.patches {
            let addend = (p.a_slot as i128 * d_slot
                + p.b_slab as i128 * d_slab
                + p.c_parity as i128 * d_parity) as u64;
            let b = image[p.cb].bytes_mut();
            let mut w = [0u8; 8];
            w.copy_from_slice(&b[p.dst..p.dst + 8]);
            let word = u64::from_le_bytes(w).wrapping_add(addend);
            b[p.dst..p.dst + 8].copy_from_slice(&word.to_le_bytes());
        }
    }
}

/// One little-endian word of a rendered image — the unit the patch model
/// works in (fields never straddle words; every section is 8-aligned).
fn word_at(image: &[ControlBlockWire], cb: usize, dst: usize) -> u64 {
    let mut w = [0u8; 8];
    w.copy_from_slice(&image[cb].bytes()[dst..dst + 8]);
    u64::from_le_bytes(w)
}

/// One moving word's three measured differences to its three affine
/// coefficients — the per-word heart of [`build_patch_plan`]'s solve, factored
/// out so the algebra is pin-testable without a render door. The
/// differentials' DEFINITIONS live at the call site (which render each came
/// from); only the solve lives here:
///
/// * `d1` — the +1 step: Δslot = +1, Δslab = 0, parity FLIPPED by
///   `dparity_slot` (±1);
/// * `d2` — the slab step: Δslab = +1, Δslot = `dslot_slab`, parity held;
/// * `d3` — the +2 step: Δslot = +2, Δslab = 0, parity held.
///
/// Returns `(a_slot, b_slab, c_parity)`. REFUSES a `d3` that is not even: a
/// word that moved an odd amount under a two-slot step with the slab and the
/// parity held is not affine in the model's scalars, and rounding a
/// coefficient in would replay a wrong address every step.
fn solve_word(
    d1: i64,
    d2: i64,
    d3: i64,
    dslot_slab: i64,
    dparity_slot: i64,
) -> Result<(i64, i64, i64)> {
    // Δslot = +2 with the slab and the parity held ⇒ the slot coefficient is half
    // the raw difference.
    if d3 % 2 != 0 {
        bail!(
            "a word moves {d3} over a +2 slot step with slab and parity held — not affine in \
             the step scalars (an even multiple of the slot term was the only lawful \
             movement). Refusing to build a patch table with a rounded coefficient."
        );
    }
    let a_slot = d3 / 2;
    // d1 — Δslot = +1 with the parity FLIPPED ⇒ what remains of the difference
    // after the slot's contribution is the parity coefficient, signed by the
    // flip's direction (`dparity_slot` is ±1, so multiplying by it is dividing
    // by it).
    let c_parity = (d1.wrapping_sub(a_slot)).wrapping_mul(dparity_slot);
    // d2 — Δslab = +1 with the slot moving `dslot_slab` and the parity held ⇒ the
    // slab coefficient is what remains after the slot's contribution is
    // subtracted.
    let b_slab = d2.wrapping_sub(a_slot.wrapping_mul(dslot_slab));
    Ok((a_slot, b_slab, c_parity))
}

/// Derive one image's patch table: render the canvas (the plan's own render —
/// the caller proves it byte-identical to the walk at this `where_`, which
/// [`patch_check`] does), render two differentials, and read every moving
/// word's coefficients off the differences. Finite differences through the
/// door: there is no second analysis of the wire layout to drift, and
/// anything that moves but is not affine in the two scalars refuses here or
/// mismatches in [`patch_check`]'s off-sample verification.
///
/// The canvas position must leave room for the differentials: the +1 step may
/// not cross a slab or page boundary, because it isolates the slot term. That
/// is the caller's choice — the gate picks it, and a violated precondition
/// is a REFUSAL, never a silently wrong coefficient.
///
/// Returns the canvas image and its patch plan: what 2c's player copies and
/// patches per step.
pub(crate) fn build_patch_plan(
    ops: &Ops,
    fold: &SessionKv,
    seg: &[Option<DevAddr>; NUM_SEGMENTS],
    banks: &[Option<DevAddr>],
    where_: &LaunchWhere,
    stream: &Stream,
) -> Result<(Vec<ControlBlockWire>, PatchPlan)> {
    let per = fold.page_slots.get().max(1);
    let pos = where_.slot_pos.0;
    if pos % fold_plan::SLAB_SLOTS >= fold_plan::SLAB_SLOTS - 2 {
        bail!(
            "cb_plan patch: canvas position {pos} sits within two of a slab boundary — the +2 \
             differential (the one that isolates the slot term from the parity term) cannot \
             hold the slab"
        );
    }
    if fold.paged && pos.rem_euclid(per) > per - 3 {
        bail!(
            "cb_plan patch: canvas position {pos} sits within two of a page boundary — the +2 \
             differential (the one that isolates the slot term from the parity term) cannot \
             hold the page"
        );
    }
    // The op-vs-session page-modulus law, checked DIRECTLY (this is the canary
    // the boundary sample used to carry): the slot scalar wraps at the
    // SESSION's modulus, `slot_write_delta` at the OP'S, and an op that
    // disagreed would wrap INSIDE an image's replayed span — steps the patch
    // model would replay linearly across the wrap. The bake emits the
    // session's page size into every op, so this agrees in every bundle; a
    // bundle that broke it refuses here, by name.
    let session_modulus = if fold.paged { per } else { 0 };
    for o in ops {
        if o.kv.slot_write && o.kv.page_slots as i64 != session_modulus {
            bail!(
                "cb_plan patch: a slot-writing op wraps at {} slots but the session wraps at {} — \
                 the two moduli disagree, so a page wrap could fall inside an image's replayed \
                 span and the patch model would replay straight across it. Refusing to patch a \
                 bundle whose ops disagree with the session about the page.",
                o.kv.page_slots,
                session_modulus
            );
        }
    }

    let plan = build_cb_plan(ops, fold, seg, banks, where_)?;
    let render = |w: &LaunchWhere| -> Result<Vec<ControlBlockWire>> {
        let (items, _) = plan.render(fold, w)?;
        Ok(stream.render_compute_batch(&items, 1)?)
    };
    // The canvas, rendered once — and the host-fence OR with it: a constant
    // of the plan (entry-keyed, the same for every render of this plan), so
    // the canvas render's value is the image's, what the player submits
    // every step of this image with.
    let (items, barrier) = plan.render(fold, where_)?;
    let canvas = stream.render_compute_batch(&items, 1)?;
    // The three differentials. Each holds every scalar but one's worth of
    // arithmetic, so each moving word's measured difference is one equation
    // in that word's coefficients. There is NO page differential: the write
    // page never moves inside an image's replayed span (the boundary step
    // walks; see [`StepScalars`]), so no coefficient for it exists to
    // measure.
    //
    // * +1 — the parity differential: slot +1, slab held, parity FLIPPED.
    // * +SLAB_SLOTS — the slab differential: slab +1, parity held
    //   (`SLAB_SLOTS` is even), slot moving `dslot_slab`.
    // * +2 — the slot differential: slot +2 with slab AND parity held (2 is
    //   even), the equation that isolates the slot term — without it, the +1
    //   difference entangles slot and parity irrecoverably (one equation,
    //   two unknowns) and the solve would guess.
    let img_slot = render(&LaunchWhere {
        slot_pos: SeqPos(pos + 1),
        ..*where_
    })?;
    let img_slab = render(&LaunchWhere {
        slot_pos: SeqPos(pos + fold_plan::SLAB_SLOTS),
        ..*where_
    })?;
    let img_parity = render(&LaunchWhere {
        slot_pos: SeqPos(pos + 2),
        ..*where_
    })?;
    for (name, img) in [
        ("slot", &img_slot),
        ("slab", &img_slab),
        ("parity", &img_parity),
    ] {
        if img.len() != canvas.len() {
            bail!(
                "cb_plan patch: the {name} differential renders {} CB(s) but the canvas holds {} \
                 — the (op, fold-pass) enumeration moved with a step scalar, so no image can key \
                 it. Refusing to build a patch table for a moving sequence.",
                img.len(),
                canvas.len()
            );
        }
    }
    // How much the slot term moves in the slab differential: +SLAB_SLOTS
    // unwrapped, or that minus one page when it crosses a page boundary — the
    // wrap is arithmetic the model already covers (the slot scalar itself
    // wraps), so the coefficient solve just needs the true delta.
    let dslot_slab = if fold.paged {
        (pos + fold_plan::SLAB_SLOTS).rem_euclid(per) - pos.rem_euclid(per)
    } else {
        fold_plan::SLAB_SLOTS
    };
    // Which way the +1 differential flips the parity — the sign the solve
    // divides by (±1, so multiplying by it again is dividing by it).
    let dparity_slot = (step_parity(SeqPos(pos + 1)) as i64) - (step_parity(SeqPos(pos)) as i64);
    let mut patches = Vec::new();
    for cb in 0..canvas.len() {
        for dst in (0..CB_NUM_BYTES).step_by(8) {
            let c = word_at(&canvas, cb, dst);
            // d1 — the +1 step: slot +1, parity flipped. d2 — the slab step. d3 — the +2
            // step: slot +2, slab and parity held.
            let d1 = word_at(&img_slot, cb, dst).wrapping_sub(c) as i64;
            let d2 = word_at(&img_slab, cb, dst).wrapping_sub(c) as i64;
            let d3 = word_at(&img_parity, cb, dst).wrapping_sub(c) as i64;
            if d1 == 0 && d2 == 0 && d3 == 0 {
                continue;
            }
            let (a_slot, b_slab, c_parity) = solve_word(d1, d2, d3, dslot_slab, dparity_slot)?;
            patches.push(Patch {
                cb,
                dst,
                a_slot,
                b_slab,
                c_parity,
            });
        }
    }
    let reference = step_scalars(fold.paged, per, pos);
    Ok((
        canvas,
        PatchPlan {
            reference,
            patches,
            barrier,
        },
    ))
}

/// One launch row's replay image — what 2c's player caches per row and
/// concatenates into a step image: the canvas it copies, and the patch plan
/// that turns the copy into any step of the same image key. The plan's
/// `barrier` is the row's host-fence OR; the player ORs the rows it merges.
pub(crate) struct RowImage {
    pub(crate) canvas: Vec<ControlBlockWire>,
    pub(crate) plan: PatchPlan,
}

/// One launch row's canvas with NO patch table — what a step-CONSTANT row
/// (prefix/suffix, the player's `follows_step == false` rows) replays from,
/// in full. A constant `where_` makes every step-differential zero, so such
/// a row's patch table is empty BY CONSTRUCTION — deriving one anyway would
/// probe a counterfactual no replay executes (measured on the first real
/// bundle: the gate's page probe walked the row's write base a full stride
/// past where any launch of that row ever puts it, the render returned a
/// structurally different allocation, and the solve refused). Parity
/// at the constant `where_` is the row's whole correctness statement; the
/// player copies this canvas verbatim and never consults the empty plan.
pub(crate) fn render_row(
    ops: &Ops,
    fold: &SessionKv,
    seg: &[Option<DevAddr>; NUM_SEGMENTS],
    banks: &[Option<DevAddr>],
    where_: &LaunchWhere,
    stream: &Stream,
) -> Result<RowImage> {
    let plan = build_cb_plan(ops, fold, seg, banks, where_)?;
    let (items, barrier) = plan.render(fold, where_)?;
    let canvas = stream.render_compute_batch(&items, 1)?;
    let per = fold.page_slots.get().max(1);
    Ok(RowImage {
        canvas,
        plan: PatchPlan {
            reference: step_scalars(fold.paged, per, where_.slot_pos.0),
            patches: Vec::new(),
            barrier,
        },
    })
}

/// ⭐ THE 2b GATE — the patch table proven off-sample, at prepare. Builds the
/// patch plan for one launch's image and then VERIFIES it against fresh walk
/// renders at steps neither the canvas nor the differentials visited: the
/// patched canvas must equal the OLD way's bytes, byte for byte, or the load
/// refuses. This is the whole correctness statement of 2c's player — replay
/// (canvas + patches) ≡ re-derive (the walk) — proven at prepare under a
/// synthetic session of the player's own shape, so no step ever pays for it.
/// The player runs this SAME check per image at the image's first use, under
/// the live session — and keeps the verified image it returns.
///
/// The sampled steps are the image's own OPEN span `(n−1)·per < p < n·per` —
/// exactly the steps the player replays this image over. The page BOUNDARY
/// (`p == n·per`, slot 0 of the next page) is deliberately not sampled: the
/// player routes it to the walk (see [`StepScalars`]), so there is no patched
/// render of it to prove — and an op that disagreed with the session about
/// the page modulus is refused outright by [`build_patch_plan`], by name,
/// rather than caught at a boundary probe.
/// Returns the number of patch sites verified and the
/// verified image itself.
pub(crate) fn patch_check(
    ops: &Ops,
    fold: &SessionKv,
    seg: &[Option<DevAddr>; NUM_SEGMENTS],
    banks: &[Option<DevAddr>],
    where_: &LaunchWhere,
    label: &str,
    stream: &Stream,
) -> Result<(usize, RowImage)> {
    // The canvas the patches are measured against is itself proven against
    // the walk at this exact `where_` — patching a canvas that already
    // disagreed with the walk would verify nothing.
    parity_check(ops, fold, seg, banks, where_, label, stream)?;
    let (canvas, patch_plan) = build_patch_plan(ops, fold, seg, banks, where_, stream)
        .map_err(|e| anyhow!("[cbpatch] {label}: patch-table derivation REFUSED — {e:#}"))?;
    let per = fold.page_slots.get().max(1);
    let pos = where_.slot_pos.0;
    let n = where_.n_fold_pages;
    // A paged step belongs to the image keyed by ceil(pos/per) — sample only
    // within the canvas's own key AND off its page boundary, which is the
    // OPEN span the player replays this image over (the boundary step walks).
    // An unpaged bundle has one image for every position.
    let in_image = |p: i64| !fold.paged || (p > (n - 1) * per && p < n * per);
    let mut samples: Vec<(i64, &str)> = vec![
        (pos + 5, "slot+5"),
        (pos + 33, "slot+33"),
        (pos + 65, "slot+65/slab+1"),
        (pos + 130, "slot+130/slab+2"),
    ];
    if pos >= 3 {
        samples.push((pos - 3, "slot-3 (negative delta)"));
    }
    let samples: Vec<_> = samples.into_iter().filter(|(p, _)| in_image(*p)).collect();
    if samples.is_empty() {
        bail!(
            "[cbpatch] {label} REFUSED: no off-sample step fits this image's position range — \
             the gate would pass vacuously"
        );
    }
    // Every sample renders the walk at the canvas's OWN page base — the base
    // every replayed step of this image writes at.
    let walk_render = |spos: i64| -> Result<Vec<ControlBlockWire>> {
        let (items, _) = walk_items(
            ops,
            fold,
            seg,
            banks,
            &LaunchWhere {
                slot_pos: SeqPos(spos),
                ..*where_
            },
        )?;
        Ok(stream.render_compute_batch(&items, 1)?)
    };
    for (spos, what) in samples {
        let expected = walk_render(spos)?;
        if expected.len() != canvas.len() {
            bail!(
                "[cbpatch] {label} REFUSED at sample `{what}`: the walk renders {} CB(s) but the \
                 image holds {} — the (op, fold-pass) enumeration moved with the step, so no \
                 image can key it",
                expected.len(),
                canvas.len()
            );
        }
        let mut got = canvas.clone();
        patch_plan.apply(&mut got, &step_scalars(fold.paged, per, spos));
        for (i, (a, b)) in got.iter().zip(expected.iter()).enumerate() {
            if a != b {
                let (mut at, mut wb, mut pb) = (0, 0u8, 0u8);
                for (j, (x, y)) in a.bytes().iter().zip(b.bytes().iter()).enumerate() {
                    if x != y {
                        (at, wb, pb) = (j, *x, *y);
                        break;
                    }
                }
                bail!(
                    "[cbpatch] {label} REFUSED at sample `{what}`: CB {i} of {} differs after \
                     patching — first differing wire byte at +{at}: patched {wb:#04x} vs the \
                     walk's {pb:#04x}. The image-plus-patches model of this launch is not \
                     byte-identical to re-deriving the step; refusing to load rather than \
                     replay a wrong image.",
                    canvas.len()
                );
            }
        }
    }
    Ok((
        patch_plan.patches.len(),
        RowImage {
            canvas,
            plan: patch_plan,
        },
    ))
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
//  #603 PR3b E — the rope parity's patch-model tests
// ══════════════════════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    /// The solve recovers all three coefficients from the three differentials,
    /// at BOTH flip directions of the +1 step (the canvas can sit at either
    /// parity) and with the slab differential's slot term wrapped or not — the
    /// 3×3 system is solved, not assumed.
    #[test]
    fn the_solve_recovers_all_three_coefficients() {
        for dparity_slot in [-1i64, 1] {
            for dslot_slab in [64i64, 64 - 2048] {
                for (a, b, c) in [
                    (0x4000i64, 0x20_0000i64, 0x1000i64), // XLAT-paddr-like magnitudes
                    (0i64, 0i64, 0x1000),                 // a parity-ONLY word
                    (128, 0, 0),                          // a slot-ONLY word
                    (0, 0x80_0000, 0),                    // a slab-ONLY word
                ] {
                    // The differentials, from their definitions: +1 (slot, parity
                    // flipped), slab (slot +dslot_slab, slab +1), +2 (slot only).
                    let d1 = a.wrapping_add(c.wrapping_mul(dparity_slot));
                    let d2 = a.wrapping_mul(dslot_slab).wrapping_add(b);
                    let d3 = a.wrapping_mul(2);
                    let got = solve_word(d1, d2, d3, dslot_slab, dparity_slot)
                        .unwrap_or_else(|e| panic!("a lawful word refused: {e}"));
                    assert_eq!(
                        got,
                        (a, b, c),
                        "solve did not recover the coefficients (flip {dparity_slot}, \
                         slab-slot {dslot_slab})"
                    );
                }
            }
        }
    }

    /// A word that moves an ODD amount under the +2 step (slab and parity held)
    /// is not affine in the model's scalars — the solve REFUSES it rather than
    /// rounding a coefficient in. Without this arm the solve would silently
    /// truncate `d3 / 2` and replay a wrong address every step.
    #[test]
    fn the_solve_refuses_a_non_affine_word() {
        assert!(
            solve_word(0, 0, 3, 64, 1).is_err(),
            "an odd +2 difference must refuse, never round"
        );
        assert!(
            solve_word(0, 0, -1, 64, -1).is_err(),
            "an odd negative +2 difference must refuse too"
        );
    }

    /// ⭐ THE DELIBERATELY-WRONG PARITY (#603 PR3b E, ruling #10's criterion):
    /// patch_check's off-sample compare is `apply(canvas, scalars(s)) ==
    /// walk_render(s)` — here driven with a hand-stated walk truth for one
    /// moving word, exactly the compare the gate runs per sample. The correct
    /// parity coefficient reproduces a FLIPPED-parity off-sample step (pos+5);
    /// a corrupted one — the pre-fix model, `c_parity = 0`, "the parity does
    /// not move the word" — does not, which is the off-sample extension
    /// CATCHING a wrong parity: the gate refuses rather than loading the image.
    /// A test that cannot fail proves nothing, so both arms are asserted: the
    /// correct plan MUST match and the corrupted one MUST NOT.
    #[test]
    fn a_wrong_parity_coefficient_fails_the_off_sample_step() {
        use flex_rs::control_block_wire::{
            AllocationTotalSize, ControlBlockWire as Cbw, DevicePhysicalAddress, Flits,
            ResponseTag, VirtualAddressSbf,
        };
        // One real compute CB, host-side — the render door's own constructor, so
        // the word under test sits inside genuinely-encoded wire bytes.
        let cb = Cbw::compute(
            ResponseTag(1),
            &[
                (
                    DevicePhysicalAddress(0x1_0000_0000),
                    AllocationTotalSize(0x0100_0000),
                ),
                (
                    DevicePhysicalAddress(0x2_0000_0000),
                    AllocationTotalSize(0x0100_0000),
                ),
                (
                    DevicePhysicalAddress(0x3_0000_0000),
                    AllocationTotalSize(0x0100_0000),
                ),
                (
                    DevicePhysicalAddress(0x4_0000_0000),
                    AllocationTotalSize(0x0100_0000),
                ),
            ],
            (
                DevicePhysicalAddress(0x7_0000_0000),
                AllocationTotalSize(0x1000),
            ),
            VirtualAddressSbf::new(7, Flits(0x40)),
        )
        .expect("a host-side compute CB renders without a device");
        let mut canvas = vec![cb; 3];
        // The word the plan patches: the routed slot's base word, at a fixed
        // 8-aligned offset, seeded with a known value.
        let (cb_i, dst) = (1usize, 64usize);
        let word0 = 0x3_0000_0000u64;
        canvas[cb_i].bytes_mut()[dst..dst + 8].copy_from_slice(&word0.to_le_bytes());
        // The coefficients the solve would measure for this word: the rope
        // stride per parity flip, a slot term, no slab term.
        let (a_slot, b_slab, stride) = (0x2000i64, 0i64, 0x1000i64);
        let plan = |c_parity: i64| PatchPlan {
            reference: step_scalars(false, 2048, 5),
            patches: vec![Patch {
                cb: cb_i,
                dst,
                a_slot,
                b_slab,
                c_parity,
            }],
            barrier: false,
        };
        // The off-sample step: position 10 — slot +5, slab 0, parity FLIPPED
        // (the canvas sits at position 5, parity 1; 10 is parity 0). The walk
        // truth for an affine word is the canvas word plus the coefficients'
        // own movement, measured from the REFERENCE scalars the plan carries.
        let s = step_scalars(false, 2048, 10);
        let reference = step_scalars(false, 2048, 5);
        let truth = (word0 as i128
            + a_slot as i128 * (s.slot_in_page as i128 - reference.slot_in_page as i128)
            + stride as i128 * (s.parity as i128 - reference.parity as i128))
            as u64;
        // The correct plan reproduces the walk at the flipped-parity step —
        // the off-sample compare passes.
        let mut got = canvas.clone();
        plan(stride).apply(&mut got, &s);
        assert_eq!(
            word_at(&got, cb_i, dst),
            truth,
            "the correct parity coefficient must reproduce the off-sample step"
        );
        // The corrupted plan — the pre-fix model, no parity term — does not:
        // the off-sample compare REFUSES, which is the catch this test exists
        // to show.
        let mut corrupted = canvas.clone();
        plan(0).apply(&mut corrupted, &s);
        assert_ne!(
            word_at(&corrupted, cb_i, dst),
            truth,
            "a zeroed parity coefficient must NOT reproduce the flipped-parity step — \
             if it does, the off-sample extension catches nothing"
        );
        // And the corrupted word is exactly the parity stride away from the
        // truth — the torn-read distance, not noise.
        assert_eq!(
            word_at(&corrupted, cb_i, dst).wrapping_sub(truth),
            stride as u64,
            "the corrupted plan's error is exactly one parity stride"
        );
    }
}
