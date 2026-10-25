# 11l — the trip-wall body split (AUDITOR RULING — exact code, final)

The wall: the two top rungs refuse ON TRIPS ONLY — 72,397 (384 rows, T=96) and 96,429
(512 rows, T=128) over `MAX_BUNDLE_TRIPS = 65_536` (superdsc_bake.rs:2013); descriptors are
fine post-11k (7,693 / 10,157 vs 16,384). The ladder must top out at 512 (wall-16) — this
ruling is REQUIRED, and it is a lowering decision (fusion/launch-count is ours), not a budget
move. The budget does not bend.

## WHERE THE SPLIT RUNS — AND WHY THERE (the three reads, ruled)

**The split runs at the END of `lower_subtile_tape_to_ktir`, on the ASSEMBLED roll, immediately
before the return — after the aliasing pre-pass, after the launch build, after the launch-table
generator and proof.** Each upstream stage argues for this by its own law:

1. **The residual-aliasing pre-pass (:2285-2337) would COLLIDE the seam if it saw the split.**
   It iterates the tape's loops, takes each body's first-in/last-out as the residual pair, and
   aliases both onto ONE placement. A split visible to it would alias the attention half's
   `hout = t_seam` AND the MLP half's `hin = t_seam` onto the residual buffer — re-placing the
   seam tensor. Running the split AFTER the pre-pass leaves every placement exactly as walked:
   the seam tensor keeps its OWN placement (READ 2: written once, read twice), and the A|B
   handoff is in-place by the existing placement. **NO pre-pass change.**
2. **The launch build (:2675-2699) refuses two bodies naming one layer (:2685-2693).** The
   split's exact shape is both halves carrying the original `layer_ids` — the duplicate-id
   refusal would fire. Running the split AFTER the launch build leaves that refusal intact for
   sibling classes; the split interleaves the ALREADY-BUILT launches positionally.
   **NO launch-build change.**
3. **The generator and proof (:3112-3212) ran correctly on the pre-split roll.** `launch_w`
   rows are `(bank as u8, shift)` — the BANK, not a body index — pushed positionally, one per
   launch (:3128, :3155); `launch_kv` likewise (:3156). The B half's rows are byte-copies of
   A's at the same iteration, and the proof relation for B's tables at that shift is EXACTLY
   what the walk proved (B's tables ⊆ the original class's tables — the generator read the
   shift off ANY of the class's tables and the proof held ALL of them, :3163-3212). The split
   adds a re-check that holds the copies to the same law again. **NO generator/proof change.**
4. **`unroll_layers` (superdsc.rs:7243-7315) is OPS-DRIVEN over the SHARED `per_layer` map**
   (:7287: `rolled.per_layer.get(&b.get())` per binding; the hidden-in special case at :7277).
   No body-scoped table sets exist at runtime — each half's ops look up their own tids in the
   one map. **The split needs NO `per_layer` PARTITION AT ALL** (the draft's partition section
   is DEAD — and would be wrong to add: removing keys from the shared map is the one change
   that could break the other half's lookups). Table lengths are unchanged (both halves carry
   the original iters).
5. **The seam tensor chains either way it is keyed.** If `t_seam ∉ per_layer`: A's write and
   B's read both keep the layer-0 tid — same tid, same placement, threads in place. If
   `t_seam ∈ per_layer`: A's write rebinds to `per_layer[t_seam][j]` (generic rebind) and B's
   read rebinds to `prev_hidden = per_layer[A.hidden_out_tid][j]` (the hidden-in special case,
   :7256-7263) — the same rebound tid. Both halves match by construction.
6. **Everything downstream is class-general already**: the bake loop
   (codegen.rs:11365-11384, `for (ci, b) in rolled.bodies.iter().enumerate()`), the fold-fused
   twin (:11420, per class), the census print (:12230, "body {bfp} + {} class(es), {}
   launch(es)" — takes `rolled.bodies.len()` / `rolled.launches.len()`), the write path's
   `(pre, bod, suf)` error arm (:12312-12318 — three write RESULTS, not a count), the loader
   (:1232/:1312, "N body class(es)"), and the executor's per-launch base bind
   (superdsc_exec.rs:4951-4958, READ 1: one shifted base per segment per launch — B's ops
   address the same single bank the original body's bank check (:3250-3265) proved, and B never
   touches the KV segment so its copied kv row is inert).

## THE SEAM LAW (final — one correction from the draft)

```
seam   = the LAST op index whose ktir bindings contain the body's hidden_in_tid
         (the pre-norm block's skip-add — the join reads the layer input)
t_seam = the UNIQUE binding of ops[seam] never referenced by ops[..seam]   (SSA-fresh output)
cross  = (∪ bindings of ops[..=seam]) ∩ (∪ bindings of ops[seam+1..])
SPLIT  iff cross == {t_seam}  — exactly one tensor crosses, and it is the join's own output
```

⛔ **THE DRAFT'S CROSS CHECK WAS WRONG** (`refs(0..=seam + 1)` folded ops[seam+1]'s own
bindings into both sides — every split would have refused). The A-side set is
`bound_tids(&ops[..=seam])`, EXACTLY. READ 2 measured granite-2b against this law: the seam
is 94_add_o737 (reads o736 × scale and o728 = the hidden input, produces o738); cross is
exactly {o738}; stagings partition (attention t729/t730/t731/t735 ‖ MLP t739/t740/t742);
per-layer tables partition (q/k/v/o + scales + input_layernorm + KV ‖ gate/up/down + scales +
post_attention_layernorm). No legal seam ⇒ the refusal stands (the measurement). Conservative
by design: a shared constant, a staging, or a weight read on both sides refuses the split
rather than risking an unproven cut.

## L1 — THE EXACT CODE (`lower_subtile_tape_to_ktir.rs`, immediately after the lowering fn's
closing brace at :3325, before `#[cfg(test)] mod scalar_mul_weight_fold_tests`)

```rust
/// The tensor ids a slice of the body's programs name — the seam scan's binding universe.
/// Every op of a bakable body carries its program (bundle creation refuses one that does
/// not), and the split's scan has already refused any body with a program-less op, so this
/// sees every binding there is.
fn bound_tids(ops: &[EmittedOp]) -> std::collections::BTreeSet<u32> {
    ops.iter()
        .filter_map(|o| o.ktir.as_ref())
        .flat_map(|k| k.bindings.iter())
        .map(|x| x.get())
        .collect()
}

/// ⭐ THE TRIP-WALL SPLIT — a body class whose TRIPS (`Σ o.time.max(1)`) would refuse the bake
/// budget ([`crate::superdsc_bake::BakeBudget::MAX_BUNDLE_TRIPS`]) becomes TWO body classes at
/// the residual seam, each baking as its own bundle under the wall. The bake, the loader, the
/// fold-fused twin and the executor's launch loop are all class-general already; the ladder's
/// two top rungs (384 and 512 rows: 72,397 and 96,429 trips over 65,536) are the measured
/// shape this exists for — the attention half and the MLP half of one layer.
///
/// ⭐ THE SEAM IS THE RESIDUAL JOIN, AND THE CUT IS LEGAL ONLY WHEN EXACTLY ONE TENSOR CROSSES
/// IT: the LAST op that reads the body's `hidden_in_tid` (the pre-norm block's skip-add) is
/// the seam; `t_seam` is its SSA-fresh output; and the tensors bound by both halves must be
/// exactly `{t_seam}`. That is the same in-place handoff the per-launch residual law threads —
/// `A` writes the seam, `B` reads it, at the seam's own placement (the aliasing pre-pass ran
/// before the split and never re-placed it), and `unroll_layers` rebinds both sides of it
/// through the SHARED `per_layer` map the same way it already threads the body→body seam.
/// Anything else — a staging, a shared constant, a weight read on both sides — refuses the
/// split, and the bake's trip refusal stands as the measurement.
///
/// ⭐ NOTHING UPSTREAM MOVES. The split runs on the assembled roll, after the aliasing
/// pre-pass, after the launch build (whose duplicate-layer-id refusal would refuse two
/// classes naming one layer), and after the launch-table generator and proof (which held
/// EVERY per-layer tensor of the original class to each launch's shift). The halves' launch
/// rows are COPIES of the rows the proof already blessed — `launch_w` is `(bank, shift)` per
/// launch, not per body — and the B half's per-layer tables are a subset of the set the proof
/// covered, so the copied row addresses them by the same law. The re-check at the end holds
/// the copies to that law again, driven the way `unroll_layers` reads it (by each body's own
/// op bindings), and any mismatch refuses with the split's name rather than shipping a
/// mis-addressed launch.
fn split_bodies_at_trip_wall(rolled: &mut RolledSuperDsc) -> Result<(), SuperDscError> {
    let wall = crate::superdsc_bake::BakeBudget::MAX_BUNDLE_TRIPS;
    // ── SCAN: every body over the wall whose residual seam is legal. Read-only — nothing
    //    moves until every split is known, so a refusal anywhere leaves the roll exactly as
    //    walked.
    let mut splits: Vec<(usize, usize, u32)> = Vec::new(); // (body, seam op, t_seam)
    for (bi, b) in rolled.bodies.iter().enumerate() {
        let trips: u64 = b.ops.iter().map(|o| o.time.max(1) as u64).sum();
        if trips <= wall || b.ops.iter().any(|o| o.ktir.is_none()) {
            continue;
        }
        let Some(seam) = b
            .ops
            .iter()
            .rposition(|o| {
                o.ktir
                    .as_ref()
                    .expect("the scan refused program-less ops above")
                    .bindings
                    .iter()
                    .any(|x| x.get() == b.hidden_in_tid)
            })
            .filter(|&s| s + 1 < b.ops.len())
        else {
            continue;
        };
        let earlier = bound_tids(&b.ops[..seam]);
        let seam_fresh: Vec<u32> = b.ops[seam]
            .ktir
            .as_ref()
            .expect("the scan refused program-less ops above")
            .bindings
            .iter()
            .map(|x| x.get())
            .filter(|t| !earlier.contains(t))
            .collect();
        if seam_fresh.len() != 1 {
            continue;
        }
        let t_seam = seam_fresh[0];
        let ahead = bound_tids(&b.ops[seam + 1..]);
        let cross: std::collections::BTreeSet<u32> = bound_tids(&b.ops[..=seam])
            .intersection(&ahead)
            .copied()
            .collect();
        if cross.len() != 1 || !cross.contains(&t_seam) {
            continue;
        }
        splits.push((bi, seam, t_seam));
    }
    if splits.is_empty() {
        return Ok(());
    }
    // ── SPLIT: descending body order, so a later split's insertion cannot disturb an earlier
    //    one's index. `A` keeps the body's slot (and its hidden_in); `B` takes the seam as its
    //    hidden_in and the body's hidden_out as its own. Both halves carry the ORIGINAL iters
    //    and layer_ids — every layer still runs both halves, and only the attention half
    //    holds the KV writer the layer ids are read from, so the MLP half could not collect
    //    them itself.
    for &(bi, seam, t_seam) in splits.iter().rev() {
        let b = rolled.bodies.remove(bi);
        let mut a_ops = b.ops;
        let b_ops = a_ops.split_off(seam + 1);
        rolled.bodies.insert(
            bi,
            RolledBody {
                ops: a_ops,
                iters: b.iters,
                layer_ids: b.layer_ids.clone(),
                hidden_in_tid: b.hidden_in_tid,
                hidden_out_tid: t_seam,
            },
        );
        rolled.bodies.insert(
            bi + 1,
            RolledBody {
                ops: b_ops,
                iters: b.iters,
                layer_ids: b.layer_ids,
                hidden_in_tid: t_seam,
                hidden_out_tid: b.hidden_out_tid,
            },
        );
    }
    // ── THE LAUNCH SEQUENCE AND ITS ROWS: every launch of a split body becomes TWO ADJACENT
    //    launches — A then B at the same class-relative iteration (true order preserved: A
    //    writes the seam, B reads it) — and every other body's launch keeps its order while
    //    its body index shifts past the inserted halves. The rows are COPIES: `launch_w` is
    //    `(bank, shift)` per launch and `launch_kv` the KV shift, and the proof held every
    //    per-layer tensor of the ORIGINAL class to exactly this entry at this iteration.
    let old_launches = std::mem::take(&mut rolled.launches);
    let old_w = std::mem::take(&mut rolled.launch_w);
    let old_kv = std::mem::take(&mut rolled.launch_kv);
    let shifted = |bi: usize| bi + splits.iter().filter(|&&(s, _, _)| s < bi).count();
    let mut launches: Vec<(u8, u32)> = Vec::with_capacity(old_launches.len() * 2);
    let mut launch_w: Vec<(u8, u64)> = Vec::with_capacity(old_launches.len() * 2);
    let mut launch_kv: Vec<u64> = Vec::with_capacity(old_launches.len() * 2);
    for (li, &(bi, it)) in old_launches.iter().enumerate() {
        let (w, kv) = (old_w[li], old_kv[li]);
        if splits.iter().any(|&(s, _, _)| s == bi as usize) {
            let a = shifted(bi as usize) as u8;
            launches.push((a, it));
            launches.push((a + 1, it));
            launch_w.push(w);
            launch_w.push(w);
            launch_kv.push(kv);
            launch_kv.push(kv);
        } else {
            launches.push((shifted(bi as usize) as u8, it));
            launch_w.push(w);
            launch_kv.push(kv);
        }
    }
    rolled.launches = launches;
    rolled.launch_w = launch_w;
    rolled.launch_kv = launch_kv;
    // ── THE COPY, HELD TO THE WALK'S OWN LAW. The re-check re-proves the launch-table
    //    relation over the post-split roll, driven the way `unroll_layers` reads it: by each
    //    launch's body's own op bindings, against the SHARED `per_layer` map. The walk proved
    //    these propositions over the pre-split roll (the B half's tables are a subset of the
    //    class the generator read), so a mismatch here is the split's own copy arithmetic —
    //    refused with its name rather than shipped as a mis-addressed launch.
    let w_seg = SegRole::Weight.segment();
    let kv_seg = SegRole::Kv.segment();
    for (li, &(bi, it)) in rolled.launches.iter().enumerate() {
        let (wb, wo) = rolled.launch_w[li];
        let ko = rolled.launch_kv[li];
        for op in &rolled.bodies[bi as usize].ops {
            let Some(k) = op.ktir.as_ref() else {
                continue;
            };
            for bind in k.bindings.iter() {
                let t0 = bind.get();
                let Some(tids) = rolled.per_layer.get(&t0) else {
                    continue;
                };
                // A table shorter than this launch's iteration is another class's (a table's
                // length is its own class's iters) — the walk proved it under that class's
                // launches.
                let Some(&tv) = tids.get(it as usize) else {
                    continue;
                };
                let (Some(p0), Some(pj)) = (
                    rolled.layout.placements.get(&t0),
                    rolled.layout.placements.get(&tv),
                ) else {
                    continue;
                };
                if p0.segment == w_seg && matches!(p0.role, SegRole::Weight) {
                    if pj.bank as u8 != wb || pj.offset != wo.wrapping_add(p0.offset) {
                        return Err(SuperDscError(format!(
                            "reroll-superdsc: the trip-wall split's launch table does not \
                             address per-layer weight t{t0} (body {bi}, iteration {it}) at bank \
                             {wb} offset {} — the walk proved the pre-split rows, so this is the \
                             split's own copy, refused here rather than shipped",
                            wo.wrapping_add(p0.offset),
                        )));
                    }
                } else if p0.segment == kv_seg && pj.offset != ko.wrapping_add(p0.offset) {
                    return Err(SuperDscError(format!(
                        "reroll-superdsc: the trip-wall split's launch table does not address \
                         per-layer KV tensor t{t0} (body {bi}, iteration {it}) at offset {} — \
                         the walk proved the pre-split rows, so this is the split's own copy, \
                         refused here rather than shipped",
                        ko.wrapping_add(p0.offset),
                    )));
                }
            }
        }
    }
    Ok(())
}
```

## L2 — THE CALL SITE (the lowering fn's tail, :3308-3325)

OLD:
```rust
    let kv_request_stride = bundle_layout.kv_request_stride_bytes;
    Ok(RolledSuperDsc {
        prefix,
        bodies,
        suffix,
        layout: bundle_layout,
        per_layer,
        launches,
        launch_w,
        launch_kv,
        kv_page_stride,
        prefix_weight_bank,
        suffix_weight_bank,
        kv_request_stride,
        suffix_in_tid,
        attn_params,
    })
}
```
NEW:
```rust
    let kv_request_stride = bundle_layout.kv_request_stride_bytes;
    let mut rolled = RolledSuperDsc {
        prefix,
        bodies,
        suffix,
        layout: bundle_layout,
        per_layer,
        launches,
        launch_w,
        launch_kv,
        kv_page_stride,
        prefix_weight_bank,
        suffix_weight_bank,
        kv_request_stride,
        suffix_in_tid,
        attn_params,
    };
    split_bodies_at_trip_wall(&mut rolled)?;
    Ok(rolled)
}
```
(The field list is untouched; the only semantic change is bind → split → return.)

## WHAT DOES NOT CHANGE

- **`per_layer` — SHARED, NOT PARTITIONED** (ruled above, fact 4). The layout (no placement
  moves). `prefix`, `suffix`, `kv_page_stride` (per_layer-derived), `prefix_weight_bank` /
  `suffix_weight_bank`, `suffix_in_tid`, `attn_params` (per-bundle facts; B's emit has no
  attention ops and does not read them).
- The walk, the pre-pass, the launch build, the generator, the proof, the per-body bank check
  (:3250-3265 — ran pre-split; each half's ops are a subset of the bank-checked set, and the
  copied rows carry the same bank).
- The bake loop, the twin, the census print, the write path, the wire, the loader, the
  executor, `unroll_layers`. If any of these surfaces an unexpected 3-hard-code at 2 classes,
  that is a REPORT (the rung-skip log names it — codegen :12336-12341), never a silent bend.

## L3 — THE TESTS (new `#[cfg(test)] mod trip_wall_split_tests` at the END of
`lower_subtile_tape_to_ktir.rs`, after the last existing test module)

The fixture follows the two verified in-tree idioms: **`minting_program`**
(superdsc.rs:9557-9688 — the `zz_` roster's law: real op kinds, hand-stated bindings
`[a, b, out]`, output last) and **`two_node_layout`** (superdsc.rs:9694-9719 —
`BundleLayout::default()` + `ids`/`placements` rows + `segment_bytes`), with the ops minted
through `ktir_superdsc::emit::whole_function::lower_function(prog, Some(&layout), &mut syms,
None)` (the two-node test's own call) and then PINNED to the two fields the split reads:
`op.ktir = Some(prog.clone())` (the walk's own attachment, stated by the fixture — idempotent
if the lowering already attaches) and `op.time`. Each minted program must lower to exactly ONE
fused op — **assert it** (`assert_eq!(ops.len(), 1, ...)`): a fixture whose shape drifted
FAILS, never quietly adapts. The coder reports the exact test diff in the apply message for
line-level audit (the K3 protocol); no `EmittedOp` struct literals enumerating 25 fields, no
production-code change to make the test compile, no `#[allow]`.

**The fixture law (every number pinned):** tids `t0` (hidden_in), `t2` (a staging), `t4`
(t_seam), `t6` (hidden_out), weights `t10/t11` and `t20/t21`. Three programs:
`seam_program(0, 10, 2, "split_a0")` (binds {t0, t10, t2}), `seam_program(2, 0, 4,
"split_join")` (binds {t2, t0, t4} — the join reads the staging AND the hidden, writes the
seam), `seam_program(4, 20, 6, "split_b0")` (binds {t4, t20, t6}). Times: 30_000 each —
trips 90_000 over the wall; A = 60_000, B = 30_000, both under. Layout: activation
placements for t0/t2/t4/t6 (segment 0, `SegRole::Activation`, size 8·64·2, two_node_layout's
spacing) and weight placements t10@0/t11@128/t20@256/t21@384 (segment
`SegRole::Weight.segment()`, `SegRole::Weight`, bank 0, size 8·64·2, `BindCover::Full`), with
`ids` rows and `segment_bytes` for both segments (the two_node_layout idiom, extended). Roll:
one body `{ops, iters: 2, layer_ids: [0, 1], hidden_in_tid: 0, hidden_out_tid: 6}`;
`per_layer: {10: [10, 11], 20: [20, 21]}`; `launches: [(0,0), (0,1)]`;
`launch_w: [(0,0), (0,128)]`; `launch_kv: [0, 0]`; the remaining fields empty/zero
(`prefix`/`suffix` `Vec::new()`, `kv_page_stride` 0, banks 0, `kv_request_stride` 0,
`suffix_in_tid: u32::MAX`, `attn_params: None`). One `syms` counter threaded through all
three `lower_function` calls.

**T1 — `a_body_over_the_trip_wall_splits_at_the_residual_seam`:** call
`split_bodies_at_trip_wall(&mut rolled).expect("the legal seam splits")`, then assert:
`bodies.len() == 2`; A = `bodies[0]` with `ops.len() == 2`, `hidden_in_tid == 0`,
`hidden_out_tid == 4`; B = `bodies[1]` with `ops.len() == 1`, `hidden_in_tid == 4`,
`hidden_out_tid == 6`; both `iters == 2` and `layer_ids == [0, 1]`; trips(A) == 60_000 and
trips(B) == 30_000 (the `Σ o.time.max(1)` sum), both `< MAX_BUNDLE_TRIPS`;
`launches == [(0,0), (1,0), (0,1), (1,1)]` (true order: A then B per layer);
`launch_w == [(0,0), (0,0), (0,128), (0,128)]` (B's rows are A's copies);
`launch_kv == [0, 0, 0, 0]`; `per_layer` UNCHANGED (the shared map — name the law in the
assert message). The re-check ran inside the call — its Ok is part of the verdict.

**T2 — `a_body_whose_seam_leaks_a_staging_refuses_the_split`:** replace `ops[2]` with
`seam_program(2, 20, 6, "split_leak")` at time 30_000 (B reads A's STAGING t2 — bindings
{t2, t20, t6}; cross = {t2} ≠ {t4}). Build the op BEFORE mutating (borrow order). Assert the
call returns Ok, `bodies.len() == 1`, `launches == [(0,0), (0,1)]`,
`launch_w == [(0,0), (0,128)]` — the conservative refusal leaves the roll as walked.

**T3 — `a_body_under_the_wall_passes_through_untouched`:** set all three `op.time = 1`
(trips 3 ≤ wall). Assert Ok, `bodies.len() == 1`, `launches`/`launch_w`/`launch_kv`
unchanged — the no-op path mutates nothing.

## Apply order and legs

L1 → L2 → L3, verbatim where quoted (rustfmt's own wrapping is the stable form — report it).
The coder reports anything else broken it sees — reports, never fixes. Then: auditor line
audit of the exact diff, local fmt pre-flight, re-sync `/work/attn-auditer-01` (oc rsync
--delete from the worktree root, cd inside the command), run-46 attempt 10 (detached setsid
nohup, verify via fresh connection), log monitor armed.

**EXPECTED at attempt 10:** all test legs green (existing fixtures are under the wall — the
fn returns before any mutation; the three new tests green). The build leg: the 384/512 rungs
bake as 4-bundle rolls — the census line prints "body {bfp} + 2 class(es), 52 launch(es)".
**Corrected per-half arithmetic (the draft's even-split estimate was wrong):** the attention
half carries the q/k/v/o GEMM chains (244 trips/tile: 73+49+49+73) and the MLP half the
gate/up/down chains (507/tile), over the 301-trip fixed tail — so A ≈ 23.5k / 31.2k and
B ≈ 48.9k / 65.0k at T=96 / T=128. The 384 rung is comfortable; **the 512 rung's B half is
≈65.0k, under the wall by only ~340-490 trips** — if the measurement refuses it, that
refusal is the next pin (11m: a three-way split or staging sharing), never a bending of the
wall. The ladder tops out at 512 (wall-16). THEN the order of record: the ONE judgment chat
(granite-3.1-2b fp8, "write about the history of france", greedy ids vs
[6082, 723, 30, 312, 10769, 316, 2481, 318], md5 a408a4f9eadd6a5fad12f85e5b4a5bc2), verdict
reported to the user before anything else.

Nothing commits, nothing pushes.
