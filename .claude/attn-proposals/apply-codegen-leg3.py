import sys

p = "crates/compiler/macros/src/codegen.rs"
src = open(p).read()

def rep(old, new, n=1):
    global src
    assert src.count(old) == n, ("MISS", n, src.count(old), old[:90])
    src = src.replace(old, new)

# ── J1. the ladder header: par_iter → serial loop + accumulator ───────────────────────
rep("""        // ⭐ THE RUNGS ARE BAKED IN PARALLEL AND COMMITTED IN ORDER. Each rung is a whole-model lowering
        // that reads nothing another rung writes — `emit_bundle_deferred` records its effects instead of
        // applying them — so the 25 bakes run side by side and `commit_bake` then applies their effects
        // rung by rung, ascending, exactly as the serial loop did (the last rung that lowered is still
        // the ceiling).
        //
        // A rung that does not lower comes back as its `(width, reason)` instead, collected in the same
        // ascending order at commit — what the window guard below names when the baked ladder stops
        // short.
        type RungBake = (to_wavefront::LoweredDecode, PrefillPick, Vec<BakeEffect>);
        let per_rung: Vec<Result<RungBake, (u32, String)>> = {
            use rayon::prelude::*;
            prefill_rung_widths.par_iter().map(|&pl| -> Result<RungBake, (u32, String)> {
            let lowered = to_wavefront::lower_decode_to_wavefront(""",
    """        // ⭐ THE RUNGS ARE LOWERED AND COMMITTED ONE AT A TIME, ASCENDING — THE SERIAL LADDER.
        // The whole-bake total budget needs a DETERMINISTIC order: each rung's keep decision
        // reads the accumulator of every rung before it, so a rung's staging cannot race a
        // narrower rung's commit. The parallelism this loop lost lives where it pays — the
        // nested chunk-ladder rayon inside every bundle's emission, and the dxp compile pool
        // behind the staging (MEASURED: the warm render is 85 s against a ~10 min compile
        // drain, so rung-level parallelism bought ~nothing while oversubscribing the 20-CPU
        // quota against the dxp children — the 63% render collapse of the 11k bake).
        //
        // A rung that does not lower, or one the bake budget refuses, comes back as its
        // `(width, reason)` instead, collected in the same ascending order — what the window
        // guard below names when the baked ladder stops short. A rung the whole-bake TOTAL
        // refuses STOPS the ladder: every later rung is wider and costs more.
        let mut refused_rungs: Vec<(u32, String)> = Vec::new();
        // ⭐ THE TOTAL-WORK ACCUMULATOR — Σ trips of every rung the ladder has STAGED, the
        //    running total of the `BakeBudget::MAX_TOTAL_TRIPS` decision below. Chunk-rung
        //    bundles never count (their tickets say so — measured, they never stage).
        let mut accumulated: u64 = 0;
        for &pl in &prefill_rung_widths {
            let lowered = to_wavefront::lower_decode_to_wavefront(""")

# ── J2. lowering match → lwd_p binding; the deferred call opens at loop level ─────────
rep("""            match lowered {
                Ok(lwd_p) => {
                    let (r, fx) = emit_bundle_deferred(""",
    """            let lwd_p = match lowered {
                Ok(lwd_p) => lwd_p,
                Err(e) if sendnn_superdsc => {
                    eprintln!(
                        "[spyre-superdsc] {stem}: prefill ladder rung m={pl} NOT LOWERED — {e} \\
                         (SKIPPED; a narrower rung still covers this width)"
                    );
                    refused_rungs.push((pl, e.to_string()));
                    continue;
                }
                Err(e) => panic!("[wavefront] {stem}: prefill (m={pl}) NOT LOWERED — {e}"),
            };
            let (r, fx, tickets) = match emit_bundle_deferred(""")

# ── J3. the commit: Budget arm, the total decision, submit, commit, accumulate ─────────
rep("""                    );
                    Ok((lwd_p, r, fx))
                }
                Err(e) if sendnn_superdsc => {
                    eprintln!(
                        "[spyre-superdsc] {stem}: prefill ladder rung m={pl} NOT LOWERED — {e} \\
                         (SKIPPED; a narrower rung still covers this width)"
                    );
                    Err((pl, e.to_string()))
                }
                Err(e) => panic!("[wavefront] {stem}: prefill (m={pl}) NOT LOWERED — {e}"),
            }
            }).collect()
        };
        let mut refused_rungs: Vec<(u32, String)> = Vec::new();
        for rung in per_rung {
            match rung {
                Ok((lwd_p, r, fx)) => {
                    commit_bake(&lwd_p, fx);
                    prefill = Some(r);
                }
                Err(refused) => refused_rungs.push(refused),
            }
        }""",
    """            ) {
                Ok(x) => x,
                // A bake-budget refusal (per-bundle or per-group) refused this rung at its
                // render — its renders were already discarded at the refusal site, so the
                // rung is unbaked and uncounted.
                Err(b) => {
                    eprintln!(
                        "[spyre-superdsc] {stem}: prefill ladder rung m={pl} BAKE BUDGET \\
                         REFUSED — {} (SKIPPED; a narrower rung still covers this width)",
                        b.0
                    );
                    refused_rungs.push((pl, b.0));
                    continue;
                }
            };
            // ⭐ THE TOTAL-WORK DECISION, AT COMMIT — the whole-bake budget judges the
            //    accumulator plus THIS rung's stageable trips BEFORE anything stages: a rung
            //    past the total is discarded unbaked (its renders cost only their render,
            //    never a compile) and the ladder STOPS — later rungs are wider and cost
            //    more. Landing exactly AT the ceiling is lawful, the `>` law.
            let trips: u64 = tickets
                .iter()
                .filter(|t| t.counts_toward_total)
                .map(|t| t.trips)
                .sum();
            if let Err(refused) =
                scratchy_target_spyre::superdsc_bake::BakeBudget::total_verdict(accumulated + trips)
            {
                superdsc::discard_stage_tickets(&tickets);
                eprintln!(
                    "[spyre-superdsc] {stem}: prefill ladder rung m={pl} REFUSED BY THE \\
                     WHOLE-BAKE TOTAL — {} (the ladder stops here; later rungs are wider \\
                     and cost more)",
                    refused.0
                );
                refused_rungs.push((pl, refused.0));
                break;
            }
            if let Err(e) = superdsc::submit_stage_tickets(&tickets) {
                panic!("[spyre-superdsc] {stem}: prefill rung m={pl} staging failed — {e}");
            }
            commit_bake(&lwd_p, fx);
            prefill = Some(r);
            accumulated += trips;
        }""")

# ── K. the census file, at the expansion's end (after the prefix ladder) ──────────────
rep("""        // ⛔⛔⛔ ONE MODEL PACKS ONE RESIDENT WEIGHT SEGMENT — the parity ledger's verdict,""",
    """        // ⭐ THE CENSUS PERSISTS — the width ladder's refusal record, on disk, because the
        //    build log is not one: cargo swallows a build script's stderr on SUCCESS, so a
        //    green build with a shortened ladder says nothing anywhere. One line per refused
        //    rung (width + refusal text), the kept total trips, and the ceiling — the
        //    calibration record for `BakeBudget::MAX_TOTAL_TRIPS`. Best-effort, same law as
        //    the dump writes: a failed write is an eprintln, never a build failure.
        if sendnn_superdsc {
            let mut census = String::new();
            for (pl, msg) in &refused_rungs {
                census.push_str(&format!("PREFILL RUNG m={pl} REFUSED: {msg}\\n"));
            }
            census.push_str(&format!(
                "KEPT TOTAL TRIPS: {accumulated}\\nTOTAL CEILING: {}\\n",
                scratchy_target_spyre::superdsc_bake::BakeBudget::MAX_TOTAL_TRIPS
            ));
            let path = std::path::Path::new("/tmp/superdsc-dump").join(format!("{stem_id}-census.txt"));
            if let Err(e) = std::fs::write(&path, census) {
                eprintln!(
                    "[spyre-superdsc] {stem}: could not write the census at {}: {e}",
                    path.display()
                );
            }
        }

        // ⛔⛔⛔ ONE MODEL PACKS ONE RESIDENT WEIGHT SEGMENT — the parity ledger's verdict,""")

open(p, "w").write(src)
print("sites J-K ok")
