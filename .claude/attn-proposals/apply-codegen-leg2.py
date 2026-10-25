import sys

p = "crates/compiler/macros/src/codegen.rs"
src = open(p).read()

def rep(old, new, n=1):
    global src
    assert src.count(old) == n, ("MISS", n, src.count(old), old[:90])
    src = src.replace(old, new)

# ── G. sweep ladder: per-rung ticket lists + per-item Budget threading ────────────────
rep("""                                                let per_rung: Vec<Option<RungRecord>> = {
                                                    use rayon::prelude::*;
                                                    ladder
                                                        .into_par_iter()
                                                        .map(|(rung, rung_chunk, rung_swept)| -> Option<RungRecord> {""",
    """                                                let per_rung: Vec<
                                                    Result<
                                                        (
                                                            Option<RungRecord>,
                                                            Vec<superdsc::StageTicket>,
                                                        ),
                                                        BudgetRefusal,
                                                    >,
                                                > = {
                                                    use rayon::prelude::*;
                                                    ladder
                                                        .into_par_iter()
                                                        .map(
                                                            |(rung, rung_chunk, rung_swept)| -> Result<
                                                            (
                                                                Option<RungRecord>,
                                                                Vec<superdsc::StageTicket>,
                                                            ),
                                                            BudgetRefusal,
                                                        > {""")

rep("""                                                            let mut rfps: Vec<String> = Vec::new();
                                                            let mut rfused: Vec<
                                                                superdsc::bundle::SiblingFp<
                                                                    'static,
                                                                >,
                                                            > = Vec::new();
                                                            let mut ok = true;""",
    """                                                            let mut rfps: Vec<String> = Vec::new();
                                                            let mut rfused: Vec<
                                                                superdsc::bundle::SiblingFp<
                                                                    'static,
                                                                >,
                                                            > = Vec::new();
                                                            let mut ok = true;
                                                            // ⭐ THE RUNG'S OWN TICKET LISTS, stageable
                                                            //    and chunk apart: a chunk fp's groups
                                                            //    always resolve through the bake's memo
                                                            //    (MEASURED across three full bakes,
                                                            //    nothing ever stages for them), so its
                                                            //    trips are not compile work the whole-
                                                            //    bake total counts.
                                                            let mut rung_staged: Vec<String> =
                                                                Vec::new();
                                                            let mut rung_chunk_fps: Vec<String> =
                                                                Vec::new();""")

rep("""                                                                    Ok(rfp) => {
                                                                        rfps.push(rfp);""",
    """                                                                    Ok(rfp) => {
                                                                        rfps.push(rfp.clone());
                                                                        if rung_chunk.is_some() {
                                                                            rung_chunk_fps.push(
                                                                                rfp.clone(),
                                                                            );
                                                                        } else {
                                                                            rung_staged.push(
                                                                                rfp.clone(),
                                                                            );
                                                                        }""")

rep("""                                                                        rfused.push(if card
                                                                            && rung_chunk.is_none()
                                                                        {
                                                                        superdsc::emit_bundle(
                                                                            &b.ops,
                                                                            Some(&rr.layout),
                                                                            superdsc::FoldGrouping::Fused,
                                                                            rr.attn_params,
                                                                        )
                                                                        .map(
                                                                            superdsc::bundle::SiblingFp::from,
                                                                        )
                                                                        .unwrap_or_else(|e| {
                                                                            eprintln!(
                                                                                "[spyre-superdsc] \\
                                                                                 {base}: rung {} fused \\
                                                                                 twin emit failed: {e}",
                                                                                rung.get()
                                                                            );
                                                                            superdsc::bundle::SiblingFp::none()
                                                                        })
                                                                    } else {
                                                                        superdsc::bundle::SiblingFp::none()
                                                                    });
                                                                    }
                                                                    Err(e) => {
                                                                        eprintln!(
                                                                            "[spyre-superdsc] {base}: \\
                                                                         ladder rung {} chunk={rung_chunk:?} \\
                                                                         body class {ci} emit \\
                                                                         failed: {e}",
                                                                            rung.get()
                                                                        );
                                                                        ok = false;
                                                                    }
                                                                }
                                                            }""",
    """                                                                        rfused.push(
                                                                            match if card
                                                                                && rung_chunk.is_none()
                                                                            {
                                                                                Some(
                                                                                    superdsc::emit_bundle(
                                                                                        &b.ops,
                                                                                        Some(&rr.layout),
                                                                                        superdsc::FoldGrouping::Fused,
                                                                                        rr.attn_params,
                                                                                    ),
                                                                                )
                                                                            } else {
                                                                                None
                                                                            } {
                                                                                // The twin is a
                                                                                // launch-grouping
                                                                                // VARIANT: a failure
                                                                                // here (budget
                                                                                // included) falls to
                                                                                // the split body,
                                                                                // the selector's
                                                                                // documented
                                                                                // behavior.
                                                                                Some(Ok(tfp)) => {
                                                                                    rung_staged.push(
                                                                                        tfp.clone(),
                                                                                    );
                                                                                    superdsc::bundle::SiblingFp::from(tfp)
                                                                                }
                                                                                Some(Err(e)) => {
                                                                                    eprintln!(
                                                                                        "[spyre-superdsc] \\
                                                                                         {base}: rung {} fused \\
                                                                                         twin emit failed: {e}",
                                                                                        rung.get()
                                                                                    );
                                                                                    superdsc::bundle::SiblingFp::none()
                                                                                }
                                                                                None => {
                                                                                    superdsc::bundle::SiblingFp::none()
                                                                                }
                                                                            },
                                                                        );
                                                                    }
                                                                    // ⭐ THE BUDGET DOOR, rung-local:
                                                                    //    this sweep rung's renders are
                                                                    //    discarded and the refusal rides
                                                                    //    out per-item — the closure
                                                                    //    files it after the collect.
                                                                    Err(superdsc::EmitError::Budget(b)) => {
                                                                        let mut tk =
                                                                            superdsc::stage_tickets_of(
                                                                                &rung_staged,
                                                                                true,
                                                                            );
                                                                        tk.extend(
                                                                            superdsc::stage_tickets_of(
                                                                                &rung_chunk_fps,
                                                                                false,
                                                                            ),
                                                                        );
                                                                        superdsc::discard_stage_tickets(&tk);
                                                                        return Err(b);
                                                                    }
                                                                    Err(e) => {
                                                                        eprintln!(
                                                                            "[spyre-superdsc] {base}: \\
                                                                         ladder rung {} chunk={rung_chunk:?} \\
                                                                         body class {ci} emit \\
                                                                         failed: {e}",
                                                                            rung.get()
                                                                        );
                                                                        ok = false;
                                                                    }
                                                                }
                                                            }
                                                            let mut tickets =
                                                                superdsc::stage_tickets_of(
                                                                    &rung_staged,
                                                                    true,
                                                                );
                                                            tickets.extend(
                                                                superdsc::stage_tickets_of(
                                                                    &rung_chunk_fps,
                                                                    false,
                                                                ),
                                                            );""")

rep("""                                                                Some((
                                                                    rec,
                                                                    (
                                                                        superdsc::bundle::SweptCols::new(rung_swept),
                                                                        scratchy_target_spyre::fold_plan::RungReach {
                                                                            folds: rung_folds,
                                                                            chunk: rung_chunk.is_some(),
                                                                            fused_twin,
                                                                        },
                                                                    ),
                                                                    (rung, rung_chunk, rr.layout),
                                                                ))
                                                            } else {
                                                                None
                                                            }
                                                        }
                                                        Err(e) => {
                                                            eprintln!(
                                                                "[spyre-superdsc] {base}: ladder rung {} lower failed: {e}",
                                                                rung.get()
                                                            );
                                                            None
                                                        }
                                                    }
                                                        })
                                                        .collect()
                                                };
                                                for (rec, reach, (rung, rung_chunk, layout)) in
                                                    per_rung.into_iter().flatten()
                                                {""",
    """                                                                Ok((
                                                                    Some((
                                                                        rec,
                                                                        (
                                                                            superdsc::bundle::SweptCols::new(rung_swept),
                                                                            scratchy_target_spyre::fold_plan::RungReach {
                                                                                folds: rung_folds,
                                                                                chunk: rung_chunk.is_some(),
                                                                                fused_twin,
                                                                            },
                                                                        ),
                                                                        (rung, rung_chunk, rr.layout),
                                                                    )),
                                                                    tickets,
                                                                ))
                                                            } else {
                                                                Ok((None, tickets))
                                                            }
                                                        }
                                                        Err(e) => {
                                                            eprintln!(
                                                                "[spyre-superdsc] {base}: ladder rung {} lower failed: {e}",
                                                                rung.get()
                                                            );
                                                            Ok((None, Vec::new()))
                                                        }
                                                    }
                                                        })
                                                        .collect()
                                                };
                                                // ⭐ A SWEEP RUNG'S BUDGET REFUSAL REFUSES THE
                                                //    WHOLE CLOSURE RUNG: the ladder this match
                                                //    builds would carry work the refusal says the
                                                //    bake cannot pay for, so the rung's renders —
                                                //    the ceiling's AND the sibling rungs' already-
                                                //    emitted ones — are discarded here, and the
                                                //    refusal rides out to the caller.
                                                let mut sweep_refusal: Option<BudgetRefusal> =
                                                    None;
                                                for item in per_rung {
                                                    let (rec, reach, (rung, rung_chunk, layout)) =
                                                        match item {
                                                            Ok((Some(x), tickets)) => {
                                                                rung_tickets.extend(tickets);
                                                                x
                                                            }
                                                            Ok((None, tickets)) => {
                                                                rung_tickets.extend(tickets);
                                                                continue;
                                                            }
                                                            Err(b) => {
                                                                sweep_refusal =
                                                                    sweep_refusal.or(Some(b));
                                                                continue;
                                                            }
                                                        };""")

rep("""                                            decode_rungs.sort_by_key(|r| r.active_cap);""",
    """                                            if let Some(b) = sweep_refusal {
                                                let mut all = superdsc::stage_tickets_of(&stage_fps, true);
                                                all.extend(rung_tickets.drain(..));
                                                superdsc::discard_stage_tickets(&all);
                                                return Err(b);
                                            }
                                            decode_rungs.sort_by_key(|r| r.active_cap);""")

# ── H. closure tail: the tickets ride the Ok payload ──────────────────────────────────
rep("""                ((man, nodes, sengraph_json, group_graphs), fx)
            };""",
    """                let mut tickets = superdsc::stage_tickets_of(&stage_fps, true);
                tickets.extend(rung_tickets.drain(..));
                Ok(((man, nodes, sengraph_json, group_graphs), fx, tickets))
            };""")

# ── I. the wrapper: submit immediately, Budget panics ─────────────────────────────────
rep("""                let (r, fx) = emit_bundle_deferred(
                    lwd,
                    base,
                    cap,
                    seq_sym,
                    prefill_mq,
                    active_cap,
                    decode_rows,
                );
                commit_bake(lwd, fx);
                r
            };""",
    """                let (r, fx, tickets) = match emit_bundle_deferred(
                    lwd,
                    base,
                    cap,
                    seq_sym,
                    prefill_mq,
                    active_cap,
                    decode_rows,
                ) {
                    Ok(x) => x,
                    // ⛔ THE PRIMARY/BATCH PATH HAS NO NARROWER RUNG TO FALL TO — a bake-budget
                    //    refusal here is the same build failure it was before the type existed
                    //    (this site panicked on the emit's Err already); the type only carries
                    //    the text verbatim now.
                    Err(b) => panic!(
                        "[spyre-superdsc] {base}: BAKE BUDGET REFUSED — {} (the decode/primary \\
                         bake has no ladder to shorten)",
                        b.0
                    ),
                };
                if let Err(e) = superdsc::submit_stage_tickets(&tickets) {
                    panic!("[spyre-superdsc] {base}: staging failed — {e}");
                }
                commit_bake(lwd, fx);
                r
            };""")

open(p, "w").write(src)
print("sites G-I ok")
