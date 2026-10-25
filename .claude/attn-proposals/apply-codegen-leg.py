import sys

p = "crates/compiler/macros/src/codegen.rs"
src = open(p).read()

def rep(old, new, n=1):
    global src
    assert src.count(old) == n, ("MISS", n, src.count(old), old[:90])
    src = src.replace(old, new)

# ── A. outer `use` + closure head: Result return + stage_fps ──────────────────────────
rep("""        // `prefill_mq`: Some(mq) marks the PREFILL path AND carries the concrete query-row count the
        // tape was lowered at — the ladder rung this bake belongs to. `None` = decode.
        let emit_bundle_deferred =""",
    """        // `prefill_mq`: Some(mq) marks the PREFILL path AND carries the concrete query-row count the
        // tape was lowered at — the ladder rung this bake belongs to. `None` = decode.
        use scratchy_target_spyre::lower_subtile_tape_to_superdsc as superdsc;
        use scratchy_target_spyre::superdsc_bake::BudgetRefusal;
        let emit_bundle_deferred =""")

rep("""             decode_rows: u32|
             -> (BakeOut, Vec<BakeEffect>) {
                let mut fx: Vec<BakeEffect> = Vec::new();""",
    """             decode_rows: u32|
             -> Result<
                (BakeOut, Vec<BakeEffect>, Vec<superdsc::StageTicket>),
                BudgetRefusal,
            > {
                let mut fx: Vec<BakeEffect> = Vec::new();
                // ⭐ THIS RUNG'S STAGE CLAIMS — every bundle the closure renders, pushed at each
                //    emit site, submitted (or discarded) by the CALLER's commit: the prefill
                //    ladder after its whole-bake total decision, the wrapper immediately. This is
                //    the render half of the stage-at-commit law; `superdsc::submit_stage_tickets`
                //    is the commit half. Chunk-rung bundles ride the per-rung ticket lists instead
                //    (their flag says uncounted), so this list holds only stageable work.
                let mut stage_fps: Vec<String> = Vec::new();
                let mut rung_tickets: Vec<superdsc::StageTicket> = Vec::new();""")

# ── A2. the closure's one early return: wrap the empty BakeOut in Ok ──────────────────
rep("""                    return ((String::new(), Vec::new(), String::new(), Vec::new()), fx);""",
    """                    // No bundle was emitted yet (the reroll never ran), so there are no
                    // stage claims to carry — the empty ticket list is the honest payload.
                    return Ok((
                        (String::new(), Vec::new(), String::new(), Vec::new()),
                        fx,
                        Vec::new(),
                    ));""")

# ── B. GATE-1: push the fp ────────────────────────────────────────────────────────────
rep("""                            Ok(fp) => eprintln!(
                                "[spyre-superdsc] GATE-1 single-matmul bundle emitted (fp {fp}) — \\
                                 dxp compiles it inline"
                            ),""",
    """                            Ok(fp) => {
                                stage_fps.push(fp.clone());
                                eprintln!(
                                    "[spyre-superdsc] GATE-1 single-matmul bundle emitted (fp {fp}) \\
                                     — dxp compiles it inline"
                                )
                            }""")

# ── C. prefix bundle: the Budget door ─────────────────────────────────────────────────
rep("""                                    let pre = if card {
                                        superdsc::emit_bundle(
                                            &rolled.prefix,
                                            Some(&rolled.layout),
                                            superdsc::FoldGrouping::Split,
                                            rolled.attn_params,
                                        )
                                    } else {
                                        Ok(String::new())
                                    };""",
    """                                    let pre = if card {
                                        match superdsc::emit_bundle(
                                            &rolled.prefix,
                                            Some(&rolled.layout),
                                            superdsc::FoldGrouping::Split,
                                            rolled.attn_params,
                                        ) {
                                            Ok(pfp) => {
                                                stage_fps.push(pfp.clone());
                                                Ok(pfp)
                                            }
                                            // ⭐ A BAKE-BUDGET REFUSAL IS A RUNG REFUSAL — the ONE
                                            //    error this closure returns as a value. The rung's
                                            //    already-rendered bundles are discarded HERE (never
                                            //    staged, never compiled — the refusal costs only its
                                            //    render), and the refusal rides out for the ladder
                                            //    to file in `refused_rungs`; a primary caller
                                            //    turns it into a panic with the same text.
                                            Err(superdsc::EmitError::Budget(b)) => {
                                                superdsc::discard_stage_tickets(
                                                    &superdsc::stage_tickets_of(&stage_fps, true),
                                                );
                                                return Err(b);
                                            }
                                            Err(e) => Err(e),
                                        }
                                    } else {
                                        Ok(String::new())
                                    };""")

# ── D. body classes: EmitError + the Budget door ──────────────────────────────────────
rep("""                                    let mut body_fps: Vec<String> = Vec::new();
                                    let mut body_err: Option<std::io::Error> = None;
                                    for (ci, b) in rolled.bodies.iter().enumerate() {
                                        let r = superdsc::emit_bundle(
                                            if card { &b.ops } else { &unrolled },
                                            Some(&rolled.layout),
                                            superdsc::FoldGrouping::Split,
                                            if card {
                                                rolled.attn_params
                                            } else {
                                                unrolled_params
                                            },
                                        );
                                        match r {
                                            Ok(fp) => body_fps.push(fp),
                                            Err(e) => {
                                                eprintln!(
                                                    "[spyre-superdsc] {base}: body class {ci} emit \\
                                                     failed: {e}"
                                                );
                                                if decode_rows <= 1 {
                                                    body_err = body_err.or(Some(e));
                                                }
                                            }
                                        }
                                    }
                                    let bod: Result<String, std::io::Error> =
                                        body_err.map(Err).unwrap_or_else(|| {
                                            Ok(body_fps.first().cloned().unwrap_or_default())
                                        });""",
    """                                    let mut body_fps: Vec<String> = Vec::new();
                                    let mut body_err: Option<superdsc::EmitError> = None;
                                    for (ci, b) in rolled.bodies.iter().enumerate() {
                                        let r = superdsc::emit_bundle(
                                            if card { &b.ops } else { &unrolled },
                                            Some(&rolled.layout),
                                            superdsc::FoldGrouping::Split,
                                            if card {
                                                rolled.attn_params
                                            } else {
                                                unrolled_params
                                            },
                                        );
                                        match r {
                                            Ok(fp) => {
                                                stage_fps.push(fp.clone());
                                                body_fps.push(fp);
                                            }
                                            // ⭐ THE BUDGET DOOR, same law as the prefix's: the
                                            //    rung's renders are discarded and the refusal
                                            //    rides out — a narrower rung still covers the
                                            //    width.
                                            Err(superdsc::EmitError::Budget(b)) => {
                                                superdsc::discard_stage_tickets(
                                                    &superdsc::stage_tickets_of(&stage_fps, true),
                                                );
                                                return Err(b);
                                            }
                                            Err(e) => {
                                                eprintln!(
                                                    "[spyre-superdsc] {base}: body class {ci} emit \\
                                                     failed: {e}"
                                                );
                                                if decode_rows <= 1 {
                                                    body_err = body_err.or(Some(e));
                                                }
                                            }
                                        }
                                    }
                                    let bod: Result<String, superdsc::EmitError> =
                                        body_err.map(Err).unwrap_or_else(|| {
                                            Ok(body_fps.first().cloned().unwrap_or_default())
                                        });""")

# ── E. ceiling twin: push on Ok, Budget falls to split ────────────────────────────────
rep("""                                                    superdsc::emit_bundle(
                                                        &b.ops,
                                                        Some(&rolled.layout),
                                                        superdsc::FoldGrouping::Fused,
                                                        rolled.attn_params,
                                                    )
                                                    .map(superdsc::bundle::SiblingFp::from)
                                                    .unwrap_or_else(|e| {
                                                        eprintln!(
                                                            "[spyre-superdsc] {base}: fold-fused \\
                                                             body twin emit failed: {e}"
                                                        );
                                                        superdsc::bundle::SiblingFp::none()
                                                    })""",
    """                                                    match superdsc::emit_bundle(
                                                        &b.ops,
                                                        Some(&rolled.layout),
                                                        superdsc::FoldGrouping::Fused,
                                                        rolled.attn_params,
                                                    ) {
                                                        Ok(tfp) => {
                                                            stage_fps.push(tfp.clone());
                                                            superdsc::bundle::SiblingFp::from(tfp)
                                                        }
                                                        // The twin is a launch-grouping VARIANT:
                                                        // an emit failure here (budget included)
                                                        // falls to the split body — one launch per
                                                        // layer dearer, the selector's documented
                                                        // behavior — never a rung refusal.
                                                        Err(e) => {
                                                            eprintln!(
                                                                "[spyre-superdsc] {base}: fold-fused \\
                                                                 body twin emit failed: {e}"
                                                            );
                                                            superdsc::bundle::SiblingFp::none()
                                                        }
                                                    }""")

# ── F. suffix: the Budget door ────────────────────────────────────────────────────────
rep("""                                    let suf = if card {
                                        superdsc::emit_bundle(
                                            &rolled.suffix,
                                            Some(&rolled.layout),
                                            superdsc::FoldGrouping::Split,
                                            rolled.attn_params,
                                        )
                                    } else {
                                        Ok(String::new())
                                    };""",
    """                                    let suf = if card {
                                        match superdsc::emit_bundle(
                                            &rolled.suffix,
                                            Some(&rolled.layout),
                                            superdsc::FoldGrouping::Split,
                                            rolled.attn_params,
                                        ) {
                                            Ok(sfp) => {
                                                stage_fps.push(sfp.clone());
                                                Ok(sfp)
                                            }
                                            // ⭐ THE BUDGET DOOR, same law as the prefix's.
                                            Err(superdsc::EmitError::Budget(b)) => {
                                                superdsc::discard_stage_tickets(
                                                    &superdsc::stage_tickets_of(&stage_fps, true),
                                                );
                                                return Err(b);
                                            }
                                            Err(e) => Err(e),
                                        }
                                    } else {
                                        Ok(String::new())
                                    };""")

open(p, "w").write(src)
print("sites A-F ok")
