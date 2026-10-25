# DEFECT-3 SURFACE — brought for ruling (2026-10-25, process call)

The uncommitted attention-contract/defect-3 work in this worktree, as raw
`git diff` against `attn-contract` HEAD (ee1ca99b3). Provenance: these
modifications were present in the worktree at the START of the coder
session that landed B and the accumulator (the session's opening git
status lists all 16 files) — they are the typed producer-consumer layout
binding for defect 3 (fp8 quant staging tears rows for Q/K/V; P::Synth /
PlaceId::Synth, the tile field) from earlier sessions of this campaign
line. This session landed NONE of them; every landing this session made
(budget-law chain, B, the accumulator) went through proposal → ruling →
audit.

## Files

- `pure-files.diff` — the 12 files whose every hunk is defect-3 work:
  ktir_tokens.rs, emit/ktir_matmul_fp8.rs, emit/lower_ktir_to_superdsc.rs,
  emit/mod.rs, pointwise.rs, place.rs, placement.rs, sdsc_abstract.rs,
  work.rs, and the three zz_a tests.
- `codegen-annotated.diff` — MIXED. Ruled hunks inside it (exclude from
  the defect-3 audit; each was proposal→ruling→audit-landed):
  the budget-law legs (the deferred closure, budget doors, sweep
  per-item Budget, first-Err collect, wrapper panic+submit, serial
  prefill ladder), B (:10456-10465 the CHUNK_FORMED_PREFILL_RUNGS const,
  the prefill_rung_widths derivation), and the accumulator threading
  (the RefCell + wrapper door + prefix-ladder deferred + census/parity
  move). Everything else in the file's diff is defect-3.
- `superdsc_bake-annotated.diff` — MIXED. Ruled: the BakeBudget block
  (MAX_* consts, total_verdict, bundle_verdict), the stage-at-commit
  registry, and the inventory comment nit. Rest: defect-3.
- `lower_ktir-annotated.diff` — MIXED. Ruled: the census leg, RolledBody
  construction, mint_trips, census_bodies_over_the_trip_wall. Rest:
  defect-3.
- `lower_superdsc-annotated.diff` — MIXED. Ruled: StageTicket,
  stage_tickets_of / submit / discard, the stage-at-commit law. Rest:
  defect-3.

## Audit protocol proposal

Chunked line-audit per your law: pure-files.diff first (self-contained,
12 files), then the annotated four file-by-file with the ruled hunks
named above treated as already-audited surface. Rulings applied
verbatim; nothing else in the tree moves until you rule.
