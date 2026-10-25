# 11i — the three lib-test failures at the 11h tree (AUDITOR-RULED exact code)

Run-46 attempt 6 (23:29 UTC): fmt/check/clippy/tests-subtile/tests-ktir GREEN,
**tests-triton GREEN** (11h's H4 holds — the rung's cover test passes), tests-spyre
and tests-forward_tape GREEN. **tests-ktir-lib: 224 passed, 3 FAILED** (RC=101; the
release-build leg is gated on the test legs and skipped). All three pin to defects
in never-run test/emission code; the fixes are ruled below as exact code. The coder
applies VERBATIM, I1 → I2 → I3.

## FORENSICS — the compile layer is done; this is the assertion layer

11h cleared all 28 compile errors; the lib tests ran for the first time (224/227
pass). The 3 failures are: one STALE PRE-REDIRECT TEST asserting the deleted
invented N-tail arm, and two OP-NAME MISMATCHES between the tiled emission and the
tiled tests — both written pre-compile, never run against each other until now.

## I1 — DELETE the stale 26b test (ktir_matmul_fp8.rs :1847-1867)

`the_26b_n2112_packed_kernel_out_stick` asserts `.expect("the n=2112 packed-kernel
fp8 chain emits (the 26b gate gemm wall is gone)")` — acceptance by the packed-gran
N-tail arm that proposal 11's D1 DELETED. The emission now refuses with the
reference's own refusal (`fp8_linear_kernel.py:90`, remainder 64 ∉ [4096, 1024,
128]) — exactly what the R1-corrected fidelity sweep put on record: the 26b gate's
2112 is a width torch-spyre never sends through this kernel (the 26b is bf16
there), so the refusal is FIDELITY, not regression. The D2d replacement test
(`the_26b_gate_n_tail_is_the_references_own_refusal`, work.rs) pins the refusal and
is among the 224 passing. The stale test is deleted, not rewritten — its subject is
already pinned, by the ruled form.

DELETE ktir_matmul_fp8.rs :1847-1867 — the 7-line doc comment (`/// ⭐⭐⭐⭐⭐ THE 26b
UP/GATE GEMM: …`), the `#[test]`, the whole fn `the_26b_n2112_packed_kernel_out_stick`,
and the trailing blank line :1867. What remains: `}` of
`an_aligned_k_emits_no_pad_op_and_no_widened_reservation` (:1845), ONE blank line
(:1846), then the `/// An operand's own HBM start …` doc (was :1868). One blank
between items — fmt-stable. NOTHING else in the file's test module changes; the
passing sibling `the_26b_k2112_pads_to_a_whole_fp8_stick` (:1771) is untouched.

## I2 — the quant chain's `opn` closure: the tile index rides BEFORE `_op` (:186-192)

The tiled arm's one naming law is `{out}_fq_<kind>{tile}_op` — stated by six of its
eight op kinds (`fq_tz{t}_op`, `fq_tc{t}g{g}_op`, `fq_dqa_t{t}_op`, `fq_dqw_t{t}_op`,
`fq_to{t}c{c}_op`) and by 11's own D2g (`fq_mm_narrow_op`). The `opn` closure's
Some(t) arm is the outlier: `format!("{out}_{s}_t{t}")` with every input ending
`_op` produces `t10_fq_qspt_op_t0` — the tile lands AFTER the suffix, so the tests'
`ends_with("_op")` census counts zero. Op names are OUR vocabulary (torch-spyre has
no op names — nothing to copy); the arbiter is the arm's own uniform law, and the
closure is fixed to it. The None (dense/expert) arm is byte-identical — every
dense-law test's names are unchanged.

ktir_matmul_fp8.rs :186-192 — OLD:
```rust
        // OP names — a DIFFERENT namespace from tensor names (a program's ops are not placed), so
        // they stay strings and take no `PlaceId`. A tiled source carries its tile index here too,
        // so a tile's ops are named apart from every other tile's.
        let opn = |s: &str| match src.tile {
            Some(t) => format!("{out}_{s}_t{t}"),
            None => format!("{out}_{s}"),
        };
```
NEW:
```rust
        // OP names — a DIFFERENT namespace from tensor names (a program's ops are not placed), so
        // they stay strings and take no `PlaceId`. A tiled source carries its tile index here too,
        // BEFORE the trailing `_op` — the tiled arm's one naming law (`fq_tz{t}_op`,
        // `fq_mm_t{t}_op`) — so a tile's ops are named apart from every other tile's.
        let opn = |s: &str| match src.tile {
            Some(t) => format!("{out}_{}_t{t}_op", s.strip_suffix("_op").unwrap_or(s)),
            None => format!("{out}_{s}"),
        };
```
Every current input ends in `_op` (fq_qspt_op, fq_qspt_p{j}_op, fq_qspt_c{i}_op,
fq_invs_op, fq_sc_op, fq_clip_op, fq_zfpad_op, fq_afp8_op), so `strip_suffix` always
engages and the closure is total — no panic path; a hypothetical `_op`-less input
still gets the correct `{out}_{s}_t{t}_op` form.

## I3 — the tile mm's op name gains `_op` (:887)

`emit_sdsc_tiled` uses its name verbatim (emit/mod.rs:3036 — nothing appends
`_op`), so the tiled mm is `t10_fq_mm_t0` while every other op in the arm ends
`_op` — the second outlier behind the tests' zero counts.

ktir_matmul_fp8.rs :887 — OLD:
```rust
        &format!("{out}_fq_mm_t{t}"),
```
NEW:
```rust
        &format!("{out}_fq_mm_t{t}_op"),
```
The dense mm (`opn("fq_mm")` at :736, the :680 closure — dense-only) and the narrow
mm (`{out}_fq_mm_narrow_op` at :942) are UNTOUCHED.

## Blast radius — verified, on the record
- No consumer of `_op_t{t}` or `fq_mm_t` exists outside ktir_matmul_fp8.rs (grep
  over crates/targets/spyre/ktir/ and crates/triton/ — hits are false positives:
  `tile_op`/`run_op_try` identifiers, unrelated test fn names).
- No passing test's match outcome changes: dense/expert sources ride the None arm
  (byte-identical); m=1 decode goes through the single-tile dense early-return;
  the tiled qspt names never matched `ends_with("fq_qspt_op")` in either form; the
  partials (`fq_qspt_p/c`) only fire at pairs>1, which the tiled arm never passes.
- Op names are labels: no bake/launcher/card consumer matches them (the DDL
  dispatch is by op_func; placements.json carries tensors, not ops). The judgment
  gate is greedy token ids, not bundle bytes.
- The two failing tests' counts move 0→6 under I2/I3 and their other asserts
  (dqa/dqw/to censuses, the afp8 negative, the mm A-walk loop, the staging
  reservations) are name-form-independent or already-passing forms.

## What does NOT change
- The emission's OP SEQUENCE, shapes, placements, bindings — zero semantic bytes.
- The dense body's :680 closure, the narrow arm, `fp8_tile_mm_op`'s spec, both
  staging types, the plan, every tensor name (`synth_tiled` Display untouched).
- The tests themselves — their censuses stand as written; the emission is brought
  to the law they encode.

## Apply order and legs
I1 → I2 → I3, verbatim, no other edits. The coder reports anything else broken it
sees — reports, never fixes. Then: auditor local fmt pre-flight, re-sync
/work/attn-auditer-01, run-46 attempt 7. Nothing commits, nothing pushes.
