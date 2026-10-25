# 09a — addendum to 09 (U1-U5 per your verdict). Still PAPER until the user lifts the
standing stop-order.

## U1 — lower_ktir_to_superdsc.rs is a BOTH-CONCERNS file (joins work.rs, split by hunk)

**Concern A** (SpraysFirst) keeps: the door refusal, the three spray tests, the
y-broadcast helpers' region. **Concern B** (fp8 retype) carries the three ripple hunks:
- `:4842` — the RoPE collapse's `MatM::of_token_rows(RowCount::of_head_major_rows(heads * mq))`
- `:6734` — the hoisted `let m_rows = RowCount::of_token_rows(m);` mint
- `:5994` — the `at_head` call's `Idx::<Head>::n(h)` argument

The by-hunk rule now covers exactly two files: work.rs and lower_ktir_to_superdsc.rs.

## U2 — Concern B's file list, completed (its whole ripple or it doesn't compile)

B additionally carries the seven test-file ripples:
- `crates/targets/spyre/tests/lm_head_trip_cost.rs` (:21)
- `crates/targets/spyre/tests/sdsc_fp8_stick_split.rs` (:29)
- `crates/targets/spyre/tests/chunk_explore_is_the_groups.rs` (:57)
- `crates/targets/spyre/tests/split_leaves_whole_operand_sticks.rs` (:41)
- `ktir-superdsc/tests/zz_a_one_row_matmul_fills_every_core.rs` (:89)
- `ktir-superdsc/tests/zz_the_index_tensors_shape_is_the_paged_dims.rs` (:222)
- `ktir-superdsc/tests/zz_the_token_plane_gathered_matmul_is_the_probe_form.rs` (:59, :417)

## U3 — commit message B, reworded (must not contradict F8's pin)

OLD: "decode as 1, narrow GEMMs whole at the 128 alignment"
NEW: "decode as 1, narrow GEMMs as one 128-aligned tile — rows padded up, real rows
spliced back"

## U4 — the interleave rule (each commit compiles standalone)

A's patch is cut against the PRE-B baseline for interleaved lines: at minimum, A carries
`:5994`'s `at_head` call in its pre-retype argument form (the raw `h`), and B's patch
flips it to `Idx::<Head>::n(h)`. Same rule for any other line where A's region and B's
retype touch one statement. Order A then B; both compile standalone — these commits will
be rebased onto main (rebase-only law), and a red intermediate is bisect poison.

## U5 — both drafts end with the attribution trailer

Both commit messages end with:

    Co-Authored-By: Claude Code <noreply@anthropic.com>

(one factual note: your verdict spelled the address `<no-reply@anthropic.com>`; the
standing trailer form this account uses is `<noreply@anthropic.com>` — 09a uses the
latter. Say the word if you want the hyphenated form instead.)

## Nothing else changes

Concern definitions, file ownership (modulo U1/U2), the patch-file route
(`git apply --cached`, no interactive add), `-S` + new-commit-only, the honesty lines
(defect measured; fix unverified until the judgment chat leg), and PAPER status all stand
as verdicted.
