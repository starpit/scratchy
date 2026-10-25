# 08 — S4: every measured number in the SpraysFirst docs cites its saved artifact

Finding S4: the doc comments carry measured numbers (620, sticks 0..51, 128 B, 310, 362,
`2t mod 21`) with NO citation of the saved artifacts — and the artifact paths were not
even at the bench_results root. VERIFIED CARRIERS (read this session, all under
`bench_results/attn-compile-restart/2026-10-24/`):

- the 620/0..51/128 B pin: `leg6i-mask-add-forensics.py` → `leg6j-stick-map.py`
  (sticks 0..51, `t=20` stutter) → `leg6k-add-descriptor.py` (the per-core
  `startAddressCoreCorelet_` arrays — the 128 B advance) → `leg6l-after-spray-split.py`
  (recounts the 620 pin in its docstring; verifies the spray split: all 672 sticks
  written, causal cut kept = t+1).
- the 310 pin: `leg6o-after-wireorder.py` — recounts leg6m's pin verbatim: "362/672
  sticks written with the wrong mask row (t' = 2t mod 21), 310 never written" — and
  verifies the wire-order fix.
- the end-to-end symptom: `run-45-judgment-chat.log` (the diverged judgment chat that
  motivated the forensics).
- the legs' report OUTPUTS are synced beside them in `pod-leg4-reports/` (e.g.
  `pod-leg4-reports/leg6l_report.txt`).

## HUNK 1 — work.rs, `distribute_cores_sprays_first`'s doc (:1041-1070)

Appended as the doc's last paragraph (after "…divides `{y:4, mb:8}` over 32 cores).",
before the `pub fn`):
```rust
///
/// Measured numbers' saved artifacts (`bench_results/attn-compile-restart/2026-10-24/`):
/// the 620/0..51/128 B pin is `leg6k-add-descriptor.py` (the per-core start arrays) with
/// `leg6j-stick-map.py` (the stick map) and `leg6i-mask-add-forensics.py` (the
/// forensics); `leg6l-after-spray-split.py` recounts the pin and verifies this splitter
/// (all 672 sticks written, causal cut kept = t+1). The diverged judgment chat that
/// motivated the forensics is `run-45-judgment-chat.log`; the legs' report outputs are
/// synced in `pod-leg4-reports/`.
```

## HUNK 2 — pointwise.rs, the door's y-major division comment (:654-678)

Appended after "…the refusal names it rather than emitting the strided slab." and before
the `if ymajor`:
```rust
    // Measured numbers' saved artifacts (`bench_results/attn-compile-restart/2026-10-24/`):
    // `leg6k-add-descriptor.py` + `leg6j-stick-map.py` hold the 620/0..51 pin;
    // `leg6l-after-spray-split.py` recounts it and verifies the spray split.
```

## HUNK 3 — pointwise.rs, the wire-order comment (:794-809)

Appended after "…the per-core starts (one 21-stick spray per core) are untouched." (the
comment's last line, at its 16-space indentation):
```rust
                // Measured numbers' saved artifact
                // (`bench_results/attn-compile-restart/2026-10-24/`):
                // `leg6o-after-wireorder.py` — recounts leg6m's pin (362/672 sticks
                // written with the wrong mask row, t' = 2t mod 21; 310 never written)
                // and verifies this wire order.
```

## HUNK 4 — lower_ktir's two test docs (stacks on 05's file state)

Test 1's doc (`the_row_split_op_at_the_prefill_mask_geometry_…`), appended after "…
the same bytes at the same bases.":
```rust
    /// Measured numbers' saved artifacts (`bench_results/attn-compile-restart/2026-10-24/`):
    /// `leg6k-add-descriptor.py` + `leg6j-stick-map.py` (the 620 pin),
    /// `leg6l-after-spray-split.py` (this splitter's verification).
```
Test 2's doc (`the_row_split_op_serializes_stick_innermost_and_reconstructs`), appended
after "…the per-core starts (one 21-stick spray per core) are untouched.":
```rust
    /// Measured numbers' saved artifact (`bench_results/attn-compile-restart/2026-10-24/`):
    /// `leg6o-after-wireorder.py` — the 310/`2t mod 21` pin and this wire order's
    /// verification.
```

## Extend-or-leave (your call)

The #549 sweep numbers (249.8/121.8 µs, 50.4/103.3 GB/s, `[256,2048]` 50.0→30.3 µs)
appear in work.rs's `EwDivision` enum doc (:606-609) and zz_a's module doc (:7-9) — both
already name "#549's sweep" (an issue citation, not a file path). If you want file paths
there too, say which artifact holds the sweep's numbers and I file the one-liners; I did
not find a saved sweep artifact in this worktree's bench_results and will not guess a
path. Issue-number citations stand as-is unless you rule otherwise.
