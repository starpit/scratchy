# 07 — S3: the outranking corners, PINNED (stacks on 06's zz_a conversion)

Finding S3: two corners of `resolve` are unpinned. (1) `sprays` outranks `core_local`
too, not just `stick_major` — `Law.resolve(true, true, true)` = SpraysFirst is nowhere in
the suite; the first match arm fires on `sprays` alone, and the pin proves the arm ORDER
is not load-bearing. (2) A STATED concrete division is its own answer even under the
spray fact — `RowsFirst.resolve(false, false, true)` = RowsFirst, with the comment
naming that the DOOR (not resolve) refuses a y-major op stated rows-first.

## HUNK — zz_a, appended inside `the_law_resolves_on_the_layout_and_the_pins`

(after the last `SpraysFirst.resolve(...)` assertion, before the closing `}` — the test
in its 06-converted form):

```rust
    // ⭐ S3 — THE OUTRANKING CORNER, PINNED: `sprays` outranks `core_local` too, not just
    // `stick_major` — the first match arm fires on `sprays` alone, so a stick-major op
    // touching a core-local pin in the row-split mode still goes sprays-first. The two
    // layout facts are disjoint by construction (see `EwFacts`'s docs); this pins that
    // the arm ORDER is not load-bearing.
    assert_eq!(
        EwDivision::Law.resolve(EwFacts {
            stick_major: true,
            core_local: true,
            sprays: true
        }),
        EwDivision::SpraysFirst
    );
    // ⭐ S3 — a STATED concrete division is its own answer even under the spray fact:
    // `resolve` does not refuse — the DOOR does. A y-major op stated rows-first is
    // refused at `pointwise_broadcast_opspec_from_tile`, pinned by the lower_ktir
    // refusal test `a_row_split_op_with_a_stated_inner_dim_split_is_refused`.
    assert_eq!(
        EwDivision::RowsFirst.resolve(EwFacts {
            stick_major: false,
            core_local: false,
            sprays: true
        }),
        EwDivision::RowsFirst
    );
```

One file, one hunk; rides 06's import (`EwFacts` already imported there).
