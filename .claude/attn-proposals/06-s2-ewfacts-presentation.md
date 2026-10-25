# 06 — S2: `EwDivision::resolve`'s three positional bools become one `EwFacts` struct

Finding S2: `resolve(stick_major: bool, core_local: bool, sprays: bool)` is stringly at
the call site — three positional bools cannot state WHICH fact is which, and a swapped
pair still type-checks while silently picking the wrong division. One struct, minted
where the facts are known (the operand/tile), spent whole at the resolve; field names
are the law's own vocabulary.

## HUNK 1 — work.rs, `EwFacts` (NEW, placed between the `EwDivision` enum's close at
:631 and `impl EwDivision {` at :633) + `resolve` retyped

OLD (work.rs :633-655, the impl head and resolve complete):
```rust
impl EwDivision {
    /// The concrete division. [`Self::Law`] resolves on the operands: `stick_major` is the builder's own
    /// rank-2 stick-major presentation decision — the SAME value that picks the operands' layout, so the
    /// layout and the division cannot disagree — `core_local` is whether any operand is a core-local
    /// LX pin (such a tensor's division is its PRODUCER's, held by `emit_sdsc_tiled`'s read law and sized
    /// by [`crate::emit::pointwise_untiled_slice_bytes`], so an op touching one under the law keeps the
    /// rows-first division those were laid out under; a pin laid out under a stated division is sized
    /// under that division instead, [`crate::emit::pointwise_untiled_slice_bytes_divided`], and its ops
    /// state it), and `sprays` is the row-split (y-major) presentation decision, the same kind of fact:
    /// the op's tensors walk `["y", "mb", "out"]`, and only a split over the leading `y` gives a core a
    /// contiguous slab in that layout. The concrete divisions resolve to themselves.
    ///
    /// ⭐ `sprays` outranks `stick_major` by construction, not by choice: the row-split frame law holds
    /// the op to ONE stick of `out` while the stick-major presentation requires wider than one, so the
    /// two facts are disjoint and their order in the match cannot matter.
    pub const fn resolve(self, stick_major: bool, core_local: bool, sprays: bool) -> EwDivision {
        match self {
            EwDivision::Law if sprays => EwDivision::SpraysFirst,
            EwDivision::Law if stick_major && !core_local => EwDivision::SticksFirst,
            EwDivision::Law => EwDivision::RowsFirst,
            concrete => concrete,
        }
    }
```
NEW:
```rust
/// ⭐⭐⭐ THE THREE LAYOUT FACTS [`EwDivision::Law`] RESOLVES ON — minted as ONE struct
/// where the facts are known (the operand/tile, at the builder), spent whole at the
/// resolve: positional bools at a call site cannot state WHICH fact is which, and a
/// swapped pair still type-checks while silently picking the wrong division. The field
/// names are the law's own vocabulary.
#[derive(Clone, Copy, Debug)]
pub struct EwFacts {
    /// The builder's own rank-2 stick-major presentation decision — the SAME value that
    /// picks the operands' layout, so the layout and the division cannot disagree.
    pub stick_major: bool,
    /// Whether any operand is a core-local LX pin: such a tensor's division is its
    /// PRODUCER's, held by `emit_sdsc_tiled`'s read law and sized by
    /// [`crate::emit::pointwise_untiled_slice_bytes`], so an op touching one under the
    /// law keeps the rows-first division those were laid out under; a pin laid out under
    /// a stated division is sized under that division instead
    /// ([`crate::emit::pointwise_untiled_slice_bytes_divided`]), and its ops state it.
    pub core_local: bool,
    /// The row-split (y-major) presentation decision, the same kind of fact: the op's
    /// tensors walk `["y", "mb", "out"]`, and only a split over the leading `y` gives a
    /// core a contiguous slab in that layout.
    pub sprays: bool,
}

impl EwDivision {
    /// The concrete division. [`Self::Law`] resolves on the operands' [`EwFacts`] — minted
    /// where those facts are known, so the layout decision and the division decision read
    /// one spelling. The concrete divisions resolve to themselves.
    ///
    /// ⭐ `facts.sprays` outranks `facts.stick_major` by construction, not by choice: the
    /// row-split frame law holds the op to ONE stick of `out` while the stick-major
    /// presentation requires wider than one, so the two facts are disjoint and their order
    /// in the match cannot matter.
    pub const fn resolve(self, facts: EwFacts) -> EwDivision {
        match self {
            EwDivision::Law if facts.sprays => EwDivision::SpraysFirst,
            EwDivision::Law if facts.stick_major && !facts.core_local => EwDivision::SticksFirst,
            EwDivision::Law => EwDivision::RowsFirst,
            concrete => concrete,
        }
    }
```

## HUNK 2 — pointwise.rs call site 1 (:447-456) + import

Import (pointwise.rs :15):
OLD: `use crate::work::{CoreDivision, EwDivision};`
NEW: `use crate::work::{CoreDivision, EwDivision, EwFacts};`

OLD (:447-456):
```rust
    pointwise_opspec_from_tile_split(
        tile_op,
        op,
        in_names,
        o_name,
        head_major,
        EwDivision::Law.resolve(stick_major, core_local, false).splitter(),
        // Whole-tensor output — every caller but the gather copy, whose runs land at a base.
        0,
    )
```
NEW:
```rust
    pointwise_opspec_from_tile_split(
        tile_op,
        op,
        in_names,
        o_name,
        head_major,
        EwDivision::Law
            .resolve(EwFacts {
                stick_major,
                core_local,
                // ⭐ `false` for the spray fact: this builder takes no `EwOperand`, so no
                // operand can state the row-split (y-broadcast) mode — the only
                // presentation whose division the fact selects.
                sprays: false,
            })
            .splitter(),
        // Whole-tensor output — every caller but the gather copy, whose runs land at a base.
        0,
    )
```

## HUNK 3 — pointwise.rs call site 2 (:697)

OLD:
```rust
                d.resolve(stickmajor, core_local, ymajor).splitter(),
```
NEW:
```rust
                d.resolve(EwFacts {
                    stick_major: stickmajor,
                    core_local,
                    sprays: ymajor,
                })
                .splitter(),
```

## HUNK 4 — emit/mod.rs call site 3 (:963-970; fully-qualified, no import change)

OLD:
```rust
    tile_op
        .tile(
            crate::superdsc_opspec::MaxCores::<{ crate::superdsc_opspec::MAX_CORES }>,
            // `false` for the spray fact: this sizing helper's tile states `y=1`, so the row-split
            // (y-major) mode — the only presentation whose division that fact selects — cannot
            // reach it.
            division.resolve(true, true, false).splitter(),
            "out",
        )
```
NEW:
```rust
    tile_op
        .tile(
            crate::superdsc_opspec::MaxCores::<{ crate::superdsc_opspec::MAX_CORES }>,
            division
                .resolve(crate::work::EwFacts {
                    // `true`/`true`: the pinned tensor is core-local by definition and its
                    // op is stick-major; `false` for the spray fact: this sizing helper's
                    // tile states `y=1`, so the row-split (y-major) mode — the only
                    // presentation whose division that fact selects — cannot reach it.
                    stick_major: true,
                    core_local: true,
                    sprays: false,
                })
                .splitter(),
            "out",
        )
```

## HUNK 5 — zz_a's resolve test converted to struct constructions

Import (zz_a :20-23):
OLD:
```rust
use ktir_superdsc::work::{
    EwDivision, core_to_wk_slice, distribute_cores, distribute_cores_sticks_first,
};
```
NEW:
```rust
use ktir_superdsc::work::{
    EwDivision, EwFacts, core_to_wk_slice, distribute_cores, distribute_cores_sticks_first,
};
```

`the_law_resolves_on_the_layout_and_the_pins` (zz_a :168-208), body REPLACED (doc
unchanged; 07's two new assertions stack after the last one):
```rust
#[test]
fn the_law_resolves_on_the_layout_and_the_pins() {
    assert_eq!(
        EwDivision::Law.resolve(EwFacts {
            stick_major: true,
            core_local: false,
            sprays: false
        }),
        EwDivision::SticksFirst
    );
    assert_eq!(
        EwDivision::Law.resolve(EwFacts {
            stick_major: true,
            core_local: true,
            sprays: false
        }),
        EwDivision::RowsFirst
    );
    assert_eq!(
        EwDivision::Law.resolve(EwFacts {
            stick_major: false,
            core_local: false,
            sprays: false
        }),
        EwDivision::RowsFirst
    );
    assert_eq!(
        EwDivision::RowsFirst.resolve(EwFacts {
            stick_major: true,
            core_local: false,
            sprays: false
        }),
        EwDivision::RowsFirst
    );
    assert_eq!(
        EwDivision::SticksFirst.resolve(EwFacts {
            stick_major: false,
            core_local: true,
            sprays: false
        }),
        EwDivision::SticksFirst
    );
    // ⭐ THE ROW-SPLIT FACT OUTRANKS THE OTHERS: a y-major op's tensors walk ["y", "mb", "out"],
    // so only a split over the leading `y` gives a core a contiguous slab — the prefill mask add
    // measured 620 of 672 sticks never written under the rows-first split (see
    // `distribute_cores_sprays_first`'s doc).
    assert_eq!(
        EwDivision::Law.resolve(EwFacts {
            stick_major: false,
            core_local: false,
            sprays: true
        }),
        EwDivision::SpraysFirst
    );
    assert_eq!(
        EwDivision::Law.resolve(EwFacts {
            stick_major: true,
            core_local: false,
            sprays: true
        }),
        EwDivision::SpraysFirst
    );
    assert_eq!(
        EwDivision::SpraysFirst.resolve(EwFacts {
            stick_major: false,
            core_local: false,
            sprays: true
        }),
        EwDivision::SpraysFirst
    );
}
```

## Inventory

grep over `crates/targets/spyre/ktir/ktir-superdsc/src` finds exactly three
`EwDivision::resolve` call sites (pointwise.rs :453/:697, emit/mod.rs :969) — all three
converted above; the only other `resolve(` in src/ is `ActiveCap::resolve`
(ktir_node.rs:357), an unrelated method. zz_a holds the only test-side calls. No others.
