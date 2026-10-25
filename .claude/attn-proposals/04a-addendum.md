# 04a — addendum to 04 (T1-T7 per your verdict). Lands WITH 04, after 02a's P1 branch
ruling (T5 rides it).

## T1 — `fp8_layout` retreats behind the geometry: `G::layout()` is the only door

`fp8_layout` (and its `place` closure) MOVES into a nested `mod fixture` inside the tests
module, private to it; the geometry types gain the layout doors as inherent impls in that
module (same crate — legal), `pub` so they are visible wherever the types are:

```rust
    // ⭐ THE LAYOUT DOOR IS THE GEOMETRY'S OWN — `fp8_layout` lives behind `fixture`,
    // private to it, so a hand-spelled triple cannot reach the helper AT ALL: the only
    // way a test states a shape to the layout is `G::layout()`.
    mod fixture {
        use super::*;

        impl<const M: u32, const K: u32, const N: u32> crate::work::Fp8GemmGeometry<M, K, N> {
            /// The fixture layout at this geometry — the ONLY door to the placer.
            pub fn layout() -> BundleLayout {
                fp8_layout(M, K, N)
            }
        }

        impl<const M: u32, const P: u32, const INN: u32, const N: u32>
            crate::work::Fp8PairTailGeometry<M, P, INN, N>
        {
            /// The fixture layout at this pair geometry — K is the DERIVED `PAIRS · INN`.
            pub fn layout() -> BundleLayout {
                fp8_layout(M, Self::FEATURES, N)
            }
        }

        fn fp8_layout(m: u32, k: u32, n: u32) -> BundleLayout {
            // body verbatim, tid literals → the T3 consts (below, hoisted to the fixture)
        }
    }
```

Every `let l = fp8_layout(G::ROWS, G::FEATURES, G::OUT);` in 04's tests becomes
`let l = G::layout();` (door tests, direct-chain tests, pair-tail tests, F6e, F8 — every
site in the conversion table too). The refusal fixture's layout comes from a named legal
geometry (`NarrowSmallMHidden::layout()`), exactly per your ruling; `NON_WHOLE_LANE_K`
stays the CHAIN INPUT only, as filed.

## T2 — the two 12b aliases in 04's block rename per 01a's R2

04's alias block lines (87/89) and the LX-overflow test's `use` (174) become
`LxOverflowControl` / `FusedOverflowControl` with 01a's docs. One home: work.rs carries
the definitions (01a), 04's block referenced them — with T1's `G::layout()` there is no
second alias block at all; 04's tests `use crate::work::LxOverflowControl as G;` etc.

## T3 — the fixture tid map, ONE home

Hoisted beside the fixture (read by `fp8_layout`'s five place calls, both facts doors,
and every assert site):
```rust
        // ⭐ THE FIXTURE'S TID MAP — ONE home: the layout places these tids and every
        // facts door and assert site READS them. A tid spelled twice is a map that can
        // disagree with itself.
        const A_TID: u32 = 7;
        const W_TID: u32 = 8;
        const WS_TID: u32 = 9;
        const OUT_TID: u32 = 10;
        const OUT2_TID: u32 = 11;
```
- `fp8_layout`'s five `place(...)` calls (7/8/9/10/11) read `A_TID`/`W_TID`/`WS_TID`/
  `OUT_TID`/`OUT2_TID`.
- `facts_out`/`facts`:
```rust
    fn facts_out(out_tid: u32) -> Fp8Facts {
        Fp8Facts {
            a_tid: A_TID,
            a_name: act_name(A_TID),
            w_name: act_name(W_TID),
            ws_name: act_name(WS_TID),
            out_tid,
        }
    }

    fn facts() -> Fp8Facts {
        facts_out(OUT_TID)
    }
```
- Assert sites: `PlaceId::Act(7)` → `PlaceId::Act(A_TID)` (wide test, both spots);
  `facts_out(11)` → `facts_out(OUT2_TID)` (sharing test); `PlaceId::Act(10)` and
  `placements.get(&10)` in F8 → `PlaceId::Act(OUT_TID)` / `placements.get(&OUT_TID)`.

## T4 — F8's mixed comment fixed

The line `/// 128-row extent (pad rows and all), and the OUTPUT keeps the geometry's own
[m, n]` becomes `//` (statement-level comment; kills the `unused_doc_comments` CI break).

## T5 — AMENDED per your 02a ruling (branch 1): the REFUSAL PIN

F8 stays at extent-128 with the census cited in its doc. The >128 unaligned case is a
REFUSAL PIN — the real 192-rung shape (the ONE unaligned prefill rung above 128, per the
census), as a named geometry in work.rs's test set:
```rust
/// The 192-rung narrow geometry — the ONE unaligned prefill rung above 128 (census in
/// `row_class`'s doc): it pads to 256, past the narrow arm's extent-128 law, and the
/// class door refuses it. No in-scope GEMM is confirmed narrow; the 26b candidates ride
/// open lowerings.
pub type NarrowHighUnaligned = Fp8GemmGeometry<192, 128, 128>;
```
and the pin, at the message's own door (the plan pads; the CLASS refuses):
```rust
    #[test]
    fn a_narrow_geometry_above_the_128_extent_refuses_to_emit() {
        let plan = crate::work::NarrowHighUnaligned::plan()
            .expect("the plan pads 192 up to 256 — the door, not the plan, refuses");
        let Err(m) = plan.row_class() else {
            panic!("192 pads to 256, above the narrow arm's extent-128 law");
        };
        assert!(
            m.contains("not a tiled ABI class")
                && m.contains("narrow padded extents above 128"),
            "names the class law and the census: {m}"
        );
    }
```
The padded-emission form originally sketched for >128 is dead under branch 1. The
emission door (`matmul_fp8_descriptors`) inherits the refusal through `row_class()` —
the work-level pin is the message's home.

## T6 — the counts gain `usize()`; `k_pad()`'s u32 exit documented

On all three work.rs newtypes (`Fp8GemmTileCount`, `Fp8GemmTileIdx`,
`Fp8GemmStickGroups`), beside `get()`:
```rust
    /// The count as a `usize` — the ONE unit exit (collection indexing); [`Self::get`]
    /// stays the u32 exit for footprint boundaries.
    pub const fn usize(self) -> usize {
        self.0 as usize
    }
```
04's casts become `.usize()`: `plan.tile_count().get() as usize` →
`plan.tile_count().usize()` (wide + both sharing tests);
`plan.k_stick_groups().get() as usize` / `n_stick_groups().get() as usize` → `.usize()`.
`k_pad()`'s doc (both geometry types):
```rust
    /// The tail tensors' padded width — the chain's `k_pad` mint. The u32 return is the
    /// STATED boundary exit (the chain's seam is u32-shaped); `k_pad_cols()` is the typed
    /// form.
```

## T7 — real vs staged, pinned at the plan level (wide test)

Added after the `total_rows` assert:
```rust
        // ⛔ REAL vs STAGED, pinned at the PLAN — defect 3 is exactly the confusion of the
        // two, so the plan's own facts get asserts, not just op-count shadows.
        let whole = plan.tile(crate::work::Fp8GemmTileIdx::new(0));
        assert_eq!(
            whole.real_rows.get(),
            crate::work::FP8_GEMM_WIDE_M,
            "a whole tile's real rows equal its staged rows — no tear"
        );
        assert_eq!(whole.staged_rows.get(), crate::work::FP8_GEMM_WIDE_M);
        let last = plan.tiles().last().expect("the wide plan has tiles");
        assert_eq!(
            last.real_rows.get(),
            1,
            "the last tile covers only the one real row — never past the footprint"
        );
```

## Your NOTES, folded

- F8 rows = 21 CONFIRMED (standing ruling) — the F8 NOTE paragraph in 04 drops.
- `Fp8PairTailGeometry`'s `N = 128` default gets its doc: pair tests don't exercise n;
  the default keeps `layout()`'s third argument honest.
- The PAIRS ≥ 1 invariant, stated on the pair type:
```rust
/// ⚠ `PAIRS ≥ 1` is an invariant of this type: `cl_reach()` computes `P − 1` and
/// underflows at 0. No alias instantiates `P = 0`; a future one may not.
```
- The literal `1` for pairs at direct-chain sites stays (Q5 follow-up note);
  `&mut 0` op-id seeds stay.
- The door's `(m, k, n)` u32 params are listed in the honest inventory as the named
  boundary (per your F6c note) — they join 03a's Q4 inventory lines.
