# 05 — S1: the prefill mask add's test geometry, MINTED ONCE as a typed fact

Finding S1: the three door tests hand-spell the mask-add presentation five-fold
(672/21/64/32/3) — the operand mints, the broadcast fan-out, and the tile op's shape
arguments re-state the same geometry in every test. One typed mint, constructed through
production's doors, consumed by all three; the wire-literal asserts stay (they pin what
the card reads, not what the mint derived).

**File: `crates/targets/spyre/ktir/ktir-superdsc/src/emit/lower_ktir_to_superdsc.rs`**
(the SpraysFirst WIP file — I land it, per your ownership ruling). The mint is placed in
the test module, immediately before
`the_row_split_op_at_the_prefill_mask_geometry_splits_one_spray_per_core` (currently
:7504).

## HUNK 1 — the typed mint (NEW, inserted before test 1)

```rust
/// ⭐⭐⭐⭐⭐ THE PREFILL MASK ADD'S OWN GEOMETRY, MINTED ONCE AND DERIVED EVERYWHERE —
/// `[y=32, mb=21, out=64]`: 32 query heads, 21 tokens, one stick of slots, three operands.
/// The score tile's 672 rows are NOT a sixth number: they are `y · mb`, minted by
/// [`Self::score_rows`], and every test below spends the same fields — so the presentation
/// is spelled once, through the production counts ([`RowCount`], [`BlockCols`]), and no
/// test re-hand-spells a number the geometry already states. The wire-literal asserts in
/// the tests stay literal: they pin what the card reads, not what the mint derived.
struct PrefillMaskAdd {
    /// The spray count — the 32 query heads, the `y_broadcast` fan-out.
    y: u32,
    /// The token count — the 21-token prefill.
    mb: RowCount,
    /// One stick of output slots.
    out: BlockCols,
    /// The add's operand count: two inputs, one output.
    n_operands: u32,
}

impl PrefillMaskAdd {
    /// The score tile's rows: every spray of every token — `y · mb` = 672, the flat
    /// row-major row count the dense operand and the output both carry.
    fn score_rows(&self) -> RowCount {
        RowCount::of_token_rows(self.y * self.mb.get())
    }

    /// The tile op at this geometry — `pointwise_tile_op`'s arms fed from the same
    /// fields (`mb` from the rows, `out` from the cols, `y` stated by this mode).
    fn tile_op(&self, df: Df) -> TileOp {
        pointwise_tile_op(self.mb.get(), self.out.get(), self.y, self.n_operands, df)
    }

    /// The dense operand: the full `[score_rows, out]` score tile.
    fn dense(&self) -> In {
        In::full(&rb("t5", self.score_rows(), self.out)).ew()
    }

    /// The mask operand: the `[mb, out]` causal mask, broadcast over the sprays.
    fn mask(&self) -> In {
        In::y_broadcast(&rb("t6", self.mb, self.out), self.y).ew()
    }
}

/// The one mint — granite-3.1-2b prefill's own mask add.
fn prefill_mask_add() -> PrefillMaskAdd {
    PrefillMaskAdd {
        y: 32,
        mb: RowCount::of_token_rows(21),
        out: BlockCols::of_one_stick(Lanes::FP16),
        n_operands: 3,
    }
}
```

(`TileOp` is already in scope — `pointwise_tile_op` returns it; `Lanes`, `RowCount`,
`BlockCols`, `rb`, `In` are the imports the tests already use.)

## HUNK 2 — test 1 converted (doc comment unchanged; body only)

OLD (the whole body, :7513-7541):
```rust
        let dense_h = rb(
            "t5",
            RowCount::of_token_rows(672),
            BlockCols::of_one_stick(Lanes::FP16),
        );
        let mask_h = rb(
            "t6",
            RowCount::of_token_rows(21),
            BlockCols::of_one_stick(Lanes::FP16),
        );
        let dense = In::full(&dense_h).ew();
        let mask = In::y_broadcast(&mask_h, 32).ew();
        let op = crate::ir::bridge::tiled_op_sdsc_op::pointwise_broadcast_opspec_from_tile(
            &pointwise_tile_op(21, 64, 32, 3, Df::Fp16),
            672,
            64,
            op_func_from_str("add"),
            &[dense, mask],
            "t7",
            crate::addr::OutCorner::ZERO,
            false,
            None,
            crate::work::CoreDivision::Ew(EwDivision::Law),
        )
        .expect("the mask add emits under the spray split");
        assert_eq!(op.iter.extent("y"), 32, "y is the 32 query heads");
        assert_eq!(op.iter.extent("mb"), 21, "mb is the 21 tokens");
        assert_eq!(op.iter.split_of("y"), 32, "ONE SPRAY PER CORE");
        assert_eq!(op.iter.split_of("mb"), 1, "the whole token axis stays in one slab");
```
NEW:
```rust
        let g = prefill_mask_add();
        let op = crate::ir::bridge::tiled_op_sdsc_op::pointwise_broadcast_opspec_from_tile(
            &g.tile_op(Df::Fp16),
            g.score_rows().get(),
            g.out.get(),
            op_func_from_str("add"),
            &[g.dense(), g.mask()],
            "t7",
            crate::addr::OutCorner::ZERO,
            false,
            None,
            crate::work::CoreDivision::Ew(EwDivision::Law),
        )
        .expect("the mask add emits under the spray split");
        assert_eq!(op.iter.extent("y"), 32, "y is the 32 query heads");
        assert_eq!(op.iter.extent("mb"), 21, "mb is the 21 tokens");
        assert_eq!(op.iter.split_of("y"), 32, "ONE SPRAY PER CORE");
        assert_eq!(op.iter.split_of("mb"), 1, "the whole token axis stays in one slab");
```

## HUNK 3 — test 2 converted (construction block only, :7559-7589; all wire asserts
and the emitted-descriptor walk below them unchanged)

OLD:
```rust
        let dense_h = rb(
            "t5",
            RowCount::of_token_rows(672),
            BlockCols::of_one_stick(Lanes::FP16),
        );
        let mask_h = rb(
            "t6",
            RowCount::of_token_rows(21),
            BlockCols::of_one_stick(Lanes::FP16),
        );
        let out_h = rb(
            "t7",
            RowCount::of_token_rows(672),
            BlockCols::of_one_stick(Lanes::FP16),
        );
        let dense = In::full(&dense_h).ew();
        let mask = In::y_broadcast(&mask_h, 32).ew();
        let mut sid = 0i64;
        let emitted =
            crate::ir::bridge::tiled_op_sdsc_op::assemble_pointwise_broadcast_off_from_tile(
                "add_o1417",
                &pointwise_tile_op(21, 64, 32, 3, Df::Fp16),
                "add",
                672,
                64,
                &[dense, mask],
                &out_h,
                crate::addr::OutCorner::ZERO,
                &mut sid,
                None,
            );
```
NEW:
```rust
        let g = prefill_mask_add();
        let out_h = rb("t7", g.score_rows(), g.out);
        let mut sid = 0i64;
        let emitted =
            crate::ir::bridge::tiled_op_sdsc_op::assemble_pointwise_broadcast_off_from_tile(
                "add_o1417",
                &g.tile_op(Df::Fp16),
                "add",
                g.score_rows().get(),
                g.out.get(),
                &[g.dense(), g.mask()],
                &out_h,
                crate::addr::OutCorner::ZERO,
                &mut sid,
                None,
            );
```

## HUNK 4 — test 3 converted (construction block only, :7629-7651)

OLD:
```rust
        let dense_h = rb(
            "t5",
            RowCount::of_token_rows(672),
            BlockCols::of_one_stick(Lanes::FP16),
        );
        let mask_h = rb(
            "t6",
            RowCount::of_token_rows(21),
            BlockCols::of_one_stick(Lanes::FP16),
        );
        let dense = In::full(&dense_h).ew();
        let mask = In::y_broadcast(&mask_h, 32).ew();
        let Err(m) = crate::ir::bridge::tiled_op_sdsc_op::pointwise_broadcast_opspec_from_tile(
            &pointwise_tile_op(21, 64, 32, 3, Df::Fp16),
            672,
            64,
            op_func_from_str("add"),
            &[dense, mask],
            "t7",
            crate::addr::OutCorner::ZERO,
            false,
            None,
            crate::work::CoreDivision::Ew(EwDivision::RowsFirst),
        ) else {
```
NEW:
```rust
        let g = prefill_mask_add();
        let Err(m) = crate::ir::bridge::tiled_op_sdsc_op::pointwise_broadcast_opspec_from_tile(
            &g.tile_op(Df::Fp16),
            g.score_rows().get(),
            g.out.get(),
            op_func_from_str("add"),
            &[g.dense(), g.mask()],
            "t7",
            crate::addr::OutCorner::ZERO,
            false,
            None,
            crate::work::CoreDivision::Ew(EwDivision::RowsFirst),
        ) else {
```
(the `panic!`/`assert!` tail of test 3 is unchanged.)

## Scope notes

- The door's own `rows`/`cols` parameters stay `u32` — the tests cross the typed counts at
  the door call (`.get()`), the same boundary form as the fp8 door's `(m, k, n)`.
- The two nearby y-broadcast tests (`the_row_split_op_states_the_y_major_frame` :7436,
  `a_tile_that_disagrees_with_the_spray_count_is_refused` :7661) use the 256/64/4
  J-slot-merge geometry, NOT the mask-add numbers — out of S1's scope, untouched.
- Stacks: none (independent of the F-set). 08's lower_ktir citations stack ON this file's
  final state.
