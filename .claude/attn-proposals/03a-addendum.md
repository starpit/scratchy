# 03a — addendum to 03 (Q1/Q3/Q4 per your verdict; Q2 resolved, Q5 noted). Lands WITH 03.

## Q1 — A3's exact code, COMPILING forms (arithmetic in u32 inside the single mint,
## typed at both ends; no new traits on the counts)

`tile()` REPLACED:
```rust
    /// Tile `t`'s placement facts — the ONE site that computes the tile's row window, so no
    /// clone/quant/mm/out site re-derives it.
    pub fn tile(&self, t: Fp8GemmTileIdx) -> Fp8GemmRowTile {
        let start = t.get() * self.rows.get();
        Fp8GemmRowTile {
            idx: t,
            row: crate::addr::Idx::<crate::addr::Row>::n(start),
            real_rows: crate::sdsc_abstract::RowCount::of_token_rows(
                self.rows.get().min(self.m.get().saturating_sub(start)),
            ),
            staged_rows: crate::sdsc_abstract::RowCount::of_token_rows(self.rows.get()),
        }
    }
```

`tiles()` REPLACED (Range over u32, not over the newtype):
```rust
    /// The plan's tiles, as minted values — the emission's ONLY tile loop: no site counts
    /// tile indices, the iterator hands each tile its own facts.
    pub fn tiles(&self) -> impl Iterator<Item = Fp8GemmRowTile> + '_ {
        (0..self.tiles.get())
            .map(Fp8GemmTileIdx::new)
            .map(|t| self.tile(t))
    }
```

`rows_run_whole()` REPLACED (typed comparison; NO `PartialEq<u32>` added):
```rust
    /// Whether the rows run WHOLE at the caller's own count — the dense arm's own test,
    /// minted here as a named fact so the door does not re-derive tiling arithmetic. Exactly
    /// `tile_count() == 1 ∧ total_rows() == m` (the CONFIRMED GOOD dense-arm byte-identity
    /// test, same predicate, one spelling).
    pub fn rows_run_whole(&self) -> bool {
        self.tiles == Fp8GemmTileCount::new(1) && self.rows == self.m
    }
```
(`RowCount`/`Fp8GemmTileCount` both derive `PartialEq`; `self.rows == self.m` is
RowCount==RowCount.) `k_stick_group_windows`/`n_stick_group_windows` stand as filed — they
already mint through `.get()` at the u32-shaped boundary you approved.

## Q3 — B1's caller inventory, CORRECTED (you verified the hole; every site below read)

**Source callers (all):**
- `matmul/opspec.rs:36` — as filed.
- `ktir_matmul_fp8.rs` ×4 (:704/:850/:1004/:1109 in the WIP tree) — as filed.
- **`lower_ktir_to_superdsc.rs:4842`** (the RoPE collapse — MISSED in 03):
  OLD:
  ```rust
            crate::sdsc_abstract::MatM::of_token_rows(heads * mq),
  ```
  NEW (head-major rows — that IS the RoPE collapse's sweep, per
  `RowCount::of_head_major_rows`'s own doc at sdsc_abstract.rs:4935):
  ```rust
            crate::sdsc_abstract::MatM::of_token_rows(
                crate::sdsc_abstract::RowCount::of_head_major_rows(heads * mq),
            ),
  ```
- **`lower_ktir_to_superdsc.rs:6734`** (MISSED in 03; the same site already mints the count
  one line below for `rb`). Hoist the one mint:
  OLD:
  ```rust
        assemble_matmul_windowed(
            &op_name,
            MatM::of_token_rows(m),
            MatN::of_out_features(n_dev),
            MatK::of_in_features(k),
            &rb(
                &a.name(),
                RowCount::of_token_rows(m),
                BlockCols::of_feature_cols(k),
  ```
  NEW:
  ```rust
        let m_rows = RowCount::of_token_rows(m);
        assemble_matmul_windowed(
            &op_name,
            MatM::of_token_rows(m_rows),
            MatN::of_out_features(n_dev),
            MatK::of_in_features(k),
            &rb(
                &a.name(),
                m_rows,
                BlockCols::of_feature_cols(k),
  ```

**Test-file callers (break with the retype; 04's scope now explicitly includes these
seven — every one is a token-row count, so every mint is `RowCount::of_token_rows`):**

- `crates/targets/spyre/tests/lm_head_trip_cost.rs:21`:
  `MatM::of_token_rows(m)` → `MatM::of_token_rows(RowCount::of_token_rows(m))`
- `crates/targets/spyre/tests/sdsc_fp8_stick_split.rs:29` (the `fn mm(m: u32) -> MatM`
  helper's body):
  `MatM::of_token_rows(m)` → `MatM::of_token_rows(RowCount::of_token_rows(m))`
- `crates/targets/spyre/tests/chunk_explore_is_the_groups.rs:57`:
  `MatM::of_token_rows(96)` → `MatM::of_token_rows(RowCount::of_token_rows(96))`
- `crates/targets/spyre/tests/split_leaves_whole_operand_sticks.rs:41`:
  `MatM::of_token_rows(m)` → `MatM::of_token_rows(RowCount::of_token_rows(m))`
- `ktir-superdsc/tests/zz_a_one_row_matmul_fills_every_core.rs:89`:
  `MatM::of_token_rows(1)` → `MatM::of_token_rows(RowCount::of_token_rows(1))`
- `ktir-superdsc/tests/zz_the_index_tensors_shape_is_the_paged_dims.rs:222`:
  `MatM::of_token_rows(8)` → `MatM::of_token_rows(RowCount::of_token_rows(8))`
- `ktir-superdsc/tests/zz_the_token_plane_gathered_matmul_is_the_probe_form.rs:59` and
  `:417`:
  `MatM::of_token_rows(96)` → `MatM::of_token_rows(RowCount::of_token_rows(96))`;
  `MatM::of_token_rows(2)` → `MatM::of_token_rows(RowCount::of_token_rows(2))`

Each file gains the `RowCount` import it lacks (they already import from
`ktir_superdsc::sdsc_abstract` or `scratchy_subtile::sdsc_abstract`; the mint rides that
path — `sdsc_abstract::RowCount` at the four spyre-test sites).

## Q4 — Fp8Facts' `a_tid`/`out_tid` u32 seam, NAMED in the honest inventory

Appended to 03's honest .get()/seam inventory:
> **Fp8Facts' `a_tid`/`out_tid` are `u32` and feed `crate::place::act_name(_)` and
> `PlaceId::Act(_)` — a NAMED known seam.** Stated boundary for this diff: `act_name`'s
> retype is a crate-wide ripple not priced here. Follow-up proposal is my call (folded
> into the MatN/MatK family proposal's scope decision, below).

## Notes acknowledged

- `MatM::of_head_major_rows(rows: u32)` at sdsc_abstract.rs:1051 is a PRE-EXISTING raw
  door, untouched by this diff — named in the inventory as a known pre-existing seam; its
  fix belongs to a future family pass.
- **MatN/MatK family retype: filing as its OWN proposal (10-matnmk-family.md) with the
  full caller inventory** — per your ruling "TAKE IT, own hunk set, not inside 03".
- Q5: `pairs: u32` → `Pairs` newtype stays a follow-up note, not this diff.
