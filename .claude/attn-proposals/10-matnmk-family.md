# 10 — the MatN/MatK family retype: `of_out_features`/`of_in_features` take `BlockCols`

Per your ruling ("TAKE IT as its own hunk set with its caller inventory"). The family is
ALREADY mostly typed — `MatN::of_kv_window/of_chunk_decode_scores/of_new_block_scores
(BlockCols)`, `one_stick(Lanes)`, `of_head_slab(SlabFeats)` — exactly TWO raw doors remain,
and one of them is hiding an existing typed door at its biggest caller (below). This set
closes both. **Stacks on 03+04's final state** (the fp8 sites read 03's minted
`BlockCols` locals).

## HUNK 1 — sdsc_abstract.rs, the two doors retyped

OLD (MatN):
```rust
    /// A projection's output features — N of `A[m,k]·W[k,n]`.
    pub const fn of_out_features(n: u32) -> MatN {
        MatN(n, OutputSplit::Planned)
    }
```
NEW:
```rust
    /// A projection's output features — N of `A[m,k]·W[k,n]`. Takes the door's own typed
    /// width, so a row count, a lane count or a stick-group count cannot fill this slot.
    pub const fn of_out_features(n: BlockCols) -> MatN {
        MatN(n.get(), OutputSplit::Planned)
    }
```
OLD (MatK):
```rust
    /// A projection's input features — K of `A[m,k]·W[k,n]` (or one K-split block of it).
    pub const fn of_in_features(k: u32) -> MatK {
        MatK(k, ContractionSplit::Planned)
    }
```
NEW:
```rust
    /// A projection's input features — K of `A[m,k]·W[k,n]` (or one K-split block of it).
    /// Takes the door's own typed width.
    pub const fn of_in_features(k: BlockCols) -> MatK {
        MatK(k.get(), ContractionSplit::Planned)
    }
```

## HUNK 2 — the caller inventory, COMPLETE (grep-verified: 34 sites, every one listed)

**Source (16):**
- `matmul/opspec.rs:37-38` and `:816-817` — the two "untyped wrapper boundary" fns (their
  own docs name the boundary; their `n`/`k` params stay u32, the KTIR facts). Mint at the
  call, both sites the same form:
  ```rust
        MatN::of_out_features(BlockCols::of_feature_cols(n)),
        MatK::of_in_features(BlockCols::of_feature_cols(k)),
  ```
- `emit/batched_gathered.rs:270-271, :687-688, :775-776` — SIX sites, all
  `MatN::of_out_features(STK)` / `MatK::of_in_features(STK)` where
  `const STK: u32 = FP16_ELEMS_PER_STICK` (batched_gathered.rs:75). **STK is one fp16
  stick, and the family already owns that door**: these become
  `MatN::one_stick(Lanes::FP16)` / `MatK::one_stick(Lanes::FP16)` — the raw door was
  hiding the typed one. (`Lanes` joins the file's sdsc_abstract import.)
- `emit/ktir_matmul_fp8.rs:705/:711` (dense door) and `:851/:852` (fp8_tile_mm_op) — in
  03's final form these read the minted `BlockCols` locals directly:
  `MatN::of_out_features(n_cols)` / `MatK::of_in_features(k_pad_cols)` (dense) and
  `MatN::of_out_features(n_cols)` / `MatK::of_in_features(k_pad_cols)` (tile mm) — no
  new mint, the 03 locals feed the door whole.
- `emit/lower_ktir_to_superdsc.rs:6735-6736` — composes with 03a's Q3 hoist at the same
  site: the hoisted `let k_cols = BlockCols::of_feature_cols(k);` (already minted there
  for `rb`) feeds `MatK::of_in_features(k_cols)`, and
  `MatN::of_out_features(BlockCols::of_feature_cols(n_dev.get()))` — the `n_dev.get()`
  is a NAMED boundary in the inventory below (DeviceWidth has no BlockCols door yet).

**Tests (18):** every one mints at the call
(`MatN::of_out_features(BlockCols::of_feature_cols(n))` etc.):
- `crates/targets/spyre/tests/lm_head_trip_cost.rs:22-23`
- `crates/targets/spyre/tests/sdsc_fp8_stick_split.rs:32,35`
- `crates/targets/spyre/tests/chunk_explore_is_the_groups.rs:58-59`
- `crates/targets/spyre/tests/split_leaves_whole_operand_sticks.rs:42-43`
- `ktir-superdsc/tests/zz_a_one_row_matmul_fills_every_core.rs:90-91`
- `ktir-superdsc/tests/zz_the_index_tensors_shape_is_the_paged_dims.rs:223-224`
- `ktir-superdsc/tests/zz_the_token_plane_gathered_matmul_is_the_probe_form.rs:60-61, :418-419`
Each file already gains the `RowCount` import via 03a's Q3 set; `BlockCols` rides the same
import lines.

## Named seams this set does NOT close (inventory, per your standing form)

- **`MatN::of_head_dim(hd: u32)` / `MatK::of_head_dim(hd: u32)`** — 66 callers
  (attn.rs ×33, lower_ktir ×22, sdsc_abstract internal ×7, tests ×8). NOT taken: the
  head-dim door needs its own typed count (a `HeadDim` vocabulary or a `BlockCols`
  ruling across 60 sites in the SpraysFirst WIP heartland) — offered as its own
  follow-up set if you want it; I did not invent the vocabulary unasked.
- **`MatM::of_head_major_rows(rows: u32)`** (sdsc_abstract.rs:1051) — pre-existing raw
  door, named per your 03-verdict note; future family pass.
- **`n_dev.get()` at lower_ktir:6735** — DeviceWidth→BlockCols has no door; if you want
  one (`BlockCols::of_device_width`), say so and I file it as a rider.
