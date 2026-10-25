# 01a — addendum to 01 (R1/R2/R3/R4 per your verdict). Lands WITH 01, as one application.

## R1 — HUNK 2's door comparison, typed (your fix, verbatim intent)

In `matmul_fp8_descriptors` (01's HUNK 2):
OLD:
```rust
    if plan.tile_count().get() == 1 && plan.total_rows() == RowCount::of_token_rows(m) {
```
NEW:
```rust
    if plan.tile_count() == crate::work::Fp8GemmTileCount::new(1)
        && plan.total_rows() == RowCount::of_token_rows(m)
    {
```
No `PartialEq<u32>` on the newtype; the `total_rows()` half stays as proposed.

## R2 — the two 12b aliases, renamed per your rule (I did the config verification)

VERIFIED FACTS (my sweep over `crates/models/arch/configs/`):
- **No config yields k=16384 as a GEMM width.** The only 16384s in the zoo are
  `max_position_embeddings` (phi-4) and a 163840 vocab (moonlight-16b) — never a
  projection width. `Gemma12bLxOverflow`'s <1024, 16384, 128> is NOT a model fact ⇒ CONTROL.
- **The 15360/3840 widths ARE a real config's**: gemma-3-12b-it.json `text_config`
  `intermediate_size = 15360`, `hidden_size = 3840` (the down projection). The rung-71
  shape <71, 15360, 3840> is that model's down-proj widths at the prime rung.
- ⚠ DISCREPANCY TO REPORT: the tree ALSO contains
  `crates/models/arch/configs/gemma4/gemma-4-12b-it.json` (text_config hidden 3840 /
  intermediate 15360 — identical widths), so "no gemma-4-12b" does not match the zoo as
  it stands. Both configs yield the same geometry, so the alias's truth is unaffected;
  I name the controls per your offered option and flag the file for your call.

HUNK 1b's two aliases REPLACED (the other fourteen unchanged):
```rust
/// The amax-reduce LX-overflow CONTROL — `m=1024, k=16384, n=128`: the reduce-residency
/// law's own stress shape. NOT a model fact: no config in `crates/models/arch/configs`
/// yields 16384 as a GEMM width (verified by sweep — the 16384s there are
/// `max_position_embeddings`/vocab, never a projection). Original test:
/// `the_quant_reduce_at_an_lx_overflowing_geometry_stays_single_shot`.
pub type LxOverflowControl = Fp8GemmGeometry<1024, 16384, 128>;
/// The rung-71 partials+combine CONTROL — `m=71, k=15360, n=3840`. The widths are a real
/// model's down projection (gemma-3-12b-it.json `text_config`: intermediate_size 15360 →
/// hidden_size 3840), but the prime-rung wall it pins is the reduce-residency law's own
/// stress shape, so it names the control, not the model. Original test:
/// `the_quant_reduce_at_a_fused_overflowing_geometry_emits_partials`.
pub type FusedOverflowControl = Fp8GemmGeometry<71, 15360, 3840>;
```
If you prefer the rung-71 alias model-named instead, the verified form is
`Gemma312bDownRung71 = Fp8GemmGeometry<71, 15360, 3840>` with the same config citation —
say the word and I swap the name at application.

## R3 — const fn: NO CHANGE NEEDED (verified)

`RowCount::of_token_rows` is already `pub const fn` (sdsc_abstract.rs:4924) — the `const fn`
on `Fp8GemmGeometry::rows()` stands exactly as 01 proposed. (You confirmed the same in the
03 verdict, B2.)

## R4 — the six unconsumed aliases: all six HAVE consumers in 04; none deleted

- `NarrowSmallMHidden` → 04's `the_quant_chain_at_a_fitting_geometry_is_one_fused_op`
- `LxOverflowControl` (renamed above) → 04's LX-overflow reduce test
- `NarrowSingleTileN` → 04's `a_second_gemm_at_a_different_row_class_refuses`
- `FusedOverflowControl` (renamed above) → 04's conversion table
  (`the_quant_reduce_at_a_fused_overflowing_geometry_emits_partials`)
- `Gemma26bDownDecode` → 04's conversion table (`the_26b_k2112_pads_to_a_whole_fp8_stick`)
- `NarrowAlignedHidden` → 04's conversion table
  (`an_aligned_k_emits_no_pad_op_and_no_widened_reservation`)

04's references at its lines 87/89/174/492 update to the R2 names at 04's application.
No named instance lands without its consumer.
