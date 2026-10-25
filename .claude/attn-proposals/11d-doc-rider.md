# 11d — the doc-consistency rider to 11c (AUDITOR-RULED exact lines)

11c deleted `SynthRole::FqTiled`; three DOC-COMMENT references to the deleted
variant survived (reported by the coder at apply-complete, NOT in 11c's
inventory — the inventory grepped constructor/code sites only). None of them
break check/clippy/tests/build (intra-doc links resolve only under rustdoc),
but every doc describes the real form or it is a lie in the tree of record —
and under `cargo doc` all three links dangle. The rider is three one-token
swaps; the coder applies VERBATIM, nothing else.

Link resolution, verified: `PlaceId` is imported in ktir_matmul_fp8.rs
(`use crate::place::{PlaceId, SynthRole};` :25) so `[`PlaceId::synth_tiled`]`
resolves there; place.rs is same-file; work.rs's mention is plain backticks
(no brackets, nothing to resolve).

## Site 1 — place.rs, FqTileRow's doc tail (:60-61)

OLD:
```rust
    /// land in a fresh buffer of their own before the splice). Always carried under
    /// [`Self::FqTiled`]: the tile index is the staging's own, known only at bake.
```
NEW:
```rust
    /// land in a fresh buffer of their own before the splice). Always minted by
    /// [`PlaceId::synth_tiled`]: the tile index is the staging's own, known only at bake.
```
("carried under" the deleted role-wrapper → "minted by" the method that now
carries the index; every FqTileRow id arrives through `synth_tiled` — wide
tiles at `t`, the narrow arm at `0`.)

## Site 2 — work.rs, Fp8GemmTileIdx's doc (:502)

OLD:
```rust
/// (`SynthRole::FqTiled`'s index takes its value). Minted from the plan's own tile count,
```
NEW:
```rust
/// (`PlaceId::synth_tiled`'s index takes its value). Minted from the plan's own tile count,
```

## Site 3 — ktir_matmul_fp8.rs, Fp8QuantSource's doc (:58)

OLD:
```rust
/// their tile index ([`SynthRole::FqTiled`]), keyed off the ACTIVATION's id either way
```
NEW:
```rust
/// their tile index ([`PlaceId::synth_tiled`]), keyed off the ACTIVATION's id either way
```

## Apply order and legs
Sites 1 → 2 → 3, verbatim, no other edits. Then the auditor re-syncs and
re-runs run-46's legs at the fixed tree. Nothing commits, nothing pushes.
