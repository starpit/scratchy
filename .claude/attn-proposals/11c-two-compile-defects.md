# 11c — the two compile defects at the applied-11 tree (AUDITOR-RULED exact code)

Run-46's check leg (builds/46, relaunched post-rider) died RC=101 on exactly two
errors. Both are pinned with repros; both fixes are ruled below as exact code.
The coder applies VERBATIM, in the order C1 → C2a → C2b → C2c → C2d.

## C1 — E0573: the multi-segment const path in the wide dispatch's turbofish

`Fp8TileEmit` is `struct Fp8TileEmit<'a, const ROWS: u32>`; the dispatch passes
`crate::work::FP8_GEMM_WIDE_M` (a `u32` const) unbraced. A single-segment const
identifier resolves in the value namespace, but a MULTI-SEGMENT path in a
turbofish resolves as a TYPE first — rustc's own help prescribes braces, and the
braced form is verified compiling (auditor repro, 2026-10-24).

ktir_matmul_fp8.rs, the wide arm of the `row_class()` dispatch (~:827):

OLD:
```rust
        crate::work::Fp8RowClass::Wide => Fp8TileEmit::<crate::work::FP8_GEMM_WIDE_M>::new(
```
NEW:
```rust
        crate::work::Fp8RowClass::Wide => Fp8TileEmit::<{ crate::work::FP8_GEMM_WIDE_M }>::new(
```

## C2 — E0072: `SynthRole::FqTiled(SynthRole, u32)` is recursive by value

The variant wraps the enum it lives in → infinite size. The rejected forms, on
the record: `Box<SynthRole>` breaks the enum's `Copy`; `&'static SynthRole`
cannot wrap the runtime-constructed `FqAmaxP(j)` (no const promotion of a
runtime payload); a flat leaf-role enum is a second spelling of every quant
role (the proposal's own law forbids it); a lifetime parameter infects every
`SynthRole` use. THE RULLED FORM: the tile index is a fact of the IDENTITY
(which tile's staging tensor), not of the ROLE (what the tensor is) — so it
rides the `PlaceId::Synth` as a third field, the `FqTiled` variant is DELETED,
and a `synth_tiled` sibling of `synth` mints tiled ids. Every rendered name is
BYTE-IDENTICAL to today's two-level form (`t{of}_{role}t{t}`).

### C2a — place.rs, the enum and its Display

DELETE the `FqTiled(SynthRole, u32)` variant AND its doc comment (the doc law
moves to `synth_tiled`), and DELETE the Display arm
`Self::FqTiled(r, t) => write!(f, "{r}t{t}"),`.

The variant gains the field:
```rust
    Synth { of: u32, role: SynthRole, tile: Option<u32> },
```
with this doc line added beside the variant's own:
```rust
    /// The target-ABI row tile this synthetic belongs to (`synth_tiled`) — `None` for
    /// every untiled derivation. The tile is a fact of the IDENTITY (which tile's
    /// staging), never of the role.
```

### C2b — place.rs, the two mint methods

`synth()` sets the new field:
```rust
    /// Derive a synthetic from this tensor.
    pub fn synth(self, role: SynthRole) -> Self {
        Self::Synth {
            of: self.tid(),
            role,
            tile: None,
        }
    }
```
and `synth_tiled` lands directly after it:
```rust
    /// Derive a synthetic from this tensor, in a target-ABI row tile's staging. ⭐ ONE
    /// METHOD TILES EVERY ROLE — the fp8 GEMM row tiling (`work::Fp8GemmRowTiles`, the
    /// copied `_m_tiles` law) quantizes per tile over per-tile fresh staging, so every
    /// tensor a tile owns carries its tile index in its ID; the role stays the fact it
    /// always was (no role gains a second spelling), and the index rides the ID.
    pub fn synth_tiled(self, role: SynthRole, t: u32) -> Self {
        Self::Synth {
            of: self.tid(),
            role,
            tile: Some(t),
        }
    }
```

### C2c — place.rs, the Display

```rust
            Self::Synth { of, role, tile } => match tile {
                None => write!(f, "t{of}_{role}"),
                Some(t) => write!(f, "t{of}_{role}t{t}"),
            },
```

### C2d — ktir_matmul_fp8.rs, the tile role map and every constructor

The `tr` closure fuses with `synth` (it returned a role only to be `synth`ed
one token later):
```rust
        // ⭐ THE TILE ID MAP — a tiled source wraps EVERY role's id in its tile index
        // (`synth_tiled`), so a tile's staging tensors are named apart from the activation's
        // own and from every other tile's; the dense source maps every role to itself
        // (byte-identical names).
        let st = |r: R| match src.tile {
            Some(t) => a_id.synth_tiled(r, t),
            None => a_id.synth(r),
        };
        let qn = |r: R| syn(layout, st(r));
```
(The old `tr`/`qn` pair is replaced by exactly these two lambdas; `qn`'s eight
callers at :195-204 are UNCHANGED.)

Every other site, mechanical (the full inventory, verified by grep 2026-10-24):
- `a_id.synth(tr(r))` → `st(r)` — the footprint loops (~:209, ~:219).
- `a_id.synth(tr(R::FqCl))` → `st(R::FqCl)` (~:216); `a_id.synth(tr(R::FqAfp8))` →
  `st(R::FqAfp8)` (~:217).
- `a_id.synth(tr(R::FqAmaxP(j)))` → `st(R::FqAmaxP(j))` (~:351, ~:355).
- Every `X.synth(SynthRole::FqTiled(SynthRole::R, t))` /
  `X.synth(R::FqTiled(R::R, t))` → `X.synth_tiled(SynthRole::R, t)` /
  `X.synth_tiled(R::R, t)` — the 22 constructor sites: ~:861, :865, :921, :925,
  :1095, :1182, :1188, :1209, :1210, :1211, :1213, :1214, :1388, :1476, :1482,
  :1501, :1502, :1503, :1505, :1506, :2319, :2331 (the last two in `#[cfg(test)]`).

### C2e — the two pre-existing test files (forced by the new field)

`PlaceId::Synth { .. }` struct literals gain `tile: None` — 7 sites:
`tests/zz_a_gather_destinations_refuse_at_the_mint.rs` (:196, :200, :305) and
`tests/zz_a_window_past_its_tensor_refuses_at_the_mint.rs` (:179, :195, :225,
:247). Mechanical field additions; nothing else in those files changes.

## What does NOT change
- Every tensor/op name renders byte-identical (`t{of}_{role}t{t}`) — the
  judgment chat's comparison is against names this form already produced.
- `SynthRole` keeps `Copy` and its one vocabulary; `FqAmaxP(j)`'s runtime
  payload flows unchanged; `qn`'s eight callers are untouched.
- `PlaceId`'s derives all stand (`Option<u32>` is Copy/Ord/Hash); the `tid()`
  and `bindable()` matches already use `..`; no serde rides `PlaceId`.

## Apply order and legs
C1 → C2a → C2b → C2c → C2d → C2e, verbatim, no other edits. Then the auditor
re-syncs and re-runs run-46's legs at the fixed tree (the failed attempt is
preserved at builds/46/build.log.check-fail-1). Nothing commits, nothing pushes.
