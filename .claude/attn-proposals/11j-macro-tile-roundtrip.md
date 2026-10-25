# 11j — the forward-macro crate's three errors at the 11i tree (AUDITOR-RULED exact code)

Run-46 attempt 7 (23:40 UTC): every test leg GREEN for the first time — fmt, check,
clippy census 0, tests-subtile, tests-ktir, **tests-ktir-lib (227/227)**, tests-spyre,
**tests-triton**, tests-forward_tape. The gated release-build leg then failed at
**scratchy-forward-compiler-macro** — the only leg that compiles that crate (the
check/clippy/test legs scope to the spyre/ktir/subtile/triton crates; the macro is
compiled solely as a build dependency of the release binary). Three errors, all one
defect class: 11c's `Synth { tile }` field and the tiled law's `FqTileRow` role never
reached the macro's VALUE-ROUND-TRIP tokenizers until now. The coder applies
VERBATIM, J1 → J2 → J3.

## FORENSICS

- codegen.rs:9764 `synth_role_tokens` — E0004: `&SynthRole::FqTileRow` not covered.
  This match is INTENTIONALLY exhaustive (its own doc, :9755-9757: "adding a role
  breaks THIS match at build time, which is the whole reason it is an enum") — the
  pre-redirect tiled diffs added `FqTileRow` to the enum (place.rs:62) and the macro
  arm was never added because the macro never compiled after that.
- codegen.rs:9879 + ktir_tokens.rs:412 `place_id_tokens` — E0027 ×2: patterns over
  `PlaceId::Synth { of, role }` missing the `tile` field 11c added. Both sites
  RECONSTRUCT the value as tokens in generated code; without `tile` in the
  reconstruction the generated code would itself fail (E0063) — the round-trip must
  be faithful.
- OWNERSHIP (on the record, as in 11f): the 11c ruling verified every rendered NAME
  byte-identical in ktir-superdsc's Display, but the macro's two reconstruction
  tokenizers are a different surface — not in any leg until the release build. The
  audit miss is recorded here, not buried.
- Both reconstruction targets are the SAME type as the pattern source: `bundle`
  (lower_subtile_tape_to_superdsc.rs:55 `pub use scratchy_spyre_bundle as bundle`)
  and `bundle_code` (spyre/src/lib.rs:30 `pub use scratchy_spyre_bundle as
  bundle_code`) are two aliases of one crate, and `__KPlace`/`__KRole`
  (ktir_tokens.rs:42) import that same `PlaceId`/`SynthRole`. The compiler's own
  E0004 note pins `SynthRole` to ktir-superdsc/src/place.rs:24. Whatever fields the
  pattern demands, the construction accepts.
- Exactly two `Synth {` sites exist in the macro crate (grep); the runtime side
  compiled green in the check/test legs, so its own Synth matches already admit the
  field. ktir_tokens' role matcher needs NOTHING — its catch-all (:457-460) renders
  any unit variant by Debug name, which is `FqTileRow`.

## J1 — codegen.rs: the `FqTileRow` arm (:9787-9791)

Insert the arm directly after the `FqAmaxP(j)` arm, before the per-head comment
block. OLD (:9787-9791):
```rust
        R::FqAmaxP(j) => {
            let j = proc_macro2::Literal::u32_unsuffixed(*j);
            quote! { #b::FqAmaxP(#j) }
        }
        // The per-head windowed-norm synthetics (the `FqAmaxP` shape: one per head,
        // all at offset zero).
```
NEW:
```rust
        R::FqAmaxP(j) => {
            let j = proc_macro2::Literal::u32_unsuffixed(*j);
            quote! { #b::FqAmaxP(#j) }
        }
        R::FqTileRow => quote! { #b::FqTileRow },
        // The per-head windowed-norm synthetics (the `FqAmaxP` shape: one per head,
        // all at offset zero).
```

## J2 — codegen.rs: `place_id_tokens`' Synth arm (:9879-9885)

OLD:
```rust
        P::Synth { of, role } => {
            let of = proc_macro2::Literal::u32_unsuffixed(*of);
            let role = synth_role_tokens(role);
            quote! {
                ::scratchy_target_spyre::bundle_code::PlaceId::Synth { of: #of, role: #role }
            }
        }
```
NEW:
```rust
        P::Synth { of, role, tile } => {
            let of = proc_macro2::Literal::u32_unsuffixed(*of);
            let role = synth_role_tokens(role);
            let tile = match tile {
                Some(t) => {
                    let t = proc_macro2::Literal::u32_unsuffixed(*t);
                    quote! { ::core::option::Option::Some(#t) }
                }
                None => quote! { ::core::option::Option::None },
            };
            quote! {
                ::scratchy_target_spyre::bundle_code::PlaceId::Synth {
                    of: #of,
                    role: #role,
                    tile: #tile,
                }
            }
        }
```
(The arm is a match over `&PlaceId`, so `tile` is `&Option<u32>`: `Some(t)` binds
`t: &u32` by match ergonomics and the literal takes `*t`. The quote's struct is
exploded one field per line — the form rustfmt itself would produce at this width;
rustfmt does not reformat `quote!` interiors, so either form is stable, and this one
is.)

## J3 — ktir_tokens.rs: `place_id_tokens`' Synth arm (:412-416)

OLD:
```rust
        PlaceId::Synth { of, role } => {
            let of = proc_macro2::Literal::u32_unsuffixed(of);
            let r = synth_role_tokens(role);
            quote! { __KPlace::Synth { of: #of, role: #r } }
        }
```
NEW:
```rust
        PlaceId::Synth { of, role, tile } => {
            let of = proc_macro2::Literal::u32_unsuffixed(of);
            let r = synth_role_tokens(role);
            let tile = match tile {
                Some(t) => {
                    let t = proc_macro2::Literal::u32_unsuffixed(t);
                    quote! { ::core::option::Option::Some(#t) }
                }
                None => quote! { ::core::option::Option::None },
            };
            quote! { __KPlace::Synth { of: #of, role: #r, tile: #tile } }
        }
```
(This file's `place_id_tokens` takes `id` BY VALUE (:404), so `tile` is an owned
`Option<u32>` and the literal takes `t` with no deref — the one deliberate
difference from J2.)

## What does NOT change
- `synth_role_tokens` in ktir_tokens.rs (the catch-all renders FqTileRow already).
- The `Act` arms, `placement_tokens`, `bind_cover_tokens`, every other matcher.
- ktir-superdsc, the spyre runtime, the emission — zero semantic bytes; these three
  hunks only make the macro's token round-trip faithful to the type it already
  pattern-matches.

## Apply order and legs
J1 → J2 → J3, verbatim, no other edits. The coder reports anything else broken it
sees — reports, never fixes. Then: auditor local fmt pre-flight (the macro crate is
in the workspace-wide fmt leg's scope; the surrounding Rust must be clean — any
drift goes through the pod rider), re-sync /work/attn-auditer-01, run-46 attempt 8.
NOTE for what may come next: after the macro compiles, the release build runs the
granite-2b `#[forward]` expansion for the first time at this tree — a geometry
refusal there is a build failure BY DESIGN (the reference's own refusals), and the
same cycle continues: I pin + rule, you apply, I audit. Nothing commits, nothing
pushes.
