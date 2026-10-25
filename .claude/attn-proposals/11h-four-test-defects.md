# 11h — the four test defects at the 11g tree (AUDITOR-RULED exact code)

Run-46 attempt 5 (23:16 UTC): fmt RC=0, check RC=0, clippy census=0, tests-subtile
RC=0, tests-ktir RC=0, tests-spyre RC=0, tests-forward_tape RC=0. TWO legs remain,
both in NEVER-COMPILED test code: **tests-ktir-lib RC=101** (log at
builds/46/build.log.test-fail-1; headers: 25×E0433 + 2×E0308 + 2×E0605, cargo
tally 27) and **tests-triton RC=101** (1×E0277, `BindCover: Ord`). All 28 errors
pin to exactly four defects; the fixes are ruled below as exact code. The coder
applies VERBATIM, H1 → H2 → H3 → H4.

## FORENSICS — why this code never compiled

- The work.rs geometry aliases and newtypes (11's D2c) were added to the OUTER
  module; the test module's import list was never extended — the whole
  01+02+11 application's first compile was run-46 attempt 3.
- The ktir_matmul_fp8.rs tiled-law test predates the typed-mint law (D-something
  typed `Fp8GemmRowTiles::of`'s signature to `RowCount`/`BlockCols`); its
  bare-u32 call sites are the ripple that first compiles now.
- The triton cover test is today's pre-redirect BindCover rung (fb52073269),
  whose verification was a RELEASE BUILD — which compiles no test binaries.
  No stop-order breach: all post-redirect changes remain working-tree only.

## H1 — work.rs:964: the test module's import list (all 25 E0433s)

`gather_split_tests` uses 18 unqualified names of the outer module; its import
carries only 4. The fix list is derived from the SOURCE SPAN (:962–1185, every
unqualified super-name the module uses — a superset of every error site), not
from the error list. `RowCount` appears only as full
`crate::sdsc_abstract::RowCount` paths and resolves — NOT added.

work.rs :964 — OLD:
```rust
    use super::{CoreDivision, Fp8GemmRowTiles, GatherSplit, ItDim};
```
NEW (rustfmt-stable; wrap verified against the legs' rustfmt):
```rust
    use super::{
        CoreDivision, Fp8GemmRowTiles, Fp8GemmStickGroups, Fp8GemmTileCount, Fp8GemmTileIdx,
        Fp8RowClass, GatherSplit, Gemma26bGateDecode, Gemma26bGatePrefill, Granite2bOProjPrefill,
        Granite2bQkvDecode, Granite2bQkvPrefill, InexpressibleNTail, ItDim, NarrowHighUnaligned,
        NarrowPadded, NarrowSmallM, SubStickContraction,
    };
```
Nothing else in the module header changes (`use crate::superdsc_opspec::Df;`
and the BTreeMap import stay).

## H2 — ktir_matmul_fp8.rs :2243-2249: typed mints, typed assert (both E0308s)

The test mod has `use super::*` (:1550), so `RowCount`/`BlockCols` are
glob-visible — the mints are the dispatch's OWN idiom (:809-810).
`Fp8GemmTileCount` is not bare-imported at file level (the dispatch uses the
full path at :814) — the test does the same. The `let (m, k, n) = (21u32,
2048, 3072);` locals STAY u32: `fp8_layout(m, k, n)` (:2250) and
`matmul_fp8_descriptors(&facts(), m, k, n, …)` (:2252) take u32s — the mints
happen only at the `of()` call, mirroring the dispatch body.

ktir_matmul_fp8.rs :2243-2249 — OLD:
```rust
        let plan =
            crate::work::Fp8GemmRowTiles::of(m, k, n).expect("the granite-2b qkv geometry plans");
        assert_eq!(
            (plan.tile_rows(), plan.tile_count(), plan.total_rows()),
            (4, 6, 24),
            "21 rows tile by four, padded UP — never a trailing short tile"
        );
```
NEW:
```rust
        let plan = crate::work::Fp8GemmRowTiles::of(
            RowCount::of_token_rows(m),
            BlockCols::of_feature_cols(k),
            BlockCols::of_feature_cols(n),
        )
        .expect("the granite-2b qkv geometry plans");
        assert_eq!(
            (plan.tile_rows(), plan.tile_count(), plan.total_rows()),
            (
                RowCount::of_token_rows(4),
                crate::work::Fp8GemmTileCount::new(6),
                RowCount::of_token_rows(24)
            ),
            "21 rows tile by four, padded UP — never a trailing short tile"
        );
```
(The assert's exploded-tuple form is work.rs's own compiled idiom, :1047-1051.)

## H3 — ktir_matmul_fp8.rs :2262 and :2279: the newtype accessor (both E0605s)

`k_stick_groups()`/`n_stick_groups()` return `Fp8GemmStickGroups` (:435/:440);
its accessor is `const fn get(self) -> u32` (:525). The cast rides the accessor,
exactly as every other newtype in the tree does.

:2262 — OLD: `            6 * plan.k_stick_groups() as usize,`
NEW:      `            6 * plan.k_stick_groups().get() as usize,`

:2279 — OLD: `            6 * plan.n_stick_groups() as usize,`
NEW:      `            6 * plan.n_stick_groups().get() as usize,`

## H4 — placement.rs :187: `BindCover` derives Ord (the E0277)

The rung's own test sorts `Vec<(u64, BindCover)>`
(zz_placements_json_carries_the_cover.rs:52) to compare the placements file
order-independently; tuple `sort()` needs `Ord` on both elements. The derived
ordering is rustc's own — variant declaration order (`Full` < `FoldBlocks`),
then `block_bytes` — no invented law: the test's expected vec (:55-58) already
lists `Full` first. REJECTED: rewriting the test to `sort_by_key` with a
hand-rolled discriminant closure — more invented code for the same mechanical
need, and the test is the rung's own, already-approved text.

placement.rs :187 — OLD:
```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
```
NEW (rustfmt's own wrap at width 103; verified against the legs' rustfmt):
```rust
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
```
`BindCover`'s variants, fields, docs, and `admits` are untouched; serde is
untouched (Ord does not touch Serialize/Deserialize).

## What does NOT change
- The emission path: zero hunks touch non-test code EXCEPT H4's derive line,
  which adds a trait impl with no behavior (the enum's own values and `admits`
  are byte-identical).
- The 11g-wrapped `admits` body, the dispatch, both emit structs, every alias
  and newtype definition.
- The triton test file itself — its text is the rung's own approved text.

## Apply order and legs
H1 → H2 → H3 → H4, verbatim, no other edits. The coder reports anything else
broken it sees while applying — reports, never fixes. Then: auditor local fmt
pre-flight (all four hunks are ruled in rustfmt-stable form; any drift goes
through the pod rider), re-sync /work/attn-auditer-01, run-46 attempt 6. If
attempt 6 surfaces errors that were masked behind the E0433 resolution
failures, the same cycle continues: I pin + rule, you apply, I audit. Nothing
commits, nothing pushes.
