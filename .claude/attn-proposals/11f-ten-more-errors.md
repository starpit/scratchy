# 11f — the next ten errors at the 11c+11d+11e tree (AUDITOR-RULED exact code)

Run-46 attempt 3 (23:00 UTC): fmt RC=0 (the 11e rider held), check RC=101 on
TEN errors in the ktir-superdsc lib — the layer that was hiding behind 11c's
two. Log preserved at builds/46/build.log.check-fail-2. All ten pin to exactly
four defects; the fixes are ruled below as exact code. The coder applies
VERBATIM, F1 → F2 → F3 → F4.

## F1/F2 — E0277+E0599 ×4: `new(...)?.emit()` at the row-class dispatch

Both emit constructors return `Self` (`Fp8TileEmit::new` :998-1018,
`Fp8NarrowEmit::new` :1289-1311) and `emit(self)` returns
`Result<Vec<EmittedOp>, SuperDscError>` — the dispatch arms apply `?` to a
`Self`, then call `.emit()` on the diverged inference. Every other `?` in the
file rides a real `Result` (verified by grep — these two lines are the only
`?.emit` sites). The `?` is deleted on both arms; the arm value is the
`emit()` Result itself, exactly like the dense arm's early return at :817.

ktir_matmul_fp8.rs, the dispatch (:825-837) — NEW, both arms:

```rust
    match plan.row_class() {
        crate::work::Fp8RowClass::Wide => Fp8TileEmit::<{ crate::work::FP8_GEMM_WIDE_M }>::new(
            facts,
            &plan,
            sym_id_base,
            layout,
            quantized,
        )
        .emit(),
        crate::work::Fp8RowClass::Narrow(padded) => {
            Fp8NarrowEmit::new(facts, &plan, padded, sym_id_base, layout, quantized).emit()
        }
    }
}
```

(OLD: line :832 is `        )?` — NEW: `        )`; line :835 is
`...quantized)?.emit()` — NEW: `...quantized).emit()`. Nothing else in the
dispatch changes.)

## F3/F4 — E0277 ×6: the staging structs' invented `#[derive(Clone, Debug)]`

`Stk<K: KindTag>`'s derive puts `K: Clone`/`K: Debug` bounds on its impl
(the imperfect-derive trap), and `RowBlockedTag` is an empty sealed marker
with no derives — so `#[derive(Clone, Debug)]` on the staging structs
demands `RowBlockedTag: Clone + Debug` (4 Clone errors, two fields × two
structs; 2 Debug errors, one per struct).

THE RULING: DELETE the derive line from BOTH staging structs. ZERO consumers
exist — `fp8_tile_mm_op`/`fp8_narrow_mm_op` take the staging BY REFERENCE
(:848, :902); each staging is minted fresh inside its emit pass (:1167,
:1460) and dropped at scope end; the second-GEMM sharing law rides the
`quantized` name-set, not struct clones; no test constructs, clones, or
formats either type (verified by grep over src/ and tests/). The derives are
invented dead weight that cannot compile — deleted, not fed.

On the record, the REJECTED alternatives:
- `#[derive(Clone, Debug)]` on `RowBlockedTag` (rustc's own help): modifies a
  pre-existing sealed-system marker to serve an invented need on
  application-added structs.
- Manual `impl<K: KindTag> Clone/Debug for Stk<K>` without the K bound: same
  flaw one floor down — rewrites main's `Stk` trait impls for the same
  invented need.

OWNERSHIP: the derives were in the 01/02/11 hunks this office approved
pre-redirect, read then as inert; the first compile of the whole application
exposes them. The audit miss is recorded here, not buried.

sdsc_abstract.rs — DELETE the line at :3091:
```rust
#[derive(Clone, Debug)]
```
(the line directly above `pub struct Fp8TileStaging<const R: u32> {`)

sdsc_abstract.rs — DELETE the line at :3144:
```rust
#[derive(Clone, Debug)]
```
(the line directly above `pub struct Fp8NarrowStaging {`)

Nothing else in either struct, impl, or doc changes; both structs become
plain (no derives), which is exactly their consumer set.

## What does NOT change
- The `staged()` constructors, accessors, and row-class refusals — the types'
  whole contract — untouched.
- `Stk`, `KindTag`, and all four tags — pre-existing, untouched.
- The dispatch's exhaustive two-arm match and its comment block — untouched.

## Apply order and legs
F1 → F2 → F3 → F4, verbatim, no other edits. The coder reports anything else
broken it sees while applying — reports, never fixes. Then: auditor local fmt
pre-flight, re-sync, run-46 attempt 4. Nothing commits, nothing pushes.
