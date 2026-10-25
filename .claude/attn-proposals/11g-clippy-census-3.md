# 11g — the clippy census=3 at the 11f tree (AUDITOR-RULED exact code)

Run-46 attempt 4 (23:07 UTC): fmt RC=0, **check RC=0** — the application
compiles end to end for the first time. Clippy died at census=3 (log at
builds/46/clippy.out). All three warnings pin and fix below; the coder
applies VERBATIM, G1 → G2 → G3.

## G1/G2 — needless_question_mark ×2 (ours, D2g-era)

`fp8_tile_mm_op` and `fp8_narrow_mm_op` both return
`Result<EmittedOp, SuperDscError>` (:855, :908) and `emit_sdsc_tiled`
returns the same type — the `Ok(...?)` wrapper around the tail call is
redundant. Delete the wrapper; the tail is the call itself.

ktir_matmul_fp8.rs — `fp8_tile_mm_op`'s tail (:886-894) becomes:

```rust
    let folds = SdscFoldSet::new(spec.iter.cores_used());
    emit_sdsc_tiled(
        &format!("{out}_fq_mm_t{t}"),
        &spec,
        &folds,
        sym_id_base,
        layout,
    )
}
```

`fp8_narrow_mm_op`'s tail (:941-949) becomes:

```rust
    let folds = SdscFoldSet::new(spec.iter.cores_used());
    emit_sdsc_tiled(
        &format!("{out}_fq_mm_narrow_op"),
        &spec,
        &folds,
        sym_id_base,
        layout,
    )
}
```

(OLD in both: `Ok(emit_sdsc_tiled(` … `)?)` — the `Ok(` and the `?` are
deleted; the argument lists are UNCHANGED.)

## G3 — manual_is_multiple_of ×1 (the BindCover rung's own line)

placement.rs:207, `BindCover::FoldBlocks::admits`:
`bytes % block_bytes == 0` trips the pod clippy's (1.99) lint. FORENSICS,
on the record: the line is NOT ancient main — it is today's pre-redirect
BindCover commit fb52073269 (13:56 EDT, +3 rung commits after it), the
campaign's own unpushed on-branch mode. Runs 41-45 (11:04-12:47 EDT)
predate the commit — that is why their census was 0 — and the rung's own
verification was a release build, never clippy. No stop-order breach:
every post-redirect change (11, 11c-11f) is working-tree only, and the
branch has no upstream (nothing pushed).

The fix is clippy's own suggested form; semantics are IDENTICAL under the
preceding guard — the `&&` chain evaluates the call only when
`*block_bytes != 0`, where `is_multiple_of` is exactly `% == 0`:

placement.rs :207, OLD:
```rust
                bytes != 0 && *block_bytes != 0 && bytes % block_bytes == 0 && bytes <= size
```
NEW:
```rust
                bytes != 0 && *block_bytes != 0 && bytes.is_multiple_of(*block_bytes) && bytes <= size
```

## What does NOT change
- `admits`' law, the guard order, and the `bytes <= size` bound.
- The mm-op bodies above the tails; the argument lists verbatim.
- Nothing else in placement.rs.

## Apply order and legs
G1 → G2 → G3, verbatim, no other edits. Then: auditor local fmt pre-flight,
re-sync, run-46 attempt 5. Nothing commits, nothing pushes.

## Rider (11g-fmt, applied by the auditor after the coder's G1-G3)
G3's one-liner crossed rustfmt's width (101 chars); the rider is the
predicted &&-chain wrap, one condition per line. Applied ON THE POD
(cargo fmt -p ktir-superdsc; receipt FMT_ALL_CHECK=CLEAN), exactly 1 file
changed (placement.rs → 6ac4cfd2…; ktir_matmul_fp8.rs untouched, md5 equal
both sides), delta audited line-by-line (the 4-line wrap, zero content
change), pulled and applied locally; local --check CLEAN.
