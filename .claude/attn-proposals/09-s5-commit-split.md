# 09 — S5: commit discipline — the per-concern split (PAPER until the user lifts the standing stop-order)

Finding S5: "the SpraysFirst half and the fp8 M-tiling half are two concerns in one worktree — commits split per concern (Conventional Commits, scoped). And note in the SpraysFirst commit/note: the DEFECT is measured; the FIX's card verification is the judgment chat leg after green — no fix-verified claim until then."

## The two concerns, and every file each owns

**Concern A — SpraysFirst (the mask-add y-major division fix):**
- crates/targets/spyre/ktir/ktir-superdsc/src/emit/lower_ktir_to_superdsc.rs (the door refusal + the three tests)
- crates/targets/spyre/ktir/ktir-superdsc/src/ir/bridge/tiled_op_sdsc_op/pointwise.rs (the builder's resolve call + measured-number docs)
- crates/targets/spyre/ktir/ktir-superdsc/src/emit/mod.rs (the sizing helper's resolve call)
- crates/targets/spyre/ktir/ktir-superdsc/tests/zz_a_stick_major_elementwise_op_splits_its_sticks_first.rs
- work.rs — ONLY its resolve()/distribute_cores_sprays_first()/sprays vocabulary hunks

**Concern B — fp8 row-tiling (the Defect-3 fix):**
- crates/targets/spyre/ktir/ktir-superdsc/src/emit/ktir_matmul_fp8.rs
- work.rs — ONLY its Fp8GemmRowTiles family hunks (incl. everything in proposals 01/03)
- crates/targets/spyre/ktir/ktir-superdsc/src/place.rs (FqTileRow/FqTiled)
- crates/targets/spyre/ktir/ktir-superdsc/src/sdsc_abstract.rs (Fp8TileStaging, MatM::of_token_rows retype)
- addr/mod.rs (Idx: Display, if approved)

**work.rs is the one file both concerns touch** — the split is BY HUNK, not by file: at
commit time, two patch files (one per concern, over the approved-final tree), applied to
the index with `git apply --cached` separately, one commit each. No interactive add, no
reordering of the working tree — the split happens at commit time only.

## The two commits (messages drafted; nothing is committed until the user lifts the stop-order)

A (SpraysFirst), Conventional Commits, scoped, `-S`, new commit only:

    fix(spyre): the mask-add's y-major division goes sprays-first

    One spray per core at [y=32, mb=21, out=64]; the wire order is dxp
    innermost-first (perfdsc/perfLdsInfo.h:29, dsc/dataOpDsc.h:347), reusing
    with_dxp_innermost_first — the batched-gathered path's own setter. The
    door refuses y-major for anything not Ew(Law|SpraysFirst), gathered
    divisions included.

    The DEFECT is measured: 620 of 672 µs (leg6l-after-spray-split.py);
    the row-stride and stick-occupancy numbers are in the op docs, each
    citing its saved artifact. The FIX's card verification is the judgment
    chat leg after green — NO fix-verified claim until that leg passes.

B (fp8 row-tiling):

    fix(spyre): fp8 GEMM rows tile at the target ABI row class

    The wide GEMM's A operand reaches the card as tiles of 4 (padded up),
    decode as 1, narrow GEMMs whole at the 128 alignment — the reference's
    own _m_tiles law (fp8_linear_kernel.py:338-346). Per tile: zero-fill the
    pad rows, clone the real rows into fresh staging, quantize THE CLONE,
    contract at the tile's class, dequant into a tile-local landing, splice
    the real rows back. The quant walk and the mm's A walk now spell ONE
    buffer's layout — the defect-3 staging tear is closed by construction.

## Sequencing note

The restructure currently sitting uncommitted in the worktree is MINE and interleaved
with the other session's SpraysFirst WIP; the F1-F8 fixes amend my half, the S1-S5 fixes
amend theirs. Each concern's commit carries only its own hunks, split as above. Until the
user lifts the standing "commit nothing" order, this file is the whole of S5: a plan,
executed only on the user's go.
