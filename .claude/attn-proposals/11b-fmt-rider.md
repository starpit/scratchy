# 11b — the fmt rider at the applied-11 tree (auditor record, 2026-10-24 ~22:4x UTC)

## What happened
Run-46's first launch (22:37:15 UTC) died at LEG fmt RC=1: `cargo fmt --all
--check` named 22 hunks across the four files the diff touches. The diff-13
pattern applies — the rewrap lands as a rider before the other legs run.

## The rider, as applied (by the auditor)
- `cargo fmt -p ktir-superdsc` ran ON THE POD (the legs' toolchain is the
  authority), then `cargo fmt --all -- --check` → CLEAN (receipt:
  FMT_ALL_CHECK=CLEAN).
- The full delta was audited line-by-line before it entered the tree of
  record: 170 changed lines across 4 files — ktir_matmul_fp8.rs 121,
  lower_ktir_to_superdsc.rs 24, work.rs 21, pointwise.rs 4. Every hunk is a
  mechanical rewrap (import reorder, argument/destructure splitting, method-
  chain splitting, expect-string dedent, iterator-chain joining). Zero
  content change: no identifier, literal, or path differs.
- The formatted files were pulled from the pod into the local worktree
  (staging + diff + apply). Local md5s now equal the pod's:
  ktir_matmul_fp8 c423aab38cd674154700344dad11191e, lower_ktir
  ef1bcbb6ba1a3f91a00de8ce24cc53dd, pointwise 6fee64b4f6675a75e6ecdc918c9e129c,
  work.rs 9cab079315dbced3325322e2baceafca.
- Tree shape after the rider: same 8 files, 2120 insertions, 81 deletions.

## Evidence kept
- The failed first attempt: /work/attn-contract-builds/46/build.log.fmt-fail-1
  (preserved before relaunch).
- The rider delta: ~/.claude/attn-audit/fmt-rider/ (staged copies) and this
  file's per-file line counts.

## Forensic note from the same window (closed)
The 17:59:57Z run-38 rewrite + 18:00:17Z launch + 18:04Z pod syncs were the
AUDITOR's own pre-compaction actions (receipt: the auditor's transcript holds
the write, the 17:59:57.934Z wc -l of the 78-line script, and the setsid
launch; the compaction summary dropped the window). The coder's 18:06–18:55Z
pod loop (rsyncs, release builds, judgment-form chats) predates the redirect
and was legitimate task work; its 18:53 scr is a pre-11 sprays-first build,
not the judgment binary. Nothing of proposal 11 reached the pod before the
auditor's 22:37Z rsync (md5-verified, 8/8 files).
