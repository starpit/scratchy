# 11e — the fmt rider at the 11c+11d tree (auditor record, 2026-10-24 ~23:0x UTC)

## What happened
11c's application (and 11d's rider) introduced fmt drift: the old
two-level `synth(SynthRole::FqTiled(SynthRole::R, t))` calls needed
multi-line wraps; the shorter `synth_tiled(SynthRole::R, t)` forms fit
one line, and the `Synth` variant's third field pushed it past
rustfmt's struct-variant threshold. The diff-13/11b pattern applies —
the rewrap lands as a rider before the legs run.

## The rider, as applied (by the auditor)
- Pre-flight (local `cargo fmt -p ktir-superdsc -- --check`) predicted
  5 hunks; saved at ~/.claude/attn-audit/fmt-preflight-11c.txt.
- The 11c+11d tree was synced to /work/attn-auditer-01 FIRST
  (oc rsync --delete, 10 diff files md5-verified IDENTICAL both sides),
  then `cargo fmt -p ktir-superdsc` ran ON THE POD (the legs' toolchain
  is the authority; pod rustfmt 1.10.0-stable b940084d7e vs local
  58cf8f9c35 — same version line, and both agreed on all of 11b's
  output and on these hunks). Receipt: `cargo fmt --all -- --check` →
  FMT_ALL_CHECK=CLEAN on the pod.
- Exactly 2 files changed: ktir_matmul_fp8.rs (1338f0d8… → 003ffc7e…),
  place.rs (7667bc0c… → ca5f3d6e…). The other 8 diff files untouched.
- The full delta was audited line-by-line before entering the local
  tree: the 5 predicted hunks, byte-for-byte — 4 over-wrapped call
  sites collapsing to one-liners (ktir_matmul_fp8.rs :855, :915, :2314,
  :2326) and the `Synth` variant exploding to multi-line (place.rs
  :251). Zero content change: no identifier, literal, or path differs.
- The formatted files were pulled via oc rsync into staging
  (~/.claude/attn-audit/fmt-rider-11e/pod/), diffed against the
  pre-rider locals, then applied. Local md5s now equal the pod's;
  local `cargo fmt -p ktir-superdsc -- --check` → CLEAN (RC=0).

## Evidence kept
- Pre-flight: ~/.claude/attn-audit/fmt-preflight-11c.txt (5 hunks).
- Pulled files + delta: ~/.claude/attn-audit/fmt-rider-11e/.
- Pod receipt: FMT_ALL_CHECK=CLEAN (in this record; the failed run-46
  attempts remain at builds/46/build.log.fmt-fail-1 and
  build.log.check-fail-1).

## Tree state after 11c+11d+11e
10 files in the diff vs HEAD. Nothing commits, nothing pushes; run-46
relaunches at this tree now.
