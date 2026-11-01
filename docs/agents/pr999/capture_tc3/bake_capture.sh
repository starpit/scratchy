#!/bin/bash
# LOCAL ONLY — the card (spyre-hw) bake of one model against the stub SDK, every group compiled by this
# machine's dxp through the capturing wrapper, once the peer's GPU lock is free (the compile is heavy
# CPU). The captured SDSC lands in .scratch/capture/<fp>/group_<i>.
set -u
cd /Users/moosevan/git/scratchy/.claude/worktrees/spyre-attn-triton
until [ ! -e /Volumes/Builds/gpu-lock ]; do
  sleep 20
done
S=$PWD/.scratch/stubsdk
[ -f "$S/senlib/lib/libsenlib.dylib" ] || bash "$S/make.sh" > /dev/null
export FLEX_RS_SPYRE_ROOT=$S BOOST_LIB=$S/boost SCRATCHY_SKIP_SENDNN_CXX=1
export DXP_STANDALONE=$PWD/.scratch/capture_tc3/dxp DEEPTOOLS_PATH=$HOME/tmp/dt_src
LOG=$PWD/.scratch/capture_tc3/bake_${MODEL}.log
echo "start $(date)" > "$LOG"
cargo clippy -j 6 -p scratchy-serving-worker --features spyre-hw,scratchy-models/${MODEL} --lib -- -D warnings >> "$LOG" 2>&1
echo "exit $? at $(date)" >> "$LOG"
