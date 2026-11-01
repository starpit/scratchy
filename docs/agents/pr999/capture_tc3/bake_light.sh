#!/bin/bash
# LOCAL ONLY — the card (spyre-hw) bake of one model against the stub SDK, every group compiled by this
# machine's dxp through the capturing wrapper, once the peer's GPU lock is free (the compile is heavy
# CPU). The captured SDSC lands in .scratch/capture/<fp>/group_<i>.
set -u
cd /Users/moosevan/git/scratchy/.claude/worktrees/spyre-attn-triton
# no lock wait: light (-j 2) incremental bake


S=$PWD/.scratch/stubsdk
[ -f "$S/senlib/lib/libsenlib.dylib" ] || bash "$S/make.sh" > /dev/null
export FLEX_RS_SPYRE_ROOT=$S BOOST_LIB=$S/boost SCRATCHY_SKIP_SENDNN_CXX=1
export DXP_STANDALONE=$PWD/.scratch/capture_tc3/dxp DEEPTOOLS_PATH=$HOME/tmp/dt_src
LOG=$PWD/.scratch/capture_tc3/bake_${MODEL}_blocks.log
echo "start $(date)" > "$LOG"
cargo clippy -j 2 -p scratchy-serving-worker --features spyre-hw,scratchy-models/${MODEL} --lib -- -D warnings >> "$LOG" 2>&1
echo "exit $? at $(date)" >> "$LOG"
