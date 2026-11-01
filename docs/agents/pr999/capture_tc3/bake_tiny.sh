#!/bin/bash
# LOCAL ONLY — the smallest-context config in scope (tinyllama-1.1b, 2048 positions = 16 pages),
# dense, to check the batched form's pool precondition at B=1.
set -u
cd /Users/moosevan/git/scratchy/.claude/worktrees/spyre-attn-triton
S=$PWD/.scratch/stubsdk
[ -f "$S/senlib/lib/libsenlib.dylib" ] || bash "$S/make.sh" > /dev/null
export FLEX_RS_SPYRE_ROOT=$S BOOST_LIB=$S/boost SCRATCHY_SKIP_SENDNN_CXX=1
export DXP_STANDALONE=$PWD/.scratch/capture_tc3/dxp DEEPTOOLS_PATH=$HOME/tmp/dt_src
LOG=$PWD/.scratch/capture_tc3/bake_tinyllama.log
echo "start $(date)" > "$LOG"
cargo check -j 6 -p scratchy-cli --features spyre-hw,model/tinyllama-1.1b,serve >> "$LOG" 2>&1
echo "exit $? at $(date)" >> "$LOG"
