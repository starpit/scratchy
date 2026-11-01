#!/bin/bash
# LOCAL ONLY — bake_nolock.sh with the card binary's fp8 preset (quant/fp8-dynamic-per-channel), so the
# captured groups and the generated layout match the card's 2b fp8 binary.
set -u
cd /Users/moosevan/git/scratchy/.claude/worktrees/spyre-attn-triton
S=$PWD/.scratch/stubsdk
[ -f "$S/senlib/lib/libsenlib.dylib" ] || bash "$S/make.sh" > /dev/null
export FLEX_RS_SPYRE_ROOT=$S BOOST_LIB=$S/boost SCRATCHY_SKIP_SENDNN_CXX=1
export DXP_STANDALONE=$PWD/.scratch/capture_tc3/dxp DEEPTOOLS_PATH=$HOME/tmp/dt_src
LOG=$PWD/.scratch/capture_tc3/bake_${MODEL}_fp8.log
echo "start $(date)" > "$LOG"
cargo check -j 6 -p scratchy-cli \
  --features spyre-hw,model/${MODEL},quant/fp8-dynamic-per-channel,serve >> "$LOG" 2>&1
echo "exit $? at $(date)" >> "$LOG"
