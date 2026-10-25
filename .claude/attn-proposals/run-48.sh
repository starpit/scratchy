#!/bin/bash
# RUN 48 — THE RE-FIRE at the landed tree (B rung cut + accumulator threading + parity move + the keep-what's-selected kill).
# One cold bake, fresh store, timed. Expected: red at the whole-bake total, or green if the twin is flat-cost;
# the census gives the real post-cut per-bundle + whole-bake numbers for the fusion decision.
cd /work/attn-auditer-01
LOG=/work/attn-contract-builds/48/build.log
mkdir -p /work/attn-contract-builds/48
: > "$LOG"
export PATH=/work/.cargo/bin:$PATH
export HF_HOME=/work/.cache/huggingface
export LD_LIBRARY_PATH=/opt/ibm/spyre/senlib/lib:/opt/ibm/spyre/runtime/lib:/opt/ibm/spyre/deeptools/lib
export DEEPTOOLS_PATH=/opt/ibm/spyre/deeptools/share
( while true; do echo "HB $(date +%H:%M:%S)" >> "$LOG"; sleep 30; done ) & HBPID=$!
STORE=/work/attn-contract-builds/48/store-cold
rm -rf "$STORE"; mkdir -p "$STORE"
echo "COLD START $(date +%H:%M:%S) (fresh SCRATCHY_SUPERDSC_CACHE=$STORE)" >> "$LOG"
T0=$(date +%s)
env -u SCRATCHY_SKIP_SENDNN_CXX SCRATCHY_SUPERDSC_CACHE="$STORE" \
  cargo build --release -p scratchy-cli \
  --features spyre-hw,serve,model/granite-3.1-2b-instruct,quant/fp8-dynamic-per-channel \
  >> "$LOG" 2>&1
RC=$?
T1=$(date +%s)
echo "COLD RC=$RC WALL=$((T1-T0))s ($(date +%H:%M:%S))" >> "$LOG"
if [ $RC -eq 0 ]; then
  echo "WARM START $(date +%H:%M:%S) (same store)" >> "$LOG"
  T2=$(date +%s)
  env -u SCRATCHY_SKIP_SENDNN_CXX SCRATCHY_SUPERDSC_CACHE="$STORE" \
    cargo build --release -p scratchy-cli \
    --features spyre-hw,serve,model/granite-3.1-2b-instruct,quant/fp8-dynamic-per-channel \
    >> "$LOG" 2>&1
  RC2=$?
  T3=$(date +%s)
  echo "WARM RC=$RC2 WALL=$((T3-T2))s ($(date +%H:%M:%S))" >> "$LOG"
fi
echo "=== CENSUS ===" >> "$LOG"
grep -E "trip-wall|refus|budget|class\(es\)|trips|WHOLE-BAKE|twin" "$LOG" | grep -v "^HB" >> "$LOG" 2>&1 || true
ls /tmp/superdsc-dump/*census* >> "$LOG" 2>&1 || true
cat /tmp/superdsc-dump/*census* >> "$LOG" 2>&1 || true
echo "LEG all DONE" >> "$LOG"
kill $HBPID
