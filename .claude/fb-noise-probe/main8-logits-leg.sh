#!/bin/bash
# front-b-noise 8B MAIN LOGITS LEG (nickm2) — the auditor's order 2026-10-27:
# "8b seeded P3 (the pair ×2, deterministic) and P5 (200 draws within 3σ of
# main's 8b logits)". P5 needs MAIN's own 8b host logits for the leg prompt —
# main reads its logits back to host, so the same print-only probe patch that
# banked the 2b reference (main-logits-dump.patch, applied to the main tree,
# NEVER committed) dumps them.
# ⚠ THE PATCH'S DUMP PATH IS FIXED (/work/fb-chat-main/logits-*.bin) — the 2b
# reference lives there. This leg moves the 2b refs aside into
# /work/fb-chat-main/2b-refs/ FIRST, runs main's 8b in-process chat, banks the
# 8b dump at /work/fb-chat-main8/logits-0000.bin, and restores the 2b refs.
# Tree: /work/front-b-chatmain (main e63c6f8ac8 + the never-committed patch —
# do not touch it). Build: ON THE POD, release, its own warm target dir (the
# main8-ref-leg build). Model: granite-3.1-8b fp8. Markers: MAIN8LOG-DONE /
# MAIN8LOG-FAILED. Log: /work/main8-logits-leg.log
set -u

LOG=/work/main8-logits-leg.log
TREE=/work/front-b-chatmain
TDIR=/work/front-b-chatmain/target
REF2B=/work/fb-chat-main
REF8B=/work/fb-chat-main8
SCR=$TDIR/release/scr
PROMPT="Write two sentences about the ocean."

if [ "${BG:-0}" != "1" ]; then
  BG=1 setsid nohup bash "$0" >>"$LOG" 2>&1 &
  echo "detached pid $! ; log $LOG"
  exit 0
fi
exec >>"$LOG" 2>&1

echo "==== main8-logits-leg start $(date -u +%FT%TZ) ===="

# ── busy check (live cargo only; zombies are the pod's normal litter) ──
if pgrep -x cargo >/dev/null 2>&1; then
  for p in $(pgrep -x cargo); do
    if ! ps -o stat= -p "$p" 2>/dev/null | grep -q Z; then
      echo "POD BUSY (live cargo pid $p)"; echo MAIN8LOG-FAILED; exit 1
    fi
  done
fi
if pgrep -f 'scr serve|scr chat' >/dev/null 2>&1; then
  echo "POD BUSY (scr)"; echo MAIN8LOG-FAILED; exit 1
fi

[ -d "$TREE/crates" ] || { echo "no main tree at $TREE"; echo MAIN8LOG-FAILED; exit 1; }
[ -s "$REF2B/logits-0000.bin" ] || { echo "no 2b logits to guard at $REF2B"; echo MAIN8LOG-FAILED; exit 1; }
cd "$TREE" || { echo "cannot cd $TREE"; echo MAIN8LOG-FAILED; exit 1; }

export PATH=/work/.cargo/bin:$PATH
export CARGO_TARGET_DIR=$TDIR
export LD_LIBRARY_PATH=/opt/ibm/spyre/senlib/lib:/opt/ibm/spyre/runtime/lib:/opt/ibm/spyre/deeptools/lib
export DEEPTOOLS_PATH=/opt/ibm/spyre/deeptools/share
export HF_HUB_OFFLINE=1

# ── leg 1: main's 8b chat stack (release, warm dir) ──
echo "==== MAIN8LOG-BUILD-START $(date -u +%FT%TZ) ===="
if ! cargo build --release -p scratchy-cli \
     --features spyre-hw,serve,model/granite-3.1-8b-instruct,quant/fp8-dynamic-per-channel; then
  echo MAIN8LOG-BUILD-FAILED; echo MAIN8LOG-FAILED; exit 1
fi
echo "==== MAIN8LOG-BUILD-OK $(date -u +%FT%TZ) ===="

# ── leg 2: guard the 2b reference, then run main's 8b chat (the dump rides
#    it: the patch writes /work/fb-chat-main/logits-{n}.bin) ──
mkdir -p "$REF2B/2b-refs" "$REF8B"
mv "$REF2B"/logits-*.bin "$REF2B/2b-refs/" || { echo "cannot guard 2b logits"; echo MAIN8LOG-FAILED; exit 1; }
echo "==== MAIN8LOG-2BREFS-GUARDED ($(ls "$REF2B/2b-refs" | wc -l) files) $(date -u +%FT%TZ) ===="

echo "==== MAIN8LOG-RUN-START $(date -u +%FT%TZ) ===="
"$SCR" chat -m RedHatAI/granite-3.1-8b-instruct-FP8-dynamic --device spyre \
  -q "$PROMPT" --max-tokens 64 \
  > "$REF8B/main8-greedy-chat.run" 2> "$REF8B/main8-greedy-chat.err"
rc=$?
echo "==== MAIN8LOG-RUN rc=$rc $(date -u +%FT%TZ) ===="

# ── restore the 2b refs and bank the 8b dump, whatever the run's rc ──
if ls "$REF2B"/logits-*.bin >/dev/null 2>&1; then
  mv "$REF2B"/logits-*.bin "$REF8B/"
  echo "==== MAIN8LOG-DUMP-BANKED $(ls "$REF8B"/logits-*.bin 2>/dev/null | wc -l) files ===="
fi
mv "$REF2B"/2b-refs/logits-*.bin "$REF2B/" && rmdir "$REF2B/2b-refs" 2>/dev/null
echo "==== MAIN8LOG-2BREFS-RESTORED ($(ls "$REF2B"/logits-*.bin 2>/dev/null | wc -l) files) ===="

if [ "$rc" -ne 0 ]; then
  echo MAIN8LOG-RUN-FAILED; tail -30 "$REF8B/main8-greedy-chat.err"; echo MAIN8LOG-FAILED; exit 1
fi
[ -s "$REF8B/logits-0000.bin" ] || { echo "MAIN8LOG-DUMP-MISSING (no $REF8B/logits-0000.bin)"; echo MAIN8LOG-FAILED; exit 1; }
ls -la "$REF8B"/logits-*.bin | head -5
tail -5 "$REF8B/main8-greedy-chat.err"
echo "==== main8-logits-leg end $(date -u +%FT%TZ) ===="
echo MAIN8LOG-DONE
