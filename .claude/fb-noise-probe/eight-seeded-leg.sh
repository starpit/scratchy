#!/bin/bash
# front-b-noise 8B FP8 SEEDED LEG (nickm2) — the auditor's order 2026-10-27,
# on the rebased tree (90cc33e893): "8b seeded P3 (the pair ×2,
# deterministic) and P5 (200 draws within 3σ of main's 8b logits)". Two parts:
#   P3. 2-CONCURRENT, seeds 1234 + 9999, temp 1.0, 64 tokens, the pair run
#       TWICE — the gate is each request's text IDENTICAL across the two pair
#       runs (ruling (b), the 2b precedent: the batched-numerics class — the
#       rows=2 rung is a different baked tape than solo, so the pair is gated
#       against ITSELF, never equated with solo; pair-vs-solo is info only).
#   P5. 200 seeded single-token draws (seeds 100000..100199, temp 1.0) —
#       each top-10 token's frequency vs softmax(MAIN's 8b host logits)
#       within 3 sigma (main8-logits-leg.sh banks the dump at
#       /work/fb-chat-main8/logits-0000.bin).
# Tree: /work/front-b-noise (the REBASED tip — rsync before firing). Build:
# ON THE POD, release, front-b's OWN warm target dir /work/front-b/target.
# Server port 8793, debug log for the seat/rung evidence. Markers (kept so
# target/main8-watch.sh's grep matches): EIGHT-DONE / EIGHT-FAILED. Log:
# /work/eight-seeded-leg.log — a FRESH log file, so no earlier run's terminal
# markers can stale-cursor a watch.
set -u

LOG=/work/eight-seeded-leg.log
TREE=/work/front-b-noise
TDIR=/work/front-b/target
PROBE=/work/fb-noise-probe
REF8B=/work/fb-chat-main8
SCR=$TDIR/release/scr
PORT=8793
PROMPT="Write two sentences about the ocean."
DRAWS=200
SEED0=100000

if [ "${BG:-0}" != "1" ]; then
  BG=1 setsid nohup bash "$0" >>"$LOG" 2>&1 &
  echo "detached pid $! ; log $LOG"
  exit 0
fi
exec >>"$LOG" 2>&1

echo "==== eight-seeded-leg start $(date -u +%FT%TZ) ===="

# ── busy check (live cargo only; zombies are the pod's normal litter) ──
if pgrep -x cargo >/dev/null 2>&1; then
  for p in $(pgrep -x cargo); do
    if ! ps -o stat= -p "$p" 2>/dev/null | grep -q Z; then
      echo "POD BUSY (live cargo pid $p)"; echo EIGHT-FAILED; exit 1
    fi
  done
fi
if pgrep -f 'scr serve|scr chat' >/dev/null 2>&1; then
  echo "POD BUSY (scr)"; echo EIGHT-FAILED; exit 1
fi

[ -d "$TREE/crates" ] || { echo "no tree at $TREE (rsync first)"; echo EIGHT-FAILED; exit 1; }
[ -s "$REF8B/logits-0000.bin" ] || { echo "no main 8b logits dump at $REF8B (run main8-logits-leg first)"; echo EIGHT-FAILED; exit 1; }
cd "$TREE" || { echo "cannot cd $TREE"; echo EIGHT-FAILED; exit 1; }

export PATH=/work/.cargo/bin:$PATH
export CARGO_TARGET_DIR=$TDIR
export LD_LIBRARY_PATH=/opt/ibm/spyre/senlib/lib:/opt/ibm/spyre/runtime/lib:/opt/ibm/spyre/deeptools/lib
export DEEPTOOLS_PATH=/opt/ibm/spyre/deeptools/share
export HF_HUB_OFFLINE=1

# ── leg 1: the REBASED tree's 8b chat stack (release, warm dir) ──
echo "==== EIGHT-SE-BUILD-START $(date -u +%FT%TZ) ===="
if ! cargo build --release -p scratchy-cli \
     --features spyre-hw,serve,model/granite-3.1-8b-instruct,quant/fp8-dynamic-per-channel; then
  echo EIGHT-SE-BUILD-FAILED; echo EIGHT-FAILED; exit 1
fi
echo "==== EIGHT-SE-BUILD-OK $(date -u +%FT%TZ) ===="

# ── leg 2: the server, at debug so the seat + rung lines land in the log ──
SRVLOG=$PROBE/eight-seeded-server.log
echo "==== EIGHT-SE-SERVE-START $(date -u +%FT%TZ) ===="
setsid nohup "$SCR" serve -m RedHatAI/granite-3.1-8b-instruct-FP8-dynamic --device spyre \
  --port $PORT --log-level debug > "$SRVLOG" 2>&1 &
SRVPID=$!
for i in $(seq 1 240); do
  if curl -s -o /dev/null "http://127.0.0.1:$PORT/v1/models"; then break; fi
  # A DEAD SERVER ENDS THE LEG — never poll a corpse for 20 minutes.
  if ! kill -0 $SRVPID 2>/dev/null; then break; fi
  sleep 5
done
if ! curl -s -o /dev/null "http://127.0.0.1:$PORT/v1/models"; then
  if ! kill -0 $SRVPID 2>/dev/null; then
    echo "EIGHT-SE-SERVE-DIED"; tail -30 "$SRVLOG"; echo EIGHT-FAILED; exit 1
  fi
  echo "EIGHT-SE-SERVE-NEVER-READY"; kill $SRVPID 2>/dev/null; tail -30 "$SRVLOG"; echo EIGHT-FAILED; exit 1
fi
echo "==== EIGHT-SE-SERVE-READY $(date -u +%FT%TZ) ===="

cleanup() { kill $SRVPID 2>/dev/null; sleep 2; kill -9 $SRVPID 2>/dev/null; }
trap cleanup EXIT

ask() {  # ask <seed> <temp> <max-tokens> <outfile>
  curl -s http://127.0.0.1:$PORT/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d "{\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"temperature\":$2,\"seed\":$1,\"max_tokens\":$3}" \
    > "$4"
}

# ── P3: 2-CONCURRENT, the two seeds at once — the pair run TWICE (ruling
#    (b)): the gate is pair-run-a == pair-run-b per request; the pair vs the
#    solo texts is information only. ──
echo "==== EIGHT-SE-P3-START $(date -u +%FT%TZ) ===="
for RUN in a b; do
  ask 1234 1.0 64 "$PROBE/eight-p3-seed1234-$RUN.json" &
  P3A=$!
  ask 9999 1.0 64 "$PROBE/eight-p3-seed9999-$RUN.json" &
  P3B=$!
  wait $P3A $P3B
done
echo "==== EIGHT-SE-P3-DONE (pair x2, $(date -u +%FT%TZ)) ===="

# ── P5: 200 seeded single-token draws ──
echo "==== EIGHT-SE-P5-START $(date -u +%FT%TZ) ===="
rm -f "$PROBE/eight-p5-draws.txt"
i=0
while [ "$i" -lt "$DRAWS" ]; do
  SEED=$((SEED0 + i))
  ask "$SEED" 1.0 1 "$PROBE/eight-p5-draw.json" || { echo "EIGHT-SE-P5-FAILED (draw $i)"; echo EIGHT-FAILED; exit 1; }
  python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); print(json.dumps(r["choices"][0]["message"]["content"]))' \
    "$PROBE/eight-p5-draw.json" >> "$PROBE/eight-p5-draws.txt" || { echo "EIGHT-SE-P5-PARSE-FAILED (draw $i)"; cat "$PROBE/eight-p5-draw.json"; echo EIGHT-FAILED; exit 1; }
  i=$((i + 1))
done
echo "==== EIGHT-SE-P5-DONE ($DRAWS draws, $(date -u +%FT%TZ)) ===="

cleanup
trap - EXIT

# ── the verdict, on the pod ──
echo "==== EIGHT-SE-CHECK $(date -u +%FT%TZ) ===="
python3 "$PROBE/eight_seeded_stats.py"
rc=$?
echo "==== eight-seeded-leg end $(date -u +%FT%TZ) checkrc=$rc ===="
if [ "$rc" -ne 0 ]; then echo EIGHT-FAILED; exit 1; fi
echo EIGHT-DONE
