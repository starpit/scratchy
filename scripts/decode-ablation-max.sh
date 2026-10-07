#!/usr/bin/env bash
# PROBE ONLY. Where a decode step's time goes on this Mac (written for the M5 Max): builds the
# probe/decode-ablation-main branch (main + a hook that skips kernels by name or grid), serves
# Qwen3.6-35B-A3B and Gemma 4 26B-A4B, and times a 300-token greedy request (EOS ignored) with
# nothing skipped and with each kernel class skipped. The text is garbage when something is
# skipped; the time is the step without that work. Two rounds. Prints each run's average ITL (the
# server's own number). Run from a scratchy checkout; nothing else should be using the GPU.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
W=.decode-ablation-main
git fetch -q https://github.com/starpit/scratchy \
  probe/decode-ablation-main:refs/decode-ablation-main/probe
[ -d "$W/tree" ] || git worktree add -q --detach "$W/tree" refs/decode-ablation-main/probe
git -C "$W/tree" checkout -q --detach refs/decode-ablation-main/probe
build() { # tag features
  (cd "$W/tree" && CARGO_TARGET_DIR=../target cargo build --release -q -p scratchy-cli --features "$2")
  cp "$W/target/release/scr" "$W/scr-$1"
}
build q36 metal,serve,model/qwen3.6-35b-a3b,quant/mlx-affine-b4-g64-qembed
build gemma metal,serve,model/gemma-4-26b-a4b-it,quant/mlx-affine-b4-g64
P="Write a detailed story of at least 900 words about a lighthouse keeper on a remote island who keeps a journal through a six-week winter storm."
ask() { # port model tokens
  curl -sf "http://127.0.0.1:$1/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$2\",\"messages\":[{\"role\":\"user\",\"content\":\"$P\"}],\"max_tokens\":$3,\"temperature\":0,\"ignore_eos\":true}" \
    > /dev/null
}
run() { # tag model label skip-set
  local port=8331 log="$W/serve.log"
  SCRATCHY_DIAG_SKIP="$4" "$W/scr-$1" serve "$2" --device metal --host 127.0.0.1 --port $port \
    --no-prefix-caching > "$log" 2>&1 &
  local pid=$!
  for _ in $(seq 1 300); do curl -sf "http://127.0.0.1:$port/v1/models" > /dev/null && break; sleep 1; done
  ask $port "$2" 60
  ask $port "$2" 300
  kill -TERM $pid
  wait $pid 2>/dev/null || true
  printf '%-38s %s\n' "$3" "$(grep -a 'completion_tokens=300' "$log" | grep -o 'avg_itl_ms="[0-9.]*"' | tail -1)"
}
Q=mlx-community/Qwen3.6-35B-A3B-4bit
G=mlx-community/gemma-4-26b-a4b-it-4bit
for i in 1 2; do
  echo "== round $i: $Q"
  run q36 $Q none ""
  run q36 $Q lm_head AffineQmvFast@1x31040x1
  run q36 $Q moe_experts MoeGateUpAct,MoeDownCombine
  run q36 $Q router NormedGemv
  run q36 $Q shared_expert_gate_up AffineQmvGated@1x64x1,AffineQmv@1x1x1
  run q36 $Q gdn_kernel GatedDeltaNet
  run q36 $Q "rows_8192 (gdn in, attn q)" AffineQmvFast@1x1024x1
  run q36 $Q "rows_2048 (gdn out, attn o, shared dn)" AffineQmvFast@1x256x1
  run q36 $Q "rows_4096" AffineQmvFast@1x512x1
  run q36 $Q "small_proj (32 / 512 rows)" AffineQmvFast@1x4x1,AffineQmvFast@1x64x1
  run q36 $Q attention AttentionDecodeGqaTq,AttentionDecodeCombine,GateSplit,GateApply,RmsNorm
  echo "== round $i: $G"
  run gemma $G none ""
  run gemma $G lm_head AffineQmvFast@1x32768x1
  run gemma $G moe_experts MoeGateUpAct,MoeDownCombine
  run gemma $G router NormedGemv
  run gemma $G q_proj AffineQmvFast@1x512x1,AffineQmvFast@1x1024x1
  run gemma $G k+v_proj AffineQmvFast@1x256x1,AffineQmvFast@1x128x1
  run gemma $G "rows_2816 (o_proj, dense down)" AffineQmvFast@1x352x1
  run gemma $G dense_gate_up AffineQmvGated@1x264x1
  run gemma $G row_programs RowProgram
  run gemma $G sliding_attention AttentionViaCache,RopeAppendNormed
  run gemma $G global_attention AttentionDecodeGqaTq,AttentionDecodeCombine,RmsNorm,RmsNormUnit
done
echo "Clean up after: git worktree remove $W/tree && rm -rf $W"
