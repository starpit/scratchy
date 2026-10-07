#!/usr/bin/env bash
# PROBE ONLY. Where a decode step's time goes on this Mac (written for the M5 Max): builds the
# probe/decode-ablation branch (main + a hook that skips kernels by name), serves Qwen3.6-35B-A3B
# and Gemma 4 26B-A4B, and times a 300-token greedy request (EOS ignored) with nothing skipped and
# with each kernel class skipped. Output text is garbage when something is skipped; the time is
# the step without that work. Two rounds. Prints each run's average ITL (the server's own number).
# Run from a scratchy checkout; nothing else should be using the GPU.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
W=.decode-ablation
git fetch -q https://github.com/starpit/scratchy probe/decode-ablation:refs/decode-ablation/probe
[ -d "$W/tree" ] || git worktree add -q --detach "$W/tree" refs/decode-ablation/probe
git -C "$W/tree" checkout -q --detach refs/decode-ablation/probe
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
  printf '%-34s %s\n' "$3" "$(grep -a 'completion_tokens=300' "$log" | grep -o 'avg_itl_ms="[0-9.]*"' | tail -1)"
}
Q=mlx-community/Qwen3.6-35B-A3B-4bit
G=mlx-community/gemma-4-26b-a4b-it-4bit
for i in 1 2; do
  echo "== round $i: $Q"
  run q36 $Q none ""
  run q36 $Q lm_head AffineQmvFast@1x31040x1
  run q36 $Q qkv+q_proj AffineQmvFast@1x1024x1
  run q36 $Q moe_experts MoeGateUpAct,MoeDownCombine
  run q36 $Q moe_down MoeDownCombine
  run q36 $Q router NormedGemv
  run q36 $Q gdn GatedDeltaNet
  run q36 $Q attention AttentionViaCacheTq
  run q36 $Q attn_small GateSplit,RmsNorm,GateApply
  run q36 $Q shared_expert AffineQmv,AffineQmvGated
  echo "== round $i: $G"
  run gemma $G none ""
  run gemma $G lm_head AffineQmv@1x32768x1
  run gemma $G q_proj AffineQmv@1x512x1,AffineQmv@1x1024x1
  run gemma $G k+v_proj AffineQmv@1x256x1,AffineQmv@1x128x1
  run gemma $G o_proj AffineQmvFast@1x352x1
  run gemma $G moe_experts MoeGateUpAct,MoeDownCombine
  run gemma $G dense_mlp AffineQmvGated@1x264x1,AffineQmv@1x352x1
  run gemma $G row_programs RowProgram
  run gemma $G attention AttentionViaCache,AttentionViaCacheTq
  run gemma $G router NormedGemv
  run gemma $G all_small RowProgram,AttentionViaCache,AttentionViaCacheTq,NormedGemv,RmsNorm,RmsNormUnit,TanhSoftCap
done
echo "Clean up after: git worktree remove $W/tree && rm -rf $W"
