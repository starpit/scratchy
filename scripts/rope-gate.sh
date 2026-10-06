#!/usr/bin/env bash
# Gate for folding rope + KV append into decode attention: `main` vs a probe build whose decode
# steps skip the rope + append dispatch entirely (its output is wrong; its time is the most the fold
# could save). Builds both in their own worktree and target dir under .rope-gate/, then runs main /
# probe pairs whose order flips every round; prints each run's ITL. Dense models only: skipping the
# step corrupts activations, which reroutes a MoE model's experts. Run from a scratchy checkout;
# nothing else should be using the GPU.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
W=.rope-gate
git fetch -q https://github.com/AI-native-Systems-Research/scratchy "main:refs/rope-gate/main"
git fetch -q https://github.com/starpit/scratchy "rope-skip-probe:refs/rope-gate/probe"
for t in main probe; do
  [ -d "$W/$t" ] || git worktree add -q --detach "$W/$t" "refs/rope-gate/$t"
  git -C "$W/$t" checkout -q --detach "refs/rope-gate/$t"
  (cd "$W/$t" && CARGO_TARGET_DIR="../target-$t" cargo build --release -q -p scratchy-cli \
    --features metal,model/llama-3.2-3b,model/qwen2.5-3b,quant/mlx-affine-b4-g64)
  cp "$W/target-$t/release/scr" "$W/scr-$t"
done
P="Write a detailed story of at least 900 words about a lighthouse keeper on a remote island who keeps a journal through a six-week winter storm."
for m in mlx-community/Llama-3.2-3B-Instruct-4bit mlx-community/Qwen2.5-3B-Instruct-4bit; do
  echo "== $m"
  for i in 1 2 3 4; do
    order="main probe"
    [ $((i % 2)) -eq 0 ] && order="probe main"
    for t in $order; do
      printf '%-6s ' "$t"
      "$W/scr-$t" chat -m "$m" --device metal -q "$P" --bench --max-tokens 400 --temperature 0 2>&1 \
        | grep -o 'ITL p10/p50/p90 : [0-9. /]*ms'
    done
  done
done
echo "Clean up after: git worktree remove $W/main && git worktree remove $W/probe && rm -rf $W"
