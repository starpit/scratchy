#!/usr/bin/env bash
# main vs the fusion-pass branch (wave order + the anchor-epilogue fold): builds both in their own
# worktree and target dir under .fusion-ab/, then runs main / pass pairs whose order flips every
# round (4 rounds per model, 400 greedy tokens); prints each run's ITL. Run from a scratchy
# checkout; nothing else should be using the GPU.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
W=.fusion-ab
git fetch -q https://github.com/AI-native-Systems-Research/scratchy "main:refs/fusion-ab/main"
git fetch -q https://github.com/starpit/scratchy "fusion-pass:refs/fusion-ab/pass"
for t in main pass; do
  [ -d "$W/$t" ] || git worktree add -q --detach "$W/$t" "refs/fusion-ab/$t"
  git -C "$W/$t" checkout -q --detach "refs/fusion-ab/$t"
  (cd "$W/$t" && CARGO_TARGET_DIR="../target-$t" cargo build --release -q -p scratchy-cli \
    --features metal,model/qwen3.6-35b-a3b,quant/mlx-affine-b4-g64-qembed)
  cp "$W/target-$t/release/scr" "$W/scr-$t-q36"
  (cd "$W/$t" && CARGO_TARGET_DIR="../target-$t" cargo build --release -q -p scratchy-cli \
    --features metal,model/llama-3.2-3b,model/qwen2.5-3b,model/gemma-4-26b-a4b-it,quant/mlx-affine-b4-g64)
  cp "$W/target-$t/release/scr" "$W/scr-$t-dense"
done
P="Write a detailed story of at least 900 words about a lighthouse keeper on a remote island who keeps a journal through a six-week winter storm."
run() { # tag model
  echo "== $2"
  for i in 1 2 3 4; do
    order="main pass"
    [ $((i % 2)) -eq 0 ] && order="pass main"
    for t in $order; do
      printf '%-5s ' "$t"
      "$W/scr-$t-$1" chat -m "$2" --device metal -q "$P" --bench --max-tokens 400 --temperature 0 2>&1 \
        | grep -o 'ITL p10/p50/p90 : [0-9. /]*ms'
    done
  done
}
run q36 mlx-community/Qwen3.6-35B-A3B-4bit
run dense mlx-community/gemma-4-26b-a4b-it-4bit
run dense mlx-community/Llama-3.2-3B-Instruct-4bit
run dense mlx-community/Qwen2.5-3B-Instruct-4bit
echo "Clean up after: git worktree remove $W/main && git worktree remove $W/pass && rm -rf $W"
