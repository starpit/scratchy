#!/usr/bin/env bash
# Decode levels on the M5 Max, each commit of starpit/scratchy `moe-levels` against upstream main:
#   levels  08396da38  independent decode work between the same barriers (wave order) + the shared
#                      expert and the residual added as the expert combine stores its rows
#   gdn     48edbe24a  + a GatedDeltaNet decode token in one command (Qwen3.6: -60 barriers)
#   norm    24ecbea2a  + the MoE block's input norm folded into its router and experts (-40)
# Builds each in its own worktree and target dir under .max-levels/, then runs the legs in an order
# that rotates every round (3 rounds, 400 greedy tokens) and prints each run's ITL p10/p50/p90,
# then whether each leg's text matches main's. The dense models run main vs norm. Run from a
# scratchy checkout; nothing else should be using the GPU.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
W=.max-levels
git fetch -q https://github.com/AI-native-Systems-Research/scratchy main:refs/max-levels/main
git fetch -q https://github.com/starpit/scratchy moe-levels:refs/max-levels/moe-levels
ref() {
  case $1 in
    main) echo refs/max-levels/main ;;
    levels) echo 08396da38 ;;
    gdn) echo 48edbe24a ;;
    norm) echo 24ecbea2a ;;
  esac
}
for t in main levels gdn norm; do
  [ -d "$W/$t" ] || git worktree add -q --detach "$W/$t" "$(ref $t)"
  git -C "$W/$t" checkout -q --detach "$(ref $t)"
done
build() { # tree tag features
  (cd "$W/$1" && CARGO_TARGET_DIR="../target-$1" cargo build --release -q -p scratchy-cli --features "$3")
  cp "$W/target-$1/release/scr" "$W/scr-$1-$2"
}
for t in main levels gdn norm; do
  build "$t" q36 metal,model/qwen3.6-35b-a3b,quant/mlx-affine-b4-g64-qembed
done
for t in main norm; do
  build "$t" dense metal,model/llama-3.2-3b,model/qwen2.5-3b,model/gemma-4-26b-a4b-it,quant/mlx-affine-b4-g64
done
P="Write a detailed story of at least 900 words about a lighthouse keeper on a remote island who keeps a journal through a six-week winter storm."
run() { # tag model legs...
  local tag=$1 m=$2
  shift 2
  local legs=("$@")
  echo "== $m"
  for i in 1 2 3; do
    local n=${#legs[@]} order=()
    for j in $(seq 0 $((n - 1))); do order+=("${legs[$(((i + j) % n))]}"); done
    for t in "${order[@]}"; do
      printf '%-7s ' "$t"
      "$W/scr-$t-$tag" chat -m "$m" --device metal -q "$P" --bench --max-tokens 400 --temperature 0 \
        > "$W/out-$t.raw" 2>&1
      grep -o 'ITL p10/p50/p90 : [0-9. /]*ms' "$W/out-$t.raw"
      sed '/^--- bench ---/,$d' "$W/out-$t.raw" | grep -v '^\[\|INFO\|WARN\|Using model' > "$W/out-$t.txt"
    done
  done
  for t in "${legs[@]}"; do
    [ "$t" = main ] && continue
    if cmp -s "$W/out-main.txt" "$W/out-$t.txt"; then echo "$t text: same as main"; else echo "$t text: differs from main"; fi
  done
}
run q36 mlx-community/Qwen3.6-35B-A3B-4bit main levels gdn norm
run dense mlx-community/gemma-4-26b-a4b-it-4bit main norm
run dense mlx-community/Llama-3.2-3B-Instruct-4bit main norm
run dense mlx-community/Qwen2.5-3B-Instruct-4bit main norm
echo "Clean up after: for t in main levels gdn norm; do git worktree remove $W/\$t; done; rm -rf $W"
