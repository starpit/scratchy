#!/usr/bin/env bash
# Probe branch only: per-launch GPU time of the Llama-3.2-3B (MLX 4-bit, TurboQuant) decode,
# dispatch vs segmented, at several streaming item widths. Needs the checkpoint in the local Hub
# cache (`scr model pull mlx-community/Llama-3.2-3B-Instruct-4bit`) and a quiet GPU.
#   scripts/mk-probe-sweep.sh [width...]      (default: default 512 1024 128)
# Writes mk-probe-<width>.log in the current directory and prints one PROBETOTAL line per width.
set -uo pipefail
widths=("$@")
[ ${#widths[@]} -gt 0 ] || widths=(default 512 1024 128)
feats=metal,turboquant,llama-3.2-3b,scratchy-quantizations/mlx-affine-b4-g64
for w in "${widths[@]}"; do
  if [ "$w" = default ]; then unset MK_PROBE_STREAM_WIDTH; else export MK_PROBE_STREAM_WIDTH=$w; fi
  cargo test --release -p scratchy-models --features "$feats" --test metal_o2_logits \
    -- --ignored --exact --nocapture --test-threads=1 perf::probe_llama_3_2_3b_mlx \
    > "mk-probe-$w.log" 2>&1
  echo "width $w: $(grep -o 'PROBETOTAL.*' "mk-probe-$w.log" || echo 'no PROBETOTAL — see the log')"
done
