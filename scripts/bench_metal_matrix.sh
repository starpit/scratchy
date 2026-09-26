#!/usr/bin/env bash
# Metal benchmark matrix — one Mac in, one JSON out.
#
#     scripts/bench_metal_matrix.sh                 # every model, every rung
#     scripts/bench_metal_matrix.sh --models llama-3.2-3b
#     scripts/bench_metal_matrix.sh --scenarios cold,warm   # no sudo needed
#
# It is a RUNNER, not a measurement tool. Every number comes from a `scr bench`
# subcommand; this script only decides what to run, in what order, and merges the
# results into one file. Two consequences worth knowing:
#
#   * It never starts or stops a server. `scr bench startup --exec` spawns and
#     reaps its own children, so this script cannot leak one. (Exception: the
#     scaling stage and the warm stage below each run ONE resident server per
#     model, solo, torn down before the next starts — the same rule
#     bench_serve_compare.sh bakes in.)
#   * It adds exactly one measurement of its own: BUILD TIME and BINARY SIZE per model. scratchy
#     compiles one model into the binary, so size is a per-model fact, and the
#     build is the cost side of doing that work ahead of time. Every startup
#     number below excludes it, which is precisely why it has to be reported.
#
# WHAT YOU GET, AND WHICH COMMAND PRODUCES IT
#
#   build seconds, binary MiB          this script (`cargo build`, `stat`)
#   ttft_exec: frozen / cold / warm    scr bench startup --exec
#   t_ready, ttft_from_send, tpot      scr bench startup --exec
#   peak RSS, major faults             scr bench startup --exec  (per child, wait4)
#   TTFT/TPOT/ITL p50+p99, tok/s       scr bench serve
#   scaling curves: tok/s, TPOT, TTFT  scr bench serve   (served axes, see below)
#   offline batch scaling: tok/s, lat  scr bench latency + mlx-lm BatchGenerator
#
# The cache ladder, the eviction proof, the priming launch, per-child resource
# attribution and the validity gates all live in `--exec`. See
# docs/BENCHMARKING.md for what each rung means.
#
# REQUIREMENTS
#   * `--scenarios frozen,...` needs `sudo` (macOS `purge`). Run `sudo -v` first,
#     or pass `--scenarios cold,warm` and leave the frozen cells blank.
#   * Weights are downloaded on first use unless you pass --offline.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ROOT="$(cd -- "${HERE}/.." &>/dev/null && pwd)"

# Every id verified against the HuggingFace API with its download size.
# The quant column is the FEATURE to enable (`quant/<name>`), not the preset
# to synthesize: `mlx` is the umbrella feature for every MLX affine preset,
# and the arch's own quantizations.json ∩ SCRATCHY_QUANTS decides which
# variants actually compile for the selected model — one per (base, preset)
# the arch declares. `mlx` here rather than a per-model preset keeps the
# build line uniform across models whose checkpoints sit on different
# group sizes (granite 4-bit builds are g32, most others g64) — and the
# variant that serves the requested HF repo is the one `scr serve` loads.
MODELS_DEFAULT=(
  "granite-3.3-2b-instruct=mlx-community/granite-3.3-2b-instruct-4bit:mlx"  #  1.4 GB
  "llama-3.2-3b=mlx-community/Llama-3.2-3B-Instruct-4bit:mlx"               #  1.8 GB
  "granite-3.3-8b-instruct=mlx-community/granite-3.3-8b-instruct-4bit:mlx"  #  4.6 GB
  "qwen2.5-7b=mlx-community/Qwen2.5-7B-Instruct-4bit:mlx"                   #  4.3 GB
  "gemma-4-26b-a4b-it=mlx-community/gemma-4-26b-a4b-it-4bit:mlx"            # 15.4 GB
  "gemma-4-31b-it=mlx-community/gemma-4-31b-it-4bit:mlx"                    # 18.4 GB
  "qwen3.5-35b-a3b=mlx-community/Qwen3.5-35B-A3B-4bit:mlx"                  # 20.4 GB
  # Frontier-ish: Moonlight is a DeepSeek-V3-architecture MoE (16B total, ~3B
  # active), so it exercises the MLA + MoE path at a size a 36 GB machine can
  # host. Listed twice on purpose — same model, two quantizations. deepseek-v3
  # declares only ggml, so the mlx umbrella synthesizes nothing there and the
  # dense base serves the 4-bit repo; the fp8-block row needs its own feature
  # because fp8-block-128x128 is outside the mlx umbrella.
  "moonlight-16b-a3b-instruct=mlx-community/Moonlight-16B-A3B-Instruct-4-bit:mlx"        #  9.0 GB
  "moonlight-16b-a3b-instruct-fp8-block=starpit/moonlight-16b-a3b-instruct-fp8-block:fp8-block-128x128" # 16.7 GB
  # DeepSeek-V2-Lite has no public 4-bit MLX build, so it would be bf16 at
  # 31.4 GB and exhaust a 36 GB machine once the KV cache is allocated. The
  # DeepSeek-V2 arch is instead reachable via
  # mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx (8.8 GB) if wanted.
)

MODELS=()
SCENARIOS="frozen,cold,warm"
REPS=3
PORT=8751
NUM_PROMPTS=20
INPUT_LEN=64
OUTPUT_LEN=32
WARM_REQUESTS=20
OFFLINE=0
SKIP_BUILD=0
OUT_DIR=""
KV_CACHE_DTYPE=""
# ---- scaling stage ----------------------------------------------------------
# One axis at a time from a base cell, NOT a full factorial: 8 models × 6 conc
# rungs × 4 input rungs × 4 output rungs is ~200 cells and none of the 3-way
# cells would tell you anything the marginal curves don't. What a scaling
# benchmark plots is the MARGINAL curve per axis, others pinned at base.
SCALING=1
CONCURRENCIES="1,2,4,8,16,32"
INPUT_LENS="128,512,2048,8192"
OUTPUT_LENS="16,64,256,1024"
BASE_INPUT=512
BASE_OUTPUT=128
BASE_CONCURRENCY=8
# A cell's num_prompts: enough requests that concurrency is actually offered
# for a while, without making the input-len=8192 cells take forever.
NUM_PROMPTS_SCALING=0   # 0 = auto: max(24, 3*concurrency)
WARMUPS=1
# Unique seed per cell, one seed space per model: same --seed means identical
# random prompts, so the prefix cache serves later cells and TTFT collapses
# toward 0 — this bit before (see bench_serve_compare.sh's BENCH LESSONS).
# Seeds are stable across models (model index salts the space) so a re-run of
# one model reproduces its own cells.
SEED_BASE=1000
# ---- offline batch axis ------------------------------------------------------
# The served concurrency axis measures scheduler + kernels together (the
# batch is OFFERED, the scheduler decides). This one COMMANDS the width: no
# server, `scr bench latency -b <rungs>` on our side and mlx-lm's
# BatchGenerator with completion_batch_size pinned on theirs — so the two
# curves differ in exactly one thing: the batching kernels.
BATCH_SIZES="1,2,4,8,16,32"
NUM_ITERS=10
NUM_ITERS_WARMUP=3
BATCH_AXIS=1
# `--exec` picks purge on macOS and fadvise on Linux by itself
EVICT=""
EVICT_PATH=()
SETTLE_S=""
# `--exec` defaults to 600 s, too tight for a ~20 GB checkpoint on this class of
# machine: gemma-4-31b-it (18.4 GB) timed out at 600 s.
READY_TIMEOUT_S="1800"
SEED=""
# Setting --mlx-python turns on the comparison column: the same ladder is run a
# second time with --backend mlx-lm.
MLX_PYTHON=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --models)         IFS=',' read -r -a MODELS <<<"$2"; shift 2 ;;
        --scenarios)      SCENARIOS="$2"; shift 2 ;;
        --reps)           REPS="$2"; shift 2 ;;
        --port)           PORT="$2"; shift 2 ;;
        --num-prompts)    NUM_PROMPTS="$2"; shift 2 ;;
        --input-len)      INPUT_LEN="$2"; shift 2 ;;
        --output-len)     OUTPUT_LEN="$2"; shift 2 ;;
        --warm-requests)  WARM_REQUESTS="$2"; shift 2 ;;
        --kv-cache-dtype) KV_CACHE_DTYPE="$2"; shift 2 ;;
        --evict)          EVICT="$2"; shift 2 ;;
        --evict-path)     EVICT_PATH+=("$2"); shift 2 ;;
        --settle-s)       SETTLE_S="$2"; shift 2 ;;
        --ready-timeout-s) READY_TIMEOUT_S="$2"; shift 2 ;;
        --seed)           SEED="$2"; shift 2 ;;
        --mlx-python)     MLX_PYTHON="$2"; shift 2 ;;
        --offline)        OFFLINE=1; shift ;;
        --skip-build)     SKIP_BUILD=1; shift ;;
        --out-dir)        OUT_DIR="$2"; shift 2 ;;
        --no-scaling)     SCALING=0; shift ;;
        --concurrencies)  CONCURRENCIES="$2"; shift 2 ;;
        --input-lens)     INPUT_LENS="$2"; shift 2 ;;
        --output-lens)    OUTPUT_LENS="$2"; shift 2 ;;
        --base-input)     BASE_INPUT="$2"; shift 2 ;;
        --base-output)    BASE_OUTPUT="$2"; shift 2 ;;
        --base-concurrency) BASE_CONCURRENCY="$2"; shift 2 ;;
        --num-prompts-scaling) NUM_PROMPTS_SCALING="$2"; shift 2 ;;
        --warmups)        WARMUPS="$2"; shift 2 ;;
        --seed-base)      SEED_BASE="$2"; shift 2 ;;
        --batch-sizes)    BATCH_SIZES="$2"; shift 2 ;;
        --num-iters)      NUM_ITERS="$2"; shift 2 ;;
        --num-iters-warmup) NUM_ITERS_WARMUP="$2"; shift 2 ;;
        --no-batch-axis)  BATCH_AXIS=0; shift ;;
        -h|--help)        grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)                echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done
[[ ${#MODELS[@]} -eq 0 ]] && MODELS=("${MODELS_DEFAULT[@]}")
(( OFFLINE )) && export HF_HUB_OFFLINE=1

# The machine block below (a python heredoc) reads the scaling config through
# the environment rather than a growing argv. ASSIGNED then EXPORTED — a
# `VAR=x cmd` prefix scopes to that one command, and the heredoc python is a
# separate process that would never see them; a bare `export VAR` with no
# assignment exports nothing.
SM_CONC="${CONCURRENCIES}"; SM_INPUT_LENS="${INPUT_LENS}"; SM_OUTPUT_LENS="${OUTPUT_LENS}"
SM_BASE_IN="${BASE_INPUT}"; SM_BASE_OUT="${BASE_OUTPUT}"; SM_BASE_CONC="${BASE_CONCURRENCY}"
SM_BATCH="${BATCH_SIZES}"
export SM_CONC SM_INPUT_LENS SM_OUTPUT_LENS SM_BASE_IN SM_BASE_OUT SM_BASE_CONC SM_BATCH

BIN="${ROOT}/target/release/scr"
chip="$(sysctl -n machdep.cpu.brand_string)"
slug="$(echo "${chip}" | tr '[:upper:] ' '[:lower:]-' | sed 's/[^a-z0-9-]//g')"
: "${OUT_DIR:="${ROOT}/bench_results/metal_matrix"}"
RAW="${OUT_DIR}/${slug}"
mkdir -p "${RAW}"
JSON="${OUT_DIR}/${slug}.json"

# ---- preflight: fail now, with a fix, rather than three models in ------------
die() { echo "error: $*" >&2; exit 1; }

command -v cargo >/dev/null || die "cargo not on PATH"
[[ "$(uname -s)" == "Darwin" ]] || die "this runner is the metal side of #3; use the cuda/spyre runner elsewhere"

# `--exec` is what measures the cache ladder.
EXEC_OK=0
if [[ -x "${BIN}" ]] && "${BIN}" bench startup --help 2>/dev/null | grep -q -- '--exec'; then
    EXEC_OK=1
fi
if [[ ",${SCENARIOS}," == *",frozen,"* ]]; then
    sudo -n true 2>/dev/null || die "frozen needs sudo for \`purge\`: run \`sudo -v\` first, or pass --scenarios cold,warm"
fi

echo "machine : ${chip}"
echo "models  : ${#MODELS[@]}"
echo "rungs   : ${SCENARIOS}"
echo "scaling : $(( SCALING ? 1 : 0 ))"
echo "output  : ${JSON}"
(( EXEC_OK )) || cat <<EOF

EOF

# ---- machine block ----------------------------------------------------------
python3 - "${JSON}" "${chip}" "${SCENARIOS}" "${EXEC_OK}" <<'PY'
import json, os, subprocess, sys, time
out, chip, scenarios, exec_ok = sys.argv[1:5]
sh = lambda *c: subprocess.run(c, capture_output=True, text=True).stdout.strip()
sysctl = lambda k: sh("sysctl", "-n", k)
batt = sh("pmset", "-g", "batt")
json.dump({
  "schema": 2, "issue": 91, "epic": 3,
  "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
  "generator": "scripts/bench_metal_matrix.sh",
  "measured_by": {
    "footprint": "this runner (cargo build, stat)",
    "cache_ladder": "scr bench startup --exec" if exec_ok == "1" else "UNAVAILABLE",
    "warm_serving": "scr bench serve",
  },
  "machine": {
    "hw_model": sysctl("hw.model"), "chip": chip,
    "cores_total": int(sysctl("hw.ncpu") or 0),
    "cores_performance": int(sysctl("hw.perflevel0.physicalcpu") or 0),
    "cores_efficiency": int(sysctl("hw.perflevel1.physicalcpu") or 0),
    "memory_gb": round(int(sysctl("hw.memsize") or 0) / 1024**3),
    "macos": sh("sw_vers", "-productVersion") + " (" + sh("sw_vers", "-buildVersion") + ")",
    "thermal_at_start": sh("pmset", "-g", "therm").replace("\n", " "),
    "power": batt.splitlines()[0] if batt else "",
  },
  "repo": {"sha": sh("git", "rev-parse", "HEAD"),
           "branch": sh("git", "rev-parse", "--abbrev-ref", "HEAD"),
           "dirty": bool(sh("git", "status", "--porcelain"))},
  "config": {"scenarios": scenarios.split(","),
             "scaling": {"rungs": {
                "concurrency": [int(x) for x in os.environ["SM_CONC"].split(",") if x],
                "input_len":   [int(x) for x in os.environ["SM_INPUT_LENS"].split(",") if x],
                "output_len":  [int(x) for x in os.environ["SM_OUTPUT_LENS"].split(",") if x],
                "batch":       [int(x) for x in os.environ["SM_BATCH"].split(",") if x]},
                "base": {"input_len": int(os.environ["SM_BASE_IN"]),
                         "output_len": int(os.environ["SM_BASE_OUT"]),
                         "concurrency": int(os.environ["SM_BASE_CONC"])},
                "note": "one axis swept at a time, others pinned at base; "
                        "concurrency is OFFERED (max-concurrency), the effective "
                        "decode batch is the scheduler's decision and is not "
                        "recorded here; the batch axis is OFFLINE and COMMANDED "
                        "(scr bench latency -b / mlx-lm completion_batch_size) "
                        "so it isolates the batching kernels from the scheduler"}},
  "methodology": "docs/BENCHMARKING.md — rung definitions, fairness rules and disclosed asymmetries live there, not here",
  "models": [],
}, open(out, "w"), indent=2)
PY

# ---- per model --------------------------------------------------------------
for entry in "${MODELS[@]}"; do
    stem="${entry%%=*}"; rest="${entry#*=}"
    id="${rest%%:*}"; quant="${rest#*:}"; [[ "${quant}" == "${rest}" ]] && quant=""
    echo; echo "################ ${stem} ################"

    feats="metal,serve,bench,model/${stem}"
    [[ -n "${quant}" ]] && feats="${feats},quant/${quant}"
    build_secs=""; bytes=""; built=1

    if (( ! SKIP_BUILD )); then
        echo "--- build -F ${feats}"
        t0=$(date +%s)
        if cargo build --release -p scratchy-cli --features "${feats}" >"${RAW}/build-${stem}.log" 2>&1; then
            build_secs=$(( $(date +%s) - t0 ))
            bytes=$(stat -f%z "${BIN}")
            echo "    ${build_secs}s · $((bytes/1024/1024)) MiB"
        else
            built=0
            echo "    BUILD FAILED — ${RAW}/build-${stem}.log" >&2
            tail -3 "${RAW}/build-${stem}.log" | sed 's/^/    /' >&2
        fi
    fi

    # Each model needs its own binary: the next build overwrites this one.
    model_bin="${RAW}/scr-${stem}"
    (( built )) && cp "${BIN}" "${model_bin}"

    exec_json="${RAW}/exec-${stem}.json"
    exec_mlx_json="${RAW}/exec-mlx-${stem}.json"
    serve_json="${RAW}/serve-${stem}.json"
    rm -f "${exec_json}" "${exec_mlx_json}" "${serve_json}"

    if (( built && EXEC_OK )); then
        serve_cmd="${model_bin} serve ${id} --port ${PORT}"
        [[ -n "${KV_CACHE_DTYPE}" ]] && serve_cmd="${serve_cmd} --kv-cache-dtype ${KV_CACHE_DTYPE}"
        mlx_cmd=""
        [[ -n "${MLX_PYTHON}" ]] && mlx_cmd="${MLX_PYTHON} -m mlx_lm.server --model ${id} --port ${PORT}"

        # Flags shared by every --exec invocation for this model.
        common=(--model "${id}" --exec
                --mode server
                --scenarios "${SCENARIOS}"
                --reps "${REPS}"
                --port "${PORT}"
                --input-len "${INPUT_LEN}"
                --output-len "${OUTPUT_LEN}"
                --warm-requests "${WARM_REQUESTS}"
                --remove-path "${HOME}/.cache/scratchy/metal-aligned-weights")
        [[ -n "${EVICT}" ]]    && common+=(--evict "${EVICT}")
        [[ -n "${SETTLE_S}" ]] && common+=(--settle-s "${SETTLE_S}")
        [[ -n "${READY_TIMEOUT_S}" ]] && common+=(--ready-timeout-s "${READY_TIMEOUT_S}")
        [[ -n "${SEED}" ]]     && common+=(--seed "${SEED}")
        for ep in "${EVICT_PATH[@]:-}"; do [[ -n "${ep}" ]] && common+=(--evict-path "${ep}"); done

        echo "--- scr bench startup --exec, backend scratchy (${SCENARIOS})"
        # --remove-path clears the derived aligned-weights sidecar, which a FROZEN
        # rung must not find warm. --exec spawns and reaps its own children.
        parity=()
        if [[ -n "${mlx_cmd}" ]]; then
            # Blocking: a broken dequant path can be fast and wrong, so the two
            # engines must agree on the same greedy prompts before either is timed.
            parity=(--parity-cmd "${mlx_cmd}" --parity-backend mlx-lm)
        fi
        "${BIN}" bench startup "${common[@]}" \
            --child-cmd "${serve_cmd}" \
            --backend scratchy \
            ${parity[@]+"${parity[@]}"} \
            --output-json "${exec_json}" \
            2>&1 | tail -8 | sed 's/^/    /' \
            || echo "    --exec returned non-zero (validity gate, or a real failure); see above" >&2

        if [[ -n "${mlx_cmd}" ]]; then
            echo "--- scr bench startup --exec, backend mlx-lm (comparison column)"
            "${BIN}" bench startup "${common[@]}" \
                --child-cmd "${mlx_cmd}" \
                --backend mlx-lm \
                --output-json "${exec_mlx_json}" \
                2>&1 | tail -6 | sed 's/^/    /' \
                || echo "    mlx-lm --exec returned non-zero; see above" >&2
        fi
    fi

    if (( built )); then
        echo "--- scr bench serve (warm steady state, greedy, conc 1)"
        # One resident server for this stage only; --exec owns the ladder above.
        "${model_bin}" serve "${id}" --port "${PORT}" >"${RAW}/serve-${stem}.log" 2>&1 &
        sp=$!
        ready=0
        deadline=$(( $(date +%s) + 1800 ))
        while (( $(date +%s) < deadline )); do
            curl -fsS -m 2 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1 && { ready=1; break; }
            kill -0 "${sp}" 2>/dev/null || break
            sleep 1
        done
        if (( ready )); then
            "${BIN}" bench serve --base-url "http://127.0.0.1:${PORT}" --model "${id}" \
                --num-prompts "${NUM_PROMPTS}" --input-len "${INPUT_LEN}" --output-len "${OUTPUT_LEN}" \
                --max-concurrency 1 --temperature 0 --seed "${RANDOM}${RANDOM}" \
                --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,99 \
                --output-json "${serve_json}" --disable-tqdm 2>&1 | tail -4 | sed 's/^/    /' \
                || echo "    bench serve failed" >&2
        else
            echo "    server never became ready — ${RAW}/serve-${stem}.log" >&2
        fi
        # Kill the whole job, not just the pid, so nothing survives this loop.
        kill -INT "${sp}" 2>/dev/null || true
        for _ in $(seq 1 30); do kill -0 "${sp}" 2>/dev/null || break; sleep 1; done
        kill -KILL "${sp}" 2>/dev/null || true
        wait "${sp}" 2>/dev/null || true
        pkill -f "${model_bin} serve" 2>/dev/null || true

        # ---- scaling stage ------------------------------------------------
        # One axis at a time from a base cell (see the defaults block for why
        # not a factorial), once PER BACKEND when --mlx-python is given, so
        # the JSON can be plotted as us-vs-mlx-lm along every dimension. The
        # same cells and the same per-cell seeds run against both servers —
        # same prompts ⇒ fair curves — the rule bench_serve_compare.sh bakes
        # in. The warm-stage server is torn down first and this stage owns
        # its own servers (solo, serial, reaped), instead of trying to reuse
        # one server across two engines.
        scaling_json="${RAW}/scaling-${stem}.json"
        rm -f "${scaling_json}"

        # Version provenance: a curve without the version that produced it
        # is unreproducible. scratchy as a git tag/commit; mlx-lm as the
        # installed package version (resolved inside the stage's python,
        # which knows how to ask the interpreter --mlx-python names).
        # Computed for the whole per-model block: both the scaling and the
        # offline batch stages record it.
        scratchy_ver="$(git -C "${ROOT}" describe --tags --always --dirty 2>/dev/null \
            || git -C "${ROOT}" rev-parse HEAD)"

        if (( SCALING && ready )); then
            # Warm stage's server must not outlive this branch of the loop.
            kill -INT "${sp}" 2>/dev/null || true
            for _ in $(seq 1 30); do kill -0 "${sp}" 2>/dev/null || break; sleep 1; done
            kill -KILL "${sp}" 2>/dev/null || true
            wait "${sp}" 2>/dev/null || true
            pkill -f "${model_bin} serve" 2>/dev/null || true

            model_idx=0
            for m in "${MODELS[@]}"; do [[ "${m}" == "${entry}" ]] && break || model_idx=$((model_idx+1)); done

            echo "--- scaling sweeps (${CONCURRENCIES} | ${INPUT_LENS} | ${OUTPUT_LENS} @ ${BASE_INPUT}x${BASE_OUTPUT} c${BASE_CONCURRENCY})"
            backends=("${model_bin} serve ${id} --port ${PORT} scratchy")
            [[ -n "${MLX_PYTHON}" ]] && backends+=("${MLX_PYTHON} -m mlx_lm.server --model ${id} --port ${PORT} mlx-lm")
            for spec in "${backends[@]}"; do
                # spec = "<server command> <backend name>" — split off the name.
                backend="${spec##* }"; serve_line="${spec% *}"
                serve_log="${RAW}/scaling-${stem}-${backend}.serve.log"
                echo "    backend: ${backend}"
                # shellcheck disable=SC2086 # serve_line is a command line by construction
                ${serve_line} >"${serve_log}" 2>&1 &
                ssp=$!
                ok=0
                deadline=$(( $(date +%s) + 1800 ))
                while (( $(date +%s) < deadline )); do
                    curl -fsS -m 2 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1 && { ok=1; break; }
                    kill -0 "${ssp}" 2>/dev/null || break
                    sleep 1
                done
                if (( ok )); then
                    python3 - "${model_bin}" "${scaling_json}" "${id}" "${model_idx}" \
                             "${CONCURRENCIES}" "${INPUT_LENS}" "${OUTPUT_LENS}" \
                             "${BASE_INPUT}" "${BASE_OUTPUT}" "${BASE_CONCURRENCY}" \
                             "${NUM_PROMPTS_SCALING}" "${WARMUPS}" "${PORT}" \
                             "${SEED_BASE}" "${backend}" \
                             "${scratchy_ver}" "${MLX_PYTHON}" <<'PY'
import json, os, subprocess, sys, time
(model_bin, out_path, mid, model_idx, concs, in_lens, out_lens,
 base_in, base_out, base_conc, num_prompts, warmups, port,
 seed_base, backend, scratchy_ver, mlx_python) = sys.argv[1:18]
concs  = [int(x) for x in concs.split(",") if x]
in_lens  = [int(x) for x in in_lens.split(",") if x]
out_lens = [int(x) for x in out_lens.split(",") if x]
base_in, base_out, base_conc = int(base_in), int(base_out), int(base_conc)
num_prompts, warmups, port, seed_base = int(num_prompts), int(warmups), int(port), int(seed_base)
model_idx = int(model_idx)

# Cell list: (axis, rung, input_len, output_len, concurrency). The base rung
# of each axis is the same physical cell; dedup by (input, output, conc) so it
# is measured once (identical seed ⇒ identical prompts anyway).
cells = []
for c in concs:
    cells.append(("concurrency", c, base_in, base_out, c))
for il in in_lens:
    cells.append(("input_len", il, il, base_out, base_conc))
for ol in out_lens:
    cells.append(("output_len", ol, base_in, ol, base_conc))

d = json.load(open(out_path)) if os.path.exists(out_path) else {"stage": "scaling", "backends": {}}
rows = d["backends"].get(backend, [])
seen = {(r["input_len"], r["output_len"], r["concurrency"]) for r in rows
        if "output_throughput" in r}

# Per-backend version provenance, recorded once in the backend's block.
# mlx-lm's version is asked of the same interpreter --mlx-python names, so the
# version recorded is the version that served the cells.
d["backends"][backend] = rows
d.setdefault("versions", {})
if backend == "scratchy":
    d["versions"][backend] = {"version": scratchy_ver}
else:
    ver = None
    if mlx_python:
        try:
            r = subprocess.run([mlx_python, "-c",
                                "import importlib.metadata as m; print(m.version('mlx-lm'))"],
                               capture_output=True, text=True, timeout=60)
            ver = r.stdout.strip() if r.returncode == 0 else None
        except Exception:
            ver = None
    d["versions"][backend] = {"version": ver or "unknown"}

for axis, rung, il, ol, c in cells:
    if (il, ol, c) in seen:
        # Same physical cell already measured under another axis — reference
        # it rather than re-running it.
        rows.append({"axis": axis, "rung": rung, "input_len": il, "output_len": ol,
                     "concurrency": c,
                     "dedup_of": {"input_len": il, "output_len": ol, "concurrency": c}})
        continue
    n = num_prompts if num_prompts else max(24, 3 * c)
    seed = seed_base + model_idx * 100000 + il * 131 + ol * 17 + c
    cell_out = f"{out_path}.{backend}.cell-{il}x{ol}x{c}.json"
    cmd = [model_bin, "bench", "serve", "--base-url", f"http://127.0.0.1:{port}",
           "--model", mid, "--num-prompts", str(n), "--input-len", str(il),
           "--output-len", str(ol), "--max-concurrency", str(c),
           "--temperature", "0", "--seed", str(seed),
           "--num-warmups", str(warmups),
           "--percentile-metrics", "ttft,tpot,itl,e2el", "--metric-percentiles", "50,99",
           "--output-json", cell_out, "--disable-tqdm"]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"    cell {axis}={rung} FAILED (rc={r.returncode}); "
              f"{r.stderr.strip().splitlines()[-1] if r.stderr.strip() else 'no stderr'}",
              file=sys.stderr)
        continue
    # bench serve writes its JSON to --output-json; stdout is the human summary.
    try:
        cell_json = json.load(open(cell_out))
    except Exception:
        print(f"    cell {axis}={rung}: no parsable JSON at {cell_out}", file=sys.stderr)
        continue
    keep = ["median_ttft_ms", "p99_ttft_ms", "median_tpot_ms", "p99_tpot_ms",
            "median_itl_ms", "p99_itl_ms", "median_e2el_ms",
            "output_throughput", "request_throughput", "completed",
            "total_output_tokens", "duration", "num_prompts"]
    rows.append({"axis": axis, "rung": rung, "input_len": il, "output_len": ol,
                 "concurrency": c, "seed": seed,
                 **{k: cell_json[k] for k in keep if k in cell_json}})
    seen.add((il, ol, c))
    print(f"    {axis}={rung:>5} · {n:>3} req · tok/s {cell_json.get('output_throughput', float('nan')):>7.1f}"
          f" · TPOT {cell_json.get('median_tpot_ms', float('nan')):>6.1f} ms"
          f" · TTFT {cell_json.get('median_ttft_ms', float('nan')):>7.1f} ms  [{time.time()-t0:.0f}s]")
d["backends"][backend] = rows
d["note"] = ("concurrency is OFFERED (max-concurrency); the effective decode "
             "batch is the scheduler's decision and is not recorded here. "
             "One axis swept at a time, others pinned at base; identical cells "
             "and seeds across backends so the curves plot us vs them directly; "
             "dedup_of marks a cell identical to one already measured under "
             "another axis.")
json.dump(d, open(out_path, "w"), indent=2)
PY
                else
                    echo "    ${backend} server never became ready — ${serve_log}" >&2
                fi
                # Solo rule: this server is reaped before the next backend.
                kill -INT "${ssp}" 2>/dev/null || true
                for _ in $(seq 1 30); do kill -0 "${ssp}" 2>/dev/null || break; sleep 1; done
                kill -KILL "${ssp}" 2>/dev/null || true
                wait "${ssp}" 2>/dev/null || true
            done
        fi

        # ---- offline batch axis ---------------------------------------------
        # The COMMANDED counterpart of the concurrency axis above: no server,
        # width set by the caller, so the curve isolates the batching kernels
        # from the online scheduler. scratchy: `scr bench latency -b <rungs>`
        # (one model load sweeps the ladder natively). mlx-lm: the same
        # BatchGenerator the library exposes, completion_batch_size pinned to
        # the rung, fed the SAME deterministic token ids the latency bench
        # feeds itself — (i*997 + j*31 + 42) % 10000, from latency.rs — so
        # both sides prefill byte-identical prompts. Both sides: greedy,
        # ignore_eos (StopSequences(None) on the mlx-lm side), prefix caching
        # OFF so every timed iteration pays its own prefill.
        if (( BATCH_AXIS )); then
            batch_json="${RAW}/batch-${stem}.json"
            rm -f "${batch_json}"
            echo "--- offline batch sweep (-b ${BATCH_SIZES} @ ${BASE_INPUT}x${BASE_OUTPUT}, ${NUM_ITERS} iters)"

            # scratchy: one process, whole ladder. Wrapped JSON: {"results":
            # [{model, batch_size, avg_latency, percentiles, latencies}]}.
            "${model_bin}" bench latency "${id}" \
                --device metal -b "${BATCH_SIZES}" \
                --input-len "${BASE_INPUT}" --output-len "${BASE_OUTPUT}" \
                --num-iters "${NUM_ITERS}" --num-iters-warmup "${NUM_ITERS_WARMUP}" \
                --no-prefix-caching --temperature 0 \
                --output-json "${batch_json}.scratchy" \
                2>&1 | tail -4 | sed 's/^/    /' \
                || echo "    bench latency (scratchy) failed" >&2

            if [[ -n "${MLX_PYTHON}" ]]; then
                "${MLX_PYTHON}" - "${batch_json}.mlx-lm" "${id}" \
                    "${BATCH_SIZES}" "${BASE_INPUT}" "${BASE_OUTPUT}" \
                    "${NUM_ITERS}" "${NUM_ITERS_WARMUP}" "${scratchy_ver}" <<'PYLX'
import json, sys, time

# Offline mlx-lm batch sweep mirroring `scr bench latency -b`:
#   - same rung list, same input/output lens, warmup + timed iters
#   - same deterministic prompts: the latency bench feeds itself
#     (i*997 + j*31 + 42) % 10000 token ids (crates/benches/src/latency.rs)
#   - EOS cannot stop generation: StopSequences(None) is an empty automaton
#     (ignore_eos parity)
#   - width COMMANDED: completion_batch_size pinned to the rung, so like the
#     latency bench every sequence decodes inside one batch of exactly `bs`
import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import BatchGenerator, StopSequences

(out_path, model_id, bss, il, ol, num_iters, num_warmups, scratchy_ver) = sys.argv[1:9]
bss = [int(x) for x in bss.split(",") if x]
il, ol, num_iters, num_warmups = int(il), int(ol), int(num_iters), int(num_warmups)

model, tokenizer = load(model_id)

def prompts(bs):
    # latency.rs's formula, verbatim: token ids (i*997 + j*31 + 42) % 10000.
    return [[((i * 997 + j * 31 + 42) % 10000) for j in range(il)]
            for i in range(bs)]

results = []
for bs in bss:
    p = prompts(bs)
    gen_kw = dict(completion_batch_size=bs, prefill_batch_size=min(bs, 8))
    # Warmup: untimed, same width.
    for _ in range(num_warmups):
        gen = BatchGenerator(model, stop_tokens=None,
                             max_tokens=ol, **gen_kw)
        gen.insert(p, [ol] * bs,
                   stop_sequences=[StopSequences(None)] * bs)
        while gen.next_generated():
            pass
        gen.close()
    # Timed.
    latencies = []
    for _ in range(num_iters):
        t0 = time.perf_counter()
        gen = BatchGenerator(model, stop_tokens=None,
                             max_tokens=ol, **gen_kw)
        gen.insert(p, [ol] * bs,
                   stop_sequences=[StopSequences(None)] * bs)
        while gen.next_generated():
            pass
        gen.close()
        latencies.append(time.perf_counter() - t0)
    results.append({
        "model": model_id, "batch_size": bs,
        "avg_latency": sum(latencies) / len(latencies),
        "latencies": latencies,
    })
    print(f"    bs={bs:>3} · avg {results[-1]['avg_latency']:.3f} s "
          f"· tok/s {bs * ol / results[-1]['avg_latency']:.1f}")

json.dump({"results": results,
           "note": "offline, COMMANDED width (completion_batch_size); prompts "
                   "and seeds identical to scr bench latency by formula "
                   "(latency.rs); StopSequences(None) = ignore_eos"},
          open(out_path, "w"), indent=2)
PYLX
            fi

            # Merge the per-backend sweeps into the recorded block:
            # {config, backends: {scratchy: ..., "mlx-lm": ...}} — the same
            # shape the served scaling block uses, so downstream consumers
            # treat the two stages alike.
            python3 - "${batch_json}" "${batch_json}.scratchy" "${batch_json}.mlx-lm" \
                     "${BASE_INPUT}" "${BASE_OUTPUT}" "${NUM_ITERS}" "${NUM_ITERS_WARMUP}" \
                     "${BATCH_SIZES}" "${scratchy_ver}" "${MLX_PYTHON}" <<'PY'
import json, os, subprocess, sys
(out, sk_p, lx_p, il, ol, iters, warmups, bss, sk_ver, mlx_python) = sys.argv[1:11]
def load(p):
    return json.load(open(p)) if p and os.path.exists(p) else None
d = {"config": {"input_len": int(il), "output_len": int(ol),
                "num_iters": int(iters), "num_iters_warmup": int(warmups),
                "batch_sizes": [int(x) for x in bss.split(",") if x]},
     "backends": {}}
sk, lx = load(sk_p), load(lx_p)
if sk:
    d["backends"]["scratchy"] = sk
if lx:
    d["backends"]["mlx-lm"] = lx
# Same version provenance rule as the served scaling block: mlx-lm's version
# is asked of the same interpreter --mlx-python names, so the recorded
# version is the version that ran the sweep.
versions = {}
if sk:
    versions["scratchy"] = {"version": sk_ver}
if lx:
    ver = None
    if mlx_python:
        try:
            r = subprocess.run([mlx_python, "-c",
                                "import importlib.metadata as m; print(m.version('mlx-lm'))"],
                               capture_output=True, text=True, timeout=60)
            ver = r.stdout.strip() if r.returncode == 0 else None
        except Exception:
            ver = None
    versions["mlx-lm"] = {"version": ver or "unknown"}
if versions:
    d["versions"] = versions
d["note"] = ("offline, COMMANDED batch width — no server, no scheduler; the "
             "served concurrency axis measures scheduler+kernels, this one "
             "isolates the kernels. Same rungs, same deterministic prompts "
             "(latency.rs formula), greedy, EOS cannot stop generation, "
             "prefix caching off on both sides")
if d["backends"]:
    json.dump(d, open(out, "w"), indent=2)
PY
        fi
    fi

    python3 - "${JSON}" "${stem}" "${id}" "${quant}" "${feats}" "${built}" \
              "${build_secs}" "${bytes}" "${exec_json}" "${serve_json}" "${exec_mlx_json}" \
              "${RAW}/scaling-${stem}.json" "${RAW}/batch-${stem}.json" <<'PY'
import json, os, sys
(js, stem, mid, quant, feats, built, secs, size,
 exec_json, serve_json, exec_mlx_json, scaling_json, batch_json) = sys.argv[1:14]
def num(x):
    for cast in (int, float):
        try: return cast(x)
        except Exception: pass
    return None
def load(p):
    return json.load(open(p)) if p and os.path.exists(p) else None
d = json.load(open(js))
warm = None
sj = load(serve_json)
if sj:
    keep = ["median_ttft_ms","p99_ttft_ms","median_tpot_ms","p99_tpot_ms","median_itl_ms",
            "p99_itl_ms","median_e2el_ms","output_throughput","request_throughput",
            "completed","total_output_tokens","duration"]
    warm = {k: sj[k] for k in keep if k in sj}
d["models"].append({
    "stem": stem, "model_id": mid, "quant": quant or None, "features": feats,
    "built": built == "1",
    "footprint": {"build_seconds": num(secs), "binary_bytes": num(size),
                  "container_image_bytes": None,
                  "note": "no container image: Metal is not available in Linux containers"},
    "cache_ladder": load(exec_json),
    "cache_ladder_mlx_lm": load(exec_mlx_json),
    "warm_serving": warm,
    "scaling": load(scaling_json),
    "offline_batch": load(batch_json),
})
json.dump(d, open(js, "w"), indent=2)
PY
    echo "    recorded"
done

# ---- summary ----------------------------------------------------------------
echo; echo "================================================================"
python3 - "${JSON}" <<'PY'
import json, statistics, sys
d = json.load(open(sys.argv[1]))
m = d["machine"]
print(f"{m['chip']} · {m['cores_total']} cores ({m['cores_performance']}P+{m['cores_efficiency']}E) "
      f"· {m['memory_gb']} GB · macOS {m['macos']}")
print(f"{'model':26}{'build':>7}{'MiB':>6}{'frozen':>9}{'cold':>8}{'TTFT':>8}{'TPOT':>8}{'tok/s':>8}")
def med(reps, scenario, field):
    vals = [r[field] for r in (reps or []) if r.get("scenario") == scenario and r.get(field) is not None]
    return statistics.median(vals) if vals else None
def fmt(v, nd=0):
    return "-" if v is None else f"{v:.{nd}f}"
for e in d["models"]:
    f, ladder, w = e["footprint"], e.get("cache_ladder"), e.get("warm_serving") or {}
    mib = round(f["binary_bytes"] / 1048576) if f["binary_bytes"] else None
    print(f"{e['stem']:26}{fmt(f['build_seconds']):>7}{fmt(mib):>6}"
          f"{fmt(med(ladder,'frozen','ttft_exec_s'), 2):>9}{fmt(med(ladder,'cold','ttft_exec_s'), 2):>8}"
          f"{fmt(w.get('median_ttft_ms')):>8}{fmt(w.get('median_tpot_ms'), 1):>8}"
          f"{fmt(w.get('output_throughput'), 1):>8}")
    if not e["built"]:
        print("    BUILD FAILED — see the build log in this machine's raw/ directory")
    elif ladder is None:
        print("    cache ladder unavailable")

# ---- scaling curves ---------------------------------------------------------
# Per model, per backend, one line per axis rung: the marginal curves the
# scaling stage exists for, plotted us vs mlx-lm when both ran. tok/s should
# rise with concurrency then flatten (the gap to linear is per-batch
# overhead); TPOT ~flat means batched decode is not charging requests for
# being batched; TTFT vs input is prefill scaling.
axes = ["concurrency", "input_len", "output_len"]
for e in d["models"]:
    sc = e.get("scaling") or {}
    bks = sc.get("backends") or {}
    if not bks:
        continue
    print(f"\n  {e['stem']} — scaling")
    for bk in sorted(bks):
        ver = (sc.get("versions") or {}).get(bk, {}).get("version", "?")
        rows = sorted((r for r in bks[bk] if "output_throughput" in r),
                      key=lambda r: (axes.index(r["axis"]) if r["axis"] in axes else 99, r["rung"]))
        if not rows:
            continue
        print(f"    [{bk} {ver}]")
        print(f"    {'axis':12}{'rung':>7}{'tok/s':>9}{'req/s':>8}{'TPOT':>9}{'TTFT':>9}")
        for r in rows:
            print(f"    {r['axis']:12}{r['rung']:>7}"
                  f"{fmt(r.get('output_throughput'), 1):>9}"
                  f"{fmt(r.get('request_throughput'), 2):>8}"
                  f"{fmt(r.get('median_tpot_ms'), 1):>9}"
                  f"{fmt(r.get('median_ttft_ms')):>9}")

# ---- offline batch curves ---------------------------------------------------
# The COMMANDED-width counterpart of the concurrency axis: same rungs, no
# scheduler. tok/s at bs should track linear with bs minus per-batch
# overhead; the gap between this curve and the served one is the scheduler's
# share. Both backends' rows share one config (input/output lens, iters).
for e in d["models"]:
    ob = e.get("offline_batch") or {}
    cfg = ob.get("config") or {}
    ol = cfg.get("output_len")
    backends = ob.get("backends") or {}
    if not backends:
        continue
    print(f"\n  {e['stem']} — offline batch (COMMANDED width, no scheduler)")
    for bk in sorted(backends):
        rows = backends[bk].get("results", [])
        if not rows:
            continue
        print(f"    [{bk}]")
        print(f"    {'bs':>5}{'avg s':>9}{'tok/s':>9}")
        for r in rows:
            bs, lat = r["batch_size"], r["avg_latency"]
            tps = bs * ol / lat if (ol and lat) else None
            print(f"    {bs:>5}{fmt(lat, 3):>9}{fmt(tps, 1):>9}")
print(f"\nfrozen/cold are ttft_exec seconds (exec -> first token); TTFT/TPOT are warm ms.")
print(f"scaling: concurrency is OFFERED (max-concurrency); TPOT ~flat across it is the goal")
print(f"json -> {sys.argv[1]}")
print("paste this file into issue #91; fill the table in #95 from it")
PY
