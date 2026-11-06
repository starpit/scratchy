#!/usr/bin/env bash
# ⭐ THE #1112/#1113 OP CARD-GATE RUNNER — runs every cell the dump test wrote
# (zz_ew_card_gate.rs / zz_reduce_card_gate.rs, this directory) on the card and checks it
# against a HOST reference computed from the op's own formula, over SEEDED RANDOM fills
# (never constants, so a wrong lane cannot hide). Test tooling: nothing here ships.
#
# usage: ew_card_gate_run.sh <gate-root> <tree-with-bundle_run_dev> <log>
#   <gate-root>   the directory of cell dirs (each: group_0/, placements.json, cell.json)
#   <tree>        a warm pod checkout with target/release/examples/bundle_run_dev built from
#                 the SAME commit the dump test ran at
#   <log>         the verdict log (one line per cell, appended — the run is resumable)
#
# THE BAR, per cell: the read-back buffer holds the card image of the op over the fills,
# decoded by the rank-2 stick-column law (a device layout is row-major with the stick dim
# last, so a host [m, c] sits on the device as [c/64, m, 64]:
#   D = (c/64)·(m·64) + t·64 + (c%64)
# — the same law the page-gather runner decodes by), against a host reference in fp32 from
# the formula:
#   gelu        the TANH form: 0.5·x·(1+tanh(√(2/π)·(x+0.044715x³)))
#   rmsnormunit x/√(mean(x²)+1e-6)                                   (eps 1e-6, no gain)
#   rmsnorm     x/√(mean(x²)+1e-5)·gain                              (eps 1e-5, Scale gain)
#   reduce_sum  Σ x per row (fp32 accumulate of the f16 values)
#   reduce_max  max x per row; the accum is one stick per row, the value at lane 0 —
#               THE DOOR'S OWN LAW, not an assumption: the single shot's output leg is a
#               row-blocked [rows, 1] region (lower_ktir_to_superdsc.rs:10600 at
#               347ee3feee — the Fp16 arm's assemble_reduce_seeded call passes
#               rb(&out.name(), rows, 1)), and a row-blocked region's column 0 of row r is
#               lane 0 of its stick: D = r·64.
# The verdict line states the program, the shape, the max abs error, the max fp16 ulp, and
# the artifact path. PASS bar: max fp16 ulp ≤ 8 (stated here, once — the numbers print
# regardless, so a tighter or looser bar is a one-line change, never a re-run).
#
# ⛔ THE FENCE PROTOCOL (the page-gather runner's own ruling): the runner holds
# /work/.spyre-bake.lock for the whole run, and ABORTS ON FENCE — after every compile and
# every run the logs are checked for the card's fence signatures, and the first one stops
# the sweep with a FENCED verdict and a non-zero exit.
#
# ⛔ THE BODY CARRIES NO SINGLE QUOTE: the whole body runs as `bash -c '…'`, so one `'`
# inside it (a python dict key, an f-string) closes the body and the outer shell sees the
# remainder as its own words — `c: unbound variable` under `set -u`. Every python string
# below is double-quoted and every value crosses as a POSITIONAL arg, exactly as
# page_gather_sweep_run.sh does it.
set -uo pipefail
[ $# -eq 3 ] || { echo "usage: ew_card_gate_run.sh <gate-root> <tree-with-bundle_run_dev> <log>" >&2; exit 2; }
ROOT=$(cd "$1" && pwd) || exit 80
TREE=$2
LOG=$3
RUNNER_SHA=$(git -C "$(dirname "$0")" rev-parse --short HEAD 2>/dev/null || echo unknown)
RUNNER_MD5=$(md5sum < "$0" 2>/dev/null | cut -c1-32 || echo unknown)
echo "# RUNNER sha=$RUNNER_SHA md5=$RUNNER_MD5 argv=$*" >> "$LOG"
exec flock /work/.spyre-bake.lock bash -c '
set -uo pipefail
ROOT=$1; TREE=$2; LOG=$3
export PATH=/work/.cargo/bin:$PATH
export LD_LIBRARY_PATH=/opt/ibm/spyre/senlib/lib:/opt/ibm/spyre/runtime/lib:/opt/ibm/spyre/deeptools/lib
export DEEPTOOLS_PATH=/opt/ibm/spyre/deeptools/share
cd "$TREE" || exit 80

fenced() {
  grep -iE "BusFence|0xa35e|PfMSIMonitor" "$1" 2>/dev/null | head -1
}

run_cell() {
  local cell=$1 name sha md5 skip; name=$(basename "$cell")
  [ -f "$cell/cell.json" ] || return 0
  [ -d "$cell/group_0" ] || return 0
  eval "$(python3 - "$cell" "$LOG" <<"PY"
import hashlib, json, os, sys
cell, log = sys.argv[1], sys.argv[2]
c = json.load(open(cell + "/cell.json"))
h = hashlib.md5()
for root, dirs, files in os.walk(cell + "/group_0"):
    dirs[:] = [d for d in dirs if d != "spyreCodeDir"]
    for f in sorted(files):
        p = os.path.join(root, f)
        h.update(p[len(cell) + 1:].encode())
        h.update(open(p, "rb").read())
md5 = h.hexdigest()
c["group_0_md5"] = md5
json.dump(c, open(cell + "/cell.json", "w"), indent=1)
skip = "False"
name = os.path.basename(cell)
if os.path.exists(log):
    for line in open(log):
        if line.startswith("VERDICT " + name + " ") and "sha=" + str(c["sha"]) in line and "md5=" + md5 in line:
            skip = "True"
print("sha=%s" % c["sha"])
print("md5=%s" % md5)
print("skip=%s" % skip)
PY
)"
  [ "$skip" = "True" ] && { echo "SKIP $name (already verdicted at sha $sha md5 $md5)"; return; }

  # The fills: SEEDED RANDOM (one seed per input, offset by the input index), standard
  # normal scaled to ±2-ish so the whole f16 mantissa is exercised; the gain row (the
  # rmsnorm cells second input) is drawn the same way.
  python3 - "$cell" <<"PY"
import json, sys
import numpy as np
cell = sys.argv[1]
c = json.load(open(cell + "/cell.json"))
n = c["m"] * c["c"]
for i, tid in enumerate(c["in_tids"]):
    rng = np.random.default_rng(c["seed"] + i)
    if c["op"] == "rmsnorm" and i == 1:
        # the gain row: [1, c], centered at 1 so the reference is a real gain
        g = (1.0 + 0.25 * rng.standard_normal(c["c"])).astype(np.float16)
        g.tofile(cell + "/in-t%d.bin" % tid)
    else:
        (2.0 * rng.standard_normal(n)).astype(np.float16).tofile(cell + "/in-t%d.bin" % tid)
PY
  [ $? -eq 0 ] || { echo "VERDICT $name NOT-RUN fill-failed sha=$sha md5=$md5" | tee -a "$LOG"; return; }

  rm -rf "$cell/group_0/spyreCodeDir"
  timeout 1800 /opt/ibm/spyre/deeptools/bin/dxp_standalone --bundle -d "$cell/group_0" -b sentient \
    > "$cell/compile.log" 2>&1
  local crc=$? sig
  sig=$(fenced "$cell/compile.log")
  if [ -n "$sig" ]; then
    echo "VERDICT $name FENCED at-compile sha=$sha md5=$md5 sig=[$sig]" | tee -a "$LOG"
    echo "FENCE-ABORT at $name (compile)" >&2
    exit 99
  fi
  [ $crc -ne 0 ] && { echo "VERDICT $name NOT-RUN compile_rc=$crc sha=$sha md5=$md5" | tee -a "$LOG"; return; }

  local out_tid n_in stage tid rrc
  eval "$(python3 - "$cell" <<"PY"
import json, sys
c = json.load(open(sys.argv[1] + "/cell.json"))
print("out_tid=%d" % c["out_tid"])
print("n_in=%d" % len(c["in_tids"]))
print("in_list=%s" % ",".join(str(t) for t in c["in_tids"]))
PY
)"
  # A Weight-role input (the rmsnorm gain) is staged through the WEIGHT path — the plain
  # t<id>=<file> bind, which stage_weights re-encodes IEEE→sen fp16 once, before prepare.
  # raw:t<id> is the per-step ACTIVATION path only: bundle_run.rs:742 refuses a raw id whose
  # placement role is Weight, by name. The role comes from the cell own placements.json,
  # one role per input tid, in order.
  local role_list
  role_list=$(python3 - "$cell" "$in_list" <<"PY"
import json, sys
cell = sys.argv[1]
tids = sys.argv[2].split(",") if sys.argv[2] else []
places = json.load(open(cell + "/placements.json"))["places"]
role = {}
for p in places:
    role[str(p["tid"])] = p["role"]
print(",".join(role.get(str(t), "Activation") for t in tids))
PY
)
  args=()
  IFS=, read -r -a in_tids <<< "$in_list"
  IFS=, read -r -a in_roles <<< "$role_list"
  for i in "${!in_tids[@]}"; do
    tid=${in_tids[$i]}
    args+=("t$tid=$cell/in-t$tid.bin")
    if [ "${in_roles[$i]}" != "Weight" ]; then
      args+=("raw:t$tid")
    fi
  done
  ./target/release/examples/bundle_run_dev "$cell" "${args[@]}" \
    --out "raw:t$out_tid=$cell/out.bin" > "$cell/run.log" 2>&1
  rrc=$?
  sig=$(fenced "$cell/run.log")
  if [ -n "$sig" ]; then
    echo "VERDICT $name FENCED at-run sha=$sha md5=$md5 sig=[$sig]" | tee -a "$LOG"
    echo "FENCE-ABORT at $name (run)" >&2
    exit 99
  fi
  [ $rrc -ne 0 ] && { echo "VERDICT $name NOT-RUN run_rc=$rrc sha=$sha md5=$md5" | tee -a "$LOG"; return; }

  python3 - "$cell" "$name" >> "$LOG" <<"PY"
import json, sys
import numpy as np
cell, name = sys.argv[1], sys.argv[2]
c = json.load(open(cell + "/cell.json"))
m, cols = c["m"], c["c"]
op = c["op"]
n = m * cols

def dev_index():
    # the rank-2 stick-column law, VECTORIZED (the tall rung is 33.5M elements — a python
    # loop would take minutes): D = (c/64)*(m*64) + t*64 + (c%64), one device offset per
    # host (t, col), flattened row-major
    colsv = np.arange(cols)
    return (
        (colsv // 64)[None, :] * (m * 64)
        + np.arange(m)[:, None] * 64
        + (colsv % 64)[None, :]
    ).reshape(m * cols)

x = np.fromfile(cell + "/in-t%d.bin" % c["in_tids"][0], dtype=np.float16, count=n).astype(np.float32)
if op == "gelu":
    k = float(np.sqrt(2.0 / np.pi))
    u = k * (x + 0.044715 * x * x * x)
    want = (0.5 * x * (1.0 + np.tanh(u))).astype(np.float16)
    got = np.fromfile(cell + "/out.bin", dtype=np.float16, count=n)
    # decode the stick-column law into host row-major order
    host = got[dev_index()]
    cmp_a, cmp_b = host.astype(np.float32), want.astype(np.float32)
elif op in ("rmsnormunit", "rmsnorm"):
    eps = 1e-6 if op == "rmsnormunit" else 1e-5
    rows = x.reshape(m, cols)
    ms = (rows * rows).mean(axis=1, keepdims=True)
    ref = rows / np.sqrt(ms + eps)
    if op == "rmsnorm":
        g = np.fromfile(cell + "/in-t%d.bin" % c["in_tids"][1], dtype=np.float16, count=cols).astype(np.float32)
        ref = ref * g[None, :]
    want = ref.astype(np.float16).reshape(n)
    got = np.fromfile(cell + "/out.bin", dtype=np.float16, count=n)
    host = got[dev_index()]
    cmp_a, cmp_b = host.astype(np.float32), want.astype(np.float32)
elif op in ("reduce_sum", "reduce_max"):
    rows = x.reshape(m, cols)
    if op == "reduce_sum":
        want = rows.sum(axis=1)
    else:
        want = rows.max(axis=1)
    want = want.astype(np.float16)
    got = np.fromfile(cell + "/out.bin", dtype=np.float16, count=m * 64)
    # THE ACCUM LAW, from the door itself: the single-shot output leg is a row-blocked
    # [rows, 1] region (lower_ktir_to_superdsc.rs:10600 at 347ee3feee — the Fp16 arm
    # assemble_reduce_seeded call passes rb(&out.name(), rows, 1)), and a row-blocked
    # region column 0 of row r is LANE 0 of its stick: D = r*64.
    host = got.reshape(m, 64)[:, 0]
    cmp_a, cmp_b = host.astype(np.float32), want.astype(np.float32)
else:
    print("VERDICT %s NOT-RUN unknown-op=%s sha=%s md5=%s" % (name, op, c["sha"], c["group_0_md5"]))
    sys.exit()

if host.size != want.size:
    print("VERDICT %s NOT-RUN short-readback %d of %d sha=%s md5=%s" % (name, host.size, want.size, c["sha"], c["group_0_md5"]))
    sys.exit()

def ulp16(a, b):
    # the fp16 ulp distance: the bit patterns as integers, same-sign; a sign flip is
    # counted as the whole range (it is never within any sane bar)
    ia = a.astype(np.float16).view(np.int16).astype(np.int32)
    ib = b.astype(np.float16).view(np.int16).astype(np.int32)
    same = (ia >= 0) == (ib >= 0)
    d = np.abs(ia - ib)
    d[~same] = 1 << 16
    return d

max_abs = float(np.max(np.abs(cmp_a - cmp_b))) if cmp_a.size else 0.0
ulps = ulp16(cmp_a, cmp_b)
max_ulp = int(ulps.max()) if ulps.size else 0
ok = max_ulp <= 8
kind = "PASS" if ok else "FAIL"
print(
    "VERDICT %s %s op=%s m=%d c=%d max_abs=%.6g max_ulp=%d artifact=%s sha=%s md5=%s"
    % (name, kind, op, m, cols, max_abs, max_ulp, cell, c["sha"], c["group_0_md5"])
)
PY
}

[ -d "$ROOT" ] || { echo "EW_CARD_GATE_RUN_INCOMPLETE — not a dir: $ROOT" >&2; exit 1; }
cell_list=$(find "$ROOT" -mindepth 1 -maxdepth 1 -type d | sort)
if [ -z "$cell_list" ]; then
  echo "EW_CARD_GATE_RUN_INCOMPLETE — no cell dirs in $ROOT" >&2
  exit 1
fi
echo "$cell_list" | while read -r c; do run_cell "$c"; done

pass=0; fail=0; notrun=0; fenced=0; runnable=0; total=0
for c in $cell_list; do
  total=$((total + 1))
  [ -d "$c/group_0" ] || continue
  runnable=$((runnable + 1))
  name=$(basename "$c")
  stamp=$(python3 - "$c" <<"PY"
import json, sys
c = json.load(open(sys.argv[1] + "/cell.json"))
print("sha=%s md5=%s" % (c["sha"], c["group_0_md5"]))
PY
)
  kind=$(grep "^VERDICT $name " "$LOG" 2>/dev/null | grep -F "$stamp" | tail -1 | awk "{print \$3}")
  case "$kind" in
    PASS) pass=$((pass + 1));;
    FAIL) fail=$((fail + 1));;
    NOT-RUN) notrun=$((notrun + 1));;
    FENCED) fenced=$((fenced + 1));;
  esac
done
echo "PASS=$pass FAIL=$fail NOT-RUN=$notrun FENCED=$fenced CELLS=$runnable TOTAL=$total"
if [ "$fail" -gt 0 ] || [ "$notrun" -gt 0 ] || [ "$fenced" -gt 0 ] || [ "$pass" -lt "$runnable" ]; then
  echo "EW_CARD_GATE_RUN_INCOMPLETE — every runnable cell needs a PASS at its own stamp" >&2
  exit 1
fi
echo EW_CARD_GATE_RUN_DONE
' _ "$ROOT" "$TREE" "$LOG"
