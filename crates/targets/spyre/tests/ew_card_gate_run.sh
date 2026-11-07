#!/usr/bin/env bash
# ⭐ THE #1112/#1113 OP CARD-GATE RUNNER — runs every cell the dump test wrote
# (zz_ew_card_gate.rs / zz_reduce_card_gate.rs, this directory) on the card and checks it
# against a HOST reference computed from the op's own definition, over SEEDED RANDOM fills
# (never constants, so a wrong lane cannot hide). Test tooling: nothing here ships.
#
# usage: ew_card_gate_run.sh <gate-root> <tree-with-bundle_run_dev> <log>
#   <gate-root>   the directory of cell dirs (each: group_0/, placements.json, cell.json)
#   <tree>        a warm pod checkout with target/release/examples/bundle_run_dev built from
#                 the SAME commit the dump test ran at
#   <log>         the verdict log (one line per cell, appended — the run is resumable)
#
# ⛔ THE CODEC LAW (run-3): the card's activation format is SEN fp16, not IEEE — `raw:t<id>`
# binds IEEE bytes verbatim, which is why run-2 measured max_abs 0.47-8.8 on cells whose
# arithmetic was right. Every input is staged and every output read back through the
# launcher's IEEE<->SEN codec instead:
#   stage    t<id>=<file> stick:t<id>=<rows>
#   readback --out t<id>=<file> stick:t<id>=<rows>
# (bundle_run.rs: the input path resticks row-major host bytes to the device stick order and
# re-encodes IEEE->SEN; `read_tensor` decodes SEN->IEEE and `restick`s back to host
# row-major on the way out). So the read-back file IS host row-major IEEE and the runner
# decodes it as the IDENTITY — the stick-column law is applied exactly once, by the
# launcher, never again below. No `raw:` remains anywhere in this runner: the page-gather
# runner kept one only because its fills were small exact integers.
# The rmsnorm gain row is staged the same way with `stick:t1=1` — which IS the [1, C] row's
# identity map (bundle_run.rs: "stick:tN=1 is the identity, and saying it out loud is the
# point"), from the cell's own gain tensor file, staged through the weight loop's
# IEEE->SEN re-encode.
#
# ⛔ THE BAR IS STATED PER OP, FROM THE OP'S DEFINITION, BEFORE THE CELL RUNS — never tuned
# after seeing a number. Metrics: max_abs everywhere; max_rel and max fp16 ulp over the
# lanes with |want| >= 1e-2 (the HEAD); tail_abs over the rest.
#   THE TAIL LAW (every op, stated once): lanes with |want| < 1e-2 are judged by tail_abs
#   <= 2^-6 (0.015625). Reason: at this runner's +-2-ish fill scale 2^-6 is 8 f16 ulp, and
#   near a zero of the function the ulp metric is defeated by sign flips and denormal
#   spacing, so the bar there is absolute.
#   rmsnorm / rmsnormunit: max_ulp <= 8 over the head. Reason: the op's mean(x^2) reduce
#   accumulates in f32, so the device value is the f16 image of an f32 computation — a few
#   ulp, with 8 leaving headroom for the rsqrt and the gain multiply's own rounding.
#   gelu (the folded cell): max_ulp <= 8 over the head against the FEST-modeled gelufwd
#   form (see (iii) below). Reason: gelufwd is ONE device primitive, and the cell must be
#   the f16 image of its own defining polynomial. The distance to the TRUE tanh form is
#   also reported.
#   gelu_longhand (the unfolded cell): the tail law plus finiteness. The distance to the
#   true tanh form is REPORTED with no invented tight bar — the longhand lowers through the
#   exp island (exp.smc, 13 SFP instructions) which this host does not model, so any tight
#   number would be tuned, not derived.
#   reduce_max: max_ulp <= 8 over the head. Reason: the selection over f16 values is exact,
#   so the output is one of the inputs, exactly.
#   reduce_sum: max_rel <= 2^-5 (3.1%) over the head. Reason: the sum accumulates in f16
#   across the row — a random walk of sqrt(cols)*2^-11 ~= 1.4% of the magnitude at 512
#   lanes — and 2^-5 is one order of headroom over that walk.
#
# ⛔ THE GELU THREE-NUMBER PROTOCOL (#1112, auditor ruling): on the same input vector,
#   (i)   the folded gelufwd cell, against
#   (iii) gelufwd's defining form, cited from the vendored deeptools tree by direct path
#         read — crates/compiler/deeptools/opaque_templates/gelu.smc at 1b5e66c1a9:
#           gelu(x) = 0.5x + 0.5x*tanh(A1*x + A2*x^3)
#         with A1 = SEN169 0x3D31 = 2^-1*(1+305/512) = 0.7978515625 (~sqrt(2/pi)) and
#         A2 = SEN169 0x3448 = 2^-5*(1+72/512) = 0.03564453125 (~0.044715*sqrt(2/pi)),
#         decoded by the sen169_bits law (emit/mod.rs:7066: sign 1, exp 6 bias 31, mantissa
#         9). The tanh is the FEST hardware approximation (gelu.smc FEST imm=8 slope,
#         imm=9 offset). RUN-4: the (iii) is NOT modeled here — the dump test itself
#         computes it on the dump-owned fills (zz_ew_card_gate.rs `fest`: senulator
#         src/sfp.rs's own segment tables and segment_lookup at 1b5e66c1a9, every smc
#         intermediate round-tripped through the tree's own SEN169 codec) and writes
#         want_iii.bin; this runner only READS the file. The gelu and gelu_longhand fills
#         are dump-owned too (`fills: "dump"` in cell.json): (i) and (ii) run the SAME x
#         vector, and this runner generates fills only for the cells the dump left to it;
#   (ii)  the SAME tanh longhand lowered op by op (the gelu_longhand cells: the unclaimed
#         spelling, no negf, so program_gelu_chains leaves it and lower_function walks it),
#         against the TRUE tanh form, distance reported.
# If (i) matches (iii) the fold is spelling the right function and the PR states the
# measured distance; if (i) does not match (iii), the fold is spelling the wrong function —
# a bug, not a bar to move. RUN-4 diagnostics on every gelu cell: the x at max ulp with
# got vs (iii) vs true, the whole cell as (x, got, want_iii, want_true) CSV in the cell
# dir, and on a failing longhand the first 8 lanes got vs want.
#
# ⛔ A DOOR REFUSAL IS MISSING-LOWERING (run-4): a cell whose cell.json carries
# `missing_lowering` (the door's refusal text — the dump records it instead of dying) is
# reported as `VERDICT <name> MISSING-LOWERING sha=<sha> reason=<text>` and counts as its
# own tally kind: the gate is INCOMPLETE while any stands, never silently skipped.
#
# ⛔ THE EVERY-BOUND-TENSOR LAW (run-5, moe-lead 2026-11-06): the runner stages EVERY
#   worker-bound tensor the bundle names, read from the launch map itself, and REFUSES BY
#   NAME to launch a cell whose bundle names a bound tensor it did not stage. The expected
#   set is the `places` rows at or above the reserved-const floor (4294966272 = u32::MAX -
#   1024, `bundle_run.rs`'s RESERVED_TID_FLOOR — the producer's reserved device constants,
#   counted down from u32::MAX); the values are the `scalarmul_scales` (tid, value) rows the
#   dump writes from the layout's own scale registry (index i at scalarmul_scale_tid(i)),
#   staged through `bundle_run`'s `const:t<id>=<value>` marker. A reserved placement with no
#   staged value is a named NOT-RUN, never a launch: `bundle_run` zeroes every segment
#   holding no staged weight, and a zero in a scale slot is SILENT — measured on card at
#   4f71242b9f (cgpx-4f71242b), where two never-staged registry scales made the longhand's
#   cubic chain ALL ZEROS (0·(x+0·x³)=0) and the exp cell ALL ONES (exp(0)=1) while every
#   launch reported success, and the rmsnormunit eps passed only because 1e-6 is below f16
#   resolution (identical f16 image at eps=0 — a latent gap, not a proof). A staged const
#   naming no placement is the same refusal the other way round: the two lists must agree
#   exactly, or the cell does not run.
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
  # ⛔ MISSING-LOWERING FIRST (run-4): the dump records a door refusal in cell.json and
  # writes no group_0 — state it by name (resumable by sha) and move on; it is never a
  # silent skip and never a dead dump.
  if grep -q "missing_lowering" "$cell/cell.json" 2>/dev/null; then
    python3 - "$cell" "$name" "$LOG" <<"PY"
import json, os, sys
cell, name, log = sys.argv[1], sys.argv[2], sys.argv[3]
c = json.load(open(cell + "/cell.json"))
sha = c["cell"]["sha"] if "cell" in c else c["sha"]
want = "VERDICT %s MISSING-LOWERING sha=%s " % (name, sha)
if os.path.exists(log) and any(line.startswith(want) for line in open(log)):
    print("SKIP %s (already verdicted MISSING-LOWERING at sha %s)" % (name, sha))
else:
    reason = " ".join(str(c.get("missing_lowering", "")).split())
    with open(log, "a") as f:
        f.write("VERDICT %s MISSING-LOWERING sha=%s reason=%s\n" % (name, sha, reason))
PY
    return
  fi
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
  # rmsnorm cells second input) is drawn the same way. A `fills: "dump"` cell already
  # carries its in-t0.bin from the dump (the gelu family: folded and longhand run the
  # SAME x vector) — the runner generates nothing for it.
  python3 - "$cell" <<"PY"
import json, sys
import numpy as np
cell = sys.argv[1]
c = json.load(open(cell + "/cell.json"))
if c.get("fills") == "dump":
    sys.exit(0)
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

  # THE STAGE MAP — every input through the codec, each with its own ROW COUNT: m for an
  # [m, c] activation, 1 for the rmsnorm gain (the [1, c] row: stick:t1=1 is its identity
  # map). The read-back states m rows for every op — the reduce accum is one stick per row,
  # so m rows of 64 lanes, its value at lane 0.
  local out_tid out_rows rrc
  eval "$(python3 - "$cell" <<"PY"
import json, sys
c = json.load(open(sys.argv[1] + "/cell.json"))
rows = []
for i, tid in enumerate(c["in_tids"]):
    if c["op"] == "rmsnorm" and i == 1:
        rows.append(1)
    else:
        rows.append(c["m"])
print("out_tid=%d" % c["out_tid"])
print("out_rows=%d" % c["m"])
print("in_list=%s" % ",".join(str(t) for t in c["in_tids"]))
print("rows_list=%s" % ",".join(str(r) for r in rows))
# THE EVERY-BOUND-TENSOR LAW (header): the expected set is the reserved-region places rows
# (4294966272 = u32::MAX - 1024, bundle_run.rs RESERVED_TID_FLOOR); the values are the
# launch map own scalarmul_scales rows. The two lists must agree exactly.
pf = json.load(open(sys.argv[1] + "/placements.json"))
FLOOR = 4294966272
expected = sorted(p["tid"] for p in pf["places"] if p["tid"] >= FLOOR)
consts = pf.get("scalarmul_scales") or []
staged = sorted(k["tid"] for k in consts)
missing = [t for t in expected if t not in staged]
stale = [t for t in staged if t not in expected]
print("const_list=%s" % ",".join("t%d=%r" % (k["tid"], k["value"]) for k in consts))
print("missing_list=%s" % ",".join(str(t) for t in missing))
print("stale_list=%s" % ",".join(str(t) for t in stale))
PY
)"
  args=()
  IFS=, read -r -a in_tids <<< "$in_list"
  IFS=, read -r -a in_rows <<< "$rows_list"
  for i in "${!in_tids[@]}"; do
    tid=${in_tids[$i]}
    args+=("t$tid=$cell/in-t$tid.bin" "stick:t$tid=${in_rows[$i]}")
  done
  # THE EVERY-BOUND-TENSOR LAW: a reserved placement with no staged value is a named
  # NOT-RUN, never a launch — a missing constant would read as a zeroed segment, silently
  # (the ALL-ZEROS cubic of run cgpx-4f71242b). A staged const naming no placement is the
  # same refusal the other way round.
  if [ -n "$missing_list" ]; then
    echo "VERDICT $name NOT-RUN unstaged-bound-tensor tids=$missing_list sha=$sha md5=$md5" | tee -a "$LOG"
    return
  fi
  if [ -n "$stale_list" ]; then
    echo "VERDICT $name NOT-RUN stale-const-row tids=$stale_list sha=$sha md5=$md5" | tee -a "$LOG"
    return
  fi
  if [ -n "$const_list" ]; then
    IFS=, read -r -a bound_consts <<< "$const_list"
    args+=("${bound_consts[@]/#/const:}")
  fi
  ./target/release/examples/bundle_run_dev "$cell" "${args[@]}" \
    --out "t$out_tid=$cell/out.bin" "stick:t$out_tid=$out_rows" > "$cell/run.log" 2>&1
  rrc=$?
  sig=$(fenced "$cell/run.log")
  if [ -n "$sig" ]; then
    echo "VERDICT $name FENCED at-run sha=$sha md5=$md5 sig=[$sig]" | tee -a "$LOG"
    echo "FENCE-ABORT at $name (run)" >&2
    exit 99
  fi
  [ $rrc -ne 0 ] && { echo "VERDICT $name NOT-RUN run_rc=$rrc sha=$sha md5=$md5" | tee -a "$LOG"; return; }

  python3 - "$cell" "$name" >> "$LOG" <<"PY"
import json, math, sys
import numpy as np
cell, name = sys.argv[1], sys.argv[2]
c = json.load(open(cell + "/cell.json"))
m, cols = c["m"], c["c"]
op = c["op"]
n = m * cols

# (iii) the gelufwd defining form — see the runner header for the full citation
# (gelu.smc at 1b5e66c1a9 + senulator/src/sfp.rs FEST tables). RUN-4: the (iii) is
# computed BY THE DUMP on these exact x values (zz_ew_card_gate.rs `fest`) and read here from
# want_iii.bin — this runner no longer models the FEST law at all.

def gelu_true(x):
    k = math.sqrt(2.0 / math.pi)
    u = k * (x + 0.044715 * x * x * x)
    return 0.5 * x * (1.0 + np.tanh(u))

x = np.fromfile(cell + "/in-t%d.bin" % c["in_tids"][0], dtype=np.float16, count=n).astype(np.float32)

# The read-back is host row-major IEEE (the launcher already un-sticked and SEN->IEEE
# decoded it) — the IDENTITY decode. The reduce accum keeps the lane law the door states.
if op in ("gelu", "gelu_longhand"):
    got = np.fromfile(cell + "/out.bin", dtype=np.float16, count=n)
elif op in ("rmsnormunit", "rmsnorm"):
    eps = 1e-6 if op == "rmsnormunit" else 1e-5
    rows = x.reshape(m, cols)
    ms = (rows * rows).mean(axis=1, keepdims=True)
    ref = rows / np.sqrt(ms + eps)
    if op == "rmsnorm":
        g = np.fromfile(cell + "/in-t%d.bin" % c["in_tids"][1], dtype=np.float16, count=cols).astype(np.float32)
        ref = ref * g[None, :]
    got = np.fromfile(cell + "/out.bin", dtype=np.float16, count=n)
    want = ref.astype(np.float16).reshape(n)
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
    # calls assemble_reduce_seeded with rb(&out.name(), rows, 1)), and column 0 of row r
    # in a row-blocked region is LANE 0 of its stick. The codec readback at m rows of
    # one stick each lands those lanes at reshape(m, 64)[:, 0].
    got = got.reshape(m, 64)[:, 0]
else:
    print("VERDICT %s NOT-RUN unknown-op=%s sha=%s md5=%s" % (name, op, c["sha"], c["group_0_md5"]))
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

TAIL_BAR = 2.0 ** -6
HEAD = 1e-2

def metrics(got, want):
    a = got.astype(np.float32)
    b = want.astype(np.float32)
    d = np.abs(a - b)
    head = np.abs(b) >= HEAD
    max_abs = float(d.max()) if d.size else 0.0
    tail_abs = float(d[~head].max()) if (~head).any() else 0.0
    max_rel = float((d[head] / np.abs(b[head])).max()) if head.any() else 0.0
    ulps = ulp16(got, want)
    max_ulp = int(ulps[head].max()) if head.any() else 0
    finite = bool(np.isfinite(a).all())
    return max_abs, tail_abs, max_rel, max_ulp, finite

if op in ("gelu", "gelu_longhand"):
    if op == "gelu":
        # (i) vs (iii): THE BAR — the (iii) is the dump-written want_iii.bin (f32), cast to
        # its f16 image for the ulp metric.
        want = np.fromfile(cell + "/want_iii.bin", dtype=np.float32, count=n).astype(np.float16)
        alt = gelu_true(x).astype(np.float16)       # reported distance to the true tanh
    else:
        want = gelu_true(x).astype(np.float16)      # (ii) vs tanh: reported, no tight bar
        alt = None
    if got.size != want.size:
        print("VERDICT %s NOT-RUN short-readback %d of %d sha=%s md5=%s" % (name, got.size, want.size, c["sha"], c["group_0_md5"]))
        sys.exit()
    max_abs, tail_abs, max_rel, max_ulp, finite = metrics(got, want)
    extra = ""
    if alt is not None:
        _, _, _, ulp_true, _ = metrics(got, alt)
        extra = " ulp_true=%d" % ulp_true
    if op == "gelu":
        ok = (max_ulp <= 8) and (tail_abs <= TAIL_BAR) and finite
    else:
        ok = (tail_abs <= TAIL_BAR) and finite
    # ⛔ RUN-4 DIAGNOSTICS — the x at max ulp (over the head, the lanes the bar covers),
    # got vs (iii) vs true; the whole cell as (x, got, want_iii, want_true) CSV; and on a
    # failing longhand the first 8 lanes got vs want (max_rel 1 with finite values means
    # the chain computes something else — the shape of the divergence, lane by lane).
    um = ulp16(got, want)
    head = np.abs(want) >= HEAD
    um_head = np.where(head, um, -1)
    i = int(np.argmax(um_head)) if um_head.size else 0
    if op == "gelu":
        print("DIAG %s x-at-max-ulp i=%d x=%.6g got=%.6g want_iii=%.6g want_true=%.6g lane_ulp=%d"
              % (name, i, x[i], got[i], want[i], alt[i], int(um_head[i])))
    else:
        print("DIAG %s x-at-max-ulp i=%d x=%.6g got=%.6g want_true=%.6g lane_ulp=%d"
              % (name, i, x[i], got[i], want[i], int(um_head[i])))
    with open(cell + "/cell.csv", "w") as f:
        f.write("x,got,want_iii,want_true\n")
        for j in range(n):
            w3 = ("%.9g" % want[j]) if op == "gelu" else ""
            wt = ("%.9g" % alt[j]) if alt is not None else ("%.9g" % want[j])
            f.write("%.9g,%.9g,%s,%s\n" % (x[j], got[j], w3, wt))
    if op == "gelu_longhand" and not ok:
        for j in range(min(8, n)):
            print("LANE %s [%d] x=%.6g got=%.6g want=%.6g" % (name, j, x[j], got[j], want[j]))
else:
    if got.size != want.size:
        print("VERDICT %s NOT-RUN short-readback %d of %d sha=%s md5=%s" % (name, got.size, want.size, c["sha"], c["group_0_md5"]))
        sys.exit()
    max_abs, tail_abs, max_rel, max_ulp, finite = metrics(got, want)
    extra = ""
    if op == "reduce_sum":
        ok = (max_rel <= 2.0 ** -5) and (tail_abs <= TAIL_BAR) and finite
    else:
        ok = (max_ulp <= 8) and (tail_abs <= TAIL_BAR) and finite

kind = "PASS" if ok else "FAIL"
print(
    "VERDICT %s %s op=%s m=%d c=%d max_abs=%.6g tail_abs=%.6g max_rel=%.6g max_ulp=%d%s finite=%d artifact=%s sha=%s md5=%s"
    % (name, kind, op, m, cols, max_abs, tail_abs, max_rel, max_ulp, extra, int(finite), cell, c["sha"], c["group_0_md5"])
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

pass=0; fail=0; notrun=0; fenced=0; missing=0; runnable=0; total=0
for c in $cell_list; do
  total=$((total + 1))
  # A MISSING-LOWERING cell has no group_0 — it counts as its own kind, never a skip.
  if grep -q "missing_lowering" "$c/cell.json" 2>/dev/null; then
    missing=$((missing + 1))
    continue
  fi
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
echo "PASS=$pass FAIL=$fail NOT-RUN=$notrun FENCED=$fenced MISSING-LOWERING=$missing CELLS=$runnable TOTAL=$total"
if [ "$fail" -gt 0 ] || [ "$notrun" -gt 0 ] || [ "$fenced" -gt 0 ] || [ "$missing" -gt 0 ] || [ "$pass" -lt "$runnable" ]; then
  echo "EW_CARD_GATE_RUN_INCOMPLETE — every runnable cell needs a PASS at its own stamp, and no cell may stand MISSING-LOWERING" >&2
  exit 1
fi
echo EW_CARD_GATE_RUN_DONE
' _ "$ROOT" "$TREE" "$LOG"
