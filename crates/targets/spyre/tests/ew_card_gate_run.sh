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
#   rmsnorm: max_ulp <= ceil(ceil(log2 c)/2) + 2 over the head, c the cell OWN swept width
#   — THE N-DEPENDENT BAR, stated BEFORE the control re-run (moe-lead 2026-11-07: the flat
#   8 was stated for widths <= 4096; a bar is a function of N derived from the op law,
#   never a constant calibrated on the widths already measured). DERIVATION, from the op
#   law: the mean(x^2) reduce is the fp16 summeanmaxexx2 template (the dump-stated accum
#   df), where EVERY accumulation level is computetype="FMA16" over SEN169_FP16-allocated
#   registers (summeanmaxexx2.ddl, the deeptools C++ tree by direct path: the tensors and
#   internal tensors are SEN169_FP16 :23-35, the chunk accumulate :277, across-subchunks
#   :336, PE->SFP :456, the core-to-core psum start/middle/end :528/:569/:547, the final
#   SFP combine :602; the f32 flavor is a SEPARATE template, summeanmaxexx2_fp32.ddl,
#   IEEE_FP32 throughout, taken only when the reduce own df is fp32) — so the accumulation
#   error unit is eps = 2^-11 per level, and:
#     E[x^2] carries a relative error <= ceil(log2 N) * eps — a tree reduction, one eps
#       per level, the same model the ruled reduce_sum bar states over this same template;
#     the E f16 write adds 1/2 eps relative; the rsqrt image, the x*r rounding and the
#     *gain rounding are each <= 1 output ulp; the rsqrt HALVES the E error.
#   At the worst lane position (just above a power of two, where one output ulp is exactly
#   the 2^-11 relative unit) that sums to ceil(ceil(log2 N)/2) + 2 ulp — 8 at N=4096 and
#   N=2048 (the old bar, now derived instead of asserted: never looser below them), 9 at
#   N >= 8192. The SAME bar is applied to the split and the unsplit rmsnorm cells (the
#   control pair): only the split differs, so if the unsplit cell also exceeds it the
#   model's eps or tree assumption is refuted — a finding to investigate, never a bar to
#   move.
#   rmsnormunit: max_ulp <= 8 over the head, unchanged — the flat bar the EW gate stated
#   for it; the N-dependent derivation above is stated for the rmsnorm cells this gate
#   adds, and no rmsnormunit cell changes.
#   gelu (the folded cell): max_ulp <= 8 over the head against the FEST-modeled gelufwd
#   form (see (iii) below). Reason: gelufwd is ONE device primitive, and the cell must be
#   the f16 image of its own defining polynomial. The distance to the TRUE tanh form is
#   also reported.
#   gelu_longhand (the unfolded cell): the tail law plus finiteness over the REPRESENTABLE
#   domain (THE OVERFLOW LAW below). The distance to the true tanh form is REPORTED with no
#   invented tight bar — the longhand lowers through the exp island (exp.smc, 13 SFP
#   instructions) which this host does not model, so any tight number would be tuned, not
#   derived.
#   ⛔ A CELL WITHOUT A STATED BAR PRINTS REPORTED, NEVER PASS (run-8, moe-lead 2026-11-07):
#   the longhand's verdict kind is REPORTED when the tail law and finiteness hold, FAIL when
#   they do not — the distance to the true form is STATED, not judged, and a judged-sounding
#   PASS on it would claim a bar nobody stated. REPORTED counts as its own tally kind: the
#   fence treats it as COMPLETE (the cell ran and its laws held) but NOT a PASS, so a
#   runnable cell needs a PASS or a REPORTED at its own stamp for the gate to close.
#   reduce_max: EXACT — bit equality with the f16 max of the STAGE IMAGE of the inputs,
#   0 ulp, no headroom. The op law is a SELECTION: the output IS one of the values the
#   device saw, exactly. The device does not see the raw f16 fills — the stage codec
#   re-encodes them IEEE->SEN169 (1 sign, 6 exp bias 31, 9 mantissa bits — the sen169_bits
#   law, emit/mod.rs), and an f16 value with an ODD mantissa sits exactly halfway between
#   two SEN169 values and the codec rounds the tie UP: measured on card (cg1113r-1a0882da,
#   the max split): the row maxima 0x48a5 and 0x4893 — SEN169 fraction slots 82.5/512 and
#   73.5/512, both EXACT ties — both came back +1 f16 ulp as 0x48a6 and 0x4894, values
#   present nowhere in the raw fills, which is why the raw-f16 reference was wrong. The
#   readback decodes a SEN169 value exactly (a 9-bit mantissa is an even-mantissa f16), so
#   the exact law compares against the STAGE IMAGE of the row max: odd mantissa +1 ulp,
#   even mantissa unchanged — and the map is monotone, so the image of the max IS the max
#   of the images. ulp_vs_rawmax is REPORTED as the diagnostic that separates a codec tie
#   from a real defect; a difference past the image is a lowering defect, never a bar to
#   widen.
#   reduce_sum (RULED by the auditor 2026-11-07 — a bar calibrated on the run it judges
#   proves nothing, so every input is door-stated, never runner-derived): per row,
#     |got - want| <= (T + ceil(log2(slice_lanes))) * eps * sum|x| + 0.5 * ulp_f16(|want|)
#   with T the split trip count, slice_lanes the slices swept reduce widths, and eps the
#   accumulator error unit: 2^-11 when the accum is fp16 on the device, 2^-24 when fp32.
#   THE FP16 UNIT IS CITED, NOT FITTED (moe-lead 2026-11-07): the fp16 sum template is
#   ddc/ddl_templates/summeanmaxexx2.ddl in the deeptools C++ tree, read by direct path —
#   every tensor is data_type="SEN169_FP16" (:23) and every sum accumulation level is
#   computetype="FMA16" over those fp16-allocated registers: the chunk accumulate :277,
#   across-subchunks :336, PE->SFP :456, the core-to-core psum start/middle/end
#   :528/:569/:547, and the final SFP combine :602 — a 16-bit FMA at every level, no f32
#   stage anywhere in the fp16 template (the f32 flavor is the separate
#   summeanmaxexx2_fp32.ddl, IEEE_FP32 throughout, taken only when the reduce own df is
#   fp32). The measured 24-abs at cg1113r-1a0882da (T=2, 2x524288 lanes, accum fp16)
#   sits inside the ruled bound at eps=2^-11, as the citation says it must.
#   T, slice_lanes and the accum df are the SPLIT FACTS the dump writes into cell.json off
#   the door OWN emitted ops (the slice descriptors N_.out_ widths and OUTPUT wordLength);
#   a reduce_sum cell without them is a named NOT-RUN, never a guessed bar. The formula is
#   the auditor own ruling, taken verbatim: T trip terms plus ceil(log2(slice_lanes))
#   reduction terms, each one eps of the row absolute magnitude sum|x|, plus half an ulp of
#   the final f16 image. err_over_abssum (err / sum|x|) and err_over_sigmasqrtn
#   (err / (sigma_row*sqrt(cols))) are REPORTED unbarred beside it.
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
# ⛔ THE EVERY-BOUND-TENSOR LAW (run-5, moe-lead 2026-11-06; read-set amendment 2026-11-07):
#   the runner stages EVERY worker-bound tensor the bundle's OPS READ, and REFUSES BY NAME
#   to launch a cell whose ops read a bound tensor it did not stage. The protected
#   invariant is NO OP READS A SOURCE NOBODY WROTE. The expected set is the `reads` rows
#   the dump writes from the emitted ops' own operand bindings (`EmittedOp`'s
#   `arg_bindings`, the same walk the dxp input states) — every reserved tid an input
#   binding names — never the placement list: a placed-but-UNREAD reserved tensor
#   (RMS_HALF/RMS_INVCOLS, placed for every RmsNorm node) is a layout fact, not a
#   correctness one, and the layout may place what the program never reads. The values are
#   the `scalarmul_scales` (tid, value) rows the dump writes from the layout's own scale
#   registry (index i at scalarmul_scale_tid(i)), staged through `bundle_run`'s
#   `const:t<id>=<value>` marker. A read tid with no staged value is a named NOT-RUN, never
#   a launch: `bundle_run` zeroes every segment holding no staged weight, and a zero in a
#   scale slot is SILENT — measured on card at 4f71242b9f (cgpx-4f71242b), where two
#   never-staged registry scales made the longhand's cubic chain ALL ZEROS (0·(x+0·x³)=0)
#   and the exp cell ALL ONES (exp(0)=1) while every launch reported success, and the
#   rmsnormunit eps passed only because 1e-6 is below f16 resolution (identical f16 image
#   at eps=0 — a latent gap, not a proof). A staged value nothing reads is the same refusal
#   the other way round: the two lists must agree exactly, or the cell does not run.
#
# ⛔ THE OVERFLOW LAW (run-6, moe-lead 2026-11-06, four clauses): the host model states its
#   chain PER STAGE, every stage at the program's own precision, and a lane whose any stage
#   f16 image is non-finite OVERFLOWED UPSTREAM — past an f16-overflow operand the device
#   arithmetic is its own law (not IEEE, not host-modelable), so:
#   (1) an overflowed lane is REPORTED — count, x, got and the stage that overflowed —
#       never barred;
#   (2) a lane whose WANT is non-finite (the stored value itself leaves f16 range) compares
#       by EQUALITY: inf == inf, nan positions match;
#   (3) every bar (head/tail/ulp/rel) applies over the REPRESENTABLE domain only — lanes
#       that overflowed upstream or carry a non-finite want are outside it;
#   (4) a NON-FINITE device value on a lane the host model says is fully representable is a
#       FAIL regardless of bar — a device or lowering defect, never overflow.
#   The overflowed decision comes from the HOST MODEL ONLY, never from the device output,
#   so the device cannot excuse its own lanes. Measured on card (cgpx-4f71242b,
#   gelu_longhand m1 c2048): the chain's exp stage overflows f16 at 46 of 2048 lanes (v up
#   to 41.5, exp(v) ~ 1e18); the device longhand returned +inf at only 3 of them (x = 5.98,
#   7.46, 6.05 — where its fdiv(x, 1+inf) gives -inf, a non-IEEE device fact: at the other
#   43 lanes the same fdiv gives tiny positives (8e-6 to 1e-7) or zero, and g = x - s comes
#   out finite and correct). No shipped kernel is affected — the shipped gelu_fwd spells
#   exp(-v).
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
  # map). The read-back states m rows for every op — the launcher reads the WHOLE placement
  # back at stick:t=m (bundle_run.rs: read_tensor at p.size/2, restick at (m, size/2/m)), and
  # the verdict half decodes the reduce accum at the door lane law (see THE ACCUM LAW
  # there): row r value at host (r, 0) of that read-back.
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
# THE EVERY-BOUND-TENSOR LAW (header): the expected set is the dump-written reads rows —
# every reserved tid an op of the bundle READS (the emitted ops own arg_bindings), never
# the placement list (a placed-but-unread reserved tensor is a layout fact); the values are
# the launch map own scalarmul_scales rows. The two lists must agree exactly.
pf = json.load(open(sys.argv[1] + "/placements.json"))
expected = sorted(int(t) for t in (pf.get("reads") or []))
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
  # THE EVERY-BOUND-TENSOR LAW: a reserved tensor an op READS with no staged value is a
  # named NOT-RUN, never a launch — a missing constant would read as a zeroed segment,
  # silently (the ALL-ZEROS cubic of run cgpx-4f71242b). A staged value nothing reads is
  # the same refusal the other way round.
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
    raw_want = None
    if op == "reduce_sum":
        want = rows.sum(axis=1).astype(np.float16)
    else:
        # THE EXACT LAW (header): a selection rounds nothing, so the want is the row max —
        # but over the STAGE IMAGE of the fills, the values the device actually saw: the
        # codec re-encodes each f16 into SEN169 (9-bit mantissa), where an odd-mantissa f16
        # sits exactly on a tie that rounds UP, and decodes a 9-bit value back exactly. The
        # map is monotone, so the image of the row max IS the row max of the images.
        raw_want = rows.max(axis=1).astype(np.float16)
        h = raw_want.view(np.uint16).astype(np.uint32)
        mag = h & 0x7FFF
        want = (((h & 0x8000) | (mag + (mag & 1))) & 0xFFFF).astype(np.uint16).view(np.float16)
    # ⛔ THE ACCUM LAW, from the door OWN out placement — never a guessed stride. Three
    # cited facts fix the decode:
    #   1. the door mints the caller accum as ONE STICK PER ROW, row-blocked —
    #      partial_accum_handle::<RowBlockedTag>(&out.name(), rows, 64, Df::Fp16) at the
    #      reduce door Fp16 arm (lower_ktir_to_superdsc.rs, `pub fn reduce`), and the split
    #      slices and combines run IN PLACE on that same handle
    #      (tiled_op_sdsc_op/reduce.rs, assemble_reduce_split combine leg);
    #   2. a RowBlocked tensor DEVICE order interleaves all rows inside each stick-group
    #      (wire.rs, `StickLayout::group_stride`: "consecutive stick-groups sit rows·eps
    #      apart"), so row r stick is at DEVICE element r·64 of the placement;
    #   3. the placement reserves row_footprint_bytes(1, Fp16) = 1024 B per row (work.rs —
    #      the one-stick width 8-splittable bump), so the t<out> places row spans
    #      stride = size/2/m elements, and the launcher read the WHOLE placement back
    #      (bundle_run.rs: read_tensor at p.size/2) and resticked it at (rows=m, cols=stride).
    # Under that restick, device element r·64+lane lands at host (r, lane) — and column 0 of
    # row r in a row-blocked region is LANE 0 of its stick (the single-shot law the door
    # own rb(&out.name(), rows, 1) states). So the value of row r is host (r, 0) of the FULL
    # [m, stride] read-back — reshape(m, stride)[:, 0] over the whole placement, never a
    # truncated m·64 prefix (run cg1113-ca8a1115: the prefix read compared row 0 reserved
    # span against all four rows and FAILed at 18684 ulp while the card row values were
    # right at their own lanes).
    pf = json.load(open(cell + "/placements.json"))
    pout = [p for p in pf["places"] if p["tid"] == c["out_tid"]]
    if len(pout) != 1:
        print("VERDICT %s NOT-RUN no-out-placement tid=%s sha=%s md5=%s" % (name, c["out_tid"], c["sha"], c.get("group_0_md5")))
        sys.exit()
    stride = pout[0]["size"] // 2 // m
    if stride * m * 2 != pout[0]["size"] or stride % 64 != 0:
        print("VERDICT %s NOT-RUN out-placement-not-row-sticks size=%d m=%d sha=%s md5=%s" % (name, pout[0]["size"], m, c["sha"], c.get("group_0_md5")))
        sys.exit()
    got = np.fromfile(cell + "/out.bin", dtype=np.float16, count=m * stride)
    got = got.reshape(m, stride)[:, 0]
    # THE SUM BAR INPUTS (header, ruled 2026-11-07): T, slice_lanes and the accum df come
    # from the SPLIT FACTS the dump wrote off the door OWN emitted ops — never re-derived
    # here. A reduce_sum cell without a complete, self-consistent set is a named NOT-RUN:
    # a bar calibrated on numbers this runner guessed proves nothing.
    bound = None
    if op == "reduce_sum":
        split = c.get("split") or {}
        T = split.get("time")
        lanes = split.get("slice_lanes") or []
        adf = split.get("accum_df")
        if not isinstance(T, int) or T < 1 or len(lanes) != T or adf not in ("fp16", "fp32"):
            print("VERDICT %s NOT-RUN no-door-stated-split-facts split=%s sha=%s md5=%s"
                  % (name, split, c["sha"], c.get("group_0_md5")))
            sys.exit()
        eps = 2.0 ** -11 if adf == "fp16" else 2.0 ** -24
        sig = rows.std(axis=1).astype(np.float32)
        abssum = np.abs(rows).sum(axis=1).astype(np.float32)
        err = np.abs(got.astype(np.float32) - want.astype(np.float32))
        # ulp_f16 at |want|: the normal spacing 2^(floor(log2 v)-10), the subnormal and
        # zero spacing 2^-24 (one min-subnormal step).
        w = np.abs(want.astype(np.float32))
        with np.errstate(divide="ignore"):
            expo = np.floor(np.log2(np.where(w >= 2.0 ** -14, w, 1.0)))
        ulp = np.where(w >= 2.0 ** -14, 2.0 ** (expo - 10), 2.0 ** -24)
        bound = (T + math.ceil(math.log2(max(lanes)))) * eps * abssum + 0.5 * ulp
        # reported, never barred: the sum error against the row absolute magnitude and
        # against the row random-walk scale sigma*sqrt(N) (a cancelling sum relative error
        # is not a metric — see the reduce_sum bar in the header).
        sum_report = " err_over_abssum=%.6g err_over_sigmasqrtn=%.6g split_T=%d split_lanes_max=%d accum_df=%s" % (
            float((err / np.maximum(abssum, 2.0 ** -24)).max()),
            float((err / np.maximum(sig * math.sqrt(cols), 2.0 ** -24)).max()),
            T, max(lanes), adf,
        )
    else:
        sum_report = ""
else:
    print("VERDICT %s NOT-RUN unknown-op=%s sha=%s md5=%s" % (name, op, c["sha"], c["group_0_md5"]))
    sys.exit()

# THE OVERFLOW LAW (header): the host chain model, PER STAGE, at the precision the program
# itself carries — the f16 image of every value the program carries in f16. The overflow
# decision comes from THIS model only, never from the device output, so the device cannot
# excuse its own lanes. An op whose host model has no f16-carried interior stage states no
# stages: every lane is representable and the law changes nothing for it. The mask is at
# the OUTPUT lane count (the longhand output is the full [m, c]).
ovf = None
ovf_stage = None
if op == "gelu_longhand":
    # the longhand spelling itself (GELU_LONGHAND_SRC): an f32 island around the cubic, then
    # e = f16(exp(v)), d = 1+e, s = fdiv(x, d), g = x-s — every value past the island is
    # carried in f16, so its f16 image IS the stage value.
    ovf = np.zeros(n, dtype=bool)
    ovf_stage = np.array([""] * n, dtype=object)
    with np.errstate(over="ignore", invalid="ignore"):
        x32 = x.astype(np.float32)
        v = 1.5957691216057308 * (x32 + 0.044715 * x32 * x32 * x32)
        e = np.exp(v).astype(np.float16)
        d = np.float16(1.0) + e
        s = (x / d.astype(np.float32)).astype(np.float16)
        g = (x - s.astype(np.float32)).astype(np.float16)
    for nm, img in (("exp", e), ("1+exp", d), ("fdiv", s), ("sub", g)):
        bad = (~np.isfinite(img)) & (~ovf)
        ovf |= bad
        ovf_stage[bad] = nm

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

def metrics(got, want, bar):
    # THE OVERFLOW LAW, clause 3: every metric is over the REPRESENTABLE domain (bar) —
    # the lanes that neither overflowed upstream nor carry a non-finite want. finite is
    # clause 4: a non-finite device value on a bar lane is a FAIL regardless of bar, a
    # device or lowering defect, never overflow.
    a = got.astype(np.float32)
    b = want.astype(np.float32)
    d = np.abs(a - b)
    head = bar & (np.abs(b) >= HEAD)
    tail = bar & (~head)
    max_abs = float(d[bar].max()) if bar.any() else 0.0
    tail_abs = float(d[tail].max()) if tail.any() else 0.0
    max_rel = float((d[head] / np.abs(b[head])).max()) if head.any() else 0.0
    ulps = ulp16(got, want)
    max_ulp = int(ulps[head].max()) if head.any() else 0
    finite = bool(np.isfinite(a[bar]).all()) if bar.any() else True
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
    # THE OVERFLOW LAW: clause 2 — non-finite want lanes compare by EQUALITY (inf == inf,
    # nan positions match); clause 3 — the bars cover the representable domain only.
    if ovf is None:
        ovf = np.zeros(want.shape, dtype=bool)
        ovf_stage = np.array([""] * want.size, dtype=object)
    want_nf = ~np.isfinite(want)
    eq_ok = True
    if want_nf.any():
        eq_ok = bool((((got == want) | (np.isnan(got) & np.isnan(want)))[want_nf]).all())
    bar = (~ovf) & (~want_nf)
    max_abs, tail_abs, max_rel, max_ulp, finite = metrics(got, want, bar)
    extra = ""
    if alt is not None:
        _, _, _, ulp_true, _ = metrics(got, alt, bar)
        extra = " ulp_true=%d" % ulp_true
    if op == "gelu":
        ok = (max_ulp <= 8) and (tail_abs <= TAIL_BAR) and finite and eq_ok
    else:
        ok = (tail_abs <= TAIL_BAR) and finite and eq_ok
    # ⛔ RUN-4 DIAGNOSTICS — the x at max ulp (over the head, the lanes the bar covers),
    # got vs (iii) vs true; the whole cell as (x, got, want_iii, want_true) CSV; and on a
    # failing longhand the first 8 lanes got vs want (max_rel 1 with finite values means
    # the chain computes something else — the shape of the divergence, lane by lane).
    um = ulp16(got, want)
    head = bar & (np.abs(want) >= HEAD)
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
    # THE OVERFLOW LAW, clauses 2 and 3 — same law, every op.
    if ovf is None:
        ovf = np.zeros(want.shape, dtype=bool)
        ovf_stage = np.array([""] * want.size, dtype=object)
    want_nf = ~np.isfinite(want)
    eq_ok = True
    if want_nf.any():
        eq_ok = bool((((got == want) | (np.isnan(got) & np.isnan(want)))[want_nf]).all())
    bar = (~ovf) & (~want_nf)
    max_abs, tail_abs, max_rel, max_ulp, finite = metrics(got, want, bar)
    extra = sum_report if op == "reduce_sum" else ""
    if op == "reduce_max":
        # the codec-tie distance (header): how many rows the stage image moved the raw max
        # — the number that separates a SEN tie from a real defect.
        extra = " ulp_vs_rawmax=%d" % int(ulp16(raw_want, want).max())
    if op == "reduce_sum":
        # THE SUM BAR (RULED 2026-11-07, header): per row
        #   |got - want| <= (T + ceil(log2(slice_lanes))) * eps * sum|x| + 0.5*ulp_f16(|want|)
        # with every input door-stated (the dump split facts) — max_abs alone is not the
        # metric: the bound is per row, against the row own absolute magnitude.
        ok = bool((err <= bound).all()) and finite and eq_ok
    elif op == "reduce_max":
        # THE EXACT LAW (header): the output IS one of the stage images — a selection
        # rounds nothing — so the bar is BIT EQUALITY with the image of the row max.
        # max_ulp (vs the image) and ulp_vs_rawmax (the codec ties) stay in the verdict
        # line as the diagnostics that name any difference; a difference past the image is
        # a lowering defect, never a bar to widen.
        ok = bool((got == want).all()) and finite and eq_ok
        if not ok:
            bad = int(np.argmax(got != want))
            print("DIAG %s max-not-exact row=%d raw_max=%.6g stage_image=%.6g got=%.6g"
                  % (name, bad, raw_want[bad], want[bad], got[bad]))
    elif op == "rmsnorm":
        # THE N-DEPENDENT RMSNORM BAR (header, stated before this run):
        #   max_ulp <= ceil(ceil(log2 c)/2) + 2 over the head
        # derived from the op law (the FMA16 accumulation of the fp16 summeanmaxexx2
        # the rsqrt halving, the f16 roundings) — applied IDENTICALLY to the split and the
        # unsplit cells of the control pair. rms_bar rides the verdict line.
        rms_bar = int(math.ceil(math.ceil(math.log2(cols)) / 2.0)) + 2
        ok = (max_ulp <= rms_bar) and (tail_abs <= TAIL_BAR) and finite and eq_ok
        extra = " rms_bar=%d" % rms_bar
    else:
        ok = (max_ulp <= 8) and (tail_abs <= TAIL_BAR) and finite and eq_ok

# THE OVERFLOW LAW, clause 1 — the REPORT, never a bar: count, x, got and the stage that
# overflowed (the first 8 lanes here; the whole cell is in cell.csv). The lanes outside the
# representable domain are stated in the log, never silently dropped.
if ovf.any():
    seen = {}
    for j in np.nonzero(ovf)[0]:
        st = ovf_stage[int(j)]
        seen[st] = seen.get(st, 0) + 1
    stages = ",".join("%s=%d" % (k, seen[k]) for k in sorted(seen))
    print("OVFL %s count=%d stages=%s sha=%s md5=%s" % (name, int(ovf.sum()), stages, c["sha"], c["group_0_md5"]))
    for j in np.nonzero(ovf)[0][:8]:
        print("OVFL-LANE %s [%d] x=%.6g got=%.6g stage=%s" % (name, int(j), x[j], got[j], ovf_stage[int(j)]))

# ⛔ A CELL WITHOUT A STATED BAR PRINTS REPORTED, NEVER PASS (run-8): the longhand is judged
# only by the tail law and finiteness — its distance numbers are stated, not barred, so the
# kind cannot say PASS. Every barred cell keeps PASS/FAIL verbatim.
if op == "gelu_longhand":
    kind = "REPORTED" if ok else "FAIL"
else:
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

pass=0; fail=0; reported=0; notrun=0; fenced=0; missing=0; runnable=0; total=0
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
    # REPORTED is its own tally kind (run-8): complete — the cell ran and its stated laws
    # held — but NOT a PASS, because no bar was stated for it.
    REPORTED) reported=$((reported + 1));;
    FAIL) fail=$((fail + 1));;
    NOT-RUN) notrun=$((notrun + 1));;
    FENCED) fenced=$((fenced + 1));;
  esac
done
echo "PASS=$pass FAIL=$fail REPORTED=$reported NOT-RUN=$notrun FENCED=$fenced MISSING-LOWERING=$missing CELLS=$runnable TOTAL=$total"
if [ "$fail" -gt 0 ] || [ "$notrun" -gt 0 ] || [ "$fenced" -gt 0 ] || [ "$missing" -gt 0 ] || [ $((pass + reported)) -lt "$runnable" ]; then
  echo "EW_CARD_GATE_RUN_INCOMPLETE — every runnable cell needs a PASS or a REPORTED at its own stamp, and no cell may stand MISSING-LOWERING" >&2
  exit 1
fi
echo EW_CARD_GATE_RUN_DONE
' _ "$ROOT" "$TREE" "$LOG"
