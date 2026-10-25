# 11k — the bake-budget refusal: the tiled law's per-stick fans fold (AUDITOR-RULED exact code)

Run-46 attempt 8: every test leg GREEN (11j holds — the macro compiles, the expansion runs),
and the gated release build fails at the granite-2b fp8 `#[forward]` bake — the FIRST run of
the tiled law at the ladder's rungs. The bake budget refuses six rungs. The refusal is BY
DESIGN ("this refusal is the measurement") and the measurement is now complete and exact.
The coder applies K1 → K2 → K3, verbatim where quoted.

## FORENSICS — the complete measurement (from build.log + the pod's /tmp/superdsc-dump)

**The per-tile law, measured.** The green rung's dump (fp `b252b13e61bd0836f`, mq=2272,
bundle_id.json on the pod) carries the exact per-program census. At the granite-2b
geometries every real GEMM is WIDE (`fp8_linear_kernel.py:338` — n decomposes over the ABI
set into >1 tile at every projection width), so every GEMM tiles by 4 rows, and per tile
per GEMM the emission mints exactly `9 + k_sticks + n_sticks` descriptors:
1 tz + k_sticks tc + 5 quant + 1 mm + 2 dequant + n_sticks to. Measured per-GEMM per-tile:
q 73 (k=2048,n=2048), k/v 49 (n=512), o 73, gate/up/down 169 (n=8192 / k=8192). Each GEMM
carries its OWN activation staging (the dump shows seven: t729..t742 — no cross-GEMM reuse),
so one layer = 751 descriptors per tile + 201 fixed, and the bundle total is exactly
`751·T + 201` where T = window rows / 4. Every refusing number in the log is this formula:

| rung window | T | bundle descriptors | verdict (16,384 / 65,536) |
|---|---|---|---|
| ~56–78 | 14–20 | 10,715 / 12,217 / 13,719 / 15,221 | BAKED GREEN |
| 112 | 28 | 21,229 | refused (descriptors) |
| 128 | 32 | 24,233 | refused (descriptors) |
| 192 | 48 | 36,249 | refused (descriptors) |
| 256 | 64 | 48,265 | refused (descriptors; trips OK) |
| 384 | 96 | 72,397 | refused (descriptors AND trips) |
| 512 | 128 | 96,429 | refused (descriptors AND trips) |

**What the reference pays here.** torch-spyre's kernel launches ONCE per GEMM at any M —
the M-tile loop is the kernel's grid and the K/N stick loops are inside the program. Our
lowering materializes those loops as separate single-trip descriptors: that fan-out is the
gap the budget measures. The budgets themselves are calibrated walls (dxp compile is
superlinear per group; the 40×-unrolled bake ended in `pthread_create` failure) — they do
not move ("budget busts ⇒ report, never bend").

**The fix is the descriptor's own fold.** The tc and to fans are windowed repetitions of
ONE op shape over ONE placement pair — source group `g` and dest stick `g` both advance by
their nest's regular head stride (`g·rows·64` elems, the law the nests spell at :1065-1069).
`EmittedOp::time` + `affine_strides` is exactly this mechanism: `concrete_trips` bumps each
fan tensor's base by `t·stride` per trip and leaves stride-less tensors (the identity
table) fixed — the per-trip files are the SAME ops at the SAME windows the per-g ops carry
today. The render is byte-identical modulo the op-name key; the dxp work (trips) is
UNCHANGED; only the bundle's descriptor count folds. This is not invention: the reference's
intra-kernel stick loop becomes our descriptor's fold, the same relationship the M-tile
grid already has to our tiles.

**Effect (arithmetic, to be measured at attempt 9).** Per tile per GEMM: 11 descriptors
(1 tz + 1 tc + 5 quant + 1 mm + 2 dq + 1 to); bundle = `77·T + 201`. The 112/128/192/256
rungs come under 16,384 (e.g. 256-rung: 5,129) and their trips (unchanged `751·T + 201`)
are under 65,536 — they BAKE. The 384/512 rungs then refuse ON TRIPS (72,397 / 96,429 over
65,536) — the NEXT measured layer, already pinned: the per-bundle trip wall needs the body
split (or staging sharing — the seven per-GEMM activations), ruled as 11l after this lands.
Nothing in 11k touches the trips.

## K1 — the tc loop folds (ktir_matmul_fp8.rs :1124-1146)

OLD (the whole `for g` loop inside staging block (2)):
```rust
                // (2) THE CLONE — one identity matmul per stick group, copying the tile's REAL rows
                // (`M = real`; the source has no rows past `m`, so a full-`ROWS` read would sweep
                // out of footprint). The pad rows keep the zero fill.
                for g in 0..k_groups {
                    ops.push(crate::ir::bridge::tiled_op_sdsc_op::assemble_matmul_placed(
                        &format!("{out}_fq_tc{t}g{g}_op"),
                        crate::sdsc_abstract::MatM::of_token_rows(real),
                        crate::sdsc_abstract::MatN::one_stick(Lanes::FP16),
                        crate::sdsc_abstract::MatK::one_stick(Lanes::FP16),
                        crate::sdsc_abstract::MatY::unbatched(),
                        crate::ir::bridge::tiled_op_sdsc_op::SharedKernelBmmForm::batch_inner_proven(
                            bmm_site::TapeLoweringSite::witness(),
                        ),
                        &a_h,
                        OperandPlacement::at_head(&src_nest, tile.row, g, 0, slabs),
                        &ident_h,
                        DevOff::ZERO,
                        &tile_h,
                        OperandPlacement::at_head(&tile_nest, Idx::<Row>::n(0), g, 0, slabs),
                        sym_id_base,
                        layout,
                    ));
                }
```
NEW:
```rust
                // (2) THE CLONE — ONE identity matmul, TIME-FOLDED over the K stick groups: the
                // g=0 op at full shape, `time = k_groups`, each fan tensor stepping by its nest's
                // head stride (`g·rows·64` elems — the group law the nests spell), the identity
                // table FIXED across trips (empty stride map — `concrete_trips`' no-bump path).
                // `concrete_trips` lands each trip at exactly the address the per-g ops
                // carried, so the render's trips are the same ops at the same windows: the fan
                // is the fold, not new work. (`M = real`; the source has no rows past `m`, so a
                // full-`ROWS` read would sweep out of footprint. The pad rows keep the zero
                // fill.)
                let mut tc = crate::ir::bridge::tiled_op_sdsc_op::assemble_matmul_placed(
                    &format!("{out}_fq_tc{t}_op"),
                    crate::sdsc_abstract::MatM::of_token_rows(real),
                    crate::sdsc_abstract::MatN::one_stick(Lanes::FP16),
                    crate::sdsc_abstract::MatK::one_stick(Lanes::FP16),
                    crate::sdsc_abstract::MatY::unbatched(),
                    crate::ir::bridge::tiled_op_sdsc_op::SharedKernelBmmForm::batch_inner_proven(
                        bmm_site::TapeLoweringSite::witness(),
                    ),
                    &a_h,
                    OperandPlacement::at_head(&src_nest, tile.row, 0, 0, slabs),
                    &ident_h,
                    DevOff::ZERO,
                    &tile_h,
                    OperandPlacement::at_head(&tile_nest, Idx::<Row>::n(0), 0, 0, slabs),
                    sym_id_base,
                    layout,
                );
                tc.time = k_groups;
                tc.affine_strides = vec![
                    BTreeMap::from([("out", nest_group_stride_bytes(m, stk))]),
                    BTreeMap::new(),
                    BTreeMap::from([("out", nest_group_stride_bytes(ROWS, stk))]),
                ];
                ops.push(tc);
```
(`k_groups` is `u32` from `plan.k_stick_groups().get()`; `EmittedOp::time` is `u32` ✓.
The strides align with `op.args` = (A, identity, C) — the tc's mint order.)

## K2 — the to loop folds (ktir_matmul_fp8.rs :1229-1251)

OLD (the whole `for c` loop, comment block (6) included):
```rust
            // (6) THE OUT-SPLICE — the clone form again, N-side: the tile's real rows of each
            // output stick group land at `tile.row` of the output. The union of the tiles' real
            // rows covers `[0, m)` exactly (the plan's own law), so every output byte is written.
            for c in 0..n_groups {
                ops.push(crate::ir::bridge::tiled_op_sdsc_op::assemble_matmul_placed(
                    &format!("{out}_fq_to{t}c{c}_op"),
                    crate::sdsc_abstract::MatM::of_token_rows(real),
                    crate::sdsc_abstract::MatN::one_stick(Lanes::FP16),
                    crate::sdsc_abstract::MatK::one_stick(Lanes::FP16),
                    crate::sdsc_abstract::MatY::unbatched(),
                    crate::ir::bridge::tiled_op_sdsc_op::SharedKernelBmmForm::batch_inner_proven(
                        bmm_site::TapeLoweringSite::witness(),
                    ),
                    &landing_h,
                    OperandPlacement::at_head(&landing_nest, Idx::<Row>::n(0), c, 0, slabs),
                    &ident_h,
                    DevOff::ZERO,
                    &out_h,
                    OperandPlacement::at_head(&out_nest, tile.row, c, 0, slabs),
                    sym_id_base,
                    layout,
                ));
            }
```
NEW:
```rust
            // (6) THE OUT-SPLICE — the clone form again, N-side, TIME-FOLDED over the output
            // stick groups: the c=0 op, `time = n_groups`, the landing side stepping by its
            // ROWS-row group stride and the output side by its m-row one — the tile's real rows
            // of each output stick group land at `tile.row` of the output. The union of the
            // tiles' real rows covers `[0, m)` exactly (the plan's own law), so every output
            // byte is written — one (tile, stick) trip at a time, as before.
            let mut to = crate::ir::bridge::tiled_op_sdsc_op::assemble_matmul_placed(
                &format!("{out}_fq_to{t}_op"),
                crate::sdsc_abstract::MatM::of_token_rows(real),
                crate::sdsc_abstract::MatN::one_stick(Lanes::FP16),
                crate::sdsc_abstract::MatK::one_stick(Lanes::FP16),
                crate::sdsc_abstract::MatY::unbatched(),
                crate::ir::bridge::tiled_op_sdsc_op::SharedKernelBmmForm::batch_inner_proven(
                    bmm_site::TapeLoweringSite::witness(),
                ),
                &landing_h,
                OperandPlacement::at_head(&landing_nest, Idx::<Row>::n(0), 0, 0, slabs),
                &ident_h,
                DevOff::ZERO,
                &out_h,
                OperandPlacement::at_head(&out_nest, tile.row, 0, 0, slabs),
                sym_id_base,
                layout,
            );
            to.time = n_groups;
            to.affine_strides = vec![
                BTreeMap::from([("out", nest_group_stride_bytes(ROWS, stk))]),
                BTreeMap::new(),
                BTreeMap::from([("out", nest_group_stride_bytes(m, stk))]),
            ];
            ops.push(to);
```

## K2b — the stride helper (immediately before `fp8_tile_mm_op`, :841's doc block)

Insert:
```rust
/// One stick-group's byte advance in a `[rows, groups, 64]` fp16 nest — the group law
/// (`g·rows·64` elems) as bytes: the per-trip stride a time-folded fan op carries, minted
/// from the same footprint law the geometry guard compares against (`synth_footprint_bytes`).
fn nest_group_stride_bytes(rows: u32, stk: u32) -> i64 {
    crate::placement::synth_footprint_bytes(&[rows, stk], Df::Fp16) as i64
}
```
If the `as i64` trips clippy as same-type at the check/clippy legs (the helper's return
type vs `i64`), drop the cast and report — the LEG is the verifier, never a silent edit
beyond this one. `BTreeMap` joins the file's `use` list if it is not already in scope
(report the exact import line as applied).

## K3 — the two wide tests move to the folded law (censuses + new assertions)

`the_wide_prefill_tiles_its_rows_and_splices_the_real_ones_back` (:2241+) and
`a_second_gemm_at_the_same_plan_reuses_the_tile_staging` (:2333+) at m=24 (6 tiles,
k=2048 → 32 groups, n=3072 → 48 groups):
- The tc census moves 192 → **6** ops named `fq_tc{t}_op` (one per tile); the to census
  288 → **6** named `fq_to{t}_op`. Every other census (tz, quant kinds, dqa/dqw, mm) is
  UNCHANGED — those ops are untouched.
- Each folded op gains the pinned assertions, in the tests' own idiom:
  `.time == 32` (tc) / `.time == 48` (to); `concrete_trips(&op).len() == 32 / 48`;
  the A-side stride `== nest_group_stride_bytes(24, 64) == 3072` and the C-side
  `== nest_group_stride_bytes(4, 64) == 512` (tc) — mirrored for to (512 / 3072); the
  identity operand's stride map EMPTY.
- The name-form asserts (`ends_with("_op")`, the `{t}` placement) carry over verbatim —
  the folded names still satisfy them (11i's law: tile index before `_op`, the stick index
  is now the fold, not the name).
The coder applies the minimal edits to the two tests' census/assert blocks and REPORTS
the exact diff in the apply message for line-level audit — nothing else in either test
changes. The stale `fq_tc{t}g{g}` / `fq_to{t}c{c}` name forms have no consumer outside
these tests and the emission (verified in 11i's blast radius; the DDL dispatch is by
op_func, placements carry tensors not ops).

## What does NOT change
- The tz, quant chain, mm, dqa/dqw ops — byte-identical, time=1.
- The trips: Σ time per tile per GEMM stays `9 + k_sticks + n_sticks` — the render, the
  dxp files, the group partition, the manifest trip order — all UNCHANGED (the folded
  trips render to the same windows in the same order).
- The narrow arm, the dense early-return, the plan, the placements, the staging law, the
  `quantized` reuse guard (the folded tc sits in the same `if !quantized.contains` block).
- The alias guard `tiled_trips_alias` now SEES the folded ops (time>1) and must PASS them:
  the tc trips write distinct sticks of one tile staging; the to trips write distinct
  (row-fixed, stick-varying) output windows — if it refuses, that is a REPORT, not a fix.

## Apply order and legs
K2b → K1 → K2 → K3, verbatim where quoted, then the K3 test edits as directed. The coder
reports anything else broken it sees — reports, never fixes. Then: auditor local fmt
pre-flight, re-sync /work/attn-auditer-01, run-46 attempt 9. EXPECTED at attempt 9: all
test legs green (the folded law pinned by K3); the build leg proceeds FURTHER — the
112/128/192/256 rungs bake (bundle counts `77·T + 201`), and the 384/512 rungs refuse ON
TRIPS with 72,397 / 96,429 — that refusal is the 11l pin (the body split), already
measured. Nothing commits, nothing pushes.
