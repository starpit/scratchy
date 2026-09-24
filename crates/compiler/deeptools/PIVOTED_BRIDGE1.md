# Pivoted bridge 1 — what is left, and what it is NOT

```
SubtileTape ──► SuperDSC ──► [ Statement tree ] ──► DataflowIR ──► init_binary
             ▲            ▲                     ▲               ▲
   lower_subtile_tape   THIS IS THE WORK   superdsc_to_        dbo-opt
   _to_superdsc.rs                         dataflow_ir         --from-dfir
   12,597 lines, works                     23,646 lines,       verified
                                           110/110 entries
```

## ⛔ CORRECTION: THIS IS NOT A 25,000-LINE ddc PORT

An earlier version of this note said the remaining work was porting `ddc` (18,830 cpp
lines) plus `L3DlOpsScheduler` (8,033). **That was wrong.** The schedule is not ddc's
code — it is **data in the DDL templates**, and all 218 are already parsed in this crate.
Op counts across `ddl_templates/*.ddl`:

| DDL op | count | the port's `Statement` variant |
|---|---|---|
| `ddl.compute` | 485 | `Compute` |
| `ddl.loop` | 465 | `Loop` |
| `ddl.data_transfer` | 464 | `Transfer` |
| `ddl.if` + `condition` + `_or` + `_and` | 871 | `Condition` |
| `ddl.sync` | 36 | `Sync` |
| `ddl.allocate` | 319 | the placement, not a `Statement` |
| `ddl.datastage_constraint` | 133 | the extents system |

ddc's bulk is the C++'s GENERALITY — 218 templates, conv2d, pooling, LSTM, topk,
depthwise, indirect/paged access, folds, multi-corelet — none of which scratchy reaches.
Scratchy emits **10 op-funcs**: Add, Sub, Mul, Matmul, Batchmatmul, Mean, Silu, Gelufwd,
Tanh, Restickifyophbm. It uses `ACTIVE_CORELETS = 1`, one fold
(`folds_needed → false`), `indirectAllocType_ = "no_indirection"` on every allocation,
and fp16 activations.

And the constraint solver is ALREADY PORTED: `e001_checkConstraints`
(`ddc/ddcv1.cpp:792`, 132 lines) in `superdsc_to_dataflow_ir/shape_constraints.rs`,
along with `e002_createDataConnectMetadata` (`:3283`), `e041_getStickSizes`,
`e071_getCumulativeStickSizes`.

## The one function to fill

`crates/targets/spyre/src/lower_superdsc_to_dataflow_ir.rs` — `Schedule::roots`. It has
the component and the `Dsc` in hand and currently returns `Vec::new()`, which is why the
path compiles and produces no program. Everything else is wired: `Dsc<'c>`'s
`cores`/`num_corelets_used`/`kind`/`folds_needed`/`view` are answered from scratchy's
`Dsc`, and `lower_superdsc_to_dataflow_ir` calls `run_translator` and prints the module.

⭐ IT IS A DATA WALK, NOT A CODE GENERATOR. The tables are already `'static` data:

```rust
pub struct Stmt    { kind: StmtKind, depth: u16, attrs: Attrs,
                     results: &'static [NameId], operands: &'static [Operand],
                     path: &'static [Enclosing] }
pub struct Program { template: Template, op_func: &'static str, bind: &'static str,
                     stmts: &'static [Stmt], roles: &'static [(NameId, Role)],
                     names: &'static [&'static str] }
```

`StmtKind` has one variant per DDL mnemonic and `path` carries the nest, so the walk is:
group `stmts` by `path` to rebuild `Loop`/`Block`/`Condition`, and dispatch each leaf on
`kind`. This runs inside the `#[forward]` expansion, so it is still compile time — there
is no runtime lowering being introduced.

⛔ ONE LIFETIME FACT THAT SHAPES THE CODE. `Emitted::Compute` holds
`ctx: &'s OperandContext<'s>` and `ComputeFamily` holds `&'f [..]` slices, while
`roots<'s>(self, ..)` takes `self` BY VALUE — so a payload built inside `roots` is
dropped at its return. The storage must outlive the walk: put an arena in the `Dsc`
implementor (`OneDsc`) and hand `&'c Arena` to each `Schedule` from `view()`, which takes
`&self` — `bumpalo`/`typed-arena` both allocate through `&self`. This is why `roots`
cannot simply build and return.

⛔ AND THE ORDER IS FORCED: `TRANSFER` BEFORE `COMPUTE`. A `ComputeInput` names a
register or wire that a transfer put there, so the compute leaf is not independently
constructible.

## The golden reference — use it instead of judgment

The C++ pipeline was run on scratchy's OWN staged bundle and succeeded end to end,
producing `init_binary.bin` (378,880 bytes). Preserved under
`~/tmp/ktir_ref/scratchy_bundle/`:

- `input_bundle/` — scratchy's real `bundle.mlir` + 809 `sdsc_*.json`
- `ref_dfir_after_sdsc_to_dfir.mlir` — **31M, 327,412 lines, 1,260 `program_unit` ops**:
  the C++'s own DataflowIR for that bundle, one `module @sdsc_N` per program
- `ref_full.mlir`

Repro:

```
DEEPTOOLS_PATH=/Users/nickm/tmp/dt_src dbo-opt --kEmitSpyreCode \
  --export-dir=<group dir> --mlir-print-ir-after=sbf-sdsc-to-dataflow-ir bundle.mlir
```

⛔ `DEEPTOOLS_PATH` IS REQUIRED FOR THIS RUN AND NOT FOR `--from-dfir`. ddc reads the
`.ddl` templates through it (`ddc/ddl/ddl_conversion.cpp:49`); the `--from-dfir` pathway
never reaches `EnsureDeviceDeclaration` and needs none. Omitting it fails as
*"sbf-ddc: DtException: Please specify DEEPTOOLS_PATH"*, which surfaces as
`sbf-run-scheduler-on-sdsc: failed on program 'sdsc_0'`.

⛔ `--export-dir` IS ALSO THE INPUT DIR: `dbo/src/Pipeline/Pipeline.cpp:187` sets
`bundleDir_ = dbo_options.export_dir`, and `SDSCLoading.cpp:49` RETURNS EARLY when it is
empty — so a run without it reports rc=0 having done nothing.

Every entry ported can be checked against `module @sdsc_N` in that file. `sdsc_16.json`
is a `mul`; the reference lowers it to **3,808 lines** binding twelve unit kinds —
`lxsu, lxlu, lx, sfp, pe, l3su, l3lu, ptrow0, l3ibr, l0su, hbm, sfpstate`.

## What the reference proves about the ABANDONED path

`subtile_to_dataflow_ir` emitted ~120 lines and seven unit kinds for that same `mul` —
about 3% of the program, with **no `l3su`**, so the result never returned to HBM. Its
layout maps were transposed too: the reference writes
`(d0,d1,d2,d3) -> (d3*64 + d2*64 + d1*64 + d0)` (first dim fastest, per
`SNTransferLowering.cpp:98-110`), and that module wrote `(d0,d1) -> (d0*512 + d1)`. Do
not mine it for logic.

Its one dbo-opt refusal is recorded only so nobody re-derives it: an HBM offset used as
an LX address, `1208320 * 2 = 2416640` against `LXSU`'s 21-bit `LRF`
(`sys-arch-spec/sysdef.cpp:369-370`, checked by `progir/progir.cpp:842-852`; LX is 2 MiB,
`sysdef.cpp:586`, less 64 KiB at `:211`).

## Recovered scaffolding

`ddl/lowering/{mod,walk,rules}.rs` (1,949 lines) are restored from `32dc4b480`: `walk.rs`
already partitions a template's statements by unit and rebuilds the nest, and `rules.rs`
carries the C++ compute rules as enums (`LOWERING_RULES.md` holds the quoted sources).
They were written to emit an invented `Emit` vocabulary and must be retargeted at
`Statement`/`Scheduled`. Two known deltas: this crate's `ResolvedCond::Position` carries
`loop_depth: u16` where the walk expects a statement IDENTITY (`loop_stmt`) — a position
standing in for an identity was a real defect — and `build.rs` has no
`lowering::codegen_lowering` hook, which the data-walk approach does not need.

## The steps, in the order the types force

**1. The arena — DONE (`8fa3e1da9`).** `OneDsc` holds `&'c bumpalo::Bump`, borrowed not owned:
`Dsc::view` takes `&self` and returns `Viewing<'c, Self>`, so `&self.arena` on an owned
field is only good for that call's anonymous borrow. `lower_superdsc_to_dataflow_ir` owns
the `Bump` and drops it after `print::run`, the last reader of anything borrowed.

**2. ONE `Transfer` leaf, no nest — the first real milestone.** Where `run_translator`
first returns a program, `print::run` first yields MLIR, and dbo-opt first gives a verdict
on the port's own output. Everything needed is already in the tree:

- ⭐ **A WORKED `DataTransfer` CONSTRUCTION TO COPY: `transfer.rs:8302-8321`**, in the port's
  own tests, with `l3lu_handlers()` beside it for the `Handlers` shape. Do not invent the
  field values — that test shows every one of the 15.
- The DDL statements: `deeptools::generated::PROGRAMS: &[Program]` (`build.rs:1600`), a
  `'static` table. Select by `(op_func, bind)` against scratchy's
  `Dsc.computeOp_[i].opFuncName` and `attributes_.dataFormat_`. `Schedule` does not carry
  the program yet — adding that lookup is part of this step.
- The placements: scratchy's `Dsc.labeledDs_` and the `AllocNode`s in `scheduleTree_`
  (`component_` is `"hbm"`/`"lx"`, `startAddressCoreCorelet_` is the `AddrFold`).
- The closures: `TransferStatement`'s `send`/`store`/`receive` should CALL the already-ported
  entries — `e090_GenerateLoadAndSendFromDataTransferNode`,
  `e091_GenerateLoadAndStoreFromDataTransferNode`,
  `e097_GenerateReceiveAndStoreFromDataTransferNode` — not reimplement them. The caller
  supplies the description; the port owns the lowering.

Verify by diffing the emitted `module` against `module @sdsc_16` in the reference, then
running dbo-opt on it.

**3. The nest.** `Loop`/`Block` from `Stmt.path` into `Scheduled::Band`/`Block`.
`ddl/lowering/walk.rs` already does this grouping and needs only retargeting.

**4. `Compute`.** Legal ONLY after 2, because a `ComputeInput` names a register or wire a
transfer placed. `ComputeFamily` from `ddl/lowering/rules.rs` (the C++ compute rules are
already encoded there, quoted in `LOWERING_RULES.md`); `OperandContext` in the arena.

**5. `Sync`, `Condition`, `StickMask`.** 36 `ddl.sync`. Conditions need
`ResolvedCond::Position` to carry a statement IDENTITY rather than `loop_depth: u16` — a
position standing in for an identity was a real defect, fixed once already in `32dc4b480`.

**6. Iterate per sdsc against the reference**, then the acceptance build on 2b AND 8b.

⭐⭐ WHY THIS CONVERGES WHERE THE LAST ATTEMPT DID NOT: every step has an oracle BEFORE it
needs judgement. Steps 2-5 each diff against `module @sdsc_N` — the C++'s own answer for
our own input — and dbo-opt sits at the end. The abandoned path had neither, which is
exactly how it reached invented addresses and a transposed layout map without anything
catching it.

### Step 2's last layer — the `store` closure, fully mapped

For an L3LU load (HBM to LX) the closure that fires is `store`; `send`/`receive` are for
wire transfers and can `panic!` naming that, as the port's own test does
(`transfer.rs:8349-8355`). The driver passes all four straight through to
`construct_data_transfer` (entry 104) at `driver.rs:698-710`.

`store: &'s dyn Fn(&mut Values, &mut Vec<DfirOp>, Component) -> LoadAndStore` should call
the ported entry 091:

```rust
generate_load_and_store_from_data_transfer_node(
    vals, handlers,
    &LoadAndStoreTransfer {            // transfer.rs:2932 — 11 fields
        storage, core, corelet,
        location: DataLocation::…,     // the (unit, storage) pair
        precision, name, node,
        view_sizes, elem, outer_loops, chunks,
    },
    LoadAndStoreSource::Zero,          // transfer.rs:2885
    |vals, ops, factor| { … },         // the address closure
) -> LoadAndStore
```

⭐⭐ **AND THE ADDRESS-GRANULARITY FACTOR IS THE PORT'S JOB, NOT THE CALLER'S.** The
`address` closure RECEIVES a `Factor`, which the port computed from
`LoadAndStoreTransfer::location` and `precision` via
`address_granularity_multiply_factor` (entry 025, `SNDSCLowering.cpp:156`) —
`addressGranularityScalePerUnit[(unit, storage)] * 8 / bits`, the table at
`sys-arch-spec/sysdef.cpp:531-550` (`L3LU`/`L3SU` to HBM and to LX are 128, `LXLU`/`LXSU`
to LX is 1, so at fp16 the factors are 64 and 0.5). So the caller hands over the RAW
placement out of the `AllocNode`'s `startAddressCoreCorelet_` and lets `Factor::apply`
scale it. Scaling it before handing it over would apply the factor twice.

⛔ THIS IS WHERE THE ABANDONED PATH'S DEFECT WAS STRUCTURALLY IMPOSSIBLE TO MAKE. That
module chose the unit and the address at the same site with no `DataLocation` in play, so
an LX view taking an HBM offset typechecked. Here the storage and the placement arrive
together and the factor is derived from the pair.
