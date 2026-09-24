# THE LOWERING RULES, EXTRACTED FROM THE C++

⛔ **The `.ddl` is the per-op DATA; the C++ is the RULE SET.** A template says *which* kind a
statement is; only the C++ says what that kind becomes. Guessing a rule from `.ddl` text is the
failure this file exists to stop — `rotate_num_elements=32` tells you the amount and nothing about
where the rotate goes, and "probably the destination" would compile, pass dbo-opt, and compute the
wrong rope.

⛔ **The `.td` is already mapped** — `src/islands/dataflow_ir/dialects/*.rs` *is* the `.td`, so op
shapes, optional attributes and legality live in the island types and are not restated here.

Authority: `~/git/deeptools` locally, the pod's `/project_src/deeptools` above it. Every rule below is
quoted, not paraphrased.

---

## 1. The compute dispatch — FIVE families, and the set is closed

`SNComputeLowering::constructComputeOperation`, `SNComputeLowering.cpp:1591-1673`.

```cpp
if (is_any_of(compute_node->type_, ComputeOpType::IMA8, ComputeOpType::IMA4,
              ComputeOpType::FMA4, ComputeOpType::FMA8,
              ComputeOpType::FMA16, ComputeOpType::FMA32,
              ComputeOpType::FNMS)) {
  if (failed(constructMACOperation(builder, *compute_node))) {
```
→ **MAC**: `IMA8 IMA4 FMA4 FMA8 FMA16 FMA32 FNMS`. `a*b + acc`, THREE operands.
⛔ `MACC` is **not** in this list. A template spelling with no `ComputeOpType` here must be deferred,
never folded into the MAC family because it looks like one.

```cpp
} else if (is_any_of(compute_node->type_, ComputeOpType::FMUL,
                     ComputeOpType::FSUB, ComputeOpType::PACKMERGE,
                     ComputeOpType::FABSMAX, ComputeOpType::GREATERTHAN,
                     ComputeOpType::GREATEREQUAL, ComputeOpType::LESSERTHAN,
                     ComputeOpType::LESSEREQUAL, ComputeOpType::EQUALTO,
                     ComputeOpType::NOTEQUAL, ComputeOpType::SELECT,
                     ComputeOpType::OR, ComputeOpType::AND)) {
  if (failed(constructBinaryOrTernaryOperation(builder, *compute_node))) {
```
→ **binary-OR-TERNARY**: `FMUL FSUB PACKMERGE FABSMAX GREATERTHAN GREATEREQUAL LESSERTHAN
LESSEREQUAL EQUALTO NOTEQUAL SELECT OR AND`.
⛔⛔ THE COMPARISONS AND `SELECT` AND `PACKMERGE` GO THROUGH THE SAME CONSTRUCTOR AS `FMUL`, and it is
"binary **or ternary**" — the operand count is the template's, not the opcode's. A separate
"compare" path that hard-codes two operands is a second dispatch the reference does not have.

```cpp
} else if (is_any_of(compute_node->type_, ComputeOpType::FMAX,
                     ComputeOpType::FMIN)) {
  if (failed(constructFMINorFMAXOperation(builder, *compute_node))) {
```
→ **FMIN/FMAX — ITS OWN CONSTRUCTOR, AND IT EMITS TWO OPS.** See §2.

```cpp
} else if (is_any_of(compute_node->type_, ComputeOpType::FEST,
                     ComputeOpType::ICVT, ComputeOpType::SPLAT,
                     ComputeOpType::REDUCE, ComputeOpType::SHUFFLE,
                     ComputeOpType::FLOOR, ComputeOpType::CAST)) {
  if (failed(constructUnaryOperation(builder, *compute_node))) {
```
→ **unary**: `FEST ICVT SPLAT REDUCE SHUFFLE FLOOR CAST`. Exactly one input:
```cpp
LogicalResult SNComputeLowering::constructUnaryOperation(
    OpBuilder &builder, const dsc2::ComputeNode &compute_op) {
  if (compute_op.inputs_.size() != 1) {
    return LogicalResult::failure();
```
(`:1274-1278`)

```cpp
} else if (is_any_of(
               compute_node->type_, ComputeOpType::RECIPROCAL,
               ComputeOpType::LAYERNORMSCALE, ComputeOpType::GELU,
               ComputeOpType::SIGMOID, ComputeOpType::EXP_P1,
               ComputeOpType::EXP_P2, ComputeOpType::EXP,
               ComputeOpType::LOG_P1, ComputeOpType::LOG_P2,
               ComputeOpType::GELU_BWD_P1, ComputeOpType::GELU_BWD_P2,
               ComputeOpType::SQRT, ComputeOpType::RSQRT,
               ... ComputeOpType::MUL_I64_TO_I64_SFP)) {
  if (failed(constructOpaqueOperation(builder, *compute_node))) {
```
→ **OPAQUE**: `RECIPROCAL LAYERNORMSCALE GELU SIGMOID EXP_P1 EXP_P2 EXP LOG_P1 LOG_P2 GELU_BWD_P1
GELU_BWD_P2 SQRT RSQRT MISH_P1 MISH_P2 EXX2_32_P1 EXX2_32_P2 EXX2_32_P3 DL16TOFP32 FP32TODL16
DL16TOBF16 SOFTPLUS_P1 SOFTPLUS_P2 IDX32TOADDR MUL_I32_TO_I32 ADD_I32_TO_I32 ADD_I64_TO_I64
MUL_I64_TO_I64_PE MUL_I64_TO_I64_SFP`.

⛔⛔ **SO `rsqrt`, `sigmoid`, `gelu`, `exp`, `sqrt`, `reciprocal` ARE `dataflow.opaque` BODIES, NOT
`vectorchain` OPS.** `constructOpaqueOperation` (`:1559-1583`) builds one op with three dictionaries
and nothing else:
```cpp
auto opaque_op = dataflow::OpaqueOp::create(
    builder, loc, builder.getStringAttr(compute_op.name_), func_name_attr,
    read_write_reg_dic_attr, read_only_reg_dic_attr, param_dic_attr);
```
Emitting `vectorchain::Estimate` for one of these is a different instruction stream.

```cpp
} else {
  emitError("Unknown compute operation in constructComputeOperation");
  return LogicalResult::failure();
}
```
→ the five families are **exhaustive** over the reference's own set. `computetype="assign"` is in
**none** of them, which is consistent with `assign` being the AutoShuffler's expansion (a layout
transform resolved before this dispatch) rather than a compute.

## 2. FMIN/FMAX is a COMPARE plus a SELECTION, and the output depends on the destination

`constructFMINorFMAXOperation`, `SNComputeLowering.cpp:1240-1269`.

```cpp
vectorchain::VectorChainElementWiseCompareOperator op_name =
    compute_op.type_ == FMAX
        ? vectorchain::VectorChainElementWiseCompareOperator::compare_gt
        : vectorchain::VectorChainElementWiseCompareOperator::compare_le;

auto compare_op = vectorchain::ElementWiseCompareOp::create(
    builder, loc, bool_type, inputs[0], inputs[1], mask_op, ..., op_name);

auto select_op = vectorchain::ElementWiseSelectionOp::create(
    builder, loc, result_type, compare_op.getResult(), inputs[0], inputs[1],
    mask_op, ...);
```
⛔ `FMAX` is `compare_gt`, `FMIN` is `compare_le` — and then a SELECTION over the two inputs. So
`FMAX -> BinaryOp::Max` is **wrong**: it is two ops, not one.

⛔ AND WHICH RESULT LEAVES DEPENDS ON THE DESTINATION COMPONENT:
```cpp
if (is_any_of(compute_op.outputs_[i], PESTATE, SFPSTATE)) {
  ... constructComputeOutputOperand(builder, loc, bool_type, compute_op,
                                    compare_op.getResult(), i)
} else {
  ... constructComputeOutputOperand(builder, loc, result_type, compute_op,
                                    select_op.getResult(), i)
```
A state-register destination takes the **i1 compare**; anything else takes the **selection**.

## 3. A compute OPERAND is constructed in the compute's OWN region, from its source component

`constructComputeInputOperandAndAddToList`, `SNComputeLowering.cpp:674-700`.

```cpp
} else if (compute_op.inputs_[index] == PT) {
  auto src_unit =
      retrieveGetUnitOpInSameCore(builder, PTROW7, core_id_, corelet_id_);
  auto data =
      dataflow::ReceiveOp::create(builder, loc, result_type, src_unit, ...);
  input_data = data.getResult();
} else if (compute_op.inputs_[index] == LXLU) {
  auto src_unit =
      retrieveGetUnitOpInSameCore(builder, LXLU, core_id_, corelet_id_);
  auto data = dataflow::ReceiveOp::create(...);
```
⛔⛔ **`PT` AS AN OPERAND SOURCE IS `PTROW7` — THE LAST ROW, NOT A SPAN.** That is where the systolic
array's result leaves. A span is not a row, and this is the reference resolving the span itself.

⛔ AND AN OPERAND WHOSE SOURCE IS ANOTHER UNIT BECOMES A `dataflow.receive` INSIDE THE COMPUTE'S OWN
REGION, with **no `ddl.data_transfer` statement involved**. A generator that only reads a local bound
by an explicit transfer misses every operand of this shape.

## 4. `rotate_num_elements` — on the SOURCE, after the load, LXLU only

`SNTransferLowering.cpp:2237-2245`.

```cpp
if (transfer_->rotateNumElements_ > 0) {
  DT_CHECK_MSG(comp_ == LXLU, "Rotation is allowed only in LXLU");
  auto rot_element = mlir::arith::ConstantIndexOp::create(
      loop_builder, load_op.getLoc(), transfer_->rotateNumElements_);
  auto rot_op = vectorchain::RotateOp::create(
      loop_builder, rot_element.getLoc(), result_type,
      load_op.getResult(), rot_element, ...);
  load_op_result = rot_op.getResult();
} else {
  load_op_result = load_op.getResult();
}
```
The rotate REPLACES the load's result, so the send carries the rotated vector. Enforced at ingestion
too — `ddl_conversion.cpp:1231-1236`:
```cpp
if (newNode->src_.unit_ != SenComponents::LXLU) {
  transfer_op.emitError(
      "Attribute rotate_num_elements was specified on the data_transfer "
      "operation. The source must be LXLU.");
```

## 5. The port identity is `data_connect=`, and producer vs consumer is POSITIONAL

`Ddc::createDataConnectMetadata`, `ddcv1.cpp:3286-3320`.

```cpp
auto& dcMap = metadata.dataConnects_;
...
if (node->nodeType_ == dsc2::ScheduleNode::TRANSFER) {
  const auto* transfer = static_cast<dsc2::TransferNode*>(node);
  if (transfer->src_.unit_ != SenComponents::CONSTANT) {
    dcMap[transfer->srcLdsAndLoopOffsets_.dataConnect_].insertConsumer(node);
  }
  for (const auto& di : transfer->dstLdsAndLoopOffsets_)
    dcMap[di.dataConnect_].insertProducer(node);
} else if (node->nodeType_ == dsc2::ScheduleNode::COMPUTE) {
  ...
    if (compute->inputs_[ind] != SenComponents::CONSTANT) {
      dcMap[compute->inputsLdsAndLoopOffsets_[ind].dataConnect_]
          .insertConsumer(node);
    }
  for (const auto& di : compute->outputsLdsAndLoopOffsets_)
    dcMap[di.dataConnect_].insertProducer(node);
```
- keyed by the `data_connect` **string** (`ddc_metadata.h:194`
  `std::unordered_map<std::string, DataConnect> dataConnects_;`)
- transfer **src** → consumer; transfer **dst** → producer
- compute **inputs** → consumer; compute **outputs** → producer
- ⛔ `SenComponents::CONSTANT` **excluded on both sides** (`:3295`, `:3304`) — a constant is an
  immediate and never joins a connect
- and every connect must have a producer (`:3323-25`):
```cpp
for (const auto& [label, dc] : dcMap) {
  if (dc.producers_.empty()) {
```

## 6. `ddl.core_to_core_communication` binds FOUR HANDLES and emits nothing

`DdlOps.td:755-780` (the op's own description):
> This op is used to generate multiple handlers that can later be used to express operations that
> include core to core communication patterns, like accumulation (e.g. psum). … as well as the next
> core id from the point of view of every core and corelet involved in an operation, completely
> automatically.

`%psum_start`/`%psum_end` are predicates a `ddl.if` branches on; `%next_core`/`%prev_core` appear in a
`ddl.unit`'s second operand slot (`:768`
`%sfp_psum_src02 = ddl.unit(%outtensor, %prev_core) {unit="sfpring", ...}`). Which core starts the
chain is a POSITION, so those `ddl.if`s cannot be flattened and need `scf.if`.

## 7. Absolute stage extents are a CONSTRAINT SYSTEM a search satisfies

`ddcv1.cpp:785-901`, and the callers at `:967`, `:1194`, `:1377`.

```cpp
for (const auto& dsInfo : currDsc->primaryDsInfo_) {
  dsTypeStickSizePerDim[dsInfo.first] =
      currDsc->getCumulativeStickSizes(dsInfo.first);
}

auto checkConstraints =
    [&](const DataStructDims& ds, const Metadata::Datastage& dsMetadata,
        PrimaryDimTypes dimToCheck, bool allowEpilogue) -> bool {
```
and the search:
```cpp
while (!checkConstraints(ds, dsMetadata, dim, false)) {
```
⛔ The ratio→element scaling (`getCumulativeStickSizes`) and the checker are ONE unit in ONE function.
Trip counts need none of it — a loop's own two stages give a dimensionless ratio — but a view's
absolute extent does. See `Tile::owed`.

## 8. Transfer kind is total over three cases

`DataTransferLowering.cpp:165-172`, `:255`, `:274`, `:293`, `:311`, `:424`, `:495`:
memref→memref is `agen.composite_load_and_store`; memref→FIFO is `agen.vector_load` +
`dataflow.send`; FIFO→memref is `dataflow.receive` + `agen.vector_store`. A memory-to-memory transfer
must run on an L3 half — `Helper.cpp:2177-2179` returns a bare `failure()` otherwise.

## 9. Precision is the compute's OPCODE, not the tensor's dtype

`DSC2ToDataflowIR.hpp:54-71` — `stringifyComputePrecision` takes a `ComputeOpType` and maps
`FMA16 -> "fp16"`, `FMA8 -> "fp8"`, `IMA4 -> "int4"`; "used to identify the MAC op code used in the
units". And `SNComputeLowering.cpp:1605-1615` prefixes `"mx"` when an operand's
`scaledLdsCategory_` is not `REGULAR_TENSOR`.

## 10. Legal wire widths are 2 bytes, 16 bytes, or a whole stick

`Helper.cpp:1690-1706` (load) and `:1766-1777` (store): `splat2b`/`masked2b`, `zpad16b`/`masked16b`,
or a full stick. A sub-stick payload rides a full stick and the AGEN access SETS carry the live
extent. Three f16 is six bytes and is none of them — that was the *"'dataflow.send' op unsupported
ldtype"* refusal.
