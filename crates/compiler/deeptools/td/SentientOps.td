//===-- SentientOps.td -- SentientOps Dialect Operation Definitions -*- tablegen ----==//
//
// IBM Confidential
// (C) Copyright IBM Corp. 2021, 2025
//
// =============================================================================
//
// This file defines the sentient operations in the MLIR operation set.
//
//===----------------------------------------------------------------------===//

#ifndef DCC_SRC_DIALECT_SENTIENT_OPS_TD
#define DCC_SRC_DIALECT_SENTIENT_OPS_TD

#ifndef OP_BASE
include "mlir/IR/OpBase.td"
#endif // OP_BASE

include "dataflow-scheduler/Dialect/Dataflow/DataflowInterfaces.td"
include "dataflow-scheduler/Dialect/OpTraits.td"
include "src/Dialect/Sentient/SentientTypes.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

// Require regions to have sentient.yield.
def ImplicitSentientTerminator
    : SingleBlockImplicitTerminator<"YieldOp">;

def YieldOp : Sentient_Op<"yield", [NoMemoryEffect, Terminator, Pure, ReturnLike,
    MemRefsNormalizable, ParentOneOf<["IfOp", "ForOp"]>]> {
  let summary = "Yield values to parent operation";
  let description = [{
    "sentient.yield yields zero or more SSA values to parent op."
  }];

  let arguments = (ins Variadic<AnyType>:$results);
  let builders = [
    OpBuilder<(ins), [{ build($_builder, $_state, {}); }]>
  ];

  let assemblyFormat = "attr-dict ($results^ `:` type($results))?";
}

def ForOp : Sentient_Op<"for", [
    ImplicitSentientTerminator,
    DeclareOpInterfaceMethods<LoopLikeOpInterface,["getInitsMutable", "getYieldedValuesMutable"]>,
    DeclareOpInterfaceMethods<RegionBranchOpInterface, ["getEntrySuccessorOperands"]>,
    RecursiveMemoryEffects,
    Dataflow_DebugNameOpInterface]> {
  let summary = "for operation";
  let description = [{
    MVLOOP equivalent with only number of iterations specified and execution order is from total iterations to zero.
  }];

  // regLocales/regIndices array convention:
  //   First entry is register info for `bound`,
  //   followed by entries for `initArgs`,
  //   followed by entries for `results`.
  let arguments = (ins Index:$bound, Variadic<AnyType>:$initArgs,
                    OptionalAttr<StrAttr>:$dbgName,
                    SentientRegTypeArrayAttr:$regLocales,
                    I32ArrayAttr:$regIndices, DefaultValuedAttr<BoolArrayAttr,"false">:$programHeader);
  let results = (outs Variadic<AnyTypeOf<[AnyInteger, Index]>>:$results);
  let regions = (region SizedRegion<1>:$region);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "Value":$bound, "::mlir::StringAttr":$dbgName, "ArrayAttr":$regLocales, "ArrayRef<int32_t>":$regIndices, "ArrayRef<bool>":$programHeader,
      CArg<"ValueRange", "{}">:$iterArgs,
      CArg<"function_ref<void(OpBuilder &, Location, Value, ValueRange)>",
           "nullptr">:$bodyBuilder)>
  ];

  let extraClassDeclaration = [{
    using BodyBuilderFn =
        function_ref<void(OpBuilder &, Location, Value, ValueRange)>;

    static StringRef getBoundAttrStrName() { return "bound"; }

    void setBound(Value bound) { getOperation()->setOperand(0, bound); }
    void setIterOperand(unsigned index, Value val) {
      DT_CHECK((hasIterOperands() && index < getNumIterOperands()) &&
             "no iter operands or index out of bound");
      getOperation()->setOperand(index + getNumControlOperands(), val);
    }

    /// Number of induction variables, always 1 for sentient::ForOp.
    unsigned getNumInductionVars() { return 1; }
    Region &getLoopBody();
    void getSuccessorRegions(std::optional<unsigned> index,
                                ArrayRef<Attribute> operands,
                                SmallVectorImpl<RegionSuccessor> &regions);
    Value getInductionVar() { return getBody()->getArgument(0); }
    Block::BlockArgListType getRegionIterArgs() {
      return getBody()->getArguments().drop_front(getNumInductionVars());
    }
    Operation::operand_range getIterOperands() {
      return getOperands().drop_front(getNumControlOperands());
    }
    /// Number of region arguments for loop-carried values
    unsigned getNumRegionIterArgs() {
      return getBody()->getNumArguments() - 1;
    }
    /// Number of operands controlling the loop: bound
    unsigned getNumControlOperands() { return 1; }
    /// Does the operation hold operands for loop-carried values
    bool hasIterOperands() {
      return getOperation()->getNumOperands() > getNumControlOperands();
    }
    /// Get Number of loop-carried values
    unsigned getNumIterOperands() {
      return getOperation()->getNumOperands() - getNumControlOperands();
    }

    /// Get the region iter arg that corresponds to an OpOperand.
    BlockArgument getRegionIterArgForOpOperand(OpOperand &opOperand) {
      DT_CHECK_MSG(opOperand.getOperandNumber() >= getNumControlOperands(),
             "expected an iter args operand");
      DT_CHECK_MSG(opOperand.getOwner() == getOperation(),
             "opOperand does not belong to this scf::ForOp operation");
      return getRegionIterArgs()[
        opOperand.getOperandNumber() - getNumControlOperands()];
    }
    /// Get the OpOperand& that corresponds to a region iter arg.
    OpOperand &getOpOperandForRegionIterArg(BlockArgument bbArg) {
      DT_CHECK_MSG(bbArg.getArgNumber() >= getNumInductionVars(),
             "expected a bbArg that is not an induction variable");
      DT_CHECK_MSG(bbArg.getOwner()->getParentOp() == getOperation(),
             "bbArg does not belong to the scf::ForOp body");
      return getOperation()->getOpOperand(
        getNumControlOperands() + bbArg.getArgNumber() - getNumInductionVars());
    }

    /// Return operands used when entering the region at 'index'. These operands
    /// correspond to the loop iterator operands, i.e., those exclusing the
    /// induction variable. LoopOp only has one region, so 0 is the only valid
    /// value for `index`.
    OperandRange getSuccessorEntryOperands(std::optional<unsigned> index);

    /// Returns the number of invocations of the body block if the loop bounds
    /// are constants. Returns `kUnknownNumRegionInvocations` otherwise.
    void getNumRegionInvocations(ArrayRef<Attribute> operands,
                                 SmallVectorImpl<int64_t> &countPerRegion);

    static StringRef getRegLocaleAttrStrName() { return "regLocales"; }
    static StringRef getProgramHeaderAttrStrName() { return "programHeader"; }


    static StringRef getRegIndicesAttrStrName() { return "regIndices"; }
  }];

  let hasCanonicalizer = 1;
  let hasCustomAssemblyFormat = 1;
}

def IfOp : Sentient_Op<"if", [
    DeclareOpInterfaceMethods<RegionBranchOpInterface>,
    SingleBlockImplicitTerminator<"sentient::YieldOp">, RecursiveMemoryEffects,
    NoRegionArguments,
    Dataflow_DebugNameOpInterface]> {
  let summary = "if-then-else operation";
  let description = [{
    "Sentient.if operation captures the conditional semantics."
  }];

  let arguments = (ins CmpIPredicateAttr:$predicate,
                       AnyTypeOf<[AnyInteger, Index]>:$lhs,
                       AnyTypeOf<[AnyInteger, Index]>:$rhs,
                       OptionalAttr<StrAttr>:$dbgName,
                       SentientRegTypeArrayAttr:$regLocales,
                       I32ArrayAttr:$regIndices);
  let results = (outs Variadic<AnyTypeOf<[AnyInteger, Index]>>:$results);
  let regions = (region SizedRegion<1>:$thenRegion, AnyRegion:$elseRegion);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "CmpIPredicateAttr":$predicate, "Value":$lhs,
                    "Value":$rhs, "ArrayAttr":$regLocales, "ArrayRef<int32_t>":$regIndices,"bool":$withElseRegion)>,
    OpBuilder<(ins "TypeRange":$resultTypes, "CmpIPredicateAttr":$predicate,
                    "Value":$lhs, "Value":$rhs, "ArrayAttr":$regLocales, "ArrayRef<int32_t>":$regIndices,
                    "bool":$withElseRegion)>,
    OpBuilder<(ins "TypeRange":$resultTypes, "CmpIPredicateAttr":$predicate,
                    "Value":$lhs, "Value":$rhs,
                    "ArrayAttr":$regLocales, "ArrayRef<int32_t>":$regIndices,
      CArg<"function_ref<void(OpBuilder &, Location)>",
           "buildTerminatedBody">:$thenBuilder,
      CArg<"function_ref<void(OpBuilder &, Location)>",
           "nullptr">:$elseBuilder)>,
    OpBuilder<(ins "CmpIPredicateAttr":$predicate, "Value":$lhs,
                "Value":$rhs,
                "ArrayAttr":$regLocales,
                "ArrayRef<int32_t>":$regIndices,
      CArg<"function_ref<void(OpBuilder &, Location)>",
           "buildTerminatedBody">:$thenBuilder,
      CArg<"function_ref<void(OpBuilder &, Location)>",
           "nullptr">:$elseBuilder)>
  ];

  let extraClassDeclaration = [{
    OpBuilder getThenBodyBuilder(OpBuilder::Listener *listener = nullptr) {
      Block* body = getBody(0);
      return getResults().empty() ? OpBuilder::atBlockTerminator(body, listener)
                               : OpBuilder::atBlockEnd(body, listener);
    }
    OpBuilder getElseBodyBuilder(OpBuilder::Listener *listener = nullptr) {
      Block* body = getBody(1);
      return getResults().empty() ? OpBuilder::atBlockTerminator(body, listener)
                               : OpBuilder::atBlockEnd(body, listener);
    }
    void getSuccessorRegions(std::optional<unsigned> index,
                               ArrayRef<Attribute> operands,
                               SmallVectorImpl<RegionSuccessor> &regions);
    static StringRef getPredicateAttrStrName() { return "predicate"; }

    static StringRef getRegLocaleAttrStrName() { return "regLocales"; }

    static StringRef getRegIndicesAttrStrName() { return "regIndices"; }
}];

  let hasCanonicalizer = 1;
  let hasCustomAssemblyFormat = 1;
}

//Vector MAC
//xrfReadIncr and xrfWriteIncr are total post-compute register updates over their initial values and can be any integer.
//The semantics of xrf pointer update is the same as sentient hardware:
//xrf read pointer has zero increment within unrolling size and increases by 1 or 2 after execution.
//xrf write pointer increments by 1 for every execution including unrolling.
def Sentient_MacOp:Sentient_Op<"vector_mac", [
    MemoryEffects<[MemWrite]>,
    AttrSizedOperandSegments,
    Dataflow_DebugNameOpInterface]> {
  let summary = "MAC operation";
  let arguments = (ins Optional<Index>:$mask,
                    // the order of pointers is xrfWritePtr and xrfReadPtr.
                    Arg<Variadic<Index>,"xrfWritePtr, xrfReadPtr",[MemWrite]>:$pointers,
                    OptionalAttr<StrAttr>:$dbgName,
                    SentientComputePortAttr:$opA,
                    SentientComputePortAttr:$opB,
                    SentientComputePortAttr:$opC,
                    SentientComputePortArrayAttr:$opAForwarding,
                    SentientComputePortArrayAttr:$opBForwarding,
                    SentientComputePortArrayAttr:$opCForwarding,
                    SentientComputePortArrayAttr:$ResultForwarding,
                    OptionalAttr<SentientFoldModeAttr>:$fold_mode,
                    DefaultValuedAttr<SentientPrecisionAttr, "SentientPrecision::fp16">:$opAPrecision,
                    DefaultValuedAttr<SentientPrecisionAttr, "SentientPrecision::fp16">:$opBPrecision,
                    DefaultValuedAttr<SentientPrecisionAttr, "SentientPrecision::fp16">:$opCPrecision,
                    DefaultValuedAttr<SentientPrecisionAttr, "SentientPrecision::fp16">:$ResultPrecision,
                    DefaultValuedAttr<SentientPrecisionAttr, "SentientPrecision::fp16">:$ComputePrecision,
                    DefaultValuedAttr<SI32Attr, "-1">:$opADataID,
                    DefaultValuedAttr<SI32Attr, "-1">:$opBDataID,
                    DefaultValuedAttr<SI32Attr, "-1">:$opCDataID,
                    DefaultValuedAttr<SentientFMAmodeAttr, "SentientFMAmode::fused_mul_add">:$mode,
                    DefaultValuedAttr<SI32Attr, "-1">:$opAPortID,
                    DefaultValuedAttr<SI32Attr, "-1">:$opBPortID,
                    DefaultValuedAttr<SI32Attr, "-1">:$opCPortID,
                    DefaultValuedAttr<SentientUnrollFactorAttr, "SentientUnrollFactor::x1">:$unrollFactor,
                    DefaultValuedAttr<BoolAttr, "false">:$unrollIncrOpA,
                    DefaultValuedAttr<BoolAttr, "false">:$unrollIncrOpB,
                    DefaultValuedAttr<BoolAttr, "false">:$unrollIncrOpC,
                    DefaultValuedAttr<BoolAttr, "false">:$unrollIncrResult,
                    DefaultValuedAttr<I32Attr, "0">:$xrfReadIncr,
                    DefaultValuedAttr<I32Attr, "0">:$xrfWriteIncr);

  let results = (outs Variadic<Index>:$results);
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    std::optional<StringRef> getOutputRegister();
    bool isXrfRdRelated();
    bool isXrfWtRelated();
    bool isXrfRelated();
    unsigned getUnrollFactorVal();

    // Custom accessors that return signed values for XRF increments
    int32_t getXrfReadIncrSigned() { return static_cast<int32_t>(getXrfReadIncr()); }
    int32_t getXrfWriteIncrSigned() { return static_cast<int32_t>(getXrfWriteIncr()); }

    // Returns the mutable address value type.
    IndexType getMaskType() { return mlir::cast<IndexType>(getMask().getType()); }
  }];
}

//Vector Unary
def Sentient_UnaryOp:Sentient_Op<"vector_unary",
  [MemoryEffects<[MemWrite]>, Dataflow_DebugNameOpInterface]> {
    let summary = "Unary operations (estimate, ...)";
    let description = [{
    "sentient.unary perform on one vector slice input data and returns
     vector slice output. The attributes are semillar to vector_binary.
      Currently it is used for lowering aux functions extimate value."
    }];
    let arguments = (ins Index:$mask,
                     OptionalAttr<StrAttr>:$dbgName,
                     SentientComputePortAttr:$opA,
                     SentientUnaryAttr:$unary_op,
                     SentientComputePortArrayAttr:$opAForwarding,
                     SentientComputePortArrayAttr:$ResultForwarding,
                     OptionalAttr<SentientFoldModeAttr>:$fold_mode,
                     DefaultValuedAttr<SentientPrecisionAttr, "SentientPrecision::fp16">:$opAPrecision,
                     DefaultValuedAttr<SentientPrecisionAttr, "SentientPrecision::fp16">:$ResultPrecision,
                     DefaultValuedAttr<SentientPrecisionAttr, "SentientPrecision::fp16">:$ComputePrecision,
                     DefaultValuedAttr<SI32Attr, "-1">:$opADataID,
                     DefaultValuedAttr<SI32Attr, "-1">:$opAPortID,
                     DefaultValuedAttr<SentientUnrollFactorAttr, "SentientUnrollFactor::x1">:$unrollFactor,
                     DefaultValuedAttr<BoolAttr, "false">:$unrollIncrOpA,
                     DefaultValuedAttr<BoolAttr, "false">:$unrollIncrResult);
  let hasCustomAssemblyFormat = 1;
  let extraClassDeclaration = [{
    // Returns the mutable address value type.
    IndexType getMaskType() { return mlir::cast<IndexType>(getMask().getType()); }
    unsigned getUnrollFactorVal();
  }];
}

//Vector Binary
def Sentient_BinaryOp:Sentient_Op<"vector_binary",
  [MemoryEffects<[MemWrite]>, Dataflow_DebugNameOpInterface]> {
    let summary = "Binary operations (and, or, min, max, ...)";
    let arguments = (ins Index:$mask,
                     OptionalAttr<StrAttr>:$dbgName,
                     SentientComputePortAttr:$opA,
                     SentientComputePortAttr:$opB,
                     SentientBinaryOperatorAttr:$binaryOp,
                     SentientComputePortArrayAttr:$opAForwarding,
                     SentientComputePortArrayAttr:$opBForwarding,
                     SentientComputePortArrayAttr:$ResultForwarding,
                     // Forwarding to internal state registers. Only applicable to
                     // FCMP/FMINMAX binary operations in PE/SFP units.
                     OptionalAttr<SentientComputePortAttr>:$LogicalResultForwarding,
                     OptionalAttr<SentientFoldModeAttr>:$fold_mode,
                     DefaultValuedAttr<SentientPrecisionAttr, "SentientPrecision::fp16">:$opAPrecision,
                     DefaultValuedAttr<SentientPrecisionAttr, "SentientPrecision::fp16">:$opBPrecision,
                     DefaultValuedAttr<SentientPrecisionAttr, "SentientPrecision::fp16">:$ResultPrecision,
                     DefaultValuedAttr<SentientPrecisionAttr, "SentientPrecision::fp16">:$ComputePrecision,
                     DefaultValuedAttr<SI32Attr, "-1">:$opADataID,
                     DefaultValuedAttr<SI32Attr, "-1">:$opBDataID,
                     DefaultValuedAttr<SI32Attr, "-1">:$opAPortID,
                     DefaultValuedAttr<SI32Attr, "-1">:$opBPortID,
                     DefaultValuedAttr<SentientUnrollFactorAttr, "SentientUnrollFactor::x1">:$unrollFactor,
                     DefaultValuedAttr<BoolAttr, "false">:$unrollIncrOpA,
                     DefaultValuedAttr<BoolAttr, "false">:$unrollIncrOpB,
                     DefaultValuedAttr<BoolAttr, "false">:$unrollIncrResult,
                     DefaultValuedAttr<BoolAttr, "false">:$unrollIncrLogicalResult);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    // Returns the mutable address value type.
    IndexType getMaskType() { return mlir::cast<IndexType>(getMask().getType()); }
    unsigned getUnrollFactorVal();
  }];
}

//Vector ternary
//Even though MAC is a ternary operation, we exclude it beacuse it has special treatment and considerations
//such as XRF read/write pointers.
def Sentient_TernaryOp:Sentient_Op<"vector_ternary",
  [MemoryEffects<[MemWrite]>, Dataflow_DebugNameOpInterface]> {
    let summary = "Ternary operations (conditional select, ...)";
    let arguments = (ins Optional<Index>:$mask,
                     OptionalAttr<StrAttr>:$dbgName,
                     SentientComputePortAttr:$opA,
                     SentientComputePortAttr:$opB,
                     SentientComputePortAttr:$opC,
                     SentientTernaryOperatorAttr:$ternaryOp,
                     SentientComputePortArrayAttr:$opAForwarding,
                     SentientComputePortArrayAttr:$opBForwarding,
                     SentientComputePortArrayAttr:$opCForwarding,
                     SentientComputePortArrayAttr:$ResultForwarding,
                     OptionalAttr<SentientFoldModeAttr>:$fold_mode,
                     DefaultValuedAttr<SentientPrecisionAttr, "SentientPrecision::fp16">:$opAPrecision,
                     DefaultValuedAttr<SentientPrecisionAttr, "SentientPrecision::fp16">:$opBPrecision,
                     DefaultValuedAttr<SentientPrecisionAttr, "SentientPrecision::fp16">:$opCPrecision,
                     DefaultValuedAttr<SentientPrecisionAttr, "SentientPrecision::fp16">:$ResultPrecision,
                     DefaultValuedAttr<SentientPrecisionAttr, "SentientPrecision::fp16">:$ComputePrecision,
                     DefaultValuedAttr<SI32Attr, "-1">:$opADataID,
                     DefaultValuedAttr<SI32Attr, "-1">:$opBDataID,
                     DefaultValuedAttr<SI32Attr, "-1">:$opCDataID,
                     DefaultValuedAttr<SI32Attr, "-1">:$opAPortID,
                     DefaultValuedAttr<SI32Attr, "-1">:$opBPortID,
                     DefaultValuedAttr<SI32Attr, "-1">:$opCPortID,
                     DefaultValuedAttr<SentientUnrollFactorAttr, "SentientUnrollFactor::x1">:$unrollFactor,
                     DefaultValuedAttr<BoolAttr, "false">:$unrollIncrOpA,
                     DefaultValuedAttr<BoolAttr, "false">:$unrollIncrOpB,
                     DefaultValuedAttr<BoolAttr, "false">:$unrollIncrOpC,
                     DefaultValuedAttr<BoolAttr, "false">:$unrollIncrResult,
                     DefaultValuedAttr<BoolAttr, "false">:$unrollIncrLogicalResult);
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    // Returns the mutable address value type.
    IndexType getMaskType() { return mlir::cast<IndexType>(getMask().getType()); }
    unsigned getUnrollFactorVal();
  }];
}


//All loads and stores in sentient are vectors.
def Sentient_LoadOp:Sentient_Op<"load",
  []> {
    let summary = "Load operation";
    let arguments = (ins Arg<Index,"the load source",[MemRead]>:$src_val,
                     I32Attr:$total_elements,
                     I32Attr:$element_size,
                     I32Attr:$chunk_size,
                     I32Attr:$chunk_stride,
                     DefaultValuedAttr<SentientShuffleModeAttr, "SentientShuffleMode::noshuffle">:$shuffle_mode);

    let assemblyFormat = [{
        $src_val attr-dict
    }];
}

def Sentient_LoadComputeAndSendOp:Sentient_Op<"load_compute_and_send",
  [MemoryEffects<[MemWrite]>, Dataflow_DebugNameOpInterface]> {
    let summary = "Load_compute_and_send operation";
    let description = [{
      "The load compute and send operation loads data from the address calculated
       by mutable address + immutable address. This address is provided in terms
       of the result type of the operation and the shuffle mode determines the load
       type to use when loading the data.
       The src_element_size and src_total_elements attributes describe how to
       interpret the input data for computation. The element_index operand
       determines which interpreted elements of the loaded data are used for the
       computation.
       The operation chooses a scale from the scale register using the
       scale_index operand to identify which scale index to use. The operation
       manipulates the selected elements with the value in the scale register
       (for example, multiplies the selected elements with the value in the scale
       register). The resulting data will be in the shape described by
       dst_element_size and dst_total_elements. The data is then sent to the unit
       indicated by the consumer operand.
       The increment operand indicates how much the mutable address should be
       incremented after the operation executes (also in terms of dst_element_size
       and dst_total_elements).

       For example, the following is an example of the operation where the loaded data
       consists of 64 16 bit elements. From each of these loaded 16 bit elements, which
       are viewed as four 4 bit elements, the second 4 bit element is selected (element
       index 1) and then multiplied by the scale register at index 1. The load type
       indicates the input data is loaded as a 2 byte splat.
      "
      ```mlir
      %res = sentient.load_compute_and_send mutable_addr(%0), immutable_addr(%1),
                increment(%2), element_index(%3), scale_index(%4), consumer(%4) {
                    dst_element_size = 16 : i32, dst_total_elements = 64 : i32,
                    regIndices = [-1 : i32],
                    regLocales = [#sentient<reg_type unknown>],
                    shuffle_mode = #sentient<shuffle_mode splat2b>,
                    src_element_size = 4 : i32,
                    src_total_elements = 256 : i32 }
                : index, index, index, index, index, index : index
      ```
    }];
    let arguments = (ins Arg<Index,"mutable address",[MemRead]>:$mutable_addr,
                     Arg<Index,"immutable address",[MemRead]>:$immutable_addr,
                     Index:$increment,
                     Index:$element_index,
                     Index:$scale_index,
                     Arg<Index,"consumer",[MemWrite]>:$consumer,
                     OptionalAttr<StrAttr>:$dbgName,
                     I32Attr:$src_total_elements,
                     I32Attr:$dst_total_elements,
                     I32Attr:$src_element_size,
                     I32Attr:$dst_element_size,
                     OptionalAttr<SentientRoutingDirectionAttr>:$dir,
                     DefaultValuedAttr<SentientShuffleModeAttr, "SentientShuffleMode::noshuffle">:$shuffle_mode,
                     DefaultValuedAttr<SentientRegTypeAttr,"SentientRegType::unknown">:$regLocale,
                     DefaultValuedAttr<I32Attr, "-1">:$regIndex);

    let results = (outs Index:$result);

    let hasCanonicalizer = 1;
    let hasCustomAssemblyFormat = 1;
    let extraClassDeclaration = [{
      // Canonicalizer pattern for setting increment to zero when possible.
      struct setIncrementToZeroIfPossible;

      // Returns the mutable address value type.
      IndexType getMutableAddrType() { return mlir::cast<IndexType>(getMutableAddr().getType()); }

      // Returns the immutable address value type.
      IndexType getImmutableAddrType() { return mlir::cast<IndexType>(getImmutableAddr().getType()); }

      // Returns the element_index value type.
      IndexType getElementIndexType() { return mlir::cast<IndexType>(getElementIndex().getType()); }

      // Returns the scale_index value type.
      IndexType getScaleIndexType() { return mlir::cast<IndexType>(getScaleIndex().getType()); }

      // Returns the increment value type.
      IndexType getIncrementType() { return mlir::cast<IndexType>(getIncrement().getType()); }

      // Returns the consumer value type.
      IndexType getConsumerType() { return mlir::cast<IndexType>(getConsumer().getType()); }
    }];
}

def Sentient_LoadAndSendOp:Sentient_Op<"load_and_send",
  [MemoryEffects<[MemWrite]>, Dataflow_DebugNameOpInterface]> {
    let summary = "Load_and_send operation";
    let arguments = (ins Arg<Index,"the mutable address",[MemRead]>:$mutable_addr,
                     Arg<Index,"the immutable address",[MemRead]>:$immutable_addr,
                     Index:$increment,
                     Arg<Index,"the consumer",[MemWrite]>:$consumer,
                     OptionalAttr<StrAttr>:$dbgName,
                     I32Attr:$total_elements,
                     I32Attr:$element_size,
                     OptionalAttr<DefaultValuedAttr<I32Attr, "0">>:$rotate_val,
                     OptionalAttr<SentientRoutingDirectionAttr>:$dir,
                     DefaultValuedAttr<I32Attr, "1">:$chunk_size,
                     DefaultValuedAttr<I32Attr, "1">:$chunk_stride,
                     DefaultValuedAttr<I32Attr, "0">:$burst_size,
                     DefaultValuedAttr<I32Attr, "0">:$interleaved_group,
                     DefaultValuedAttr<SentientShuffleModeAttr, "SentientShuffleMode::noshuffle">:$shuffle_mode,
                     DefaultValuedAttr<SentientRegTypeAttr,"SentientRegType::unknown">:$regLocale,
                     DefaultValuedAttr<I32Attr, "-1">:$regIndex);

    let results = (outs Index:$result);

    let hasCanonicalizer = 1;
    let hasCustomAssemblyFormat = 1;
    let extraClassDeclaration = [{
      // Canonicalizer pattern for setting increment to zero when possible.
      struct setIncrementToZeroIfPossible;

      // Returns the mutable address value type.
      IndexType getMutableAddrType() { return mlir::cast<IndexType>(getMutableAddr().getType()); }

      // Returns the immutable address value type.
      IndexType getImmutableAddrType() { return mlir::cast<IndexType>(getImmutableAddr().getType()); }

      // Returns the increment value type.
      IndexType getIncrementType() { return mlir::cast<IndexType>(getIncrement().getType()); }

      // Returns the consumer value type.
      IndexType getConsumerType() { return mlir::cast<IndexType>(getConsumer().getType()); }
    }];
}

def Sentient_ReceiveAndStoreOp:Sentient_Op<"receive_and_store", [
        AttrSizedOperandSegments,
        MemoryEffects<[MemWrite]>, Dataflow_DebugNameOpInterface]> {
    let summary = "Receive_and_store operation";
    let arguments = (ins Arg<Index,"the mutable address",[MemWrite]>:$mutable_addr,
                     Arg<Index,"the immutable address",[MemWrite]>:$immutable_addr,
                     Index:$increment,
                     Arg<AnyTypeOf<[AnyInteger, AnyFloat, Index]>,"the producer",[MemRead]>:$producer,
                     OptionalAttr<StrAttr>:$dbgName,
                     Optional<Index>:$dst,
                     Optional<Index>:$drop_first,
                     Optional<Index>:$multicast_info,
                     I32Attr:$total_elements,
                     I32Attr:$element_size,
                     DefaultValuedAttr<I32Attr, "0">:$burst_size,
                     DefaultValuedAttr<I32Attr, "0">:$interleaved_group,
                     DefaultValuedAttr<BoolAttr, "false">:$coalesce,
                     DefaultValuedAttr<I32Attr, "1">:$subword_length,
                     DefaultValuedAttr<I32Attr, "1">:$stride,
                     DefaultValuedAttr<BoolAttr, "false">:$permute,
                     OptionalAttr<DefaultValuedAttr<SentientShuffleModeAttr, "SentientShuffleMode::noshuffle">>:$shuffle_mode,
                     DefaultValuedAttr<SentientRegTypeAttr,"SentientRegType::unknown">:$regLocale,
                     DefaultValuedAttr<I32Attr, "-1">:$regIndex);

    let results = (outs Index:$result);

    let builders = [
      OpBuilder<(ins "::mlir::IndexType":$result, "::mlir::Value":$mutable_addr, "::mlir::Value":$immutable_addr, "::mlir::Value":$increment,
      "::mlir::OpResult":$producer, "::mlir::StringAttr":$dbgName, "::mlir::Value":$dst, "::mlir::Value":$drop_first, "::mlir::Value":$multicast_info, "int":$total_elements, "int": $element_size,
      "int":$burst_size, "int":$interleaved_group, "bool":$coalesce, "int":$subword_length, "int":$stride),
      [{build($_builder, $_state, result, mutable_addr, immutable_addr, increment, producer, dbgName, dst, drop_first, multicast_info, total_elements, element_size,
      burst_size, interleaved_group, coalesce, subword_length, stride, false, nullptr, SentientRegType::unknown, -1);}]>
    ];

    let hasCanonicalizer = 1;
    let hasCustomAssemblyFormat = 1;
    let extraClassDeclaration = [{
      // Canonicalizer pattern for setting increment to zero when possible.
      struct setIncrementToZeroIfPossible;

      // Returns the mutable_addr value type.
      IndexType getMutableAddrType() { return mlir::cast<IndexType>(getMutableAddr().getType()); }

      // Returns the immutable value type.
      IndexType getImmutableAddrType() { return mlir::cast<IndexType>(getImmutableAddr().getType()); }

      // Returns the increment value type.
      IndexType getIncrementType() { return mlir::cast<IndexType>(getIncrement().getType()); }

      // Returns the producer value type.
      Type getProducerType() { return getProducer().getType(); }

      // Returns the drop_first value type.
      IndexType getDropFirstType() { return mlir::cast<IndexType>(getDropFirst().getType()); }

      // Returns the multicast_info value type.
      IndexType getMulticastInfoType() { return mlir::cast<IndexType>(getMulticastInfo().getType()); }
    }];
}

def Sentient_LoadAndExtractScalarOp:Sentient_Op<"load_and_extract_scalar",
  [MemoryEffects<[MemWrite]>, Dataflow_DebugNameOpInterface]> {
    let summary = "Load_and_extract_scalar operation";
    let arguments = (ins Arg<Index,"the mutable address",[MemRead]>:$mutable_addr,
                     Arg<Index,"the immutable address",[MemRead]>:$immutable_addr,
                     Index:$increment,
                     Arg<Index,"the consumer",[MemWrite]>:$consumer,
                     OptionalAttr<StrAttr>:$dbgName,
                     I32Attr:$total_elements,
                     I32Attr:$element_size,
                     SentientRegTypeArrayAttr:$regLocales,
                     I32ArrayAttr:$regIndices);

    // addr returns the updated address according to the operation (mutable_addr + increment)
    // data returns the data contained at the memory location represented by this
    // operation (data stored at mutable_addr + immutable_addr)
    let results = (outs Index:$addr, Index:$data);

    let hasCanonicalizer = 1;
    let hasCustomAssemblyFormat = 1;
    let builders = [
      OpBuilder<(ins "::mlir::IndexType":$addr, "::mlir::IndexType":$data,
      "::mlir::Value":$mutable_addr, "::mlir::Value":$immutable_addr, "::mlir::Value":$increment,
      "::mlir::Value":$consumer, "::mlir::StringAttr":$dbgName, "int":$total_elements, "int": $element_size),
      [{build($_builder, $_state, addr, data, mutable_addr, immutable_addr, increment,
      consumer, dbgName, total_elements, element_size,
      $_builder.getArrayAttr({SentientRegTypeAttr::get($_builder.getContext(), SentientRegType::unknown),
                              SentientRegTypeAttr::get($_builder.getContext(), SentientRegType::unknown)}),
                              $_builder.getI32ArrayAttr({-1,-1}));}]>
    ];

    let extraClassDeclaration = [{
      // Canonicalizer pattern for setting increment to zero when possible.
      struct setIncrementToZeroIfPossible;

      // Returns the mutable address value type.
      IndexType getMutableAddrType() { return mlir::cast<IndexType>(getMutableAddr().getType()); }

      // Returns the immutable address value type.
      IndexType getImmutableAddrType() { return mlir::cast<IndexType>(getImmutableAddr().getType()); }

      // Returns the increment value type.
      IndexType getIncrementType() { return mlir::cast<IndexType>(getIncrement().getType()); }

      // Returns the consumer value type.
      IndexType getConsumerType() { return mlir::cast<IndexType>(getConsumer().getType()); }

      // Returns the address result.
      Value getAddrResult() { return getResult(0); }

      // Returns the data result.
      Value getDataResult() { return getResult(1); }

      // Returns result index for address result.
      int getAddrResultIdx() { return 0; }

      // Returns result index for data result.
      int getDataResultIdx() { return 1; }

      // Helper functions to standardize load_and_extract_scalar op with other memory ops.
      // Some functions are templated to operate in a generic fashion on any memory op.
      // load_and_extract_scalar ops don't contain all the same attributes as the other
      // memory ops. To allow those templated functions to work nicely, the following
      // functions simulate the attributes needed.
      int getBurstSize() { return 0; }
      int getInterleavedGroup() { return 0; }
    }];
}

def Sentient_ReceiveAndExtractScalarOp:Sentient_Op<"receive_and_extract_scalar",
  [MemoryEffects<[MemWrite]>, Dataflow_DebugNameOpInterface]> {
    let summary = "Receive_and_extract_scalar operation";
    let arguments = (ins Arg<Index,"the unit to receive data from",[MemRead]>:$unit,
                     Arg<Index,"the position of the object to extract",[MemRead]>:$position,
                     OptionalAttr<DefaultValuedAttr<StrAttr, "">>:$dbgName,
                     DefaultValuedAttr<SentientRegTypeAttr,"SentientRegType::unknown">:$regLocale,
                     DefaultValuedAttr<I32Attr, "-1">:$regIndex);

    let results = (outs Index:$result);

    let hasCustomAssemblyFormat = 1;
    let extraClassDeclaration = [{
      // Canonicalizer pattern for setting increment to zero when possible.
      struct setIncrementToZeroIfPossible;

      // Returns the unit value type.
      IndexType getUnitType() { return mlir::cast<IndexType>(getUnit().getType()); }

      // Returns the position value type.
      IndexType getPositionType() { return mlir::cast<IndexType>(getPosition().getType()); }
    }];
}

//Scalar Add : TODO -- Group all of the scalar binary ops into a single op
def Sentient_AddOp:Sentient_Op<"scalar_add",
  [SameOperandsAndResultType, Commutative, NoMemoryEffect, ScalarValue]> {
    let summary = "Adding two 1D variables";
    let arguments = (ins AnyTypeOf<[AnyInteger, Index]>:$inp1,
                            AnyTypeOf<[AnyInteger, Index]>:$inp2,
                            DefaultValuedAttr<SentientRegTypeAttr,"SentientRegType::unknown">:$regLocale,
                            DefaultValuedAttr<I32Attr, "-1">:$regIndex);
    let results = (outs AnyTypeOf<[AnyInteger, Index]>:$out);

    let assemblyFormat = [{
        $inp1 `,` $inp2  attr-dict `:` type($inp1) `,` type($inp2)
    }];
    let hasFolder = 1;
}

def Sentient_LoadAndStoreOp:Sentient_Op<"load_and_store",
  [MemoryEffects<[MemWrite]>, Dataflow_DebugNameOpInterface]> {
    let summary = "Load_and_store operation";
    let description = [{
      "sentient.load_and_store operation loads data from src and stores it in dst.
       The operation returns the final, potentially updated, addresses of src and dst."
    }];
    let arguments = (ins Index:$src,
                     Index:$dst,
                     Arg<Index,"the source mutable address",[MemRead]>:$src_mutable_addr,
                     Arg<Index,"the source immutable address",[MemRead]>:$src_immutable_addr,
                     Index:$src_inc,
                     Arg<Index,"the destination mutable address",[MemWrite]>:$dst_mutable_addr,
                     Arg<Index,"the destination immutable address",[MemWrite]>:$dst_immutable_addr,
                     Index:$dst_inc,
                     Optional<Index>:$multicast_info,
                     OptionalAttr<StrAttr>:$dbgName,
                     I32Attr:$total_elements,
                     I32Attr:$element_size,
                     OptionalAttr<DefaultValuedAttr<I32Attr, "0">>:$rotate_val,
                     DefaultValuedAttr<I32Attr, "1">:$chunk_size,
                     DefaultValuedAttr<I32Attr, "1">:$chunk_stride,
                     DefaultValuedAttr<I32Attr, "0">:$burst_size,
                     DefaultValuedAttr<SentientShuffleModeAttr, "SentientShuffleMode::noshuffle">:$shuffle_mode,
                     DefaultValuedAttr<I32Attr, "1">:$stride,
                     SentientRegTypeArrayAttr:$regLocales,
                     I32ArrayAttr:$regIndices,
                     OptionalAttr<SentientRoutingDirectionAttr>:$dir);

    // src_res returns the final, potentially updated, address of src
    // dst_res returns the final, potentially updated, address of dst
    let results = (outs Index:$src_res, Index:$dst_res);

    let hasCanonicalizer = 1;
    let hasCustomAssemblyFormat = 1;

    let builders = [
      OpBuilder<(ins "::mlir::IndexType":$src_res, "::mlir::IndexType":$dst_res,
      "::mlir::Value":$src, "::mlir::Value":$dst, "::mlir::Value":$src_mutable_addr, "::mlir::Value":$src_immutable_addr,
      "::mlir::Value":$src_inc, "::mlir::Value":$dst_mutable_addr, "::mlir::Value":$dst_immutable_addr, "::mlir::Value":$dst_inc,
      "::mlir::Value":$multicast_info, "::mlir::StringAttr":$dbgName, "int":$total_elements, "int": $element_size, "::mlir::IntegerAttr": $rotate_val, "int":$chunk_size,
      "int":$chunk_stride, "int":$burst_size, "SentientShuffleMode":$shuffle_mode, "int":$stride, "SentientRoutingDirectionAttr":$dir),
      [{build($_builder, $_state, src_res, dst_res, src, dst, src_mutable_addr, src_immutable_addr, src_inc,
      dst_mutable_addr, dst_immutable_addr, dst_inc, multicast_info, dbgName, total_elements, element_size, rotate_val, chunk_size,
      chunk_stride, burst_size, shuffle_mode, stride,
      $_builder.getArrayAttr({SentientRegTypeAttr::get($_builder.getContext(), SentientRegType::unknown),
                              SentientRegTypeAttr::get($_builder.getContext(), SentientRegType::unknown)}),
      $_builder.getI32ArrayAttr({-1,-1}), dir);}]>
    ];

    let extraClassDeclaration = [{
      // Canonicalizer pattern for setting increments to zero when possible.
      struct setIncrementToZeroIfPossible;

      // Returns the src value type.
      IndexType getSrcType() { return mlir::cast<IndexType>(getSrc().getType()); }

      // Returns the dst value type.
      IndexType getDstType() { return mlir::cast<IndexType>(getDst().getType()); }

      // Returns the src_mutable_addr value type.
      IndexType getSrcMutableAddrType() { return mlir::cast<IndexType>(getSrcMutableAddr().getType()); }

      // Returns the src_immutable_addr value type.
      IndexType getSrcImmutableAddrType() { return mlir::cast<IndexType>(getSrcImmutableAddr().getType()); }

      // Returns the src_inc value type.
      IndexType getSrcIncType() { return mlir::cast<IndexType>(getSrcInc().getType()); }

      // Returns the dst_mutable_addr value type.
      IndexType getDstMutableAddrType() { return mlir::cast<IndexType>(getDstMutableAddr().getType()); }

      // Returns the dst_immutable_addr value type.
      IndexType getDstImmutableAddrType() { return mlir::cast<IndexType>(getDstImmutableAddr().getType()); }

      // Returns the dst_inc value type.
      IndexType getDstIncType() { return mlir::cast<IndexType>(getDstInc().getType()); }

      // Returns the multicast_info value type.
      IndexType getMulticastInfoType() { return mlir::cast<IndexType>(getMulticastInfo().getType()); }

      static StringRef getIsIBRWriteAttrStrName() { return "is-ibr-write"; }
    }];
}

//Scalar Sub
def Sentient_SubOp:Sentient_Op<"scalar_sub",
  [SameOperandsAndResultType, NoMemoryEffect, ScalarValue]> {
    let summary = "Subtracting two 1D variables";
    let arguments = (ins AnyTypeOf<[AnyInteger, Index]>:$inp1,
                            AnyTypeOf<[AnyInteger, Index]>:$inp2,
                            DefaultValuedAttr<SentientRegTypeAttr,"SentientRegType::unknown">:$regLocale,
                            DefaultValuedAttr<I32Attr, "-1">:$regIndex);
    let results = (outs AnyTypeOf<[AnyInteger, Index]>:$out);

    let assemblyFormat = [{
         $inp1 `,` $inp2  attr-dict `:` type($inp1) `,` type($inp2)
    }];
}

//Scalar Mul
def Sentient_MulOp:Sentient_Op<"scalar_mul",
  [SameOperandsAndResultType, Commutative, NoMemoryEffect, ScalarValue]> {
    let summary = "Multiplying two 1D variables";
    let arguments = (ins AnyTypeOf<[AnyInteger, Index]>:$inp1,
                            AnyTypeOf<[AnyInteger, Index]>:$inp2,
                            DefaultValuedAttr<SentientRegTypeAttr,"SentientRegType::unknown">:$regLocale);
    let results = (outs AnyTypeOf<[AnyInteger, Index]>:$out);

    let assemblyFormat = [{
         $inp1 `,` $inp2  attr-dict `:` type($inp1) `,` type($inp2)
    }];
}

//Scalar Copy
def Sentient_CopyOp:Sentient_Op<"scalar_copy",
  [SameOperandsAndResultType, NoMemoryEffect, ScalarValue]> {
    let summary = "Copying variables";
    let arguments = (ins AnyTypeOf<[AnyInteger, Index]>:$inp,
                            DefaultValuedAttr<SentientRegTypeAttr,"SentientRegType::unknown">:$regLocale,
                            DefaultValuedAttr<I32Attr, "-1">:$regIndex,
                            DefaultValuedAttr<BoolAttr,"false">:$programHeader);
    let results = (outs AnyTypeOf<[AnyInteger, Index]>:$out);

    let assemblyFormat = [{
         $inp  attr-dict `:` type($inp)
    }];

    let extraClassDeclaration = [{
      static StringRef getProgramHeaderAttrStrName() { return "programHeader"; }
    }];
}

def Sentient_ConstantOp:Sentient_Op<"scalar_constant",
  [ConstantLike, NoMemoryEffect, ScalarValue]> {
    let summary = "Constant operation";
    let arguments = (ins SI64Attr:$value,
                         DefaultValuedAttr<SentientRegTypeAttr,"SentientRegType::imm">:$regLocale);
    let results = (outs AnyTypeOf<[AnyInteger, AnyFloat, Index]>:$out);

    let hasCustomAssemblyFormat = 1;

    let extraClassDeclaration = [{
        /// Returns true if a constant operation can be built with the given value
        /// and result type.
        static bool isBuildableWith(Attribute value, Type type);
    }];

    let hasFolder = 1;
}

def Sentient_VectorConstantOp:Sentient_Op<"vector_constant",
  [ConstantLike, NoMemoryEffect]> {
    let summary = "Constant operation";
    let arguments = (ins ArrayAttr:$value);
    let results = (outs AnyVectorOfAnyRank:$out);

    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
}

def Sentient_SyncOp:Sentient_Op<"sync",
  [Dataflow_DebugNameOpInterface]> {
    let summary = "generalized Sync operation";
    let arguments = (ins OptionalAttr<StrAttr>:$dbgName,
                         SentientSyncModeAttr:$mode,
                         SentientConsumerArrayAttr:$units,
                         OptionalAttr<DefaultValuedAttr<BoolAttr, "false"> >:$soft,
                         OptionalAttr<DefaultValuedAttr<SI32Attr, "-1">>:$implicit_sync_memory_boundary);
    let assemblyFormat = [{
         attr-dict
    }];

    let extraClassDeclaration = [{
        static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::StringAttr dbgName, ::mlir::sentient::SentientSyncModeAttr mode, ::mlir::ArrayAttr units, /*optional*/::mlir::BoolAttr soft) {
            build(odsBuilder, odsState, dbgName, mode, units, soft, odsBuilder.getSI32IntegerAttr(-1));
        }
    }];
}

def Sentient_NOPOp:Sentient_Op<"nop",
  [Dataflow_DebugNameOpInterface]> {
    let summary = "NOP operation";
    let arguments = (ins OptionalAttr<StrAttr>:$dbgName);
    let assemblyFormat = [{
        attr-dict
    }];
}

def Sentient_SetSendDestinationOp:Sentient_Op<"set_send_dst",
  []> {
    let summary = "Operation to set the destination for data being sent from one unit to the other.";
    let description = [{
      "This operation globally sets the destination of following data transfer
       operations to one or more units. In cases were more than one unit is being
       specified as the destination, the input to this operation could be the
       result of a dataflow.create_group."
    }];

    let arguments = (ins Index:$units);
    let assemblyFormat = [{
        `(` $units `)` attr-dict
    }];
}

def Sentient_LogicalPortOp:Sentient_Op<"logical_port",
  [NoMemoryEffect]> {
    let summary = "Getting a handle to a sentient port";
    let description = [{
      "This operation gets an ssa variable that specifies a sentient port."
    }];
    let arguments = (ins SentientComputePortAttr:$portName);
    let results = (outs Index:$port);
    let assemblyFormat = [{
         attr-dict `:` type($port)
    }];
}

def Sentient_SplatOp:Sentient_Op<"splat",
  [MemoryEffects<[MemWrite]>, Dataflow_DebugNameOpInterface]> {
    let summary = "Creates a vector by splatting the input and stores/forwards the result.";
    let description = [{
      "This operation constructs a vector by splatting (and optionally padding)
       the input. This operation conceptually creates a subvector wide enough for
       a vector slice (128-bits) and then splats that subvector across all the
       vector slices. The resulting full vector is then stored to a register file
       or forwarded to another unit. If input is a scalar constant, the initial
       vector slice is created either by splatting that constant across all lanes
       of the slice (when pad=false) or by inserting the input into one element
       of the vector slice and padding the remaining elements with zeros.
       If input is a vector coming from a port, `n` right-most bits of the input
       are used to populate the subvector, where `n` is based on the precision
       attribute."
    }];
    let arguments = (ins Arg<AnyTypeOf<[Index, AnyInteger, AnyFloat, AnyVectorOfAnyRank]>,"input",[MemWrite]>:$input,
                     Arg<Index,"output",[MemWrite]>:$output, Index:$mask,
                     OptionalAttr<StrAttr>:$dbgName,
                     SentientSplatPadAttr:$pad,
                     DefaultValuedAttr<SentientPrecisionAttr, "SentientPrecision::fp16">:$precision,
                     DefaultValuedAttr<BoolAttr,"false">:$programHeader,
                     DefaultValuedAttr<SentientUnrollFactorAttr, "SentientUnrollFactor::x1">:$unrollFactor,
                     DefaultValuedAttr<BoolAttr, "false">:$unrollIncrResult
    );

  let hasCustomAssemblyFormat = 1;

}

def Sentient_OpaqueOp:Sentient_Op<"opaque",
  [Dataflow_DebugNameOpInterface]> {
    let summary = "Opaque operation";
    let description = [{
      "Opaque operation for auxiliary functions."
      ```mlir
      sentient.opaque {func_name = "reciprocal",
                       read_write_register_dictionary = {p0="R0",p1="R1"},
                       read_only_register_dictionary = {c0="R2"},
                       parameter_dictionary = {unroll_factor="2"}}
      ```
    }];
    let arguments = (    ins OptionalAttr<StrAttr>:$dbgName,
                         StrAttr:$func_name,
                         Arg<DictionaryAttr,"read write reg dictionary",[MemWrite]>:$read_write_register_dictionary,
                         Arg<DictionaryAttr,"read only reg dictionary",[MemWrite]>:$read_only_register_dictionary,
                         DictionaryAttr:$parameter_dictionary);

    let hasCustomAssemblyFormat = 1;
}

def Sentient_SetActiveMaskValueOp:Sentient_Op<"samv", [
        MemoryEffects<[MemWrite]>,
        Dataflow_DebugNameOpInterface]> {
    let summary = "SAMV operation";
    let description = [{
      "Set Active Mask and Value (SAMV) operation"
      ```mlir
      %mask = sentient.samv { maskall = "no", numvalidentry = 5 : i32,
                              sliceid-xsl = 4 : i32, xslinner = 0 : i32,
                              wsllen = 2 : i32,
                              precision = "16b",
                              regIndices = [0 : i32],
                              regLocales = [#sentient<reg_type mvr>]} : index, index : index
      ```

      Mask all is represented by the following IR:
      ```mlir
      %mask = sentient.samv { maskall = "yes", numvalidentry = 0 : i32,
                              sliceid-xsl = 0 : i32, xslinner = 0 : i32,
                              wsllen = 1 : i32,
                              precision = "16b",
                              regIndices = [0 : i32],
                              regLocales = [#sentient<reg_type mvr>]} : index, index : index
      ```

      Unmask all is represented by the following IR:
      ```mlir
      %mask = sentient.samv { maskall = "no", numvalidentry = 0 : i32,
                              sliceid-xsl = 7 : i32, xslinner = 0 : i32,
                              wsllen = 1 : i32,
                              precision = "16b",
                              regIndices = [0 : i32],
                              regLocales = [#sentient<reg_type mvr>]} : index, index : index
      ```
    }];
    let arguments = (ins AnyTypeOf<[Index, AnyInteger, AnyFloat]>:$mask_value,
                         BoolAttr:$maskall,
                         I32Attr:$numvalidentry,
                         I32Attr:$sliceid_xsl,
                         BoolAttr:$xslinner,
                         I32Attr:$wsllen,
                         I32Attr:$precision,
                         OptionalAttr<StrAttr>:$dbgName);

    let hasVerifier = 1;
    let hasCustomAssemblyFormat = 1;

    let extraClassDeclaration = [{
      IndexType getMaskValueType() { return mlir::cast<IndexType>(getMaskValue().getType()); }
      static StringRef getMaskallAttrStrName() { return "maskall"; }
      static StringRef getXslinnerAttrStrName() { return "xslinner"; }
    }];
}

def Sentient_SetMaskOp:Sentient_Op<"set_mask", [
        MemoryEffects<[MemWrite]>,
        Dataflow_DebugNameOpInterface]> {
    let summary = "set mask operation";
    let description = [{
      "Set mask operation for PT masking."
      ```mlir
      sentient.set_mask %c0
      ```
    }];
    let arguments = (ins AnyTypeOf<[AnyInteger, Index]>:$mask_value,
                         OptionalAttr<StrAttr>:$dbgName);

    let hasVerifier = 1;
    let hasCustomAssemblyFormat = 1;

    let extraClassDeclaration = [{
      Type getMaskValueType() { return getMaskValue().getType(); }
    }];
}

def Sentient_IncrMaskOp:Sentient_Op<"incrmask", [
        MemoryEffects<[MemWrite]>,
        Dataflow_DebugNameOpInterface]> {
    let summary = "increment mask operation";
    let description = [{
      "Increment mask operation for PT masking."
      ```mlir
      sentient.incrmask
      ```
    }];
    let arguments = (ins OptionalAttr<StrAttr>:$dbgName);
    let hasCustomAssemblyFormat = 1;

    let extraClassDeclaration = [{
      // Return how much this operation increments the mask.
      int getIncrement() { return 1; }
    }];
}

#endif // DCC_SRC_DIALECT_SENTIENT_OPS_TD
