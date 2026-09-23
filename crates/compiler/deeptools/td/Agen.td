//===-- Agen.td ---------------------------------------------*- tablegen -*-==//
//
// Part of the Dataflow Scheduler MLIR Dialects project.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// This file defines the agen dialect.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_DIALECT_AGEN_AGEN_TD
#define DATAFLOW_SCHEDULER_DIALECT_AGEN_AGEN_TD

include "mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "dataflow-scheduler/Dialect/Agen/AgenEnums.td"
include "dataflow-scheduler/Dialect/Dataflow/DataflowInterfaces.td"
include "dataflow-scheduler/Dialect/Dataflow/DataflowTypes.td"

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

def Agen_Dialect : Dialect {
  let name = "agen";
  let cppNamespace = "::mlir::agen";

  let useDefaultAttributePrinterParser = 1;

  let dependentDialects = [
    "arith::ArithDialect",
    "dataflow::DataflowDialect",
    "vectorchain::VectorChainDialect"
  ];
}

//===----------------------------------------------------------------------===//
// Attributes
//===----------------------------------------------------------------------===//

def AgenRoutingDirectionAttr : EnumAttr<Agen_Dialect, AgenRoutingDirection, "direction">;

//===----------------------------------------------------------------------===//
// Operations
//===----------------------------------------------------------------------===//

// Base class for Agen dialect operations. This operation inherits from the
// base `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class Agen_Op<string mnemonic, list<Trait> traits = []> :
    Op<Agen_Dialect, mnemonic, traits> {
}

// Require regions to have sentient.yield.
def ImplicitAgenTerminator
    : SingleBlockImplicitTerminator<"YieldOp">;

def YieldOp : Agen_Op<"yield", [NoMemoryEffect, Terminator, ReturnLike,
    MemRefsNormalizable]> {
  let summary = "Yield values to parent operation";
  let description = [{
    "agen.yield yields zero or more SSA values to parent op."
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [
    OpBuilder<(ins), [{ build($_builder, $_state, ValueRange{}); }]>
  ];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

def Agen_VectorLoadOp : Agen_Op<"vector_load", [
    DeclareOpInterfaceMethods<AffineReadOpInterface>,
    DeclareOpInterfaceMethods<AffineMapAccessInterface>,
    MemRefsNormalizable,
    AttrSizedOperandSegments,
    Dataflow_DebugNameOpInterface]> {
  let summary = "agen vector load operation";
  let description = [{
    Loads a vector of data from a memory view (typically produced by
    `dataflow.get_logical_memory_view`).

    The index subscripts on `$mem_ref` give the starting address of the load.
    `$load_set` is an affine set that dictates which elements (relative to the
    starting address) are gathered into the vector, and `$load_order` is an
    affine map that dictates how those elements are packed into the result
    vector. The load is contiguous when the set walks the innermost memory
    dimension; selecting an outer dimension instead yields a strided load.

    `$multicast_info`, when present, carries routing information for transfers
    that fan the loaded vector out to multiple consumers.

    Example:
    ```mlir
    %data = agen.vector_load %l1_view[0, 0]
      {load_order = affine_map<(d0, d1) -> (d1, d0)>,
       load_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 7 >= 0, d1 == 0)>}
      : memref<8x16xi8>, vector<8xi8>
    ```
  }];
  let arguments = (ins
    Arg<AnyMemRef, "the reference to load from", [MemRead]>:$mem_ref,
    OptionalAttr<StrAttr>:$dbgName,
    AffineMapAttr:$affine_map,
    Variadic<Index>:$map_operands,
    Builtin_IntegerSetAttr:$load_set,
    AffineMapAttr:$load_order,
    Optional<Index>:$multicast_info);

  let results = (outs AnyTypeOf<[AnyVectorOfAnyRank, Dataflow_CustomVectorType]>:$result);

  let builders = [
    /// Builds an agen vector load op with the specified map and operands.
    OpBuilder<(ins
        "Type":$resultType, "Value":$mem_ref, "StringAttr":$dbg_name,
        "AffineMap":$map, "ValueRange":$map_operands, "IntegerSet":$load_set,
        "AffineMap":$load_order, CArg<"Value", "{}">:$multicast_info), [{
      build($_builder, $_state, resultType, mem_ref, dbg_name, map, map_operands,
            IntegerSetAttr::get(load_set), load_order, multicast_info);
    }]>
  ];

  let hasVerifier = 1;
  let assemblyFormat = [{
    $mem_ref``custom<AffineMapOfSSAIds>($affine_map, $map_operands)
    (`multicast_info` `=` $multicast_info^)?
    attr-dict `:` type($mem_ref) `,` type($result)
  }];

  let extraClassDeclaration =  [{
    VectorType getVectorType() {
      return dyn_cast<VectorType>(getResult().getType());
    }

    /// Returns the affine map for the agen vector load.
    [[deprecated("use getLoadSetAttr()")]]
    IntegerSetAttr getAffineLoadSetAttr() {
      return getLoadSetAttr();
    }

    [[deprecated("use getMapOperands()")]]
    ::mlir::Operation::operand_range getMapIndices() {
      return getMapOperands();
    }

    [[deprecated("use getMapOperands()")]]
    ::mlir::Operation::operand_range indices() {
      return getMapOperands();
    }

    SmallVector<Operation *> getUseChain();
    void cloneUseChainToNewOp(OpBuilder &builder, Operation *new_op);
    VectorLoadOp cloneWithNewAccessInfo(OpBuilder &builder, Value mem_view,
        const AffineMap &subscripts_map, ValueRange indices);
    void eraseOpAndUseChain();

    //===------------------------------------------------------------------===//
    // AffineMapAccessInterface
    //===------------------------------------------------------------------===//

    [[nodiscard]] auto getMapAttrStrName() -> StringRef {
      return getAffineMapAttrName();
    }
  }];
}

def Agen_VectorStoreOp : Agen_Op<"vector_store", [
    DeclareOpInterfaceMethods<AffineWriteOpInterface>,
    DeclareOpInterfaceMethods<AffineMapAccessInterface>,
    MemRefsNormalizable,
    Dataflow_DebugNameOpInterface]> {
  let summary = "agen vector store operation";
  let description = [{
    Stores a vector of data into a memory view (typically produced by
    `dataflow.get_logical_memory_view`). This is the counterpart of
    `agen.vector_load` and carries the analogous fields.

    The index subscripts on `$mem_ref` give the starting address of the store.
    `$store_set` is an affine set that dictates which elements (relative to the
    starting address) are written, and `$store_order` is an affine map that
    dictates the order in which lanes of `$value_to_store` map onto those
    elements.

    Note that the starting address may change across iterations, as arises with
    double-buffering or streaming transfers.

    Example:
    ```mlir
    agen.vector_store %data, %l1_view[0, 0]
      {store_order = affine_map<(d0, d1) -> (d1, d0)>,
       store_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 7 >= 0, d1 == 0)>}
      : memref<8x16xi8>, vector<8xi8>
    ```
  }];
  let arguments = (ins
    AnyTypeOf<[AnyVectorOfAnyRank, Dataflow_CustomVectorType]>:$value_to_store,
    Arg<AnyMemRef, "the reference to store to", [MemWrite]>:$mem_ref,
    OptionalAttr<StrAttr>:$dbgName,
    AffineMapAttr:$affine_map,
    Variadic<Index>:$map_operands,
    Builtin_IntegerSetAttr:$store_set,
    AffineMapAttr:$store_order);

  let builders = [
    OpBuilder<(ins
        "Value":$value_to_store, "Value":$mem_ref, "StringAttr":$dbg_name, "AffineMap":$map,
        "ValueRange":$map_operands, "IntegerSet":$store_set, "AffineMap":$store_order), [{
      build($_builder, $_state, value_to_store, mem_ref, dbg_name, map, map_operands,
            IntegerSetAttr::get(store_set), store_order);
    }]>,
    OpBuilder<(ins
        "Value":$value_to_store, "Value":$mem_ref, "mlir::StringAttr":$dbg_name,
        "ValueRange":$indices, "IntegerSet":$store_set, "mlir::AffineMap":$store_order)>
  ];

  let hasVerifier = 1;
  let assemblyFormat = [{
    $value_to_store `,` $mem_ref``custom<AffineMapOfSSAIds>($affine_map, $map_operands)
    attr-dict `:` type($mem_ref) `,` type($value_to_store)
  }];

  let extraClassDeclaration =  [{
    VectorType getVectorType() {
      return dyn_cast<VectorType>(getValueToStore().getType());
    }

    /// Returns the affine map for then agen vector store.
    [[deprecated("use getStoreSetAttr()")]]
    IntegerSetAttr getAffineStoreSetAttr() {
        return getStoreSetAttr();
    }

    [[deprecated("use getMapOperands().size()")]]
    int getNumIndices() {
      return getMapOperands().size();
    }

    [[deprecated("use getMapOperands()")]]
    // Match other op functions to allow better generalization.
    ::mlir::Operation::operand_range getMapIndices() {
      return getMapOperands();
    }

    SmallVector<Operation *> getUseChain();
    void cloneUseChainToNewOp(OpBuilder &builder, Operation *new_op);
    VectorStoreOp cloneWithNewAccessInfo(OpBuilder &builder, Value mem_view,
        const AffineMap &subscripts_map, ValueRange indices);
    void eraseOpAndUseChain();

    //===------------------------------------------------------------------===//
    // AffineMapAccessInterface
    //===------------------------------------------------------------------===//

    [[nodiscard]] auto getMapAttrStrName() -> StringRef {
      return getAffineMapAttrName();
    }
  }];
}

def Agen_CompositeLoadAndStoreOp : Agen_Op<"composite_load_and_store", [
    MemRefsNormalizable,
    ImplicitAgenTerminator,
    DeclareOpInterfaceMethods<AffineMapAccessInterface, ["getAffineMapAttrForMemRef"]>,
    AttrSizedOperandSegments,
    Dataflow_DebugNameOpInterface]> {
  let summary = "agen composite load_and_store operation";
  let description = [{
    Captures a sequence of individual vector loads and stores: at each time
    step it loads from one memory address and stores into another. This is used
    to model core-to-memory or memory-to-core data transfers.

    The element dimension is described by `$load_set`/`$store_set` (which
    elements form each vector) and `$load_order`/`$store_order` (how those
    elements are packed), mirroring `agen.vector_load`/`agen.vector_store`.

    The time dimension is described by:
      * `$time_set` - the set of time steps involved in the composite operation.
      * `$time_order` - the order among those time steps.
      * `$load_time_addr_map`/`$store_time_addr_map` - the address offset from
        the base address for each time step.

    The single region is entered once per time step. Its induction argument (see
    `getLoadInductionVar`) carries the vector loaded at that time step, and the
    region body describes what to do with it (for example, sending it to another
    unit, or storing it via the terminating `agen.yield`). `$dir` and
    `$multicast_info`, when present, carry routing information for the transfer.

    Example (three time steps loading columns 0, 1, 2):
    ```mlir
    agen.composite_load_and_store src:%src[0, 0] dst:%dst[0, 0]
      time_symbols(), load_iv(%load:vector<8xi8>)
      {load_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 7 >= 0, d1 == 0)>,
       load_order = affine_map<(d0, d1) -> (d1, d0)>,
       time_set = affine_set<(t0) : (t0 >= 0, -t0 + 2 >= 0)>,
       time_order = affine_map<(t0) -> (t0)>,
       load_time_addr_map = affine_map<(t0) -> (0, t0)>, ...}
    {
      // region: consume %load, e.g. agen.yield
    } : memref<8x16xi8>, memref<8x16xi8>
    ```
  }];
  let arguments = (ins
    Arg<AnyMemRef, "the reference to load from", [MemRead]>:$src_mem_ref,
    Arg<AnyMemRef, "the reference to store into", [MemWrite]>:$dst_mem_ref,
    OptionalAttr<StrAttr>:$dbgName,
    AffineMapAttr:$src_map,
    Variadic<Index>:$src_map_operands,
    AffineMapAttr:$dst_map,
    Variadic<Index>:$dst_map_operands,
    Builtin_IntegerSetAttr:$load_set,
    AffineMapAttr:$load_order,
    Builtin_IntegerSetAttr:$store_set,
    AffineMapAttr:$store_order,
    Variadic<Index>:$time_symbols,
    Builtin_IntegerSetAttr:$time_set,
    AffineMapAttr:$time_order,
    AffineMapAttr:$load_time_addr_map,
    AffineMapAttr:$store_time_addr_map,
    OptionalAttr<AgenRoutingDirectionAttr>:$dir,
    Optional<Index>:$multicast_info);

  let regions = (region SizedRegion<1>:$region);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
        "Value":$src_mem_ref, "Value":$dst_mem_ref, "StringAttr":$dbg_name,
        "AffineMapAttr":$src_map, "ValueRange":$src_map_operands, "AffineMapAttr":$dst_map,
        "ValueRange":$dst_map_operands, "IntegerSetAttr":$load_set, "AffineMapAttr":$load_order,
        "IntegerSetAttr":$store_set, "AffineMapAttr":$store_order, "ValueRange":$time_symbols,
        "IntegerSetAttr":$time_set, "AffineMapAttr":$time_order, "AffineMapAttr":$load_time_addr_map,
        "AffineMapAttr":$store_time_addr_map, "Type":$type,
        CArg<"AgenRoutingDirectionAttr", "nullptr">:$dir,
        CArg<"Value", "{}">:$multicast_info,
        CArg<"function_ref<void(OpBuilder &, Location, Value)>", "nullptr">:$body_builder)>,
    /// Builds a composite load_and_store op with the specified map and operands.
    OpBuilder<(ins
        "Value":$src_mem_ref, "Value":$dst_mem_ref, "StringAttr":$dbg_name,
        "AffineMap":$src_map, "ValueRange":$src_map_operands, "AffineMap":$dst_map,
        "ValueRange":$dst_map_operands, "IntegerSet":$load_set, "AffineMap":$load_order,
        "IntegerSet":$store_set, "AffineMap":$store_order, "ValueRange":$time_symbols,
        "IntegerSet":$time_set, "AffineMap":$time_order, "AffineMap":$load_time_addr_map,
        "AffineMap":$store_time_addr_map, "Type":$type,
        CArg<"AgenRoutingDirectionAttr", "nullptr">:$dir,
        CArg<"Value", "{}">:$multicast_info,
        CArg<"function_ref<void(OpBuilder &, Location, Value)>", "nullptr">:$body_builder)>
  ];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let extraClassDeclaration =  [{
    // Returns the src/dst memref value type.
    [[deprecated("use getSrcMemRef().getType()")]]
    MemRefType getSrcMemRefType() {return getSrcMemRef().getType(); }
    [[deprecated("use getDstMemRef().getType()")]]
    MemRefType getDstMemRefType() {return getDstMemRef().getType(); }

    // Returns the affine map used to index the src/dst memref for this operation.
    [[deprecated("use getSrcMapAttr()")]]
    AffineMapAttr getSrcAffineMapAttr() {
      return getSrcMapAttr();
    }
    [[deprecated("use getDstMapAttr()")]]
    AffineMapAttr getDstAffineMapAttr() {
      return getDstMapAttr();
    }

    Value getLoadInductionVar() {
      return getRegion().getBlocks().begin()->getArgument(0);
    }
    Type getLoadInductionVarType() {
      return getRegion().getBlocks().begin()->getArgumentTypes()[0];
    }

    [[deprecated("use getSrcMapOperands()")]]
    ::mlir::Operation::operand_range getSrcMapIndices() {
      return getSrcMapOperands();
    }
    [[deprecated("use getDstMapOperands()")]]
    ::mlir::Operation::operand_range getDstMapIndices() {
      return getDstMapOperands();
    }
    [[deprecated("use getSrcMapOperands()")]]
    ::mlir::Operation::operand_range src_indices() {
      return getSrcMapOperands();
    }
    [[deprecated("use getDstMapOperands()")]]
    ::mlir::Operation::operand_range dst_indices() {
      return getDstMapOperands();
    }

    CompositeLoadAndStoreOp cloneWithNewAccessInfo(OpBuilder &builder,
        Value src_mem_view, Value dst_mem_view,
        const AffineMap &src_subscripts_map, const AffineMap &dst_subscripts_map,
        ValueRange src_indices,
        ValueRange dst_indices,
        const IntegerSet &time_set);
  }];
}

def Agen_CompositeLoadOp:Agen_Op<"composite_load", [
    DeclareOpInterfaceMethods<AffineMapAccessInterface>,
    MemRefsNormalizable,
    ImplicitAgenTerminator,
    AttrSizedOperandSegments,
    Dataflow_DebugNameOpInterface]> {
  let summary = "agen composite load operation";
  let arguments = (ins
    Arg<AnyMemRef, "the reference to load from", [MemRead]>:$mem_ref,
    OptionalAttr<StrAttr>:$dbgName,
    AffineMapAttr:$affine_map,
    Variadic<Index>:$map_operands,
    Builtin_IntegerSetAttr:$load_set,
    AffineMapAttr:$load_order,
    Variadic<Index>:$time_symbols,
    Builtin_IntegerSetAttr:$time_set,
    AffineMapAttr:$time_order,
    AffineMapAttr:$time_addr_map);
  let regions = (region SizedRegion<1>:$region);

  let skipDefaultBuilders = 1;
  let builders = [
    /// Builds a composite load op with the specified map and operands.
    OpBuilder<(ins
      "Value":$mem_ref, "StringAttr":$dbg_name, "AffineMap":$affine_map,
      "ValueRange":$map_operands, "Type":$load_type, "IntegerSet":$load_set,
      "AffineMap":$load_order, "ValueRange":$time_symbols, "IntegerSet":$time_set,
      "mlir::AffineMap":$time_order, "mlir::AffineMap":$time_addr_map)>
  ];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let extraClassDeclaration =  [{
    // Returns the memref value type.
    [[deprecated("use getMemRef().getType()")]]
    MemRefType getMemRefType() { return getMemRef().getType(); }

    Value getLoadInductionVar() {
      return getRegion().getBlocks().begin()->getArgument(0);
    }
    Type getLoadInductionVarType() {
      return getRegion().getBlocks().begin()->getArgumentTypes()[0];
    }

    [[deprecated("use getMapOperands()")]]
    ::mlir::Operation::operand_range getMapIndices() {
      return getMapOperands();
    }

    CompositeLoadOp cloneWithNewAccessInfo(OpBuilder &builder,
        Value mem_view, const AffineMap &subscripts_map,
        ValueRange indices, const IntegerSet &time_set);

    //===------------------------------------------------------------------===//
    // AffineMapAccessInterface
    //===------------------------------------------------------------------===//

    [[nodiscard]] auto getMapAttrStrName() -> StringRef {
      return getAffineMapAttrName();
    }
  }];
}

def Agen_CompositeStoreOp:Agen_Op<"composite_store", [
    DeclareOpInterfaceMethods<AffineMapAccessInterface>,
    MemRefsNormalizable,
    ImplicitAgenTerminator,
    AttrSizedOperandSegments,
    Dataflow_DebugNameOpInterface]> {
  let summary = "agen composite store operation";
  let arguments = (ins
    Arg<AnyMemRef, "the reference to store to", [MemWrite]>:$mem_ref,
    OptionalAttr<StrAttr>:$dbgName,
    AffineMapAttr:$affine_map,
    Variadic<Index>:$map_operands,
    Optional<AnyTypeOf<[AnyVectorOfAnyRank, Dataflow_CustomVectorType]>>:$input_vector,
    OptionalAttr<Builtin_IntegerSetAttr>:$store_set,
    OptionalAttr<AffineMapAttr>:$store_order,
    Variadic<Index>:$time_symbols,
    Builtin_IntegerSetAttr:$time_set,
    AffineMapAttr:$time_order,
    AffineMapAttr:$time_addr_map);
  let regions = (region AnyRegion:$region);

  let skipDefaultBuilders = 1;
  let builders = [
    /// Builds a composite store op with store_set, store_order and region but no input vector.
    OpBuilder<(ins
      "Value":$mem_ref, "StringAttr":$dbg_name, "AffineMap":$map, "ValueRange":$map_operands,
      "IntegerSet":$store_set, "AffineMap":$store_order, "ValueRange":$time_symbols,
      "IntegerSet":$time_set, "AffineMap":$time_order, "AffineMap":$time_addr_map)>,
    /// Builds a composite store op with input vector but no store_set, store_order or region.
    OpBuilder<(ins
      "Value":$mem_ref, "StringAttr":$dbg_name, "AffineMap":$map, "ValueRange":$map_operands,
      "Value":$input_vector, "ValueRange":$time_symbols, "IntegerSet":$time_set,
      "AffineMap":$time_order, "AffineMap":$time_addr_map)>
  ];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let extraClassDeclaration =  [{
    // Returns the memref value type.
    [[deprecated("use getMemRef().getType()")]]
    MemRefType getMemRefType() { return getMemRef().getType(); }

    [[deprecated("use getMapOperands()")]]
    ::mlir::Operation::operand_range getMapIndices() {
      return getMapOperands();
    }

    // TODO need to consider Non-constant time symbols...
    unsigned int getTimeIterNum() {
      unsigned num = 1;
      for (auto t : getTimeSymbols()) {
        num *= mlir::dyn_cast<IntegerAttr>(t.getDefiningOp<arith::ConstantOp>().getValue()).getInt();
      }
      return num;
    }

    CompositeStoreOp cloneWithNewAccessInfo(OpBuilder &builder,
        Value mem_view, const AffineMap &subscripts_map,
        ValueRange indices, const IntegerSet &time_set);

    //===------------------------------------------------------------------===//
    // AffineMapAccessInterface
    //===------------------------------------------------------------------===//

    [[nodiscard]] auto getMapAttrStrName() -> StringRef {
      return getAffineMapAttrName();
    }
  }];
}

def Agen_CompositeIndirectLoadAndStoreOp:Agen_Op<"composite_indirect_load_and_store",
        [MemRefsNormalizable,
        ImplicitAgenTerminator,
        AttrSizedOperandSegments,
        Dataflow_DebugNameOpInterface]> {
  // TODO:
  // The `AffineMapAccessInterface` trait (and its uses in MLIR) expect a single
  // dereferenceable memref, whereas this operations dereferences upto four memrefs.
  // For safety we cannot inherit this trait, which means MLIR may assume that
  // memrefs used in `composite_indirect_load_and_store` operations escape. To improve
  // this we'd need to enhance the trait (or add a new one) and update a few
  // places in MLIR where they check for this trait.

  let summary = "agen composite indirect load_and_store operation";
  let description = [{
    composite_indirect_load_and_store operations are used for [indirectly] loading from one memory
    address and [indirectly] storing into another as part of each time step.
    All of the indirect access functions are optional.
  }];
  let arguments = (ins
        Optional<AnyMemRef>:$indirect_src_memref,
        AnyMemRef:$direct_src_memref,
        Optional<AnyMemRef>:$indirect_dst_memref,
        AnyMemRef:$direct_dst_memref,
        Variadic<Index>:$indirect_src_map_indices,
        Variadic<Index>:$direct_src_map_indices,
        Variadic<Index>:$indirect_dst_map_indices,
        Variadic<Index>:$direct_dst_map_indices,
        Optional<Index>:$multicast_info,
        Variadic<Index>:$time_symbols,
        // attrs:
        OptionalAttr<StrAttr>:$dbgName,
        OptionalAttr<AffineMapAttr>:$indirect_src_map,
        OptionalAttr<AffineMapAttr>:$direct_src_map,
        OptionalAttr<AffineMapAttr>:$indirect_dst_map,
        OptionalAttr<AffineMapAttr>:$direct_dst_map,
        Builtin_IntegerSetAttr:$load_set,
        AffineMapAttr:$load_order,
        Builtin_IntegerSetAttr:$store_set,
        AffineMapAttr:$store_order,
        Builtin_IntegerSetAttr:$time_set,
        AffineMapAttr:$time_order,
        OptionalAttr<AffineMapAttr>:$load_indirect_time_addr_map,
        AffineMapAttr:$load_direct_time_addr_map,
        OptionalAttr<AffineMapAttr>:$store_indirect_time_addr_map,
        AffineMapAttr:$store_direct_time_addr_map
  );
  let regions = (region SizedRegion<1>:$region);

  let skipDefaultBuilders = 1;
  let hasCustomAssemblyFormat = 1;
  let builders = [
    /// Builds a composite indirect load_and_store op with the specified map and operands.
    /// The $operands argument is the list of operands corresponding to $indirect_src_map_indices, $direct_src_map_indices,$indirect_dst_map_indices, $direct_dst_map_indices, $multicast_info, and $time_symbols in that order.
    /// Since some operands are optional and/or variadic, the number of SSA values in each segment of the operand list is passed to the builder as argument (eg. see $num_ind_src_memref_indices, etc).
    OpBuilder<(ins "Value":$indirect_src_memref, "Value":$direct_src_memref, "Value":$indirect_dst_memref, "Value":$direct_dst_memref, "mlir::StringAttr":$dbgName,
      "mlir::AffineMap":$indirect_src_map, "mlir::AffineMap":$direct_src_map, "mlir::AffineMap":$indirect_dst_map, "mlir::AffineMap":$direct_dst_map, "ValueRange":$operands,
      "Type":$type, "IntegerSet":$load_set, "mlir::AffineMap":$load_order,  "IntegerSet":$store_set, "mlir::AffineMap":$store_order,
      "IntegerSet":$time_set, "mlir::AffineMap":$time_order, "mlir::AffineMap":$indirect_load_time_addr_map, "mlir::AffineMap":$direct_load_time_addr_map,
      "mlir::AffineMap":$indirect_store_time_addr_map, "mlir::AffineMap":$direct_store_time_addr_map,
      "uint32_t":$num_ind_src_memref_indices, "uint32_t":$num_dir_src_memref_indices, "uint32_t":$num_ind_dst_memref_indices, "uint32_t":$num_dir_dst_memref_indices,
      "uint32_t":$num_multicast_info, "uint32_t":$num_time_symbols,
      CArg<"function_ref<void(OpBuilder &, Location, Value, ValueRange)>",
           "nullptr">:$bodyBuilder)>
  ];

  let extraClassDeclaration =  [{
    using BodyBuilderFn =
        function_ref<void(OpBuilder &, Location, Value, ValueRange)>;

    // Returns the src/dst memref value type.
    MemRefType getIndirectSrcMemRefType() {return mlir::cast<MemRefType>(getIndirectSrcMemref().getType()); }
    MemRefType getDirectSrcMemRefType() {return mlir::cast<MemRefType>(getDirectSrcMemref().getType()); }
    MemRefType getIndirectDstMemRefType() {return mlir::cast<MemRefType>(getIndirectDstMemref().getType()); }
    MemRefType getDirectDstMemRefType() {return mlir::cast<MemRefType>(getDirectDstMemref().getType()); }
    IndexType getMulticastInfoType() { return mlir::cast<IndexType>(getMulticastInfo().getType()); }

    // Returns the affine map used to index the src/dst memref for this operation.
    [[deprecated("use getIndirectSrcMapAttr()")]]
    AffineMapAttr getIndirectSrcAffineMapAttr() {
      return getIndirectSrcMapAttr();
    }
    [[deprecated("use getDirectSrcMapAttr()")]]
    AffineMapAttr getDirectSrcAffineMapAttr() {
      return getDirectSrcMapAttr();
    }
    [[deprecated("use getIndirectDstMapAttr()")]]
    AffineMapAttr getIndirectDstAffineMapAttr() {
      return getIndirectDstMapAttr();
    }
    [[deprecated("use getDirectDstMapAttr()")]]
    AffineMapAttr getDirectDstAffineMapAttr() {
      return getDirectDstMapAttr();
    }

    Value getLoadInductionVar() { return getRegion().getBlocks().begin()->getArgument(0); }
    Type getLoadInductionVarType() { return getRegion().getBlocks().begin()->getArgumentTypes()[0]; }

    bool hasIndirectSrc() {
      return static_cast<bool>(getIndirectSrcMemref());
    }
    bool hasIndirectDst() {
      return static_cast<bool>(getIndirectDstMemref());
    }
    bool hasMulticastInfo() {
      return static_cast<bool>(getMulticastInfo());
    }

    CompositeIndirectLoadAndStoreOp cloneWithNewAccessInfo(
        OpBuilder &builder, const Value &ind_src_mem_view,
        const Value &dir_src_mem_view,
        const Value &ind_dst_mem_view, const Value &dir_dst_mem_view,
        const AffineMap &ind_src_map, const AffineMap &dir_src_map,
        const AffineMap &ind_dst_map, const AffineMap &dir_dst_map,
        const SmallVectorImpl<Value> &dir_src_indices,
        const SmallVectorImpl<Value> &dir_dst_indices, const IntegerSet time_set);
  }];
}

def Agen_CompositeIndirectLoadOp:Agen_Op<"composite_indirect_load",
        [MemRefsNormalizable,
        ImplicitAgenTerminator,
        AttrSizedOperandSegments,
        Dataflow_DebugNameOpInterface]> {
  // TODO:
  // The `AffineMapAccessInterface` trait (and its uses in MLIR) expect a single
  // dereferenceable memref, whereas this operations dereferences two memrefs.
  // For safety we cannot inherit this trait, which means MLIR may assume that
  // memrefs used in `composite_indirect_load` operations escape. To improve
  // this we'd need to enhance the trait (or add a new one) and update a few
  // places in MLIR where they check for this trait.

  let summary = "agen composite indirect load operation";
  let description = [{
    composite_indirect_load operations are used for [indirectly] loading from a memory
    address as part of each time step.
  }];
  let arguments = (ins
        AnyMemRef:$indirect_memref,
        AnyMemRef:$direct_memref,
        Variadic<Index>:$indirect_map_indices,
        Variadic<Index>:$direct_map_indices,
        Optional<Index>:$multicast_info,
        Variadic<Index>:$time_symbols,
        // attrs:
        OptionalAttr<StrAttr>:$dbgName,
        AffineMapAttr:$indirect_map,
        AffineMapAttr:$direct_map,
        Builtin_IntegerSetAttr:$load_set,
        AffineMapAttr:$load_order,
        Builtin_IntegerSetAttr:$time_set,
        AffineMapAttr:$time_order,
        AffineMapAttr:$time_addr_map
  );
  let regions = (region SizedRegion<1>:$region);

  let skipDefaultBuilders = 1;
  let hasCustomAssemblyFormat = 1;
  let builders = [
    /// Builds a composite indirect load op with the specified map and operands.
    /// The $operands argument is the list of operands corresponding to $indirect_map_indices, $direct_map_indices, $multicast_info, and $time_symbols in that order.
    /// Since some operands are optional and/or variadic, the number of SSA values in each segment of the operand list is passed to the builder as argument (eg. see $num_ind_memref_indices, etc).
    OpBuilder<(ins "Value":$indirect_memref, "Value":$direct_memref, "mlir::StringAttr":$dbgName,
      "mlir::AffineMap":$indirect_map, "mlir::AffineMap":$direct_map, "ValueRange":$operands,
      "Type":$type, "IntegerSet":$load_set, "mlir::AffineMap":$load_order,
      "IntegerSet":$time_set, "mlir::AffineMap":$time_order, "mlir::AffineMap":$time_addr_map,
      "uint32_t":$num_ind_memref_indices, "uint32_t":$num_dir_memref_indices,
      "uint32_t":$num_multicast_info, "uint32_t":$num_time_symbols,
      CArg<"function_ref<void(OpBuilder &, Location, Value, ValueRange)>",
           "nullptr">:$body_builder)>
  ];

  let extraClassDeclaration =  [{
    using BodyBuilderFn =
        function_ref<void(OpBuilder &, Location, Value, ValueRange)>;

    // Returns the memref value type.
    MemRefType getIndirectMemrefType() {return mlir::cast<MemRefType>(getIndirectMemref().getType()); }
    MemRefType getDirectMemrefType() {return mlir::cast<MemRefType>(getDirectMemref().getType()); }
    IndexType getMulticastInfoType() { return mlir::cast<IndexType>(getMulticastInfo().getType()); }

    // Returns the affine map used to index the src memref for this operation.
    [[deprecated("use getIndirectMapAttr()")]]
    AffineMapAttr getIndirectAffineMapAttr() {
      return getIndirectMapAttr();
    }
    [[deprecated("use getDirectMapAttr()")]]
    AffineMapAttr getDirectAffineMapAttr() {
      return getDirectMapAttr();
    }

    Value getLoadInductionVar() { return getRegion().getBlocks().begin()->getArgument(0); }
    Type getLoadInductionVarType() { return getRegion().getBlocks().begin()->getArgumentTypes()[0]; }

    bool hasMulticastInfo() {
      return static_cast<bool>(getMulticastInfo());
    }
  }];
}

def Agen_CompositeIndirectStoreOp:Agen_Op<"composite_indirect_store",
        [MemRefsNormalizable,
        ImplicitAgenTerminator,
        AttrSizedOperandSegments,
        Dataflow_DebugNameOpInterface]> {
  // TODO:
  // The `AffineMapAccessInterface` trait (and its uses in MLIR) expect a single
  // dereferenceable memref, whereas this operations dereferences two memrefs.
  // For safety we cannot inherit this trait, which means MLIR may assume that
  // memrefs used in `composite_indirect_store` operations escape. To improve
  // this we'd need to enhance the trait (or add a new one) and update a few
  // places in MLIR where they check for this trait.

  let summary = "agen composite indirect store operation";
  let description = [{
    composite_indirect_store operations are used for [indirectly] storing into a memory
    address as part of each time step.
  }];
  let arguments = (ins

        AnyMemRef:$indirect_memref,
        AnyMemRef:$direct_memref,
        Variadic<Index>:$indirect_map_indices,
        Variadic<Index>:$direct_map_indices,
        Optional<Index>:$multicast_info,
        Variadic<Index>:$time_symbols,
        // attrs:
        OptionalAttr<StrAttr>:$dbgName,
        AffineMapAttr:$indirect_map,
        AffineMapAttr:$direct_map,
        Builtin_IntegerSetAttr:$store_set,
        AffineMapAttr:$store_order,
        Builtin_IntegerSetAttr:$time_set,
        AffineMapAttr:$time_order,
        AffineMapAttr:$time_addr_map
  );
  let regions = (region SizedRegion<1>:$region);

  let skipDefaultBuilders = 1;
  let hasCustomAssemblyFormat = 1;
  let builders = [
    /// Builds a composite indirect store op with the specified map and operands.
    /// The $operands argument is the list of operands corresponding to $indirect_map_indices, $direct_map_indices, $multicast_info, and $time_symbols in that order.
    /// Since some operands are optional and/or variadic, the number of SSA values in each segment of the operand list is passed to the builder as argument (eg. see $num_ind_memref_indices, etc).
    OpBuilder<(ins "Value":$indirect_memref, "Value":$direct_memref, "mlir::StringAttr":$dbgName,
      "mlir::AffineMap":$indirect_map, "mlir::AffineMap":$direct_map, "ValueRange":$operands,
      "Type":$type, "IntegerSet":$store_set, "mlir::AffineMap":$store_order,
      "IntegerSet":$time_set, "mlir::AffineMap":$time_order,
      "mlir::AffineMap":$time_addr_map,
      "uint32_t":$num_ind_memref_indices, "uint32_t":$num_dir_memref_indices,
      "uint32_t":$num_multicast_info, "uint32_t":$num_time_symbols,
      CArg<"function_ref<void(OpBuilder &, Location, Value, ValueRange)>",
           "nullptr">:$body_builder)>
  ];

  let extraClassDeclaration =  [{
    using BodyBuilderFn =
        function_ref<void(OpBuilder &, Location, Value, ValueRange)>;

    // Returns the memref value type.
    MemRefType getIndirectMemrefType() {return mlir::cast<MemRefType>(getIndirectMemref().getType()); }
    MemRefType getDirectMemrefType() {return mlir::cast<MemRefType>(getDirectMemref().getType()); }
    IndexType getMulticastInfoType() { return mlir::cast<IndexType>(getMulticastInfo().getType()); }

    // Returns the affine map used to index the memref for this operation.
    [[deprecated("use getIndirectMapAttr()")]]
    AffineMapAttr getIndirectAffineMapAttr() {
      return getIndirectMapAttr();
    }
    [[deprecated("use getDirectMapAttr()")]]
    AffineMapAttr getDirectAffineMapAttr() {
      return getDirectMapAttr();
    }

    bool hasMulticastInfo() {
      return static_cast<bool>(getMulticastInfo());
    }
  }];
}

def Agen_IndirectVectorLoadOp:Agen_Op<"indirect_vector_load",
        [MemRefsNormalizable,
        ImplicitAgenTerminator,
        AttrSizedOperandSegments,
        Dataflow_DebugNameOpInterface]> {
  // TODO:
  // The `AffineMapAccessInterface` trait (and its uses in MLIR) expect a single
  // dereferenceable memref, whereas this operations dereferences two memrefs.
  // For safety we cannot inherit this trait, which means MLIR may assume that
  // memrefs used in `indirect_vector_load` operations escape. To improve
  // this we'd need to enhance the trait (or add a new one) and update a few
  // places in MLIR where they check for this trait.

  let summary = "agen indirect vector_load operation";
  let description = [{
    indirect_vector_load operations are used for [indirectly] loading from a memory
    address.
  }];
  let arguments = (ins
        AnyMemRef:$indirect_memref,
        AnyMemRef:$direct_memref,
        Variadic<Index>:$indirect_map_indices,
        Variadic<Index>:$direct_map_indices,
        Optional<Index>:$multicast_info,
        // attrs:
        OptionalAttr<StrAttr>:$dbgName,
        AffineMapAttr:$indirect_map,
        AffineMapAttr:$direct_map,
        Builtin_IntegerSetAttr:$load_set,
        AffineMapAttr:$load_order
  );

  let results = (outs AnyTypeOf<[AnyVectorOfAnyRank, Dataflow_CustomVectorType]>:$result);
  let skipDefaultBuilders = 1;
  let hasCustomAssemblyFormat = 1;
  let builders = [
    /// Builds an indirect vector_load op with the specified map and operands.
    /// The $operands argument is the list of operands corresponding to $indirect_map_indices, $direct_map_indices, $multicast_info
    /// in that order.
    /// Since some operands are optional and/or variadic, the number of SSA values in each segment of the operand list is passed to
    /// the builder as argument (eg. see $num_ind_memref_indices, etc).
    OpBuilder<(ins "Type":$resultType, "Value":$indirect_memref, "Value":$direct_memref, "mlir::StringAttr":$dbgName,
      "mlir::AffineMap":$indirect_map, "mlir::AffineMap":$direct_map,
      "ValueRange":$operands, "Type":$type, "IntegerSet":$load_set, "mlir::AffineMap":$load_order,
      "uint32_t":$num_ind_memref_indices, "uint32_t":$num_dir_memref_indices, "uint32_t":$num_multicast_info)>
  ];

  let extraClassDeclaration =  [{
    VectorType getVectorType() {
      return dyn_cast<VectorType>(getResult().getType());
    }

    // Returns the memref value type.
    MemRefType getIndirectMemrefType() {return mlir::cast<MemRefType>(getIndirectMemref().getType()); }
    MemRefType getDirectMemrefType() {return mlir::cast<MemRefType>(getDirectMemref().getType()); }
    IndexType getMulticastInfoType() { return mlir::cast<IndexType>(getMulticastInfo().getType()); }

    // Returns the affine map used to index the memref for this operation.
    [[deprecated("use getIndirectMapAttr()")]]
    AffineMapAttr getIndirectAffineMapAttr() {
      return getIndirectMapAttr();
    }
    [[deprecated("use getDirectMapAttr()")]]
    AffineMapAttr getDirectAffineMapAttr() {
      return getDirectMapAttr();
    }

    bool hasMulticastInfo() {
      return static_cast<bool>(getMulticastInfo());
    }
  }];
}

def Agen_IndirectVectorStoreOp:Agen_Op<"indirect_vector_store",
        [MemRefsNormalizable,
        ImplicitAgenTerminator,
        AttrSizedOperandSegments,
        Dataflow_DebugNameOpInterface]> {
  // TODO:
  // The `AffineMapAccessInterface` trait (and its uses in MLIR) expect a single
  // dereferenceable memref, whereas this operations dereferences two memrefs.
  // For safety we cannot inherit this trait, which means MLIR may assume that
  // memrefs used in `indirect_vector_store` operations escape. To improve
  // this we'd need to enhance the trait (or add a new one) and update a few
  // places in MLIR where they check for this trait.

  let summary = "agen indirect vector_store operation";
  let description = [{
    indirect_vector_store operations are used for [indirectly] storing into a memory
    address.
  }];
  let arguments = (ins
        AnyType:$value,
        AnyMemRef:$indirect_memref,
        AnyMemRef:$direct_memref,
        Variadic<Index>:$indirect_map_indices,
        Variadic<Index>:$direct_map_indices,
        Optional<Index>:$multicast_info,
        // attrs:
        OptionalAttr<StrAttr>:$dbgName,
        AffineMapAttr:$indirect_map,
        AffineMapAttr:$direct_map,
        Builtin_IntegerSetAttr:$store_set,
        AffineMapAttr:$store_order
  );

  let skipDefaultBuilders = 1;
  let hasCustomAssemblyFormat = 1;
  let builders = [
    /// Builds an indirect vector_store op with the specified map and operands.
    /// The $operands argument is the list of operands corresponding to $indirect_map_indices, $direct_map_indices, $multicast_info
    /// in that order.
    /// Since some operands are optional and/or variadic, the number of SSA values in each segment of the operand list is passed to
    /// the builder as argument (eg. see $num_ind_src_memref_indices, etc).
    OpBuilder<(ins "Value":$value, "Value":$indirect_memref, "Value":$direct_memref, "mlir::StringAttr":$dbgName, "mlir::AffineMap":$indirect_map,
       "mlir::AffineMap":$direct_map, "ValueRange":$operands, "Type":$type, "IntegerSet":$store_set, "mlir::AffineMap":$store_order,
      "uint32_t":$num_ind_memref_indices, "uint32_t":$num_dir_memref_indices, "uint32_t":$num_multicast_info)>
  ];

  let extraClassDeclaration =  [{
    VectorType getVectorType() {
      return dyn_cast<VectorType>(getValue().getType());
    }
    // Returns the memref value type.
    MemRefType getIndirectMemrefType() {return mlir::cast<MemRefType>(getIndirectMemref().getType()); }
    MemRefType getDirectMemrefType() {return mlir::cast<MemRefType>(getDirectMemref().getType()); }
    IndexType getMulticastInfoType() { return mlir::cast<IndexType>(getMulticastInfo().getType()); }

    // Returns the affine map used to index the memref for this operation.
    [[deprecated("use getIndirectMapAttr()")]]
    AffineMapAttr getIndirectAffineMapAttr() {
      return getIndirectMapAttr();
    }
    [[deprecated("use getDirectMapAttr()")]]
    AffineMapAttr getDirectAffineMapAttr() {
      return getDirectMapAttr();
    }

    bool hasMulticastInfo() {
      return static_cast<bool>(getMulticastInfo());
    }
  }];
}

def Agen_CompositeMemoryInterleaveOp:Agen_Op<"composite_memory_interleave",
    [ImplicitAgenTerminator, Dataflow_DebugNameOpInterface]> {
  let summary = "agen composite memory interleaving operation";
  let description = [{
    The composite_memory_interleave operation interleaves the composite memory
    operations contained in its region in bursts specified by the granularity
    attribute. If no granularity is specified, granularity is assumed to be the
    maximum burst amount supported by the hardware.

    For example:
    agen.composite_memory_interleave [granularity = 64] {
      composite memory operation A {burst = 80}
      composite memory operation B {burst = 80}
    }
    -->
    composite memory operation A {burst = 64}
    composite memory operation B {burst = 64}
    composite memory operation A {burst = 16}
    composite memory operation B {burst = 16}

    The granularity cannot exceed the burst of the contained operations. If the
    granularity is set to a value greater than the burst of the contained
    operations, it will be interpreted as being equal to the burst. This
    behaviour is required because some of the upstream components may insert
    CompositeMemoryInterleaveOps without knowing the burst of the contained
    operations first.

    For example:
    agen.composite_memory_interleave [granularity = 32] {
      composite memory operation A {burst = 16}
      composite memory operation B {burst = 16}
    }
    -->
    composite memory operation A {burst = 16}
    composite memory operation B {burst = 16}

    Restrictions:
    - only composite memory operations are allowed in containing region
    - the granularity cannot exceed maximum burst amount supported by hardware
    - all composite memory operations inside the containing region need to
      utilize the same burst
    - all composite memory operations inside the containing region need to be
      the same operation type
  }];

  let arguments = (ins OptionalAttr<StrAttr>:$dbgName, OptionalAttr<I32Attr>:$granularity);
  let regions = (region SizedRegion<1>:$region);

  let hasCustomAssemblyFormat = 1;
  let skipDefaultBuilders = 1;

  let builders = [
    // Builds the CompositeMemoryInterleaveOp with a specific granularity.
    OpBuilder<(ins "mlir::StringAttr":$dbgName, "uint32_t":$granularity)>,
    // Builds the CompositeMemoryInterleaveOp with no granularity attribute.
    OpBuilder<(ins "mlir::StringAttr":$dbgName)>
  ];
}

def Agen_SetTransferMaskStateOp:Agen_Op<"set_transfer_mask_state", [
    MemoryEffects<[MemWrite]>,
    Dataflow_DebugNameOpInterface]> {

  let summary = "agen set transfer mask state operation";
  let description = [{
    The set_transfer_mask_state operation collects information from various
    mask patterns and forms one final mask that will be applied to transfers
    for the given unit.

    mask_value: The value that is applied over the elements that are masked.
    num_slices: The mask can be separated into a number of equal sections (slices).
                Each slice may have a different mask.
    slice_mask_map: A description of the mask that will be applied to each
                    slice. Each slice may have a different masking pattern.
                    Some slices may consist of combinations of multiple masks.

                    Each slice is delimited by parentheses. A slice containing
                    a 0 indicates no masking. A slice containing a letter
                    corresponds to the indicated mask. Lettering starts with A
                    and continues sequentially for each mask thereafter.

                    Current support for combinations of masks include:
                      OR operation: (A|B)

                    Example: (0)(0)(A|B)(A)(B)(1) describes a mask containing
                             6 slices. The first two slices will have no masking.
                             The third slice will be masked by a mask pattern
                             determined by OR'ing masks A and B. The fourth slice
                             will be masked by mask A. The fifth slice will be
                             masked by a mask B. The sixth slice will be fully
                             masked.

    *num_unmasked_elements: Number of consecutive unmasked elements in the mask
                            pattern. If multiple masks are involved, the number of
                            consecutive unmasked elements for each mask is provided
                            in an ordered list.

    *num_masked_elements*: Number of consecutive masked elements in the mask pattern.
                           If multiple masks are involved, the number of consecutive
                           masked elements for each mask is provided in an ordered
                           list.

    * If a mask is shorter than the number of elements in a slice, it is repeated to
      fill the slice. The sum of num_unmasked_elements and num_masked elements at a
      given index must evenly divide into the number of elements in a slice.

    Example IR:
      %mask = agen.set_transfer_mask_state mask_unit(%l1lu_c0) mask_value(%c0) {
                  num_slices = 8 : i32, slice_mask_map = "(A)(A)(A)(A)(A)(A|B)(1)(1)",
                  maskA = "(unmasked = 6 : i32, masked = 2 : i32)",
                  maskB = "(unmasked = 1 : i32, masked = 1 : i32)" } :
                  index, index : vector<64xf16>
  }];
  let arguments = (ins
        Index:$mask_value,
        OptionalAttr<StrAttr>:$dbgName,
        I32Attr:$num_slices,
        StrAttr:$slice_mask_map,
        OptionalAttr<I32ArrayAttr>:$num_unmasked_elements,
        OptionalAttr<I32ArrayAttr>:$num_masked_elements);

  let results = (outs AnyVectorOfAnyRank:$result);

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration =  [{
    IndexType getMaskValueType() { return mlir::cast<IndexType>(getMaskValue().getType()); }

    static StringRef getMaskPrefixAttrStrName() { return "mask"; }

    bool isUnmask();
    bool isFullMask();
  }];
}

// TODO: Check if any additional traits are required.
def Agen_SymbolicVectorLoadOp:Agen_Op<"symbolic_vector_load",
[MemRefsNormalizable, Dataflow_DebugNameOpInterface]> {

  let summary = "agen symbolic vector load operation";
  let arguments = (ins Arg<AnyMemRef, "the reference to load from",
        [MemRead]>:$memref,
        OptionalAttr<StrAttr>:$dbgName,
        Builtin_IntegerSetAttr:$load_set,
        AffineMapAttr:$load_order,
        I32Attr:$num_indices,
        I32Attr:$num_strides,
        Variadic<Index>:$operands1);

  let results = (outs AnyTypeOf<[AnyVectorOfAnyRank, Dataflow_CustomVectorType]>:$result);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let extraClassDeclaration =  [{
    VectorType getVectorType() {
      return dyn_cast<VectorType>(getResult().getType());
    }
    /// Returns the operand index of the memref.
    unsigned getMemrefOperandIndex() { return 0; }

    /// Returns the operand index of the operands1 operands list.
    unsigned getOperands1OperandIndex() { return 1; }

    void setMemref(Value value) { setOperand(getMemrefOperandIndex(), value); }

    /// Returns the affine map for the agen vector load.
    [[deprecated("use getLoadSetAttr()")]]
    IntegerSetAttr getAffineLoadSetAttr() {
      return getLoadSetAttr();
    }

    ::mlir::Operation::operand_range getIndices() {
      return getOperands1().slice(0, getNumIndices());
    }

    ::mlir::Operation::operand_range getStrides() {
      return getOperands1().slice(getNumIndices(), getNumStrides());
    }

    std::optional<Value> getMulticastInfo() {
      auto num_other_operands = getNumIndices() + getNumStrides();
      auto multicast_size = getOperands1().size() - num_other_operands;
      if (multicast_size == 1)
        return getOperands1().slice(num_other_operands, 1)[0];
      else if (multicast_size == 0)
        return std::optional<Value>();
      else
        llvm_unreachable("expect one or no multicast info");
    }

  }];
}

def Agen_SymbolicVectorStoreOp : Agen_Op<"symbolic_vector_store",
[MemRefsNormalizable, Dataflow_DebugNameOpInterface]> {

  let summary = "agen symbolic vector store operation";
  let arguments = (ins AnyType:$value_to_store,
      Arg<AnyMemRef, "the reference to store to",
      [MemWrite]>:$memref,
      OptionalAttr<StrAttr>:$dbgName,
      Builtin_IntegerSetAttr:$store_set,
      AffineMapAttr:$store_order,
      I32Attr:$num_indices,
      I32Attr:$num_strides,
      Variadic<Index>:$operands1);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let extraClassDeclaration =  [{
    /// Returns the operand index of the value to be stored.
    unsigned getStoredValOperandIndex() { return 0; }

    VectorType getVectorType() {
        return dyn_cast<VectorType>(getValueToStore().getType());
    }

    /// Returns the operand index of the memref.
    unsigned getMemrefOperandIndex() { return 1; }

    // Returns the operand index of the indices operands.
    unsigned getIndicesOperandIndex() { return 2; }

    // Returns the operand index of the strides operands.
    unsigned getStridesOperandIndex() { return 2 + getNumIndices(); }

    void setMemref(Value value) { setOperand(getMemrefOperandIndex(), value); }

    /// Returns the affine map for then agen vector store.
    [[deprecated("use getStoreSetAttr()")]]
    IntegerSetAttr getAffineStoreSetAttr() {
        return getStoreSetAttr();
    }

    ::mlir::Operation::operand_range getIndices() {
      return getOperands1().slice(0, getNumIndices());
    }

    ::mlir::Operation::operand_range getStrides() {
      return getOperands1().slice(getNumIndices(), getNumStrides());
    }
  }];
}

#endif // DATAFLOW_SCHEDULER_DIALECT_AGEN_AGEN_TD
