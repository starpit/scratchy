//===-- KTDF.td --------------------------------------------*- tablegen -*-===//
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
// This file defines the operations in the ktdf dialect.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_DIALECTS_KTDF_KTDF_TD
#define DATAFLOW_SCHEDULER_DIALECTS_KTDF_KTDF_TD

include "mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "dataflow-scheduler/Dialect/KTDF/KTDFTypes.td"

// Base class for KTDF dialect operations
class KTDF_Op<string mnemonic, list<Trait> traits = []> :
    Op<KTDF_Dialect, mnemonic, traits> {
}

//===----------------------------------------------------------------------===//
// FIFO Operations
//===----------------------------------------------------------------------===//

def KTDF_FifoAllocateOp : KTDF_Op<"fifo.allocate"> {
  let summary = "Allocate FIFO slots for dataflow communication";
  let description = [{
    The `ktdf.fifo.allocate` operation allocates one or more FIFO slots for
    dataflow communication between stages. Each slot has a specific FIFO type
    (e.g., "L1LU" -> "SFU") and size (capacity).

    ## FIFO Slot Semantics

    From the IR's perspective, FIFOs are modeled as having infinite depth (i.e.,
    an unbounded number of slots) from which slots are allocated and accessed in
    a rolling fashion. This means:

    1. **Ordered Access**: The order in which slots are allocated must match the
       order in which they are used in data transfers.

    2. **Capacity & Frequency Enforcement**: The total amount of data transferred through a
       given slot must exactly match the slot's size before the next slot can be
       used in a subsequent transfer.

    3. **Rolling Behavior**: After all allocated slots have been used, the FIFO
       rolls back to the first slot, allowing it to be reused for the next round
       of transfers.

    For example:
    ```mlir
    %slot1, %slot2 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 2>,
                                              !ktdf.fifo.slot<"L1LU" -> "SFU", 4>
    ```

    This allocation means:
    - Exactly 2 elements of data from l1lu must be transferred to sfu via %slot1
      before any data can be transferred via %slot2
    - Exactly 4 elements of data from l1lu must be transferred via %slot2 before
      the FIFO rolls back, allowing data to be transferred via %slot1 again

    **Loop Semantics**: Allocation of FIFO slots within a loop does not affect
    the lifetime of slots allocated in previous iterations. Each iteration's
    allocations are independent, enabling pipelined execution across loop iterations.

    ## Size Specification

    Sizes can be specified either as:
    1. Static values in the result type (e.g., `!ktdf.fifo.slot<"L1LU" -> "SFU", 4>`)
    2. Dynamic values using SSA operands, with `?` in the result type

    When using dynamic sizes, provide one index operand per dynamic slot.
    The sizes are matched to the result types in order.

    ## Examples

    ```mlir
    // Allocate two FIFO slots with static sizes
    %slot0, %slot1 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 1>,
                                              !ktdf.fifo.slot<"L1LU" -> "SFU", 4>
    
    // Allocate a single FIFO slot with dynamic size
    %size = arith.constant 8 : index
    %slot = ktdf.fifo.allocate(%size) -> !ktdf.fifo.slot<"SFU" -> "L1SU", ?>
    
    // Mix static and dynamic sizes
    %dyn_size = arith.constant 16 : index
    %s0, %s1 = ktdf.fifo.allocate(%dyn_size) -> !ktdf.fifo.slot<"L1LU" -> "SFU", ?>,
                                                 !ktdf.fifo.slot<"L1LU" -> "SFU", 4>
    ```
  }];

  let arguments = (ins
    Variadic<Index>:$dynamic_sizes
  );

  let results = (outs
    Variadic<KTDF_FifoSlotType>:$slots
  );

  let assemblyFormat = [{
    `(` $dynamic_sizes `)` attr-dict `->` type($slots)
  }];

  let hasVerifier = 1;
  let hasCanonicalizer = 1;
  
  let extraClassDeclaration = [{
    /// Returns the number of FIFO slots being allocated.
    unsigned getNumSlots() { return getSlots().size(); }
  }];
}

def KTDF_ReadFromFifoOp : KTDF_Op<"read_from_fifo", [
    AllMatchSameOperatorTrait<
      ["fifo_slot", "result"], 
      KTDF_ElementType<"_self">.result, 
      "element type">
  ]> {
  let summary = "Read data from a FIFO slot into a tensor";
  let description = [{
    The `ktdf.read_from_fifo` operation reads data from a FIFO slot and returns
    it as a tensor. This operation is used to consume data that has been written
    to a FIFO slot by a previous stage in the dataflow pipeline.

    The operation takes a FIFO slot as input and produces a tensor with the same
    element type as the FIFO slot. The shape of the resulting tensor is determined
    by the number of elements in the FIFO slot.

    Example:
    ```mlir
    // Read from a FIFO slot with 64 f16 elements
    %slot = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
    %data = ktdf.read_from_fifo %slot : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16> -> tensor<64xf16>
    
    // Read from a FIFO slot with dynamic size
    %size = arith.constant 8 : index
    %dyn_slot = ktdf.fifo.allocate(%size) -> !ktdf.fifo.slot<"SFU" -> "L1SU", ?xf16>
    %dyn_data = ktdf.read_from_fifo %dyn_slot : !ktdf.fifo.slot<"SFU" -> "L1SU", ?xf16> -> tensor<?xf16>
    ```
  }];

  let arguments = (ins
    KTDF_FifoSlotType:$fifo_slot
  );

  let results = (outs
    AnyRankedTensor:$result
  );

  let assemblyFormat = [{
    $fifo_slot attr-dict `:` type($fifo_slot) `->` type($result)
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    /// Returns the element type of the FIFO slot
    auto getFifoElementType() -> Type {
      return getFifoSlot().getType().getElementType();
    }
    
    /// Returns the number of elements in the FIFO slot
    auto getFifoNumElements() -> int64_t {
      return getFifoSlot().getType().getNumElements();
    }
  }];
}

def KTDF_WriteToFifoOp : KTDF_Op<"write_to_fifo", [
    AllMatchSameOperatorTrait<
      ["data", "fifo_slot"], 
      KTDF_ElementType<"_self">.result, 
      "element type">
  ]> {
  let summary = "Write data from a tensor to a FIFO slot";
  let description = [{
    The `ktdf.write_to_fifo` operation writes data from a tensor to a FIFO slot.
    This operation is used to produce data that will be consumed by a subsequent
    stage in the dataflow pipeline.

    The operation takes a tensor and a FIFO slot as inputs. The tensor's element
    type and shape must match the FIFO slot's element type and capacity.

    Example:
    ```mlir
    // Write to a FIFO slot with 64 f16 elements
    %data = arith.constant dense<1.0> : tensor<64xf16>
    %slot = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
    ktdf.write_to_fifo %data, %slot : tensor<64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
    
    // Write to a FIFO slot with dynamic size
    %size = arith.constant 8 : index
    %dyn_data = tensor.empty(%size) : tensor<?xf16>
    %dyn_slot = ktdf.fifo.allocate(%size) -> !ktdf.fifo.slot<"SFU" -> "L1SU", ?xf16>
    ktdf.write_to_fifo %dyn_data, %dyn_slot : tensor<?xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", ?xf16>
    ```
  }];

  let arguments = (ins
    AnyRankedTensor:$data,
    KTDF_FifoSlotType:$fifo_slot
  );

  let assemblyFormat = [{
    $data `,` $fifo_slot attr-dict `:` type($data) `,` type($fifo_slot)
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    /// Returns the element type of the FIFO slot
    auto getFifoElementType() -> Type {
      return getFifoSlot().getType().getElementType();
    }
    
    /// Returns the number of elements in the FIFO slot
    auto getFifoNumElements() -> int64_t {
      return getFifoSlot().getType().getNumElements();
    }
  }];
}

//===----------------------------------------------------------------------===//
// Token Operations
//===----------------------------------------------------------------------===//

def KTDF_CreateTokenOp : KTDF_Op<"create_token", [Pure]> {
  let summary = "Create a new token for dataflow synchronization";
  let description = [{
    The `ktdf.create_token` operation creates a new token value that can be used
    for dataflow synchronization and dependency tracking. Tokens represent
    abstract synchronization points in the dataflow graph and do not carry data.

    Example:
    ```mlir
    %token = ktdf.create_token : !ktdf.token
    ```
  }];

  let arguments = (ins KTDF_UniqueProp:$unique_id);
  let results = (outs KTDF_TokenType:$result);

  let assemblyFormat = "attr-dict `:` type($result)";

  let builders = [
    OpBuilder<(ins), [{
      build($_builder, $_state, TokenType::get($_builder.getContext()));
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// Tile Size Operations
//===----------------------------------------------------------------------===//

def KTDF_TilingReserveSizeOp : KTDF_Op<"tiling.reserve_size", [Pure]> {
  let summary = "Create a placeholder for a tile size to be determined by the scheduler";
  let description = [{
    The `ktdf.tiling.reserve_size` operation creates a placeholder value of index 
    type that represents a tile size whose concrete value will be determined by the
    scheduler at a later stage. This operation enables loop tiling transformations
    while deferring the actual tile size selection until scratchpad usage can be
    computed more accurately.

    The operation takes two named attributes:
    - `min_value`: Specifies the minimum acceptable value for the tile size
    - `divisibility`: Specifies a value that the selected tile size must evenly divide

    The scheduler will later replace this placeholder with a concrete tile size value
    that satisfies both constraints and optimizes for scratchpad usage.

    Example:
    ```mlir
    // Create a tile size with minimum value 8 that must divide 4
    %tile_size = ktdf.tiling.reserve_size {min_value = 8 : index, divisibility = 4 : index} : index
    
    // Use in loop tiling
    scf.for %i = %c0 to %N step %tile_size {
      // Loop body
    }
    ```
  }];

  let arguments = (ins
    IndexAttr:$min_value,
    IndexAttr:$divisibility,
    KTDF_UniqueProp:$unique_id
  );

  let results = (outs
    Index:$result
  );

  let assemblyFormat = "attr-dict `:` type($result)";

  let builders = [
    OpBuilder<(ins "int64_t":$minValue, "int64_t":$divisibility), [{
      auto indexType = $_builder.getIndexType();
      $_state.addAttribute(getMinValueAttrName($_state.name),
                          $_builder.getIndexAttr(minValue));
      $_state.addAttribute(getDivisibilityAttrName($_state.name),
                          $_builder.getIndexAttr(divisibility));
      $_state.addTypes(indexType);
    }]>
  ];

  let extraClassDeclaration = [{
    /// Returns the minimum value for the tile size
    int64_t getMinValueInt() {
      return getMinValue().getSExtValue();
    }
    
    /// Returns the divisibility constraint
    int64_t getDivisibilityInt() {
      return getDivisibility().getSExtValue();
    }
  }];
}

def KTDF_TilingDeriveSizeOp : KTDF_Op<"tiling.derive_size", [Pure, SameVariadicOperandSize]> {
  let summary = "Get the effective tile size (trip count) of the current inner tiled loop";
  let description = [{
    The `ktdf.tiling.derive_size` operation returns the number of iterations that
    the current (inner) loop should execute, given the current values of all
    enclosing tile loop induction variables and the original total loop size.

    Each `[%iv : %tile_size]` pair associates a tile loop IV with its tile
    size. Pairs must appear outermost-first; their count determines the tiling
    level. `total_size` is the original loop trip count before any tiling.

    Semantics: the result equals `tile_sizes[last]` (full tile) when all
    enclosing tile loops are in steady-state regime, or the remaining size
    otherwise. Result is 0 when a prior epilogue has already exhausted the
    iteration space.
    A zero result is valid — an `scf.for` with upper bound 0 is a no-op and
    the inner loop simply does not execute.
    Result is always <= the last tile_size operand.

    All IVs must be 0-based, stride-1 normalized iterators.

    Example:
    ```mlir
    %x0 = arith.ceildivui %c23, %c5 : index
    scf.for %i0 = 0 to %x0 step 1 {
      %x1 = ktdf.tiling.derive_size [%i0 : %c5], total_size = %c23 : index
      scf.for %i1 = 0 to %x1 step 1 {
        %x2 = ktdf.tiling.derive_size [%i0 : %c5], [%i1 : %c3], total_size = %c23 : index
        scf.for %i2 = 0 to %x2 step 1 {
          // ...
        }
      }
    }
    ```
  }];

  let skipDefaultBuilders = 1;

  let builders = [
    OpBuilder<(ins "ValueRange":$ivs, "ValueRange":$tile_sizes, "Value":$total_size), [{
      $_state.addOperands(ivs);
      $_state.addOperands(tile_sizes);
      $_state.addOperands(total_size);
      $_state.addTypes($_builder.getIndexType());
    }]>
  ];

  let arguments = (ins
    Variadic<Index>:$ivs,
    Variadic<Index>:$tile_sizes,
    Index:$total_size
  );

  let results = (outs Index:$result);

  let hasVerifier = 1;
  let hasFolder = 1;
  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// Pipeline and Stage Operations
//===----------------------------------------------------------------------===//

def KTDF_PipelineOp : KTDF_Op<"pipeline", [
    NoTerminator,
    SingleBlock
  ]> {
  let summary = "Defines a pipeline containing stages";
  let description = [{
    The `ktdf.pipeline` operation represents a structured directed acyclic graph
    (DAG) of execution steps that process data in stages. Instead of executing one
    step completely before starting the next, pipelines allow different stages to
    operate concurrently on different iterations of work, thereby overlapping
    computation and data movement. In dataflow architectures, pipelines reflect
    how hardware engines operate: one unit may be loading data, another computing,
    and another storing results simultaneously.

    The immediate children of a pipeline must be `ktdf.stage` operations.

    The optional `modulo` attribute declares the pipeline's rolling-window depth.
    With `modulo(size: N)`, the pipeline rotates through `N` physical resource
    sets per `N` consecutive iterations.

    Example:
    ```mlir
    ktdf.pipeline modulo(size: 2) {
      ktdf.stage {
        // Stage 1 operations
      }
      ktdf.stage {
        // Stage 2 operations
      }
    }
    ```
  }];

  let arguments = (ins
    OptionalAttr<ConfinedAttr<I64Attr, [IntMinValue<1>]>>:$modulo
  );

  let regions = (region SizedRegion<1>:$region);

  let assemblyFormat = [{
    (`modulo` `(` `size` `:` $modulo^ `)`)? $region attr-dict
  }];

  let hasRegionVerifier = 1;

  let builders = [
    OpBuilder<(ins CArg<"function_ref<void(OpBuilder&, Location)>", "{}">:$body_builder)>
  ];

  let extraClassDeclaration = [{
    using StageRange = llvm::iterator_range<Block::op_iterator<StageOp>>;

    auto getPrivateOp() -> PrivateOp;

    auto getStages() -> StageRange;
    auto getNumStages() -> unsigned;
  }];
}

def KTDF_StageOp : KTDF_Op<"stage", [
    NoTerminator,
    SingleBlock,
    HasParent<"ktdf::PipelineOp">,
    AttrSizedOperandSegments
  ]> {
  let summary = "Defines a stage within a pipeline";
  let description = [{
    The `ktdf.stage` operation defines a single stage within a pipeline.
    A stage represents a logical step within a pipeline where a specific
    type of work is performed. Each stage "eventually" corresponds to a
    particular hardware capability such as memory transfer or computation.
    Stages form the building blocks of the pipeline and define how work flows
    through the system.

    Stages can specify dataflow dependencies using tokens:
    - `depends_in`: Stages which must be performed before this stage
    - `depends_out`: Stages which must be performed upon completion of current stage

    The immediate parent of a stage is always a `ktdf.pipeline`, and the immediate
    siblings of a stage are other stages.

    Example:
    ```mlir
    %token_a = ktdf.create_token : !ktdf.token
    %token_b = ktdf.create_token : !ktdf.token
    
    ktdf.pipeline {
      ktdf.stage depends_in(%token_a, %token_b) depends_out(none) {
        // Stage operations
      }
      ktdf.stage depends_in(none) depends_out(%token_a) {
        // Stage operations
      }
      ktdf.stage depends_in(none) depends_out(%token_b) {
        // Stage operations
      }
    }
    ```
  }];

  let arguments = (ins
    Variadic<KTDF_TokenType>:$depends_in,
    Variadic<KTDF_TokenType>:$depends_out,
    OptionalAttr<ArrayAttr>:$applicable_units
  );

  let regions = (region SizedRegion<1>:$region);

  let assemblyFormat = [{
    `depends_in` `(` ($depends_in^):(```none`)? `)` 
    `depends_out` `(` ($depends_out^):(```none`)? `)`
    $region
    attr-dict
  }];
  
  let hasVerifier = 1;
  let skipDefaultBuilders = 1;

  let builders = [
    OpBuilder<(ins 
        "ValueRange":$depends_in, 
        "ValueRange":$depends_out, 
        CArg<"function_ref<void(OpBuilder&, Location)>", "{}">:$body_builder)>
  ];
}

//===----------------------------------------------------------------------===//
// Data Transfer Operation
//===----------------------------------------------------------------------===//

def KTDF_DataTransferOp : KTDF_Op<"data_transfer", [
    AllMatchSameOperatorTrait<
      ["source", "destination"], 
      KTDF_ElementType<"_self">.result, 
      "element type">,
    AttrSizedOperandSegments,
    AffineMapAccessInterface
  ]> {
  let summary = "Transfer data between memory locations and FIFO slots";
  let description = [{
    The `ktdf.data_transfer` operation represents explicit data movement in the
    dataflow pipeline. It supports three types of transfers:
    
    1. **memref to memref**: Transfer data between different memory spaces
       (e.g., DDR to L1, L1 to DDR)
    2. **memref to fifo**: Transfer data from memory to a FIFO slot
       (e.g., L1 to FIFO for streaming to compute units)
    3. **fifo to memref**: Transfer data from a FIFO slot to memory
       (e.g., FIFO from compute units to SFU registers or L1)

    The operation explicitly specifies the source and destination, along with
    the size of data being transferred. This makes data movement visible in the
    IR and enables analysis of memory traffic patterns and pipeline behavior.

    ## Syntax

    ```
    data-transfer-op ::= `ktdf.data_transfer`
                         `from` ssa-value (`[` ssa-use-list `]`)?
                         (`size` `[` ssa-use-list `]`)?
                         `to` ssa-value (`[` ssa-use-list `]`)?
                         (`size` `[` ssa-use-list `]`)?
                         `:` type(source) `,` type(destination)
    ```

    ## Examples

    ```mlir
    // memref to memref: DDR to L1
    ktdf.data_transfer from %A[%m, %n] size [1, 64] to %l1_A[%c0] size [64]
      : memref<1x64xf16, "DDR">, memref<64xf16, "L1">

    // memref to fifo: L1 to FIFO slot
    ktdf.data_transfer from %l1_A[%c0] size [64] to %fifo_slot_a size [64]
      : memref<64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64>

    // fifo to memref: FIFO slot to SFU register
    ktdf.data_transfer from %fifo_slot_a to %sfu_A[%c0] size [64]
      : !ktdf.fifo.slot<"L1LU" -> "SFU", 64>, memref<64xf16, "SFU_REG">

    // memref to memref: L1 to DDR
    ktdf.data_transfer from %l1_C[%m1, %n1] size [1, 64] to %C[%m0 + %m1, %n0 + %n1] size [1, 64]
      : memref<?x?xf16, "L1">, memref<?x?xf16, "DDR">
    ```

    ## Semantics

    - The transfer is atomic from the perspective of the stage containing it
    - Source and destination types must be compatible (same element type)
    - For FIFO transfers, the size must not exceed the FIFO slot capacity constraints
    - Indices and sizes are used to specify the exact region being transferred
  }];

  let arguments = (ins
    AnyTypeOf<[AnyMemRef, KTDF_FifoSlotType]>:$source,
    Variadic<Index>:$source_indices,
    Variadic<Index>:$source_sizes,
    AnyTypeOf<[AnyMemRef, KTDF_FifoSlotType]>:$destination,
    Variadic<Index>:$dest_indices,
    Variadic<Index>:$dest_sizes,
    OptionalAttr<DenseI64ArrayAttr>:$static_source_sizes,
    OptionalAttr<DenseI64ArrayAttr>:$static_dest_sizes,
    OptionalAttr<AffineMapAttr>:$source_map,
    OptionalAttr<AffineMapAttr>:$dest_map
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    `from` custom<CellOrSlot>($source, $source_indices, $source_map, $source_sizes, $static_source_sizes) 
    `to` custom<CellOrSlot>($destination, $dest_indices, $dest_map, $dest_sizes, $static_dest_sizes) 
    attr-dict `:` type($source) `,` type($destination)
  }];

  let builders = [
    OpBuilder<(ins "Value":$source, "AffineMap":$source_map,
      "ValueRange":$source_indices,
      "ArrayRef<OpFoldResult>":$source_sizes,
      "Value":$destination, "AffineMap":$dest_map,
      "ValueRange":$dest_indices,
      "ArrayRef<OpFoldResult>":$dest_sizes)>,
    OpBuilder<(ins "Value":$source, "AffineMap":$source_map,
      "ValueRange":$source_indices,
      "ArrayRef<int64_t>":$source_sizes,
      "Value":$destination, "AffineMap":$dest_map,
      "ValueRange":$dest_indices,
      "ArrayRef<int64_t>":$dest_sizes)>
  ];

  let extraClassDeclaration = [{
    /// Returns true if the source is a memref
    bool isSourceMemRef() {
      return llvm::isa<mlir::MemRefType>(getSource().getType());
    }

    /// Returns true if the source is a FIFO slot
    bool isSourceFifo() {
      return llvm::isa<FifoSlotType>(getSource().getType());
    }

    /// Returns true if the destination is a memref
    bool isDestMemRef() {
      return llvm::isa<mlir::MemRefType>(getDestination().getType());
    }

    /// Returns true if the destination is a FIFO slot
    bool isDestFifo() {
      return llvm::isa<FifoSlotType>(getDestination().getType());
    }

    /// Returns the transfer type (memref-to-memref, memref-to-fifo, fifo-to-memref)
    enum class TransferType { MemRefToMemRef, MemRefToFifo, FifoToMemRef };
    
    TransferType getTransferType() {
      if (isSourceMemRef() && isDestMemRef())
        return TransferType::MemRefToMemRef;
      if (isSourceMemRef() && isDestFifo())
        return TransferType::MemRefToFifo;
      return TransferType::FifoToMemRef;
    }
    
    /// Get the mixed source sizes (combining static and dynamic)
    SmallVector<OpFoldResult> getMixedSourceSizes() {
      SmallVector<OpFoldResult> result;
      if (auto staticSizes = getStaticSourceSizes()) {
        unsigned dynamicIdx = 0;
        for (int64_t size : *staticSizes) {
          if (ShapedType::isDynamic(size)) {
            result.push_back(getSourceSizes()[dynamicIdx++]);
          } else {
            result.push_back(IntegerAttr::get(IndexType::get(getContext()), size));
          }
        }
      } else {
        // All dynamic
        for (Value size : getSourceSizes()) {
          result.push_back(size);
        }
      }
      return result;
    }
    
    /// Get the mixed destination sizes (combining static and dynamic)
    SmallVector<OpFoldResult> getMixedDestSizes() {
      SmallVector<OpFoldResult> result;
      if (auto staticSizes = getStaticDestSizes()) {
        unsigned dynamicIdx = 0;
        for (int64_t size : *staticSizes) {
          if (ShapedType::isDynamic(size)) {
            result.push_back(getDestSizes()[dynamicIdx++]);
          } else {
            result.push_back(IntegerAttr::get(IndexType::get(getContext()), size));
          }
        }
      } else {
        // All dynamic
        for (Value size : getDestSizes()) {
          result.push_back(size);
        }
      }
      return result;
    }
    
    /// Returns true if all source sizes are static (compile-time constants)
    bool hasAllStaticSourceSizes() {
      if (!getStaticSourceSizes())
        return false;
      return llvm::none_of(*getStaticSourceSizes(),
                          [](int64_t size) { return ShapedType::isDynamic(size); });
    }
    
    /// Returns true if all destination sizes are static (compile-time constants)
    bool hasAllStaticDestSizes() {
      if (!getStaticDestSizes())
        return false;
      return llvm::none_of(*getStaticDestSizes(),
                          [](int64_t size) { return ShapedType::isDynamic(size); });
    }
    
    /// Get source sizes as static integers. Returns std::nullopt if any size is dynamic.
    std::optional<SmallVector<int64_t>> getStaticSourceSizesArray() {
      if (!hasAllStaticSourceSizes())
        return std::nullopt;
      ArrayRef<int64_t> sizes = *getStaticSourceSizes();
      return SmallVector<int64_t>(sizes.begin(), sizes.end());
    }
    
    /// Get destination sizes as static integers. Returns std::nullopt if any size is dynamic.
    std::optional<SmallVector<int64_t>> getStaticDestSizesArray() {
      if (!hasAllStaticDestSizes())
        return std::nullopt;
      ArrayRef<int64_t> sizes = *getStaticDestSizes();
      return SmallVector<int64_t>(sizes.begin(), sizes.end());
    }

    /// Implements AffineMapAccessInterface.
    /// Returns the AffineMapAttr associated with 'memref'.
    NamedAttribute getAffineMapAttrForMemRef(Value memref);
  }];
}

//===----------------------------------------------------------------------===//
// Private Region Operations
//===----------------------------------------------------------------------===//

def KTDF_PrivateYieldOp : KTDF_Op<"private_yield", [
    Pure,
    ReturnLike,
    Terminator,
    HasParent<"ktdf::PrivateOp">,
    DeclareOpInterfaceMethods<RegionBranchTerminatorOpInterface>
  ]> {
  let summary = "Yield operation for private regions";
  let description = [{
    The `ktdf.private_yield` operation is a terminator for `ktdf.private`
    regions. It yields zero or more values from the private region back to
    the parent operation.
  }];

  let arguments = (ins
    Variadic<AnyType>:$operands
  );

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";

  let builders = [
    OpBuilder<(ins), [{
      build($_builder, $_state, ValueRange{});
    }]>
  ];
}

def KTDF_PrivateOp : KTDF_Op<"private", [
    SingleBlockImplicitTerminator<"ktdf::PrivateYieldOp">,
    HasParent<"ktdf::PipelineOp">,
    DeclareOpInterfaceMethods<RegionBranchOpInterface>
  ]> {
  let summary = "Define a private region with yielded results";
  let description = [{
    The `ktdf.private` operation defines a region yielding resources
    (tokens, memref allocations) private to its parent pipeline operation.
    
    Example:
    ```mlir
    ktdf.pipeline {
      %1:2 = ktdf.private -> (memref<64xf16, "SFU_REG">, !ktdf.token) {
        %sfu_A = memref.alloc() : memref<64xf16, "SFU_REG">
        %t = ktdf.create_token : !ktdf.token
        ktdf.private_yield %sfu_A, %t : memref<64xf16, "SFU_REG">, !ktdf.token
      }

      // Use the token in a stage
      ktdf.stage depends_in(%1#1) depends_out(none) {
        // Stage operations
      }
    }
    ```
  }];

  let results = (outs
    Variadic<AnyType>:$results
  );

  let regions = (region SizedRegion<1>:$region);

  let assemblyFormat = [{
    (`->` `(` type($results)^ `)`)?  $region attr-dict-with-keyword
  }];

  let hasRegionVerifier = 1;

  let builders = [
    OpBuilder<(ins CArg<"::mlir::TypeRange", "{}">:$results), [{
     Region* bodyRegion = $_state.addRegion();
     bodyRegion->emplaceBlock();
     $_state.addTypes(results);
    }]>
  ];

  let extraClassDeclaration = [{
    /// Get the yield operation terminating this private region.
    auto getYieldOp() -> PrivateYieldOp;
  }];
}

//===----------------------------------------------------------------------===//
// Parallel Operation
//===----------------------------------------------------------------------===//

def KTDF_ParallelOp : KTDF_Op<"parallel", [
    SingleBlock,
    SameVariadicOperandSize
  ]> {
  let summary = "N-dim parallel iteration space distributed across hardware instances";
  let description = [{
    The `ktdf.parallel` operation expresses an N-dimensional parallel
    iteration space together with a pure-block distribution of those
    iterations across `num_instances` hardware instances. It is target
    agnostic — it does not say whether the instances are corelets, cores,
    or anything else; lowering binds the distribution axis to a concrete
    hardware axis.

    The body is a single block whose arguments are, in order:
      - N induction variables (the iteration coordinates), all `index`.
      - One trailing instance id, `index`-typed, ranging over
        `[0, num_instances)`.

    The op produces no SSA results. Control flow falls through after every
    instance has finished its assigned chunk (implicit barrier).

    ## Distribution semantics

    Iterations enumerate the Cartesian product
    `[lb_0, ub_0) x ... x [lb_{N-1}, ub_{N-1})` in lex order. With `T`
    total iterations and `N = num_instances`, the lex enumeration is split
    into `N` contiguous chunks of size `ceil(T / N)` (with the last chunk
    holding any leftover). Instance `i` runs chunk `i`. Distribution is
    always pure block — no cyclic or block-cyclic policy.

    ## Example

    ```mlir
    ktdf.parallel (%m, %n, %inst) = (%c0, %c0) to (%c32, %c1) step (%c1, %c1)
                                    distribute(num_instances = 2) {
      // Body uses %m, %n as induction vars and %inst as the instance id.
      ktdf.pipeline { /* ... */ }
      ktdf.parallel_yield
    }
    ```
  }];

  let arguments = (ins
    Variadic<Index>:$lower_bounds,
    Variadic<Index>:$upper_bounds,
    Variadic<Index>:$steps,
    I64Attr:$num_instances
  );

  let regions = (region SizedRegion<1>:$region);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let hasRegionVerifier = 1;
  let skipDefaultBuilders = 1;

  let builders = [
    OpBuilder<(ins
      "ValueRange":$lower_bounds,
      "ValueRange":$upper_bounds,
      "ValueRange":$steps,
      "int64_t":$num_instances,
      CArg<"function_ref<void(OpBuilder&, Location, ValueRange, Value)>", "{}">:$body_builder)>
  ];

  let extraClassDeclaration = [{
    /// Number of induction variables (== rank of the iteration space).
    auto getNumLoops() -> unsigned {
      return static_cast<unsigned>(getLowerBounds().size());
    }

    /// Block argument for induction variable `dim`.
    auto getInductionVar(unsigned dim) -> BlockArgument {
      return getBody()->getArgument(dim);
    }

    /// Block argument for the trailing instance id.
    auto getInstanceId() -> BlockArgument {
      return getBody()->getArgument(getNumLoops());
    }
  }];
}

def KTDF_ParallelYieldOp : KTDF_Op<"parallel_yield", [
    Pure,
    ReturnLike,
    Terminator,
    HasParent<"ParallelOp">
  ]> {
  let summary = "Terminator for ktdf.parallel regions";
  let description = [{
    The `ktdf.parallel_yield` operation is a terminator for `ktdf.parallel`
    regions. It takes no operands and yields no values; the parent
    `ktdf.parallel` op produces no SSA results.

    Example:
    ```mlir
    ktdf.parallel (%m, %inst) = (%c0) to (%c64) step (%c1)
                                distribute(num_instances = 2) {
      ktdf.parallel_yield
    }
    ```
  }];

  let arguments = (ins);

  let assemblyFormat = "attr-dict";
  // No custom builder: the auto-generated `build(OpBuilder&, OperationState&)`
  // is sufficient for a no-operand, no-result op. A previously-defined custom
  // OpBuilder<(ins), [{ build($_builder, $_state); }]> recursed into itself
  // and crashed when invoked via `ParallelYieldOp::create(builder, loc)`.
}

//===----------------------------------------------------------------------===//
// Double-Buffering Operations
//===----------------------------------------------------------------------===//

def KTDF_BufferPhaseOp : KTDF_Op<"buffer_phase", [Pure]> {
  let summary = "Compute the rotation phase for double-buffered memrefs";
  let description = [{
    The `ktdf.buffer_phase` operation produces an `index` value that selects
    which physical buffer a double-buffered (or N-buffered) memref handoff uses
    on the current pipeline iteration. The result is the linearized pipeline
    iteration count modulo `num_phases`.

    The operation takes one or more `index` operands; each operand must be the
    induction variable of an `scf.for` (the verifier enforces this).

    Intended semantics (to be enforced by the lowering pass that consumes
    this op):
    ```
    phase = linearized_pipeline_iteration_count % num_phases
    ```
    where the linearized count is the global iteration index of the pipeline
    over its enclosing loop nest. The lowering pass is responsible for
    deriving each operand IV's bounds and step from its enclosing `scf.for`.

    Example:
    ```mlir
    %phase = ktdf.buffer_phase(%m0, %n0) {num_phases = 2 : i64} : index
    ```
  }];

  let arguments = (ins
    Variadic<Index>:$ivs,
    ConfinedAttr<I64Attr, [IntMinValue<2>]>:$num_phases
  );

  let results = (outs Index:$result);

  let assemblyFormat = [{
    `(` $ivs `)` attr-dict `:` type($result)
  }];

  let hasVerifier = 1;
  let hasFolder = 1;
}

def KTDF_SelectMemrefOp : KTDF_Op<"select_memref", [
    Pure,
    TypesMatchWith<"", "result", "candidates", "$_self", "::allEqualTo">
  ]> {
  let summary = "Select one of N candidate memrefs by index";
  let description = [{
    The `ktdf.select_memref` operation picks one of `N >= 2` candidate
    memrefs based on a phase index produced by `ktdf.buffer_phase`. All
    candidate memrefs must have identical type (element type, shape including
    dynamic dims, and memory space); the result has the same type.

    Semantics:
    ```
    result = candidates[phase]
    ```
    The phase is semantically in `[0, N)` by `ktdf.buffer_phase`'s definition
    (`iteration_count % num_phases`); no wrap is needed in the result selection.

    Example:
    ```mlir
    %X_sel = ktdf.select_memref %phase[%X0, %X1] : memref<?x?xf16, "L1">
    ```
  }];

  let arguments = (ins
    Index:$phase,
    Variadic<AnyMemRef>:$candidates
  );

  let results = (outs AnyMemRef:$result);

  let assemblyFormat = [{
    $phase `[` $candidates `]` attr-dict `:` type($result)
  }];

  let hasVerifier = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Tile Construct Operations
//===----------------------------------------------------------------------===//

def KTDF_TilingLinearizeIndexOp : KTDF_Op<"tiling.linearize_index",
    [Pure, SameVariadicOperandSize]> {
  let summary = "Compute flat index into original iteration space from normalized tile loop IVs";
  let description = [{
    The `ktdf.tiling.linearize_index` operation computes a flat index into the
    original (pre-tiling) iteration space as a weighted sum of all loop IVs
    across all tiling levels including the innermost point loop.

    Each `[%iv : %stride]` pair associates a loop IV with its stride. Pairs
    must appear outermost-first. The last pair is always the point loop with
    stride 1 (explicit).

    Semantics:
    ```
    result = iv[0]*stride[0] + iv[1]*stride[1] + ... + iv[k]*stride[k]
    ```

    All inputs may be SSA values. The result is a pure weighted sum — no
    modulo, no conditionals — and is correct across all steady-state/epilogue
    regime combinations. Once tile sizes are known, this lowers trivially to
    an `affine.apply` or chain of `arith.muli` + `arith.addi`.

    Example:
    ```mlir
    %idx = ktdf.tiling.linearize_index [%i0 : %c5], [%i1 : %c3], [%i2 : %c1] : index
    ```
  }];

  let skipDefaultBuilders = 1;

  let builders = [
    OpBuilder<(ins "ValueRange":$ivs, "ValueRange":$strides), [{
      $_state.addOperands(ivs);
      $_state.addOperands(strides);
      $_state.addTypes($_builder.getIndexType());
    }]>
  ];

  let arguments = (ins
    Variadic<Index>:$ivs,
    Variadic<Index>:$strides
  );

  let results = (outs Index:$result);

  let assemblyFormat = [{
    custom<BracketPairList>($ivs, $strides) attr-dict `:` type($result)
  }];
  let hasVerifier = 1;
}

#endif // DATAFLOW_SCHEDULER_DIALECTS_KTDF_KTDF_TD
