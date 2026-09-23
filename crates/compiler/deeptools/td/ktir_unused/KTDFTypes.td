//===-- KTDFTypes.td ---------------------------------------*- tablegen -*-===//
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
// This file defines the types in the ktdf dialect.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_DIALECT_KTDF_KTDFTYPES_TD
#define DATAFLOW_SCHEDULER_DIALECT_KTDF_KTDFTYPES_TD

include "dataflow-scheduler/Dialect/KTDF/KTDFAttributes.td"

// Base class for KTDF dialect types
class KTDF_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<KTDF_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

class KTDF_ElementType<string name> :
  StrFunc<"mlir::ktdf::getElementTypeOrSelf($" # name # ".getType())">;

//===----------------------------------------------------------------------===//
// TokenType
//===----------------------------------------------------------------------===//

def KTDF_TokenType : KTDF_Type<"Token", "token"> {
  let summary = "KTDF token type for dataflow synchronization";
  let description = [{
    Syntax:

    ```
    token-type ::= `token`
    ```

    The token type represents a synchronization token in the KTDF dialect.
    It is used to model dataflow dependencies and control flow in dataflow
    graphs. Tokens have no runtime representation and are used purely for
    scheduling and dependency tracking.

    Example:

    ```mlir
    // A token value
    %token = ktdf.create_token : !ktdf.token
    ...
    // Token used for synchronization
    ktdf.stage depend_in(%token) ...
    ```
  }];

  let parameters = (ins);

  let assemblyFormat = "";
}

//===----------------------------------------------------------------------===//
// FifoSlotType
//===----------------------------------------------------------------------===//

def KTDF_FifoSlotType : KTDF_Type<"FifoSlot", "fifo.slot"> {
  let summary = "KTDF FIFO slot type for dataflow buffering";
  let description = [{
    Syntax:

    ```
    fifo.slot-type ::= `fifo.slot` `<` src-attr `->` dest-attr `,` shaped-element-type `>`
    src-attr ::= component-type | memory-space-attr
    dest-attr ::= component-type | memory-space-attr
    memory-space-attr ::= `#` dialect-namespace `.` attr-name `<` ... `>`
    shaped-element-type ::= (num-elements `x`)? element-type
    num-elements ::= `?` | decimal-literal
    element-type ::= type
    ```

    The fifo.slot type represents a slot in a FIFO buffer used for dataflow
    communication between stages. Each slot has:
    - A source attribute indicating where data flows from (component type or memory space)
    - A destination attribute indicating where data flows to (component type or memory space)
    - A fifo slot capacity in terms of number of elements that can be static or dynamic
    - An element type that specifies the type of data being transferred

    The source and destination can be component types or memory space attributes.
    The number of elements can be:
    - A static positive integer (e.g., 4, 8, 16, 64)
    - Dynamic, indicated by `?` (similar to memref dimensions)

    Example:

    ```mlir
    // A FIFO slot from L1LU to SFU with 64 f16 elements
    %slot0 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<L1LU -> SFU, 64xf16>
    
    // A FIFO slot from SFU to L1SU with some dynamic number of f16 elements
    %size = arith.constant 8 : index
    %slot1 = ktdf.fifo.allocate(%size) -> !ktdf.fifo.slot<SFU -> L1SU, ?xf16>

    // A FIFO slot from DDR memory space to SFU
    %slot2 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<#ktdp.spyre_memory_space<DDR> -> SFU, 64xf16>
    ```
  }];

  let parameters = (ins
    "Attribute":$src,
    "Attribute":$dest,
    "int64_t":$numElements,
    "Type":$elementType
  );

  let genStorageClass = 1;
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    /// Returns true if the number of elements is dynamic
    bool isDynamicNumElements() const {
      return mlir::ShapedType::isDynamic(getNumElements());
    }
    
    /// Returns the static number of elements, or kDynamic if dynamic
    int64_t getStaticNumElements() const { return getNumElements(); }
  }];
}

#endif // DATAFLOW_SCHEDULER_DIALECT_KTDF_KTDFTYPES_TD