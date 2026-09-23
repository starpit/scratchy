//===-- Uniform.td ------------------------------------------*- tablegen -*-==//
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
// This file defines the uniform dialect.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_DIALECT_UNIFORM_UNIFORM_TD
#define DATAFLOW_SCHEDULER_DIALECT_UNIFORM_UNIFORM_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

def Uniform_Dialect : Dialect {
  let name = "uniform";
  let cppNamespace = "::mlir::uniform";

  let dependentDialects = [
    "dataflow::DataflowDialect"
  ];
}

//===----------------------------------------------------------------------===//
// Operations
//===----------------------------------------------------------------------===//

// FIXME: A lot weird ODS going on here.

// Base class for Uniform dialect operations. This operation inherits from the
// base `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class Uniform_Op<string mnemonic, list<Trait> traits = []> :
    Op<Uniform_Dialect, mnemonic, traits> {

}

// Require regions to have uniform.yield.
def ImplicitUniformTerminator
    : SingleBlockImplicitTerminator<"YieldOp">;

def YieldOp : Uniform_Op<"yield", [NoMemoryEffect, Terminator, ReturnLike,
    MemRefsNormalizable]> {
  let summary = "Yield values to parent operation";
  let description = [{
    `uniform.yield` yields zero or more SSA values to its parent operation. It
    is the implicit terminator for regions in the uniform dialect.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);
  let builders = [
    OpBuilder<(ins), [{ build($_builder, $_state, ValueRange{}); }]>
  ];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

def UniformizeRegionsOp : Uniform_Op<"uniformize_regions",
      [ImplicitUniformTerminator]> {
  let summary = "uniformize_regions operation";
  let description = [{
    This is the main operation of the uniform dialect to map parallel regions to different units, aiming to consolidate programs from multiple units into a single program.
  }];

  let arguments = (ins Variadic<AnyType>:$units,
                       I32ArrayAttr:$list_sizes,
                       OptionalAttr<I32ArrayAttr>:$regIndices,
                       OptionalAttr<ArrayAttr>:$regLocales);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region VariadicRegion<AnyRegion>:$regions);

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    Value getRegionArg(int i) {return getRegion(i).getArgument(0);}
    ValueRange getRegionUnitList(int);
    std::optional<ValueRange> getRegionUnitList(Value);
    std::optional<Region *> getSpecificRegion(Value);
    Region *getRegionFromUnit(Value);

    ArrayAttr getRegIndicesIfExist() {
      if (getRegIndices().has_value()) return getRegIndices().value();
      return nullptr;
    }
    ArrayAttr getRegLocalesIfExist() {
      if (getRegLocales().has_value()) return getRegLocales().value();
      return nullptr;
    }
  }];

  let hasVerifier = 1;
}

def DefImmutableMappingOp : Uniform_Op<"def_immutable_mapping",
  [NoMemoryEffect, IsolatedFromAbove, AttrSizedOperandSegments]> {
  let summary = "def_immutable_mapping operation";
  let description = [{
    Constructs an immutable key-value mapping from one set of SSA values to
    another and returns a handle to it. It is used to capture *constant*
    differences between the per-core programs of a unit (for example differing
    loop bounds or addresses): each core's value is associated with a common
    key, so the program can be expressed once and specialized per core via
    `uniform.query_map`. The `keys` and `values` operands are paired
    positionally.

    ```mlir
    %map = uniform.def_immutable_mapping([%8 -> %0], [%9 -> %0]) : index
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$keys,
                       Variadic<AnyType>:$values);
  let results = (outs Index:$result);

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    /// Returns the value mapped to \p key, or std::nullopt if absent.
    std::optional<Value> getValue(Value key);

    llvm::DenseMap<Value, Value> toMap();

    void getValuesFromKeys(ValueRange keys, SmallVectorImpl<std::optional<Value>> &result);
    SmallVector<std::optional<Value>> getValuesFromKeys(ValueRange keys) {
      SmallVector<std::optional<Value>> result;
      getValuesFromKeys(keys, result);
      return result;
    }

    void getNonNullValuesFromKeys(ValueRange keys, SmallVectorImpl<Value> &result);
    SmallVector<Value> getNonNullValuesFromKeys(ValueRange keys) {
      SmallVector<Value> result;
      getNonNullValuesFromKeys(keys, result);
      return result;
    }
  }];

  let hasVerifier = 1;
}

def QueryMapOp : Uniform_Op<"query_map",
  [NoMemoryEffect, IsolatedFromAbove]> {
  let summary = "query_map operation";
  let description = [{
    Queries a mapping produced by `uniform.def_immutable_mapping`. Given the map
    handle and a key, it returns the value associated with that key, allowing a
    single uniform program to be specialized to a particular core's constants.

    ```mlir
    %v = uniform.query_map(map: %map, key: %arg0) : index
    ```
  }];

  let arguments = (ins Index:$map,
                       Index:$key);
  let results = (outs AnyType:$result);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    void getAllQueriedValues(SmallVectorImpl<Value> &result);
  }];
}

def EqualizePatternOp : Uniform_Op<"equalize_pattern",
      [ImplicitUniformTerminator]> {
  let summary = "equalize_pattern operation";
  let description = [{
    This operation captures the variations in transfer or compute patterns
    between units and encapsulates them within parallel regions.
    All regions must exhibit an identical pattern of operation sequence.
  }];

  let arguments = (ins Variadic<AnyType>:$units,
                       I32ArrayAttr:$list_sizes);
  let regions = (region VariadicRegion<AnyRegion>:$regions);

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    Value getRegionArg(int i) {return getRegion(i).getArgument(0);}
    ValueRange getRegionUnitList(int);
    std::optional<ValueRange> getRegionUnitList(Value);
    std::optional<Region *> getSpecificRegion(Value);
    Region *getRegionFromUnit(Value unit);
  }];

  let hasVerifier = 1;
}

#endif // DATAFLOW_SCHEDULER_DIALECT_UNIFORM_UNIFORM_TD
