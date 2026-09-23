//===-- Symbol.td -------------------------------------------*- tablegen -*-==//
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
// This file defines the symbol dialect.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_DIALECT_SYMBOL_SYMBOL_TD
#define DATAFLOW_SCHEDULER_DIALECT_SYMBOL_SYMBOL_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "dataflow-scheduler/Dialect/OpTraits.td"

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

def Symbol_Dialect : Dialect {
  let name = "symbol";
  let cppNamespace = "::mlir::symbol";
}

//===----------------------------------------------------------------------===//
// Operations
//===----------------------------------------------------------------------===//

// Base class for Symbol dialect operations. This operation inherits from the
// base `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class Symbol_Op<string mnemonic, list<Trait> traits = []> :
    Op<Symbol_Dialect, mnemonic, traits> {
}

def Symbol_CreateSymbolOp : Symbol_Op<"create_symbol",
  [NoMemoryEffect, ScalarValue]> {
    let summary = "Creates a single symbol";
    let description = [{
    "This operation creates a single symbol."
    ```mlir
        %symbol = symbol.create_symbol : index
    ```
    }];

    let results = (outs Index:$result);
    let assemblyFormat = [{
      attr-dict `:` type(results)
    }];
    let extraClassDeclaration = [{
    int64_t getSymbolID() { return mlir::cast<mlir::IntegerAttr>(getOperation()->getAttr("SymbolId")).getInt(); }
  }];
}

def Symbol_CreateIdOp : Symbol_Op<"create_id",
  [MemoryEffects<[MemWrite<DefaultResource>]>]> {
    let summary = "Associates a symbol ID with a value";
    let description = [{
    "This operation takes a single operand of any type and attaches a mandatory
    integer symbol ID to it."
    ```mlir
        symbol.create_id %value { symbol_id = -100 : i64 } : <type>
    ```
    }];

    let arguments = (ins AnyType:$input,
                         SI64Attr:$symbol_id);
    let assemblyFormat = [{
      $input attr-dict `:` type($input)
    }];
}

def SymbolImmutableMappingOp : Symbol_Op<"symbol_immutable_mapping",
  [NoMemoryEffect, IsolatedFromAbove, SameVariadicOperandSize]> {
  let summary = "symbol_immutable_mapping operation";
  let description = [{
    This operation constructs a key-value dictionary to map arith.constant SSA values to symbols.
  }];

  let arguments = (ins Variadic<Index>:$keys,
                       Variadic<Index>:$values);
  let results = (outs Index:$result);

  let assemblyFormat = [{
    custom<ImmutableMap>($keys, $values) attr-dict `:` type($result)
  }];

  let hasVerifier = 1;
}

def SymbolQueryMapOp : Symbol_Op<"query_map",
  [NoMemoryEffect, IsolatedFromAbove]> {
  let summary = "symbol query_map operation";
  let description = [{
    This operation querries a symbol_immutable_mapping based on an iter arg key.
  }];

  let arguments = (ins Index:$map,
                       Index:$key);
  let results = (outs Index:$result);

  let assemblyFormat = [{
    `(````map````:```$map `,` `key````:```$key ```)` attr-dict `:` type($result)
  }];

  let hasVerifier = 1;
}

#endif  // DATAFLOW_SCHEDULER_DIALECT_SYMBOL_SYMBOL_TD
