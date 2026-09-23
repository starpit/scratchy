//===-- DataflowInterfaces.td ------------------------------*- tablegen -*-===//
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

#ifndef DATAFLOW_SCHEDULER_DIALECT_DATAFLOW_DATAFLOWINTERFACES_TD
#define DATAFLOW_SCHEDULER_DIALECT_DATAFLOW_DATAFLOWINTERFACES_TD

include "mlir/IR/OpBase.td"

def Dataflow_DebugNameOpInterface : OpInterface<"DebugNameOpInterface"> {
  let cppNamespace = "::mlir::dataflow";
  let description = [{
    Interface for an operation with an inherent `dbgName` attribute.
  }];

  let methods = [
    InterfaceMethod<[{
        Gets the optional debug name.
      }],
      "std::optional<StringRef>", "getDbgName", (ins),
      "",
      "">,
    InterfaceMethod<[{
        Gets the optional debug name attribute.
      }],
      "StringAttr", "getDbgNameAttr", (ins),
      "",
      "">,
    InterfaceMethod<[{
        Sets the optional debug name.
      }],
      "void", "setDbgName", (ins
        "std::optional<StringRef>":$name),
      "",
      "">,
    InterfaceMethod<[{
        Sets the optional debug name attribute.
      }],
      "void", "setDbgNameAttr", (ins "StringAttr":$name),
      "",
      "">
  ];

  let verify = [{
    static_assert(
        ::mlir::dataflow::detail::has_dbgName_property_v<ConcreteOp>,
        "op must declare an inherent attribute with name `dbgName`");

    return ::mlir::success();
  }];

  let extraClassDeclaration = [{
    /// The canonical, required name of the `dbgName` attribute.
    constexpr const static ::llvm::StringLiteral kDbgNameAttrName = "dbgName";
  }];
}

#endif // DATAFLOW_SCHEDULER_DIALECT_DATAFLOW_DATAFLOWINTERFACES_TD
