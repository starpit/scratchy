//===-- KTDFEnums.td ---------------------------------------*- tablegen -*-===//
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
// This file defines the enumerations in the ktdf dialect.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_DIALECT_KTDF_KTDFENUMS_TD
#define DATAFLOW_SCHEDULER_DIALECT_KTDF_KTDFENUMS_TD

include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// Loop Type Enum
//===----------------------------------------------------------------------===//

def KTDF_LoopTypeEnum : I32EnumAttr<"LoopType",
    "Loop type classification",
    [
      I32EnumAttrCase<"ParallelLoop", 0, "parallel_loop">,
      I32EnumAttrCase<"ReductionLoop", 1, "reduction_loop">
    ]> {
  let cppNamespace = "::mlir::ktdf";
  let genSpecializedAttr = 0;
}

#endif // DATAFLOW_SCHEDULER_DIALECT_KTDF_KTDFENUMS_TD
