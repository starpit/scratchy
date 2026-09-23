//===-- KTDFAttributes.td ----------------------------------*- tablegen -*-===//
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
// This file defines the attributes in the ktdf dialect.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_DIALECT_KTDF_KTDFATTRIBUTES_TD
#define DATAFLOW_SCHEDULER_DIALECT_KTDF_KTDFATTRIBUTES_TD

include "dataflow-scheduler/Dialect/KTDF/KTDFDialect.td"
include "dataflow-scheduler/Dialect/KTDF/KTDFEnums.td"

// Helper that allows defining unique operations.
def KTDF_UniqueProp : Property<"::mlir::ktdf::UniqueProp"> {
  let defaultValue = "::mlir::ktdf::UniqueProp()";
  let convertToAttribute = "return $_storage.asAttr($_ctxt);";
  let convertFromAttribute = [{
    return ::mlir::ktdf::UniqueProp::setFromAttr($_storage, $_attr, $_diag);
  }];
  let writeToMlirBytecode = "$_storage.write($_writer)";
  let readFromMlirBytecode = [{
    return ::mlir::ktdf::UniqueProp::read($_reader, $_storage);
  }];
}

// Base class for KTDF dialect attributes
class KTDF_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<KTDF_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

//===----------------------------------------------------------------------===//
// LoopTypeAttr
//===----------------------------------------------------------------------===//

def KTDF_LoopTypeAttr : EnumAttr<KTDF_Dialect, KTDF_LoopTypeEnum, "loop_type"> {
  let summary = "KTDF loop type attribute for classifying loops";
  let description = [{
    The loop type attribute provides an enum-backed classification for loops,
    distinguishing between parallel and reduction loops.

    Example:

    ```mlir
    scf.for %i = %c0 to %c10 step %c1 {loop_type = #ktdf.loop_type<parallel_loop>} {
      // parallel loop body
    }
    ```
  }];
  let assemblyFormat = "`<` $value `>`";
}

#endif // DATAFLOW_SCHEDULER_DIALECT_KTDF_KTDFATTRIBUTES_TD
