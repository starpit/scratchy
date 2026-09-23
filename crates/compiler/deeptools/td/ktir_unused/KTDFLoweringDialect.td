//===-- KTDFLoweringDialect.td -----------------------------*- tablegen -*-===//
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
// This file defines the KTDFLowering dialect for temporary operations during
// KTIR to Dataflow IR lowering.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_DIALECT_KTDFLOWERING_KTDFLOWERINGDIALECT_TD
#define DATAFLOW_SCHEDULER_DIALECT_KTDFLOWERING_KTDFLOWERINGDIALECT_TD

include "mlir/IR/OpBase.td"

def KTDFLowering_Dialect : Dialect {
  let name = "ktdf_lowering";
  let cppNamespace = "::mlir::ktdf_lowering";

  let description = [{
    The KTDFLowering dialect provides temporary operations used during the
    lowering process from KTIR (KTDF dialect) to Dataflow IR. These operations
    are intermediate representations that facilitate the multi-phase lowering
    transformation.
    
    Phase 1: Stage-to-KTDF_Lowering
      - Replaces stages/pipelines with execute_on operations
      - Introduces synchronization signals
    
    Phase 2: Program Unit Creation
      - Creates dataflow.program_unit for each unit type
      - Filters execute_on operations per unit
    
    Phase 3: DFIR Lowering
      - Lowers ktdf_lowering operations to final dataflow IR
  }];

  let extraClassDeclaration = [{
    void registerOps();
  }];
}

#endif // DATAFLOW_SCHEDULER_DIALECT_KTDFLOWERING_KTDFLOWERINGDIALECT_TD