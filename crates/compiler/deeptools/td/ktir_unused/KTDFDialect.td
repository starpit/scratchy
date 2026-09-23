//===-- KTDFDialect.td -------------------------------------*- tablegen -*-===//
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
// This file defines the KTDF (Kernel Tiled Data Flow) dialect.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_DIALECT_KTDF_KTDFDIALECT_TD
#define DATAFLOW_SCHEDULER_DIALECT_KTDF_KTDFDIALECT_TD

include "mlir/IR/OpBase.td"

def KTDF_Dialect : Dialect {
  let name = "ktdf";
  let cppNamespace = "::mlir::ktdf";
  
  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;

  let dependentDialects = [
    "scf::SCFDialect",
    "arith::ArithDialect"
  ];

  let hasConstantMaterializer = 1;

  let extraClassDeclaration = [{
    void registerOps();
    void registerTypes();
    void registerAttributes();
  }];
}

#endif // DATAFLOW_SCHEDULER_DIALECT_KTDF_KTDFDIALECT_TD
