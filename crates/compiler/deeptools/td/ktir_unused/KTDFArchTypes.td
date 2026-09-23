//===-- KTDFArchTypes.td -----------------------------------*- tablegen -*-===//
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
// This file defines the types in the ktdf_arch dialect.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_DIALECT_KTDFARCH_KTDFARCHTYPES_TD
#define DATAFLOW_SCHEDULER_DIALECT_KTDFARCH_KTDFARCHTYPES_TD

include "mlir/IR/OpBase.td"

include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchAttributes.td"

class KTDFArch_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<KTDFArch_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def KTDFArch_MemoryType : KTDFArch_Type<"Memory", "memory"> {
  let summary = "Type of a memory resource.";
}

def KTDFArch_ExecutionUnitType : KTDFArch_Type<"ExecutionUnit", "exec_unit"> {
  let summary = "Type of an execution unit resource.";
}

def KTDFArch_PortType : KTDFArch_Type<"Port", "port"> {
  let summary = "Type of a switch port resource.";
}

def KTDFArch_EndpointType : AnyTypeOf<[KTDFArch_MemoryType, KTDFArch_ExecutionUnitType, KTDFArch_PortType]>;

#endif // DATAFLOW_SCHEDULER_DIALECT_KTDFARCH_KTDFARCHTYPES_TD