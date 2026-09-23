//===-- DataflowTypes.td ----------------------------------------*- c++ -*-===//
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

#ifndef DATAFLOW_SCHEDULER_DIALECT_DATAFLOW_DATAFLOWTYPES_TD
#define DATAFLOW_SCHEDULER_DIALECT_DATAFLOW_DATAFLOWTYPES_TD

include "dataflow-scheduler/Dialect/Dataflow/DataflowAttributes.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

class Dataflow_Type<string name, string typeMnemonic, list<Trait> traits = []>
        : TypeDef<Dataflow_Dialect, name, traits> {
    let mnemonic = typeMnemonic;
}

def Dataflow_CustomVectorType : Dataflow_Type<"CustomVector", "custom_vector"> {
    let summary = "Dataflow custom vector type";

    let parameters = (ins "Type":$element_type, "int64_t":$num_elements);

    let hasCustomAssemblyFormat = 1;

    let builders = [
        TypeBuilderWithInferredContext<(ins "Type":$element_type, "int64_t":$num_elements), [{
            return $_get(element_type.getContext(), element_type, num_elements);
        }]>
    ];
}

def Dataflow_CustomMXFloatType : Dataflow_Type<"CustomMXFloat", "mxfp", [
        MemRefElementTypeInterface]> {
    let summary = "Dataflow custom MX float type";

    let parameters = (ins "unsigned":$width);

    let assemblyFormat = "`<` $width `>`";

    let extraClassDeclaration = [{
        [[nodiscard]] auto getIntOrFloatBitWidth() const -> unsigned {
            return getWidth();
        }
    }];
}

def Dataflow_CustomMXIntType : Dataflow_Type<"CustomMXInt", "mxint", [
        MemRefElementTypeInterface]> {
    let summary = "Dataflow custom MX float type";

    let parameters = (ins "unsigned":$width);

    let assemblyFormat = "`<` $width `>`";

    let extraClassDeclaration = [{
        [[nodiscard]] auto getIntOrFloatBitWidth() const -> unsigned {
            return getWidth();
        }
    }];
}

#endif // DATAFLOW_SCHEDULER_DIALECT_DATAFLOW_DATAFLOWTYPES_TD
