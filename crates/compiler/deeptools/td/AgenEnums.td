//===-- AgenEnums.td ----------------------------------------*- tablegen -*-==//
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
// This file defines the enumerations in the agen dialect.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_DIALECT_AGEN_AGENENUMS_TD
#define DATAFLOW_SCHEDULER_DIALECT_AGEN_AGENENUMS_TD

include "mlir/IR/EnumAttr.td"

// FIXME: Why don't we use dataflow::DataflowRoutingDirection?

def AgenRoutingDirPseudoRandom : I32EnumAttrCase<"PseudoRandom", 0>;
def AgenRoutingDirCounterClockwise : I32EnumAttrCase<"CounterClockwise", 1>;
def AgenRoutingDirClockwise : I32EnumAttrCase<"Clockwise", 2>;
def AgenRoutingDirBoth : I32EnumAttrCase<"BothWays", 3>;

def AgenRoutingDirection : I32EnumAttr<"AgenRoutingDirection",
    "Agen routing direction:",
    [
      AgenRoutingDirPseudoRandom,
      AgenRoutingDirCounterClockwise,
      AgenRoutingDirClockwise,
      AgenRoutingDirBoth
    ]> {
    let genSpecializedAttr = 0;
    let cppNamespace = "::mlir::agen";
}

#endif // DATAFLOW_SCHEDULER_DIALECT_AGEN_AGENENUMS_TD
