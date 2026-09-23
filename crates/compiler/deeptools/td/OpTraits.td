//===-- OpTraits.td -----------------------------------------*- tablegen -*-==//
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
// This file defines some shared operation traits.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_DIALECT_OPTRAITS_TD
#define DATAFLOW_SCHEDULER_DIALECT_OPTRAITS_TD

include "mlir/IR/OpBase.td"

class SchedulerOpTrait<string name, list<Trait> traits = []> 
    : NativeOpTrait<name, traits> {
  let cppNamespace = "::scheduler::OpTrait";
}

def ScalarValue : SchedulerOpTrait<"ScalarValue">;

#endif  // DATAFLOW_SCHEDULER_DIALECT_OPTRAITS_TD
