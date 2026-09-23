//===-- KTDFLowering.td ------------------------------------*- tablegen -*-===//
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
// This file defines KTDFLowering operations in the MLIR operation set.
// These are intermediate operations used only during the lowering from
// KTIR into Dataflow IR.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_DIALECT_KTDFLOWERING_KTDFLOWERING_TD
#define DATAFLOW_SCHEDULER_DIALECT_KTDFLOWERING_KTDFLOWERING_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

include "dataflow-scheduler/Dialect/KTDFLowering/KTDFLoweringDialect.td"
include "dataflow-scheduler/Dialect/KTDF/KTDFEnums.td"
include "dataflow-scheduler/Dialect/KTDF/KTDFAttributes.td"

class KTDFLowering_Op<string mnemonic, list<Trait> traits = []> :
    Op<KTDFLowering_Dialect, mnemonic, traits> {
}

def KTDFLowering_ExecuteOnOp : KTDFLowering_Op<"execute_on", [
    NoTerminator,
    SingleBlock
  ]> {
  let summary = "Represents execution on one or more hardware units with a body region";
  let description = [{
    The `ktdf_lowering.execute_on` operation represents execution on one or more
    hardware units. It wraps a body region containing operations that should be
    executed on the specified units.

    This operation takes index-typed SSA values as operands representing the units
    to execute on. These are typically results from dataflow.get_unit operations
    and uniform.query_map results in Phase 1.5.

    This is a temporary operation used during the lowering process from KTIR to
    Dataflow IR. In Phase 2, pipelines and stages are replaced with execute_on
    operations. In Phase 3, these are used to create separate program_unit
    operations for each unit type.

    Example:
    ```mlir
    // Single unit operand
    ktdf_lowering.execute_on(%sfu_unit) {
      %result = arith.addf %a, %b : f32
    }

    // Multiple unit operands
    ktdf_lowering.execute_on(%mnilu_unit, %mnisu_unit) {
      // Operations for multiple units
    }
    ```
  }];

  let arguments = (ins
    Variadic<Index>:$units
  );

  let regions = (region SizedRegion<1>:$body);

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    /// Get the block containing the body.
    Block *getBodyBlock() { return &getBody().front(); }

    /// Get the number of unit operands
    size_t getNumUnits() { return getUnits().size(); }
  }];
}

def KTDFLowering_SignalOp : KTDFLowering_Op<"signal", []> {
  let summary = "Synchronization signal between units";
  let description = [{
    The `ktdf_lowering.signal` operation represents a synchronization point
    between hardware units. It takes index-typed SSA values as operands representing
    the units that need to synchronize.

    This is a temporary operation used during the lowering process. In Phase 2,
    signals are inserted between stages based on their dependencies and scratchpad
    conflicts. In Phase 3, they are lowered to dataflow.sync_send/recv operations.

    This operation has side effects to prevent DCE (Dead Code Elimination) from
    removing it before it can be lowered to sync operations.

    Example:
    ```mlir
    // Signal between MNILU and L1LU
    ktdf_lowering.signal(%mnilu_unit, %l1lu_unit)

    // Signal between L1LU and SFU
    ktdf_lowering.signal(%l1lu_unit, %sfu_unit)
    ```
  }];

  let arguments = (ins
    Variadic<Index>:$units
  );

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    /// Get the number of unit operands
    size_t getNumUnits() { return getUnits().size(); }
  }];
}

#endif // DATAFLOW_SCHEDULER_DIALECT_KTDFLOWERING_KTDFLOWERING_TD