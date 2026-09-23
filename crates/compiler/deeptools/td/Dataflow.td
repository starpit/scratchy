//===-- Dataflow.td -----------------------------------------*- tablegen -*-==//
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
// This file defines the dataflow dialect attributes and operations.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_DIALECT_DATAFLOW_DATAFLOW_TD
#define DATAFLOW_SCHEDULER_DIALECT_DATAFLOW_DATAFLOW_TD

include "mlir/Interfaces/SideEffectInterfaces.td"

include "dataflow-scheduler/Dialect/Dataflow/DataflowInterfaces.td"
include "dataflow-scheduler/Dialect/Dataflow/DataflowTypes.td"

// Base class for KTDF dialect operations
class Dataflow_Op<string mnemonic, list<Trait> traits = []> :
    Op<Dataflow_Dialect, mnemonic, traits> {
}

// FIXME: A lot weird ODS going on here.

def Dataflow_GetUnitOp:Dataflow_Op<"get_unit",
  [NoMemoryEffect, IsolatedFromAbove]> {
    let summary = "Getting a handle to a global unit";
    let description = [{
    Returns an SSA handle corresponding to a hardware unit. The unit is
    identified by its attributes - typically `core`, `corelet`, `type`, and
    `name`. The handle is then used by other dataflow operations (such as
    `program_unit`, `send`, `receive`, and `get_logical_memory_view`) to refer
    to that unit.

    A `get_unit` may produce multiple results, one per program time step.

    ```mlir
    %l1lu = dataflow.get_unit {core = 0, corelet = 0, type = "L1LU", name = "L1-LU"} : index
    ```
    }];
    let arguments = (ins StrAttr:$name, StrAttr:$type);

    //Variadic has been added to represent multiple return values of GetUnit operation with each return
    //value corresponding to an instance of program time steps.
    let results = (outs Variadic<Index>:$units);
    let assemblyFormat = [{
        attr-dict `:` type(results)
    }];

    let hasVerifier = 1;

    let extraClassDeclaration = [{
      Value getUnit();
    }];
}

def Dataflow_GetLocalUnitOp:Dataflow_Op<"get_local_unit",
  [NoMemoryEffect, IsolatedFromAbove]> {
    let summary = "Getting a handle to a local unit";
    let description = [{
    "This operation gets a handle to a local unit for operations on the unit."
    ```mlir
        %unit = dataflow.get_local_unit {name="L1-LU"} : Index
    ```
    }];
    let arguments = (ins Index:$unitId, StrAttr:$name);
    let results = (outs Index:$unit);
    let assemblyFormat = [{
         $unitId attr-dict `:` type(results)
    }];
}

def Dataflow_ProgramUnitOp:Dataflow_Op<"program_unit",
  [SingleBlockImplicitTerminator<"ReturnOp">]> {
    let summary = "Definition of a unit";
    let description = [{
    Holds the program associated with one or more units. The region carries the
    body of work executed on the given unit(s) and is terminated by a
    `dataflow.return`. The optional `precision` attribute records the numeric
    precision of the unit's program.

    ```mlir
    dataflow.program_unit (%l1lu) { #region } : index
    ```

    Multiple units can be bound through iter-arg style operands, one per program
    time step:

    ```mlir
    dataflow.program_unit iter_arg : %unit -> (%l1lu0, %l1lu1) { #region } : index, index
    ```
    }];

    let arguments = (ins Variadic<Index>:$units, OptionalAttr<StrAttr>:$precision);
    let regions = (region SizedRegion<1>:$region);
    let skipDefaultBuilders = 1;
    let builders = [
    OpBuilder<(ins "ValueRange":$units,
      CArg<"function_ref<void(OpBuilder &, Location, ValueRange)>",
           "nullptr">:$bodyBuilder)>,
      OpBuilder<(ins "ValueRange":$units, "StringAttr":$precision,
            CArg<"function_ref<void(OpBuilder &, Location, ValueRange)>",
                 "nullptr">:$bodyBuilder)>
    ];

    let extraClassDeclaration = [{
    using BodyBuilderFn =
        function_ref<void(OpBuilder &, Location, ValueRange)>;
    }];

    let hasCustomAssemblyFormat = 1;
}

def Dataflow_ReturnOp : Dataflow_Op<"return", [Terminator]> {
  let summary = [{
    The `return` operation terminates the definition.
  }];
  let description = [{
    The `return` operation is the terminator of a `dataflow.program_unit`
    region. It marks the end of the unit definition and carries no operands.
    It is the implicit terminator for that region, so it is normally created
    automatically rather than written by hand.
  }];
}

def Dataflow_CreateGroupOp:Dataflow_Op<"create_group",
  [NoMemoryEffect, IsolatedFromAbove]> {
    let summary = "Creates a group of unit IDs";
    let description = [{
    Creates a logical group of units from the given unit handles and returns a
    single group handle. The group can then be used wherever a collective of
    units is addressed (for example as a multicast destination).

    ```mlir
    %group = dataflow.create_group (%l1lu0 : index, %l1lu1 : index) : index
    ```
    }];

    let arguments = (ins Variadic<Index>:$unit_ids);
    let results = (outs Index:$group_id);
    let assemblyFormat = [{
       `(` ($unit_ids^ `:` type($unit_ids))? `)`  attr-dict `:` type(results)
    }];
}

def Dataflow_CreateMulticastGroupOp:Dataflow_Op<"create_multicast_group",
  [NoMemoryEffect, IsolatedFromAbove]> {
    let summary = "Creates a multicast group.";
    let description = [{
    "This operation creats a multicast group which specifies the unit ID of the
     producer of data, list of unit IDs of the consumers, and additional
     information such as `group_id`, `count` and the number of sharers. Note that the
     list of consumers could be left empty, however when it is not empty, the
     size of that list must match the `num_consumers` (number of sharers) field.
     `count` field specifies the initial number of outstanding requests."
    ```mlir
        %group = dataflow.create_multicast_group (%1 -> (%2, %3, %4)) {count = 0 : i32, group_id = 1 : i32, num_consumers = 2 : i32} : index
    ```
    }];

    let arguments = (ins Index:$producer,
                     Variadic<Index>:$consumers,
                     I32Attr:$num_consumers,
                     I32Attr:$group_id,
                     I32Attr:$count
                     );
    let results = (outs Index:$result);
    let assemblyFormat = [{
       `(` $producer `->` `(` ($consumers^)? `)` `)`  attr-dict `:` type(results)
    }];
}

def Dataflow_SyncSendOp:Dataflow_Op<"sync_send",
  [Dataflow_DebugNameOpInterface]> {
    let summary = "Send synchronization signal for explicit synchronization";
    let description = [{
    Sends a synchronization signal to another unit, used for explicit
    synchronization between units. The optional
    `wait_immediately_for_async_transfers` attribute controls whether the sender
    waits for outstanding asynchronous transfers before signalling.

    ```mlir
    dataflow.sync_send %l1su0 {wait_immediately_for_async_transfers = false} : index
    ```
    }];
    let arguments = (ins Index:$to_unit, OptionalAttr<StrAttr>:$dbgName,
                         OptionalAttr<BoolAttr>:$wait_immediately_for_async_transfers);
    let assemblyFormat = [{
         $to_unit attr-dict `:` type($to_unit)
    }];
}

def Dataflow_SyncRecvOp:Dataflow_Op<"sync_recv",
  [Dataflow_DebugNameOpInterface]> {
    let summary = "Receive synchronization signal for explicit synchronization";
    let description = [{
    Receives a synchronization signal from another unit, used for explicit
    synchronization between units. This is a blocking call: it does not return
    until the matching signal has been received.

    ```mlir
    dataflow.sync_recv %l1su0 : index
    ```
    }];
    let arguments = (ins Index:$from_unit,
                         OptionalAttr<StrAttr>:$dbgName);
    let assemblyFormat = [{
         $from_unit  attr-dict `:` type($from_unit)
    }];
}

def Dataflow_ImplicitSyncOnStreamingBufferOp:Dataflow_Op<"implicit_sync_on_streaming_buffer",
                                      [Dataflow_DebugNameOpInterface]> {
    let summary = "Performs sync send/recv operations at the buffer granularity level";
    let description = [{
      "This operation is meant to enable synchronization implicitly at the boundary of streaming buffer."
    ```mlir
          %handler = dataflow.implicit_sync_on_streaming_buffer %l1_view, %l1su, %size : (memref<4xi8>, Index, Index, Index)
    ```
    }];
    let arguments = (ins AnyTypeOf<[AnyMemRef]>:$memory_unit, Index:$dst_unit, Index:$buffer_size,
                         OptionalAttr<StrAttr>:$dbgName);
    let assemblyFormat = [{
      $memory_unit `,` $dst_unit `,` $buffer_size attr-dict `:`
                  type($memory_unit) `,` type($dst_unit)  `,` type($buffer_size)
    }];
}

def Dataflow_GetLogicalMemoryViewOp:Dataflow_Op<"get_logical_memory_view",
  [NoMemoryEffect, IsolatedFromAbove]> {
    let summary = "Operation to get a logical view of the memory";
    let description = [{
    Views a hardware memory region, addressed linearly from `$start_address`, as
    a multi-dimensional memref. The `layout_map` affine map describes how the
    multi-dimensional indices of the view map back onto the 1-D address space.

    Addresses in Dataflow IR are in *element* granularity, not bytes. For
    example, with `layout_map = affine_map<(i, j) -> (i + j * 8)>`, index
    `(0, 1)` refers to element `0 + 1 * 8 = 8` of the underlying 1-D region.

    ```mlir
    %l1_view = dataflow.get_logical_memory_view (%l1_mem, %start_address)
        {layout_map = affine_map<(i, j) -> (i + j * 8)>}
        : index, index, memref<8x16xi8>
    ```
    }];
    let arguments = (ins Index:$from_unit, Index:$start_address, AffineMapAttr:$layout_map);
    let results = (outs AnyTypeOf<[AnyMemRef]>:$data);
    let assemblyFormat = [{
         $from_unit `,` $start_address  attr-dict `:` type($from_unit) `,` type($start_address) `,` type(results)
    }];
}

def Dataflow_GetPagedLogicalMemoryViewOp:Dataflow_Op<"get_paged_logical_memory_view",
  [NoMemoryEffect, IsolatedFromAbove]> {
    let summary = "Operation to get logical views of paged memory";
    let description = [{
    "This operation is useful to get logical views of paged memory."
    ```mlir
    %data = dataflow.get_paged_logical_memory_view %from, %start_address {#layout_map}
                {segments =
                    page0: (idx_set = (d0 >= 0, d0 < 64, d1 >= 0, d1 <= 1, d2 = 0>),
                            start_address = %pg0_start_address),
                    page1: (idx_set = (d0 >= 0, d0 < 64, d1 >= 2, d1 <= 3, d2 = 0>),
                            start_address = %pg1_start_address),
                    page2: (idx_set = (d0 >= 0, d0 < 64, d1 >= 0, d1 <= 1, d2 = 1>),
                            start_address = %pg2_start_address),
                    ...
                    page63: (idx_set = (d0 >= 0, d0 < 64, d1 >= 2, d1 <= 3, d2 = 31>),
                            start_address = %pg63_start_address)}
                : index, index, memref<?x?x?xf16>
    ```
    }];
    let arguments = (ins Index:$unit,
                         Index:$start_addr,
                         Variadic<Index>:$page_start_addrs,
                         AffineMapAttr:$layout_map,
                         ArrayAttr:$idx_sets);
    let results = (outs AnyTypeOf<[AnyMemRef]>:$data);
    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;

    let extraClassDeclaration = [{
      StringRef getIdxSetsStrName() { return "idx_sets"; }
    }];
}

def Dataflow_SendOp:Dataflow_Op<"send",
  [Dataflow_DebugNameOpInterface]> {
    let summary = "Send a vector of data to another unit";
    let description = [{
    Sends a vector of data to another unit. The optional `dir` attribute records
    the routing direction of the transfer.

    ```mlir
    dataflow.send %l0su, %data : index, vector<128xi8>
    ```
    }];

    let arguments = (ins Index:$to_unit,
                     AnyTypeOf<[AnyVectorOfAnyRank, Dataflow_CustomVectorType]>:$send_data,
                     OptionalAttr<DataflowRoutingDirectionAttr>:$dir,
                     OptionalAttr<StrAttr>:$dbgName
                     );
    let assemblyFormat = [{
         $to_unit `,` $send_data attr-dict `:` type($send_data)
    }];
}

def Dataflow_ReceiveOp:Dataflow_Op<"receive",
  [Dataflow_DebugNameOpInterface]> {
    let summary = "Receive a vector of data from another unit";
    let description = [{
    Receives a vector of data from another unit and returns it as the result.

    ```mlir
    %data = dataflow.receive %l1lu : vector<128xi8>
    ```
    }];
    let arguments = (ins Index:$from_unit,
                         OptionalAttr<StrAttr>:$dbgName);
    let results = (outs AnyTypeOf<[AnyVectorOfAnyRank, Dataflow_CustomVectorType]>:$data);

    let assemblyFormat = [{
         $from_unit attr-dict `:` type(results)
    }];
}

def Dataflow_OpaqueOp:Dataflow_Op<"opaque",
  [Dataflow_DebugNameOpInterface]> {
    let summary = "Opaque operation";
    let description = [{
      "Opaque operation for auxiliary functions."
      ```mlir
      dataflow.opaque {func_name = "reciprocal",
                       read_write_register_dictionary = {p0="R0",p1="R1"},
                       read_only_register_dictionary = {c0="R2"},
                       parameter_dictionary = {unroll_factor="2"}}
      ```
    }];
    let arguments = (ins OptionalAttr<StrAttr>:$dbgName,
                         StrAttr:$func_name,
                         DictionaryAttr:$read_write_register_dictionary,
                         DictionaryAttr:$read_only_register_dictionary,
                         DictionaryAttr:$parameter_dictionary);
    let assemblyFormat = [{
        attr-dict
    }];
}

#endif // DATAFLOW_SCHEDULER_DIALECT_DATAFLOW_DATAFLOW_TD
