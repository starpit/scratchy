//===-- KTDFArch.td ----------------------------------------*- tablegen -*-===//
//
// Copyright 2026 The KTIR Scheduler Authors.
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
// This file defines the attributes, types and operations in the ktdf_arch 
// dialect.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_DIALECT_KTDFARCH_KTDFARCH_TD
#define DATAFLOW_SCHEDULER_DIALECT_KTDFARCH_KTDFARCH_TD

include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchTypes.td"

//===----------------------------------------------------------------------===//
// Traits
//===----------------------------------------------------------------------===//

def KTDFArch_IsSubgraph : NativeOpTrait<"IsSubgraph">, StructuralOpTrait {
  let cppNamespace = "::mlir::ktdf_arch";
  let extraConcreteClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // OpAsmOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Allow the dialect prefix to be omitted.
    [[nodiscard]] static auto getDefaultDialect() -> StringRef { 
      return KTDFArchDialect::getDialectNamespace(); 
    }
  }];
}

def KTDFArch_Subgraph : TraitList<[OpAsmOpInterface, KTDFArch_IsSubgraph]>;

//===----------------------------------------------------------------------===//
// Operations
//===----------------------------------------------------------------------===//

class KTDFArch_Op<string mnemonic, list<Trait> traits = []> :
    Op<KTDFArch_Dialect, mnemonic, traits> {
}

def KTDFArch_DeviceOp : KTDFArch_Op<"device", [
  Symbol,
  IsolatedFromAbove,
  KTDFArch_Subgraph,
  NoTerminator
]> {
  let summary = "Defines an accelerator device.";
  let description = [{
    A `ktdf_arch.device` operation provides a definition of an an accelerator
    target device's architecture, suitable for scheduling dataflow on it.

    Every device requires a unique name by which it is referred to within its
    enclosing `SymbolTable` scope. The device architecture is represented using
    resources, which are MLIR ops nested underneath the `ktdf_arch.device` op. 
    The architecture is given as a graph of nodes and links between them.

    There are three different kinds of nodes:

      - Memory, as defined by `ktdf_arch.memory`, which can store data.
      - Execution Units, as defined by `ktdf_arch.exec_unit`, which can process
        or move data.
      - Switches, as defined by `ktdf_arch.switch`, which can relay data.

    There is one kind of link:

      - Datapath, as defined by `ktdf_arch.datapath`, which can relay data.

    There is one additional kind of resource:

      - Groups, as defined by `ktdf_arch.group`, which group private resources.

    Each resource can have a unique identifier, which allows for finding it via 
    name-based lookup. Each resource can also have a class, which relates it to
    resources that are functionally similar.

    Device descriptions can also be imported from disk when needed, by provding
    a `import(...)` declaration instead of a body. The file path is relative to
    the source file (or CWD if reading from the CLI), and must lead to an MLIR 
    file that contains a non-imported device with the same name. The 
    `DeviceManager` analysis provided by this dialect implements loading and 
    caching of devices on the module scope.

    Example:
    ```
    ktdf_arch.device @my_device {
      // Define one memory and one compute resource.
      %main_memory = memory { kind = 1 }
      %main_exec = exec_unit

      // Establish a bi-directional link between the two.
      datapath %main_memory to %main_exec : memory, exec_unit
      datapath %main_exec to %main_memory : exec_unit, memory
    }

    // Import an architecture graph that is read from disk later.
    ktdf_arch.device @my_imported_device import("./relative/to/input.mlir")
    ```
  }];

  let arguments = (ins 
    SymbolNameAttr:$sym_name,
    OptionalAttr<StrAttr>:$import_path,
    KTDFArch_ErasedAttr<OptionalAttr<I64Attr>>:$version
  );
  let regions = (region MaxSizedRegion<1>:$region);

  let hasVerifier = 1;
  let hasRegionVerifier = 1;

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "StringAttr":$sym_name, CArg<"StringAttr", "nullptr">:$import_path)>,
    OpBuilder<(ins "StringRef":$sym_name, CArg<"StringRef", "{}">:$import_path), [{
      const auto import_path_attr = import_path.empty() 
        ? StringAttr{} 
        : $_builder.getStringAttr(import_path);
      build($_builder, $_state, $_builder.getStringAttr(sym_name), import_path_attr);
    }]>
  ];

  let assemblyFormat = [{
    $sym_name attr-dict-with-keyword 
    custom<GraphOrImport>($region, $import_path)
  }];

  let extraClassDeclaration = [{
    /// Gets a value indicating whether the device graph is to be imported.
    [[nodiscard]] auto isImported() -> bool { 
      return getRegion().empty(); 
    }

    /// Gets the body Region.
    [[nodiscard]] auto getBodyRegion() -> Region& { 
      return getRegion(); 
    }
    /// Gets the body block, or `nullptr` if imported.
    [[nodiscard]] auto getBody() -> Block* { 
      return isImported() ? nullptr : &getRegion().front(); 
    }
  }];
}

//===----------------------------------------------------------------------===//
// Resources
//===----------------------------------------------------------------------===//

// Base class for defining resources.
//
// Since all inherent attributes are stored as properties, but assigning the
// inherent attributes from a property dict is silently throwing away errors,
// we need to type-erase all inherent attributes in storage.
class KTDFArch_ResourceOp<
  string mnemonic, 
  list<Trait> traits = [], 
  dag inherentAttrs = (ins)
> : KTDFArch_Op<mnemonic, traits # [KTDFArch_Resource]> {
  // NOTE: Unfortunately, tablegen-lsp can't handle !foreach on dags. So, we
  //       use this work-around to improve the developer experience instead.
  defvar inherentAttrNames = !foreach(i, !range(!size(inherentAttrs)), !getdagname(inherentAttrs, i));
  defvar inherentAttrTypes = !foreach(i, !range(!size(inherentAttrs)), !getdagarg<Attr>(inherentAttrs, i));
  defvar erasedAttrs = !dag(ins, !foreach(attr, inherentAttrTypes, KTDFArch_ErasedAttr<attr>), inherentAttrNames);

  // By default, every entity has:
  //   - an optional ID attribute and
  //   - an optional kind (class is not allowed as a field name).
  // which it can extend by declaring more type-erased attribute properties.
  let arguments = !con(
    (ins 
      OptionalAttr<SymbolNameAttr>:$id,
      DefaultValuedAttr<AnyAttr, "nullptr">:$kind
    ),
    erasedAttrs
  );

  // By default, an entity stores all of its inherent attributes in its
  // attribute dictionary, for forward compatiblity. Only attributes mandated
  // by the V0 spec are exempted from this.
  let assemblyFormat = "($id^)? ``custom<AttrDictOrAlias>(attr-dict)";
}

def KTDFArch_GroupOp : KTDFArch_ResourceOp<"group", [
  IsolatedFromAbove,
  SingleBlock,
  KTDFArch_Subgraph,
  SingleBlockImplicitTerminator<"YieldOp">,
  DeclareOpInterfaceMethods<RegionBranchOpInterface, ["getEntrySuccessorOperands"]>
]> {
  let summary = "Defines a delimited subgraph of resources.";
  let description = [{
    A `ktdf_arch.group` operation establishes a hierarchical grouping of 
    entities, such that the contained resources are considered private.

    Two resources are equivalent if they have the same capabilities, such as two
    cores with the same ISA and the same amount of local private memory. Local
    similarity relaxes this constraint by allowing hierarchically-bounded
    exceptions: two entities are locally similar iff their private resources
    are equivalent.

    The `ktdf_arch.group` operation declares a level in this hierarchy by
    requiring explicit capture of shared resources that are not private to it.
    When two groups have isomoprhic subgraphs nested underneath them, they can
    be declared as having the same class. Further, these two groups are then
    known to be equivalent, iff they capture the same resources.

    Groups are anchored by the `ktdf_arch.compute` resources nested underneath
    them. To indicate communication between these resources of different, 
    groups, they can be returned from the class and linked outside. In this 
    case, the parent of the `ktdf_arch.datapath` operation indicates the scope
    to which the link is private.

    Example:
    ```
    ktdf_arch.device @my_device {
      %shared_a = memory { kind = "shared_a" }
      %shared_b = memory { kind = "shared_b" }

      // The following 3 entities are locally similar, i.e., their internal
      // resources are the same. Furthermore:
      // - @grp0 and @grp1 are similar, but their list of captured 
      //   resources differs, meaning that they are _not_ identical.
      // - @grp0 and @grp2 are similar, and their list of captures
      //   resources is the same, meaning that they _must be_ identical.

      %c0 = group @grp0 { kind = "core" } share(%shared_a) {
        %private = memory { kind = "private" }
        datapath %shared_a to %private : memory, memory
        datapath %private to %shared_a : memory, memory

        %compute = exec_unit
        datapath %private to %compute : memory, exec_unit
        datapath %compute to %private : exec_unit, memory
        yield %compute
      } -> exec_unit
      group @grp1 { kind = "core" } share(%shared_b) {
        %private = memory { kind = "private" }
        datapath %shared_b to %private : memory, memory
        datapath %private to %shared_b : memory, memory

        %compute = exec_unit
        datapath %private to %compute : memory, exec_unit
        datapath %compute to %private : exec_unit, memory
      }
      %c2 = group @grp2 { kind = "core" } share(%shared_a) {
        %private = memory { kind = "private" }
        datapath %shared_a to %private : memory, memory
        datapath %private to %shared_a : memory, memory

        %compute = exec_unit
        datapath %private to %compute : memory, exec_unit
        datapath %compute to %private : exec_unit, memory
        yield %compute
      } -> exec_unit

      datapath %c0 to %c2 : exec_unit, exec_unit
      datapath %c2 to %c0 : exec_unit, exec_unit
    }
    ```
  }];

  let arguments = (ins 
    Variadic<KTDFArch_MemoryType>:$shared_memory,
    OptionalAttr<SymbolNameAttr>:$id,
    DefaultValuedAttr<AnyAttr, "nullptr">:$kind
  );
  let results = (outs Variadic<KTDFArch_ExecutionUnitType>:$shared_exec);
  let regions = (region SizedRegion<1>:$region);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins 
        CArg<"ValueRange", "{}">:$shared_memory, 
        CArg<"unsigned", "0">:$num_results, 
        CArg<"function_ref<void(OpBuilder&, Location, ValueRange)>", "nullptr">:$body_builder
    )>
  ];

  let assemblyFormat = ?;
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // OpAsmOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Group the results, and name them using the type mnemonic.
    void getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
      if ((*this)->getNumResults() > 0) {
        setNameFn((*this)->getResult(0), ExecutionUnitType::getMnemonic());
      }
    }
  }];
}

def KTDFArch_YieldOp : KTDFArch_Op<"yield", [
  Pure, 
  ReturnLike, 
  Terminator, 
  HasParent<"GroupOp">
]> {
  let summary = "Yields execution unit resources from a group.";
  let description = [{
    Example:
    ```
    ktdf_arch.device @my_device {
      %c0 = exec_unit
      
      %c1 = group share() {
        %c1 = exec_unit
        yield %c1
      } -> exec_unit

      datapath %c0 to %c1 : exec_unit, exec_unit
      datapath %c1 to %c0 : exec_unit, exec_unit
    }
    ```
  }];

  let arguments = (ins Variadic<KTDFArch_ExecutionUnitType>:$operands);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins CArg<"ValueRange", "{}">:$operands), [{
      $_state.operands.append(operands.begin(), operands.end());
    }]>
  ];

  let assemblyFormat = "$operands attr-dict";
}

//===----------------------------------------------------------------------===//
// Nodes
//===----------------------------------------------------------------------===//

// Base class for defining resource node operations.
class KTDFArch_NodeOp<
  string mnemonic, 
  Type type, 
  list<Trait> traits = [], 
  dag inherentAttrs = (ins)
> : KTDFArch_ResourceOp<
  mnemonic, 
  traits # [KTDFArch_Node, OpAsmOpInterface], 
  inherentAttrs
> {
  // By default, every resource returns a single SSA value of its type.
  let results = (outs type:$result);

  string defaultOpAsmImpl = !subst("$_type", type.cppType, [{
    //===------------------------------------------------------------------===//
    // OpAsmOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Allow the dialect prefix to be omitted.
    [[nodiscard]] static auto getDefaultDialect() -> StringRef { 
      return KTDFArchDialect::getDialectNamespace(); 
    }

    /// Group the results, and name them using the type mnemonic.
    void getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
      if ((*this)->getNumResults() > 0) {
        setNameFn((*this)->getResult(0), $_type::getMnemonic());
      }
    }
  }]);
  let extraClassDeclaration = defaultOpAsmImpl;
}

def KTDFArch_MemoryOp : KTDFArch_NodeOp<
  "memory", 
  KTDFArch_MemoryType, 
  [],
  (ins OptionalAttr<ConfinedAttr<AnyI64Attr, [IntMinValue<1>]>>:$size)
> {
  let summary = "Defines a memory resource.";
  let description = [{
    A `ktdf_arch.memory` operation defines a memory resource, with an associated
    memory space, in which the scheduler can place data.

    The class of a memory must be a valid memory space, e.g., for use with the
    `memref` type, and must always be specified.

    Memories can be passed downward the hierarchical structure, i.e., they
    can be captured by `ktdf_arch.group` operations, but they can't propagate
    back upwards. This means every memory resource has a deepest legal scope
    where none if its users lies above. By definition, the memory is _private_
    to that scope.

    Memories can have an optional `size` >= 1, specified in bytes.

    Example:
    ```
    ktdf_arch.device @my_device {
      %ram = memory { kind = #ptr.generic_space }
      %ddr = memory { kind = "DDR", size = 17179869184 }
    }
    ```
  }];

  let hasVerifier = 1;

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins 
        "Attribute":$kind, 
        CArg<"std::optional<int64_t>", "std::nullopt">:$size
    ), [{
      $_state.types.push_back(MemoryType::get($_builder.getContext()));
      $_state.addAttribute(getKindAttrName($_state.name), kind);
      if (size) {
        $_state.addAttribute(getSizeAttrName($_state.name), $_builder.getI64IntegerAttr(*size));
      }
    }]>
  ];
}

def KTDFArch_ExecutionUnitOp : KTDFArch_NodeOp<
  "exec_unit", 
  KTDFArch_ExecutionUnitType,
  [],
  (ins OptionalAttr<UnitAttr>:$load_store)
> {
  let summary = "Defines an execution unit resource.";
  let description = [{
    A `ktdf_arch.exec_unit` operation defines an execution unit resource, which 
    the scheduler can assign work to.

    Execution units can be passed upward the hierarchical structure, i.e., 
    they can be released by `ktdf_arch.group` operations. This allows them
    to establish connections with other compute resources outside of their
    scope without needing to break the local symmetry by hoisting them.

    Execution units can be designated as `load_store` units, which indicates
    that they should not be considered for tasks not involving data movement.

    Example:
    ```
    ktdf_arch.device @my_device {
      %exec_unit = exec_unit
      %load_store = exec_unit { load_store }
    }
    ```
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins CArg<"bool", "false">:$load_store), [{
      $_state.types.push_back(ExecutionUnitType::get($_builder.getContext()));
      if (load_store) {
        $_state.addAttribute(getLoadStoreAttrName($_state.name), $_builder.getUnitAttr());
      }
    }]>
  ];

  let extraClassDeclaration = defaultOpAsmImpl # [{
    /// Gets a value indicating whether this is a load/store unit.
    [[nodiscard]] auto isLoadStore() -> bool {
      return getLoadStore().value_or(false);
    }
  }];
}

def KTDFArch_SwitchOp : KTDFArch_NodeOp<
  "switch", 
  KTDFArch_PortType,
  [],
  (ins KTDFArch_AdjacencyMatrixAttr:$connectivity)
> {
  let summary = "Defines a switch resource.";
  let description = [{
    A `ktdf_arch.switch` operation defines a routing switch resource, which
    the scheduler can use to plan data movement.

    Compared to other resources, switches have 0 or more dedicated ports that
    can each be connected to individual links. The connectivity between the
    ports is encoded on the switch as an adjacency matrix. If none is specified,
    it defaults to full connectivity (`dense<1>`).

    Example:
    ```
    ktdf_arch.device @my_device {
      %sw0:2 = switch [2]
      %sw1:2 = switch [2] { connectivity = dense<[[0, 1],[1, 0]]> : tensor<2x2xi1> }
      %sw2:4 = switch [4] { connectivity = sparse<[[0, 2], [0, 3], [1, 2]], true> : tensor<4x4xi1> }
    }
    ```
  }];

  let results = (outs Variadic<KTDFArch_PortType>:$result);

  let hasVerifier = 1;

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "unsigned":$num_ports, CArg<"ElementsAttr", "{}">:$connectivity)>
  ];

  let assemblyFormat = ?;
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // OpAsmOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Group the results, and name them using the type mnemonic.
    void getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
      if ((*this)->getNumResults() > 0) {
        setNameFn((*this)->getResult(0), "switch");
      }
    }
  }];
}

//===----------------------------------------------------------------------===//
// Links
//===----------------------------------------------------------------------===//

// Base class for defining link node operations.
class KTDFArch_LinkOp<
  string mnemonic, 
  list<Trait> traits = [], 
  dag inherentAttrs = (ins)
> : KTDFArch_ResourceOp<mnemonic, traits # [KTDFArch_Link], inherentAttrs> {}

def KTDFArch_DatapathOp : KTDFArch_LinkOp<"datapath"> {
  let summary = "Declares a datapath between two endpoints.";
  let description = [{
    A `ktdf_arch.datapath` operation defines a direct point-to-point conection 
    between two resource endpoints.

    It is illegal to connect an endpoint to itself.

    Example:
    ```
    ktdf_arch.device @my_device {
      %m = memory { kind = 1 }
      %c = exec_unit
      %sw:2 = switch [2]

      datapath @c_dma %m to %c : memory, exec_unit
      datapath { kind = "fifo" } %c to %sw#0 : exec_unit, port
    }
    ```
  }];

  let arguments = (ins 
    KTDFArch_EndpointType:$source, 
    KTDFArch_EndpointType:$target,
    OptionalAttr<SymbolNameAttr>:$id,
    DefaultValuedAttr<AnyAttr, "nullptr">:$kind
  );

  let hasVerifier = 1;

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "Value":$source, "Value":$target), [{
      $_state.operands.push_back(source);
      $_state.operands.push_back(target);
    }]>
  ];

  let assemblyFormat = [{
    ($id^)? ``custom<AttrDictOrAlias>(attr-dict)
    $source `to` $target
    `:` custom<ShortType>(type($source)) `,` custom<ShortType>(type($target)) 
  }];

  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // Link Methods
    //===------------------------------------------------------------------===//

    /// Gets the source operands.
    [[nodiscard]] auto getSources() -> OperandRange { 
      return OperandRange{getOperands().begin(), getOperands().begin() + 1}; 
    }

    /// Gets the target operands.
    [[nodiscard]] auto getTargets() -> OperandRange { 
      return OperandRange{getOperands().begin() + 1, getOperands().end()}; 
    }
  }];
}

#endif // DATAFLOW_SCHEDULER_DIALECT_KTDFARCH_KTDFARCH_TD