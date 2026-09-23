//===-- TraceOps.td -- Trace Dialect Operation Definitions -*- tablegen ----==//
//
// IBM Confidential
// (C) Copyright IBM Corp. 2025
//
// =============================================================================
//
// This file defines the Trace dialect operations in the MLIR operation set.
//
//===----------------------------------------------------------------------===//

#ifndef DCC_SRC_DIALECT_TRACE_OPS_TD
#define DCC_SRC_DIALECT_TRACE_OPS_TD

#ifndef OP_BASE
include "mlir/IR/OpBase.td"
#endif // OP_BASE

include "mlir/IR/BuiltinAttributes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Trace_Dialect : Dialect {
  let name = "trace";
  let cppNamespace = "::mlir::trace";
}

// Base class for Trace dialect operations.
class Trace_Op<string mnemonic, list<Trait> traits = []> :
    Op<Trace_Dialect, mnemonic, traits> {}

def Trace_ReturnOp : Trace_Op<"return", [Terminator]> {
  let summary = [{
    The `return` operation terminates a trace region.
  }];
}

def Trace_ProgramUnitTraceOp : Trace_Op<"program_unit",
  [SingleBlockImplicitTerminator<"ReturnOp">]> {
    let summary = "Static trace of a unit";
    let description = [{
    "This operation holds the static trace of a unit.  It is symmetric to
     dataflow.program_unit but used for holding the `unrolled` form of the
     function that it traces.
     Attribute `tracename` is a human readable name to annotate the trace,
     typically set to the name of the pass that was applied prior to the trace.
     Set to `standalone-trace` with `--kEmitTraceIR`.
     Set to `initial-pass` for the first trace in a pipeline.
     Attribute `compare_pass` is only set if a trace comparison was performed.
     The tracer runs a comparison after generating a trace if a
     corresponding prior trace was available in the IR prior.
     True if the traces compared equivalent; false if they did not.
    ```mlir
    trace.program_unit %unit_id : (Index) { }
    ```
    }];

    let arguments = (ins Variadic<Index>:$units,
                         StrAttr:$tracename,
                         OptionalAttr<BoolAttr>:$compare_pass);
    let regions = (region SizedRegion<1>:$region);
    let skipDefaultBuilders = 1;
    let builders = [
    OpBuilder<(ins "ValueRange":$units, "StringAttr":$tracename,
                   CArg<"function_ref<void(OpBuilder &, Location, ValueRange)>",
                        "nullptr">:$bodyBuilder)>,
    ];

    let extraClassDeclaration = [{
    using BodyBuilderFn =
        function_ref<void(OpBuilder &, Location, ValueRange)>;
        static StringRef getTracenameAttrStrName() { return "tracename"; }
        static StringRef getComparePassAttrStrName() { return "compare_pass"; }
    }];

    let hasCustomAssemblyFormat = 1;
}

#endif // DCC_SRC_DIALECT_TRACE_OPS_TD
