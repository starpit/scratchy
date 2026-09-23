//===-- ExPlanOps.td -- ExPlan Dialect Operation Definitions -*- tablegen -==//
//
// IBM Confidential
// (C) Copyright IBM Corp. 2026
//
// =============================================================================
//
// This file defines the execution plan dialect's operations.
//
// A compiled run is not only the programs it is made of: something has to get
// their packets onto the device, compute what only the host can compute, and
// launch them in an order. That sequence is the execution plan, and these are
// the steps it is made of -- so that what the run does is in the IR, in order,
// rather than only in the job plan written out at the end.
//
//===----------------------------------------------------------------------===//

#ifndef DIALECTS_DIALECT_EXPLAN_OPS_TD
#define DIALECTS_DIALECT_EXPLAN_OPS_TD

#ifndef OP_BASE
include "mlir/IR/OpBase.td"
#endif // OP_BASE

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def ExPlan_Dialect : Dialect {
  let name = "explan";
  let cppNamespace = "::mlir::explan";
  let summary = "The steps a compiled run is executed in";
}

// Which of the job's two plans a step belongs to. A run is prepared once and
// then executed, so a step is either setting the device up before the job or
// part of the job itself.
def ExPlan_Phase_Preparation : I32EnumAttrCase<"preparation", 0>;
def ExPlan_Phase_Execution : I32EnumAttrCase<"execution", 1>;

def ExPlan_Phase : I32EnumAttr<"Phase",
    "When a step of the plan happens",
    [ExPlan_Phase_Preparation, ExPlan_Phase_Execution]> {
  let cppNamespace = "::mlir::explan";
}

// Base class for ExPlan dialect operations.
class ExPlan_Op<string mnemonic, list<Trait> traits = []> :
    Op<ExPlan_Dialect, mnemonic, traits> {
}

def HostToDeviceOp : ExPlan_Op<"host_to_device",
    [MemoryEffects<[MemWrite<DefaultResource>]>]> {
  let summary = "Sends a run of bytes from the host to where it will be read";
  let description = [{
    The `explan.host_to_device` operation sends a run of bytes to device memory,
    where what follows it will read them: the bytes themselves as its operand, and
    the device address they start at.

    The operand is the whole description of what is sent. How many bytes there are
    is its type, and what they are is whatever produced it -- a program's packet,
    or the flits a correction's host compute prepared. Neither is worth repeating
    in an attribute, where it could come to disagree with the value.

    `addr` is in bytes. A program is placed at stick granularity and the address
    kept on its frame counts sticks, but nothing performing a transfer deals in
    those, so the conversion happens before it reaches here.

    `phase` says which of the job's two plans this belongs to. A program's packet
    is sent while the device is being prepared, since it is the same every run; the
    flits a correction's host compute produced depend on the run's inputs, so
    sending those is part of executing the job.

    Example:
    ```mlir
    explan.host_to_device(%packet) {addr = 68719476736 : i64,
                                    phase = 0 : i32} : tensor<1024xi8>
    ```
  }];

  let arguments = (ins AnyRankedTensor:$bytes, I64Attr:$addr,
                       ExPlan_Phase:$phase);
  let hasVerifier = 1;

  let assemblyFormat = "`(` $bytes `)` attr-dict `:` type($bytes)";
}

def DeviceComputeOp : ExPlan_Op<"devicecompute",
    [MemoryEffects<[MemWrite<DefaultResource>]>]> {
  let summary = "Runs what was transferred to the device";
  let description = [{
    The `explan.devicecompute` operation is the device's part of the run: the
    packet at `address` is executed there.

    It stands after the transfer that put it there, and says nothing about what it
    is beyond where it is -- which is all the device is told. What was placed at
    that address, and how it got there, is the transfer above it and the program
    the transfer's bytes came from.

    `phase` says which of the job's two plans this belongs to, which for running
    something is always the execution one.

    Example:
    ```mlir
    explan.devicecompute {address = 120259088512 : i64, phase = 1 : i32}
    ```
  }];

  let arguments = (ins I64Attr:$address, ExPlan_Phase:$phase);

  let assemblyFormat = "attr-dict";
}

def HostComputeExecuteOp : ExPlan_Op<"host_compute_execute",
    [MemoryEffects<[MemWrite<DefaultResource>]>]> {
  let summary = "The host computation itself: symbols in, a buffer out";
  let description = [{
    The `explan.host_compute_execute` operation is the host's part of the run: it
    takes the symbol values the computation is over and produces the buffer the
    host hands back.

    It stands in the body of the function that names that computation, as what
    that function returns. Everything above it in the body works out the symbol
    values -- the ones the run supplies arrive as the function's arguments, and
    the rest are expressions over those -- and this is what consumes them. So the
    body reads as what it is: the symbols, then the compute over them.

    What actually produces the bytes is the host function the job plan names. This
    says which values it is a function of, and how many bytes come back; it does
    not say how, and nothing in the IR computes it.

    Example:
    ```mlir
    %flits = explan.host_compute_execute(%arg0, %arg1, %0)
        : (index, index, index) -> tensor<4224xi8>
    ```
  }];

  let arguments = (ins Variadic<Index>:$symbols);
  let results = (outs AnyRankedTensor:$output);

  let assemblyFormat = [{
      `(` $symbols `)` attr-dict `:` functional-type($symbols, $output)
  }];
}

#endif // DIALECTS_DIALECT_EXPLAN_OPS_TD
