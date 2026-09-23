//===-- InitOps.td -- Init Dialect Operation Definitions -*- tablegen ----==//
//
// IBM Confidential
// (C) Copyright IBM Corp. 2026
//
// =============================================================================
//
// This file defines the init dialect operations in the MLIR operation set.
//
//===----------------------------------------------------------------------===//

#ifndef DIALECTS_DIALECT_INIT_OPS_TD
#define DIALECTS_DIALECT_INIT_OPS_TD

#ifndef OP_BASE
include "mlir/IR/OpBase.td"
#endif // OP_BASE

include "mlir/IR/BuiltinAttributes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Init_Dialect : Dialect {
  let name = "init";
  let cppNamespace = "::mlir::init";
}

// Base class for Init dialect operations.
class Init_Op<string mnemonic, list<Trait> traits = []> :
    Op<Init_Dialect, mnemonic, traits> {
}

def SmcOp : Init_Op<"smc",
    [MemoryEffects<[MemWrite<DefaultResource>]>]> {
  let summary = "References an SMC artifact by name";
  let description = [{
    The `init.smc` operation holds the name of an SMC artifact. The actual
    artifact is stored externally in a context map keyed by this name.

    Example:
    ```mlir
    init.smc { name = "my_smc" }
    ```
  }];

  let arguments = (ins StrAttr:$name);

  let assemblyFormat = "attr-dict";
}

def BinOp : Init_Op<"bin",
    [MemoryEffects<[MemWrite<DefaultResource>]>]> {
  let summary = "References a binary artifact by name, and is its bytes";
  let description = [{
    The `init.bin` operation holds the name of a binary artifact. The actual
    artifact is stored externally in a context map keyed by this name.

    It produces that artifact's bytes: a `tensor<size x i8>`, where `size` is how
    long the packet is. So a program is a value the run can pass on -- what has to
    reach the device is what the program evaluated to, rather than something a
    later stage has to go and look up.

    It also carries the symbols the program the artifact was built for stands
    behind, the same way `sdscbundle.sdsc_execute` carries them: the packet holds
    addresses that are only settled when the program runs, and this is what says
    in the IR which symbol each of those is. A program reduced to its packet has
    no other record of them left, and the correction the packet needs is written
    against exactly these.

    Examples:
    ```mlir
    %packet = init.bin { name = "my_bin", size = 1024 } : tensor<1024xi8>
    %packet = init.bin(%0, %1) { name = "my_bin", size = 1024,
                                 symbol_ids = [-1, -2] } : tensor<1024xi8>
    ```
  }];

  let arguments = (ins Variadic<Index>:$symbol_definitions,
                       StrAttr:$name,
                       I64Attr:$size,
                       DefaultValuedAttr<I64ArrayAttr, "{}">:$symbol_ids);
  let results = (outs AnyRankedTensor:$packet);
  let hasVerifier = 1;

  // The operands are only printed when there are some, so a packet reference
  // that stands behind no symbols reads as it always has.
  let assemblyFormat = [{
      (`(`$symbol_definitions^`)`)? attr-dict `:` type($packet)
  }];
}

def SymLocOp : Init_Op<"symloc",
    [MemoryEffects<[MemWrite<DefaultResource>]>]> {
  let summary = "References a correction table artifact by name";
  let description = [{
    The `init.symloc` operation holds the name of a correction table artifact.
    The actual artifact is stored externally in a context map keyed by this name.

    Example:
    ```mlir
    init.symloc { name = "my_corr_table" }
    ```
  }];

  let arguments = (ins StrAttr:$name);

  let assemblyFormat = "attr-dict";
}

#endif // DIALECTS_DIALECT_INIT_OPS_TD
