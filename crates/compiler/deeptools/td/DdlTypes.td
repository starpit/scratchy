//===-- DdlTypes.td -- DdlTypes Dialect Operation Definitions -*- tablegen ----==//
//
// IBM Confidential
// (C) Copyright IBM Corp. 2023
//
// =============================================================================
//
// This file defines the ddl types in the MLIR operation set.
//
//===----------------------------------------------------------------------===//

#ifndef DDC_DDL_DIALECT_DDL_TYPES_TD
#define DDC_DDL_DIALECT_DDL_TYPES_TD

#ifndef OP_BASE
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
#endif // OP_BASE

include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/AttrTypeBase.td"


def Ddl_Dialect : Dialect {
  let name = "ddl";
  let cppNamespace = "::mlir::ddl";
  let useDefaultAttributePrinterParser = 1;
}

// Base class for Ddl dialect operations. This operation inherits from the
// base `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class Ddl_Op<string mnemonic, list<Trait> traits = []> :
    Op<Ddl_Dialect, mnemonic, traits> {
  // For every standard op, there needs to be a:
  //   * void print(OpAsmPrinter &p, ${C++ class of Op} op)
  //   * LogicalResult verify(${C++ class of Op} op)
  //   * ParseResult parse${C++ class of Op}(OpAsmParser &parser,
  //                                         OperationState &result)
  // functions.
}


def dummy : I32EnumAttrCase<"dummy", 0>;


def DdlDummy : I32EnumAttr<"DdlDummy",
    "built-in precisions for ddl.",
    [
      dummy
    ]> {
        let genSpecializedAttr = 0;
    }

def DdlDummyAttr : EnumAttr<Ddl_Dialect, DdlDummy, "dummy">;

#endif // DDC_DDL_DIALECT_DDL_TYPES_TD
