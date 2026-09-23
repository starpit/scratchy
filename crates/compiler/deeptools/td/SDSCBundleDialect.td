//===----------------------------------------------------------------------===//
//
// IBM Confidential
// (C) Copyright IBM Corp. 2026
//
//===----------------------------------------------------------------------===//
//
// This file defines the SDSCBundle dialect.
//
//===----------------------------------------------------------------------===//

#ifndef SDSCBUNDLE_DIALECT_TD
#define SDSCBUNDLE_DIALECT_TD

#ifndef OP_BASE
include "mlir/IR/OpBase.td"
#endif // OP_BASE

def SDSCBundle_Dialect : Dialect {
  let name = "sdscbundle";
  let cppNamespace = "::mlir::sdscbundle";
  let useDefaultTypePrinterParser = 1;
}

#endif // SDSCBUNDLE_DIALECT_TD