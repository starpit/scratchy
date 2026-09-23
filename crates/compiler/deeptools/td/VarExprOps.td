//===-- VarExprOps.td -- VarExpr Dialect Operation Definitions -*- tablegen -==//
//
// IBM Confidential
// (C) Copyright IBM Corp. 2026
//
// =============================================================================
//
// This file defines the varexpr dialect: the VariableOperator expressions that
// have no equivalent in arith. Everything a symbol can be defined as is either
// arithmetic, and written with arith, or one of these.
//
//===----------------------------------------------------------------------===//

#ifndef DIALECTS_DIALECT_VAREXPR_OPS_TD
#define DIALECTS_DIALECT_VAREXPR_OPS_TD

#ifndef OP_BASE
include "mlir/IR/OpBase.td"
#endif // OP_BASE

include "mlir/IR/BuiltinAttributes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def VarExpr_Dialect : Dialect {
  let name = "varexpr";
  let cppNamespace = "::mlir::varexpr";
  let description = [{
    What a symbol stands for, where arith cannot say it. A symbol's definition is
    an operator over operands that are either literals or other symbols, and most
    of those operators are ordinary arithmetic -- add, min, a ceiling division --
    which the IR writes with arith and reads back the same way. The rest are here:
    a range lookup, a compile-time iteration choice, and three that reinterpret an
    integer's bits as a float. They are ops rather than arith because writing them
    out of arith would spread one decision over a dozen operations and leave
    nothing to read back.
  }];
}

class VarExpr_Op<string mnemonic, list<Trait> traits = []> :
    Op<VarExpr_Dialect, mnemonic, traits> {
}

def RangeUpperBoundOp : VarExpr_Op<"range_upper_bound", [Pure]> {
  let summary = "The upper bound of the range a value falls in";
  let description = [{
    `{RANGE_UPPER_BOUND, V, low0, high0, ..., lowN, highN}`: the first `highI`
    whose `lowI <= V <= highI`. Undefined when no range matches, which
    VariableDefinition treats as an error rather than a value.

    ```mlir
    %r = varexpr.range_upper_bound %v, %low0, %high0 : index
    ```
  }];
  let arguments = (ins Index:$value, Variadic<Index>:$bounds);
  let results = (outs Index:$result);
  let assemblyFormat = "$value `,` $bounds attr-dict `:` type($result)";
}

def SelectCompileIterOp : VarExpr_Op<"select_compile_iter", [Pure]> {
  let summary = "The compile-time iteration that fits a set of input sizes";
  let description = [{
    `{SELECT_COMPILE_ITER, numSymbols, inputs..., (min, max, gran) per input per
    iteration, iterId}`: the iteration whose maxima cover the inputs with the
    least padding, breaking a tie on the largest granularity of the first input.
    The operands keep VariableDefinition's order, since the shape of the thing is
    the order.

    ```mlir
    %i = varexpr.select_compile_iter %operands : index
    ```
  }];
  let arguments = (ins Variadic<Index>:$operands);
  let results = (outs Index:$result);
  let assemblyFormat = "$operands attr-dict `:` type($result)";
}

def Uint32To16LowAsFp32Op : VarExpr_Op<"uint32_to_16low_as_fp32", [Pure]> {
  let summary = "The low 16 bits of a uint32, as the bits of the float it equals";
  let description = [{
    Takes the low 16 bits of a value that has to be in uint32 range, converts that to a
    float, and gives back the bits of that float read as an integer. A conversion
    rather than arithmetic: the result is only meaningful as bits.

    ```mlir
    %r = varexpr.uint32_to_16low_as_fp32 %v : index
    ```
  }];
  let arguments = (ins Index:$input);
  let results = (outs Index:$result);
  let assemblyFormat = "$input attr-dict `:` type($result)";
}

def Uint32To16HighAsFp32Op : VarExpr_Op<"uint32_to_16high_as_fp32", [Pure]> {
  let summary = "The high 16 bits of a uint32, as the bits of the float it equals";
  let description = [{
    Takes the high 16 bits of a value that has to be in uint32 range, converts that to a
    float, and gives back the bits of that float read as an integer. A conversion
    rather than arithmetic: the result is only meaningful as bits.

    ```mlir
    %r = varexpr.uint32_to_16high_as_fp32 %v : index
    ```
  }];
  let arguments = (ins Index:$input);
  let results = (outs Index:$result);
  let assemblyFormat = "$input attr-dict `:` type($result)";
}

def Uint32To8AsFp32Op : VarExpr_Op<"uint32_to_8_as_fp32", [Pure]> {
  let summary = "One byte of a uint32, as the bits of the float it equals";
  let description = [{
    Takes byte `$byte` of a value that has to be in uint32 range -- 0 is the
    lowest -- converts that byte to a float, and gives back the bits of that
    float read as an integer. A conversion rather than arithmetic: the result is
    only meaningful as bits. This is how a symbolic address reaches an
    index-to-address conversion, a byte at a time.

    ```mlir
    %r = varexpr.uint32_to_8_as_fp32 %v, %byte : index
    ```
  }];
  let arguments = (ins Index:$input, Index:$byte);
  let results = (outs Index:$result);
  let assemblyFormat = "$input `,` $byte attr-dict `:` type($result)";
}

#endif // DIALECTS_DIALECT_VAREXPR_OPS_TD
