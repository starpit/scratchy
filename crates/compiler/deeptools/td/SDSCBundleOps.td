//===----------------------------------------------------------------------===//
//
// IBM Confidential
// (C) Copyright IBM Corp. 2026
//
// =============================================================================
//
// This file defines the SDSCBundle operations in the MLIR operation set.
//
//===----------------------------------------------------------------------===//

#ifndef SDSCBUNDLE_OPS_TD
#define SDSCBUNDLE_OPS_TD

#ifndef OP_BASE
include "mlir/IR/OpBase.td"
#endif // OP_BASE
include "mlir/IR/EnumAttr.td"

//include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Import the SDSCBundle dialect definition
include "SDSCBundleDialect.td"
include "SDSCBundleTypes.td"

// Base class for SDSCBundle dialect operations. This operation inherits from the
// base `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class SDSCBundle_Op<string mnemonic, list<Trait> traits = []> :
    Op<SDSCBundle_Dialect, mnemonic, traits> {

}

def SDSCBundle_RegionTerminatorOp : SDSCBundle_Op<"region_terminator", [Terminator]> {
  let assemblyFormat = "attr-dict";
}

// Input argument extraction kind
def SDSCBundle_InputArgExtractKind_Value : I32EnumAttrCase<"value", 0>;
def SDSCBundle_InputArgExtractKind_Granularity : I32EnumAttrCase<"granularity", 1>;
def SDSCBundle_InputArgExtractKind_MaxValue : I32EnumAttrCase<"max_value", 2>;

def SDSCBundle_InputArgExtractKind : I32EnumAttr<"InputArgExtractKind",
    "Kind of extraction from input_arg type",
    [
       SDSCBundle_InputArgExtractKind_Value,
       SDSCBundle_InputArgExtractKind_Granularity,
       SDSCBundle_InputArgExtractKind_MaxValue
    ]> {
  let cppNamespace = "::mlir::sdscbundle";
}

// Input argument extract operation
def SDSCBundle_InputArgExtractOp : SDSCBundle_Op<"input_arg_extract", [Pure]> {
  let summary = "Extract a value or attribute from an input_arg";
  let description = [{
    The `sdscbundle.input_arg_extract` operation extracts either the underlying value
    or one of the optional attributes (granularity or max_value) from an input_arg type.

    The operation takes an input_arg operand and an extraction kind attribute that
    specifies what to extract:
    - `value`: Extracts the underlying SSA value with the input_arg's underlying type
    - `granularity`: Extracts the granularity attribute as an index value
    - `max_value`: Extracts the max_value attribute as an index value

    Examples:

    ```mlir
    // Extract the value
    %val = sdscbundle.input_arg_extract value from %M_sym : !sdscbundle.input_arg<index, granularity=3, max_value=300> -> index

    // Extract granularity
    %gran = sdscbundle.input_arg_extract granularity from %M_sym : !sdscbundle.input_arg<index, granularity=3, max_value=300> -> index

    // Extract max_value
    %max = sdscbundle.input_arg_extract max_value from %M_sym : !sdscbundle.input_arg<index, granularity=3, max_value=300> -> index
    ```

    Note: Attempting to extract granularity or max_value from an input_arg that
    doesn't have those attributes will result in a verification error.
  }];

  let arguments = (ins
    SDSCBundle_InputArgExtractKind:$kind,
    SDSCBundle_InputArgType:$input
  );

  let results = (outs AnyType:$result);

  let assemblyFormat = [{
    $kind `from` $input `:` type($input) attr-dict `->` type($result)
  }];

  let hasVerifier = 1;

  let builders = [
    // Builder that infers the result type from the input and kind
    OpBuilder<(ins "InputArgExtractKind":$kind, "Value":$input)>
  ];
}

// Device memory pool allocation
def SDSCBundle_DeviceMemAllocateOp : SDSCBundle_Op<"device_mem_allocate",
    [MemoryEffects<[MemAlloc]>]> {
  let summary = "Allocate device memory and yield its start address";
  let description = [{
    The `sdscbundle.device_mem_allocate` operation requests a contiguous buffer of
    `size` bytes of device memory and yields the address of its first byte.

    A bundle uses it to obtain the base address of the memory pool its SDSCs work
    out of, so the individual buffer addresses can be expressed as
    `pool base + offset` instead of being baked into the bundle as absolute
    addresses.

    Example:

    ```mlir
    %pool = sdscbundle.device_mem_allocate 65536 bytes : index
    %offset = arith.constant 16384 : index
    %buffer = arith.addi %pool, %offset : index
    ```
  }];

  let arguments = (ins I64Attr:$size);

  let results = (outs Index:$address);

  let assemblyFormat = [{
    $size `bytes` attr-dict `:` type($address)
  }];

  let hasVerifier = 1;
}

// SDSC execution
def SDSCBundle_SDSCExecuteOp:SDSCBundle_Op<"sdsc_execute"> {
  let summary = "Operation to execute a SuperDSC.";
  let description = [{
    The `sdscbundle.sdsc_execute` operation allows to specify execution of a program,
    identified by a sdsc file, while passing the value of symbols used inside it.
    In this way the symbol formulas that depend on loop iterations are explicitly
    visible in the mlir file, improving readability.
    for, e.g.,
    ```
      sdscbundle.sdsc_execute (%A_start_address, %B_start_address) {sdsc_filename="sdscA.json", symbol_ids=[-1, -2]}
    ```
  }];

  let arguments = (ins Variadic<Index>:$symbol_definitions,
                       StrAttr:$sdsc_filename,
                       DefaultValuedAttr<I64ArrayAttr, "{}">:$symbol_ids);
  let hasVerifier = 1;
  let assemblyFormat = [{
      `(`$symbol_definitions`)` attr-dict
  }];
  
  let extraClassDeclaration = [{
    /// Check if this operation has the correction_program attribute
    bool isCorrectionProgram();
    
    /// Check if this operation has the loop_latch_program attribute
    bool isLoopLatchProgram();

    /// Check if this operation has the dummy_program attribute
    bool isDummyProgram();

    /// Check if this operation has the loop_exit_program attribute
    bool isLoopExitProgram();
  }];
}

#endif // SDSCBUNDLE_OPS_TD