//===-- VectorChain.td -------------------------------------*- tablegen ----==//
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
// This file defines the vectorchain dialect.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_DIALECT_VECTORCHAIN_VECTORCHAIN_TD
#define DATAFLOW_SCHEDULER_DIALECT_VECTORCHAIN_VECTORCHAIN_TD

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "dataflow-scheduler/Dialect/Dataflow/DataflowInterfaces.td"
include "dataflow-scheduler/Dialect/Dataflow/DataflowTypes.td"
include "dataflow-scheduler/Dialect/VectorChain/VectorChainEnums.td"

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

def VectorChain_Dialect : Dialect {
  let name = "vectorchain";
  let cppNamespace = "::mlir::vectorchain";

  let useDefaultAttributePrinterParser = 1;

  let dependentDialects = [
    "dataflow::DataflowDialect"
  ];
}

//===----------------------------------------------------------------------===//
// Attributes
//===----------------------------------------------------------------------===//

def VectorChainBinaryOperatorAttr : EnumAttr<VectorChain_Dialect, VectorChainBinaryOperator, "binary_operator">;

def VectorChainElementWiseCompareOperatorAttr : EnumAttr<VectorChain_Dialect, VectorChainElementWiseCompareOperator, "element_wise_compare_operator">;

def VectorChainExpEstimateAttr : EnumAttr<VectorChain_Dialect, VectorChainExpEstimate, "exp_estimate">;

def VectorChainEstimateVersionsAttr : EnumAttr<VectorChain_Dialect, VectorChainEstimateVersions, "estimate_versions">;

def VectorChainScanOpEvalOrdersAttr : EnumAttr<VectorChain_Dialect, VectorChainScanOpEvalOrders, "eval_order">;

//===----------------------------------------------------------------------===//
// Operations
//===----------------------------------------------------------------------===//

// FIXME: A lot weird ODS going on here.

// Base class for VectorChain dialect operations. This operation inherits from the
// base `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class VectorChain_Op<string mnemonic, list<Trait> traits = []> :
    Op<VectorChain_Dialect, mnemonic, traits> {
  // For every standard op, there needs to be a:
  //   * void print(OpAsmPrinter &p, ${C++ class of Op} op)
  //   * LogicalResult verify(${C++ class of Op} op)
  //   * ParseResult parse${C++ class of Op}(OpAsmParser &parser,
  //                                         OperationState &result)
  // functions.
}

def VectorChain_SelectOp:VectorChain_Op<"select",
  [NoMemoryEffect, IsolatedFromAbove]> {
    let summary = "Operation to shuffle/select required data";
    let description = [{
    "This operation selects/shuffle the required data."
    ```mlir
        %data = vectorchain.select %data {selection_map} : vector<4xi8>, vector<128xi8>
    ```
    }];
    let arguments = (ins AnyTypeOf<[Dataflow_CustomVectorType, AnyVectorOfAnyRank]>:$data, AffineMapAttr:$selection_map);
    let results = (outs AnyTypeOf<[Dataflow_CustomVectorType, AnyVectorOfAnyRank]>:$selected_data);

    let assemblyFormat = [{
         $data attr-dict `:` type($data) `,` type(results)
    }];
}

def VectorChain_MultiplyOp:VectorChain_Op<"multiply",
  [NoMemoryEffect, IsolatedFromAbove, Dataflow_DebugNameOpInterface]> {
    let summary = "Element-wise multiply operation";
    let description = [{
    "This operation performs an element-wise multiply operations and reduce using the reduction map."
    ```mlir
        %data = vectorchain.multiply"(%op1, %op2) [%mask] {#reduction_map}
            : vector<128xi8>, vector<128xi8>, vector<64xi8>
    ```
    }];
    let arguments = (ins AnyTypeOf<[Dataflow_CustomVectorType, AnyVectorOfAnyRank]>:$op1,
                         AnyTypeOf<[Dataflow_CustomVectorType, AnyVectorOfAnyRank]>:$op2,
                         Optional<AnyVectorOfAnyRank>:$mask,
                         OptionalAttr<StrAttr>:$dbgName, AffineMapAttr:$reduction_map);
    let results = (outs AnyTypeOf<[Dataflow_CustomVectorType, AnyVectorOfAnyRank]>:$data);
    let assemblyFormat = [{
          $op1 `,` $op2 (`[` $mask^ `:` type($mask) `]`)?   attr-dict `:` type($op1) `,` type($op2) `,` type(results)
    }];
}

def VectorChain_BinaryOp:VectorChain_Op<"binary",
  [NoMemoryEffect, IsolatedFromAbove, Dataflow_DebugNameOpInterface]> {
    let summary = "Element-wise binary operation";
    let description = [{
    Takes two operand vectors, performs the element-wise operation named by the
    `binary_op` attribute (for example `multiply`, `and`, `or`, `min`), and
    reduces the result. `op_specific_map` is an affine map describing the
    operation-specific reduction/indexing. The optional `mask` selects which
    lanes of the result participate in the computation.

    ```mlir
        %data = vectorchain.binary"(%op1, %op2) [%mask] {#op_specific_map}
            : vector<128xi8>, vector<128xi8>, vector<128xi8>
    ```
    }];
    let arguments = (ins AnyTypeOf<[Dataflow_CustomVectorType, AnyVectorOfAnyRank]>:$op1,
                         AnyTypeOf<[Dataflow_CustomVectorType, AnyVectorOfAnyRank]>:$op2,
                         Optional<AnyVectorOfAnyRank>:$mask,
                         OptionalAttr<StrAttr>:$dbgName,
                         VectorChainBinaryOperatorAttr:$binary_op,
                         AffineMapAttr:$op_specific_map);
    let results = (outs AnyTypeOf<[Dataflow_CustomVectorType, AnyVectorOfAnyRank]>:$data);
    let assemblyFormat = [{
          $op1 `,` $op2  (`[` $mask^ `:` type($mask) `]`)?   attr-dict `:` type($op1) `,` type($op2) `,` type(results)
    }];
}

def VectorChain_ElementWiseCompareOp:VectorChain_Op<"element_wise_compare",
  [NoMemoryEffect, IsolatedFromAbove, Dataflow_DebugNameOpInterface]> {
    let summary = "Element-wise compare operation";
    let description = [{
    "This operation performs a element-wise compare operation"
    }];
    let arguments = (ins AnyVectorOfAnyRank:$op1, AnyVectorOfAnyRank:$op2, Optional<VectorOfRankAndType<[1], [I1]>>:$mask,
                         OptionalAttr<StrAttr>:$dbgName,
                         VectorChainElementWiseCompareOperatorAttr:$compare_op);
    let results = (outs AnyVectorOfAnyRank:$data);
    let assemblyFormat = [{
          $op1 `,` $op2  (`[` $mask^ `:` type($mask) `]`)?   attr-dict `:` type($op1) `,` type($op2) `,` type(results)
    }];
}

def VectorChain_MergeOp:VectorChain_Op<"merge",
  [NoMemoryEffect, IsolatedFromAbove, Dataflow_DebugNameOpInterface]> {
    let summary = "Merge operation";
    let description = [{
    "This operation performs a merge between two vectors based on the polyhedral mini-scop representation (where the order doesn't matter)."
    }];

    let arguments = (ins AnyTypeOf<[Dataflow_CustomVectorType, AnyVectorOfAnyRank]>:$op1, AnyTypeOf<[Dataflow_CustomVectorType, AnyVectorOfAnyRank]>:$op2,
                         OptionalAttr<StrAttr>:$dbgName,
                         Builtin_IntegerSetAttr:$iteration_space_CA,
                         AffineMapAttr:$access_function_CA,
                         AffineMapAttr:$access_function_A,
                         Builtin_IntegerSetAttr:$iteration_space_CB,
                         AffineMapAttr:$access_function_CB,
                         AffineMapAttr:$access_function_B
                         );

    let results = (outs AnyTypeOf<[Dataflow_CustomVectorType, AnyVectorOfAnyRank]>:$data);
    let assemblyFormat = [{
          $op1 `,` $op2    attr-dict `:` type($op1) `,` type($op2) `,` type(results)
    }];
}

def VectorChain_CreateAffineMaskOp:VectorChain_Op<"create_affine_mask",
  [NoMemoryEffect, IsolatedFromAbove]> {
    let summary = "Create affine mask operation";
    let description = [{"Create affine mask operation"}];

    let arguments = (ins Optional<Index>:$mask_parameter,
                         Builtin_IntegerSetAttr:$mask_set);

    let results = (outs VectorOfRankAndType<[1], [I1]>:$data);
    let assemblyFormat = [{
        ($mask_parameter^ )? attr-dict `:` type(results)
    }];
}

def VectorChain_PackOp:VectorChain_Op<"pack",
  [NoMemoryEffect, IsolatedFromAbove, Dataflow_DebugNameOpInterface]> {
    let summary = "Pack operation";
    let description = [{
    "This operation packs two vectors into a one based on the array of indices."
    }];

    let arguments = (ins AnyTypeOf<[Dataflow_CustomVectorType, AnyVectorOfAnyRank]>:$op1,
                         AnyTypeOf<[Dataflow_CustomVectorType, AnyVectorOfAnyRank]>:$op2,
                         Optional<AnyVectorOfAnyRank>:$mask,
                         OptionalAttr<StrAttr>:$dbgName,
                         ArrayAttr:$indices,
                         IndexAttr:$repetition,
                         DefaultValuedAttr<BoolAttr, "false">:$sign_extend);

    let results = (outs AnyTypeOf<[Dataflow_CustomVectorType, AnyVectorOfAnyRank]>:$data);
    let assemblyFormat = [{
          $op1 `,` $op2 (`[` $mask^ `:` type($mask) `]`)?  attr-dict `:` type($op1) `,` type($op2) `,` type(results)
    }];
}

def VectorChain_CastOp:VectorChain_Op<"cast",
  [NoMemoryEffect, IsolatedFromAbove, Dataflow_DebugNameOpInterface]> {
    let summary = "Cast Operation";
    let description = [{
    "This operation casts a vector of one type to a vector of another."
    }];

    let arguments = (ins AnyTypeOf<[Dataflow_CustomVectorType, AnyVectorOfAnyRank]>:$input, Optional<AnyVectorOfAnyRank>:$mask, OptionalAttr<StrAttr>:$dbgName);
    let results = (outs AnyTypeOf<[Dataflow_CustomVectorType, AnyVectorOfAnyRank]>:$data);
    let assemblyFormat = [{
          $input (`[` $mask^ `:` type($mask) `]`)?  attr-dict `:` type($input) `,` type(results)
    }];
    let hasVerifier = 1;
}

def VectorChain_FastExpOp:VectorChain_Op<"fast_exp",
  [NoMemoryEffect, IsolatedFromAbove, Dataflow_DebugNameOpInterface]> {
    let summary = "Fast Exponent Operation";
    let description = [{
    "This operation computes fast exp auxiliary function."
    }];

    let arguments = (ins AnyVectorOfAnyRank:$op, Optional<VectorOfRankAndType<[1], [I1]>>:$mask, OptionalAttr<StrAttr>:$dbgName);
    let results = (outs AnyVectorOfAnyRank:$data);
    let assemblyFormat = [{
          $op (`[` $mask^ `:` type($mask) `]`)?  attr-dict `:` type($op) `,` type(results)
    }];
}

def VectorChain_ExpEstimateOp:VectorChain_Op<"exp_estimate",
  [NoMemoryEffect, IsolatedFromAbove, Dataflow_DebugNameOpInterface]> {
    let summary = "Estimate operation";
    let description = [{
    "This operation estimates exp auxiliary function."
    }];

    let arguments = (ins AnyVectorOfAnyRank:$op, Optional<VectorOfRankAndType<[1], [I1]>>:$mask, OptionalAttr<StrAttr>:$dbgName,
                         VectorChainExpEstimateAttr:$version);

    let results = (outs AnyVectorOfAnyRank:$data);
    let assemblyFormat = [{
          $op (`[` $mask^ `:` type($mask) `]`)?  attr-dict `:` type($op) `,` type(results)
    }];
}

def VectorChain_FloorOp:VectorChain_Op<"floor",
  [NoMemoryEffect, IsolatedFromAbove, Dataflow_DebugNameOpInterface]> {
    let summary = "Floor operation";
    let description = [{
    "This operation computes the floor function."
    }];

    let arguments = (ins AnyVectorOfAnyRank:$op, Optional<VectorOfRankAndType<[1], [I1]>>:$mask, OptionalAttr<StrAttr>:$dbgName);

    let results = (outs AnyVectorOfAnyRank:$data);
    let assemblyFormat = [{
          $op (`[` $mask^ `:` type($mask) `]`)?  attr-dict `:` type($op) `,` type(results)
    }];
}

def VectorChain_RecEstimateOp:VectorChain_Op<"rec_estimate",
  [NoMemoryEffect, IsolatedFromAbove, Dataflow_DebugNameOpInterface]> {
    let summary = "Estimate operation";
    let description = [{
    "This operation estimates rec auxiliary function."
    }];

    let arguments = (ins AnyVectorOfAnyRank:$op, Optional<VectorOfRankAndType<[1], [I1]>>:$mask, OptionalAttr<StrAttr>:$dbgName);

    let results = (outs AnyVectorOfAnyRank:$data);
    let assemblyFormat = [{
          $op (`[` $mask^ `:` type($mask) `]`)?  attr-dict `:` type($op) `,` type(results)
    }];
}

def VectorChain_LnEstimateOp:VectorChain_Op<"ln_estimate",
  [NoMemoryEffect, IsolatedFromAbove, Dataflow_DebugNameOpInterface]> {
    let summary = "Estimate operation";
    let description = [{
    "This operation estimates ln auxiliary function."
    }];

    let arguments = (ins AnyVectorOfAnyRank:$op, Optional<VectorOfRankAndType<[1], [I1]>>:$mask, OptionalAttr<StrAttr>:$dbgName);

    let results = (outs AnyVectorOfAnyRank:$data);
    let assemblyFormat = [{
          $op (`[` $mask^ `:` type($mask) `]`)?  attr-dict `:` type($op) `,` type(results)
    }];
}

def VectorChain_RsqrtEstimateOp:VectorChain_Op<"rsqrt_estimate",
  [NoMemoryEffect, IsolatedFromAbove, Dataflow_DebugNameOpInterface]> {
    let summary = "Estimate operation";
    let description = [{
    "This operation estimates rsqrt auxiliary function."
    }];

    let arguments = (ins AnyVectorOfAnyRank:$op, Optional<VectorOfRankAndType<[1], [I1]>>:$mask, OptionalAttr<StrAttr>:$dbgName);

    let results = (outs AnyVectorOfAnyRank:$data);
    let assemblyFormat = [{
          $op (`[` $mask^ `:` type($mask) `]`)?  attr-dict `:` type($op) `,` type(results)
    }];
}

def VectorChain_SigmoidEstimateOp:VectorChain_Op<"sigmoid_estimate",
  [NoMemoryEffect, IsolatedFromAbove, Dataflow_DebugNameOpInterface]> {
    let summary = "Estimate operation";
    let description = [{
    "This operation estimates sigmoid auxiliary function."
    }];

    let arguments = (ins AnyVectorOfAnyRank:$op, Optional<VectorOfRankAndType<[1], [I1]>>:$mask, OptionalAttr<StrAttr>:$dbgName,
                         VectorChainEstimateVersionsAttr:$version);

    let results = (outs AnyVectorOfAnyRank:$data);
    let assemblyFormat = [{
          $op (`[` $mask^ `:` type($mask) `]`)?  attr-dict `:` type($op) `,` type(results)
    }];
}

def VectorChain_TanhEstimateOp:VectorChain_Op<"tanh_estimate",
  [NoMemoryEffect, IsolatedFromAbove, Dataflow_DebugNameOpInterface]> {
    let summary = "Estimate operation";
    let description = [{
    "This operation estimates tanh auxiliary function."
    }];

    let arguments = (ins AnyVectorOfAnyRank:$op, Optional<VectorOfRankAndType<[1], [I1]>>:$mask, OptionalAttr<StrAttr>:$dbgName,
                         VectorChainEstimateVersionsAttr:$version);

    let results = (outs AnyVectorOfAnyRank:$data);
    let assemblyFormat = [{
          $op (`[` $mask^ `:` type($mask) `]`)?  attr-dict `:` type($op) `,` type(results)
    }];
}

def VectorChain_NegOp:VectorChain_Op<"neg",
  [NoMemoryEffect, IsolatedFromAbove, Dataflow_DebugNameOpInterface]> {
    let summary = "Negation operation";
    let description = [{
    "This operation negates the input SSA variable."
    }];

    let arguments = (ins AnyVectorOfAnyRank:$op, Optional<VectorOfRankAndType<[1], [I1]>>:$mask,
                         OptionalAttr<StrAttr>:$dbgName);

    let results = (outs AnyVectorOfAnyRank:$data);
    let assemblyFormat = [{
          $op (`[` $mask^ `:` type($mask) `]`)?  attr-dict `:` type($op) `,` type(results)
    }];
}

def VectorChain_MultiplyAndAccumulateOp:VectorChain_Op<"multiply_and_accumulate",
  [NoMemoryEffect, IsolatedFromAbove, Dataflow_DebugNameOpInterface]> {
    let summary = "Multiply and accumulate operation";
    let description = [{
    "This operation performs a mac operations -- op1 x op2 + op3 with reduction map."
    ```mlir
    %data = vectorchain.multiply_and_accumulate(%op1, %op2, %op3) [%mask] {#reduction_map}
            : vector<128xi8>, vector<128xi8>, vector<256xi8>, vector<64xi8>
    ```
    }];
    let arguments = (ins AnyTypeOf<[Dataflow_CustomVectorType, AnyVectorOfAnyRank]>:$op1, AnyTypeOf<[Dataflow_CustomVectorType, AnyVectorOfAnyRank]>:$op2,
                         AnyTypeOf<[Dataflow_CustomVectorType, AnyVectorOfAnyRank]>:$op3, Optional<AnyVectorOfAnyRank>:$mask,
                         OptionalAttr<StrAttr>:$dbgName,
                         AffineMapAttr:$reduction_map);
    let results = (outs AnyTypeOf<[Dataflow_CustomVectorType, AnyVectorOfAnyRank]>:$data);
    let assemblyFormat = [{
          $op1 `,` $op2 `,` $op3 (`[` $mask^ `:` type($mask) `]`)?  attr-dict `:` type($op1) `,` type($op2) `,` type($op3) `,` type(results)
    }];
}

def VectorChain_ElementWiseSelectionOp:VectorChain_Op<"element_wise_selection",
  [NoMemoryEffect, IsolatedFromAbove, Dataflow_DebugNameOpInterface]> {
    let summary = "Element wise selection";
    let description = [{
    "This operation performs a element-wise selection b/w lhs and rhs operands based on the condition."
    }];
    let arguments = (ins AnyVectorOfAnyRank:$cond, AnyTypeOf<[Dataflow_CustomVectorType, AnyVectorOfAnyRank]>:$lhs,
                         AnyTypeOf<[Dataflow_CustomVectorType, AnyVectorOfAnyRank]>:$rhs, Optional<AnyVectorOfAnyRank>:$mask,
                         OptionalAttr<StrAttr>:$dbgName);
    let results = (outs AnyTypeOf<[Dataflow_CustomVectorType, AnyVectorOfAnyRank]>:$data);
    let assemblyFormat = [{
          $cond `?` $lhs `:` $rhs (`[` $mask^ `:` type($mask) `]`)?  attr-dict `:` type($cond) `,` type($lhs) `,` type($rhs) `,` type(results)
    }];
}

def VectorChain_ConstantBitstreamOp:VectorChain_Op<"constant_bitstream",
  [NoMemoryEffect, IsolatedFromAbove]> {
    let summary = "Constant bitstream operation";
    let description = [{
    "This operation interprets a constant bit stream as a value with the given vector type.

    The value is interpretted as a signed 64 bit integer.

    The number of elements of the value attribute should match the number of elements in the
    result vector.

    The bitwidth of values specified by the value attribute should not exceed the bitwidth
    of the output element type."
    }];

    let arguments = (ins ArrayAttr:$value);

    let results = (outs AnyTypeOf<[Dataflow_CustomVectorType, AnyVectorOfAnyRank]>:$result);

    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;
}

def VectorChain_ShuffleOp:VectorChain_Op<"shuffle",
  [NoMemoryEffect, IsolatedFromAbove, AttrSizedOperandSegments,
   Dataflow_DebugNameOpInterface]> {
    let summary = "Shuffle operation";
    let description = [{
    "This operation shuffles the elements of a vector based on the array of indices.

    Each element of `indices` refers to an element of the input vector. The
    `indices` pattern is repeated `repetition` times, so the product of the
    number of `indices` and `repetitions` must match the number of elements in
    the output.

    Negative values indicate indices of the variable and pad operands, starting at -1.

    Examples:
      %0 = vectorchain.shuffle input(%inp), variable(%arg0, %arg1, %arg2), pad(%c0)
               { indices: [-1 : i32, -2 : i32, -3 : i32, -4 : i32], repetition = 16 : i32 } :
               vector<64xf16>, vector<3xindex>, index, vector<64xf16>

      %0 = vectorchain.shuffle input(%inp), pad(%c0, %c1)
               { indices: [0 : i32, 1 : i32, -1 : i32, -2 : i32], repetition = 16 : i32 } :
               vector<64xf16>, index, index, vector<64xf16>
    "
    }];

    let arguments = (ins AnyTypeOf<[Dataflow_CustomVectorType, AnyVectorOfAnyRank]>:$input,
                         Variadic<AnyTypeOf<[Index, AnyInteger, AnyFloat]>>:$variable,
                         Variadic<AnyTypeOf<[Index, AnyInteger, AnyFloat]>>:$pad,
                         Optional<AnyVectorOfAnyRank>:$mask,
                         OptionalAttr<StrAttr>:$dbgName,
                         ArrayAttr:$indices,
                         I32Attr:$repetition);

    let results = (outs AnyTypeOf<[Dataflow_CustomVectorType, AnyVectorOfAnyRank]>:$output);
    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;

    let builders = [
      OpBuilder<(ins "Type":$resultType, "Value":$input, "ValueRange":$variable,
                     "ValueRange":$pad, "Value":$mask, "ArrayAttr":$indices,
                     "IntegerAttr":$repetition, "StringAttr":$dbgName)>,
      OpBuilder<(ins "Type":$resultType, "Value":$input, "ArrayAttr":$indices,
                     "IntegerAttr":$repetition, "StringAttr":$dbgName)>,
      OpBuilder<(ins "Type":$resultType, "Value":$input, "Value":$mask,
                     "ArrayAttr":$indices, "IntegerAttr":$repetition,
                     "StringAttr":$dbgName)>,
      OpBuilder<(ins "Type":$resultType, "Value":$input, "ValueRange":$variable,
                     "ValueRange":$pad, "ArrayAttr":$indices,
                     "IntegerAttr":$repetition, "StringAttr":$dbgName)>
    ];

    let extraClassDeclaration = [{
      std::optional<int64_t> getFirstVariableIndex() {
        if (getVariable().empty())
          return std::nullopt;
        return -1;
      }

      std::optional<int64_t> getFirstPadIndex() {
        if (getPad().empty())
          return std::nullopt;
        if (!getVariable().empty())
          return -static_cast<int64_t>(getVariable().size()) - 1;
        return -1;
      }

      Value getVariableOrPadOperand(int64_t index) {
        auto variable = getVariable();
        auto pad = getPad();
        auto operand_offset = -index - 1;
        auto total_operands = variable.size() + pad.size();

        assert(index < 0 &&
               static_cast<size_t>(operand_offset) < total_operands);

        if (static_cast<size_t>(operand_offset) < variable.size())
          return variable[operand_offset];
        return pad[operand_offset - variable.size()];
      }
    }];
}

def VectorChain_ScanWithGapOp:VectorChain_Op<"scan_with_gap",
  [NoMemoryEffect, IsolatedFromAbove, Dataflow_DebugNameOpInterface]> {
    let summary = "Scan (reduction) operation with gap";
    let description = [{
    "This operation performs reduction using the given binary operation
    in reduction_op attribute. The gap attribute shows how many element
    one needs to jump to get to the next element. The eval_order is giving
    of reduction.
    Mathematical definition of the operation on n elements for eval_order=left-to-right:
    O[i] = sum {I[i - k * gap] | k = 0, ..., floor_div(i, gap)} for i=0,1,...,n
    "
    }];

    let arguments = (ins AnyTypeOf<[Dataflow_CustomVectorType, AnyVectorOfAnyRank]>:$input,
                         OptionalAttr<StrAttr>:$dbgName,
                         VectorChainBinaryOperatorAttr:$reduction_op,
                         IndexAttr:$gap,
                         VectorChainScanOpEvalOrdersAttr:$eval_order);

    let results = (outs AnyTypeOf<[Dataflow_CustomVectorType, AnyVectorOfAnyRank]>:$output);
    let assemblyFormat = [{
          $input attr-dict `:` type($input) `,` type(results)
    }];
}

def VectorChain_RotateOp:VectorChain_Op<"rotate", [
        NoMemoryEffect,
        IsolatedFromAbove,
        Dataflow_DebugNameOpInterface]> {
    let summary = "Rotation operation";
    let description = [{
        "This operation rotates the elements of a vector from a given location either left
        or right depending on the right_shift attribute. The operation views the vector
        as a circular array. If its a mult-dimensional array, then this operation views
        the vector in a linearized form and performs the operation."
    }];

    let arguments = (ins AnyTypeOf<[Dataflow_CustomVectorType, AnyVectorOfAnyRank]>:$input,
                       Index:$position,
                       OptionalAttr<StrAttr>:$dbgName,
                       DefaultValuedAttr<BoolAttr, "true">:$right_shift);

    let results = (outs AnyTypeOf<[Dataflow_CustomVectorType, AnyVectorOfAnyRank]>:$output);
    let assemblyFormat = [{
        $input `,` $position attr-dict `:` type($input) `,` type($position) `,` type(results)
    }];
}

#endif // DATAFLOW_SCHEDULER_DIALECT_VECTORCHAIN_VECTORCHAIN_TD
