//===-- DdlOps.td -- DdlOps Dialect Operation Definitions -*- tablegen ----==//
//
// IBM Confidential
// (C) Copyright IBM Corp. 2023, 2025
//
// =============================================================================
//
// This file defines the ddl operations in the MLIR operation set.
//
//===----------------------------------------------------------------------===//

#ifndef DDC_DDL_DIALECT_DDL_OPS_TD
#define DDC_DDL_DIALECT_DDL_OPS_TD

include "DdlTypes.td"

include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// ops for definitions section
def YieldOp : Ddl_Op<"yield", [NoMemoryEffect, Terminator, ReturnLike,
    MemRefsNormalizable]> {
  let summary = "Yield values to parent operation";
  let description = [{
    "ddl.yield yields zero or more SSA values to parent op."
  }];

  let arguments = (ins Variadic<AnyType>:$operands);
  let builders = [
    OpBuilder<(ins), [{ build($_builder, $_state, ValueRange{}); }]>
  ];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

// Require regions to have ddl.yield.
def ImplicitDdlTerminator
    : SingleBlockImplicitTerminator<"YieldOp">;


def Ddl_DimensionOp:Ddl_Op<"dimension"> {
    let summary = "dimension";
    let description = [{
        This operation creates one or multiple dimensions to be used throughout the ddl template.
        The optional attribute `dim_property` enables the dimension(s) to be
        used in the specified meta context (window or padding-related
        operation, for example).
        Accepted values for `dim_property`:
          - unpadded   : default
          - window     : the dimension can be used in a window operation
          - stride     : the dimension can be used in a stride operation
          - dilation   : the dimension can be used in a dilation operation
          - pad_front  : the dimension can be used in a padding operation that
                         refers to the front-padding
          - pad_back   : the dimension can be used in a padding operation that
                         refers to the back-padding
          - pad_valid  : the dimension can be used in an operation that refers
                         to the valid part of a padded dimension

        ```mlir
        %d:4 = ddl.dimension {} : index, index, index, index
        %wrd = ddl.dimension {} : index
        // These dimensions can be used in a window operation context
        %w:2 = ddl.dimension { dim_property = "window" } : index, index
        // These dimensions can be used in a front-padding related context
        %pf:2 = ddl.dimension { dim_property = "pad_front" } : index, index
        ```
    }];
  let arguments = (ins DefaultValuedAttr<StrAttr, "\"unpadded\""> : $dim_property);
  let results = (outs Variadic<Index>);
  let assemblyFormat = [{ attr-dict `:` type(results)  }];
}

def Ddl_PaddedDimensionOp:Ddl_Op<"padded_dimension",[AttrSizedOperandSegments]> {
    let summary = "padded_dimension";
    let description = [{
        This operation creates the padded version of a non-padded dimension.
        This operation takes an optional list of dimensions. The list
          - must include a non-padded dimension and
          - may include window and padding dimensions.
        The result is the padded dimension that correspond to the input
        non-padded dimension.
        When specified, the operation also sets up the window and
        padding dimensions based on the input non-padded dimension.
        ```mlir
        %d0pad = ddl.padded_dimension(primary = %d0)
        %d1pad = ddl.padded_dimension(primary = %d1, padding=[%d1_pad_front, %d1_pad_back, %d1_pad_valid], window=[%d1_window, %d1_stride, %d1_dilation])

        ```
    }];
  let arguments = (ins Index:$primary_dim, Variadic<Index>:$padding_dim, Variadic<Index>:$window_dim );
  let results = (outs Index);
  let assemblyFormat = [{
    `(`
        `primary` `=` $primary_dim
         `,` `padding` `=` `[` $padding_dim `]` 
         `,` `window` `=` `[` $window_dim `]` 
    `)` attr-dict
  }];
}

def Ddl_LayoutOp:Ddl_Op<"layout"> {
    let summary = "tensor layout";
    let description = [{
        This operation creates a layout that can be used as slice/stick/global layout in a ddl.tensor op.
        It has an optional attribute is_order_fixed (default false) to specify if the order of dimensions specified in the op has to be respected or they can be reordered.
        ```mlir
        %l = ddl.layout(%d1, %d2, %wrd#1) {is_order_fixed = true}
        ```
    }];
    let arguments = (ins DefaultValuedAttr<BoolAttr, "false"> : $is_order_fixed, Variadic<Index>:$dimensions);
    let results = (outs Index);
    let assemblyFormat = [{
        `(`$dimensions`)` `{` oilist(`is_order_fixed` `=` $is_order_fixed) `}` attr-dict
    }];
}


def Ddl_DatatypeOp : Ddl_Op<"type"> {
    let summary = "data type";
    let description =[{
        This Operation creates a type, specified with a name from the DataFormats enum in Deeptools.
        Optionally it is possible to override the default bit-width.
        ```mlir
        %type_fp16 = ddl.type {data_type="SEN169_FP16"}
        %type_fp8_wide = ddl.type {data_type="SEN143_FP8", bit_width=16}
        ```
    }];
    let hasVerifier = 1;
    let arguments = (ins OptionalAttr<SI64Attr> : $bit_width, StrAttr : $data_type);
    let results = (outs Index);
    let assemblyFormat = [{
        `{` `data_type` `=` $data_type (`,` `bit_width` `=` $bit_width^ )? `}` attr-dict
    }];
}

def Ddl_TensorOp:Ddl_Op<"tensor"> {
    let summary = "tensor";
    let description = [{
        This operation creates one or multiple tensors with same layout and type.
        The layout of a tensor is specified through three layout ops, one that defines the layout in the slice,
        one that defines how slices are composed to build a stick, and one that defines how sticks are ordered in the global layout of the tensor.
        ```mlir
        %inptensor, %outtensor = ddl.tensor(%slice_layout, %stick_layout, %global_layout_all, %type_fp16) : index, index
        ```
    }];
  let arguments = (ins Index:$slice_layout, Index:$stick_layout, Index:$global_layout, Variadic<Index>:$type);
  let results = (outs Variadic<Index>);
  let assemblyFormat = [{ `(`$slice_layout `,` $stick_layout `,` $global_layout `,` `[` $type `]` `)` attr-dict `:` type(results)  }];
}

def Ddl_InternalTensorOp:Ddl_Op<"internal_tensor"> {
    let summary = "internal_tensor";
    let description = [{
        This operation creates one or multiple tensors for use in the dataflow as temporary or intermediate results.
        The layout is copied from the `reference_tensor`, while the type can be customized. Local tensors can then be used
        as `interim` tensors in operation_bind ops.
        ```mlir
        %ptsum = ddl.tensor(%outtensor, %type_int24) : index, index
        ```
    }];
  let arguments = (ins Index:$reference_tensor, Variadic<Index>:$type);
  let results = (outs Variadic<Index>);
  let assemblyFormat = [{ `(`$reference_tensor `,` `[` $type `]` `)` attr-dict `:` type(results)  }];
}

def Ddl_AliasOneTensorOfOp : Ddl_Op<"alias_one_tensor_of"> {
    let summary = "alias_one_tensor_of";
    let description = [{
        This operation creates an alias of a tensor. This is very useful when we need to define multiple versions of a tensor,
        usually because we are trying to capture multiple precisions of the same operation in the same ddl template (e.g. bmmint4, bmmint8,...)
        but all these tensors have all the same role in the dataflow.
        ```mlir
        %inptensor = ddl.alias_one_tensor_of(%inptensor_int8, %inptensor_fp16, %inptensor_fp8, %inptensor_int4)
        ```
    }];
    let arguments = (ins Variadic<Index>:$tensors);
    let results = (outs Index);
    let assemblyFormat = [{
        `(`$tensors`)` attr-dict
    }];

}

def Ddl_AliasOneConstantOfOp : Ddl_Op<"alias_one_constant_of"> {
    let summary = "alias_one_constant_of";
    let description = [{
        This operation creates an alias of a constant. This is very useful when we need to define multiple versions of a constant,
        usually because we are trying to capture multiple precisions of the same operation in the same ddl template (e.g. bmmint4, bmmint8,...)
        but all these constants have all the same role in the dataflow.
        ```mlir
        %ffff_const = ddl.alias_one_constant_of(%ffff_const_fp16, %ffff_const_fp32)
        ```
    }];
    let arguments = (ins Variadic<Index>:$constants);
    let results = (outs Index);
    let assemblyFormat = [{
        `(`$constants`)` attr-dict
    }];
}

def Ddl_OperationBindOp : Ddl_Op<"operation_bind", [AttrSizedOperandSegments]> {
    let summary = "operation bind";
    let description = [{
        This create a binding to a DSC operation, identified by `opFuncName`.
        The tensors involved in the operation needs to be passed as input to this op and they need to match what the DSC captures.
        Inputs and outputs are mandatory, while interim tensors (like ptsum) are optional.
        The `required` attribute allows to enforce the presence of the op in the DSC to allow this ddl template to match.
        ```mlir
        %bmm_int8_op = ddl.operation_bind([%inptensor_int8, %kertensor_int8], [%outtensor]) {opFuncName="batchmatmulint8", required=false}
        ```
    }];
    let hasVerifier = 1;
    let arguments = (ins StrAttr: $opFuncName, BoolAttr : $required, Variadic<Index> : $dataFormats, Variadic<Index> : $outputs, Variadic<Index>: $inputs, Variadic<Index>: $interim);
    let results = (outs Index);
    let assemblyFormat =  [{
        `(` `[`$dataFormats`]` `,` `[`$inputs`]` `,` `[`$outputs`]` (`,` `[`$interim^`]`)? `)`  attr-dict
    }];
}

def Ddl_ConstraintOp : Ddl_Op<"constraint"> {
    let summary = "constraint";
    let description = [{
        This operation checks a constraint among its operands by looking at the property.
        The op can be used to capture several different types of constrants, as in the examples below.
        If a constraint is not met, the ddl template is considered unsuitable for the input DSC, and DDC will
        need to look for a different template to match (if available).
        ```mlir
        ddl.constraint(%input, %shift, %scale) {property = "slice", cmp = "equal"}
        ddl.constraint(%input, %shift, %scale, %output) {property = "stick", cmp = "equal"}
        ddl.constraint(%inptensor_int8, %inptensor_fp8) {property = "slice", dim_idx = 0, cmp = "equal", value = 8}
        ddl.constraint() {min_num_cores = 1}
        ddl.constraint(%bmm_int8_mbkg3_op, %bmm_int8_op, %bmm_fp16_op) {min_num_valid = 1, max_num_valid = 1}
        ```
    }];
    let arguments = (ins OptionalAttr<I64Attr>:$min_num_cores,  OptionalAttr<StrAttr>:$property, OptionalAttr<StrAttr>:$cmp,
                         OptionalAttr<I64Attr>:$dim_idx, OptionalAttr<I64Attr>:$value,
                         OptionalAttr<I64Attr>:$min_num_valid, OptionalAttr<I64Attr>:$max_num_valid, OptionalAttr<BoolAttr>:$relative_op_order, Variadic<Index>: $operands);
    let assemblyFormat = [{
        `(`$operands`)` attr-dict
    }];
}

def Ddl_GetExternalDataTransferAllocationOp : Ddl_Op<"get_external_data_transfer_allocation"> {
    let summary = "get_external_data_transfer_allocation";
    let description = [{
        This op references how an external data structure is provided to us.
        We need to specify the tensor we are referring to and the memory in which we expect it to be.
        The op provides us the handler to the memory allocation for this data structure, to use in the rest of the ddl template.
        Additionally we need to provide the data_connect label for the external transfer that brings the data in or out.
        ```mlir
        %inptensor_lx_allocation = ddl.get_external_data_transfer_allocation (%inptensor) {memory="lx", data_connect="l3_lx_input"}
        ```
    }];
    let arguments = (ins StrAttr:$memory, StrAttr:$data_connect, Index:$tensor);
    let results = (outs Index);
    let assemblyFormat = [{
        `(` $tensor `)` `{` `memory` `=` $memory `,` `data_connect` `=` $data_connect `}` attr-dict
    }];
}

def Ddl_GetExternalConstantOp : Ddl_Op<"get_external_constant"> {
    let summary = "get_external_constant";
    let description = [{
        This op works similarly to a tensor op, but instead it refers to a constant value coming through the DSC.
        The constant is identified by its name, and we additionally need to provide type and number of elements.
        ```mlir
        %leak_const = ddl.get_external_constant(%type_fp16){name="leakconst", num_elements=1}
        ```
    }];
    let arguments = (ins Variadic<Index>:$datatype, StrAttr:$name, I64Attr : $num_elements);
    let results = (outs Index);
    let assemblyFormat = [{
        `(`$datatype`)``{` `name` `=` $name `,` `num_elements` `=` $num_elements`}` attr-dict
    }];
}

def Ddl_DefineConstantOp : Ddl_Op<"define_constant"> {
    let summary = "define_constant";
    let description = [{
        This op defines a new data constant, not provided by DSC.
        The constant value can be give in decimal or hexadecimal (with 0x) but it should be already encoded in sentient format.
        Multiple values can be provided in case the constant changes across a stick. We additionally need to provide the type.
        ```mlir
        %one_const = ddl.define_constant(%type_fp16){name="oneConst", value=[0x3e00]}
        ```
    }];
    let arguments = (ins Index:$datatype, OptionalAttr<StrAttr>:$name, ArrayAttr:$value);
    let results = (outs Index);
    let assemblyFormat = [{
        `(`$datatype`)` attr-dict
    }];
}

def Ddl_OperandConstantOp : Ddl_Op<"operand_constant"> {
    let summary = "operand_constant";
    let description = [{
        This op is used to substitute a unit op whenever we want to use a constant operand such as 0.0 or 1.0 in FMA or similar.
        The name needs to correspond to something allowed by the ISA.
        ```mlir
        %zero_const = ddl.operand_constant{name="0.0"}
        ```
    }];
    let arguments = (ins StrAttr:$name);
    let results = (outs Index);
    let assemblyFormat = [{
        attr-dict
    }];
}



// Ops for dataflow sections
def Ddl_DataflowOp:Ddl_Op<"dataflow", [ImplicitDdlTerminator]> {
    let summary = "Dataflow";
    let description = [{
        This operation is the root of the entire datalflow section of the ddl template.
        ```mlir
        dataflow {
            
        }
        ```
    }];
    let regions = (region AnyRegion:$body);
    let assemblyFormat = [{
       $body attr-dict
    }];
}

def Ddl_GetExternalDatastageOp:Ddl_Op<"get_external_datastage"> {
    let summary = "get_external_datastage";
    let description = [{
        This op allows us to reference a datastage from DSC.
        We can choose between `core` or `chunk`.
        ```mlir
        %b_datastage = get_external_datastage {property = "core"}
        ```
    }];
    let arguments = (ins StrAttr: $property);
    let results = (outs Index);
    let assemblyFormat = [{
        `{``property` `=` $property `}` attr-dict
    }];
}

def Ddl_DatastageOp : Ddl_Op<"datastage"> {
    let summary = "datastage";
    let description = [{
        This op creates a new datastage. We can specify if the strategy is to minimize or maximize its size.
        Every datastage has natural upper and lower bounds defined by the datastages used above and below it.
        The order among datastages is determined automatically from their use in loops.
        An additional natural lower bound is provided by the minimum size of the transfers and computes that sit inside loops that use that datastage as denominator.
        In case of maximize strategy, an additional natural upper bound comes from memory sizes. Allocate op locations link
        tensor memory footprints to datastages and therefore the max size of a memory can limit the growth of a datastage.
        If maximization of the datastage is crucial for performance (e.g. interleaving) we can additionally explicitly allow epilogue.
        ```mlir
        %bottom_datastage = datastage(strategy=minimize)
        ```
    }];
    let arguments = (ins StrAttr : $strategy, DefaultValuedAttr<BoolAttr, "false">: $allow_epilogue);
    let results = (outs Index);
    let assemblyFormat = [{
        attr-dict
    }];
}

def Ddl_DatastageConstraintOp : Ddl_Op<"datastage_constraint"> {
    let summary = "datastage_constraint";
    let description = [{
        The op adds constraints to the possible values that will be assigned to `datastage`. It is possible to set a minimum value, a max,
        an explicit list of possible values, or a combination of thereof.
        The constraint applies to the `dimensions`. When multiple dimensions are listed, the constraints are intended as applied to the volume
        defined by all those dimensions, i.e. the multiplication of their value.
        When `reference` is a tensor, the constraints are intended as number of sticks, based on the stick composition of that tensor.
        When `reference` is another datastage, the constraints are intended as a factor compared to the dimension (or volume) in the other
        datastage, e.g. `min="2"` would mean that the size should be at least double.
        This op can be put inside ddl.if based on operation_bind so that we can give different constraints based on the op the ddl is applied to.
        ```mlir
        ddl.datastage_constraint(%accum2_datastage, %outtensor, %d#0, %d#1) {min="1", max="4", values=["1", "2", "4"]} //absolute number of sticks
        ddl.datastage_constraint(%accum2_datastage, %bottom_datastage, %d#0, %d#1) {values=["2"]} //relative size compared to other datastage
        ddl.datastage_constraint(%accum2_datastage, %blk_datastage, %d#0, %d#1) {max="0.5"} //relative size compared to other datastage
        ```
    }];
    let arguments = (ins Index:$datastage, Index:$reference, Variadic<Index>:$dimensions, OptionalAttr<StrAttr>:$min, OptionalAttr<StrAttr>:$max, DefaultValuedAttr<StrArrayAttr,"{}">:$values);
    let results = (outs Index);

    let assemblyFormat = [{
        `(` $datastage `,` $reference `,` $dimensions `)` attr-dict
    }];
}

def Ddl_GenericOp : Ddl_Op<"generic_node"> {
    let summary = "generic_node";
    let description = [{
        This node is kind of placeholder for a schedule node which has no representation in ddl, inorder to still show some data during back translation this op will be emitted.
	```mlir
        ddl.generic_node() {name="name", nodetype="nodetype"}
        ```
    }];

    let assemblyFormat = [{
        `(``)` attr-dict
    }];
}

def Ddl_AllocateOp : Ddl_Op<"allocate",[AttrSizedOperandSegments]> {
    let summary = "allocate";
    let description = [{
        The op location links the size of a tensor in a memory with datastages used in the loops it sits into.
        Additionally, padding and buffer-style information can be provided. `num_buffers=-1:si64` means an infinite circular buffer.
        In cases where an external allocation may or may not be present, it is possible to pass the external allocation
        as an input to this op so that it can be used when available.
        The `padding_dim` argument is a list of dimensions that require
        non-default padding-type specification. Each dimension in the list must
        be a padded dimension, i.e. the result of a `ddl.padded_dimeension` op.
        The `padding_type` attribute specifies the padding status per dimension
        of the allocated storage. The attribute accepts a list of strings where
        each string specifies the padding style for a dimension in the
        padding_dim list.
        Allowed values for the padding status:
          “nopad”            : Default
          “padded_nozeropad” : Only valid elements are stored in the allocated
                               storage.
          “padded_wzeropad”  : Both zero-padded parts and the valid part are
                               stored in the allocated storage.
          “lowered_padded”   : The tensor is stored in the lowered form
                               (suitable for window operations).

        The padding types can be specified in two ways.
        - Attribute `padding_type` list can specify a single padding style. In
          this case, the specified padding-type is applied to all dimensions in
          the `padding_dim` list.
        - The number of elements in `padding_type` list and the number of
          elements in the `padding_dim` list are the same. In this case,
          each padding-type is applied to the corresponding dimension in
          `padding_dim` list, in order.

        ```mlir
        %shift_pe_allocation = ddl.allocate(%shift_const) {memory="pelrf", num_buffers=2:si64} : index
        %shift_sfp_allocation = ddl.allocate(%scale_const, [], %external_allocate) {memory="sfplrf"} : index

        // Padded_NoZeroPad is applied to both %d1_pad and %d2_pad
        %l0_allocation = ddl.allocate(%work_tensor, [%d1_pad, %d2_pad]) {memory="l0", padding_type=["padded_nozeropad"]} : index

        // Padded_NoZeroPad is applied to %d1_pad and Padded_WZeroPad is applied to %d2_pad
        %l0_allocation_1 = ddl.allocate(%work_tensor, [%d1_pad, %d2_pad]) {memory="l0", padding_type=["padded_nozeropad", "padded_wzeropad"]} : index

        ```
    }];
    let arguments = (ins Index:$tensor, Variadic<Index> : $padding_dim,
                         Optional<Index>:$external_allocate, StrAttr : $memory,
                         DefaultValuedAttr<SI64Attr,"1">:$num_buffers,
                         OptionalAttr<StrArrayAttr>:$padding_type,
                         OptionalAttr<SI64Attr>:$replication);
    let results = (outs Index);
    let hasCustomAssemblyFormat = 1;
    //let assemblyFormat = [{
    //    `(` $tensor  ( `,` `[` $padding_dim^ `]` )?  (`,` $external_allocate^ )?`)` attr-dict
    //}];
}

def Ddl_ForceInnermostDimensionsOp : Ddl_Op<"force_innermost_dimensions"> {
    let summary = "force_innermost_dimensions";
    let description = [{
        The op allows to force specific dimensions to appear as innermost in a layout in allocations that we fully control.
        The `allocate` input is used to specify with memory and tensor will be affected by this operation.
        `dims` specify which dimensions, in order, should appear as innermost in the layout. Their position in the layout
        is not moved, but they are rather repeated, so they appear as innermost and then again in their regular order.
        The `datastage` is used to determine what is the maximum size that these dimensions will have in their repeated appearance
        as innermost dimensions. This is needed so that we know how the total number of elements in a dimension will be split
        among the repetitions of that dimensions.
        ```mlir
        ddl.force_innermost_dimensions(%kertensor_xrf_allocation, %datastage, %in, %mb)
        ```
        In the example, if the layout was originally `[X,IN,Y,MB]` it will then become `[IN,MB,X,IN,Y,MB]`.
    }];
    let hasVerifier = 1;
    let arguments = (ins Index : $allocate, Index : $datastage, Variadic<Index> : $dims);
    let hasCustomAssemblyFormat = 1;
    let assemblyFormat = [{
       `(` $allocate `,` $datastage `,` $dims `)` attr-dict
    }];
}

def Ddl_UnitOp : Ddl_Op<"unit", [AttrSizedOperandSegments]> {
    let summary = "unit";
    let description = [{
        The op defines units and optionally memories used as inputs or outputs for computes, or as source or destinations in transfers.
        If an allocation is provided, the data interacts with a memory, otherwise it only goes through a fifo. When the unit is `sfpring`,
        the allocation can be substituted with the next_core handler from a core_to_core_communication op, to easily generate the correct
        core destination information for each core and corelet involved in the operation.
        The data_connect label is used to link producers and consumers of data.
        Optionally it is possible to add a constant offsets to be added to the start address, only applicable to reduced dimensions in the
        stick (scale=-2 in DSC). This sould to be used in rare situations (e.g. exx2), as the padding infrastructure
        should allow for automatic start address management in most cases.
        The `vias` field allows to specify which units the data needs to go through to reach the destination. It is only allowed in unit ops 
        that are used as destination in data_transfer ops.
        ```mlir
        %src_inp_lxpe = ddl.unit(%inptensor, %inptensor_lx_allocation) {unit="lxlu", data_connect="l3_lx_input"}
        %dst_inp_lxpe = ddl.unit(%inptensor) {unit="pe", data_connect="pe_lx_input", vias=["sfp"]}
        %dst_outp_lxsu = ddl.unit(%outtensor, %outtensor_lx_allocation) {unit="lxsu", data_connect="lxsu_output", stick_replicated_dim_offset_elements=8}
        %sfp_psum_dst00_sfpring = ddl.unit(%outtensor, %next_core) {unit="sfpring", data_connect="sfpring_output"}
        ```
    }];
    let arguments = (ins  Optional<Index>:$tensor, Optional<Index>:$allocation, StrAttr : $unit, StrAttr : $data_connect, OptionalAttr<I64Attr>:$stick_replicated_dim_offset_elements, DefaultValuedAttr<StrArrayAttr,"{}">:$vias);
    let results = (outs Index);
    let assemblyFormat = [{
        `(` $tensor (`,` $allocation^)? `)` attr-dict
    }];
}

// def Ddl_AccessPattenOp : Ddl_Op<"access_pattern"> {
//     let summary = "access_pattern";
//     let description = [{
//         "specify access pattern of tensor"
//     }];
//     let arguments = (ins StrAttr:$style, Variadic<Index>:$operands);
//     let results = (outs Index);
//     let assemblyFormat = [{
//         `(`$operands`)` attr-dict
//     }];
// }

def Ddl_DataTransferOp : Ddl_Op<"data_transfer",[AttrSizedOperandSegments]> {
    let summary = "data_transfer";
    let description = [{
        Represents a data transfer from one source to one or multiple destinations, defined through unit ops.
        The size of the transfer is defined by the location of the op.
        The optional attribute `rotate_num_elements_`, when specified, represents the
        amount of right rotation. The source content of the transfer is rotated by
        `rotate_num_elements_` elements to the right.
        It is optionally possible to add information about the access pattern
        for each dimension.
        The optional argument `access_pattern_dim` can specify a list of
        dimensions that require non-default access-pattern specification.
        The optional attribute `access_pattern_style` can specify a list of
        strings that describe the corresponding access-patterns.

        In general the access pattern style is inferred automatically from the
        source/destination allocations. When we have an allocation in both src
        and dst, the style is locked and determined automatically. When one of
        the two ends of a transfer is a fifo, we default to assume the same
        padding type as the allocation at the other end.
        The `access pattern style` field in the data_transfer op can be used to
        override this default selection.

        The access pattern styles can be specified in two ways.
        - Attribute `access_pattern_style` list can specify a single
          access-pattern. In this case, the specified access-pattern is applied
          to all dimensions in the `access_pattern_dim` list.
        - The number of elements in `access_pattern_style` list and the number
          of elements in the `access_pattern_dim` list are the same. In this
          case, each access-pattern is applied to the corresponding dimension
          in `access_pattern_dim` list, in order.

        Each access pattern is specified as
          `<source-form>-to-<destination-form>`.
        
        The following access-patterns are accepted.

        Source form/       Destination form/  Transfer mode
        allocation         allocation
        -----------------------------------------------------------------------

        NOPAD              NOPAD              One-to-one: Every co-ordinate “i”
                                              in source corresponds to
                                              co-ordinate “i” in destination
        PADDED_NOZEROPAD   PADDED_NOZEROPAD   One-to-one: Every co-ordinate “i”
                                              in source corresponds to
                                              co-ordinate “i” in destination
        PADDED_NOZEROPAD   PADDED_WZEROPAD    Co-ordinate offset: Every
                                              co-ordinate “i” in source
                                              corresponds to coordinate “i+zpf”
                                              is destination
        PADDED_WZEROPAD    PADDED_WZEROPAD    One-to-one: Every co-ordinate “i”
                                              in source corresponds to
                                              co-ordinate “i” in destination
        PADDED_WZEROPAD    LOWERED_PADDED     Redundant copy with lowering:
                                              Source {0…N}
                                              M=nopad size W= window S=stride
                                              Note: both M an d W can be
                                              “data-staged”
                                              Destination content will be
                                              {0…W-1, S…S+W-1, 2S….2S+W-1,… (M-1)*S…(M*S)+W-1}
        LOWERED_PADDED     LOWERED_PADDED     One-to-one: Every co-ordinate “i”
                                              in source corresponds to
                                              co-ordinate “i” in destination

        Optionally it is also possible to force the number of elements being read/written in a single load/store,
        only applicable to reduced dimensions in the stick (scale=-2 in DSC). This sould to be used in rare
        situations (e.g. exx2), as broadcasted dimensions and padding are handled automatically in most cases.
        ```mlir
        // Default access pattern for all dimensions
        ddl.data_transfer(%src_inp_lxpe, [%dst_inp_lxpe])
        // Non-default access pattern for the specified dimensions.
        ddl.data_transfer(%src_inp_lxpe, [%dst_inp_lxpe], [%d1, %d2]) {
          access_pattern_style=["padded_nozeropad-to-padded_wzeropad", "padded_wzeropad-to-padded_wzeropad"]}
        ddl.data_transfer(%src_outp_sfplx, [%dst_outp_sfplx]) {limit_num_elements_stick_replicated_dim=8}
        ddl.data_transfer(%src_lxlu, [%dst_sfp]) {rotate_num_elements=16}
        ```
    }];
    let arguments = (ins  OptionalAttr<StrArrayAttr>:$access_pattern_style, OptionalAttr<I64Attr>:$limit_num_elements_stick_replicated_dim,
                     OptionalAttr<I64Attr>:$rotate_num_elements,
                     Index : $source,
                     Variadic<Index> : $destinations,
                     Variadic<Index> : $access_pattern_dim);
    let assemblyFormat = [{
        `(` $source`,` `[` $destinations `]` (`,` `[` $access_pattern_dim^ `]`)? `)` attr-dict
    }];
}

def Ddl_ComputeOp : Ddl_Op<"compute",[AttrSizedOperandSegments]> {
    let summary = "compute";
    let description = [{
        Represents a computation. The op requires one or more inputs and can have one or more destinations for the output.
        The actual operation requested is defined by `computeType`, and it will be executed on the specified unit.
        It is also possible to add additional information, such as `mode`, needed to control several computation types.
        ```mlir
        ddl.compute([%sfp_fma_src00, %sfp_fma_src01, %sfp_fma_src02], [%sfp_fma_dst00]) {computetype="FMA16", unit="sfp"}
        ```
    }];
    let hasVerifier = 1;
    let arguments = (ins StrAttr:$computetype, 
                    StrAttr:$unit, 
                    OptionalAttr<I64Attr>:$mode, 
                    OptionalAttr<I64Attr>:$mask, 
                    OptionalAttr<I64Attr>:$repetition,
                    OptionalAttr<ArrayAttr>:$indices,
                    Variadic<Index> : $inputs, Variadic<Index>:$outputs);
    let assemblyFormat = [{
        `(``[`$inputs `]` `,` `[`$outputs`]``)` attr-dict
    }];
}

def Ddl_OpaqueOp : Ddl_Op<"opaque"> {
    let summary = "opaque";
    let description = [{
        Represent a computation through opaque op, identified by the `op` attribute and executed in `unit`.
        `params` captures the parameters that are not registers, such as fifo name or other fields. The "unroll" parameter
        does not need to be filled, as that will be filled automatically based on the data stage parameter selection.
        `internal_registers` captures the list of parameters corresponding to registers that are used internally by the opaque op,
        without interaction with other parts of the dataflow. Only the parameters for an unroll of 1 should be listed, and those
        with a name ending with "_unroll" will be automatically renamed with the following ending "_0", "_1", etc., based on the unroll
        factor. Registers in an unroll group are guaranteed consecutive registers.
        `input_output_registers` captures registers used to read and write data that are used in other parts of the dataflow, such as 
        constants and potentially input/outputs if using registers instead of fifos. The same "_unroll" naming auto-expansion applies here.
        Additionally, for every parameter is in list, it is necessary to list the corresponding allocations through `input_output_allocations`,
        in the same order in which the `input_output_registers` are given.
        `reference_tensor` is needed to help during the register allocation phase, and normally the output tensor can be used.
        `max_unroll_factor` defines the largest unroll factor that the opaque op template supports.
        ```mlir
        ddl.opaque(%outtensor, %zero_pe_allocation, %ffff_pe_allocation) {unit="pe", op="RECIPROCAL", input_output_registers=["c1", "c2"],
            internal_registers=["p0_unroll", "t0_unroll", "t2_unroll"], max_unroll_factor=2, params={"in0"="lxlu", "out0"="result"}}
        ```
    }];
    let arguments = (ins Index:$reference_tensor, Variadic<Index>:$input_output_allocations, StrAttr:$unit, StrAttr:$op, StrArrayAttr:$input_output_registers, StrArrayAttr:$internal_registers, I64Attr:$max_unroll_factor, DictionaryAttr:$params, StrArrayAttr:$input_data_connects, StrArrayAttr:$output_data_connects);
    let assemblyFormat = [{
        `(` $reference_tensor (`,` $input_output_allocations^)? `)` attr-dict
    }];
}

def Ddl_SyncOp : Ddl_Op<"sync"> {
    let summary = "sync";
    let description = [{
        This op represents a group of units all sending or receiving a sync signal. This op needs to be paired with one or multiple other
        sync ops with the opposite value of `is_receive` and same `signal_name`. It is optionally possible to set `separate_corelets` to true
        to automatically split the sync such that corelet0 units only interact with other corelet0 units, and corelet1 units only
        interact with corelet1 units.
        ```mlir
        ddl.sync {units=["lxlu"], is_receive=false, signal_name="sub-chunk-sync"}
        ```
    }];
    let arguments = (ins StrArrayAttr:$units, BoolAttr:$is_receive, StrAttr:$signal_name, DefaultValuedAttr<BoolAttr, "false">:$separate_corelets);
    let assemblyFormat = [{
        attr-dict
    }];
}

def Ddl_ImplicitSyncOp : Ddl_Op<"implicit_sync"> {
    let summary = "implicit_sync";
    let description = [{
        This op is used to set-up the implicit syncs between LU and SU units of a specific memory. The tool will automatically identify the
        transfers involved in writing and reading from the memory, that is expected to be used in a circular buffer fashion (`num_buffers=-1`).
        ```mlir
        ddl.implicit_sync(%l0_allocation)
        ```
    }];
    let hasVerifier = 1;
    let arguments = (ins Index:$allocate);
    let assemblyFormat = [{
        `(`$allocate`)` attr-dict
    }];
}

def Ddl_LoopOp : Ddl_Op<"loop", [ImplicitDdlTerminator]> {
    let summary = "loop";
    let description = [{
        A loop op is defined by a pair of datastages and a list of dimensions. It represents a band of perfectly nested loops
        whose number of iterations are determined by the division of the two datastages for each of the mentioned dimensions.
        The dimensions inside the same loop op can be reordered to optimize memory accesses. If order matters, use separate loops.
        Example: the code below
        ```mlir
        ddl.loop (%d_datastage, %b_datastage, %d#0, %d#1, %d#2, %d#3) {} {

        }
        ```
        corresponds to the following pseudo-code
        ```
        for (1...d_d0/b_d0){
            for (1...d_d1/b_d1){
                for (1...d_d2/b_d2){
                    for (1...d_d3/b_d3){
                        
                    }
                }
            }
        }
        ```
    }];
    let arguments = (ins OptionalAttr<StrAttr>:$label, Index:$numerator_ds, Index:$denominator_ds, Variadic<Index>:$dimensions);
    let regions = (region AnyRegion:$body);
    let assemblyFormat = [{
        `(`$numerator_ds `,` $denominator_ds `,` $dimensions`)` `{` (`label` `=` $label^ )? `}` $body attr-dict 
    }];
}

def Ddl_ParametricLoopOp : Ddl_Op<"parametric_loop", [ImplicitDdlTerminator]> {
    let summary = "parametric_loop";
    let description = [{
        Example: the code below
        ```mlir
        ddl.parametric_loop (%d_pad, %reference_tensor) {} {
          <loop-body>
        }
        ```
        corresponds to the following pseudo-code
        ```
        Let `loop_stride` be number of elements along dimension `d_pad` in a
          stick of `%reference_tensor`. In case, `d_pad` does not appear in the
          stick layout, `loop_stride` is 1.
        for (1...d_pad / loop_stride){
          <loop body>
        }
        ```
    }];
    let arguments = (ins OptionalAttr<StrAttr>:$label, Index:$dimension, Index:$reference_tensor);
    let regions = (region AnyRegion:$body);
    let assemblyFormat = [{
        `(` $dimension `,` $reference_tensor `)` `{` (`label` `=` $label^ )? `}` $body attr-dict 
    }];
}

def Ddl_CoreToCoreCommunicationOp : Ddl_Op<"core_to_core_communication"> {
    let summary = "core_to_core_communication";
    let description = [{
        This op is used to generate multiple handlers that can later be used to express operations that include
        core to core communication patterns, like accumulation (e.g. psum). The only required input is a list of dimensions,
        that will be used to detect the work division across cores and determine chain start/end core ids, as well
        as the next core id from the point of view of every core and corelet involved in an operation, completely
        automatically. For now, only one of the dimensions in the list can actually be split across cores.
        ```mlir
        %psum_start, %psum_end, %next_core, %prev_core = ddl.core_to_core_communication(%in)
        ```
        The three handlers returned by the op can then be used like in the example below
        ```mlir
        %sfp_psum_src02 = ddl.unit(%outtensor, %prev_core) {unit="sfpring", data_connect="sfpring_output"}
        %sfp_psum_dst00_sfpring = ddl.unit(%outtensor, %next_core) {unit="sfpring", data_connect="sfpring_output"}
        ddl.if (%psum_start) {
            // psum: 0 + pe -> sfpring
            ddl.compute([%sfp_psum_src00, %one_const, %zero_const], [%sfp_psum_dst00_sfpring]) {computetype="FMA16", unit="sfp"}
        } else {
            ddl.if (%psum_end) {
                // psum: sfp-ring + pe -> lxsu
                ddl.compute([%sfp_psum_src00, %one_const, %sfp_psum_src02], [%sfp_psum_dst00_lxsu]) {computetype="FMA16", unit="sfp"}
            } else { // middle cores
                // psum: sfp-ring + pe -> sfpring
                ddl.compute([%sfp_psum_src00, %one_const, %sfp_psum_src02], [%sfp_psum_dst00_sfpring]) {computetype="FMA16", unit="sfp"}
            }
        }
        ```
    }];
    let arguments = (ins Variadic<Index>:$dimensions, DefaultValuedAttr<StrAttr, "\"linear\"">:$strategy);
    let results = (outs Index:$chain_start_core, Index:$chain_end_core, Index:$next_core_in_chain, Index:$prev_core_in_chain);
    let assemblyFormat = [{
        `(`$dimensions`)` attr-dict
    }];
}

def Ddl_CoreCoreletCondOp : Ddl_Op<"core_corelet_cond"> {
    let summary = "core_corelet_cond";
    let description = [{
        This op is used to create a core to corelet map and used as condition for a compute Op.
        Key of attribute dictionary represent a core id and values corelet id.
        This Op should not be used by ddl author, it will be used by reverse translator.
        ```mlir
        %out = ddl.core_corelet_cond() {corecorelet = {"1" = [0, 1], "2" = [0], "3" = [1]}}
        ```
    }];
    let arguments = (ins DictionaryAttr:$corecorelet);
    let results = (outs Index);
    let assemblyFormat = [{
        attr-dict
    }];
}

def Ddl_ConditionOp : Ddl_Op<"condition"> {
    let summary = "condition";
    let description = [{
        This op is used to create a conditional based on a loop iterator. We need to provide a `loop_label` and
        a `dimension` to uniquely identify a loop iterator, then we need to specify the `condition` (`eq`, `lt`,...)
        and a value for the comparison. `first` and `last` are special keywords that can be used to reference the
        first or last iteration of the loop, without knowing the loop count.
        ```mlir
        %out = ddl.condition(%in) {loop_label="outermost_loop", condition="eq", value_expr="last"}
        ```
    }];
    let arguments = (ins Index:$dimension, StrAttr:$loop_label, StrAttr:$condition, StrAttr:$value_expr);
    let results = (outs Index);
    let assemblyFormat = [{
        `(`$dimension`)` attr-dict
    }];
}

def Ddl_ConditionAndOp : Ddl_Op<"condition_and"> {
    let summary = "condition_and";
    let description = [{
        Multiple conditions can be combined through a logical AND. Inputs can be other condition[_and/_or/_not] ops,
        get_external_data_transfer_allocation ops, or operation_bind ops. In general, if constructing a complex condition 
        based on loop iterators, the limit is a two-level "OR of ANDs", with optionally an overall negation. There is no
        limit on combining conditions not based on loop iterators.
        ```mlir
        %out = ddl.condition_and(%c1, %c2, %t2)
        ```
    }];
    let arguments = (ins Variadic<Index>:$operands);
    let results = (outs Index);
    let assemblyFormat = [{
        `(`$operands`)` attr-dict
    }];
}

def Ddl_ConditionOrOp : Ddl_Op<"condition_or"> {
    let summary = "condition_or";
    let description = [{
        Multiple conditions can be combined through a logical OR. Inputs can be other condition[_and/_or/_not] ops,
        get_external_data_transfer_allocation ops, or operation_bind ops. In general, if constructing a complex condition 
        based on loop iterators, the limit is a two-level "OR of ANDs", with optionally an overall negation. There is no
        limit on combining conditions not based on loop iterators.
        ```mlir
        %out = ddl.condition_or(%c1, %c2, %t2)
        ```
    }];
    let arguments = (ins Variadic<Index>:$operands);
    let results = (outs Index);
    let assemblyFormat = [{
        `(`$operands`)` attr-dict
    }];
}

def Ddl_ConditionNotOp : Ddl_Op<"condition_not"> {
    let summary = "condition_not";
    let description = [{
        Apply a logical NOT to the input condition. Inputs can be other condition[_and/_or/_not] ops,
        get_external_data_transfer_allocation ops, or operation_bind ops. In general, if constructing a complex condition 
        based on loop iterators, the limit is a two-level "OR of ANDs", with optionally an overall negation. There is no
        limit on combining conditions not based on loop iterators.
        ```mlir
        %out = ddl.condition_not(%c1)
        ```
    }];
    let arguments = (ins Index:$operand);
    let results = (outs Index);
    let assemblyFormat = [{
        `(`$operand`)` attr-dict
    }];
}

def Ddl_IfOp : Ddl_Op<"if",[ImplicitDdlTerminator]> {
    let summary = "if";
    let description = [{
        Your typical if[/else], controlled by a condition input.
        Input can be a condition[_and/_or/_not] op, an operation_bind op, an external allocation handler,
        or a chain_start/end handler from a core_to_core_communication op.
        ```mlir
        ddl.if (%conditionA) {

        } else {

        }
        ddl.if (%conditionB) {

        }
        ```
    }];
    let arguments = (ins Index:$condition);
    let regions = (region AnyRegion:$thenRegion, AnyRegion:$elseRegion);
    let assemblyFormat = [{
        `(`$condition`)` $thenRegion (`else` $elseRegion^)?  attr-dict
    }];

}

// Ops related to transformations
def Ddl_TransformationsOp:Ddl_Op<"transformations", [ImplicitDdlTerminator]> {
    let summary = "Transformations Block";
    let description = [{
        This operation is the root of the entire transformations section of the ddl template.
        ```mlir
        transformations {
            
        }
        ```
    }];
    let regions = (region AnyRegion:$body);
    let assemblyFormat = [{
       $body attr-dict
    }];
}

def Ddl_DisableTransferPromotion:Ddl_Op<"disable_transfer_promotion"> {
    let summary = "";
    let description = [{
        Disables the transformation(s) that moves data_transfers to some outer loop.
        The operation must appear in the `ddl.transformations` section.
        ```mlir
        ddl.disable_transfer_promotion()
        ```
    }];
    let assemblyFormat = [{
        `(``)` attr-dict
    }];
}


#endif // DDC_DDL_DIALECT_DDL_OPS_TD
