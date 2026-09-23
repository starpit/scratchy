//===----------------------------------------------------------------------===//
//
// IBM Confidential
// (C) Copyright IBM Corp. 2026
//
//===----------------------------------------------------------------------===//
//
// This file defines the SDSCBundle dialect types.
//
//===----------------------------------------------------------------------===//

#ifndef SDSCBUNDLE_TYPES_TD
#define SDSCBUNDLE_TYPES_TD

#ifndef OP_BASE
include "mlir/IR/OpBase.td"
#endif // OP_BASE

include "mlir/IR/AttrTypeBase.td"
include "SDSCBundleDialect.td"

//===----------------------------------------------------------------------===//
// SDSCBundle Type Definitions
//===----------------------------------------------------------------------===//

class SDSCBundle_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<SDSCBundle_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// InputArg Type
//===----------------------------------------------------------------------===//

def SDSCBundle_InputArgType : SDSCBundle_Type<"InputArg", "input_arg"> {
  let summary = "SDSCBundle input argument type with optional metadata";
  let description = [{
    The InputArg type represents an input argument to an SDSC Bundle operation
    with optional metadata for optimization and validation purposes.
    
    The type includes:
    - An underlying type (required): The actual data type of the argument
    - Optional granularity: Specifies the granularity constraint for the argument
    - Optional max_value: Specifies the maximum value constraint for the argument
    
    Example:
    ```mlir
      !sdscbundle.input_arg<index>
      !sdscbundle.input_arg<i32, granularity = 32, max_value = 2048>
    ```
  }];

  let parameters = (ins
    "Type":$underlyingType,
    OptionalParameter<"std::optional<int64_t>">:$granularity,
    OptionalParameter<"std::optional<int64_t>">:$max_value
  );

  let assemblyFormat = [{
    `<` $underlyingType
    (`,` `granularity` `=` $granularity^)?
    (`,` `max_value` `=` $max_value^)?
    `>`
  }];

  let genVerifyDecl = 1;

  let builders = [
    TypeBuilderWithInferredContext<(ins
      "Type":$underlyingType
    ), [{
      return $_get(underlyingType.getContext(), underlyingType,
                    std::nullopt, std::nullopt);
    }]>,
    TypeBuilderWithInferredContext<(ins
      "Type":$underlyingType,
      "std::optional<int64_t>":$granularity,
      "std::optional<int64_t>":$max_value
    ), [{
      return $_get(underlyingType.getContext(), underlyingType,
                    granularity, max_value);
    }]>
  ];
}

#endif // SDSCBUNDLE_TYPES_TD