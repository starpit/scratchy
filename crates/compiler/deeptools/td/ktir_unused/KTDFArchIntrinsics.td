//===-- KTDFArchIntrinsics.td ------------------------------*- tablegen -*-===//
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
// This file implements ktdf_arch intrinsic code generation facilities.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_DIALECT_KTDFARCH_KTDFARCHINTRINSICS_TD
#define DATAFLOW_SCHEDULER_DIALECT_KTDFARCH_KTDFARCHINTRINSICS_TD

include "mlir/IR/OpBase.td"

class Capitalize<string str> {
  string ret = !if(
    !eq(str, ""), 
    "", 
    !strconcat(!toupper(!substr(str, 0, 1)), !substr(str, 1))
  );
}

class SplitString<string str, string at> {
  defvar index = !find(str, at);
  defvar end = !add(!size(at), index);

  list<string> ret = !if(
    !eq(index, -1), 
    [str],
    !listconcat(
      [!substr(str, 0, index)], 
      SplitString<!substr(str, end), at>.ret
    )
  );
}

class PascalCase<string str> {
  defvar input = !subst(".", "_", str);
  defvar parts = SplitString<input, "_">.ret;
  defvar pascalParts = !foreach(part, parts, Capitalize<part>.ret);

  string ret = !interleave(pascalParts, "");
}

class KTDFArch_Intrinsic<
    string mnemonic, 
    string name = PascalCase<mnemonic>.ret
> {
  string attrName = !strconcat("ktdf_arch.", mnemonic);
  string cppName = name;
  string fieldName = !strconcat(!subst(".", "_", mnemonic), "_");

  bit genVerifyDecl = 0;
  string verifierMap = ?;

  bit genTestDecl = 0;
}

class KTDFArch_IntrinsicProperty<
    string mnemonic, 
    string name = PascalCase<mnemonic>.ret
> : KTDFArch_Intrinsic<mnemonic, name> {
  let genVerifyDecl = 1;
  let verifierMap = "property";
}

class KTDFArch_IntrinsicFeature<
    string mnemonic, 
    string name = PascalCase<mnemonic>.ret
> : KTDFArch_Intrinsic<"feature." # mnemonic, "Feature" # name> {
  let genVerifyDecl = 1;
  let verifierMap = "feature";
  let genTestDecl = 1;
}

class KTDFArch_IntrinsicCodeGen<list<KTDFArch_Intrinsic> intrinsics> {
  defvar fieldTemplate = 
  "    StringAttr $_fieldName;";
  defvar getAttrNameTemplate = 
  "    [[nodiscard]] auto get$_cppNameAttrName() const -> StringAttr { return $_fieldName; }";
  defvar cacheAttrNameTemplate =
  "      $_fieldName = StringAttr::get(getContext(), \"$_attrName\");";

  defvar verifyAttrTemplate = 
  "    static auto verify$_cppNameAttr(Operation* op, const NamedAttribute& attr) -> LogicalResult;";
  defvar registerAttrVerifierTemplate =
  "      $_verifierMap_verifiers_[$_fieldName] = verify$_cppNameAttr;";

  defvar featureTestTemplate = 
  "    [[nodiscard]] static auto test$_cppName(Attribute provided, const Feature& required) -> bool;";
  defvar registerFeatureTestTemplate =
  "      feature_tests_[$_fieldName] = test$_cppName;";

  defvar private = !foreach(intrinsic, intrinsics, 
    !subst("$_fieldName", intrinsic.fieldName, 
      fieldTemplate
    )
  );
  defvar register = !listconcat(
    !foreach(intrinsic, intrinsics,
      !subst("$_fieldName", intrinsic.fieldName, 
        !subst("$_attrName", intrinsic.attrName, 
          cacheAttrNameTemplate)
      )
    ),
    !foreach(intrinsic, !filter(intrinsic, intrinsics, intrinsic.genVerifyDecl),
      !subst("$_fieldName", intrinsic.fieldName, 
        !subst("$_cppName", intrinsic.cppName, 
          !subst("$_verifierMap", intrinsic.verifierMap,
            registerAttrVerifierTemplate)
        )
      )
    ),
    !foreach(intrinsic, !filter(intrinsic, intrinsics, intrinsic.genTestDecl),
      !subst("$_fieldName", intrinsic.fieldName, 
        !subst("$_cppName", intrinsic.cppName, 
          registerFeatureTestTemplate
        )
      )
    )
  );
  defvar public = !listconcat(
    !foreach(intrinsic, intrinsics,
      !subst("$_fieldName", intrinsic.fieldName, 
        !subst("$_cppName", intrinsic.cppName, 
          getAttrNameTemplate)
      )
    ),
    !foreach(intrinsic, !filter(intrinsic, intrinsics, intrinsic.genVerifyDecl),
      !subst("$_cppName", intrinsic.cppName, 
        verifyAttrTemplate)
    ),
    !foreach(intrinsic, !filter(intrinsic, intrinsics, intrinsic.genTestDecl),
      !subst("$_cppName", intrinsic.cppName, 
        featureTestTemplate
      )
    ) 
  );

  code ret = [{
   private:
    AttrVerifyMap property_verifiers_;
    AttrVerifyMap feature_verifiers_;
    FeatureTestMap feature_tests_;
}] # !interleave(private, "\n") # [{

    void registerIntrinsics() { 
}] # !interleave(register, "\n") # [{
    }

   public:
}] # !interleave(public, "\n") # [{
  }];
}

#endif // DATAFLOW_SCHEDULER_DIALECT_KTDFARCH_KTDFARCHINTRINSICS_TD
