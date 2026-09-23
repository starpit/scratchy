//===-- KTDFArchAttributes.td ------------------------------*- tablegen -*-===//
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
// This file defines the attributes in the ktdf_arch dialect.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_DIALECT_KTDFARCH_KTDFARCHATTRIBUTES_TD
#define DATAFLOW_SCHEDULER_DIALECT_KTDFARCH_KTDFARCHATTRIBUTES_TD

include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchDialect.td"
include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchInterfaces.td"

//===----------------------------------------------------------------------===//
// Attribute constraints
//===----------------------------------------------------------------------===//

// This helper erases the type of an attribute, preventing setInherentAttr from
// silently discarding mismatched attributes, but keeping ODS intact.
class KTDFArch_ErasedAttr<Attr attr> : Attr<attr.predicate, attr.summary> {
  let cppFunctionName = attr.cppFunctionName;
  let storageType = "::mlir::Attribute";
  let returnType = attr.returnType;
  let convertFromStorage = !subst("$_self", "::llvm::cast<" # attr.storageType # ">($_self)", attr.convertFromStorage);
  let constBuilderCall = !if(!eq(attr.constBuilderCall, "?"), "$0", attr.constBuilderCall);
  let defaultValue = attr.defaultValue;
  let valueType = attr.valueType;
  let isOptional = attr.isOptional;
  let baseAttr = attr.baseAttr;
  let cppNamespace = attr.cppNamespace;
  let description = attr.description;
}

def KTDFArch_AdjacencyMatrixAttr : Attr<
    CPred<"::llvm::isa<::mlir::ktdf_arch::AdjacencyMatrixAttr>($_self)">, 
    "directed sparse or dense adjacency matrix"
  > {
  let storageType = "::mlir::ktdf_arch::AdjacencyMatrixAttr";
  let returnType = "::mlir::ktdf_arch::AdjacencyMatrixAttr";
  let convertFromStorage = "$_self";
  let constBuilderCall = "?";
}

// Base class for KTDF dialect attributes
class KTDFArch_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<KTDFArch_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

//===----------------------------------------------------------------------===//
// MapAttr
//===----------------------------------------------------------------------===//

def KTDFArch_MapAttr : KTDFArch_Attr<"Map", "map"> {
  let summary = "Maps attributes to other attributes.";
  let description = [{
    The `ktdf_arch.map` attribute is a utility that allows defining an ad-hoc
    mapping between arbitrary attributes. It is illegal to map the same key more 
    than once.

    The order of the entries is preserved, i.e., no canonicalization of the 
    storage takes place, and access is always performed using a linear scan. 
    Usage of this attribute should be limited to scenarios where there are few
    entries, or the data is consumed once. 

    Example:
    ```
    func.func private @map_attr() attributes { 
      map = #ktdf_arch.map<f32 = 1, { hello } = "world"> 
    }
    ```
  }];

  let parameters = (ins 
    ArrayRefParameter<"std::pair<Attribute, Attribute>">:$entries
  );

  let genVerifyDecl = 1;
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    /// Type of the map entries.
    using Entry = std::pair<Attribute, Attribute>;

    /// Gets the @p Attr associated with @p key , if any.
    template <class Attr = Attribute>
    [[nodiscard]] auto getAttr(Attribute key) const -> Attr {
      for (auto& entry : getEntries()) {
        if (entry.first == key) {
          return dyn_cast<Attr>(entry.second);
        }
      }

      return nullptr;
    }

    /// Gets the @p Attr value associated with @p key , if any.
    template <class Attr>
    [[nodiscard]] 
    auto getValue(Attribute key) const -> std::optional<typename Attr::ValueType> {
      const auto attr = getAttr<Attr>(key);
      if (attr) {
        return attr.getValue();
      }

      return std::nullopt;
    }

    //===------------------------------------------------------------------===//
    // Container Interface
    //===------------------------------------------------------------------===//

    using value_type = typename ArrayRef<Entry>::value_type;
    using size_type = typename ArrayRef<Entry>::size_type;
    using iterator = typename ArrayRef<Entry>::iterator;

    [[nodiscard]] auto empty() const -> bool { return getEntries().empty(); }
    [[nodiscard]] auto size() const -> size_type { return getEntries().size(); }

    [[nodiscard]] auto begin() const -> iterator { return getEntries().begin(); }
    [[nodiscard]] auto end() const -> iterator { return getEntries().end(); }
  }];
}

#endif // DATAFLOW_SCHEDULER_DIALECT_KTDFARCH_KTDFARCHATTRIBUTES_TD
