//===-- KTDFArchInterfaces.td ------------------------------*- tablegen -*-===//
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
// This file defines the interfaces of the ktdf_arch dialect.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_DIALECT_KTDFARCH_KTDFARCHINTERFACES_TD
#define DATAFLOW_SCHEDULER_DIALECT_KTDFARCH_KTDFARCHINTERFACES_TD

include "mlir/IR/OpBase.td"

def KTDFArch_Resource : OpInterface<"Resource"> {
  let cppNamespace = "::mlir::ktdf_arch";
  let description = [{
    Interface for an allocatable resource in an architecture graph.

    A resource is an entity that can be allocated by the scheduler to map a
    compute task, data residency or data routing onto.

    WARNING: At the moment, this interface should not be implemented by foreign
             operations. Only the operations defined in the `ktdf_arch` dialect
             are supposed to implement it.
  }];

  let methods = [
    InterfaceMethod<[{
        Gets the optional unique resource identifier.
      }], 
      "std::optional<StringRef>", "getId", (ins), 
      "", 
      [{
        if (const auto id_attr = getIdAttr(); id_attr) {
          return id_attr.strref();
        }
        return std::nullopt;
      }]>,
    InterfaceMethod<[{
        Gets the optional unique resource identifier attribute.
      }], 
      "StringAttr", "getIdAttr", (ins), 
      "", 
      [{
        return $_op->template getAttrOfType<StringAttr>("id");
      }]>,
    InterfaceMethod<[{
        Sets the unique resource identifier attribute.
      }],
      "void", "setIdAttr", (ins "StringAttr":$id),
      "", [{ 
        $_op->setAttr("id", id);
      }]>,
    InterfaceMethod<[{
        Removes the optional unique resource identifier attribute.
      }], 
      "Attribute", "removeIdAttr", (ins), 
      "", 
      [{
        return $_op->removeAttr("id");
      }]>,
    InterfaceMethod<[{
        Sets the optional unique resource identifier.
      }], 
      "void", "setId", (ins "std::optional<StringRef>":$id), 
      "", 
      [{
        if (!id) {
          removeIdAttr();
        } else {
          setIdAttr(StringAttr::get($_op->getContext(), *id));
        }
      }]>,
    InterfaceMethod<[{
        Gets the resource kind attribute.

        If the resource is in its own unique kind, this attribute is `nullptr`.
      }], 
      "Attribute", "getKind", (ins), 
      "", 
      "">,
  ];

  let verify = [{
    return mlir::ktdf_arch::verifyResource($_op);
  }];

  let extraSharedClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // Properties
    //===------------------------------------------------------------------===//

    /// Gets the intrinsic @p PropertyAttr, if it is set.
    template <class PropertyAttr>
    auto getProperty() -> PropertyAttr {
      return mlir::ktdf_arch::getProperty<PropertyAttr>($_op);
    }

    /// Sets the intrinsic @p PropertyAttr .
    template <class PropertyAttr>
    void setProperty(PropertyAttr attr) {
      return mlir::ktdf_arch::setProperty($_op, attr);
    }

    /// Removes the intrinsic @p PropertyAttr and returns it, if any.
    template <class PropertyAttr>
    auto removeProperty() -> PropertyAttr {
      return mlir::ktdf_arch::removeProperty<PropertyAttr>($_op);
    }

    //===------------------------------------------------------------------===//
    // Features
    //===------------------------------------------------------------------===//

    /// Gets the Feature that satisfies @p required , if any.
    [[nodiscard]] auto getFeature(const Feature& required) -> std::optional<Feature> {
      return mlir::ktdf_arch::getFeature($_op, required);
    }
    /// Gets the Feature with @p name , if any.
    [[nodiscard]] auto getFeature(StringRef name) -> std::optional<Feature> {
      return mlir::ktdf_arch::getFeature($_op, name);
    }
    /// Gets the Feature with @p name , if any.
    [[nodiscard]] auto getFeature(StringAttr name) -> std::optional<Feature> {
      return mlir::ktdf_arch::getFeature($_op, name);
    }
    /// Gets the @p FeatureAttr , if any.
    template <class FeatureAttr>
    [[nodiscard]] auto getFeature() -> FeatureAttr {
      return mlir::ktdf_arch::getFeature<FeatureAttr>($_op);
    }
    /// Gets the @p FeatureAttr that satisfies @p required , if any.
    template <class FeatureAttr>
    [[nodiscard]] auto getFeature(FeatureAttr required) 
      -> std::enable_if_t<std::is_base_of_v<Attribute, FeatureAttr>, FeatureAttr> {
      return mlir::ktdf_arch::getFeature($_op, required);
    }
  }];
}

def KTDFArch_Node : OpInterface<"Node", [KTDFArch_Resource]> {
  let cppNamespace = "::mlir::ktdf_arch";
  let description = [{
    Interface for a resource that declares endpoints in the architecture graph.

    WARNING: At the moment, this interface should not be implemented by foreign
             operations. Only the operations defined in the `ktdf_arch` dialect
             are supposed to implement it.
  }];
}

def KTDFArch_Link : OpInterface<"Link", [KTDFArch_Resource]> {
  let cppNamespace = "::mlir::ktdf_arch";
  let description = [{
    Interface for a link between entities in an architecture graph.

    A link is an entity that establishes a direct connection between two 
    resources, usually for the purposes of data movement.

    WARNING: At the moment, this interface should not be implemented by foreign
             operations. Only the operations defined in the `ktdf_arch` dialect
             are supposed to implement it.
  }];

  let methods = [
    InterfaceMethod<[{
        Gets the source endpoints associated with this link.
      }], 
      "OperandRange", "getSources", (ins), 
      "", 
      "">,
    InterfaceMethod<[{
        Gets the target endpoints associated with this link.
      }], 
      "OperandRange", "getTargets", (ins), 
      "", 
      "">,
  ];

  let verify = [{
    return mlir::ktdf_arch::verifyLink($_op);
  }];
}

#endif // DATAFLOW_SCHEDULER_DIALECT_KTDFARCH_KTDFARCHINTERFACES_TD
