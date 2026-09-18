//! Re-ported from the C++ authority. See crustify-scheduler/AGENT-BRIEF.md.
//!
//! `util/foldManager/foldInfrastructure.h` — a folded quantity is a TREE of fold functions, one
//! level per folded dimension: [`FoldDimProp`] says how wide the level is, its fold function says
//! what the value is at a coordinate in it.

/// One folded dimension's extent — `FoldDimProp::factor_`, `uint32_t` (`foldInfrastructure.h:153`).
///
/// ⛔ NOT a [`Cardinality`](crate::schedule::dsc2::Cardinality): this is the storage a child
/// vector's length is taken from (`:1877`, `:1898`, `:1920`), narrowed to `int` on the way out
/// (`:2631-2633`); an unfolded dim is spelled 1, not 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FoldDimSize(pub u32);

/// Replaces: e009_FoldDimProp
///
/// One folded dimension's extent and label (`foldInfrastructure.h:119-155`).
///
/// ⛔ NO `Default`, AND THAT ABSENCE IS THE GUARD: `FoldDimProp() {}` (`:121`) leaves `factor_`
/// INDETERMINATE (`:153`, no member initialiser). Both placeholder sites fill it immediately
/// (`dsc/dsc2.h:340-342`; `dsc/dsc2.cpp:4664-4675`), so [`new`](Self::new) loses no caller.
/// ⛔ The manager BORROWS these — `fm_dim_prop` is `vector<pair<const FoldDimProp*, ..>>` (`:888`).
/// ⛔ `importFromJson` (`:136-146`) is field-name dispatch onto the two setters; this crate has no
/// JSON reader, so it has no counterpart to port into.
///
/// The control doctest is what makes the `compile_fail` one evidence: stable rustdoc does not check
/// the annotated error code.
/// ```compile_fail,E0599
/// let _ = deeptools::schedule::fold::FoldDimProp::default();
/// ```
/// ```
/// use deeptools::schedule::fold::{FoldDimProp, FoldDimSize};
/// assert_eq!(FoldDimProp::new(FoldDimSize(32), "core_fold_dim").size(), FoldDimSize(32));
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FoldDimProp {
    /// Field: e009_FoldDimProp.factor_
    factor: FoldDimSize,
    /// Field: e009_FoldDimProp.label_
    ///
    /// An open set, never matched against: only forwarded into
    /// [`FoldParamInfoType::fold_dim_label`](crate::schedule::dsc2::FoldParamInfoType) and printed.
    /// "optinal" (`:154`) — the authority's default argument is `""` (`:123`).
    label: String,
}

impl FoldDimProp {
    /// `FoldDimProp(uint32_t factor, std::string label = "")` (`:123-124`).
    pub fn new(factor: FoldDimSize, label: &str) -> Self {
        Self {
            factor,
            label: label.to_owned(),
        }
    }

    /// `getSize()` (`:126`).
    pub const fn size(&self) -> FoldDimSize {
        self.factor
    }

    /// `Label()` (`:127`).
    pub fn label(&self) -> &str {
        &self.label
    }

    /// `setLabel()` (`:128`).
    pub fn set_label(&mut self, label: &str) {
        self.label.clear();
        self.label.push_str(label);
    }

    /// `setSize()` (`:129`).
    pub const fn set_size(&mut self, factor: FoldDimSize) {
        self.factor = factor;
    }

    /// `print()` (`:131-135`) — appends one fold dim's JSON fragment.
    ///
    /// ⛔ No braces and no trailing separator: the caller supplies both (`:2216-2219`,
    /// `dsc/dsc2.h:372`).
    pub fn print(&self, out: &mut String) {
        out.push_str("\"factor_\" : ");
        out.push_str(&self.factor.0.to_string());
        out.push_str(", \"label_\" : \"");
        out.push_str(&self.label);
        out.push('"');
    }
}

/// Replaces: FoldFunction::FuncType
///
/// Which of the seven concrete fold functions a tree node is (`foldInfrastructure.h:166-175`); the
/// discriminant every dispatch site compares `Type()` against (`:1740-1749`, `dsc/dsc2.h:376`).
///
/// ⛔ The order is NOT leaves-then-non-leaves: `WkSplit_leaf` is 6, after all three non-leaves.
/// ⛔ NO `Default`. `Unknown` has NO PRODUCER — the base's `type_ = Unknown` (`:180`) is dead, the
/// only constructor takes a kind (`:177`) and every derived class passes its own.
/// ⛔ There is no WkSplit NON-leaf: `createNonLeafFunc` `DT_ERROR`s on it (`:1911-1927`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FuncType {
    ConstantLeaf = 0,
    MapLeaf = 1,
    AffineLeaf = 2,
    ConstantNonLeaf = 3,
    MapNonLeaf = 4,
    AffineNonLeaf = 5,
    WkSplitLeaf = 6,
    /// Reachable only through the dead member initialiser at `:180`.
    Unknown = 7,
}

/// ⛔ E0080 if a kind is inserted, dropped or reordered: these are the authority's values.
const _: () = {
    assert!(FuncType::ConstantLeaf as u8 == 0);
    assert!(FuncType::MapLeaf as u8 == 1);
    assert!(FuncType::AffineLeaf as u8 == 2);
    assert!(FuncType::ConstantNonLeaf as u8 == 3);
    assert!(FuncType::MapNonLeaf as u8 == 4);
    assert!(FuncType::AffineNonLeaf as u8 == 5);
    assert!(FuncType::WkSplitLeaf as u8 == 6);
    assert!(FuncType::Unknown as u8 == 7);
};

/// Replaces: e010_FoldFunction
///
/// The `FoldFunction<Dtype>` base subobject (`foldInfrastructure.h:163-257`): the kind tag every
/// fold function carries, plus its two predicates. The seven kinds (e013-e019) carry it and add
/// their own data; element-type-free because `type_` is the base's only state.
///
/// ⛔ WHERE THE BASE'S OTHER 14 MEMBERS GO, so none reads as dropped: `getData`/`insertData`
/// (`:200`, `:230`) are PURE VIRTUAL, per kind; their variadic overloads (`:195`, `:225`) only pack
/// coordinates into a `deque`, which one Rust slice already is; the ten `DT_ERROR` stubs
/// (`:203-256`) belong to the kinds that override them, where absence is a compile error instead of
/// a throw; `getFunc()` (`:207`) has zero callers.
/// ⛔ The depth-equals-coordinate-count contract (`:277` non-leaf vs `:570` leaf) is the shared
/// walk's, not the tag's: depth is built at run time from a `fm_dim_prop` (`:888`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FoldFunction {
    /// Field: e010_FoldFunction.type_
    ///
    /// Public in the authority (`:180`) but written only by the constructor (`:177`) and read only
    /// through `Type()`; private here, which makes "fixed at construction" a property of the type.
    ty: FuncType,
}

impl FoldFunction {
    /// `FoldFunction(FuncType type)` (`:177`) — the one constructor.
    pub const fn new(ty: FuncType) -> Self {
        Self { ty }
    }

    /// `Type()` (`:179`).
    pub const fn ty(&self) -> FuncType {
        self.ty
    }

    /// `isLeaf()` (`:182-185`) — verbatim, including what it leaves out.
    ///
    /// ⛔ `WkSplitLeaf` ANSWERS FALSE though it is a leaf by construction (`:1911-1927`): the
    /// authority lists only the other three. Unobservable, hence not a divergence to correct —
    /// `FoldFunction::isLeaf()` has ZERO callers tree-wide, so no C++ run distinguishes an answer.
    pub const fn is_leaf(&self) -> bool {
        matches!(
            self.ty,
            FuncType::ConstantLeaf | FuncType::MapLeaf | FuncType::AffineLeaf
        )
    }

    /// `isNonLeaf()` (`:187-190`) — whether a tree walk descends through this node.
    ///
    /// The only predicate with readers, and both are the same walk (`:1807`, `:1854`): children are
    /// pushed only on true, so `WkSplitLeaf`'s false is right there.
    pub const fn is_non_leaf(&self) -> bool {
        matches!(
            self.ty,
            FuncType::ConstantNonLeaf | FuncType::MapNonLeaf | FuncType::AffineNonLeaf
        )
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    /// IBM's own props (`util/foldManager/test/test_fold_infrastructure.cpp:324-336`), the setter
    /// order of `buildTransferFoldDim` (`dsc/dsc2.cpp:4669-4675`), and `print`'s exact bytes
    /// (`foldInfrastructure.h:131-135`). ⛔ The label is part of equality (`:148-150`).
    #[test]
    fn a_fold_dim_prop_carries_its_extent_and_its_label_and_prints_both() {
        let mut core = FoldDimProp::new(FoldDimSize(32), "core_fold_dim");
        assert_eq!(core.size(), FoldDimSize(32));
        assert_eq!(core.label(), "core_fold_dim");

        let mut out = String::new();
        core.print(&mut out);
        assert_eq!(out, "\"factor_\" : 32, \"label_\" : \"core_fold_dim\"");

        // An unfolded dim is 1, not 0, and the default label is empty.
        let mut out = String::new();
        FoldDimProp::new(FoldDimSize(1), "").print(&mut out);
        assert_eq!(out, "\"factor_\" : 1, \"label_\" : \"\"");

        core.set_size(FoldDimSize(64));
        core.set_label("loop_3_dim");
        assert_eq!(core, FoldDimProp::new(FoldDimSize(64), "loop_3_dim"));

        assert_ne!(
            FoldDimProp::new(FoldDimSize(1), "loop_1_dim"),
            FoldDimProp::new(FoldDimSize(1), "loop_2_dim"),
            "`operator==` compares the label too"
        );
        assert_ne!(
            FoldDimProp::new(FoldDimSize(1), "loop_1_dim"),
            FoldDimProp::new(FoldDimSize(2), "loop_1_dim"),
            "and the extent"
        );
    }

    /// The negative: ⛔ the two predicates are NOT complements and do not cover the kinds
    /// (`foldInfrastructure.h:182-190`) — `WkSplit_leaf` is absent from BOTH lists, so the one kind
    /// that exists only as a leaf answers false to `isLeaf()`.
    #[test]
    fn the_two_leaf_predicates_are_not_complements_and_wksplit_is_in_neither() {
        // (kind, isLeaf, isNonLeaf), read off `:183-189`.
        let expected = [
            (FuncType::ConstantLeaf, true, false),
            (FuncType::MapLeaf, true, false),
            (FuncType::AffineLeaf, true, false),
            (FuncType::ConstantNonLeaf, false, true),
            (FuncType::MapNonLeaf, false, true),
            (FuncType::AffineNonLeaf, false, true),
            (FuncType::WkSplitLeaf, false, false),
            (FuncType::Unknown, false, false),
        ];
        for (ty, is_leaf, is_non_leaf) in expected {
            let ff = FoldFunction::new(ty);
            assert_eq!(ff.ty(), ty, "the tag is fixed by the constructor");
            assert_eq!(ff.is_leaf(), is_leaf, "isLeaf({ty:?})");
            assert_eq!(ff.is_non_leaf(), is_non_leaf, "isNonLeaf({ty:?})");
        }

        assert_eq!(
            expected
                .iter()
                .filter(|(_, leaf, non_leaf)| !leaf && !non_leaf)
                .count(),
            2,
            "`WkSplit_leaf` and `Unknown` are in neither list"
        );
    }
}
