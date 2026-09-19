//! Re-ported from the C++ authority. See crustify-scheduler/AGENT-BRIEF.md.
//!
//! `util/foldManager/foldInfrastructure.h` — a folded quantity is a TREE of fold functions, one
//! level per folded dimension: [`FoldDimProp`] says how wide the level is, its fold function says
//! what the value is at a coordinate in it.

// ⛔ THE SCHEDULER'S FIELD ANCHORS FOR THESE TWO UNITS NAME ONE DECLARED FIELD BETWEEN THEM. The
// census matched `Type name = init;` inside method bodies, so it wrote a local and missed the field
// the local sits next to. Named here so the removal is not silent, and neither is the addition:
//   e018_MapFoldFunction_Leaf.data_vec_   `:782`  DECLARED — the one anchor that is a field
//   e019_WkSplitFoldFunction_Leaf.temp_data
//                                        `:828`  local of the `DT_ERROR`-only `getFoldedData`
//                                                overload, a declaration the compiler reaches only
//                                                after an abort
//   e019_WkSplitFoldFunction_Leaf.wksplit_param_
//                                        `:885`  DECLARED, and UNSCHEDULED — it is this type's
//                                                whole state, so it carries a `Field:` anchor the
//                                                worklist never asked for

use crate::schedule::wk_division::{Cid, CoordRange, WkSize, WkSplitParam};
use core::fmt;
use std::collections::{BTreeMap, VecDeque};

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
    /// An open set, never matched against: `foldDimLabel` (`dsc/dsc2.h:1084`) is assigned,
    /// forwarded and printed at every one of its sites and compared at none. Read into
    /// [`FoldParamInfoType::fold_dim_label`](crate::schedule::dsc2::FoldParamInfoType)
    /// (`ddc/ddc_fold.cpp:2234`), into `addFold` (`dsc/dsc2.h:177`), by the copy that rebuilds a
    /// prop out of `getSize()` and `Label()` (`dsc/superdsc.cpp:60-67` — that is [`Clone`]), and by
    /// [`print`](Self::print).
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
    /// ⛔ No braces and no trailing separator. `dim_prop_attr` supplies both around each fragment
    /// (`:2216-2219`); `debugPrint` supplies NEITHER, printing it straight after
    /// `"\n  Fold dimension= "` (`dsc/dsc2.h:371-372`).
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
/// ⛔ WHERE THE BASE'S OTHER 15 MEMBERS GO, so none reads as dropped: `getData`/`insertData`
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

/// One coordinate within one folded dimension — an element of the `std::deque<int64_t>
/// fold_dim_indices` every fold function is indexed by (`foldInfrastructure.h:755`, `:824`).
///
/// ⛔ NOT a [`FoldDimSize`]: that is a level's extent, this is a position in it, and the authority
/// spells the extent `uint32_t` and the position `int64_t`. A negative one is reachable — see
/// [`MapFoldFunctionLeaf::data`].
/// ⛔ NOT a [`Cid`] either, though the WkSplit leaf reinterprets it as one: `getFoldedData` passes
/// this straight into `WkSplitParam`'s core-indexed readers (`:838`, `:844`). That reinterpretation
/// is written once, in `WkSplitFoldFunctionLeaf`'s private `as_cid`, so a loop-dim coordinate cannot
/// become a core id by accident anywhere else.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FoldDimIndex(pub i64);

/// Replaces: e018_MapFoldFunction_Leaf
///
/// The map fold `f(a1)` — one stored value per coordinate of one folded dimension
/// (`foldInfrastructure.h:739-783`). The only fold function whose payload IS its data, which is why
/// it is the one that carries the element type.
///
/// ⛔ `Dtype` IS UNBOUNDED, and that is the authority's own contract: `FoldManager` is instantiated
/// at `uint64_t`, `int64_t`, `bool`, `std::string`, `Dsc*` and a dozen more, and this class does
/// nothing to a `Dtype` but store, hand back and overwrite it. The bounds sit on the two
/// constructors that need them and nowhere else, so a non-`Default` element type still gets
/// [`new`](Self::new).
/// ⛔ `getFoldFunc` (`:776-779`) returns `this` — an identity the walk that called it already holds,
/// so it has no counterpart here.
/// ⛔ THE BASE'S `getData` TAKES THE WHOLE DEQUE AND `DT_CHECK`s `idx == size - 1` (`:757`, `:766`).
/// A leaf reads exactly one coordinate, so it takes exactly one here and the check has nothing left
/// to compare: reaching the last level is the walk's job, and e020 is where that lands.
///
/// ```
/// use deeptools::schedule::fold::{FoldDimIndex, FoldDimSize, MapFoldFunctionLeaf};
/// let mut leaf = MapFoldFunctionLeaf::filled(FoldDimSize(4), &7_u64);
/// *leaf.data_mut(FoldDimIndex(2)).unwrap() = 11;
/// assert_eq!(leaf.data_vec(), &[7, 7, 11, 7]);
/// assert_eq!(leaf.data(FoldDimIndex(4)), None, "past the end");
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MapFoldFunctionLeaf<Dtype> {
    /// Field: e018_MapFoldFunction_Leaf.data_vec_
    ///
    /// One value per coordinate (`:782`). Its length is the fold dim's extent — set by the extent
    /// in two of the three constructors (`:744`, `:748`) and taken from the vector in the third
    /// (`:752`), which is why [`size`](Self::size) can read it back.
    data_vec: Vec<Dtype>,
}

impl<Dtype> MapFoldFunctionLeaf<Dtype> {
    /// The `FoldFunction<Dtype>` base subobject every constructor passes `Map_leaf` to (`:743`,
    /// `:747`, `:751`).
    ///
    /// ⭐ AN ASSOCIATED CONST, NOT A FIELD: the authority gives the base subobject no name, so a
    /// field would be one invented, which Rule 2 of the brief forbids. Naming the tag once here is
    /// also what stops e020 from spelling a construction site's kind itself and getting it wrong.
    pub const FUNCTION: FoldFunction = FoldFunction::new(FuncType::MapLeaf);

    /// `MapFoldFunction_Leaf(const std::vector<Dtype>& new_data_vec)` (`:750-752`).
    ///
    /// ⛔ [`None`] FOR A VECTOR LONGER THAN A [`FoldDimSize`] CAN HOLD, which is what lets
    /// [`size`](Self::size) be total. IBM's `int getSize()` (`:772`) narrows `size_t` to `int`
    /// instead and says nothing.
    /// ⛔ THIS CONSTRUCTOR HAS NO CALLER IN THE AUTHORITY — `createLeafFunc` uses the other two
    /// (`:1877`, `:1898`). It is kept because it is the one that needs no bound on `Dtype`, and the
    /// other two are it plus a way of filling the run.
    pub fn new(data_vec: Vec<Dtype>) -> Option<Self> {
        u32::try_from(data_vec.len()).ok()?;
        Some(Self { data_vec })
    }

    /// `getData` (`:755-761`) — the value at one coordinate.
    ///
    /// ⛔ [`None`] IS BOTH OF IBM'S REFUSALS AT ONCE. `DT_CHECK(data_vec_.size() > dim_index)`
    /// (`:759`) compares a `size_t` against an `int64_t`, so the index is converted to unsigned and
    /// a NEGATIVE one fails the check rather than reaching `.at()`'s throw; the conversion here
    /// refuses it for the same reason and in the same place.
    pub fn data(&self, dim_index: FoldDimIndex) -> Option<&Dtype> {
        self.data_vec.get(usize::try_from(dim_index.0).ok()?)
    }

    /// `insertData` (`:763-770`) and the mutable `getDataVec` (`:773`), which between them only ever
    /// overwrite one element: `data_vec_.at(dim_index) = new_data` (`:769`).
    ///
    /// A place to write rather than a taker of the value, because that is the same lookup as
    /// [`data`](Self::data) with the same two refusals; handing out `&mut Vec<Dtype>` as the
    /// authority does would let a caller change the fold dim's extent instead.
    pub fn data_mut(&mut self, dim_index: FoldDimIndex) -> Option<&mut Dtype> {
        self.data_vec.get_mut(usize::try_from(dim_index.0).ok()?)
    }

    /// `getSize()` (`:772`) — the folded dimension's extent, which is this vector's length.
    pub fn size(&self) -> FoldDimSize {
        // Total: `new` refuses a longer vector and the other two constructors take the length as a
        // `FoldDimSize` in the first place, so the narrowing has no lossy case to reach.
        FoldDimSize(self.data_vec.len() as u32)
    }

    /// The const `getDataVec()` (`:774`), whose readers only ever compare (`:1164`) or copy
    /// (`:1085`, `:2519`) the whole run.
    pub fn data_vec(&self) -> &[Dtype] {
        &self.data_vec
    }
}

impl<Dtype: Clone> MapFoldFunctionLeaf<Dtype> {
    /// `MapFoldFunction_Leaf(int dim_size, const Dtype& new_data)` (`:746-748`) — every coordinate
    /// starts at the same value. `createLeafFunc`'s data-carrying overload (`:1898`).
    pub fn filled(dim_size: FoldDimSize, new_data: &Dtype) -> Self {
        Self {
            data_vec: vec![new_data.clone(); dim_size.0 as usize],
        }
    }

    /// `this_leaf->getDataVec() = rhs_leaf->getDataVec()` (`:1085`, `:2519`) — the whole-run
    /// assignment the mutable `getDataVec` (`:773`) exists for, and its only other use.
    ///
    /// ⛔ THE EXTENT COMES WITH IT, exactly as the C++ assignment does: both sites reach this after
    /// their walk has already matched the two trees' fold dim props, so the lengths agree there and
    /// a length that did not would be that walk's defect, not a case to refuse here.
    pub fn copy_data_vec_from(&mut self, source: &Self) {
        self.data_vec.clear();
        self.data_vec.extend_from_slice(&source.data_vec);
    }
}

impl<Dtype: Default> MapFoldFunctionLeaf<Dtype> {
    /// `MapFoldFunction_Leaf(int dim_size)` (`:742-744`) — `std::vector<Dtype>(dim_size)` VALUE
    /// initialises, so this is the zero of the element type and not indeterminate storage.
    ///
    /// The constructor `createLeafFunc` reaches for a map fold with no data yet (`:1877`).
    pub fn with_default_data(dim_size: FoldDimSize) -> Self {
        Self {
            data_vec: (0..dim_size.0).map(|_| Dtype::default()).collect(),
        }
    }
}

/// Replaces: e019_WkSplitFoldFunction_Leaf
///
/// The work-split fold `f(a1)` — a leaf that stores no data at all and answers from a
/// [`WkSplitParam`] instead (`foldInfrastructure.h:794-886`). The one fold function whose value at a
/// coordinate is COMPUTED, which is why it is the one that is not a container.
///
/// ⛔ NO ELEMENT TYPE, AND THAT IS THE POINT: `Dtype` reaches this class only to pick between two
/// readings of the one stored param — `getCoordVec` for a `vector<pair<int64_t, int64_t>>` (`:838`),
/// `getSize` cast to the element type for anything arithmetic (`:844`) — and to `DT_ERROR` on any
/// third choice, in BOTH constructors (`:800-805`, `:809-813`) AND in a `getFoldedData` overload
/// that exists only to abort (`:824-830`). Two named readers, [`folded_size`](Self::folded_size) and
/// [`folded_coord_vec`](Self::folded_coord_vec), spell the two readings and leave the third
/// unspellable, so all three `DT_ERROR`s go with the type parameter.
/// ⛔ WHICH READER IS RIGHT IS THE MANAGER'S CHOICE, NOT THE LEAF'S — a `FoldManager<uint64_t>` and
/// a `FoldManager<vector<pair<..>>>` build this leaf from the same param and read it differently
/// (`util/foldManager/test/test_fold_infrastructure.cpp:131-190`). e020 selects; this type offers.
/// ⛔ `getFoldFunc` (`:869-872`) returns `this`, and `insertData` (`:863-867`) is a `DT_ERROR` — see
/// the `compile_fail` doctest below for where that one went.
///
/// The controls are what make the `compile_fail`s evidence: stable rustdoc does not check the
/// annotated error code.
/// ```compile_fail,E0599
/// // `DT_ERROR("Illegal use of insertData on WkSplitFoldFunction_Leaf")` (`:866`): this leaf holds
/// // no data to insert into, so it has no writer to call.
/// let mut leaf = deeptools::schedule::fold::WkSplitFoldFunctionLeaf::unbuilt();
/// let _ = leaf.data_mut(deeptools::schedule::fold::FoldDimIndex(0));
/// ```
/// ```compile_fail,E0107
/// // The constructors' `DT_ERROR` (`:800-805`): there is no element type to instantiate wrongly.
/// let _: deeptools::schedule::fold::WkSplitFoldFunctionLeaf<String>;
/// ```
/// ```
/// // The control: the map leaf, which does own its data, has both.
/// use deeptools::schedule::fold::{FoldDimIndex, FoldDimSize, MapFoldFunctionLeaf};
/// let mut leaf: MapFoldFunctionLeaf<String> = MapFoldFunctionLeaf::with_default_data(FoldDimSize(2));
/// *leaf.data_mut(FoldDimIndex(0)).unwrap() = "written".to_owned();
/// assert_eq!(leaf.data(FoldDimIndex(0)).unwrap(), "written");
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WkSplitFoldFunctionLeaf {
    /// Field: e019_WkSplitFoldFunction_Leaf.wksplit_param_
    ///
    /// The work split this leaf reports (`:885`). ⛔ THE WORKLIST DID NOT SCHEDULE THIS FIELD and
    /// scheduled a method-body local instead; it is this type's whole state, so it is anchored here.
    ///
    /// [`None`] is IBM's `WkSplitParam wksplit_param_{}` before anything built it — the state the
    /// default constructor (`:807-814`) leaves behind, and the state
    /// `DT_CHECK(wksplit_param_.isBuilt())` (`:849`) aborts on.
    wk_split_param: Option<WkSplitParam>,
}

impl WkSplitFoldFunctionLeaf {
    /// The `FoldFunction<Dtype>` base subobject both constructors pass `WkSplit_leaf` to (`:798`,
    /// `:808`) — see [`MapFoldFunctionLeaf::FUNCTION`] for why it is a const and not a field.
    ///
    /// ⛔ ITS TAG ANSWERS FALSE TO [`FoldFunction::is_leaf`] though this type is a leaf in its name
    /// and by construction; the authority's list omits it (`:182-185`) and nothing reads it.
    pub const FUNCTION: FoldFunction = FoldFunction::new(FuncType::WkSplitLeaf);

    /// `WkSplitFoldFunction_Leaf(const WkSplitParam& wksplit_param)` (`:797-805`).
    pub const fn new(wk_split_param: WkSplitParam) -> Self {
        Self {
            wk_split_param: Some(wk_split_param),
        }
    }

    /// `WkSplitFoldFunction_Leaf()` (`:807-814`) — the constructor `createLeafFunc` reaches for a
    /// work-split dim (`:1883`, `:1904`), which is every one of them: `buildWkSplitDim`
    /// (`:1332-1334`) always arrives before any param does.
    ///
    /// ⭐ NOT A `Default`: an unbuilt leaf is one of two states this type has, so it is named. It is
    /// also the one that answers nothing — every reader below is [`None`] until
    /// [`insert_wk_split_param`](Self::insert_wk_split_param) lands.
    pub const fn unbuilt() -> Self {
        Self {
            wk_split_param: None,
        }
    }

    /// `getWkSplitParam()` (`:855-857`).
    pub const fn wk_split_param(&self) -> Option<&WkSplitParam> {
        self.wk_split_param.as_ref()
    }

    /// `getWkSplitParamMutable()` (`:858`), which has three readers: the two fold merges that pass
    /// it straight into another leaf's [`insert_wk_split_param`](Self::insert_wk_split_param)
    /// (`:1071`, `:2515`), and the manager's own mutable accessor (`:2679-2682`).
    pub const fn wk_split_param_mut(&mut self) -> Option<&mut WkSplitParam> {
        self.wk_split_param.as_mut()
    }

    /// `insertWkSplitParam` (`:859-861`) — `wksplit_param_.build(new_params)`, a whole-value
    /// assignment that also flips `isBuilt_`.
    ///
    /// ⛔ THE ABORT INSIDE `build` IS UNREACHABLE FROM HERE. `build(const WkSplitParam&)` re-checks
    /// the fields it copied (`util/foldManager/wkDivisionParams.h:123-124`), so a source that was
    /// never built aborts on `DT_CHECK(outer_repeat_factor_ > 0)` — and the merge sites (`:1071`,
    /// `:2515`) pass whatever the other leaf happens to hold. Taking a `&WkSplitParam` means an
    /// unbuilt source has nothing to pass.
    pub fn insert_wk_split_param(&mut self, new_params: &WkSplitParam) {
        match &mut self.wk_split_param {
            Some(param) => param.overwrite_with(new_params),
            None => self.wk_split_param = Some(new_params.clone()),
        }
    }

    /// The one reinterpretation of a fold coordinate as a core id, which is what `getFoldedData`
    /// does by passing `dim_index` into a `cid` parameter (`:838`, `:844`).
    ///
    /// ⛔ THE AUTHORITY NARROWS HERE AND THIS DOES NOT: `getFoldedData` takes an `int64_t` (`:837`,
    /// `:843`), both readers take an `int32_t cid` (`util/foldManager/wkDivisionParams.h:235`,
    /// `:250`), so a coordinate outside `int32_t` WRAPS ONTO ANOTHER CORE — a compiled
    /// `getSize(1 << 32)` answers `10`, core 0's work, on a split whose cores stop at 15. [`Cid`]
    /// keeps the caller's width, so here that coordinate is simply out of range.
    const fn as_cid(dim_index: FoldDimIndex) -> Cid {
        Cid(dim_index.0)
    }

    /// `getFoldedData` for an arithmetic `Dtype` (`:841-845`) — the work the core at this coordinate
    /// does, which is `getData`'s answer (`:847-853`) once the walk has taken the last coordinate.
    ///
    /// ⛔ [`None`] IS `DT_CHECK(wksplit_param_.isBuilt())` (`:849`), the whole of it: a gap core is
    /// `Some(WkSize(0))` and not an absence (`util/foldManager/wkDivisionParams.h:235-248`).
    /// ⛔ THE AUTHORITY CASTS TO `Dtype` (`:844`), so an arithmetic element type narrower than
    /// `int64_t` truncates silently at the call site. [`WkSize`] does not, and its consumer converts
    /// where it must.
    pub fn folded_size(&self, dim_index: FoldDimIndex) -> Option<WkSize> {
        Some(self.wk_split_param.as_ref()?.size(Self::as_cid(dim_index)))
    }

    /// `getFoldedData` for `Dtype = std::vector<std::pair<int64_t, int64_t>>` (`:832-839`) — the
    /// real element ranges the core at this coordinate covers.
    ///
    /// ⛔ ONE RANGE PER REAL RUN, and a gap core yields one [`CoordRange::GAP`] rather than an empty
    /// vector (`util/foldManager/wkDivisionParams.h:250-284`), so an empty result means neither a gap
    /// nor an absence.
    /// ⛔ WHAT IT DOES MEAN: the core's work lies entirely past the dimension's real-coordinate
    /// table, so it has real work and no real elements to do it on
    /// (`util/foldManager/wkDivisionParams.h:257-281`). A caller that treats empty as "no work" is
    /// reading a gap core's answer into a working core.
    pub fn folded_coord_vec(&self, dim_index: FoldDimIndex) -> Option<Vec<CoordRange>> {
        Some(
            self.wk_split_param
                .as_ref()?
                .coord_vec(Self::as_cid(dim_index)),
        )
    }

    /// `printMetaData` (`:874-882`) — delegates.
    ///
    /// ⛔ NO `isBuilt` GUARD, DELIBERATELY: the authority delegates unconditionally, so an unbuilt
    /// leaf DOES print, with `"isBuilt_" : 0` and every declared initialiser behind it. That is what
    /// [`WkSplitParam::unbuilt_meta_data`] is for, and it is the only thing this type's [`None`]
    /// state is observable through.
    pub fn print_meta_data(&self, ps: &str) -> String {
        match &self.wk_split_param {
            Some(param) => param.print_meta_data(ps),
            None => WkSplitParam::unbuilt_meta_data(ps),
        }
    }
}

impl From<WkSplitParam> for WkSplitFoldFunctionLeaf {
    fn from(wk_split_param: WkSplitParam) -> Self {
        Self::new(wk_split_param)
    }
}

/// The dynamic type behind every `FoldFunction<Dtype>*` the authority holds — `parent_func_`
/// (`:2908`), `child_ff_` (`:300`), `child_ff_vec_` (`:729`) — i.e. WHICH concrete fold function a
/// tree node is. Named for `getFoldFunc` (`:209`), the authority's own accessor for one.
///
/// ⛔ NOT [`FoldFunction`]: that is the base subobject each kind embeds, and it is what
/// [`base`](Self::base) hands back. The variant IS the tag, so the two cannot disagree — the C++
/// pairing of a `Type()` test with a `static_cast` (`dsc/dsc2.h:375-390`) becomes one `match`.
/// ⛔ SIX OF THE SEVEN KINDS ARE HERE, AND THE SEVENTH CANNOT BE. `WkSplitFoldFunction_Leaf` (e019)
/// stores no value: its value at a coordinate is COMPUTED, and WHICH reading — `getCoordVec`
/// (`:838`) or `getSize` cast to the element type (`:844`) — is chosen by the `FoldManager<Dtype>`
/// that built it, not by the leaf. [`WkSplitFoldFunctionLeaf`] is therefore not generic at all, so a
/// variant holding one could not answer [`get_data`](Self::get_data) without naming that choice
/// here, and the choice is e026's. There is no eighth: `createNonLeafFunc` cannot build a WkSplit
/// non-leaf (`:1911-1927`), so [`FuncType`] has no name for one either.
/// ⛔ THE BASE'S TEN `DT_ERROR` STUBS (`:203-256`) ARE DELIBERATELY NOT HERE. `getChild`,
/// `getChildren`, `insertFunc`, `insertAlpha`, `insertBeta`, `printMetaData` and the three
/// `WkSplitParam` accessors live on the kinds that override them, so asking the wrong kind is a
/// compile error where the authority throws at run time. Only the two pure-virtual members
/// (`getData` `:200`, `insertData` `:230`) and `getFoldFunc` — stubbed in the base but overridden by
/// all seven — dispatch from here.
///
/// `D` is the authority's own `Dtype` (`:163`), and it carries DATA, not facts: the scheduler
/// instantiates the tree at `int64_t` for coordinates, addresses and next-core
/// (`dsc/dsc2.h:431`, `:723`, `:985`), at `int` for transfer pad sizes (`:810-811`) and at
/// `std::vector<int64_t>` for constant data (`:49`). Those must not be interchangeable, which is why
/// the walk below asks nothing of `D` beyond [`Clone`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FoldFunc<D> {
    /// A [`ConstFoldFunctionNonLeaf`] — e013.
    ConstantNonLeaf(ConstFoldFunctionNonLeaf<D>),
    /// A [`ConstFoldFunctionLeaf`] — e014.
    ConstantLeaf(ConstFoldFunctionLeaf<D>),
    /// An [`AffineFoldFunctionNonLeaf`] — e015.
    AffineNonLeaf(AffineFoldFunctionNonLeaf<D>),
    /// An [`AffineFoldFunctionLeaf`] — e016.
    AffineLeaf(AffineFoldFunctionLeaf<D>),
    /// A [`MapFoldFunctionNonLeaf`] — e017.
    MapNonLeaf(MapFoldFunctionNonLeaf<D>),
    /// A [`MapFoldFunctionLeaf`] — e018, the bottom of every Map-base dim's subtree
    /// (`createLeafFunc`, `:1877`, `:1898`).
    MapLeaf(MapFoldFunctionLeaf<D>),
}

impl<D> FoldFunc<D> {
    /// `Type()` (`:179`) — read off the variant, never stored twice.
    pub const fn ty(&self) -> FuncType {
        match self {
            Self::ConstantNonLeaf(_) => FuncType::ConstantNonLeaf,
            Self::ConstantLeaf(_) => FuncType::ConstantLeaf,
            Self::AffineNonLeaf(_) => FuncType::AffineNonLeaf,
            Self::AffineLeaf(_) => FuncType::AffineLeaf,
            Self::MapNonLeaf(_) => FuncType::MapNonLeaf,
            Self::MapLeaf(_) => FuncType::MapLeaf,
        }
    }

    /// The `FoldFunction<Dtype>` base subobject this node carries (`:163-190`) — its tag and the two
    /// predicates over it.
    pub const fn base(&self) -> FoldFunction {
        FoldFunction::new(self.ty())
    }

    /// `getData(fold_dim_indices, idx)` (`:200`) — the walk from THIS node down.
    ///
    /// ⛔ [`None`] IS EXACTLY WHERE THE AUTHORITY THROWS, and never a substituted value. It covers
    /// the four throwing outcomes and nothing else: the non-leaf depth `DT_CHECK`
    /// (`:277`, `:409`, `:701`), the leaf one (`:579`, `:757`), a Map level's coordinate outside its
    /// own extent (`:703`) and a map leaf's outside its data run (`:759`).
    /// ⛔ AND `FoldManager::isLegal` (`:1666-1681`), which `getData` runs first (`:1687`, `:1692`),
    /// DOES NOT MAKE THEM UNREACHABLE. It refuses a coordinate count that differs from
    /// `dim_prop_.size()` (`:1668-1669`), so the depth outcomes are the manager's own error
    /// instead — but its range test is signed (`:1677`), so a NEGATIVE coordinate reaches a Map level
    /// or a map leaf and that node's `DT_CHECK` is what raises.
    /// ⛔ THE COORDINATE LIST SHRINKS FROM THE FRONT instead of carrying an `idx`, which is the same
    /// walk with one exception, recorded on [`non_leaf_rest`]: the authority's `idx < size() - 1`
    /// underflows on an EMPTY list.
    pub fn get_data(&self, fold_dim_indices: &[FoldDimIndex]) -> Option<D>
    where
        D: Clone,
    {
        match self {
            Self::ConstantNonLeaf(ff) => ff.get_data(fold_dim_indices),
            Self::ConstantLeaf(ff) => ff.get_data(),
            Self::AffineNonLeaf(ff) => ff.get_data(fold_dim_indices),
            Self::AffineLeaf(ff) => ff.get_data(fold_dim_indices),
            Self::MapNonLeaf(ff) => ff.get_data(fold_dim_indices),
            // `DT_CHECK(idx == fold_dim_indices.size() - 1)` (`:757`) IS THE WALK'S TO MAKE:
            // [`MapFoldFunctionLeaf::data`] takes the one coordinate it reads (`:758-760`), so
            // "this is the last level" is stated here, once, for every kind that has no `idx`.
            Self::MapLeaf(ff) => {
                let [dim_index] = fold_dim_indices else {
                    return None;
                };
                ff.data(*dim_index).cloned()
            }
        }
    }

    /// `insertData(new_data, fold_dim_indices, idx)` (`:230`) — writes one leaf of the walk.
    ///
    /// ⛔ `Some(())` DOES NOT MEAN THE VALUE LANDED: `AffineFoldFunction_Leaf::insertData` is
    /// `// ignored` (`:613-617`), so an affine leaf accepts the write and keeps its own
    /// `alpha_ * i + beta_`. That is not a divergence to fix — an affine leaf is written through
    /// [`insert_alpha`](AffineFoldFunctionLeaf::insert_alpha) and
    /// [`insert_beta`](AffineFoldFunctionLeaf::insert_beta), and `createLeafFunc` discards the
    /// `new_data` it is handed for one (`:1899-1901`) for the same reason.
    /// ⛔ [`None`] is the authority's throw, as on [`get_data`](Self::get_data), except that the
    /// affine non-leaf never reads its own coordinate here (`:452-457` calls no `.at`), so on an
    /// empty list it descends where [`get_data`](Self::get_data) throws.
    pub fn insert_data(&mut self, new_data: D, fold_dim_indices: &[FoldDimIndex]) -> Option<()> {
        match self {
            Self::ConstantNonLeaf(ff) => ff.insert_data(new_data, fold_dim_indices),
            Self::ConstantLeaf(ff) => {
                ff.insert_data(new_data);
                Some(())
            }
            Self::AffineNonLeaf(ff) => ff.insert_data(new_data, fold_dim_indices),
            Self::AffineLeaf(_) => Some(()),
            Self::MapNonLeaf(ff) => ff.insert_data(new_data, fold_dim_indices),
            // The same two checks `getData` makes, on the same two lines of the write
            // (`:766`, `:768`) — a map leaf is the one kind where the write can miss.
            Self::MapLeaf(ff) => {
                let [dim_index] = fold_dim_indices else {
                    return None;
                };
                *ff.data_mut(*dim_index)? = new_data;
                Some(())
            }
        }
    }

    /// `getFoldFunc(fold_dim_indices, idx)` (`:209`) — the node one coordinate list reaches.
    ///
    /// Its only three consumers all want a `WkSplitFoldFunction_Leaf` (e019) at the end of the walk:
    /// `insertWkSplitParam` and both `getWkSplitParam` overloads (`:2655-2681`). The kinds here are
    /// the descent, and a leaf answers itself with NO depth check of its own (`:328-331`,
    /// `:619-622`, `:776-779`) — unlike [`get_data`](Self::get_data).
    pub fn fold_func(&self, fold_dim_indices: &[FoldDimIndex]) -> Option<&Self> {
        match self {
            Self::ConstantNonLeaf(ff) => ff.fold_func(fold_dim_indices),
            Self::AffineNonLeaf(ff) => ff.fold_func(fold_dim_indices),
            Self::MapNonLeaf(ff) => ff.fold_func(fold_dim_indices),
            Self::ConstantLeaf(_) | Self::AffineLeaf(_) | Self::MapLeaf(_) => Some(self),
        }
    }

    /// `getFoldFunc` again, for the two `WkSplitParam` writers (`:2655-2658`, `:2679-2681`).
    pub fn fold_func_mut(&mut self, fold_dim_indices: &[FoldDimIndex]) -> Option<&mut Self> {
        match self {
            Self::ConstantNonLeaf(ff) => ff.fold_func_mut(fold_dim_indices),
            Self::AffineNonLeaf(ff) => ff.fold_func_mut(fold_dim_indices),
            Self::MapNonLeaf(ff) => ff.fold_func_mut(fold_dim_indices),
            Self::ConstantLeaf(_) | Self::AffineLeaf(_) | Self::MapLeaf(_) => Some(self),
        }
    }
}

/// The non-leaf coordinate contract — `DT_CHECK(idx < fold_dim_indices.size() - 1)` (`:277`, `:284`,
/// `:295`, `:409`, `:455`, `:461`, `:701`, `:713`, `:722`) — and the remainder to hand the child.
///
/// ⛔ THAT CHECK IS A `size_t` UNDERFLOW AND THE UNDERFLOW IS REACHABLE. On an EMPTY coordinate list
/// `size() - 1` is `SIZE_MAX`, so every non-leaf passes it and the walk descends with the list still
/// empty. Measured: a two-level constant tree holding 77 answers 77 for `getData()` with no
/// coordinates at all, while the same walk with exactly ONE coordinate throws. So the condition is
/// not "at least two coordinates remain", it is "not exactly one", and an empty remainder stays
/// empty on the way down. The affine and Map non-leaves throw one line later anyway, on the
/// `.at(idx)` that reads the coordinate (`:410`, `:702`).
fn non_leaf_rest(fold_dim_indices: &[FoldDimIndex]) -> Option<&[FoldDimIndex]> {
    match fold_dim_indices.len() {
        1 => None,
        _ => Some(fold_dim_indices.get(1..).unwrap_or(fold_dim_indices)),
    }
}

/// Replaces: e013_ConstFoldFunction_NonLeaf
///
/// A fold level whose value does not depend on its own coordinate: `getData(a1, .., aN)` is
/// `child_ff_->getData(a2, .., aN)` (`:269-301`).
///
/// ⛔ ITS COORDINATE IS NEVER READ — `getData` does not even call `.at` (`:275-279`) — so a
/// coordinate outside this level's `FoldDimProp::factor_` is not an error at a constant level, and
/// there is nothing here for [`FoldDimProp`] to bound.
/// ⛔ THE CHILD IS OWNED HERE AND THE AUTHORITY'S IS NOT: `child_ff_` is a raw
/// `FoldFunction<Dtype>*` that `clear()` deletes through `deleteSubTree` (`:2917-2921`). The
/// authority's tree is a strict hierarchy with one owner per node all the same — `createTree` hands
/// each new node straight to its parent (`:2227-2248`) — so a [`Box`] loses no aliasing the
/// scheduler uses, and it drops the double-free `clear()` invites.
/// ⛔ NO NULL CHILD, AND NO CONSTRUCTOR THAT LEAVES ONE. `createNonLeafFunc`'s `next_child`
/// defaults to `nullptr` (`:1911-1913`) and `createTree` fills it three lines later (`:2242`,
/// `:2244`); [`new`](Self::new) takes the child instead, so the transient null is unrepresentable.
/// Building bottom-up — deepest level first, then wrap — gets that for free.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConstFoldFunctionNonLeaf<D> {
    /// Field: e013_ConstFoldFunction_NonLeaf.child_ff_
    child: Box<FoldFunc<D>>,
}

impl<D> ConstFoldFunctionNonLeaf<D> {
    /// `ConstFoldFunction_NonLeaf(FoldFunction<Dtype>* new_child)` (`:271-273`).
    pub fn new(child: FoldFunc<D>) -> Self {
        Self {
            child: Box::new(child),
        }
    }

    /// `getChild()` (`:288`).
    pub fn child(&self) -> &FoldFunc<D> {
        &self.child
    }

    /// `getChild()` (`:288`) — the authority hands back a mutable pointer, so this is the faithful
    /// half and [`child`](Self::child) the shared reborrow.
    pub fn child_mut(&mut self) -> &mut FoldFunc<D> {
        &mut self.child
    }

    /// `insertFunc(FoldFunction<Dtype>* new_func)` (`:289-291`) — replaces the whole subtree.
    ///
    /// ⛔ The old subtree is dropped here. The authority leaks it: `insertFunc` overwrites the
    /// pointer and nothing deletes what it pointed at.
    pub fn insert_func(&mut self, new_func: FoldFunc<D>) {
        *self.child = new_func;
    }

    /// `getData` (`:275-279`).
    fn get_data(&self, fold_dim_indices: &[FoldDimIndex]) -> Option<D>
    where
        D: Clone,
    {
        self.child.get_data(non_leaf_rest(fold_dim_indices)?)
    }

    /// `insertData` (`:281-286`).
    fn insert_data(&mut self, new_data: D, fold_dim_indices: &[FoldDimIndex]) -> Option<()> {
        self.child
            .insert_data(new_data, non_leaf_rest(fold_dim_indices)?)
    }

    /// `getFoldFunc` (`:293-297`).
    fn fold_func(&self, fold_dim_indices: &[FoldDimIndex]) -> Option<&FoldFunc<D>> {
        self.child.fold_func(non_leaf_rest(fold_dim_indices)?)
    }

    /// `getFoldFunc` (`:293-297`), mutably.
    fn fold_func_mut(&mut self, fold_dim_indices: &[FoldDimIndex]) -> Option<&mut FoldFunc<D>> {
        self.child.fold_func_mut(non_leaf_rest(fold_dim_indices)?)
    }
}

/// Replaces: e014_ConstFoldFunction_Leaf
///
/// The one fold function that is just a value: `getData(a1)` returns `data_` (`:310-335`). Every
/// constant fold ends in one, and it is what `createLeafFunc` falls back to for an unknown base type
/// (`:1885-1887`).
///
/// ⛔ ITS WALK HAS NO COORDINATE CONTRACT AT ALL. Alone among the leaves, `getData` (`:318-321`)
/// omits the `DT_CHECK(idx == fold_dim_indices.size() - 1)` its affine sibling carries (`:579`), so a
/// WRONG-DEPTH walk gets this value instead of a throw: measured, a leaf holding 77 answers 77 for
/// `{5}`, for `{}` and for `{1, 2, 3}` alike. Ported verbatim — `FoldManager::getData` runs
/// `isLegal` first (`:1687`, `:1692`), so no production walk observes it, and hiding it here would
/// make our tree stricter than the trees IBM's own tests build.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ConstFoldFunctionLeaf<D> {
    /// Field: e014_ConstFoldFunction_Leaf.data_
    ///
    /// `Dtype data_{}` (`:334`) — brace-initialised, so the no-argument constructor (`:315-316`, the
    /// one `createLeafFunc` uses at `:1873-1874`) is [`Default`] and not an indeterminate value.
    data: D,
}

impl<D> ConstFoldFunctionLeaf<D> {
    /// `ConstFoldFunction_Leaf(const Dtype& d)` (`:312-314`).
    pub fn new(data: D) -> Self {
        Self { data }
    }

    /// `getData` (`:318-321`) — the indices are not a parameter here, because the authority does not
    /// read them.
    fn get_data(&self) -> Option<D>
    where
        D: Clone,
    {
        Some(self.data.clone())
    }

    /// `insertData` (`:322-326`) — no depth check either, and it overwrites.
    fn insert_data(&mut self, new_data: D) {
        self.data = new_data;
    }
}

/// `getDataAffine`'s overload set (`:390-441` non-leaf, `:561-606` leaf) resolved for ONE payload
/// type: `alpha_ * dim_index + beta_`, plus the child's data at a non-leaf.
///
/// ⛔ WHY THIS IS A STORED `fn` AND NOT A BOUND ON THE WALK. The authority picks between four
/// overloads on `Dtype` by SFINAE, so both the arithmetic AND its truncation are properties of the
/// concrete `Dtype` and of nothing else. Resolving the overload once, where the node is built, keeps
/// that out of [`FoldFunc::get_data`]: a bound on the walk would ALSO take away the constant and Map
/// walks at a payload no affine fold uses, which `FoldManager<std::vector<int64_t>>`
/// (`ConstantInfo::data_`, `dsc/dsc2.h:49`) needs and the authority provides. A trait is the other
/// way to write this, and this campaign forbids one.
///
/// ⛔ IT RETURNS `D` AND NOT `Option<D>`, AND THAT IS THIS BATCH'S CORRECTION. It was ONE generic
/// body reducing the coordinate with `D::try_from(i64)` and answering [`None`] when it did not fit —
/// a REFUSAL WHERE THE AUTHORITY COMPUTES. Measured against the authority's own header: an
/// `AffineFoldFunction_Leaf<int>` with `alpha_ = 1` at coordinate `2^32 + 5` answers **5**, and a
/// `<uint64_t>` one with `(3, 10)` at coordinate **-2** answers **4** — C++ CONVERTS a coordinate,
/// it never refuses one. A negative coordinate is not exotic either: `FoldManager::isLegal`'s range
/// test is signed (`:1677`), `PadSizeFold::data` records exactly that
/// (`crate::schedule::dsc2`), and the affine leaf computing `3 * -2 + 10 = 4` is pinned two tests
/// below. The old justification — "a coordinate is bounded by its level's `FoldDimProp::factor_`" —
/// argued about MAGNITUDE and said nothing about SIGN, which is the case that is reachable.
type GetDataAffine<D> = fn(&D, &D, FoldDimIndex, Option<&D>) -> D;

/// THE AUTHORITY'S WHOLE AFFINE CENSUS, and it is what lets [`GetDataAffine`] be total: an affine
/// fold exists at exactly TWO payloads tree-wide, and one `affine_payload!` line per payload is the
/// compile-time guard that replaced the run-time refusal.
///
/// | payload | the callsite that builds one |
/// |---|---|
/// | `int64_t` | `CoordinateType<CoordinateBaseType>::addFold` (`dsc/dsc2.h:120-126`), the one production caller of `buildAffineDim`; `#define CoordinateBaseType int64_t` (`dsc/dsc2.h:442`) |
/// | `int` | `TransferPadInfo::buildTransferFoldDim` (`dsc/dsc2.cpp:4677-4681`) through `MapWithFMHelper<PrimaryDimTypes, int>` (`dsc/dsc2.h:808`), AND `FoldManager<SdscFoldId>::buildAffineDim` (`dsm/translators/perfDscToSdsc/perfDscToSdsc.cpp:6487`) with `using SdscFoldId = int` (`util/sendefs/sendefs.h:197`) |
///
/// ⛔ THE CENSUS IS NARROWER THAN `std::is_arithmetic` AND THE GAP IS DELIBERATE. The authority's
/// constructor guard (`:353-359`, `:536-542`) admits EVERY arithmetic type, so
/// `AffineFoldFunction_Leaf<double>(0.5, 0.25)` compiles, constructs and answers **1.75** at
/// coordinate 3 — measured, which is why the guard here must not be described as that `DT_ERROR`'s
/// compile-time twin. It is strictly stronger. Nothing is lost: no `FoldManager<float>`, `<double>`
/// or unsigned instantiation exists tree-wide. What is gained is that a payload whose truncation
/// nobody has written down is a COMPILE ERROR instead of a wrong number or a silent [`None`].
/// ⛔ `std::pair<int64_t, int64_t>` HAS NO INSTANTIATION EITHER, and the
/// `std::vector<std::pair<int64_t, int64_t>>` overload is `DT_ERROR("Not yet implemented")` in the
/// authority as well (`:431-441`, `:596-606`) — so both are unported, not dropped.
macro_rules! affine_payload {
    ($D:ty, $get_data_affine:ident, $reduce:expr) => {
        /// The arithmetic overload (`:405-416` with a child, `:575-582` without) at one payload.
        ///
        /// ⛔ WRAPPING, BECAUSE THE AUTHORITY TRUNCATES. C++ promotes `alpha_`, `beta_` and the
        /// child's data to the common type of `Dtype` and `int64_t`, computes there, and narrows the
        /// result back to `Dtype` on return, so a `FoldManager<int>` level with `alpha_ = 2^20` at
        /// coordinate `2^20` answers 0, not `2^40`, and a non-leaf's `2^32 + 1` answers 1 —
        /// measured, both. Truncation to the payload's width is a ring homomorphism, so the same
        /// bits come out of wrapping arithmetic in the payload; plain `*` would instead panic in a
        /// debug build on a value the authority accepts.
        #[allow(
            clippy::cast_possible_truncation,
            clippy::unnecessary_cast,
            reason = "$reduce IS the authority's coordinate conversion, and it is the identity at \
                      the payload that is already `int64_t`"
        )]
        fn $get_data_affine(
            alpha: &$D,
            beta: &$D,
            dim_index: FoldDimIndex,
            child_data: Option<&$D>,
        ) -> $D {
            let index: $D = ($reduce)(dim_index.0);
            let affine = alpha.wrapping_mul(index).wrapping_add(*beta);
            match child_data {
                Some(data) => affine.wrapping_add(*data),
                None => affine,
            }
        }

        impl AffineFoldFunctionNonLeaf<$D> {
            /// `AffineFoldFunction_NonLeaf(Dtype alpha, Dtype beta, FoldFunction<Dtype>* new_child)`
            /// (`:347-359`), and — with two zeroes — the child-only form at `:373-382`.
            pub fn new(alpha: $D, beta: $D, child: FoldFunc<$D>) -> Self {
                Self {
                    alpha,
                    beta,
                    child: Box::new(child),
                    get_data_affine: $get_data_affine,
                }
            }
        }

        impl AffineFoldFunctionLeaf<$D> {
            /// `AffineFoldFunction_Leaf(Dtype alpha, Dtype beta)` (`:532-542`).
            pub fn new(alpha: $D, beta: $D) -> Self {
                Self {
                    alpha,
                    beta,
                    get_data_affine: $get_data_affine,
                }
            }
        }

        /// `AffineFoldFunction_Leaf()` (`:544-552`) — `alpha_{}`, `beta_{}`.
        impl Default for AffineFoldFunctionLeaf<$D> {
            fn default() -> Self {
                Self::new(0, 0)
            }
        }
    };
}

affine_payload!(i64, get_data_affine_i64, |index: i64| index as i64);
affine_payload!(i32, get_data_affine_i32, |index: i64| index as i32);

// ⛔ THE SCHEDULER ALSO SCHEDULED `e015_AffineFoldFunction_NonLeaf.val`. THERE IS NO SUCH FIELD:
// `val` is a METHOD-BODY LOCAL — `Dtype val; return val;` (`:401-402`) — the uninitialised return of
// the NON-arithmetic `getDataAffine` overload, on the line after a `DT_ERROR` that has already
// thrown. It holds no state and has no accessor, so it is removed rather than invented.
// ⛔ AND TWO REAL FIELDS WERE MISSING FROM BOTH AFFINE UNITS: `alpha_` and `beta_` (`:517-518`,
// `:676-677`) are declared brace-initialised, which is why e015 was scheduled with `child_ff_` and
// this local, and why e016 was scheduled with NO fields at all. Anchored below as the fields their
// classes declare.

/// Replaces: e015_AffineFoldFunction_NonLeaf
///
/// An affine fold level that adds its own term to its child's: `getData(a1, .., aN)` is
/// `(alpha_ * a1 + beta_) + child_ff_->getData(a2, .., aN)` (`:345-520`). A chain of these over an
/// [`AffineFoldFunctionLeaf`] is how a folded coordinate becomes a sum of per-level strides —
/// `CoordinateType::addFold` builds exactly that (`dsc/dsc2.h:120-126`).
///
/// ⛔ ONE OF THE AUTHORITY'S THREE CONSTRUCTORS IS OMITTED, AND IT IS A LANDMINE:
/// `AffineFoldFunction_NonLeaf(Dtype alpha, Dtype beta)` (`:361-371`) sets `alpha_` and `beta_` and
/// leaves `child_ff_` INDETERMINATE — not null, indeterminate, because `child_ff_` (`:519`) is the
/// one member of this class declared with no initialiser. Its `getData` would dereference that. It
/// has NO caller tree-wide: `createNonLeafFunc` uses only the child-taking form (`:1920-1922`), and
/// nothing outside `FoldManager` constructs these nodes at all — `dsc/dsc2.h:378-386` only
/// `static_cast`s an existing one to read `alpha_` and `beta_`. Omitted for the same reason
/// [`FoldDimProp`] has no [`Default`].
/// ⛔ THE THIRD CONSTRUCTOR IS [`new`](Self::new) WITH DEFAULTS:
/// `AffineFoldFunction_NonLeaf(FoldFunction<Dtype>* new_child)` (`:373-382`) is the one
/// `createNonLeafFunc` calls, and it leaves `alpha_{}` and `beta_{}` at zero (`:517-518`), so
/// `new(D::default(), D::default(), child)` is it.
/// ⛔ ITS `Dtype` GUARD IS A COMPILE ERROR HERE, AND A STRICTLY STRONGER ONE: all three constructors
/// `DT_ERROR` unless `Dtype` is arithmetic, `std::pair<int64_t, int64_t>` or a vector of those
/// (`:353-359`), while `affine_payload!` admits only the two payloads an affine fold is ever BUILT
/// at. An `AffineFoldFunction_NonLeaf<double>` constructs and computes in the authority — refusing it
/// is this port's choice, not IBM's. See [`GetDataAffine`].
#[derive(Clone)]
pub struct AffineFoldFunctionNonLeaf<D> {
    /// Field: e015_AffineFoldFunction_NonLeaf.alpha_
    ///
    /// This level's stride per step — [`Alpha`](crate::schedule::dsc2::Alpha) at the `int64_t`
    /// instantiation (`dsc/dsc2.h:442`), an `int` at the pad-size one (`dsc/dsc2.h:810-811`).
    alpha: D,
    /// Field: e015_AffineFoldFunction_NonLeaf.beta_
    ///
    /// This level's offset — [`Beta`](crate::schedule::dsc2::Beta) at the `int64_t` instantiation.
    beta: D,
    /// Field: e015_AffineFoldFunction_NonLeaf.child_ff_
    child: Box<FoldFunc<D>>,
    /// The overload `getDataAffine` resolves to for `D` (`:390-441`), fixed at construction. See
    /// [`GetDataAffine`]; it is not one of the class's members.
    get_data_affine: GetDataAffine<D>,
}

impl<D> AffineFoldFunctionNonLeaf<D> {
    /// `getAlpha()` (`:444`).
    pub const fn alpha(&self) -> &D {
        &self.alpha
    }

    /// `getBeta()` (`:445`).
    pub const fn beta(&self) -> &D {
        &self.beta
    }

    /// `insertAlpha()` (`:446`) — reached through `FoldManager::insertAlpha` (`:2278-2294`), which
    /// refuses a non-affine level first (`:2286-2287`).
    pub fn insert_alpha(&mut self, new_alpha: D) {
        self.alpha = new_alpha;
    }

    /// `insertBeta()` (`:447`).
    pub fn insert_beta(&mut self, new_beta: D) {
        self.beta = new_beta;
    }

    /// `getChild()` (`:443`).
    pub fn child(&self) -> &FoldFunc<D> {
        &self.child
    }

    /// `getChild()` (`:443`), mutably.
    pub fn child_mut(&mut self) -> &mut FoldFunc<D> {
        &mut self.child
    }

    /// `insertFunc()` (`:448-450`).
    pub fn insert_func(&mut self, new_func: FoldFunc<D>) {
        *self.child = new_func;
    }

    /// `printMetaData()` (`:465-467`) via `printMetaDataAffine`'s arithmetic overload (`:487-493`).
    ///
    /// ⛔ THE PREFIX IS EMITTED TWICE — before `"alpha_"` AND before `"beta_"`, mid-line after the
    /// separator (`:491-492`). Verbatim: the only caller passes no prefix (`dsc/dsc2.h:383`), so
    /// nothing in the scheduler reads the odd form, and "fixing" it here would make our output
    /// differ from IBM's for the caller that does.
    /// ⛔ THE OTHER THREE `printMetaDataAffine` OVERLOADS ARE NOT PORTED, in line with
    /// [`GetDataAffine`]: one is `DT_ERROR("Unsupported")` (`:469-478`), one prints nothing at all
    /// (`:507-514`), and the `std::pair` one has no instantiation.
    pub fn print_meta_data(&self, out: &mut String, ps: &str)
    where
        D: fmt::Display,
    {
        print_meta_data_affine(&self.alpha, &self.beta, out, ps);
    }

    /// `getData` (`:385-388`) through `getDataAffine` (`:405-416`).
    ///
    /// ⛔ The authority's `if (!std::is_arithmetic<decltype(child_ff_->getData(..))>::value)`
    /// (`:411-413`) tests the CHILD's type, which is this same `Dtype` — always arithmetic when this
    /// overload was selected at all, so it is a tautology with nothing to port.
    fn get_data(&self, fold_dim_indices: &[FoldDimIndex]) -> Option<D>
    where
        D: Clone,
    {
        let rest = non_leaf_rest(fold_dim_indices)?;
        let dim_index = *fold_dim_indices.first()?;
        let child_data = self.child.get_data(rest)?;
        Some((self.get_data_affine)(
            &self.alpha,
            &self.beta,
            dim_index,
            Some(&child_data),
        ))
    }

    /// `insertData` (`:452-457`) — forwards, and unlike [`get_data`](Self::get_data) never reads its
    /// own coordinate.
    fn insert_data(&mut self, new_data: D, fold_dim_indices: &[FoldDimIndex]) -> Option<()> {
        self.child
            .insert_data(new_data, non_leaf_rest(fold_dim_indices)?)
    }

    /// `getFoldFunc` (`:459-463`).
    fn fold_func(&self, fold_dim_indices: &[FoldDimIndex]) -> Option<&FoldFunc<D>> {
        self.child.fold_func(non_leaf_rest(fold_dim_indices)?)
    }

    /// `getFoldFunc` (`:459-463`), mutably.
    fn fold_func_mut(&mut self, fold_dim_indices: &[FoldDimIndex]) -> Option<&mut FoldFunc<D>> {
        self.child.fold_func_mut(non_leaf_rest(fold_dim_indices)?)
    }
}

/// ⛔ MANUAL, AND IT IGNORES THE RESOLVED OVERLOAD: two affine levels with the same `alpha_`,
/// `beta_` and subtree ARE the same fold function, and Rust does not guarantee that one `fn` item has
/// one address across codegen units, so deriving this would make equality a build detail.
impl<D: PartialEq> PartialEq for AffineFoldFunctionNonLeaf<D> {
    fn eq(&self, other: &Self) -> bool {
        self.alpha == other.alpha && self.beta == other.beta && self.child == other.child
    }
}

impl<D: Eq> Eq for AffineFoldFunctionNonLeaf<D> {}

/// ⛔ MANUAL for the same reason: a code address is not part of this node's state.
impl<D: fmt::Debug> fmt::Debug for AffineFoldFunctionNonLeaf<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AffineFoldFunctionNonLeaf")
            .field("alpha", &self.alpha)
            .field("beta", &self.beta)
            .field("child", &self.child)
            .finish_non_exhaustive()
    }
}

/// Replaces: e016_AffineFoldFunction_Leaf
///
/// The affine leaf: `getData(a1)` is `alpha_ * a1 + beta_` (`:530-678`), the innermost level of a
/// folded coordinate.
///
/// ⛔ `insertData` IS SILENTLY `// ignored` (`:613-617`) AND `createLeafFunc` DISCARDS THE DATA IT IS
/// GIVEN FOR ONE (`:1899-1901`: the `new_data` overload builds a bare `AffineFoldFunction_Leaf()`).
/// Both are consistent, not defects — an affine leaf's value IS `alpha_` and `beta_`, and
/// [`insert_alpha`](Self::insert_alpha) / [`insert_beta`](Self::insert_beta) are its only writers —
/// but a caller that expects `insertData` to land a value on an affine fold gets the old one back.
/// ⛔ ITS NON-ARITHMETIC OVERLOAD RETURNS `alpha_` WHERE THE NON-LEAF'S `DT_ERROR`s (`:568-573` vs
/// `:397-403`): the two disagree about the same unsupported payload. Neither is reachable — the
/// constructor has already thrown for such a `Dtype` — and here neither exists to be reached.
/// ⛔ NO INDETERMINATE STATE TO GUARD: both `alpha_` and `beta_` are brace-initialised (`:676-677`),
/// so the no-argument constructor (`:544-552`, `createLeafFunc`'s at `:1879-1880`) is [`Default`] —
/// measured, it answers 0 at every coordinate.
///
/// The `Dtype` guard is a compile error here, and per [`GetDataAffine`] a STRONGER one than the
/// authority's `DT_ERROR` — `3u64, 10u64` below is a pair C++ accepts and computes with. The controls
/// are what make these `compile_fail` blocks evidence; stable rustdoc does not check the annotated
/// error code:
/// ```compile_fail,E0599
/// let _ = deeptools::schedule::fold::AffineFoldFunctionLeaf::<Vec<i64>>::new(vec![1], vec![2]);
/// ```
/// ```compile_fail,E0599
/// let _ = deeptools::schedule::fold::AffineFoldFunctionLeaf::<u64>::new(3, 10);
/// ```
/// ```
/// let _ = deeptools::schedule::fold::AffineFoldFunctionLeaf::<i64>::new(1, 2);
/// let _ = deeptools::schedule::fold::AffineFoldFunctionLeaf::<i32>::new(1, 2);
/// ```
/// And the walk that guard must NOT take away — a CONSTANT fold over that same payload
/// (`ConstantInfo::data_`, `dsc/dsc2.h:49`) still reads:
/// ```
/// use deeptools::schedule::fold::{ConstFoldFunctionLeaf, FoldDimIndex, FoldFunc};
/// let leaf = FoldFunc::ConstantLeaf(ConstFoldFunctionLeaf::new(vec![7i64, 8]));
/// assert_eq!(leaf.get_data(&[FoldDimIndex(0)]), Some(vec![7, 8]));
/// ```
#[derive(Clone)]
pub struct AffineFoldFunctionLeaf<D> {
    /// Field: e016_AffineFoldFunction_Leaf.alpha_
    alpha: D,
    /// Field: e016_AffineFoldFunction_Leaf.beta_
    beta: D,
    /// The overload `getDataAffine` resolves to for `D` (`:561-606`), fixed at construction.
    get_data_affine: GetDataAffine<D>,
}

impl<D> AffineFoldFunctionLeaf<D> {
    /// `getAlpha()` (`:608`).
    pub const fn alpha(&self) -> &D {
        &self.alpha
    }

    /// `getBeta()` (`:609`).
    pub const fn beta(&self) -> &D {
        &self.beta
    }

    /// `insertAlpha()` (`:610`).
    pub fn insert_alpha(&mut self, new_alpha: D) {
        self.alpha = new_alpha;
    }

    /// `insertBeta()` (`:611`).
    pub fn insert_beta(&mut self, new_beta: D) {
        self.beta = new_beta;
    }

    /// `printMetaData()` (`:631-633`) via `printMetaDataAffine` (`:646-652`) — the prefix is emitted
    /// twice here too.
    pub fn print_meta_data(&self, out: &mut String, ps: &str)
    where
        D: fmt::Display,
    {
        print_meta_data_affine(&self.alpha, &self.beta, out, ps);
    }

    /// `getData` (`:556-559`) through `getDataAffine` (`:575-582`).
    ///
    /// ⛔ `DT_CHECK(idx == fold_dim_indices.size() - 1)` (`:579`) is EXACTLY one remaining
    /// coordinate: an empty list throws here (`0 == SIZE_MAX` is false) where the non-leaf check lets
    /// it through. See [`non_leaf_rest`].
    fn get_data(&self, fold_dim_indices: &[FoldDimIndex]) -> Option<D> {
        match fold_dim_indices {
            [dim_index] => Some((self.get_data_affine)(
                &self.alpha,
                &self.beta,
                *dim_index,
                None,
            )),
            _ => None,
        }
    }
}

impl<D: PartialEq> PartialEq for AffineFoldFunctionLeaf<D> {
    fn eq(&self, other: &Self) -> bool {
        self.alpha == other.alpha && self.beta == other.beta
    }
}

impl<D: Eq> Eq for AffineFoldFunctionLeaf<D> {}

impl<D: fmt::Debug> fmt::Debug for AffineFoldFunctionLeaf<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AffineFoldFunctionLeaf")
            .field("alpha", &self.alpha)
            .field("beta", &self.beta)
            .finish_non_exhaustive()
    }
}

/// `printMetaDataAffine`'s arithmetic overload, shared by both affine kinds (`:487-493`, `:646-652`)
/// because the authority's two copies are byte-identical.
fn print_meta_data_affine<D: fmt::Display>(alpha: &D, beta: &D, out: &mut String, ps: &str) {
    out.push_str(ps);
    out.push_str("\"alpha_\" : ");
    out.push_str(&alpha.to_string());
    out.push_str(", ");
    out.push_str(ps);
    out.push_str("\"beta_\" : ");
    out.push_str(&beta.to_string());
}

/// Replaces: e017_MapFoldFunction_NonLeaf
///
/// The fold level that SELECTS: `getData(a1, .., aN)` is `child_ff_vec_.at(a1)->getData(a2, .., aN)`
/// (`:687-730`) — one independent subtree per coordinate of this level, and the only kind whose
/// children can differ from each other.
///
/// ⛔ NO HOLES, AND THAT REMOVES THE ONE UNDEFINED BEHAVIOUR IN THIS FILE. The authority's
/// `MapFoldFunction_NonLeaf(int dim_size)` constructor fills `child_ff_vec_` with `nullptr`
/// (`:690-693`) and `getData` dereferences the selected entry with no null test (`:704`), so a
/// coordinate for an unfilled child is a null dereference. Every producer fills every entry at once —
/// `createSubTreeForEachMapChild` assigns `createTree(...)` to all of them (`:2250-2257`), which is
/// the only thing `createNonLeafFunc`'s Map branch (`:1917-1919`) is ever followed by — so
/// [`new`](Self::new) takes the children the authority's OTHER constructor takes (`:695-697`) and the
/// `dim_size` one is omitted with the nulls it makes. Bottom-up: build each subtree, then this.
/// ⛔ ITS RANGE `DT_CHECK` IS AN UNSIGNED COMPARISON: `child_ff_vec_.size() > dim_index` (`:703`)
/// converts a negative `int64_t` coordinate to a huge `size_t`, so -1 IS rejected — the right answer
/// for the wrong reason, and this is the only kind that rejects one at all.
/// ⛔ IT OVERRIDES NEITHER `insertFunc` NOR `getChild` (`:687-730`), so both throw in the authority.
/// Here they are simply absent, and [`children_mut`](Self::children_mut) is how a child is replaced —
/// `createSubTreeForEachMapChild`'s own route (`:2254-2256`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MapFoldFunctionNonLeaf<D> {
    /// Field: e017_MapFoldFunction_NonLeaf.child_ff_vec_
    children: Vec<FoldFunc<D>>,
}

impl<D> MapFoldFunctionNonLeaf<D> {
    /// `MapFoldFunction_NonLeaf(std::vector<FoldFunction<Dtype>*> ff_ptrs)` (`:695-697`).
    ///
    /// ⛔ The length is this level's `FoldDimProp::factor_` (`:1919`), and it is what bounds a
    /// coordinate here — the only kind where the extent is enforced at all.
    pub fn new(children: Vec<FoldFunc<D>>) -> Self {
        Self { children }
    }

    /// `getChildren()` (`:706-708`).
    pub fn children(&self) -> &[FoldFunc<D>] {
        &self.children
    }

    /// `getChildren()` (`:706-708`) — the authority returns the `vector` itself, so a caller COULD
    /// resize it and break the level's extent. None does: the two callers assign entries (`:2256`)
    /// or read them (`:1862`), so a slice is the whole surface they use.
    pub fn children_mut(&mut self) -> &mut [FoldFunc<D>] {
        &mut self.children
    }

    /// The child a coordinate selects — `DT_CHECK(child_ff_vec_.size() > dim_index)` and
    /// `.at(dim_index)` (`:702-704`), which are one bounds test in Rust.
    fn child_at(&self, dim_index: FoldDimIndex) -> Option<usize> {
        let index = usize::try_from(dim_index.0).ok()?;
        (index < self.children.len()).then_some(index)
    }

    /// `getData` (`:699-705`).
    fn get_data(&self, fold_dim_indices: &[FoldDimIndex]) -> Option<D>
    where
        D: Clone,
    {
        let rest = non_leaf_rest(fold_dim_indices)?;
        let index = self.child_at(*fold_dim_indices.first()?)?;
        self.children[index].get_data(rest)
    }

    /// `insertData` (`:710-718`).
    fn insert_data(&mut self, new_data: D, fold_dim_indices: &[FoldDimIndex]) -> Option<()> {
        let rest = non_leaf_rest(fold_dim_indices)?;
        let index = self.child_at(*fold_dim_indices.first()?)?;
        self.children[index].insert_data(new_data, rest)
    }

    /// `getFoldFunc` (`:720-726`).
    fn fold_func(&self, fold_dim_indices: &[FoldDimIndex]) -> Option<&FoldFunc<D>> {
        let rest = non_leaf_rest(fold_dim_indices)?;
        let index = self.child_at(*fold_dim_indices.first()?)?;
        self.children[index].fold_func(rest)
    }

    /// `getFoldFunc` (`:720-726`), mutably.
    fn fold_func_mut(&mut self, fold_dim_indices: &[FoldDimIndex]) -> Option<&mut FoldFunc<D>> {
        let rest = non_leaf_rest(fold_dim_indices)?;
        let index = self.child_at(*fold_dim_indices.first()?)?;
        self.children[index].fold_func_mut(rest)
    }
}

// ⛔ THE SCHEDULER'S 42 FIELD ANCHORS FOR e026 NAME ONE DECLARED FIELD BETWEEN THEM, AND MISS THE
// OTHER. `FoldManager` declares exactly two (`:2908-2909`); the census matched `Type name = init;`
// inside 2,025 lines of method bodies, so 41 of the 42 are method-body locals and the second real
// field was never scheduled. Named here so neither the removal nor the addition is silent:
//   e026_FoldManager.parent_func_      `:2908`  DECLARED — the one anchor that is a field
//   e026_FoldManager.dim_prop_         `:2909`  DECLARED, and UNSCHEDULED — the fold space itself,
//                                               so it carries a `Field:` anchor the worklist never
//                                               asked for
//   all_data `:2111`, areAllConstant `:2837`, base_func_types `:2589`, can_copy `:1512`,
//   const_ffs_at_pos `:1777`, coord_to_data `:2838`, coordinates `:2047`, curr_leaf_pos `:1374`,
//   data_and_coord `:2093`, dim_prop_sub_tree `:1354`, do_expand `:1989`, ff `:2438`,
//   ffs_at_last_non_leaf_pos `:1377`, ffs_at_pos `:2202`, ffs_at_pre_leaf_pos `:1411`, fifo `:1729`,
//   fold_dim_props `:2611`, l_so2 `:2733`, last_non_leaf_pos `:1376`, linear_ff_list `:2531`,
//   linear_ff_list_rhs `:965`, linear_ff_list_this `:966`, need_to_rebuild `:1504`,
//   new_dim_prop `:928`, next_pos `:1806`, old_parent `:1358`, pos `:958`, posOfAffineFolds `:2780`,
//   pre_leaf_pos `:1410`, ps2 `:2190`, reduced_coord `:2846`, reduced_coord_to_data `:2843`,
//   repeat_factor `:2019`, time `:1735`, total_count `:1951`, unique_data_coords `:2000`,
//   wasCompressed `:2892`, zero_fold_coord `:2067`
//                                        — thirty-eight names, each cited at its FIRST declaration
//                                          inside the class body (`:897-2922`); four of them
//                                          (`coordinates`, `data_and_coord`, `ffs_at_pos`,
//                                          `linear_ff_list`) are PARAMETER names as well, which is
//                                          further evidence they are not fields. With `break`,
//                                          `continue` and `endl` — KEYWORDS the same regex matched —
//                                          and `parent_func_`, that is all 42.
// ⛔ AND A SECOND, IDENTICAL 42-ANCHOR BLOCK SAT IN THIS FILE UNDER `e020_FoldManager`, this type's
// entry number before the campaign renumbered (`e020` is now `DesignSpaceConfig`). Removed with this
// one: one type cannot have two entry numbers, and leaving the stale set would hand
// `DesignSpaceConfig`'s port 42 anchors naming FoldManager's locals.

/// One position in a [`FoldManager`]'s fold-dimension list — the authority's `int pos`, where `[0]`
/// is the OUTERMOST fold dim (`:2909`) and a negative value counts back from the innermost, `-1`
/// being the last.
///
/// ⛔ NOT a [`FoldDimIndex`]: that is a coordinate WITHIN one level, this selects the level.
/// ⛔ AND THE AUTHORITY DOES NOT ACCEPT A NEGATIVE ONE EVERYWHERE, which is why this carries two
/// resolutions instead of one. `insertAlpha`, `getAlpha`, `rebuildDim` and `insertWkSplitParam` open
/// with `if (pos < 0) pos = dim_prop_.size() + pos;` (`:2279-2280`, `:2326-2327`, `:1474-1475`) —
/// that is [`resolve`](Self::resolve). `getFuncType`, `getFoldDimProp` and `getFoldDimSize` index
/// straight through an unsigned comparison or `.at()` (`:2579-2581`, `:2624`, `:2631-2633`), so a
/// negative one becomes a huge unsigned value and throws — measured, `getFuncType(-1)` on a
/// one-dimension manager throws. That is [`index`](Self::index).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FoldDimPos(pub i32);

impl FoldDimPos {
    /// The from-the-end pass plus `DT_CHECK(pos >= 0 && pos <= dim_prop_.size() - 1)`.
    ///
    /// ⛔ THAT CHECK IS UNSIGNED AND UNDERFLOWS AT ZERO DIMENSIONS: `dim_prop_.size() - 1` is
    /// `SIZE_MAX` there, so the guard PASSES and the `dim_prop_.at(pos)` one line later throws
    /// instead — measured, `rebuildDim(0, Map)` on a fresh manager throws rather than answering
    /// `false`. Both outcomes are [`None`] here, and the two are told apart where it matters, in
    /// [`FoldManager::rebuild_dim`].
    fn resolve(self, num_dims: usize) -> Option<usize> {
        let len = i64::try_from(num_dims).ok()?;
        let pos = i64::from(self.0);
        let pos = if pos < 0 { len + pos } else { pos };
        if pos < 0 || pos >= len {
            return None;
        }
        usize::try_from(pos).ok()
    }

    /// `.at(pos)` with no from-the-end pass — the accessors that compare against an unsigned size.
    fn index(self, num_dims: usize) -> Option<usize> {
        let pos = usize::try_from(self.0).ok()?;
        (pos < num_dims).then_some(pos)
    }
}

/// `scan_inner_outer` (`:1986`) — `true` makes the INNERMOST fold dim vary fastest in the
/// coordinate list a manager flattens to.
///
/// ⛔ A NEWTYPE BECAUSE IT TRAVELS BESIDE [`ExpandIfAnyMap`]: `getFlattenedCoordinates(coords, fix,
/// scan_inner_outer, expand_if_any_map)` takes the two adjacent and in that order, and transposing
/// them silently reorders every address a transfer node reads. Here it is `E0308`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScanInnerOuter(pub bool);

/// `expand_if_any_map` (`:1987`) — `true` enumerates a Map level's every coordinate instead of
/// collapsing the level to one, and ONLY if some unfixed Map level is wider than 1 (`:1990-1998`).
/// [`FoldManager::all_data_with_map_unrolled`] is the `getAllData` that passes it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExpandIfAnyMap(pub bool);

/// Replaces: BaseFuncType
///
/// WHICH fold function a level of the tree is built from — `enum class BaseFuncType` (`:39-45`), the
/// second half of every `dim_prop_` entry (`:888`).
///
/// ⛔ NOT [`FuncType`]: that names all SEVEN concrete classes plus `Unknown`, and it is read off a
/// node that exists. This names the FOUR FAMILIES a level is built FROM, and `Unknown` here is a
/// sentinel a caller passes IN — `rebuildDim`'s "leave the type alone" (`:1494`) — which is why
/// [`as_str`](Self::as_str) has no name for it and `createLeafFunc` throws on it (`:1884-1885`).
/// ⛔ UNSCHEDULED. It is `dim_prop_`'s element type, so the field cannot be carried without it, and
/// the campaign never gave it an entry of its own: `FoldInfraUtils` (`:46-113`) was scheduled as
/// neither a type nor a symbol. Its two maps are accounted for in [`as_str`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BaseFuncType {
    /// One value for the whole level — `ConstFoldFunction_{NonLeaf,Leaf}`.
    Constant = 0,
    /// One subtree, or one value, per coordinate of the level — `MapFoldFunction_{NonLeaf,Leaf}`.
    Map = 1,
    /// `alpha * coordinate + beta` — `AffineFoldFunction_{NonLeaf,Leaf}`.
    Affine = 2,
    /// The work split of `WkSplitFoldFunction_Leaf` — a leaf family, with no non-leaf form
    /// (`:1911-1927` builds three kinds, not four).
    WkSplit = 3,
    /// The sentinel, never a built level. See the type's own note.
    Unknown = 4,
}

// The discriminants are the authority's own (`:40-44`), and `printMetaData` keys a map with them
// (`:2193-2194`), so they are pinned rather than left to declaration order.
const _: () = {
    assert!(BaseFuncType::Constant as u8 == 0);
    assert!(BaseFuncType::Map as u8 == 1);
    assert!(BaseFuncType::Affine as u8 == 2);
    assert!(BaseFuncType::WkSplit as u8 == 3);
    assert!(BaseFuncType::Unknown as u8 == 4);
};

impl BaseFuncType {
    /// `FoldInfraUtils::baseFuncTypeToString` (`:47-51`) — FOUR entries, and [`None`] for the fifth
    /// because the authority's map has no `Unknown` key and `.at()` throws on it (measured).
    ///
    /// ⛔ `stringToBaseFuncType` (`:52-53`) IS NOT PORTED WITH IT. It is `flipMap` of this, and its
    /// only readers are `importFromJson` (`:2760-2818`) and `perfdsc`'s JSON reader — this crate has
    /// no JSON reader at all, the same reason [`FoldDimProp`] carries no `importFromJson`.
    pub const fn as_str(self) -> Option<&'static str> {
        match self {
            Self::Constant => Some("Const"),
            Self::Map => Some("Map"),
            Self::Affine => Some("Affine"),
            Self::WkSplit => Some("WkSplit"),
            Self::Unknown => None,
        }
    }
}

/// Replaces: e026_FoldManager
///
/// The fold space of one dataspace: a list of fold dimensions, outermost first, and the tree of
/// [`FoldFunc`]s that maps a coordinate in that space to a value — `template <typename Dtype> class
/// FoldManager` (`:897-2922`).
///
/// ⛔ THE TWO FIELDS ALIAS IN C++ AND ARE ONE OWNER HERE. `dim_prop_` is
/// `std::vector<std::pair<const FoldDimProp*, BaseFuncType>>` (`:888`, `:2909`) — BORROWED pointers,
/// and every caller in scope owns the `FoldDimProp` elsewhere and hands over its address, so the
/// manager's own lifetime says nothing about the props'. `parent_func_` (`:2908`) is a raw `new`d tree
/// the destructor `deleteSubTree`s (`:914`). This type owns both outright: the props are CLONED in,
/// which is what lets the tree and the list be rebuilt together without a caller's prop outliving or
/// under-living the manager that reads its size. [`PadSizeFold`](super::dsc2::PadSizeFold) already
/// records the same choice for the same reason.
///
/// ⛔ `Dtype` IS NOT FREE, and the third field is why. Six of the eight fold-function kinds are
/// payload-agnostic, but the two affine ones need `Dtype::operator*` and `operator+` — expressed in
/// this file as the [`GetDataAffine`] function pointer an affine node carries, because the authority
/// gets them from `Dtype` itself and Rust cannot ask a type parameter for them without the trait
/// [RULE 1] forbids. So a manager carries the constructor its own affine levels would need, and
/// carries [`None`] at a payload that has no such constructor. A payload with no affine arithmetic
/// therefore has NO `build_affine_dim` AT ALL — `E0599`, at the call site, instead of the authority's
/// `DT_ERROR` at run time — and that is the whole reason the constructors are per-payload.
///
/// The controls are what make the `compile_fail` blocks evidence: stable rustdoc checks no error
/// code, so both codes below were read from `rustc` against the built rlib. First the payload guard
/// — `E0599`, "no method named `build_affine_dim` found for struct `FoldManager<Vec<i64>>`":
/// ```compile_fail,E0599
/// use deeptools::schedule::fold::{FoldDimPos, FoldDimProp, FoldDimSize, FoldManager};
/// let mut fm = FoldManager::<Vec<i64>>::new();
/// let prop = FoldDimProp::new(FoldDimSize(2), "a");
/// let _ = fm.build_affine_dim(&prop, FoldDimPos(0));
/// ```
/// ```
/// use deeptools::schedule::fold::{FoldDimPos, FoldDimProp, FoldDimSize, FoldManager};
/// let mut fm = FoldManager::<i64>::new();
/// let prop = FoldDimProp::new(FoldDimSize(2), "a");
/// assert_eq!(fm.build_affine_dim(&prop, FoldDimPos(0)), Some(()));
/// // And the level a payload without the arithmetic DOES get to build, so the guard takes away the
/// // affine kind and nothing else:
/// let mut vecs = FoldManager::<Vec<i64>>::new();
/// assert_eq!(vecs.build_const_dim(&prop, FoldDimPos(0)), Some(()));
/// ```
/// And then the two `i32` newtypes this class indexes with, which a C++ `int` does not tell apart —
/// `E0308`, "expected `FoldDimPos`, found `FoldDimIndex`". [`FoldDimPos`] selects a LEVEL of the
/// tree; [`FoldDimIndex`] is a coordinate WITHIN one:
/// ```compile_fail,E0308
/// use deeptools::schedule::fold::{FoldDimIndex, FoldDimProp, FoldDimSize, FoldManager};
/// let mut fm = FoldManager::<i64>::new();
/// let prop = FoldDimProp::new(FoldDimSize(2), "a");
/// let _ = fm.build_const_dim(&prop, FoldDimIndex(0));
/// ```
/// ```
/// use deeptools::schedule::fold::{FoldDimIndex, FoldDimPos, FoldDimProp, FoldDimSize, FoldManager};
/// let mut fm = FoldManager::<i64>::new();
/// let prop = FoldDimProp::new(FoldDimSize(2), "a");
/// assert_eq!(fm.build_const_dim(&prop, FoldDimPos(0)), Some(()));
/// assert_eq!(fm.get_data(&[FoldDimIndex(0)]), Some(0));
/// ```
///
/// Field: e026_FoldManager.parent_func_
/// Field: e026_FoldManager.dim_prop_
#[derive(Clone, Debug)]
pub struct FoldManager<D> {
    /// `FoldFunction<Dtype>* parent_func_ = nullptr;` (`:2908`) — the root of the tree, one level per
    /// entry of `dim_prop`, a leaf kind at the last.
    ///
    /// ⛔ NEVER ABSENT HERE, where the authority has a window in which it is null: `clear()`
    /// (`:2917-2921`) sets it to `nullptr`, and `buildFoldSpace({})` clears and then throws out of
    /// `createTree` — `DT_ERROR("Need at least one fold dim to build a tree")` (`:2228-2229`,
    /// measured), leaving a manager whose every reader segfaults. A cleared manager here holds the
    /// default constant leaf its own constructor builds (`:901-904`),
    /// which is the state the very next `buildDim` puts it in anyway.
    parent_func: FoldFunc<D>,
    /// `fm_dim_prop dim_prop_;  // [0] --> outer most` (`:2909`, the alias at `:888`) — one entry per
    /// fold dimension, each pairing the dimension's extent and label with the kind of fold function
    /// built at that level. EMPTY is legal and means a zero-dimension fold space: one value, and the
    /// coordinate `[-1]` (`:2067`).
    dim_prop: Vec<(FoldDimProp, BaseFuncType)>,
    /// The affine constructor this payload has, if it has one. See the type's note; the same
    /// justification [`AffineFoldFunctionNonLeaf`] already carries for holding one applies here —
    /// it is not one of the class's members, it is the arithmetic the class asks `Dtype` for.
    get_data_affine: Option<GetDataAffine<D>>,
}

/// The four constructors and `reset` (`:901-914`, `:1032-1038`), per payload, plus the affine builder
/// only where an affine level can exist. See [`FoldManager`]'s note on `Dtype`; the census of which
/// payloads exist is [`affine_payload`]'s.
macro_rules! fold_manager_payload {
    (@ctors $D:ty, $affine:expr) => {
        impl FoldManager<$D> {
            /// `FoldManager()` (`:901-904`) — a zero-dimension space holding one default value.
            pub fn new() -> Self {
                Self {
                    parent_func: FoldFunc::ConstantLeaf(ConstFoldFunctionLeaf::new(
                        <$D>::default(),
                    )),
                    dim_prop: Vec::new(),
                    get_data_affine: $affine,
                }
            }

            /// `FoldManager(const Dtype& data)` (`:905-908`) — a zero-dimension space holding `data`.
            pub fn with_data(data: $D) -> Self {
                Self {
                    parent_func: FoldFunc::ConstantLeaf(ConstFoldFunctionLeaf::new(data)),
                    dim_prop: Vec::new(),
                    get_data_affine: $affine,
                }
            }

            /// `reset(Args&&... args)` (`:1032-1038`) — the authority destroys the tree in place and
            /// placement-news the manager over itself, forwarding to a constructor. Assignment is
            /// that, without the in-place part.
            pub fn reset(&mut self) {
                *self = Self::new();
            }

            /// `reset` forwarding to `FoldManager(const Dtype&)`. See [`reset`](Self::reset).
            pub fn reset_with_data(&mut self, data: $D) {
                *self = Self::with_data(data);
            }
        }

        impl Default for FoldManager<$D> {
            fn default() -> Self {
                Self::new()
            }
        }
    };
    ($D:ty) => {
        fold_manager_payload!(@ctors $D, None);
    };
    ($D:ty, affine = $get_data_affine:ident) => {
        fold_manager_payload!(@ctors $D, Some($get_data_affine as GetDataAffine<$D>));

        impl FoldManager<$D> {
            /// `buildAffineDim(const FoldDimProp* prop, int pos = 0)` (`:1328-1330`).
            ///
            /// ⛔ THIS METHOD IS THE PAYLOAD GUARD. It exists only where the payload has affine
            /// arithmetic; at a payload that does not, calling it is `E0599` rather than the
            /// authority's `DT_ERROR` out of `createLeafFunc` (`:1884-1885`) at run time.
            pub fn build_affine_dim(
                &mut self,
                prop: &FoldDimProp,
                pos: FoldDimPos,
            ) -> Option<()> {
                self.build_dim(prop, BaseFuncType::Affine, pos)
            }
        }
    };
}

fold_manager_payload!(i64, affine = get_data_affine_i64);
fold_manager_payload!(i32, affine = get_data_affine_i32);
fold_manager_payload!(Vec<i64>);

/// `operator==` (`:1101-1169`).
///
/// ⛔ IT DOES NOT COMPARE LABELS, only `getSize()` and the `BaseFuncType` per dimension (`:1104-1109`)
/// and then the payloads down the tree (`:1130-1166`) — measured: two managers whose dimensions are
/// labelled "a" and "b" compare EQUAL, while two whose constant leaves hold 3 and 4 do not.
/// `FoldDimProp`'s own `operator==` (`:148-150`) compares the label; a manager's does not reach it.
impl<D: PartialEq> PartialEq for FoldManager<D> {
    fn eq(&self, rhs: &Self) -> bool {
        if self.dim_prop.len() != rhs.dim_prop.len() {
            return false;
        }
        for (lhs_dim, rhs_dim) in self.dim_prop.iter().zip(rhs.dim_prop.iter()) {
            if lhs_dim.0.size() != rhs_dim.0.size() || lhs_dim.1 != rhs_dim.1 {
                return false;
            }
        }
        // The authority walks the two BFS lists and compares payloads per kind (`:1136-1165`). With
        // the sizes and the kinds already equal the two trees have the same shape, so structural
        // equality of the roots compares exactly those payloads — and skips `get_data_affine`, which
        // the affine nodes' own hand-written `PartialEq` drops.
        self.parent_func == rhs.parent_func
    }
}

impl<D: Eq> Eq for FoldManager<D> {}

impl<D> FoldManager<D> {
    /// `getNumDims()` (`:2270`).
    pub fn num_dims(&self) -> usize {
        self.dim_prop.len()
    }

    /// `hasZeroFoldDim()` (`:2603`).
    pub fn has_zero_fold_dim(&self) -> bool {
        self.dim_prop.is_empty()
    }

    /// `getDimProp()` (`:2820`) — the whole list, outermost first.
    pub fn dim_prop(&self) -> &[(FoldDimProp, BaseFuncType)] {
        &self.dim_prop
    }

    /// `getFoldDimProp()` (`:2610-2616`) — the authority's `std::deque<const FoldDimProp*>`, here
    /// borrows into the list this manager owns.
    pub fn fold_dim_props(&self) -> Vec<&FoldDimProp> {
        self.dim_prop.iter().map(|(prop, _)| prop).collect()
    }

    /// `getFoldDimProp(int pos)` (`:2624`) — `.at()`, so [`FoldDimPos::index`] and NOT a
    /// from-the-end position.
    pub fn fold_dim_prop(&self, pos: FoldDimPos) -> Option<&FoldDimProp> {
        let idx = pos.index(self.dim_prop.len())?;
        Some(&self.dim_prop[idx].0)
    }

    /// `getFoldDimSize(int pos)` (`:2631-2633`) — `.at()` again.
    ///
    /// ⛔ The authority narrows it to `int` on the way out while `getFoldSpaceSize` (`:2457`) keeps
    /// the same value as `int64_t`. Both are the dimension's `uint32_t factor_`, so both are
    /// [`FoldDimSize`] here and the two cannot disagree.
    pub fn fold_dim_size(&self, pos: FoldDimPos) -> Option<FoldDimSize> {
        Some(self.fold_dim_prop(pos)?.size())
    }

    /// `getFoldSpaceSize()` (`:2457-2462`) — every dimension's extent, outermost first.
    pub fn fold_space_size(&self) -> Vec<FoldDimSize> {
        self.dim_prop.iter().map(|(prop, _)| prop.size()).collect()
    }

    /// `getFuncType(int pos)` (`:2578-2581`).
    ///
    /// ⛔ ITS RANGE TEST IS `pos >= dim_prop_.size()`, AN UNSIGNED COMPARISON, so `-1` is not the last
    /// dimension here the way it is for `getAlpha` — it is `SIZE_MAX` and throws (measured). Hence
    /// [`FoldDimPos::index`].
    pub fn func_type(&self, pos: FoldDimPos) -> Option<BaseFuncType> {
        let idx = pos.index(self.dim_prop.len())?;
        Some(self.dim_prop[idx].1)
    }

    /// `getFuncType()` (`:2588-2594`) — every level's kind, outermost first.
    pub fn func_types(&self) -> Vec<BaseFuncType> {
        self.dim_prop.iter().map(|(_, ty)| *ty).collect()
    }

    /// `isLegal(const std::deque<int64_t>&)` (`:1666-1681`) — whether a coordinate list addresses this
    /// fold space.
    ///
    /// ⛔ IT CANNOT ANSWER `false`. The authority declares it `bool` and returns `true` on the only
    /// path that reaches a return; every failing case is a `DT_ERROR` — measured, `isLegal({2})` on a
    /// two-wide dimension THROWS rather than answering false, and so does a list of the wrong length.
    /// So the answer is a unit and the throw, and no caller can mistake "illegal" for "false"; the
    /// authority's own callers ignore the value and rely on the throw (`:1687`, `:1692`).
    /// ⛔ A ZERO-DIMENSION MANAGER CALLS EVERY LIST LEGAL, INCLUDING A NON-EMPTY ONE (`:1667` returns
    /// before the arity test), which is why `getData({7})` on one answers its single value instead of
    /// refusing — measured.
    /// ⛔ AND THE PER-DIMENSION TEST IS SIGNED: `getSize() <= fold_dim_indices.at(idx)` promotes the
    /// `uint32_t` extent to `int64_t`, so a NEGATIVE coordinate passes it (measured, `isLegal({-1})` is
    /// legal on a three-wide Map dim and `getData({-1})` then throws in the level itself). Ported as it
    /// stands; what a negative then does is the level's business, and the two MAP kinds are the ones
    /// that reject it — `DT_CHECK(child_ff_vec_.size() > dim_index)` (`:703`) and
    /// `DT_CHECK(data_vec_.size() > dim_index)` (`:759`), both unsigned comparisons a `-1` fails. A
    /// Constant level ignores the coordinate and an Affine one computes with it.
    /// ⛔ ITS THIRD CHECK IS DEAD CODE (`:1671-1674`): an empty list with a non-Constant root. A list
    /// is empty only where `dim_prop_` is, and that case returned `true` at the top.
    pub fn is_legal(&self, fold_dim_indices: &[FoldDimIndex]) -> Option<()> {
        if self.dim_prop.is_empty() {
            return Some(());
        }
        // `DT_ERROR("number of dimensions in query and fold space are different")` (`:1668-1669`).
        if fold_dim_indices.len() != self.dim_prop.len() {
            return None;
        }
        for (idx, (prop, _)) in self.dim_prop.iter().enumerate() {
            // `DT_ERROR("query fold dimension with higher fold factor")` (`:1676-1679`).
            if i64::from(prop.size().0) <= fold_dim_indices[idx].0 {
                return None;
            }
        }
        Some(())
    }

    /// `clear()` (`:2917-2921`).
    ///
    /// ⛔ THE AUTHORITY LEAVES `parent_func_` NULL AND THIS DOES NOT. See the field's own note: every
    /// caller of `clear` rebuilds a tree on the next line, and the one that does not —
    /// `buildFoldSpace` with an empty list — throws with the manager still cleared, so the null is
    /// observable only through a manager the authority has already thrown out of.
    fn clear(&mut self)
    where
        D: Default,
    {
        self.parent_func = FoldFunc::ConstantLeaf(ConstFoldFunctionLeaf::new(D::default()));
        self.dim_prop.clear();
    }

    /// `createLeafFunc(const FoldDimProp* prop, BaseFuncType)` (`:1870-1888`) — and its
    /// `const Dtype&` overload (`:1890-1909`), which differs only in seeding the constant leaf and is
    /// inlined at the one place that wants it, [`clone_ignoring`](Self::clone_ignoring)'s zero-dim
    /// case.
    ///
    /// The affine constructor travels as an argument rather than off `self` because every caller is
    /// midway through rebuilding `self.parent_func`.
    fn create_leaf_func(
        affine: Option<GetDataAffine<D>>,
        prop: &FoldDimProp,
        ty: BaseFuncType,
    ) -> Option<FoldFunc<D>>
    where
        D: Default,
    {
        match ty {
            BaseFuncType::Constant => Some(FoldFunc::ConstantLeaf(ConstFoldFunctionLeaf::new(
                D::default(),
            ))),
            BaseFuncType::Map => Some(FoldFunc::MapLeaf(MapFoldFunctionLeaf::with_default_data(
                prop.size(),
            ))),
            BaseFuncType::Affine => Some(FoldFunc::AffineLeaf(AffineFoldFunctionLeaf {
                alpha: D::default(),
                beta: D::default(),
                get_data_affine: affine?,
            })),
            // `DT_ERROR("Unknown base fold function\n")` (`:1884-1885`) for `Unknown`, and for
            // `WkSplit` the kind this tree has no variant for — see [`build_dim`](Self::build_dim).
            BaseFuncType::WkSplit | BaseFuncType::Unknown => None,
        }
    }

    /// `createNonLeafFunc(const FoldDimProp*, BaseFuncType, FoldFunction<Dtype>* next_child)`
    /// (`:1911-1927`), with `next_child` already built — the authority's `nullptr` default and the
    /// `insertFunc` that fills it one call later are one step here.
    ///
    /// ⛔ MAP IGNORES `next_child` IN THE AUTHORITY (`:1917-1919` passes only the width) and
    /// `createSubTreeForEachMapChild` fills the children afterwards (`:2250-2257`). Here the child is
    /// CLONED into every slot, which is what that pair adds up to: each child is a fresh subtree over
    /// the same dimension properties, then `copySubTree`d from the node being replaced (`:1364`,
    /// `:1432`) — measured, both children of a new Map level carry the old level's data.
    /// ⛔ AND THERE IS NO WKSPLIT ARM, in the authority either (`:1911-1927` builds three kinds).
    fn wrap(
        affine: Option<GetDataAffine<D>>,
        prop: &FoldDimProp,
        ty: BaseFuncType,
        child: FoldFunc<D>,
    ) -> Option<FoldFunc<D>>
    where
        D: Clone + Default,
    {
        match ty {
            BaseFuncType::Constant => Some(FoldFunc::ConstantNonLeaf(
                ConstFoldFunctionNonLeaf::new(child),
            )),
            BaseFuncType::Affine => Some(FoldFunc::AffineNonLeaf(AffineFoldFunctionNonLeaf {
                alpha: D::default(),
                beta: D::default(),
                child: Box::new(child),
                get_data_affine: affine?,
            })),
            BaseFuncType::Map => {
                let width = usize::try_from(prop.size().0).ok()?;
                Some(FoldFunc::MapNonLeaf(MapFoldFunctionNonLeaf::new(vec![
                    child;
                    width
                ])))
            }
            BaseFuncType::WkSplit | BaseFuncType::Unknown => None,
        }
    }

    /// `createTree(std::deque<...> dim_prop)` (`:2227-2248`) — one node per entry, leaf at the last.
    ///
    /// ⛔ AN EMPTY LIST IS `DT_ERROR("Need at least one fold dim to build a tree")` (`:2228-2229`),
    /// (measured: `buildFoldSpace({})` throws). That is [`None`] here, and every caller checks it
    /// BEFORE mutating rather than after.
    fn create_tree(
        affine: Option<GetDataAffine<D>>,
        dim_prop: &[(FoldDimProp, BaseFuncType)],
    ) -> Option<FoldFunc<D>>
    where
        D: Clone + Default,
    {
        let (prop, ty) = dim_prop.first()?;
        if dim_prop.len() == 1 {
            return Self::create_leaf_func(affine, prop, *ty);
        }
        let child = Self::create_tree(affine, &dim_prop[1..])?;
        Self::wrap(affine, prop, *ty, child)
    }

    /// Whether [`create_tree`](Self::create_tree) can build every level of `dim_prop` — the guard
    /// that keeps a refused kind from being discovered halfway through a rebuild, where the authority
    /// throws out of a manager it has already half-destroyed.
    fn buildable(
        affine: Option<GetDataAffine<D>>,
        dim_prop: &[(FoldDimProp, BaseFuncType)],
    ) -> bool {
        !dim_prop.is_empty()
            && dim_prop.iter().all(|(_, ty)| match ty {
                BaseFuncType::Constant | BaseFuncType::Map => true,
                BaseFuncType::Affine => affine.is_some(),
                BaseFuncType::WkSplit | BaseFuncType::Unknown => false,
            })
    }

    /// `collectFoldFunctionAtLevel(int pos, std::vector<FoldFunction<Dtype>*>&)` (`:1726-1766`) —
    /// every node at depth `pos`, `pos == 0` being the root alone (`:1731-1732`).
    ///
    /// ⭐ `pub(crate)` FOR ONE CALLER: `CoordinateType::debug_print` walks a coordinate's levels
    /// through it (`dsc/dsc2.h:373-374`), which is the authority's only use outside this class.
    /// ⛔ ITS 2^20-ITERATION TIMEOUT (`:1735`, `:1763`) IS NOT PORTED. It guards against a cyclic
    /// `child_ff_` graph, and a [`Box`]ed child cannot be one.
    pub(crate) fn collect_at_level(&self, pos: usize) -> Option<Vec<&FoldFunc<D>>> {
        // `DT_CHECK(pos < dim_prop_.size())` (`:1728`) — measured, `collectFoldFunctionAtLevel(1)` on
        // a one-dimension manager throws.
        if pos >= self.dim_prop.len() {
            return None;
        }
        let mut level = vec![&self.parent_func];
        for _ in 0..pos {
            let mut next = Vec::new();
            for node in level {
                match node {
                    FoldFunc::ConstantNonLeaf(c) => next.push(c.child()),
                    FoldFunc::AffineNonLeaf(a) => next.push(a.child()),
                    FoldFunc::MapNonLeaf(m) => next.extend(m.children()),
                    // A leaf above the last level cannot happen: `dim_prop_` decides the depth.
                    FoldFunc::ConstantLeaf(_) | FoldFunc::AffineLeaf(_) | FoldFunc::MapLeaf(_) => {}
                }
            }
            level = next;
        }
        Some(level)
    }

    /// [`collect_at_level`](Self::collect_at_level) for the rebuilders, which replace what they find.
    fn collect_at_level_mut(root: &mut FoldFunc<D>, pos: usize) -> Vec<&mut FoldFunc<D>> {
        let mut level = vec![root];
        for _ in 0..pos {
            let mut next = Vec::new();
            for node in level {
                match node {
                    FoldFunc::ConstantNonLeaf(c) => next.push(c.child_mut()),
                    FoldFunc::AffineNonLeaf(a) => next.push(a.child_mut()),
                    FoldFunc::MapNonLeaf(m) => next.extend(m.children_mut().iter_mut()),
                    FoldFunc::ConstantLeaf(_) | FoldFunc::AffineLeaf(_) | FoldFunc::MapLeaf(_) => {}
                }
            }
            level = next;
        }
        level
    }

    /// `getLinearFuncList(FoldFunction<Dtype>*, std::vector<FoldFunction<Dtype>*>&)` (`:1842-1868`) —
    /// breadth-first from one node, that node included.
    fn linear_list(root: &FoldFunc<D>) -> Vec<&FoldFunc<D>> {
        let mut out = Vec::new();
        let mut fifo = VecDeque::from([root]);
        while let Some(node) = fifo.pop_front() {
            out.push(node);
            match node {
                FoldFunc::ConstantNonLeaf(c) => fifo.push_back(c.child()),
                FoldFunc::AffineNonLeaf(a) => fifo.push_back(a.child()),
                FoldFunc::MapNonLeaf(m) => fifo.extend(m.children()),
                FoldFunc::ConstantLeaf(_) | FoldFunc::AffineLeaf(_) | FoldFunc::MapLeaf(_) => {}
            }
        }
        out
    }

    /// `getLinearFuncList(std::map<int64_t, int64_t> ignore_dim_idx_and_dim_val, ...)`
    /// (`:1793-1834`) — the whole tree breadth-first, with the levels in `ignore` LEFT OUT of the
    /// list, and a Map level in `ignore` descending through its named child only (`:1820-1828`).
    ///
    /// ⛔ AN IGNORED NON-MAP LEVEL IS STILL DESCENDED THROUGH; only its own node is dropped.
    /// ⛔ AND AT ZERO DIMENSIONS THE AUTHORITY ANSWERS AN EMPTY LIST, NOT A ONE-NODE ONE. Both
    /// overloads guard the initial push with `if (dim_prop_.size())` (`:1797`, `:1845`) — even the one
    /// that takes the node to start from — so the root is left out of its own list and everything built
    /// on the list does nothing: `copy` copies no payload, `operator==` compares none, and
    /// `deleteSubTree` leaks the tree. This port always lists the root. No caller can see the
    /// difference: [`clone_ignoring`](Self::clone_ignoring) and [`assign`](Self::assign) are its only
    /// two, and both answer the zero-dimension case before the list is built — which is what the
    /// authority's own `getData()`-comparing branch in `operator==` (`:1112-1117`) does too.
    fn linear_list_ignoring(&self, ignore: &BTreeMap<usize, usize>) -> Vec<&FoldFunc<D>> {
        let mut out = Vec::new();
        let mut fifo = VecDeque::from([(&self.parent_func, 0usize)]);
        while let Some((node, pos)) = fifo.pop_front() {
            let ignored = ignore.get(&pos);
            if ignored.is_none() {
                out.push(node);
            }
            match node {
                FoldFunc::ConstantNonLeaf(c) => fifo.push_back((c.child(), pos + 1)),
                FoldFunc::AffineNonLeaf(a) => fifo.push_back((a.child(), pos + 1)),
                FoldFunc::MapNonLeaf(m) => match ignored {
                    None => fifo.extend(m.children().iter().map(|child| (child, pos + 1))),
                    Some(&child_idx) => {
                        if let Some(child) = m.children().get(child_idx) {
                            fifo.push_back((child, pos + 1));
                        }
                    }
                },
                FoldFunc::ConstantLeaf(_) | FoldFunc::AffineLeaf(_) | FoldFunc::MapLeaf(_) => {}
            }
        }
        out
    }
    /// `static copy(std::vector<FoldFunction<Dtype>*>& lhs_list, ... rhs_list)` (`:1046-1088`) — the
    /// payload of every node of one tree into the node at the same POSITION of the other's list.
    ///
    /// ⛔ IT PAIRS BY POSITION, NOT BY SHAPE, and that is observable: cloning a manager while ignoring
    /// a Map level pairs the surviving `Map_NonLeaf` with a `Map_Leaf`, the per-kind `dynamic_cast`
    /// misses, and the copy SILENTLY DOES NOTHING — measured, an affine level's alpha survives such a
    /// clone while a map leaf's data does not.
    /// ⛔ AND ONLY FOUR PAIRINGS CARRY ANYTHING. `Const_NonLeaf` and `Map_NonLeaf` hold no payload, and
    /// the authority's `else` is an explicit "do nothing" (`:1086`). Its fifth arm is `WkSplit_leaf`
    /// (`:1067-1072`), the kind this tree has no variant for — see [`build_dim`](Self::build_dim).
    /// ⛔ THE AUTHORITY SWITCHES ON THE **RHS**'s kind and `dynamic_cast`s the lhs (`:1057-1086`), so a
    /// mismatched pair dereferences a null there. Testing both sides is the same copy without the
    /// undefined behaviour.
    ///
    /// The left side is walked rather than collected because a node and its child cannot both be held
    /// mutably; the walk visits in the same breadth-first order [`linear_list`](Self::linear_list)
    /// builds, so the pairing is the authority's.
    fn copy_lists(lhs_root: &mut FoldFunc<D>, rhs_list: &[&FoldFunc<D>]) -> Option<()>
    where
        D: Clone,
    {
        // `DT_ERROR("Unexpected")` (`:1048-1049`) — and before anything is written, as there.
        if Self::linear_list(lhs_root).len() != rhs_list.len() {
            return None;
        }
        let mut idx = 0;
        let mut fifo = VecDeque::from([lhs_root]);
        while let Some(node) = fifo.pop_front() {
            let rhs = *rhs_list.get(idx)?;
            idx += 1;
            match node {
                FoldFunc::ConstantLeaf(lhs) => {
                    if let FoldFunc::ConstantLeaf(rhs) = rhs {
                        lhs.insert_data(rhs.get_data()?);
                    }
                }
                FoldFunc::AffineLeaf(lhs) => {
                    if let FoldFunc::AffineLeaf(rhs) = rhs {
                        lhs.insert_alpha(rhs.alpha().clone());
                        lhs.insert_beta(rhs.beta().clone());
                    }
                }
                FoldFunc::MapLeaf(lhs) => {
                    if let FoldFunc::MapLeaf(rhs) = rhs {
                        lhs.copy_data_vec_from(rhs);
                    }
                }
                FoldFunc::AffineNonLeaf(lhs) => {
                    if let FoldFunc::AffineNonLeaf(rhs) = rhs {
                        lhs.insert_alpha(rhs.alpha().clone());
                        lhs.insert_beta(rhs.beta().clone());
                    }
                    fifo.push_back(lhs.child_mut());
                }
                FoldFunc::ConstantNonLeaf(lhs) => fifo.push_back(lhs.child_mut()),
                FoldFunc::MapNonLeaf(lhs) => fifo.extend(lhs.children_mut().iter_mut()),
            }
        }
        Some(())
    }

    /// `copySubTree(FoldFunction<Dtype>* sub_tree, FoldFunction<Dtype>* ref_sub_tree)`
    /// (`:2470-2522`) — two linear lists and the per-kind copy over them.
    ///
    /// ⛔ IT DOES NOT CALL `copy` (`:1046-1088`); it INLINES the same five-arm dispatch (`:2491-2520`
    /// against `:1057-1086`), the two arm orders differing and nothing else. So it is
    /// [`copy_lists`](Self::copy_lists) here, and a second transcription would only be a second place
    /// for the arms to drift.
    fn copy_sub_tree(sub_tree: &mut FoldFunc<D>, ref_sub_tree: &FoldFunc<D>) -> Option<()>
    where
        D: Clone,
    {
        Self::copy_lists(sub_tree, &Self::linear_list(ref_sub_tree))
    }
}

impl<D> FoldManager<D> {
    /// `getData(const std::deque<int64_t>&)` (`:1691-1703`) — the value at one coordinate.
    ///
    /// ⛔ THE AUTHORITY'S `dynamic_cast` CHAIN (`:1693-1700`) IS REDUNDANT: both affine classes
    /// override the virtual `getData(deque, idx)` to call `getDataAffine` (`:385-388`, `:556-559`), so
    /// the fallback at `:1702` reaches the same body the chain does. Every kind's walk is one dispatch
    /// here, the `match` in [`FoldFunc::get_data`], which is what the chain and the fallback agree on.
    pub fn get_data(&self, fold_dim_indices: &[FoldDimIndex]) -> Option<D>
    where
        D: Clone,
    {
        // `isLegal(fold_dim_indices)` for its throw alone (`:1692`) — the walk never starts on an
        // illegal coordinate, which is what keeps `ConstFoldFunction_Leaf`'s missing depth check
        // unobservable.
        self.is_legal(fold_dim_indices)?;
        self.parent_func.get_data(fold_dim_indices)
    }

    /// `insertData(const Dtype&, const std::deque<int64_t>&)` (`:1712-1716`).
    pub fn insert_data(&mut self, new_data: D, fold_dim_indices: &[FoldDimIndex]) -> Option<()> {
        self.is_legal(fold_dim_indices)?;
        self.parent_func.insert_data(new_data, fold_dim_indices)
    }

    /// `getSingleData(const std::map<int64_t, int64_t>& pos_to_fixCoord)` (`:1934-1939`) — the value
    /// at coordinate zero in every dimension the map does not fix.
    ///
    /// ⛔ `fold_dim_indices.at(dim)` (`:1937`) THROWS for a position past the last dimension, and a
    /// zero-dimension manager with a non-empty map is exactly that case (measured).
    pub fn single_data(&self, pos_to_fix_coord: &BTreeMap<FoldDimPos, FoldDimIndex>) -> Option<D>
    where
        D: Clone,
    {
        let mut fold_dim_indices = vec![FoldDimIndex(0); self.dim_prop.len()];
        for (pos, coord) in pos_to_fix_coord {
            let idx = pos.index(fold_dim_indices.len())?;
            fold_dim_indices[idx] = *coord;
        }
        self.get_data(&fold_dim_indices)
    }

    /// `getNumUniqueCoordsInEachFold(std::vector<int64_t>&)` (`:1947-1961`) — how many DISTINCT values
    /// each dimension can hold, and their product.
    ///
    /// ⛔ A CONSTANT DIMENSION CONTRIBUTES 1, NOT ITS EXTENT, and it does not enter the product at all
    /// — that is the whole point of the Constant kind, and why a 3x2 space with a constant outer dim
    /// flattens to two coordinates and not six.
    /// ⛔ THE AUTHORITY ACCUMULATES THE PRODUCT IN AN `int` AND RETURNS IT AS `int64_t` (`:1951`), so
    /// a fold space of more than 2^31 coordinates overflows there. Accumulated at the return width
    /// here; no fold space in scope is within six orders of magnitude of that.
    fn unique_coords_in_each_fold(&self) -> (Vec<i64>, i64) {
        let mut unique_data_coords = Vec::new();
        let mut total_count: i64 = 1;
        for (prop, ty) in &self.dim_prop {
            if *ty == BaseFuncType::Constant {
                unique_data_coords.push(1);
            } else {
                unique_data_coords.push(i64::from(prop.size().0));
                total_count = total_count.saturating_mul(i64::from(prop.size().0));
            }
        }
        (unique_data_coords, total_count)
    }

    /// `getAllCoordsInEachFold(std::vector<int64_t>&)` (`:1963-1972`) — the same, but a Constant
    /// dimension counts its full extent, which is how a Map level gets unrolled.
    fn all_coords_in_each_fold(&self) -> (Vec<i64>, i64) {
        let mut all_fold_coords = Vec::new();
        let mut total_count: i64 = 1;
        for (prop, _) in &self.dim_prop {
            all_fold_coords.push(i64::from(prop.size().0));
            total_count = total_count.saturating_mul(i64::from(prop.size().0));
        }
        (all_fold_coords, total_count)
    }

    /// `getFlattenedCoordinates(pos_to_fixCoord, scan_inner_outer, expand_if_any_map)`
    /// (`:1983-2032`, and the returning overload `:2043-2051`) — every coordinate of the fold space,
    /// in the order a caller walking the data will see them.
    ///
    /// ⛔ A ZERO-DIMENSION MANAGER YIELDS NO COORDINATES AT ALL — the authority leaves the output
    /// vector untouched (`:1988` guards the whole body), so the `[-1]` coordinate is
    /// [`data_and_fold_coordinates`](Self::data_and_fold_coordinates)'s doing and not this method's.
    /// ⛔ `expand_if_any_map` DOES NOTHING UNLESS SOME UNFIXED MAP LEVEL IS WIDER THAN ONE
    /// (`:1990-1998`) — a fixed Map dimension is not an expansion, which is why fixing the Map level
    /// and unrolling it give different lengths.
    pub fn flattened_coordinates(
        &self,
        pos_to_fix_coord: &BTreeMap<FoldDimPos, FoldDimIndex>,
        scan_inner_outer: ScanInnerOuter,
        expand_if_any_map: ExpandIfAnyMap,
    ) -> Option<Vec<Vec<FoldDimIndex>>> {
        if self.dim_prop.is_empty() {
            return Some(Vec::new());
        }
        let mut do_expand = false;
        if expand_if_any_map.0 {
            for (i, (prop, ty)) in self.dim_prop.iter().enumerate() {
                if !pos_to_fix_coord.contains_key(&FoldDimPos(i32::try_from(i).ok()?))
                    && *ty == BaseFuncType::Map
                    && prop.size().0 > 1
                {
                    do_expand = true;
                    break;
                }
            }
        }
        let (mut unique_data_coords, mut total_count) = if do_expand {
            self.all_coords_in_each_fold()
        } else {
            self.unique_coords_in_each_fold()
        };
        for pos in pos_to_fix_coord.keys() {
            // `unique_data_coords.at(kv.first)` (`:2008-2010`) throws past the last dimension, and the
            // `DT_CHECK` beside it rejects a fixed dimension that does not divide the product.
            let idx = pos.index(unique_data_coords.len())?;
            if unique_data_coords[idx] == 0 || total_count % unique_data_coords[idx] != 0 {
                return None;
            }
            total_count /= unique_data_coords[idx];
            unique_data_coords[idx] = 1;
        }
        let total_count = usize::try_from(total_count).ok()?;
        let mut coordinates = vec![vec![FoldDimIndex(0); unique_data_coords.len()]; total_count];
        let mut repeat_factor: i64 = 1;
        for i in 0..unique_data_coords.len() {
            let dim_idx = if scan_inner_outer.0 {
                unique_data_coords.len() - 1 - i
            } else {
                i
            };
            // Every extent here is at least 1 whenever there is a coordinate to write: a zero extent
            // makes the product zero and the loop body never runs, which is why the authority's
            // `% unique_data_coords.at(dim_idx)` never divides by zero either.
            for (coord_idx, coord) in coordinates.iter_mut().enumerate() {
                let coord_idx = i64::try_from(coord_idx).ok()?;
                coord[dim_idx] =
                    FoldDimIndex((coord_idx / repeat_factor) % unique_data_coords[dim_idx]);
            }
            repeat_factor = repeat_factor.saturating_mul(unique_data_coords[dim_idx]);
        }
        // `FoldInfraUtils::fixCoordinatesAtPos(coordinates, pos_to_fixCoord)` (`:100-111`), inlined
        // because it has no other caller in the authority either. Its `DT_CHECK(coord.at(dim_idx) ==
        // 0)` (`:106`) is what makes fixing a dimension the flattening did not collapse a refusal
        // rather than a silent overwrite.
        for coord in &mut coordinates {
            for (pos, fixed) in pos_to_fix_coord {
                let idx = pos.index(coord.len())?;
                if coord[idx] != FoldDimIndex(0) {
                    return None;
                }
                coord[idx] = *fixed;
            }
        }
        Some(coordinates)
    }

    /// `getDataAndFoldCoordinates(pos_to_fixCoord, scan_inner_outer)` (`:2062-2079`, and the
    /// returning overload `:2090-2097`) — every coordinate paired with the value there.
    ///
    /// ⛔ A ZERO-DIMENSION MANAGER ANSWERS ONE PAIR AT THE COORDINATE `[-1]` (`:2066-2069`). That `-1`
    /// is a SENTINEL, not an index — `isLegal` waves it through because a zero-dimension manager
    /// calls every list legal, and `ConstFoldFunction_Leaf` never reads it.
    pub fn data_and_fold_coordinates(
        &self,
        pos_to_fix_coord: &BTreeMap<FoldDimPos, FoldDimIndex>,
        scan_inner_outer: ScanInnerOuter,
    ) -> Option<Vec<(Vec<FoldDimIndex>, D)>>
    where
        D: Clone,
    {
        if self.dim_prop.is_empty() {
            let zero_fold_coord = vec![FoldDimIndex(-1)];
            let data = self.get_data(&[])?;
            return Some(vec![(zero_fold_coord, data)]);
        }
        let coordinates =
            self.flattened_coordinates(pos_to_fix_coord, scan_inner_outer, ExpandIfAnyMap(false))?;
        let mut data_and_coord = Vec::with_capacity(coordinates.len());
        for coord in coordinates {
            let data = self.get_data(&coord)?;
            data_and_coord.push((coord, data));
        }
        Some(data_and_coord)
    }

    /// `getAllData(pos_to_fixCoord, scan_inner_outer)` (`:2108-2125`) — the values alone.
    pub fn all_data(
        &self,
        pos_to_fix_coord: &BTreeMap<FoldDimPos, FoldDimIndex>,
        scan_inner_outer: ScanInnerOuter,
    ) -> Option<Vec<D>>
    where
        D: Clone,
    {
        self.all_data_expanding(pos_to_fix_coord, scan_inner_outer, ExpandIfAnyMap(false))
    }

    /// `getAllDataWithMapUnrolled(pos_to_fixCoord, scan_inner_outer)` (`:2127-2145`) — `getAllData`
    /// with `expand_if_any_map` set, so a Map level contributes one value per coordinate instead of
    /// one per distinct value.
    pub fn all_data_with_map_unrolled(
        &self,
        pos_to_fix_coord: &BTreeMap<FoldDimPos, FoldDimIndex>,
        scan_inner_outer: ScanInnerOuter,
    ) -> Option<Vec<D>>
    where
        D: Clone,
    {
        self.all_data_expanding(pos_to_fix_coord, scan_inner_outer, ExpandIfAnyMap(true))
    }

    /// The body the two `getAllData` forms share — identical but for the flag (`:2118` against
    /// `:2137-2138`).
    fn all_data_expanding(
        &self,
        pos_to_fix_coord: &BTreeMap<FoldDimPos, FoldDimIndex>,
        scan_inner_outer: ScanInnerOuter,
        expand_if_any_map: ExpandIfAnyMap,
    ) -> Option<Vec<D>>
    where
        D: Clone,
    {
        if self.dim_prop.is_empty() {
            return Some(vec![self.get_data(&[])?]);
        }
        let coordinates =
            self.flattened_coordinates(pos_to_fix_coord, scan_inner_outer, expand_if_any_map)?;
        coordinates
            .iter()
            .map(|coord| self.get_data(coord))
            .collect()
    }

    /// `apply(pos_to_fixCoord, func, funcArgs...)` (`:1257-1270`) — replace every value by
    /// `func(value)`.
    ///
    /// ⛔ THE VARIADIC `funcArgs` ARE NOT A PARAMETER HERE. They are C++'s way of binding extra
    /// arguments to a function object; a Rust closure captures them, and every in-scope caller passes
    /// a lambda.
    /// ⛔ A ZERO-DIMENSION MANAGER APPLIES ONCE AT THE EMPTY COORDINATE (`:1260-1262`), NOT at the
    /// `[-1]` one `getDataAndFoldCoordinates` reports.
    pub fn apply(
        &mut self,
        pos_to_fix_coord: &BTreeMap<FoldDimPos, FoldDimIndex>,
        mut func: impl FnMut(D) -> D,
    ) -> Option<()>
    where
        D: Clone,
    {
        if self.dim_prop.is_empty() {
            let data = self.get_data(&[])?;
            return self.insert_data(func(data), &[]);
        }
        let data_and_coord =
            self.data_and_fold_coordinates(pos_to_fix_coord, ScanInnerOuter(false))?;
        for (coord, data) in data_and_coord {
            self.insert_data(func(data), &coord)?;
        }
        Some(())
    }

    /// `apply(const FoldManager<Dtype2>& rhs, pos_to_fixCoord, func, funcArgs...)` (`:1287-1317`) —
    /// combine every value with the value at the same coordinate of another manager, whose payload
    /// may be a different type.
    ///
    /// ⛔ THE COMPATIBILITY GATE IS ASYMMETRIC AND IT IS ABOUT **THIS** MANAGER'S KINDS
    /// (`:1296-1305`): a Constant level here may not meet a non-Constant level there, because the one
    /// value it holds would be written once per coordinate of the other; and an Affine or WkSplit
    /// level here is refused outright, because its value is computed and cannot be written back at
    /// all. A Map level here meets anything.
    pub fn apply_with<D2>(
        &mut self,
        rhs: &FoldManager<D2>,
        pos_to_fix_coord: &BTreeMap<FoldDimPos, FoldDimIndex>,
        mut func: impl FnMut(D, D2) -> D,
    ) -> Option<()>
    where
        D: Clone,
        D2: Clone,
    {
        // `DT_ERROR("Fold managers are not the same in apply function")` (`:1291-1293`).
        if rhs.fold_space_size() != self.fold_space_size() {
            return None;
        }
        for (mine, theirs) in self.func_types().iter().zip(rhs.func_types()) {
            let compatible = match mine {
                BaseFuncType::Constant => theirs == BaseFuncType::Constant,
                BaseFuncType::Map | BaseFuncType::Unknown => true,
                BaseFuncType::Affine | BaseFuncType::WkSplit => false,
            };
            if !compatible {
                return None;
            }
        }
        if self.dim_prop.is_empty() {
            let data = self.get_data(&[])?;
            let rhs_data = rhs.get_data(&[])?;
            return self.insert_data(func(data, rhs_data), &[]);
        }
        let data_and_coord =
            self.data_and_fold_coordinates(pos_to_fix_coord, ScanInnerOuter(false))?;
        for (coord, data) in data_and_coord {
            let rhs_data = rhs.get_data(&coord)?;
            self.insert_data(func(data, rhs_data), &coord)?;
        }
        Some(())
    }

    /// `getAllDimProFromPos(int pos, fm_dim_prop&)` (`:2259-2263`) — the dimension list from `pos`
    /// inward, which is the list a rebuild of that level and everything under it is built from.
    pub fn dim_prop_from(&self, pos: FoldDimPos) -> Option<&[(FoldDimProp, BaseFuncType)]> {
        // `DT_CHECK(pos >= 0)` (`:2260`); a `pos` past the end yields an empty list there, and here.
        let pos = usize::try_from(pos.0).ok()?;
        Some(self.dim_prop.get(pos..).unwrap_or(&[]))
    }

    /// `printMetaData(std::ostream&, std::string ps, bool add_comma, bool compressed)`
    /// (`:2186-2225`) — the fold space as the two JSON arrays `dim_prop_func` and `dim_prop_attr`.
    ///
    /// ⛔ ONLY THE FIRST NODE OF A LEVEL IS PRINTED (`:2205`), so a Map level above an Affine one
    /// reports one alpha for a level that has one per Map child. Faithful; the authority's own
    /// `getAlpha` refuses that shape outright (`:2339`) rather than reporting the first.
    /// ⛔ AND A MAP OR CONSTANT LEVEL PRINTS NO METADATA AT ALL (`:2196-2198`), not even an empty
    /// object — the braces are the value.
    pub fn print_meta_data(
        &self,
        out: &mut String,
        ps: &str,
        add_comma: AddComma,
        compressed: Compressed,
    ) -> Option<()>
    where
        D: fmt::Display,
    {
        let ps2 = format!("{ps}  ");
        out.push_str(ps);
        out.push_str("\"dim_prop_func\" : [\n");
        for (idx, (_, ty)) in self.dim_prop.iter().enumerate() {
            out.push_str(&ps2);
            out.push_str("{ \"");
            // `baseFuncTypeToString.at(...)` (`:2193-2194`) throws on the `Unknown` sentinel.
            out.push_str(ty.as_str()?);
            out.push_str("\" : {");
            if *ty != BaseFuncType::Map && *ty != BaseFuncType::Constant {
                // `DT_CHECK(ffs_at_pos.size())` (`:2204`).
                match self.collect_at_level(idx)?.first()? {
                    FoldFunc::AffineNonLeaf(affine) => affine.print_meta_data(out, ""),
                    FoldFunc::AffineLeaf(affine) => affine.print_meta_data(out, ""),
                    // The base's `printMetaData` is a `DT_ERROR` stub (`:254-256`) and no other kind
                    // overrides it, but no other kind can be at a non-Map non-Constant level either.
                    FoldFunc::ConstantNonLeaf(_)
                    | FoldFunc::ConstantLeaf(_)
                    | FoldFunc::MapNonLeaf(_)
                    | FoldFunc::MapLeaf(_) => return None,
                }
            }
            out.push_str("} }");
            if idx + 1 != self.dim_prop.len() {
                out.push(',');
            }
            out.push('\n');
        }
        out.push_str(ps);
        out.push(']');
        if !compressed.0 {
            out.push_str(",\n");
            out.push_str(ps);
            out.push_str("\"dim_prop_attr\" : [");
            for (idx, (prop, _)) in self.dim_prop.iter().enumerate() {
                out.push('\n');
                out.push_str(&ps2);
                out.push_str("{ ");
                prop.print(out);
                out.push_str(" }");
                if idx + 1 != self.dim_prop.len() {
                    out.push(',');
                }
            }
            out.push('\n');
            out.push_str(ps);
            out.push(']');
        }
        if add_comma.0 {
            out.push(',');
        }
        out.push('\n');
        Some(())
    }

    /// `print(std::ostream&, std::string ps, bool printContent)` (`:2726-2750`) — the whole manager as
    /// one JSON object: the fold space [`print_meta_data`](Self::print_meta_data) writes, then every
    /// value keyed by the coordinate it sits at.
    ///
    /// ⛔ A ZERO-DIMENSION MANAGER PRINTS ITS VALUE AND NOTHING ELSE (`:2730-2731`) — no braces, no
    /// metadata, and `ps` is not used at all. Measured: `"42"`. So this is not a JSON object at every
    /// shape, and a caller that concatenates it is relying on that.
    /// ⛔ `printContent` IS PASSED AS `printMetaData`'s **`add_comma`** (`:2735`), not as its
    /// `compressed`. One flag, two jobs: whether the values follow, and the comma before them.
    /// ⛔ AND THE VALUES ARE ALWAYS INNER-OUTER (`:2737` passes `scan_inner_outer`), whichever order
    /// the caller reads data in elsewhere.
    pub fn print(&self, out: &mut String, ps: &str, print_content: PrintContent) -> Option<()>
    where
        D: Clone + fmt::Display,
    {
        if self.has_zero_fold_dim() {
            Self::print_data(out, &self.get_data(&[])?);
            return Some(());
        }
        let ps2 = format!("{ps}  ");
        out.push_str(" {\n");
        self.print_meta_data(out, &ps2, AddComma(print_content.0), Compressed(false))?;
        if print_content.0 {
            let data_and_coord =
                self.data_and_fold_coordinates(&BTreeMap::new(), ScanInnerOuter(true))?;
            out.push_str(&ps2);
            out.push_str("\"data_\" : {\n");
            let mut remaining = data_and_coord.len();
            for (coord, data) in data_and_coord {
                // `PrintUtil::printVec` over the coordinate deque (`util/print_utils.h:313-323`) —
                // ", "-separated and NOT quoted, unlike the value beside it.
                let coord: Vec<String> = coord.iter().map(|c| c.0.to_string()).collect();
                out.push_str(&ps2);
                out.push_str("  \"[");
                out.push_str(&coord.join(", "));
                out.push_str("]\" :");
                Self::print_data(out, &data);
                remaining -= 1;
                out.push_str(if remaining > 0 { ",\n" } else { "\n" });
            }
            out.push_str(&ps2);
            out.push_str("}\n");
        }
        out.push_str(ps);
        out.push('}');
        Some(())
    }

    /// `printData(std::ostream&, const Dtype&)` (`:2684-2689`) — one value, quoted.
    ///
    /// ⛔ IT IS FOUR SFINAE ARMS AND THIS IS THE ARITHMETIC ONE. A `std::set` (`:2691-2698`) or
    /// `std::vector` (`:2700-2707`) payload prints as a bracketed list of QUOTED elements (measured:
    /// `["1", "2", "3"]`), and any other payload is
    /// `DT_ERROR("Print util is not available for this Dtype\n")` (`:2709-2718`) — a run-time throw for
    /// a question the payload already answers, so here it is the [`fmt::Display`] bound: a payload with
    /// no printable form has no [`print`](Self::print) at all. That leaves the two list arms
    /// UNREACHABLE rather than unported — `Vec<i64>` is the only such payload in scope and is not
    /// [`fmt::Display`], and all three call sites in scope (`dsc/pcfg.cpp:583`, `dsc/pcfg.cpp:592`,
    /// `dsc/dsc2.h:281`) print address and coordinate managers whose payload is `int64_t`.
    fn print_data(out: &mut String, data: &D)
    where
        D: fmt::Display,
    {
        out.push('"');
        out.push_str(&data.to_string());
        out.push('"');
    }
}

/// `add_comma` (`:2187`) — whether `printMetaData` ends its last line with a comma, because its
/// caller may have another member to print after it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AddComma(pub bool);

/// `compressed` (`:2187`) — `true` omits the `dim_prop_attr` array, printing which fold function each
/// level is without the extents and labels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Compressed(pub bool);

/// `printContent` (`:2727`) — whether [`FoldManager::print`] prints the values after the fold space.
///
/// ⛔ IT IS ALSO THAT CALL'S [`AddComma`] (`:2735`), which is why it cannot be one of the two flags
/// `printMetaData` already has: it decides both whether the values follow and the comma that would
/// separate them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrintContent(pub bool);

impl<D: Clone + Default> FoldManager<D> {
    /// `buildDim(const FoldDimProp* prop, BaseFuncType, int pos)` (`:1337-1458`) — insert ONE fold
    /// dimension at `pos`, keeping what is already built where it can.
    ///
    /// The four cases are the authority's, and two of them are one here:
    /// - **no dimensions yet** (`:1339-1348`): the tree becomes this level's leaf. A Constant level
    ///   KEEPS the existing constant leaf and its value (`:1341-1343`, measured — a manager built with
    ///   9 still holds 9); any other kind replaces it and the value is gone.
    /// - **`pos` inside the list** (`:1351-1371` for 0, `:1407-1456` for the middle): each node at
    ///   depth `pos` is wrapped in a new node of this kind, so everything below it survives. The
    ///   authority writes the two separately — `copySubTree` from the old parent at 0, from the
    ///   current child in the middle — and both say "the node that was here becomes this new node's
    ///   child", which is what [`wrap`](Self::wrap) does.
    /// - **`pos == dim_prop_.size()`** (`:1372-1406`): appending. ⛔ THE OLD LEAF LEVEL IS REBUILT
    ///   FROM SCRATCH AND ITS DATA IS DISCARDED — there is no `copySubTree` on this path, measured: a
    ///   manager holding 55 holds 0 after appending a dimension, while inserting one in the middle
    ///   keeps 66. That is the authority's behaviour and callers depend on the shape, not the values.
    ///
    /// ⛔ A NEGATIVE `pos` IS UNDEFINED BEHAVIOUR IN THE AUTHORITY, not a from-the-end position: the
    /// `pos == 0` and `pos == size()` tests both fail, `DT_CHECK(pos < dim_prop_.size())` (`:1408`)
    /// negative one, and `dim_prop_.insert(begin() + pos, ...)` then writes before the vector. [`None`]
    /// here. [`FoldDimPos`] records which members do accept one.
    pub fn build_dim(
        &mut self,
        prop: &FoldDimProp,
        ty: BaseFuncType,
        pos: FoldDimPos,
    ) -> Option<()> {
        let affine = self.get_data_affine;
        let len = self.dim_prop.len();
        let pos = usize::try_from(pos.0).ok()?;
        // `DT_CHECK(pos == 0)` at zero dimensions (`:1340`) and `DT_CHECK(pos < dim_prop_.size())` on
        // the middle path (`:1408`); `pos == dim_prop_.size()` is the append case and legal.
        if pos > len {
            return None;
        }
        let new_dim = (prop.clone(), ty);
        // Whether this kind can be built at all is settled BEFORE anything is mutated: the authority
        // discovers it inside `createLeafFunc`/`createNonLeafFunc` and throws out of a half-rebuilt
        // tree.
        if !Self::buildable(affine, core::slice::from_ref(&new_dim)) {
            return None;
        }
        if len == 0 {
            if ty == BaseFuncType::Constant {
                // `DT_CHECK(parent_func_->Type() == FoldFunction<Dtype>::ConstFoldFunction_Leaf)`
                // (`:1341-1343`) — and a zero-dimension manager holds nothing else.
                if self.parent_func.ty() != FuncType::ConstantLeaf {
                    return None;
                }
            } else {
                self.parent_func = Self::create_leaf_func(affine, prop, ty)?;
            }
            self.dim_prop.push(new_dim);
            return Some(());
        }
        if pos == len {
            // The new leaf level and the level that used to be the leaf, rebuilt as a pair.
            let sub = vec![self.dim_prop[len - 1].clone(), new_dim.clone()];
            if !Self::buildable(affine, &sub) {
                return None;
            }
            self.dim_prop.push(new_dim);
            for node in Self::collect_at_level_mut(&mut self.parent_func, len - 1) {
                *node = Self::create_tree(affine, &sub)?;
            }
            return Some(());
        }
        self.dim_prop.insert(pos, new_dim);
        for node in Self::collect_at_level_mut(&mut self.parent_func, pos) {
            let old = core::mem::replace(
                node,
                FoldFunc::ConstantLeaf(ConstFoldFunctionLeaf::new(D::default())),
            );
            *node = Self::wrap(affine, prop, ty, old)?;
        }
        Some(())
    }

    /// `buildConstDim(const FoldDimProp*, int pos = 0)` (`:1320-1322`).
    pub fn build_const_dim(&mut self, prop: &FoldDimProp, pos: FoldDimPos) -> Option<()> {
        self.build_dim(prop, BaseFuncType::Constant, pos)
    }

    /// `buildMapDim(const FoldDimProp*, int pos = 0)` (`:1324-1326`).
    pub fn build_map_dim(&mut self, prop: &FoldDimProp, pos: FoldDimPos) -> Option<()> {
        self.build_dim(prop, BaseFuncType::Map, pos)
    }

    /// `buildFoldSpace(fm_dim_prop dim_prop)` (`:1653-1657`) — discard the tree and build a whole new
    /// fold space.
    ///
    /// ⛔ AN EMPTY LIST IS NOT A ZERO-DIMENSION FOLD SPACE. The authority clears, then throws out of
    /// `createTree` (measured), leaving a manager with no tree at all; here the refusal happens before
    /// the clear, so the manager is untouched. [`reset`] is how a zero-dimension space is reached.
    ///
    /// [`reset`]: FoldManager::reset
    pub fn build_fold_space_dims(
        &mut self,
        dim_prop: &[(FoldDimProp, BaseFuncType)],
    ) -> Option<()> {
        let affine = self.get_data_affine;
        if !Self::buildable(affine, dim_prop) {
            return None;
        }
        self.clear();
        self.dim_prop = dim_prop.to_vec();
        self.parent_func = Self::create_tree(affine, dim_prop)?;
        Some(())
    }

    /// `buildFoldSpace(std::deque<const FoldDimProp*>, std::deque<BaseFuncType>)` (`:1635-1645`).
    pub fn build_fold_space(
        &mut self,
        props: &[FoldDimProp],
        types: &[BaseFuncType],
    ) -> Option<()> {
        // `DT_ERROR("Size of props and func_base_types should be same")` (`:1637-1638`) — measured.
        if props.len() != types.len() {
            return None;
        }
        let dim_prop: Vec<_> = props.iter().cloned().zip(types.iter().copied()).collect();
        self.build_fold_space_dims(&dim_prop)
    }

    /// `buildAllConstantFoldSpace(std::deque<const FoldDimProp*>)` (`:1599-1609`).
    pub fn build_all_constant_fold_space(&mut self, props: &[FoldDimProp]) -> Option<()> {
        let dim_prop: Vec<_> = props
            .iter()
            .cloned()
            .map(|prop| (prop, BaseFuncType::Constant))
            .collect();
        self.build_fold_space_dims(&dim_prop)
    }

    /// `buildAllMapFoldSpace(std::deque<const FoldDimProp*>)` (`:1617-1626`).
    pub fn build_all_map_fold_space(&mut self, props: &[FoldDimProp]) -> Option<()> {
        let dim_prop: Vec<_> = props
            .iter()
            .cloned()
            .map(|prop| (prop, BaseFuncType::Map))
            .collect();
        self.build_fold_space_dims(&dim_prop)
    }

    /// `rebuildDim(int pos, BaseFuncType)` (`:1473-1484`) — change ONE level's fold function, keeping
    /// the data below it where the two kinds allow. `false` when `pos` names no dimension.
    ///
    /// ⛔ ITS RANGE TEST UNDERFLOWS AT ZERO DIMENSIONS and the throw that follows is NOT the `false`
    /// this returns for an out-of-range `pos` — measured: `rebuildDim(5, Map)` on a two-dimension
    /// manager answers `false`, `rebuildDim(0, Map)` on a zero-dimension one throws. Hence
    /// [`Option<bool>`]: [`None`] is the throw, `Some(false)` the refusal.
    /// ⛔ THE THREE-ARGUMENT FORM (`:1494-1591`) IS NOT SEPARATELY PUBLIC. Its extra parameter is the
    /// `FoldDimProp*` the manager already holds, asserted equal to it by POINTER
    /// (`DT_CHECK(dim_prop_.at(pos).first == prop)`, `:1503`) — an identity of the caller's own
    /// storage, which this port cannot spell because it owns its props instead of borrowing them. The
    /// two-argument form passes `dim_prop_.at(pos).first` (`:1479`), so it always holds, and the body
    /// is [`rebuild_dim_at`](Self::rebuild_dim_at).
    pub fn rebuild_dim(&mut self, pos: FoldDimPos, ty: BaseFuncType) -> Option<bool>
    where
        D: Clone,
    {
        if self.dim_prop.is_empty() {
            return None;
        }
        let Some(idx) = pos.resolve(self.dim_prop.len()) else {
            return Some(false);
        };
        self.rebuild_dim_at(idx, ty)?;
        Some(true)
    }

    /// The body of `rebuildDim(const FoldDimProp*, int pos, BaseFuncType)` (`:1494-1591`).
    ///
    /// ⛔ TWO INDEPENDENT DECISIONS, AND THEY DISAGREE. `need_to_rebuild` (`:1504-1510`) is true when
    /// either side is a Map or the kinds differ, because a Map level's node count changes and a
    /// different kind is a different class. `can_copy` (`:1512-1519`) is true only when the CURRENT
    /// kind is not Map, the target is not Map unless it is replacing a Constant, and the level is not
    /// the innermost. So `Const -> Affine` at an outer level keeps the data below; `Map -> Const`
    /// discards it; and rebuilding the LAST level always discards, measured — a two-level space's
    /// values all read 0 after its inner level is rebuilt to Map.
    fn rebuild_dim_at(&mut self, idx: usize, ty: BaseFuncType) -> Option<()>
    where
        D: Clone,
    {
        let current = self.dim_prop.get(idx)?.1;
        let need_to_rebuild = current == BaseFuncType::Map
            || ty == BaseFuncType::Map
            || (current != ty && ty != BaseFuncType::Unknown);
        let can_copy = current != BaseFuncType::Map
            && (ty != BaseFuncType::Map || current == BaseFuncType::Constant)
            && idx + 1 != self.dim_prop.len();
        let affine = self.get_data_affine;
        // The dimension list this rebuild would leave behind, built BEFORE anything is written:
        // `if (func_base_type != BaseFuncType::Unknown) dim_prop_.at(pos).second = func_base_type;`
        // (`:1521-1523`) — the sentinel means "rebuild this level as the kind it already is" — and
        // the authority assigns it there and only then throws out of `createLeafFunc`, leaving a
        // level LABELLED with a kind its tree does not have. A refusal here leaves the manager where
        // it was.
        let mut sub = self.dim_prop[idx..].to_vec();
        if ty != BaseFuncType::Unknown {
            sub[0].1 = ty;
        }
        if need_to_rebuild && !Self::buildable(affine, &sub) {
            return None;
        }
        if ty != BaseFuncType::Unknown {
            self.dim_prop[idx].1 = ty;
        }
        if !need_to_rebuild {
            return Some(());
        }
        for node in Self::collect_at_level_mut(&mut self.parent_func, idx) {
            let mut new_node = Self::create_tree(affine, &sub)?;
            if can_copy {
                // `copySubTree(new_child->getChild(), old_child->getChild())` (`:1560`, `:1577`), and
                // for a Map target every new child gets the OLD single child (`:1562-1565`,
                // `:1579-1582`). `can_copy`
                // already guarantees the old node is a Constant or Affine non-leaf, so it has exactly
                // one child to copy from — the level below, which this rebuild did not touch.
                let source = match &*node {
                    FoldFunc::ConstantNonLeaf(c) => c.child(),
                    FoldFunc::AffineNonLeaf(a) => a.child(),
                    _ => return None,
                };
                match &mut new_node {
                    FoldFunc::ConstantNonLeaf(c) => Self::copy_sub_tree(c.child_mut(), source)?,
                    FoldFunc::AffineNonLeaf(a) => Self::copy_sub_tree(a.child_mut(), source)?,
                    FoldFunc::MapNonLeaf(m) => {
                        for child in m.children_mut() {
                            Self::copy_sub_tree(child, source)?;
                        }
                    }
                    _ => return None,
                }
            }
            *node = new_node;
        }
        Some(())
    }

    /// `clone(const FoldManager<Dtype>& rhs, std::map<int64_t, int64_t> ignore_dim_idx_and_dim_val)`
    /// (`:987-1019`) — become a copy of `rhs`, with the named dimensions LEFT OUT and each replaced by
    /// the one coordinate the map names.
    ///
    /// ⛔ THE DATA COPY PAIRS BY LIST POSITION, so dropping a level shifts every node below it against
    /// its counterpart and the per-kind copy silently declines the mismatched pairs — see
    /// [`copy_lists`](Self::copy_lists). Measured: cloning a two-level manager while ignoring its Map
    /// level keeps the affine level's alpha and LOSES the map leaf's data.
    /// ⛔ THE UNIGNORED CLONE IS [`Clone`] ITSELF. `clone(rhs)` with an empty map rebuilds rhs's
    /// dimension list bottom-up and copies every payload into the same position, which is a deep copy
    /// of `rhs` — the authority's copy constructor (`:910`) is exactly that call.
    pub fn clone_ignoring(
        &mut self,
        rhs: &Self,
        ignore_dim_idx_and_dim_val: &BTreeMap<FoldDimPos, FoldDimIndex>,
    ) -> Option<()> {
        self.clear();
        if rhs.dim_prop.is_empty() {
            // `clone`'s zero-dimension case (`:992-997`) — a fresh constant leaf holding rhs's value,
            // is BUILT INLINE THERE (`:995-996`) rather than through `createLeafFunc`'s data-taking
            // overload (`:1890-1909`), which has no caller in the authority at all.
            self.parent_func =
                FoldFunc::ConstantLeaf(ConstFoldFunctionLeaf::new(rhs.get_data(&[])?));
            return Some(());
        }
        let mut ignore = BTreeMap::new();
        for (pos, coord) in ignore_dim_idx_and_dim_val {
            let idx = pos.index(rhs.dim_prop.len())?;
            ignore.insert(idx, usize::try_from(coord.0).ok()?);
        }
        // `for (int idx = rhs.dim_prop_.size() - 1; idx >= 0; idx--) buildDim(..., 0)` (`:1002-1007`) —
        // innermost first, each new level wrapping what is already there.
        for idx in (0..rhs.dim_prop.len()).rev() {
            if ignore.contains_key(&idx) {
                continue;
            }
            let (prop, ty) = &rhs.dim_prop[idx];
            self.build_dim(prop, *ty, FoldDimPos(0))?;
        }
        let rhs_list = rhs.linear_list_ignoring(&ignore);
        Self::copy_lists(&mut self.parent_func, &rhs_list)
    }

    /// `operator=(const FoldManager<Dtype>& rhs)` (`:922-975`).
    ///
    /// ⛔ NOT RUST'S `=`, WHICH IS THE AUTHORITY'S DEFAULTED MOVE ASSIGNMENT (`:912`). This one takes
    /// rhs's fold FUNCTIONS and data but KEEPS THIS MANAGER'S OWN `FoldDimProp`s (`:935-936`) —
    /// measured, a dimension labelled "a" is still labelled "a" after assigning a manager whose
    /// dimension is labelled "b". The labels are this manager's identity; the tree is the value.
    /// ⛔ AND IT REFUSES A MANAGER OF A DIFFERENT SHAPE: a different number of dimensions
    /// (`:923-926`) or a different extent in any of them (`:930-933`) is `DT_ERROR`, measured. It is
    /// an in-place overwrite of a value that already has this shape, not a replacement.
    pub fn assign(&mut self, rhs: &Self) -> Option<()> {
        if self.dim_prop.len() != rhs.dim_prop.len() {
            return None;
        }
        let mut new_dim_prop = Vec::with_capacity(rhs.dim_prop.len());
        for (mine, theirs) in self.dim_prop.iter().zip(rhs.dim_prop.iter()) {
            if mine.0.size() != theirs.0.size() {
                return None;
            }
            new_dim_prop.push((mine.0.clone(), theirs.1));
        }
        if new_dim_prop.is_empty() {
            // `:947-953` — the zero-dimension case assigns the single value and nothing else.
            self.parent_func =
                FoldFunc::ConstantLeaf(ConstFoldFunctionLeaf::new(rhs.get_data(&[])?));
            return Some(());
        }
        let affine = self.get_data_affine;
        if !Self::buildable(affine, &new_dim_prop) {
            return None;
        }
        self.clear();
        // `:957-972`, the same bottom-up rebuild `clone` does, from the merged list.
        for idx in (0..new_dim_prop.len()).rev() {
            let (prop, ty) = &new_dim_prop[idx];
            self.build_dim(prop, *ty, FoldDimPos(0))?;
        }
        let rhs_list = rhs.linear_list_ignoring(&BTreeMap::new());
        Self::copy_lists(&mut self.parent_func, &rhs_list)
    }
}

impl<D: Clone> FoldManager<D> {
    /// `insertAlpha(const Dtype& new_alpha, int pos)` (`:2278-2294`) — one alpha for EVERY node at the
    /// level, which is what makes an Affine level under a Map level still a single fold function.
    ///
    /// ⛔ `DT_ERROR` UNLESS THE LEVEL IS AFFINE (`:2286-2287`), measured — the authority would
    /// otherwise reach the base's `insertAlpha` stub (`:235-236`) node by node.
    pub fn insert_alpha(&mut self, new_alpha: &D, pos: FoldDimPos) -> Option<()> {
        self.each_affine_at(pos, |node| match node {
            FoldFunc::AffineNonLeaf(affine) => affine.insert_alpha(new_alpha.clone()),
            FoldFunc::AffineLeaf(affine) => affine.insert_alpha(new_alpha.clone()),
            _ => {}
        })
    }

    /// `insertBeta(const Dtype& new_beta, int pos)` (`:2302-2318`).
    pub fn insert_beta(&mut self, new_beta: &D, pos: FoldDimPos) -> Option<()> {
        self.each_affine_at(pos, |node| match node {
            FoldFunc::AffineNonLeaf(affine) => affine.insert_beta(new_beta.clone()),
            FoldFunc::AffineLeaf(affine) => affine.insert_beta(new_beta.clone()),
            _ => {}
        })
    }

    /// `insertAlphaBeta(const Dtype& new_alpha, const Dtype& new_beta, int pos)` (`:2394-2413`) — both
    /// in one level walk.
    pub fn insert_alpha_beta(
        &mut self,
        new_alpha: &D,
        new_beta: &D,
        pos: FoldDimPos,
    ) -> Option<()> {
        self.each_affine_at(pos, |node| match node {
            FoldFunc::AffineNonLeaf(affine) => {
                affine.insert_alpha(new_alpha.clone());
                affine.insert_beta(new_beta.clone());
            }
            FoldFunc::AffineLeaf(affine) => {
                affine.insert_alpha(new_alpha.clone());
                affine.insert_beta(new_beta.clone());
            }
            _ => {}
        })
    }

    /// The level walk the three inserters share: resolve `pos`, refuse a level that is not Affine,
    /// then visit every node at it. The `_` arms above are unreachable — `dim_prop_` saying Affine is
    /// what makes every node at the level one of the two affine kinds.
    fn each_affine_at(
        &mut self,
        pos: FoldDimPos,
        mut visit: impl FnMut(&mut FoldFunc<D>),
    ) -> Option<()> {
        let idx = pos.resolve(self.dim_prop.len())?;
        if self.dim_prop[idx].1 != BaseFuncType::Affine {
            return None;
        }
        for node in Self::collect_at_level_mut(&mut self.parent_func, idx) {
            visit(node);
        }
        Some(())
    }

    /// `getAlpha(int pos)` (`:2325-2352`).
    ///
    /// ⛔ IT REFUSES A LEVEL WITH MORE THAN ONE NODE: `DT_CHECK_MSG(ffs_at_pos.size() == 1, ...)`
    /// (`:2339`), so an Affine level anywhere below a Map level has no readable alpha at all,
    /// while [`insert_alpha`](Self::insert_alpha) writes every node of it. Asymmetric in the authority
    /// and kept.
    pub fn alpha(&self, pos: FoldDimPos) -> Option<&D> {
        match self.single_affine_at(pos)? {
            FoldFunc::AffineNonLeaf(affine) => Some(affine.alpha()),
            FoldFunc::AffineLeaf(affine) => Some(affine.alpha()),
            _ => None,
        }
    }

    /// `getBeta(int pos)` (`:2359-2385`), with the same one-node restriction.
    pub fn beta(&self, pos: FoldDimPos) -> Option<&D> {
        match self.single_affine_at(pos)? {
            FoldFunc::AffineNonLeaf(affine) => Some(affine.beta()),
            FoldFunc::AffineLeaf(affine) => Some(affine.beta()),
            _ => None,
        }
    }

    /// `getAlphaBeta(int pos)` (`:2422-2450`) — both, in one walk.
    ///
    /// ⛔ THIS ONE DOES NOT RESTRICT THE LEVEL TO ONE NODE; it reads `ffs_at_pos.front()`
    /// (`:2437-2438`), so under a Map level it answers the first child's pair where `getAlpha`
    /// refuses. Kept: the two are separate members with separate contracts.
    pub fn alpha_beta(&self, pos: FoldDimPos) -> Option<(&D, &D)> {
        let idx = pos.resolve(self.dim_prop.len())?;
        if self.dim_prop[idx].1 != BaseFuncType::Affine {
            return None;
        }
        match self.collect_at_level(idx)?.first()? {
            FoldFunc::AffineNonLeaf(affine) => Some((affine.alpha(), affine.beta())),
            FoldFunc::AffineLeaf(affine) => Some((affine.alpha(), affine.beta())),
            _ => None,
        }
    }

    /// The reader half of [`each_affine_at`](Self::each_affine_at), with `getAlpha`'s extra
    /// restriction.
    fn single_affine_at(&self, pos: FoldDimPos) -> Option<&FoldFunc<D>> {
        let idx = pos.resolve(self.dim_prop.len())?;
        if self.dim_prop[idx].1 != BaseFuncType::Affine {
            return None;
        }
        let ffs_at_pos = self.collect_at_level(idx)?;
        if ffs_at_pos.len() != 1 {
            return None;
        }
        ffs_at_pos.first().copied()
    }
}

// ⛔ WHAT THIS PORT LEAVES OUT OF `FoldManager`, with the count of call sites each has across
// `dsc/`, `ddc/`, `dcg/dcg_fe/scheduler/` and `dsc-based-utils/` — the campaign's whole scope:
//   `buildWkSplitDim` `:1332`, `insertWkSplitParam` `:2554` and `:2655`, `getWkSplitParam` `:2667`
//     and `:2679`                                                                  0 callers each
//       ⛔ THIS IS A DIVERGENCE, NOT AN OMISSION, AND IT IS THE ONE THING IN THIS UNIT THE PORT
//       REFUSES WHERE THE AUTHORITY SUCCEEDS. `buildWkSplitDim` builds a real level — measured, one
//       dimension whose `getFuncType` is 3 — and this tree has no variant to hold it, for the reason
//       [`FoldFunc`] states: a `WkSplitFoldFunction_Leaf`'s value is COMPUTED, and which of its two
//       readings applies is a choice that belongs here. It cannot be made yet. The conversion is
//       `getFoldedData`, and it is THREE SFINAE overloads whose two value-producing arms read the
//       level's `WkSplitParam` differently: `wksplit_param_.getSize(dim_index)` at an arithmetic
//       payload (`:843`) against `getCoordVec(dim_index)` at a `vector<pair<int64_t, int64_t>>` one
//       (`:837`), the third arm being `DT_ERROR` for any other payload (`:824-827`). Neither reading
//       has a caller to pick it: measured, a freshly built WkSplit level throws on `getData` — its
//       `DT_CHECK(wksplit_param_.isBuilt())` (`:849`) — on `insertData`, and on `insertWkSplitParam`
//       of a default-constructed param, and answers only after a caller inserts a BUILT one. No such
//       caller exists anywhere in scope, and no `foldTypes` vector in scope contains `WkSplit`, so
//       there is no call site whose reading would settle it. Building the level without the conversion
//       would be a manager whose every reader refuses — which is the placeholder RULE 5 forbids. So
//       [`build_dim`](FoldManager::build_dim) refuses the kind, and the first real WkSplit caller is
//       what makes it buildable.
//   `copyFoldedSubSpace` `:1180`                                                   0
//       Copies a sub-space between managers, `DT_ERROR`ing when the dimensions skipped are Affine or
//       WkSplit (`:1193-1197`). Its callers are `perfdsc`'s, outside the campaign.
//   `compressMapToConst` `:2831`, `compressMapToConstForAllDims` `:2891`           0
//       Both would need `rebuildDim`'s three-argument form and the same pointer identity; neither has
//       a caller to fix the reading of `areAllConstant`'s tolerance.
//   `importFromJson` `:2760`                                                       0 (on this class)
//       This crate has no JSON reader — the same reason [`FoldDimProp`] carries none.
//   `printDimPropPtr` `:2640`                                                      0
//       Prints the addresses of the borrowed `FoldDimProp*`s, which this port does not have.
//   `getFoldCoordinatesAndDataMap` `:2155`, `:2179`                                0
//       `getDataAndFoldCoordinates` keyed by coordinate instead of ordered by it.
//   `getFoldFunc(const std::deque<int64_t>&)` `:2911`                              private
//       The authority's own private accessor, whose only readers are the five WkSplit members above.
//       [`FoldFunc::fold_func`] is the same walk, on the node.
//   `getLinearFuncList` `:1793`, `:1842`, `collectFoldFunctionAtLevel` `:1726`,
//     `createTree` `:2227`, `createSubTreeForEachMapChild` `:2250`, `createLeafFunc` `:1870`,
//     `createNonLeafFunc` `:1911`, `copySubTree` `:2470`, `copy` `:1046`, `clear` `:2917`,
//     `getNumUniqueCoordsInEachFold` `:1947`, `getAllCoordsInEachFold` `:1963`
//       Ported, but private: every one of them is a step of a member above, and the authority's own
//       call sites for all of them are inside this class.
//   `FoldInfraUtils::getFlattenedCoordinatesWithConstraints` `:64`                 0
//       A free function over a coordinate list, not a member of this class, and no caller in scope.
//
// And what is NOT left out, named because a reader counting declarations would otherwise miss it:
//   `printData`'s four SFINAE arms (`:2684-2689`, `:2691-2698`, `:2700-2707`, `:2709-2718`) are ONE
//     method here, [`FoldManager::print_data`]. The arithmetic arm is the body and the `DT_ERROR` arm
//     is its [`fmt::Display`] bound, so the two list arms are UNREACHABLE rather than unported — see
//     that method's own note.
//   `~FoldManager()` (`:914`) and `deleteSubTree` (`:2529-2539`) are [`Drop`]: the tree is owned
//     [`Box`]es, so there is no list to walk and nothing to `delete`.
//   `FoldManager(FoldManager&&)` (`:912`) is Rust's move, which needs no declaration.
//   `operator!=` (`:1090-1092`) is [`PartialEq`]'s provided `ne`.
//   The variadic `isLegal` (`:1660-1665`), `getData` (`:1684-1689`) and `insertData` (`:1706-1710`)
//     all end at the deque form. `isLegal` builds the deque itself (`:1663`); the other two forward to
//     `FoldFunction`'s own variadic overloads, which build it (`:196`, `:226`) and call the same
//     virtual `getData(deque, 0)` / `insertData(new_data, deque, 0)`. A slice is that deque, so the
//     deque form is the only form here.

#[cfg(test)]
mod unit_tests {
    use super::*;

    /// The vendor's two eight-core work splits, transcribed from its own `build` arguments
    /// (`util/foldManager/test/test_fold_infrastructure.cpp:161-165`): `ij` shares each of two
    /// ten-element slices across four cores, `out` gives each of four four-element slices one core
    /// and lays the set down twice. ⛔ THE THIRTEEN ARGUMENTS ARE THE VENDOR'S ORDER — see
    /// [`WkSplitParam::new`] for why each has its own type.
    fn vendor_split(
        wk_ss: i32,
        num_ss_slices: i32,
        repeat_inner: i32,
        outer_repeat: i32,
    ) -> WkSplitParam {
        use crate::schedule::wk_division::{
            GapAfterAllSlices, GapAfterInnerRepeat, GapWithinInnerRepeat, MaxCores,
            NumEpilogueSlices, NumSsSlices, OuterRepeatFactor, RepeatFactorInner, WkEpilogue, WkSs,
        };
        WkSplitParam::new(
            WkSs(wk_ss),
            WkEpilogue(0),
            MaxCores(8),
            Cid(0),
            NumSsSlices(num_ss_slices),
            NumEpilogueSlices(0),
            GapWithinInnerRepeat::NONE,
            RepeatFactorInner::new(repeat_inner).unwrap(),
            GapAfterInnerRepeat::NONE,
            GapAfterAllSlices(0),
            OuterRepeatFactor::new(outer_repeat).unwrap(),
            Vec::new(),
            None,
        )
        .unwrap()
    }

    /// ⭐ THE VENDOR READS ONE WORK SPLIT IN TWO VOCABULARIES: `constructor_test_wksplit` builds a
    /// `FoldManager<uint64_t>` and a `FoldManager<vector<pair<int64_t, int64_t>>>` over the SAME
    /// `WkSplitParam` and expects a size from one and coordinate ranges from the other
    /// (`util/foldManager/test/test_fold_infrastructure.cpp:176-189`). Here that is one leaf and two
    /// readers, so the goldens below are the vendor's for both of its instantiations at once.
    #[test]
    fn one_wksplit_leaf_answers_in_both_of_the_vendors_two_vocabularies() {
        use crate::schedule::wk_division::Coord;

        let ij = WkSplitFoldFunctionLeaf::from(vendor_split(10, 2, 4, 1));
        let out = WkSplitFoldFunctionLeaf::new(vendor_split(4, 4, 1, 2));

        assert_eq!(
            WkSplitFoldFunctionLeaf::FUNCTION.ty(),
            FuncType::WkSplitLeaf,
            "the base subobject's tag, `:798` and `:808`"
        );

        for i in 0..8 {
            let core = FoldDimIndex(i);
            assert_eq!(ij.folded_size(core), Some(WkSize(10)), "`getSize` at {i}");
            assert_eq!(out.folded_size(core), Some(WkSize(4)));

            // `g_val.push_back({10 * (i / 4), 9 + 10 * (i / 4)})` (`:181`) and `{4 * (i % 4),
            // 3 + 4 * (i % 4)}` (`:187`) — ONE range per core, not an empty vec.
            assert_eq!(
                ij.folded_coord_vec(core),
                Some(vec![CoordRange {
                    start: Coord(10 * (i / 4)),
                    end: Coord(9 + 10 * (i / 4)),
                }]),
                "`getCoordVec` at {i}"
            );
            assert_eq!(
                out.folded_coord_vec(core),
                Some(vec![CoordRange {
                    start: Coord(4 * (i % 4)),
                    end: Coord(3 + 4 * (i % 4)),
                }])
            );
        }
    }

    /// The negative: ⛔ an unbuilt leaf ANSWERS NOTHING BUT STILL PRINTS — `getData`'s
    /// `DT_CHECK(wksplit_param_.isBuilt())` (`:849`) aborts, while `printMetaData` (`:880-882`)
    /// delegates with no check at all. Both readers are [`None`] and the text is the declared
    /// initialisers, which is the whole of what this state is observable as.
    #[test]
    fn an_unbuilt_wksplit_leaf_answers_nothing_and_still_prints_its_metadata() {
        let mut leaf = WkSplitFoldFunctionLeaf::unbuilt();
        assert_eq!(leaf.wk_split_param(), None);
        assert_eq!(leaf.folded_size(FoldDimIndex(0)), None);
        assert_eq!(leaf.folded_coord_vec(FoldDimIndex(0)), None);
        assert_eq!(
            leaf.print_meta_data("\n  "),
            WkSplitParam::unbuilt_meta_data("\n  "),
            "`printMetaData` has no `isBuilt` guard to stop it"
        );

        // `insertWkSplitParam` (`:859-861`) is what builds it, and the state flips whole.
        let ij = vendor_split(10, 2, 4, 1);
        leaf.insert_wk_split_param(&ij);
        assert_eq!(leaf.wk_split_param(), Some(&ij));
        assert_eq!(leaf.folded_size(FoldDimIndex(0)), Some(WkSize(10)));
        assert_eq!(leaf.print_meta_data("\n  "), ij.print_meta_data("\n  "));

        // And on an ALREADY built leaf it overwrites rather than merges: the answer becomes the new
        // param's alone. `wkDivisionParams.h:109-125` copies all thirteen fields.
        let out = vendor_split(4, 4, 1, 2);
        leaf.insert_wk_split_param(&out);
        assert_eq!(leaf.folded_size(FoldDimIndex(0)), Some(WkSize(4)));
        assert_eq!(leaf.wk_split_param(), Some(&out));

        // `getWkSplitParamMutable` (`:858`): the merge sites reach the source through it (`:1071`,
        // `:2515`), and an unbuilt leaf has nothing for them to reach.
        assert!(leaf.wk_split_param_mut().is_some());
        assert!(
            WkSplitFoldFunctionLeaf::unbuilt()
                .wk_split_param_mut()
                .is_none()
        );
    }

    /// The vendor's SIXTEEN-core split, transcribed from its own `build` arguments
    /// (`util/foldManager/test/test_fold_infrastructure.cpp:223-226`): three unused cores after each
    /// set of slices and two more at the end, which is where a gap core comes from.
    fn vendor_split_16(
        wk_ss: i32,
        wk_epilogue: i32,
        num_ss_slices: i32,
        num_epilogue_slices: i32,
        repeat_inner: i32,
        outer_repeat: i32,
    ) -> WkSplitParam {
        use crate::schedule::wk_division::{
            GapAfterAllSlices, GapAfterInnerRepeat, GapWithinInnerRepeat, MaxCores,
            NumEpilogueSlices, NumSsSlices, OuterRepeatFactor, RepeatFactorInner, WkEpilogue, WkSs,
        };
        WkSplitParam::new(
            WkSs(wk_ss),
            WkEpilogue(wk_epilogue),
            MaxCores(16),
            Cid(0),
            NumSsSlices(num_ss_slices),
            NumEpilogueSlices(num_epilogue_slices),
            GapWithinInnerRepeat::NONE,
            RepeatFactorInner::new(repeat_inner).unwrap(),
            GapAfterInnerRepeat::NONE,
            GapAfterAllSlices(3),
            OuterRepeatFactor::new(outer_repeat).unwrap(),
            Vec::new(),
            None,
        )
        .unwrap()
    }

    /// ⭐ THE VENDOR'S SECOND GOLDEN IS THE ONE WITH GAPS IN IT, and nothing reached it until now:
    /// `constructor_test_wksplit2` asserts a work split over sixteen cores of which only seven do
    /// work per outer repeat, so ⛔ A GAP CORE'S SIZE IS `Some(WkSize(0))` AND ITS RANGE IS ONE
    /// [`CoordRange::GAP`] — the two claims [`WkSplitFoldFunctionLeaf::folded_size`] and
    /// [`WkSplitFoldFunctionLeaf::folded_coord_vec`] make and neither could show. Nine of the
    /// sixteen are gaps in `ij` and the same nine in `out`
    /// (`util/foldManager/test/test_fold_infrastructure.cpp:239-253`).
    ///
    /// It is also the only fixture with an EPILOGUE slice: `out`'s fourth slice is two elements
    /// where its first three are three, which is `getSize`'s `slice_id < num_ss_slices_` branch
    /// (`util/foldManager/wkDivisionParams.h:241`).
    #[test]
    fn the_vendors_sixteen_core_split_gives_nine_gap_cores_a_size_and_a_range() {
        use crate::schedule::wk_division::Coord;

        let range = |start, end| {
            Some(vec![CoordRange {
                start: Coord(start),
                end: Coord(end),
            }])
        };

        let ij = WkSplitFoldFunctionLeaf::from(vendor_split_16(10, 0, 2, 0, 2, 2));
        let out = WkSplitFoldFunctionLeaf::from(vendor_split_16(3, 2, 3, 1, 1, 2));

        // `golden_vals_ij` and `golden_vals_out` (`:239-243`) — the zeros are the gap cores.
        let golden_size_ij = [10, 10, 10, 10, 0, 0, 0, 10, 10, 10, 10, 0, 0, 0, 0, 0];
        let golden_size_out = [3, 3, 3, 2, 0, 0, 0, 3, 3, 3, 2, 0, 0, 0, 0, 0];

        // `golden_vals_coord_ij` and `golden_vals_coord_out` (`:245-253`), each pushed as the one
        // entry of the vector the core gets back (`util/foldManager/wkDivisionParams.h:253-254`).
        let golden_coord_ij = [
            range(0, 9),
            range(0, 9),
            range(10, 19),
            range(10, 19),
            None,
            None,
            None,
            range(0, 9),
            range(0, 9),
            range(10, 19),
            range(10, 19),
            None,
            None,
            None,
            None,
            None,
        ];
        let golden_coord_out = [
            range(0, 2),
            range(3, 5),
            range(6, 8),
            range(9, 10),
            None,
            None,
            None,
            range(0, 2),
            range(3, 5),
            range(6, 8),
            range(9, 10),
            None,
            None,
            None,
            None,
            None,
        ];

        for cid in 0..16_i64 {
            let core = FoldDimIndex(cid);
            let i = cid as usize;

            assert_eq!(
                ij.folded_size(core),
                Some(WkSize(golden_size_ij[i])),
                "`golden_vals_ij` at {cid}"
            );
            assert_eq!(
                out.folded_size(core),
                Some(WkSize(golden_size_out[i])),
                "`golden_vals_out` at {cid}"
            );

            // A gap core answers ONE `{-1, -1}`, never an empty vector and never `None`.
            let want_ij = golden_coord_ij[i]
                .clone()
                .unwrap_or_else(|| vec![CoordRange::GAP]);
            let want_out = golden_coord_out[i]
                .clone()
                .unwrap_or_else(|| vec![CoordRange::GAP]);
            assert_eq!(
                ij.folded_coord_vec(core),
                Some(want_ij),
                "`golden_vals_coord_ij` at {cid}"
            );
            assert_eq!(
                out.folded_coord_vec(core),
                Some(want_out),
                "`golden_vals_coord_out` at {cid}"
            );
        }
    }

    /// The negative for the one narrowing on this seam: ⛔ `getFoldedData` HANDS AN `int64_t` TO AN
    /// `int32_t cid` (`foldInfrastructure.h:837`/`:843` into
    /// `util/foldManager/wkDivisionParams.h:235`/`:250`), so a coordinate past `INT32_MAX` WRAPS
    /// ONTO A WORKING CORE. Measured through the leaf itself, compiled against the fixture above:
    /// `getData({1 << 32})` answers `10` over coordinates `(0, 9)` — core 0's — and
    /// `getData({(1 << 32) + 3})` answers `10` over `(10, 19)`, core 3's.
    ///
    /// [`Cid`] keeps the caller's width, so neither coordinate is in the gang here and both get a
    /// gap core's answer. The divergence is deliberate: handing back another core's elements for a
    /// coordinate that has no core is worse than reporting no work.
    #[test]
    fn a_fold_coordinate_past_int32_does_not_wrap_onto_another_core() {
        use crate::schedule::wk_division::Coord;

        let ij = WkSplitFoldFunctionLeaf::from(vendor_split_16(10, 0, 2, 0, 2, 2));

        // The two cores `1 << 32` and `(1 << 32) + 3` truncate onto, and what they really hold.
        assert_eq!(ij.folded_size(FoldDimIndex(0)), Some(WkSize(10)));
        assert_eq!(
            ij.folded_coord_vec(FoldDimIndex(0)),
            Some(vec![CoordRange {
                start: Coord(0),
                end: Coord(9),
            }])
        );
        assert_eq!(
            ij.folded_coord_vec(FoldDimIndex(3)),
            Some(vec![CoordRange {
                start: Coord(10),
                end: Coord(19),
            }])
        );

        // ⭐ `i64::MAX` IS NOT ONE OF THE DIVERGING CASES — it truncates to `-1`, which `adjustCID`
        // (`util/foldManager/wkDivisionParams.h:145-147`) puts outside the gang as well, so the
        // authority answers a gap core there too. It is here to show where the two ends do agree.
        for past in [1_i64 << 32, (1 << 32) + 3, i64::MAX] {
            assert_eq!(
                ij.folded_size(FoldDimIndex(past)),
                Some(WkSize(0)),
                "no core {past}"
            );
            assert_eq!(
                ij.folded_coord_vec(FoldDimIndex(past)),
                Some(vec![CoordRange::GAP])
            );
        }
    }

    /// A map leaf IS its `data_vec_` (`:782`), one value per coordinate, and ⛔ IBM'S BOUNDS CHECK
    /// `data_vec_.size() > dim_index` (`:759`) IS A SIGNED/UNSIGNED COMPARISON — a negative index is
    /// converted to a huge unsigned and fails the check, so it never reaches `.at()`'s throw. Both
    /// ends refuse here for that one reason.
    #[test]
    fn a_map_leaf_is_its_data_vec_and_refuses_an_index_outside_it() {
        assert_eq!(
            MapFoldFunctionLeaf::<u64>::FUNCTION.ty(),
            FuncType::MapLeaf,
            "the base subobject's tag, `:743`/`:747`/`:751`"
        );

        // `std::vector<Dtype>(dim_size)` VALUE initialises (`:744`), reached from `createLeafFunc`
        // with no data yet (`:1877`).
        let mut leaf: MapFoldFunctionLeaf<u64> =
            MapFoldFunctionLeaf::with_default_data(FoldDimSize(4));
        assert_eq!(leaf.size(), FoldDimSize(4));
        assert_eq!(leaf.data_vec(), &[0, 0, 0, 0]);

        // `insertData` (`:769`) writes one coordinate and leaves the extent alone.
        *leaf.data_mut(FoldDimIndex(1)).unwrap() = 41;
        *leaf.data_mut(FoldDimIndex(3)).unwrap() = 43;
        assert_eq!(leaf.data(FoldDimIndex(1)), Some(&41));
        assert_eq!(leaf.data_vec(), &[0, 41, 0, 43]);
        assert_eq!(leaf.size(), FoldDimSize(4));

        // Both refusals, and they are the same one.
        assert_eq!(leaf.data(FoldDimIndex(4)), None, "one past the end");
        assert_eq!(leaf.data(FoldDimIndex(-1)), None, "and a negative index");
        assert!(leaf.data_mut(FoldDimIndex(-1)).is_none());

        // `MapFoldFunction_Leaf(dim_size, new_data)` (`:748`), `createLeafFunc`'s data-carrying
        // overload (`:1898`).
        let filled = MapFoldFunctionLeaf::filled(FoldDimSize(4), &7_u64);
        assert_eq!(filled.data_vec(), &[7, 7, 7, 7]);
        assert_eq!(
            MapFoldFunctionLeaf::new(vec![7_u64; 4]),
            Some(filled.clone())
        );

        // `this_leaf->getDataVec() = rhs_leaf->getDataVec()` (`:1085`, `:2519`) takes the whole run,
        // and `!=` on the two vecs is what `:1164` compares.
        assert_ne!(leaf, filled);
        leaf.copy_data_vec_from(&filled);
        assert_eq!(leaf, filled);
        assert_eq!(leaf.data_vec(), filled.data_vec());

        // An extent of one is an unfolded dim, and it still indexes.
        let single = MapFoldFunctionLeaf::filled(FoldDimSize(1), &"only".to_owned());
        assert_eq!(single.size(), FoldDimSize(1));
        assert_eq!(single.data(FoldDimIndex(0)).unwrap(), "only");
        assert_eq!(single.data(FoldDimIndex(1)), None);
    }

    /// Every kind's variant, its `type_` tag and the two predicates over it are ONE fact
    /// (`foldInfrastructure.h:166-190`): the C++ pairing of a `Type()` test with a `static_cast`
    /// (`dsc/dsc2.h:375-390`) can disagree, and a `match` on [`FoldFunc`] cannot.
    ///
    /// ⛔ `WkSplitLeaf` HAS NO VARIANT TO CHECK HERE and `Unknown` has no producer at all (`:180`) —
    /// see [`FoldFunc`] for why the work-split leaf cannot be one of its variants.
    #[test]
    fn every_fold_func_variant_agrees_with_the_tag_and_the_predicates_it_carries() {
        let const_leaf = ConstFoldFunctionLeaf::new(1i64);
        let variants = [
            (
                FoldFunc::ConstantLeaf(const_leaf),
                FuncType::ConstantLeaf,
                true,
            ),
            (
                FoldFunc::AffineLeaf(AffineFoldFunctionLeaf::<i64>::new(1i64, 2)),
                FuncType::AffineLeaf,
                true,
            ),
            (
                FoldFunc::MapLeaf(MapFoldFunctionLeaf::filled(FoldDimSize(2), &1i64)),
                FuncType::MapLeaf,
                true,
            ),
            (
                FoldFunc::ConstantNonLeaf(ConstFoldFunctionNonLeaf::new(FoldFunc::ConstantLeaf(
                    const_leaf,
                ))),
                FuncType::ConstantNonLeaf,
                false,
            ),
            (
                FoldFunc::AffineNonLeaf(AffineFoldFunctionNonLeaf::<i64>::new(
                    1i64,
                    2,
                    FoldFunc::ConstantLeaf(const_leaf),
                )),
                FuncType::AffineNonLeaf,
                false,
            ),
            (
                FoldFunc::MapNonLeaf(MapFoldFunctionNonLeaf::new(vec![FoldFunc::ConstantLeaf(
                    const_leaf,
                )])),
                FuncType::MapNonLeaf,
                false,
            ),
        ];
        for (ff, ty, is_leaf) in variants {
            assert_eq!(ff.ty(), ty, "the variant IS the tag");
            assert_eq!(
                ff.base(),
                FoldFunction::new(ty),
                "and so is the base subobject"
            );
            assert_eq!(ff.base().is_leaf(), is_leaf, "isLeaf({ty:?})");
            assert_eq!(ff.base().is_non_leaf(), !is_leaf, "isNonLeaf({ty:?})");
        }
    }

    /// An affine level's state is `alpha_`, `beta_` and its subtree (`foldInfrastructure.h:517-519`,
    /// `:676-677`).
    ///
    /// ⛔ THE RESOLVED [`GetDataAffine`] OVERLOAD IS NOT STATE: Rust does not guarantee one address
    /// per `fn` item across codegen units, so if equality or [`Debug`](fmt::Debug) were derived, two
    /// levels the authority cannot tell apart could compare unequal from one build to the next.
    #[test]
    fn an_affine_level_is_its_alpha_beta_and_subtree_and_never_the_overload_it_resolved() {
        let leaf = AffineFoldFunctionLeaf::<i64>::new(3i64, 10);
        assert_eq!(leaf, AffineFoldFunctionLeaf::<i64>::new(3i64, 10));
        assert_eq!(leaf, leaf.clone());
        assert_ne!(
            leaf,
            AffineFoldFunctionLeaf::<i64>::new(10i64, 3),
            "alpha_ and beta_ do not commute"
        );
        assert_eq!(
            AffineFoldFunctionLeaf::<i64>::default(),
            AffineFoldFunctionLeaf::<i64>::new(0i64, 0)
        );

        let level =
            AffineFoldFunctionNonLeaf::<i64>::new(1i64, 0, FoldFunc::AffineLeaf(leaf.clone()));
        assert_eq!(
            level,
            AffineFoldFunctionNonLeaf::<i64>::new(1i64, 0, FoldFunc::AffineLeaf(leaf))
        );
        assert_ne!(
            level,
            AffineFoldFunctionNonLeaf::<i64>::new(
                1i64,
                0,
                FoldFunc::AffineLeaf(AffineFoldFunctionLeaf::<i64>::new(3i64, 11))
            ),
            "the subtree is part of the fold function"
        );

        // The overload is not printed either, and `..` is where it would have been.
        let shown = format!("{level:?}");
        assert!(shown.contains("alpha: 1"), "{shown}");
        assert!(shown.contains("beta: 0"), "{shown}");
        assert!(shown.contains(".."), "{shown}");
    }

    /// The one fold-function kind a [`FoldManager`] here will not build, and the measurements that
    /// say declining it costs no reader anything.
    ///
    /// ⛔ THIS IS A DIVERGENCE, NOT A REPRODUCTION. `buildWkSplitDim` SUCCEEDS in the authority —
    /// measured `w.build = OK`, `w.dims = 1`, `w.func_type = 3` (`BaseFuncType::WkSplit`), and a
    /// Constant level nests over it: `w.build_outer_const = OK`, `w.two_dims = 2`. `buildDim(&a,
    /// WkSplit, 0)` does the same: `c11.wksplit = 1`.
    /// ⛔ AND EVERY READER OF THE LEVEL IT BUILT THROWS. `w.get_data_no_param = THROW` is
    /// `DT_CHECK(wksplit_param_.isBuilt())` (`:849`), `w.all_data = THROW` is that same check reached
    /// through the walk, `w.insert_data = THROW` is `DT_ERROR("Illegal use of insertData on
    /// WkSplitFoldFunction_Leaf")` (`:865-867`), and `w.insert_param = THROW` is
    /// `WkSplitParam::build`'s own `checkLegality` rejecting a default-constructed source
    /// (`wkDivisionParams.h:109-125`).
    /// ⛔ THE ONLY WAY TO A VALUE IS THROUGH THAT LAST THROW. `build` sets `isBuilt_ = true`
    /// (`wkDivisionParams.h:123`) BEFORE it calls `checkLegality` (`:124`), so the throw leaves the
    /// leaf claiming to be built over thirteen zeroed parameters — which is why
    /// `w.get_data_empty_param = 0` and `w.two_get_data = 0` answer at all, having thrown one line
    /// earlier. That half-built state is UNREPRESENTABLE here:
    /// [`WkSplitFoldFunctionLeaf::insert_wk_split_param`] takes a [`WkSplitParam`] that is already a
    /// legal whole, because [`WkSplitParam::new`] is the only way to make one.
    ///
    /// So the capability declined is a level whose value no in-scope caller can read without going
    /// through a throw, and no `foldTypes` vector in scope contains `WkSplit` — see the omission
    /// block above this module for the census. [RULE 5]: the two readings of `getFoldedData`
    /// (`:824-845`) cannot be chosen without a caller to choose them, and there is none.
    #[test]
    fn a_wk_split_level_is_the_one_kind_this_fold_manager_refuses_to_build() {
        let a = FoldDimProp::new(FoldDimSize(2), "a");
        let b = FoldDimProp::new(FoldDimSize(3), "b");

        let mut fm = FoldManager::<i64>::with_data(9);
        assert_eq!(
            fm.build_dim(&a, BaseFuncType::WkSplit, FoldDimPos(0)),
            None,
            "`w.build = OK` there; refused here"
        );
        assert_eq!(fm.num_dims(), 0, "and the refusal came before any mutation");
        assert_eq!(fm.get_data(&[]), Some(9), "the value it already held");

        assert_eq!(
            fm.build_fold_space(
                &[a.clone(), b.clone()],
                &[BaseFuncType::Map, BaseFuncType::WkSplit]
            ),
            None,
            "one refused level refuses the whole fold space"
        );
        assert_eq!(fm.num_dims(), 0);
        assert_eq!(fm.func_types(), Vec::new());

        // A rebuild INTO the kind is the same refusal, and it too leaves the level labelled as it
        // was — where the authority assigns the label first (`:1521-1523`) and throws after.
        assert_eq!(fm.build_const_dim(&a, FoldDimPos(0)), Some(()));
        assert_eq!(fm.build_map_dim(&b, FoldDimPos(0)), Some(()));
        assert_eq!(fm.rebuild_dim(FoldDimPos(1), BaseFuncType::WkSplit), None);
        assert_eq!(
            fm.func_types(),
            vec![BaseFuncType::Map, BaseFuncType::Constant],
            "the refused rebuild did not relabel the level"
        );
        assert_eq!(
            fm.get_data(&[FoldDimIndex(2), FoldDimIndex(0)]),
            Some(9),
            "and the 9 the constant leaf held before either build is still there (`:1341-1343`)"
        );

        // And the leaf type itself is fully ported (e019) — it is only the MANAGER that will not put
        // one in a tree, so nothing above is a gap in the work split.
        assert_eq!(
            WkSplitFoldFunctionLeaf::FUNCTION.ty(),
            FuncType::WkSplitLeaf
        );
        assert_eq!(WkSplitFoldFunctionLeaf::unbuilt().wk_split_param(), None);
    }

    /// [`FoldDimPos`]'s two resolutions, side by side on the same list — the reason it carries two
    /// instead of one.
    ///
    /// ⛔ A NEGATIVE POSITION IS THE INNERMOST LEVEL TO FOUR MEMBERS AND A THROW TO THREE.
    /// `insertAlpha`, `getAlpha`, `rebuildDim` and `insertWkSplitParam` open with `if (pos < 0) pos =
    /// dim_prop_.size() + pos;` (`:2279-2280`, `:2326-2327`, `:1474-1475`, `:2555-2557`);
    /// `getFuncType`, `getFoldDimProp` and `getFoldDimSize` compare against an unsigned `size()` or
    /// call `.at(pos)` (`:2579-2581`, `:2624`, `:2631-2633`), where `-1` becomes a huge unsigned value
    /// — measured, `o.getFuncType_neg = THROW` on a one-dimension manager.
    /// ⛔ SO THE TWO ARE NOT INTERCHANGEABLE AT ANY LIST LENGTH BUT ONE: they agree only where the
    /// position is non-negative and in range, which is exactly the case no caller had to think about.
    #[test]
    fn the_two_fold_dim_pos_resolutions_are_not_interchangeable() {
        // Where they agree: a non-negative position inside the list.
        for pos in 0..3 {
            assert_eq!(FoldDimPos(pos).resolve(3), Some(pos as usize));
            assert_eq!(FoldDimPos(pos).index(3), Some(pos as usize));
        }
        // Where they do not: from the end.
        assert_eq!(FoldDimPos(-1).resolve(3), Some(2), "the innermost level");
        assert_eq!(FoldDimPos(-1).index(3), None, "`.at((size_t)-1)`");
        assert_eq!(FoldDimPos(-3).resolve(3), Some(0), "the outermost");
        assert_eq!(FoldDimPos(-4).resolve(3), None, "past the outermost");
        // And past the end, where they agree again by two different routes.
        assert_eq!(FoldDimPos(3).resolve(3), None);
        assert_eq!(FoldDimPos(3).index(3), None);
        // At zero dimensions neither answers, which is what makes `rebuild_dim`'s own
        // zero-dimension test — not this one — the place the authority's underflow is told from a
        // genuine out-of-range refusal.
        assert_eq!(FoldDimPos(0).resolve(0), None);
        assert_eq!(FoldDimPos(0).index(0), None);
        assert_eq!(FoldDimPos(-1).resolve(0), None);

        // The split as the public surface shows it, on one three-level manager.
        let mut fm = FoldManager::<i64>::new();
        assert_eq!(
            fm.build_fold_space(
                &[
                    FoldDimProp::new(FoldDimSize(2), "o"),
                    FoldDimProp::new(FoldDimSize(3), "m"),
                    FoldDimProp::new(FoldDimSize(4), "i"),
                ],
                &[
                    BaseFuncType::Constant,
                    BaseFuncType::Constant,
                    BaseFuncType::Affine,
                ]
            ),
            Some(())
        );
        assert_eq!(
            fm.func_type(FoldDimPos(-1)),
            None,
            "`o.getFuncType_neg = THROW`"
        );
        assert_eq!(fm.fold_dim_prop(FoldDimPos(-1)), None);
        assert_eq!(fm.fold_dim_size(FoldDimPos(-1)), None);
        assert_eq!(fm.func_type(FoldDimPos(2)), Some(BaseFuncType::Affine));

        // `getAlpha` restricts the level to ONE node (`:2339`), which is why neither level above the
        // affine one here is a Map: a Map above an affine level is `m.alpha_under_map`'s case.

        assert_eq!(
            fm.insert_alpha_beta(&5, &1, FoldDimPos(-1)),
            Some(()),
            "the same level, reached from the end"
        );
        assert_eq!(fm.alpha(FoldDimPos(-1)), Some(&5));
        assert_eq!(fm.alpha(FoldDimPos(2)), Some(&5));
        assert_eq!(
            fm.rebuild_dim(FoldDimPos(-1), BaseFuncType::Constant),
            Some(true)
        );
        assert_eq!(fm.func_type(FoldDimPos(2)), Some(BaseFuncType::Constant));
    }

    /// Where a refused build leaves the manager — the one place this port will not follow the
    /// authority, because the state the authority lands in has no readers at all.
    ///
    /// ⛔ `clear()` SETS `parent_func_ = nullptr` (`:2917-2921`) AND EVERY BUILDER CALLS IT FIRST:
    /// `buildFoldSpace` (`:1653-1657`), `buildAllConstantFoldSpace` (`:1599-1609`) and
    /// `buildAllMapFoldSpace` (`:1617-1626`) all clear, fill `dim_prop_`, and only then call
    /// `createTree` — whose empty-list arm is `DT_ERROR("Need at least one fold dim to build a tree")`
    /// (`:2228-2229`). Measured: `c12.build_empty_props = THROW`, `c9.build_mismatched_sizes = THROW`.
    /// So an empty props list leaves a manager with a NULL tree and an empty dimension list, and its
    /// next reader dereferences that null.
    /// ⛔ HERE THE REFUSAL COMES BEFORE THE CLEAR, so a refused build is a no-op and the manager still
    /// holds what it held. There is no null to represent: `parent_func` is a [`FoldFunc`], not a
    /// pointer, and `clear` installs the default constant leaf the constructor would.
    #[test]
    fn a_refused_build_leaves_the_manager_intact_where_the_authority_leaves_it_cleared() {
        let a = FoldDimProp::new(FoldDimSize(2), "a");
        let b = FoldDimProp::new(FoldDimSize(3), "b");

        let mut fm = FoldManager::<i64>::new();
        assert_eq!(
            fm.build_fold_space(&[a.clone(), b.clone()], &[BaseFuncType::Map; 2]),
            Some(())
        );
        assert_eq!(
            fm.insert_data(41, &[FoldDimIndex(1), FoldDimIndex(2)]),
            Some(())
        );

        // `c12.build_empty_props = THROW` — and there, a cleared manager.
        assert_eq!(fm.build_all_constant_fold_space(&[]), None);
        assert_eq!(fm.build_all_map_fold_space(&[]), None);
        assert_eq!(fm.build_fold_space(&[], &[]), None);
        // `c9.build_mismatched_sizes = THROW` — `DT_ERROR("Size of props and func_base_types should
        // be same")` (`:1634-1635`), which the authority raises BEFORE clearing, so this one case it
        // too leaves intact.
        assert_eq!(
            fm.build_fold_space(&[a.clone()], &[BaseFuncType::Map; 2]),
            None
        );
        // And the refused kind, which the authority discovers only after clearing.
        assert_eq!(
            fm.build_fold_space(&[a.clone()], &[BaseFuncType::Unknown]),
            None
        );

        assert_eq!(fm.num_dims(), 2, "five refusals, nothing cleared");
        assert_eq!(fm.func_types(), vec![BaseFuncType::Map, BaseFuncType::Map]);
        assert_eq!(fm.get_data(&[FoldDimIndex(1), FoldDimIndex(2)]), Some(41));

        // `clear` is only reachable through a build that will succeed, and the zero-dimension space
        // the authority's `nullptr` window cannot serve is `reset`'s (`:1032-1038`).
        fm.reset_with_data(7);
        assert_eq!(fm.num_dims(), 0);
        assert_eq!(fm.get_data(&[]), Some(7));
    }
}

/// Every value pinned here was MEASURED, not reasoned about: the authority's own header was compiled
/// and run.
///
/// `clang++ -std=c++17 -I/Users/nickm/git/deeptools-src` over a probe that includes
/// `util/foldManager/foldInfrastructure.h` at revision `a0d29abbed` and drives the five classes
/// directly — the header is self-contained, so no other translation unit is linked. Each test names
/// the case it reproduces and the number the C++ printed; `THROW` there (a `DtException` out of
/// `DT_CHECK`/`DT_ERROR`, `util/dt_exception.hpp:105-121`) is [`None`] here.
///
/// ⛔ THESE ARE THE AUTHORITY'S ANSWERS, INCLUDING THE ONES NOBODY WANTED: a constant leaf answering
/// at the wrong depth, a `size_t` underflow on an empty coordinate list, an affine leaf swallowing
/// `insertData`, `int` arithmetic truncating, and a prefix printed twice. A test that "corrected" any
/// of them would pin OUR divergence as the reference.
#[cfg(test)]
mod equivalence {
    use super::*;

    /// `c14.one = 77`, `c14.wrong_index = 77`, `c14.empty = 77`, `c14.too_many = 77`,
    /// `c14.default = 0`, `c14.after_insert = -9`.
    ///
    /// ⛔ FOUR COORDINATE LISTS, ONE ANSWER. `ConstFoldFunction_Leaf::getData` (`:318-321`) reads
    /// neither the list nor its length, so it is the one kind with no depth contract at all — its
    /// affine sibling throws for three of these four (`:579`).
    #[test]
    fn e014_a_constant_leaf_answers_its_value_for_any_coordinate_list() {
        let mut leaf = FoldFunc::ConstantLeaf(ConstFoldFunctionLeaf::new(77i64));
        assert_eq!(leaf.get_data(&[FoldDimIndex(0)]), Some(77));
        assert_eq!(leaf.get_data(&[FoldDimIndex(5)]), Some(77));
        assert_eq!(leaf.get_data(&[]), Some(77));
        assert_eq!(
            leaf.get_data(&[FoldDimIndex(1), FoldDimIndex(2), FoldDimIndex(3)]),
            Some(77)
        );

        assert_eq!(
            FoldFunc::ConstantLeaf(ConstFoldFunctionLeaf::<i64>::default())
                .get_data(&[FoldDimIndex(0)]),
            Some(0),
            "`Dtype data_{{}}` (:334)"
        );

        assert_eq!(leaf.insert_data(-9, &[FoldDimIndex(0)]), Some(()));
        assert_eq!(leaf.get_data(&[FoldDimIndex(0)]), Some(-9));
    }

    /// `c13.forward = 77`, `c13.own_coord_ignored = 77`, `c13.empty_underflow = 77`,
    /// `c13.single = THROW`, `c13.after_insert_forwarded = -3`, `c13.get_child_is_leaf = 1`,
    /// `c13.after_insert_func = 5`, `c13.get_fold_func_is_leaf2 = 1`.
    ///
    /// ⛔ `12345` IS OUT OF EVERY REAL LEVEL'S EXTENT AND THE ANSWER IS STILL 77: a constant level
    /// does not read its own coordinate (`:275-279`).
    /// ⛔ AN EMPTY LIST ANSWERS WHERE A ONE-ENTRY LIST THROWS. `idx < fold_dim_indices.size() - 1`
    /// (`:277`) underflows to `SIZE_MAX` on an empty `deque`, which is the whole reason
    /// [`non_leaf_rest`] tests for "exactly one" instead of "at least two".
    #[test]
    fn e013_a_constant_level_forwards_and_never_reads_its_own_coordinate() {
        let mut tree = FoldFunc::ConstantNonLeaf(ConstFoldFunctionNonLeaf::new(
            FoldFunc::ConstantLeaf(ConstFoldFunctionLeaf::new(77i64)),
        ));
        assert_eq!(tree.get_data(&[FoldDimIndex(0), FoldDimIndex(0)]), Some(77));
        assert_eq!(
            tree.get_data(&[FoldDimIndex(12345), FoldDimIndex(0)]),
            Some(77)
        );
        assert_eq!(tree.get_data(&[]), Some(77), "size() - 1 underflows (:277)");
        assert_eq!(tree.get_data(&[FoldDimIndex(0)]), None, "DT_CHECK throws");

        assert_eq!(
            tree.insert_data(-3, &[FoldDimIndex(0), FoldDimIndex(0)]),
            Some(())
        );
        assert_eq!(tree.get_data(&[FoldDimIndex(0), FoldDimIndex(0)]), Some(-3));

        if let FoldFunc::ConstantNonLeaf(nl) = &mut tree {
            assert_eq!(
                nl.child(),
                &FoldFunc::ConstantLeaf(ConstFoldFunctionLeaf::new(-3i64)),
                "getChild() (:288) sees the write"
            );
            nl.insert_func(FoldFunc::ConstantLeaf(ConstFoldFunctionLeaf::new(5i64)));
        }
        assert_eq!(tree.get_data(&[FoldDimIndex(0), FoldDimIndex(0)]), Some(5));

        // `getFoldFunc` reaches the CHILD, not this level (:293-297) — the same object `getChild`
        // hands back, which is what the `WkSplitParam` accessors (:2655-2681) rely on.
        if let FoldFunc::ConstantNonLeaf(nl) = &tree {
            let reached = tree
                .fold_func(&[FoldDimIndex(0), FoldDimIndex(0)])
                .map(std::ptr::from_ref);
            assert_eq!(reached, Some(std::ptr::from_ref(nl.child())));
        }
    }

    /// `c16.i0 = 10`, `c16.i7 = 31`, `c16.negative_index = 4`, `c16.wrong_depth = THROW`,
    /// `c16.empty = THROW`, `c16.after_ignored_insert = 31`, `c16.after_insert_alpha_beta = -8`,
    /// `c16.get_alpha = -1`, `c16.get_beta = -1`, `c16.default = 0`,
    /// `c16.print = ["alpha_" : -1, "beta_" : -1]`,
    /// `c16.print_ps = [  "alpha_" : -1,   "beta_" : -1]`.
    ///
    /// ⛔ A NEGATIVE COORDINATE IS COMPUTED WITH, NOT REJECTED: `3 * -2 + 10` is 4 (`:575-582`).
    /// ⛔ `insertData(999)` IS ACCEPTED AND DISCARDED (`:613-617`) — the leaf still answers 31. Only
    /// `insertAlpha`/`insertBeta` change an affine leaf.
    /// ⛔ THE PREFIX IS EMITTED TWICE, mid-line after the separator (`:650-651`).
    #[test]
    fn e016_an_affine_leaf_is_alpha_times_its_coordinate_plus_beta() {
        let mut al = AffineFoldFunctionLeaf::<i64>::new(3i64, 10);
        let data = |al: &AffineFoldFunctionLeaf<i64>, coords: &[FoldDimIndex]| {
            FoldFunc::AffineLeaf(al.clone()).get_data(coords)
        };
        assert_eq!(data(&al, &[FoldDimIndex(0)]), Some(10));
        assert_eq!(data(&al, &[FoldDimIndex(7)]), Some(31));
        assert_eq!(data(&al, &[FoldDimIndex(-2)]), Some(4));
        assert_eq!(
            data(&al, &[FoldDimIndex(0), FoldDimIndex(0)]),
            None,
            "idx == size() - 1 (:579)"
        );
        assert_eq!(data(&al, &[]), None, "0 == SIZE_MAX is false");

        let mut ignored = FoldFunc::AffineLeaf(al.clone());
        assert_eq!(ignored.insert_data(999, &[FoldDimIndex(0)]), Some(()));
        assert_eq!(ignored.get_data(&[FoldDimIndex(7)]), Some(31));

        al.insert_alpha(-1);
        al.insert_beta(-1);
        assert_eq!(data(&al, &[FoldDimIndex(7)]), Some(-8));
        assert_eq!((*al.alpha(), *al.beta()), (-1, -1));

        assert_eq!(
            data(
                &AffineFoldFunctionLeaf::<i64>::default(),
                &[FoldDimIndex(9)]
            ),
            Some(0),
            "alpha_{{}}, beta_{{}} (:676-677)"
        );

        let mut out = String::new();
        al.print_meta_data(&mut out, "");
        assert_eq!(out, "\"alpha_\" : -1, \"beta_\" : -1");

        let mut out = String::new();
        al.print_meta_data(&mut out, "  ");
        assert_eq!(out, "  \"alpha_\" : -1,   \"beta_\" : -1");
    }

    /// `c15.i0_i0 = 17`, `c15.i2_i7 = 238`, `c15.wrong_depth = THROW`,
    /// `c15.empty_underflow = THROW`, `c15.get_alpha = 100`, `c15.get_beta = 7`,
    /// `c15.after_insert_alpha_beta = 33`, `c15.after_insert_func = 1002`,
    /// `c15.after_insert_data = -2`, `c15.print = ["alpha_" : 1, "beta_" : 0]`,
    /// `c15.three_levels = 159`.
    ///
    /// ⛔ AN EMPTY LIST THROWS HERE AND NOT AT A CONSTANT LEVEL: the depth `DT_CHECK` (`:409`) lets
    /// it past exactly as it does at `:277`, and the `.at(idx)` on the next line (`:410`) is what
    /// raises — a different line, the same [`None`].
    /// ⛔ THE SUM IS OVER THE WHOLE CHAIN: `{1, 2, 7}` through `(100, 7)`, `(10, 1)` and a `(3, 10)`
    /// leaf is `107 + 21 + 31`.
    #[test]
    fn e015_an_affine_level_adds_its_own_term_to_its_child_s() {
        let mut anl = AffineFoldFunctionNonLeaf::<i64>::new(
            100i64,
            7,
            FoldFunc::AffineLeaf(AffineFoldFunctionLeaf::<i64>::new(3i64, 10)),
        );
        let data = |anl: &AffineFoldFunctionNonLeaf<i64>, coords: &[FoldDimIndex]| {
            FoldFunc::AffineNonLeaf(anl.clone()).get_data(coords)
        };
        assert_eq!(data(&anl, &[FoldDimIndex(0), FoldDimIndex(0)]), Some(17));
        assert_eq!(data(&anl, &[FoldDimIndex(2), FoldDimIndex(7)]), Some(238));
        assert_eq!(data(&anl, &[FoldDimIndex(5)]), None, "DT_CHECK (:409)");
        assert_eq!(data(&anl, &[]), None, "the .at(idx) one line later (:410)");
        assert_eq!((*anl.alpha(), *anl.beta()), (100, 7));

        anl.insert_alpha(1);
        anl.insert_beta(0);
        assert_eq!(data(&anl, &[FoldDimIndex(2), FoldDimIndex(7)]), Some(33));

        anl.insert_func(FoldFunc::ConstantLeaf(ConstFoldFunctionLeaf::new(1000i64)));
        assert_eq!(data(&anl, &[FoldDimIndex(2), FoldDimIndex(7)]), Some(1002));

        // `insertData` forwards to the constant leaf and never reads its own coordinate (:452-457).
        let mut tree = FoldFunc::AffineNonLeaf(anl.clone());
        assert_eq!(
            tree.insert_data(-4, &[FoldDimIndex(2), FoldDimIndex(7)]),
            Some(())
        );
        assert_eq!(tree.get_data(&[FoldDimIndex(2), FoldDimIndex(7)]), Some(-2));

        let mut out = String::new();
        anl.print_meta_data(&mut out, "");
        assert_eq!(out, "\"alpha_\" : 1, \"beta_\" : 0");

        let top = AffineFoldFunctionNonLeaf::<i64>::new(
            100i64,
            7,
            FoldFunc::AffineNonLeaf(AffineFoldFunctionNonLeaf::<i64>::new(
                10i64,
                1,
                FoldFunc::AffineLeaf(AffineFoldFunctionLeaf::<i64>::new(3i64, 10)),
            )),
        );
        assert_eq!(
            data(&top, &[FoldDimIndex(1), FoldDimIndex(2), FoldDimIndex(7)]),
            Some(159)
        );
    }

    /// `c16.int_truncates = 0`, `c16.int_normal = 11`, `c15.int_truncates = 1`.
    ///
    /// ⛔ THE AUTHORITY PROMOTES TO `int64_t` AND NARROWS ON RETURN, so a `FoldManager<int>` level
    /// (`TransferPadInfo`, `dsc/dsc2.h:810-811`) with `alpha_ = 2^20` at coordinate `2^20` answers
    /// **0**, and a non-leaf's `2^32 + 1` answers **1** — not the mathematical product. Wrapping
    /// arithmetic in `D` reproduces both bit for bit; plain `*` would panic in a debug build on a
    /// value IBM accepts.
    #[test]
    fn the_affine_arithmetic_truncates_to_the_payload_type_the_way_the_authority_does() {
        let truncating = FoldFunc::AffineLeaf(AffineFoldFunctionLeaf::<i32>::new(1i32 << 20, 0));
        assert_eq!(truncating.get_data(&[FoldDimIndex(1 << 20)]), Some(0));

        let normal = FoldFunc::AffineLeaf(AffineFoldFunctionLeaf::<i32>::new(2i32, 3));
        assert_eq!(normal.get_data(&[FoldDimIndex(4)]), Some(11));

        let level = FoldFunc::AffineNonLeaf(AffineFoldFunctionNonLeaf::<i32>::new(
            1i32 << 16,
            0,
            FoldFunc::ConstantLeaf(ConstFoldFunctionLeaf::new(1i32)),
        ));
        assert_eq!(
            level.get_data(&[FoldDimIndex(1 << 16), FoldDimIndex(0)]),
            Some(1)
        );
    }

    /// `arith.int_negative = 4`, `arith.int_coord_over_i32max = 5`, `c15.negative_own_coord = -277`.
    ///
    /// ⛔ ONE OF THE THREE ANSWERED [`None`] BEFORE THIS REVIEW PASS, AND IT IS `2^32 + 5` — measured
    /// by putting these assertions against the old body, which failed on that line alone with
    /// `left: None, right: Some(5)`. [`GetDataAffine`] reduced the coordinate with `D::try_from(i64)`
    /// and refused when it did not fit, which is a REFUSAL WHERE THE AUTHORITY CONVERTS: `:575-582`
    /// narrows on return, it never validates. The other two are here because they are what the old
    /// justification — "a coordinate is bounded by its level's `FoldDimProp::factor_`" — was about:
    /// MAGNITUDE. It was wrong on its own terms, since the coordinate that overflows `int` is exactly
    /// the reachable case; and it was silent about SIGN, where the refusal was worse still — at a
    /// `uint32_t` or `uint64_t` payload `try_from` failed for EVERY negative coordinate while the
    /// authority answered 4 (`arith.u64_negative`, `arith.u32_negative`). Those two payloads are now a
    /// compile error rather than a wrong answer, pinned by [`AffineFoldFunctionLeaf`]'s
    /// `compile_fail` block.
    /// ⛔ A NEGATIVE COORDINATE IS LEGAL ALL THE WAY DOWN: `FoldManager::isLegal` widens a `uint32_t`
    /// extent to `int64_t` for its range test (`:1677`), and
    /// [`PadSizeFold`](crate::schedule::dsc2) — the live `FoldManager<int>` consumer — records the
    /// same fact in its own header.
    #[test]
    fn an_affine_coordinate_is_converted_at_the_payload_s_width_and_never_refused() {
        let narrow = FoldFunc::AffineLeaf(AffineFoldFunctionLeaf::<i32>::new(3, 10));
        assert_eq!(narrow.get_data(&[FoldDimIndex(-2)]), Some(4), "3 * -2 + 10");

        let past_i32_max = FoldFunc::AffineLeaf(AffineFoldFunctionLeaf::<i32>::new(1, 0));
        assert_eq!(
            past_i32_max.get_data(&[FoldDimIndex((1 << 32) | 5)]),
            Some(5),
            "static_cast<int>(2^32 + 5)"
        );

        let level = FoldFunc::AffineNonLeaf(AffineFoldFunctionNonLeaf::<i64>::new(
            100,
            7,
            FoldFunc::AffineLeaf(AffineFoldFunctionLeaf::<i64>::new(3, 10)),
        ));
        assert_eq!(
            level.get_data(&[FoldDimIndex(-3), FoldDimIndex(2)]),
            Some(-277),
            "(100 * -3 + 7) + (3 * 2 + 10)"
        );
    }

    /// `c13.three_coords = 77`, `c13.over_affine_two = 31`, `c13.over_affine_three = THROW`,
    /// `c13.two_levels_three = 31`, `c15.three_coords = THROW`, `c15.over_const_three = 1002`,
    /// `c17.three_coords = 20`.
    ///
    /// ⛔ NO LEVEL BOUNDS THE LIST FROM ABOVE, SO THE LEAF DECIDES. Every non-leaf's `DT_CHECK` only
    /// asks that it is not itself the last (`:277`, `:409`, `:701`), so whether a too-deep list is
    /// refused depends entirely on what the LEAF demands: a constant leaf reads neither the list nor
    /// its length (`:318-321`) and answers, an affine leaf insists on being last (`:579`) and throws.
    /// ⛔ THE SAME AFFINE LEVEL THEREFORE ANSWERS 1002 OVER A CONSTANT LEAF AND THROWS OVER AN AFFINE
    /// ONE — which is why [`non_leaf_rest`] hands the tail down instead of validating a depth it
    /// cannot know.
    /// ⛔ THIS TEST PASSES AGAINST THE PRE-REVIEW BODY TOO — measured. It is COVERAGE, not a
    /// regression pin: the depth contract was already right and nothing asserted it past two
    /// coordinates.
    #[test]
    fn a_too_deep_coordinate_list_is_refused_by_the_leaf_and_never_by_the_level_above_it() {
        let three = [FoldDimIndex(1), FoldDimIndex(2), FoldDimIndex(3)];

        let over_const = FoldFunc::ConstantNonLeaf(ConstFoldFunctionNonLeaf::new(
            FoldFunc::ConstantLeaf(ConstFoldFunctionLeaf::new(77i64)),
        ));
        assert_eq!(over_const.get_data(&three), Some(77));

        let over_affine = FoldFunc::ConstantNonLeaf(ConstFoldFunctionNonLeaf::new(
            FoldFunc::AffineLeaf(AffineFoldFunctionLeaf::<i64>::new(3, 10)),
        ));
        assert_eq!(
            over_affine.get_data(&[FoldDimIndex(5), FoldDimIndex(7)]),
            Some(31)
        );
        assert_eq!(
            over_affine.get_data(&three),
            None,
            "the leaf's :579, not :277"
        );

        let two_levels = FoldFunc::ConstantNonLeaf(ConstFoldFunctionNonLeaf::new(
            FoldFunc::ConstantNonLeaf(ConstFoldFunctionNonLeaf::new(FoldFunc::AffineLeaf(
                AffineFoldFunctionLeaf::<i64>::new(3, 10),
            ))),
        ));
        assert_eq!(
            two_levels.get_data(&[FoldDimIndex(1), FoldDimIndex(2), FoldDimIndex(7)]),
            Some(31),
            "two levels shed two coordinates: 3 * 7 + 10"
        );

        let affine_over_affine = FoldFunc::AffineNonLeaf(AffineFoldFunctionNonLeaf::<i64>::new(
            100,
            7,
            FoldFunc::AffineLeaf(AffineFoldFunctionLeaf::<i64>::new(3, 10)),
        ));
        assert_eq!(affine_over_affine.get_data(&three), None);

        let affine_over_const = FoldFunc::AffineNonLeaf(AffineFoldFunctionNonLeaf::<i64>::new(
            1,
            0,
            FoldFunc::ConstantLeaf(ConstFoldFunctionLeaf::new(1000i64)),
        ));
        assert_eq!(
            affine_over_const.get_data(&[FoldDimIndex(2), FoldDimIndex(7), FoldDimIndex(9)]),
            Some(1002),
            "the same kind of level, and it answers"
        );

        let map = FoldFunc::MapNonLeaf(MapFoldFunctionNonLeaf::new(vec![
            FoldFunc::ConstantLeaf(ConstFoldFunctionLeaf::new(10i64)),
            FoldFunc::ConstantLeaf(ConstFoldFunctionLeaf::new(20i64)),
            FoldFunc::ConstantLeaf(ConstFoldFunctionLeaf::new(30i64)),
        ]));
        assert_eq!(map.get_data(&three), Some(20));
    }

    /// `c13.empty_underflow = 77`, `c13.insert_empty_list = OK` then `c13.after_insert_empty = -3`,
    /// `c13.get_fold_func_empty_is_leaf2 = 1`, `c13.over_affine_empty = THROW`,
    /// `c13.two_levels_empty = THROW`, `c15.insert_data_empty = OK` then
    /// `c15.after_insert_data_empty = -2`, `c15.insert_empty_over_affine_leaf = OK`,
    /// `c15.insert_empty_over_map_leaf = THROW`, `c17.insert_empty = THROW`,
    /// `c17.get_fold_func_empty = THROW`.
    ///
    /// ⛔ THE EMPTY LIST IS [`non_leaf_rest`]'S WHOLE REASON FOR EXISTING AND ONLY `getData` PINNED
    /// IT. `idx < fold_dim_indices.size() - 1` underflows to `SIZE_MAX` on an empty `deque` (`:277`,
    /// `:409`, `:701`), so every non-leaf passes its OWN check and hands the same empty list down —
    /// the tail never shrinks, which is why the slice model answers `Some(&[])` rather than refusing.
    /// Two constant levels over an affine leaf therefore throw at the LEAF, not at either level.
    /// ⛔ A LEVEL THAT READS THE COORDINATE IN ITS OWN BODY THROWS AT ONCE: the Map one's `.at(idx)`
    /// (`:702`) is a different line from its `DT_CHECK`, and `insertData`/`getFoldFunc` reach it the
    /// same way `getData` does — the two walks the `WkSplitParam` accessors use (`:2655-2681`).
    /// ⛔ THIS TEST ALSO PASSES AGAINST THE PRE-REVIEW BODY — measured. [`non_leaf_rest`] was already
    /// correct; what was missing is that only `getData` exercised it, so the two walks below rested on
    /// an unasserted claim.
    #[test]
    fn an_empty_coordinate_list_passes_every_level_and_stops_at_the_first_one_that_reads_it() {
        let mut const_over_const = FoldFunc::ConstantNonLeaf(ConstFoldFunctionNonLeaf::new(
            FoldFunc::ConstantLeaf(ConstFoldFunctionLeaf::new(77i64)),
        ));
        assert_eq!(const_over_const.get_data(&[]), Some(77));
        assert_eq!(const_over_const.insert_data(-3, &[]), Some(()));
        assert_eq!(const_over_const.get_data(&[]), Some(-3));
        if let FoldFunc::ConstantNonLeaf(nl) = &const_over_const {
            let reached = const_over_const.fold_func(&[]).map(std::ptr::from_ref);
            assert_eq!(
                reached,
                Some(std::ptr::from_ref(nl.child())),
                "getFoldFunc still reaches the child (:293-297)"
            );
        }

        let over_affine = FoldFunc::ConstantNonLeaf(ConstFoldFunctionNonLeaf::new(
            FoldFunc::AffineLeaf(AffineFoldFunctionLeaf::<i64>::new(3, 10)),
        ));
        assert_eq!(over_affine.get_data(&[]), None, "the leaf's :579");

        let two_levels = FoldFunc::ConstantNonLeaf(ConstFoldFunctionNonLeaf::new(
            FoldFunc::ConstantNonLeaf(ConstFoldFunctionNonLeaf::new(FoldFunc::AffineLeaf(
                AffineFoldFunctionLeaf::<i64>::new(3, 10),
            ))),
        ));
        assert_eq!(
            two_levels.get_data(&[]),
            None,
            "both levels pass their own check and neither shrinks the tail"
        );

        let mut affine_over_const = FoldFunc::AffineNonLeaf(AffineFoldFunctionNonLeaf::<i64>::new(
            1,
            0,
            FoldFunc::ConstantLeaf(ConstFoldFunctionLeaf::new(1000i64)),
        ));
        assert_eq!(affine_over_const.insert_data(-4, &[]), Some(()));
        assert_eq!(
            affine_over_const.get_data(&[FoldDimIndex(2), FoldDimIndex(7)]),
            Some(-2),
            "1 * 2 + 0 + -4 — the write landed through an empty list"
        );

        let mut affine_over_affine =
            FoldFunc::AffineNonLeaf(AffineFoldFunctionNonLeaf::<i64>::new(
                1,
                0,
                FoldFunc::AffineLeaf(AffineFoldFunctionLeaf::<i64>::new(3, 10)),
            ));
        assert_eq!(
            affine_over_affine.insert_data(7, &[]),
            Some(()),
            "an affine leaf's insertData is `// ignored` and checks no depth (:613-617)"
        );

        let mut affine_over_map = FoldFunc::AffineNonLeaf(AffineFoldFunctionNonLeaf::<i64>::new(
            1,
            0,
            FoldFunc::MapLeaf(MapFoldFunctionLeaf::new(vec![1i64, 2, 3]).unwrap()),
        ));
        assert_eq!(
            affine_over_map.insert_data(7, &[]),
            None,
            "a map leaf's insertData does check its depth"
        );

        let mut map =
            FoldFunc::MapNonLeaf(MapFoldFunctionNonLeaf::new(vec![FoldFunc::ConstantLeaf(
                ConstFoldFunctionLeaf::new(10i64),
            )]));
        assert_eq!(map.insert_data(-8, &[]), None, "the .at(idx) at :702");
        assert_eq!(map.fold_func(&[]), None);
    }

    /// `c17.k0 = 10`, `c17.k2 = 30`, `c17.out_of_range = THROW`, `c17.negative = THROW`,
    /// `c17.wrong_depth = THROW`, `c17.empty_underflow = THROW`, `c17.after_insert = -7`,
    /// `c17.children = 3`, `c17.after_children_write = -11`, `c17.get_fold_func_is_kid2 = 1`,
    /// `c17.affine_kid0 = 9`, `c17.affine_kid1 = 100`.
    ///
    /// ⛔ THE ONLY KIND THAT BOUNDS ITS OWN COORDINATE (`:703`), and the only one whose children may
    /// differ from each other — `affine_kid0`/`affine_kid1` are two different fold functions under one
    /// level, which is what `createSubTreeForEachMapChild` builds (`:2250-2257`).
    /// ⛔ -1 IS REJECTED BY AN UNSIGNED COMPARISON, not by a sign test: `child_ff_vec_.size() >
    /// dim_index` reads it as `SIZE_MAX`. `usize::try_from` gives the same answer for the honest
    /// reason.
    /// ⛔ `c17.hole_children = 3, first_is_null = 1` IS NOT REPRODUCED — it is the
    /// `MapFoldFunction_NonLeaf(int dim_size)` constructor (`:690-693`), whose `nullptr` children
    /// `getData` would dereference (`:704`). It is omitted, so there is no Rust expression to pin.
    #[test]
    fn e017_a_map_level_selects_one_subtree_per_coordinate() {
        let mut tree = FoldFunc::MapNonLeaf(MapFoldFunctionNonLeaf::new(vec![
            FoldFunc::ConstantLeaf(ConstFoldFunctionLeaf::new(10i64)),
            FoldFunc::ConstantLeaf(ConstFoldFunctionLeaf::new(20i64)),
            FoldFunc::ConstantLeaf(ConstFoldFunctionLeaf::new(30i64)),
        ]));
        assert_eq!(tree.get_data(&[FoldDimIndex(0), FoldDimIndex(0)]), Some(10));
        assert_eq!(tree.get_data(&[FoldDimIndex(2), FoldDimIndex(0)]), Some(30));
        assert_eq!(tree.get_data(&[FoldDimIndex(3), FoldDimIndex(0)]), None);
        assert_eq!(tree.get_data(&[FoldDimIndex(-1), FoldDimIndex(0)]), None);
        assert_eq!(tree.get_data(&[FoldDimIndex(1)]), None, "DT_CHECK (:701)");
        assert_eq!(tree.get_data(&[]), None, "the .at(idx) at :702");

        assert_eq!(
            tree.insert_data(-7, &[FoldDimIndex(1), FoldDimIndex(0)]),
            Some(())
        );
        assert_eq!(tree.get_data(&[FoldDimIndex(1), FoldDimIndex(0)]), Some(-7));

        if let FoldFunc::MapNonLeaf(m) = &mut tree {
            assert_eq!(m.children().len(), 3, "one child per coordinate (:1919)");
            m.children_mut()[0] = FoldFunc::ConstantLeaf(ConstFoldFunctionLeaf::new(-11i64));
        }
        assert_eq!(
            tree.get_data(&[FoldDimIndex(0), FoldDimIndex(0)]),
            Some(-11)
        );

        if let FoldFunc::MapNonLeaf(m) = &tree {
            let reached = tree
                .fold_func(&[FoldDimIndex(2), FoldDimIndex(0)])
                .map(std::ptr::from_ref);
            assert_eq!(reached, Some(std::ptr::from_ref(&m.children()[2])));
        }

        let affine = FoldFunc::MapNonLeaf(MapFoldFunctionNonLeaf::new(vec![
            FoldFunc::AffineLeaf(AffineFoldFunctionLeaf::<i64>::new(1i64, 0)),
            FoldFunc::AffineLeaf(AffineFoldFunctionLeaf::<i64>::new(0i64, 100)),
        ]));
        assert_eq!(
            affine.get_data(&[FoldDimIndex(0), FoldDimIndex(9)]),
            Some(9)
        );
        assert_eq!(
            affine.get_data(&[FoldDimIndex(1), FoldDimIndex(9)]),
            Some(100)
        );
    }

    /// `mapleaf.under_const_level = 30`, `mapleaf.out_of_range = THROW`, `mapleaf.negative = THROW`,
    /// `mapleaf.too_deep = THROW`, `mapleaf.after_insert_through_level = -5`,
    /// `mapleaf.get_fold_func_is_leaf = 1`, `mapleaf.root_one_coord = 30`,
    /// `mapleaf.root_two_coords = THROW`, `mapleaf.root_empty = THROW`,
    /// `mapleaf.under_affine_level = 227`, `mapleaf.under_map_level_1_1 = 4`.
    ///
    /// ⛔ THE MAP LEAF IS e018 AND ITS TYPE IS NOT PORTED HERE — this pins the WALK, which is: the
    /// depth `DT_CHECK` at `:757` that [`MapFoldFunctionLeaf::data`]'s one-coordinate signature
    /// leaves to its caller. Every level above it is one of the four ported here, so a leaf that
    /// answered at the wrong depth would be this file's defect and nothing else would catch it:
    /// `mapleaf.root_two_coords` and `mapleaf.too_deep` are that check, from both sides.
    #[test]
    fn a_map_leaf_reached_through_a_level_answers_only_at_the_last_coordinate() {
        let run = || MapFoldFunctionLeaf::new(vec![10i64, 20, 30]).unwrap();
        let mut level =
            FoldFunc::ConstantNonLeaf(ConstFoldFunctionNonLeaf::new(FoldFunc::MapLeaf(run())));
        assert_eq!(
            level.get_data(&[FoldDimIndex(7), FoldDimIndex(2)]),
            Some(30)
        );
        assert_eq!(level.get_data(&[FoldDimIndex(0), FoldDimIndex(3)]), None);
        assert_eq!(level.get_data(&[FoldDimIndex(0), FoldDimIndex(-1)]), None);
        assert_eq!(
            level.get_data(&[FoldDimIndex(0), FoldDimIndex(1), FoldDimIndex(1)]),
            None,
            "the leaf is not the last level (:757)"
        );

        assert_eq!(
            level.insert_data(-5, &[FoldDimIndex(0), FoldDimIndex(1)]),
            Some(())
        );
        assert_eq!(
            level.get_data(&[FoldDimIndex(0), FoldDimIndex(1)]),
            Some(-5)
        );
        assert_eq!(
            level.insert_data(-5, &[FoldDimIndex(0), FoldDimIndex(3)]),
            None,
            "and the write can miss (:768)"
        );

        if let FoldFunc::ConstantNonLeaf(nl) = &level {
            let reached = level
                .fold_func(&[FoldDimIndex(0), FoldDimIndex(1)])
                .map(std::ptr::from_ref);
            assert_eq!(reached, Some(std::ptr::from_ref(nl.child())));
        }

        let root = FoldFunc::MapLeaf(run());
        assert_eq!(root.get_data(&[FoldDimIndex(2)]), Some(30));
        assert_eq!(root.get_data(&[FoldDimIndex(2), FoldDimIndex(0)]), None);
        assert_eq!(root.get_data(&[]), None);

        let affine = FoldFunc::AffineNonLeaf(AffineFoldFunctionNonLeaf::<i64>::new(
            100i64,
            7,
            FoldFunc::MapLeaf(run()),
        ));
        assert_eq!(
            affine.get_data(&[FoldDimIndex(2), FoldDimIndex(1)]),
            Some(227)
        );

        let map = FoldFunc::MapNonLeaf(MapFoldFunctionNonLeaf::new(vec![
            FoldFunc::MapLeaf(MapFoldFunctionLeaf::new(vec![1i64, 2]).unwrap()),
            FoldFunc::MapLeaf(MapFoldFunctionLeaf::new(vec![3i64, 4]).unwrap()),
        ]));
        assert_eq!(map.get_data(&[FoldDimIndex(1), FoldDimIndex(1)]), Some(4));
    }

    /// `e009.print_core = ["factor_" : 32, "label_" : "core_fold_dim"]`,
    /// `e009.print_unfolded = ["factor_" : 1, "label_" : ""]`,
    /// `e009.print_after_setters = ["factor_" : 64, "label_" : "loop_3_dim"]`,
    /// `e009.eq_after_setters = 1`, `e009.eq_same_size_diff_label = 0`,
    /// `e009.eq_same_label_diff_size = 0`, `e009.eq_identical = 1`, `e009.copy_roundtrip_eq = 1`.
    ///
    /// The props are IBM's own (`util/foldManager/test/test_fold_infrastructure.cpp:324-336`) and the
    /// setter order is `buildTransferFoldDim`'s (`dsc/dsc2.cpp:4669-4675`).
    /// ⛔ THE LABEL IS PART OF EQUALITY (`:148-150`), so each `assert_ne!` below moves ONE field.
    #[test]
    fn e009_a_fold_dim_prop_carries_its_extent_and_its_label_and_prints_both() {
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

        // `dsc/superdsc.cpp:60-67` copies a prop by rebuilding it out of its two getters.
        let src = FoldDimProp::new(FoldDimSize(8), "dims");
        assert_eq!(FoldDimProp::new(src.size(), src.label()), src);
        assert_eq!(src.clone(), src, "and `Clone` is that same copy");
    }

    /// `e010.Constant_leaf = { Type=0, isLeaf=1, isNonLeaf=0 }`, `Map_leaf = { 1, 1, 0 }`,
    /// `Affine_leaf = { 2, 1, 0 }`, `Constant_nonleaf = { 3, 0, 1 }`, `Map_nonleaf = { 4, 0, 1 }`,
    /// `Affine_nonleaf = { 5, 0, 1 }`, `WkSplit_leaf = { 6, 0, 0 }`, `Unknown = { 7, 0, 0 }`.
    ///
    /// All eight rows came off ONE instance: `type_` is public (`:180`), so the probe reassigned it
    /// per row — the only way to reach a tag no constructor produces, `Unknown` being the one.
    /// ⛔ THE TWO PREDICATES ARE NEITHER COMPLEMENTS NOR A COVER (`:182-190`): `WkSplit_leaf` is
    /// absent from BOTH lists, so the one kind that exists only as a leaf answers false to `isLeaf`.
    #[test]
    fn e010_the_two_leaf_predicates_are_not_complements_and_wksplit_is_in_neither() {
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

    /// `FoldDimProp::new` in the probe's own argument order, so a dimension reads here as it does
    /// there: `FoldDimProp p0(2, "outer")`.
    fn prop(factor: u32, label: &str) -> FoldDimProp {
        FoldDimProp::new(FoldDimSize(factor), label)
    }

    /// The probe's `{{pos, coord}, ...}` initialiser list — `std::map<int64_t, int64_t>
    /// pos_to_fixCoord` (`:1983`), whose key is a fold dim POSITION and whose value is a coordinate
    /// in it.
    fn fixed(pairs: &[(i32, i64)]) -> BTreeMap<FoldDimPos, FoldDimIndex> {
        pairs
            .iter()
            .map(|&(pos, coord)| (FoldDimPos(pos), FoldDimIndex(coord)))
            .collect()
    }

    /// The probe's `coords()` formatter, so a coordinate list can be transcribed exactly as the C++
    /// printed it instead of being re-derived as nested vectors.
    fn coords(cs: &[Vec<FoldDimIndex>]) -> String {
        let cs: Vec<String> = cs
            .iter()
            .map(|c| {
                let c: Vec<String> = c.iter().map(|i| i.0.to_string()).collect();
                format!("({})", c.join(" "))
            })
            .collect();
        format!("[{}]", cs.join(","))
    }

    /// `c1.default_data = 0`, `c1.default_dims = 0`, `c1.default_zero = 1`, `c1.seeded_data = 3`,
    /// `c1.seeded_legal_empty = 1`, `c1.seeded_legal_one = 1`, `c1.seeded_size = []`,
    /// `c1.seeded_data_and_coord = (-1)=3`, `c1.seeded_all_data = [3]`, `r.reset = OK`, `r.dims = 0`,
    /// `r.data = 0`, `r.reset_with_data = OK`, `r.data_after = 11`.
    ///
    /// ⛔ A ZERO-DIMENSION MANAGER IS NOT AN EMPTY ONE: it holds exactly one value, and the
    /// coordinate it reports for that value is `[-1]` — a sentinel `getDataAndFoldCoordinates`
    /// invents (`:2067`) and `getFlattenedCoordinates` does not, which is why the two disagree on the
    /// same manager.
    #[test]
    fn e026_a_fresh_fold_manager_is_a_zero_dimension_space_holding_one_value() {
        let fm = FoldManager::<i64>::new();
        assert_eq!(fm.get_data(&[]), Some(0));
        assert_eq!(fm.num_dims(), 0);
        assert!(fm.has_zero_fold_dim());

        let seeded = FoldManager::<i64>::with_data(3);
        assert_eq!(seeded.get_data(&[]), Some(3));
        assert_eq!(seeded.is_legal(&[]), Some(()));
        assert_eq!(
            seeded.is_legal(&[FoldDimIndex(7)]),
            Some(()),
            "a zero-dimension manager calls EVERY list legal, including a non-empty one"
        );
        assert!(seeded.fold_space_size().is_empty());
        assert_eq!(
            seeded.data_and_fold_coordinates(&fixed(&[]), ScanInnerOuter(false)),
            Some(vec![(vec![FoldDimIndex(-1)], 3)])
        );
        assert_eq!(
            seeded.all_data(&fixed(&[]), ScanInnerOuter(false)),
            Some(vec![3])
        );
        assert_eq!(
            seeded
                .flattened_coordinates(&fixed(&[]), ScanInnerOuter(false), ExpandIfAnyMap(false))
                .as_deref(),
            Some(&[][..]),
            "the `[-1]` above belongs to `getDataAndFoldCoordinates` alone"
        );

        let mut fm = FoldManager::<i64>::with_data(7);
        assert_eq!(fm.build_map_dim(&prop(2, "a"), FoldDimPos(0)), Some(()));
        fm.reset();
        assert_eq!(fm.num_dims(), 0);
        assert_eq!(fm.get_data(&[]), Some(0));
        fm.reset_with_data(11);
        assert_eq!(fm.get_data(&[]), Some(11));
    }

    /// `c2.build_const_0 = OK`, `c2.after_const_dims = 1`, `c2.after_const_data = 9`,
    /// `c2.legal_past_end = THROW`, `c2.legal_negative = 1`, `c2.data_negative = 9`,
    /// `c2.build_const_1 = OK`, `c2.two_dims = 2`, `c2.two_data = 9`, `c2.func_types = 0`,
    /// `c2.space = [3,2]`, `c2.coords = [(0 0)]`, `c2.all_data = [9]`, `c2.insert = OK`,
    /// `c2.after_insert = -4`.
    ///
    /// ⛔ THE SEEDED VALUE SURVIVES THE FIRST BUILD. `buildConstDim` on a zero-dimension manager
    /// wraps the constant leaf it already has (`:1446-1450`), so a manager built from
    /// `FoldManager(9)` answers 9 at every coordinate of its new dimension.
    /// ⛔ AND THE ONE CELL IS SHARED: writing at `{1,1}` is readable at `{0,0}`. That is what a
    /// Constant fold dimension MEANS, and it is why `getAllData` on a two-dimension constant space
    /// answers one value and not six.
    #[test]
    fn e026_a_constant_dimension_answers_its_one_value_at_every_coordinate() {
        let mut fm = FoldManager::<i64>::with_data(9);
        assert_eq!(
            fm.build_const_dim(&prop(2, "outer"), FoldDimPos(0)),
            Some(())
        );
        assert_eq!(fm.num_dims(), 1);
        assert_eq!(fm.get_data(&[FoldDimIndex(1)]), Some(9));
        assert_eq!(fm.is_legal(&[FoldDimIndex(2)]), None, "past the extent");
        assert_eq!(
            fm.is_legal(&[FoldDimIndex(-1)]),
            Some(()),
            "the per-dimension test is signed, so a negative coordinate passes it"
        );
        assert_eq!(fm.get_data(&[FoldDimIndex(-1)]), Some(9));

        assert_eq!(
            fm.build_const_dim(&prop(3, "inner"), FoldDimPos(0)),
            Some(())
        );
        assert_eq!(fm.num_dims(), 2);
        assert_eq!(fm.get_data(&[FoldDimIndex(2), FoldDimIndex(1)]), Some(9));
        assert_eq!(
            fm.func_types(),
            vec![BaseFuncType::Constant, BaseFuncType::Constant]
        );
        assert_eq!(fm.fold_space_size(), vec![FoldDimSize(3), FoldDimSize(2)]);
        assert_eq!(
            coords(
                &fm.flattened_coordinates(
                    &fixed(&[]),
                    ScanInnerOuter(false),
                    ExpandIfAnyMap(false)
                )
                .unwrap()
            ),
            "[(0 0)]"
        );
        assert_eq!(
            fm.all_data(&fixed(&[]), ScanInnerOuter(false)),
            Some(vec![9])
        );
        assert_eq!(
            fm.insert_data(-4, &[FoldDimIndex(1), FoldDimIndex(1)]),
            Some(())
        );
        assert_eq!(fm.get_data(&[FoldDimIndex(0), FoldDimIndex(0)]), Some(-4));
    }

    /// `c3.build = OK`, `c3.dims = 2`, `c3.coords = [(0 0),(1 0),(0 1),(1 1),(0 2),(1 2)]`,
    /// `c3.coords_inner_outer = [(0 0),(0 1),(0 2),(1 0),(1 1),(1 2)]`,
    /// `c3.all_data = [0,10,1,11,2,12]`, `c3.all_data_inner_outer = [0,1,2,10,11,12]`,
    /// `c3.fixed_outer_1 = [10,11,12]`, `c3.single = 0`, `c3.single_fixed = 12`, `c3.data = 12`,
    /// `c3.dim_size_1 = 3`.
    ///
    /// ⛔ THE DEFAULT SCAN VARIES THE OUTERMOST DIMENSION FASTEST — `(0 0),(1 0),(0 1)`, not
    /// `(0 0),(0 1),(0 2)`. `scan_inner_outer` is the flag that gives the ORDER A C LOOP NEST WOULD
    /// PRODUCE, and it is not the default; every caller that walks data in memory order has to pass
    /// it (`:2016-2021`).
    #[test]
    fn e026_an_all_map_fold_space_varies_the_outermost_dimension_fastest_by_default() {
        let mut fm = FoldManager::<i64>::new();
        assert_eq!(
            fm.build_all_map_fold_space(&[prop(2, "outer"), prop(3, "inner")]),
            Some(())
        );
        assert_eq!(fm.num_dims(), 2);
        let all = fm
            .flattened_coordinates(&fixed(&[]), ScanInnerOuter(false), ExpandIfAnyMap(false))
            .unwrap();
        assert_eq!(coords(&all), "[(0 0),(1 0),(0 1),(1 1),(0 2),(1 2)]");
        let inner_outer = fm
            .flattened_coordinates(&fixed(&[]), ScanInnerOuter(true), ExpandIfAnyMap(false))
            .unwrap();
        assert_eq!(
            coords(&inner_outer),
            "[(0 0),(0 1),(0 2),(1 0),(1 1),(1 2)]"
        );

        for i in 0..2i64 {
            for j in 0..3i64 {
                assert_eq!(
                    fm.insert_data(10 * i + j, &[FoldDimIndex(i), FoldDimIndex(j)]),
                    Some(())
                );
            }
        }
        assert_eq!(
            fm.all_data(&fixed(&[]), ScanInnerOuter(false)),
            Some(vec![0, 10, 1, 11, 2, 12])
        );
        assert_eq!(
            fm.all_data(&fixed(&[]), ScanInnerOuter(true)),
            Some(vec![0, 1, 2, 10, 11, 12])
        );
        assert_eq!(
            fm.all_data(&fixed(&[(0, 1)]), ScanInnerOuter(false)),
            Some(vec![10, 11, 12]),
            "fixing the outer dimension drops it from the walk"
        );
        assert_eq!(fm.single_data(&fixed(&[])), Some(0));
        assert_eq!(fm.single_data(&fixed(&[(0, 1), (1, 2)])), Some(12));
        assert_eq!(fm.get_data(&[FoldDimIndex(1), FoldDimIndex(2)]), Some(12));
        assert_eq!(fm.fold_dim_size(FoldDimPos(1)), Some(FoldDimSize(3)));
    }

    /// `c4.build = OK`, `c4.alpha_beta = OK`, `c4.data_3 = 7`, `c4.alpha = 2`, `c4.beta = 1`,
    /// `c4.all_data = [1,3,5,7]`, `c4.build_outer = OK`, `c4.outer_alpha_beta = OK`,
    /// `c4.two_level_data = 107`, `c4.two_level_all = [1,101,3,103,5,105,7,107]`,
    /// `c4.alpha_outer = 100`, `c4.alpha_inner = 2`, `c4.get_alpha_beta_values = 2001`,
    /// `c4.alpha_on_const = THROW`.
    ///
    /// ⛔ AN AFFINE LEVEL'S VALUE IS ITS OWN TERM PLUS ITS CHILD'S, so a two-level affine space
    /// holds no stored data at all and `getAllData` computes all eight values from four numbers.
    #[test]
    fn e026_an_affine_level_is_read_through_the_alpha_and_beta_of_that_level() {
        let mut fm = FoldManager::<i64>::new();
        assert_eq!(
            fm.build_affine_dim(&prop(4, "affine"), FoldDimPos(0)),
            Some(())
        );
        assert_eq!(fm.insert_alpha_beta(&2, &1, FoldDimPos(0)), Some(()));
        assert_eq!(fm.get_data(&[FoldDimIndex(3)]), Some(7));
        assert_eq!(fm.alpha(FoldDimPos(0)), Some(&2));
        assert_eq!(
            fm.beta(FoldDimPos(-1)),
            Some(&1),
            "the affine accessors take a from-the-end position"
        );
        assert_eq!(
            fm.all_data(&fixed(&[]), ScanInnerOuter(false)),
            Some(vec![1, 3, 5, 7])
        );

        assert_eq!(
            fm.build_affine_dim(&prop(2, "outer_affine"), FoldDimPos(0)),
            Some(())
        );
        assert_eq!(fm.insert_alpha_beta(&100, &0, FoldDimPos(0)), Some(()));
        assert_eq!(fm.get_data(&[FoldDimIndex(1), FoldDimIndex(3)]), Some(107));
        assert_eq!(
            fm.all_data(&fixed(&[]), ScanInnerOuter(false)),
            Some(vec![1, 101, 3, 103, 5, 105, 7, 107])
        );
        assert_eq!(fm.alpha(FoldDimPos(0)), Some(&100));
        assert_eq!(fm.alpha(FoldDimPos(1)), Some(&2));
        assert_eq!(fm.alpha_beta(FoldDimPos(1)), Some((&2, &1)));
        assert_eq!(
            FoldManager::<i64>::new().alpha(FoldDimPos(0)),
            None,
            "a zero-dimension manager has no level to read an alpha from"
        );
    }

    /// `m.insert_alpha_beta = OK`, `m.alpha_under_map = THROW`,
    /// `m.get_alpha_beta_under_map = OK`, `m.get_alpha_beta_values = 7002`,
    /// `m.all_data = [2,2,9,9,16,16]`, `m.data = 16`.
    ///
    /// ⛔ `getAlpha` AND `getAlphaBeta` DISAGREE ON THE SAME LEVEL. An Affine level under a Map level
    /// has one node per map child; `insertAlphaBeta` writes all of them (`:2381-2404`), `getAlphaBeta`
    /// reads the first (`:2428`), and `getAlpha` REFUSES because it asserts the level has exactly one
    /// node (`:2330-2332`). Two readers of one field with two different contracts, kept as they are.
    #[test]
    fn e026_an_affine_level_under_a_map_level_has_an_alpha_only_one_of_its_two_readers_will_report()
    {
        let mut fm = FoldManager::<i64>::new();
        assert_eq!(fm.build_affine_dim(&prop(3, "i"), FoldDimPos(0)), Some(()));
        assert_eq!(fm.build_map_dim(&prop(2, "o"), FoldDimPos(0)), Some(()));
        assert_eq!(fm.insert_alpha_beta(&7, &2, FoldDimPos(1)), Some(()));
        assert_eq!(fm.alpha(FoldDimPos(1)), None);
        assert_eq!(fm.alpha_beta(FoldDimPos(1)), Some((&7, &2)));
        assert_eq!(
            fm.all_data(&fixed(&[]), ScanInnerOuter(false)),
            Some(vec![2, 2, 9, 9, 16, 16]),
            "both map children carry the alpha the one insert wrote"
        );
        assert_eq!(fm.get_data(&[FoldDimIndex(1), FoldDimIndex(2)]), Some(16));
    }

    /// `c5.build_map_front = OK`, `c5.dims = 2`, `c5.data_0 = 7`, `c5.data_1 = 7`,
    /// `c5.after_insert_0 = 7`, `c5.after_insert_1 = -1`, `c5.all_data = [7,-1]`,
    /// `c5.all_data_unrolled = [7,-1,7,-1,7,-1]`.
    ///
    /// ⛔ A NEW MAP LEVEL CLONES THE SUBTREE UNDER IT INTO EVERY CHILD, so the value that was there
    /// once is now there twice and the two are then independent. In the authority that is
    /// `createNonLeafFunc` plus `createSubTreeForEachMapChild` plus `copySubTree` (`:1360-1367`); here
    /// it is one `Clone`, and the measured pair `data_0 == data_1 == 7` is what says the children were
    /// SEEDED rather than defaulted.
    #[test]
    fn e026_a_map_level_built_in_front_of_a_tree_clones_that_tree_into_every_child() {
        let mut fm = FoldManager::<i64>::new();
        assert_eq!(
            fm.build_const_dim(&prop(3, "inner"), FoldDimPos(0)),
            Some(())
        );
        assert_eq!(fm.insert_data(7, &[FoldDimIndex(0)]), Some(()));
        assert_eq!(fm.build_map_dim(&prop(2, "outer"), FoldDimPos(0)), Some(()));
        assert_eq!(fm.num_dims(), 2);
        assert_eq!(fm.get_data(&[FoldDimIndex(0), FoldDimIndex(0)]), Some(7));
        assert_eq!(fm.get_data(&[FoldDimIndex(1), FoldDimIndex(2)]), Some(7));
        assert_eq!(
            fm.insert_data(-1, &[FoldDimIndex(1), FoldDimIndex(0)]),
            Some(())
        );
        assert_eq!(fm.get_data(&[FoldDimIndex(0), FoldDimIndex(0)]), Some(7));
        assert_eq!(fm.get_data(&[FoldDimIndex(1), FoldDimIndex(1)]), Some(-1));
        assert_eq!(
            fm.all_data(&fixed(&[]), ScanInnerOuter(false)),
            Some(vec![7, -1]),
            "the Constant dimension contributes one coordinate, not three"
        );
        assert_eq!(
            fm.all_data_with_map_unrolled(&fixed(&[]), ScanInnerOuter(false)),
            Some(vec![7, -1, 7, -1, 7, -1]),
            "and unrolling puts its extent back"
        );
    }

    /// `c6.before = 55`, `c6.append = OK`, `c6.dims = 2`, `c6.after = 0`, `c6.mid_before = 66`,
    /// `c6.mid_insert = OK`, `c6.mid_dims = 3`, `c6.mid_after = 66`, `c6.mid_space = [2,2,3]`.
    ///
    /// ⛔ WHERE A DIMENSION IS ADDED DECIDES WHETHER THE DATA SURVIVES. A level built at the position
    /// of the current leaf REPLACES that leaf, so its payload is gone (`:1420-1441`); a level built
    /// anywhere above it wraps what is there and `copySubTree`s the payload across. Measured: 55
    /// becomes 0, 66 stays 66.
    #[test]
    fn e026_appending_a_dimension_at_the_leaf_discards_the_data_and_inserting_one_above_keeps_it() {
        let mut fm = FoldManager::<i64>::new();
        assert_eq!(
            fm.build_const_dim(&prop(2, "outer"), FoldDimPos(0)),
            Some(())
        );
        assert_eq!(fm.insert_data(55, &[FoldDimIndex(0)]), Some(()));
        assert_eq!(fm.get_data(&[FoldDimIndex(1)]), Some(55));
        assert_eq!(
            fm.build_const_dim(&prop(3, "inner"), FoldDimPos(1)),
            Some(())
        );
        assert_eq!(fm.num_dims(), 2);
        assert_eq!(fm.get_data(&[FoldDimIndex(1), FoldDimIndex(2)]), Some(0));

        let mut fm = FoldManager::<i64>::new();
        assert_eq!(fm.build_const_dim(&prop(2, "a"), FoldDimPos(0)), Some(()));
        assert_eq!(fm.build_const_dim(&prop(3, "b"), FoldDimPos(1)), Some(()));
        assert_eq!(
            fm.insert_data(66, &[FoldDimIndex(0), FoldDimIndex(0)]),
            Some(())
        );
        assert_eq!(fm.get_data(&[FoldDimIndex(1), FoldDimIndex(2)]), Some(66));
        assert_eq!(fm.build_const_dim(&prop(2, "c"), FoldDimPos(1)), Some(()));
        assert_eq!(fm.num_dims(), 3);
        assert_eq!(
            fm.get_data(&[FoldDimIndex(1), FoldDimIndex(1), FoldDimIndex(2)]),
            Some(66)
        );
        assert_eq!(
            fm.fold_space_size(),
            vec![FoldDimSize(2), FoldDimSize(2), FoldDimSize(3)]
        );
    }

    /// `c7.build = OK`, `c7.all_data = [1,2]`, `c7.rebuild_last_to_map = 1`, `c7.after_types = 11`,
    /// `c7.after_all_data = [0,0,0,0,0,0]`, `c7.rebuild_out_of_range = 0`,
    /// `c7.rebuild_first_to_const = 1`, `c7.after_first_types = 1`,
    /// `c7.after_first_all_data = [0,0,0]`, `c7.rebuild_zero_dims = THROW`.
    ///
    /// ⛔ THE TWO FAILURES ARE DIFFERENT ANSWERS, WHICH IS WHY THIS RETURNS [`Option<bool>`]:
    /// `rebuildDim(2, ...)` on a two-dimension manager answers FALSE, while `rebuildDim(0, ...)` on a
    /// zero-dimension one THROWS — its range check is `pos <= dim_prop_.size() - 1` on an unsigned
    /// size, so at zero dimensions the bound is `SIZE_MAX` and the `.at(pos)` after it is what fails.
    /// ⛔ AND A REBUILT LEVEL LOSES ITS DATA whenever the two kinds do not pair: Constant to Map
    /// leaves the new map leaves at their defaults, measured as six zeroes where there were two
    /// values.
    #[test]
    fn e026_rebuilding_one_level_answers_false_out_of_range_and_throws_at_zero_dimensions() {
        let mut fm = FoldManager::<i64>::new();
        assert_eq!(
            fm.build_fold_space(
                &[prop(2, "a"), prop(3, "b")],
                &[BaseFuncType::Map, BaseFuncType::Constant]
            ),
            Some(())
        );
        for i in 0..2i64 {
            assert_eq!(
                fm.insert_data(i + 1, &[FoldDimIndex(i), FoldDimIndex(0)]),
                Some(())
            );
        }
        assert_eq!(
            fm.all_data(&fixed(&[]), ScanInnerOuter(false)),
            Some(vec![1, 2])
        );

        assert_eq!(
            fm.rebuild_dim(FoldDimPos(-1), BaseFuncType::Map),
            Some(true)
        );
        assert_eq!(fm.func_types(), vec![BaseFuncType::Map, BaseFuncType::Map]);
        assert_eq!(
            fm.all_data(&fixed(&[]), ScanInnerOuter(false)),
            Some(vec![0, 0, 0, 0, 0, 0])
        );
        assert_eq!(
            fm.rebuild_dim(FoldDimPos(2), BaseFuncType::Map),
            Some(false),
            "out of range is a refusal"
        );
        assert_eq!(
            fm.rebuild_dim(FoldDimPos(0), BaseFuncType::Constant),
            Some(true)
        );
        assert_eq!(
            fm.func_types(),
            vec![BaseFuncType::Constant, BaseFuncType::Map]
        );
        assert_eq!(
            fm.all_data(&fixed(&[]), ScanInnerOuter(false)),
            Some(vec![0, 0, 0])
        );
        assert_eq!(
            FoldManager::<i64>::new().rebuild_dim(FoldDimPos(0), BaseFuncType::Map),
            None,
            "zero dimensions is the throw, not the refusal"
        );
    }

    /// `c8.label_ignored = 1`, `c8.data_compared = 0`, `c8.data_equal = 1`, `c8.size_compared = 0`,
    /// `c8.copy_ctor = 1`, `c8.copy_data = 4`, `d.copy_equal = 1`,
    /// `d.copy_all_data = [0,10,1,11,2,12]`, `d.independent = 0`, `d.copy_after = 99`.
    ///
    /// ⛔ EQUALITY IGNORES THE LABELS. Two managers whose dimensions are named "a" and
    /// "DIFFERENT LABEL" compare EQUAL, and `FoldDimProp`'s own `operator==` — which DOES compare the
    /// label (`:143-145`) — is never reached. What a manager compares is extents, kinds and payloads.
    #[test]
    fn e026_equality_compares_extents_kinds_and_payloads_and_never_the_labels() {
        let mut lhs = FoldManager::<i64>::new();
        let mut rhs = FoldManager::<i64>::new();
        assert_eq!(lhs.build_const_dim(&prop(2, "a"), FoldDimPos(0)), Some(()));
        assert_eq!(
            rhs.build_const_dim(&prop(2, "DIFFERENT LABEL"), FoldDimPos(0)),
            Some(())
        );
        assert_eq!(lhs, rhs);
        assert_eq!(lhs.insert_data(4, &[FoldDimIndex(0)]), Some(()));
        assert_ne!(lhs, rhs, "the payload is compared");
        assert_eq!(rhs.insert_data(4, &[FoldDimIndex(0)]), Some(()));
        assert_eq!(lhs, rhs);
        let mut wider = FoldManager::<i64>::new();
        assert_eq!(
            wider.build_const_dim(&prop(3, "b"), FoldDimPos(0)),
            Some(())
        );
        assert_ne!(lhs, wider, "and so is the extent");

        let cloned = lhs.clone();
        assert_eq!(cloned, lhs);
        assert_eq!(cloned.get_data(&[FoldDimIndex(1)]), Some(4));

        let mut fm = FoldManager::<i64>::new();
        assert_eq!(
            fm.build_all_map_fold_space(&[prop(2, "o"), prop(3, "i")]),
            Some(())
        );
        for x in 0..2i64 {
            for y in 0..3i64 {
                assert_eq!(
                    fm.insert_data(x * 10 + y, &[FoldDimIndex(x), FoldDimIndex(y)]),
                    Some(())
                );
            }
        }
        let mut copied = fm.clone();
        assert_eq!(copied, fm);
        assert_eq!(
            copied.all_data(&fixed(&[]), ScanInnerOuter(false)),
            Some(vec![0, 10, 1, 11, 2, 12])
        );
        assert_eq!(
            copied.insert_data(99, &[FoldDimIndex(0), FoldDimIndex(0)]),
            Some(())
        );
        assert_eq!(
            fm.get_data(&[FoldDimIndex(0), FoldDimIndex(0)]),
            Some(0),
            "a deep copy"
        );
        assert_eq!(
            copied.get_data(&[FoldDimIndex(0), FoldDimIndex(0)]),
            Some(99)
        );
    }

    /// `c8.clone_ignoring = OK`, `c8.clone_ignoring_dims = 1`, `c8.clone_ignoring_data = 2`,
    /// `c8.clone_ignoring_space = [3]`, `a.clone_ignoring_map = 1`,
    /// `a.clone_ignoring_map_alpha = 5`, `a.clone_ignoring_map_data = 11`,
    /// `a.clone_ignoring_affine = 1`, `a.clone_ignoring_affine_data = 0`.
    ///
    /// ⛔ THE COPY PAIRS NODES BY POSITION IN A BREADTH-FIRST LIST, NOT BY SHAPE, and dropping a
    /// level shifts every pair after it. Ignoring the Map level of a Map-over-Affine manager keeps
    /// the affine level's alpha (the surviving `Affine_Leaf` still lands opposite one); ignoring the
    /// AFFINE level pairs the surviving `Map_NonLeaf` against an `Affine_NonLeaf`, the per-kind copy
    /// declines, and the data is SILENTLY the default. `a.clone_ignoring_affine_data = 0` is that
    /// silence, measured.
    #[test]
    fn e026_a_clone_that_drops_a_level_shifts_every_pair_after_it_and_the_copy_declines_quietly() {
        let mut two = FoldManager::<i64>::new();
        assert_eq!(two.build_const_dim(&prop(3, "i"), FoldDimPos(0)), Some(()));
        assert_eq!(two.build_map_dim(&prop(2, "o"), FoldDimPos(0)), Some(()));
        for k in 0..2i64 {
            assert_eq!(
                two.insert_data(k + 1, &[FoldDimIndex(k), FoldDimIndex(0)]),
                Some(())
            );
        }
        let mut sub = FoldManager::<i64>::new();
        assert_eq!(sub.clone_ignoring(&two, &fixed(&[(0, 1)])), Some(()));
        assert_eq!(sub.num_dims(), 1);
        assert_eq!(
            sub.get_data(&[FoldDimIndex(0)]),
            Some(2),
            "map child 1's subtree"
        );
        assert_eq!(sub.fold_space_size(), vec![FoldDimSize(3)]);

        let mut two = FoldManager::<i64>::new();
        assert_eq!(two.build_affine_dim(&prop(3, "i"), FoldDimPos(0)), Some(()));
        assert_eq!(two.build_map_dim(&prop(2, "o"), FoldDimPos(0)), Some(()));
        assert_eq!(two.insert_alpha_beta(&5, &1, FoldDimPos(1)), Some(()));
        let mut sub = FoldManager::<i64>::new();
        assert_eq!(sub.clone_ignoring(&two, &fixed(&[(0, 1)])), Some(()));
        assert_eq!(sub.num_dims(), 1);
        assert_eq!(sub.alpha(FoldDimPos(0)), Some(&5));
        assert_eq!(sub.get_data(&[FoldDimIndex(2)]), Some(11));

        let mut sub = FoldManager::<i64>::new();
        assert_eq!(sub.clone_ignoring(&two, &fixed(&[(1, 1)])), Some(()));
        assert_eq!(sub.num_dims(), 1);
        assert_eq!(
            sub.get_data(&[FoldDimIndex(1)]),
            Some(0),
            "the surviving Map level was paired with an Affine one, so nothing was copied"
        );
    }

    /// `c9.coords = [(0 0 0),(1 0 0),(0 0 1),(1 0 1)]`,
    /// `c9.coords_fixed = [(0 2 0),(1 2 0),(0 2 1),(1 2 1)]`,
    /// `c9.coords_expand = [(0 0 0),(1 0 0),(0 1 0),(1 1 0),(0 2 0),(1 2 0),(0 0 1),(1 0 1),(0 1 1),(1 1 1),(0 2 1),(1 2 1)]`,
    /// `c9.coords_expand_inner = [(0 0 0),(0 0 1),(0 1 0),(0 1 1),(0 2 0),(0 2 1),(1 0 0),(1 0 1),(1 1 0),(1 1 1),(1 2 0),(1 2 1)]`,
    /// `c9.space = [2,3,2]`, `c9.all_data_len = 4`, `c9.unrolled_len = 12`,
    /// `c9.func_type_out_of_range = THROW`, `c9.dim_prop_label = b`,
    /// `c9.build_mismatched_sizes = THROW`.
    ///
    /// ⛔ `expand_if_any_map` EXPANDS THE CONSTANT DIMENSIONS, not the map ones. Its test is whether
    /// SOME unfixed Map dimension is wider than one (`:1990-1998`); if one is, EVERY dimension
    /// contributes its full extent, which is how the constant middle dimension here goes from one
    /// coordinate to three. A fold space with no Map dimension at all ignores the flag entirely.
    #[test]
    fn e026_a_constant_dimension_collapses_the_walk_and_a_map_dimension_anywhere_expands_it_back() {
        let mut fm = FoldManager::<i64>::new();
        let props = [prop(2, "a"), prop(3, "b"), prop(2, "c")];
        let types = [BaseFuncType::Map, BaseFuncType::Constant, BaseFuncType::Map];
        assert_eq!(fm.build_fold_space(&props, &types), Some(()));
        assert_eq!(
            coords(
                &fm.flattened_coordinates(
                    &fixed(&[]),
                    ScanInnerOuter(false),
                    ExpandIfAnyMap(false)
                )
                .unwrap()
            ),
            "[(0 0 0),(1 0 0),(0 0 1),(1 0 1)]"
        );
        assert_eq!(
            coords(
                &fm.flattened_coordinates(
                    &fixed(&[(1, 2)]),
                    ScanInnerOuter(false),
                    ExpandIfAnyMap(false)
                )
                .unwrap()
            ),
            "[(0 2 0),(1 2 0),(0 2 1),(1 2 1)]"
        );
        assert_eq!(
            coords(
                &fm.flattened_coordinates(&fixed(&[]), ScanInnerOuter(false), ExpandIfAnyMap(true))
                    .unwrap()
            ),
            "[(0 0 0),(1 0 0),(0 1 0),(1 1 0),(0 2 0),(1 2 0),(0 0 1),(1 0 1),(0 1 1),(1 1 1),(0 2 1),(1 2 1)]"
        );
        assert_eq!(
            coords(
                &fm.flattened_coordinates(&fixed(&[]), ScanInnerOuter(true), ExpandIfAnyMap(true))
                    .unwrap()
            ),
            "[(0 0 0),(0 0 1),(0 1 0),(0 1 1),(0 2 0),(0 2 1),(1 0 0),(1 0 1),(1 1 0),(1 1 1),(1 2 0),(1 2 1)]"
        );
        assert_eq!(
            fm.fold_space_size(),
            vec![FoldDimSize(2), FoldDimSize(3), FoldDimSize(2)]
        );
        assert_eq!(
            fm.all_data(&fixed(&[]), ScanInnerOuter(false))
                .map(|d| d.len()),
            Some(4)
        );
        assert_eq!(
            fm.all_data_with_map_unrolled(&fixed(&[]), ScanInnerOuter(false))
                .map(|d| d.len()),
            Some(12)
        );
        assert_eq!(fm.func_type(FoldDimPos(3)), None);
        assert_eq!(
            fm.fold_dim_prop(FoldDimPos(1)).map(FoldDimProp::label),
            Some("b")
        );
        assert_eq!(
            fm.build_fold_space(&props, &[BaseFuncType::Map]),
            None,
            "one kind for three dimensions"
        );
    }

    /// `c10.apply = OK`, `c10.applied = [30,40]`, `c10.apply_zero_dim = OK`,
    /// `c10.applied_zero_dim = 6`, `t.apply = OK`, `t.applied = [13,24]`,
    /// `t.const_meets_map = THROW`, `t.affine_lhs = THROW`, `t.shape_mismatch = THROW`,
    /// `t.const_meets_const = OK`, `t.const_applied = [6]`.
    ///
    /// ⛔ THE TWO-MANAGER GATE IS ABOUT **THIS** MANAGER'S KINDS AND IS ASYMMETRIC (`:1295-1305`): a
    /// Constant level here refuses a Map level there, because the one cell it has would be written
    /// once per coordinate of the other and only the last write would survive; an Affine level here is
    /// refused outright, because its value is COMPUTED and there is nowhere to write it. A Map level
    /// here accepts anything — including the Constant level that would have refused it.
    #[test]
    fn e026_apply_rewrites_every_value_and_the_two_manager_form_gates_on_this_managers_kinds() {
        let mut fm = FoldManager::<i64>::new();
        assert_eq!(fm.build_map_dim(&prop(2, "a"), FoldDimPos(0)), Some(()));
        assert_eq!(fm.insert_data(3, &[FoldDimIndex(0)]), Some(()));
        assert_eq!(fm.insert_data(4, &[FoldDimIndex(1)]), Some(()));
        assert_eq!(fm.apply(&fixed(&[]), |d| d * 10), Some(()));
        assert_eq!(
            fm.all_data(&fixed(&[]), ScanInnerOuter(false)),
            Some(vec![30, 40])
        );
        let mut zero = FoldManager::<i64>::with_data(5);
        assert_eq!(zero.apply(&fixed(&[]), |d| d + 1), Some(()));
        assert_eq!(zero.get_data(&[]), Some(6));

        let props = [prop(2, "a")];
        let mut lhs = FoldManager::<i64>::new();
        let mut rhs = FoldManager::<i64>::new();
        assert_eq!(lhs.build_all_map_fold_space(&props), Some(()));
        assert_eq!(rhs.build_all_map_fold_space(&props), Some(()));
        assert_eq!(lhs.insert_data(10, &[FoldDimIndex(0)]), Some(()));
        assert_eq!(lhs.insert_data(20, &[FoldDimIndex(1)]), Some(()));
        assert_eq!(rhs.insert_data(3, &[FoldDimIndex(0)]), Some(()));
        assert_eq!(rhs.insert_data(4, &[FoldDimIndex(1)]), Some(()));
        assert_eq!(lhs.apply_with(&rhs, &fixed(&[]), |l, r| l + r), Some(()));
        assert_eq!(
            lhs.all_data(&fixed(&[]), ScanInnerOuter(false)),
            Some(vec![13, 24])
        );

        let mut constant = FoldManager::<i64>::new();
        assert_eq!(constant.build_all_constant_fold_space(&props), Some(()));
        assert_eq!(
            constant.apply_with(&rhs, &fixed(&[]), |l, r| l + r),
            None,
            "a Constant level here may not meet a Map level there"
        );
        let mut affine = FoldManager::<i64>::new();
        assert_eq!(
            affine.build_affine_dim(&prop(2, "a"), FoldDimPos(0)),
            Some(())
        );
        assert_eq!(
            affine.apply_with(&rhs, &fixed(&[]), |l, r| l + r),
            None,
            "and an Affine level here meets nothing at all"
        );
        let mut wide = FoldManager::<i64>::new();
        assert_eq!(wide.build_all_map_fold_space(&[prop(5, "wide")]), Some(()));
        assert_eq!(lhs.apply_with(&wide, &fixed(&[]), |l, r| l + r), None);

        let mut other = FoldManager::<i64>::new();
        assert_eq!(other.build_all_constant_fold_space(&props), Some(()));
        assert_eq!(other.insert_data(6, &[FoldDimIndex(0)]), Some(()));
        assert_eq!(
            constant.apply_with(&other, &fixed(&[]), |l, r| l + r),
            Some(())
        );
        assert_eq!(
            constant.all_data(&fixed(&[]), ScanInnerOuter(false)),
            Some(vec![6]),
            "the refused attempt left it untouched"
        );
    }

    /// `c11.unknown = THROW`, `c11.unknown_nonleaf = THROW`, `c11.wksplit_string = WkSplit`,
    /// `c11.unknown_string = THROW`, `c11.build_bad_pos = THROW`.
    ///
    /// ⛔ `Unknown` IS A SENTINEL WITH NO NAME AND NO FOLD FUNCTION. It is not in
    /// `baseFuncTypeToString` (`:46-50`), so `printMetaData` throws on a level that carries it, and
    /// neither `createLeafFunc` nor `createNonLeafFunc` has an arm for it. The fifth enumerator exists
    /// to be compared against, never to be built from.
    #[test]
    fn e026_the_unknown_kind_has_no_name_and_no_builder_and_a_position_past_the_end_is_refused() {
        let mut fm = FoldManager::<i64>::new();
        assert_eq!(
            fm.build_dim(&prop(2, "a"), BaseFuncType::Unknown, FoldDimPos(0)),
            None
        );
        assert_eq!(fm.num_dims(), 0);
        assert_eq!(fm.build_const_dim(&prop(2, "a"), FoldDimPos(0)), Some(()));
        assert_eq!(
            fm.build_dim(&prop(2, "a"), BaseFuncType::Unknown, FoldDimPos(0)),
            None
        );
        assert_eq!(BaseFuncType::WkSplit.as_str(), Some("WkSplit"));
        assert_eq!(BaseFuncType::Unknown.as_str(), None);
        assert_eq!(
            fm.build_dim(&prop(2, "a"), BaseFuncType::Constant, FoldDimPos(5)),
            None,
            "a position past the end is not an append"
        );
    }

    /// `c12.build = OK`, `c12.dims = 1`, `c12.empty_data = 0`, `c12.data = [1,2,3]`,
    /// `c12.all_data_len = 1`, `c12.build_empty_props = THROW`.
    ///
    /// ⛔ A PAYLOAD WITH NO ARITHMETIC IS A REAL INSTANTIATION of this class in the authority
    /// (`FoldManager<std::vector<int64_t>>`, `dsc/dsc2.h`), and it is the reason the affine
    /// arithmetic cannot be a bound on `Dtype`: five of the six fold functions never touch the
    /// payload. Here it is the payload with no `build_affine_dim` at all.
    #[test]
    fn e026_a_payload_with_no_arithmetic_still_has_a_whole_constant_fold_space() {
        let mut fm = FoldManager::<Vec<i64>>::new();
        assert_eq!(fm.build_all_constant_fold_space(&[prop(2, "a")]), Some(()));
        assert_eq!(fm.num_dims(), 1);
        assert_eq!(fm.get_data(&[FoldDimIndex(0)]), Some(Vec::new()));
        assert_eq!(fm.insert_data(vec![1, 2, 3], &[FoldDimIndex(1)]), Some(()));
        assert_eq!(fm.get_data(&[FoldDimIndex(0)]), Some(vec![1, 2, 3]));
        assert_eq!(
            fm.all_data(&fixed(&[]), ScanInnerOuter(false))
                .map(|d| d.len()),
            Some(1)
        );
        assert_eq!(
            fm.build_all_constant_fold_space(&[]),
            None,
            "an empty dimension list"
        );
    }

    /// `o.fold_dim_prop_oor = THROW`, `o.fold_dim_size_oor = THROW`, `o.insert_wrong_arity = THROW`,
    /// `o.get_wrong_arity = THROW`, `o.alpha_oor = THROW`, `o.insert_alpha_nonaffine = THROW`,
    /// `o.dim_prop_len = 1`, `o.getFuncType_neg = THROW`, `n.legal_negative = 1`,
    /// `n.get_negative = THROW`, `g.from_1_len = 2`, `g.from_1_first = 3`, `g.from_past_end = 0`,
    /// `g.from_negative = THROW`.
    ///
    /// ⛔ `-1` IS THE LAST DIMENSION TO SOME OF THESE MEMBERS AND A HUGE UNSIGNED INDEX TO THE
    /// OTHERS. `getBeta(-1)` answers the innermost level's beta while `getFuncType(-1)` throws, on the
    /// same manager, because only the first group runs `if (pos < 0) pos = size + pos` first. The two
    /// conventions are [`FoldDimPos::resolve`] and [`FoldDimPos::index`], and the split is not
    /// cosmetic — see the sibling unit test.
    /// ⛔ AND `getAllDimProFromPos` PAST THE END IS AN EMPTY LIST, NOT A THROW (`:2259-2263`): its own
    /// `DT_CHECK` is only `pos >= 0`, and the loop from a `pos` above the size simply never runs.
    #[test]
    fn e026_a_position_past_the_end_is_a_throw_and_a_negative_one_is_only_sometimes_the_last() {
        let mut fm = FoldManager::<i64>::new();
        assert_eq!(fm.build_const_dim(&prop(2, "a"), FoldDimPos(0)), Some(()));
        assert_eq!(fm.fold_dim_prop(FoldDimPos(3)), None);
        assert_eq!(fm.fold_dim_size(FoldDimPos(3)), None);
        assert_eq!(fm.insert_data(1, &[FoldDimIndex(0), FoldDimIndex(0)]), None);
        assert_eq!(fm.get_data(&[]), None);
        assert_eq!(fm.alpha(FoldDimPos(3)), None);
        assert_eq!(
            fm.insert_alpha(&1, FoldDimPos(0)),
            None,
            "the level is not Affine"
        );
        assert_eq!(fm.dim_prop().len(), 1);
        assert_eq!(
            fm.func_type(FoldDimPos(-1)),
            None,
            "`getFuncType` indexes unsigned"
        );

        let mut fm = FoldManager::<i64>::new();
        assert_eq!(fm.build_map_dim(&prop(3, "a"), FoldDimPos(0)), Some(()));
        assert_eq!(fm.is_legal(&[FoldDimIndex(-1)]), Some(()));
        assert_eq!(
            fm.get_data(&[FoldDimIndex(-1)]),
            None,
            "`isLegal` passes it and the map leaf is what refuses it"
        );

        let mut fm = FoldManager::<i64>::new();
        assert_eq!(
            fm.build_all_constant_fold_space(&[prop(2, "a"), prop(3, "b"), prop(4, "c")]),
            Some(())
        );
        let from_1 = fm.dim_prop_from(FoldDimPos(1)).unwrap();
        assert_eq!(from_1.len(), 2);
        assert_eq!(from_1[0].0.size(), FoldDimSize(3));
        assert_eq!(
            fm.dim_prop_from(FoldDimPos(9)).map(|dims| dims.len()),
            Some(0),
            "past the end is empty, not a throw"
        );
        assert_eq!(fm.dim_prop_from(FoldDimPos(-1)), None);
    }

    /// `s.assign = OK`, `s.assign_type = 1`, `s.assign_data = 8`, `s.assign_label_kept = a`,
    /// `s.assign_size_mismatch = THROW`, `s.assign_zero_dims = OK`, `s.assign_zero_data = 9`.
    ///
    /// ⛔ `operator=` KEEPS THIS MANAGER'S OWN `FoldDimProp`s AND TAKES RHS'S TREE (`:930-932`): a
    /// dimension labelled "a" is still labelled "a" afterwards, though its fold function is now
    /// rhs's Map and its data is rhs's 8. The labels are this manager's identity and the tree is the
    /// value, which is why this is `assign` and not Rust's `=` — that one is the authority's
    /// DEFAULTED MOVE assignment (`:912`) and replaces everything.
    /// ⛔ AND IT REFUSES A DIFFERENT SHAPE outright (`:923-928`, `:933-940`), so it is an in-place
    /// overwrite of a value that already has this shape rather than a replacement.
    #[test]
    fn e026_assignment_takes_the_tree_and_the_data_but_keeps_this_managers_own_labels() {
        let mut lhs = FoldManager::<i64>::new();
        let mut rhs = FoldManager::<i64>::new();
        assert_eq!(lhs.build_const_dim(&prop(2, "a"), FoldDimPos(0)), Some(()));
        assert_eq!(
            rhs.build_map_dim(&prop(2, "b_other_label"), FoldDimPos(0)),
            Some(())
        );
        assert_eq!(rhs.insert_data(8, &[FoldDimIndex(1)]), Some(()));
        assert_eq!(lhs.assign(&rhs), Some(()));
        assert_eq!(lhs.func_type(FoldDimPos(0)), Some(BaseFuncType::Map));
        assert_eq!(lhs.get_data(&[FoldDimIndex(1)]), Some(8));
        assert_eq!(
            lhs.fold_dim_prop(FoldDimPos(0)).map(FoldDimProp::label),
            Some("a")
        );

        let mut wider = FoldManager::<i64>::new();
        assert_eq!(
            wider.build_const_dim(&prop(3, "c"), FoldDimPos(0)),
            Some(())
        );
        assert_eq!(lhs.assign(&wider), None, "a different extent is refused");

        let mut zero = FoldManager::<i64>::with_data(4);
        assert_eq!(zero.assign(&FoldManager::<i64>::with_data(9)), Some(()));
        assert_eq!(zero.get_data(&[]), Some(9));
    }

    /// `l.collect_0 = OK`, `l.collect_0_len = 1`, `l.collect_1 = OK`, `l.collect_1_len = 2`,
    /// `l.collect_oor = THROW`.
    ///
    /// ⛔ LEVEL `0` IS THE ROOT ALONE, and the count at a level is the product of the extents of the
    /// MAP dimensions above it — one Map dimension of two above level 1 makes two nodes there. That
    /// count is what `getAlpha` refuses to read from and what `insertAlpha` writes all of.
    #[test]
    fn e026_a_level_holds_one_node_per_coordinate_of_the_map_dimensions_above_it() {
        let mut fm = FoldManager::<i64>::new();
        assert_eq!(fm.build_const_dim(&prop(3, "i"), FoldDimPos(0)), Some(()));
        assert_eq!(fm.build_map_dim(&prop(2, "o"), FoldDimPos(0)), Some(()));
        assert_eq!(fm.collect_at_level(0).map(|ffs| ffs.len()), Some(1));
        assert_eq!(fm.collect_at_level(1).map(|ffs| ffs.len()), Some(2));
        assert_eq!(fm.collect_at_level(2).map(|ffs| ffs.len()), None);
    }

    /// `p.default`, `p.compressed` and `p.zero_dim`, transcribed whole from the probe's
    /// `std::ostringstream`.
    ///
    /// ⛔ THE PREFIX GOES ON THE TWO ARRAY LINES AND, THROUGH `ps2`, ON THE ENTRIES — and an Affine
    /// entry's own `printMetaData` is called with the EMPTY prefix (`:2205`), which is why
    /// `"alpha_"` and `"beta_"` sit on one line while the entries are indented.
    /// ⛔ A MAP OR CONSTANT LEVEL PRINTS NO METADATA AT ALL (`:2197-2200`) — `{ "Map" : {} }`, with
    /// the empty braces standing in for the fields it has none of.
    /// ⛔ AND `compressed` DROPS THE WHOLE `dim_prop_attr` ARRAY, extents and labels together, so a
    /// compressed dump cannot be read back into a fold space.
    #[test]
    fn e026_the_meta_data_dump_is_two_json_arrays_and_the_compressed_form_keeps_only_the_first() {
        let mut fm = FoldManager::<i64>::new();
        assert_eq!(
            fm.build_fold_space(
                &[prop(2, "a"), prop(3, "b")],
                &[BaseFuncType::Map, BaseFuncType::Affine]
            ),
            Some(())
        );
        assert_eq!(fm.insert_alpha_beta(&4, &5, FoldDimPos(1)), Some(()));

        let mut out = String::new();
        assert_eq!(
            fm.print_meta_data(&mut out, "", AddComma(true), Compressed(false)),
            Some(())
        );
        assert_eq!(
            out,
            concat!(
                "\"dim_prop_func\" : [\n",
                "  { \"Map\" : {} },\n",
                "  { \"Affine\" : {\"alpha_\" : 4, \"beta_\" : 5} }\n",
                "],\n",
                "\"dim_prop_attr\" : [\n",
                "  { \"factor_\" : 2, \"label_\" : \"a\" },\n",
                "  { \"factor_\" : 3, \"label_\" : \"b\" }\n",
                "],\n",
            )
        );

        let mut out = String::new();
        assert_eq!(
            fm.print_meta_data(&mut out, "__", AddComma(false), Compressed(true)),
            Some(())
        );
        assert_eq!(
            out,
            concat!(
                "__\"dim_prop_func\" : [\n",
                "__  { \"Map\" : {} },\n",
                "__  { \"Affine\" : {\"alpha_\" : 4, \"beta_\" : 5} }\n",
                "__]\n",
            )
        );

        let mut out = String::new();
        assert_eq!(
            FoldManager::<i64>::with_data(3).print_meta_data(
                &mut out,
                "",
                AddComma(true),
                Compressed(false)
            ),
            Some(())
        );
        assert_eq!(
            out,
            concat!(
                "\"dim_prop_func\" : [\n",
                "],\n",
                "\"dim_prop_attr\" : [\n",
                "],\n",
            ),
            "a zero-dimension manager prints two empty arrays and not its value"
        );
    }

    /// `q.content`, `q.no_content`, `q.zero` and `q.vector`, transcribed line for line from the
    /// probe's `ostringstream`.
    ///
    /// ⛔ THE OPENING BRACE IS NOT PREFIXED AND THE CLOSING ONE IS (`:2737` against `:2747`), so a
    /// dump nested under a non-empty `ps` is not indented consistently with itself: with `ps = "__"`
    /// the first line is `" {"` and the last is `"__}"`.
    /// ⛔ AND A ZERO-DIMENSION MANAGER PRINTS `"42"` — no braces, no metadata, `ps` unused
    /// (`:2730-2731`). A caller splicing this into a larger object gets a bare string where every
    /// other shape gives it an object.
    ///
    /// `q.vector` is the measured shape of the arm this port makes unreachable: a
    /// `FoldManager<std::vector<int64_t>>` prints `"[0]" :["1", "2", "3"],` and `"[1]" :[]` from
    /// `printData`'s vector overload (`:2700-2707`). [`FoldManager::print_data`]'s [`fmt::Display`]
    /// bound is what excludes it, and `Vec<i64>` — the one list payload in scope — is not
    /// [`fmt::Display`], so no in-scope caller can ask for that output.
    #[test]
    fn e026_the_dump_prefixes_its_closing_brace_but_not_its_opening_one() {
        let mut fm = FoldManager::<i64>::new();
        assert_eq!(
            fm.build_fold_space(
                &[prop(2, "a"), prop(3, "b")],
                &[BaseFuncType::Map, BaseFuncType::Constant]
            ),
            Some(())
        );
        for i in 0..2 {
            assert_eq!(
                fm.insert_data(i * 10 + 1, &[FoldDimIndex(i), FoldDimIndex(0)]),
                Some(())
            );
        }

        let mut out = String::new();
        assert_eq!(fm.print(&mut out, "", PrintContent(true)), Some(()));
        assert_eq!(
            out,
            concat!(
                " {\n",
                "  \"dim_prop_func\" : [\n",
                "    { \"Map\" : {} },\n",
                "    { \"Const\" : {} }\n",
                "  ],\n",
                "  \"dim_prop_attr\" : [\n",
                "    { \"factor_\" : 2, \"label_\" : \"a\" },\n",
                "    { \"factor_\" : 3, \"label_\" : \"b\" }\n",
                "  ],\n",
                "  \"data_\" : {\n",
                "    \"[0, 0]\" :\"1\",\n",
                "    \"[1, 0]\" :\"11\"\n",
                "  }\n",
                "}",
            ),
            "q.content"
        );

        let mut out = String::new();
        assert_eq!(fm.print(&mut out, "__", PrintContent(false)), Some(()));
        assert_eq!(
            out,
            concat!(
                " {\n",
                "__  \"dim_prop_func\" : [\n",
                "__    { \"Map\" : {} },\n",
                "__    { \"Const\" : {} }\n",
                "__  ],\n",
                "__  \"dim_prop_attr\" : [\n",
                "__    { \"factor_\" : 2, \"label_\" : \"a\" },\n",
                "__    { \"factor_\" : 3, \"label_\" : \"b\" }\n",
                "__  ]\n",
                "__}",
            ),
            "q.no_content — one flag drops the values AND the comma before them"
        );

        let mut out = String::new();
        assert_eq!(
            FoldManager::<i64>::with_data(42).print(&mut out, "  ", PrintContent(true)),
            Some(())
        );
        assert_eq!(out, "\"42\"", "q.zero");
    }
}
