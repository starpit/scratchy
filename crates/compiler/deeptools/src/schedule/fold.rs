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
/// here, and the choice is e020's. There is no eighth: `createNonLeafFunc` cannot build a WkSplit
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
}

// crustify:todo: e020_FoldManager

// crustify:todo: e020_FoldManager.all_data

// crustify:todo: e020_FoldManager.areAllConstant

// crustify:todo: e020_FoldManager.base_func_types

// crustify:todo: e020_FoldManager.break

// crustify:todo: e020_FoldManager.can_copy

// crustify:todo: e020_FoldManager.const_ffs_at_pos

// crustify:todo: e020_FoldManager.continue

// crustify:todo: e020_FoldManager.coord_to_data

// crustify:todo: e020_FoldManager.coordinates

// crustify:todo: e020_FoldManager.curr_leaf_pos

// crustify:todo: e020_FoldManager.data_and_coord

// crustify:todo: e020_FoldManager.dim_prop_sub_tree

// crustify:todo: e020_FoldManager.do_expand

// crustify:todo: e020_FoldManager.endl

// crustify:todo: e020_FoldManager.ff

// crustify:todo: e020_FoldManager.ffs_at_last_non_leaf_pos

// crustify:todo: e020_FoldManager.ffs_at_pos

// crustify:todo: e020_FoldManager.ffs_at_pre_leaf_pos

// crustify:todo: e020_FoldManager.fifo

// crustify:todo: e020_FoldManager.fold_dim_props

// crustify:todo: e020_FoldManager.l_so2

// crustify:todo: e020_FoldManager.last_non_leaf_pos

// crustify:todo: e020_FoldManager.linear_ff_list

// crustify:todo: e020_FoldManager.linear_ff_list_rhs

// crustify:todo: e020_FoldManager.linear_ff_list_this

// crustify:todo: e020_FoldManager.need_to_rebuild

// crustify:todo: e020_FoldManager.new_dim_prop

// crustify:todo: e020_FoldManager.next_pos

// crustify:todo: e020_FoldManager.old_parent

// crustify:todo: e020_FoldManager.parent_func_

// crustify:todo: e020_FoldManager.pos

// crustify:todo: e020_FoldManager.posOfAffineFolds

// crustify:todo: e020_FoldManager.pre_leaf_pos

// crustify:todo: e020_FoldManager.ps2

// crustify:todo: e020_FoldManager.reduced_coord

// crustify:todo: e020_FoldManager.reduced_coord_to_data

// crustify:todo: e020_FoldManager.repeat_factor

// crustify:todo: e020_FoldManager.time

// crustify:todo: e020_FoldManager.total_count

// crustify:todo: e020_FoldManager.unique_data_coords

// crustify:todo: e020_FoldManager.wasCompressed

// crustify:todo: e020_FoldManager.zero_fold_coord
