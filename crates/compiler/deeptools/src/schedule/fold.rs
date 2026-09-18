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
    /// vector (`util/foldManager/wkDivisionParams.h:250-284`), so an empty result would mean neither
    /// a gap nor an absence.
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
