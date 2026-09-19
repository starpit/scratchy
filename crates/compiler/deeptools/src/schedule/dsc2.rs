//! Re-ported from the C++ authority. See crustify-scheduler/AGENT-BRIEF.md.

use crate::schedule::ddc::LatchDataId;
use crate::schedule::dims::{
    DataStructDims, PadType, PaddingFormType, PrimaryDimAndKind, PrimaryDimTypes,
};
use crate::schedule::fold::{
    BaseFuncType, FoldDimIndex, FoldDimPos, FoldDimProp, FoldDimSize, FoldFunc, FoldManager,
    PrintContent,
};
use core::num::{NonZeroUsize, Wrapping};
use std::collections::btree_map::Entry;
use std::collections::{BTreeMap, BTreeSet};
use sys_arch_spec::arch_enums::{DataLocation, SenComponent};
use sys_arch_spec::fields::Gen;
use sys_arch_spec::{CoreId, CoreletId, SFP_SLICES};

/// A group tag register's group id (`dsc/dsc2.h:35`). `DesignSpaceConfig::gtrIdsUsed_`
/// (`dsc/designSpaceConfig.h:116`, a `std::set<int>`) holds the set of them, and only a PRESENT id
/// is inserted (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:4833-4834`, `:5257-5258`).
///
/// ⭐ NEVER THE OUT-OF-RANGE SEED. `getSharesAndGroupName` starts its group name at
/// `sysDef.maxGroupID + 1` (`:4705`), one past the last legal id, but that seed survives only for a
/// single sharer — the very case the producer answers -1 for — and on the shared path the name is
/// `DT_CHECK`ed `<= maxGroupID` (`:4710-4711`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GroupId(pub u32);

/// How many cores share one group tag register (`dsc/dsc2.h:36`).
///
/// ⭐ AT LEAST ONE WHENEVER PRODUCED: both writers assign `sharesCoreIds.size()` (`:4701`, used at
/// `:4828` and `:5252`) from a set `DT_CHECK_MSG`ed non-empty (`:4697`). [`None`] is reachable only
/// on an untouched record or a JSON `-1` (`dsc/dsc2.cpp:1596-1597`), never from the scheduler.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NumSharers(pub u32);

/// A fold level's coordinate stride per step — `CoordinateBaseType` (`dsc/dsc2.h:442`, `int64_t`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Alpha(pub i64);

/// A fold level's coordinate offset — `CoordinateBaseType` (`dsc/dsc2.h:442`, `int64_t`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Beta(pub i64);

/// A fold level's trip count — `int64_t` on the field itself (`dsc/dsc2.h:1083`).
///
/// ⛔ THE FOLD MANAGER'S COPY IS NARROWER, so this is not `getFoldDimSize`'s currency: a cardinality
/// reaches a fold through `CoordinateType::addFold`'s `int foldCardinality` (`dsc/dsc2.h:120-123`),
/// is stored in `FoldDimProp::factor_` as `uint32_t` (`util/foldManager/foldInfrastructure.h:153`)
/// and comes back out as `int` (`:2631-2633`).
///
/// ⛔ AND 0 IS PRODUCED, NOT MERELY DEFAULTED — "Parametric loops may have 0 iteration count"
/// (`dsc/dsc2.cpp:6251`) — and both readers spell it as the multiplicative identity 1: one skips it
/// in the inner-cardinality product (`:6249-6253`, `:6460-6464`), the other writes
/// `cardinality == 0 ? 1 : cardinality` (`:6359-6361`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Cardinality(pub i64);

/// A loop's element stride after distribution, read straight into `loopEleOffsets_`
/// (`ddc/ddcv1.cpp:2455-2460`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TemporalStride(pub i64);

/// An element-arrangement fold level, counted INNERMOST-FIRST as `currElemArrLevel`
/// (`ddc/ddc_fold.cpp:2666-2689`).
///
/// ⛔ NOT A FOLD INDEX, AND NOT OUTERMOST-FIRST. The authority states it in prose — "Element
/// arrangement at index 0 is the innermost and at index (size - 1) is the outermost"
/// (`dsc/dsc2.cpp:6012-6013`). Over an OUTER-to-inner fold vector IBM's own diagram gives the
/// conversion, `loop_elem_arr_level = foldNumDims - i - 1` over the absolute fold index `i`
/// (`ddc/ddc_fold.cpp:2656-2660`), which `dsc/dsc2.cpp:6728` inverts the same way; over the
/// inner-to-outer `elemArrParamsAfterDistribution` the level IS the index, assigned straight across
/// (`dsc/dsc2.cpp:6074`, `:6352`). The walk also starts at `origNumElemArrFoldsOfRefNode`, not 0
/// (`:2667`).
///
/// ⭐ UNSIGNED IS SOUND: the writers are a `foldIdx` from a `foldIdx >= 0` loop
/// (`dsc/dsc2.cpp:6016-6017`) and a `currElemArrLevel` that cannot go negative because
/// `elemArrParamsAfterDistribution.at(nextFoldIdx)` throws first (`:6441-6443`).
///
/// ⛔ ONE AUTHORITY SITE IS OFF BY ONE IN THIS CURRENCY, so port it deliberately rather than
/// transcribe it. `constructAllocElemArrLayout` inserts a fold at `begin() + currElemArrIndex + 1`
/// (`ddc/ddc_fold.cpp:371-372`), which raises the level of every fold from `currElemArrIndex`
/// OUTWARDS — every level `>= size - 1 - currElemArrIndex` — but it bumps only the loops whose level
/// is `>= currElemArrLevel`, and `:341` fixed that at `size - currElemArrIndex` before the insertion
/// (`:377-379`). The split fold's OWN level is the one level the threshold skips, so a loop related
/// to it resolves (`dsc/dsc2.cpp:6728`) to the newly inserted inner fold instead of to the residue
/// whose `alpha` is scaled for it (`ddc/ddc_fold.cpp:386`). Only the FIRST insertion is wrong; by the
/// second the split fold's level has caught up with the stale threshold. Derived in
/// `unit_tests::the_level_shift_guard_skips_the_split_folds_own_level`, and unreachable from this
/// crate so far because the containing function is unported.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ElemArrLevel(pub u32);

/// One dimension's extent inside a unit view or a stick layout — the `int` of
/// `ScheduleNode::Size` (`dsc/dsc2.h:488`).
///
/// ⛔ A MULTIPLIER, NEVER AN INDEX. `calculateSizeIdxAndOffset` accumulates it as
/// `newDimSizeSoFar *= size_` while the `sizeIdx_` it writes beside it indexes `sizesNoGaps_`
/// (`dsc/dsc2.cpp:2732-2746`), and `UnitView::LoopInfo` carries both as bare `int`s
/// (`dsc/dsc2.h:503-504`).
///
/// ⛔ AND WHAT IT COUNTS DEPENDS ON THE DIM, so it is not a `Sticks` or an `Elements`. In one and
/// the same `sizesNoGaps_`, `buildUnitView` fills the stick dims from `getStickSizes` clamped to the
/// stick's capacity — ELEMENTS within a stick (`dsc/dsc2.cpp:2776-2781`) — and then fills the layout
/// dims with `ceil(remainingDimSize / stickDimSize.at(dim))`, a count of STICKS along that dim
/// (`:2792-2816`). Only the dim the extent is paired with says which.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DimSize(pub i32);

/// An index INTO a unit view's size vector — `UnitView::LoopInfo::sizeIdx_` (`dsc/dsc2.h:503-504`)
/// and a chunk entry's `srcSizeIdx_`/`dstSizeIdx_` (`dsc/dsc2.h:822`) are the same currency.
///
/// ⛔ A POSITION, NOT A DIM AND NOT A [`DimSize`]. Bridge 1 walks `view_sizes` positionally and
/// matches this against the POSITION `dim_id`, never against a `Size`'s dim
/// (`dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:333-348`) — which is why a chunk
/// entry carries an index AND a dim, and why the two are not interchangeable.
///
/// ⭐ UNSIGNED BECAUSE THE ONE LIVE PRODUCER IS: `populateUnitTimeTransfers` pushes the stick-size
/// loop counter (`ddc/ddcv1.cpp:524-525`) and bridge 1 assigns `i`
/// (`SNTransferLowering.cpp:631`). The authority's `-1` initialiser is [`None`].
///
/// ⛔ THE SECOND DDC WRITER IS NEITHER LIVE NOR NON-NEGATIVE, so it is not evidence for that
/// choice and must not be cited as such. `sizeIdx = getDimIndexInLayoutOrder(dsType, IN)
/// + tensorSizes.size()` (`ddc/ddcv1.cpp:1588-1591`, pushed at `:1597-1598`) sits inside the
/// lambda `checkAndResetUnitTimeTransfer` (`:1560-1645`) whose only callsite is commented out
/// (`:1649-1650`), and `getDimIndexInLayoutOrder` returns `-1` for a dim absent from
/// `layoutDimOrder_` (`dsc/designSpaceConfig.cpp:429-438`) — so that expression is `-1 + n`, and
/// is `-1` itself whenever `getStickSizes` came back empty. Were the callsite ever restored, this
/// type would have to tell that computed `-1` apart from the initialiser's absent one, which
/// [`Option<SizeIdx>`](Option) cannot.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SizeIdx(pub u32);

/// How many elements one trip of a loop moves along one extent of a
/// [`UnitView`] — `ScheduleNode::UnitView::LoopInfo::elemOffset_` (`dsc/dsc2.h:504`).
///
/// ⛔ A STRIDE, NOT AN ELEMENT COUNT. Every reader multiplies it by an induction value: bridge 1
/// builds the address expression as `iv_expr * elemOffset_`
/// (`dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:157`) and scales a composite loop's
/// contribution by it (`:246`). Its seed is `DataInfo::loopEleOffsets_`, an offset "in number of
/// elements in that dim (e.g. 4 mb)" per loop and dim (`dsc/dsc2.h:730-734`), which
/// `calculateSizeIdxAndOffset` then DIVIDES down by the running product of the extents it has already
/// passed, so what is stored is the step in units of the extent at
/// [`LoopInfo::size_idx`] (`dsc/dsc2.cpp:2732-2746`).
///
/// ⛔ AND IT IS REWRITTEN AFTER THAT: when a gap spreads the extent this loop steps, the gap pass
/// multiplies the stored value by that dim's `gapStickSpread_` (`dsc/dsc2.cpp:2880-2897`) — so the
/// value the bridge reads is in gapped elements, not in the elements the seed named.
///
/// `i32` and not `u32` because the authority's initialiser is `-1` (`dsc/dsc2.h:504`) and because `0`
/// is a live value meaning "reuse is expected" (`:732-734`); absence is
/// [`Option<ElemOffset>`](Option) and the sign is not how it is spelled.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ElemOffset(pub i32);

/// One data stage's id — the key of `DesignSpaceConfig::dataStageParam_`
/// (`dsc/designSpaceConfig.h:105`). `metadata_.core_dstgid` and `metadata_.chunk_dstgid` name the
/// two the DDL conversion compares a minted loop against (`ddc/ddl/ddl_conversion.cpp:1114-1115`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DataStageId(pub i32);

/// An index into `DesignSpaceConfig::labeledDs_` (`dsc/designSpaceConfig.h:86`) — the `ldsIdx` that
/// `parametricStride` looks up (`dsc/dsc2.cpp:4203`) and that `DataInfo::myLdsIdx_` holds
/// (`dsc/dsc2.h:722`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LdsIdx(pub i32);

/// One symbol id in `SuperDsc::symbolDefinitions_` — IBM's `using VariableSymbol = int64_t`
/// (`util/variabledefinition/VariableDefinition.h:32`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VariableSymbol(pub i64);

/// Replaces: e006_GroupTagRegInfo
///
/// GTR (group tag register) info, one per core on a transfer node — `coreIdToGTRInfo_` is a
/// `std::map<int, GroupTagRegInfo>` on `TransferNode` (`dsc/dsc2.h:34-37`, `:840`).
/// TRAP: the two -1s are NOT symmetric. `groupId_` is genuinely written -1, whenever there is only
/// one sharer i.e. no share (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:4831-4832`, `:5255-5256`),
/// so [`group_id`](Self::group_id) is [`Some`] exactly when `numSharers_ > 1`; `numSharers_` is
/// never written -1 at all (`:4828`, `:5252`). The JSON round trip carries both verbatim
/// (`dsc/dsc2.cpp:631-632`, `:1594-1597`), so neither field may be inferred from the other.
///
/// ⛔ AND THE PAIRING IS NOT ENFORCEABLE FROM ITS ONE CONSUMER, which is why both fields stay
/// independently optional: `dsc/dsc2Pcfg.cpp:1258-1266` copies each straight onto `GTRAndBurst` and
/// keeps the -1s, and that is DCG/PCFG — off this campaign's path. Nothing on our path reads either
/// field yet, so `(Some(_), Some(NumSharers(1)))` is merely unproduced, not unsound.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GroupTagRegInfo {
    /// Field: e006_GroupTagRegInfo.groupId_
    pub group_id: Option<GroupId>,
    /// Field: e006_GroupTagRegInfo.numSharers_
    pub num_sharers: Option<NumSharers>,
}

/// Replaces: e007_FoldParamInfoType
///
/// One fold level's affine parameters, trip count and label (`dsc/dsc2.h:1081-1085`).
/// TRAP: the declared default is the identity in NEITHER field, and `cardinality`'s 0 is not an
/// empty fold — every reader takes a 0 as 1 (`dsc/dsc2.cpp:6249-6253`, `:6359-6361`; see
/// [`Cardinality`]). The identity a caller actually wants is the one `getDefaultRowSplitFold`
/// writes, `alpha = 0` with `cardinality = 1` (`ddc/ddc_fold.cpp:2154-2159`), which overrides BOTH
/// declared initialisers.
///
/// ⭐ AGGREGATE-INITIALISED STRAIGHT OUT OF A [`LoopDistributionParamType`]'s `alpha` and `beta`
/// (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:7654-7655`), which is why the two types share these
/// newtypes instead of each declaring its own.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FoldParamInfoType {
    /// Field: e007_FoldParamInfoType.alpha
    pub alpha: Alpha,
    /// Field: e007_FoldParamInfoType.beta
    pub beta: Beta,
    /// Field: e007_FoldParamInfoType.cardinality
    pub cardinality: Cardinality,
    /// Field: e007_FoldParamInfoType.foldDimLabel
    ///
    /// Open set, not a closed one: `FoldDimProp::importFromJson` reads it from JSON. ⭐ But where the
    /// element arrangements are relabelled it is not free text — it is `"elem_arr_" +
    /// std::to_string(<level>)`, and both producers spell the same [`ElemArrLevel`] over an
    /// outer-to-inner fold vector: `c` counted up from the last index (`ddc/ddc_fold.cpp:2696-2698`)
    /// and `foldParams.size() - 1 - i` written out
    /// (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:7493-7497`).
    pub fold_dim_label: String,
}

impl Default for FoldParamInfoType {
    /// The authority's default member initializers, `dsc/dsc2.h:1082-1084`.
    fn default() -> Self {
        Self {
            alpha: Alpha(1),
            beta: Beta(0),
            cardinality: Cardinality(0),
            fold_dim_label: String::new(),
        }
    }
}

/// Replaces: e008_LoopDistributionParamType
///
/// One loop's affine parameters for one dimension after loop distribution (`dsc/dsc2.h:1110-1114`),
/// filled in two goes: alpha, beta and the level together (`ddc/ddc_fold.cpp:2687-2689`), the stride
/// afterwards (`dsc/dsc2.cpp:6748-6751`).
/// TRAP: the authority's `= -1` on the last two fields is UNSET, not a value, and no reader compares
/// against -1. An unset `relatedElemArrLevel` is LOUD, not silent: `allocFm.getNumDims() - level - 1`
/// (`dsc/dsc2.cpp:6727-6728`) makes the position `getNumDims()`, which `FoldManager::getAlpha`
/// `DT_CHECK`s (`util/foldManager/foldInfrastructure.h:2325-2329`). What IS silent is the opposite
/// direction: `getAlpha` remaps a negative position to `size() + pos`, so a level past the last
/// fold reads a DIFFERENT fold without complaint.
///
/// ⭐ THE ONE READER THAT ORDERS AGAINST THE LEVEL rather than indexing with it is
/// `relatedElemArrLevel >= currElemArrLevel`, which then increments the field in place
/// (`ddc/ddc_fold.cpp:377-379`); [`Option`] stays faithful to it because `None < Some(_)` and that
/// `currElemArrLevel` is at least 1 (`:330-341`). ⛔ But the threshold itself is one level too high
/// for the shift it guards — see [`ElemArrLevel`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LoopDistributionParamType {
    /// Field: e008_LoopDistributionParamType.alpha
    pub alpha: Alpha,
    /// Field: e008_LoopDistributionParamType.beta
    pub beta: Beta,
    /// Field: e008_LoopDistributionParamType.temporalStridePostDistribution
    pub temporal_stride_post_distribution: Option<TemporalStride>,
    /// Field: e008_LoopDistributionParamType.relatedElemArrLevel
    pub related_elem_arr_level: Option<ElemArrLevel>,
}

impl Default for LoopDistributionParamType {
    /// The authority's default member initializers, `dsc/dsc2.h:1111-1113`.
    fn default() -> Self {
        Self {
            alpha: Alpha(1),
            beta: Beta(0),
            temporal_stride_post_distribution: None,
            related_elem_arr_level: None,
        }
    }
}

/// Replaces: ScheduleNode::NodeType
///
/// `dsc/dsc2.h:446-456`. Which kind of node this is. It is the discriminant of the whole
/// `ScheduleNode` hierarchy: every concrete node fixes it in its constructor
/// (`dsc/dsc2.h:482`), it is `const` on the base (`:460`) so it never changes after construction,
/// and it is what both the exporter and the importer dispatch on
/// (`dsc/dsc2.cpp:376-377`, `:1337-1358`).
///
/// ⛔ THE DISCRIMINANTS ARE THE AUTHORITY'S. `INVALID` is 0 and is the base class's own default
/// (`dsc/dsc2.h:460`), so a node whose kind was never set reads as `Invalid` rather than as a real
/// kind, and `Invalid` is a full member of both spelling maps rather than an absent case.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum NodeType {
    /// The base class's default (`dsc/dsc2.h:460`); no concrete node constructs with it.
    #[default]
    Invalid = 0,
    Block = 1,
    Loop = 2,
    Transfer = 3,
    Compute = 4,
    Sync = 5,
    Condition = 6,
    Allocate = 7,
    StickMask = 8,
}

/// ⛔ E0080 IF A KIND IS EVER INSERTED, DROPPED, REORDERED OR LEFT OUT OF `ALL`: the discriminants
/// are what `nodeTypeToString` is keyed by and what the JSON dispatch compares against, and `ALL`
/// is positional against them.
const _: () = {
    let mut i = 0;
    while i < NodeType::ALL.len() {
        assert!(
            NodeType::ALL[i] as usize == i,
            "NodeType::ALL is out of declaration order"
        );
        i += 1;
    }
};

impl NodeType {
    /// Every kind in the authority's order, `Invalid` included — unlike `PrimaryDimTypes` there is
    /// no `NodeTypeCount` sentinel here, so all nine are real values.
    pub const ALL: [Self; 9] = [
        Self::Invalid,
        Self::Block,
        Self::Loop,
        Self::Transfer,
        Self::Compute,
        Self::Sync,
        Self::Condition,
        Self::Allocate,
        Self::StickMask,
    ];

    /// Field: e029_ScheduleNode.nodeTypeToString
    /// Field: e041_ScheduleNode.nodeTypeToString
    ///
    /// The spelling `ScheduleNode::nodeTypeToString` gives this kind
    /// (`dsc/dsc2.h:457`, defined `dsc/dsc2.cpp:1878-1888`).
    ///
    /// ⭐ TOTAL, AND THE AUTHORITY'S MAP IS TOO: all nine kinds have an entry, so none of the ten
    /// `nodeTypeToString.at(...)` callsites can throw (`dsc/dsc2.cpp:377`, `:1358`, `:1867`,
    /// `ddc/ddc_fold.cpp:1968`, `:2069`, `:3307`, `:3314`, `ddc/ddc_transformation_util.cpp:899`,
    /// `:1014`, `ddc/ddl/ddl_conversion.cpp:3484`). Neither map is ever iterated — only `.at()` —
    /// so their insertion order is not observable and only the mapping itself is ported.
    pub fn name(self) -> &'static str {
        match self {
            Self::Invalid => "invalid",
            Self::Block => "block",
            Self::Loop => "loop",
            Self::Transfer => "transfer",
            Self::Compute => "compute",
            Self::Sync => "sync",
            Self::Condition => "condition",
            Self::Allocate => "allocate",
            Self::StickMask => "stickmask",
        }
    }

    /// Field: e029_ScheduleNode.stringToNodeType
    /// Field: e041_ScheduleNode.stringToNodeType
    ///
    /// `ScheduleNode::stringToNodeType`, the flip of the above — IBM builds it with `flipMap`
    /// (`dsc/dsc2.h:458`, `dsc/dsc2.cpp:1889-1890`). An unknown spelling is absent, where IBM's
    /// `.at()` throws on the JSON import path (`dsc/dsc2.cpp:1337-1338`).
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "invalid" => Some(Self::Invalid),
            "block" => Some(Self::Block),
            "loop" => Some(Self::Loop),
            "transfer" => Some(Self::Transfer),
            "compute" => Some(Self::Compute),
            "sync" => Some(Self::Sync),
            "condition" => Some(Self::Condition),
            "allocate" => Some(Self::Allocate),
            "stickmask" => Some(Self::StickMask),
            _ => None,
        }
    }

    /// `ScheduleNode::isBlockNode` (`dsc/dsc2.h:479-481`), which reads nothing but `nodeType_`.
    ///
    /// ⭐ THESE THREE KINDS ARE EXACTLY THE ONES THAT OWN CHILDREN: `BlockNode` holds the
    /// `unique_ptr` child vector `next_` (`dsc/dsc2.h:526-538`), and `LoopNode` (`:563`) and
    /// `ConditionNode` (`:685`) are its only two derivations. `Transfer` (`:814`), `Compute`
    /// (`:900`), `Sync` (`:964`), `Allocate` (`:974`) and `StickMask` (`:1059`) derive from
    /// `ScheduleNode` directly and are leaves, and `Invalid` is not a block either.
    pub fn is_block_node(self) -> bool {
        matches!(self, Self::Block | Self::Loop | Self::Condition)
    }
}

/// Replaces: ScheduleNode::Size
///
/// `dsc/dsc2.h:486-498`. One dimension paired with its extent — an entry of a unit view's
/// `sizesNoGaps_`/`sizesWithGaps_` (`:506`, `:509`), of `StickMaskNode::stickLayout_` (`:1063`),
/// and the whole of one unit-time transfer chunk's `sizeDim_` (`:821`). Those three are every
/// `Size`-typed declaration in the authority.
///
/// ⛔ NO DEFAULT, DELIBERATELY, AND THE AUTHORITY'S ONE IS A HAZARD. `Size` is the only
/// dim-carrying struct in this header whose `dim_` has no member initialiser (`:487`) — the
/// `LoopInfo` beside it initialises its own to `PrimaryDimTypesCount` (`:502`). So `Size() =
/// default` (`:490`) leaves `dim_` indeterminate under default-initialisation and zero — the `in`
/// dim, *not* `PrimaryDimTypesCount` — under the value-initialisation that
/// `sizesNoGaps_.emplace_back()` performs. FIVE sites construct one that way, not the three this
/// anchor used to name: `sizesNoGaps_`, `sizesWithGaps_` and `stickLayout_` on the JSON import path
/// (`dsc/dsc2.cpp:1304`, `:1321`, `:1849`), a transfer chunk's `sizeDim_` on the same path
/// (`:1563`, `:1578`), and bridge 1's own `SizeAndIndex dim;`
/// (`dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:630`, filled at `:632-633`). Every
/// site that means a value calls the two-argument form (`dsc/dsc2.cpp:2780`, `:2816`), so the
/// placeholder CONSTRUCTOR is not ported:
///
/// ```compile_fail
/// // E0599, for the reader only: stable rustdoc parses the code an annotation names and ignores
/// // it, so the annotation is documentation and the positive control below is the check.
/// use deeptools::schedule::dsc2::Size;
/// let _ = Size::default();
/// ```
///
/// ⭐ AND ITS POSITIVE CONTROL, which rustdoc DOES enforce — the same path and the same construct
/// with the intended constructor. Without it the `compile_fail` above would pass just as happily on
/// a misspelled module path or an item that stopped being `pub`, i.e. exactly when it had stopped
/// testing anything:
///
/// ```
/// use deeptools::schedule::dims::PrimaryDimTypes;
/// use deeptools::schedule::dsc2::{DimSize, Size};
/// let _ = Size::new(PrimaryDimTypes::Ij, DimSize(64));
/// ```
///
/// ⛔ AND `size_ = -1` IS NOT A SIZE. Every reader multiplies it — `newDimSizeSoFar *= size_`
/// (`dsc/dsc2.cpp:2739`), `loopStride *= size_` (`:2962`), `elements *= dim_size.sizeDim_.size_`
/// (`dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:34`) — not one of them tests it. So
/// it is a plain [`DimSize`] and not an `Option`: unlike `e006`'s and `e010`'s `-1`s there is no
/// absent state for one to carry.
///
/// ⛔ AND THE GUARANTOR OF THAT IS THE EXPORTER, NOT `importSize`. `importSize` is a per-field `if`
/// chain that assigns only the keys the JSON object carries (`dsc/dsc2.cpp:1029-1038`), so a size
/// object written without `"size_"` would keep the `-1` and reach `calculateSizeIdxAndOffset` as a
/// negative multiplier (`:2739`). What keeps that out of the authority's own round trip is
/// `exportSize`, which writes `dim_` AND `size_` unconditionally (`:202-206`). ⚠️ And the port does
/// not exclude the VALUE either: `Size::new(PrimaryDimTypes::In, DimSize(-1))` spells the C++
/// value-initialised placeholder exactly. What is absent from the port is the constructor, not the
/// value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Size {
    /// Field: e029_ScheduleNode.dim_
    /// Field: e013_ScheduleNode.dim_
    /// Field: e041_ScheduleNode.dim_
    ///
    /// ⚠️ THE SCHEDULER'S `.dim_` ANCHOR COVERS TWO C++ FIELDS, so both sites carry it. This is
    /// `ScheduleNode::Size::dim_` (`dsc/dsc2.h:487`), which is a dim of the data structure the extent
    /// belongs to and has no default at all. [`LoopInfo::dim`] (`:502`) spells the same name, is a
    /// different field, defaults to `PrimaryDimTypesCount`, and is the dim of a LOOP rather than of
    /// an extent — `buildUnitView` can store one in the `LoopInfo` while resolving the extent under
    /// another (`dsc/dsc2.cpp:2852-2868`).
    pub dim: PrimaryDimTypes,
    /// Field: e029_ScheduleNode.size_
    /// Field: e013_ScheduleNode.size_
    /// Field: e041_ScheduleNode.size_
    pub size: DimSize,
}

impl Size {
    /// The two-argument form (`dsc/dsc2.h:491`), which is what every value-producing site uses
    /// (`dsc/dsc2.cpp:2780`, `:2816`).
    pub fn new(dim: PrimaryDimTypes, size: DimSize) -> Self {
        Self { dim, size }
    }
}

/// IBM's converting constructor from a `std::pair<PrimaryDimTypes, int>` (`dsc/dsc2.h:492-493`).
/// It is not `explicit`, and its live use is the range construction at `ddc/ddcv1.cpp:3546-3547`,
/// which builds a `vector<Size>` straight out of `getStickSizes`' `vector<pair<PrimaryDimTypes,
/// int>>` (`dsc/designSpaceConfig.h:256`).
impl From<(PrimaryDimTypes, DimSize)> for Size {
    fn from((dim, size): (PrimaryDimTypes, DimSize)) -> Self {
        Self::new(dim, size)
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    /// `dsc/dsc2.h:35-36`: both fields start absent.
    #[test]
    fn group_tag_reg_info_starts_with_neither_field() {
        assert_eq!(
            GroupTagRegInfo::default(),
            GroupTagRegInfo {
                group_id: None,
                num_sharers: None
            }
        );
    }

    /// `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:4827-4836` and its twin `:5251-5260`, over the real
    /// map and the real set: `gtrIdsUsed_` gains an id only from a record that HAS one (`:4833-4834`),
    /// so two cores sharing one group contribute one id and the unshared core contributes none.
    #[test]
    fn only_a_present_group_id_reaches_gtr_ids_used() {
        use crate::schedule::dsc::DesignSpaceConfig;

        let mut node = TransferNode::default();
        for (core, shares, group_name) in [(0, 2, 3), (1, 2, 3), (2, 1, 9)] {
            // `numSharers_ = shares; groupId_ = shares > 1 ? groupName : -1;`, with `shares >= 1`
            // guaranteed at `:4697` and the shared path's name `DT_CHECK`ed at `:4710-4711`.
            node.core_id_to_gtr_info.insert(
                CoreId(core),
                GroupTagRegInfo {
                    group_id: (shares > 1).then_some(GroupId(group_name)),
                    num_sharers: Some(NumSharers(shares)),
                },
            );
        }

        let mut dsc = DesignSpaceConfig::default();
        dsc.gtr_ids_used.extend(
            node.core_id_to_gtr_info
                .values()
                .filter_map(|info| info.group_id),
        );

        assert_eq!(dsc.gtr_ids_used, BTreeSet::from([GroupId(3)]));
        assert_eq!(
            node.core_id_to_gtr_info[&CoreId(2)].num_sharers,
            Some(NumSharers(1)),
            "the unshared core still records its one sharer"
        );
    }

    /// `ddc/ddc_fold.cpp:2656-2660` — IBM's own diagram, seven folds `S S T T T T E`, with
    /// `loop_elem_arr_level = foldNumDims - i - 1` over the absolute fold index `i`, inverted the
    /// same way at `dsc/dsc2.cpp:6728`. ⛔ Level 0 is the INNERMOST fold, so an [`ElemArrLevel`] is
    /// neither a fold index nor an outermost-first count.
    #[test]
    fn the_elem_arr_level_is_counted_from_the_innermost_fold() {
        const FOLD_NUM_DIMS: u32 = 7;
        let level_of = |fold_index: u32| ElemArrLevel(FOLD_NUM_DIMS - fold_index - 1);
        let fold_index_of = |level: ElemArrLevel| FOLD_NUM_DIMS - level.0 - 1;

        // The diagram's own worked case, "For loops related to positions [4-5]".
        assert_eq!(level_of(4), ElemArrLevel(2));
        assert_eq!(level_of(5), ElemArrLevel(1));
        // The two ends, which is what innermost-first means.
        assert_eq!(level_of(FOLD_NUM_DIMS - 1), ElemArrLevel(0));
        assert_eq!(level_of(0), ElemArrLevel(FOLD_NUM_DIMS - 1));

        for fold_index in 0..FOLD_NUM_DIMS {
            assert_eq!(fold_index_of(level_of(fold_index)), fold_index);
        }
    }

    /// `ddc/ddc_fold.cpp:2696-2698` and `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:7493-7497`
    /// relabel the element-arrangement folds independently — one counting `c` up from the innermost,
    /// the other writing `foldParams.size() - 1 - i` — and both spell the same [`ElemArrLevel`].
    #[test]
    fn the_elem_arr_label_numbers_folds_by_their_level() {
        // Outer to inner, the order both loops walk backwards over.
        const NUM_ELEM_ARR_FOLDS: usize = 3;
        let mut folds = vec![FoldParamInfoType::default(); 5];

        // `for (int i = size - 1, c = 0; c < numElemArrFoldsOfAllocNode; --i, ++c)`.
        for (c, i) in (0..NUM_ELEM_ARR_FOLDS).zip((0..folds.len()).rev()) {
            folds[i].fold_dim_label = format!("elem_arr_{c}");
        }
        // `foldDimLabel = "elem_arr_" + std::to_string(foldParams.size() - 1 - i)`, same tail.
        let l3_labels: Vec<String> = (folds.len() - NUM_ELEM_ARR_FOLDS..folds.len())
            .map(|i| format!("elem_arr_{}", folds.len() - 1 - i))
            .collect();
        assert_eq!(
            folds[folds.len() - NUM_ELEM_ARR_FOLDS..]
                .iter()
                .map(|fold| fold.fold_dim_label.clone())
                .collect::<Vec<_>>(),
            l3_labels
        );

        // The suffix IS the level, so the INNERMOST element arrangement is `elem_arr_0`.
        let level_of = |i: usize| ElemArrLevel((folds.len() - 1 - i) as u32);
        assert_eq!(level_of(4), ElemArrLevel(0));
        assert_eq!(folds[4].fold_dim_label, "elem_arr_0");
        assert_eq!(level_of(2), ElemArrLevel(2));
        assert_eq!(folds[2].fold_dim_label, "elem_arr_2");
        // The outer, non-element-arrangement folds keep the declared default's empty label.
        assert_eq!(folds[0].fold_dim_label, String::new());
    }

    /// `ddc/ddc_fold.cpp:341` fixes the shift threshold at `foldParams.size() - currElemArrIndex`,
    /// but the insertion at `:371-372` raises the level of every fold from `currElemArrIndex`
    /// OUTWARDS, i.e. every level `>= size - 1 - currElemArrIndex`. ⛔ So `:377-379` skips exactly
    /// one level — the split fold's own — and a loop related to it lands on the newly inserted inner
    /// fold instead of the residue whose `alpha` `:386` scales. A derivation, not a port: the
    /// containing function is unported, so nothing in this crate reaches it yet.
    #[test]
    fn the_level_shift_guard_skips_the_split_folds_own_level() {
        // Outer to inner; `currElemArrIndex` is the innermost fold with cardinality != 1 (`:330-335`).
        const SIZE: i64 = 5;
        const CURR_ELEM_ARR_INDEX: i64 = 3;
        let level_of = |index: i64, size: i64| ElemArrLevel((size - 1 - index) as u32);

        // Which levels `insert(begin() + currElemArrIndex + 1, ..)` actually moves: a fold at index
        // <= currElemArrIndex keeps its index while the size grows, so its level gains one.
        let shifted = |index: i64| {
            index <= CURR_ELEM_ARR_INDEX
                && level_of(index, SIZE + 1) == ElemArrLevel(level_of(index, SIZE).0 + 1)
        };
        assert!(
            (0..=CURR_ELEM_ARR_INDEX).all(shifted),
            "the split fold and everything outside it move one level out"
        );
        let lowest_shifted_level = level_of(CURR_ELEM_ARR_INDEX, SIZE);

        // What `:377-379` bumps instead, from the threshold `:341` fixed before the insertion.
        let curr_elem_arr_level = ElemArrLevel((SIZE - CURR_ELEM_ARR_INDEX) as u32);
        assert_eq!(curr_elem_arr_level.0, lowest_shifted_level.0 + 1);
        let skipped: Vec<u32> = (lowest_shifted_level.0..curr_elem_arr_level.0).collect();
        assert_eq!(
            skipped,
            vec![lowest_shifted_level.0],
            "one level wide, and it is the split fold's own"
        );

        // A loop left on that level converts back (`dsc/dsc2.cpp:6728`) to the INSERTED fold, not to
        // the residue at `currElemArrIndex` whose alpha `:386` scales for it.
        let index_in_folds = |level: ElemArrLevel, size: i64| size - i64::from(level.0) - 1;
        assert_eq!(
            index_in_folds(lowest_shifted_level, SIZE + 1),
            CURR_ELEM_ARR_INDEX + 1
        );
    }

    /// `dsc/dsc2.cpp:6249-6253` and `:6359-6361`: a 0 `cardinality` is PRODUCED — "Parametric loops
    /// may have 0 iteration count" (`:6251`) — and both readers take it as the multiplicative
    /// identity 1. ⛔ It is not an empty fold, and it does not annihilate the product.
    #[test]
    fn a_zero_cardinality_reads_as_one_not_as_an_empty_fold() {
        let inner_folds = [
            FoldParamInfoType {
                cardinality: Cardinality(4),
                ..Default::default()
            },
            // A parametric loop's 0 iteration count is the declared default's cardinality.
            FoldParamInfoType::default(),
            FoldParamInfoType {
                cardinality: Cardinality(8),
                ..Default::default()
            },
        ];
        assert_eq!(inner_folds[1].cardinality, Cardinality(0));

        // `:6250-6252` skips a 0 in the inner-cardinality product.
        let skipping_zero: i64 = inner_folds
            .iter()
            .filter(|fold| fold.cardinality != Cardinality(0))
            .map(|fold| fold.cardinality.0)
            .product();
        // `:6359-6361` spells `cardinality == 0 ? 1 : cardinality` over the same folds.
        let coercing_zero_to_one: i64 = inner_folds
            .iter()
            .map(|fold| {
                if fold.cardinality == Cardinality(0) {
                    1
                } else {
                    fold.cardinality.0
                }
            })
            .product();

        assert_eq!(skipping_zero, 32, "the 0 contributed the identity, not 0");
        assert_eq!(skipping_zero, coercing_zero_to_one, "both readers agree");
    }

    /// `dsc/dsc2.h:1082-1084` against `getDefaultRowSplitFold`, `ddc/ddc_fold.cpp:2154-2159`:
    /// the declared default is NOT the identity fold that function writes.
    #[test]
    fn fold_param_info_default_is_not_the_identity_row_split_fold() {
        assert_eq!(
            FoldParamInfoType::default(),
            FoldParamInfoType {
                alpha: Alpha(1),
                beta: Beta(0),
                cardinality: Cardinality(0),
                fold_dim_label: String::new(),
            }
        );

        let row_split_fold = FoldParamInfoType {
            alpha: Alpha(0),
            beta: Beta(0),
            cardinality: Cardinality(1),
            fold_dim_label: "rowsplit_fold".to_string(),
        };
        assert_ne!(FoldParamInfoType::default(), row_split_fold);
    }

    /// `dsc/dsc2.h:1111-1113`: alpha starts at 1 and beta at 0, and the two -1 fields start unset.
    #[test]
    fn loop_distribution_param_starts_with_an_identity_affine_and_nothing_distributed() {
        assert_eq!(
            LoopDistributionParamType::default(),
            LoopDistributionParamType {
                alpha: Alpha(1),
                beta: Beta(0),
                temporal_stride_post_distribution: None,
                related_elem_arr_level: None,
            }
        );
    }

    /// `dsc/dsc2.cpp:6727-6728` against `FoldManager::getAlpha`
    /// (`util/foldManager/foldInfrastructure.h:2325-2329`): an unset `relatedElemArrLevel` indexes
    /// ONE PAST the folds and trips the `DT_CHECK`, while a level past the OUTERMOST fold indexes
    /// NEGATIVE and is silently remapped to the innermost one. ⛔ The two -1s fail differently.
    #[test]
    fn an_unset_related_elem_arr_level_indexes_off_the_end_of_the_folds() {
        const NUM_DIMS: i64 = 4;
        // `allocFm.getNumDims() - currLoopParam.relatedElemArrLevel - 1`, with the authority's -1
        // for an unset level substituted verbatim.
        let position_of = |param: LoopDistributionParamType| {
            let level = param
                .related_elem_arr_level
                .map_or(-1, |level| i64::from(level.0));
            NUM_DIMS - level - 1
        };
        // `getAlpha(pos)`: `if (pos < 0) pos = size + pos;` then `DT_CHECK(0 <= pos < size)`.
        let get_alpha_reads = |pos: i64| {
            let pos = if pos < 0 { NUM_DIMS + pos } else { pos };
            (0..NUM_DIMS).contains(&pos).then_some(pos)
        };

        let unset = LoopDistributionParamType::default();
        assert_eq!(unset.related_elem_arr_level, None);
        assert_eq!(position_of(unset), NUM_DIMS);
        assert_eq!(get_alpha_reads(NUM_DIMS), None, "the DT_CHECK, loudly");

        let innermost = LoopDistributionParamType {
            related_elem_arr_level: Some(ElemArrLevel(0)),
            ..Default::default()
        };
        assert_eq!(position_of(innermost), NUM_DIMS - 1);
        assert_eq!(get_alpha_reads(NUM_DIMS - 1), Some(NUM_DIMS - 1));

        let past_the_outermost = LoopDistributionParamType {
            related_elem_arr_level: Some(ElemArrLevel(NUM_DIMS as u32)),
            ..Default::default()
        };
        assert_eq!(position_of(past_the_outermost), -1);
        assert_eq!(
            get_alpha_reads(-1),
            Some(NUM_DIMS - 1),
            "silently the innermost fold, not a refusal"
        );
    }

    /// The nine node kinds in the authority's declaration order (`dsc/dsc2.h:446-456`).
    const EVERY_NODE_TYPE: [NodeType; 9] = [
        NodeType::Invalid,
        NodeType::Block,
        NodeType::Loop,
        NodeType::Transfer,
        NodeType::Compute,
        NodeType::Sync,
        NodeType::Condition,
        NodeType::Allocate,
        NodeType::StickMask,
    ];

    /// `dsc/dsc2.h:446-456` — the discriminants `nodeTypeToString` is keyed by, with `INVALID`
    /// first so that the base class's own default (`:460`) is `Invalid`. ⛔ `ALLOCATE` precedes
    /// `STICKMASK` in the enum even though `nodeTypeToString`'s initialiser list spells them the
    /// other way round (`dsc/dsc2.cpp:1886-1888`); the enum is what fixes the values.
    #[test]
    fn the_node_type_discriminants_are_the_authoritys() {
        for (i, node_type) in EVERY_NODE_TYPE.into_iter().enumerate() {
            assert_eq!(node_type as usize, i, "{node_type:?} moved");
        }
        assert_eq!(NodeType::ALL, EVERY_NODE_TYPE);
        assert_eq!(NodeType::default(), NodeType::Invalid);
        assert_eq!(
            NodeType::Allocate as usize + 1,
            NodeType::StickMask as usize
        );
    }

    /// `dsc/dsc2.cpp:1878-1890`: all nine kinds have a spelling, so `nodeTypeToString.at()` cannot
    /// throw, and `stringToNodeType` is exactly its `flipMap`.
    #[test]
    fn every_node_type_name_round_trips() {
        for node_type in EVERY_NODE_TYPE {
            assert_eq!(NodeType::from_name(node_type.name()), Some(node_type));
        }

        // `flipMap` is injective here: nine distinct spellings for nine kinds, so no kind is lost
        // on the JSON import path (`dsc/dsc2.cpp:1337-1338`).
        let spellings: std::collections::BTreeSet<_> =
            EVERY_NODE_TYPE.iter().map(|t| t.name()).collect();
        assert_eq!(spellings.len(), EVERY_NODE_TYPE.len());

        // The one spelling that is not the variant name lower-cased word for word.
        assert_eq!(NodeType::StickMask.name(), "stickmask");
        assert_eq!(NodeType::from_name("stick_mask"), None);
        assert_eq!(NodeType::from_name("STICKMASK"), None);
    }

    /// `dsc/dsc2.h:479-481` against the derivations at `:526`, `:563`, `:685`, `:814`, `:900`,
    /// `:964`, `:974` and `:1059`: exactly the three kinds that derive from `BlockNode` — the only
    /// class holding a child vector (`:538`) — answer yes. `Invalid` included, every other kind is
    /// a leaf.
    #[test]
    fn only_block_loop_and_condition_are_block_nodes() {
        let blocks: Vec<NodeType> = EVERY_NODE_TYPE
            .into_iter()
            .filter(|t| t.is_block_node())
            .collect();
        assert_eq!(
            blocks,
            [NodeType::Block, NodeType::Loop, NodeType::Condition]
        );
        assert!(!NodeType::Invalid.is_block_node());
        assert!(!NodeType::Allocate.is_block_node());
        assert!(!NodeType::StickMask.is_block_node());
    }

    /// `dsc/dsc2.h:491-497`: the two-argument form carries both fields, the pair conversion
    /// (`:492-493`) agrees with it, and `operator==` sees both — so the same extent on another dim
    /// is a different `Size`, which is what `stickLayout_ != stickDimSizes` relies on
    /// (`ddc/ddcv1.cpp:3556`).
    #[test]
    fn a_size_is_one_dim_with_its_extent_and_equality_sees_both() {
        let size = Size::new(PrimaryDimTypes::Ij, DimSize(64));
        assert_eq!(size.dim, PrimaryDimTypes::Ij);
        assert_eq!(size.size, DimSize(64));

        // `ddc/ddcv1.cpp:3546-3547` range-constructs a `vector<Size>` out of `getStickSizes`' pairs.
        assert_eq!(Size::from((PrimaryDimTypes::Ij, DimSize(64))), size);
        assert_ne!(Size::new(PrimaryDimTypes::Ki, DimSize(64)), size);
        assert_ne!(Size::new(PrimaryDimTypes::Ij, DimSize(32)), size);

        // A stick layout compares elementwise AND in order, per `stickLayout_ != stickDimSizes`.
        let layout = [
            Size::new(PrimaryDimTypes::In, DimSize(4)),
            Size::new(PrimaryDimTypes::Ij, DimSize(32)),
        ];
        assert_eq!(layout, [layout[0], layout[1]]);
        assert_ne!(layout, [layout[1], layout[0]]);
    }

    /// `ddc/ddcv1.cpp:544-546` and `:1637-1643`: the DDC's two writes to a chunk's [`Size`] are
    /// different shapes. The splat mutates the STORED entry; the hole split mutates a COPY, sends
    /// that to the strides and erases the source — so the chunk-size vector never holds the 1.
    #[test]
    fn the_splat_mutates_the_stored_chunk_extent_and_the_hole_split_mutates_a_copy() {
        let entry = |dim, size, idx| SizeAndIndex {
            size_dim: Size::new(dim, DimSize(size)),
            src_size_idx: Some(SizeIdx(idx)),
            dst_size_idx: Some(SizeIdx(idx)),
        };
        let mut chunk_size = vec![entry(PrimaryDimTypes::In, 1, 0), entry(PrimaryDimTypes::Ij, 8, 1)];

        // `:544-546`, in place on the first entry, from the 1 its `DT_CHECK` requires.
        chunk_size[0].size_dim.size = DimSize(chunk_size[0].size_dim.size.0 * 4);
        assert_eq!(chunk_size[0], entry(PrimaryDimTypes::In, 4, 0));

        // `:1637-1643`: copy, accumulate, set the COPY to 1, append to the strides, erase the source.
        let hole_end_pos = 1;
        let mut num_chunks = 1;
        let mut chunk_stride = Vec::new();
        for stored in &chunk_size[hole_end_pos..] {
            let mut size_dim = *stored;
            num_chunks *= size_dim.size_dim.size.0;
            size_dim.size_dim.size = DimSize(1);
            chunk_stride.push(size_dim);
        }
        chunk_size.truncate(hole_end_pos);

        assert_eq!(num_chunks, 8);
        assert_eq!(chunk_stride, vec![entry(PrimaryDimTypes::Ij, 1, 1)]);
        // The 1 lives in the strides only; the entry it was copied from is gone, not set to 1.
        assert_eq!(chunk_size, vec![entry(PrimaryDimTypes::In, 4, 0)]);
    }

    /// `dsc/dsc2.h:43` against all THREE naming conventions: the `+ "el"` suffix the DDC gives the
    /// stages it mints by number (`ddc/ddc_transformation_util.cpp:121-122`, `:131-132`,
    /// `ddc/ddc_transformation.cpp:1018-1019`, `ddc/ddcv1.cpp:1236-1237`), the shared name all three
    /// NAMED stages carry in both halves — `"core"` (`fillLoopLatchSdsc`,
    /// `dbo/src/Utils/sdsc_bundle/ProgramCorrection.cpp:1074-1075`) and `"chunk"`
    /// (`addOrUpdateDataStageParam` called with one name for both,
    /// `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:1419-1420`, `:1477-1480`) — and the DDL's own
    /// by-number stage, which names `ss_` and leaves `el_` unnamed
    /// (`ddc/ddl/ddl_conversion.cpp:1512-1515`).
    #[test]
    fn a_data_stages_name_is_its_steady_states_and_the_el_suffix_is_not_an_invariant() {
        // `dsc2::DataStage newDstg;` (`dsc/dsc2.cpp:3619`) and `emplace(index, dsc2::DataStage())`
        // (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:731`): both halves start empty and unnamed.
        let mut stage = DataStage::default();
        assert_eq!(stage.name(), "");
        assert!(stage.ss.empty() && stage.el.empty());

        // `ddc/ddc_transformation_util.cpp:117-124`: stage 7 is named "7", its epilogue "7el".
        stage.ss.name = "7".to_string();
        stage.el.name = format!("{}el", stage.ss.name);
        assert_eq!(stage.name(), "7");
        assert_eq!(stage.el.name, "7el");

        // `ddc/ddcv1.cpp:1236-1237`: the epilogue starts as a copy of the steady state, and stays
        // equal to it until a relevant dim shrinks — `DataStructDims`' equality omits `name_`.
        stage.ss.out = crate::schedule::dims::DimSize::new(64.0);
        stage.el = stage.ss.clone();
        stage.el.name = format!("{}el", stage.ss.name);
        assert_eq!(stage.el, stage.ss);
        assert_ne!(stage.el.name, stage.ss.name);
        stage.el.out = crate::schedule::dims::DimSize::new(16.0);
        assert_ne!(stage.el, stage.ss);

        // The suffix is not an invariant: `fillLoopLatchSdsc` names both halves "core", and
        // `getSizeDataStageForNode` checks only the steady state's (`dsc/dsc2.cpp:3638-3639`).
        let named = |name: &str| DataStage {
            ss: DataStructDims {
                name: name.to_string(),
                ..DataStructDims::default()
            },
            el: DataStructDims {
                name: name.to_string(),
                ..DataStructDims::default()
            },
        };
        let core = named("core");
        assert_eq!(core.name(), "core");
        assert_eq!(core.el.name, core.name());

        // ⛔ THE CHUNK STAGE IS THE ONE THAT BITES, and it is on this campaign's own port path: both
        // callers of `addOrUpdateDataStageParam` pass one `chunkDsName` for `ssName` AND `elName`
        // (`L3DlOpsScheduler.cpp:1419-1420`, `:1477-1480`), so the stage that
        // `attachToPrefilledSchedule` requires to be named "chunk" (`ddc/ddcv1.cpp:2285`) has an
        // epilogue named "chunk" too. A port deriving the epilogue's name would write "chunkel".
        let chunk = named("chunk");
        assert_eq!(chunk.name(), "chunk");
        assert_eq!(chunk.el.name, "chunk");
        assert_ne!(chunk.el.name, format!("{}el", chunk.name()));

        // ⛔ AND THE DDL'S OWN BY-NUMBER STAGE NAMES THE STEADY STATE ONLY: the `DatastageOp` handler
        // writes `ss_.name_ = to_string(id)` and never assigns `el_.name_`
        // (`ddc/ddl/ddl_conversion.cpp:1512-1515`), so stage 7 there has an epilogue named "" — the
        // third convention, and the one a derived `+ "el"` suffix gets wrong in the other direction.
        let mut ddl_minted = DataStage::default();
        ddl_minted.ss.name = "7".to_string();
        assert_eq!(ddl_minted.name(), "7");
        assert_eq!(ddl_minted.el.name, "");
    }

    /// `ddc/ddl/ddl_conversion.cpp:2974` reads `dataStageParam_[loopnode->numId_]` and `:2998` reads
    /// `[loopnode->denId_]`, both through the NON-CONST `operator[]`, and the DFS that reaches them
    /// (`:2951-3082`) has no parametric guard — so the `-1` a `ParametricLoopOp` loop carries
    /// (`:1129-1130`, linked at `:1162`) is INSERTED by the very test that asks whether its name is
    /// empty, once per id.
    #[test]
    fn the_emissions_emptiness_test_mints_the_absent_stage_it_reads() {
        use crate::schedule::dims::DimVal;
        use crate::schedule::metadata::Datastage;

        // `:1510-1511`: the id the DDL mints next starts from `dataStageParam_.size()`.
        let mut param: BTreeMap<DataStageId, DataStage> = BTreeMap::new();
        param.insert(DataStageId(0), DataStage::default());
        assert_eq!(param.len(), 1);

        // `:2974`, `dataStageParam_[loopnode->numId_]` with `numId_ == -1`: the READ IS A WRITE, and
        // what it reads back is the empty name that selects the `DatastageOp` branch at `:2985`.
        let minted = param.entry(DataStageId(-1)).or_default().clone();
        assert_eq!(minted.name(), "");
        assert_eq!(param.len(), 2);

        // Which is what keeps `.at(denId_)` / `.at(numId_)` at `:3066-3067` — the SAME iteration —
        // from throwing. ⛔ AND WHAT THEY FIND ANSWERS `-1`, NOT ABSENT: the authority's unfilled dim
        // (`dsc/dims.h:162-193`, and `calculate_padded` returns `-1` for any negative,
        // `dsc/dims.cpp:567-568`), reproduced here in both halves. `:3070-3071` reads `numstg.ss_` and
        // `denstg.ss_`, which with both ids `-1` is this one stage's steady state twice, so the
        // `ss_loop_count` the emission attaches to a parametric loop is `ceil(-1.0 / -1) == 1`.
        assert!(minted.ss.empty() && minted.el.empty());
        let ss = minted.ss.primary_dim_to_val(PrimaryDimTypes::In);
        assert_eq!(ss, Some(DimVal(-1)));
        assert_eq!(
            minted.el.primary_dim_to_val(PrimaryDimTypes::In),
            Some(DimVal(-1))
        );
        let count = ss.map(|DimVal(v)| (f64::from(v) / f64::from(v)).ceil());
        assert_eq!(count, Some(1.0));

        // ⛔ AND THE BRANCH THAT EMPTY NAME SELECTS MINTS AGAIN, SOMEWHERE ELSE: `:2986-2988` reads
        // `metadata_.datastages_[numId_].strategyMinimize_`, a SECOND default-inserting container
        // (`ddc/ddc_metadata.h:82`), whose freshly minted value answers `true` (`:77`) — so the
        // `DatastageOp` the emission creates for a parametric loop is spelled "minimize", never the
        // "maximize" the line above it initialises. `denId_` repeats both reads at `:2998` and `:3011`.
        let mut datastages: BTreeMap<DataStageId, Datastage> = BTreeMap::new();
        let strategy = if datastages
            .entry(DataStageId(-1))
            .or_default()
            .strategy_minimize
        {
            "minimize"
        } else {
            "maximize"
        };
        assert_eq!(strategy, "minimize");
        assert_eq!(
            datastages.len(),
            1,
            "one absent id, one insert per container"
        );
    }

    /// `dsc/dsc2.h:1088`: the declared order and the field's `NOT_PROCESSED` initialiser (`:1093`).
    /// Which states the authority ever writes is a C++ fact this cannot reach; it is recorded on
    /// [`PropStateType`] with its evidence.
    #[test]
    fn the_prop_state_discriminants_are_the_authoritys_and_the_default_is_not_processed() {
        let declared = [
            PropStateType::NotProcessed,
            PropStateType::RolledBack,
            PropStateType::Overridden,
            PropStateType::Complete,
        ];
        for (i, state) in declared.into_iter().enumerate() {
            assert_eq!(state as usize, i, "{state:?} moved");
        }
        assert_eq!(PropStateType::default(), PropStateType::NotProcessed);
    }

    /// The three shapes `ddc/ddl/ddl_conversion.cpp:1076-1164` mints plus the head `ScheduleTree()`
    /// leaves behind (`dsc/dsc2.h:629`), against the declared defaults (`dsc/dsc2.h:573-578`,
    /// `:617-618`) and the dummy loop at
    /// `dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:734`.
    #[test]
    fn the_minted_loop_shapes_are_four_independent_fields_and_none_excludes_another() {
        use crate::schedule::dims::MetaDimKind;

        // `new dsc2::LoopNode()` (`ddl_conversion.cpp:1076`): every id and index absent, no dims.
        let fresh = LoopNode::default();
        assert_eq!(
            (fresh.num_id, fresh.den_id, fresh.parametric_lds_idx()),
            (None, None, None)
        );
        assert!(!fresh.is_parametric_loop());
        assert!(fresh.dims.is_empty() && fresh.loop_count_symbol_ids.is_empty());

        // A `LoopOp` (`:1096-1103`): both stage ids set, and no lds index — it is not parametric.
        let ds_loop = LoopNode {
            num_id: Some(DataStageId(0)),
            den_id: Some(DataStageId(2)),
            dims: vec![PrimaryDimTypes::Ij.into()],
            ..LoopNode::default()
        };
        assert!(!ds_loop.is_parametric_loop());
        assert_eq!(ds_loop.parametric_lds_idx(), None);

        // A `ParametricLoopOp` (`:1127-1161`): NEITHER id, exactly one dim, and an lds index.
        let mut parametric = LoopNode::default();
        parametric.mark_as_parametric_loop();
        parametric.dims.push(PrimaryDimAndKind::new(
            PrimaryDimTypes::Ij,
            MetaDimKind::PadFront,
        ));
        parametric.set_parametric_lds_idx(Some(LdsIdx(3)));
        assert!(parametric.is_parametric_loop());
        assert_eq!((parametric.num_id, parametric.den_id), (None, None));
        assert_eq!(parametric.dims.len(), 1);
        assert_eq!(parametric.parametric_lds_idx(), Some(LdsIdx(3)));

        // ⛔ The head carries `denId_` ALONE. This is the shape behind every `>= 0` guard on a
        // climbed parent (`ddc/ddcv1.cpp:634`, `:645`). It is NOT what makes an absent numerator a
        // throw everywhere: `dsc/dsc2.cpp:2995` throws only because `:2994` has already excluded
        // parametric loops, and `ddl_conversion.cpp:2974` MINTS the key instead — pinned by
        // `the_emissions_emptiness_test_mints_the_absent_stage_it_reads`.
        let head = LoopNode {
            den_id: Some(DataStageId(0)),
            ..LoopNode::default()
        };
        assert_eq!((head.num_id, head.den_id), (None, Some(DataStageId(0))));
        assert!(head.dims.is_empty());

        // ⛔ NO SHAPE EXCLUDES ANOTHER. The four fields are independent, so the state a tagged enum
        // would have to reject IS representable here — the importer builds exactly this one entry at
        // a time (`dsc/dsc2.cpp:1405-1426`): marked parametric, both ids still set, no lds index.
        let mut mixed = ds_loop.clone();
        mixed.mark_as_parametric_loop();
        assert!(mixed.is_parametric_loop());
        assert_eq!(mixed.num_id, Some(DataStageId(0)));
        assert_eq!(mixed.den_id, Some(DataStageId(2)));
        assert_eq!(mixed.parametric_lds_idx(), None);

        // `SNTransferLowering.cpp:734`, `LoopNode(-1, -1, {dim})`: a bare dim means it unpadded.
        let dummy = LoopNode::new(None, None, vec![PrimaryDimTypes::In.into()], false);
        assert_eq!((dummy.num_id, dummy.den_id), (None, None));
        assert_eq!(
            dummy.dims,
            [PrimaryDimAndKind::new(
                PrimaryDimTypes::In,
                MetaDimKind::Unpadded
            )]
        );
        assert_eq!(dummy.parametric_lds_idx(), None);
    }

    /// `dsc/dsc2.h:592` against `dsc/dsc2.cpp:4223-4228`. The constructor keeps the order it is
    /// handed — inner to outer (`dsc/dsc2.h:575`, `dsc/dsc2Pcfg.cpp:517`), which is why bridge 1
    /// opens index `size() - 1` first and reads that one back as the outermost
    /// (`SNControlFlowLowering.cpp:893`, `:898`) — and `hasLoopDim` matches the dim ALONE.
    #[test]
    fn the_ctor_keeps_the_dim_order_it_is_given_and_has_loop_dim_matches_the_dim_alone() {
        use crate::schedule::dims::MetaDimKind;

        // `dsc2.h:592` inserts at `begin()` of an empty vector, so the order survives unchanged.
        let dims = vec![
            PrimaryDimAndKind::new(PrimaryDimTypes::Ij, MetaDimKind::Unpadded),
            PrimaryDimAndKind::new(PrimaryDimTypes::Mb, MetaDimKind::WindowDim),
            PrimaryDimAndKind::new(PrimaryDimTypes::In, MetaDimKind::PadBack),
        ];
        let loop_node = LoopNode::new(
            Some(DataStageId(0)),
            Some(DataStageId(1)),
            dims.clone(),
            false,
        );
        assert_eq!(loop_node.dims, dims);

        // ⛔ THE KIND IS IGNORED. `Mb` and `In` are carried under kinds that are NOT the default one
        // `From<PrimaryDimTypes>` supplies, so an implementation comparing whole
        // `PrimaryDimAndKind`s answers no here where the authority's dim-only compare answers yes.
        assert!(loop_node.has_loop_dim(PrimaryDimTypes::Ij));
        assert!(loop_node.has_loop_dim(PrimaryDimTypes::Mb));
        assert!(loop_node.has_loop_dim(PrimaryDimTypes::In));
        assert!(!loop_node.has_loop_dim(PrimaryDimTypes::Ki));
    }

    /// `dsc/dsc2.h:595-597` against the one site that fills the map (`dsc/dsc2.cpp:2992-3010`) and
    /// the one that reads a bound out of it (`SNControlFlowLowering.cpp:921-923`).
    #[test]
    fn a_symbolic_dim_is_the_map_alone_and_an_empty_symbol_list_still_reads_symbolic() {
        let mut loop_node = LoopNode::new(
            Some(DataStageId(0)),
            Some(DataStageId(1)),
            vec![PrimaryDimTypes::Ij.into(), PrimaryDimTypes::Mb.into()],
            false,
        );

        // Nothing is symbolic until `finalizeScheduleTree` fills the map.
        assert!(!loop_node.is_dim_symbolic(PrimaryDimTypes::Ij));
        loop_node
            .loop_count_symbol_ids
            .insert(PrimaryDimTypes::Ij, vec![VariableSymbol(7)]);
        assert!(loop_node.is_dim_symbolic(PrimaryDimTypes::Ij));
        assert!(!loop_node.is_dim_symbolic(PrimaryDimTypes::Mb));

        // ⛔ IT IS `count(dim)` ON THE MAP AND NOTHING ELSE, so it is independent of `hasLoopDim`: a
        // dim this loop does not even iterate reads symbolic once it has an entry.
        assert!(!loop_node.has_loop_dim(PrimaryDimTypes::Ki));
        loop_node
            .loop_count_symbol_ids
            .insert(PrimaryDimTypes::Ki, vec![VariableSymbol(9)]);
        assert!(loop_node.is_dim_symbolic(PrimaryDimTypes::Ki));

        // ⛔ The `operator[]` write at `:3001` leaves an EMPTY vector for a dim with no mapping, and
        // `isDimSymbolic` still answers yes — bridge 1's `DT_CHECK(size() == 1)` is what then fails.
        loop_node
            .loop_count_symbol_ids
            .insert(PrimaryDimTypes::Mb, Vec::new());
        assert!(loop_node.is_dim_symbolic(PrimaryDimTypes::Mb));
        assert!(loop_node.loop_count_symbol_ids[&PrimaryDimTypes::Mb].is_empty());
    }

    /// `util/utils.h:105-107`: `clone()` is the copy constructor, so all six declared fields come
    /// across and the two containers are copies, not shares. Each field is moved on the copy ALONE
    /// and one at a time — a `Clone` that dropped or shared any single one of them fails here, which
    /// a whole-object compare would not show.
    #[test]
    fn cloning_a_loop_carries_all_six_declared_fields_and_shares_none() {
        use crate::schedule::dims::MetaDimKind;

        let mut original = LoopNode::new(
            Some(DataStageId(4)),
            Some(DataStageId(5)),
            vec![PrimaryDimAndKind::new(
                PrimaryDimTypes::Ij,
                MetaDimKind::PadValid,
            )],
            true,
        );
        original.set_parametric_lds_idx(Some(LdsIdx(6)));
        original
            .loop_count_symbol_ids
            .insert(PrimaryDimTypes::Ij, vec![VariableSymbol(11)]);

        let copy = original.clone();
        assert_eq!(copy.num_id, Some(DataStageId(4)));
        assert_eq!(copy.den_id, Some(DataStageId(5)));
        assert_eq!(copy.dims, original.dims);
        assert_eq!(copy.loop_count_symbol_ids, original.loop_count_symbol_ids);
        assert!(copy.is_parametric_loop());
        assert_eq!(copy.parametric_lds_idx(), Some(LdsIdx(6)));

        let mut moved = copy.clone();
        moved.num_id = Some(DataStageId(40));
        assert_eq!(moved.num_id, Some(DataStageId(40)));
        assert_eq!(original.num_id, Some(DataStageId(4)));

        let mut moved = copy.clone();
        moved.den_id = None;
        assert_eq!(moved.den_id, None);
        assert_eq!(original.den_id, Some(DataStageId(5)));

        let mut moved = copy.clone();
        moved.dims.push(PrimaryDimTypes::Mb.into());
        assert_eq!(moved.dims.len(), 2);
        assert_eq!(original.dims.len(), 1);

        let mut moved = copy.clone();
        moved
            .loop_count_symbol_ids
            .insert(PrimaryDimTypes::Mb, vec![VariableSymbol(12)]);
        assert_eq!(moved.loop_count_symbol_ids.len(), 2);
        assert_eq!(original.loop_count_symbol_ids.len(), 1);

        let mut moved = copy.clone();
        moved.set_parametric_lds_idx(None);
        assert_eq!(moved.parametric_lds_idx(), None);
        assert_eq!(original.parametric_lds_idx(), Some(LdsIdx(6)));

        // `isParametricLoop_` is one-way (`dsc/dsc2.h:600`), so the only move available is marking a
        // loop the copy was taken from before.
        let mut unmarked = LoopNode::default();
        let unmarked_copy = unmarked.clone();
        unmarked.mark_as_parametric_loop();
        assert!(unmarked.is_parametric_loop());
        assert!(!unmarked_copy.is_parametric_loop());
    }

    /// The eleven operators in the authority's declaration order (`dsc/dscdefn.h:95-107`) paired
    /// with the spellings `EnumsConversion::condOpToString` gives them (`dsc/dscdefn.cpp:19-28`).
    const EVERY_COND_OP: [(CondOp, &str); 11] = [
        (CondOp::Eq, "eq"),
        (CondOp::Ne, "ne"),
        (CondOp::Lt, "lt"),
        (CondOp::Le, "le"),
        (CondOp::Gt, "gt"),
        (CondOp::Ge, "ge"),
        (CondOp::Toggle, "toggle"),
        (CondOp::Always, "always"),
        (CondOp::Never, "never"),
        (CondOp::Const, "const"),
        (CondOp::Default, "default"),
    ];

    /// `dsc/dscdefn.h:95-107` against `dsc/dscdefn.cpp:19-31`: the discriminants are positional,
    /// every operator has a spelling, and `flipMap` makes the pair a round trip.
    #[test]
    fn cond_op_spellings_round_trip_and_the_discriminants_are_the_authoritys() {
        assert_eq!(CondOp::ALL.len(), EVERY_COND_OP.len());
        for (i, (cond_op, spelling)) in EVERY_COND_OP.into_iter().enumerate() {
            assert_eq!(cond_op as usize, i, "{spelling} moved");
            assert_eq!(CondOp::ALL[i], cond_op);
            assert_eq!(cond_op.name(), spelling);
            assert_eq!(CondOp::from_name(spelling), Some(cond_op));
        }

        // `stringToCondOp` is `flipMap`ped, so a spelling it does not hold is absent rather than a
        // different operator — the miss the DDL conversion tests for at
        // `ddc/ddl/ddl_conversion.cpp:254-258`.
        assert_eq!(CondOp::from_name("=="), None);
        assert_eq!(CondOp::from_name("EQ"), None);
        assert_eq!(CondOp::from_name(""), None);

        // `LoopCond::condOp_`'s initialiser, `dsc/dsc2.h:661`.
        assert_eq!(CondOp::default(), CondOp::Default);
    }

    /// `ddc/ddl/ddl_conversion.cpp:260-264` and
    /// `dsc-based-utils/DSC2ToDataflowIR/V3/SNControlFlowLowering.cpp:23-44` state the same set
    /// twice, independently: exactly the six relational operators reach a `LoopCond` on this path.
    #[test]
    fn only_the_six_relational_cond_ops_survive_both_ends_of_the_path() {
        assert_eq!(
            CondOp::COMPARISONS,
            [
                CondOp::Eq,
                CondOp::Ne,
                CondOp::Lt,
                CondOp::Le,
                CondOp::Gt,
                CondOp::Ge
            ]
        );

        // The five the DDL rejects and `getCmpIPredicate_dup` fails on are the rest of `ALL`, so the
        // two lists together account for every operator and neither grew a member of the other.
        let refused: Vec<CondOp> = CondOp::ALL
            .into_iter()
            .filter(|op| !CondOp::COMPARISONS.contains(op))
            .collect();
        assert_eq!(
            refused,
            vec![
                CondOp::Toggle,
                CondOp::Always,
                CondOp::Never,
                CondOp::Const,
                CondOp::Default
            ]
        );

        // ⛔ The default is one of the refused ones: a `LoopCond` that nobody filled cannot be
        // lowered, which is why `dsc/dsc2.h:661` is a "not chosen yet" marker and not an operator.
        assert!(!CondOp::COMPARISONS.contains(&CondOp::default()));
    }

    /// `dsc/dsc2.h:655` and `:662` against `dsc/dsc2.cpp:21-25` and
    /// `ddc/ddl/ddl_conversion.cpp:266-274`.
    #[test]
    fn cond_val_type_defaults_to_int_and_an_unknown_spelling_is_the_ddls_integer_case() {
        assert_eq!(CondValType::default(), CondValType::Int);
        assert_eq!(
            CondValType::ALL,
            [CondValType::Int, CondValType::First, CondValType::Last]
        );

        for (value_type, spelling) in [
            (CondValType::Int, "int"),
            (CondValType::First, "first"),
            (CondValType::Last, "last"),
        ] {
            assert_eq!(value_type.name(), spelling);
            assert_eq!(CondValType::from_name(spelling), Some(value_type));
        }

        // ⭐ THE MISS IS THE LIVE PATH, not an error: `processExpression` parses the spelling as an
        // integer whenever `stringToCondValType.find` fails (`ddc/ddl/ddl_conversion.cpp:268-274`),
        // and the result is an `INT` condition rather than a rejected DDL.
        assert_eq!(CondValType::from_name("4"), None);
        assert_eq!(CondValType::from_name("chunk_count-1"), None);
        assert_eq!(CondValType::from_name("INT"), None);
    }

    /// `dsc/dsc2.h:1138` and its `print` switch at `:1151-1167`.
    #[test]
    fn loop_distribution_cat_print_spellings_carry_the_authoritys_misspelling() {
        for (i, cat) in LoopDistributionCat::ALL.into_iter().enumerate() {
            assert_eq!(cat as usize, i);
        }

        // ⛔ `"Unknwon"` is the authority's, `dsc/dsc2.h:1153`.
        assert_eq!(LoopDistributionCat::Unknown.name(), "Unknwon");
        assert_eq!(LoopDistributionCat::AboveChunk.name(), "Above_chunk");
        assert_eq!(LoopDistributionCat::BelowChunk.name(), "Below_chunk");
        assert_eq!(LoopDistributionCat::CoreletSlice.name(), "Corelet_slice");
    }

    /// `dsc/dsc2.cpp:6612-6627` and `:6545-6561` against `dsc/dims.h:79-81`: the shape of the
    /// synthetic `CORELET_SLICE` chain entry that `e021_LoopDistributionInfo` has to be able to hold.
    #[test]
    fn the_corelet_slice_chain_entry_carries_its_split_dim_unpadded() {
        use crate::schedule::dims::MetaDimKind;

        // ⛔ THE WRITER NEVER SPELLS A KIND — `dsc/dsc2.cpp:6620-6622` pushes a bare
        // `PrimaryDimTypes` into a `PrimaryDimAndKind` slot — so the entry's kind is whatever IBM's
        // non-`explicit` constructor supplies, and a port that had to name one at the callsite would
        // be guessing.
        let split_dim = PrimaryDimTypes::Mb;
        assert_eq!(
            PrimaryDimAndKind::from(split_dim),
            PrimaryDimAndKind::new(split_dim, MetaDimKind::Unpadded)
        );

        // ⛔ AND `Unpadded` IS LOAD-BEARING: `isLoopDimRelated` matches on the dim alone
        // (`dsc/dsc2.cpp:6550-6551`) EXCEPT for a `WindowDim` entry, which ALSO matches a different
        // dim through the padding window of `loop->denId_` (`:6554-6559`). A synthetic entry minted
        // `WindowDim` would be collected for dims it has nothing to do with, and
        // `ddc/ddc_fold.cpp:2543-2557` searches exactly that collection for it — erroring out when
        // it is missing.
        assert_ne!(
            PrimaryDimAndKind::from(split_dim),
            PrimaryDimAndKind::new(split_dim, MetaDimKind::WindowDim)
        );
    }

    /// `dsc/dsc2.cpp:4364-4383` over the set at `dsc/dscdefn.cpp:142-144`.
    #[test]
    fn the_memory_questions_test_the_storage_half_and_take_the_first_non_memory_dst() {
        // ⛔ `unit` IS NEVER TESTED: a read by the LXLU (a unit) out of LX (a memory) has a memory
        // source, and swapping the two halves flips the answer.
        let lxlu_reads_lx = DataLocation {
            unit: SenComponent::Lxlu,
            storage: SenComponent::Lx,
        };
        let mut node = TransferNode {
            src: lxlu_reads_lx,
            ..TransferNode::default()
        };
        assert!(!node.has_non_memory_source());
        node.src = lxlu_reads_lx.swapped();
        assert!(node.has_non_memory_source());

        // `dsc/dsc2Pcfg.cpp:1039` tests `storage_ != LATCH` beside this very call, so LATCH is a
        // storage that reaches it and is not in the set.
        node.src = DataLocation {
            unit: SenComponent::Lxlu,
            storage: SenComponent::Latch,
        };
        assert!(node.has_non_memory_source());

        // No destination at all is "no non-memory result" — the `-1` both callers test for
        // (`ddc/ddc_transformation.cpp:1570-1572`, `:1818-1819`).
        assert_eq!(node.non_memory_result_index(), None);
        assert!(!node.has_non_memory_result());

        let landing_in = |storage| DstVia {
            loc: DataLocation {
                unit: SenComponent::Lxlu,
                storage,
            },
            ..DstVia::default()
        };
        node.dst_vias = vec![
            landing_in(SenComponent::Lx),
            landing_in(SenComponent::Latch),
            landing_in(SenComponent::Constant),
        ];

        // ⭐ THE FIRST MATCH, not any match: `getNonMemoryResultIndex` returns inside the loop
        // (`dsc/dsc2.cpp:4376-4383`), and `hoistTransfersUpForReuse` indexes another vector with it.
        assert_eq!(node.non_memory_result_index(), Some(1));
        assert!(node.has_non_memory_result());
        assert!(!node.check_non_memory_result_index(0));
        assert!(node.check_non_memory_result_index(1));
        assert!(node.check_non_memory_result_index(2));

        // Every component IBM lists is a memory, and these four are not.
        for storage in MEMORIES {
            node.src = DataLocation {
                unit: SenComponent::Lxlu,
                storage,
            };
            assert!(!node.has_non_memory_source());
        }
        for storage in [
            SenComponent::Latch,
            SenComponent::Constant,
            SenComponent::Zero,
            SenComponent::Lxlu,
        ] {
            assert!(!MEMORIES.contains(&storage));
        }
    }

    /// `dsc/dsc2.h:877` and `:879-883`, and the route bridge 1 walks at
    /// `dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:2631-2639`.
    #[test]
    fn indirect_is_the_unit_half_being_set_and_via_runs_source_to_destination() {
        let mut node = TransferNode::default();
        assert!(!node.is_src_indirect());

        // ⛔ THE `unit` HALF — the opposite half of the same type from the memory questions above.
        node.src_indirect = DataLocation {
            unit: SenComponent::Lxlu,
            storage: SenComponent::NoComponent,
        };
        assert!(node.is_src_indirect());

        node.dst_vias = vec![
            DstVia::default(),
            DstVia {
                loc_indirect: node.src_indirect,
                via: vec![SenComponent::Lxlu, SenComponent::Ptxrf],
                ..DstVia::default()
            },
        ];
        assert!(!node.is_dst_indirect_at_index(0));
        assert!(node.is_dst_indirect_at_index(1));

        // `via_.front()` is the first hop out of the source and `.back()` the last unit before
        // `loc_` (`SNTransferLowering.cpp:2530`, `:2572`); empty means the destination's own unit.
        assert_eq!(node.dst_vias[1].via.first(), Some(&SenComponent::Lxlu));
        assert_eq!(node.dst_vias[1].via.last(), Some(&SenComponent::Ptxrf));
        assert!(node.dst_vias[0].via.is_empty());
    }

    /// The authority's member initialisers, `dsc/dsc2.h:834`, `:837`, `:839`, `:817`.
    #[test]
    fn a_default_transfer_node_carries_the_authoritys_initialisers() {
        let node = TransferNode::default();
        assert_eq!(node.replication_factor, 1);
        assert_eq!(node.unit_time_transfer_num_chunks, 1);
        assert_eq!(node.rotate_num_elements, 0);
        assert_eq!(node.src, DataLocation::UNSET);
        assert_eq!(DstVia::default().loc_indirect, DataLocation::UNSET);

        // ⛔ EMPTY, AND IT STAYS EMPTY ON EVERY PATH BUT THE JSON IMPORTER'S: the one C++ writer is
        // inside a lambda whose only callsite is commented out (`ddc/ddcv1.cpp:1640`, `:1649-1650`).
        assert!(node.unit_time_transfer_chunk_stride.is_empty());

        // ⛔ EMPTY ONLY AS AN INITIALISER — THE JUSTIFICATION ABOVE DOES NOT EXTEND TO THIS FIELD.
        // It has a live writer, `populateUnitTimeTransfers` (`ddc/ddcv1.cpp:524-525`), and three
        // live rewriters; what stays empty is the default, not the field.
        assert!(node.unit_time_transfer_chunk_size.is_empty());
    }

    /// `util/utils.h:105-106` reaching [`TransferPadInfo`]'s do-nothing copy constructor
    /// (`dsc/dsc2.h:761-764`) — the three `DT_CHECK_MSG(newNode->paddingInfo_.isEmpty())` the
    /// authority puts after its two `TransferNode::clone()` calls (`dsc/dsc2.cpp:5445`, `:5692`,
    /// `:5792`), as a property of the type rather than a check at runtime.
    #[test]
    fn cloning_a_transfer_node_drops_its_padding_info_and_copies_everything_else() {
        let mut node = TransferNode {
            replication_factor: 8,
            rotate_num_elements: 16,
            dst_vias: vec![DstVia::default(), DstVia::default()],
            repetition: TransferRepetition {
                src: Some(Repetition(2)),
                dsts: vec![Repetition(3), Repetition(4)],
            },
            ..TransferNode::default()
        };
        node.transfer_size.insert(PrimaryDimTypes::X, DimSize(64));
        assert_eq!(
            node.padding_info.build_pad_sizes(
                PadEnd::Front,
                PrimaryDimTypes::X,
                [FoldDimSize(2), FoldDimSize(3)],
                [PadSize(-40), PadSize(-10)],
                [PadSize(25), PadSize(0)],
            ),
            Some(())
        );
        assert!(!node.padding_info.is_empty());

        let clone = node.clone();

        // ⛔ THE ONE FIELD THE COPY CONSTRUCTOR DROPS, and the source keeps its own.
        assert!(clone.padding_info.is_empty());
        assert!(!node.padding_info.is_empty());

        // Every other member is a plain member-wise copy.
        assert_eq!(clone.replication_factor, 8);
        assert_eq!(clone.rotate_num_elements, 16);
        assert_eq!(clone.dst_vias, node.dst_vias);
        assert_eq!(clone.repetition, node.repetition);
        assert_eq!(clone.transfer_size, node.transfer_size);
    }

    /// `dsc/dsc2.h:826-829` and its only two writers, which append one `dstReps_` entry per
    /// `dstVias_` entry in the same loop iteration (`ddc/ddl/ddl_conversion.cpp:1171-1189`).
    #[test]
    fn the_transfer_repetition_starts_absent_and_is_indexed_like_the_destinations() {
        // ⛔ ABSENT, NOT `Repetition(1)`: `int srcRep_;` carries no member initialiser and
        // `TransferNode()` does not name `repetition_`, so the authority's default-constructed node
        // reads uninitialised storage here (`dsc/dsc2.h:827`, `:815`) — unlike `dstReps_`, whose
        // `std::vector` default IS empty.
        assert_eq!(
            TransferNode::default().repetition,
            TransferRepetition::default()
        );
        assert_eq!(TransferRepetition::default().src, None);
        assert!(TransferRepetition::default().dsts.is_empty());

        let landing_in = |storage| DstVia {
            loc: DataLocation {
                unit: SenComponent::Lxlu,
                storage,
            },
            ..DstVia::default()
        };
        let node = TransferNode {
            dst_vias: vec![
                landing_in(SenComponent::Lx),
                landing_in(SenComponent::Latch),
            ],
            repetition: TransferRepetition {
                src: Some(Repetition(1)),
                dsts: vec![Repetition(1), Repetition(4)],
            },
            ..TransferNode::default()
        };

        // ⭐ ONE POSITION SPEAKS FOR EVERY DESTINATION VECTOR: `non_memory_result_index` is an index
        // into `dstVias_`, and `hoistTransfersUpForReuse` reads the vector emplaced beside it with
        // that very index (`ddc/ddc_transformation.cpp:1570-1575`).
        let fifo = node
            .non_memory_result_index()
            .expect("the LATCH destination");
        assert_eq!(node.repetition.dsts.len(), node.dst_vias.len());
        assert_eq!(node.repetition.dsts[fifo], Repetition(4));
    }

    /// `dsc/dsc2.h:932-941` and `:950-954` — every default member initializer of a compute node,
    /// including the nested `InstrAttribute`'s eight slices and all-on compute mask.
    #[test]
    fn compute_node_defaults_are_the_authoritys_initializers() {
        let node = ComputeNode::default();
        assert_eq!(node.ex_unit, SenComponent::NoComponent);
        assert_eq!(node.r#type, ComputeOpType::Count);
        assert_eq!(node.data_format, DataFormats::Sen169Fp16);
        assert_eq!(node.num_folds_engaged, NumFoldsEngaged(1));
        assert!(!node.is_opaque_op);
        assert!(node.inputs.is_empty());
        assert!(node.outputs.is_empty());

        // `dsc/dsc2.h:907`, `:915-916`.
        assert_eq!(node.instr_attribute.repetition, Repetition(8));
        assert_eq!(node.instr_attribute.compute_mask, ComputeMask(255));
        assert_eq!(node.instr_attribute.mode, None);
        assert!(!node.instr_attribute.sign_extend);
        assert!(node.instr_attribute.indices.is_empty());
        assert!(node.instr_attribute.param_map.is_empty());
        assert!(node.instr_attribute.input_data_connects.is_empty());
        assert_eq!(node.repetition_with_offset, RepetitionWithOffset::default());
    }

    /// `dsc/dsc2.cpp:2291-2346`. The vendor's own dispatch: PTWEST and the memories split at
    /// RCUDD1A, PTNORTH does not, PELRF is a memory this rule excludes, `MACC` alone reads the data
    /// format, and the output element count comes LAST.
    #[test]
    fn operand_sizes_split_at_rcudd1a_and_the_output_comes_last() {
        let fma16 = ComputeNode {
            r#type: ComputeOpType::Fma16,
            inputs: vec![
                SenComponent::Ptwest,
                SenComponent::Lx,
                SenComponent::Ptnorth,
                SenComponent::Pelrf,
            ],
            num_folds_engaged: NumFoldsEngaged(2),
            ..ComputeNode::default()
        };

        // PTWEST 8, LX 64, PTNORTH 64, PELRF the 1024/16 default, output 64 — all doubled by the
        // two folds (`:2333`, `:2344`).
        assert_eq!(
            fma16.operand_sizes(Gen::Rcudd1a),
            Some(vec![
                OperandSize(16),
                OperandSize(128),
                OperandSize(128),
                OperandSize(128),
                OperandSize(128)
            ])
        );
        // SEN1P5 lifts PTWEST to 32 and LX to 256; PTNORTH and the default do not move.
        assert_eq!(
            fma16.operand_sizes(Gen::Sen1p5),
            Some(vec![
                OperandSize(64),
                OperandSize(512),
                OperandSize(128),
                OperandSize(128),
                OperandSize(128)
            ])
        );

        // ⛔ `MACC` IS DISPATCHED BY ITS FORMAT (`:2317-2331`): fp8 takes the FMA8 sizes.
        let macc_fp8 = ComputeNode {
            r#type: ComputeOpType::Macc,
            data_format: DataFormats::Sen143Fp8,
            inputs: vec![SenComponent::Lx],
            ..ComputeNode::default()
        };
        assert_eq!(
            macc_fp8.operand_sizes(Gen::Rcudd1a),
            Some(vec![OperandSize(128), OperandSize(64)])
        );
        assert_eq!(
            macc_fp8.operand_sizes(Gen::Sen1p5),
            Some(vec![OperandSize(1024), OperandSize(64)])
        );

        // An fp32 `MACC` matches no size rule, so the input takes the 1024/32 default and the
        // output is 32 rather than 64 (`:2295-2296`, `:2338-2342`).
        let macc_fp32 = ComputeNode {
            r#type: ComputeOpType::Macc,
            data_format: DataFormats::IeeeFp32,
            inputs: vec![SenComponent::Lx],
            ..ComputeNode::default()
        };
        assert_eq!(
            macc_fp32.operand_sizes(Gen::Sen1p5),
            Some(vec![OperandSize(32), OperandSize(32)])
        );
    }

    /// The two places the authority stops: a PTWEST input it cannot size (`dsc/dsc2.cpp:2308-2312`)
    /// and the -1 bit width of `INVALID` (`:2295`).
    #[test]
    fn operand_sizes_are_absent_where_the_authority_refuses() {
        // ⛔ "This case is not correctly handled at the moment" — a PTWEST `MACC`.
        let ptwest_macc = ComputeNode {
            r#type: ComputeOpType::Macc,
            inputs: vec![SenComponent::Ptwest],
            ..ComputeNode::default()
        };
        assert_eq!(ptwest_macc.operand_sizes(Gen::Rcudd1a), None);
        assert_eq!(ptwest_macc.operand_sizes(Gen::Sen1p5), None);

        // ⛔ "Unexpected PT operation in view calculation" — anything outside the five FMA/IMA ops.
        let ptwest_fcmp = ComputeNode {
            r#type: ComputeOpType::Fcmp,
            inputs: vec![SenComponent::Ptwest],
            ..ComputeNode::default()
        };
        assert_eq!(ptwest_fcmp.operand_sizes(Gen::Sen1p5), None);

        // ⛔ `INVALID` divides 1024 by -1 in the authority; a negative operand size is not one.
        let invalid_format = ComputeNode {
            r#type: ComputeOpType::Fmul,
            data_format: DataFormats::Invalid,
            inputs: vec![SenComponent::Lx],
            ..ComputeNode::default()
        };
        assert_eq!(invalid_format.operand_sizes(Gen::Sen1p5), None);

        // ⭐ WITH NO INPUTS THE AUTHORITY NEVER DIVIDES, so the output alone survives even on
        // `INVALID` — the bit width is read per input (`:2294-2295`).
        let no_inputs = ComputeNode {
            data_format: DataFormats::Invalid,
            ..ComputeNode::default()
        };
        assert_eq!(
            no_inputs.operand_sizes(Gen::Sen1p5),
            Some(vec![OperandSize(64)])
        );
    }

    /// ⛔ THE AUTHORITY'S OWN PACK/MERGE TABLE AND WHAT ITS EXPANSION DOES TO THE -1 SLOTS. `pack12`
    /// is `{0, -1, 1, -1, ...}` (`ddc/transformations/automatic_shuffle/shuffle.cpp:34-35`) and
    /// reaches `insert_packmerge` through `unary_op`, which omits the expand argument and so takes
    /// its `true` default (`:434`, `:438`, `:167`, `shuffle.h:194`). `expand_indices` then scales
    /// EVERY entry as `compact_indices[i] * scale + j` with no -1 guard
    /// (`ddc/ddc_transformation.cpp:1947-1952`). At 4-bit elements `scale` is 2, so ONE -1 slot
    /// becomes TWO DIFFERENT negative entries — which is why [`InstrAttribute::indices`] is a plain
    /// [`PackMergeIndex`] and not an `Option`.
    #[test]
    fn a_packmerge_index_is_not_a_sentinel_because_the_expansion_scales_it() {
        let pack12 = [0, -1, 1, -1, 2, -1, 3, -1, 4, -1, 5, -1, 6, -1, 7, -1];

        // `ddc/ddc_transformation.cpp:1941-1947`: 128 bits per slice over the entry count, then
        // over the input element's width.
        let entry_bits = 128 / pack12.len() as i32;
        let scale = entry_bits / DataFormats::Senint4.bit_width().unwrap().0;
        assert_eq!(scale, 2);

        let attr = InstrAttribute {
            indices: pack12
                .iter()
                .flat_map(|index| (0..scale).map(move |j| PackMergeIndex(index * scale + j)))
                .collect(),
            ..InstrAttribute::default()
        };

        // ⛔ THE HOLE EXPANDED INTO TWO UNEQUAL HOLES, and an `Option` cannot hold either of them.
        assert_eq!(attr.indices.len(), 32);
        assert_eq!(attr.indices[0], PackMergeIndex(0));
        assert_eq!(attr.indices[1], PackMergeIndex(1));
        assert_eq!(attr.indices[2], PackMergeIndex(-2));
        assert_eq!(attr.indices[3], PackMergeIndex(-1));
        assert_ne!(attr.indices[2], attr.indices[3]);

        // ⭐ AND THE LENGTH IS LOAD-BEARING: `expand_indices` reads it back to derive the width
        // (`:1942`), so a representation that dropped or merged slots would change every entry.
        assert_eq!(128 / attr.indices.len() as i32, 4);
    }

    /// `dsc/dsc2.cpp:2297-2332`, every row of both size tables at both sides of the RCUDD1A split —
    /// the PTWEST rows straight, the memory rows also through the `MACC` format pairing that shares
    /// each row, and PELRF/SFPLRF proving the exclusion at `:2315`.
    #[test]
    fn every_operand_size_row_holds_at_both_sides_of_the_rcudd1a_split() {
        // `:2298-2307`. FMA4 is the one row with no split — "only available from SEN1P5".
        let ptwest = [
            (ComputeOpType::Fma16, 8, 32),
            (ComputeOpType::Fma8, 16, 128),
            (ComputeOpType::Ima8, 32, 128),
            (ComputeOpType::Fma4, 256, 256),
            (ComputeOpType::Ima4, 64, 256),
        ];
        for (r#type, below, above) in ptwest {
            let node = ComputeNode {
                r#type,
                inputs: vec![SenComponent::Ptwest],
                ..ComputeNode::default()
            };
            let output = OperandSize(64);
            for (arch, want) in [
                (Gen::Mpw2, below),
                (Gen::Rcudd1a, below),
                (Gen::Sen1p5, above),
            ] {
                assert_eq!(
                    node.operand_sizes(arch),
                    Some(vec![OperandSize(want), output]),
                    "PTWEST {:?} at {:?}",
                    r#type,
                    arch
                );
            }
        }

        // `:2316-2331`. Each row is reached BOTH by its op and by `MACC` at the row's format, and
        // every `above` here differs from the `1024 / bit_width` default the row displaces.
        let memory = [
            (ComputeOpType::Fma8, DataFormats::Sen143Fp8, 128, 1024),
            (ComputeOpType::Ima8, DataFormats::Senint8, 256, 1024),
            (ComputeOpType::Ima4, DataFormats::Senint4, 512, 2048),
            (ComputeOpType::Fma16, DataFormats::Sen169Fp16, 64, 256),
            (ComputeOpType::Fma4, DataFormats::Sen121Fp4, 2048, 2048),
        ];
        for (r#type, format, below, above) in memory {
            let default = OperandSize(1024 / format.bit_width().unwrap().0);
            assert_ne!(
                OperandSize(above),
                default,
                "{:?} row is the default",
                r#type
            );
            for op in [r#type, ComputeOpType::Macc] {
                let node = ComputeNode {
                    r#type: op,
                    data_format: format,
                    inputs: vec![SenComponent::Lx, SenComponent::Pelrf, SenComponent::Sfplrf],
                    ..ComputeNode::default()
                };
                // ⛔ PELRF AND SFPLRF ARE MEMORIES THE ROW EXCLUDES, so they take the default.
                for (arch, want) in [(Gen::Rcudd1a, below), (Gen::Sen1p5, above)] {
                    assert_eq!(
                        node.operand_sizes(arch),
                        Some(vec![OperandSize(want), default, default, OperandSize(64)]),
                        "{:?}/{:?} at {:?}",
                        op,
                        format,
                        arch
                    );
                }
            }
        }
    }

    /// `dsc/dscdefn.h:134-207` against `dsc/dscdefn.cpp:33-107`: 70 of the 71 ops have a spelling,
    /// and `flipMap` makes each one a round trip.
    #[test]
    fn compute_op_type_spellings_round_trip_and_only_fcvt_has_none() {
        let mut unspelled = Vec::new();
        for op in ComputeOpType::ALL {
            match op.name() {
                Some(name) => assert_eq!(ComputeOpType::from_name(name), Some(op), "{name}"),
                None => unspelled.push(op),
            }
        }
        // ⛔ `computeTypeToString.at(FCVT)` throws; it is the map's one hole.
        assert_eq!(unspelled, vec![ComputeOpType::Fcvt]);
        assert_eq!(ComputeOpType::from_name("fcvt"), None);

        // ⛔ The authority spells `EQUALTO` `"equal"` (`dsc/dscdefn.cpp:85`).
        assert_eq!(ComputeOpType::Equalto.name(), Some("equal"));
        assert_eq!(ComputeOpType::from_name("equalto"), None);

        // `COUNT` is `type_`'s initialiser and a spelled value, not a count sentinel.
        assert_eq!(ComputeOpType::default(), ComputeOpType::Count);
        assert_eq!(ComputeOpType::Count.name(), Some("undefined"));
        assert_eq!(ComputeOpType::from_name(""), None);
    }

    /// `util/sendefs/sendefs.h:30-54` against `sendefs.cpp:18-67` and `:129-141`.
    #[test]
    fn data_format_widths_are_the_authoritys_table_and_unknown_text_is_invalid() {
        assert_eq!(DataFormats::ALL.len(), DataFormats::COUNT);
        for format in DataFormats::ALL {
            assert_eq!(DataFormats::from_name(format.name()), format);
        }
        assert_eq!(DataFormats::default(), DataFormats::Sen169Fp16);

        // ⛔ SENINT24 IS 16 BITS in the table (`sendefs.cpp:135`).
        assert_eq!(DataFormats::Senint24.bit_width(), Some(BitWidth(16)));
        assert_eq!(DataFormats::Sen169Fp16.bit_width(), Some(BitWidth(16)));
        assert_eq!(DataFormats::IeeeFp32.bit_width(), Some(BitWidth(32)));
        assert_eq!(DataFormats::Sen121Fp4.bit_width(), Some(BitWidth(4)));
        assert_eq!(DataFormats::Sen153Fp9.bit_width(), Some(BitWidth(9)));
        // ⛔ The `-1` entry is not a width (`sendefs.cpp:131`).
        assert_eq!(DataFormats::Invalid.bit_width(), None);

        // ⭐ `FromString`'s own `else`: unrecognised text is INVALID rather than absent
        // (`util/sendefs/sendefs.h:294-296`).
        assert_eq!(DataFormats::from_name("fp16"), DataFormats::Invalid);
        assert_eq!(DataFormats::from_name(""), DataFormats::Invalid);
    }

    /// `dsc/dscdefn.cpp:142-144`. The set `getComputeOperandSizes` tests against, and it is not
    /// `ddc::memories` (`ddc/ddc_metadata.h:20-21`).
    #[test]
    fn memories_is_dsc2s_sixteen_and_holds_the_two_the_size_rule_excludes_by_name() {
        assert_eq!(MEMORIES.len(), 16);
        assert!(MEMORIES.contains(&SenComponent::Lx));
        assert!(MEMORIES.contains(&SenComponent::Hbm));

        // ⛔ BOTH ARE MEMORIES, which is exactly why `dsc/dsc2.cpp:2315` excludes them by name
        // instead of relying on the set.
        assert!(MEMORIES.contains(&SenComponent::Pelrf));
        assert!(MEMORIES.contains(&SenComponent::Sfplrf));

        // The two PT endpoints have their own size rules and are not in the set at all.
        assert!(!MEMORIES.contains(&SenComponent::Ptwest));
        assert!(!MEMORIES.contains(&SenComponent::Ptnorth));

        // ⛔ The eight `ddc::memories` does NOT hold — reading the ddc set here would size a
        // register-file operand as the 1024/bitWidth default.
        for extra in [
            SenComponent::L0Scale,
            SenComponent::Lrfreg,
            SenComponent::L3luibr,
            SenComponent::L3suibr,
            SenComponent::Pestate,
            SenComponent::Sfpstate,
            SenComponent::Lxluscalereg,
            SenComponent::Qgi,
        ] {
            assert!(MEMORIES.contains(&extra), "{extra:?}");
        }
    }

    /// `dsc/dsc2.h:693-695`: the discriminator reads the LOOP side alone. So a node carrying NEITHER
    /// guard is a core/corelet condition selecting no core — the state the DDL resolves to the
    /// CONSTANT FALSE before any node is minted (`ddc/ddl/ddl_conversion.cpp:438-441`) — and a node
    /// carrying BOTH is a loop condition whose `coreClCond_` every gated reader skips
    /// (`dsc/dsc2.cpp:2663`, `ddc/ddc_transformation_util.cpp:489`,
    /// `ddc/ddl/ddl_conversion.cpp:3248`).
    #[test]
    fn a_condition_nodes_discriminator_reads_the_loop_guard_alone() {
        let mut node = ConditionNode::default();
        assert!(node.loop_cond.is_none() && node.core_cl_cond.is_empty());
        assert!(node.has_core_cl_cond());

        node.core_cl_cond.insert(CoreId(0), BTreeSet::new());
        node.core_cl_cond
            .insert(CoreId(1), BTreeSet::from([CoreletId(0)]));
        assert!(node.has_core_cl_cond());

        node.loop_cond = Some(
            LoopCondConjunction::new(LoopCond {
                dim: PrimaryDimTypes::Y,
                cond_op: LoopCondOp::Eq,
                cond_val: CondVal::Last,
            })
            .into(),
        );
        assert!(!node.has_core_cl_cond());

        node.core_cl_cond.clear();
        assert!(!node.has_core_cl_cond());
    }

    /// `dsc/dsc2.h:964-972`, and the ordering divergence on `units_`: the node's JSON array
    /// (`dsc/dsc2.cpp:814-819`) and the DDL export (`ddc/ddl/ddl_conversion.cpp:3319-3325`) print
    /// this set in libstdc++ bucket order, and in `SenComponent` declaration order here.
    #[test]
    fn sync_node_units_iterate_in_component_declaration_order() {
        let node = SyncNode {
            units: BTreeSet::from([
                SenComponent::L3lu,
                SenComponent::Sfp,
                SenComponent::Ring,
                SenComponent::Pt,
            ]),
            ..SyncNode::default()
        };
        assert!(!node.is_receive);
        assert!(!node.is_soft);
        assert_eq!(
            node.units.iter().copied().collect::<Vec<_>>(),
            [
                SenComponent::Sfp,
                SenComponent::Pt,
                SenComponent::Ring,
                SenComponent::L3lu
            ]
        );
    }

    /// A SAMV node whose stick is `layout` and whose first masked coordinate per dim is
    /// `first_masked` (`ddc/ddcv1.cpp:3546-3547`, `:3589-3597`).
    fn samv(
        layout: &[(PrimaryDimTypes, i32)],
        first_masked: &[(PrimaryDimTypes, i32)],
    ) -> StickMaskNode {
        StickMaskNode {
            stick_layout: layout
                .iter()
                .map(|&(dim, size)| Size::new(dim, DimSize(size)))
                .collect(),
            first_stick_coord_to_mask_per_dim: first_masked
                .iter()
                .map(|&(dim, coord)| (dim, StickCoord(coord)))
                .collect(),
            ..StickMaskNode::default()
        }
    }

    /// `dsc/dsc2.cpp:2440-2491` over a two-dim stick — `Out` within the slice, `In` across the eight.
    /// Masking `In` from coordinate 5 transitions in slice 2 and scales maskB by the wsl extent;
    /// masking `Out` instead leaves the cross-slice dim whole, so maskB masks nothing and the
    /// transition is the last slice.
    #[test]
    fn stick_mask_view_splits_the_masked_dim_and_scales_it_by_the_other() {
        let layout = [(PrimaryDimTypes::Out, 4), (PrimaryDimTypes::In, 16)];
        assert_eq!(
            samv(&layout, &[(PrimaryDimTypes::In, 5)]).view(),
            Some(StickMaskView {
                mask_a: MaskSplit {
                    unmasked: MaskElements(4),
                    masked: MaskElements(0),
                },
                mask_b: MaskSplit {
                    unmasked: MaskElements(4),
                    masked: MaskElements(4),
                },
                transition_slice: SliceId(2),
            })
        );
        assert_eq!(
            samv(&layout, &[(PrimaryDimTypes::Out, 3)]).view(),
            Some(StickMaskView {
                mask_a: MaskSplit {
                    unmasked: MaskElements(3),
                    masked: MaskElements(1),
                },
                mask_b: MaskSplit {
                    unmasked: MaskElements(8),
                    masked: MaskElements(0),
                },
                transition_slice: SliceId(SLICES_PER_STICK - 1),
            })
        );
    }

    /// The three "SAMV not possible with current stick layout" refusals — over three dims, a second
    /// within-slice dim, and a cross-slice extent under eight (`dsc/dsc2.cpp:2442-2463`) — plus the
    /// empty layout the authority reads off the end of (`:2446`).
    #[test]
    fn stick_mask_view_is_absent_where_the_layout_defeats_masking() {
        let unmasked: [(PrimaryDimTypes, i32); 0] = [];
        assert_eq!(samv(&[], &unmasked).view(), None);
        assert_eq!(
            samv(
                &[
                    (PrimaryDimTypes::Mb, 2),
                    (PrimaryDimTypes::Out, 2),
                    (PrimaryDimTypes::Y, 2),
                    (PrimaryDimTypes::In, 16),
                ],
                &unmasked,
            )
            .view(),
            None
        );
        assert_eq!(
            samv(
                &[
                    (PrimaryDimTypes::Out, 4),
                    (PrimaryDimTypes::Mb, 2),
                    (PrimaryDimTypes::In, 16),
                ],
                &unmasked,
            )
            .view(),
            None
        );
        assert_eq!(
            samv(
                &[(PrimaryDimTypes::Out, 4), (PrimaryDimTypes::In, 4)],
                &unmasked,
            )
            .view(),
            None
        );
    }

    /// `dsc/dsc2.cpp:2469-2472`, `:2486-2489`: a THREE-entry layout is the "xsl inner" case, which
    /// scales maskA by the per-slice cross-slice extent instead of scaling maskB by the whole
    /// within-slice one. Same masked dim and same coordinate as the two-dim vector above, so
    /// transposing the two scales would answer with that test's `{4, 0}` / `{4, 4}` and fail here.
    #[test]
    fn stick_mask_view_scales_mask_a_when_the_cross_slice_dim_is_inner() {
        let layout = [
            (PrimaryDimTypes::In, 4),
            (PrimaryDimTypes::Out, 4),
            (PrimaryDimTypes::In, 4),
        ];
        assert_eq!(
            samv(&layout, &[(PrimaryDimTypes::In, 5)]).view(),
            Some(StickMaskView {
                mask_a: MaskSplit {
                    unmasked: MaskElements(8),
                    masked: MaskElements(0),
                },
                mask_b: MaskSplit {
                    unmasked: MaskElements(1),
                    masked: MaskElements(1),
                },
                transition_slice: SliceId(2),
            })
        );
    }

    /// `dsc/dsc2.cpp:2478-2484`: a cross-slice coordinate ON a slice boundary leaves remainder 0, so
    /// maskB spans the whole slice and the transition names the slice BEFORE the quotient. That is
    /// the encoding the DCC decodes by reading a fully-valid count back out of a zero
    /// (`dcc/src/Conversion/AgenToSentient/Helper.cpp:2679-2700`), not an off-by-one.
    ///
    /// ⛔ AND AT COORDINATE 0 IT IS `-1`, which bridge 1 turns into the eight `(1)` slices of the
    /// full-mask pattern (`SNStickMaskLowering.cpp:51-58`) while still attaching both masks
    /// (`:74-75`) — and the DCC's full-mask path accepts it only with no masks attached
    /// (`Helper.cpp:2607-2612`). Reachable: the DDC refuses only `numMaskedElem > size`, so an
    /// entirely masked dim mints coordinate 0 (`ddc/ddcv1.cpp:3589-3596`).
    #[test]
    fn stick_mask_view_puts_a_boundary_coordinate_on_the_previous_slice() {
        let layout = [(PrimaryDimTypes::Out, 4), (PrimaryDimTypes::In, 16)];
        let at_slice = |slice: i32| StickMaskView {
            mask_a: MaskSplit {
                unmasked: MaskElements(4),
                masked: MaskElements(0),
            },
            mask_b: MaskSplit {
                unmasked: MaskElements(0),
                masked: MaskElements(8),
            },
            transition_slice: SliceId(slice),
        };
        assert_eq!(
            samv(&layout, &[(PrimaryDimTypes::In, 4)]).view(),
            Some(at_slice(1))
        );
        assert_eq!(
            samv(&layout, &[(PrimaryDimTypes::In, 0)]).view(),
            Some(at_slice(-1))
        );
    }

    /// `StickMaskNode::getView` transcribed from `dsc/dsc2.cpp:2440-2491` over raw `int`s, in the
    /// authority's own order, as `(maskA, maskB, transitionSliceId_)` flattened. [`None`] stands for
    /// each of the three `DT_ERROR`s and for the empty layout the authority reads `back()` off the
    /// end of.
    fn get_view_transcribed(
        layout: &[(PrimaryDimTypes, i32)],
        first_masked: &[(PrimaryDimTypes, i32)],
    ) -> Option<(i32, i32, i32, i32, i32)> {
        let count = |dim| {
            first_masked
                .iter()
                .find(|&&(d, _)| d == dim)
                .map(|&(_, c)| c)
        };
        if layout.len() > 3 {
            return None;
        }
        let mut wsl_dim = PrimaryDimTypes::Undefined;
        let xsl_dim = layout.last()?.0;
        let (mut wsl_size, mut xsl_size) = (1, 1);
        for &(dim, size) in layout {
            if dim == xsl_dim {
                xsl_size *= size;
            } else {
                wsl_size *= size;
                if wsl_dim == PrimaryDimTypes::Undefined {
                    wsl_dim = dim;
                } else if wsl_dim != dim {
                    return None;
                }
            }
        }
        let xsl_per_slice = xsl_size / 8;
        if xsl_per_slice == 0 {
            return None;
        }
        let first_coord = count(wsl_dim).unwrap_or(wsl_size);
        let mut mask_a = (first_coord, wsl_size - first_coord);
        if layout.len() == 3 {
            mask_a = (mask_a.0 * xsl_per_slice, mask_a.1 * xsl_per_slice);
        }
        let (mut mask_b, transition) = match count(xsl_dim) {
            None => ((xsl_per_slice, 0), 7),
            Some(coord) => {
                let (quot, rem) = (coord / xsl_per_slice, coord % xsl_per_slice);
                (
                    (rem, xsl_per_slice - rem),
                    if rem == 0 { quot - 1 } else { quot },
                )
            }
        };
        if layout.len() != 3 {
            mask_b = (mask_b.0 * wsl_size, mask_b.1 * wsl_size);
        }
        Some((mask_a.0, mask_a.1, mask_b.0, mask_b.1, transition))
    }

    /// ⭐ THE FOUR TESTS ABOVE PIN SEVEN HAND-COMPUTED POINTS; THIS ONE SWEEPS THE WHOLE SHORT-LAYOUT
    /// SPACE against the transcription — every layout of one to four entries over three dims and four
    /// extents, each with no masked dim and with one masked dim at six coordinates. Both scale
    /// branches, all three refusals and the `std::div` remainder fall out of the sweep rather than
    /// being chosen, and truncating division is the same operation in both languages over the
    /// non-negative coordinates a `firstStickCoordToMaskPerDim_` holds.
    #[test]
    fn stick_mask_view_matches_get_view_transcribed_over_every_short_layout() {
        const DIMS: [PrimaryDimTypes; 3] = [
            PrimaryDimTypes::Out,
            PrimaryDimTypes::In,
            PrimaryDimTypes::Mb,
        ];
        let entries: Vec<(PrimaryDimTypes, i32)> = DIMS
            .iter()
            .flat_map(|&dim| [1, 2, 8, 16].map(move |size| (dim, size)))
            .collect();
        let mut masked: Vec<Vec<(PrimaryDimTypes, i32)>> = vec![Vec::new()];
        for &dim in &DIMS {
            masked.extend([0, 1, 3, 4, 5, 8].map(|coord| vec![(dim, coord)]));
        }
        let (mut layouts, mut frontier) = (Vec::new(), vec![Vec::new()]);
        for _ in 0..4 {
            frontier = frontier
                .iter()
                .flat_map(|l: &Vec<(PrimaryDimTypes, i32)>| {
                    entries.iter().map(|&e| [l.as_slice(), &[e][..]].concat())
                })
                .collect();
            layouts.extend(frontier.iter().cloned());
        }
        assert_eq!(
            (entries.len(), masked.len(), layouts.len()),
            (12, 19, 22620)
        );
        let mut viewed = 0usize;
        for layout in &layouts {
            for first_masked in &masked {
                let flat = samv(layout, first_masked).view().map(|v| {
                    (
                        v.mask_a.unmasked.0,
                        v.mask_a.masked.0,
                        v.mask_b.unmasked.0,
                        v.mask_b.masked.0,
                        v.transition_slice.0,
                    )
                });
                assert_eq!(
                    flat,
                    get_view_transcribed(layout, first_masked),
                    "{layout:?} masked at {first_masked:?}"
                );
                viewed += usize::from(flat.is_some());
            }
        }
        // ⛔ AND THE SWEEP IS NOT VACUOUS: a space that had drifted to all-refusal, or to no refusal
        // at all, would agree with the transcription on every case and check nothing.
        assert!(
            viewed > 0 && viewed < layouts.len() * masked.len(),
            "{viewed}"
        );
    }

    /// `dsc/dsc2.h:976-1005`: every declared initialiser at once. ⛔ `numBuffers_` STARTS AT ONE, NOT
    /// AT [`NumBuffers::STREAMING`] — a default-constructed allocation is unbuffered, and bridge 1
    /// reads mode 1 for it (`dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:58-64`).
    #[test]
    fn allocate_node_defaults_to_an_unidentified_unbuffered_direct_allocation() {
        let node = AllocateNode::default();
        // The three-way identity starts empty in both of its ported arms (`ddc/ddcv1.cpp:20-29`).
        assert_eq!(node.lds_idx, None);
        assert_eq!(node.const_idx, None);
        assert_eq!(node.component, SenComponent::NoComponent);
        assert_eq!(node.padding, PaddingFormType::default());
        // One empty `Vec` is both `layoutDimOrder_` and `maxDimSizes_` empty — the `resize` every
        // producer performs (`ddc/ddl/ddl_conversion.cpp:803`) has nothing left to do.
        assert!(node.layout_dim_order.is_empty());
        assert_eq!(node.num_buffers, NumBuffers(1));
        assert_ne!(node.num_buffers, NumBuffers::STREAMING);
        assert!(!node.is_start_addr_symbolic);
        assert!(node.buffer_offset_core_corelet.is_empty());
        assert!(node.back_gap_core.is_empty());
        assert_eq!(node.indirect_alloc_type, IndirectAllocType::NoIndirection);
        assert_eq!(node.index_tensor_type, IndexTensorType::Address);
        assert!(node.gap_stick_spread.is_empty());
        assert!(!node.ignore_symbolic_volume_limits);
        assert!(!node.non_unified_alloc_in_hbm);
    }

    /// The two indirection enums against their four string maps (`dsc/dsc2.cpp:2423-2438`). ⛔ THE
    /// `IndexTensorType` MAP LISTS `INDEX` BEFORE `ADDRESS` while the enum declares `ADDRESS` first
    /// (`dsc/dsc2.h:995-998`), so the map's initialiser order is not the discriminant order.
    #[test]
    fn the_indirection_enums_round_trip_every_spelling_in_declaration_order() {
        assert_eq!(
            IndirectAllocType::ALL.map(IndirectAllocType::name),
            ["no_indirection", "value_tensor", "index_tensor"]
        );
        assert_eq!(
            IndexTensorType::ALL.map(IndexTensorType::name),
            ["address", "index"]
        );
        for role in IndirectAllocType::ALL {
            assert_eq!(IndirectAllocType::from_name(role.name()), Some(role));
        }
        for form in IndexTensorType::ALL {
            assert_eq!(IndexTensorType::from_name(form.name()), Some(form));
        }
        assert_eq!(IndirectAllocType::from_name("index"), None);
        assert_eq!(IndexTensorType::from_name("index_tensor"), None);
    }

    /// `dsc/dsc2.h:989` ("HBM is -1") against the reader that demands that key on an HBM allocation
    /// and takes the first entry otherwise (`dsc/dsc2.cpp:3937-3955`). The iteration order is
    /// exported (`dsc/dsc2.cpp:893-906`) and linearized into the SuperDsc fingerprint
    /// (`dsc/superdsc.cpp:1458-1464`), and `-1` leads there as `None` leads here.
    #[test]
    fn back_gap_core_puts_hbms_pseudo_core_before_every_real_core() {
        let node = AllocateNode {
            component: SenComponent::Hbm,
            back_gap_core: BTreeMap::from([(
                PrimaryDimTypes::Out,
                BTreeMap::from([
                    (Some(CoreId(3)), DimSize(16)),
                    (None, DimSize(4)),
                    (Some(CoreId(0)), DimSize(8)),
                ]),
            )]),
            ..AllocateNode::default()
        };
        let out_gaps = &node.back_gap_core[&PrimaryDimTypes::Out];
        assert_eq!(
            out_gaps.keys().copied().collect::<Vec<_>>(),
            [None, Some(CoreId(0)), Some(CoreId(3))]
        );
        // The HBM read takes the `-1` entry; the LX read takes the first, which is that same entry
        // only because no real core can precede it.
        assert_eq!(out_gaps.get(&None), Some(&DimSize(4)));
        assert_eq!(out_gaps.values().next(), Some(&DimSize(4)));
    }

    /// `ForceInnermostDimensionsOp` inserting at `begin()` (`ddc/ddl/ddl_conversion.cpp:1900-1905`),
    /// over an allocation the DDL conversion sized with `-1`s (`:803`). ⭐ THE AUTHORITY INSERTS INTO
    /// TWO VECTORS AND THIS IS ONE INSERT, which is the whole point of the merge: it cannot insert
    /// the dim and forget the size. The pass refuses to run twice by testing `any_of(maxDimSizes_,
    /// >= 0)` (`:1879-1885`), which is [`Option::is_some`] here — and the entry that separates that
    /// predicate from `buildUnitView`'s cap is pinned below, in
    /// `a_zero_max_dim_size_is_filled_to_every_writer_and_absent_to_the_only_cap`.
    #[test]
    fn forcing_an_inner_dim_prepends_one_pair_and_is_refused_twice() {
        let mut node = AllocateNode {
            layout_dim_order: vec![(PrimaryDimTypes::Y, None), (PrimaryDimTypes::Out, None)],
            ..AllocateNode::default()
        };
        assert!(!node.layout_dim_order.iter().any(|(_, max)| max.is_some()));

        // `:1900-1905`: the forced dim becomes the innermost, and it carries a data stage index —
        // not an extent — until `finalizeAllocateLayouts` overwrites it (`ddc/ddcv1.cpp:1710-1732`).
        node.layout_dim_order
            .insert(0, (PrimaryDimTypes::In, Some(MaxDimSize(2))));
        assert_eq!(
            node.layout_dim_order[0],
            (PrimaryDimTypes::In, Some(MaxDimSize(2)))
        );
        assert_eq!(node.layout_dim_order.len(), 3);
        assert!(node.layout_dim_order.iter().any(|(_, max)| max.is_some()));
    }

    /// ⛔ THE THREE READERS OF ONE `maxDimSizes_` ENTRY DRAW THREE DIFFERENT BOUNDARIES, and a ZERO
    /// extent is what separates them. `getPageSize` calls only a negative entry unbounded
    /// (`dsc/dsc2.cpp:4501`); `ForceInnermostDimensionsOp` and `finalizeAllocateLayouts` call every
    /// non-negative entry filled (`ddc/ddl/ddl_conversion.cpp:1879-1881`, `ddc/ddcv1.cpp:1719`); and
    /// `buildUnitView` caps a dim only on a STRICTLY POSITIVE one (`dsc/dsc2.cpp:2806`). So
    /// [`Option::is_some`] is the writers' predicate and never the cap's, and the zero it lets
    /// through is the one entry `buildUnitView` neither caps nor `DT_CHECK`s (`:2810`) while
    /// `getPageSize` multiplies it into the page size two `DesignSpaceConfig` readers then divide by
    /// (`:4508`, `:3569`, `:3899`).
    #[test]
    fn a_zero_max_dim_size_is_filled_to_every_writer_and_absent_to_the_only_cap() {
        let node = AllocateNode {
            layout_dim_order: vec![
                (PrimaryDimTypes::Y, None),
                (PrimaryDimTypes::Out, Some(MaxDimSize(0))),
                (PrimaryDimTypes::In, Some(MaxDimSize(4))),
            ],
            indirect_alloc_type: IndirectAllocType::ValueTensor,
            ..AllocateNode::default()
        };

        // `getPageSize`'s `maxSize < 0`: only the absent entry leaves its dim unbounded.
        let unbounded = node
            .layout_dim_order
            .iter()
            .filter_map(|&(dim, max)| max.is_none().then_some(dim))
            .collect::<Vec<_>>();
        assert_eq!(unbounded, [PrimaryDimTypes::Y]);

        // The writers' `>= 0`: the zero counts as ALREADY WRITTEN, so the DDL pass refuses to run a
        // second time over it and `finalizeAllocateLayouts` overwrites it in place.
        let filled = node
            .layout_dim_order
            .iter()
            .filter(|(_, max)| max.is_some())
            .count();
        assert!(node.layout_dim_order.iter().any(|(_, max)| max.is_some()));
        assert_eq!(filled, 2);

        // `buildUnitView`'s `> 0`: the zero is NOT a cap, and it is `is_some` all the same.
        let caps = node
            .layout_dim_order
            .iter()
            .filter_map(|&(dim, max)| max.is_some_and(|MaxDimSize(size)| size > 0).then_some(dim))
            .collect::<Vec<_>>();
        assert_eq!(caps, [PrimaryDimTypes::In]);

        // And the entry that cap ignored is a bound of ZERO downstream, not an absence: one dim's
        // page size is the product of its own entries (`dsc/dsc2.cpp:4507-4508`), read out of
        // [`AllocateNode::page_size`] itself rather than recomputed here.
        let page_size = node.page_size(&node);
        assert_eq!(page_size[&PrimaryDimTypes::Out], PageSize(0));
        assert_eq!(page_size[&PrimaryDimTypes::In], PageSize(4));
        assert_eq!(page_size.get(&PrimaryDimTypes::Y), None);
    }

    /// `getPageSize`'s two arms that read this node (`dsc/dsc2.cpp:4483-4488`): a direct allocation
    /// has no page at all, and a value tensor's page is the product of its own layout's entries per
    /// dim (`:4507-4508`). ⭐ NEITHER ARM READS `relatedIndirectAccessAlloc_`, so passing the node as
    /// its own link is not a fixture cheat here — the arm that reads it is pinned next.
    #[test]
    fn a_direct_allocation_has_no_page_and_a_value_tensors_page_is_its_own_layout() {
        let direct = AllocateNode {
            layout_dim_order: vec![
                (PrimaryDimTypes::In, Some(MaxDimSize(4))),
                (PrimaryDimTypes::Out, Some(MaxDimSize(3))),
                (PrimaryDimTypes::In, Some(MaxDimSize(5))),
            ],
            ..AllocateNode::default()
        };
        assert_eq!(direct.indirect_alloc_type, IndirectAllocType::NoIndirection);
        assert!(direct.page_size(&direct).is_empty());

        let value = AllocateNode {
            indirect_alloc_type: IndirectAllocType::ValueTensor,
            ..direct.clone()
        };
        // A repeated dim multiplies, so `In` is 4 * 5 and neither entry alone (`:4507-4508`).
        assert_eq!(
            value.page_size(&value),
            BTreeMap::from([
                (PrimaryDimTypes::In, PageSize(20)),
                (PrimaryDimTypes::Out, PageSize(3)),
            ])
        );
    }

    /// ⛔ THE ANSWER FOR AN INDEX TENSOR IS THE VALUE TENSOR'S LAYOUT, read through
    /// `relatedIndirectAccessAlloc_` after `DT_CHECK`ing it non-null (`dsc/dsc2.cpp:4489-4492`) —
    /// which is why that link is [`AllocateNode::page_size`]'s parameter. Both layouts are filled and
    /// they disagree, so an answer taken from the index allocation's own layout would be a plausible
    /// wrong number rather than an empty map; it is pinned here as the number NOT to answer.
    #[test]
    fn an_index_tensors_page_size_is_the_value_tensors_and_never_its_own() {
        let value = AllocateNode {
            indirect_alloc_type: IndirectAllocType::ValueTensor,
            layout_dim_order: vec![(PrimaryDimTypes::In, Some(MaxDimSize(64)))],
            ..AllocateNode::default()
        };
        let index = AllocateNode {
            indirect_alloc_type: IndirectAllocType::IndexTensor,
            index_tensor_type: IndexTensorType::Address,
            layout_dim_order: vec![(PrimaryDimTypes::Out, Some(MaxDimSize(2)))],
            ..AllocateNode::default()
        };
        assert_eq!(
            index.page_size(&value),
            BTreeMap::from([(PrimaryDimTypes::In, PageSize(64))])
        );
        // The wrong answer, spelled out: `self`'s own layout is a different dim and a different size.
        assert_eq!(
            index.page_size(&index),
            BTreeMap::from([(PrimaryDimTypes::Out, PageSize(2))])
        );
    }

    /// ⛔ ONE ABSENT ENTRY UNBOUNDS ITS DIM IN BOTH DIRECTIONS: `pageSize.erase(dim)` throws away
    /// what earlier positions of that dim accumulated ("safe even if key not present",
    /// `dsc/dsc2.cpp:4503`) and `unboundedDims` blocks every later one (`:4505`). Both directions are
    /// pinned because only the second is a plain skip, and a dim may repeat.
    #[test]
    fn one_absent_entry_unbounds_its_dim_before_and_after_itself() {
        let node = AllocateNode {
            indirect_alloc_type: IndirectAllocType::ValueTensor,
            layout_dim_order: vec![
                // Accumulated, then erased by the absent entry that follows it (`:4503`).
                (PrimaryDimTypes::In, Some(MaxDimSize(4))),
                (PrimaryDimTypes::In, None),
                // Blocked by `unboundedDims` rather than multiplied in (`:4505`).
                (PrimaryDimTypes::In, Some(MaxDimSize(7))),
                // A different dim is untouched by either.
                (PrimaryDimTypes::Out, Some(MaxDimSize(9))),
            ],
            ..AllocateNode::default()
        };
        assert_eq!(
            node.page_size(&node),
            BTreeMap::from([(PrimaryDimTypes::Out, PageSize(9))])
        );
    }

    /// `allocAllMem`'s buffer arithmetic end to end (`ddc/ddcv1.cpp:224-226`, `:244`, `:317-328`,
    /// `:340`, `:352-356`) for a streaming allocation — the one case where the REQUEST and the
    /// RESERVATION differ. ⛔ THE STRIDE IS THE REQUEST OVER THE COUNT, NOT THE RESERVATION OVER THE
    /// COUNT: `kv.second` at `:356` is what was pushed at `:244`, while the widening at `:327` only
    /// reached the local `mySize` that `checkAndAddDs` places at `:340`. The quotient lands in a
    /// FUNCTION-LOCAL map that the committing tail copies onto the node (`:407-408`), see
    /// [`AllocateNode::buffer_offset_core_corelet`].
    #[test]
    fn a_streaming_buffer_offset_is_one_buffers_capacity_and_not_half_the_reservation() {
        // `:224-226`: the count `-1` becomes 2 before anything is sized with it.
        let declared = NumBuffers::STREAMING;
        let buffers = i64::from(if declared == NumBuffers::STREAMING {
            2
        } else {
            declared.0
        });
        assert_eq!(buffers, 2);

        // `:244`: the request is that count times one buffer's capacity.
        let capacity = 6 * 1024;
        let requested = buffers * capacity;

        // `:317-328`: only a streaming allocation has its reservation widened to the whole memory,
        // and `:340` places it at that widened size.
        let mem_capacity = 256 * 1024;
        let reserved = requested.max(mem_capacity);
        assert_eq!(reserved, mem_capacity);
        assert_ne!(reserved, requested);

        // `:355-356`: the divisor is the count, but the dividend is the REQUEST.
        let offset = BufferOffset(requested / buffers);
        assert_eq!(offset, BufferOffset(capacity));
        assert_ne!(offset, BufferOffset(reserved / buffers));

        // A non-streaming count is the case where the two agree, which is how reading the stride as
        // half of the reservation survives every fixture that holds no streaming allocation.
        let plain = NumBuffers(2);
        let plain_reserved = i64::from(plain.0) * capacity;
        assert_eq!(BufferOffset(plain_reserved / i64::from(plain.0)), offset);
    }

    /// `finalizeAllocateLayouts`'s inner loop transcribed onto the ported readers the authority
    /// calls (`ddc/ddcv1.cpp:1718-1730`): the entry names a sizing data stage, that stage's extent
    /// for the dim beside it REPLACES the stage index, and a dim inside the stick is divided by its
    /// cumulative stick size. ⛔ THAT DIVISION IS `int` DIVISION, truncating toward zero in both
    /// languages.
    fn finalize_allocate_layouts_entry(
        sizing_stage: &DataStructDims,
        component: SenComponent,
        dim: PrimaryDimTypes,
        stick_sizes: &BTreeMap<PrimaryDimTypes, DimSize>,
    ) -> Option<MaxDimSize> {
        use crate::schedule::dims::DimDensity;
        use sys_arch_spec::RowId;

        // `:1720-1722`: `ss_.primaryDimToVal_st(dim, component_, 0, 0)` — PT row 0, corelet 0.
        let mut size = sizing_stage
            .primary_dim_to_val_for_component(
                dim,
                component,
                Some(RowId(0)),
                Some(CoreletId(0)),
                &PaddingFormType::default(),
                DimDensity::FULL,
                false,
            )
            .expect("`primaryDimToVal_st` answers an `int` for every real dim")
            .0;
        // `:1723-1728`: only a dim that is part of the stick divides.
        if let Some(stick) = stick_sizes.get(&dim) {
            size /= stick.0;
        }
        // `:1729` writes the `int` straight back into `maxDimSizes_`, where negative is [`None`].
        u32::try_from(size).ok().map(MaxDimSize)
    }

    /// ⛔ AN UNFILLED SIZING DIM IS NOT AN UNBOUNDED ONE: on a stick dim the SAME absence becomes a
    /// filled ZERO. `primaryDimToVal_st` answers `-1` for a dim the sizing stage does not carry
    /// (`dsc/dims.cpp:567-568`, over dims born `-1`, `dsc/dims.h:162-192`) and
    /// `finalizeAllocateLayouts` divides that `-1` by the cumulative stick size before writing it
    /// back (`ddc/ddcv1.cpp:1723-1729`), so `-1 / 4 == 0`. Both halves come out of the ported code
    /// the authority would call — [`DesignSpaceConfig::cumulative_stick_sizes`] for the divisor and
    /// [`DataStructDims::primary_dim_to_val_for_component`] for the extent — so what is pinned is
    /// that a porter of that pass cannot pass the reader's answer through: one absence reaches
    /// [`MaxDimSize`] as TWO different values, and `as u32` would make it 4294967295.
    #[test]
    fn an_unfilled_sizing_extent_is_a_zero_page_on_a_stick_dim_and_no_page_off_it() {
        use crate::schedule::dims::{DimDensity, DimVal};
        use crate::schedule::dsc::{
            DesignSpaceConfig, DsTypes, PrimaryDsInfo, StickRepl, StickSize, StickSizeScope,
        };
        use sys_arch_spec::RowId;

        let mut dsc = DesignSpaceConfig::default();
        dsc.primary_ds_info.insert(
            DsTypes::Input,
            PrimaryDsInfo {
                layout_dim_order: Vec::new(),
                stick_dim_order: vec![PrimaryDimTypes::In, PrimaryDimTypes::Out],
                stick_size: vec![StickSize(8.0), StickSize(4.0)],
                stick_repl: vec![StickRepl(1), StickRepl(1)],
            },
        );
        let stick_sizes = dsc
            .cumulative_stick_sizes(DsTypes::Input, StickSizeScope::WholeStick)
            .expect("the stick is fully described");
        assert_eq!(
            stick_sizes,
            BTreeMap::from([
                (PrimaryDimTypes::In, DimSize(8)),
                (PrimaryDimTypes::Out, DimSize(4)),
            ])
        );

        // The sizing stage carries `In` and neither `Out` nor `Y`. The two it does not carry are the
        // authority's `-1`, which the ported reader reports as the VALUE it is and not as an absence.
        let mut stage = DataStructDims::default();
        *stage
            .primary_dim_to_val_handler_mut(PrimaryDimTypes::In)
            .expect("`In` has a field") = crate::schedule::dims::DimSize::new(64.0);
        let sized = |dim| {
            stage.primary_dim_to_val_for_component(
                dim,
                SenComponent::Lx,
                Some(RowId(0)),
                Some(CoreletId(0)),
                &PaddingFormType::default(),
                DimDensity::FULL,
                false,
            )
        };
        assert_eq!(sized(PrimaryDimTypes::In), Some(DimVal(64)));
        assert_eq!(sized(PrimaryDimTypes::Out), Some(DimVal(-1)));
        assert_eq!(sized(PrimaryDimTypes::Y), Some(DimVal(-1)));

        let entry =
            |dim| finalize_allocate_layouts_entry(&stage, SenComponent::Lx, dim, &stick_sizes);
        // A dim the stage carries is an extent counted in sticks: 64 / 8.
        assert_eq!(entry(PrimaryDimTypes::In), Some(MaxDimSize(8)));
        // ⛔ The absence, on a stick dim: `-1 / 4` truncates to a FILLED zero.
        assert_eq!(entry(PrimaryDimTypes::Out), Some(MaxDimSize(0)));
        // The same absence, off the stick: the negative survives, and that is the unbounded dim.
        assert_eq!(entry(PrimaryDimTypes::Y), None);

        // And the two answers part company downstream: the zero is a page size of zero, the
        // negative erases its dim (`dsc/dsc2.cpp:4501-4508`).
        let node = AllocateNode {
            indirect_alloc_type: IndirectAllocType::ValueTensor,
            layout_dim_order: [
                PrimaryDimTypes::In,
                PrimaryDimTypes::Out,
                PrimaryDimTypes::Y,
            ]
            .map(|dim| (dim, entry(dim)))
            .to_vec(),
            ..AllocateNode::default()
        };
        assert_eq!(
            node.page_size(&node),
            BTreeMap::from([
                (PrimaryDimTypes::In, PageSize(8)),
                (PrimaryDimTypes::Out, PageSize(0)),
            ])
        );
    }

    /// `allocAllMem`'s commit of the buffer strides transcribed as the authority writes it
    /// (`ddc/ddcv1.cpp:407-429`): the strides it computed live in a function-local map (`:136-137`)
    /// that only the committing branch copies onto the node, corelet 0 is then replicated to every
    /// used corelet, and the head core to the other used cores — the last gated on
    /// `numBuffers_ != 1`.
    fn commit_buffer_offsets(
        node: &mut AllocateNode,
        placed: &BTreeMap<CoreId, BTreeMap<CoreletId, BufferOffset>>,
        commit: bool,
        copy_core: bool,
        copy_corelet: bool,
        cores_used: &[CoreId],
        num_corelets_used: u8,
    ) {
        // `:351`: the writes of the local are gated on `commitIfValid` and so is this whole tail, so
        // a speculative run reaches neither.
        if !commit {
            return;
        }
        // `:407-408`: the node takes the local's per-node value wholesale.
        node.buffer_offset_core_corelet = placed.clone();
        if copy_corelet {
            // `:413-417`, whose `.at(0)` is this index.
            for per_corelet in node.buffer_offset_core_corelet.values_mut() {
                let corelet0 = per_corelet[&CoreletId(0)];
                for cl in 1..num_corelets_used {
                    per_corelet.insert(CoreletId(cl), corelet0);
                }
            }
        }
        if copy_core {
            // `:421-428`, including the gate at `:424`.
            let (head, others) = cores_used.split_first().expect("`coreIdsUsed_.front()`");
            for &core in others {
                if node.num_buffers != NumBuffers(1) {
                    let from_head = node.buffer_offset_core_corelet[head].clone();
                    node.buffer_offset_core_corelet.insert(core, from_head);
                }
            }
        }
    }

    /// ⛔ THE PER-CORE WRITE AT `ddc/ddcv1.cpp:355-356` IS NOT THIS FIELD, and two consequences are
    /// pinned over the transcription above. A speculative `allocAllMem(false)` — fifteen of the
    /// seventeen call sites — leaves the map EMPTY although the placement itself ran, because the
    /// write lands in the function-local homonym (`:136-137`). And a committed UNBUFFERED allocation
    /// keeps a SINGLE core key, because the cross-core replication is gated on `numBuffers_ != 1`
    /// (`:424`). ⭐ WHICH IS WHY THE READER THAT INDEXES THIS MAP BY AN ARBITRARY CORE DEMANDS LX AND
    /// `numBuffers_` 1 OR 2 FIRST (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:4929`, `:4940-4944`):
    /// off that path the core it asks for need not be a key at all.
    #[test]
    fn a_speculative_placement_writes_no_stride_and_an_unbuffered_one_keeps_one_core() {
        const HEAD: CoreId = CoreId(0);
        const CORES: [CoreId; 3] = [HEAD, CoreId(1), CoreId(2)];
        const STRIDE: BufferOffset = BufferOffset(6 * 1024);
        let placed = BTreeMap::from([(HEAD, BTreeMap::from([(CoreletId(0), STRIDE)]))]);

        let mut speculative = AllocateNode::default();
        commit_buffer_offsets(&mut speculative, &placed, false, true, true, &CORES, 2);
        assert!(speculative.buffer_offset_core_corelet.is_empty());

        let mut unbuffered = AllocateNode::default();
        assert_eq!(unbuffered.num_buffers, NumBuffers(1));
        commit_buffer_offsets(&mut unbuffered, &placed, true, true, true, &CORES, 2);
        assert_eq!(
            unbuffered.buffer_offset_core_corelet,
            BTreeMap::from([(
                HEAD,
                BTreeMap::from([(CoreletId(0), STRIDE), (CoreletId(1), STRIDE)])
            )])
        );

        // The same placement, double buffered: every used core is a key, and each is the head's.
        let mut doubled = AllocateNode {
            num_buffers: NumBuffers(2),
            ..AllocateNode::default()
        };
        commit_buffer_offsets(&mut doubled, &placed, true, true, true, &CORES, 2);
        assert_eq!(
            doubled
                .buffer_offset_core_corelet
                .keys()
                .copied()
                .collect::<Vec<_>>(),
            CORES
        );
        assert_eq!(
            doubled.buffer_offset_core_corelet[&CoreId(2)],
            doubled.buffer_offset_core_corelet[&HEAD]
        );
    }

    /// `dsc/dsc2.h:47-51`: a fresh constant is INVALID, unnamed and not symbolic. ⛔ ITS FORMAT IS
    /// NOT [`DataFormats::default()`], which is `SEN169_FP16` (`dsc/dsc2.h:934`). ⛔ AND THE
    /// AUTHORITY'S TABLE DOES NOT REFUSE THAT INITIALISER: `INVALID` maps to `-1`
    /// (`util/sendefs/sendefs.cpp:131`), so both sites that divide by this field's width answer a
    /// NEGATIVE count instead of stopping — `1024 / width` for a `CONSTANT_TO_CONSTANT` transfer
    /// (`dsc/dsc2.cpp:3486-3492`) and `bitsPerElem * numElems`, which then slips UNDER the `> 4 * 32`
    /// gate that exists to bound it (`ddc/ddl/ddl_conversion.cpp:697-704`). [`None`] is this port's
    /// boundary because no reader can divide by it.
    #[test]
    fn a_fresh_constants_invalid_format_has_no_width_and_the_authority_divides_by_minus_one() {
        let constant = ConstantInfo::default();
        assert_eq!(constant.data_format, DataFormats::Invalid);
        assert_ne!(constant.data_format, DataFormats::default());
        assert!(constant.name.is_empty());
        assert!(!constant.is_data_symbolic);
        assert_eq!(constant.data_format.bit_width(), None);

        // The authority's entry, and neither reader refuses it.
        let authority_width = -1;
        assert_eq!(1024 / authority_width, -1024);
        let num_elems = 16;
        assert!(authority_width * num_elems <= 4 * 32);

        // A STATED format is what makes either quantity a bound at all.
        let stated = DataFormats::Sen169Fp16.bit_width().unwrap().0;
        assert_eq!(1024 / stated, 64);
        assert!(stated * num_elems > 4 * 32);
    }

    /// ⛔ THE PAD PATH WRITES THIS FIELD FROM THE LABELED DS AND NEVER READS IT BACK FOR A WIDTH:
    /// `transformLxZeroPadInfoInScheduleTree` unpacks the op-const with `lds.dataFormat_`
    /// (`dsc/dsc2.cpp:5260-5265`, over the `lds` bound at `:4773`), mints the entry with that same
    /// format under the reserved name (`:5281-5283`), and on a REUSED `padval` entry `DT_CHECK`s
    /// that the two agree (`:5229-5235`). So the constant's format equals the labeled DS's by
    /// construction, and a disagreeing entry is what that check refuses.
    #[test]
    fn the_padval_entry_carries_the_labeled_dss_format_and_a_disagreement_is_what_is_refused() {
        let lds_data_format = DataFormats::Sen143Fp8;
        let minted = ConstantInfo {
            data_format: lds_data_format,
            name: "padval".to_string(),
            ..ConstantInfo::default()
        };
        assert_eq!(minted.data_format, lds_data_format);

        // `:5229-5230`: the reuse arm keys on the name, then requires the format to agree.
        let constants = BTreeMap::from([(ConstantId(0), minted)]);
        let reused = constants.values().find(|c| c.name == "padval").unwrap();
        assert_eq!(reused.data_format, lds_data_format);
        assert_ne!(reused.data_format, DataFormats::Sen169Fp16);

        // The mask the unpacker builds is the LABELED DS's width, which this entry only mirrors.
        assert_eq!(lds_data_format.bit_width(), Some(BitWidth(8)));
    }

    /// ⛔ ONE FLAG, TWO OPPOSITE VERDICTS: bridge 1 emits `is_symbol` exactly when it is set
    /// (`dsc-based-utils/DSC2ToDataflowIR/V3/SNDSCLowering.cpp:479-480`, `:515-516`) and the PCFG
    /// translator `DT_CHECK`s that it is NOT (`dsc/dsc2Pcfg.cpp:1366`, `:1945`), so `stzJumpAddr`
    /// (`dbo/src/Utils/sdsc_bundle/ProgramCorrection.cpp:1151-1154`) is a constant one consumer
    /// emits and the other refuses.
    #[test]
    fn the_symbolic_flag_is_bridge_ones_attribute_and_the_pcfg_paths_refusal() {
        let stz_jump_addr = ConstantInfo {
            name: "stzJumpAddr".to_string(),
            is_data_symbolic: true,
            ..ConstantInfo::default()
        };
        let padval = ConstantInfo {
            name: "padval".to_string(),
            ..ConstantInfo::default()
        };

        let emits_is_symbol = |constant: &ConstantInfo| constant.is_data_symbolic;
        let pcfg_admits = |constant: &ConstantInfo| !constant.is_data_symbolic;

        assert!(emits_is_symbol(&stz_jump_addr));
        assert!(!pcfg_admits(&stz_jump_addr));
        assert!(!emits_is_symbol(&padval));
        assert!(pcfg_admits(&padval));
    }

    /// `dsc/dsc2.h:54-60` against `:46`: the copy ASSIGNMENT never touches `isDataSymbolic_`, while
    /// the copy CONSTRUCTOR carries every field — and the constructor is what
    /// `emplace(myId, std::move(myConstInfo))` resolves to, because declaring that assignment
    /// operator suppresses the implicit move constructor (`ddc/ddl/ddl_conversion.cpp:725`).
    /// [`Clone`] is that constructor, so `stzJumpAddr`'s flag travels
    /// (`dbo/src/Utils/sdsc_bundle/ProgramCorrection.cpp:1151-1154`).
    #[test]
    fn cloning_a_symbolic_constant_carries_the_flag_the_authoritys_assignment_drops() {
        let symbolic = ConstantInfo {
            data_format: DataFormats::Sen169Fp16,
            name: "stzJumpAddr".to_string(),
            is_data_symbolic: true,
            ..ConstantInfo::default()
        };
        let copied = symbolic.clone();
        assert_eq!(copied.data_format, DataFormats::Sen169Fp16);
        assert_eq!(copied.name, "stzJumpAddr");
        assert!(copied.is_data_symbolic);
    }

    /// The two in-scope readers that key on the name, over `constantInfo_`'s own shape
    /// (`dsc/designSpaceConfig.h:90`): the padding path reuses the entry named `padval` and requires
    /// its format to agree with the labeled ds's (`dsc/dsc2.cpp:5228-5235`), and a constant named
    /// `useZeroMean` turns an `EXX2` op into `EXX2_ZEROMEAN` (`ddc/ddcv1.cpp:2064-2072`) — that
    /// second one also tests the datum, [`data`](ConstantInfo::data), which neither entry here builds.
    #[test]
    fn a_constants_name_is_the_key_both_in_scope_readers_match_on() {
        let constants = BTreeMap::from([
            (
                ConstantId(0),
                ConstantInfo {
                    data_format: DataFormats::Sen143Fp8,
                    name: "useZeroMean".to_string(),
                    is_data_symbolic: false,
                    ..ConstantInfo::default()
                },
            ),
            (
                ConstantId(1),
                ConstantInfo {
                    data_format: DataFormats::Sen169Fp16,
                    name: "padval".to_string(),
                    is_data_symbolic: false,
                    ..ConstantInfo::default()
                },
            ),
        ]);

        let padval = constants.iter().find(|(_, c)| c.name == "padval");
        assert_eq!(padval.map(|(id, _)| *id), Some(ConstantId(1)));
        assert_eq!(
            padval.map(|(_, c)| c.data_format),
            Some(DataFormats::Sen169Fp16)
        );
        assert!(constants.values().any(|c| c.name == "useZeroMean"));
    }

    /// ⛔ BOTH OF A TRANSFER NODE'S MAPS ARE ITERATED BY THE JSON EXPORTER, so their key order is
    /// observable output and not an implementation detail: `coreIdToGTRInfo_` emits one object per
    /// core keyed by `std::to_string(coreId)` (`dsc/dsc2.cpp:627-629`) and `transferSize_` emits
    /// `primaryDimToString.at(dim)` keys inline (`:641-647`). This pins the two [`BTreeMap`]s to the
    /// `std::map` order the authority writes — insertion order must NOT survive.
    #[test]
    fn a_transfer_nodes_two_maps_emit_their_keys_in_the_authoritys_order() {
        let mut node = TransferNode::default();

        // Inserted in neither key order nor reverse key order, so passing means the map reordered.
        for dim in [
            PrimaryDimTypes::X1,
            PrimaryDimTypes::In,
            PrimaryDimTypes::Kij,
            PrimaryDimTypes::Undefined,
            PrimaryDimTypes::Mb,
        ] {
            node.transfer_size.insert(dim, DimSize(dim as i32));
        }
        // The spelling and the sequence the exporter's `for` loop writes, verbatim.
        assert_eq!(
            node.transfer_size
                .keys()
                .map(|dim| dim.name())
                .collect::<Vec<_>>(),
            ["in", "mb", "kij", "x1", "undefined"],
            "transferSize_ must emit in the authority's discriminant order (`dsc/dims.h:34-48`)"
        );

        for core in [CoreId(9), CoreId(0), CoreId(4)] {
            node.core_id_to_gtr_info
                .insert(core, GroupTagRegInfo::default());
        }
        // `std::to_string(coreId)` over an ascending `int` key, which is `CoreId`'s derived `Ord`.
        assert_eq!(
            node.core_id_to_gtr_info.keys().copied().collect::<Vec<_>>(),
            [CoreId(0), CoreId(4), CoreId(9)],
            "coreIdToGTRInfo_ must emit in ascending core id order"
        );
    }

    /// ⛔ A DELIBERATE DIVERGENCE, AND THE AUTHORITY'S SIDE IS UNDEFINED BEHAVIOUR.
    /// `TransferPadInfo(TransferPadInfo&&) = default` (`dsc/dsc2.h:760`) moves the two maps but
    /// bitwise-copies `MapWithFMHelper`, whose only member is a REFERENCE to the sibling map
    /// (`util/foldManager/mapWithFMHelper.h:829-831`) — so the moved-TO helper still refers to the
    /// moved-FROM object's storage. Measured on the authority: `dst.isEmpty = 0` with
    /// `dst.frontKeys = 0` and every helper-routed query throwing, `dst.frontKeys = 1` again after
    /// rebuilding on the SOURCE, and an AddressSanitizer use-after-scope inside `getAllKeys`
    /// once the source was destroyed. A Rust move carries the storage, so all four of these answer
    /// from the moved-to object — there is no reference to leave behind.
    #[test]
    fn a_moved_transfer_pad_info_carries_its_own_storage_where_the_authoritys_aliases_the_source() {
        let mut src = TransferPadInfo::default();
        src.build_pad_sizes(
            PadEnd::Front,
            PrimaryDimTypes::X,
            [FoldDimSize(2), FoldDimSize(3)],
            [PadSize(-40), PadSize(-10)],
            [PadSize(25), PadSize(0)],
        );
        let moved = src;

        assert!(!moved.is_empty(), "`dst.isEmpty = 0`, as in C++");
        assert_eq!(
            moved.pad_dims(PadEnd::Front).collect::<Vec<_>>(),
            [PrimaryDimTypes::X],
            "C++ answers 0 keys here"
        );
        assert_eq!(
            moved.transfer_pad_size(
                PadEnd::Front,
                PrimaryDimTypes::X,
                WkSliceIdx(0),
                ChunkIdx(0),
                ChunkSizePadded(10)
            ),
            Some(PadSize(10)),
            "C++ throws here"
        );

        // The two `FoldDimProp`s the fold was built over — `wkslice_index` outer, `chunk_index`
        // inner (`dsc/dsc2.cpp:4669-4675`) — now read off the manager's own `dim_prop_`, which is
        // where they live.
        let fm = moved.front[&PrimaryDimTypes::X].manager();
        assert_eq!(fm.num_dims(), FoldDimPosition::COUNT);
        assert_eq!(
            fm.func_types(),
            vec![BaseFuncType::Affine; FoldDimPosition::COUNT]
        );
        let props = fm.fold_dim_props();
        assert_eq!(props[0].size(), FoldDimSize(2));
        assert_eq!(props[0].label(), "wkslice_index");
        assert_eq!(props[1].size(), FoldDimSize(3));
        assert_eq!(props[1].label(), "chunk_index");
    }

    /// `isNodeRelevant`'s component reading, and the divergence between it and `getRelevantComps`: a
    /// core present with an EMPTY corelet set is relevant to the first and invisible to the second
    /// (`dsc/dsc2.cpp:1929-1931` against `:1964-1967`). The state is hand-built here; the writer that
    /// actually mints it is the subject of the test below.
    #[test]
    fn a_core_with_no_corelets_is_relevant_but_absent_from_get_relevant_comps() {
        let mut node = ScheduleNode::new(NodeType::Transfer);
        node.relevant_comps_mut().insert(
            SenComponent::Lx,
            BTreeMap::from([
                (CoreId(0), BTreeSet::from([CoreletId(0), CoreletId(1)])),
                (CoreId(1), BTreeSet::new()),
            ]),
        );

        assert!(node.is_relevant(SenComponent::Lx));
        assert!(!node.is_relevant(SenComponent::L0));

        // `isNodeRelevant(LX, -1, 1)` — the core's presence alone answers true.
        let cores = node
            .relevant_cores(SenComponent::Lx)
            .expect("LX is relevant");
        assert!(cores.contains_key(&CoreId(1)));
        // `getRelevantComps(1)` — the same core, and LX does not count.
        assert!(node.relevant_comps_of_core(CoreId(1)).is_empty());
        assert_eq!(
            node.relevant_comps_of_core(CoreId(0)),
            BTreeSet::from([SenComponent::Lx])
        );
        assert_eq!(
            node.relevant_comps_of_corelet(CoreId(0), CoreletId(1)),
            BTreeSet::from([SenComponent::Lx])
        );
        assert!(
            node.relevant_comps_of_corelet(CoreId(0), CoreletId(2))
                .is_empty()
        );
        // `getRelevantComps()` — LX has cores, so it counts (`dsc/dsc2.cpp:1969-1971`).
        assert_eq!(
            node.relevant_comps_any_core(),
            BTreeSet::from([SenComponent::Lx])
        );
    }

    /// ⭐ THE WRITER THAT MINTS AN EMPTY CORELET SET, the one `setRelevantCompCoreCl` cannot be: the
    /// importer's head-node union over a child whose own core came from `"1":[]`
    /// (`dsc/dsc2.cpp:1373-1382`, `:1389-1395`). The head it leaves behind still satisfies
    /// `DT_CHECK(!getHead()->relevantComps_.empty())` (`ddc/ddcv1.cpp:3458`).
    #[test]
    fn the_head_union_mints_an_empty_corelet_set_that_the_ddcs_check_still_accepts() {
        // Two children as the importer built them: `{"LX": {"0": [0], "1": []}}`, `{"L0": {"0": [1]}}`.
        let mut child_a = ScheduleNode::new(NodeType::Transfer);
        child_a.relevant_comps_mut().insert(
            SenComponent::Lx,
            BTreeMap::from([
                (CoreId(0), BTreeSet::from([CoreletId(0)])),
                (CoreId(1), BTreeSet::new()),
            ]),
        );
        let mut child_b = ScheduleNode::new(NodeType::Compute);
        child_b.relevant_comps_mut().insert(
            SenComponent::L0,
            BTreeMap::from([(CoreId(0), BTreeSet::from([CoreletId(1)]))]),
        );

        // `:1389-1395`, with `entry().or_default()` standing in for `operator[]`: the core's entry is
        // created before the range goes in, so an empty range leaves it empty.
        let mut head = ScheduleNode::new(NodeType::Block);
        for child in [&child_a, &child_b] {
            for (comp, core_cls) in child.relevant_comps() {
                let head_comp = head.relevant_comps_mut().entry(*comp).or_default();
                for (core, cls) in core_cls {
                    head_comp
                        .entry(*core)
                        .or_default()
                        .extend(cls.iter().copied());
                }
            }
        }

        assert_eq!(
            head.relevant_comps()[&SenComponent::Lx][&CoreId(1)],
            BTreeSet::new()
        );
        // `ddc/ddcv1.cpp:3458` tests the MAP, which is non-empty whatever the corelet sets hold.
        assert!(!head.relevant_comps().is_empty());

        // And the three readers disagree about that core on the HEAD node too.
        assert!(
            head.relevant_cores(SenComponent::Lx)
                .is_some_and(|cores| cores.contains_key(&CoreId(1)))
        );
        assert!(head.relevant_comps_of_core(CoreId(1)).is_empty());
        assert!(
            !head
                .relevant_core_cl_of_comp(SenComponent::Lx)
                .contains_key(&CoreId(1))
        );
    }

    /// `ALL` is the authority's "do not filter", answered by an early return and never by a lookup
    /// (`dsc/dsc2.cpp:1918-1922`, `:1939`) — so a node with an empty `relevantComps_`, which is what
    /// every node has before `setRelevantCompCoreCl` runs (`dsc/dsc2.cpp:2647-2729`), is still
    /// relevant to it.
    #[test]
    fn the_all_component_is_relevant_without_ever_being_a_key() {
        let node = ScheduleNode::new(NodeType::Sync);

        assert!(node.relevant_comps().is_empty());
        assert!(node.is_relevant(SenComponent::All));
        assert!(!node.is_relevant(SenComponent::Lx));
        assert_eq!(node.relevant_cores(SenComponent::All), None);
        assert_eq!(
            node.relevant_core_cl_of_comp(SenComponent::All),
            node.relevant_core_cl()
        );
        assert_eq!(node.node_type(), NodeType::Sync);
        assert!(!node.is_block_node());
    }

    /// `getRelevantCoreCl()` merges the corelets of EVERY component onto one core map and drops a
    /// core whose corelet set is empty, `if (!cls.empty())` (`dsc/dsc2.cpp:1940-1942`); named a
    /// component it filters, and named `ALL` it filters nothing (`:1939`).
    #[test]
    fn relevant_core_cl_merges_every_component_and_drops_a_core_with_no_corelet() {
        let mut node = ScheduleNode::new(NodeType::Compute);
        node.relevant_comps_mut().extend([
            (
                SenComponent::Pe,
                BTreeMap::from([
                    (CoreId(0), BTreeSet::from([CoreletId(0)])),
                    (CoreId(2), BTreeSet::new()),
                ]),
            ),
            (
                SenComponent::Sfp,
                BTreeMap::from([(CoreId(0), BTreeSet::from([CoreletId(1)]))]),
            ),
        ]);

        assert_eq!(
            node.relevant_core_cl(),
            BTreeMap::from([(CoreId(0), BTreeSet::from([CoreletId(0), CoreletId(1)]))])
        );
        assert_eq!(
            node.relevant_core_cl_of_comp(SenComponent::Pe),
            BTreeMap::from([(CoreId(0), BTreeSet::from([CoreletId(0)]))])
        );
        assert!(node.relevant_core_cl_of_comp(SenComponent::Lx).is_empty());
        assert_eq!(
            node.relevant_core_cl_of_comp(SenComponent::All),
            node.relevant_core_cl()
        );
    }

    /// `getSizesForCoreId`'s three-step fallback (`dsc/dsc2.cpp:2398-2405`): this core, then HBM's
    /// `-1` pseudo-core, then the gapless view.
    #[test]
    fn a_unit_views_sizes_fall_back_from_the_core_to_hbm_to_the_gapless_view() {
        let no_gaps = vec![Size::new(PrimaryDimTypes::Ij, DimSize(4))];
        let hbm = vec![Size::new(PrimaryDimTypes::Ij, DimSize(6))];
        let core_three = vec![Size::new(PrimaryDimTypes::Ij, DimSize(8))];
        let view = UnitView {
            sizes_no_gaps: no_gaps.clone(),
            sizes_with_gaps: BTreeMap::from([
                (None, hbm.clone()),
                (Some(CoreId(3)), core_three.clone()),
            ]),
            ..UnitView::default()
        };

        assert_eq!(view.sizes_for_core(CoreId(3)), core_three.as_slice());
        assert_eq!(view.sizes_for_core(CoreId(0)), hbm.as_slice());

        let gapless = UnitView {
            sizes_no_gaps: no_gaps.clone(),
            ..UnitView::default()
        };
        assert_eq!(gapless.sizes_for_core(CoreId(0)), no_gaps.as_slice());
    }

    /// The gap rescale multiplies the offset of every loop entry whose `sizeIdx_` names the gapped
    /// extent (`dsc/dsc2.cpp:2880-2897`), and an unrelated loop's `{currLoop, dim, -1, -1}` entry
    /// (`:2872`) is not one of them — which [`None`] states rather than relies on `-1` failing the
    /// comparison.
    #[test]
    fn the_gap_rescale_passes_over_a_loop_that_names_no_extent() {
        let unrelated = LoopInfo {
            dim: PrimaryDimTypes::Mb,
            ..LoopInfo::default()
        };
        let stepping = LoopInfo {
            dim: PrimaryDimTypes::Ij,
            size_idx: Some(SizeIdx(1)),
            elem_offset: Some(ElemOffset(4)),
        };
        assert_eq!(unrelated.size_idx, None);
        assert_eq!(unrelated.elem_offset, None);
        assert_eq!(LoopInfo::default().dim, PrimaryDimTypes::Undefined);

        let mut composite_loops = vec![unrelated, stepping];
        for loop_info in &mut composite_loops {
            if loop_info.size_idx == Some(SizeIdx(1)) {
                loop_info.elem_offset = loop_info.elem_offset.map(|o| ElemOffset(o.0 * 3));
            }
        }

        assert_eq!(composite_loops[0].elem_offset, None);
        assert_eq!(composite_loops[1].elem_offset, Some(ElemOffset(12)));
    }

    /// `dsc/dsc2.h:741-750`. Neither predicate holds on a default-constructed `DataInfo` — IBM's
    /// `-1`/`-1` — each holds for exactly its own variant, and the state both `DT_CHECK`s refuse has
    /// no spelling at all.
    ///
    /// ⭐ THE ONE REWRITE THAT MOVES A SOURCE FROM A TENSOR TO A CONSTANT is two adjacent
    /// assignments in the authority (`dsc/dsc2.cpp:5379-5380`) and ONE here, so the refused state is
    /// not even momentarily reachable.
    #[test]
    fn e033_a_data_info_is_a_labeled_ds_or_a_constant_and_never_both() {
        let neither = DataInfo::default();
        assert!(!neither.is_labeled_ds());
        assert!(!neither.is_constant());
        assert_eq!(neither.latch_data_id, None);
        assert_eq!(neither.data_connect, DataConnect(String::new()));
        assert!(!neither.is_start_addr_symbolic);

        let lds = DataInfo {
            lds_or_const: Some(LdsOrConst::LabeledDs(LdsIdx(3))),
            ..DataInfo::default()
        };
        assert!(lds.is_labeled_ds());
        assert!(!lds.is_constant());

        let mut constant = lds.clone();
        constant.lds_or_const = Some(LdsOrConst::Constant(ConstantId(7)));
        assert!(!constant.is_labeled_ds());
        assert!(constant.is_constant());
    }

    /// ⛔ PINS THE DIVERGENCE `const_ele_offsets`' own doc names. The outer two maps are `std::map`
    /// in the authority and this reproduces their key order; the INNERMOST is `std::unordered_map`
    /// whose iteration order reaches IBM's exported JSON (`dsc/dsc2.cpp:237-241`), so the dims come
    /// out here in [`PrimaryDimTypes`] declaration order — `In` before `Ki` — where the reference
    /// spells them in libstdc++ hash order. The reader that takes three `begin()`s
    /// (`ddc/ddl/ddl_conversion.cpp:3105-3111`) therefore reads the LOWEST dim here and an arbitrary
    /// one there.
    #[test]
    fn e033_the_const_element_offsets_walk_core_then_corelet_then_dim_in_key_order() {
        let mut di = DataInfo::default();
        for (core, corelet, dim) in [
            (1u8, 1u8, PrimaryDimTypes::Ij),
            (0, 1, PrimaryDimTypes::In),
            (0, 0, PrimaryDimTypes::Ki),
            (0, 0, PrimaryDimTypes::In),
        ] {
            di.const_ele_offsets
                .entry(CoreId(core))
                .or_default()
                .entry(CoreletId(corelet))
                .or_default()
                .insert(dim, ConstEleOffset(1));
        }

        let walk: Vec<_> = di
            .const_ele_offsets
            .iter()
            .flat_map(|(core, corelets)| {
                corelets.iter().flat_map(move |(corelet, dims)| {
                    dims.keys().map(move |dim| (*core, *corelet, *dim))
                })
            })
            .collect();
        assert_eq!(
            walk,
            [
                (CoreId(0), CoreletId(0), PrimaryDimTypes::In),
                (CoreId(0), CoreletId(0), PrimaryDimTypes::Ki),
                (CoreId(0), CoreletId(1), PrimaryDimTypes::In),
                (CoreId(1), CoreletId(1), PrimaryDimTypes::Ij),
            ]
        );
    }

    /// ⛔ A CONSTANT OPERAND GETS NO OFFSETS AT ALL, AND THIS TYPE CANNOT SAY SO. Both stages' fillers
    /// return before the offset half for anything that is not a labeled ds — `if (di.myLdsIdx_ < 0)
    /// return;  // no offsets for constants` (`ddc/ddcv1.cpp:2410`,
    /// `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:5991`) — and the rewrite that turns a tensor source
    /// into a constant one asserts that result with four `DT_CHECK`s in a row
    /// (`dsc/dsc2.cpp:5386-5397`). Of those four fields this type carries two, and neither
    /// [`LdsOrConst`] nor [`DataInfo::default`] refuses the state the authority checks for.
    #[test]
    fn e033_a_constant_operand_carries_no_offsets_and_this_type_cannot_refuse_one() {
        // What `dsc/dsc2.cpp:5378-5397` produces and then checks: the constant source of a zero-pad
        // transfer, every offset empty.
        let checked = DataInfo {
            lds_or_const: Some(LdsOrConst::Constant(ConstantId(4))),
            ..DataInfo::default()
        };
        assert!(checked.is_constant());
        assert!(checked.const_ele_offsets.is_empty());
        assert!(checked.buffer_addr_offset.is_empty());

        // ⛔ AND THE STATE THOSE FOUR `DT_CHECK`S EXIST TO CATCH IS REPRESENTABLE HERE.
        // `transformLxZeroPadInfoInScheduleTree` rewrites the source of a DataInfo it copied from a
        // TENSOR transfer (`dsc/dsc2.cpp:5379-5380`), so the offsets it asserts empty are the ones the
        // old source left behind — which is why the authority checks instead of assuming, and the one
        // assignment that makes this a constant here leaves both carried offsets standing.
        let mut rewritten = DataInfo {
            lds_or_const: Some(LdsOrConst::LabeledDs(LdsIdx(2))),
            ..DataInfo::default()
        };
        rewritten
            .const_ele_offsets
            .entry(CoreId(0))
            .or_default()
            .entry(CoreletId(0))
            .or_default()
            .insert(PrimaryDimTypes::In, ConstEleOffset(8));
        rewritten
            .buffer_addr_offset
            .entry(CoreId(0))
            .or_default()
            .insert(CoreletId(0), BufferAddrOffset(512));
        rewritten.lds_or_const = Some(LdsOrConst::Constant(ConstantId(4)));
        assert!(rewritten.is_constant());
        assert!(!rewritten.const_ele_offsets.is_empty());
        assert!(!rewritten.buffer_addr_offset.is_empty());

        // ⭐ AND ABSENT IS A THIRD CASE, NOT THE CONSTANT ONE: for it the fillers return before even the
        // start address (`ddc/ddcv1.cpp:2364`, `L3DlOpsScheduler.cpp:5786`), so the whole address half
        // of a `None` operand is the declared default and nothing else.
        let neither = DataInfo::default();
        assert!(!neither.is_constant() && !neither.is_labeled_ds());
        assert!(!neither.is_start_addr_symbolic);
        assert!(neither.const_ele_offsets.is_empty() && neither.buffer_addr_offset.is_empty());
    }

    /// `dsc/dsc2.h:884-896`, every arm of the chain in its own order. ⛔ THE ORDER IS LOAD-BEARING
    /// AND THIS IS WHERE THAT SHOWS: a constant source with a constant destination matches the
    /// `CONSTANT_TO_TENSOR` test's first half too, and only the sequence decides.
    ///
    /// ⛔ AN EMPTY DESTINATION LIST IS NOT A NON-TENSOR DESTINATION, IT IS BOTH ANSWERS AT ONCE:
    /// `isDstLabeledDs` and `isDstConstant` each open with `!dstLdsAndLoopOffsets_.empty()`
    /// (`:868-876`), so a constant source with no destination falls all the way through to
    /// `INVALID_TRANSFER_TYPE` while a TENSOR source with no destination is
    /// `NO_TRANSFER_FROM_TENSOR`.
    #[test]
    fn e034_the_transfer_type_chain_is_ordered_and_a_constant_source_takes_the_first_two_arms() {
        let lds = || DataInfo {
            lds_or_const: Some(LdsOrConst::LabeledDs(LdsIdx(0))),
            ..DataInfo::default()
        };
        let constant = || DataInfo {
            lds_or_const: Some(LdsOrConst::Constant(ConstantId(0))),
            ..DataInfo::default()
        };
        let node = |src: DataInfo, dst: Vec<DataInfo>| TransferNode {
            src_lds_and_loop_offsets: src,
            dst_lds_and_loop_offsets: dst,
            ..TransferNode::default()
        };

        assert_eq!(
            node(constant(), vec![constant()]).transfer_type(),
            TransferType::ConstantToConstant
        );
        assert_eq!(
            node(constant(), vec![lds()]).transfer_type(),
            TransferType::ConstantToTensor
        );
        assert_eq!(
            node(lds(), vec![lds()]).transfer_type(),
            TransferType::TensorToTensor
        );
        assert_eq!(
            node(DataInfo::default(), vec![lds()]).transfer_type(),
            TransferType::NoTransferToTensor
        );
        assert_eq!(
            node(lds(), Vec::new()).transfer_type(),
            TransferType::NoTransferFromTensor
        );
        assert_eq!(
            node(lds(), vec![constant()]).transfer_type(),
            TransferType::Invalid,
            "`tensor -> constant` is INVALID: `isDstConstant` fails the fifth arm's second half"
        );
        assert_eq!(
            node(DataInfo::default(), Vec::new()).transfer_type(),
            TransferType::Invalid
        );
        assert_eq!(
            node(constant(), Vec::new()).transfer_type(),
            TransferType::Invalid,
            "`constant -> nothing` is INVALID, not NO_TRANSFER_FROM_TENSOR"
        );

        // Only the FIRST destination is asked, though a multicast has several (`:868-870`).
        assert_eq!(
            node(lds(), vec![DataInfo::default(), lds()]).transfer_type(),
            TransferType::NoTransferFromTensor
        );
    }

    /// `dsc/dsc2.h:867-878`. The five predicates over the four `DataInfo` fields, each reading the
    /// one field the authority names — and `isDstIndirect` reading EMPTINESS alone, never a
    /// destination's contents.
    #[test]
    fn e034_the_data_info_predicates_each_read_their_own_operand() {
        let lds = DataInfo {
            lds_or_const: Some(LdsOrConst::LabeledDs(LdsIdx(1))),
            ..DataInfo::default()
        };

        let mut node = TransferNode::default();
        assert!(!node.is_src_labeled_ds());
        assert!(!node.is_dst_labeled_ds());
        assert!(!node.is_src_constant());
        assert!(!node.is_dst_constant());
        assert!(!node.is_dst_indirect());

        node.src_indirect_lds_and_loop_offsets = lds.clone();
        assert!(
            !node.is_src_labeled_ds(),
            "the INDIRECT source is a different field (`:832`)"
        );

        node.dst_indirect_lds_and_loop_offsets = vec![DataInfo::default()];
        assert!(
            node.is_dst_indirect(),
            "a default `DataInfo` still makes the list non-empty"
        );
        assert!(!node.is_dst_labeled_ds());
    }

    /// ⭐ THE COUPLING THE THREE FIELDS' DOCS NAME, AS A PROPERTY RATHER THAN A SPOT CHECK. The
    /// element count bridge 1 reassembles — `Π extents × numChunks × replicationFactor_`
    /// (`SNTransferLowering.cpp:32-38`, `:963-965`) — is what each DDC writer preserves; the extent
    /// product ALONE is not. One input, the stick sizes `getStickSizes` hands
    /// `populateUnitTimeTransfers`, through every writer the authority applies to this vector.
    #[test]
    fn e034_the_chunk_extents_and_the_replication_factor_are_one_load_size() {
        // The stick sizes of one 256-element load (`ddc/ddcv1.cpp:513`, `dsc/dsc2.h:835-836`).
        let stick_sizes = [(PrimaryDimTypes::In, 4), (PrimaryDimTypes::Ij, 64)];
        let entry = |(dim, size): (PrimaryDimTypes, i32), idx| SizeAndIndex {
            size_dim: Size::new(dim, DimSize(size)),
            src_size_idx: Some(SizeIdx(idx)),
            dst_size_idx: Some(SizeIdx(idx)),
        };
        let extents = |node: &TransferNode| -> i32 {
            node.unit_time_transfer_chunk_size
                .iter()
                .map(|chunk| chunk.size_dim.size.0)
                .product()
        };
        // One `agen` access's element count (`SNTransferLowering.cpp:32-38`, `:963-965`).
        let elements = |node: &TransferNode| {
            extents(node) * node.unit_time_transfer_num_chunks * node.replication_factor
        };

        // `:510-525` with `do2BSplat == false`: one entry per stick size, at its own position.
        let plain = TransferNode {
            unit_time_transfer_chunk_size: stick_sizes
                .iter()
                .enumerate()
                .map(|(i, &size)| entry(size, i as u32))
                .collect(),
            ..TransferNode::default()
        };
        assert_eq!(elements(&plain), 256);
        assert_eq!(extents(&plain), 256);

        // `:525` with `do2BSplat == true` pushes extent 1 for every dim, and `:532-535` puts the
        // product of the stick sizes in the factor instead — the same load, on the OTHER field.
        let mut splat = TransferNode {
            unit_time_transfer_chunk_size: stick_sizes
                .iter()
                .enumerate()
                .map(|(i, &(dim, _))| entry((dim, 1), i as u32))
                .collect(),
            replication_factor: stick_sizes.iter().map(|&(_, size)| size).product(),
            ..TransferNode::default()
        };
        assert_eq!(elements(&splat), 256);
        assert_eq!(extents(&splat), 1, "the extents alone lost the whole load");

        // `:544-548`, the fp32 fixup: a 4 moves BACK out of the factor into entry 0.
        splat.unit_time_transfer_chunk_size[0].size_dim.size = DimSize(4);
        splat.replication_factor /= 4;
        assert_eq!(splat.replication_factor, 64);
        assert_eq!(elements(&splat), 256);
        assert_eq!(extents(&splat), 4);

        // `:1549-1552`, the data-stage shrink of entry 0 from 4 to `dsDim = 2`, in place, with
        // `replicationFactor_ *= size / dsDim`.
        let mut shrunk = plain.clone();
        shrunk.unit_time_transfer_chunk_size[0].size_dim.size = DimSize(2);
        shrunk.replication_factor *= 2;
        assert_eq!(elements(&shrunk), 256);
        assert_eq!(extents(&shrunk), 128);

        // `:1669-1677`, the `reduce2B` tail WITH `doSplat`: every extent to 1, each folded in.
        let mut reduced = splat.clone();
        let mut factor = reduced.replication_factor;
        for chunk in &mut reduced.unit_time_transfer_chunk_size {
            factor *= chunk.size_dim.size.0;
            chunk.size_dim.size = DimSize(1);
        }
        reduced.replication_factor = factor;
        assert_eq!(elements(&reduced), 256);

        // ⛔ AND WITHOUT `doSplat` THAT SAME LOOP DISCARDS IT (`:1673`) — the transfer really did get
        // smaller, which is why the product is an invariant of the SPLAT path alone.
        let mut dropped = splat.clone();
        for chunk in &mut dropped.unit_time_transfer_chunk_size {
            chunk.size_dim.size = DimSize(1);
        }
        assert_eq!(elements(&dropped), 64);
    }

    /// [`LoopCondOp`] IS `CondOp::COMPARISONS`, positionally and by spelling, and the narrowing is
    /// total on both sides. Which operators make up that set is
    /// [`only_the_six_relational_cond_ops_survive_both_ends_of_the_path`]'s job, not this one's.
    #[test]
    fn only_the_six_relational_operators_narrow_onto_a_loop_condition() {
        for (i, op) in CondOp::COMPARISONS.into_iter().enumerate() {
            let narrowed = LoopCondOp::from_cond_op(op);
            assert_eq!(narrowed, Some(LoopCondOp::ALL[i]), "{}", op.name());
            assert_eq!(CondOp::from(LoopCondOp::ALL[i]), op);
            assert_eq!(LoopCondOp::ALL[i].name(), op.name());
        }
        for op in [
            CondOp::Toggle,
            CondOp::Always,
            CondOp::Never,
            CondOp::Const,
            CondOp::Default,
        ] {
            assert_eq!(LoopCondOp::from_cond_op(op), None, "{}", op.name());
        }
    }

    /// The fused value reproduces both JSON fields (`dsc/dsc2.cpp:464-468`) and survives the
    /// importer's join (`:1444-1449`) — plus the one wire pair the fusion deliberately collapses.
    #[test]
    fn a_first_or_last_condition_value_derives_the_authoritys_minus_one() {
        assert_eq!(CondVal::First.val_type(), CondValType::First);
        assert_eq!(CondVal::Last.val_type(), CondValType::Last);
        assert_eq!(CondVal::First.val_int(), CondVal::ABSENT_VAL_INT);
        assert_eq!(CondVal::Last.val_int(), CondVal::ABSENT_VAL_INT);
        assert_eq!(CondVal::ABSENT_VAL_INT, -1);

        for val in [
            CondVal::First,
            CondVal::Last,
            CondVal::Iteration(IterationIdx(0)),
            CondVal::Iteration(IterationIdx(7)),
            // ⭐ a literal `-1` under `INT` stays an iteration index, not the absent marker
            CondVal::Iteration(IterationIdx(CondVal::ABSENT_VAL_INT)),
        ] {
            assert_eq!(CondVal::from_wire(val.val_type(), val.val_int()), val);
        }

        // ⛔ THE ONE DIVERGENCE: an integer beside `FIRST`/`LAST` is discarded, because every reader
        // of those two forms already ignores it and no producer writes one.
        assert_eq!(CondVal::from_wire(CondValType::Last, 4), CondVal::Last);
    }

    /// `addFold` reaches `coordinates_[dim]` before anything can fail (`dsc/dsc2.h:126`), so a refusal
    /// leaves the dim's key behind with an empty tower — the state
    /// [`CoordinateType::clear_fold_for_dim`] produces deliberately. Measured on the authority through
    /// its own refusal, the `UNKNOWN_COORD` `DT_ERROR` (`:140`), which this port cannot spell; the
    /// refusal reachable here is [`FoldManager::build_affine_dim`]'s, a position past the end of the
    /// tower.
    #[test]
    fn a_refused_fold_leaves_the_dims_key_behind() {
        let mut coord = CoordinateType::default();
        assert_eq!(
            coord.add_fold(
                PrimaryDimTypes::Y,
                CoordinateCategory::Spatial,
                FoldDimSize(9),
                "unknown",
                Alpha(1),
                Beta(1),
                FoldDimPos(5),
            ),
            None,
            "position 5 of an empty tower"
        );
        assert!(coord.has_coord_for_dim(PrimaryDimTypes::Y));
        assert_eq!(coord.coordinates()[&PrimaryDimTypes::Y].num_dims(), 0);
        assert_eq!(coord.num_of_spatial_folds(PrimaryDimTypes::Y), 0);
        assert_eq!(
            coord.add_fold(
                PrimaryDimTypes::Y,
                CoordinateCategory::Spatial,
                FoldDimSize(9),
                "unknown",
                Alpha(1),
                Beta(1),
                FoldDimPos(0),
            ),
            Some(()),
            "the control: the same call at a position the tower has"
        );
    }

    /// The five node fields a coordinate is (`dsc/dsc2.h:852`, `:948-949`, `:1008-1009`), each starting
    /// as an empty one, and `TransferNode`'s own copy carrying its coordinate across
    /// (`dsc/dsc2.cpp:1387-1420`) while the pad info it does NOT copy stays default.
    #[test]
    fn the_five_node_coordinate_fields_start_empty_and_a_transfer_copies_its_own() {
        let compute = ComputeNode::default();
        assert!(compute.input_coordinates.is_empty());
        assert_eq!(compute.output_coordinate, CoordinateType::default());

        let allocate = AllocateNode::default();
        assert_eq!(allocate.allocate_coordinates, CoordinateType::default());
        assert_eq!(allocate.slice_view_coordinates, CoordinateType::default());

        let mut transfer = TransferNode::default();
        assert_eq!(transfer.transfer_coordinates, CoordinateType::default());
        assert_eq!(
            transfer.transfer_coordinates.add_fold(
                PrimaryDimTypes::X,
                CoordinateCategory::Spatial,
                FoldDimSize(4),
                "core",
                Alpha(10),
                Beta(1),
                FoldDimPos(0),
            ),
            Some(())
        );
        let copy = transfer.clone();
        assert_eq!(
            copy.transfer_coordinates.tensor_dims(),
            vec![PrimaryDimTypes::X]
        );
        assert_eq!(
            copy.transfer_coordinates
                .num_of_spatial_folds(PrimaryDimTypes::X),
            1
        );
        assert_eq!(
            copy.padding_info,
            TransferPadInfo::default(),
            "the control: the field the authority's copy does not carry"
        );
    }

    /// `coordinates_` and `getTensorDims()` are one map and its keys (`:431`, `:246-252`), so the dim
    /// order `printCoordinates` exports (`:265`) is the same order both readers give — which is why the
    /// field is a [`BTreeMap`] and the dims come out in enum order rather than insertion order.
    #[test]
    fn the_dim_order_is_the_enum_order_not_the_insertion_order() {
        let mut coord = CoordinateType::default();
        for dim in [PrimaryDimTypes::Y, PrimaryDimTypes::In, PrimaryDimTypes::X] {
            assert_eq!(
                coord.add_fold(
                    dim,
                    CoordinateCategory::Spatial,
                    FoldDimSize(2),
                    dim.name(),
                    Alpha(1),
                    Beta(0),
                    FoldDimPos(0),
                ),
                Some(())
            );
        }
        assert_eq!(
            coord.tensor_dims(),
            vec![PrimaryDimTypes::In, PrimaryDimTypes::X, PrimaryDimTypes::Y]
        );
        assert_eq!(
            coord.coordinates().keys().copied().collect::<Vec<_>>(),
            coord.tensor_dims()
        );
    }

    /// `setPadding(PaddingFormType)` (`:240`) replaces every dim's form at once, where
    /// [`CoordinateType::set_padding`] replaces one — IBM's two overloads, which differ only in their
    /// argument and so need two names here.
    #[test]
    fn the_padding_form_setter_replaces_every_dims_form() {
        let mut coord = CoordinateType::default();
        coord.set_padding(PrimaryDimTypes::X, PadType::PaddedNoZeroPad);
        coord.set_padding(PrimaryDimTypes::Y, PadType::PaddedWZeroPad);
        assert_eq!(coord.padding(PrimaryDimTypes::X), PadType::PaddedNoZeroPad);

        let mut replacement = PaddingFormType::default();
        replacement.set_padding(PrimaryDimTypes::Y, PadType::PaddedFullSpan);
        coord.set_padding_form(replacement.clone());
        assert_eq!(
            coord.padding(PrimaryDimTypes::X),
            PadType::NoPad,
            "the form X had is gone, not merged"
        );
        assert_eq!(coord.padding(PrimaryDimTypes::Y), PadType::PaddedFullSpan);
        assert_eq!(coord.padding_form(), &replacement);
    }

    /// `addChildNode`'s sibling arm at the one state the authority cannot survive: a reference that is
    /// not a child of this node. There the scan dereferences BEFORE it tests `end()`
    /// (`dsc/dsc2.cpp:2019-2021`), so it runs off the vector instead of reaching the `DT_ERROR` below
    /// it; here the index is simply out of range and the node comes back whole.
    ///
    /// ⭐ `After(0)` IS `end()` ON A ONE-CHILD LIST AND IS IN RANGE, where `Before(1)` is not — the
    /// asymmetry `insertion_point` gets from the authority's `it` against `it + 1` (`:2023-2027`).
    #[test]
    fn a_refused_child_insertion_hands_the_node_back_and_changes_nothing() {
        let mut block = BlockNode::default();
        block.base_class.name = "B".to_owned();
        let mut first = SyncNode::default();
        first.base_class.name = "s".to_owned();
        assert!(
            block
                .add_child_node(InsertionPoint::Back, ChildNode::Sync(first))
                .is_none()
        );

        let mut orphan = ComputeNode::default();
        orphan.base_class.name = "NEW".to_owned();
        let handed_back = block
            .add_child_node(InsertionPoint::Before(1), ChildNode::Compute(orphan))
            .expect("index 1 is past the only child");
        assert_eq!(handed_back.base().name, "NEW");
        assert_eq!(block.children().len(), 1, "the refusal inserted nothing");

        assert!(
            block
                .add_child_node(InsertionPoint::After(0), handed_back)
                .is_none(),
            "`After(0)` is one past the only child, which is in range"
        );
        assert_eq!(
            block
                .children()
                .iter()
                .map(|child| child.base().name.as_str())
                .collect::<Vec<_>>(),
            ["s", "NEW"]
        );
    }

    /// `deleteChildNode(ownerDsc, node, /*nonDestructive*/ true)` is `childIt->release()` before the
    /// erase (`dsc/dsc2.cpp:2202-2206`) — take the node out and hand it to the caller, which is the
    /// arm `moveChildNode` uses (`:2038`). The DESTRUCTIVE arm is blocked on
    /// `DesignSpaceConfig::cleanupAllocation` and stays off this type, and it is the arm that cleans
    /// up BEFORE it discovers the node is not a child at all (`:2189` against `:2191-2199`).
    #[test]
    fn taking_a_child_out_hands_it_over_and_an_index_past_the_end_is_refused() {
        let mut block = BlockNode::default();
        for name in ["a", "b"] {
            let mut leaf = SyncNode::default();
            leaf.base_class.name = name.to_owned();
            assert!(
                block
                    .add_child_node(InsertionPoint::Back, ChildNode::Sync(leaf))
                    .is_none()
            );
        }

        assert!(
            block.take_child_node(2).is_none(),
            "only two children exist"
        );
        let taken = block.take_child_node(0).expect("index 0 is a child");
        assert_eq!(taken.base().name, "a");
        assert_eq!(block.children().len(), 1);
        assert_eq!(block.children()[0].base().name, "b");
    }

    /// `traverseTreeDFS` seeds its work list from `head_.next_` and never pushes the head itself
    /// (`dsc/dsc2.cpp:2228-2231`), so a filter naming LOOP still does not report the tree's own loop
    /// head — which is why [`ScheduleTree::traverse_dfs`] iterates the children rather than the head.
    #[test]
    fn the_tree_walk_starts_below_the_head_even_when_the_head_matches_the_filter() {
        let mut tree = ScheduleTree::default();
        tree.head_mut().base_class.base_class.name = "head".to_owned();
        let mut inner = LoopNode::default();
        inner.base_class.base_class.name = "inner".to_owned();
        assert!(
            tree.head_mut()
                .base_class
                .add_child_node(InsertionPoint::Back, ChildNode::Loop(inner))
                .is_none()
        );

        assert_eq!(
            tree.head().base_class.base_class.node_type(),
            NodeType::Loop,
            "the head is itself a loop node"
        );
        let visited = tree.traverse_dfs(&[NodeType::Loop], SenComponent::All);
        assert_eq!(
            visited
                .iter()
                .map(|node| node.base().name.as_str())
                .collect::<Vec<_>>(),
            ["inner"]
        );
    }

    /// The exclude list is an `unordered_set<const ScheduleNode*>` tested with `count(currNode)`
    /// (`dsc/dsc2.cpp:2242`), i.e. IDENTITY — so two nodes with the same name are two different
    /// entries. [`ChildNode::traverse_dfs_excluding`] uses [`std::ptr::eq`] for exactly that reason,
    /// and this tree has two `twin`s to prove a name would not do.
    #[test]
    fn the_exclude_list_prunes_by_identity_and_never_by_name() {
        let mut root = BlockNode::default();
        root.base_class.name = "root".to_owned();
        for leaf_name in ["a", "b"] {
            let mut twin = BlockNode::default();
            twin.base_class.name = "twin".to_owned();
            let mut leaf = SyncNode::default();
            leaf.base_class.name = leaf_name.to_owned();
            assert!(
                twin.add_child_node(InsertionPoint::Back, ChildNode::Sync(leaf))
                    .is_none()
            );
            assert!(
                root.add_child_node(InsertionPoint::Back, ChildNode::Block(twin))
                    .is_none()
            );
        }
        let root = ChildNode::Block(root);

        let names = |order: &[&ChildNode]| {
            order
                .iter()
                .map(|node| node.base().name.to_owned())
                .collect::<Vec<_>>()
        };
        assert_eq!(
            names(&root.traverse_dfs(&[], SenComponent::All)),
            ["root", "twin", "a", "twin", "b"]
        );

        let second_twin = &root.as_block().expect("the root is a block").children()[1];
        assert_eq!(
            names(&root.traverse_dfs_excluding(&[], SenComponent::All, &[second_twin])),
            ["root", "twin", "a"],
            "the FIRST twin survives an exclusion of the second"
        );
    }

    /// `insertPerfectlyNestedBlockNode`'s guard is `!nodeToAdd->next_.empty()`
    /// (`dsc/dsc2.cpp:2043-2045`), and the authority THROWS out of it — leaving the caller holding a
    /// node it must still free. Both variants hand it back here, with its children intact, and the
    /// parent untouched.
    ///
    /// ⚠️ THE LOOP VARIANT IS THE ONE WITH A LIVE CALLER: the authority's parameter is `BlockNode*`
    /// and its single caller passes a `LoopNode*` (`ddc/ddc_transformation_util.cpp:287-303`).
    #[test]
    fn a_refused_nesting_hands_the_node_back_with_its_children() {
        let mut parent = BlockNode::default();
        let mut existing = SyncNode::default();
        existing.base_class.name = "existing".to_owned();
        assert!(
            parent
                .add_child_node(InsertionPoint::Back, ChildNode::Sync(existing))
                .is_none()
        );

        let mut nest = BlockNode::default();
        nest.base_class.name = "NEST".to_owned();
        let mut filler = SyncNode::default();
        filler.base_class.name = "filler".to_owned();
        assert!(
            nest.add_child_node(InsertionPoint::Back, ChildNode::Sync(filler))
                .is_none()
        );
        let nest = parent
            .insert_perfectly_nested_block_node(nest)
            .expect("a node that already has children is refused");
        assert_eq!(nest.children()[0].base().name, "filler");
        assert_eq!(parent.children()[0].base().name, "existing");

        let nest_loop = LoopNode {
            base_class: nest,
            ..LoopNode::default()
        };
        let nest_loop = parent
            .insert_perfectly_nested_loop_node(nest_loop)
            .expect("the loop variant has the same guard");
        assert_eq!(nest_loop.base_class.children().len(), 1);
        assert_eq!(parent.children().len(), 1);

        // Emptied, the same node is accepted, and the parent's children move down into it.
        let mut nest_loop = nest_loop;
        assert!(nest_loop.base_class.take_child_node(0).is_some());
        assert!(
            parent
                .insert_perfectly_nested_loop_node(nest_loop)
                .is_none()
        );
        assert_eq!(parent.children().len(), 1);
        let nested = parent.children()[0]
            .as_loop()
            .expect("the nested node is a loop");
        assert_eq!(nested.base_class.children()[0].base().name, "existing");
    }

    /// `moveChildren` is `toNode->next_ = std::move(next_)` with no test at all
    /// (`dsc/dsc2.cpp:2053-2056`), so a destination that already has children LOSES them — which is
    /// what makes [`BlockNode::move_children_to`] an overwrite and not an append.
    #[test]
    fn moving_children_overwrites_the_destinations_own_list() {
        let mut source = BlockNode::default();
        let mut moved = SyncNode::default();
        moved.base_class.name = "moved".to_owned();
        assert!(
            source
                .add_child_node(InsertionPoint::Back, ChildNode::Sync(moved))
                .is_none()
        );

        let mut destination = BlockNode::default();
        let mut dropped = SyncNode::default();
        dropped.base_class.name = "dropped".to_owned();
        assert!(
            destination
                .add_child_node(InsertionPoint::Back, ChildNode::Sync(dropped))
                .is_none()
        );

        source.move_children_to(&mut destination);
        assert!(source.children().is_empty());
        assert_eq!(
            destination
                .children()
                .iter()
                .map(|child| child.base().name.as_str())
                .collect::<Vec<_>>(),
            ["moved"],
            "`dropped` is gone, not appended after"
        );
    }

    /// The condition node's three writers are three DIFFERENT predicates
    /// (`dsc/dsc2.cpp:2143-2167`), and an empty condition is the state that separates them:
    /// `addElseRegion` requires exactly one child (`:2160-2163`) and refuses, while the general
    /// insertion accepts. At two children all three refuse.
    #[test]
    fn an_else_region_needs_a_then_region_and_a_third_region_is_refused() {
        let mut cond = ConditionNode::default();
        let mut early = BlockNode::default();
        early.base_class.name = "early".to_owned();
        let early = cond
            .add_else_region(early)
            .expect("an else region needs a then region first");
        assert_eq!(early.base_class.name, "early");
        assert!(cond.base().children().is_empty());
        assert!(cond.then_branch().is_none());
        assert!(cond.then_core_cl(SenComponent::All).is_none());

        assert!(cond.add_then_region(early).is_none());
        let mut second = BlockNode::default();
        second.base_class.name = "second".to_owned();
        assert!(cond.add_else_region(second).is_none());
        assert_eq!(cond.then_branch().expect("then").base().name, "early");
        assert_eq!(cond.else_branch().expect("else").base().name, "second");

        let mut third = BlockNode::default();
        third.base_class.name = "third".to_owned();
        let third = cond
            .add_child_node(InsertionPoint::Back, third)
            .expect("a third region is refused");
        assert_eq!(third.base_class.name, "third");
        assert!(
            cond.add_then_region(third).is_some(),
            "and so is a then region on a full condition"
        );
        assert_eq!(cond.base().children().len(), 2);
    }

    /// `ConditionNode::getNextView`'s loop-guarded arm discards the caller's component, core AND
    /// corelet and reads the whole child list (`dsc/dsc2.cpp:1996-2001`), so on a loop-guarded
    /// condition the per-corelet reading is the UNFILTERED one — the same widening
    /// [`ConditionNode::next_view`] does, reached through the other entry point.
    #[test]
    fn a_loop_guarded_conditions_per_corelet_view_ignores_the_corelet() {
        let mut cond = ConditionNode::default();
        let mut then_region = BlockNode::default();
        then_region.base_class.name = "then".to_owned();
        then_region.base_class.relevant_comps_mut().insert(
            SenComponent::Lx,
            BTreeMap::from([(CoreId(0), BTreeSet::from([CoreletId(0)]))]),
        );
        let mut else_region = BlockNode::default();
        else_region.base_class.name = "else".to_owned();
        else_region.base_class.relevant_comps_mut().insert(
            SenComponent::Lx,
            BTreeMap::from([(CoreId(0), BTreeSet::from([CoreletId(1)]))]),
        );
        assert!(cond.add_then_region(then_region).is_none());
        assert!(cond.add_else_region(else_region).is_none());

        assert!(cond.has_core_cl_cond(), "`loopCond_` is unset");
        let filtered = cond.next_view_of_corelet(SenComponent::Lx, CoreId(0), Some(CoreletId(1)));
        assert_eq!(
            filtered
                .iter()
                .map(|child| child.base().name.as_str())
                .collect::<Vec<_>>(),
            ["else"]
        );

        cond.loop_cond = Some(LoopCondComposite::from(LoopCondConjunction::new(
            LoopCond {
                dim: PrimaryDimTypes::X,
                cond_op: LoopCondOp::Eq,
                cond_val: CondVal::First,
            },
        )));
        assert!(!cond.has_core_cl_cond());
        assert_eq!(
            cond.next_view_of_corelet(SenComponent::Lx, CoreId(0), Some(CoreletId(1)))
                .len(),
            2,
            "a loop-guarded condition hands back both regions"
        );
        assert_eq!(
            cond.next_view_of_corelet(SenComponent::NoComponent, CoreId(9), None)
                .len(),
            2,
            "including for a component no region is relevant to"
        );
    }

    /// `isParametricLoop_` and `parametricLdsIdx_` are private with one getter and one setter each
    /// (`dsc/dsc2.h:604-619`), and `markAsParametricLoop` only ever SETS — the authority has no
    /// clearing path, which is why there is no `mark_as_non_parametric` here. The index is `-1` until
    /// written, which is [`None`].
    #[test]
    fn the_loops_parametric_marker_only_sets_and_its_lds_index_is_absent_until_written() {
        let mut node = LoopNode::default();
        assert!(!node.is_parametric_loop());
        assert_eq!(node.parametric_lds_idx(), None);

        node.mark_as_parametric_loop();
        assert!(node.is_parametric_loop());
        node.mark_as_parametric_loop();
        assert!(node.is_parametric_loop(), "setting twice is the same state");

        node.set_parametric_lds_idx(Some(LdsIdx(3)));
        assert_eq!(node.parametric_lds_idx(), Some(LdsIdx(3)));
        node.set_parametric_lds_idx(None);
        assert_eq!(
            node.parametric_lds_idx(),
            None,
            "the authority's -1 is expressible, unlike clearing the flag above"
        );

        // `LoopNode(numId, denId, dims, isParametricLoop)` (`dsc/dsc2.h:607-614`) is the only
        // constructor that takes the flag, and it is the DSC2-to-DataflowIR translator's.
        let built = LoopNode::new(
            Some(DataStageId(1)),
            Some(DataStageId(0)),
            vec![PrimaryDimAndKind::new(
                PrimaryDimTypes::X,
                crate::schedule::dims::MetaDimKind::Unpadded,
            )],
            true,
        );
        assert!(built.is_parametric_loop());
        assert_eq!(built.parametric_lds_idx(), None);
        assert!(built.has_loop_dim(PrimaryDimTypes::X));
        assert!(!built.has_loop_dim(PrimaryDimTypes::Y));
    }

    /// `traverseTreeDFSMutable` is the same walk handing out non-const pointers
    /// (`dsc/dsc2.cpp:2208-2220`), and its one live caller rewrites the lds index of every node
    /// beneath a given one (`ddc/ddc_transformation_util.cpp:1920-1930`). ⛔ A VISITOR RATHER THAN A
    /// LIST, because the authority's list ALIASES the tree: an ancestor and its descendant are both
    /// in it, so two `&mut` to the same storage would be live at once.
    #[test]
    fn the_mutable_walk_reaches_every_node_the_read_only_one_does() {
        let mut tree = ScheduleTree::default();
        let mut outer = LoopNode::default();
        outer.base_class.base_class.name = "outer".to_owned();
        let mut leaf = ComputeNode::default();
        leaf.base_class.name = "leaf".to_owned();
        assert!(
            outer
                .base_class
                .add_child_node(InsertionPoint::Back, ChildNode::Compute(leaf))
                .is_none()
        );
        assert!(
            tree.head_mut()
                .base_class
                .add_child_node(InsertionPoint::Back, ChildNode::Loop(outer))
                .is_none()
        );

        let mut seen = Vec::new();
        tree.for_each_dfs_mut(&[], SenComponent::All, |node| {
            seen.push(node.base().name.clone());
            node.base_mut().name.push('!');
        });
        assert_eq!(seen, ["outer", "leaf"]);
        assert_eq!(
            tree.traverse_dfs(&[], SenComponent::All)
                .iter()
                .map(|node| node.base().name.as_str())
                .collect::<Vec<_>>(),
            ["outer!", "leaf!"],
            "every node the read-only walk reports was reached mutably"
        );
    }
}

/// Replaces: CoordinateCategory
///
/// Which third of a dim's fold tower one level belongs to — IBM's `CoordinateCategory`
/// (`dsc/dsc2.h:66-71`), the argument [`CoordinateType::add_fold`] counts the level under. The
/// spatial levels are the outermost, then the temporal ones, then the element-arrangement ones
/// (`ddc/ddc_fold.cpp:2754-2765`, whose `i > temporalFoldEnds` / `i > spatialFoldEnds` chain hands the
/// low positions `SPATIAL_COORD`), which is what makes the three counts a pair of thresholds rather
/// than a per-level tag.
///
/// ⛔ IBM'S FOURTH LITERAL, `UNKNOWN_COORD = 0` (`:67`), IS NOT A CATEGORY AND IS NOT A VARIANT HERE.
/// It has two jobs in the authority and neither is a value this enum must carry: it is
/// `getFoldCategory`'s out-of-range answer (`:224-226`), which
/// [`fold_category`](CoordinateType::fold_category) spells [`None`]; and it is the initialiser a
/// caller gives a local before a TOTAL if/else chain overwrites it — all seven sites do that
/// (`ddc/ddc_fold.cpp:579-586`, `:669`, `:2754-2765`, `:3266-3276`, `:3626`,
/// `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:7492`, `:7708`).
///
/// ⛔ AND LEAVING IT OUT IS WHAT MAKES A DESTRUCTIVE ERROR UNSPELLABLE.
/// `DT_ERROR("[CoordinateType::addFold] Unsupported coordinate category.")` (`:140`) fires AFTER the
/// fold level and its alpha and beta are in place (`:124-137`), so the authority throws out of a
/// coordinate that now holds a level no count covers: measured on the reference, a fourth level added
/// with `UNKNOWN_COORD` survives the throw as position 0 with the label it was given, the three
/// counts unchanged, and `getCoordinateCategoryOfPos(dim, 0)` reporting `SPATIAL_COORD` for it. On a
/// dim with no tower the same call creates the key before throwing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CoordinateCategory {
    /// `SPATIAL_COORD` (`:68`) — a level counted in `numOfSpatialFolds_`, outermost.
    Spatial,
    /// `TEMPORAL_COORD` (`:69`) — a level counted in `numOfTemporalFolds_`, after the spatial ones.
    Temporal,
    /// `ELEM_ARR_COORD` (`:70`) — a level counted in `numOfElemArrFolds_`, innermost.
    ElemArr,
}

impl CoordinateCategory {
    /// The three categories in the authority's discriminant order (`:68-70`).
    pub const ALL: [Self; 3] = [Self::Spatial, Self::Temporal, Self::ElemArr];
}

/// Replaces: e011_CoordinateType
/// Replaces: e012_CoordinateType
///
/// `dsc/dsc2.h:75-440`. How one node sees a tensor's dims folded: per dim, a tower of affine fold
/// levels, how many of those levels are spatial, temporal and element-arrangement, and that dim's
/// padding form. Five node fields are one of these — a transfer's `transferCoordinates_` (`:852`), a
/// compute node's `inputCoordinates_` and `outputCoordinate_` (`:948-949`) and an allocation's
/// `allocateCoordinates_` and `sliceViewCoordinates_` (`:1008-1009`) — and `ddc/ddc_fold.cpp` is the
/// pass that fills them: `:2748-2769` builds an allocation's tower out of its fold params,
/// `:4300-4370` propagates a compute node's input tower onto its output.
///
/// ⭐ NON-GENERIC WHERE IBM IS A TEMPLATE. `CoordinateType<Dtype>` has exactly one instantiation,
/// `CoordinateType<CoordinateBaseType>`, and that macro is `int64_t` (`:442`) — 62 uses tree-wide and
/// no other payload — so the fold payload is [`i64`] and what a level carries is [`Alpha`] and
/// [`Beta`].
///
/// ⛔ `coordinates_` IS PUBLIC IN THE AUTHORITY AND PRIVATE HERE, AND THAT IS WHAT MAKES EVERY LEVEL
/// OF EVERY TOWER AFFINE. [`add_fold`](Self::add_fold) is the only builder and it only ever calls
/// [`FoldManager::build_affine_dim`]; no site outside this class builds through the field — the uses
/// of `coordinates_.at(...)` elsewhere read alphas, betas, sizes and labels (`ddc/ddc_fold.cpp:906`,
/// `:1192-1197`, `:4016-4024`, `dsc/dsc2.cpp:3848-3871`), and the one non-const binding,
/// `auto& allocFoldCurrDim = ...coordinates_[currDim.dim_]` (`ddc/ddc_fold.cpp:2276`), is never read
/// again. [`coordinates`](Self::coordinates) is that read access. The invariant is load-bearing:
/// [`Clone`] is total only because of it.
///
/// ⛔ AND THE JSON IMPORTER IS NOT PORTED. `dsc_import_json` (`:318-358`) rebuilds a coordinate from
/// the object [`print_coordinates`](Self::print_coordinates) writes, through
/// `FoldManager::importFromJson` and `FoldDimProp::importFromJson`
/// (`util/foldManager/foldInfrastructure.h:1553`, `:141`) — neither of which is ported either,
/// because this crate has no JSON reader. Nothing else in the class needs one: the scheduler mints
/// coordinates with [`add_fold`](Self::add_fold).
#[derive(Debug, Default, Eq)]
pub struct CoordinateType {
    /// Field: e011_CoordinateType.coordinates_
    /// Field: e012_CoordinateType.coordinates_
    ///
    /// One affine fold tower per folded dim (`:431`). Ordered, and the order reaches IBM's exported
    /// JSON (`:265`), which is why it is a [`BTreeMap`].
    coordinates: BTreeMap<PrimaryDimTypes, FoldManager<i64>>,
    /// Field: e011_CoordinateType.coreIdToWkSlice_
    /// Field: e012_CoordinateType.coreIdToWkSlice_
    ///
    /// Which work slice of each dim a core covers, for the cores this coordinate is addressed by
    /// (`:432`). Public in the authority too.
    ///
    /// ⛔ EMPTY MEANS "USE THE DSC'S", NOT "NO CORES". Every reader spells that out —
    /// `coordinates.coreIdToWkSlice_.empty() ? sdsc_->coreIdToWkSlice_ : coordinates.coreIdToWkSlice_`
    /// (`ddc/ddcv1.cpp:2690-2700`, `ddc/ddc_fold.cpp:1376-1378`) — and a non-empty one is a CUSTOM map
    /// that `buildFoldFromAllocation` refuses to propagate coordinates through when it varies on the
    /// corelet-split dim (`ddc/ddc_fold.cpp:3110-3120`). The distinction is this map's emptiness and
    /// nothing else, so it stays a map rather than becoming an [`Option`].
    /// ⛔ AND [`clear`](Self::clear) DOES NOT TOUCH IT (`:81-96`) — measured: the folds, the counts and
    /// the padding all go, this map stays.
    pub core_id_to_wk_slice: BTreeMap<CoreId, BTreeMap<PrimaryDimTypes, WkSliceIdx>>,
    /// Field: e011_CoordinateType.foldConstructed_
    /// Field: e012_CoordinateType.foldConstructed_
    ///
    /// Whether the pass that builds this coordinate decided the tower is finished (`:436`), set by
    /// [`complete_fold_construction`](Self::complete_fold_construction) once every dim the allocation
    /// needs is covered (`ddc/ddc_fold.cpp:3282-3284`) and read as a precondition downstream
    /// (`ddc/ddc_fold.cpp:1340`, `:2481`).
    ///
    /// ⛔ NOT PART OF EQUALITY: `operator==` (`:188-213`) never compares it, measured — two towers
    /// built alike compare equal with one of them marked constructed.
    fold_constructed: bool,
    /// Field: e011_CoordinateType.numOfSpatialFolds_
    /// Field: e012_CoordinateType.numOfSpatialFolds_
    ///
    /// How many of a dim's levels are spatial (`:437`), counted up by [`add_fold`](Self::add_fold).
    /// Positions `0..spatial` are the spatial ones.
    num_of_spatial_folds: BTreeMap<PrimaryDimTypes, usize>,
    /// Field: e011_CoordinateType.numOfTemporalFolds_
    /// Field: e012_CoordinateType.numOfTemporalFolds_
    ///
    /// How many of a dim's levels are temporal (`:438`) — positions `spatial..spatial + temporal`.
    num_of_temporal_folds: BTreeMap<PrimaryDimTypes, usize>,
    /// Field: e011_CoordinateType.numOfElemArrFolds_
    /// Field: e012_CoordinateType.numOfElemArrFolds_
    ///
    /// How many of a dim's levels are element-arrangement ones (`:439`). ⛔ NO READER USES IT AS A
    /// THRESHOLD: every position at or past `spatial + temporal` is element arrangement whatever this
    /// says (`:160-163`, `:231-232`), so an entry that disagrees with `getNumDims()` is invisible to
    /// the two category readers and visible to `printCoordinates` and `operator==`.
    num_of_elem_arr_folds: BTreeMap<PrimaryDimTypes, usize>,
    /// Field: e011_CoordinateType.padding_
    /// Field: e012_CoordinateType.padding_
    ///
    /// Each dim's padding form (`:440`). ⛔ COMPARED ONLY FOR THE DIMS THAT HAVE A TOWER: `operator==`
    /// reads `getPadding(coordDim)` inside its `coordinates_` loop (`:202`), so a padding form on a
    /// dim with no folds is not part of equality — and
    /// [`clear_fold_for_dim`](Self::clear_fold_for_dim) deliberately leaves such a form behind
    /// (`:97-99`).
    padding: PaddingFormType,
}

impl CoordinateType {
    /// `clear()` (`:81-96`) — drop every dim's tower, every count and every padding form, and mark the
    /// coordinate unconstructed.
    ///
    /// ⛔ IT DOES NOT CLEAR [`core_id_to_wk_slice`](Self::core_id_to_wk_slice), so this is not
    /// `*self = Self::default()`. Measured: a coordinate cleared after a core map was written still
    /// answers that map.
    /// ⭐ THE `delete` LOOP (`:82-91`) HAS NO COUNTERPART. It frees the `FoldDimProp*`s `addFold`
    /// allocated, walking them out with `getAllDimProFromPos`; here a [`FoldDimProp`] is owned by
    /// value inside [`FoldManager`], so dropping the manager is the whole job — and the destructor
    /// that calls this (`:79`) is [`Drop`] doing the same.
    pub fn clear(&mut self) {
        self.coordinates.clear();
        self.num_of_spatial_folds.clear();
        self.num_of_temporal_folds.clear();
        self.num_of_elem_arr_folds.clear();
        self.padding.clear();
        self.fold_constructed = false;
    }

    /// `clearFoldForDim(dim)` (`:98-114`) — zero one dim's three counts and reset its tower to a
    /// zero-dimension fold space, keeping the dim's key and its padding form.
    ///
    /// ⛔ THE KEY SURVIVING IS OBSERVABLE, AND A COPY THEN DROPS IT.
    /// [`has_coord_for_dim`](Self::has_coord_for_dim) still answers `true` and
    /// [`tensor_dims`](Self::tensor_dims) still lists the dim, but [`Clone`] skips a tower with no
    /// levels, so the copy has neither — measured, and the two therefore compare unequal.
    /// ⛔ AND A COUNT IS ZEROED ONLY IF THE DIM ALREADY HAD ONE (`:101-111`): the authority's
    /// `count(dim)` guards mean this never inserts, which is why it adds nothing.
    pub fn clear_fold_for_dim(&mut self, dim: PrimaryDimTypes) {
        for counts in [
            &mut self.num_of_spatial_folds,
            &mut self.num_of_temporal_folds,
            &mut self.num_of_elem_arr_folds,
        ] {
            if let Some(count) = counts.get_mut(&dim) {
                *count = 0;
            }
        }
        if let Some(fold_manager) = self.coordinates.get_mut(&dim) {
            fold_manager.reset();
        }
    }

    /// `completeFoldConstruction()` (`:116`) — mark the tower finished (`ddc/ddc_fold.cpp:3282-3284`).
    pub fn complete_fold_construction(&mut self) {
        self.fold_constructed = true;
    }

    /// `foldConstructed()` (`:117`).
    pub fn fold_constructed(&self) -> bool {
        self.fold_constructed
    }

    /// `addFold(dim, coordCat, foldCardinality, foldLabel, alpha, beta, pos)` (`:118-142`) — insert one
    /// affine fold level into `dim`'s tower at `pos` and count it under `coord_cat`.
    ///
    /// The one production shape is a loop from the last fold param down to the first, always at
    /// position 0, which leaves the levels in fold-param order with the spatial ones outermost
    /// (`ddc/ddc_fold.cpp:2757-2769`).
    ///
    /// ⛔ [`None`] IS THE FOLD MANAGER'S OWN TWO REFUSALS AND NOTHING ELSE: a `pos` past the end of the
    /// tower or a negative one ([`FoldManager::build_affine_dim`]), and an alpha or beta inserted at a
    /// level that is not affine ([`FoldManager::insert_alpha_beta`]). The category's `DT_ERROR`
    /// (`:140`) is not among them — [`CoordinateCategory`] has no variant for it.
    /// ⛔ AND A REFUSAL LEAVES THE DIM'S KEY BEHIND, because `coordinates_[dim]` default-constructs
    /// before anything can fail (`:126`). Measured on the reference, which creates the key and then
    /// throws.
    /// ⛔ THE CARDINALITY IS [`FoldDimSize`], NOT [`Cardinality`]: IBM takes an `int` and stores it in
    /// `FoldDimProp::factor_`, a `uint32_t` (`util/foldManager/foldInfrastructure.h:153`), so a
    /// negative cardinality is a wrap there and unspellable here.
    ///
    /// Transposing the alpha and the beta is `E0308` — measured 2026-09-19 by compiling the block
    /// below against the built rlibs, one error, "arguments to this method are incorrect", rather than
    /// inferred from the annotation, which stable rustdoc does not check. The control beside it passes
    /// the same two values the right way round through the same public path, so the failure is
    /// attributable to the order and not to a renamed method or a moved parameter.
    /// ```compile_fail,E0308
    /// use deeptools::schedule::dims::PrimaryDimTypes;
    /// use deeptools::schedule::dsc2::{Alpha, Beta, CoordinateCategory, CoordinateType};
    /// use deeptools::schedule::fold::{FoldDimPos, FoldDimSize};
    /// let mut coord = CoordinateType::default();
    /// coord.add_fold(
    ///     PrimaryDimTypes::X,
    ///     CoordinateCategory::Spatial,
    ///     FoldDimSize(4),
    ///     "core",
    ///     Beta(1),
    ///     Alpha(10),
    ///     FoldDimPos(0),
    /// );
    /// ```
    /// ```
    /// use deeptools::schedule::dims::PrimaryDimTypes;
    /// use deeptools::schedule::dsc2::{Alpha, Beta, CoordinateCategory, CoordinateType};
    /// use deeptools::schedule::fold::{FoldDimPos, FoldDimSize};
    /// let mut coord = CoordinateType::default();
    /// assert_eq!(
    ///     coord.add_fold(
    ///         PrimaryDimTypes::X,
    ///         CoordinateCategory::Spatial,
    ///         FoldDimSize(4),
    ///         "core",
    ///         Alpha(10),
    ///         Beta(1),
    ///         FoldDimPos(0),
    ///     ),
    ///     Some(())
    /// );
    /// ```
    pub fn add_fold(
        &mut self,
        dim: PrimaryDimTypes,
        coord_cat: CoordinateCategory,
        fold_cardinality: FoldDimSize,
        fold_label: &str,
        alpha: Alpha,
        beta: Beta,
        pos: FoldDimPos,
    ) -> Option<()> {
        let fold_dim = FoldDimProp::new(fold_cardinality, fold_label);
        let fold_for_curr_dim = self.coordinates.entry(dim).or_default();
        fold_for_curr_dim.build_affine_dim(&fold_dim, pos)?;
        fold_for_curr_dim.insert_alpha_beta(&alpha.0, &beta.0, pos)?;
        let counts = match coord_cat {
            CoordinateCategory::Spatial => &mut self.num_of_spatial_folds,
            CoordinateCategory::Temporal => &mut self.num_of_temporal_folds,
            CoordinateCategory::ElemArr => &mut self.num_of_elem_arr_folds,
        };
        *counts.entry(dim).or_default() += 1;
        Some(())
    }

    /// `getNumOfSpatialFolds(dim)` (`:144-146`) — zero for a dim with no entry.
    pub fn num_of_spatial_folds(&self, dim: PrimaryDimTypes) -> usize {
        self.num_of_spatial_folds.get(&dim).copied().unwrap_or(0)
    }

    /// `getNumOfTemporalFolds(dim)` (`:147-149`).
    pub fn num_of_temporal_folds(&self, dim: PrimaryDimTypes) -> usize {
        self.num_of_temporal_folds.get(&dim).copied().unwrap_or(0)
    }

    /// `getNumOfElemArrFolds(dim)` (`:150-152`).
    pub fn num_of_elem_arr_folds(&self, dim: PrimaryDimTypes) -> usize {
        self.num_of_elem_arr_folds.get(&dim).copied().unwrap_or(0)
    }

    /// `getCoordinateCategoryOfPos(dim, pos)` (`:154-164`) — which category the level at `pos` is
    /// counted under, by the two thresholds the counts define. `dsc/dsc2.cpp:3857` is the caller,
    /// asking whether a fold is an element-arrangement one before it multiplies that level's
    /// cardinality into a dim size.
    ///
    /// ⛔ [`None`] IS THE TWO THROWS AND NOTHING ELSE: `coordinates_.at(dim)` on a dim with no tower
    /// (`:157`) and `DT_CHECK(pos < coord.getNumDims())` (`:158`). This method has no `UNKNOWN_COORD`
    /// arm — its three branches are total over everything that passes the check.
    /// ⛔ A NEGATIVE `pos` PASSES THAT CHECK AND IS REPORTED AS A CATEGORY, and this port keeps that:
    /// measured, position `-1` of a three-level tower answers `SPATIAL_COORD`. Which is why `pos` is a
    /// raw [`FoldDimPos`] here and NOT one of its two resolutions — neither the from-the-end
    /// [`resolve`](FoldDimPos::resolve) nor [`index`](FoldDimPos::index).
    /// ⛔ AND IT DISAGREES WITH [`fold_category`](Self::fold_category) ON EXACTLY THAT POSITION, which
    /// is a difference between the two methods rather than a defect in either.
    pub fn coordinate_category_of_pos(
        &self,
        dim: PrimaryDimTypes,
        pos: FoldDimPos,
    ) -> Option<CoordinateCategory> {
        let num_dims = i64::try_from(self.coordinates.get(&dim)?.num_dims()).ok()?;
        let pos = i64::from(pos.0);
        if pos >= num_dims {
            return None;
        }
        Some(self.category_at(dim, pos))
    }

    /// `getFoldCategory(dim, pos)` (`:222-234`) — the same two thresholds, but answering
    /// `UNKNOWN_COORD` for a position outside the tower instead of throwing. Its three callers read a
    /// level's category off one coordinate and add a fold to another with it
    /// (`ddc/ddc_fold.cpp:4016-4024`, `:4301-4310`, `:4322-4370`).
    ///
    /// ⛔ [`None`] IS BOTH OF THE AUTHORITY'S TWO OUTCOMES HERE — `UNKNOWN_COORD` (`:225`) and the
    /// `coordinates_.at(dim)` throw (`:224`) — and that collapse is safe because it is not observable
    /// downstream: all three callers hand the answer straight to `addFold`, whose `DT_ERROR` on
    /// `UNKNOWN_COORD` (`:140`) ends the run exactly as the `.at` would have. The sentinel cannot
    /// travel further than that in the authority, and here it cannot be spelled at all, since
    /// [`add_fold`](Self::add_fold) takes a [`CoordinateCategory`] that has no such variant.
    /// ⛔ AND ONLY ONE OF THE THREE CALLERS CAN REACH IT: `:4301` and `:4322` bound their loop by the
    /// very tower they then read (`inputFm` IS `computeCoord.coordinates_.at(dim)`), while `:4017`
    /// bounds it by a DIFFERENT coordinate's tower, so a longer right-hand side asks this coordinate
    /// for a level it does not have.
    pub fn fold_category(
        &self,
        dim: PrimaryDimTypes,
        pos: FoldDimPos,
    ) -> Option<CoordinateCategory> {
        let num_dims = i64::try_from(self.coordinates.get(&dim)?.num_dims()).ok()?;
        let pos = i64::from(pos.0);
        if pos < 0 || pos >= num_dims {
            return None;
        }
        Some(self.category_at(dim, pos))
    }

    /// The two thresholds both category readers apply (`:159-163`, `:227-233`), over a position each
    /// has already accepted.
    fn category_at(&self, dim: PrimaryDimTypes, pos: i64) -> CoordinateCategory {
        let spatial = self.num_of_spatial_folds(dim) as i64;
        let temporal = self.num_of_temporal_folds(dim) as i64;
        if pos < spatial {
            CoordinateCategory::Spatial
        } else if pos < spatial + temporal {
            CoordinateCategory::Temporal
        } else {
            CoordinateCategory::ElemArr
        }
    }

    /// `setNumOfTemporalFoldPerDim(dim, num)` (`:215-217`) — overwrite a dim's temporal count.
    ///
    /// ⛔ NO CALLER TREE-WIDE, AND THE AUTHORITY SAYS SO ITSELF: "TEMP. remove these two after fold
    /// type vector is used" (`:214`). Ported because it is this class's own method over this class's
    /// own field, and because what it does to the category readers is a fact about them — measured,
    /// setting the temporal count to 4 on a three-level tower moves BOTH inner positions into
    /// `TEMPORAL_COORD` and leaves the element-arrangement count describing nothing.
    /// ⛔ [`None`] IS `numOfTemporalFolds_.at(dim)` (`:216`): a dim that has never been counted under
    /// this category throws rather than gaining an entry.
    pub fn set_num_of_temporal_fold_per_dim(
        &mut self,
        dim: PrimaryDimTypes,
        num: usize,
    ) -> Option<()> {
        *self.num_of_temporal_folds.get_mut(&dim)? = num;
        Some(())
    }

    /// `setNumOfElemArrFoldPerDim(dim, num)` (`:218-220`) — the same for the element-arrangement
    /// count, with the same absent caller and the same [`None`].
    pub fn set_num_of_elem_arr_fold_per_dim(
        &mut self,
        dim: PrimaryDimTypes,
        num: usize,
    ) -> Option<()> {
        *self.num_of_elem_arr_folds.get_mut(&dim)? = num;
        Some(())
    }

    /// `setPadding(dim, pad)` (`:236-238`).
    pub fn set_padding(&mut self, dim: PrimaryDimTypes, pad: PadType) {
        self.padding.set_padding(dim, pad);
    }

    /// `setPadding(padding)` (`:240`) — replace every dim's form at once. Named apart from
    /// [`set_padding`](Self::set_padding) because IBM's two overloads differ only in their argument.
    pub fn set_padding_form(&mut self, padding: PaddingFormType) {
        self.padding = padding;
    }

    /// `getPadding(dim)` (`:242-244`) — `NoPad` for a dim with no form.
    pub fn padding(&self, dim: PrimaryDimTypes) -> PadType {
        self.padding.padding(dim)
    }

    /// `getPadding()` (`:245`) — every dim's form. Borrowed where IBM returns a copy; a caller that
    /// needs an independent one writes `.clone()`.
    pub fn padding_form(&self) -> &PaddingFormType {
        &self.padding
    }

    /// `getTensorDims()` (`:246-252`) — every dim that has a tower, in dim order.
    ///
    /// ⛔ A DIM WHOSE TOWER WAS CLEARED IS STILL ONE OF THEM, exactly as in
    /// [`has_coord_for_dim`](Self::has_coord_for_dim).
    pub fn tensor_dims(&self) -> Vec<PrimaryDimTypes> {
        self.coordinates.keys().copied().collect()
    }

    /// `hasCoordForDim(dim)` (`:254-256`) — whether this dim has an entry, which is not the same as
    /// having a level: see [`clear_fold_for_dim`](Self::clear_fold_for_dim).
    pub fn has_coord_for_dim(&self, dim: PrimaryDimTypes) -> bool {
        self.coordinates.contains_key(&dim)
    }

    /// The `coordinates_` field's read access (`:431`), which is public in the authority and how every
    /// pass outside this class reaches a level's alpha, beta, size or label.
    ///
    /// ⛔ SHARED ON PURPOSE. A `&mut` here would let a caller build a Map or Constant level and break
    /// the all-affine invariant [`Clone`] rests on; no in-scope site needs one, and the two that
    /// `const_cast` (`dsc/dsc2.cpp:3849`, `ddc/ddc_fold.cpp:4016`) do it only because IBM's readers are
    /// not `const`.
    pub fn coordinates(&self) -> &BTreeMap<PrimaryDimTypes, FoldManager<i64>> {
        &self.coordinates
    }

    /// `printCoordinates(out, printContent, ps)` (`:257-315`) — the coordinate as the JSON object
    /// `dsc_import_json` reads back, `ps` indenting every line of it.
    ///
    /// ⛔ [`None`] IS [`FoldManager::print`]'s REFUSAL and nothing this method decides.
    /// ⭐ A DIM WITH NO LEVELS PRINTS ITS VALUE WHERE THE OTHERS PRINT AN OBJECT — `"folds" : "0"` —
    /// because a zero-dimension manager prints just its datum and ignores the prefix. Measured, and it
    /// is why the output is not JSON-uniform across dims.
    /// ⛔ IBM'S `int foldLevelcount = foldManager.getNumDims()` (`:266`) IS DEAD HERE AND ONLY HERE:
    /// nothing in this body reads it. `debugPrint` computes the same thing and uses it as its loop
    /// bound (`:365`).
    pub fn print_coordinates(
        &self,
        out: &mut String,
        print_content: PrintContent,
        ps: &str,
    ) -> Option<()> {
        let indent = "  ";
        let ps1 = format!("{ps}{indent}");
        let ps2 = format!("{ps1}{indent}");
        let ps3 = format!("{ps2}{indent}");
        out.push('\n');
        out.push_str(ps);
        out.push_str("\"coordinates_\" : {\n");
        out.push_str(&ps1);
        out.push_str("\"coordInfo\" : {\n");
        let mut remaining = self.coordinates.len();
        for (&curr_dim, fold_manager) in &self.coordinates {
            out.push_str(&ps2);
            out.push_str(&format!("\"{}\" : {{\n", curr_dim.name()));
            for (key, count) in [
                ("spatial", self.num_of_spatial_folds(curr_dim)),
                ("temporal", self.num_of_temporal_folds(curr_dim)),
                ("elemArr", self.num_of_elem_arr_folds(curr_dim)),
            ] {
                out.push_str(&ps3);
                out.push_str(&format!("\"{key}\" : {count},\n"));
            }
            out.push_str(&ps3);
            out.push_str(&format!(
                "\"padding\" : \"{}\",\n",
                self.padding.padding_as_str(curr_dim)
            ));
            out.push_str(&ps3);
            out.push_str("\"folds\" : ");
            fold_manager.print(out, &ps3, print_content)?;
            out.push('\n');
            out.push_str(&ps2);
            out.push('}');
            remaining -= 1;
            if remaining > 0 {
                out.push_str(", ");
            }
            out.push('\n');
        }
        out.push_str(&ps1);
        out.push_str("},\n");
        out.push_str(&ps1);
        out.push_str("\"coreIdToWkSlice_\" : { \n");
        let mut remaining = self.core_id_to_wk_slice.len();
        for (core_id, wk_slice) in &self.core_id_to_wk_slice {
            out.push_str(&ps2);
            out.push_str(&format!("\"{}\" : {{ ", core_id.0));
            let mut inner = wk_slice.len();
            for (dim, slice) in wk_slice {
                out.push_str(&format!("\"{}\" : {}", dim.name(), slice.0));
                inner -= 1;
                if inner > 0 {
                    out.push_str(", ");
                }
            }
            out.push_str(" }");
            remaining -= 1;
            if remaining > 0 {
                out.push_str(", ");
            }
            out.push('\n');
        }
        out.push_str(&ps1);
        out.push_str("} \n");
        out.push_str(ps);
        out.push_str("}\n");
        Some(())
    }

    /// `debugPrint(out, printContent, ps)` (`:360-425`) — the human-readable dump the fold passes write
    /// under `coordPropReportLevel_ > 2` (`ddc/ddc_fold.cpp:593`, `:2773`, `:3283`, `:4381`).
    ///
    /// ⭐ IBM'S `printContent` AND `ps` ARE GONE BECAUSE THE BODY NEVER READS EITHER. Measured: the
    /// output with `(true, "IGNORED")` is byte-identical to the output with the defaults, and every
    /// call site passes the defaults anyway.
    /// ⛔ [`None`] IS `foldDims.at(i)` (`:372`) AND `collectFoldFunctionAtLevel`'s `DT_CHECK` (`:374`),
    /// both unreachable from here: the loop is bounded by the same `getNumDims()` that sizes the one
    /// and guards the other.
    /// ⛔ AND THE `WkSplit_leaf` ARM (`:377-378`) IS UNREACHABLE RATHER THAN UNPORTED, for the reason
    /// [`FoldManager::build_dim`] gives: nothing builds a WkSplit level, so no tower holds one. A
    /// Constant or Map level cannot be here either — [`add_fold`](Self::add_fold) builds affine levels
    /// only — which leaves the two affine arms as the whole of this dump.
    pub fn debug_print(&self, out: &mut String) -> Option<()> {
        out.push_str("\nDDC Coordinates<int64_t>: ");
        out.push_str(&format!("{} coordinate entries", self.coordinates.len()));
        for (&curr_dim, fold_manager) in &self.coordinates {
            out.push_str(&format!("\n\nPrimary Dim= {}", curr_dim.name()));
            let fold_dims = fold_manager.fold_dim_props();
            for pos in 0..fold_manager.num_dims() {
                out.push_str("\n  Fold dimension= ");
                fold_dims.get(pos)?.print(out);
                for fold_function in fold_manager.collect_at_level(pos)? {
                    match fold_function {
                        FoldFunc::AffineNonLeaf(affine) => {
                            out.push_str("\n    Affine:");
                            affine.print_meta_data(out, "");
                        }
                        FoldFunc::AffineLeaf(affine) => {
                            out.push_str("\n    Affine: ");
                            affine.print_meta_data(out, "");
                        }
                        FoldFunc::ConstantNonLeaf(_) => out.push_str("\n    Constant "),
                        FoldFunc::ConstantLeaf(_)
                        | FoldFunc::MapNonLeaf(_)
                        | FoldFunc::MapLeaf(_) => {}
                    }
                }
            }
            out.push_str(&format!(
                "\n  #Spatial  = {}\n  #Temporal = {}\n  #ElemArr  = {}",
                self.num_of_spatial_folds(curr_dim),
                self.num_of_temporal_folds(curr_dim),
                self.num_of_elem_arr_folds(curr_dim)
            ));
            out.push_str(&format!(
                "\n  Padding: {{ ({}, {}) }}",
                curr_dim.name(),
                self.padding.padding_as_str(curr_dim)
            ));
        }
        if !self.core_id_to_wk_slice.is_empty() {
            out.push_str("\n  coreIdToWkSlice_ : { \n");
            let mut remaining = self.core_id_to_wk_slice.len();
            for (core_id, wk_slice) in &self.core_id_to_wk_slice {
                out.push_str(&format!("    {} : {{ ", core_id.0));
                let mut inner = wk_slice.len();
                for (dim, slice) in wk_slice {
                    out.push_str(&format!("{} : {}", dim.name(), slice.0));
                    inner -= 1;
                    if inner > 0 {
                        out.push_str(", ");
                    }
                }
                out.push_str(" }");
                remaining -= 1;
                if remaining > 0 {
                    out.push_str(", ");
                }
                out.push('\n');
            }
            out.push_str("  } \n");
        }
        Some(())
    }
}

impl Clone for CoordinateType {
    /// `operator=(const CoordinateType&)` (`:166-184`) and the copy constructor that delegates to it
    /// (`:186`) — the only way a coordinate is copied in the authority, and what every node's own copy
    /// reaches.
    ///
    /// ⛔ IT IS A REBUILD, NOT A MEMBERWISE COPY, AND IT DROPS A DIM WHOSE TOWER HAS NO LEVELS. The
    /// authority clears, then walks each dim's levels from the innermost out and replays them through
    /// `addFold` (`:170-179`), so a dim with zero levels contributes no call and never reaches the
    /// copy's `coordinates_` — measured: a coordinate whose one dim was cleared with
    /// [`clear_fold_for_dim`](Self::clear_fold_for_dim) copies to an empty one that compares UNEQUAL to
    /// its source, while the padding form for that same dropped dim survives.
    /// ⭐ THE REPLAY IS A LEVEL-FOR-LEVEL IDENTITY ON AN AFFINE TOWER, which is what lets this be a
    /// [`Clone`] at all: `addFold` re-inserts each level at position 0 in innermost-to-outermost order,
    /// so the copy's sizes, labels, alphas and betas come back in the source's order — measured against
    /// the reference for a three-level tower. Cloning the [`FoldManager`] instead of replaying it is
    /// therefore the same tower, and it needs neither `getAlphaBeta` nor the `insertAlphaBeta` that
    /// could refuse. The invariant it rests on is [`coordinates`](Self::coordinates) being read-only.
    /// ⛔ BUT THE COUNTS ARE RECOMPUTED, NOT COPIED, and that is visible whenever they disagree with the
    /// tower: the replay counts each level under the category the SOURCE reports for its position, so a
    /// temporal count raised past the tower's depth by
    /// [`set_num_of_temporal_fold_per_dim`](Self::set_num_of_temporal_fold_per_dim) comes back as the
    /// number of positions that actually read as temporal.
    fn clone(&self) -> Self {
        let mut copy = Self::default();
        for (&dim, fold_manager) in &self.coordinates {
            let num_dims = fold_manager.num_dims();
            if num_dims == 0 {
                continue;
            }
            copy.coordinates.insert(dim, fold_manager.clone());
            for pos in 0..num_dims {
                let counts = match self.category_at(dim, pos as i64) {
                    CoordinateCategory::Spatial => &mut copy.num_of_spatial_folds,
                    CoordinateCategory::Temporal => &mut copy.num_of_temporal_folds,
                    CoordinateCategory::ElemArr => &mut copy.num_of_elem_arr_folds,
                };
                *counts.entry(dim).or_default() += 1;
            }
        }
        copy.padding = self.padding.clone();
        copy.core_id_to_wk_slice = self.core_id_to_wk_slice.clone();
        copy.fold_constructed = self.fold_constructed;
        copy
    }
}

impl PartialEq for CoordinateType {
    /// `operator==(const CoordinateType&)` (`:188-213`) — the same dims, each with the same three
    /// counts, the same padding form and the same tower, plus the same core-to-work-slice map.
    ///
    /// ⛔ IT IGNORES `foldConstructed_`, measured — two coordinates built alike are equal with one of
    /// them marked constructed. A derived [`PartialEq`] would compare it.
    /// ⛔ AND IT COMPARES PADDING ONLY FOR THE DIMS THAT HAVE A TOWER (`:202`), so a form left behind on
    /// a dim with no folds is invisible to equality — which is exactly the state
    /// [`clear_fold_for_dim`](Self::clear_fold_for_dim) leaves. A derived one would compare that too.
    fn eq(&self, rhs: &Self) -> bool {
        if self.coordinates.len() != rhs.coordinates.len() {
            return false;
        }
        for (&coord_dim, lhs_fm) in &self.coordinates {
            let Some(rhs_fm) = rhs.coordinates.get(&coord_dim) else {
                return false;
            };
            if self.num_of_spatial_folds(coord_dim) != rhs.num_of_spatial_folds(coord_dim)
                || self.num_of_temporal_folds(coord_dim) != rhs.num_of_temporal_folds(coord_dim)
                || self.num_of_elem_arr_folds(coord_dim) != rhs.num_of_elem_arr_folds(coord_dim)
                || self.padding(coord_dim) != rhs.padding(coord_dim)
            {
                return false;
            }
            if lhs_fm != rhs_fm {
                return false;
            }
        }
        self.core_id_to_wk_slice == rhs.core_id_to_wk_slice
    }
}

/// `dsc2::ScheduleNode` (`dsc/dsc2.h:444-524`) — the base every node in a schedule tree derives
/// from. `BlockNode` (`:526`) derives from it and owns the children; `LoopNode` (`:563`) and
/// `ConditionNode` (`:685`) derive from `BlockNode`; `TransferNode` (`:814`), `ComputeNode`
/// (`:900`), `SyncNode` (`:964`), `AllocateNode` (`:974`) and `StickMaskNode` (`:1059`) derive from
/// it directly. Those eight are the whole hierarchy, and [`NodeType`] is its discriminant — the
/// importer's `new`-per-kind chain is the exhaustive list (`dsc/dsc2.cpp:1337-1358`).
///
/// ⛔ THIS CARRIES 3 OF SCHEDULENODE'S 4 FIELDS, so the `e041_ScheduleNode`, `e029_ScheduleNode` and
/// `e013_ScheduleNode` anchors below stay open. `prev_` (`dsc/dsc2.h:515`) is a `BlockNode*` pointing back at the parent
/// that OWNS this node, through `BlockNode::next_`, a `VectorOfChildren` of `unique_ptr`s (`:538`).
/// Every reader of it is a tree operation, not a question about one node: `getPrev` and
/// `getMutableParent` hand it straight out (`:463-464`), `getOwnerLoop` climbs it to the nearest
/// `LOOP` (`dsc/dsc2.cpp:1896-1900`), `getParentDimLoop` climbs on from there to the nearest loop
/// carrying a dim (`:1906-1914`), `insertLoopAbove` finds `this` in `prev_->next_` by ADDRESS,
/// `nodePtr.get() == this`, and splices a loop into its slot (`:2169-2186`), and `moveNode` forwards
/// the whole job to `prev_->moveChildNode` (`:1977-1982`). So the identity a Rust parent link would
/// need is the one `ScheduleTree::head_` (`dsc/dsc2.h:623`) and `BlockNode::next_` have to define,
/// and both of those anchors are open — `e030_BlockNode.next_` and `e032_ScheduleTree.head_`, whose
/// superseded `e015_`/`e007_` duplicates this changeset removed.
///
/// ⛔ NINE OF THE `e041_ScheduleNode` GENERATION'S TWENTY-ONE FIELD ANCHORS STAY OPEN, AND ONLY TWO OF
/// THEM NAME A MEMBER — the scheduler's field scan is not a declaration parser. The two are `prev_`
/// above and `UnitView::LoopInfo::loop_` (`dsc/dsc2.h:501`), the `const LoopNode*` that stops
/// [`UnitView`]'s `print` as well. FIVE ARE THE CLASS'S `friend` CLASSES — `BlockNode`, `Ddc`,
/// `DesignSpaceConfig`, `L3DlOpsScheduler` and `ScheduleTree` (`:518-522`) — while the sixth friend, a
/// function (`:523`), is not scanned. THE LAST TWO ARE METHOD PARAMETERS carrying a default argument,
/// `comp` (`:470`, `:472`) and `clId` (`:470`, `:474`), and `coreId` carries one on those same two
/// declarations (`:470`, `:473`) yet is not scanned. Nor is the real member `sizesWithGaps_` (`:509`),
/// which the two earlier generations do name and [`UnitView::sizes_with_gaps`] carries.
///
/// ⛔ AND `name_` IS NOT THAT IDENTITY IN MEMORY, ONLY ON THE JSON SEAM. Names are made unique by
/// `finalizeScheduleTree`, which appends `__1`, `__2`, … as it walks and `DT_ERROR`s on a node with
/// no name at all (`dsc/dsc2.cpp:2976-2992`), and the tree importer refuses a duplicate outright
/// (`:1369-1371`). Before that pass runs the authority mints colliding names deliberately — every
/// condition node the DDL conversion builds is named the literal `"condition"`
/// (`ddc/ddl/ddl_conversion.cpp:1556`), and every implicit L0 sync `"sync_implicit_L0"` (`:1790`).
///
/// ⛔ NO `Default`: the authority's only constructor takes the kind (`dsc/dsc2.h:482`), every
/// concrete node passes its own, and the field is `const` (`:460`), so [`NodeType::Invalid`] is what
/// a base that was never constructed reads as rather than a kind any C++ path produces. [`Size`]
/// above refuses one for the same reason.
///
/// ⛔ NO `PartialEq`: the authority declares none, and a derive would answer "the same node" for two
/// distinct nodes agreeing on kind, name and relevance — exactly what `insertLoopAbove`'s
/// `nodePtr.get() == this` (`dsc/dsc2.cpp:2177`) and `finalizeScheduleTree`'s uniquifier
/// (`:2988-2992`) exist to tell apart.
///
/// ⛔ `clone()` IS NOT PORTED AND IS NOT [`Clone`]. The authority's is pure virtual (`dsc/dsc2.h:484`)
/// and every node in the hierarchy inherits its override from `InheritWithClone<Base, Derived>`
/// (`:526`, `:563`, `:685`, `:814`), which `new`s a node of the DERIVED type — an operation this base
/// cannot answer. [`Clone`] here copies the three carried fields, which is the implicit copy
/// constructor, not `clone()`.
#[derive(Clone, Debug)]
pub struct ScheduleNode {
    /// Field: e029_ScheduleNode.nodeType_
    /// Field: e013_ScheduleNode.nodeType_
    /// Field: e041_ScheduleNode.nodeType_
    ///
    /// Which kind of node this is (`dsc/dsc2.h:460`). PRIVATE because the authority's is `const`:
    /// it is fixed by the constructor and there is no path that rewrites it, which is what lets the
    /// JSON exporter and importer dispatch on it (`dsc/dsc2.cpp:376-377`, `:1337-1358`).
    ///
    /// ⚠️ AND C++ CANNOT SPELL A WHOLE-VALUE OVERWRITE AT ALL, WHERE RUST CAN. The `const` member
    /// deletes the implicitly-declared copy assignment and the class declares none
    /// (`dsc/dsc2.h:444-524`), so the authority copies a node only through the pure-virtual `clone()`
    /// (`:484`) — which is what [`Clone`] is here. A holder of `&mut ScheduleNode` can still write
    /// `*node = ScheduleNode::new(other)`, and no Rust construct forbids that; it is the one way this
    /// field moves, and it moves the whole node with it.
    node_type: NodeType,
    /// Field: e029_ScheduleNode.name_
    /// Field: e013_ScheduleNode.name_
    /// Field: e041_ScheduleNode.name_
    ///
    /// The node's name (`dsc/dsc2.h:461`). It is the node's identity ON THE JSON SEAM: the exporter
    /// writes a `LoopInfo`'s loop as `li.loop_->name_` (`dsc/dsc2.cpp:209-210`) and the importer
    /// resolves it back through `nodeNamePtrMap` (`:1284-1286`), the same map that refuses a
    /// duplicate name (`:1369-1371`).
    ///
    /// ⛔ AND IT IS WRITTEN AFTER THE FACT, so it is not a construction-time identity: every node
    /// reaches `finalizeScheduleTree` with whatever name minted it, and that pass renames the
    /// collisions (`dsc/dsc2.cpp:2988-2992`). A `String` and not an enum because the set is open —
    /// the DDL conversion builds one per template loop, `loop_ds<numId>_ds<denId>`
    /// (`ddc/ddl/ddl_conversion.cpp:1104`), and one per allocation, `allocate_<node>` (`:1627`).
    pub name: String,
    /// Field: e029_ScheduleNode.relevantComps_
    /// Field: e013_ScheduleNode.relevantComps_
    /// Field: e041_ScheduleNode.relevantComps_
    ///
    /// Which components, cores and corelets this node is relevant to (`dsc/dsc2.h:516`) — a
    /// component, then that component's cores, then each core's corelets.
    ///
    /// NOT A PUBLIC FIELD, because the authority's is `protected` behind five friend classes and one
    /// friend function (`dsc/dsc2.h:514-523`) while every reading below is a public method: a caller
    /// that wants "is this node relevant to LX" gets that answer, and only the two whole-map
    /// accessors, [`Self::relevant_comps`] and [`Self::relevant_comps_mut`], hand over the map.
    ///
    /// ⛔ [`SenComponent::All`] IS NEVER A KEY, AND [`SenComponent::NoComponent`] IS ONE ONLY UNTIL
    /// FINALIZATION. `setRelevantCompCoreCl` seeds the head and then every node with a
    /// `NO_COMPONENT` entry, splits a condition node's two branches under that same key, and only
    /// then fills the real components from each location, "Leave NO_COMPONENT in relevantComps_ as
    /// it is useful for analysis" (`dsc/dsc2.cpp:2647-2729`) — and `finalizeScheduleTree` erases that
    /// key from the head and from every node (`:2977-2980`). `ALL` reaches the map from neither
    /// writer, nor from the importer that rebuilds it key by key (`:1373-1382`), which is why
    /// [`Self::is_relevant`] answers it from the authority's early return instead of a lookup.
    ///
    /// ⛔ AN EMPTY CORELET SET IS A REAL STATE AND THE THREE READERS DISAGREE ABOUT IT.
    /// `isNodeRelevant(comp, -1, coreId)` returns true on nothing but that core's PRESENCE
    /// (`:1929-1931`), while `getRelevantComps(coreId)` requires `!clSet.empty()` (`:1964-1967`) and
    /// `getRelevantCoreCl` drops such a core from its result altogether (`:1940-1942`). So neither
    /// reader can be composed out of the other, and both are ported.
    ///
    /// ⛔ BUT `setRelevantCompCoreCl` IS NOT THE WRITER THAT MINTS IT — THREE GUARDS CLOSE THAT
    /// ROUTE. `relCoreCls[core].insert(cls.begin(), cls.end())` does create the core's entry before
    /// it inserts anything (`:2692-2694`), but its `cls` comes from the `NO_COMPONENT` map, and every
    /// core there is seeded `coreCorelets[coreId] = clIds` with `clIds` never empty (`:2648`,
    /// `:2652`); a condition node's then-branch `emplace`s only a NON-EMPTY intersection (`:2671`),
    /// and its else-branch ERASES a core whose difference came out empty (`:2681`). So that `insert`
    /// always carries at least one corelet.
    ///
    /// ⭐ THE WRITER THAT DOES MINT IT IS THE JSON IMPORTER, AND THE STATE ROUND-TRIPS: `auto& core =
    /// comp[std::stoi(corePair.first)]` creates the core from the key alone, and an empty `[]` leaves
    /// it empty (`:1373-1382`) — which the exporter writes straight back out as `"0":[]`
    /// (`:387-393`). The head node's union over its children mints it a second time — an empty `cls`
    /// in `headComp[core].insert(cls.begin(), cls.end())` (`:1389-1395`) — on the one node the ddc
    /// then `DT_CHECK`s, and an empty corelet set satisfies that check because what it tests is that
    /// the MAP is non-empty (`ddc/ddcv1.cpp:3458`).
    relevant_comps: BTreeMap<SenComponent, BTreeMap<CoreId, BTreeSet<CoreletId>>>,
}

impl ScheduleNode {
    /// `ScheduleNode(NodeType)` (`dsc/dsc2.h:482`), the authority's only constructor — `name_` starts
    /// empty (`:461`) and `relevantComps_` starts empty (`:516`).
    pub fn new(node_type: NodeType) -> Self {
        Self {
            node_type,
            name: String::new(),
            relevant_comps: BTreeMap::new(),
        }
    }

    /// `nodeType_` (`dsc/dsc2.h:460`), read-only because the authority's is `const`.
    pub fn node_type(&self) -> NodeType {
        self.node_type
    }

    /// `isBlockNode` (`dsc/dsc2.h:479-481`), which reads nothing but the kind — see
    /// [`NodeType::is_block_node`] for which three kinds those are and why.
    pub fn is_block_node(&self) -> bool {
        self.node_type.is_block_node()
    }

    /// `isNodeRelevant(comp)` with both filters left at their `-1` defaults
    /// (`dsc/dsc2.h:470`, `dsc/dsc2.cpp:1916-1928`): [`SenComponent::All`] means "do not filter by
    /// component" and every node answers yes to it (`:1918-1922`), and any other component is
    /// relevant exactly when it has an entry (`:1923-1928`). This is the reading the ddc and the L3
    /// scheduler take to decide whether a transfer belongs to a unit
    /// (`ddc/ddcv1.cpp:2818`, `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:6285`).
    ///
    /// ⛔ THE CORE- AND CORELET-FILTERED READINGS ARE [`Self::relevant_cores`]'s, NOT THIS ONE'S. The
    /// authority's `DT_ERROR("Cannot filter node by clId/coreId and not by SenComponent")`
    /// (`dsc/dsc2.cpp:1919-1921`) fires for one argument combination — a core or corelet filter
    /// beside a component filter of `ALL` — and no function here takes both a component and a core,
    /// so that combination cannot be spelled. `isNodeRelevant(comp, -1, coreId)` is
    /// `relevant_cores(comp).is_some_and(|cores| cores.contains_key(&core))` and
    /// `isNodeRelevant(comp, clId, coreId)` continues into the corelet set — the two readings the
    /// bridge's sync lowering and the tree traversals call for
    /// (`dsc-based-utils/DSC2ToDataflowIR/V3/SNSyncLowering.cpp:109`, `dsc/dsc2.cpp:1988`, `:2245`).
    ///
    /// ```
    /// use deeptools::schedule::dsc2::{NodeType, ScheduleNode};
    /// use sys_arch_spec::arch_enums::SenComponent;
    ///
    /// let node = ScheduleNode::new(NodeType::Transfer);
    /// assert!(node.is_relevant(SenComponent::All));
    /// assert!(!node.is_relevant(SenComponent::Lx));
    /// ```
    pub fn is_relevant(&self, comp: SenComponent) -> bool {
        comp == SenComponent::All || self.relevant_comps.contains_key(&comp)
    }

    /// `isNodeRelevant(comp, clId, coreId)` with a real core (`dsc/dsc2.cpp:1916-1933`): whether this
    /// node belongs to ONE unit. `getNextView`'s filtered reading is this
    /// (`dsc/dsc2.cpp:1988`) and so is the sync lowering's test that a signal's end is in the unit
    /// being emitted (`dsc-based-utils/DSC2ToDataflowIR/V3/SNSyncLowering.cpp:109`).
    ///
    /// ⛔ AN ABSENT CORELET IS "ANY CORELET OF THAT CORE", WHICH IS THE `clId < 0 ||` AT `:1931` — the
    /// core has to be listed either way, and only the corelet set is skipped. ⛔ AND A COMPONENT OF
    /// [`SenComponent::All`] IS THE AUTHORITY'S ONE REFUSAL HERE (`:1919-1921`), narrowed to `false`:
    /// `ALL` is never a key in `relevantComps_` (see [`Self::is_relevant`]), so a core-filtered
    /// question about "every component" answers no. Both live callers pass the component of the unit
    /// they are lowering (`DSC2ToDataflowIR.cpp:291-300`, `SNSyncLowering.cpp:105-109`).
    pub fn is_relevant_to_corelet(
        &self,
        comp: SenComponent,
        core: CoreId,
        corelet: Option<CoreletId>,
    ) -> bool {
        self.relevant_cores(comp)
            .and_then(|cores| cores.get(&core))
            .is_some_and(|corelets| corelet.is_none_or(|cl| corelets.contains(&cl)))
    }

    /// `isNodeRelevant`'s `relevantComps_.find(comp)` (`dsc/dsc2.cpp:1923-1924`) — this component's
    /// cores and their corelets, absent when the component is not relevant at all. See
    /// [`Self::is_relevant`] for why the core-filtered readings are spelled from here.
    pub fn relevant_cores(
        &self,
        comp: SenComponent,
    ) -> Option<&BTreeMap<CoreId, BTreeSet<CoreletId>>> {
        self.relevant_comps.get(&comp)
    }

    /// `getRelevantCoreCl()` with its default `ALL`, i.e. no component filter
    /// (`dsc/dsc2.h:471-472`, `dsc/dsc2.cpp:1935-1945`): every relevant core, with the corelets of
    /// ALL components merged. This is the reading the ddc and the DDL conversion take
    /// (`ddc/ddcv1.cpp:2701`, `:3467`, `:3476`, `ddc/ddl/ddl_conversion.cpp:3233`).
    ///
    /// ⛔ A CORE WITH AN EMPTY CORELET SET IS DROPPED, not carried empty — `if (!cls.empty())`
    /// (`dsc/dsc2.cpp:1941`) — so this is not `relevantComps_` flattened.
    pub fn relevant_core_cl(&self) -> BTreeMap<CoreId, BTreeSet<CoreletId>> {
        let mut all_core_cl: BTreeMap<CoreId, BTreeSet<CoreletId>> = BTreeMap::new();
        for core_cl in self.relevant_comps.values() {
            for (core, cls) in core_cl {
                if !cls.is_empty() {
                    all_core_cl
                        .entry(*core)
                        .or_default()
                        .extend(cls.iter().copied());
                }
            }
        }
        all_core_cl
    }

    /// `getRelevantCoreCl(comp)` with a component named (`dsc/dsc2.cpp:1935-1945`) — the fold reads
    /// it for one component and then again for `NO_COMPONENT` (`ddc/ddc_fold.cpp:1371-1374`).
    ///
    /// ⛔ [`SenComponent::All`] IS THE AUTHORITY'S "NO FILTER" HERE, NOT A COMPONENT TO MATCH:
    /// `if (comp != SenComponents::ALL && comp != relevantComp) continue` (`dsc/dsc2.cpp:1939`)
    /// skips the test entirely for it, so it delegates to [`Self::relevant_core_cl`] rather than
    /// looking for a key that no writer creates.
    pub fn relevant_core_cl_of_comp(
        &self,
        comp: SenComponent,
    ) -> BTreeMap<CoreId, BTreeSet<CoreletId>> {
        if comp == SenComponent::All {
            return self.relevant_core_cl();
        }
        let mut all_core_cl: BTreeMap<CoreId, BTreeSet<CoreletId>> = BTreeMap::new();
        if let Some(core_cl) = self.relevant_comps.get(&comp) {
            for (core, cls) in core_cl {
                if !cls.is_empty() {
                    all_core_cl.insert(*core, cls.clone());
                }
            }
        }
        all_core_cl
    }

    /// `getRelevantComps()` with both filters at their `-1` defaults
    /// (`dsc/dsc2.h:473-474`, `dsc/dsc2.cpp:1947-1975`): the components relevant to ANY core, which
    /// is every component whose core map is non-empty (`:1969-1971`).
    ///
    /// ⚠️ `getRelevantComps`'s ONLY CALLER TREE-WIDE IS `dsc/dsc2Pcfg.cpp:64`, and DCG/PCFG is off
    /// this campaign's path — so all three readings below are ported because the method belongs to
    /// this unit, not because anything in scope reads them yet.
    pub fn relevant_comps_any_core(&self) -> BTreeSet<SenComponent> {
        self.relevant_comps
            .iter()
            .filter(|(_, core_cl)| !core_cl.is_empty())
            .map(|(comp, _)| *comp)
            .collect()
    }

    /// `getRelevantComps(coreId)` (`dsc/dsc2.cpp:1955-1968`) — the components relevant to THAT core
    /// through any of its corelets, which is the reading `dsc/dsc2Pcfg.cpp:64` takes.
    ///
    /// ⛔ A CORE PRESENT WITH AN EMPTY CORELET SET COUNTS FOR NOTHING HERE, `!clSet.empty()`
    /// (`dsc/dsc2.cpp:1964`), where `isNodeRelevant` on the same core answers true — see
    /// [`Self::relevant_comps`].
    pub fn relevant_comps_of_core(&self, core: CoreId) -> BTreeSet<SenComponent> {
        self.relevant_comps
            .iter()
            .filter(|(_, core_cl)| core_cl.get(&core).is_some_and(|cls| !cls.is_empty()))
            .map(|(comp, _)| *comp)
            .collect()
    }

    /// `getRelevantComps(coreId, clId)` with both named (`dsc/dsc2.cpp:1959-1963`) — the components
    /// relevant to that one corelet of that one core.
    ///
    /// ⛔ THE AUTHORITY'S `DT_ERROR("Cannot filter comps by clId and not by coreId")` (`:1950-1952`)
    /// IS UNSPELLABLE ACROSS THESE THREE READERS: a corelet filter here always arrives with the core
    /// it belongs to, and the reader that takes no core takes no corelet either.
    pub fn relevant_comps_of_corelet(
        &self,
        core: CoreId,
        corelet: CoreletId,
    ) -> BTreeSet<SenComponent> {
        self.relevant_comps
            .iter()
            .filter(|(_, core_cl)| core_cl.get(&core).is_some_and(|cls| cls.contains(&corelet)))
            .map(|(comp, _)| *comp)
            .collect()
    }

    /// `relevantComps_` itself (`dsc/dsc2.h:516`) — a named accessor rather than a public field,
    /// because the authority's is `protected` with five friend classes and one friend function
    /// (`:514-523`) and Rust has neither. Its whole-map readers are the ddc's assertion that the
    /// head's is non-empty (`ddc/ddcv1.cpp:3458`) and `SyncNode::getComponentsFromOtherEnds`, which
    /// merges the other end's map component by component (`dsc/dsc2.cpp:2411-2417`).
    pub fn relevant_comps(&self) -> &BTreeMap<SenComponent, BTreeMap<CoreId, BTreeSet<CoreletId>>> {
        &self.relevant_comps
    }

    /// The write side, which those same friends need: `setRelevantCompCoreCl` writes and reads the
    /// `NO_COMPONENT` entry and then fills the real components (`dsc/dsc2.cpp:2647-2729`),
    /// `finalizeScheduleTree` erases that entry (`:2977-2980`), the tree importer builds the map key
    /// by key (`:1373-1382`) and then unions every child's into the HEAD's (`:1389-1395`), and the
    /// work split copies a node's whole map onto its clone (`:5355`).
    ///
    /// ⛔ THE FIELD IS UNWRITEABLE WITHOUT IT, and a carried field with no writer is the defect this
    /// campaign already booked once: every one of those five writers is a method of a type whose own
    /// anchor is still open, so this accessor is what they will write through rather than something
    /// added for them later.
    pub fn relevant_comps_mut(
        &mut self,
    ) -> &mut BTreeMap<SenComponent, BTreeMap<CoreId, BTreeSet<CoreletId>>> {
        &mut self.relevant_comps
    }
}

/// `ScheduleNode::UnitView` (`dsc/dsc2.h:499-512`) — what one unit sees of one data structure at one
/// point in the schedule: the extents it addresses, and the enclosing loops that step through them.
/// `buildUnitView` produces every one of them, from a default-constructed value
/// (`dsc/dsc2.cpp:2759`), and a transfer's and a compute node's per-corelet views are vectors of
/// these (`dsc/dsc2.h:846-851`, `:943-947`).
///
/// ⛔ THE TWO LOOP VECTORS ARE A PARTITION OF THE SAME CLIMB, NOT TWO KINDS OF LOOP. `buildUnitView`
/// walks `getOwnerLoop()` upwards and pushes into `compositeLoops_` while the walk is still at or
/// below `lastFusableParentLoop`, into `outerLoops_` after it, flipping `compLoop` off at that node
/// (`dsc/dsc2.cpp:2842-2877`). So which vector a loop lands in is a property of the FUSION boundary,
/// and a view built with no such parent starts with `compLoop` already false (`:2843`).
///
/// ⛔ AND `print` IS NOT PORTED: it dereferences `it.loop_->name_` for every entry of both vectors
/// (`dsc/dsc2.cpp:4588`, `:4596`) — unconditionally, where the exporter beside it tests the pointer
/// first (`:209`) — so it needs the field this port does not carry. See the `loop_` anchor below.
#[derive(Clone, Debug, Default)]
pub struct UnitView {
    /// Field: e029_ScheduleNode.sizesNoGaps_
    /// Field: e013_ScheduleNode.sizesNoGaps_
    /// Field: e041_ScheduleNode.sizesNoGaps_
    ///
    /// The unit's extents, innermost first, with no per-core gap folded in (`dsc/dsc2.h:506`).
    /// `buildUnitView` fills the stick dims first, clamped to what is left of the stick's capacity,
    /// and then the layout dims in `layoutOrder` (`dsc/dsc2.cpp:2776-2818`) — which is why
    /// [`DimSize`] counts elements in the leading entries and sticks in the trailing ones, and why
    /// the same dim can appear twice.
    ///
    /// ⭐ POSITION IS THE CURRENCY THAT LEAVES THIS VECTOR: [`SizeIdx`] indexes it, bridge 1 walks it
    /// positionally against `srcSizeIdx_`/`dstSizeIdx_`
    /// (`dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:333-349`), and
    /// `calculateSizeIdxAndOffset` divides the offset down by the running product of the entries it
    /// has passed (`dsc/dsc2.cpp:2732-2746`). Innermost-first is what the gap writer relies on when
    /// it takes the HIGHEST index of a repeated dim as the outermost one (`:2905-2924`).
    pub sizes_no_gaps: Vec<Size>,
    /// Field: e029_ScheduleNode.compositeLoops_
    /// Field: e013_ScheduleNode.compositeLoops_
    /// Field: e041_ScheduleNode.compositeLoops_
    ///
    /// The enclosing loops up to and including the last fusable parent (`dsc/dsc2.h:507`). Unlike
    /// [`Self::outer_loops`] this one also carries loops that touch NONE of the data structure's
    /// dims, pushed with no extent and no offset at all (`dsc/dsc2.cpp:2869-2873`).
    pub composite_loops: Vec<LoopInfo>,
    /// Field: e029_ScheduleNode.outerLoops_
    /// Field: e013_ScheduleNode.outerLoops_
    /// Field: e041_ScheduleNode.outerLoops_
    ///
    /// The enclosing loops beyond the fusion boundary (`dsc/dsc2.h:508`). Only loops that address a
    /// dim of this data structure reach it — the `else if (compLoop)` arm that admits an unrelated
    /// loop pushes into `compositeLoops_` and never here (`dsc/dsc2.cpp:2869-2873`).
    pub outer_loops: Vec<LoopInfo>,
    /// Field: e029_ScheduleNode.sizesWithGaps_
    /// Field: e013_ScheduleNode.sizesWithGaps_
    ///
    /// The same extents per core, with that core's back gap added to the OUTERMOST entry of the
    /// gapped dim (`dsc/dsc2.h:509`, filled at `dsc/dsc2.cpp:2899-2926`).
    ///
    /// ⛔ [`None`] IS THE AUTHORITY'S `-1`, WHICH IS HBM. The keys come from an allocation's
    /// [`AllocateNode::back_gap_core`], whose own header comment says "HBM is -1"
    /// (`dsc/dsc2.h:989`), copied key for key by `try_emplace(core, sizesNoGaps_)`
    /// (`dsc/dsc2.cpp:2903`), and [`Self::sizes_for_core`] falls back to that entry for any core,
    /// which is the whole reason the pseudo-key exists. [`CoreId`] is unsigned, so it cannot spell
    /// one, and `None` sorting before every `Some` is where `-1` sits in the authority's
    /// `std::map`.
    ///
    /// ⚠️ THE PLAN DROPPED THIS FIELD BETWEEN WAVES: `e013_ScheduleNode` scheduled it, and
    /// `e029_ScheduleNode` — the same class, re-scheduled — lists `sizesNoGaps_`, `compositeLoops_`
    /// and `outerLoops_` but not this fourth member of the same struct (`dsc/dsc2.h:509`), which
    /// `getSizesForCoreId` reads before either of the others.
    pub sizes_with_gaps: BTreeMap<Option<CoreId>, Vec<Size>>,
}

impl UnitView {
    /// `getSizesForCoreId` (`dsc/dsc2.h:510`, defined `dsc/dsc2.cpp:2398-2405`): this core's gapped
    /// extents if it has any, else HBM's `-1` entry, else the gapless view. Bridge 1 reads it for
    /// the core it is lowering for, at nine sites
    /// (`dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:872`, `:1227`, `:1413`, `:1528`,
    /// `:2076`, `:2356`, `:2447`, and
    /// `dsc-based-utils/DSC2ToDataflowIR/V3/SNComputeLowering.cpp:575`, `:843`).
    ///
    /// ⭐ THE FALLBACK CHAIN IS WHY THE WILDCARD CANNOT BE ASKED FOR AS A CORE: taking a [`CoreId`]
    /// makes the authority's `getSizesForCoreId(-1)`, which would find the pseudo-key by its first
    /// lookup rather than its second, unspellable.
    pub fn sizes_for_core(&self, core: CoreId) -> &[Size] {
        self.sizes_with_gaps
            .get(&Some(core))
            .or_else(|| self.sizes_with_gaps.get(&None))
            .map_or(self.sizes_no_gaps.as_slice(), Vec::as_slice)
    }
}

/// `ScheduleNode::UnitView::LoopInfo` (`dsc/dsc2.h:500-505`) — one enclosing loop's view of one
/// dimension of the unit: which dim it steps, which extent of [`UnitView::sizes_no_gaps`] that dim
/// resolved to, and how many elements one trip moves.
///
/// ⛔ ONE LOOP CONTRIBUTES ONE ENTRY PER DIM IT CARRIES, not one entry per loop: `buildUnitView`
/// iterates `currLoop->dims_` and pushes inside that iteration (`dsc/dsc2.cpp:2850-2874`).
///
/// ⛔ AND [`Self::dim`] IS THE LOOP'S OWN DIM, NOT THE ONE THE EXTENT WAS FOUND UNDER. When the
/// allocation pads a window dim, `buildUnitView` resolves the extent and the offset against the
/// PADDED dim, `effectiveDim`, and then stores the unpadded `dim` beside them
/// (`dsc/dsc2.cpp:2852-2868`). So this field does not index `sizes_no_gaps` — [`Self::size_idx`]
/// does.
#[derive(Clone, Copy, Debug, Default)]
pub struct LoopInfo {
    /// Field: e029_ScheduleNode.dim_
    /// Field: e041_ScheduleNode.dim_
    ///
    /// Which dimension of the data structure this entry is about (`dsc/dsc2.h:502`). Its initialiser
    /// is `PrimaryDimTypesCount`, [`PrimaryDimTypes::Undefined`] here, which is a live value and not
    /// an absent one — the exporter spells it through `primaryDimToString` like any other dim
    /// (`dsc/dsc2.cpp:211-212`), where that enumerator maps to `"undefined"` (`dsc/dims.cpp:23`).
    pub dim: PrimaryDimTypes,
    /// Field: e029_ScheduleNode.sizeIdx_
    /// Field: e013_ScheduleNode.sizeIdx_
    /// Field: e041_ScheduleNode.sizeIdx_
    ///
    /// Which entry of [`UnitView::sizes_no_gaps`] this loop steps (`dsc/dsc2.h:503`), as
    /// `calculateSizeIdxAndOffset` resolved it (`dsc/dsc2.cpp:2732-2746`).
    ///
    /// ⛔ [`None`] IS THE AUTHORITY'S `-1` AND IT IS REACHABLE IN A STORED ENTRY, BY EXACTLY ONE
    /// ROUTE: a loop that touches none of the data structure's dims is pushed as
    /// `{currLoop, dim, -1, -1}` WITHOUT the call (`dsc/dsc2.cpp:2872`) — the only way, because
    /// `calculateSizeIdxAndOffset` ends in `DT_CHECK(sizeIdx >= 0)` (`:2746`) and so never returns
    /// one. The gap rescale then multiplies the offset of every entry whose `sizeIdx_ == i` for a
    /// gapped layout entry `i` (`:2880-2897`), a test the `-1` silently fails and a [`None`] cannot
    /// be mistaken for a position.
    pub size_idx: Option<SizeIdx>,
    /// Field: e029_ScheduleNode.elemOffset_
    /// Field: e013_ScheduleNode.elemOffset_
    /// Field: e041_ScheduleNode.elemOffset_
    ///
    /// How many elements of the extent at [`Self::size_idx`] one trip of this loop moves
    /// (`dsc/dsc2.h:504`) — see [`ElemOffset`] for the currency and its rescale.
    ///
    /// ⛔ [`None`] IS THE `-1` OF THE SAME UNRELATED-LOOP ENTRY (`dsc/dsc2.cpp:2872`), AND `0` IS A
    /// VALUE, NOT AN ABSENCE: the seed's own declaration says so — "put 1 if popping next element, or
    /// 0 if reuse is expected" (`dsc/dsc2.h:732-734`) — so absent and reuse are different states and
    /// only one of them is spelled by [`None`].
    pub elem_offset: Option<ElemOffset>,
}

// crustify:todo: e029_ScheduleNode

// crustify:todo: e029_ScheduleNode.loop_

// crustify:todo: e029_ScheduleNode.prev_

// crustify:todo: e013_ScheduleNode

// crustify:todo: e013_ScheduleNode.loop_

// crustify:todo: e013_ScheduleNode.prev_

// crustify:todo: e041_ScheduleNode

// crustify:todo: e041_ScheduleNode.loop_

// crustify:todo: e041_ScheduleNode.prev_

// crustify:todo: e041_ScheduleNode.BlockNode

// crustify:todo: e041_ScheduleNode.Ddc

// crustify:todo: e041_ScheduleNode.DesignSpaceConfig

// crustify:todo: e041_ScheduleNode.L3DlOpsScheduler

// crustify:todo: e041_ScheduleNode.ScheduleTree

// crustify:todo: e041_ScheduleNode.clId

// crustify:todo: e041_ScheduleNode.comp

/// Replaces: e022_DataStage
///
/// `dsc/dsc2.h:39-44`. One data stage's two halves — the steady-state dims and the epilogue dims of
/// the same data structure. `DesignSpaceConfig::dataStageParam_` keys them by id
/// (`dsc/designSpaceConfig.h:105`); id 0 is the core stage, whose name `getSizeDataStageForNode`
/// `DT_CHECK`s to be `"core"` (`dsc/dsc2.cpp:3638-3639`).
///
/// `e014_DataStage` is this same class under the superseded numbering; its two filled field anchors
/// are RENUMBERED onto e022 here, not deleted.
///
/// ⛔ [`name`](Self::name) IS THE STEADY STATE'S NAME ALONE, and WHETHER THE EPILOGUE CARRIES A
/// DIFFERENT ONE DEPENDS ON WHO MINTED THE STAGE. The `+ "el"` suffix belongs to the stages the DDC
/// mints by NUMBER — `constructDatastage` writes `to_string(id)` and `to_string(id) + "el"`
/// (`ddc/ddc_transformation_util.cpp:121-122`, `:131-132`, and the L3 scheduler's own copy at
/// `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:7730-7731`), the chunk split does the same from `ss_`
/// (`ddc/ddc_transformation.cpp:1018-1019`), and `calculateEpilogues` copies `ss_` and appends to
/// the copy's name (`ddc/ddcv1.cpp:1236-1237`). ⛔ BUT NOT EVEN EVERY BY-NUMBER STAGE HAS AN
/// EPILOGUE NAME: the DDL's own `DatastageOp` handler writes `ss_.name_ = to_string(id)` and never
/// assigns `el_.name_` at all (`ddc/ddl/ddl_conversion.cpp:1509-1515`), so there the epilogue name is
/// `""` and not `"<id>el"`. ⛔ ALL THREE **NAMED** STAGES CARRY THE SAME NAME IN
/// BOTH HALVES, and two of the three are written by the L3 scheduler this campaign ports: `"core"`
/// (`fillLoopLatchSdsc`, `dbo/src/Utils/sdsc_bundle/ProgramCorrection.cpp:1074-1075`), `"chunk"`
/// (both callers of `addOrUpdateDataStageParam` pass one `chunkDsName` for BOTH names,
/// `L3DlOpsScheduler.cpp:1419-1420`, `:1477-1480`) and `"superchunk"` (`ss_.name_ = el_.name_`,
/// `:2815`). So deriving the epilogue's name from the steady state's is wrong: the chunk stage that
/// `attachToPrefilledSchedule` requires to be named `"chunk"` (`ddc/ddcv1.cpp:2285`) has an epilogue
/// named `"chunk"`, not `"chunkel"`. Nothing relates the two — `addOrUpdateDataStageParam` takes
/// `ssName` and `elName` as independent parameters (`:721-726`).
///
/// ⛔ NO `PartialEq`: IBM declares none (`dsc/dsc2.h:40-44` has no `operator==` and no `tie()`), and
/// a derive would compare the two halves through `DataStructDims`' own equality, which deliberately
/// omits `name_` (`dsc/dims.h:221-228`) — so two stages with different names would compare equal.
#[derive(Clone, Debug, Default)]
pub struct DataStage {
    /// Field: e022_DataStage.ss_
    ///
    /// The steady state: the dims of every trip but the last. It is the half readers reach for by
    /// default (`ddc/ddc_fold.cpp:2113`, `ddc/ddcv1.cpp:1924`,
    /// `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:4848`).
    pub ss: DataStructDims,
    /// Field: e022_DataStage.el_
    ///
    /// The epilogue: the dims of the last, short trip. `calculateEpilogues` seeds it from `ss_` and
    /// then shrinks only the dims the metadata calls relevant (`ddc/ddcv1.cpp:1230-1330`), so an
    /// untouched epilogue equals the steady state rather than being empty.
    pub el: DataStructDims,
}

impl DataStage {
    /// `DataStage::name` (`dsc/dsc2.h:43`) — tested against `"core"` and `"chunk"` by
    /// `attachToPrefilledSchedule` (`ddc/ddcv1.cpp:2283-2285`) and built into a loop label
    /// (`dsc-based-utils/DSC2ToDataflowIR/V3/SNControlFlowLowering.cpp:953-954`).
    ///
    /// ⛔ THE DDL EMISSION'S TWO EMPTINESS TESTS **MINT** THE STAGE THEY TEST: they reach it through
    /// the non-const `dataStageParam_[numId_]` / `[denId_]` (`ddc/ddl/ddl_conversion.cpp:2974`,
    /// `:2998`), so a parametric loop's absent id inserts a default stage and this answers `""`. That
    /// insert is LOAD-BEARING: `.at(denId_)` / `.at(numId_)` later in the SAME iteration
    /// (`:3066-3067`) only succeeds because of it; it grows the `dataStageParam_.size()` the next
    /// minted id starts from (`:1510-1511`); and the stage's unfilled dims are what the emitted
    /// `ss_loop_count` divides, `ceil(-1.0 / -1) == 1` (`:3070-3078`) — a trip count this port
    /// reproduces, since an unfilled dim answers `-1` here too (`dsc/dims.cpp:567-568`).
    pub fn name(&self) -> &str {
        &self.ss.name
    }
}

// crustify:todo: e016_CoordPropInfoType

/// Replaces: CoordPropInfoType::PropStateType
///
/// `dsc/dsc2.h:1088`. How far one coordinate-propagation work item got. The queue pushes
/// `NOT_PROCESSED` (`ddc/ddc.h:424-427`, `:438-441`), `getCurrItem` marks the item it hands out
/// `COMPLETE` (`:468-469`) and `rollbackToPos` writes `ROLLED_BACK` (`:485-486`).
///
/// ⛔ THE WHOLE `propState` FIELD IS WRITE-ONLY IN THE AUTHORITY, not just the one dead state.
/// `propState` occurs exactly three times tree-wide — its declaration (`dsc/dsc2.h:1093`) and those
/// two writes — so NOTHING READS IT, and `CoordPropInfoType::print` does not print it either
/// (`dsc/dsc2.h:1098-1107`). A port must not grow a reader: no propagation decision is taken on this
/// value anywhere. `OVERRIDDEN` is deader still — `dsc/dsc2.h:1088` is its ONLY occurrence, with no
/// writer at all — and is ported because it holds the discriminant `COMPLETE` sits behind.
///
/// ⛔ `rollbackToPos` DOES NOT MARK THE RANGE IT ROLLS BACK, and this is an authority defect: the
/// loop runs `for (int i = newPos; i <= currItemToProcess_; ++i)` and clears each `[i]`'s
/// coordinates, but the `propState` write on the same iterations indexes
/// `itemsToProcess_.at(currItemToProcess_)` — `currItemToProcess_`, NOT `i` (`ddc/ddc.h:484-486`).
/// Only the last item is marked, once per iteration, and every other rolled-back item keeps
/// `COMPLETE` from `getCurrItem`. Inert today because nothing reads the field; do not reproduce the
/// aliasing when porting it, and do not "fix" it into a reader either.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PropStateType {
    /// The field's own initialiser (`dsc/dsc2.h:1093`) and what both push sites state.
    #[default]
    NotProcessed = 0,
    RolledBack = 1,
    Overridden = 2,
    Complete = 3,
}

/// `dsc/dsc2.h:1087-1108`. One work item of the coordinate-propagation queue: fold one schedule
/// node's coordinates from another's, over a named set of dims. `Ddc::CoordPropTracker` holds a
/// `std::deque` of them (`ddc/ddc.h:537`), `addPropInfo` pushes one per unseen dim set (`:402-428`)
/// and `getCurrItem` hands the next one out (`:463-474`).
///
/// ⛔ THIS CARRIES FIVE OF COORDPROPINFOTYPE'S SEVEN FIELDS, so its type anchor and two field anchors
/// stay open in all three generations. `refNode` and `nodeToFold` (`:1089-1090`) are `ScheduleNode*`
/// held as tree identity: they key `CoordPropTracker::refsAdded_` (`ddc/ddc.h:410-418`), and
/// `rollbackToPos` `static_cast`s `nodeToFold` to the concrete node and CLEARS its coordinates
/// through the pointer (`:487-503`). ⛔ AND THE TREE NOW ANSWERS WHAT THAT IDENTITY IS, WHICH IS WHY
/// THESE TWO STAY OPEN RATHER THAN BECOMING POINTERS: `BlockNode::next_` owns its children by value
/// and `ScheduleTree::head_` owns the root, so what names a node is its POSITION — [`InsertionPoint`]
/// for the three sites that search the list by address, and `std::ptr::eq` over live borrows for the
/// traversal's exclude set. The type's one method goes out with them: `print` (`:1098-1107`)
/// dereferences both for their `name_`.
///
/// ⛔ THE AUTHORITY'S DEFAULT CONSTRUCTION LEAVES THOSE TWO POINTERS INDETERMINATE — neither
/// declaration carries an initialiser — and both default-construction sites assign both on the next
/// two lines (`ddc/ddc_fold.cpp:1222-1224`, `:1924-1926`). [`Default`] here is the five initialisers
/// those sites do rely on, and nothing else.
///
/// ⛔ AND WHAT THOSE TWO SITES DO **NOT** FORWARD IS READ DOWNSTREAM. Each builds a reverse item out
/// of `dataConnect` and a flipped `refIsProducer` only, so [`scale_down`](Self::scale_down) and
/// [`dims_to_propagate`](Self::dims_to_propagate) come back DEFAULT on it: the reverse walk of a
/// scaled propagation is unscaled where `gatherRelatedPTRowsBase` divides the row-split beta by
/// `mxInfo_.blkSize` (`ddc/ddc_fold.cpp:907-911`).
///
/// ⛔ NO `PartialEq`: the authority declares none, and what identifies a work item is the two
/// pointers it names.
#[derive(Clone, Debug)]
pub struct CoordPropInfoType {
    /// Field: e038_CoordPropInfoType.dataConnect
    ///
    /// Field: e010_CoordPropInfoType.dataConnect
    ///
    /// Which operand of the two nodes the propagation runs over (`dsc/dsc2.h:1091`).
    ///
    /// ⛔ ITS TWO READERS DISAGREE ABOUT THE EMPTY NAME: `matchDataStream` reads it as a filter it
    /// skips, `dataConnect != "" && dataConnect == di.dataConnect_` (`ddc/ddc_fold.cpp:434-435`),
    /// while `getRelatedComputeCoord` REFUSES an empty one outright, twice — "data_connect is
    /// missing" on an opaque compute (`:1987-1990`) and "dataconnect is missing" on every other
    /// (`:2075-2078`). A [`DataConnect`] and not an [`Option`] of one for the same reason
    /// [`DataInfo::data_connect`] is: the empty string is the ordinary unset state.
    pub data_connect: DataConnect,
    /// Field: e038_CoordPropInfoType.refIsProducer
    ///
    /// Field: e010_CoordPropInfoType.refIsProducer
    ///
    /// Whether `refNode` PRODUCES the data connect and `nodeToFold` consumes it (`dsc/dsc2.h:1092`)
    /// — which end of a transfer or a compute the walk reads. Eight readers branch on it
    /// (`ddc/ddc_fold.cpp:777`, `:833`, `:985`, `:1005`, `:1072`, `:1124`, `:2833`, `:2933`), and
    /// both reverse items flip it (`:1226`, `:1928`).
    ///
    /// ⛔ `true` IS THE INITIALISER AND SO IS `addPropInfo`'S DEFAULT ARGUMENT (`ddc/ddc.h:406`), so a
    /// caller that names neither queues the producer walk.
    pub ref_is_producer: bool,
    /// Field: e038_CoordPropInfoType.propState
    ///
    /// Field: e010_CoordPropInfoType.propState
    ///
    /// How far this item got (`dsc/dsc2.h:1093`) — see [`PropStateType`], which no reader anywhere
    /// in the authority tree reads back.
    pub prop_state: PropStateType,
    /// Field: e038_CoordPropInfoType.dimsToPropagate
    ///
    /// Field: e010_CoordPropInfoType.dimsToPropagate
    ///
    /// The dims this item propagates (`dsc/dsc2.h:1094`). Every reader tests membership,
    /// `is_any_of(dim, coordPropInfo.dimsToPropagate)` (`ddc/ddc_fold.cpp:2514`, `:3137`, `:3348`),
    /// and `computeLoopElemOffsetsFromCoordinates` takes the whole list (`:1943`).
    ///
    /// ⛔ A `Vec` AND NOT A SET, BECAUSE ONLY ONE OF THE TWO PUSH SITES DEDUPS: `addPropInfo` records
    /// each dim in `refsAdded_` as it collects it, so a repeat in its own input is skipped
    /// (`ddc/ddc.h:409-423`), while `retry` pushes the caller's vector verbatim (`:438-441`).
    pub dims_to_propagate: Vec<PrimaryDimTypes>,
    /// Field: e038_CoordPropInfoType.scaleDown
    ///
    /// Field: e010_CoordPropInfoType.scaleDown
    ///
    /// Whether the reference coordinate is scaled down to the value tensor's block size before the
    /// fold is built (`dsc/dsc2.h:1095`). Its one writer sets it while re-targeting the item at the
    /// scale allocation (`ddc/ddc_fold.cpp:1744-1748`); `buildFoldForAllocation` then calls
    /// `scaleDownCoord` (`:2402-2407`) and `gatherRelatedPTRowsBase` divides the row-split beta by
    /// `mxInfo_.blkSize` (`:907-911`).
    ///
    /// ⛔ `retry` LOSES IT: it re-queues SIX of the seven initialisers and omits this one
    /// (`ddc/ddc.h:438-441`), so the aggregate's own `= false` wins, while `addPropInfo(rhs, dims)`
    /// beside it forwards `rhs.scaleDown` (`:430-434`). Both retries in `buildFoldForAllocation`
    /// (`ddc/ddc_fold.cpp:2437`, `:2460`) are on the path that reads it.
    pub scale_down: bool,
}

impl Default for CoordPropInfoType {
    /// The authority's five member initialisers (`dsc/dsc2.h:1091-1095`). ⛔ NOT
    /// `#[derive(Default)]`: `refIsProducer` starts `true`, and [`DataConnect`] derives no
    /// [`Default`] of its own.
    fn default() -> Self {
        Self {
            data_connect: DataConnect(String::new()),
            ref_is_producer: true,
            prop_state: PropStateType::NotProcessed,
            dims_to_propagate: Vec::new(),
            scale_down: false,
        }
    }
}

/// One entry of a [`BlockNode`]'s child list: the `std::unique_ptr<ScheduleNode>` the authority's
/// `VectorOfChildren` owns (`dsc/dsc2.h:529-537`), resolved to the CONCRETE node behind it.
///
/// ⭐ THE NAME IS `addChildNode`'S. The three operations that put a node into that list or take it
/// out all spell it — `addChildNode(ScheduleNode* nodeToAdd, ..)` (`dsc/dsc2.h:544-545`),
/// `moveChildNode` (`:546-549`) and `deleteChildNode` (`:552-553`) — and `deleteChildNode`'s own
/// refusal message calls the node "a child of node" `name_` (`dsc/dsc2.cpp:2197-2198`).
///
/// ⛔ THE VARIANTS ARE THE WHOLE HIERARCHY AND [`NodeType`] IS THEIR TAG, so the authority's pairing
/// of a `nodeType_` with a `dynamic_cast` is ONE fact here. The JSON importer is that pairing written
/// out once per kind (`dsc/dsc2.cpp:1337-1358`) and it is the exhaustive list; [`Self::node_type`]
/// READS the variant rather than carrying a second copy of the tag, and every [`Default`] in the
/// hierarchy passes its own kind to the base, so the two agree at every constructor. ⚠️ THE ONE WAY
/// TO BREAK THAT AGREEMENT IS THE WHOLE-BASE OVERWRITE `ScheduleNode`'s own field records
/// (`*node.base_mut() = ScheduleNode::new(..)`, see [`ScheduleNode::node_type`]) — C++ cannot spell it
/// because `nodeType_` is `const`, and nothing in Rust forbids it. ⛔ `INVALID` IS NOT A VARIANT: it
/// is `ScheduleNode`'s initialiser for a base nobody constructed (`dsc/dsc2.h:460`) and no minting
/// site produces it.
///
/// ⚠️ AND A `dynamic_cast<BlockNode*>` IS THREE VARIANTS, NOT ONE — `BLOCK`, `LOOP` and `CONDITION`,
/// which is exactly what `isBlockNode()` states (`dsc/dsc2.h:475-477`) and what [`Self::as_block`]
/// answers. The traversal leans on it: `if (auto* currBlock = dynamic_cast<const BlockNode*>(...))`
/// is what descends into a condition node's two regions (`dsc/dsc2.cpp:2243`).
///
/// ⛔ NO `PartialEq`, BECAUSE NODE IDENTITY IN THE AUTHORITY IS THE ADDRESS and the tree owns its
/// nodes here: `insertLoopAbove` finds `this` in its parent's list with `nodePtr.get() == this`
/// (`dsc/dsc2.cpp:2177`), `deleteChildNode` finds the child with `x.get() == nodeToDelete`
/// (`:2191-2193`), `addChildNode` finds the sibling with `insertionPoint->get() != siblingRefNode`
/// (`:2019`), and `traverseTreeDFS`'s exclude list is a set of pointers (`:2265`). What replaces the
/// address is the POSITION — [`InsertionPoint`] for the first three, and
/// [`traverse_dfs_excluding`](Self::traverse_dfs_excluding)'s `std::ptr::eq` over live borrows for
/// the fourth.
#[derive(Clone, Debug)]
pub enum ChildNode {
    Block(BlockNode),
    Loop(LoopNode),
    Transfer(TransferNode),
    Compute(ComputeNode),
    Sync(SyncNode),
    Condition(ConditionNode),
    Allocate(AllocateNode),
    StickMask(StickMaskNode),
}

impl ChildNode {
    /// The tag `ScheduleNode::nodeType_` holds (`dsc/dsc2.h:460`), read off the variant. It agrees
    /// with `self.base().node_type()` by construction, because every [`Default`] in this hierarchy
    /// passes its own kind to the base exactly as the authority's constructors do
    /// (`dsc/dsc2.h:554`, `:583`, `:686`, `:815`, `:901`, `:965`, `:975`, `:1060`).
    pub fn node_type(&self) -> NodeType {
        match self {
            Self::Block(_) => NodeType::Block,
            Self::Loop(_) => NodeType::Loop,
            Self::Transfer(_) => NodeType::Transfer,
            Self::Compute(_) => NodeType::Compute,
            Self::Sync(_) => NodeType::Sync,
            Self::Condition(_) => NodeType::Condition,
            Self::Allocate(_) => NodeType::Allocate,
            Self::StickMask(_) => NodeType::StickMask,
        }
    }

    /// The `ScheduleNode` subobject every node in the hierarchy has, reached through the two
    /// `BlockNode`-derived kinds' own base (`dsc/dsc2.h:563`, `:685`). It is the implicit conversion
    /// to a public base, which is how every caller of `name_`, `isNodeRelevant` or
    /// `getRelevantCoreCl` reaches those from a concrete node.
    pub fn base(&self) -> &ScheduleNode {
        match self {
            Self::Block(node) => &node.base_class,
            Self::Loop(node) => &node.base_class.base_class,
            Self::Condition(node) => &node.base().base_class,
            Self::Transfer(node) => &node.base_class,
            Self::Compute(node) => &node.base_class,
            Self::Sync(node) => &node.base_class,
            Self::Allocate(node) => &node.base_class,
            Self::StickMask(node) => &node.base_class,
        }
    }

    /// The mutable base, which is how a name is written: `finalizeScheduleTree` renames colliding
    /// nodes as it walks (`dsc/dsc2.cpp:2988-2992`) and `setRelevantCompCoreCl` fills
    /// `relevantComps_` node by node (`:2647-2729`).
    pub fn base_mut(&mut self) -> &mut ScheduleNode {
        match self {
            Self::Block(node) => &mut node.base_class,
            Self::Loop(node) => &mut node.base_class.base_class,
            Self::Condition(node) => &mut node.base_mut().base_class,
            Self::Transfer(node) => &mut node.base_class,
            Self::Compute(node) => &mut node.base_class,
            Self::Sync(node) => &mut node.base_class,
            Self::Allocate(node) => &mut node.base_class,
            Self::StickMask(node) => &mut node.base_class,
        }
    }

    /// `dynamic_cast<const BlockNode*>` (`dsc/dsc2.cpp:2243`), which succeeds for all three
    /// `isBlockNode()` kinds (`dsc/dsc2.h:475-477`) — a loop and a condition node ARE block nodes
    /// and own children through the same list.
    pub fn as_block(&self) -> Option<&BlockNode> {
        match self {
            Self::Block(node) => Some(node),
            Self::Loop(node) => Some(&node.base_class),
            Self::Condition(node) => Some(node.base()),
            _ => None,
        }
    }

    /// The mutable `static_cast<BlockNode*>` (`ddc/ddl/ddl_conversion.cpp:2934`,
    /// `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:4204`).
    ///
    /// ⛔ AND IT REFUSES A CONDITION NODE, WHERE THE AUTHORITY'S CAST DOES NOT. `ConditionNode`
    /// OVERRIDES `addChildNode` to refuse anything but a `BLOCK` and any third child
    /// (`dsc/dsc2.h:697`, body `dsc/dsc2.cpp:2143-2150`), and that override is NOT reached through a
    /// `BlockNode*` — the base method is `virtual` (`dsc/dsc2.h:544`), so it IS reached there, but
    /// `BlockNode::addChildNode` is then reachable directly as `node->BlockNode::addChildNode(..)`
    /// and every non-virtual sibling of it (`moveChildren`, `insertPerfectlyNestedBlockNode`'s own
    /// `addChildNode` call) can still put a third non-block child into a condition node. Handing out
    /// `&mut BlockNode` for a condition node would reopen that; [`ConditionNode`]'s own
    /// [`add_child_node`](ConditionNode::add_child_node) is the only mutable route to its children,
    /// and it takes a [`BlockNode`] BY VALUE, which is the "only 2 BlockNodes" rule as a type.
    pub fn as_block_mut(&mut self) -> Option<&mut BlockNode> {
        match self {
            Self::Block(node) => Some(node),
            Self::Loop(node) => Some(&mut node.base_class),
            _ => None,
        }
    }

    /// `static_cast<dsc2::LoopNode*>` after a `nodeType_ == LOOP` test (`ddc/ddcv1.cpp:596-597`).
    pub fn as_loop(&self) -> Option<&LoopNode> {
        match self {
            Self::Loop(node) => Some(node),
            _ => None,
        }
    }

    /// The mutable form (`ddc/ddcv1.cpp:596-597`, reached from `traverseTreeDFSMutable`).
    pub fn as_loop_mut(&mut self) -> Option<&mut LoopNode> {
        match self {
            Self::Loop(node) => Some(node),
            _ => None,
        }
    }

    /// `static_cast<dsc2::ConditionNode *>` after a `CONDITION` test
    /// (`ddc/ddc_transformation_util.cpp:262-263`).
    pub fn as_condition(&self) -> Option<&ConditionNode> {
        match self {
            Self::Condition(node) => Some(node),
            _ => None,
        }
    }

    /// The mutable form, which is how `addThenRegion`/`addElseRegion` are reached from a traversal
    /// (`ddc/ddc_transformation_util.cpp:262-263`, `:587-588`).
    pub fn as_condition_mut(&mut self) -> Option<&mut ConditionNode> {
        match self {
            Self::Condition(node) => Some(node),
            _ => None,
        }
    }

    /// `dynamic_cast<const dsc2::TransferNode *>` (`DSC2ToDataflowIR.cpp:265`) and its
    /// `static_cast` twin (`ddc/ddcv1.cpp:441-442`).
    pub fn as_transfer(&self) -> Option<&TransferNode> {
        match self {
            Self::Transfer(node) => Some(node),
            _ => None,
        }
    }

    /// The mutable form (`ddc/ddcv1.cpp:441-442`).
    pub fn as_transfer_mut(&mut self) -> Option<&mut TransferNode> {
        match self {
            Self::Transfer(node) => Some(node),
            _ => None,
        }
    }

    /// `static_cast<dsc2::ComputeNode *>` (`ddc/ddc_fold.cpp:1628-1629`).
    pub fn as_compute(&self) -> Option<&ComputeNode> {
        match self {
            Self::Compute(node) => Some(node),
            _ => None,
        }
    }

    /// The mutable form (`ddc/ddc_fold.cpp:1628-1629`).
    pub fn as_compute_mut(&mut self) -> Option<&mut ComputeNode> {
        match self {
            Self::Compute(node) => Some(node),
            _ => None,
        }
    }

    /// `static_cast<dsc2::SyncNode *>` (`ddc/ddc_transformation.cpp:1528-1529`).
    pub fn as_sync(&self) -> Option<&SyncNode> {
        match self {
            Self::Sync(node) => Some(node),
            _ => None,
        }
    }

    /// The mutable form (`ddc/ddc_transformation.cpp:1528-1529`).
    pub fn as_sync_mut(&mut self) -> Option<&mut SyncNode> {
        match self {
            Self::Sync(node) => Some(node),
            _ => None,
        }
    }

    /// `static_cast<dsc2::AllocateNode*>` (`ddc/ddcv1.cpp:48-49`).
    pub fn as_allocate(&self) -> Option<&AllocateNode> {
        match self {
            Self::Allocate(node) => Some(node),
            _ => None,
        }
    }

    /// The mutable form (`ddc/ddcv1.cpp:48-49`).
    pub fn as_allocate_mut(&mut self) -> Option<&mut AllocateNode> {
        match self {
            Self::Allocate(node) => Some(node),
            _ => None,
        }
    }

    /// `static_cast<dsc2::StickMaskNode*>` (`ddc/ddcv1.cpp:3662-3663`).
    pub fn as_stick_mask(&self) -> Option<&StickMaskNode> {
        match self {
            Self::StickMask(node) => Some(node),
            _ => None,
        }
    }

    /// The mutable form (`ddc/ddcv1.cpp:3662-3663`).
    pub fn as_stick_mask_mut(&mut self) -> Option<&mut StickMaskNode> {
        match self {
            Self::StickMask(node) => Some(node),
            _ => None,
        }
    }

    /// `traverseTreeDFS(startNode, nodeTypes, comp)` over the subtree rooted at THIS node, the node
    /// itself included (`dsc/dsc2.cpp:2222-2265`): the authority seeds `nodesToVisit` with
    /// `startNode` and tests it like any other (`:2228`, `:2237`).
    ///
    /// ⭐ IT IS A PURE SUBTREE WALK, WHICH IS WHY IT IS A METHOD ON THE NODE AND NOT ON THE TREE:
    /// with a `startNode` whose `prev_` is set, nothing in the body reaches `head_` again. The
    /// `startNode == nullptr || startNode->prev_ == nullptr` arm (`:2224`) is the whole-tree walk,
    /// [`ScheduleTree::traverse_dfs`], and the second half of that test — an ORPHAN node silently
    /// walking the whole tree instead of itself — is unspellable here.
    ///
    /// ⛔ A NODE THAT IS NOT RELEVANT PRUNES ITS WHOLE SUBTREE, not just itself (`:2237-2240`
    /// `continue`s before the children are pushed), and `nodeTypes` filters only what is COLLECTED —
    /// a non-matching block node is still descended into (`:2241-2253`).
    ///
    /// ⛔ AND `maxLoopDepth` IS NOT PORTED BECAUSE NOTHING PASSES ONE. Tree-wide, every one of the
    /// ~60 `traverseTreeDFS`/`traverseTreeDFSMutable` calls either stops before that parameter or
    /// passes the `-1` default, and it is the ONLY reader of the `loopDepth` stack the body carries
    /// (`:2224`, `:2231-2235`, `:2244-2250`). ⭐ THAT STACK IS ALSO THE TRAVERSAL'S ONLY USE OF
    /// `prev_` — `while (currNode->prev_ != loopDepth.back().first) loopDepth.pop_back()` (`:2235`)
    /// — so dropping the dead parameter is what makes a `prev_`-free tree walk EXACT rather than
    /// approximate. ⛔ AND THAT SAME LINE IS UNSOUND ON A STALE `prev_`: `insertLoopAbove` splices a
    /// loop in without updating either node's (`:2169-2186`), and `pop_back()` on an emptied vector
    /// is undefined behaviour, so a walk through a spliced node has no defined result at all. It has
    /// no callers either (`insertLoopAbove` is unreferenced tree-wide).
    ///
    /// ⛔ AND THE CORE/CORELET FILTER IS NOT PORTED FOR THE SAME REASON: every in-scope caller
    /// passes `-1, -1` (`DSC2ToDataflowIR.cpp:262-263`, `ddc/ddcv1.cpp:3416-3417`,
    /// `ddc/ddc_transformation.cpp:1525-1527`, `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:1596`).
    /// The only callers that pass a core or a corelet are in `dsc/dsc2Pcfg.cpp` (`:144`, `:2253`,
    /// `:2257-2258`), and DCG/PCFG is off this campaign's path. The COMPONENT filter stays because
    /// two in-scope callers do use it: bridge 1 passes the component it is lowering
    /// (`DSC2ToDataflowIR.cpp:262-263`) and the DDC passes a transfer's destination component
    /// (`ddc/ddcv1.cpp:3416-3417`).
    pub fn traverse_dfs(&self, node_types: &[NodeType], comp: SenComponent) -> Vec<&ChildNode> {
        let mut order = Vec::new();
        self.visit_dfs(node_types, comp, &[], &mut order);
        order
    }

    /// `traverseTreeDFS`'s `excludeList` (`dsc/dsc2.cpp:2265`), whose one live caller scans a loop
    /// for `SYNC` nodes while skipping the inner loop it scanned on the previous iteration
    /// (`ddc/ddc_transformation.cpp:1525-1527`, filled at `:1656-1657`).
    ///
    /// ⛔ EXCLUDING A NODE PRUNES ITS SUBTREE, which is the whole point at that caller: the
    /// `continue` at `:2239` happens before the children are pushed, so an excluded loop hides the
    /// syncs beneath it. ⭐ AND THE IDENTITY IS THE ADDRESS, as it is in the authority's
    /// `excludeList.count(currNode)` over a set of pointers: `std::ptr::eq` on live borrows says the
    /// same thing, and the borrow is what keeps the excluded node alive for the walk.
    pub fn traverse_dfs_excluding(
        &self,
        node_types: &[NodeType],
        comp: SenComponent,
        exclude: &[&ChildNode],
    ) -> Vec<&ChildNode> {
        let mut order = Vec::new();
        self.visit_dfs(node_types, comp, exclude, &mut order);
        order
    }

    /// The body of `traverseTreeDFS` (`dsc/dsc2.cpp:2236-2254`) with its work list turned into
    /// recursion: the authority pops the FRONT of a deque and pushes a block's children back onto
    /// the front in reverse (`:2251-2253`), which is pre-order depth-first over the children in
    /// order — what this produces.
    fn visit_dfs<'tree>(
        &'tree self,
        node_types: &[NodeType],
        comp: SenComponent,
        exclude: &[&ChildNode],
        order: &mut Vec<&'tree ChildNode>,
    ) {
        if !self.base().is_relevant(comp) || exclude.iter().any(|node| std::ptr::eq(*node, self)) {
            return;
        }
        if node_types.is_empty() || node_types.contains(&self.node_type()) {
            order.push(self);
        }
        for child in self.as_block().map_or(&[][..], BlockNode::children) {
            child.visit_dfs(node_types, comp, exclude, order);
        }
    }

    /// `traverseTreeDFSMutable(startNode, nodeTypes, comp)` over this subtree
    /// (`dsc/dsc2.cpp:2208-2220`), whose live caller takes the start node as a PARAMETER and rewrites
    /// the lds index of every allocate, transfer and compute beneath it
    /// (`ddc/ddc_transformation_util.cpp:1920-1930`).
    ///
    /// ⛔ A VISITOR RATHER THAN A LIST OF `&mut`, for the reason [`ScheduleTree::for_each_dfs_mut`]
    /// records: the authority's list aliases the tree, and an ancestor and its descendant appear in it
    /// together.
    pub fn for_each_dfs_mut(
        &mut self,
        node_types: &[NodeType],
        comp: SenComponent,
        mut f: impl FnMut(&mut ChildNode),
    ) {
        self.visit_dfs_mut(node_types, comp, &mut f);
    }

    /// The mutable walk itself. `&mut dyn FnMut` rather than a second type parameter because the
    /// recursion would otherwise instantiate itself infinitely through `&mut F`.
    fn visit_dfs_mut(
        &mut self,
        node_types: &[NodeType],
        comp: SenComponent,
        f: &mut dyn FnMut(&mut ChildNode),
    ) {
        if !self.base().is_relevant(comp) {
            return;
        }
        if node_types.is_empty() || node_types.contains(&self.node_type()) {
            f(self);
        }
        let Some(children) = self.children_mut() else {
            return;
        };
        for child in children.iter_mut() {
            child.visit_dfs_mut(node_types, comp, f);
        }
    }

    /// The child list of a block node by value, or [`None`] for a leaf — what
    /// [`Self::visit_dfs_mut`] descends through, the mutable twin of
    /// `dynamic_cast<const BlockNode*>` at `dsc/dsc2.cpp:2243`.
    ///
    /// ⛔ MODULE-PRIVATE, AND THAT IS WHAT MAKES [`ConditionNode`]'S TWO-CHILD RULE HOLD: a public
    /// `&mut Vec<ChildNode>` would let any caller `push` a third child, or a non-`BLOCK` one, into a
    /// condition node — exactly what `ConditionNode::addChildNode` `DT_ERROR`s on
    /// (`dsc/dsc2.cpp:2143-2150`). [`ChildNode::as_block_mut`] refuses a condition node for the same
    /// reason.
    fn children_mut(&mut self) -> Option<&mut Vec<ChildNode>> {
        match self {
            Self::Block(node) => Some(&mut node.next),
            Self::Loop(node) => Some(&mut node.base_class.next),
            Self::Condition(node) => Some(node.children_mut()),
            _ => None,
        }
    }
}

/// Where in a [`BlockNode`]'s child list `addChildNode` puts the node — the authority's own
/// `insertionPoint` local, which is an ITERATOR into `next_` (`dsc/dsc2.cpp:2015-2025`).
///
/// ⭐ FOUR SHAPES, ALL FOUR WITH LIVE CALLERS, and they are what the `(addBefore, siblingRefNode)`
/// pair spells rather than two independent flags:
///  * [`Self::Back`] is `(false, nullptr)`, the default and the common case
///    (`ddc/ddc_transformation.cpp:986`, `:1195-1197`, `ddc/ddcv1.cpp:3656`);
///  * [`Self::Front`] is `(true, nullptr)` — `next_.begin()` is where `insertionPoint` starts and
///    `addBefore` is what stops it being reassigned to `end()` (`dsc/dsc2.cpp:2016-2017`), used to
///    put a `"root_level_operations"` block at the head of the tree
///    (`ddc/ddl/ddl_conversion.cpp:2779-2782`);
///  * [`Self::Before`] is `(true, sibling)` (`ddc/ddc_transformation_util.cpp:861`, `:1187`,
///    `ddc/ddcv1.cpp:2158`, `ddc/ddc_transformation.cpp:1011`);
///  * [`Self::After`] is `(false, sibling)`, the `insertionPoint++` at `dsc/dsc2.cpp:2026`
///    (`ddc/ddc_transformation_util.cpp:1036`, `:1366`, `ddc/ddc_transformation.cpp:1012`).
///
/// ⛔ THE SIBLING IS A POSITION HERE AND AN ADDRESS THERE, AND THAT IS WHAT CLOSES A REAL HOLE.
/// The authority walks `for (; insertionPoint->get() != siblingRefNode; ++insertionPoint)` and tests
/// `insertionPoint == next_.end()` INSIDE the body (`dsc/dsc2.cpp:2019-2023`), so the dereference
/// happens BEFORE the end test: a sibling that is not in this list runs off the vector instead of
/// reaching the `DT_ERROR` one line below. Compiled and run under ASAN over the extracted body, that
/// is a heap-buffer-overflow read followed by a member call on the null it loads — never the
/// diagnostic. An index cannot express it, and an out-of-range one is refused by
/// [`BlockNode::add_child_node`] handing the node back.
///
/// ⚠️ AND AN INDEX IS ONLY VALID UNTIL THE LIST CHANGES, where the authority's pointer survives a
/// sibling insertion. Every live caller resolves the sibling and inserts in the same statement, so
/// none of them holds one across a mutation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InsertionPoint {
    Front,
    Back,
    Before(usize),
    After(usize),
}

/// Replaces: e004_BlockNode
/// Replaces: e030_BlockNode
///
/// `dsc2::BlockNode` (`dsc/dsc2.h:526-561`) — the node that OWNS other nodes. Every interior node of
/// a schedule tree is one: `LoopNode` (`:563`) and `ConditionNode` (`:685`) derive from it, the tree's
/// root IS one (`ScheduleTree::head_` is a `LoopNode`, `:623`), and the DDL conversion mints plain
/// ones to group operations (`ddc/ddl/ddl_conversion.cpp:2779`, `:1738-1745`).
///
/// ⭐ THIS IS WHERE THE SCHEDULE TREE BECOMES A TREE, and it is the shape the deleted port never had:
/// the five leaf node types carried none of the base's fields and nothing owned a child. With this
/// type there is exactly one owner of a node — the `Vec<ChildNode>` below — and the parent link,
/// the visit order and the lifetime of every node follow from it.
///
/// ⛔ `getNextView` IS NOT VIRTUAL, AND `ConditionNode` DECLARES ITS OWN (`dsc/dsc2.h:541-543`,
/// `:701-703`), so which body runs depends on the STATIC type of the handle: through a `BlockNode*`
/// a condition node gets the FILTERED reading, and only a `ConditionNode*` gets the one that widens a
/// loop-guarded node's view to `ALL, -1, -1` (`dsc/dsc2.cpp:1995-2002`). Rust method resolution is
/// the same rule, so [`Self::next_view`] on a condition node's base and
/// [`ConditionNode::next_view`] on the node itself preserve the split exactly — including at the
/// ~25 sites that hold a `BlockNode*`.
///
/// ⛔ TWO OF THIS CLASS'S SIX METHODS ARE BLOCKED ON e027_DesignSpaceConfig, which is why the
/// `e004_BlockNode`/`e030_BlockNode` TODO anchors at the end of this file stay open.
/// `deleteChildNode` calls `ownerDsc->cleanupAllocation(nodeToDelete)` (`dsc/dsc2.cpp:2188-2190`)
/// and `moveChildNode` forwards to it (`:2031-2039`); `DesignSpaceConfig::cleanupAllocation`
/// (`dsc/designSpaceConfig.h:262`) is unported. ⭐ THE OTHER HALF OF `deleteChildNode` IS HERE:
/// `nonDestructive` is `childIt->release()` before the erase (`:2202-2206`), i.e. "take the node out
/// and hand it to the caller", which is [`Self::take_child_node`] and which is the arm
/// `moveChildNode` uses (`:2038`).
///
/// ⚠️ AND THE AUTHORITY'S DESTRUCTIVE ARM CLEANS UP BEFORE IT CHECKS: `cleanupAllocation` runs on
/// `nodeToDelete` at `:2189`, and only then does the `find_if` at `:2191-2196` discover the node is
/// not a child of this one and `DT_ERROR` (`:2197-2199`). A failed delete has therefore already
/// unregistered the allocation.
///
/// ⛔ NO `prev_`, AND NOTHING THIS UNIT PORTS READS ONE. The four operations that do —
/// `getOwnerLoop`, `getParentDimLoop`, `moveNode` and `insertLoopAbove` (`dsc/dsc2.cpp:1892-1933`,
/// `:1977-1982`, `:2169-2186`) — are `ScheduleNode`'s, and they stay with e029_ScheduleNode's open
/// anchor. In an owning tree the parent is the walker's, not the node's, which is what makes
/// `insertLoopAbove`'s measured defect unrepresentable here: it splices a loop between a node and its
/// parent and updates NEITHER `prev_`, so both the node's and the new loop's are stale the moment it
/// returns.
///
/// ⛔ NO `PartialEq`: see [`ChildNode`]. `Default` is `BlockNode() : BaseClass(BLOCK)`
/// (`dsc/dsc2.h:554`) and `Clone` is IBM's, which DROPS THE CHILDREN — see [`Self::clone`].
#[derive(Debug)]
pub struct BlockNode {
    /// Field: e004_BlockNode.BaseClass
    /// Field: e030_BlockNode.BaseClass
    ///
    /// The `ScheduleNode` subobject (`dsc/dsc2.h:526`, `InheritWithClone<ScheduleNode, BlockNode>`).
    /// The anchor names `using BaseClass::BaseClass;` (`:528`), the PROTECTED inherited constructor
    /// that is how a derived class passes its own kind down — `LoopNode() : BaseClass(LOOP)`
    /// (`:583`), `ConditionNode() : BaseClass(CONDITION)` (`:686`) — while `BlockNode()` itself
    /// passes `BLOCK` (`:554`). [`Self::new`] is that protected constructor and is module-private for
    /// the same reason: outside this module [`Default`] is the only way to make one, so no `BlockNode`
    /// is CONSTRUCTED with a kind other than its own.
    ///
    /// PUBLIC, because the authority's inheritance is public and `name_` with it: the DDL conversion
    /// writes `initialInsertionBlock->name_ = "root_level_operations"` straight through a
    /// `BlockNode*` (`ddc/ddl/ddl_conversion.cpp:2782`). ⚠️ `BaseClass` IS ALSO A TYPEDEF, `typedef
    /// Base BaseClass` in `InheritWithClone` (`util/utils.h:100`), and the scheduler's field scan
    /// anchored the `using` line rather than a field; what the field holds is the base's four
    /// members, three of which e029_ScheduleNode carries.
    pub base_class: ScheduleNode,
    /// Field: e004_BlockNode.next_
    /// Field: e030_BlockNode.next_
    ///
    /// The children, in order (`dsc/dsc2.h:538`). The authority's `VectorOfChildren` is a
    /// `std::vector<std::unique_ptr<ScheduleNode>>` (`:529`), so this list OWNS its nodes and is the
    /// only owner — which is what `unique_ptr` states and what `deleteChildNode`'s
    /// `childIt->release()` has to defeat to hand a node out alive (`dsc/dsc2.cpp:2202-2206`).
    ///
    /// PRIVATE, because the authority's is `protected` behind five friend classes (`:556-560`) while
    /// the readings are public methods: [`Self::children`] hands out the list, [`Self::next_view`]
    /// the relevance-filtered one, and every mutation goes through an operation that keeps the
    /// invariants — which is what makes [`ConditionNode`]'s "at most two BLOCK children" true rather
    /// than documented.
    ///
    /// ⛔ ORDER IS THE SCHEDULE, so this is a `Vec` and never a set: bridge 1 emits the children of a
    /// block in list order and that is program order (`SNControlFlowLowering.cpp:577-600`), and
    /// `getThenBranchNode`/`getElseBranchNode` are positions 0 and 1 (`dsc/dsc2.h:707-718`).
    next: Vec<ChildNode>,
}

impl Default for BlockNode {
    /// `BlockNode() : BaseClass(BLOCK) {}` (`dsc/dsc2.h:554`) over `VectorOfChildren() = default`
    /// (`:530`).
    fn default() -> Self {
        Self::new(NodeType::Block)
    }
}

impl Clone for BlockNode {
    /// IBM's `clone()` is `new BlockNode(static_cast<BlockNode const&>(*this))`, the implicit COPY
    /// CONSTRUCTOR reached through `InheritWithClone` (`util/utils.h:105-107`, `dsc/dsc2.h:526`).
    ///
    /// ⛔ SO A CLONED BLOCK NODE HAS NO CHILDREN, and that is load-bearing rather than a leak to
    /// tidy up: `VectorOfChildren(const VectorOfChildren&) {}` is "do nothing on purpose", with the
    /// authority's own comment "when copying, it is up to the caller to manually insert copies of
    /// the children" (`dsc/dsc2.h:533-536`). Both `BlockNode`-derived clone sites depend on it —
    /// `ddc/ddc_transformation.cpp:984-986` clones a loop and then `addChildNode`s one cloned child,
    /// and `ddc/ddc_transformation_util.cpp:580-588` clones a `ConditionNode` and then
    /// `addThenRegion`s a fresh block, which `DT_ERROR`s outright if `next_` is non-empty
    /// (`dsc/dsc2.cpp:2152-2155`). A deep `Clone` would turn a working DDC path into a fatal error.
    ///
    /// ⛔ AND C++ CANNOT ASSIGN ONE: `VectorOfChildren`'s copy assignment is `= delete` (`:536`),
    /// which implicitly deletes `BlockNode`'s, so `*a = b` does not compile there and always
    /// compiles here — the same asymmetry [`TransferNode::clone`] records for `paddingInfo_`.
    /// [`ScheduleTree`] is where it matters, and there it is closed: the tree has no [`Clone`] at
    /// all.
    fn clone(&self) -> Self {
        Self {
            base_class: self.base_class.clone(),
            next: Vec::new(),
        }
    }
}

impl BlockNode {
    /// `using BaseClass::BaseClass;` (`dsc/dsc2.h:528`), the PROTECTED inherited constructor. Module
    /// -private, so the only callers are the two derived kinds' [`Default`] impls and this one's —
    /// which is what `protected` buys the authority.
    fn new(node_type: NodeType) -> Self {
        Self {
            base_class: ScheduleNode::new(node_type),
            next: Vec::new(),
        }
    }

    /// The child list in order (`dsc/dsc2.h:538`). The authority reaches it directly through
    /// friendship — `for (const auto& child : scheduleTree_.getHeadMutable()->next_)`
    /// (`dsc/dsc2.cpp:1389`), and `setRelevantCompCoreCl` walks it at `:2658-2686`.
    ///
    /// ⭐ NAMED FOR `moveChildren` (`dsc/dsc2.h:551`), whose own comment calls these "children of the
    /// current node" (`dsc/dsc2.cpp:2042-2043`).
    pub fn children(&self) -> &[ChildNode] {
        &self.next
    }

    /// `getNextView(comp)` (`dsc/dsc2.cpp:1984-1993`): the children this component can see. Bridge 1
    /// takes the roots of a uniformized program unit this way (`DSC2ToDataflowIR.cpp:380`).
    ///
    /// ⛔ [`SenComponent::All`] RETURNS EVERY CHILD rather than looking anything up, which is
    /// `isNodeRelevant`'s first arm (`dsc/dsc2.cpp:1919-1923`) and is why `ALL` is never a key in
    /// `relevantComps_`. ⛔ AND A CHILD WITH AN EMPTY `relevantComps_` IS IN NOBODY'S VIEW: before
    /// `setRelevantCompCoreCl` has run (`:2647-2729`) every view except `ALL`'s is empty.
    pub fn next_view(&self, comp: SenComponent) -> Vec<&ChildNode> {
        self.next
            .iter()
            .filter(|child| child.base().is_relevant(comp))
            .collect()
    }

    /// `getNextView(comp, clId, coreId)` with a real core (`dsc/dsc2.cpp:1984-1993`), the reading
    /// bridge 1 uses to find the roots of ONE program unit — a core, a corelet and a component
    /// (`DSC2ToDataflowIR.cpp:298-300`, `SNControlFlowLowering.cpp:577`, `:1213`).
    ///
    /// ⛔ A SEPARATE METHOD BECAUSE THE AUTHORITY REFUSES THE MIXED CALL: `isNodeRelevant` `DT_ERROR`s
    /// on `ALL` together with a core or corelet filter, and on a corelet with no core
    /// (`dsc/dsc2.cpp:1919-1928`). The second of those two is unspellable here — a corelet without a
    /// core has nowhere to go in this signature — and bridge 1's own call sites are the split: the
    /// one that has a unit passes all three (`:298-300`), the one that does not calls
    /// [`Self::next_view`] (`:380`).
    ///
    /// ⚠️ THE FIRST REFUSAL IS A NARROWING HERE, NOT AN ABORT: `comp == All` cannot be a key in
    /// `relevantComps_`, so this answers with an EMPTY view where the authority ends the process. No
    /// live caller reaches it — a `ProgramUnitOp` is built per component (`DSC2ToDataflowIR.cpp:291`)
    /// and `comp` is that component.
    pub fn next_view_of_corelet(
        &self,
        comp: SenComponent,
        core: CoreId,
        corelet: Option<CoreletId>,
    ) -> Vec<&ChildNode> {
        self.next
            .iter()
            .filter(|child| child.base().is_relevant_to_corelet(comp, core, corelet))
            .collect()
    }

    /// `addChildNode(nodeToAdd, addBefore, siblingRefNode)` (`dsc/dsc2.cpp:2013-2029`), the one way
    /// a node enters a tree. 60-odd call sites; the four [`InsertionPoint`] shapes are all of them.
    ///
    /// ⭐ THE REFUSAL HANDS THE NODE BACK. An [`InsertionPoint::Before`] or
    /// [`InsertionPoint::After`] past the end of the list is the authority's "Sibling reference node
    /// not found in parent node" (`:2021`) — except that the authority never reaches that
    /// `DT_ERROR`, because the loop above it dereferences before testing for the end (see
    /// [`InsertionPoint`]). Returning the node means a refused insertion loses nothing and nothing
    /// needs to unwind; [`None`] is the success.
    ///
    /// ⛔ IT DOES NOT CHECK THE KIND, AND A CONDITION NODE IS WHERE THAT MATTERS: this body accepts
    /// any node, and `ConditionNode::addChildNode` is the `override` that narrows it to two BLOCKs
    /// (`dsc/dsc2.h:697`, `dsc/dsc2.cpp:2143-2150`). [`ChildNode::as_block_mut`] refusing a
    /// condition node is what stops this method being reached on one.
    ///
    /// ⛔ AND `nodeToAdd->prev_ = this` (`:2028`) HAS NO COUNTERPART: the parent is the position in
    /// this list.
    #[must_use = "a refused insertion hands the node back and it is lost if dropped"]
    pub fn add_child_node(&mut self, at: InsertionPoint, node: ChildNode) -> Option<ChildNode> {
        match self.insertion_point(at) {
            Some(index) => {
                self.next.insert(index, node);
                None
            }
            None => Some(node),
        }
    }

    /// The authority's `insertionPoint` iterator resolved to an index (`dsc/dsc2.cpp:2015-2026`).
    /// [`None`] is the sibling it cannot find.
    fn insertion_point(&self, at: InsertionPoint) -> Option<usize> {
        match at {
            InsertionPoint::Front => Some(0),
            InsertionPoint::Back => Some(self.next.len()),
            InsertionPoint::Before(sibling) => (sibling < self.next.len()).then_some(sibling),
            InsertionPoint::After(sibling) => (sibling < self.next.len()).then_some(sibling + 1),
        }
    }

    /// `deleteChildNode(ownerDsc, nodeToDelete, /*nonDestructive=*/true)`
    /// (`dsc/dsc2.cpp:2188-2207`): take the child out of the list and hand it to the caller alive.
    /// That arm skips `cleanupAllocation` entirely (`:2188-2190`) and `release()`s the `unique_ptr`
    /// before erasing it (`:2202-2206`), which is a MOVE out of the tree — what returning the
    /// [`ChildNode`] by value is.
    ///
    /// ⭐ ITS CALLER IS `moveChildNode` (`:2038`), and taking the node out rather than deleting it is
    /// what lets a move be spelled as a take plus an [`Self::add_child_node`] — two operations on
    /// two objects, where the authority's one takes the destination parent as a third pointer.
    ///
    /// ⛔ THE DESTRUCTIVE ARM IS NOT HERE: it needs `DesignSpaceConfig::cleanupAllocation`
    /// (`dsc/designSpaceConfig.h:262`), which is unported, and dropping the returned node is not the
    /// same thing — that cleanup unregisters the allocation from the DSC.
    #[must_use = "the node is removed from the tree and is lost if dropped"]
    pub fn take_child_node(&mut self, child: usize) -> Option<ChildNode> {
        (child < self.next.len()).then(|| self.next.remove(child))
    }

    /// `moveChildren(toNode)` (`dsc/dsc2.cpp:2053-2059`).
    ///
    /// ⛔ IT OVERWRITES THE DESTINATION'S CHILDREN, IT DOES NOT APPEND: `toNode->next_ =
    /// std::move(this->next_)` (`:2057`), so a non-empty destination silently loses its own. Both
    /// live callers pass a freshly minted, still-detached loop
    /// (`ddc/ddc_transformation_util.cpp:236-245`,
    /// `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:6852-6859`), and
    /// [`Self::insert_perfectly_nested_block_node`] REFUSES a non-empty node before calling this
    /// (`dsc/dsc2.cpp:2044-2048`) — the authority's own guard against its own overwrite.
    ///
    /// ⭐ AND "STILL DETACHED" IS WHAT MAKES THIS EXPRESSIBLE AT ALL: two `&mut BlockNode`s are two
    /// distinct objects only because every caller moves the children into the new node BEFORE
    /// linking it into the tree (`ddc/ddc_transformation_util.cpp:245` then `:247`).
    ///
    /// ⛔ `child->prev_ = toNode` (`:2055-2056`) has no counterpart: the children's parent is the
    /// list they are in, and they change lists here.
    pub fn move_children_to(&mut self, to: &mut BlockNode) {
        to.next = std::mem::take(&mut self.next);
    }

    /// `insertPerfectlyNestedBlockNode(nodeToAdd)` (`dsc/dsc2.cpp:2041-2051`): make `node` this
    /// node's ONLY child and give it everything that was here.
    ///
    /// ⭐ THE REFUSAL HANDS THE NODE BACK, as [`Self::add_child_node`] does: "Nested node must not
    /// have any children" (`:2044-2046`) is the guard that keeps [`Self::move_children_to`]'s
    /// overwrite from losing them. ⛔ AND IT IS NOT REDUNDANT WITH THE MOVE: a node with children
    /// would keep them in the authority too — the overwrite happens on the way IN, so the caller's
    /// children would be the ones dropped.
    ///
    /// ⚠️ THE AUTHORITY'S PARAMETER IS `BlockNode*` AND ITS ONE CALLER PASSES A `LoopNode*`
    /// (`dsc/dsc2.h:550`, `ddc/ddc_transformation_util.cpp:301`), so WHICH KIND of node gets nested
    /// is the caller's static choice and not something this body can recover — see
    /// [`Self::insert_perfectly_nested_loop_node`], which is the reading that one caller needs. This
    /// variant is the parameter's own declared type and has no live caller in the authority.
    ///
    /// ⛔ NEITHER VARIANT TAKES A [`ChildNode`]: the authority's parameter is `BlockNode*` because a
    /// leaf cannot hold the children it is handed.
    #[must_use = "a refused nesting hands the node back and it is lost if dropped"]
    pub fn insert_perfectly_nested_block_node(&mut self, mut node: BlockNode) -> Option<BlockNode> {
        if !node.next.is_empty() {
            return Some(node);
        }
        self.move_children_to(&mut node);
        self.next.push(ChildNode::Block(node));
        None
    }

    /// `insertPerfectlyNestedBlockNode(nodeToAdd)` reached with a `LoopNode*`, which is what its ONLY
    /// caller passes (`dsc/dsc2.cpp:2041-2051`): `Ddc::splitLoopBandOnDatastage` mints a loop over the
    /// base loop's own dims and nests it inside that loop, then moves the denominator datastage down
    /// into it (`ddc/ddc_transformation_util.cpp:287-303`, the call at `:301`).
    ///
    /// ⭐ A SECOND NAME RATHER THAN A WIDER PARAMETER, because the kind of the nested node SURVIVES
    /// here: `BlockNode*` erases it in the authority and every later reader gets it back with a
    /// `dynamic_cast`, whereas the node enters this tree as [`ChildNode::Loop`] and stays a loop for
    /// the walk that follows.
    #[must_use = "a refused nesting hands the node back and it is lost if dropped"]
    pub fn insert_perfectly_nested_loop_node(&mut self, mut node: LoopNode) -> Option<LoopNode> {
        if !node.base_class.next.is_empty() {
            return Some(node);
        }
        self.move_children_to(&mut node.base_class);
        self.next.push(ChildNode::Loop(node));
        None
    }
}

/// Replaces: e034_LoopNode
/// Replaces: e031_LoopNode
///
/// `dsc/dsc2.h:563-619`. One loop of the schedule tree: the dims it iterates and the two data
/// stages whose ratio is its trip count (`getTripCount(dsc, dim, numId_, denId_)`,
/// `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:1840`). The minting site is
/// `ddc/ddl/ddl_conversion.cpp:1076-1164`: a DDL `LoopOp` becomes a datastage loop and a
/// `ParametricLoopOp` a parametric one.
///
/// ⛔ THIS CARRIES ALL SIX OF LOOPNODE'S OWN DECLARED FIELDS AND ITS BASE SUBOBJECT. `next_` is
/// `BlockNode`'s (`dsc/dsc2.h:538`) and `nodeType_`, `name_` and `relevantComps_` are
/// `ScheduleNode`'s (`:460-461`, `:516`); all four arrive through
/// [`base_class`](Self::base_class), because `LoopNode : public InheritWithClone<BlockNode,
/// LoopNode>` (`:563`) and a loop IS a block node — `isBlockNode()` says so (`:475-477`) and the
/// traversal descends into a loop's children through exactly that cast (`dsc/dsc2.cpp:2243`).
/// ⛔ `prev_` (`:515`) IS THE ONE FIELD NOT CARRIED, and not because an owner is missing: in an
/// owning tree the parent is the list a node is in, so `prev_` would be a second, independently
/// writable answer to a question [`BlockNode::children`] already answers — see [`ScheduleTree`].
///
/// ⛔ AND IT IS THE METHODS, NOT THE FIELD COUNT, THAT KEEP THE `e031_LoopNode`/`e034_LoopNode`
/// ANCHORS AT THE END OF THIS FILE OPEN. A unit is its fields AND its methods together, and three of
/// this class's nine still cannot be written: `parametricIterCount` (`dsc/dsc2.cpp:4126`) and
/// `parametricStride` (`:4197`) read `DesignSpaceConfig::dataStageParam_` and `labeledDs_`
/// (e027_DesignSpaceConfig), and `print` (`:4284`) prints `this` — a raw address (`:4288`).
///
/// ⚠️ FIVE OF EACH ANCHOR SET NAME NOTHING THIS TYPE CAN CARRY: `Ddc`, `DesignSpaceConfig`,
/// `ScheduleTree` and `ScheduleNode` are the four `friend class` declarations (`dsc/dsc2.h:611-614`),
/// and `rowId` is `parametricIterCount`'s fourth PARAMETER, `int rowId = -1` (`:602`). They stay open
/// because a deleted anchor cannot be told from a finished one.
///
/// ⛔ AND THE `Clone` DERIVE IS EXACT ONLY BECAUSE [`BlockNode`] HAND-WRITES ITS OWN. IBM's `clone()`
/// is `new Derived(static_cast<Derived const&>(*this))` (`util/utils.h:105-107`), i.e. the copy
/// constructor, and `BlockNode::next_`'s copy constructor is EMPTY ON PURPOSE
/// (`VectorOfChildren(const VectorOfChildren&) {}`, `dsc/dsc2.h:533-536`) — so cloning a loop yields
/// a loop with NO CHILDREN and the caller re-inserts them. `ddc/ddc_transformation.cpp:984-986` is
/// that caller: it clones a loop and then `addChildNode`s one cloned child. Deriving `Clone` here
/// reproduces it because [`BlockNode::clone`] drops the children and copies the rest.
///
/// ⛔ THE THREE SHAPES BELOW ARE DISJOINT AND THIS TYPE CANNOT ENFORCE IT — IBM declares four
/// independent fields and the JSON importer writes them one entry at a time
/// (`dsc/dsc2.cpp:1405-1426`), so a tagged enum would have to reject a legal intermediate state:
///  * a datastage loop has both ids and no `parametricLdsIdx_` (`ddc/ddl/ddl_conversion.cpp:1096-1103`);
///  * a parametric loop has NEITHER id, exactly one dim and a `parametricLdsIdx_` (`:1127-1161`),
///    and cannot also be symbolic (`ddc/ddcv1.cpp:2830-2831`);
///  * the tree's head has `denId_` alone — `ScheduleTree()` writes the core stage into it and
///    leaves `numId_` absent (`dsc/dsc2.h:629`).
///
/// ⛔ NO `PartialEq`: IBM declares none, and the node identity every consumer uses is the POINTER.
/// `dsc_loops_to_mlir_loops_map_` (`dsc-based-utils/DSC2ToDataflowIR/V3/SNDSCLowering.hpp:168-169`)
/// and `LoopDistributionParamPerNodeType` (`dsc/dsc2.h:1118-1119`) are keyed by `const LoopNode*`,
/// and `loop_labels_` holds one as its VALUE under the DDL's label string
/// (`ddc/ddl/ddl_conversion.h:409-410`). ⛔ TRAP: `LoopDistributionParamPerLoopType`, declared three
/// lines above the per-node one, is NOT a loop key at all — it is keyed by `const PrimaryDimTypes`
/// (`dsc/dsc2.h:1115-1116`).
#[derive(Clone, Debug)]
pub struct LoopNode {
    /// The `BlockNode` subobject (`dsc/dsc2.h:563`), which is where a loop's CHILDREN live — the
    /// loop body. ⭐ EVERY INTERIOR NODE OF A SCHEDULE TREE IS ONE OF THESE, and the tree's root is a
    /// bare `LoopNode` with nothing above it (`ScheduleTree::head_`, `:623`).
    ///
    /// PUBLIC, because the authority's inheritance is public and its own code reaches straight
    /// through: `scheduleTree_.getHeadMutable()->next_` (`dsc/dsc2.cpp:1389`) and
    /// `loopNode->getNextView(comp)` (`ddc/ddcv1.cpp:3413`) are both a loop used as a block node.
    pub base_class: BlockNode,
    /// Field: e031_LoopNode.numId_
    /// Field: e034_LoopNode.numId_
    ///
    /// The numerator stage (`dsc/dsc2.h:573`). ⛔ `-1` IS ABSENT, AND A READER EITHER THROWS ON IT OR
    /// **MINTS** IT. The `.at()` readers throw (`dsc/dsc2.cpp:2995`, itself behind
    /// `!isParametricLoop()` at `:2994`; `:6104`, `ddc/ddcv1.cpp:2496`, `:2835`,
    /// `ddc/ddc_fold.cpp:2349`, `:3235`, `:3604`); the DDL emission instead reads
    /// `dataStageParam_[numId_]` through the NON-CONST `operator[]`, with NO parametric guard
    /// (`ddc/ddl/ddl_conversion.cpp:2974`, in the DFS at `:2951-3082`), and a parametric loop always
    /// arrives with `-1` (minted `:1129-1130`, linked into the tree `:1162`) — so the key is
    /// DEFAULT-INSERTED and the `else` branch runs on a stage named `""`. ⭐ [`DataStage::name`]
    /// carries what that insert then makes true.
    ///
    /// ⛔ AND THAT `else` BRANCH MINTS IN A SECOND CONTAINER. It reads
    /// `metadata_.datastages_[numId_].strategyMinimize_` (`:2987`), an `unordered_map<int, Datastage>`
    /// (`ddc/ddc_metadata.h:82`), where the emptiness test above it read a
    /// `map<int, dsc2::DataStage>` (`dsc/designSpaceConfig.h:105`) — so ONE parametric loop inserts
    /// `-1` into BOTH, and the strategy the emission then attaches is decided by the value that insert
    /// default-constructed: `strategyMinimize_ = true` (`ddc/ddc_metadata.h:77`), hence `"minimize"`,
    /// never the `"maximize"` the line above it states. [`den_id`](Self::den_id) takes the same pair of
    /// reads thirteen lines on (`:2998`, `:3011`).
    ///
    /// ⭐ AND `-1` IS NOT A SENTINEL EVERYWHERE IT IS A KEY. `constraints_[loop->numId_]`
    /// (`ddc/ddcv1.cpp:610`, `:649`) is the same `operator[]` shape, but there `-1` is the DECLARED
    /// key for "absolute constraints" (`ddc/ddc_metadata.h:74`), written as such directly at `:689` —
    /// so a parametric loop's constraints land in the absolute bucket, and `metadata.rs` recording
    /// that key as an `Option` is what keeps the two spellings one.
    pub num_id: Option<DataStageId>,
    /// Field: e031_LoopNode.denId_
    /// Field: e034_LoopNode.denId_
    ///
    /// The denominator stage (`dsc/dsc2.h:574`). ⛔ NOT SYMMETRIC WITH [`num_id`](Self::num_id):
    /// this is the id with `>= 0` guards, and all three sit where the reader has climbed
    /// `getOwnerLoop()` to a PARENT loop and can therefore reach the head — `exploreAssignDataStages`
    /// at `ddc/ddcv1.cpp:634` and `:645`, and `parametricIterCount` at `dsc/dsc2.cpp:4144-4147`. Its
    /// other uses in that same function carry no such test (`ddc/ddcv1.cpp:694`, `:701`, `:727-729`),
    /// so the guard marks the climb, not the field.
    ///
    /// ⛔ AND ABSENCE IS READ THREE WAYS, NOT TWO. Besides those guards and the `.at()` readers that
    /// throw, `-1` ABORTS at `ddc/ddcv1.cpp:607-608`, where `metadata.datastages_.find(denId_)` is
    /// followed by a `DT_CHECK` on the iterator — the same container the guarded site then tests with
    /// `count` (`:646`) — and it MINTS in the DDL emission, which reads `dataStageParam_[denId_]` and
    /// `metadata_.datastages_[denId_]` through the non-const `operator[]` in the SAME
    /// parametric-guard-free DFS that does it to [`num_id`](Self::num_id)
    /// (`ddc/ddl/ddl_conversion.cpp:2998`, `:3011`, in the walk at `:2951-3082`).
    /// `ddc/ddc_transformation.cpp:988` and `:1023` are that same `operator[]` shape and go on to WRITE
    /// the value they inserted, though both take their id from a stage the caller has just minted
    /// rather than from a parametric loop.
    pub den_id: Option<DataStageId>,
    /// Field: e031_LoopNode.dims_
    ///
    /// ⛔ ORDERED INNER TO OUTER (`dsc/dsc2.h:575`, and `dsc/dsc2Pcfg.cpp:517` says `// Inner to
    /// outer` verbatim over a front-to-back walk), and bridge 1 depends on that: it walks `dim_idx`
    /// from `dims_.size() - 1` down to 0, and the loop it opens FIRST is the one it later retrieves
    /// as `.at(0)` = outermost, so the LAST entry becomes the OUTERMOST loop of the emitted nest
    /// (`dsc-based-utils/DSC2ToDataflowIR/V3/SNControlFlowLowering.cpp:893`, `:898-957`).
    pub dims: Vec<PrimaryDimAndKind>,
    /// Field: e031_LoopNode.loopCountSymbolIds_
    ///
    /// One entry per symbolic dim: a single symbol for a pure symbolic or pivot dim, several
    /// (max-pivot) for an irregular one (`dsc/dsc2.h:576-578`).
    ///
    /// ⛔ FILLED LATE, AND BY ONE SITE ONLY: `finalizeScheduleTree` writes it for a non-parametric
    /// loop whose numerator stage is symbolic in a dim and whose denominator is not
    /// (`dsc/dsc2.cpp:2992-3010`); `ddc/ddcv1.cpp:2832-2833` reimplements the test rather than read
    /// this map for exactly that reason. ⛔ AND IT IS WRITTEN THROUGH `operator[]` (`:3001`), so a
    /// dim whose `dimToSymbolMapping_` entry is empty leaves an EMPTY vector behind —
    /// [`is_dim_symbolic`](Self::is_dim_symbolic) then answers yes while bridge 1's
    /// `DT_CHECK(size() == 1)` on the bound is what fails (`SNControlFlowLowering.cpp:921-923`).
    pub loop_count_symbol_ids: BTreeMap<PrimaryDimTypes, Vec<VariableSymbol>>,
    /// Field: e031_LoopNode.isParametricLoop_
    /// Field: e034_LoopNode.isParametricLoop_
    ///
    /// Private in IBM's declaration (`dsc/dsc2.h:617`) and ONE-WAY: `markAsParametricLoop` is the
    /// only writer tree-wide (`ddc/ddl/ddl_conversion.cpp:1128`, `dsc/dsc2.cpp:1412`) and nothing
    /// clears it. Readers go through `isParametricLoop()` (`ddc/ddcv1.cpp:2830`,
    /// `dsc/dsc2.cpp:2994`, `:4129`) except the JSON exporter, which reaches the field itself
    /// through friendship (`dsc/dsc2.cpp:414`, `:416`) — a read, so the getter below still covers it
    /// and this stays private.
    is_parametric_loop: bool,
    /// Field: e031_LoopNode.parametricLdsIdx_
    /// Field: e034_LoopNode.parametricLdsIdx_
    ///
    /// The reference tensor whose cumulative stick size along the loop dim IS the parametric loop's
    /// stride (`dsc/dsc2.h:618`, read at `dsc/dsc2.cpp:4198-4210`). ⛔ `-1` IS ABSENT and
    /// `parametricStride` refuses on it (`:4199-4202`); the DDL conversion sets it from the
    /// reference tensor's `ldsIdx_` and rejects a tensor without one
    /// (`ddc/ddl/ddl_conversion.cpp:1155-1161`).
    parametric_lds_idx: Option<LdsIdx>,
}

impl Default for LoopNode {
    /// `LoopNode() : BaseClass(LOOP) {}` (`dsc/dsc2.h:583`) over the authority's member initialisers
    /// (`:573-578`). ⭐ THE TAG IS SET HERE, AND THAT IS WHAT MAKES IT UNFORGEABLE: the only public
    /// route to a fresh loop is this one, so no caller can mint a `LoopNode` whose base says
    /// `TRANSFER` — [`BlockNode::new`], the protected inherited constructor that takes a kind, is
    /// module-private.
    fn default() -> Self {
        Self {
            base_class: BlockNode::new(NodeType::Loop),
            num_id: None,
            den_id: None,
            dims: Vec::new(),
            loop_count_symbol_ids: BTreeMap::new(),
            is_parametric_loop: false,
            parametric_lds_idx: None,
        }
    }
}

impl LoopNode {
    /// `dsc/dsc2.h:586-593`, the constructor added for the DSC2.1-to-Dataflow translator. Its one
    /// caller mints a DUMMY loop for an implicit transfer dim with both ids absent
    /// (`LoopNode(-1, -1, {view_sizes[i].dim_})`,
    /// `dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:734`), which is why the ids are
    /// nullable here. ⛔ It leaves `parametricLdsIdx_` absent even when `isParametricLoop` is set,
    /// so a loop built parametric this way cannot answer `parametricStride`.
    pub fn new(
        num_id: Option<DataStageId>,
        den_id: Option<DataStageId>,
        dims: Vec<PrimaryDimAndKind>,
        is_parametric_loop: bool,
    ) -> Self {
        Self {
            num_id,
            den_id,
            dims,
            is_parametric_loop,
            ..Self::default()
        }
    }

    /// `dsc/dsc2.h:595-597`. Whether this loop's count for `dim` comes from a symbol rather than
    /// from the two stages' dims. ⛔ It is `count(dim)` on the map and nothing else: the dim need
    /// not be one of [`dims`](Self::dims) for this to answer yes.
    pub fn is_dim_symbolic(&self, dim: PrimaryDimTypes) -> bool {
        self.loop_count_symbol_ids.contains_key(&dim)
    }

    /// `dsc/dsc2.h:599`.
    pub fn is_parametric_loop(&self) -> bool {
        self.is_parametric_loop
    }

    /// `dsc/dsc2.h:600`. One-way: nothing tree-wide clears the flag.
    pub fn mark_as_parametric_loop(&mut self) {
        self.is_parametric_loop = true;
    }

    /// `dsc/dsc2.h:603`.
    pub fn parametric_lds_idx(&self) -> Option<LdsIdx> {
        self.parametric_lds_idx
    }

    /// `dsc/dsc2.h:604`. It takes the absent case because the JSON importer feeds back the `-1`
    /// that every non-parametric loop exports (`dsc/dsc2.cpp:420-421`, `:1414-1415`).
    pub fn set_parametric_lds_idx(&mut self, idx: Option<LdsIdx>) {
        self.parametric_lds_idx = idx;
    }

    /// `dsc/dsc2.cpp:4223-4228`. Whether `dim` is one of this loop's, THE KIND IGNORED — the
    /// authority destructures `dims_` and compares `loopDim` alone. Its one caller is
    /// `ScheduleNode::getParentDimLoop`, which climbs owner loops until one answers yes
    /// (`dsc/dsc2.cpp:1906-1914`).
    pub fn has_loop_dim(&self, dim: PrimaryDimTypes) -> bool {
        self.dims.iter().any(|d| d.dim == dim)
    }
}

/// Replaces: e042_ScheduleTree
/// Replaces: e032_ScheduleTree
///
/// `dsc2::ScheduleTree` (`dsc/dsc2.h:621-652`) — THE SCHEDULE. One object, one root, and every node
/// of the program underneath it. `DesignSpaceConfig::scheduleTree_` (`dsc/designSpaceConfig.h:115`)
/// is the only holder, and it is what bridge 1 walks to emit a program
/// (`DSC2ToDataflowIR.cpp:262-263`, `:380`) and what the DDL conversion fills
/// (`ddc/ddl/ddl_conversion.cpp:2779-2782`).
///
/// ⭐ THE NUMBER THAT MATTERS HANGS OFF THIS TYPE: IBM's 187 `g0` fixture trees hold 14,711 nodes,
/// and every one of them is reached from here through [`BlockNode::children`].
///
/// ⛔ NO `Clone`, AND THAT IS THE AUTHORITY'S OWN REFUSAL MADE STATIC. `copyFrom` is
/// `DT_ERROR("Not yet able to deep copy a schedule tree")` on any non-empty tree
/// (`dsc/dsc2.cpp:2267-2281`) — the deep copy it would have performed is entirely commented out
/// there, including its own `/// TODO: add update of pointers throughout the various nodes` — and
/// `operator=` is `= delete` "so that copy needs to be more voluntary" (`dsc/dsc2.h:636-637`). The
/// copy CONSTRUCTOR is the only route in and it forwards to `copyFrom` (`:633`), so a holder's
/// implicit copy is exactly what aborts. ⚠️ AN EMPTY TREE DOES COPY THERE AND CANNOT HERE: the guard
/// is `!oldTree.head_.next_.empty()`, so copying an empty one returns `*this` UNMODIFIED — it does
/// not even clear the destination. Nothing tree-wide calls `copyFrom` and nothing copies a
/// `DesignSpaceConfig`, so the divergence is a refusal nobody reaches, and a `Clone` that panicked on
/// a non-empty tree would be a run-time refusal where this is a compile error.
///
/// ⛔ AND THE HEAD IS NEVER A VISITED NODE. `traverseTreeDFS` seeds its work list with `head_.next_`
/// and never pushes `head_` itself (`dsc/dsc2.cpp:2224-2226`), so the root loop is a container for the
/// program and not a loop in it — which is why [`Self::default`] leaves `numId_` absent and why
/// [`LoopNode`] records that shape as one of its three.
pub struct ScheduleTree {
    /// Field: e042_ScheduleTree.head_
    /// Field: e032_ScheduleTree.head_
    ///
    /// The root (`dsc/dsc2.h:623`). PRIVATE, as the authority's is — the only `private:` section in
    /// this class — with [`Self::head`] and [`Self::head_mut`] the two public readings
    /// (`:635-636`).
    ///
    /// ⭐ AND IT IS A `LoopNode` BY VALUE, NOT A POINTER: the tree always has a root, an empty tree is
    /// a root with no children (`:625-626`), and every node in the program is owned transitively from
    /// here. That is what makes [`Self::clear`] `head_.next_.clear()` (`:625`) and nothing else.
    head: LoopNode,
}

impl Default for ScheduleTree {
    /// `ScheduleTree() { head_.denId_ = 0; }` — the root's denominator is the CORE DATASTAGE, which is
    /// what the authority's own trailing comment on that line says (`dsc/dsc2.h:629`).
    ///
    /// ⛔ AND `numId_` STAYS ABSENT, which is the head's whole shape: `denId_` alone. Every `>= 0`
    /// guard on a climbed parent's numerator exists for it (`ddc/ddcv1.cpp:634`, `:645`).
    fn default() -> Self {
        Self {
            head: LoopNode {
                den_id: Some(DataStageId(0)),
                ..LoopNode::default()
            },
        }
    }
}

impl ScheduleTree {
    /// `clear()` (`dsc/dsc2.h:625`): drop every node, keep the root. ⛔ IT DOES NOT RESET THE ROOT —
    /// `denId_` and any `relevantComps_` set on the head survive, because only `next_` is cleared.
    pub fn clear(&mut self) {
        self.head.base_class.next.clear();
    }

    /// `empty()` (`dsc/dsc2.h:626`), which is the root having no children and says nothing about the
    /// root itself. `copyFrom`'s refusal is gated on it (`dsc/dsc2.cpp:2277`).
    pub fn is_empty(&self) -> bool {
        self.head.base_class.next.is_empty()
    }

    /// `getHead()` (`dsc/dsc2.h:635`), read by the DDL conversion to find where to insert
    /// (`ddc/ddl/ddl_conversion.cpp:2777`) and by the L3 scheduler's walkers
    /// (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:1596`).
    pub fn head(&self) -> &LoopNode {
        &self.head
    }

    /// `getHeadMutable()` (`dsc/dsc2.h:636`), which is how every minting site attaches its node:
    /// `scheduleTree_.getHeadMutable()->addChildNode(..)` (`ddc/ddl/ddl_conversion.cpp:2781`,
    /// `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:3225`, `dsc/dsc2.cpp:1389`).
    pub fn head_mut(&mut self) -> &mut LoopNode {
        &mut self.head
    }

    /// `traverseTreeDFS()` with no start node (`dsc/dsc2.cpp:2222-2265`): every node of the tree in
    /// pre-order, the root excluded. Bridge 1 collects a component's transfers this way
    /// (`DSC2ToDataflowIR.cpp:262-263`) and the DDC its allocations (`ddc/ddcv1.cpp:3416-3417`).
    ///
    /// ⭐ ROOT EXCLUDED BECAUSE THE AUTHORITY SEEDS WITH `head_.next_` (`:2225`), so the head is the
    /// only node of the tree no walk ever returns. Everything else about the walk — the relevance
    /// pruning, the empty `nodeTypes` meaning "collect all", the dead `maxLoopDepth` — is
    /// [`ChildNode::traverse_dfs`]'s, which this delegates to per root child.
    pub fn traverse_dfs(&self, node_types: &[NodeType], comp: SenComponent) -> Vec<&ChildNode> {
        let mut order = Vec::new();
        for child in self.head.base_class.children() {
            child.visit_dfs(node_types, comp, &[], &mut order);
        }
        order
    }

    /// `traverseTreeDFSMutable()` (`dsc/dsc2.cpp:2208-2220`), which is `traverseTreeDFS` with every
    /// pointer `const_cast` back to mutable (`:2216`). Its live callers rename nodes
    /// (`dsc/dsc2.cpp:2985-2992`), mark loops parametric (`ddc/ddcv1.cpp:2830-2831`) and rewrite
    /// transfer offsets (`:441-442`).
    ///
    /// ⛔ IT IS A VISITOR, NOT A `Vec<&mut ChildNode>`, AND THAT IS THE ONE DIVERGENCE THIS METHOD
    /// CARRIES. The authority hands back a flat list of mutable pointers ALIASING the tree it walked;
    /// two of them can be held at once and one can be an ancestor of another, which is what a
    /// `Vec<&mut _>` cannot be. Every live caller uses the list as a `for` loop over one node at a
    /// time, so the visitor serves all of them — what it forbids is holding node `i` while touching
    /// node `j`. ⚠️ AND THE INTERLEAVING DIFFERS: the authority walks the tree ONCE, collects, and
    /// then the caller's loop runs, so a mutation cannot change what is visited; here the mutation
    /// happens DURING the walk, so a `f` that adds children sees them. No live caller adds one —
    /// `finalizeScheduleTree` writes `name_` (`dsc/dsc2.cpp:2988-2992`), `ddcv1` writes a flag and an
    /// offset — and adding one through `f` is impossible anyway, because `f` receives the node and not
    /// its parent.
    pub fn for_each_dfs_mut(
        &mut self,
        node_types: &[NodeType],
        comp: SenComponent,
        mut f: impl FnMut(&mut ChildNode),
    ) {
        for child in self.head.base_class.next.iter_mut() {
            child.visit_dfs_mut(node_types, comp, &mut f);
        }
    }
}

/// Replaces: CondOp
///
/// `dsc/dscdefn.h:95-107`. The comparison a conditional region's guard applies to a loop's
/// iteration variable — the operator half of a `LoopCond` (`dsc/dsc2.h:661`).
///
/// ⛔ ONLY THE SIX RELATIONAL OPERATORS EVER REACH A `LoopCond`, AND BOTH ENDS OF OUR PATH REFUSE
/// THE REST. The DDL conversion rejects the DDL outright for anything outside
/// `{EQ, NE, GE, LE, GT, LT}` (`ddc/ddl/ddl_conversion.cpp:260-264`), and bridge 1's
/// `getCmpIPredicate_dup` maps exactly those six onto an `arith::CmpIPredicate` and fails on every
/// other value (`dsc-based-utils/DSC2ToDataflowIR/V3/SNControlFlowLowering.cpp:23-44`), refused at
/// `:141-143` and again at `:251-253`. `TOGGLE`, `ALWAYS`, `NEVER` and `CONST` carry the DCG/PCFG
/// conditional vocabulary, which is off this campaign's path, and `DEFAULT` is the field's own
/// initialiser (`dsc/dsc2.h:661`) — no comparison chosen yet.
///
/// ⛔ THE DISCRIMINANTS ARE NOT OBSERVABLE, so `ALL`'s order and the E0080 guard below are a SHAPE
/// guard and not a wire guard: `condOpToString` is a `std::map` keyed by this enum
/// (`dsc/dscdefn.h:110`), but neither map is ever iterated — every use is `.at()` or `.find()`
/// (`dsc/dsc2.cpp:462`, `:1442`, `ddc/ddl/ddl_conversion.cpp:254`, `:3273`) — and nothing orders
/// operators relationally: neither `LccrCond` (`dsc/dscdefn.h:114-119`) nor `PcfgLccrCond`
/// (`dsc/pcfg.h:39-44`) declares a comparison operator at all.
///
/// ⛔ AND THE DERIVED `Default` IS `LoopCond::condOp_`'s INITIALISER, NOT AN ENUM-WIDE ONE: both of
/// those other carriers of a `CondOp` leave the field UNINITIALISED (`dsc/dscdefn.h:117`,
/// `dsc/pcfg.h:42`), so a port of either must not read its default out of this enum.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CondOp {
    /// `==`
    Eq = 0,
    /// `!=`
    Ne = 1,
    /// `<`
    Lt = 2,
    /// `<=`
    Le = 3,
    /// `>`
    Gt = 4,
    /// `>=`
    Ge = 5,
    Toggle = 6,
    Always = 7,
    Never = 8,
    /// The authority's `CONST`, "used when the condition should not be evaluated"
    /// (`dsc/dscdefn.h:105`).
    Const = 9,
    /// `LoopCond::condOp_`'s initialiser (`dsc/dsc2.h:661`).
    #[default]
    Default = 10,
}

/// ⛔ E0080 IF AN OPERATOR IS EVER INSERTED, DROPPED, REORDERED OR LEFT OUT OF `ALL`: `ALL` is
/// positional against the discriminants, so this is what keeps it a faithful copy of
/// `dsc/dscdefn.h:95-107` rather than a list that merely happens to be the right length.
const _: () = {
    let mut i = 0;
    while i < CondOp::ALL.len() {
        assert!(
            CondOp::ALL[i] as usize == i,
            "CondOp::ALL is out of declaration order"
        );
        i += 1;
    }
};

impl CondOp {
    /// Every operator in the authority's declaration order (`dsc/dscdefn.h:95-107`). There is no
    /// count sentinel here, so all eleven are real values.
    pub const ALL: [Self; 11] = [
        Self::Eq,
        Self::Ne,
        Self::Lt,
        Self::Le,
        Self::Gt,
        Self::Ge,
        Self::Toggle,
        Self::Always,
        Self::Never,
        Self::Const,
        Self::Default,
    ];

    /// The six a `LoopCond` can carry on this campaign's path, in the authority's declaration order
    /// rather than the `is_any_of` argument order. Both ends state the same set independently: the
    /// DDL conversion accepts only these (`ddc/ddl/ddl_conversion.cpp:260-261`) and bridge 1 maps
    /// only these onto a predicate (`dsc-based-utils/DSC2ToDataflowIR/V3/SNControlFlowLowering.cpp:23-41`).
    pub const COMPARISONS: [Self; 6] = [Self::Eq, Self::Ne, Self::Lt, Self::Le, Self::Gt, Self::Ge];

    /// The spelling `EnumsConversion::condOpToString` gives this operator (`dsc/dscdefn.h:110`,
    /// defined `dsc/dscdefn.cpp:19-28`).
    ///
    /// ⭐ TOTAL, AND THE AUTHORITY'S MAP IS TOO: all eleven operators have an entry, so none of the
    /// `condOpToString.at(...)` callsites can throw.
    pub fn name(self) -> &'static str {
        match self {
            Self::Eq => "eq",
            Self::Ne => "ne",
            Self::Lt => "lt",
            Self::Le => "le",
            Self::Gt => "gt",
            Self::Ge => "ge",
            Self::Toggle => "toggle",
            Self::Always => "always",
            Self::Never => "never",
            Self::Const => "const",
            Self::Default => "default",
        }
    }

    /// `EnumsConversion::stringToCondOp`, the flip of the above — IBM builds it with `flipMap`
    /// (`dsc/dscdefn.h:111`, `dsc/dscdefn.cpp:30-31`). An unknown spelling is absent, which is what
    /// both readers already test for: the DDL conversion errors out on a `find` miss
    /// (`ddc/ddl/ddl_conversion.cpp:254-258`) while the JSON importer's `.at()` throws
    /// (`dsc/dsc2.cpp:1442-1443`).
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "eq" => Some(Self::Eq),
            "ne" => Some(Self::Ne),
            "lt" => Some(Self::Lt),
            "le" => Some(Self::Le),
            "gt" => Some(Self::Gt),
            "ge" => Some(Self::Ge),
            "toggle" => Some(Self::Toggle),
            "always" => Some(Self::Always),
            "never" => Some(Self::Never),
            "const" => Some(Self::Const),
            "default" => Some(Self::Default),
            _ => None,
        }
    }
}

/// Replaces: LoopCond::CondValType
///
/// `dsc/dsc2.h:655`. What the right-hand side of a loop condition IS: a literal iteration index, or
/// the loop's first or last iteration whatever its bounds turn out to be.
///
/// ⛔ `condValInt_` IS ONLY READ UNDER `INT`, and bridge 1 guards every read with a
/// `DT_CHECK(... == INT)` — for an `affine.for` it takes the constant bounds instead
/// (`dsc-based-utils/DSC2ToDataflowIR/V3/SNControlFlowLowering.cpp:99-114`), for an `scf.for` the
/// bound values (`:121-135`), and the single-`and` path repeats both (`:214-226`, `:234-245`).
///
/// ⛔ AND `LAST` IS `upperBound - 1`, NOT THE UPPER BOUND: constant-folded at `:101-103` and
/// emitted as an `arith.subi` against `1` at `:122-126`. Comparing against the bound itself is a
/// legal-looking program that guards the wrong trip.
///
/// `INT` is the fallback as well as the declared default (`dsc/dsc2.h:662`): the DDL conversion
/// looks the value expression up in `stringToCondValType` and, on a miss, parses it as an integer
/// expression instead (`ddc/ddl/ddl_conversion.cpp:266-274`).
///
/// ⛔ ON A DROPPED DIM `FIRST`/`LAST` NEVER REACH A `LoopCond` AT ALL: the dim is treated as a loop
/// of size one, both forms collapse to `condVal = 0`, and the condition is resolved to a bool against
/// 0 right there (`ddc/ddl/ddl_conversion.cpp:275-294`). So `LAST = upperBound - 1` is the real-loop
/// rule; where the loop does not exist, first and last are the same iteration.
///
/// ⛔ AND `condValInt_` IS EXPORTED UNCONDITIONALLY (`dsc/dsc2.cpp:468`) then imported field by field
/// into a default-constructed `LoopCond` (`:1438-1449`), so the `-1` an unparsed `FIRST`/`LAST`
/// carries (`ddc/ddl/ddl_conversion.cpp:267`) has to round-trip.
///
/// ⛔ THE REVERSAL: this doc used to say a fused `CondVal` "would still have to hold that `-1`". It
/// does not, and [`CondVal`] is that fusion — it answers the `-1` FROM THE FORM, because `-1` is
/// `condValInt_`'s own initialiser (`dsc/dsc2.h:663`) and the one mint of an unparsed `FIRST`/`LAST`
/// passes exactly it (`ddc/ddl/ddl_conversion.cpp:267`). Holding the integer would reproduce the
/// JSON; deriving it reproduces the JSON AND makes bridge 1's four `DT_CHECK(condValType_ == INT)`
/// unspellable, which holding it does not.
///
/// ⛔ THE DISCRIMINANTS ARE NOT OBSERVABLE HERE, unlike [`NodeType`]'s: both maps are
/// `std::unordered_map` (`dsc/dsc2.h:656-657`), neither is iterated, and the JSON round trip
/// carries the spelling rather than the value (`dsc/dsc2.cpp:464-466`, `:1444-1447`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CondValType {
    /// `LoopCond::condValType_`'s initialiser (`dsc/dsc2.h:662`) and the DDL's fallback.
    #[default]
    Int = 0,
    First = 1,
    Last = 2,
}

impl CondValType {
    /// Every value form in the authority's declaration order (`dsc/dsc2.h:655`).
    pub const ALL: [Self; 3] = [Self::Int, Self::First, Self::Last];

    /// Field: e039_LoopCond.condValTypeToString
    /// Field: e030_LoopCond.condValTypeToString
    ///
    /// The spelling `LoopCond::condValTypeToString` gives this form (`dsc/dsc2.h:656`, defined
    /// `dsc/dsc2.cpp:21-23`). ⭐ TOTAL, AND THE AUTHORITY'S MAP IS TOO — all three have an entry.
    pub fn name(self) -> &'static str {
        match self {
            Self::Int => "int",
            Self::First => "first",
            Self::Last => "last",
        }
    }

    /// Field: e039_LoopCond.stringToCondValType
    /// Field: e030_LoopCond.stringToCondValType
    ///
    /// `LoopCond::stringToCondValType`, the flip of the above (`dsc/dsc2.h:657`, built with
    /// `flipMap` at `dsc/dsc2.cpp:24-25`). ⭐ AN ABSENT SPELLING IS A LIVE ANSWER HERE, not an
    /// error: the DDL conversion treats the miss as "this is an integer expression"
    /// (`ddc/ddl/ddl_conversion.cpp:268-274`). Only the JSON importer's `.at()` throws
    /// (`dsc/dsc2.cpp:1445-1447`).
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "int" => Some(Self::Int),
            "first" => Some(Self::First),
            "last" => Some(Self::Last),
            _ => None,
        }
    }
}

/// The comparison one [`LoopCond`] carries: [`CondOp`] narrowed to the six relational operators,
/// which is the whole of that enum a loop condition can hold.
///
/// ⛔ THE NARROWING DELETES TWO LIVE REFUSALS ON OUR OWN LOWERING PATH. Bridge 1's
/// `getCmpIPredicate_dup` maps exactly these six onto an `arith::CmpIPredicate` and FAILS THE WHOLE
/// LOWERING on anything else (`dsc-based-utils/DSC2ToDataflowIR/V3/SNControlFlowLowering.cpp:23-44`),
/// refused at `:141-143` and again at `:251-253`; the DDL conversion's `is_any_of` over the same six
/// (`ddc/ddl/ddl_conversion.cpp:260-264`) becomes the parse-time [`from_cond_op`](Self::from_cond_op)
/// rather than a check a built condition can still fail.
///
/// ⛔ AND IT MAKES [`CondOp::Default`] UNSPELLABLE ON A CONDITION. It is `condOp_`'s declared
/// initialiser (`dsc/dsc2.h:661`), yet every mint names a relational operator
/// (`ddc/ddcv1.cpp:3643`, `ddc/ddc_transformation.cpp:1060-1062`, `dsc/dsc2.cpp:5095-5101`,
/// `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:4811`, `ddc/ddl/ddl_conversion.cpp:317-319`), so
/// `DEFAULT` on a `LoopCond` is a state only IBM's own default constructor reaches.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LoopCondOp {
    Eq = 0,
    Ne = 1,
    Lt = 2,
    Le = 3,
    Gt = 4,
    Ge = 5,
}

/// ⛔ E0080 IF THE NARROWING EVER STOPS AGREEING WITH [`CondOp`] ON A DISCRIMINANT OR AN ORDER: the
/// two are positional against each other, which is what lets [`LoopCondOp::name`] hand its operator
/// straight to [`CondOp::name`] and keeps a container keyed by either one iterating the same way.
const _: () = {
    let mut i = 0;
    while i < LoopCondOp::ALL.len() {
        assert!(
            LoopCondOp::ALL[i] as usize == CondOp::COMPARISONS[i] as usize,
            "LoopCondOp is no longer CondOp::COMPARISONS"
        );
        i += 1;
    }
};

impl LoopCondOp {
    /// Every operator a loop condition can carry, in the authority's declaration order — the same
    /// six as [`CondOp::COMPARISONS`], positionally.
    pub const ALL: [Self; 6] = [Self::Eq, Self::Ne, Self::Lt, Self::Le, Self::Gt, Self::Ge];

    /// The narrowing both readers of a `condOp_` already perform: the DDL conversion's `is_any_of`
    /// (`ddc/ddl/ddl_conversion.cpp:260-264`) and bridge 1's predicate map
    /// (`SNControlFlowLowering.cpp:23-44`). [`None`] is the state those two refuse.
    pub fn from_cond_op(op: CondOp) -> Option<Self> {
        match op {
            CondOp::Eq => Some(Self::Eq),
            CondOp::Ne => Some(Self::Ne),
            CondOp::Lt => Some(Self::Lt),
            CondOp::Le => Some(Self::Le),
            CondOp::Gt => Some(Self::Gt),
            CondOp::Ge => Some(Self::Ge),
            CondOp::Toggle | CondOp::Always | CondOp::Never | CondOp::Const | CondOp::Default => {
                None
            }
        }
    }

    /// The spelling `EnumsConversion::condOpToString` gives this operator — [`CondOp::name`]'s, which
    /// is what the JSON exporter writes for a `condOp_` (`dsc/dsc2.cpp:464-466`).
    pub fn name(self) -> &'static str {
        CondOp::from(self).name()
    }
}

impl From<LoopCondOp> for CondOp {
    fn from(op: LoopCondOp) -> Self {
        match op {
            LoopCondOp::Eq => Self::Eq,
            LoopCondOp::Ne => Self::Ne,
            LoopCondOp::Lt => Self::Lt,
            LoopCondOp::Le => Self::Le,
            LoopCondOp::Gt => Self::Gt,
            LoopCondOp::Ge => Self::Ge,
        }
    }
}

/// One iteration of a loop dim — the `int` of `LoopCond::condValInt_` (`dsc/dsc2.h:663`).
///
/// ⛔ ITS DIRECTION IS THE READER'S, NOT THE FIELD'S, and the authority's two readers disagree:
/// bridge 1 compares the induction variable against this value verbatim, counting UP from the loop's
/// lower bound (`SNControlFlowLowering.cpp:111-114`, `:132-134`), while the PCFG translator answers
/// `loopCount - 1 - condValInt_`, counting DOWN from the last iteration (`dsc/dsc2Pcfg.cpp:788-806`).
/// This newtype carries bridge 1's — an induction value on our own lowering path — and nothing here
/// converts between the two.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct IterationIdx(pub i32);

/// What the right-hand side of a loop condition IS: `LoopCond`'s `condValType_` and `condValInt_`
/// (`dsc/dsc2.h:662-663`) as ONE value, since the integer exists only under [`CondValType::Int`].
///
/// ⛔ THIS EXISTS TO MAKE BRIDGE 1'S OWN `DT_CHECK` UNSPELLABLE. Every read of `condValInt_` on our
/// path is guarded by `DT_CHECK(condValType_ == dsc2::LoopCond::INT)` — four of them, at
/// `SNControlFlowLowering.cpp:110` and `:131`, then again at `:224` and `:243` on the single-`and`
/// path — and the state they refuse is one this enum has no variant for. [`CondValType`] records why the merge is sound at
/// all: `-1` is what `condValInt_` initialises to and what the only unparsed `FIRST`/`LAST` mint
/// passes, so [`val_int`](Self::val_int) DERIVES the JSON's integer instead of storing it.
///
/// ⛔ AND `FIRST`/`LAST` ARE NOT ITERATION INDICES, so this is not an [`Option<IterationIdx>`]: both
/// resolve against the loop's bounds at lowering time, `LAST` to `upperBound - 1` and `FIRST` to the
/// lower bound (`SNControlFlowLowering.cpp:99-109`, `:121-130`), which are values no producer of a
/// `LoopCond` knows.
///
/// ⛔ AND THAT PAIRING IS BRIDGE 1'S, NOT THE FIELD'S: the PCFG translator counts DOWN, so it
/// resolves `FIRST` to `loopCount - 1` and `LAST` to `0` — the OPPOSITE ends of the same loop
/// (`dsc/dsc2Pcfg.cpp:792-799`). Down-counting is that reader's own convention throughout, which
/// [`IterationIdx`] already records for the `INT` arm; neither variant means an index, and nothing in
/// this module converts one reader's answer into the other's.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CondVal {
    /// `INT`, carrying the `condValInt_` that form is the only reader of.
    Iteration(IterationIdx),
    /// `FIRST` — the loop's first iteration, whatever its bounds turn out to be.
    First,
    /// `LAST` — its last, which is `upperBound - 1` and not the bound.
    Last,
}

impl CondVal {
    /// `condValInt_`'s declared initialiser (`dsc/dsc2.h:663`), which is also what the DDL conversion
    /// leaves on a `FIRST`/`LAST` it never parses an integer for
    /// (`ddc/ddl/ddl_conversion.cpp:267`) — so it is the value the JSON seam carries for both forms.
    pub const ABSENT_VAL_INT: i32 = -1;

    /// Which `condValType_` this form is (`dsc/dsc2.h:662`) — the half of the pair the JSON exporter
    /// writes as a spelling (`dsc/dsc2.cpp:464-466`).
    pub fn val_type(self) -> CondValType {
        match self {
            Self::Iteration(_) => CondValType::Int,
            Self::First => CondValType::First,
            Self::Last => CondValType::Last,
        }
    }

    /// The `condValInt_` the JSON exporter writes for this form, unconditionally
    /// (`dsc/dsc2.cpp:468`). A raw `i32` and not an [`IterationIdx`] deliberately: under
    /// `FIRST`/`LAST` the number is [`ABSENT_VAL_INT`](Self::ABSENT_VAL_INT), which is not an
    /// iteration of anything.
    pub fn val_int(self) -> i32 {
        match self {
            Self::Iteration(idx) => idx.0,
            Self::First | Self::Last => Self::ABSENT_VAL_INT,
        }
    }

    /// The JSON importer's two fields joined (`dsc/dsc2.cpp:1444-1449`).
    ///
    /// ⛔ IT DISCARDS `val_int` UNDER `FIRST`/`LAST`, and that is the fusion's whole divergence: a
    /// wire pair carrying anything but [`ABSENT_VAL_INT`](Self::ABSENT_VAL_INT) there does not
    /// round-trip. No producer writes one, and every reader of those two forms already ignores the
    /// integer — BOTH of its arms, not just the first (`SNControlFlowLowering.cpp:99-109`,
    /// `dsc/dsc2Pcfg.cpp:792-799`).
    pub fn from_wire(val_type: CondValType, val_int: i32) -> Self {
        match val_type {
            CondValType::Int => Self::Iteration(IterationIdx(val_int)),
            CondValType::First => Self::First,
            CondValType::Last => Self::Last,
        }
    }
}

/// One term of a condition node's guard: WHICH iteration of one loop dim the guarded region applies
/// to (`dsc/dsc2.h:654-673`).
///
/// ⛔ PARTIAL, AND `e039_LoopCond`/`e030_LoopCond`/`e018_LoopCond` STAY OPEN BELOW: `loopComp_` is
/// the `const LoopNode*` this term is a condition ON, used as pure pointer identity — compared
/// against a loop (`ddc/ddc_transformation_util.cpp:266`, `:555`, `dsc/dsc2.cpp:2071`, `:2128`),
/// inserted into a set (`:329`) and keyed into a map (`dsc/dsc2Pcfg.cpp:746`).
///
/// ⛔ A BORROW CANNOT SERVE NOW THAT THE TREE OWNS ITS NODES, AND A NAME CANNOT EITHER: the
/// compare at `:266` runs inside a `traverseTreeDFSMutable` walk holding `&mut` on the node that
/// holds this very `LoopCond` ([`ConditionNode`], `dsc/dsc2.h:690`), and while `name_` is identity
/// on the JSON seam (`dsc/dsc2.cpp:456`, resolved at `:1445`) that same caller RENAMES its loop
/// twelve lines before the compare (`ddc/ddc_transformation_util.cpp:251-256`) and the uniquifier
/// runs LAST, after all scheduling (`dsc/dsc2.cpp:2986-2992` from `ddc/ddcv1.cpp:3790`).
///
/// ⭐ THE VALUE HALF LANDS ANYWAY BECAUSE TWO READERS NEVER TOUCH THE LOOP: `convertCondValToInt`
/// takes the loop's trip count as a parameter rather than following the pointer
/// (`dsc/dsc2Pcfg.cpp:788-806`), and the reverse-DDL emitter writes a LITERAL `"label"` where the
/// loop's name belongs (`ddc/ddl/ddl_conversion.cpp:3272-3281`).
///
/// ⛔ NO `Default`, unlike the authority's `LoopCond() = default` (`dsc/dsc2.h:672`): that
/// constructor exists for the JSON importer, which overwrites all five fields before the value is
/// used (`dsc/dsc2.cpp:1438-1449`), and its [`CondOp::Default`] operator is a state
/// [`LoopCondOp`] has no variant for.
///
/// ```compile_fail
/// // E0599, for the reader only: stable rustdoc parses the code an annotation names and ignores
/// // it, so the annotation is documentation and the positive control below is the check.
/// use deeptools::schedule::dsc2::LoopCond;
/// let _ = LoopCond::default();
/// ```
///
/// ⭐ AND ITS POSITIVE CONTROL, which rustdoc DOES enforce — the same path, reached the only way a
/// value of this type exists. Without it the `compile_fail` above would pass just as happily on a
/// misspelled module path or an item that stopped being `pub`:
///
/// ```
/// use deeptools::schedule::dims::PrimaryDimTypes;
/// use deeptools::schedule::dsc2::{CondVal, LoopCond, LoopCondOp};
/// let _ = LoopCond {
///     dim: PrimaryDimTypes::Y,
///     cond_op: LoopCondOp::Eq,
///     cond_val: CondVal::Last,
/// };
/// ```
///
/// ⛔ NO `PartialEq` EITHER, and the authority declares none: two terms agreeing on dim, operator and
/// value are the same condition only on the same loop, and that is the field this type is missing.
#[derive(Clone, Copy, Debug)]
pub struct LoopCond {
    /// Field: e039_LoopCond.dim_
    /// Field: e030_LoopCond.dim_
    /// Field: e018_LoopCond.dim_
    ///
    /// Which of `loopComp_`'s dims the term is on (`dsc/dsc2.h:660`) — a loop node carries several,
    /// and bridge 1 resolves the pair to one MLIR loop (`SNControlFlowLowering.cpp:92-94`).
    ///
    /// ⭐ [`PrimaryDimTypes::Undefined`] IS THE AUTHORITY'S OWN INITIALISER, `PrimaryDimTypesCount`
    /// (`dsc/dsc2.h:660`), so no [`Option`] is needed: the live "no dimension" key already spells it.
    pub dim: PrimaryDimTypes,
    /// Field: e039_LoopCond.condOp_
    /// Field: e030_LoopCond.condOp_
    /// Field: e018_LoopCond.condOp_
    ///
    /// How the loop's iteration is compared against [`cond_val`](Self::cond_val)
    /// (`dsc/dsc2.h:661`). See [`LoopCondOp`] for what the narrowing deletes.
    pub cond_op: LoopCondOp,
    /// Field: e039_LoopCond.condValType_
    /// Field: e039_LoopCond.condValInt_
    /// Field: e030_LoopCond.condValType_
    /// Field: e030_LoopCond.condValInt_
    /// Field: e018_LoopCond.condValType_
    /// Field: e018_LoopCond.condValInt_
    ///
    /// What it is compared AGAINST — the authority's two fields as one (`dsc/dsc2.h:662-663`). See
    /// [`CondVal`].
    pub cond_val: CondVal,
}

// crustify:todo: e018_LoopCond

// crustify:todo: e018_LoopCond.loopComp_

/// One conjunction of a condition node's guard — one entry of `twoLevelOrOfAnds_`
/// (`dsc/dsc2.h:676`): the [`LoopCond`] terms that must ALL hold for the guarded region to run.
///
/// ⛔ NON-EMPTY BY SHAPE, BECAUSE AN EMPTY CONJUNCTION IS A SILENTLY DROPPED GUARD OR A CRASH:
/// bridge 1's single-clause path emits no comparison at all for an empty term list, leaves `if_op`
/// DEFAULT-CONSTRUCTED and then builds the "then" region with the UNCHANGED builder — that is,
/// unconditionally (`SNControlFlowLowering.cpp:203-272`, mapped at `:1063`) — while its multi-clause
/// path reads `cmp_list[cmp_list.size() - 1]` with `cmp_list` still empty (`:180`).
///
/// ⭐ AND THE AUTHORITY CHECKS NEITHER: one minter's term vector is non-empty only because its first
/// dim is seeded before the filter that fills it (`ddc/ddc_transformation.cpp:919`, `:1056-1065`),
/// and the SAMV minter's is a walk over enclosing loops that may match no dim at all
/// (`ddc/ddcv1.cpp:3638-3649`). A mandatory first term is the check nobody wrote.
#[derive(Clone, Debug)]
pub struct LoopCondConjunction {
    first: LoopCond,
    rest: Vec<LoopCond>,
}

impl LoopCondConjunction {
    /// The one-term conjunction a fresh condition's clause starts from — FOUR of the SIX producers of
    /// a new clause (`ddc/ddl/ddl_conversion.cpp:317-319`, `dsc/dsc2.cpp:5092-5098`,
    /// `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:4811-4815`, and
    /// `ddc/ddc_transformation.cpp:1056-1065` by way of the `:919` seed above).
    ///
    /// ⛔ THE OTHER TWO EMPLACE THE CLAUSE EMPTY AND FILL IT AFTERWARDS, which is the state above:
    /// the SAMV minter's walk over enclosing loops may match no dim (`ddc/ddcv1.cpp:3638-3649`) and
    /// the JSON importer takes one term per array element (`dsc/dsc2.cpp:1436-1438`). ⛔ AND THAT
    /// SECOND ONE IS NOT A MINT AT ALL — it is the import half of the round trip, so it was never
    /// evidence for how a condition is built.
    pub fn new(term: LoopCond) -> Self {
        Self {
            first: term,
            rest: Vec::new(),
        }
    }

    /// One further term, ANDed on the end — the SAMV minter's inner `emplace_back`
    /// (`ddc/ddcv1.cpp:3643-3644`).
    pub fn and_term(mut self, term: LoopCond) -> Self {
        self.rest.push(term);
        self
    }

    /// ⭐ THE DDL'S `ConditionAndOp`, AND TOTAL WHERE THE AUTHORITY'S IS GUARDED: it appends the
    /// operand's terms to ours (`ddc/ddl/ddl_conversion.cpp:390-392`) behind a four-part check that
    /// each side be a SINGLE, UN-NEGATED clause (`:380-389`) — which is what having this type at all
    /// says, so the check has no input left to reject.
    pub fn and(mut self, other: Self) -> Self {
        self.rest.push(other.first);
        self.rest.extend(other.rest);
        self
    }

    /// Its terms in declaration order, which is OBSERVABLE and therefore not a set: the PCFG
    /// translator spells a condition node's NAME from the dims it walks in this order
    /// (`dsc/dsc2Pcfg.cpp:721-725`).
    pub fn terms(&self) -> impl Iterator<Item = LoopCond> + '_ {
        core::iter::once(self.first).chain(self.rest.iter().copied())
    }

    /// How many terms, never zero.
    pub fn term_count(&self) -> NonZeroUsize {
        NonZeroUsize::MIN.saturating_add(self.rest.len())
    }
}

impl From<LoopCond> for LoopCondConjunction {
    fn from(term: LoopCond) -> Self {
        Self::new(term)
    }
}

/// The whole of `twoLevelOrOfAnds_` (`dsc/dsc2.h:676`): the conjunctions, ANY of which lets the
/// guarded region run.
///
/// ⛔ NON-EMPTY FOR THE SAME REASON ONE LEVEL UP, and here the empty state is not a weaker condition
/// but a DIFFERENT KIND OF NODE: `hasCoreClCond()` IS that emptiness (`dsc/dsc2.h:693-695`), so what
/// e044 carries as an [`Option`] — see [`ConditionNode::has_core_cl_cond`], now filled. ⭐ THAT
/// DELETES A CHECK IN BOTH DSC-TO-DATAFLOW-IR
/// LOWERINGS, each re-asserting `twoLevelOrOfAnds_.empty()` inside the arm the same predicate
/// already selected (`SNControlFlowLowering.cpp:1049`, `:1079`; `DSC2ToDataflowIR.cpp:91`, `:113`).
#[derive(Clone, Debug)]
pub struct LoopCondDisjunction {
    first: LoopCondConjunction,
    rest: Vec<LoopCondConjunction>,
}

impl LoopCondDisjunction {
    /// The one-clause disjunction, which is every minter's whole condition: all five push exactly
    /// one clause.
    pub fn new(clause: LoopCondConjunction) -> Self {
        Self {
            first: clause,
            rest: Vec::new(),
        }
    }

    /// One further clause, ORed on the end — `adjustConditionForSplitLoop`'s outer `push_back`
    /// (`dsc/dsc2.cpp:2134`).
    ///
    /// ⛔ AND NOT ONLY ON THE SPLIT PATH: the chunk-condition minter emplaces one clause per
    /// enclosing loop of the dim, outer to inner (`dsc/dsc2.cpp:5077-5104`), so a MINT alone reaches
    /// two clauses and `:81`'s `size() > 1` disjunct is live with no loop splitting anywhere. Its own
    /// `DT_CHECK` caps that at two and names THIS field as what would have to change to lift the cap
    /// (`:5049-5052`).
    pub fn or_clause(mut self, clause: LoopCondConjunction) -> Self {
        self.rest.push(clause);
        self
    }

    /// ⭐ THE DDL'S `ConditionOrOp`, TOTAL AGAIN: it concatenates the clause lists
    /// (`ddc/ddl/ddl_conversion.cpp:401-403`) behind a check that NEITHER side be negated
    /// (`:393-399`), and a negation cannot reach this level — it lives one up, in
    /// [`LoopCondComposite`].
    pub fn or(mut self, other: Self) -> Self {
        self.rest.push(other.first);
        self.rest.extend(other.rest);
        self
    }

    /// Back down to one conjunction when that is all this is — the narrowing that makes a clause
    /// composable with [`LoopCondConjunction::and`] again, and exactly what the authority's
    /// `size() != 1` half asks (`ddc/ddl/ddl_conversion.cpp:380-382`).
    pub fn into_conjunction(self) -> Option<LoopCondConjunction> {
        if self.rest.is_empty() {
            Some(self.first)
        } else {
            None
        }
    }

    /// Its clauses in declaration order, for the same reason [`LoopCondConjunction::terms`] is
    /// ordered.
    pub fn clauses(&self) -> impl Iterator<Item = &LoopCondConjunction> + '_ {
        core::iter::once(&self.first).chain(self.rest.iter())
    }

    /// How many clauses, never zero.
    ///
    /// ⛔ NOT BRIDGE 1'S PATH SELECTOR, THOUGH IT IS ONE THIRD OF ONE: `:81` is
    /// `twoLevelOrOfAnds_.size() > 1 || cond.negated_ || has_else_branch`
    /// (`SNControlFlowLowering.cpp:81`), and `has_else_branch` is only `children.size() == 2`
    /// (`:1052`) — so a SINGLE un-negated clause takes the multi-clause path whenever the condition
    /// node has an else region, and a lowering dispatched on this count alone would send those to the
    /// single-clause body (`:203-272`), which creates its `scf.if` with no else region at all
    /// (`:258-259`, `:262-263`).
    pub fn clause_count(&self) -> NonZeroUsize {
        NonZeroUsize::MIN.saturating_add(self.rest.len())
    }
}

impl From<LoopCondConjunction> for LoopCondDisjunction {
    fn from(clause: LoopCondConjunction) -> Self {
        Self::new(clause)
    }
}

/// A condition node's loop guard: `LoopCondComposite` (`dsc/dsc2.h:675-683`), an OR of ANDs under
/// one overall negation — which is the shape the DDL's own diagnostic names, "two-level OR of ANDs,
/// with optionally an overall negation" (`ddc/ddl/ddl_conversion.cpp:385-388`).
///
/// ⭐ THREE GRAMMAR LEVELS, THREE TYPES, AND THAT DELETES BOTH OF THAT DIAGNOSTIC'S REFUSALS: the
/// authority keeps all three in one flat struct, so `ConditionAndOp` must re-check at run time that
/// each operand is a single un-negated clause (`:380-389`) and `ConditionOrOp` that neither operand
/// is negated (`:393-399`). [`LoopCondConjunction::and`] and [`LoopCondDisjunction::or`] take
/// operands of the level they compose, so both checks lose their inputs. The two narrowings a
/// doubly negated operand needs to compose again — [`without_negation`](Self::without_negation) and
/// [`LoopCondDisjunction::into_conjunction`] — are those same two questions asked as PROJECTIONS,
/// and they answer a value where the authority aborts the compiler.
///
/// ⛔ AND NEGATED-WITH-NO-CONDITION IS A CRASH, WHICH IS WHY THE FLAG SITS AT THIS LEVEL: bridge 1
/// enters its multi-clause path on `negated_` ALONE (`SNControlFlowLowering.cpp:81`) and then reads
/// `cmp_list[cmp_list.size() - 1]` on an empty `cmp_list` (`:180`). Both writers already guard the
/// toggle on exactly the emptiness this type cannot spell — explicitly
/// (`ddc/ddl/ddl_conversion.cpp:325-326`, where a `condNot` over an empty composite negates
/// `coreClCond_` instead) and through `hasCoreClCond()`'s else arm
/// (`ddc/ddc_transformation_util.cpp:489`, `:592`).
///
/// ⛔ PARTIAL, AND THE `e022`/`e043` ANCHORS STAY OPEN BELOW: every term is [`LoopCond`]'s value
/// half, so a composite still cannot name the loops it is a condition ON, and
/// `adjustConditionForSplitLoop` selects and rebuilds its terms by that pointer
/// (`dsc/dsc2.cpp:2071-2076`, `:2126-2132`).
///
/// ⚠️ AND THAT DISPATCH HAS NO FIFTH REFUSAL: it has the FOUR the e044 anchor lists, and that
/// anchor's fourth IS `:2136` — "anything outside `EQ` / `NE` / `(GT,FIRST)` / `(LT,LAST)`". What is
/// worth recording is WHICH pairs reach it, because there are only two: over the six operators a
/// ported condition can spell times `FIRST`/`LAST`, `:2087-2101` takes the four always-true/false
/// pairings and `:2136` is left with `(LE, FIRST)` and `(GE, LAST)` alone. ⭐ BOTH ARE EQUALITIES on
/// an index that cannot leave its own bounds, so each belongs in the ANDed-term arm `:2109-2113` —
/// NOT in the new-OR-clause arm `:2114-2134`, which yields the other shape entirely.
///
/// ⛔ TWO CARRIERS, and both must reach this type: `ConditionNode::loopCond_` (`dsc/dsc2.h:690`) and
/// `DdlInterface::CondProp::loopCond_` (`ddc/ddl/ddl_conversion.h:420-421`).
///
/// ⛔ NO `Default`, unlike the authority's aggregate: `myCp.loopCond_ = {}` is how the DDL DROPS a
/// condition that resolved to a constant (`ddc/ddl/ddl_conversion.cpp:353`, `:359`), and what it
/// leaves behind is the discriminator, not a guard that holds trivially.
///
/// ```compile_fail
/// // E0599, for the reader only: stable rustdoc parses the code an annotation names and ignores
/// // it, so the annotation is documentation and the positive control below is the check.
/// use deeptools::schedule::dsc2::LoopCondComposite;
/// let _ = LoopCondComposite::default();
/// ```
///
/// ⭐ AND ITS POSITIVE CONTROL, which rustdoc DOES enforce — the one-clause guard every minter
/// builds, reached the only way a value of this type exists:
///
/// ```
/// use deeptools::schedule::dims::PrimaryDimTypes;
/// use deeptools::schedule::dsc2::{
///     CondVal, LoopCond, LoopCondComposite, LoopCondConjunction, LoopCondOp,
/// };
/// let term = LoopCond {
///     dim: PrimaryDimTypes::Y,
///     cond_op: LoopCondOp::Eq,
///     cond_val: CondVal::Last,
/// };
/// let cond: LoopCondComposite = LoopCondConjunction::new(term).into();
/// assert!(!cond.negated);
/// assert_eq!(cond.or_of_ands.clause_count().get(), 1);
/// ```
///
/// ⛔ NO `PartialEq`, for [`LoopCond`]'s reason: two guards agreeing on dims, operators and values
/// are the same guard only on the same loops, and that is the field the terms are missing.
#[derive(Clone, Debug)]
pub struct LoopCondComposite {
    /// Field: e043_LoopCondComposite.twoLevelOrOfAnds_
    /// Field: e022_LoopCondComposite.twoLevelOrOfAnds_
    ///
    /// The OR of ANDs itself (`dsc/dsc2.h:676`) — non-empty, per [`LoopCondDisjunction`].
    pub or_of_ands: LoopCondDisjunction,
    /// Field: e043_LoopCondComposite.negated_
    /// Field: e022_LoopCondComposite.negated_
    ///
    /// Whether the whole disjunction is inverted (`dsc/dsc2.h:677`).
    ///
    /// ⛔ A PARITY, NEVER A SET: both writers TOGGLE it (`ddc/ddc_transformation_util.cpp:592`, where
    /// the flip is what makes a cloned condition guard the ELSE region, and
    /// `ddc/ddl/ddl_conversion.cpp:326`), so compose through [`negate`](Self::negate) — two flips
    /// must be none. ⚠️ AND ON THE WIRE IT IS AN INTEGER, not a JSON bool: the exporter writes `0`
    /// or `1` through `operator<<` with no `boolalpha` anywhere (`dsc/dsc2.cpp:483`) and the importer
    /// reads `int_value()` (`:1462`), which json11 answers `0` for on a JSON `true`
    /// (`external/json11/json11.cpp:193-196`, `:283`) — so a hand-written `true` imports as FALSE
    /// and inverts the guard.
    pub negated: bool,
}

impl LoopCondComposite {
    /// The `condNot` toggle (`ddc/ddl/ddl_conversion.cpp:326`,
    /// `ddc/ddc_transformation_util.cpp:592`).
    pub fn negate(mut self) -> Self {
        self.negated = !self.negated;
        self
    }

    /// The disjunction back out, when the negation is off — the other half of what the authority's
    /// composition checks ask (`ddc/ddl/ddl_conversion.cpp:381`, `:394`), which read the FLAG and
    /// not the history of it, so a doubly negated operand composes again.
    pub fn without_negation(self) -> Option<LoopCondDisjunction> {
        if self.negated {
            None
        } else {
            Some(self.or_of_ands)
        }
    }
}

impl From<LoopCondDisjunction> for LoopCondComposite {
    fn from(or_of_ands: LoopCondDisjunction) -> Self {
        Self {
            or_of_ands,
            negated: false,
        }
    }
}

impl From<LoopCondConjunction> for LoopCondComposite {
    fn from(clause: LoopCondConjunction) -> Self {
        LoopCondDisjunction::new(clause).into()
    }
}

/// One constant element offset added on top of a start address — the `int` of `constEleOffsets_`
/// (`dsc/dsc2.h:727-729`).
///
/// ⛔ ELEMENTS ALONG ONE DIM, NOT BYTES AND NOT A STICK COUNT, which is what the field's own comment
/// says ("constant offset on top of the start address") and what every writer produces: a padding
/// front size (`ddc/ddcv1.cpp:2655-2656`), a padded dim size less its back pad (`:2670-2673`), a
/// row-split offset the L3 path stores under `metadata.rowSplitDim` (`:2887`, `:2905`) and a
/// `factor * size` product (`:3076`). ⭐ IT IS THEREFORE NOT [`BufferAddrOffset`]'s currency: bridge 1
/// scales this one by the operand's element size and adds that one to an address as it stands.
///
/// ⛔ SIGNED, AND THE NEGATIVES ARE THE ERROR PATH, NOT A SENTINEL: `:2657-2660` and `:2674-2677`
/// `DT_ERROR` on a negative pad AFTER storing it, so a port that made this unsigned would move the
/// refusal earlier than IBM's and lose the loop name it reports.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ConstEleOffset(pub i32);

/// The address step from one buffer to the next, per core and corelet — `bufferAddrOffset_`
/// (`dsc/dsc2.h:735-737`).
///
/// ⛔ NOT [`BufferOffset`], THOUGH IT IS COPIED FROM ONE: both writers take
/// `allocation->bufferOffsetCoreCorelet_` and then DIVIDE every entry by the unit's
/// `addressGranularityScalePerUnit` (`ddc/ddcv1.cpp:2801-2806`,
/// `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:6267-6272`), so [`BufferOffset`]'s bytes and this are
/// two currencies and assigning one to the other is `E0308`. ⭐ THE PROOF THAT THIS IS THE SAME
/// CURRENCY AS `startAddr_` is that bridge 1 ADDS THEM: `bufferAddrOffset_.at(core).at(corelet)
/// + 2 * getSingleDataStrict(startAddr_, ...)`
/// (`dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:1174-1180`).
///
/// ⛔ IT SURVIVES THE JSON ROUND TRIP AS A QUOTED STRING, not a number: the export writes
/// `QUOTE(std::to_string(...))` (`dsc/dsc2.cpp:279-280`) and the import reads `std::stoll` of a
/// `string_value()` (`:1267-1268`) — because the value is 64-bit and the JSON reader's numbers are
/// not.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BufferAddrOffset(pub i64);

/// What a [`DataInfo`] IS: an entry in `DesignSpaceConfig::labeledDs_` or one in its
/// `constantInfo_`. IBM spells the pair as two `int` fields, `myLdsIdx_` and `constantId_`
/// (`dsc/dsc2.h:722`, `:726`), and the name is `getLdsOrConstNameOfAllocNode`'s
/// (`ddc/ddcv1.cpp:20-29`).
///
/// ⛔ THIS EXISTS TO MAKE THE AUTHORITY'S OWN `DT_CHECK` UNSPELLABLE. Both `isLabeledDs` and
/// `isConstant` open with `DT_CHECK_MSG(!(myLdsIdx_ >= 0 && constantId_ >= 0), "Cannot be both
/// labeledDs and constant.")` (`dsc/dsc2.h:741-750`) — the state it refuses is one this enum has no
/// variant for, and the one rewrite that moves a `DataInfo` from the first to the second sets both
/// halves in adjacent statements (`dsc/dsc2.cpp:5379-5380`), which here is ONE assignment and so
/// cannot be half-done.
///
/// ⛔ [`AllocateNode`] DELIBERATELY DOES NOT MERGE ITS OWN PAIR, and that is not an inconsistency:
/// `getLdsOrConstNameOfAllocNode` tests `tempStorageForCompute_` FIRST and only then `ldsIdx_` and
/// `constIdx_` (`ddc/ddcv1.cpp:20-29`), so an allocation has three states, no `DT_CHECK` holds its
/// two indices apart, and merging them there would delete a reachable combination.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LdsOrConst {
    /// `myLdsIdx_ >= 0` — an index into `DesignSpaceConfig::labeledDs_`, which is what
    /// `getComputeOperandFormats` and `parametricStride` look the operand's format and stride up
    /// with (`dsc/dsc2.cpp:2348-2357`, `:4203`).
    LabeledDs(LdsIdx),
    /// `constantId_ >= 0` — an index into `DesignSpaceConfig::constantInfo_`, taken when the
    /// transfer's source is zero padding rather than a tensor (`dsc/dsc2.cpp:5375-5380`).
    Constant(ConstantId),
}

/// Replaces: e033_DataInfo
///
/// `dsc/dsc2.h:721-753`. WHICH data one operand of a node refers to and HOW its address is formed —
/// the struct [`TransferNode`] holds four of and [`ComputeNode`] a vector of per operand
/// (`dsc/dsc2.h:832-833`, `:937-938`). `fillLoopOffsetsAndAddresses` is what fills the address half,
/// walking the enclosing loops of the allocation that backs the operand (`ddc/ddcv1.cpp:2360-2807`,
/// and again for stage 2a at `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:5780-6272`).
///
/// ⛔ SEVEN OF TEN DECLARED FIELDS ARE HERE. The three that are not, with the reason:
///  * `startAddr_` (`:723`) is a `FoldManager<int64_t>` — e026_FoldManager, which HAS now landed in
///    `src/schedule/fold.rs`, so this one's stated blocker is gone and the field is schedulable;
///  * `loopEleOffsets_` (`:730-734`) is keyed by `const LoopNode*` — POINTER IDENTITY, needing
///    e029_ScheduleNode's `name_`, which is also the only thing the JSON export can order those keys
///    by (`dsc/dsc2.cpp:251-256`). Its value currency is already here as [`TemporalStride`];
///  * `bufferSwitchPosition_` (`:738`) is a `const LoopNode*` for the same reason.
///
/// ⛔ AND THOSE TWO POINTERS AND `bufferAddrOffset_` ARE ONE STATE, WRITTEN IN ONE BLOCK PER STAGE.
/// Both stages write the three together under `allocation->numBuffers_ != 1` — stage 2b at
/// `ddc/ddcv1.cpp:2796-2807`, stage 2a at `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:6262-6274` — so
/// an empty [`buffer_addr_offset`](Self::buffer_addr_offset) and an absent buffer-switch loop mean the
/// same thing, single-buffered, and a reader that finds one without the other is looking at a
/// half-filled node. ⛔ IT IS NOT THE SOURCE MAP THAT IS EMPTY: `bufferOffsetCoreCorelet_` is filled
/// for single-buffered allocations too, under a `DT_CHECK` that admits `numBuffers_ == 1` outright
/// (`L3DlOpsScheduler.cpp:4966-4967`, written at `:4991`, `:5008`, `:5035`), so the emptiness here is
/// made by the guard on the COPY and by nothing else.
///
/// ⛔ THE TWO STAGES REACH THAT OFFSET BY DIFFERENT ROUTES AND DIVERGE ON THE START ADDRESS. Stage 2b
/// folds the L0LU row count into `addrScale` before anything uses it (`ddc/ddcv1.cpp:2392-2394`) and
/// divides the buffer offset by the product (`:2804`); stage 2a takes `addrScale` without it
/// (`L3DlOpsScheduler.cpp:5850-5852`) and divides by `numPTRows` separately, inside the block
/// (`:6271`). Truncating division makes those two equal — but `startAddr_` is scaled by that same
/// `addrScale` in both (`ddc/ddcv1.cpp:2398-2406`, `L3DlOpsScheduler.cpp:5854-5862`), so on L0LU
/// stage 2b divides the start address by `numPTRows` and stage 2a does not. That is this field's own
/// to carry, and it is why one citation per stage is the minimum here.
///
/// ⛔ AND WHICH ADDRESS FIELDS GET FILLED AT ALL IS A THREE-WAY MATCH ON
/// [`lds_or_const`](Self::lds_or_const), the same in both stages. [`None`] fills NOTHING — the filler
/// returns before even the start address (`ddc/ddcv1.cpp:2364`, `L3DlOpsScheduler.cpp:5786`), as it
/// also does for a non-memory storage (`ddcv1.cpp:2365`). [`LdsOrConst::Constant`] gets the start
/// address and then returns, `// no offsets for constants` (`ddcv1.cpp:2410`,
/// `L3DlOpsScheduler.cpp:5991`), leaving `constEleOffsets_`, `loopEleOffsets_`, `bufferAddrOffset_`
/// and `bufferSwitchPosition_` empty — which is exactly what the rewrite that turns a tensor source
/// into a constant one asserts next, four `DT_CHECK`s in a row (`dsc/dsc2.cpp:5386-5397`). Only
/// [`LdsOrConst::LabeledDs`] reaches the offsets. ⛔ THIS TYPE DOES NOT ENFORCE THAT: a `Constant`
/// holding a non-empty [`const_ele_offsets`](Self::const_ele_offsets) is representable here, and
/// `e033_a_constant_operand_carries_no_offsets_and_this_type_cannot_refuse_one` pins it.
///
/// ⛔ NO `PartialEq`, as no node type here has one: the authority declares no `operator==` for
/// `DataInfo` and its consumers compare the parts they care about, `dataConnect_` above all
/// (`ddc/ddc_fold.cpp:435`, `:1794`, `ddc/ddc_transformation_util.cpp:891`, `:997`).
#[derive(Clone, Debug)]
pub struct DataInfo {
    /// Field: e033_DataInfo.myLdsIdx_
    ///
    /// Field: e033_DataInfo.constantId_
    ///
    /// The two `-1`-initialised indices of `dsc/dsc2.h:722` and `:726` as the one thing they encode.
    /// [`None`] is IBM's `-1`/`-1`: a `DataInfo` that is neither, which is what
    /// [`is_labeled_ds`](Self::is_labeled_ds) and [`is_constant`](Self::is_constant) both answer
    /// `false` for. See [`LdsOrConst`] for why they are one field.
    pub lds_or_const: Option<LdsOrConst>,
    /// Field: e033_DataInfo.isStartAddrSymbolic_
    ///
    /// Whether `startAddr_` is a symbol to be resolved rather than a number (`dsc/dsc2.h:724`),
    /// copied from the backing allocation (`ddc/ddcv1.cpp:2397`,
    /// `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:5817`) and read immediately after to branch on
    /// (`ddc/ddcv1.cpp:2399`, `L3DlOpsScheduler.cpp:5855`).
    ///
    /// ⛔ ONE OF THOSE BRANCHES IS A REFUSAL, NOT A PATH. Stage 2a `DT_ERROR`s "Currently no support;
    /// work in progress" when a symbolic start address meets cross-core reduction with a corelet split
    /// (`L3DlOpsScheduler.cpp:5833-5834`). Stage 2b has no such gap (`ddc/ddcv1.cpp:2399-2405`), so the
    /// flag means "resolve a symbol" in one stage and "abort the compile" in the other.
    pub is_start_addr_symbolic: bool,
    /// Field: e033_DataInfo.latchDataId_
    ///
    /// The LATCH link between a producer and a consumer (`dsc/dsc2.h:725`). [`None`] is the field's
    /// `-1` — "not latched" — exactly as [`LatchDataId`]'s own doc states, and the counter that
    /// mints these only counts up (`ddc/ddc_transformation_util.cpp:967`).
    ///
    /// ⛔ THE `Option` IS WHERE FIVE `DT_CHECK`S GO. `latch_id != -1` is asserted at
    /// `dsc-based-utils/DSC2ToDataflowIR/V3/SNComputeLowering.cpp:724`, `:903` and
    /// `SNTransferLowering.cpp:957`, `:1660`, `:2154`, all with "latch id cannot be negative" — a
    /// bridge that has to unwrap this cannot reach any of them.
    ///
    /// ⭐ ONE REWRITE WRITES ALL THREE HALVES OF A LATCH: `currLatchDataId` goes onto the producing
    /// transfer's destination, the consuming transfer's source and every consuming compute's input in
    /// one walk (`ddc/ddc_transformation_util.cpp:965-1010`).
    pub latch_data_id: Option<LatchDataId>,
    /// Field: e033_DataInfo.constEleOffsets_
    ///
    /// A constant element offset per core, corelet and dim (`dsc/dsc2.h:727-729`).
    ///
    /// ⚠️ THE SCHEDULER LISTED NO ANCHOR FOR IT, nor for `startAddr_`, `latchDataId_`, `constantId_`,
    /// `loopEleOffsets_` or `bufferAddrOffset_` — and the cause is a TRAILING COMMENT, not the wrapped
    /// declaration. `plan.py`'s field pattern is anchored at `;\s*$`, so a declaration line ending in
    /// `// …` never matches, and it scans EVERY line of the class body, which is why wrapping the type
    /// onto its own line costs nothing. The six names it missed are exactly the six `DataInfo`
    /// declarations carrying a trailing comment (`dsc/dsc2.h:723`, `:725`, `:726`, `:728`, `:732`,
    /// `:736`); the four it anchored — `myLdsIdx_`, `isStartAddrSymbolic_`, `bufferSwitchPosition_`,
    /// `dataConnect_` — are exactly the four carrying none (`:722`, `:724`, `:738`, `:739`), and
    /// `bufferSwitchPosition_` wraps no more and no less than `bufferAddrOffset_` does.
    ///
    /// ⛔ EMPTY IS A DECISION, NOT AN ABSENCE OF ONE. `fillLoopOffsetsAndAddresses` remembers whether
    /// any offset existed before it starts broadcasting them across cores and corelets and CLEARS the
    /// whole map again if none did (`ddc/ddcv1.cpp:2689`, `:2793`), and four readers test
    /// `constEleOffsets_.empty()` to decide whether to write their own (`:2883`, `:2897`, `:2968`,
    /// `:3001`).
    ///
    /// ⛔ THE INNERMOST MAP IS `std::unordered_map` IN C++ AND ITS ITERATION ORDER REACHES IBM'S
    /// EXPORTED JSON (`dsc/dsc2.cpp:237-241`), so a [`BTreeMap`] here spells the dims in
    /// [`PrimaryDimTypes`] declaration order where the reference spells them in libstdc++ hash order.
    /// That is a text divergence in the exported node and it is deliberate: the outer two maps ARE
    /// `std::map` and ordered, and one live reader takes `constEleOffsets_.begin()->second.begin()
    /// ->second.begin()->second` (`ddc/ddl/ddl_conversion.cpp:3105-3111`, `:3167-3174`), which is
    /// unordered in the authority and lowest-dim-first here.
    pub const_ele_offsets:
        BTreeMap<CoreId, BTreeMap<CoreletId, BTreeMap<PrimaryDimTypes, ConstEleOffset>>>,
    /// Field: e033_DataInfo.bufferAddrOffset_
    ///
    /// The step to the next buffer, per core and corelet (`dsc/dsc2.h:735-737`). Ordered for the
    /// same reason [`AllocateNode::buffer_offset_core_corelet`] is: both keys are `int` in the
    /// authority, both are exported in key order (`dsc/dsc2.cpp:273-286`), and no writer uses a
    /// negative pseudo-key. See [`BufferAddrOffset`]: it is NOT the allocation's byte stride.
    pub buffer_addr_offset: BTreeMap<CoreId, BTreeMap<CoreletId, BufferAddrOffset>>,
    /// Field: e033_DataInfo.dataConnect_
    ///
    /// The DDL wire name of this operand (`dsc/dsc2.h:739`) — the key under which the DDC indexes
    /// producers and consumers (`ddc/ddcv1.cpp:3293-3308`) and the name it reports when it cannot
    /// place an allocation (`:2424`, `:2627`).
    ///
    /// ⛔ NOT STABLE AND NOT UNIQUE: cloning a node for reuse APPENDS a suffix to it in place
    /// (`ddc/ddc_transformation_util.cpp:1593-1594`), and the empty string is the ordinary "no data
    /// connect" state that same line tests for — which is why this is a [`DataConnect`] and not an
    /// [`Option`] of one.
    pub data_connect: DataConnect,
}

impl Default for DataInfo {
    /// The authority's member initialisers (`dsc/dsc2.h:722-739`). ⛔ NOT `#[derive(Default)]`:
    /// [`DataConnect`] wraps a [`String`] and deliberately derives no [`Default`], because a data
    /// connect is a name the DDL authored and only a default-constructed node has none.
    fn default() -> Self {
        Self {
            lds_or_const: None,
            is_start_addr_symbolic: false,
            latch_data_id: None,
            const_ele_offsets: BTreeMap::new(),
            buffer_addr_offset: BTreeMap::new(),
            data_connect: DataConnect(String::new()),
        }
    }
}

impl DataInfo {
    /// `dsc/dsc2.h:741-745`. Whether this operand is an entry of `DesignSpaceConfig::labeledDs_`.
    ///
    /// ⭐ THE `DT_CHECK` THE AUTHORITY OPENS WITH IS DISCHARGED BY THE TYPE, so this is a total
    /// function where IBM's throws — see [`LdsOrConst`].
    pub fn is_labeled_ds(&self) -> bool {
        matches!(self.lds_or_const, Some(LdsOrConst::LabeledDs(_)))
    }

    /// `dsc/dsc2.h:746-750`. Whether this operand is an entry of `DesignSpaceConfig::constantInfo_`,
    /// on the same terms as [`is_labeled_ds`](Self::is_labeled_ds).
    pub fn is_constant(&self) -> bool {
        matches!(self.lds_or_const, Some(LdsOrConst::Constant(_)))
    }
}

// crustify:todo: e020_DistributionStatusInfo

/// Replaces: LoopDistributionInfo::LoopDistributionCat
///
/// `dsc/dsc2.h:1138`. Where one entry of an enclosing-loop chain sits relative to the chunk
/// boundary, which is what decides whether that loop's fold is distributed over the element
/// arrangement.
///
/// ⛔ THE TAG IS PER-WRITER, AND ONLY TWO OF THE EIGHT WRITERS LOOK ANYTHING UP. Those two are
/// `getEnclosingLoopsAndRelatedDims`'s lambda, which takes `BELOW_CHUNK` when the loop is in the
/// caller's `loopsBelowChunkBoundary` and `ABOVE_CHUNK` otherwise (`dsc/dsc2.cpp:6598-6604`), and
/// `ddc/ddc_fold.cpp:3513-3517`, which writes that ternary again. FIVE state a constant instead:
/// `ddc/ddc_fold.cpp:3549-3551` tags every loop it re-distributes a transfer size over `BELOW_CHUNK`
/// with NO membership test, and the L3 scheduler's four sites all write `ABOVE_CHUNK`
/// (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:7316-7318`, `:7325-7327`, `:7586-7588`,
/// `:7608-7610`). The eighth is `CORELET_SLICE` below, whose own set test (`dsc/dsc2.cpp:6616-6617`)
/// gates whether the entry EXISTS, not which tag it gets.
///
/// ⛔ SO A PORT MUST NOT DERIVE THIS FIELD FROM THE SET — that would be right for two writers and
/// wrong for six. Nor is the set chain-relative: `loopsBelowChunkBoundary` is one `Ddc` member
/// (`ddc/ddc.h:108`) cleared and refilled ONCE PER DSC from the DFS LOOP subtree of the block named
/// `"lx_below_schedule"` (`ddc/ddcv1.cpp:3683-3689`, block found at `:2343-2345`, called at `:3783`
/// ahead of both consumers at `:3786-3787`), and all five DDC chain builders pass that same member
/// (`ddc/ddc_fold.cpp:539-540`, `:2255-2256`, `:2476-2477`, `:2482-2483`, `:3122-3123`). Under those
/// two writers the tag is a stable property of where the loop sits in the tree.
///
/// ⛔ AND `BELOW_CHUNK` IS UNREACHABLE ON THE L3 HALF — stage 2a of this campaign. That half's own
/// `loopsBelowChunkBoundary` is a local declared EMPTY with its filler commented out and the comment
/// "there are no loops below chunk boundary in ALxS"
/// (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:5765-5781`), and its four writers name `ABOVE_CHUNK`
/// unconditionally, so every L3 chain entry means `coreletId = -1` at the reader below.
///
/// ⛔ WHAT THE TAG SELECTS IS A CORELET ID, and that reader is why the field exists: `ABOVE_CHUNK`
/// → -1, `BELOW_CHUNK` → `targetCoreletId`, `CORELET_SLICE` → 0, and a fourth value is
/// `DT_ERROR("Unsupported loop distribution category for loop " + loopNode->name_)`
/// (`dsc/dsc2.cpp:6050-6061`). ⭐ THIS CLOSED ENUM IS WHAT MAKES THAT `else` ARM UNWRITABLE — the
/// crate's no-runtime-refusal rule is discharged by the type, not by a check. The remaining readers
/// branch `CORELET_SLICE` against the rest (`:6028-6031`, `:6062-6065`, `:6067`, `:6093-6099`,
/// `:6340`, `ddc/ddc_fold.cpp:2546-2547`, `:3221-3228`).
///
/// ⛔ `CORELET_SLICE` IS A SYNTHETIC CHAIN ENTRY, NOT A REAL LOOP'S TAG — but it carries a real
/// loop's pointer. Its one writer inserts it while climbing from the node's owner loop, at the first
/// loop NOT below the chunk boundary and BEFORE that loop's own per-dim entries
/// (`dsc/dsc2.cpp:6612-6627`); the chain runs innermost-first, so the entry sits inside the innermost
/// chunk loop, which is what IBM's own comment says (`dsc/dsc2.h:1215-1217`). Its dim is the corelet
/// split dim and is NEVER the `PrimaryDimTypesCount` seed of `:6607`: both callers that pass
/// `includeCoreletSplit` require a non-empty `coreletSplit_` on data stage 0 first
/// (`ddc/ddc_fold.cpp:2478-2483`, `:3119-3123` over `ddc/ddcv1.cpp:3673-3680`, whose
/// `metadata.core_dstgid` is the same constant 0 as `dataStageCoreIdx` — `ddc/ddc_metadata.h:211`,
/// `dsc/dsc2.cpp:18`). Its kind is `Unpadded`, supplied by IBM's non-`explicit` constructor rather
/// than named at the callsite (`dsc/dims.h:79-81`), and that matters: a `WindowDim` entry would also
/// match a DIFFERENT dim through the padding window of `loop->denId_` (`dsc/dsc2.cpp:6554-6559`).
/// Its reader takes `loopNode->denId_` for BOTH halves of the size data stage, clears
/// `coreletSplit_` on the numerator copy, counts `numCoreletsUsed_DSC2_` iterations and files the
/// result under `nullptr` (`dsc/dsc2.cpp:6093-6099`, `ddc/ddc_fold.cpp:3223-3228`).
///
/// ⛔ `UNKNOWN` IS UNREACHABLE BY CONSTRUCTION, so the `cat = UNKNOWN` initialiser at
/// `dsc/dsc2.h:1144` is dead: the only constructor takes the category (`:1139-1141`) and, being
/// user-declared, suppresses the implicit default constructor; none of the eight writers names
/// `UNKNOWN`; and `cat` is never assigned after construction — the only reads are
/// `dsc/dsc2.cpp:6027`, `ddc/ddc_fold.cpp:2546` and the structured binding at `:3219`. (⛔ NOT
/// because the name is rare: `UNKNOWN` occurs 54 times across 27 files tree-wide, twice in this enum
/// alone.) That is why this enum derives no `Default` even though the authority writes one, exactly
/// as [`Size`] does not:
///
/// ```compile_fail
/// use deeptools::schedule::dsc2::LoopDistributionCat;
/// let _ = LoopDistributionCat::default();
/// ```
///
/// It is still ported as an enumerator, because it holds the discriminant the other three sit
/// behind. ⭐ AND NO E0080 ORDER GUARD BELOW, unlike [`CondOp`]'s: nothing serializes, parses,
/// iterates or relationally compares a category, so the only load-bearing discriminant is
/// `UNKNOWN`'s zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LoopDistributionCat {
    Unknown = 0,
    AboveChunk = 1,
    BelowChunk = 2,
    CoreletSlice = 3,
}

impl LoopDistributionCat {
    /// Every category in the authority's declaration order (`dsc/dsc2.h:1138`).
    pub const ALL: [Self; 4] = [
        Self::Unknown,
        Self::AboveChunk,
        Self::BelowChunk,
        Self::CoreletSlice,
    ];

    /// The spelling `LoopDistributionInfo::print` gives this category (`dsc/dsc2.h:1151-1167`).
    ///
    /// ⛔ THERE IS NO MAP BEHIND THIS ONE, unlike [`NodeType::name`]: the spellings live only in
    /// that `switch`, nothing parses them back, and `UNKNOWN`'s is the authority's misspelling
    /// `"Unknwon"` (`:1153`) — reproduced verbatim, because correcting a debug spelling diverges the
    /// reference's own log output for no gain.
    ///
    /// ⭐ TOTAL WHERE THE `switch` NEEDED A FALLBACK: its `default: "ERROR"` arm (`:1164-1166`) has
    /// no reachable input over the four enumerators, so this match carries no error case.
    pub fn name(self) -> &'static str {
        match self {
            Self::Unknown => "Unknwon",
            Self::AboveChunk => "Above_chunk",
            Self::BelowChunk => "Below_chunk",
            Self::CoreletSlice => "Corelet_slice",
        }
    }
}

// crustify:todo: e021_LoopDistributionInfo

// crustify:todo: e022_LoopCondComposite

/// Replaces: dsc2::memories
///
/// The components that ARE memories — declared `dsc/dscdefn.h:518`, filled `dsc/dscdefn.cpp:142-144`.
///
/// ⛔ THE TEST IS ALWAYS ON THE `storage` HALF OF A [`DataLocation`] AND NEVER THE `unit`: all four
/// of [`TransferNode`]'s memory questions are `memories.count(x.storage_) == 0`
/// (`dsc/dsc2.cpp:4364-4383`). Both halves hold the same enum, so only the callsite says which.
///
/// ⛔ `nonCoreletMemories` (five components) AND `directAddressableMemories` (seven) ARE DIFFERENT,
/// SMALLER SETS declared beside it (`dsc/dscdefn.h:519-520`, filled `dsc/dscdefn.cpp:145-146` and
/// `:149-150`), AND BOTH ARE READ BY IN-SCOPE UNITS — not, as recorded before, by none:
/// `e307_exploreAssignDataStages` reads `directAddressableMemories` (`ddc/ddcv1.cpp:677`) and
/// `e260_fillLoopOffsetsAndAddresses` reads `nonCoreletMemories` (`:2415`). Each gets its own
/// constant when its unit lands. Neither may be substituted for this one, and this one may not be
/// substituted for either.
///
/// ⛔ NOT `ddc::memories`, WHICH IS A DIFFERENT AND SMALLER SET — eight components, without
/// `L0_SCALE`, `LRFREG`, `L3LUIBR`, `L3SUIBR`, `PESTATE`, `SFPSTATE`, `LXLUSCALEREG` or `QGI`
/// (`ddc/ddc_metadata.h:20-21`). `dsc/dsc2.cpp` includes `dscdefn.h` and not `ddc_metadata.h`, so
/// the `memories.count(input)` in `getComputeOperandSizes` (`dsc/dsc2.cpp:2315`) is THIS set.
/// ⭐ Only membership is ever read, so the `std::set`'s ordering is not observable.
pub const MEMORIES: [SenComponent; 16] = [
    SenComponent::Lx,
    SenComponent::L0,
    SenComponent::L0Scale,
    SenComponent::Lrfreg,
    SenComponent::Pelrf,
    SenComponent::Sfplrf,
    SenComponent::Ptarf,
    SenComponent::Ptxrf,
    SenComponent::Ptirf,
    SenComponent::Hbm,
    SenComponent::L3luibr,
    SenComponent::L3suibr,
    SenComponent::Pestate,
    SenComponent::Sfpstate,
    SenComponent::Lxluscalereg,
    SenComponent::Qgi,
];

/// ⛔ E0080 IF A COMPONENT IS EVER LISTED TWICE: IBM's is a `std::set`, which collapses a duplicate
/// and keeps its `size()` honest; this array would keep both and make its `len()` a lie.
const _: () = {
    let mut i = 0;
    while i < MEMORIES.len() {
        let mut j = i + 1;
        while j < MEMORIES.len() {
            assert!(
                MEMORIES[i] as i32 != MEMORIES[j] as i32,
                "MEMORIES lists one component twice"
            );
            j += 1;
        }
        i += 1;
    }
};

/// Replaces: TransferNode::DstVia
///
/// One destination of a transfer: where the data lands, the location that addresses it indirectly,
/// and the units it routes through on the way (`dsc/dsc2.h:816-819`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DstVia {
    /// Field: e034_TransferNode.loc_
    ///
    /// Where this destination writes (`dsc/dsc2.h:817`). Its `storage` is what decides whether the
    /// destination is a memory at all (`dsc/dsc2.cpp:4372-4383`).
    pub loc: DataLocation,
    /// Field: e034_TransferNode.locIndirect_
    ///
    /// The location holding the address when this destination is indirect, or
    /// [`DataLocation::UNSET`] when it is not —
    /// [`is_dst_indirect_at_index`](TransferNode::is_dst_indirect_at_index) tests the `unit` half
    /// against `NO_COMPONENT` (`dsc/dsc2.h:879-883`).
    pub loc_indirect: DataLocation,
    /// Field: e034_TransferNode.via_
    ///
    /// ⛔ ORDERED SOURCE TO DESTINATION, and bridge 1 walks it as a route: at the component it is
    /// lowering for, `via_[i + 1]` is the next hop and `via_[i - 1]` the previous
    /// (`dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:2631-2639`), `via_.front()` is
    /// the first hop out of the source (`:2530`, `:2764`) and `via_.back()` the last unit before
    /// [`loc`](Self::loc) (`:2572`, `ddc/ddc_transformation.cpp:24-28`). Empty means "no hop": the
    /// readers then take `loc_.unit_` itself (`SNTransferLowering.cpp:2764`).
    pub via: Vec<SenComponent>,
}

impl Default for DstVia {
    /// `dsc/dsc2.h:817-818` writes no member initialisers, but [`DataLocation`]'s own are both
    /// `NO_COMPONENT` (`sys-arch-spec/arch_enums.h:390-391`), so the DDL conversion's
    /// `dstVias_.emplace_back()` (`ddc/ddl/ddl_conversion.cpp:1176`) lands exactly here and
    /// `setDataLocAndInfo` fills `loc_` and `via_` immediately after (`:1178-1179`).
    fn default() -> Self {
        Self {
            loc: DataLocation::UNSET,
            loc_indirect: DataLocation::UNSET,
            via: Vec::new(),
        }
    }
}

/// Replaces: TransferNode::SizeAndIndex
///
/// One dimension of a unit-time transfer chunk: its extent, and where that dimension sits in the
/// source's and in the destination's view (`dsc/dsc2.h:820-823`).
///
/// ⛔ NO `Default`, FOR THE SAME REASON [`Size`] HAS NONE — `sizeDim_.dim_` has no initialiser, so
/// the JSON importer's `unitTimeTransferChunkStride_.emplace_back()` (`dsc/dsc2.cpp:1575`) leaves an
/// indeterminate dim behind until the entry's fields are read out of the JSON:
///
/// ```compile_fail
/// use deeptools::schedule::dsc2::SizeAndIndex;
/// let _ = SizeAndIndex::default();
/// ```
///
/// ⭐ THE TWO INDICES ARE EQUAL AT EVERY PRODUCER — `{{dim, size}, i, i}` (`ddc/ddcv1.cpp:524-525`,
/// `:1597-1598`) and `dim.srcSizeIdx_ = dim.dstSizeIdx_ = i`
/// (`dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:631`). Only the JSON round trip can
/// make them differ (`dsc/dsc2.cpp:1564-1567`), and only bridge 1 tells them apart, by direction:
/// the load path matches `srcSizeIdx_` and the store path `dstSizeIdx_` (`:338-348`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SizeAndIndex {
    /// Field: e034_TransferNode.sizeDim_
    ///
    /// The dim and how much of it this chunk covers (`dsc/dsc2.h:821`).
    ///
    /// ⛔ THE DDC'S TWO WRITES ARE NOT THE SAME SHAPE, AND ONLY ONE IS IN PLACE. The 4B-splat path
    /// mutates the STORED entry: `unitTimeTransferChunkSize_[0].sizeDim_.size_ *= 4`, guarded by a
    /// `DT_CHECK` that it was 1 (`ddc/ddcv1.cpp:544-546`). The hole split does not: it takes a COPY
    /// (`auto sizeDim = uttChunkSize[i]`, `:1637`), accumulates `unitTimeTransferNumChunks_` from
    /// it, sets THE COPY's extent to 1, appends that to `unitTimeTransferChunkStride_`
    /// (`:1638-1640`), and then ERASES the source entries from `unitTimeTransferChunkSize_`
    /// (`:1642-1643`) — so nothing in the chunk-size vector is ever left holding the 1.
    pub size_dim: Size,
    /// Field: e034_TransferNode.srcSizeIdx_
    ///
    /// This dim's position in the SOURCE's view sizes, absent as the authority's `-1`
    /// (`dsc/dsc2.h:822`). Bridge 1's load path searches for the entry whose index equals the
    /// position it is emitting (`SNTransferLowering.cpp:338-341`), so an absent index simply never
    /// matches — and the miss is a live answer there, not a refusal (`:349-370`).
    pub src_size_idx: Option<SizeIdx>,
    /// Field: e034_TransferNode.dstSizeIdx_
    ///
    /// The same position in the DESTINATION's view sizes, read by the store path
    /// (`SNTransferLowering.cpp:344-347`).
    pub dst_size_idx: Option<SizeIdx>,
}

/// Replaces: TransferNode::TransferType
///
/// `dsc/dsc2.h:854-861`. Which ends of a transfer are tensors — the answer `getTransferType`
/// derives from whether each side is a labeled data structure or a constant (`dsc/dsc2.h:884-896`).
///
/// ⛔ THE DISCRIMINANTS ARE NOT OBSERVABLE: no map is keyed by this enum, it has no spelling, the
/// JSON round trip does not carry it, and every use is an `==` or `!=` against a named enumerator
/// (`ddc/ddcv1.cpp:448`, `:471-477`, `:1050`, `:2325-2329`,
/// `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:4135`, `:7108`, `:7886`, `dsc/dsc2.cpp:3483-3501`).
///
/// ⛔ `INVALID_TRANSFER_TYPE` MEANS "NEITHER END IS A TENSOR", AND IT IS A REFUSAL — NOT, as
/// recorded before, a handled case answered with an empty size map. It is the fallthrough of the
/// five tests (`dsc/dsc2.h:895`), and the one reader that can see it,
/// `getBlockTransferSizePerDimCustomLocation` (`dsc/dsc2.cpp:3471-3472`), answers it with
/// `DT_ERROR("Transfer must have lds or constant id set")` at the very lines cited before,
/// `:3483-3484`, before it puts anything into its size map. So the variant exists to be MATCHED AND
/// REJECTED: a port of that function must refuse here and must not fabricate an empty map. The
/// other six `getTransferType` readers compare against a named enumerator and take no branch for it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TransferType {
    ConstantToConstant = 0,
    ConstantToTensor = 1,
    TensorToTensor = 2,
    NoTransferToTensor = 3,
    NoTransferFromTensor = 4,
    /// The authority's `INVALID_TRANSFER_TYPE` — the enum's own name already says "transfer type".
    Invalid = 5,
}

impl TransferType {
    /// Every form in the authority's declaration order (`dsc/dsc2.h:854-861`).
    pub const ALL: [Self; 6] = [
        Self::ConstantToConstant,
        Self::ConstantToTensor,
        Self::TensorToTensor,
        Self::NoTransferToTensor,
        Self::NoTransferFromTensor,
        Self::Invalid,
    ];
}

/// `dsc/dsc2.h:826-829` — the DDL allocation's `replication` for each end of one transfer. IBM
/// declares it as an UNNAMED struct member `repetition_`, so carrying it means naming it.
///
/// ⛔ WRITTEN ONCE, READ NOWHERE, AND NOT ON THE WIRE: `ddc/ddl/ddl_conversion.cpp:1171-1172` and
/// `:1189` are the only two mentions tree-wide outside the declaration, and the JSON round trip
/// carries neither member — the `"repetition_"` entries at `dsc/dsc2.cpp:132` and `:1172` are
/// `ComputeNode::instrAttribute_.repetition_` (`dsc/dsc2.h:907`), a different field. Carried because
/// the node declares it, and portable for exactly that reason: with no reader, ownership of the two
/// counts is the whole contract.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TransferRepetition {
    /// Field: e034_TransferNode.srcRep_
    ///
    /// The source end's count, from `getRepetitionIfExists(transfer_op.getSource())`
    /// (`ddc/ddl/ddl_conversion.cpp:1171-1172`).
    ///
    /// ⛔ [`None`] IS AN INDETERMINATE READ, NOT THE AUTHORITY'S `-1`. `int srcRep_;` carries no
    /// member initialiser (`dsc/dsc2.h:827`) and `TransferNode()` does not name `repetition_` in its
    /// mem-init list (`:815`), so on a default-constructed or JSON-imported node this storage is
    /// uninitialised and IBM's own `clone()` copies whatever it holds. Everywhere else in this file
    /// an [`Option`] spells the authority's `-1` sentinel; here it spells storage no writer has
    /// touched, which is why it is not simply `Repetition(1)`.
    pub src: Option<Repetition>,
    /// Field: e034_TransferNode.dstReps_
    ///
    /// ⛔ INDEX-PARALLEL WITH [`TransferNode::dst_vias`], NOT KEYED: the minting loop pushes one
    /// entry per `dstVias_.emplace_back()` in the same iteration
    /// (`ddc/ddl/ddl_conversion.cpp:1176-1189`). Empty by default, and that IS faithful where
    /// [`src`](Self::src) is absent — a `std::vector` member is default-constructed where an `int`
    /// without an initialiser is not.
    pub dsts: Vec<Repetition>,
}

/// Replaces: e034_TransferNode
///
/// `dsc/dsc2.h:814-898`. A transfer of one data stage from one source to one or more destinations —
/// the node bridge 1 lowers into an `agen` load or store. Its minting site is
/// `ddc/ddl/ddl_conversion.cpp:1166-1195`: a DDL `DataTransferOp` becomes one of these, with one
/// [`DstVia`] per declared destination. e023_TransferNode is this same class under the superseded
/// numbering, which listed ten of its fields; those are renumbered onto e034 below, and this batch
/// adds `repetition_`, `paddingInfo_` and the four `DataInfo` operands.
///
/// ⛔ THIS CARRIES SEVENTEEN OF TRANSFERNODE'S OWN TWENTY DECLARED FIELDS AND ITS BASE SUBOBJECT.
/// `nodeType_`, `name_` and `relevantComps_` arrive through [`base_class`](Self::base_class)
/// (`dsc/dsc2.h:460-461`, `:516`); `prev_` (`:515`) does not, because in an owning tree the parent is
/// the [`BlockNode`] whose child list holds the node. The other three of its own, with the reason:
///  * `lastFusableParentLoopSrc_` (`:830`) and `lastFusableParentLoopDst_` (`:831`) are
///    `const LoopNode*` held as POINTER IDENTITY, which needs e029_ScheduleNode's `name_`, exactly as
///    `DataInfo::bufferSwitchPosition_` does;
///  * `coreletViews_` (`:851`) is a map of `CoreletView`, four `UnitView`s (`:847-850`) —
///    e029_ScheduleNode.
///
/// ⚠️ AND THE OPEN ANCHORS DO NOT MATCH THAT SET, in both directions, because the scheduler's field
/// scan takes one declarator per declaration:
///  * `srcLdsAndLoopOffsets_` and `dstLdsAndLoopOffsets_` (`:832-833`) never got an anchor at all —
///    each lost it to the `Indirect` twin sharing its line — yet both are now CARRIED, so they are
///    anchored here with that fact noted on the field;
///  * `srcIndirectLoopsAndSize_` and `dstIndirectLoopsAndSizes_` ARE anchored as if they were this
///    class's, but they are `CoreletView`'s (`:848-849`); they stay open with `coreletViews_`, and
///    their line-sharing twins `srcLoopsAndSize_` and `dstLoopsAndSizes_` have no anchor either.
///
/// ⭐ THE FIVE `DataInfo` PREDICATES AND `getTransferType` ARE HERE, because the four [`DataInfo`]
/// fields they read landed with e033_DataInfo (`dsc/dsc2.h:867-896`). ⛔ ONLY `print` STAYS OUT, AND
/// NOT FOR `name_`: it prints `this`, the node's own ADDRESS (`dsc/dsc2.cpp:4385-4389`), which no
/// Rust value can reproduce. All three node `print`s do (`:4387`, `:4445`, `:4517`).
///
/// ⛔ `dstVias_`, `dstLdsAndLoopOffsets_` AND [`TransferRepetition::dsts`] ARE THREE PARALLEL VECTORS
/// ON ONE INDEX, and all three are now carried: the DDL conversion emplaces one of each per
/// destination in the same iteration (`ddc/ddl/ddl_conversion.cpp:1176-1189`), and
/// `hoistTransfersUpForReuse` takes [`non_memory_result_index`](Self::non_memory_result_index) — an
/// index into `dstVias_` — and indexes `dstLdsAndLoopOffsets_` with it
/// (`ddc/ddc_transformation.cpp:1570-1575`). Nothing in the type keeps their lengths equal, because
/// nothing in the authority does.
///
/// ⛔ NO `PartialEq`, as [`LoopNode`] has none: IBM declares no `operator==` and every consumer keys
/// on the POINTER. ⛔ AND NO `Clone` DERIVE either, now that `paddingInfo_` is carried — see
/// [`clone`](Self::clone).
#[derive(Debug)]
pub struct TransferNode {
    /// The `ScheduleNode` subobject (`dsc/dsc2.h:814`,
    /// `InheritWithClone<ScheduleNode, TransferNode>`), tagged `TRANSFER` by `TransferNode()`
    /// (`:815`). A transfer is a LEAF: it derives from `ScheduleNode` directly and owns no children.
    pub base_class: ScheduleNode,
    /// Field: e034_TransferNode.src_
    ///
    /// Where the data comes from (`dsc/dsc2.h:824`), written by `setDataLocAndInfo`
    /// (`ddc/ddl/ddl_conversion.cpp:1169`).
    pub src: DataLocation,
    /// Field: e034_TransferNode.srcIndirect_
    ///
    /// The location holding the source address when the read is indirect, or
    /// [`DataLocation::UNSET`] when it is direct (`dsc/dsc2.h:824`).
    /// [`is_src_indirect`](Self::is_src_indirect) is the test every reader applies before touching
    /// it (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:6302-6305`).
    pub src_indirect: DataLocation,
    /// Field: e034_TransferNode.dstVias_
    ///
    /// One entry per destination, in the DDL's declaration order (`dsc/dsc2.h:825`,
    /// `ddc/ddl/ddl_conversion.cpp:1175-1190`). Several entries is a multicast; the row-expansion
    /// path rejects several destinations at once (`:1183-1187`).
    pub dst_vias: Vec<DstVia>,
    /// Field: e034_TransferNode.repetition_
    ///
    /// The DDL allocation's replication for both ends (`dsc/dsc2.h:826-829`) — see
    /// [`TransferRepetition`], which names IBM's unnamed struct.
    pub repetition: TransferRepetition,
    /// Field: e034_TransferNode.srcLdsAndLoopOffsets_
    ///
    /// What the source operand IS and how it is addressed (`dsc/dsc2.h:832`). `setDataLocAndInfo`
    /// fills it from the DDL's source operand (`ddc/ddl/ddl_conversion.cpp:1169`) and
    /// `fillLoopOffsetsAndAddresses` puts the addresses in (`ddc/ddcv1.cpp:2360-2806`).
    ///
    /// ⚠️ THE SCHEDULER LISTED NO ANCHOR FOR IT, nor for
    /// [`dst_lds_and_loop_offsets`](Self::dst_lds_and_loop_offsets): both are declared two-per-line
    /// (`:832-833`) and `plan.py` takes only the first name of such a declaration, which is why the
    /// two `Indirect` halves were anchored and these two were not.
    pub src_lds_and_loop_offsets: DataInfo,
    /// Field: e034_TransferNode.srcIndirectLdsAndLoopOffsets_
    ///
    /// The same, for the operand that HOLDS THE SOURCE ADDRESS when the read is indirect
    /// (`dsc/dsc2.h:832`). [`is_src_indirect`](Self::is_src_indirect) is the test every reader applies
    /// before touching it (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:6302-6305`).
    pub src_indirect_lds_and_loop_offsets: DataInfo,
    /// Field: e034_TransferNode.dstLdsAndLoopOffsets_
    ///
    /// One entry per destination (`dsc/dsc2.h:833`).
    ///
    /// ⛔ PARALLEL TO [`dst_vias`](Self::dst_vias), AND THE PAIRING IS BY INDEX: the DDL conversion
    /// emplaces one of each per destination in the same iteration
    /// (`ddc/ddl/ddl_conversion.cpp:1176-1177`), `hoistTransfersUpForReuse` takes
    /// [`non_memory_result_index`](Self::non_memory_result_index) — an index into `dstVias_` — and
    /// indexes THIS vector with it (`ddc/ddc_transformation.cpp:1570-1575`), and bridge 1 walks both
    /// under one `dst_idx` (`dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:2150-2153`).
    /// They are two vectors here because IBM declares two and every writer pushes to them separately;
    /// nothing in the type keeps their lengths equal.
    pub dst_lds_and_loop_offsets: Vec<DataInfo>,
    /// Field: e034_TransferNode.dstIndirectLdsAndLoopOffsets_
    ///
    /// The operands that hold the destination addresses when the writes are indirect
    /// (`dsc/dsc2.h:833`). ⛔ ITS EMPTINESS IS THE WHOLE OF
    /// [`is_dst_indirect`](Self::is_dst_indirect) (`dsc/dsc2.h:878`).
    pub dst_indirect_lds_and_loop_offsets: Vec<DataInfo>,
    /// Field: e034_TransferNode.replicationFactor_
    ///
    /// How many times the loaded chunk is splatted, `1` for no splat (`dsc/dsc2.h:834`). Bridge 1
    /// multiplies one `agen` access's element count by it, refusing an LXLU splat it cannot express
    /// (`dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:963-971`), and divides the
    /// recorded `OUT` stick and element counts by it (`:727-729`).
    ///
    /// ⛔ IT IS THE OTHER HALF OF
    /// [`unit_time_transfer_chunk_size`](Self::unit_time_transfer_chunk_size) AND NOT AN
    /// INDEPENDENT COUNT — every DDC writer moves magnitude between the two. Read that field's doc
    /// before reading either alone.
    pub replication_factor: i32,
    /// Field: e034_TransferNode.unitTimeTransferChunkSize_
    ///
    /// "Continuous elements within a stick" (`dsc/dsc2.h:835-836`) — the contiguous dims of one
    /// unit-time transfer, in stick order, each entry's index equal to its own position, minted by
    /// `e126_populateUnitTimeTransfers` (`ddc/ddcv1.cpp:524-525`).
    ///
    /// ⛔ THE EXTENTS ARE NOT THE STICK SIZES AND THIS VECTOR ALONE IS NOT THE LOAD. What bridge 1
    /// reassembles is `Π extents × unitTimeTransferNumChunks_` (`SNTransferLowering.cpp:32-38`)
    /// `× replicationFactor_` (`:963-965`), and every DDC writer moves magnitude between the extents
    /// and [`replication_factor`](Self::replication_factor) rather than setting either alone: a
    /// `do2BSplat` mint pushes extent `1` per dim and the product of the stick sizes into the factor
    /// (`ddc/ddcv1.cpp:510-525`, `:527-535`), the fp32 fixup hands 4 of it back to entry 0
    /// (`:544-548`), `e307_exploreAssignDataStages` shrinks each entry IN PLACE to its data-stage
    /// extent with `replicationFactor_ *= size / dsDim; size = dsDim` (`:1549-1552`), and its
    /// `reduce2B` tail sets EVERY extent to `1` (`:1669-1677`). ⛔ Both shrinks DISCARD the
    /// magnitude when `!doSplat` (`:1549`, `:1673`) — there the transfer really did get smaller — so
    /// the product holds for a splat only. A reader that takes the extents for the stick geometry
    /// reads a transfer already reduced; one that drops the factor loses the splat.
    ///
    /// ⛔ NOR IS THE LENGTH ALWAYS THE STICK COUNT, as recorded before: minting stops as soon as
    /// `elemSoFar >= numElemLimit`, dividing the extent it is pushing when it oversteps
    /// (`:510-522`), `numElemLimit` being the metadata's `force_num_elements_` (`:443-446`). What
    /// cannot change the length again is the DEAD writer — the out-of-stick `IN` append
    /// (`:1597-1598`) and the erase (`:1642-1643`) both sit inside `checkAndResetUnitTimeTransfer`
    /// (`:1560-1645`), the same lambda that owns
    /// [`unit_time_transfer_chunk_stride`](Self::unit_time_transfer_chunk_stride)'s only writer and
    /// whose callsite is commented out (`:1649-1650`), which is also why the
    /// `DT_CHECK(uttChunkSize.size() == tensorSizes.size())` at `:1568` is unreachable.
    pub unit_time_transfer_chunk_size: Vec<SizeAndIndex>,
    /// Field: e034_TransferNode.unitTimeTransferNumChunks_
    ///
    /// How many chunks one unit-time transfer covers, `1` for a single contiguous chunk
    /// (`dsc/dsc2.h:837`), which bridge 1 multiplies into the element count
    /// (`SNTransferLowering.cpp:32-38`) — the third factor of
    /// [`unit_time_transfer_chunk_size`](Self::unit_time_transfer_chunk_size)'s product.
    ///
    /// ⛔ NO LIVE C++ PRODUCER SETS IT EITHER, and the citation recorded here before hid that. Both
    /// writers — the `= 1` and the product of the extents the hole split moves out of the chunk
    /// sizes (`ddc/ddcv1.cpp:1633-1643`) — are inside the same dead `checkAndResetUnitTimeTransfer`
    /// as [`unit_time_transfer_chunk_stride`](Self::unit_time_transfer_chunk_stride)'s
    /// (`:1649-1650`), so outside the JSON importer (`dsc/dsc2.cpp:1571-1572`) it holds the
    /// initialiser `1` and `loadSize *= unitTimeTransferNumChunks_` at `:1663` multiplies by one.
    pub unit_time_transfer_num_chunks: i32,
    /// Field: e034_TransferNode.unitTimeTransferChunkStride_
    ///
    /// The dims the chunks stride over — the entries the hole split removed from
    /// [`unit_time_transfer_chunk_size`](Self::unit_time_transfer_chunk_size), each with its extent
    /// set to 1 (`dsc/dsc2.h:838`, `ddc/ddcv1.cpp:1636-1643`).
    ///
    /// ⛔ NO LIVE C++ PRODUCER FILLS THIS. Its one writer is inside the lambda
    /// `checkAndResetUnitTimeTransfer` (`ddc/ddcv1.cpp:1640`), whose sole callsite is commented out
    /// (`:1649-1650`), so outside the JSON importer (`dsc/dsc2.cpp:1573-1584`) it is always empty —
    /// which is why the `size() <= 1` check at `dsc/dsc2.cpp:3555` never fires and why bridge 1's
    /// own refusal of more than one stride dim (`SNTransferLowering.cpp:324-331`) is never reached.
    /// It is carried rather than dropped because bridge 1 reads it in seven places (`:930`, `:1270`,
    /// `:1713`, `:1723`, `:2227`, `:2256`, `:2454`) and a JSON-imported tree can carry it.
    pub unit_time_transfer_chunk_stride: Vec<SizeAndIndex>,
    /// Field: e034_TransferNode.rotateNumElements_
    ///
    /// How far the LXLU rotates the loaded data, `0` for no rotation (`dsc/dsc2.h:839`). Every
    /// reader guards on `> 0` (`SNTransferLowering.cpp:991`, `:1097`, `:2239`, `:2277`).
    pub rotate_num_elements: i32,
    /// Field: e034_TransferNode.coreIdToGTRInfo_
    ///
    /// The group tag register each core uses for this transfer — L3 only, as the authority's own
    /// comment says (`dsc/dsc2.h:840`). The L3 scheduler writes it one core at a time and refuses to
    /// overwrite an entry (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:4836`, `:5249-5260`).
    ///
    /// ⛔ AND ITS KEY ORDER IS OBSERVABLE, WHICH IS WHY IT IS A [`BTreeMap`] AND NOT A HASH MAP: the
    /// JSON exporter ITERATES it and emits one object per core (`dsc/dsc2.cpp:627-629`), so the
    /// `std::map`'s ascending-core-id order reaches the emitted DSC. [`CoreId`]'s derived [`Ord`] is
    /// that same numeric order over the `int` key `std::stoi` reads back (`dsc/dsc2.cpp:1599`).
    pub core_id_to_gtr_info: BTreeMap<CoreId, GroupTagRegInfo>,
    /// Field: e034_TransferNode.transferSize_
    ///
    /// "Explicit transfer size. If filled, use this size rather than derived from data stage"
    /// (`dsc/dsc2.h:841-843`). ⛔ ABSENCE IS THE COMMON CASE AND IS TESTED PER DIM, never for the
    /// whole map: `getBlockTransferSizePerDimCustomLocation` overrides one dim's extent only where
    /// `count(dim)` says so (`dsc/dsc2.cpp:3561-3562`), while two fill sites require the map to be
    /// EMPTY first (`dsc/dsc2.cpp:4776-4777`,
    /// `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:7897-7905`).
    ///
    /// ⛔ BUT `count`/`at` ARE NOT ITS ONLY READERS, AND THE ONE THAT WAS MISSED IS THE ORDER-BEARING
    /// ONE: the JSON exporter iterates this map too and emits `primaryDimToString.at(dim)` keys in
    /// `std::map` order (`dsc/dsc2.cpp:641-647`). [`BTreeMap`] reproduces that order only because
    /// [`PrimaryDimTypes`]'s Rust discriminants are the authority's (`dsc/dims.h:34-48`), so the
    /// derived [`Ord`] and `operator<` on the C++ enum agree. A hash map here — or a reordering of
    /// that enum — would silently reorder emitted JSON, which is why the enum's own discriminant
    /// guard and this container are one decision and not two.
    pub transfer_size: BTreeMap<PrimaryDimTypes, DimSize>,
    /// Field: e034_TransferNode.paddingInfo_
    ///
    /// "Zero padding sizes in L3 transfers" (`dsc/dsc2.h:844-845`) — what the L3 scheduler builds per
    /// padded dim and end (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:5359-5473`) and what
    /// `dsc/dsc2.cpp:4761-4990` then turns into condition and transfer nodes.
    ///
    /// ⛔ EMPTY IS THE GATE, NOT A VALUE: every reader but that one transformation does nothing at
    /// all unless [`TransferPadInfo::is_empty`] is false (`ddc/ddcv1.cpp:2860`,
    /// `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:6298`, `dsc/dsc2.cpp:4761`).
    ///
    /// ⛔ AND IT IS WHY THIS TYPE HAS NO `Clone` DERIVE — see [`clone`](Self::clone).
    pub padding_info: TransferPadInfo,
    /// Field: e034_TransferNode.transferCoordinates_
    ///
    /// How this transfer's dims are folded (`dsc/dsc2.h:852`). `buildFoldForTransferNode` is the
    /// writer — it copies the allocation's coordinate for each transferred dim and then rebuilds the
    /// levels the transfer itself splits (`ddc/ddc_fold.cpp:1340-1420`) — and `getTransferFoldParams`
    /// and the LDS lowering are the readers (`ddc/ddc_fold.cpp:906`, `dsc/dsc2.cpp:3848-3871`).
    ///
    /// ⛔ AND IT IS COPIED BY [`clone`](Self::clone) WHERE `paddingInfo_` IS NOT: IBM's copy
    /// constructor is implicit, so every member with a real copy is copied, and
    /// [`CoordinateType`]'s is the authority's `operator=` replay.
    pub transfer_coordinates: CoordinateType,
}

impl Default for TransferNode {
    /// The authority's member initialisers (`dsc/dsc2.h:834`, `:837`, `:839`) over
    /// [`DataLocation`]'s own (`sys-arch-spec/arch_enums.h:390-391`).
    ///
    /// ⭐ AND IT IS `TransferNode()`, BASE TAG INCLUDED: that constructor passes `TRANSFER` to
    /// `ScheduleNode` (`dsc/dsc2.h:815`), which is [`base_class`](TransferNode::base_class)'s
    /// `nodeType_`.
    ///
    /// ⛔ AND ONE MEMBER HAS NO INITIALISER TO REPRODUCE: `repetition_.srcRep_` (`:827`) is left
    /// uninitialised by that constructor, which is what [`TransferRepetition::src`]'s [`None`] spells.
    fn default() -> Self {
        Self {
            base_class: ScheduleNode::new(NodeType::Transfer),
            src: DataLocation::UNSET,
            src_indirect: DataLocation::UNSET,
            dst_vias: Vec::new(),
            repetition: TransferRepetition::default(),
            src_lds_and_loop_offsets: DataInfo::default(),
            src_indirect_lds_and_loop_offsets: DataInfo::default(),
            dst_lds_and_loop_offsets: Vec::new(),
            dst_indirect_lds_and_loop_offsets: Vec::new(),
            replication_factor: 1,
            unit_time_transfer_chunk_size: Vec::new(),
            unit_time_transfer_num_chunks: 1,
            unit_time_transfer_chunk_stride: Vec::new(),
            rotate_num_elements: 0,
            core_id_to_gtr_info: BTreeMap::new(),
            transfer_size: BTreeMap::new(),
            padding_info: TransferPadInfo::default(),
            transfer_coordinates: CoordinateType::default(),
        }
    }
}

impl Clone for TransferNode {
    /// IBM's `clone()` is `new TransferNode(static_cast<TransferNode const&>(*this))` — the implicit
    /// COPY CONSTRUCTOR, reached through `InheritWithClone` (`util/utils.h:105-106`,
    /// `dsc/dsc2.h:814`).
    ///
    /// ⛔ SO A CLONED TRANSFER NODE HAS NO PADDING INFO, and that is load-bearing rather than a leak
    /// this port may tidy up: [`TransferPadInfo`]'s copy constructor is "Do nothing on purpose"
    /// (`dsc/dsc2.h:761-764`), and the authority CHECKS the consequence three times — `dsc/dsc2.cpp`
    /// clones a padded transfer at `:5546` and `:5702` and then `DT_CHECK_MSG`s the clone `isEmpty()`
    /// at `:5445`, `:5692` and `:5792`. A deep copy would turn all three into fatal errors.
    /// [`TransferPadInfo`]'s own absent `Clone` is what makes the derive here impossible, so this is
    /// a compile error the port cannot walk past rather than a convention.
    ///
    /// ⛔ AND C++ CANNOT ASSIGN ONE AT ALL, WHICH IS UNREPRESENTABLE HERE: `TransferPadInfo`'s copy
    /// assignment is `= delete` (`dsc/dsc2.h:766`), which implicitly deletes `TransferNode`'s, so
    /// `*a = b` does not compile there and always compiles here. Measured on a control reproducing
    /// that shape — a do-nothing copy constructor plus a deleted copy assignment inside the same CRTP
    /// `clone()` — the clone reported the pad empty with every other member copied, and the
    /// assignment was rejected as "copy assignment operator is implicitly deleted".
    fn clone(&self) -> Self {
        Self {
            base_class: self.base_class.clone(),
            src: self.src,
            src_indirect: self.src_indirect,
            dst_vias: self.dst_vias.clone(),
            repetition: self.repetition.clone(),
            src_lds_and_loop_offsets: self.src_lds_and_loop_offsets.clone(),
            src_indirect_lds_and_loop_offsets: self.src_indirect_lds_and_loop_offsets.clone(),
            dst_lds_and_loop_offsets: self.dst_lds_and_loop_offsets.clone(),
            dst_indirect_lds_and_loop_offsets: self.dst_indirect_lds_and_loop_offsets.clone(),
            replication_factor: self.replication_factor,
            unit_time_transfer_chunk_size: self.unit_time_transfer_chunk_size.clone(),
            unit_time_transfer_num_chunks: self.unit_time_transfer_num_chunks,
            unit_time_transfer_chunk_stride: self.unit_time_transfer_chunk_stride.clone(),
            rotate_num_elements: self.rotate_num_elements,
            core_id_to_gtr_info: self.core_id_to_gtr_info.clone(),
            transfer_size: self.transfer_size.clone(),
            padding_info: TransferPadInfo::default(),
            transfer_coordinates: self.transfer_coordinates.clone(),
        }
    }
}

impl TransferNode {
    /// `dsc/dsc2.cpp:4364-4366`. Whether the source's storage is not one of [`MEMORIES`] — a FIFO, a
    /// latch or a constant rather than a memory.
    ///
    /// ⛔ ITS ONLY CALLER IS OFF THIS CAMPAIGN'S PATH (`dsc/dsc2Pcfg.cpp:1039`, and DCG/PCFG is not
    /// ours). It is ported because it is this class's own method over this class's own field.
    pub fn has_non_memory_source(&self) -> bool {
        !MEMORIES.contains(&self.src.storage)
    }

    /// `dsc/dsc2.cpp:4368-4370`. Whether ANY destination is not a memory, i.e. whether
    /// [`non_memory_result_index`](Self::non_memory_result_index) found one. Its caller is
    /// `moveTransferNode`, which refuses to hoist a transfer that has one
    /// (`ddc/ddc_transformation_util.cpp:397`).
    pub fn has_non_memory_result(&self) -> bool {
        self.non_memory_result_index().is_some()
    }

    /// `dsc/dsc2.cpp:4372-4374`. Whether the destination at `index` is not a memory.
    ///
    /// ⛔ ITS ONLY CALLERS ARE OFF THIS CAMPAIGN'S PATH (`dsc/dsc2Pcfg.cpp:1041`, `:1309`), and IBM's
    /// `dstVias_[i]` is unchecked there — an out-of-range index is undefined behaviour in the
    /// authority, where here it is the slice's own bound.
    pub fn check_non_memory_result_index(&self, index: usize) -> bool {
        !MEMORIES.contains(&self.dst_vias[index].loc.storage)
    }

    /// `dsc/dsc2.cpp:4376-4383`. The FIRST destination that is not a memory, as an index into
    /// [`dst_vias`](Self::dst_vias).
    ///
    /// ⭐ THE AUTHORITY'S `-1` IS [`None`], AND BOTH CALLERS ALREADY TREAT IT THAT WAY: each writes
    /// `int fifoInd` and guards every use with `fifoInd != -1`
    /// (`ddc/ddc_transformation.cpp:1570-1572`, `:1818-1819`).
    pub fn non_memory_result_index(&self) -> Option<usize> {
        self.dst_vias
            .iter()
            .position(|dst| !MEMORIES.contains(&dst.loc.storage))
    }

    /// `dsc/dsc2.h:877`. Whether the source address comes from
    /// [`src_indirect`](Self::src_indirect) rather than from the data stage. Its in-scope callers
    /// gate every read of the indirect fields on it (`dsc/dsc2.cpp:506`, `:558`, `:656`, `:3066`,
    /// `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:6302-6305`).
    pub fn is_src_indirect(&self) -> bool {
        self.src_indirect.unit != SenComponent::NoComponent
    }

    /// `dsc/dsc2.h:879-883`. Whether the destination at `dst_vias_idx` is addressed indirectly.
    /// Called per destination by `fillLoopOffsetsAndAddresses`
    /// (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:6317`).
    ///
    /// ⭐ HALF OF THE AUTHORITY'S `DT_CHECK_MSG(dstViasIdx >= 0 && ... < size())` (`:880-881`) IS A
    /// TYPE GUARD HERE: `usize` makes a negative index unrepresentable, so only the upper bound is
    /// left to check and the slice checks it.
    ///
    /// ```compile_fail
    /// use deeptools::schedule::dsc2::TransferNode;
    /// let _ = TransferNode::default().is_dst_indirect_at_index(-1);
    /// ```
    pub fn is_dst_indirect_at_index(&self, dst_vias_idx: usize) -> bool {
        self.dst_vias[dst_vias_idx].loc_indirect.unit != SenComponent::NoComponent
    }

    /// `dsc/dsc2.h:866`. Whether the source is a labeled data structure. Its callers gate every
    /// `labeledDs_.at(srcLdsAndLoopOffsets_.myLdsIdx_)` on it (`ddc/ddc_transformation.cpp:728`,
    /// `ddc/ddcv1.cpp:1441`).
    pub fn is_src_labeled_ds(&self) -> bool {
        self.src_lds_and_loop_offsets.is_labeled_ds()
    }

    /// `dsc/dsc2.h:867-870`. Whether the FIRST destination is a labeled data structure — the
    /// authority's `!dstLdsAndLoopOffsets_.empty() && front().isLabeledDs()`, which is what
    /// [`Option::is_some_and`] over [`slice::first`] states.
    ///
    /// ⛔ IT ASKS ONLY THE FIRST DESTINATION THOUGH A MULTICAST HAS SEVERAL, so a transfer whose
    /// SECOND destination is the tensor answers no. Verbatim, and
    /// [`transfer_type`](Self::transfer_type) inherits it.
    pub fn is_dst_labeled_ds(&self) -> bool {
        self.dst_lds_and_loop_offsets
            .first()
            .is_some_and(DataInfo::is_labeled_ds)
    }

    /// `dsc/dsc2.h:871`. Whether the source is a constant container
    /// (`dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:2683-2687` reads the id it
    /// promises).
    pub fn is_src_constant(&self) -> bool {
        self.src_lds_and_loop_offsets.is_constant()
    }

    /// `dsc/dsc2.h:872-875`. Whether the FIRST destination is a constant, on the same terms and with
    /// the same first-only reach as [`is_dst_labeled_ds`](Self::is_dst_labeled_ds).
    pub fn is_dst_constant(&self) -> bool {
        self.dst_lds_and_loop_offsets
            .first()
            .is_some_and(DataInfo::is_constant)
    }

    /// `dsc/dsc2.h:878`. Whether ANY destination is addressed indirectly.
    ///
    /// ⛔ IT TESTS A VECTOR'S EMPTINESS, NOT A LOCATION, which is why it does not mirror
    /// [`is_src_indirect`](Self::is_src_indirect): that one reads `srcIndirect_.unit_`, and the
    /// per-destination form is [`is_dst_indirect_at_index`](Self::is_dst_indirect_at_index), which
    /// reads `dstVias_[i].locIndirect_`. So the three are three different questions and the
    /// authority answers them from three different fields.
    pub fn is_dst_indirect(&self) -> bool {
        !self.dst_indirect_lds_and_loop_offsets.is_empty()
    }

    /// `dsc/dsc2.h:884-896`. Which ends of the transfer are tensors, derived from the four predicates
    /// above and from nothing else.
    ///
    /// ⛔ THE TESTS ARE ORDERED AND THE ORDER IS LOAD-BEARING: constant-to-constant is asked before
    /// constant-to-tensor, so it wins where both hold. [`TransferType::Invalid`] is the fallthrough
    /// and it is a REFUSAL rather than a case — read that variant's own doc before matching it.
    ///
    /// ⛔ TENSOR-TO-CONSTANT AND CONSTANT-TO-NOTHING BOTH FALL THROUGH TO THE REFUSAL. There is no
    /// arm for a constant DESTINATION other than the first, so a tensor source writing one fails the
    /// fifth arm on `!isDstConstant()` and a constant source with an empty destination list fails
    /// every arm that asks about a destination at all — the enum names five forms of transfer and
    /// this is a five-way chain, not a matrix.
    pub fn transfer_type(&self) -> TransferType {
        match (
            self.is_src_labeled_ds(),
            self.is_src_constant(),
            self.is_dst_labeled_ds(),
            self.is_dst_constant(),
        ) {
            (_, true, _, true) => TransferType::ConstantToConstant,
            (_, true, true, _) => TransferType::ConstantToTensor,
            (true, _, true, _) => TransferType::TensorToTensor,
            (false, false, true, _) => TransferType::NoTransferToTensor,
            (true, _, false, false) => TransferType::NoTransferFromTensor,
            _ => TransferType::Invalid,
        }
    }
}

// crustify:todo: e034_TransferNode.coreletViews_

// crustify:todo: e034_TransferNode.dstIndirectLoopsAndSizes_

// crustify:todo: e034_TransferNode.lastFusableParentLoopDst_

// crustify:todo: e034_TransferNode.lastFusableParentLoopSrc_

// crustify:todo: e034_TransferNode.srcIndirectLoopsAndSize_

/// How many folds one compute node engages — `numFoldsEngaged` (`dsc/dsc2.h:940`, `int`), filled
/// from `sysDef.numFoldsPerUnit` (`ddc/ddcv1.cpp:1887`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NumFoldsEngaged(pub i32);

/// One operand's element count, as [`ComputeNode::operand_sizes`] reports it — "number of elements
/// read or written by compute node" (`dsc/dsc2.h:956-958`, `int`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OperandSize(pub i32);

/// One element's width in bits — the value of `EnumsConversion::dataFormatsToBitWidth`
/// (`util/sendefs/sendefs.cpp:129-141`, `int`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BitWidth(pub i32);

/// A compute instruction's general SRC1/IMM field — `InstrAttribute::mode_` (`dsc/dsc2.h:915`).
///
/// ⛔ AN OPCODE-SPECIFIC ENCODING, NOT A CLOSED SET: bridge 1 reads it as an FMUL divide selector at
/// 11 (`dsc-based-utils/DSC2ToDataflowIR/V3/SNComputeLowering.cpp:1102`), as a FEST flavour at 0
/// through 9 (`:1317-1385`) and as a convert flavour at 1, 8, 10, 12 and 14 (`:1454-1462`), and the
/// DDL states it verbatim (`ddc/ddl/ddl_conversion.cpp:1368-1370`).
///
/// ⛔ AND -1 IS EMITTED, NOT ELIDED: two writers put the field on the wire whatever it holds —
/// `ddc/ddl/ddl_conversion.cpp:3145` builds `APInt(64, mode_)` for the DataflowIR `ComputeOp`, and
/// `dsc/dsc2.cpp:169` writes `"mode_" : -1` into the JSON its own importer reads straight back
/// (`:1196-1197`). So [`InstrAttribute::mode`]'s `None` denotes BOTH "matches no encoding" to the
/// branching readers above and the integer -1 to those two.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Mode(pub i32);

/// Which compute slices an instruction runs on — `InstrAttribute::compute_mask_`
/// (`dsc/dsc2.h:916`, `size_t`, all eight slices by default).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ComputeMask(pub u64);

/// A repetition count, as the DDL's `getRepetitionIfExists` yields it
/// (`ddc/ddl/ddl_conversion.cpp:1393-1394`).
///
/// ⛔ TWO ROLES, ONE UNIT: `InstrAttribute::repetition_` counts the slices one PACK/MERGE
/// instruction repeats over (`dsc/dsc2.h:907`), while a `RepetitionWithOffset` entry is the SPREAD
/// of one operand — `ddc/ddc_transformation.cpp:1358-1379` clones the node `entry - 1` further
/// times and leaves 1 behind.
///
/// ⭐ AND THAT SECOND ROLE HAS TWO HOLDERS, filled by the same lambda: `TransferNode::repetition_`
/// takes one per transfer end (`ddc/ddl/ddl_conversion.cpp:1171-1189`) — see [`TransferRepetition`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Repetition(pub i32);

/// One entry of a PACK/MERGE mapping (`dsc/dsc2.h:906`) — a source element position within the
/// 128-bit slice, or a negative value standing for a zero/sign-extended slot.
///
/// ⛔ THE NEGATIVE ENTRIES ARE NOT ONE SENTINEL. The authority's own tables spell an extend slot -1
/// (`ddc/transformations/automatic_shuffle/shuffle.cpp:19-41` — `pack24`, `pack8`, `pack9` and
/// `pack12`-`pack15`), and `expand_indices` then scales EVERY entry with no -1 guard,
/// `compact_indices[i] * scale + j` (`ddc/ddc_transformation.cpp:1952`). `scale` is
/// `128 / indices.size() / element_bit_width` (`:1941-1947`), so a 4-bit `pack12` expands at scale 2
/// and its one -1 slot becomes the PAIR -2, -1 before it is stored (`:1969-1970`). See
/// [`InstrAttribute::indices`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PackMergeIndex(pub i32);

/// A DDL data-connect name — the wire identity of one operand (`dsc/dsc2.h:926-929`). ⛔ OPEN TEXT,
/// NOT A CLOSED SET: `ddc/ddc_fold.cpp:1996` and `:4099` compare these names to each other and
/// `ddc/ddc_transformation_util.cpp:1520-1525` rewrites them, all as the DDL authored them.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DataConnect(pub String);

/// Replaces: ComputeOpType
///
/// `dsc/dscdefn.h:134-207`. Which compute instruction a [`ComputeNode`] issues.
///
/// ⛔ `COUNT` IS A LIVE VALUE HERE, NOT A COUNT SENTINEL: it is `ComputeNode::type_`'s initialiser
/// (`dsc/dsc2.h:933`) and `computeTypeToString` gives it the spelling `"undefined"`
/// (`dsc/dscdefn.cpp:87`), so an unfilled node has a name rather than a hole.
///
/// ⛔ THE DISCRIMINANTS ARE THE AUTHORITY'S: `computeTypeToString` is a `std::map` keyed by this
/// enum (`dsc/dscdefn.h:210`), so declaration order is its iteration order, and [`Self::ALL`] is
/// positional against it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ComputeOpType {
    /// Precision-independent fma/ima — `dataFormat_` picks the precision, which is why
    /// [`ComputeNode::operand_sizes`] dispatches on the format for this one op alone.
    Macc,
    Fma32,
    Fma16,
    Fma8,
    Fma4,
    Ima8,
    Ima4,
    Fmax,
    Fmin,
    Fabsmax,
    Fsignedabsmineq,
    Fnms,
    Fsub,
    Fmul,
    And,
    Or,
    Xnorround,
    Andnot,
    Fcmp,
    Select,
    Packmerge,
    Reduce,
    Reciprocal,
    Layernormscale,
    Layernormscale32,
    Fest,
    Icvt,
    Shr,
    Gcvt,
    /// ⛔ THE ONE OP WITH NO SPELLING — see [`Self::name`].
    Fcvt,
    Splat,
    Ime,
    Ee,
    ExpP1,
    ExpP2,
    LogP1,
    LogP2,
    Realdiv,
    Gelu,
    GeluBwdP1,
    GeluBwdP2,
    Where3,
    Sqrt,
    Rsqrt,
    MishP1,
    MishP2,
    Greaterequal,
    Lesserequal,
    Greaterthan,
    Lesserthan,
    Equalto,
    Notequal,
    Exx232P1,
    Exx232P2,
    Exx232P3,
    Fp32todl16,
    Dl16tofp32,
    Sigmoid,
    Exp,
    Shuffle,
    Dl16tobf16,
    SoftplusP1,
    SoftplusP2,
    AutomaticShuffling,
    Assign,
    Floor,
    Idx32toaddr,
    AddI32ToI32,
    AddI64ToI64,
    MulI32ToI32,
    /// `ComputeNode::type_`'s initialiser (`dsc/dsc2.h:933`), spelled `"undefined"`.
    #[default]
    Count,
}

/// ⛔ E0080 IF AN OP IS EVER INSERTED, DROPPED, REORDERED OR LEFT OUT OF `ALL`: the discriminants
/// are what `computeTypeToString`'s `std::map` orders by, and `ALL` is positional against them.
const _: () = {
    let mut i = 0;
    while i < ComputeOpType::ALL.len() {
        assert!(
            ComputeOpType::ALL[i] as usize == i,
            "ComputeOpType::ALL is out of declaration order"
        );
        i += 1;
    }
};

impl ComputeOpType {
    /// Every op in the authority's declaration order (`dsc/dscdefn.h:134-207`), `COUNT` included
    /// because it is a value this campaign's nodes actually hold.
    pub const ALL: [Self; 71] = [
        Self::Macc,
        Self::Fma32,
        Self::Fma16,
        Self::Fma8,
        Self::Fma4,
        Self::Ima8,
        Self::Ima4,
        Self::Fmax,
        Self::Fmin,
        Self::Fabsmax,
        Self::Fsignedabsmineq,
        Self::Fnms,
        Self::Fsub,
        Self::Fmul,
        Self::And,
        Self::Or,
        Self::Xnorround,
        Self::Andnot,
        Self::Fcmp,
        Self::Select,
        Self::Packmerge,
        Self::Reduce,
        Self::Reciprocal,
        Self::Layernormscale,
        Self::Layernormscale32,
        Self::Fest,
        Self::Icvt,
        Self::Shr,
        Self::Gcvt,
        Self::Fcvt,
        Self::Splat,
        Self::Ime,
        Self::Ee,
        Self::ExpP1,
        Self::ExpP2,
        Self::LogP1,
        Self::LogP2,
        Self::Realdiv,
        Self::Gelu,
        Self::GeluBwdP1,
        Self::GeluBwdP2,
        Self::Where3,
        Self::Sqrt,
        Self::Rsqrt,
        Self::MishP1,
        Self::MishP2,
        Self::Greaterequal,
        Self::Lesserequal,
        Self::Greaterthan,
        Self::Lesserthan,
        Self::Equalto,
        Self::Notequal,
        Self::Exx232P1,
        Self::Exx232P2,
        Self::Exx232P3,
        Self::Fp32todl16,
        Self::Dl16tofp32,
        Self::Sigmoid,
        Self::Exp,
        Self::Shuffle,
        Self::Dl16tobf16,
        Self::SoftplusP1,
        Self::SoftplusP2,
        Self::AutomaticShuffling,
        Self::Assign,
        Self::Floor,
        Self::Idx32toaddr,
        Self::AddI32ToI32,
        Self::AddI64ToI64,
        Self::MulI32ToI32,
        Self::Count,
    ];

    /// The spelling `EnumsConversion::computeTypeToString` gives this op (`dsc/dscdefn.h:210`,
    /// defined `dsc/dscdefn.cpp:33-105`).
    ///
    /// ⛔ `FCVT` HAS NO ENTRY: the map holds 70 of the 71 ops, so `computeTypeToString.at(FCVT)`
    /// throws — absent here rather than a throw.
    /// ⛔ `EQUALTO` IS SPELLED `"equal"`, not `"equalto"` (`dsc/dscdefn.cpp:85`), and the DDL is
    /// matched against these spellings (`ddc/ddl/ddl_conversion.cpp:1432`).
    pub fn name(self) -> Option<&'static str> {
        let name = match self {
            Self::Macc => "macc",
            Self::Fma32 => "fma32",
            Self::Fma16 => "fma16",
            Self::Fma8 => "fma8",
            Self::Fma4 => "fma4",
            Self::Ima8 => "ima8",
            Self::Ima4 => "ima4",
            Self::Fmax => "fmax",
            Self::Fmin => "fmin",
            Self::Fabsmax => "fabsmax",
            Self::Fsignedabsmineq => "fsignedabsmineq",
            Self::Fnms => "fnms",
            Self::Fsub => "fsub",
            Self::Fmul => "fmul",
            Self::And => "and",
            Self::Or => "or",
            Self::Xnorround => "xnorround",
            Self::Andnot => "andnot",
            Self::Fcmp => "fcmp",
            Self::Select => "select",
            Self::Packmerge => "packmerge",
            Self::Reduce => "reduce",
            Self::Reciprocal => "reciprocal",
            Self::Layernormscale => "layernormscale",
            Self::Layernormscale32 => "layernormscale32",
            Self::Fest => "fest",
            Self::Icvt => "icvt",
            Self::Shr => "shr",
            Self::Gcvt => "gcvt",
            Self::Fcvt => return None,
            Self::Splat => "splat",
            Self::Ime => "ime",
            Self::Ee => "ee",
            Self::ExpP1 => "exp_p1",
            Self::ExpP2 => "exp_p2",
            Self::LogP1 => "log_p1",
            Self::LogP2 => "log_p2",
            Self::Realdiv => "realdiv",
            Self::Gelu => "gelu",
            Self::GeluBwdP1 => "gelu_bwd_p1",
            Self::GeluBwdP2 => "gelu_bwd_p2",
            Self::Where3 => "where3",
            Self::Sqrt => "sqrt",
            Self::Rsqrt => "rsqrt",
            Self::MishP1 => "mish_p1",
            Self::MishP2 => "mish_p2",
            Self::Greaterequal => "greaterequal",
            Self::Lesserequal => "lesserequal",
            Self::Greaterthan => "greaterthan",
            Self::Lesserthan => "lesserthan",
            Self::Equalto => "equal",
            Self::Notequal => "notequal",
            Self::Exx232P1 => "exx2_32_p1",
            Self::Exx232P2 => "exx2_32_p2",
            Self::Exx232P3 => "exx2_32_p3",
            Self::Fp32todl16 => "fp32todl16",
            Self::Dl16tofp32 => "dl16tofp32",
            Self::Sigmoid => "sigmoid",
            Self::Exp => "exp",
            Self::Shuffle => "shuffle",
            Self::Dl16tobf16 => "dl16tobf16",
            Self::SoftplusP1 => "softplus_p1",
            Self::SoftplusP2 => "softplus_p2",
            Self::AutomaticShuffling => "automatic_shuffling",
            Self::Assign => "assign",
            Self::Floor => "floor",
            Self::Idx32toaddr => "idx32toaddr",
            Self::AddI32ToI32 => "addi32toi32",
            Self::AddI64ToI64 => "addi64toi64",
            Self::MulI32ToI32 => "muli32toi32",
            Self::Count => "undefined",
        };
        Some(name)
    }

    /// `EnumsConversion::stringToComputeType`, the `flipMap` of the above (`dsc/dscdefn.h:211`,
    /// `dsc/dscdefn.cpp:106-107`). An unknown spelling is absent, which is the DDL rejection at
    /// `ddc/ddl/ddl_conversion.cpp:1432-1436`; the DDL's own `"macc"` never reaches here, because
    /// that spelling picks an FMA/IMA op from the precision instead (`:1410-1430`).
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|op| op.name() == Some(name))
    }
}

/// Replaces: DataFormats
///
/// `util/sendefs/sendefs.h:30-54`. One operand's element format. It is `util/sendefs`' type rather
/// than dsc2's, and it lives here beside its only ported reader, [`ComputeNode::data_format`].
///
/// ⛔ `NUM_DATA_FORMATS` IS A PURE SENTINEL, so it is [`Self::COUNT`] and not a variant: it has no
/// spelling (`dataFormatsToString` `DT_ERROR`s in its `default:` arm,
/// `util/sendefs/sendefs.cpp:64-65`) and no bit-width entry.
/// ⛔ `INVALID` IS A LIVE VALUE, though: `FromString` returns it for every unrecognised spelling
/// (`util/sendefs/sendefs.h:294-296`) and the DDL conversion parks it on a node as a transient
/// marker it resolves from the operands' own formats (`ddc/ddl/ddl_conversion.cpp:1438-1468`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DataFormats {
    /// `ComputeNode::dataFormat_`'s initialiser (`dsc/dsc2.h:934`).
    #[default]
    Sen169Fp16,
    IeeeFp32,
    Invalid,
    Sen143Fp8,
    Sen152Fp8,
    Sen153Fp9,
    Senint2,
    Senint4,
    Senint8,
    Senint16,
    Senint24,
    IeeeInt64,
    IeeeInt32,
    Senuint32,
    Senuint2,
    IeeeFp16,
    Bool,
    Bfloat16,
    Sen18fFp24,
    /// For an MX scale (`util/sendefs/sendefs.h:50`).
    Sen080Fp8,
    /// For an MX scale (`util/sendefs/sendefs.h:51`).
    Sen053Fp8,
    /// For MX fp4 (`util/sendefs/sendefs.h:52`).
    Sen121Fp4,
}

/// ⛔ E0080 IF A FORMAT IS EVER INSERTED, DROPPED, REORDERED OR LEFT OUT OF `ALL`: the
/// discriminants order `dataFormatsToBitWidth`'s `std::map` and every other map keyed by this enum.
const _: () = {
    let mut i = 0;
    while i < DataFormats::ALL.len() {
        assert!(
            DataFormats::ALL[i] as usize == i,
            "DataFormats::ALL is out of declaration order"
        );
        i += 1;
    }
};

impl DataFormats {
    /// `NUM_DATA_FORMATS` (`util/sendefs/sendefs.h:53`) — the count of real formats, which is what
    /// the sentinel's discriminant is.
    pub const COUNT: usize = 22;

    /// Every format in the authority's declaration order (`util/sendefs/sendefs.h:30-54`).
    pub const ALL: [Self; Self::COUNT] = [
        Self::Sen169Fp16,
        Self::IeeeFp32,
        Self::Invalid,
        Self::Sen143Fp8,
        Self::Sen152Fp8,
        Self::Sen153Fp9,
        Self::Senint2,
        Self::Senint4,
        Self::Senint8,
        Self::Senint16,
        Self::Senint24,
        Self::IeeeInt64,
        Self::IeeeInt32,
        Self::Senuint32,
        Self::Senuint2,
        Self::IeeeFp16,
        Self::Bool,
        Self::Bfloat16,
        Self::Sen18fFp24,
        Self::Sen080Fp8,
        Self::Sen053Fp8,
        Self::Sen121Fp4,
    ];

    /// `EnumsConversion::dataFormatsToString` (`util/sendefs/sendefs.cpp:18-67`). Total over the
    /// real formats — all 22 have a `case`.
    pub fn name(self) -> &'static str {
        match self {
            Self::Sen169Fp16 => "SEN169_FP16",
            Self::IeeeFp32 => "IEEE_FP32",
            Self::Invalid => "INVALID",
            Self::Sen143Fp8 => "SEN143_FP8",
            Self::Sen152Fp8 => "SEN152_FP8",
            Self::Sen153Fp9 => "SEN153_FP9",
            Self::Senint2 => "SENINT2",
            Self::Senint4 => "SENINT4",
            Self::Senint8 => "SENINT8",
            Self::Senint16 => "SENINT16",
            Self::Senint24 => "SENINT24",
            Self::IeeeInt64 => "IEEE_INT64",
            Self::IeeeInt32 => "IEEE_INT32",
            Self::Senuint32 => "SENUINT32",
            Self::Senuint2 => "SENUINT2",
            Self::IeeeFp16 => "IEEE_FP16",
            Self::Bool => "BOOL",
            Self::Bfloat16 => "BFLOAT16",
            Self::Sen18fFp24 => "SEN18F_FP24",
            Self::Sen080Fp8 => "SEN080_FP8",
            Self::Sen053Fp8 => "SEN053_FP8",
            Self::Sen121Fp4 => "SEN121_FP4",
        }
    }

    /// `FromString<DataFormats>` (`util/sendefs/sendefs.h:250-297`), which the DDL conversion parses
    /// every stated type with (`ddc/ddl/ddl_conversion.cpp:459`).
    ///
    /// ⛔ TOTAL, AND AN UNKNOWN SPELLING IS `INVALID` RATHER THAN ABSENT — the authority's own
    /// `else` (`util/sendefs/sendefs.h:294-296`). That is the value the DDL conversion then resolves
    /// from the operands, so a typo in a DDL type reads as "not stated yet", not as a refusal.
    pub fn from_name(name: &str) -> Self {
        Self::ALL
            .into_iter()
            .find(|format| format.name() == name)
            .unwrap_or(Self::Invalid)
    }

    /// `EnumsConversion::dataFormatsToBitWidth` (`util/sendefs/sendefs.cpp:129-141`).
    ///
    /// ⛔ `SENINT24` IS 16 BITS IN THE AUTHORITY'S TABLE (`:135`), not 24. Every caller reads the
    /// table, so this reproduces the table.
    /// ⛔ `INVALID`'S ENTRY IS `-1` (`:131`) — not a width, so it is ABSENT here. FOUR readers in
    /// `dsc/dsc2.cpp` alone divide by it: `getComputeOperandSizes` twice (`:2295`, `:2310`),
    /// `getBlockTransferSizePerDim`'s constant transfer (`:3490`) and the pad unpacker (`:5261`), and
    /// `replicationFactor_` divides by a CONSTANT's width outside this file (`ddc/ddcv1.cpp:457`). On
    /// `INVALID` each yields a NEGATIVE count in the authority rather than refusing, and a negative
    /// operand size is not one — see [`ComputeNode::operand_sizes`] and [`ConstantInfo::data_format`].
    pub fn bit_width(self) -> Option<BitWidth> {
        let bits = match self {
            Self::Invalid => return None,
            Self::Senint2 | Self::Senuint2 => 2,
            Self::Senint4 | Self::Sen121Fp4 => 4,
            Self::Sen143Fp8
            | Self::Sen152Fp8
            | Self::Senint8
            | Self::Bool
            | Self::Sen080Fp8
            | Self::Sen053Fp8 => 8,
            Self::Sen153Fp9 => 9,
            Self::Sen169Fp16
            | Self::Senint16
            | Self::Senint24
            | Self::Bfloat16
            | Self::IeeeFp16 => 16,
            Self::Sen18fFp24 => 24,
            Self::IeeeFp32 | Self::IeeeInt32 | Self::Senuint32 => 32,
            Self::IeeeInt64 => 64,
        };
        Some(BitWidth(bits))
    }
}
/// Replaces: ComputeNode::InstrAttribute
///
/// `dsc/dsc2.h:905-930`. The instruction-level attributes of one compute node: what the DDL states
/// verbatim about the instruction word (`ddc/ddl/ddl_conversion.cpp:1364-1383`) plus the three
/// opaque alias maps an OPAQUE op carries through to the emitter.
///
/// ⛔ THE MAPS ARE ORDERED, AND THAT IS OBSERVABLE: the JSON exporter walks all three in map order
/// (`dsc/dsc2.cpp:136-167`) and bridge 1 turns them into a `DictionaryAttr`
/// (`dsc-based-utils/DSC2ToDataflowIR/V3/SNComputeLowering.cpp:1542-1550`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InstrAttribute {
    /// Field: e035_ComputeNode.indices_
    ///
    /// The PACK/MERGE mapping (`dsc/dsc2.h:906`), one entry per slot of the instruction word. The
    /// LENGTH is load-bearing: `expand_indices` derives each entry's width from it as
    /// `128 / indices.size()` (`ddc/ddc_transformation.cpp:1941-1942`).
    ///
    /// ⛔ NOT `Option`, THOUGH `dsc/dsc2.h:903` INVITES IT. That comment — "-1 for zero/sign
    /// extend" — says what the hardware does with the slot, and NOTHING IN THE TREE BRANCHES ON -1:
    /// all five readers emit the entry verbatim as an integer (`getI32ArrayAttr` at
    /// `dsc-based-utils/DSC2ToDataflowIR/V3/SNComputeLowering.cpp:1126` and `:1495`,
    /// `getI64IntegerAttr` at `ddc/ddl/ddl_conversion.cpp:3136`, the JSON exporter at
    /// `dsc/dsc2.cpp:126` and its importer at `:1176`, and `dsc/dsc2Pcfg.cpp:1640`). The one writer
    /// that computes rather than copies multiplies the sentinel like any other index, turning one -1
    /// into -2, -1 at scale 2 (`ddc/ddc_transformation.cpp:1952`) — and a hole cannot expand into
    /// two unequal holes. See [`PackMergeIndex`].
    pub indices: Vec<PackMergeIndex>,
    /// Field: e035_ComputeNode.repetition_
    ///
    /// `dsc/dsc2.h:907` — "default 8 slices works the same".
    pub repetition: Repetition,
    /// Field: e035_ComputeNode.sign_extend_
    ///
    /// Whether a PACK/MERGE extends signed (`dsc/dsc2.h:908`), read as a bool attribute by bridge 1
    /// (`dsc-based-utils/DSC2ToDataflowIR/V3/SNComputeLowering.cpp:1124`).
    pub sign_extend: bool,
    /// Field: e035_ComputeNode.read_write_reg_map_
    ///
    /// An OPAQUE op's read/write register alias map (`dsc/dsc2.h:909-910`); `Ddc::finalizeOps` sizes
    /// its register window from it (`ddc/ddcv1.cpp:3376-3377`).
    pub read_write_reg_map: BTreeMap<String, String>,
    /// Field: e035_ComputeNode.read_only_reg_map_
    ///
    /// The read-only half of the same (`dsc/dsc2.h:911-912`, `ddc/ddcv1.cpp:3391`).
    pub read_only_reg_map: BTreeMap<String, String>,
    /// Field: e035_ComputeNode.param_map_
    ///
    /// An OPAQUE op's parameter alias map (`dsc/dsc2.h:913-914`). ⛔ THE SCHEDULER WRITES INTO IT:
    /// `Ddc::finalizeOps` sets `"unroll"` and `"prec"` (`ddc/ddcv1.cpp:3343`, `:3395-3397`).
    pub param_map: BTreeMap<String, String>,
    /// Field: e035_ComputeNode.mode_
    ///
    /// `dsc/dsc2.h:915`. `-1` IS ABSENT TO EVERY BRANCHING READER — the DDL writes the field only
    /// when it states one (`ddc/ddl/ddl_conversion.cpp:1368-1370`), and each bridge-1 reader tests
    /// it against a specific non-negative encoding
    /// (`dsc-based-utils/DSC2ToDataflowIR/V3/SNComputeLowering.cpp:1102`, `:1317-1385`, `:1454-1462`),
    /// which -1 never is. ⛔ IT IS STILL AN INTEGER ON THE WIRE, so `None` MUST re-serialize as -1
    /// and not be skipped — unlike [`Self::indices`], where the same "-1 means absent" reading is
    /// wrong outright. See [`Mode`] for both wire writers.
    pub mode: Option<Mode>,
    /// Field: e035_ComputeNode.compute_mask_
    ///
    /// `dsc/dsc2.h:916`, all eight slices unless the DDL states a mask
    /// (`ddc/ddl/ddl_conversion.cpp:1371-1373`).
    pub compute_mask: ComputeMask,
    // crustify:todo: e035_ComputeNode.computeMaskLoopOffsets_
    /// Field: e035_ComputeNode.input_data_connects_
    ///
    /// One name per input of an OPAQUE op (`dsc/dsc2.h:926-927`), index-parallel with
    /// [`ComputeNode::inputs`] where the fold pass reads them together
    /// (`ddc/ddc_fold.cpp:1632`, `:1856`).
    pub input_data_connects: Vec<DataConnect>,
    /// Field: e035_ComputeNode.output_data_connects_
    ///
    /// The output half of the same (`dsc/dsc2.h:928-929`, `ddc/ddc_fold.cpp:4202-4226`).
    pub output_data_connects: Vec<DataConnect>,
}

impl Default for InstrAttribute {
    /// The authority's initialisers (`dsc/dsc2.h:906-929`): eight slices of repetition, all eight
    /// compute slices, no mode, and every collection empty.
    fn default() -> Self {
        Self {
            indices: Vec::new(),
            repetition: Repetition(8),
            sign_extend: false,
            read_write_reg_map: BTreeMap::new(),
            read_only_reg_map: BTreeMap::new(),
            param_map: BTreeMap::new(),
            mode: None,
            compute_mask: ComputeMask(255),
            input_data_connects: Vec::new(),
            output_data_connects: Vec::new(),
        }
    }
}

/// Replaces: ComputeNode::RepetitionWithOffset
///
/// `dsc/dsc2.h:950-953`. How many times each operand repeats with an offset, one entry per operand.
///
/// ⛔ INDEX-PARALLEL WITH THE OPERAND LISTS, NOT A MAP: the DDL conversion pushes an entry beside
/// every `inputs_`/`outputs_` entry it appends (`ddc/ddl/ddl_conversion.cpp:1385-1406`), and both
/// readers index it with an operand position (`ddc/ddcv1.cpp:3059`,
/// `ddc/ddc_transformation.cpp:1358-1379`).
///
/// ⛔ AND BOTH OF THOSE READERS READ `forOutputs_`. `forInputs_` HAS NO READER AT ALL: tree-wide its
/// only mention outside the declaration is the DDL's `push_back`
/// (`ddc/ddl/ddl_conversion.cpp:1393-1394`), so it is carried because the node declares it, not
/// because a pass consults it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RepetitionWithOffset {
    /// Field: e035_ComputeNode.forInputs_
    ///
    /// `dsc/dsc2.h:951`. ⛔ WRITTEN AND NEVER READ — see the type's note above.
    pub for_inputs: Vec<Repetition>,
    /// Field: e035_ComputeNode.forOutputs_
    ///
    /// `dsc/dsc2.h:952`. ⛔ A SPREAD THE TRANSFORMATION CONSUMES: it clones the node
    /// `for_outputs[idx] - 1` further times and writes 1 back into the clone
    /// (`ddc/ddc_transformation.cpp:1358-1379`).
    pub for_outputs: Vec<Repetition>,
}

/// `ComputeNode::CoreletView` (`dsc/dsc2.h:943-947`) — what ONE corelet sees of this instruction's
/// operands. Named for its owner because [`TransferNode`]'s nested `CoreletView` (`:847-851`) is a
/// different type with different members.
///
/// ⛔ BOTH VECTORS ARE INDEX-PARALLEL WITH THE OPERAND LIST THEY DESCRIBE, and bridge 1 depends on
/// that: it reads `inputsLoopsAndSizes_[index]` for input `index`
/// (`dsc-based-utils/DSC2ToDataflowIR/V3/SNComputeLowering.cpp:552-553`, `:571-572`) — EXCEPT that an
/// input arriving on `NFWD0`/`NFWD2` reads `outputsLoopsAndSizes_.front()` instead (`:548-550`,
/// `:568-570`), a forwarded operand being the output's own view.
#[derive(Clone, Debug, Default)]
pub struct ComputeCoreletView {
    /// Field: e035_ComputeNode.inputsLoopsAndSizes_
    ///
    /// One view per entry of [`ComputeNode::inputs`], built from that input's own `DataInfo`
    /// (`dsc/dsc2.cpp:3021-3025`).
    pub inputs_loops_and_sizes: Vec<UnitView>,
    /// Field: e035_ComputeNode.outputsLoopsAndSizes_
    ///
    /// One view per entry of [`ComputeNode::outputs`], on the same terms (`dsc/dsc2.cpp:3026-3031`).
    pub outputs_loops_and_sizes: Vec<UnitView>,
}

/// Replaces: e035_ComputeNode
///
/// `dsc/dsc2.h:900-962`. One compute instruction in the schedule tree: the unit it issues on, the
/// op, the operand components and the instruction attributes.
///
/// `e024_ComputeNode` is this same class under the superseded numbering; its 21 filled field anchors
/// are RENUMBERED onto e035 here, not deleted — each has an e035 counterpart. It came back for the
/// reason `0aa459e2e` records: `plan.py` matches `/// Replaces: (e\d{3}_[A-Za-z0-9_]+)` on the class
/// name, so a port emitting only `/// Field:` anchors is never in the set.
///
/// ⛔ THIS CARRIES COMPUTENODE'S OWN DECLARED FIELDS AND NOTHING INHERITED — IBM derives it from
/// `InheritWithClone<ScheduleNode, ComputeNode>` and its constructor tags the base with `COMPUTE`
/// (`dsc/dsc2.h:900-901`), and the base's thirteen fields are e029's. ONE field anchor stays OPEN:
/// `instrAttribute_.computeMaskLoopOffsets_` below, which is a TREE fact, not a node fact.
/// ⭐ `inputCoordinates_` AND `outputCoordinate_` (`:948-949`) ARE CARRIED NOW. They were blocked on
/// `CoordinateType` — `e023_CoordinateType` under the numbering `port.json` used, `e012` in the
/// anchors this file carried, and `e011` in the one that filled it — which is
/// `std::map<PrimaryDimTypes, FoldManager<Dtype>>` (`dsc/dsc2.h:431`) and needed e026_FoldManager
/// first. Both have landed in `src/schedule/dsc2.rs` and `src/schedule/fold.rs`.
/// ⭐ `coreletViews_` AND ITS TWO SEPARATELY ANCHORED HALVES ARE PORTED HERE and were not portable
/// when e024 ran: `ScheduleNode::UnitView` (`:943-947`) landed with e029 in `625e761da`.
/// ⭐ AND `inputsLdsAndLoopOffsets_` / `outputsLdsAndLoopOffsets_` (`:937-938`) ARE CARRIED NOW: the
/// `std::vector<DataInfo>` that blocked them is ported as e033_DataInfo in this same changeset.
///
/// ⛔ AND `instrAttribute_.computeMaskLoopOffsets_` IS KEYED BY A `const LoopNode*` WHOSE
/// KEY MAY BE NULL (`:923-925`), which is a TREE fact and not a node fact. Its wire form keys by the
/// loop's `ScheduleNode::name_` — the exporter substitutes `""` for a null key and re-sorts that
/// level by name (`dsc/dsc2.cpp:172-190`), the importer resolves it through a `nodeNamePtrMap` seeded
/// `{"", nullptr}` (`:1198-1205`, `:1369`) — and `name_` has landed, so the blocker is no longer the
/// name. It stays open because the CHOICE is the campaign's and e029 made it the other way: that port
/// left `e029_ScheduleNode.loop_`, the `const LoopNode*` inside [`LoopInfo`], open rather than
/// substituting a name, and this key must agree with that one. The pointer is load-bearing at the far
/// end rather than mere identity: bridge 1 hands it to `getMLIRLoopFromLoopNode` for the enclosing
/// MLIR loop's induction variable and refuses more than one entry
/// (`dsc-based-utils/DSC2ToDataflowIR/V3/SNComputeLowering.cpp:60-80`, from `:997-1008`).
///
/// ⛔ `print` IS THE ONE METHOD LEFT OUT, and what blocks it is now inside [`DataInfo`] rather than
/// `DataInfo` itself: it prints the base's `name_`, which has landed, and then each `DataInfo`'s own
/// `print` (`dsc/dsc2.cpp:4443-4477`), which walks `loopEleOffsets_` — dereferencing every
/// `const LoopNode*` key for its `name_` — and then `bufferSwitchPosition_->print`
/// (`dsc/dsc2.cpp:4335-4356`). Those are two of the three fields e033_DataInfo left open; the third,
/// `startAddr_`, is not a blocker at all, because its section of that function is COMMENTED OUT
/// (`:4315-4321`).
///
/// ⚠️ AND ELEVEN OF THE 24 FILLED ANCHORS NAME A FIELD THE e035 LIST DOES NOT CARRY, so the
/// scheduled count undercounts this class: `instrAttribute_`, and all ten `InstrAttribute` members
/// but `computeMaskLoopOffsets_` (`:906-929`). No rule for that is derived here — `type_`, `inputs_`
/// and `outputs_` are listed and equally plain. They keep anchors because the fields are carried.
///
/// ⛔ NO `PartialEq`: node identity in the authority is the POINTER. `AllocateNode::allocUsers_` is
/// a `std::vector<std::pair<const ScheduleNode*, int>>` and all three of its operations match a user
/// with `node == userNode`, an ADDRESS compare (`dsc/dsc2.h:1007`, `:1014`, `:1023`, `:1039`); the
/// fold pass then walks that pointer-keyed list (`ddc/ddc_fold.cpp:1688`).
#[derive(Clone, Debug)]
pub struct ComputeNode {
    /// The `ScheduleNode` subobject (`dsc/dsc2.h:900`, `InheritWithClone<ScheduleNode, ComputeNode>`),
    /// tagged `COMPUTE` by `ComputeNode()` (`:901`). A compute is a LEAF.
    pub base_class: ScheduleNode,
    /// Field: e035_ComputeNode.exUnit_
    ///
    /// The execution unit the instruction issues on (`dsc/dsc2.h:932`). ⛔ A COMPUTE WHOSE OWN
    /// `exUnit_` APPEARS IN ITS OPERANDS IS ILLEGAL DDL (`ddc/ddl/ddl_conversion.cpp:1478-1487`).
    pub ex_unit: SenComponent,
    /// Field: e035_ComputeNode.type_
    ///
    /// The op (`dsc/dsc2.h:933`). Its `COUNT` initialiser means "not chosen yet"; the DDL conversion
    /// always overwrites it (`ddc/ddl/ddl_conversion.cpp:1410-1437`).
    pub r#type: ComputeOpType,
    /// Field: e035_ComputeNode.dataFormat_
    ///
    /// The precision the op runs at (`dsc/dsc2.h:934`). ⛔ IT IS THE OP'S PRECISION FOR `MACC`
    /// ALONE, which is why [`Self::operand_sizes`] dispatches on it for that one op: a `"macc"` in
    /// the DDL picks the FMA/IMA variant from the stated precision and then keeps it
    /// (`ddc/ddl/ddl_conversion.cpp:1410-1430`).
    ///
    /// ⛔ FOR EVERY OTHER OP IT IS ONE OF EXACTLY TWO VALUES, NOT WHAT THE OPERANDS HOLD. The DDL
    /// does scan the operands' labelled-DS formats (`:1441-1466`) and then THROWS THE SCANNED VALUE
    /// AWAY: `dataFormat_` is left `IEEE_FP32` if that is what it reached and overwritten with
    /// `SEN169_FP16` otherwise (`:1467-1468`). So `INVALID` never survives onto a DDL-minted node,
    /// and the `1024 / bit_width` default in [`Self::operand_sizes`] only ever divides by 32 or 16
    /// there.
    pub data_format: DataFormats,
    /// Field: e035_ComputeNode.inputs_
    ///
    /// Where each input comes from (`dsc/dsc2.h:935`), pushed in lockstep with
    /// `inputsLdsAndLoopOffsets_` and `repetitionWithOffset_.forInputs_`
    /// (`ddc/ddl/ddl_conversion.cpp:1385-1395`).
    pub inputs: Vec<SenComponent>,
    /// Field: e035_ComputeNode.outputs_
    ///
    /// Where each output goes (`dsc/dsc2.h:936`), on the same terms
    /// (`ddc/ddl/ddl_conversion.cpp:1396-1407`).
    pub outputs: Vec<SenComponent>,
    /// Field: e035_ComputeNode.inputsLdsAndLoopOffsets_
    ///
    /// One [`DataInfo`] per input, pushed in lockstep with [`inputs`](Self::inputs) and
    /// `repetitionWithOffset_.forInputs_` (`dsc/dsc2.h:937`,
    /// `ddc/ddl/ddl_conversion.cpp:1385-1395`).
    ///
    /// ⛔ THE FOLD PASS INDEXES THIS AND [`inputs`](Self::inputs) WITH THE SAME `i`
    /// (`ddc/ddc_fold.cpp:170-172`, `:2044`, `:2089`), and the latch rewrite writes both halves in one
    /// walk (`ddc/ddc_transformation_util.cpp:995-1010`) — so the two vectors are parallel and a
    /// reader that shortens one has broken the other.
    pub inputs_lds_and_loop_offsets: Vec<DataInfo>,
    /// Field: e035_ComputeNode.outputsLdsAndLoopOffsets_
    ///
    /// One [`DataInfo`] per output, on the same terms (`dsc/dsc2.h:938`,
    /// `ddc/ddl/ddl_conversion.cpp:1396-1407`).
    ///
    /// ⛔ ENTRY 0 IS THE ONE `getComputeOperandFormats` READS FOR A PACKMERGE
    /// (`dsc/dsc2.cpp:2348-2357`), and it reads it as `labeledDs_.at(myLdsIdx_).dataFormat_` — which
    /// is why [`operand_formats`](Self::operand_formats) takes the resolved format as its argument:
    /// `DesignSpaceConfig::labeledDs_`, not this field, is the hop still missing.
    pub outputs_lds_and_loop_offsets: Vec<DataInfo>,
    /// Field: e035_ComputeNode.instrAttribute_
    ///
    /// `dsc/dsc2.h:939`.
    pub instr_attribute: InstrAttribute,
    /// Field: e035_ComputeNode.numFoldsEngaged
    ///
    /// `dsc/dsc2.h:940`. ⛔ IT SCALES EVERY OPERAND SIZE (`dsc/dsc2.cpp:2333`, `:2344`); `Ddc` sets
    /// it from the unit's fold count (`ddc/ddcv1.cpp:1887`).
    pub num_folds_engaged: NumFoldsEngaged,
    /// Field: e035_ComputeNode.isOpaqueOp_
    ///
    /// Whether the DDL supplied the instruction verbatim (`dsc/dsc2.h:941`). ⛔ THE FOLD AND
    /// TRANSFORMATION PASSES BRANCH ON IT before reading the data connects
    /// (`ddc/ddc_fold.cpp:1630`, `:1853`, `ddc/ddc_transformation_util.cpp:1466`).
    pub is_opaque_op: bool,
    /// Field: e035_ComputeNode.coreletViews_
    ///
    /// One [`ComputeCoreletView`] per corelet (`dsc/dsc2.h:948`), filled for every corelet in
    /// `0..numCoreletsUsed_DSC2_` by `finalizeScheduleTree` (`dsc/dsc2.cpp:3019-3032`).
    ///
    /// ⛔ KEYED BY CORELET ALONE, AND EVERY CORELET'S VIEWS ARE BUILT FROM ONE CORE: the writer
    /// iterates `coreIdsUsed_.front()` only, and says why — "do not insert view for each core until
    /// we have expanded coreletViews to coreCoreletViews" (`dsc/dsc2.cpp:3015-3018`), an assumption
    /// bridge 1 restates where it takes `.begin()` for the uniform case
    /// (`dsc-based-utils/DSC2ToDataflowIR/V3/SNComputeLowering.cpp:546-547`). The per-core answer
    /// comes from [`UnitView::sizes_for_core`] inside the view, not from a second key.
    pub corelet_views: BTreeMap<CoreletId, ComputeCoreletView>,
    /// Field: e035_ComputeNode.inputCoordinates_
    ///
    /// How each input operand's dims are folded, one coordinate per input (`dsc/dsc2.h:948`) —
    /// parallel to [`inputs`](Self::inputs) on the same index.
    ///
    /// ⛔ THE PROPAGATION READS THIS AND WRITES [`output_coordinate`](Self::output_coordinate), PER
    /// LEVEL AND PER CATEGORY: `buildFoldForComputeNode` takes the input tower's category at each
    /// position with `getFoldCategory` and adds the matching level to the output
    /// (`ddc/ddc_fold.cpp:4300-4370`), which is the one path on which
    /// [`CoordinateType::fold_category`]'s `UNKNOWN_COORD` can be reached — see its note.
    pub input_coordinates: Vec<CoordinateType>,
    /// Field: e035_ComputeNode.outputCoordinate_
    ///
    /// How the output operand's dims are folded (`dsc/dsc2.h:949`).
    ///
    /// ⛔ SINGULAR WHERE [`input_coordinates`](Self::input_coordinates) IS A VECTOR, and the
    /// authority's field name says so: one coordinate however many entries
    /// [`outputs`](Self::outputs) has.
    pub output_coordinate: CoordinateType,
    /// Field: e035_ComputeNode.repetitionWithOffset_
    ///
    /// `dsc/dsc2.h:954`.
    pub repetition_with_offset: RepetitionWithOffset,
}

impl Default for ComputeNode {
    /// The authority's default member initializers (`dsc/dsc2.h:932-941`, `:950-954`). ⭐ AND IT IS
    /// `ComputeNode()`, BASE TAG INCLUDED: that constructor passes `COMPUTE` to `ScheduleNode`
    /// (`dsc/dsc2.h:901`).
    fn default() -> Self {
        Self {
            base_class: ScheduleNode::new(NodeType::Compute),
            ex_unit: SenComponent::NoComponent,
            r#type: ComputeOpType::Count,
            data_format: DataFormats::Sen169Fp16,
            inputs: Vec::new(),
            outputs: Vec::new(),
            inputs_lds_and_loop_offsets: Vec::new(),
            outputs_lds_and_loop_offsets: Vec::new(),
            instr_attribute: InstrAttribute::default(),
            num_folds_engaged: NumFoldsEngaged(1),
            is_opaque_op: false,
            corelet_views: BTreeMap::new(),
            repetition_with_offset: RepetitionWithOffset::default(),
            input_coordinates: Vec::new(),
            output_coordinate: CoordinateType::default(),
        }
    }
}

impl ComputeNode {
    /// `dsc/dsc2.cpp:2291-2346`. The element count of every operand, inputs in order and the OUTPUT
    /// LAST (`dsc/dsc2.h:956-957`).
    ///
    /// It takes the core generation rather than the `const SenSystemDef&` of the declaration
    /// (`dsc/dsc2.h:958`), because `sysDef.coreArch` is the only thing the body reads
    /// (`sys-arch-spec/sysdef.h:93`), and [`Gen`] is `IsaCoreGen` already ported
    /// (`sys-arch-spec/isa/isa.hpp:25-32`).
    ///
    /// ⛔ ABSENT WHERE THE AUTHORITY REFUSES, and it refuses on a PTWEST input twice: `MACC` is
    /// "not correctly handled at the moment" and anything outside the five FMA/IMA ops is an
    /// "Unexpected PT operation" (`dsc/dsc2.cpp:2308-2312`).
    /// ⛔ AND ABSENT ON AN `INVALID` FORMAT, where the authority divides 1024 by the -1 in
    /// `dataFormatsToBitWidth` and carries -1024 elements forward (`:2295`) — see
    /// [`DataFormats::bit_width`]. Nothing downstream can use a negative operand size, and the DDL
    /// conversion resolves `INVALID` away before a node is scheduled
    /// (`ddc/ddl/ddl_conversion.cpp:1442-1468`).
    pub fn operand_sizes(&self, core_arch: Gen) -> Option<Vec<OperandSize>> {
        let up_to_rcudd1a = core_arch <= Gen::Rcudd1a;
        let mut sizes = Vec::new();
        for input in &self.inputs {
            let size = match input {
                // `dsc/dsc2.cpp:2297-2312`.
                SenComponent::Ptwest => match self.r#type {
                    ComputeOpType::Fma16 => {
                        if up_to_rcudd1a {
                            8
                        } else {
                            32
                        }
                    }
                    ComputeOpType::Fma8 => {
                        if up_to_rcudd1a {
                            16
                        } else {
                            128
                        }
                    }
                    ComputeOpType::Ima8 => {
                        if up_to_rcudd1a {
                            32
                        } else {
                            128
                        }
                    }
                    // Only available from SEN1P5.
                    ComputeOpType::Fma4 => 256,
                    ComputeOpType::Ima4 => {
                        if up_to_rcudd1a {
                            64
                        } else {
                            256
                        }
                    }
                    _ => return None,
                },
                // `:2313-2314`.
                SenComponent::Ptnorth => 64,
                // `:2315-2332`. PELRF and SFPLRF are memories that this branch excludes, so they
                // take the default below.
                input
                    if MEMORIES.contains(input)
                        && !matches!(input, SenComponent::Pelrf | SenComponent::Sfplrf) =>
                {
                    match (self.r#type, self.data_format) {
                        (ComputeOpType::Fma8, _)
                        | (ComputeOpType::Macc, DataFormats::Sen143Fp8 | DataFormats::Sen152Fp8) => {
                            if up_to_rcudd1a {
                                128
                            } else {
                                1024
                            }
                        }
                        (ComputeOpType::Ima8, _) | (ComputeOpType::Macc, DataFormats::Senint8) => {
                            if up_to_rcudd1a {
                                256
                            } else {
                                1024
                            }
                        }
                        (ComputeOpType::Ima4, _) | (ComputeOpType::Macc, DataFormats::Senint4) => {
                            if up_to_rcudd1a {
                                512
                            } else {
                                2048
                            }
                        }
                        (ComputeOpType::Fma16, _)
                        | (ComputeOpType::Macc, DataFormats::Sen169Fp16) => {
                            if up_to_rcudd1a {
                                64
                            } else {
                                256
                            }
                        }
                        // Only available from SEN1P5.
                        (ComputeOpType::Fma4, _)
                        | (ComputeOpType::Macc, DataFormats::Sen121Fp4) => 2048,
                        // The default for fp16/int24, `:2295-2296`.
                        _ => 1024 / self.data_format.bit_width()?.0,
                    }
                }
                _ => 1024 / self.data_format.bit_width()?.0,
            };
            sizes.push(OperandSize(size * self.num_folds_engaged.0));
        }

        // `:2337-2344`: all reduced-precision ops produce fp16, so the output is 64 elements unless
        // the op runs in fp32.
        let output = if self.data_format == DataFormats::IeeeFp32 {
            32
        } else {
            64
        };
        sizes.push(OperandSize(output * self.num_folds_engaged.0));
        Some(sizes)
    }

    /// `dsc/dsc2.cpp:2348-2396`. Every operand's FORMAT, inputs in order and the output LAST.
    ///
    /// ⭐ THE PARAMETER IS IBM'S `const DesignSpaceConfig&` NARROWED TO THE ONE VALUE THE BODY READS
    /// OUT OF IT — `dsc.labeledDs_.at(outputsLdsAndLoopOffsets_.at(0).myLdsIdx_).dataFormat_`
    /// (`:2352-2353`), and only for `PACKMERGE` — the same narrowing
    /// [`operand_sizes`](Self::operand_sizes) performs on `const SenSystemDef&`. It is what keeps
    /// this method WHOLE, and only ONE of that lookup's two hops is unported now: the first is here
    /// as [`outputs_lds_and_loop_offsets`](Self::outputs_lds_and_loop_offsets)`[0].lds_or_const`,
    /// since e033_DataInfo lands in this changeset, but
    /// [`DesignSpaceConfig`](crate::schedule::dsc::DesignSpaceConfig) carries no `labeledDs_` to
    /// resolve the format with.
    /// ⛔ SO THE ARGUMENT IS IGNORED ON EVERY OTHER OP, exactly as IBM's `dsc` is.
    /// ⛔ AND THE TWO THROWS THAT LOOKUP CARRIES GO WITH IT: `outputsLdsAndLoopOffsets_.at(0)` on an
    /// empty vector and `labeledDs_.at()` on an unregistered index both refuse (`:2352-2353`), and a
    /// resolved `DataFormats` cannot express either — so whoever lands `labeledDs_` inherits them.
    ///
    /// ⛔ FOUR INPUTS YIELD FEWER FORMATS THAN THREE (`:2372-2373`, `:2392-2393`), and every caller
    /// indexes by operand position (`.../V3/SNComputeLowering.cpp:766`, `:789`, `:1058`, `:1207`,
    /// `:1291`, `:1484`), so a fourth input reads the OUTPUT's format as its own in the authority.
    /// ⛔ AND `FMA16` IS IN THE ACCUMULATOR CHAIN BUT NOT THE INPUT ONE (`:2360-2371`, `:2385`), so
    /// its inputs stay at `dataFormat_`. That accumulator is the DDL's `%ptsum_fp`/`%ptsum_int` pair
    /// seen from the C++ side (`ddc/ddl_templates/bmm.ddl:47-48`, aliased at `:107`).
    pub fn operand_formats(&self, packmerge_output_format: DataFormats) -> Vec<DataFormats> {
        // `:2351-2357`. The one branch that consults the design space config, and the one that does
        // not answer per operand: three copies whatever the operand count is.
        if self.r#type == ComputeOpType::Packmerge {
            return vec![packmerge_output_format; 3];
        }

        // `:2360-2371`.
        let input_format = match self.r#type {
            ComputeOpType::Fma8 => DataFormats::Sen143Fp8,
            ComputeOpType::Fma4 => DataFormats::Sen121Fp4,
            ComputeOpType::Ima8 => DataFormats::Senint8,
            ComputeOpType::Ima4 => DataFormats::Senint4,
            ComputeOpType::Fma32 => DataFormats::IeeeFp32,
            _ => self.data_format,
        };

        // `:2377-2391`, in the authority's order — a `match` is first-arm-wins like its `else if`
        // chain, so `IMA8`/`IMA4` reach `SENINT24` before the FMA arm can claim them.
        let accumulator_format = match (self.r#type, self.data_format) {
            (ComputeOpType::Ima8 | ComputeOpType::Ima4, _)
            | (ComputeOpType::Macc, DataFormats::Senint8 | DataFormats::Senint4) => {
                DataFormats::Senint24
            }
            (ComputeOpType::Fma32, _) | (ComputeOpType::Macc, DataFormats::IeeeFp32) => {
                DataFormats::IeeeFp32
            }
            (ComputeOpType::Fma4 | ComputeOpType::Fma8 | ComputeOpType::Fma16, _)
            | (
                ComputeOpType::Macc,
                DataFormats::Sen121Fp4
                | DataFormats::Sen143Fp8
                | DataFormats::Sen152Fp8
                | DataFormats::Sen169Fp16
                | DataFormats::Bfloat16,
            ) => DataFormats::Sen169Fp16,
            _ => self.data_format,
        };

        // `:2372-2393`.
        let mut formats = vec![input_format; self.inputs.len().min(2)];
        if self.inputs.len() == 3 {
            formats.push(accumulator_format);
        }
        formats.push(accumulator_format);
        formats
    }
}

/// Replaces: e044_ConditionNode
///
/// `dsc/dsc2.h:685-719`. A two-way branch in the schedule tree: a `BlockNode` whose at most two
/// children are the "then" and the "else" region (`:688`, `:698-699`) — in THAT order, because
/// `getThenBranchNode` is `next_[0]` and `getElseBranchNode` is `next_[1]` (`:707-718`).
///
/// `e025_ConditionNode` is this same class under the superseded numbering, and e025 is now `Ddc`
/// (`crustify-scheduler/UNITS.tsv:26`) — a COLLISION, not a merely stale number. Both of its filled
/// anchors and its open type anchor are RENUMBERED onto e044 here (`UNITS.tsv:45`); none is deleted
/// and none changes meaning.
///
/// ⭐ AND BOTH GUARDS NOW LAND, BECAUSE THE BLOCK WAS ON THE TERM AND NOT ON THIS FIELD: `loopCond_`
/// (`:690`) is [`LoopCondComposite`], which landed carrying its value half with `loopComp_` still
/// open on [`LoopCond`] — schedule-node pointer identity, recorded there, on the type that is
/// missing the pointer. Carrying the composite here adds no gap of its own.
///
/// ⭐ WHAT THIS TYPE ADDS IS THE [`Option`], AND THAT IS THE DISCRIMINATOR ITSELF: `hasCoreClCond()`
/// IS `loopCond_.twoLevelOrOfAnds_.empty()` (`:693-695`) and never looks at `coreClCond_`, and a
/// landed composite cannot BE empty ([`LoopCondDisjunction`] is non-empty by shape), so the absence
/// of one spells that predicate exactly — see [`has_core_cl_cond`](Self::has_core_cl_cond).
///
/// ⛔ `negated_` IS A PARITY TOGGLE AT BOTH WRITERS AND IS NEVER SET: `^= true` when the else branch
/// carries the condition (`ddc/ddc_transformation_util.cpp:592`) and `= !` under a `condNot`
/// (`ddc/ddl/ddl_conversion.cpp:326`). ⭐ THAT SECOND ONE IS GUARDED BY THE SAME DISCRIMINATOR AS
/// `hasCoreClCond()` — `!twoLevelOrOfAnds_.empty()` (`:325`) — so a `condNot` over an EMPTY
/// composite negates `coreClCond_` instead and leaves this flag alone. A port that toggles
/// unconditionally inverts a core/corelet condition twice.
///
/// ⛔ AND THE SPLIT DISPATCH IS CLOSED, WITH FOUR REFUSALS, none of which is a branch a caller can
/// take: an empty `newLoops` (`dsc/dsc2.cpp:2064`), a `condValType_` outside `FIRST`/`LAST`
/// (`:2082-2084`), the four always-true/always-false pairings `(GT,LAST)`, `(LT,FIRST)`,
/// `(LE,LAST)`, `(GE,FIRST)` (`:2087-2099`), and anything outside `EQ` / `NE` / `(GT,FIRST)` /
/// `(LT,LAST)` (`:2136`) — which is `(LE,FIRST)` and `(GE,LAST)` alone, both of them equalities, as
/// [`LoopCondComposite`] records. The first two are total over closed enums and belong in the type;
/// the always-true/false four are a `(CondOp, CondValType)` PAIR, and the authority's own comment
/// says the DDL parser was supposed to simplify them away (`:2097-2098`), so they are a guard on the
/// pairing and not on either enum alone.
///
/// ⚠️ AND THE HEADER'S "only one is filled" (`:689`) IS ENFORCED — ON THE OTHER CARRIER, NOT ON THIS
/// NODE. The DDL front end refuses a mixed composition outright, `emitError("And/or op is mixing
/// incompatible types")` then `DT_ERROR` in both directions
/// (`ddc/ddl/ddl_conversion.cpp:369-373`, `:405-409`), and it is that same `DdlInterface::CondProp`
/// — [`LoopCondComposite`]'s second carrier — that it then copies into a fresh node's two fields in
/// consecutive statements (`:1557-1558`). What is unchecked is the JSON importer, which fills both
/// from the wire in one pass with no cross-field test (`dsc/dsc2.cpp:1431-1467`).
/// ⭐ AND IF BOTH DO ARRIVE FILLED, THE LOOP GUARD WINS AND THE MAP GOES SILENTLY DEAD: every reader
/// of `coreClCond_` except the JSON and DDL exporters is gated on `hasCoreClCond()`
/// (`dsc/dsc2.cpp:2663`, `ddc/ddc_transformation_util.cpp:489`, `ddc/ddl/ddl_conversion.cpp:3248`).
///
/// ⭐ AND BOTH-EMPTY IS A CORE/CORELET CONDITION SELECTING NO CORE, not an unconditional region —
/// which the authority says twice: `hasCoreClCond()` answers `true` there, and the DDL resolves that
/// exact pair of empties to the CONSTANT FALSE (`ddc/ddl/ddl_conversion.cpp:438-441`) before any node
/// is minted.
///
/// ⭐ AND ALL EIGHT OF THIS CLASS'S METHODS NOW LAND, BECAUSE `next_` LANDED WITH IT. Each one
/// reaches the child list, and the list is [`BlockNode`]'s: `addChildNode` (the `override` that
/// refuses anything but a `BLOCK` and any third child, `dsc/dsc2.cpp:2143-2150`), `addThenRegion` and
/// `addElseRegion` (one refusal and two more, `:2152-2155`, `:2157-2167`), `getThenBranchNode`,
/// `getElseBranchNode`, `getThenCoreCl`, `getElseCoreCl` and `getNextView`, which widens the base
/// view to `ALL, -1, -1` whenever the guard is a loop condition (`:1995-2001`).
///
/// ⛔ AND THE HEADER'S "max 2 children in next_, of type BLOCK" (`dsc/dsc2.h:687-688`) IS A TYPE HERE
/// RATHER THAN A COMMENT: [`Self::add_child_node`] takes a [`BlockNode`] BY VALUE, and it is the only
/// mutable route to this node's children — [`ChildNode::as_block_mut`] refuses a condition node
/// precisely so that `BlockNode`'s own non-virtual `moveChildren` and
/// `insertPerfectlyNestedBlockNode` cannot reach one and put a third, non-block child in.
///
/// ⚠️ AND TWO OF THE EIGHT REFUSE BY THROWING IN THE AUTHORITY, WHERE THE TWO BESIDE THEM ANSWER
/// `nullptr`: `getThenCoreCl` is `next_.at(0)` and `getElseCoreCl` is `next_.at(1)`
/// (`dsc/dsc2.cpp:2004-2011`) over a `VectorOfChildren` deriving from `std::vector`
/// (`dsc/dsc2.h:529`), so each throws `std::out_of_range` on the very node
/// `getThenBranchNode`/`getElseBranchNode` report absent (`:707-718`). ⛔ AND BRIDGE 1 REACHES THE
/// `at(1)` UNGUARDED: `DSC2ToDataflowIR.cpp:144-146` calls `getThenCoreCl` and `getElseCoreCl` in
/// consecutive statements behind nothing but `DT_CHECK_MSG(1 <= num_regions && num_regions <= 2)`, so
/// a ONE-region condition node throws there — while the V3 lowering of the same read guards it with
/// `if (max_num_regions == 2)` (`SNControlFlowLowering.cpp:1116`). [`Self::then_core_cl`] and
/// [`Self::else_core_cl`] answer [`None`], which is what the guarded site tests for.
///
/// ⚠️ AND THE SCHEDULER'S FOUR FIELD ANCHORS FOR THIS UNIT NAME ONE FIELD: `loopCond_`, beside
/// `comp`, `coreId` and `siblingRefNode`, which are method-signature continuations of the class
/// already tallied on e042 below — two ending `) const;` (`dsc/dsc2.h:702`, `:704`) and one ending
/// `) override;` (`:697`), which that tally's `) const;` count does not reach. `coreClCond_` itself
/// is absent from the list because its declaration carries a trailing comment (`:691-692`), the
/// class tallied on e031. Both real fields are carried here regardless.
///
/// ⛔ NO `PartialEq`: node identity in the authority is the pointer. `Clone` is IBM's own, through
/// `InheritWithClone` (`:685`).
#[derive(Clone, Debug)]
pub struct ConditionNode {
    /// The `BlockNode` subobject (`dsc/dsc2.h:685`), holding the at most two region blocks.
    ///
    /// PRIVATE, WHERE [`LoopNode`]'S IS PUBLIC, and that is the whole enforcement of this class's
    /// two-child rule: `BlockNode::addChildNode` is `virtual` so the authority's override is reached
    /// through a `BlockNode*` (`dsc/dsc2.h:544`, `:697`), but `moveChildren` and
    /// `insertPerfectlyNestedBlockNode` are NOT overridden, and a `BlockNode::addChildNode(..)`
    /// qualified call bypasses the override outright — three routes that put a third or a non-block
    /// child into a condition node there and none of which exist here. [`Self::base`] is the public
    /// read-only conversion, which is what the ~25 sites holding a `const BlockNode*` need.
    base_class: BlockNode,
    /// Field: e044_ConditionNode.loopCond_
    ///
    /// The loop guard (`dsc/dsc2.h:690`), ABSENT exactly when this node's condition is the
    /// core/corelet one. ⭐ `= {}` IS HOW THE DDL DROPS ONE it resolved to a constant
    /// (`ddc/ddl/ddl_conversion.cpp:353`, `:359`), and what that leaves behind is the discriminator,
    /// not a guard that holds trivially — which is why [`LoopCondComposite`] has no `Default`.
    pub loop_cond: Option<LoopCondComposite>,
    /// Field: e044_ConditionNode.coreClCond_
    ///
    /// The cores and corelets the "then" region applies to (`dsc/dsc2.h:691-692`).
    ///
    /// ⭐ AN ABSENT CORE IS AN EXCLUDED ONE, NOT AN UNCONSTRAINED ONE: `setRelevantCompCoreCl`
    /// INTERSECTS this map into the "then" region's `relevantComps_` and hands the complement to the
    /// "else" region (`dsc/dsc2.cpp:2663-2683`), so an EMPTY map excludes every core from the "then"
    /// side — and not because the intersection came out empty but because the `operator[]` at `:2665`
    /// CREATES that child's `NO_COMPONENT` entry, which is what then stops it inheriting its parent's
    /// set at `:2658-2660`.
    ///
    /// ⛔ WHAT IS NOT TRUE IS THAT A CORE LISTED WITH NO CORELETS IS A DIFFERENT CONDITION FROM AN
    /// ABSENT ONE. Executed over the extracted body, 35,625 of 37,500 absent-versus-listed-empty
    /// pairs propagate IDENTICALLY, and all 1,875 that differ need the PARENT's own set to list that
    /// core with no corelets — a state this function cannot produce (the head seeds `{0}` or `{0,1}`
    /// at `:2648-2653`, the "then" arm `emplace`s only a NON-EMPTY intersection at `:2671`, and the
    /// "else" arm ERASES a core whose difference empties at `:2681`). The four direct minters cannot
    /// produce it either — each fills `0..numCoreletsUsed_DSC2_` (`ddc/ddl/ddl_conversion.cpp:1738`,
    /// `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:3223`, `:3267`, and `dsc/dsc2.cpp:4746-4748`
    /// emplaced at `:4912`) — and every composition ERASES a core whose set empties
    /// (`ddl_conversion.cpp:337`, `:419`, `:424`, `ddc/ddc_transformation_util.cpp:504`, `:507`,
    /// `:541`). ⚠️ TWO ROUTES SURVIVE THAT, AND BOTH ARE AN `operator[]` REACHED BEFORE AN `insert`:
    /// the JSON importer creates the entry and then inserts the wire's array, so an EMPTY array mints
    /// one (`dsc/dsc2.cpp:1465-1471`), and the DDL's OR arm inserts one operand's set into an entry it
    /// just created, which leaves it empty when that operand's was (`ddl_conversion.cpp:428-429`) — a
    /// propagation, not a first cause. The sweep and its control are in
    /// `equivalence::e044_core_cl_cond_propagation_matches_the_executed_authority`.
    ///
    /// ⭐ SO THE NESTING IS THE AUTHORITY'S DECLARATION AND THE WIRE'S, not a distinction the
    /// propagation draws: `std::map<int, std::set<int>>` (`dsc/dsc2.h:691-692`), exported as a nested
    /// object (`dsc/dsc2.cpp:486-497`) and imported back the same way (`:1465-1471`).
    pub core_cl_cond: BTreeMap<CoreId, BTreeSet<CoreletId>>,
}

impl Default for ConditionNode {
    /// `ConditionNode() : BaseClass(CONDITION) {}` (`dsc/dsc2.h:686`) over an empty composite and an
    /// empty map (`:690-692`).
    fn default() -> Self {
        Self {
            base_class: BlockNode::new(NodeType::Condition),
            loop_cond: None,
            core_cl_cond: BTreeMap::new(),
        }
    }
}

impl ConditionNode {
    /// `hasCoreClCond()` (`dsc/dsc2.h:693-695`): WHICH of the two guards this node carries, answered
    /// off the loop side alone. Nine live sites read it — `getNextView` (`dsc/dsc2.cpp:1998`),
    /// `setRelevantCompCoreCl` (`:2663`), `ddc/ddcv1.cpp:3472`,
    /// `ddc/ddc_transformation_util.cpp:324`, `:489`, `ddc/ddl/ddl_conversion.cpp:3248`,
    /// `dsc/dsc2Pcfg.cpp:290`, `DSC2ToDataflowIR.cpp:91` and `SNControlFlowLowering.cpp:1049`.
    /// ⚠️ A TENTH IS BLOCK-COMMENTED INSIDE A CONDITION and is not a reader, so the arm it would have
    /// narrowed breaks on EVERY condition node (`ddc/ddcv1.cpp:2822-2825`).
    pub fn has_core_cl_cond(&self) -> bool {
        self.loop_cond.is_none()
    }

    /// The public `BlockNode` base (`dsc/dsc2.h:685`) — the conversion every site holding a
    /// `const BlockNode*` performs, and the read-only half of
    /// [`base_class`](ConditionNode::base_class).
    pub fn base(&self) -> &BlockNode {
        &self.base_class
    }

    /// The mutable base, MODULE-PRIVATE so that `BlockNode`'s own child operations cannot be reached
    /// on a condition node — see [`base_class`](ConditionNode::base_class).
    fn base_mut(&mut self) -> &mut BlockNode {
        &mut self.base_class
    }

    /// The region list by value, for [`ChildNode::children_mut`] alone.
    fn children_mut(&mut self) -> &mut Vec<ChildNode> {
        &mut self.base_class.next
    }

    /// `ConditionNode::addChildNode` (`dsc/dsc2.h:697`, body `dsc/dsc2.cpp:2143-2151`), the `override`
    /// that narrows `BlockNode`'s to two `BLOCK`s.
    ///
    /// ⭐ THE `nodeType_ != BLOCK` HALF OF THAT REFUSAL IS THE PARAMETER TYPE: a [`BlockNode`] by
    /// value cannot be a transfer. What remains at run time is the count, and it hands the node back
    /// exactly as [`BlockNode::add_child_node`] does.
    ///
    /// ⚠️ IT KEEPS THE [`InsertionPoint`] THE OVERRIDE TAKES, so the "else" region can still be
    /// inserted BEFORE the "then" one — the authority's signature allows it (`dsc/dsc2.h:697-698`) and
    /// the positional readings at `:707-718` are what would then disagree with the caller's intent.
    /// Both region helpers below pass [`InsertionPoint::Back`], which is the default the authority's
    /// own `addChildNode(nodeToAdd)` calls supply (`dsc/dsc2.cpp:2155`, `:2166`).
    #[must_use = "a refused insertion hands the node back and it is lost if dropped"]
    pub fn add_child_node(&mut self, at: InsertionPoint, node: BlockNode) -> Option<BlockNode> {
        if self.base_class.next.len() >= 2 {
            return Some(node);
        }
        let Some(index) = self.base_class.insertion_point(at) else {
            return Some(node);
        };
        self.base_class.next.insert(index, ChildNode::Block(node));
        None
    }

    /// `addThenRegion` (`dsc/dsc2.cpp:2152-2156`): the FIRST region, refused once one exists. Its
    /// callers build a conditional around code they are about to move
    /// (`ddc/ddc_transformation_util.cpp:265`, `:588`, `ddc/ddl/ddl_conversion.cpp:1745`).
    ///
    /// ⭐ AND `ddc/ddc_transformation_util.cpp:580-588` IS WHY [`BlockNode::clone`] MUST DROP THE
    /// CHILDREN: it clones a condition node and then calls this on the clone, which the authority
    /// refuses outright if the clone kept its regions.
    #[must_use = "a refused insertion hands the node back and it is lost if dropped"]
    pub fn add_then_region(&mut self, node: BlockNode) -> Option<BlockNode> {
        if !self.base_class.next.is_empty() {
            return Some(node);
        }
        self.add_child_node(InsertionPoint::Back, node)
    }

    /// `addElseRegion` (`dsc/dsc2.cpp:2158-2167`): the SECOND region, refused both when one already
    /// exists and when the "then" region does not (`:2159-2165`) — two `DT_ERROR`s with one answer
    /// here, because `next_.size() != 1` is exactly their union and neither leaves the node changed.
    #[must_use = "a refused insertion hands the node back and it is lost if dropped"]
    pub fn add_else_region(&mut self, node: BlockNode) -> Option<BlockNode> {
        if self.base_class.next.len() != 1 {
            return Some(node);
        }
        self.add_child_node(InsertionPoint::Back, node)
    }

    /// `getThenBranchNode()` (`dsc/dsc2.h:707-712`): `next_[0]`, absent on an empty node. Bridge 1
    /// emits the region's body from it (`SNControlFlowLowering.cpp:1096`) and the DDC reads it to
    /// decide whether a transfer is inside the guarded half (`ddc/ddc_transformation_util.cpp:326`).
    pub fn then_branch(&self) -> Option<&ChildNode> {
        self.base_class.next.first()
    }

    /// `getElseBranchNode()` (`dsc/dsc2.h:713-718`): `next_[1]`, absent on a node with fewer than two
    /// regions — which is the common shape, since a condition with no else region is one child
    /// (`ddc/ddl/ddl_conversion.cpp:1745`).
    pub fn else_branch(&self) -> Option<&ChildNode> {
        self.base_class.next.get(1)
    }

    /// `getThenCoreCl(comp)` (`dsc/dsc2.cpp:2003-2006`): the cores and corelets the "then" region is
    /// relevant to, which is that CHILD's `getRelevantCoreCl(comp)` and not this node's
    /// `coreClCond_` — the two agree only after `setRelevantCompCoreCl` has intersected one into the
    /// other (`:2663-2683`). [`None`] is the `next_.at(0)` throw.
    pub fn then_core_cl(
        &self,
        comp: SenComponent,
    ) -> Option<BTreeMap<CoreId, BTreeSet<CoreletId>>> {
        self.then_branch()
            .map(|node| node.base().relevant_core_cl_of_comp(comp))
    }

    /// `getElseCoreCl(comp)` (`dsc/dsc2.cpp:2007-2011`). [`None`] is the `next_.at(1)` throw — the one
    /// bridge 1 reaches unguarded at `DSC2ToDataflowIR.cpp:146`.
    pub fn else_core_cl(
        &self,
        comp: SenComponent,
    ) -> Option<BTreeMap<CoreId, BTreeSet<CoreletId>>> {
        self.else_branch()
            .map(|node| node.base().relevant_core_cl_of_comp(comp))
    }

    /// `ConditionNode::getNextView(comp)` (`dsc/dsc2.cpp:1995-2002`).
    ///
    /// ⛔ A LOOP-GUARDED CONDITION NODE RETURNS BOTH REGIONS TO EVERY COMPONENT — "always return both
    /// children for loop cond" (`:1998`) — by delegating with `ALL, -1, -1` and DISCARDING the
    /// caller's filter. A core/corelet-guarded one filters normally.
    ///
    /// ⛔ AND THIS METHOD IS NOT `virtual`, WHICH IS LOAD-BEARING: `BlockNode::getNextView`
    /// (`dsc/dsc2.h:541-543`) and this one (`:701-703`) are two unrelated functions, so a caller
    /// holding a `BlockNode*` gets the FILTERED reading even on a condition node. Rust's static method
    /// resolution reproduces that exactly — [`BlockNode::next_view`] through [`Self::base`], this one
    /// through the node itself.
    pub fn next_view(&self, comp: SenComponent) -> Vec<&ChildNode> {
        if self.has_core_cl_cond() {
            self.base_class.next_view(comp)
        } else {
            self.base_class.next_view(SenComponent::All)
        }
    }

    /// `ConditionNode::getNextView(comp, clId, coreId)` with a real core
    /// (`dsc/dsc2.cpp:1995-2002`), the form bridge 1's control-flow lowering calls per program unit
    /// (`SNControlFlowLowering.cpp:1213`). The loop-guarded arm discards all three filters, as above.
    pub fn next_view_of_corelet(
        &self,
        comp: SenComponent,
        core: CoreId,
        corelet: Option<CoreletId>,
    ) -> Vec<&ChildNode> {
        if self.has_core_cl_cond() {
            self.base_class.next_view_of_corelet(comp, core, corelet)
        } else {
            self.base_class.next_view(SenComponent::All)
        }
    }
}

/// Replaces: e036_SyncNode
///
/// `dsc/dsc2.h:964-972`. One end of a signal: which units it signals to or waits on, and which end
/// it is. The DDL conversion mints one per `SyncOp` (`ddc/ddl/ddl_conversion.cpp:1708-1732`) and the
/// L3 scheduler mints them in send/receive pairs
/// (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:640-660`).
///
/// `e026_SyncNode` is this same class under the superseded numbering, and e026 is now `Metadata`
/// (`crustify-scheduler/UNITS.tsv:27`) — a COLLISION, not a merely stale number. All eight of its
/// anchors are RENUMBERED onto e036 here (`UNITS.tsv:37`), the three UNFILLED ones included, so the
/// open work stays attributed to the live entry. None is deleted and none changes meaning.
///
/// ⛔ THIS CARRIES THREE OF SYNCNODE'S FIVE FIELDS, so the `e036_SyncNode` anchor below stays open.
/// Both of the others are schedule-node pointer identity: `implicitSyncRefTransfer_` (`:968`) is a
/// `const TransferNode*` its reader dereferences for that transfer's TILE SIZE and for its SOURCE
/// labeled DS's precision (`dsc-based-utils/DSC2ToDataflowIR/V3/SNSyncLowering.cpp:156-158`,
/// `:179-184`) — never its destination, which is the lowering's own component (`:144-145`); its JSON
/// round trip goes through the node's `name_` (`dsc/dsc2.cpp:823-826`), e013's field;
/// `otherEndOfTheSignals_` (`:969`) is the `vector<const SyncNode*>` linking the two ends.
///
/// ⛔ AND `getComponentsFromOtherEnds` STAYS OUT WITH THEM: it walks those pointers and unions each
/// other end's `relevantComps_` (`dsc/dsc2.cpp:2408-2421`), e013's field, which has no ported
/// writer. ⛔ THAT, NOT [`units`](Self::units), is where bridge 1 gets the units it emits a
/// `sync_send`/`sync_recv` against (`SNSyncLowering.cpp:20-42`).
///
/// ⛔ NO `PartialEq`: node identity in the authority is the pointer, and here it is what links the
/// ends. `Clone` is IBM's own, through `InheritWithClone` (`:964`).
#[derive(Clone, Debug)]
pub struct SyncNode {
    /// The `ScheduleNode` subobject (`dsc/dsc2.h:964`, `InheritWithClone<ScheduleNode, SyncNode>`),
    /// tagged `SYNC` by `SyncNode()` (`:965`). A sync is a LEAF.
    pub base_class: ScheduleNode,
    /// Field: e036_SyncNode.units_
    ///
    /// "all to all signals" (`dsc/dsc2.h:966`): every unit this end signals to or waits on.
    ///
    /// ⛔ ORDERED HERE, HASH-ORDERED IN THE AUTHORITY, where it is an `unordered_set` (`:966`). THREE
    /// consumers put that order in their output — the node's JSON array (`dsc/dsc2.cpp:814-819`), the
    /// `SyncOp` unit-name `ArrayAttr` of the DSC-to-DDL export
    /// (`ddc/ddl/ddl_conversion.cpp:3319-3325`), and the ordered `dstUnits` vector the PCFG's
    /// `createSyncNode` receives (`dcg/dcg_fe/pcfg_gen/dlOpsNew.cpp:2687-2690`) — so all three follow
    /// libstdc++ bucket order there and [`SenComponent`]'s declaration order here. A fourth site puts
    /// it in a diagnostic, naming whichever colliding unit it reaches first
    /// (`ddc/ddl/ddl_conversion.cpp:2806-2810`).
    ///
    /// ⚠️ THE REMAINING READS ARE ORDER-FREE — but they are not six, as this anchor claimed, and
    /// NINETEEN of them are the membership test it gave to two. `units_.count(<a named component>)`
    /// appears THIRTEEN times in the PCFG sync builder (`dcg/dcg_fe/pcfg_gen/dlOpsNew.cpp:2651`,
    /// `:2674`, `:2680`, `:2683`, `:2706`, `:2711`, `:2736`, `:2741`, `:2774`, `:2782`, `:2790`,
    /// `:2792`, `:2798`) and SIX times in the L3 scheduler's L3SU/L3LU pair removal
    /// (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:4094-4100`). Three more iterate to accumulate into
    /// a set or to ask an any-of (`dsc/dsc2.cpp:2702-2704`, `dsc/dsc2Pcfg.cpp:2050-2053`,
    /// `ddc/ddc_transformation.cpp:1531-1535`). The order-free conclusion holds for every one of
    /// them — it is the count that was wrong, and it was wrong by thirteen.
    ///
    /// ⛔ AND THREE READS ARE CARDINALITY, WHICH THIS TYPE CANNOT GUARANTEE AND MUST NOT NARROW TO:
    /// `!units_.empty()` guards the membership walk (`dlOpsNew.cpp:2650`) and `units_.size() == 1` is
    /// asserted on BOTH ends of every sync node in the tree (`L3DlOpsScheduler.cpp:4086`, `:4092`,
    /// beside the same guard on `otherEndOfTheSignals_` at `:4088`). ⭐ A ONE-COMPONENT FIELD WOULD BE
    /// WRONG ANYWAY: the DDL conversion mints an implicit L0 sync whose units are `{L0SU}` on one end
    /// and one `L0LUROW<i>` PER ROW on the other (`ddc/ddl/ddl_conversion.cpp:1785-1788`), so that
    /// guard is a PHASE constraint on the nodes the L3 scheduler is willing to delete, not an
    /// invariant of the class.
    pub units: BTreeSet<SenComponent>,
    /// Field: e036_SyncNode.isReceive_
    ///
    /// Which end this is (`dsc/dsc2.h:967`): bridge 1 emits a `sync_send` when it is false and a
    /// `sync_recv` when it is true (`SNSyncLowering.cpp:209`, `:239`).
    ///
    /// ⛔ BOTH ARMS ARE ALSO GATED ON A NULL `implicitSyncRefTransfer_`, so this flag selects NEITHER
    /// when one is set: `is_implicit_sync` is that pointer against `nullptr` (`:208`), both arms carry
    /// `&& !is_implicit_sync`, and the third arm reads this field not at all (`:264-269`).
    pub is_receive: bool,
    /// Field: e036_SyncNode.isSoft_
    ///
    /// Whether the send may run ahead of the transfers it covers (`dsc/dsc2.h:967`): bridge 1 sets
    /// the emitted send's `wait_immediately_for_async_transfers` to its NEGATION
    /// (`SNSyncLowering.cpp:210-211`). TWO sites write it — the L3 scheduler's minter
    /// (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:658`, which only ever writes `true`) and the JSON
    /// importer (`dsc/dsc2.cpp:1721-1722`), round-tripping the exporter at `:822`.
    ///
    /// ⛔ AND THE PCFG TRANSLATOR READS IT FIVE TIMES ACROSS THREE MINT PATHS, not once. The EXPLICIT
    /// path passes it beside `isReceive_` for the single sync it builds (`dsc/dsc2Pcfg.cpp:478-479`)
    /// — the primary reader, which this anchor left out — and the IMPLICIT path carries it onto both
    /// syncs of the L0SU pair (`:2056`, `:2058`) AND both of the L0LU pair (`:2067`, `:2069`).
    /// ⭐ THOSE FOUR CORROBORATE [`is_receive`](Self::is_receive)'s NOTE FROM THE OTHER SIDE: each
    /// passes a LITERAL `false`/`true` for the receive flag and never reads that field, while still
    /// propagating this one.
    ///
    /// ⚠️ THE SCHEDULER LISTED NO ANCHOR FOR IT: it is declared on the same line as
    /// [`is_receive`](Self::is_receive), and that bridge-1 read is on this campaign's path.
    pub is_soft: bool,
}

impl Default for SyncNode {
    /// `SyncNode() : BaseClass(SYNC) {}` (`dsc/dsc2.h:965`) over the authority's member initialisers
    /// (`:966-967`). ⛔ HAND-WRITTEN RATHER THAN DERIVED, because a derived one would leave the base's
    /// tag `INVALID` — the kind a node of this class can never have.
    fn default() -> Self {
        Self {
            base_class: ScheduleNode::new(NodeType::Sync),
            units: BTreeSet::new(),
            is_receive: false,
            is_soft: false,
        }
    }
}

// crustify:todo: e036_SyncNode

// crustify:todo: e036_SyncNode.implicitSyncRefTransfer_

// crustify:todo: e036_SyncNode.otherEndOfTheSignals_

/// A constant container the mask value is read out of — an index into
/// `DesignSpaceConfig::constantInfo_` (`dsc/designSpaceConfig.h:90`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ConstantId(pub i32);

/// A coordinate INSIDE one stick, counted in elements: the DDC stores
/// `cumulative stick size - masked elements` (`ddc/ddcv1.cpp:3589-3597`), so it is the first
/// coordinate the mask covers along that dim.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StickCoord(pub i32);

/// A count of ELEMENTS on one side of a stick-mask split — not a coordinate: bridge 1 emits these as
/// `agen.set_transfer_mask_state`'s two offset arrays (`SNStickMaskLowering.cpp:25-30`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MaskElements(pub i32);

/// Which of a stick's slices the mask changes at.
///
/// ⛔ SIGNED, AND `-1` IS REACHABLE: masking a whole dim leaves remainder 0 in slice 0, and the
/// authority then names the PREVIOUS slice (`dsc/dsc2.cpp:2481-2483`). It is not an absent value —
/// bridge 1 compares against it per slice and emits `(1)` for every slice past it
/// (`SNStickMaskLowering.cpp:51-58`), so `-1` yields `(1)(1)(1)(1)(1)(1)(1)(1)`.
///
/// ⛔ AND THAT STRING IS THE ONE THE DCC REFUSES, so `-1` is reachable here and unlowerable there. It
/// matches `isFullMask()` rather than the generic SAMV pattern
/// `^(\(A\)){0,7}(\(A\|B\))(\(1\)){0,7}$` (`dialect_utils/Agen/Utils.cpp:36-51`, and `isFullMask` at
/// `dataflow-scheduler/external/dataflow-scheduler-dialects/lib/Dialect/Agen/Agen.cpp:2737-2740`),
/// and the full-mask path requires that NO mask attributes be attached
/// (`dcc/src/Conversion/AgenToSentient/Helper.cpp:2607-2612`) while bridge 1 always attaches both
/// (`SNStickMaskLowering.cpp:74-75`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SliceId(pub i32);

/// A stick spans the SFP's eight slices — [`SFP_SLICES`], the `numSlicesPerStick` of
/// `sys-arch-spec/sysdef.cpp:229`, which `getView` and `getStickSizes` both spell as a literal `8`
/// (`dsc/dsc2.cpp:2460`, `:4080`).
pub const SLICES_PER_STICK: i32 = SFP_SLICES as i32;

/// One half of a stick mask: how many elements it leaves valid and how many it masks. IBM declares
/// each half as a `std::pair<int, int>` commented `<unmasked, masked>` (`dsc/dsc2.h:1068`); naming
/// them is what makes bridge 1's two parallel offset arrays impossible to transpose
/// (`SNStickMaskLowering.cpp:25-30`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MaskSplit {
    pub unmasked: MaskElements,
    pub masked: MaskElements,
}

/// Replaces: StickMaskNode::View
///
/// `dsc/dsc2.h:1067-1070`. What one SAMV node programs: a mask for the within-slice dim, one for the
/// cross-slice dim, and the slice the second takes effect at. Derived by
/// [`StickMaskNode::view`](StickMaskNode::view), never stored and never serialized.
///
/// ⛔ NO `Default`, BECAUSE THE AUTHORITY'S HAS NO VALUE: `View view;` (`dsc/dsc2.cpp:2441`)
/// default-initializes, leaving `transitionSliceId_` indeterminate until one of the two branches
/// assigns it. Every field of every value we hand out comes from [`StickMaskNode::view`].
///
/// ```compile_fail
/// // E0599, for the reader only: stable rustdoc parses the code an annotation names and ignores
/// // it, so the annotation is documentation and the positive control below is the check.
/// use deeptools::schedule::dsc2::StickMaskView;
/// let _ = StickMaskView::default();
/// ```
///
/// ⭐ AND ITS POSITIVE CONTROL, which rustdoc DOES enforce — the same path, reached the only way a
/// value of this type exists. Without it the `compile_fail` above would pass just as happily on a
/// misspelled module path or an item that stopped being `pub`, i.e. exactly when it had stopped
/// testing anything:
///
/// ```
/// use deeptools::schedule::dsc2::{StickMaskNode, StickMaskView};
/// let _: Option<StickMaskView> = StickMaskNode::default().view();
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StickMaskView {
    /// Field: e027_StickMaskNode.maskA_
    /// Field: e040_StickMaskNode.maskA_
    ///
    /// The within-slice (wsl) dim's mask (`dsc/dsc2.h:1068`, `dsc/dsc2.cpp:2463-2472`).
    ///
    /// ⚠️ THE SCHEDULER LISTS NO ANCHOR FOR EITHER HALF — not for `maskB_` either, though an earlier
    /// revision of the plan did. `maskA_, maskB_;` carries a TRAILING `// <unmasked, masked>`
    /// (`dsc/dsc2.h:1068`), so the field scan finds nothing once its pattern is anchored to the end
    /// of the line; the same change that anchored it also dropped the trailing-underscore
    /// requirement, which is what lets `break;` in as `e042_LoopDistributionInfo.break`. So one
    /// tightening lost a real field and gained a phantom. Bridge 1 reads both halves of both
    /// (`SNStickMaskLowering.cpp:25-30`).
    pub mask_a: MaskSplit,
    /// Field: e027_StickMaskNode.maskB_
    /// Field: e040_StickMaskNode.maskB_
    ///
    /// The cross-slice (xsl) dim's mask (`dsc/dsc2.h:1068`, `dsc/dsc2.cpp:2473-2489`).
    pub mask_b: MaskSplit,
    /// Field: e027_StickMaskNode.transitionSliceId_
    /// Field: e040_StickMaskNode.transitionSliceId_
    ///
    /// The slice `mask_b` transitions at (`dsc/dsc2.h:1069`): bridge 1 emits `(A)` for every slice
    /// before it, `(A|B)` at it, and `(1)` after (`SNStickMaskLowering.cpp:51-58`).
    ///
    /// ⛔ `(1)` IS FULLY MASKED, NOT UNMASKED. `(0)` is the token for no masking, and the op's own
    /// documentation reads `(1)` as "will be fully masked", with an example IR of exactly this shape:
    /// `"(A)(A)(A)(A)(A)(A|B)(1)(1)"` for a transition at slice 5. Both are in the op's TableGen
    /// definition, `Agen.td:1058-1072` and `:1090`, under
    /// `dataflow-scheduler/external/dataflow-scheduler-dialects/include/dataflow-scheduler/Dialect/Agen/`.
    /// The slices PAST the transition are the masked tail, which is what a SAMV is for.
    pub transition_slice: SliceId,
}

/// Replaces: e027_StickMaskNode
/// Replaces: e040_StickMaskNode
///
/// `dsc/dsc2.h:1059-1072`. A SAMV node: the mask an LXLU transfer applies to the tail of a stick so
/// that the elements past the tensor's real extent read the mask value instead. `constructSAMVNodes`
/// mints one reference node out of `DesignSpaceConfig::coordinateMasking_`, clones it into the "then"
/// region of a [`ConditionNode`], and clones a RESET copy — same fields, an empty
/// [`first_stick_coord_to_mask_per_dim`](Self::first_stick_coord_to_mask_per_dim) — into the "else"
/// region (`ddc/ddcv1.cpp:3527-3665`).
///
/// ⛔ THIS CARRIES FOUR OF STICKMASKNODE'S FIVE FIELDS, so the `e027_StickMaskNode` anchor below
/// stays open. `affectedTransfers_` (`:1065`) is a `vector<const dsc2::TransferNode*>` held as
/// schedule-node pointer identity, and its JSON round trip goes through each transfer's `name_`
/// (`dsc/dsc2.cpp:980-987`), e013's field.
///
/// ⚠️ AND THE SCHEDULER RE-LABELLED THIS UNIT `e040_StickMaskNode` WITHOUT EVER PUTTING THAT ANCHOR
/// IN THE TREE — no revision of this file has carried an `e040` anchor, so the second label is added
/// here beside the first rather than replacing it. The remainder schedule did not notice: it matches
/// a landed unit by CLASS NAME, not by number, so `e027_StickMaskNode` satisfied it — but an anchor
/// is keyed by its eNNN name, so nothing in the tree answered for `e040` at all.
///
/// ⛔ NO `PartialEq`: node identity in the authority is the pointer. `Clone` is IBM's own, through
/// `InheritWithClone` (`:1059`), and the DDC leans on it for the reset copy (`ddc/ddcv1.cpp:3663`).
#[derive(Clone, Debug)]
pub struct StickMaskNode {
    /// The `ScheduleNode` subobject (`dsc/dsc2.h:1059`,
    /// `InheritWithClone<ScheduleNode, StickMaskNode>`), tagged `STICKMASK` by `StickMaskNode()`
    /// (`:1060`). A stick mask is a LEAF.
    pub base_class: ScheduleNode,
    /// Field: e027_StickMaskNode.maskValConstId_
    /// Field: e040_StickMaskNode.maskValConstId_
    ///
    /// The constant holding the value written into the masked elements (`dsc/dsc2.h:1061`), taken
    /// from `DesignSpaceConfig::maskingConstId_` (`ddc/ddcv1.cpp:3531`).
    ///
    /// ⛔ THE AUTHORITY'S `-1` IS ABSENT, and both readers refuse it rather than indexing with it:
    /// bridge 1 checks `>= 0` before `constantInfo_.at` (`SNStickMaskLowering.cpp:32-36`) and the
    /// PCFG translator repeats the check (`dsc/dsc2Pcfg.cpp:2202-2203`).
    pub mask_val_const_id: Option<ConstantId>,
    /// Field: e027_StickMaskNode.dataFormat_
    /// Field: e040_StickMaskNode.dataFormat_
    ///
    /// The precision of the masked tensor (`dsc/dsc2.h:1062`), copied from the affected transfer's
    /// labeled data structure (`ddc/ddcv1.cpp:3577`).
    ///
    /// ⛔ ITS INITIALISER IS NOT [`DataFormats`]' OWN DEFAULT: this field starts `INVALID` (`:1062`)
    /// where [`ComputeNode::data_format`] starts at fp16, so a node minted without a transfer
    /// carries no width at all — see [`StickMaskNode::default`].
    pub data_format: DataFormats,
    /// Field: e027_StickMaskNode.stickLayout_
    /// Field: e040_StickMaskNode.stickLayout_
    ///
    /// What one stick is made of: each dim inside it with its extent in elements (`dsc/dsc2.h:1063`),
    /// range-built out of `getStickSizes` (`ddc/ddcv1.cpp:3546-3547`) — the conversion [`Size`]
    /// carries a [`From`] impl for.
    ///
    /// ⛔ ITS LAST ENTRY IS THE CROSS-SLICE DIM (`dsc/dsc2.cpp:2445-2446`), which is the whole reason
    /// this is a `Vec` and not a map: [`view`](Self::view) reads the layout's order and its length.
    pub stick_layout: Vec<Size>,
    /// Field: e027_StickMaskNode.firstStickCoordToMaskPerDim_
    /// Field: e040_StickMaskNode.firstStickCoordToMaskPerDim_
    ///
    /// Per dim, the first coordinate inside the stick the mask covers (`dsc/dsc2.h:1064`).
    ///
    /// ⛔ EMPTY MEANS "MASK NOTHING", AND THAT IS THE RESET NODE: the else-region clone clears it
    /// (`ddc/ddcv1.cpp:3663`) and the PCFG translator checks the else node's map IS empty
    /// (`dsc/dsc2Pcfg.cpp:2167`). ⛔ AND THE DDC MINTS AT MOST ONE ENTRY — "Cannot currently mask
    /// more than one dim at a time" (`ddc/ddcv1.cpp:3629-3631`) — while [`view`](Self::view) reads
    /// two dims out of it and scales one mask by the other's extent.
    pub first_stick_coord_to_mask_per_dim: BTreeMap<PrimaryDimTypes, StickCoord>,
}

/// `dsc/dsc2.h:1060-1062`: the base is tagged `STICKMASK` by `StickMaskNode()`, the mask value is
/// absent and the precision is `INVALID`, unlike [`DataFormats`]' own default.
impl Default for StickMaskNode {
    fn default() -> Self {
        Self {
            base_class: ScheduleNode::new(NodeType::StickMask),
            mask_val_const_id: None,
            data_format: DataFormats::Invalid,
            stick_layout: Vec::new(),
            first_stick_coord_to_mask_per_dim: BTreeMap::new(),
        }
    }
}

impl StickMaskNode {
    /// `dsc/dsc2.cpp:2440-2491`. The two masks and the transition slice this node programs, derived
    /// from [`stick_layout`](Self::stick_layout) and
    /// [`first_stick_coord_to_mask_per_dim`](Self::first_stick_coord_to_mask_per_dim). Bridge 1 turns
    /// the result into `agen.set_transfer_mask_state`'s offsets and its per-slice mask map
    /// (`SNStickMaskLowering.cpp:22-74`).
    ///
    /// ⛔ ABSENT WHERE THE AUTHORITY REFUSES — "SAMV not possible with current stick layout", three
    /// times: over three dims in the stick (`:2442-2444`), a second dim outside the cross-slice one
    /// (`:2455-2457`), and a cross-slice extent under [`SLICES_PER_STICK`], where no whole element
    /// falls in a slice (`:2460-2463`).
    ///
    /// ⛔ AND ABSENT ON AN EMPTY LAYOUT, where the authority reads `stickLayout_.back()` off the end
    /// (`:2446`) — undefined behaviour there, [`None`] here, and reachable through the JSON importer,
    /// which fills the vector entry by entry (`dsc/dsc2.cpp:1847-1851`).
    pub fn view(&self) -> Option<StickMaskView> {
        if self.stick_layout.len() > 3 {
            return None;
        }
        let xsl_dim = self.stick_layout.last()?.dim;
        // `:2445`. ⛔ THE "NO WSL DIM YET" SENTINEL IS A LIVE DIM VALUE, the authority's
        // `PrimaryDimTypesCount`: a layout entry carrying it is a second wsl dim to the loop below,
        // and an `Option` here would silently accept what the authority refuses.
        let mut wsl_dim = PrimaryDimTypes::Undefined;
        let (mut wsl_size, mut xsl_size) = (1, 1);
        for entry in &self.stick_layout {
            if entry.dim == xsl_dim {
                xsl_size *= entry.size.0;
            } else {
                wsl_size *= entry.size.0;
                if wsl_dim == PrimaryDimTypes::Undefined {
                    wsl_dim = entry.dim;
                } else if wsl_dim != entry.dim {
                    return None;
                }
            }
        }
        let xsl_per_slice = xsl_size / SLICES_PER_STICK;
        if xsl_per_slice == 0 {
            return None;
        }

        // `:2464-2472`: maskA covers the within-slice dim's tail. An unmasked dim's first masked
        // coordinate is its whole extent, so nothing is masked.
        let first_coord = self
            .first_stick_coord_to_mask_per_dim
            .get(&wsl_dim)
            .map_or(wsl_size, |coord| coord.0);
        // `:2469`, `:2486`: with three dims the cross-slice dim is the inner one, and each mask is
        // scaled by the other's extent.
        let xsl_inner = self.stick_layout.len() == 3;
        let scale_a = if xsl_inner { xsl_per_slice } else { 1 };
        let mask_a = MaskSplit {
            unmasked: MaskElements(first_coord * scale_a),
            masked: MaskElements((wsl_size - first_coord) * scale_a),
        };

        // `:2473-2485`: maskB covers the cross-slice dim.
        let (unmasked, transition_slice) =
            match self.first_stick_coord_to_mask_per_dim.get(&xsl_dim) {
                // No masking in this dim: transition at the last slice, and mask nothing there.
                None => (xsl_per_slice, SliceId(SLICES_PER_STICK - 1)),
                // ⛔ REMAINDER 0 MEANS THE WHOLE SLICE IS VALID, so the PREVIOUS slice transitions.
                Some(coord) => {
                    let (quot, rem) = (coord.0 / xsl_per_slice, coord.0 % xsl_per_slice);
                    (rem, SliceId(if rem == 0 { quot - 1 } else { quot }))
                }
            };
        let scale_b = if xsl_inner { 1 } else { wsl_size };
        let mask_b = MaskSplit {
            unmasked: MaskElements(unmasked * scale_b),
            masked: MaskElements((xsl_per_slice - unmasked) * scale_b),
        };

        Some(StickMaskView {
            mask_a,
            mask_b,
            transition_slice,
        })
    }
}

// crustify:todo: e027_StickMaskNode

// crustify:todo: e040_StickMaskNode

// crustify:todo: e027_StickMaskNode.affectedTransfers_

// crustify:todo: e040_StickMaskNode.affectedTransfers_

/// Which half of an indirect access an allocation is — `AllocateNode::IndirectAllocType`
/// (`dsc/dsc2.h:990-994`). It is the paged-access discriminator the L3 scheduler reads: `isPagedLds`
/// is `VALUE_TENSOR` on an HBM allocation (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:6596-6604`)
/// and `isIndexLds` is `INDEX_TENSOR` on one (`:6582-6594`).
///
/// ⛔ AN INDIRECT ACCESS IS TWO ALLOCATIONS AND THIS ONLY NAMES ONE HALF: the value tensor holds the
/// paged data, the index tensor holds the addresses into it, and the link between them is
/// `relatedIndirectAccessAlloc_` (`dsc/dsc2.h:999-1001`) — schedule-node pointer identity, not
/// ported here — so nothing in this port can walk from one half to the other. The JSON exporter
/// `DT_CHECK`s that link non-null whenever this is not `NO_INDIRECTION` (`dsc/dsc2.cpp:916-921`).
///
/// ⛔ THE DISCRIMINANTS ARE NOT OBSERVABLE, unlike [`NodeType`]'s: both string maps are
/// `std::unordered_map` (`dsc/dsc2.h:1049-1052`), the JSON round trip carries the spelling
/// (`dsc/dsc2.cpp:907-909`, `:1792-1794`), and `fillAllocateNode` does not put this field into the
/// SuperDsc fingerprint at all (`dsc/superdsc.cpp:1446-1467`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum IndirectAllocType {
    /// `indirectAllocType_`'s initialiser (`dsc/dsc2.h:994`): an ordinary, directly addressed
    /// allocation. `getPageSize` answers the empty map for it (`dsc/dsc2.cpp:4481-4485`).
    #[default]
    NoIndirection = 0,
    /// The paged data itself, whose own layout gives the page size.
    ValueTensor = 1,
    /// The addresses into a value tensor.
    IndexTensor = 2,
}

impl IndirectAllocType {
    /// Every role in the authority's declaration order (`dsc/dsc2.h:990-994`).
    pub const ALL: [Self; 3] = [Self::NoIndirection, Self::ValueTensor, Self::IndexTensor];

    /// Field: e037_AllocateNode.indirectAllocTypeToString
    ///
    /// Field: e003_AllocateNode.indirectAllocTypeToString
    ///
    /// The spelling `indirectAllocTypeToString` gives this role (`dsc/dsc2.h:1049-1050`, filled
    /// `dsc/dsc2.cpp:2423-2427`). ⭐ TOTAL, AND THE AUTHORITY'S MAP IS TOO — all three have an entry.
    pub fn name(self) -> &'static str {
        match self {
            Self::NoIndirection => "no_indirection",
            Self::ValueTensor => "value_tensor",
            Self::IndexTensor => "index_tensor",
        }
    }

    /// Field: e037_AllocateNode.stringToIndirectAllocType
    ///
    /// Field: e003_AllocateNode.stringToIndirectAllocType
    ///
    /// `stringToIndirectAllocType`, the `flipMap` of the above (`dsc/dsc2.h:1051-1052`, built
    /// `dsc/dsc2.cpp:2428-2430`). ⛔ THE AUTHORITY'S ONLY CALLER IS AN `.at()` THAT THROWS on a miss
    /// (`dsc/dsc2.cpp:1793-1794`), so [`None`] here is that throw's input, never a live answer.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "no_indirection" => Some(Self::NoIndirection),
            "value_tensor" => Some(Self::ValueTensor),
            "index_tensor" => Some(Self::IndexTensor),
            _ => None,
        }
    }
}

/// What one entry of an index tensor holds — `AllocateNode::IndexTensorType` (`dsc/dsc2.h:995-998`).
///
/// ⛔ ONLY `ADDRESS` IS SUPPORTED: `isIndexLds` `DT_CHECK_MSG`s it on every index allocation it
/// recognises, "Only index tensors of type address are supported"
/// (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:6587-6589`), and it is also the declared default
/// (`dsc/dsc2.h:998`). `INDEX` is ported because it holds the discriminant that check rejects.
///
/// ⛔ THE FIELD IS ONLY MEANINGFUL UNDER [`IndirectAllocType::IndexTensor`]: the JSON exporter
/// writes it only then (`dsc/dsc2.cpp:910-915`), so an imported non-index allocation always reads
/// back the default rather than whatever it held.
///
/// ⛔ THE STRING MAP LISTS `INDEX` FIRST (`dsc/dsc2.cpp:2432-2435`) while the enum declares
/// `ADDRESS` first — the map is an `unordered_map` and its initialiser order is not the enum's, so
/// the discriminants come from the declaration and nothing else.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum IndexTensorType {
    /// `indexTensorType_`'s initialiser (`dsc/dsc2.h:998`): the entry is an address.
    #[default]
    Address = 0,
    /// The entry is an index the hardware still has to convert to an address.
    Index = 1,
}

impl IndexTensorType {
    /// Both forms in the authority's declaration order (`dsc/dsc2.h:995-998`).
    pub const ALL: [Self; 2] = [Self::Address, Self::Index];

    /// Field: e037_AllocateNode.indexTensorTypeToString
    ///
    /// Field: e003_AllocateNode.indexTensorTypeToString
    ///
    /// The spelling `indexTensorTypeToString` gives this form (`dsc/dsc2.h:1053-1054`, filled
    /// `dsc/dsc2.cpp:2432-2435`).
    pub fn name(self) -> &'static str {
        match self {
            Self::Address => "address",
            Self::Index => "index",
        }
    }

    /// Field: e037_AllocateNode.stringToIndexTensorType
    ///
    /// Field: e003_AllocateNode.stringToIndexTensorType
    ///
    /// `stringToIndexTensorType`, the `flipMap` of the above (`dsc/dsc2.h:1055-1056`, built
    /// `dsc/dsc2.cpp:2436-2438`). ⛔ THE AUTHORITY'S ONLY CALLER IS AN `.at()` THAT THROWS on a miss
    /// (`dsc/dsc2.cpp:1796-1797`).
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "address" => Some(Self::Address),
            "index" => Some(Self::Index),
            _ => None,
        }
    }
}

/// How many buffers one allocation reserves — `AllocateNode::numBuffers_` (`dsc/dsc2.h:984`), whose
/// own comment enumerates the encoding: "1:no buffering, 2:double-buffer, -1:streaming buffer".
///
/// ⛔ NOT AN ENUM OF THOSE THREE, because the set is not closed: the DDL conversion assigns it
/// straight from the `AllocateOp`'s `num_buffers` attribute (`ddc/ddl/ddl_conversion.cpp:804`,
/// `:833`), so a template may state any count, and `allocAllMem` multiplies one buffer's capacity by
/// it to size the request (`ddc/ddcv1.cpp:224-226`, `:244`).
///
/// ⛔ AND NOT AN [`Option`] EITHER: [`STREAMING`](Self::STREAMING) is a live third mode, not the
/// absence of a count, and its readers keep it distinct from `2` even while mapping it to `2` —
/// `allocAllMem` sizes its REQUEST with `2` (`ddc/ddcv1.cpp:224-226`) and then widens the
/// RESERVATION to the memory's whole capacity — half of it on a post-RCUDD1A L0 or L0_SCALE that is
/// not tethered — which no other count gets (`ddc/ddcv1.cpp:317-328`), and `processImplicitSync`
/// refuses an implicit sync on anything else, "Implicit syncs are only
/// possible on circular buffers (num_buffers=-1)"
/// (`ddc/ddl/ddl_conversion.cpp:1777-1782`).
///
/// ⛔ THE L3 SCHEDULER ADMITS ONLY 1 OR 2: `DT_CHECK_MSG` "Expect no buffering or double buffering"
/// on an HBM-pinned LX allocation and "Expect no buffering" on any other
/// (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:4940-4947`).
///
/// ⭐ A BARE COUNT CANNOT REACH THE FIELD:
///
/// ```compile_fail
/// use deeptools::schedule::dsc2::AllocateNode;
/// let mut node = AllocateNode::default();
/// node.num_buffers = -1;
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NumBuffers(pub i32);

impl NumBuffers {
    /// The streaming (circular) buffer's encoding, `-1` (`dsc/dsc2.h:984`). ⭐ IT IS THE ONE VALUE
    /// BRIDGE 1 TESTS: `getBufferingOrStreamingMode` answers mode 2 for it and mode 1 for every
    /// other count (`dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:58-64`, `:76-82`).
    pub const STREAMING: Self = Self(-1);
}

/// The size limit on one dim of an allocation's layout — one entry of
/// `AllocateNode::maxDimSizes_` (`dsc/dsc2.h:983`), carried inside
/// [`layout_dim_order`](AllocateNode::layout_dim_order) beside the dim it limits.
///
/// ⛔⛔ IT HOLDS TWO DIFFERENT CURRENCIES AND THE PASS ORDER IS WHAT SAYS WHICH, which is why it is
/// neither a [`DataStageId`] nor a [`DimSize`]. The DDL conversion stores a DATA-STAGE INDEX here —
/// its own comment says "store the Datastage index in the maxDimSizes vector. It will be later
/// converted into an actual size" (`ddc/ddl/ddl_conversion.cpp:1902-1905`) — and
/// `finalizeAllocateLayouts` overwrites each non-negative entry in place with that stage's EXTENT
/// for the paired dim, divided by the cumulative stick size when the dim is a stick dim, so the
/// result counts sticks there and elements elsewhere (`ddc/ddcv1.cpp:1710-1732`).
///
/// ⛔ AND THE JSON IMPORTER CANNOT TELL THEM APART: it pushes the bare integer
/// (`dsc/dsc2.cpp:1761-1764`), so a dump taken before that pass reimports stage indices into the
/// same slots an extent would occupy.
///
/// ⭐ A FILLED ENTRY IS NEVER NEGATIVE, so this is `u32` and the authority's negative is the
/// [`None`] beside the dim: a data-stage index is an index, and the extent that overwrites it is a
/// size. ⛔ BUT THE SIZING READER ANSWERS `-1`, SO THE CONVERSION IS NOT A CAST.
/// `finalizeAllocateLayouts` sizes each entry with `ss_.primaryDimToVal_st(dim, component_, 0, 0)`
/// (`ddc/ddcv1.cpp:1720-1722`), and that is `-1` — a VALUE, not an absence — for a dim the sizing
/// data stage does not carry: every dim is born `-1` (`dsc/dims.h:162-192`) and `calculate_padded`
/// returns `-1` for any negative before every other branch (`dsc/dims.cpp:567-568`), which is what
/// [`DataStructDims::primary_dim_to_val_for_component`] reports as `Some(DimVal(-1))`. A porter of
/// that pass has to map the negative to [`None`], because `size as u32` would write 4294967295 into
/// a slot every reader below treats as a bound.
///
/// ⛔⛔ AND ON A STICK DIM THAT SAME ABSENCE BECOMES A FILLED ZERO: the `-1` is divided by the dim's
/// cumulative stick size before being written back, with INTEGER division truncating toward zero
/// (`ddc/ddcv1.cpp:1723-1729`), and `-1 / 8 == 0`. So ONE unfilled sizing dim lands as
/// `Some(MaxDimSize(0))` when the dim is part of the stick and as [`None`] when it is not — a zero
/// page size arrived at from an absence rather than from a small extent. Pinned by
/// `an_unfilled_sizing_extent_is_a_zero_page_on_a_stick_dim_and_no_page_off_it`.
///
/// ⛔⛔ AND ITS READERS DO NOT AGREE ON WHERE ABSENCE STOPS. [`None`] here is the authority's
/// NEGATIVE entry, which is the boundary `getPageSize` draws (`maxSize < 0` is the unbounded dim,
/// `dsc/dsc2.cpp:4501`) and the boundary both `>= 0` writers draw (`finalizeAllocateLayouts`,
/// `ddc/ddcv1.cpp:1719`, and `ForceInnermostDimensionsOp`'s already-applied refusal,
/// `ddc/ddl/ddl_conversion.cpp:1879-1881`). `buildUnitView`'s is `> 0` —
/// `if (maxDimSize > 0 && size > maxDimSize)` (`dsc/dsc2.cpp:2806`) — so a ZERO entry takes the
/// `else` branch: it is never capped, and it never reaches the `DT_CHECK` that the remainder divides
/// (`:2810`), which is the one guard that would have refused it. `constructAllocElemArrLayout` draws
/// the same `> 0` boundary over the same `DT_CHECK` and turns a capping entry into a fold level's
/// `cardinality` (`ddc/ddc_fold.cpp:353-366`).
///
/// ⭐ ZERO IS REACHABLE AND IT IS NOT INERT. The pass that fills these entries divides by the
/// cumulative stick size with INTEGER division, so any extent below one stick lands on zero — an
/// unfilled dim's `-1` included, as above (`ddc/ddcv1.cpp:1723-1729`) — and the JSON importer pushes
/// back whatever the dump held (`dsc/dsc2.cpp:1761-1764`). Downstream, that zero is a bound and not
/// an absence: `getPageSize` multiplies it into the dim's page size (`dsc/dsc2.cpp:4508`), and both
/// in-file readers of that map then divide a per-dim size by it under `INDEX_TENSOR`
/// (`dsc/dsc2.cpp:3569`, `:3899`) or `DT_CHECK` `dimSize <= 0` under `VALUE_TENSOR` (`:3572-3573`).
/// So the three boundaries are three different predicates over the same [`Option`], and
/// [`Option::is_some`] is the writers' test, NEVER `buildUnitView`'s cap test.
///
/// ⭐ NEITHER OTHER CURRENCY CAN REACH THE LAYOUT:
///
/// ```compile_fail
/// use deeptools::schedule::dims::PrimaryDimTypes;
/// use deeptools::schedule::dsc2::{AllocateNode, DimSize};
/// let mut node = AllocateNode::default();
/// node.layout_dim_order.push((PrimaryDimTypes::In, Some(DimSize(8))));
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MaxDimSize(pub u32);

/// How many of one dim's elements one page of an indirect access holds — one value of
/// `getPageSize`'s answer (`dsc/dsc2.cpp:4480`, the local `pageSize`), built as the product of that
/// dim's [`MaxDimSize`] entries (`:4507-4508`) and therefore in whichever currency those entries
/// were in.
///
/// ⛔ NOT A [`DimSize`] EVEN THOUGH ITS READERS COMPARE IT WITH ONE: they compare it with a dim size
/// only after dividing or clamping — `std::ceil(float(dimSize) / pageSize.at(dim))` under
/// `INDEX_TENSOR` and `DT_CHECK_MSG(dimSize <= pageSize.at(dim), "A transfer cannot move more than
/// one page at time")` under `VALUE_TENSOR` (`dsc/dsc2.cpp:3568-3574`, `:3893-3919`), so a value of
/// this type is a page GRANULARITY and the transfer size it bounds is the [`DimSize`].
///
/// ⛔ ZERO IS A LEGAL VALUE OF IT, and it is the numerator's undoing: a single zero entry of the dim
/// makes the product zero (`:4508`) and that division is in `float`, so the dim size becomes INF
/// before `std::ceil` truncates it back into an `int`. The authority guards neither, and neither can
/// this type — see [`MaxDimSize`] for why the zero cannot be excluded upstream.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PageSize(pub u32);

/// The distance in bytes between one buffer of an allocation and the next — one value of
/// `AllocateNode::bufferOffsetCoreCorelet_` (`dsc/dsc2.h:988`).
///
/// ⛔ A STRIDE, NOT A BASE ADDRESS: `allocAllMem` computes one buffer's own capacity for this while
/// the base goes to `startAddressCoreCorelet_` beside it (`ddc/ddcv1.cpp:351-356`) — both into
/// FUNCTION-LOCAL maps that only the committing tail copies onto the node, see
/// [`AllocateNode::buffer_offset_core_corelet`] — and the L3 scheduler reads the pair back together
/// (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:4942-4944`).
///
/// ⛔ ITS NUMERATOR IS THE REQUESTED SIZE, NOT THE RESERVED ONE, and the two differ on exactly the
/// allocations [`NumBuffers::STREAMING`] marks. The request is `numBuffers * capacity` with `-1`
/// already mapped to `2` before the multiply (`ddc/ddcv1.cpp:224-226`, `:244`); a streaming
/// allocation then has its RESERVATION widened to the whole memory capacity (`:317-328`) and is
/// placed at that widened size (`:340`); but the division at `:355-356` is over `kv.second`, the
/// REQUEST pushed at `:244`, so the stride is one buffer's capacity and not half of what was
/// reserved. ⭐ A READER THAT RECOVERED THE BUFFER COUNT AS `reserved / offset` WOULD GET THE MEMORY
/// CAPACITY DIVIDED BY ONE BUFFER instead of `2`, which is why the reservation is not ported as a
/// second currency of this newtype.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BufferOffset(pub i64);

/// How many sticks one dim's data is spread across when it carries gaps — one value of
/// `AllocateNode::gapStickSpread_` (`dsc/dsc2.h:1006`).
///
/// ⛔ IT IS A MULTIPLIER IN ONE READER AND A DIVISOR IN THE OTHER, over the same dim.
/// `buildUnitView` multiplies the dim's unit-view size and every matching loop's `elemOffset_` by it
/// (`dsc/dsc2.cpp:2880-2897`), while `getBufferCapacityForNodePerDimCustomLocation` divides that
/// dim's capacity by it (`dsc/dsc2.cpp:3958-3961`) — the spread inflates the addresses and deflates
/// the capacity, so it is not a size in either direction.
///
/// Its in-scope writers are the masked-compute pass, which puts `8` on the INNERMOST layout dim
/// (`ddc/ddcv1.cpp:1704`); the internal-register transformation, which puts one new stick dim's own
/// size on every input, output and internal-register allocation of the compute
/// (`ddc/ddc_transformation.cpp:1132-1135`); and `cloneComputeForOffsetAdjustment`, which puts the
/// OUTPUT REPETITION COUNT on the innermost layout dim of the output allocation the clone now
/// shares with its original (`ddc/ddc_transformation.cpp:1356`, written at `:1378-1380`) — so a
/// filled entry is not always a stick count, and reading one as the masked pass's `8` would be
/// wrong for every cloned compute.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StickSpread(pub i32);

/// One region of one memory, reserved for one labeled data structure, one constant or one compute
/// temporary — `dsc/dsc2.h:974-1057`. The DDL conversion mints one per `AllocateOp`
/// (`ddc/ddl/ddl_conversion.cpp:780-840`), `allocAllMem` places it (`ddc/ddcv1.cpp:218-360`), and
/// the L3 scheduler reads its addresses back
/// (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:4926-4950`).
///
/// ⛔ THIS CARRIES 17 OF ALLOCATENODE'S 21 FIELDS, so four `e028_AllocateNode` and
/// `e037_AllocateNode` anchors below stay open. Three are schedule-node pointer identity, e013's
/// `name_` and the tree it hangs on: `tempStorageForCompute_` (`:978`), the `ComputeNode` whose
/// temporary this region is; `relatedIndirectAccessAlloc_` (`:999-1001`), the other half of an
/// indirect access; and `allocUsers_` (`:1007`), the reference-counted list of nodes that read or
/// write the region. All three serialize by node name and re-resolve through `nodeNamePtrMap`
/// (`dsc/dsc2.cpp:840-843`, `:919-920`, `:934-944`, `:1743-1745`, `:1798-1801`, `:1811-1823`). The
/// fourth is `startAddressCoreCorelet_` (`:985-986`), a `FoldManager<int64_t>`; ⭐ THAT TYPE HAS
/// SINCE LANDED as e026_FoldManager (`src/schedule/fold.rs`), so it is open work rather than blocked
/// work, as are the two coordinates' own former blockers — `allocateCoordinates_` and
/// `sliceViewCoordinates_` (`:1008-1009`) are CARRIED below.
///
/// ⛔ NAME IDENTITY WOULD NOT SUBSTITUTE FOR THE POINTER IN `allocUsers_`, and one pass proves it:
/// `cloneComputeForOffsetAdjustment` pushes a `clone()`d compute straight onto the list
/// (`ddc/ddc_transformation.cpp:1373`, bypassing `addAllocUser`) while the original is still on it,
/// and the clone carries the original's `name_` because the uniquifier runs later — the clone pass
/// is called at `ddc/ddcv1.cpp:3732` and `finalizeScheduleTree` only at `:3790`, where the suffix
/// is appended (`dsc/dsc2.cpp:2988-2992`). Keyed by name, the two users would collapse into one,
/// and the live-range walk that reads this list would then span one clone instead of both
/// (`ddc/ddcv1.cpp:48-56`). ⛔ THE AUTHORITY'S OWN JSON ROUND TRIP ALREADY COLLAPSES THEM: the
/// exporter `emplace`s into a `std::map<std::string, int>` keyed by name, so the colliding second
/// refcount is DROPPED and the list comes back in NAME order (`dsc/dsc2.cpp:934-944`, `:1811-1823`)
/// — which changes who `allocUsers_.begin()->first` is, and that reader `static_cast`s it to a
/// `TransferNode*` unchecked (`ddc/ddcv1.cpp:2139-2140`).
///
/// ⭐ `getPageSize` IS PORTED AND TOTAL (`:1011`, defined `dsc/dsc2.cpp:4480-4513`) — see
/// [`page_size`](Self::page_size), which takes the one unported field it reads as a parameter. Its
/// six remaining methods stay out with `allocUsers_`: `addAllocUser`, `removeAllocUser`,
/// `hasAllocUsers`, `hasAllocUser` and `clearAllocUsers` (`:1012-1046`) are that list's five
/// operations, THREE of which compare `node == userNode` BY POINTER — `addAllocUser` (`:1014`),
/// `removeAllocUser` (`:1024`) and `hasAllocUser` (`:1039`), while `hasAllocUsers` and
/// `clearAllocUsers` only test and clear it; and `print` (`:1048`, defined
/// `dsc/dsc2.cpp:4515-4573`) streams `this`, recurses into `tempStorageForCompute_` and prints
/// every `allocUsers_` name.
///
/// ⛔ NO `PartialEq`: node identity in the authority is the pointer, and `allocUsers_` and
/// `relatedIndirectAccessAlloc_` compare by it. `Clone` is IBM's own, through `InheritWithClone`
/// (`:974`).
#[derive(Clone, Debug)]
pub struct AllocateNode {
    /// The `ScheduleNode` subobject (`dsc/dsc2.h:974`,
    /// `InheritWithClone<ScheduleNode, AllocateNode>`), tagged `ALLOCATE` by `AllocateNode()`
    /// (`:975`). An allocate is a LEAF.
    pub base_class: ScheduleNode,
    /// Field: e028_AllocateNode.ldsIdx_
    ///
    /// Field: e037_AllocateNode.ldsIdx_
    ///
    /// Field: e003_AllocateNode.ldsIdx_
    ///
    /// The labeled data structure this region holds, or [`None`] for the authority's `-1`
    /// (`dsc/dsc2.h:976`). The DDL conversion sets it for a tensor allocation
    /// (`ddc/ddl/ddl_conversion.cpp:805`) and `ForceInnermostDimensionsOp` refuses any allocation
    /// without one, "Op can be applied only to tensor allocations"
    /// (`ddc/ddl/ddl_conversion.cpp:1869-1872`).
    ///
    /// ⛔ THIS IS ONE THIRD OF A THREE-WAY IDENTITY, AND THE ORDER IS FIXED:
    /// `getLdsOrConstNameOfAllocNode` names the region by `tempStorageForCompute_`'s node first,
    /// then by this, then by [`const_idx`](Self::const_idx), and answers the empty string when all
    /// three are absent (`ddc/ddcv1.cpp:20-29`). That first arm is the unported pointer, so no
    /// ported reader can reproduce the whole discriminator.
    pub lds_idx: Option<LdsIdx>,
    /// Field: e028_AllocateNode.constIdx_
    ///
    /// Field: e037_AllocateNode.constIdx_
    ///
    /// Field: e003_AllocateNode.constIdx_
    ///
    /// The constant this region holds, or [`None`] for the authority's `-1` (`dsc/dsc2.h:977`). It
    /// indexes `DesignSpaceConfig::constantInfo_`, whose entry supplies the region's name
    /// (`ddc/ddcv1.cpp:26-27`), and the DDL conversion names such a node
    /// `allocate_const<idx>_<component>` (`ddc/ddl/ddl_conversion.cpp:836-838`).
    pub const_idx: Option<ConstantId>,
    /// Field: e028_AllocateNode.component_
    ///
    /// Field: e037_AllocateNode.component_
    ///
    /// Field: e003_AllocateNode.component_
    ///
    /// Which memory the region is in (`dsc/dsc2.h:979`), taken from the `AllocateOp`'s storage
    /// (`ddc/ddl/ddl_conversion.cpp:806`).
    ///
    /// ⛔ `HBM` IS A DIFFERENT SHAPE OF ALLOCATION, NOT JUST A DIFFERENT PLACE: unless
    /// [`non_unified_alloc_in_hbm`](Self::non_unified_alloc_in_hbm) is set, its size data stage is
    /// forced to `N_` in both halves and the node is `DT_CHECK`ed to sit at the schedule tree's root
    /// (`dsc/dsc2.cpp:3624-3633`); and it is the component under which
    /// [`back_gap_core`](Self::back_gap_core) is keyed by `-1` instead of by a core
    /// (`dsc/dsc2.cpp:3943-3946`).
    pub component: SenComponent,
    /// Field: e028_AllocateNode.padding_
    ///
    /// Field: e037_AllocateNode.padding_
    ///
    /// Field: e003_AllocateNode.padding_
    ///
    /// The padding form of each dim of the region (`dsc/dsc2.h:981`), written from the `AllocateOp`
    /// (`ddc/ddl/ddl_conversion.cpp:793`). `getSizeDataStageForNode` passes it on to size the
    /// allocation (`dsc/dsc2.cpp:3613`).
    pub padding: PaddingFormType,
    /// Field: e028_AllocateNode.layoutDimOrder_
    ///
    /// Field: e028_AllocateNode.maxDimSizes_
    ///
    /// Field: e037_AllocateNode.layoutDimOrder_
    ///
    /// Field: e037_AllocateNode.maxDimSizes_
    ///
    /// Field: e003_AllocateNode.layoutDimOrder_
    ///
    /// Field: e003_AllocateNode.maxDimSizes_
    ///
    /// The dims the region is laid out over, each with its own size limit — [`None`] for the
    /// authority's negative "no limit" (`dsc/dsc2.h:982-983`). Read [`MaxDimSize`] before touching a
    /// filled one: the integer means a data-stage index before `finalizeAllocateLayouts` and an
    /// extent after.
    ///
    /// ⛔⛔ TWO OF THE AUTHORITY'S FIELDS ARE ONE FIELD HERE, BECAUSE THEIR EQUAL LENGTH IS AN
    /// INVARIANT AND THE AUTHORITY CHECKS IT AT RUNTIME THREE TIMES. Every producer `resize`s the
    /// sizes to `layoutDimOrder_.size()` with `-1` (`ddc/ddl/ddl_conversion.cpp:803`, `:1641`,
    /// `ddc/ddc_transformation_util.cpp:52`, `ddc/ddc_transformation.cpp:2116`, `:2347`,
    /// `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:561`) and `ForceInnermostDimensionsOp` inserts
    /// into both at `begin()` (`ddc/ddl/ddl_conversion.cpp:1900-1905`); against that,
    /// `finalizeAllocateLayouts` raises `DT_ERROR("Mismatch in allocate layout vectors")`
    /// (`ddc/ddcv1.cpp:1715-1717`), `getPageSize` `DT_CHECK`s the same lengths
    /// (`dsc/dsc2.cpp:4496`), and `SdscCoreletSplit` skips checking at all — it `std::find`s the dim
    /// in the layout and indexes the SIZES by that distance, unchecked
    /// (`dbo/src/Utils/sdsc_bundle/SdscCoreletSplit.cpp:76-80`). One `Vec` of pairs is what deletes
    /// all three: the mismatch is unconstructible, and that third read becomes one `find` that
    /// cannot leave the vector.
    ///
    /// ⭐ AND NEITHER OF THE TWO READS THAT WANT THE VECTORS SEPARATELY NEEDS THEM TO BE:
    /// `SdscCoreletSplit`'s is a search on the dim answering the size beside it, and the SuperDsc
    /// fingerprint linearizes the dims as one run and the sizes as another
    /// (`dsc/superdsc.cpp:1451-1452`) — two passes over one `Vec`, in the same index order, since a
    /// pair `Vec` preserves exactly the positional pairing the two runs are read back in
    /// (`dsc/superdsc.cpp:1446-1456`). `print` streams them as two runs the same way
    /// (`dsc/dsc2.cpp:4531-4539`).
    ///
    /// ⛔ INDEX 0 IS THE INNERMOST DIM: `ForceInnermostDimensionsOp` `insert`s at `begin()`
    /// (`ddc/ddl/ddl_conversion.cpp:1900-1901`), the masked-compute pass puts its stick spread on
    /// `at(0)` (`ddc/ddcv1.cpp:1704`), and `buildUnitView` appends these dims to the unit view AFTER
    /// the stick dims (`dsc/dsc2.cpp:2880-2881`, whose walk starts at `getStickSizes(...).size()`).
    ///
    /// ⛔ A DIM MAY REPEAT — `backGapCore_`'s reader says so outright, "sizes may have dimensions
    /// repeated. Add gaps to outermost" (`dsc/dsc2.cpp:2905`), and `getPageSize` multiplies every
    /// entry of one dim together (`dsc/dsc2.cpp:4507-4508`). What the DDL forbids is a repeat in the
    /// layout it copies out of the labeled data structure (`ddc/ddl/ddl_conversion.cpp:797`),
    /// "Handling of external allocations with repeated dimensions is not yet implemented" (`:798-802`)
    /// — the repeats the readers above tolerate are the ones `ForceInnermostDimensionsOp` prepends,
    /// which it inserts without ever testing whether the layout already holds that dim (`:1886-1906`).
    ///
    /// ⛔ A NEGATIVE ENTRY IS ALSO WHAT MAKES A DIM UNBOUNDED IN `getPageSize`, and it wins over
    /// every other entry of the same dim, erasing what earlier positions accumulated
    /// (`dsc/dsc2.cpp:4498-4509`); see [`page_size`](Self::page_size).
    ///
    /// ⭐ THE PAIRING IS THE GUARD, AND THE CONTROL IS THE SAME PUSH ONE FIELD APART:
    ///
    /// ```
    /// use deeptools::schedule::dims::PrimaryDimTypes;
    /// use deeptools::schedule::dsc2::{AllocateNode, MaxDimSize};
    /// let mut node = AllocateNode::default();
    /// node.layout_dim_order.push((PrimaryDimTypes::In, Some(MaxDimSize(8))));
    /// node.layout_dim_order.push((PrimaryDimTypes::Out, None));
    /// assert_eq!(node.layout_dim_order.len(), 2);
    /// ```
    ///
    /// ```compile_fail
    /// use deeptools::schedule::dims::PrimaryDimTypes;
    /// use deeptools::schedule::dsc2::AllocateNode;
    /// let mut node = AllocateNode::default();
    /// node.layout_dim_order.push(PrimaryDimTypes::In);
    /// ```
    pub layout_dim_order: Vec<(PrimaryDimTypes, Option<MaxDimSize>)>,
    /// Field: e028_AllocateNode.numBuffers_
    ///
    /// How many buffers the region holds (`dsc/dsc2.h:984`); see [`NumBuffers`] for the encoding and
    /// [`NumBuffers::STREAMING`] for the one value bridge 1 tests.
    ///
    /// ⚠️ E037 LISTED NO ANCHOR FOR IT, nor for [`back_gap_core`](Self::back_gap_core), though e028
    /// listed both.
    pub num_buffers: NumBuffers,
    /// Field: e028_AllocateNode.isStartAddrSymbolic_
    ///
    /// Field: e037_AllocateNode.isStartAddrSymbolic_
    ///
    /// Field: e003_AllocateNode.isStartAddrSymbolic_
    ///
    /// Whether the region's start address is a symbol rather than a placed address
    /// (`dsc/dsc2.h:987`).
    ///
    /// ⛔ THE INDIRECT PATH REFUSES IT: `DT_CHECK(!indAllocation->isStartAddrSymbolic_)` before the
    /// L3 scheduler reads an index allocation's address
    /// (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:5882`).
    pub is_start_addr_symbolic: bool,
    /// Field: e028_AllocateNode.bufferOffsetCoreCorelet_
    ///
    /// Field: e037_AllocateNode.bufferOffsetCoreCorelet_
    ///
    /// Field: e003_AllocateNode.bufferOffsetCoreCorelet_
    ///
    /// The buffer stride per core and corelet (`dsc/dsc2.h:988`), read as
    /// `.at(coord.at(0)).at(corelet0Id)` beside the start address
    /// (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:4942-4944`). See [`BufferOffset`]: it is a stride
    /// in bytes, not a base.
    ///
    /// ⛔ `allocAllMem`'S PER-CORE WRITE DOES NOT REACH THIS FIELD, AND THAT IS WHAT MAKES ITS
    /// SPECULATIVE RUNS SAFE. `bufferOffsetCoreCorelet_[kv.first][core][corelet] = kv.second /
    /// numBuffers` (`ddc/ddcv1.cpp:355-356`) names a FUNCTION-LOCAL homonym declared at `:136-137`
    /// and keyed by `AllocateNode*`, so the fifteen calls that pass `false` (`ddc/ddcv1.cpp:1335`,
    /// `:1381` and thirteen in the L3 scheduler's copy of the function) leave every node's map
    /// untouched even though the placement itself ran. This field's own writers are all in the
    /// `commitIfValid` tail: `kv.first->bufferOffsetCoreCorelet_ = kv.second` assigns the local's
    /// per-node value wholesale (`:407-408`), corelet 0 is then replicated to every
    /// `numCoreletsUsed_` corelet (`:413-417`) and the head core to the other used cores
    /// (`:421-428`). The L3 scheduler repeats all three (`:5714-5715`, `:5720-5724`, `:5728-5735`).
    ///
    /// ⛔⛔ SO ITS KEY SET IS NOT "EVERY USED CORE": the cross-core replication is gated on
    /// `alloc->numBuffers_ != 1` (`ddc/ddcv1.cpp:424`,
    /// `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:5731`), and only an `LX`, `L0` or `L0_SCALE`
    /// allocation is placed on every core to begin with — every other component is placed on
    /// `coreIdsUsed_.front()` as a proxy (`ddc/ddcv1.cpp:189-198`). An unbuffered non-LX allocation
    /// therefore keeps a SINGLE core key while `startAddressCoreCorelet_` beside it gets every core
    /// (`:396-405`), which is why the reader that indexes this map by an arbitrary core demands LX
    /// and `DT_CHECK_MSG`s `numBuffers_ == 1 || numBuffers_ == 2` first
    /// (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:4929`, `:4940-4941`). Pinned by
    /// `a_speculative_placement_writes_no_stride_and_an_unbuffered_one_keeps_one_core`.
    ///
    /// ⭐ ORDERED, AND THE ORDER IS EXPORTED: the authority's nested `std::map`s print in key order
    /// in the node's JSON (`dsc/dsc2.cpp:879-892`), which a [`BTreeMap`] reproduces. Both keys are
    /// `int` there and non-negative in every writer — the placement loop draws its cores from
    /// `coreIdsUsed_` and its corelets from `0` up to `numCoreletsUsed_DSC2_`
    /// (`ddc/ddcv1.cpp:189-213`, and note that the replication above bounds itself by the OTHER
    /// counter, `numCoreletsUsed_`), the replication only copies those keys, and the L3 scheduler
    /// writes `[coreId][coreletId]` (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:4991`, `:5008`,
    /// `:5035`, `:5130`) — so unlike [`back_gap_core`](Self::back_gap_core) this map has no `-1`
    /// pseudo-key and needs no [`Option`].
    pub buffer_offset_core_corelet: BTreeMap<CoreId, BTreeMap<CoreletId, BufferOffset>>,
    /// Field: e028_AllocateNode.backGapCore_
    ///
    /// The gap left after each dim, per core (`dsc/dsc2.h:989`).
    ///
    /// ⛔ [`None`] IS THE AUTHORITY'S `-1`, WHICH IS HBM — the header says so ("HBM is -1") and
    /// `getBufferCapacityForNodePerDimCustomLocation` `DT_CHECK`s that key present and reads only it
    /// when the component is HBM, taking the first entry otherwise and requiring every used core to
    /// agree, "only uniform LX back gap across cores is supported" (`dsc/dsc2.cpp:3937-3955`).
    /// [`CoreId`] is unsigned, so the pseudo-core cannot be one, and `None` sorting before every
    /// `Some` is the position `-1` takes in the authority's `std::map` — an order that reaches both
    /// the node's JSON (`dsc/dsc2.cpp:893-906`) and the SuperDsc fingerprint
    /// (`dsc/superdsc.cpp:1458-1464`).
    ///
    /// The gap is added to the dim's size in that dim's own currency, hence [`DimSize`]
    /// (`dsc/dsc2.cpp:3956`); `buildUnitView` turns each core's entry into that core's
    /// `sizesWithGaps_` (`dsc/dsc2.cpp:2900-2910`).
    pub back_gap_core: BTreeMap<PrimaryDimTypes, BTreeMap<Option<CoreId>, DimSize>>,
    /// Field: e028_AllocateNode.indirectAllocType_
    ///
    /// Field: e037_AllocateNode.indirectAllocType_
    ///
    /// Field: e003_AllocateNode.indirectAllocType_
    ///
    /// Which half of an indirect access this region is (`dsc/dsc2.h:990-994`); see
    /// [`IndirectAllocType`]. It is what [`page_size`](Self::page_size) dispatches on.
    pub indirect_alloc_type: IndirectAllocType,
    /// Field: e028_AllocateNode.indexTensorType_
    ///
    /// Field: e037_AllocateNode.indexTensorType_
    ///
    /// Field: e003_AllocateNode.indexTensorType_
    ///
    /// What an index tensor's entries hold (`dsc/dsc2.h:995-998`); see [`IndexTensorType`]. It is
    /// only meaningful under [`IndirectAllocType::IndexTensor`].
    pub index_tensor_type: IndexTensorType,
    /// Field: e028_AllocateNode.gapStickSpread_
    ///
    /// Field: e037_AllocateNode.gapStickSpread_
    ///
    /// Field: e003_AllocateNode.gapStickSpread_
    ///
    /// Per dim, how many sticks that dim's data is spread across (`dsc/dsc2.h:1006`); see
    /// [`StickSpread`] for the multiplier/divisor split between its two readers.
    ///
    /// ⭐ ORDERED, AND THE ORDER IS EXPORTED (`dsc/dsc2.cpp:926-933`). ⛔ ITS EMPTINESS IS ITSELF A
    /// TEST: `ddc/ddc_transformation.cpp:1716`, `:1723` gate a transformation on whether either side
    /// of a transfer has any spread at all.
    pub gap_stick_spread: BTreeMap<PrimaryDimTypes, StickSpread>,
    /// Whether to place this region as a plain rectangle, ignoring the data stage's symbolic volume
    /// limits — the authority's "force this allocation to be 'ghost rectangular'"
    /// (`dsc/dsc2.h:1002-1003`).
    ///
    /// It gates the whole symbolic-volume collection in
    /// `getBufferCapacityForNodePerDimCustomLocation` (`dsc/dsc2.cpp:3782`), where a gap on a dim
    /// that still has a symbolic limit is refused, "Gaps on dims with symbolic volume limit not
    /// handled" (`dsc/dsc2.cpp:3939-3940`). Its in-scope writer copies it from a reference
    /// allocation (`dsc/designSpaceConfig.cpp:129-130`).
    ///
    /// ⚠️ NEITHER WAVE'S SCHEDULER LISTED AN ANCHOR FOR IT, and none for
    /// [`non_unified_alloc_in_hbm`](Self::non_unified_alloc_in_hbm) either: both are declared across
    /// two lines with the initialiser on the second. Both have in-scope readers.
    pub ignore_symbolic_volume_limits: bool,
    /// Whether each core's slice of an HBM region lives somewhere different — the authority's "HBM
    /// allocation for each core is residing in different locations" (`dsc/dsc2.h:1004-1005`).
    ///
    /// ⛔ IT IS THE EXEMPTION FROM HBM'S FORCED `N_` SIZE: `getSizeDataStageForNode` returns `N_` in
    /// both data-stage halves for an HBM allocation ONLY while this is clear, and otherwise falls
    /// through to the ordinary sizing (`dsc/dsc2.cpp:3624-3633`).
    ///
    /// ⛔ NO IN-SCOPE WRITER SETS IT. Its only writer tree-wide is the perf-DSC translator
    /// (`dsm/translators/perfDscToSdsc/perfDscToSdsc.cpp:1870`), which this campaign does not scope,
    /// so on our path it is whatever the JSON importer read (`dsc/dsc2.cpp:1804-1805`).
    pub non_unified_alloc_in_hbm: bool,
    /// Field: e028_AllocateNode.allocateCoordinates_
    ///
    /// Field: e037_AllocateNode.allocateCoordinates_
    ///
    /// Field: e003_AllocateNode.allocateCoordinates_
    ///
    /// How this region's dims are folded (`dsc/dsc2.h:1008`) — the coordinate every other node's is
    /// derived from. `buildFoldFromAllocation` is the writer (`ddc/ddc_fold.cpp:2748-2769`, which
    /// walks the allocation's fold params and calls `addFold` per level), and
    /// `buildFoldForTransferNode` and the compute propagation are the readers
    /// (`ddc/ddc_fold.cpp:1340-1420`, `:4300-4370`).
    ///
    /// ⛔ ITS `foldConstructed_` IS THE GATE THE READERS CHECK FIRST (`ddc/ddc_fold.cpp:1340`,
    /// `:2481`), so an allocation whose tower is half built is distinguishable from one with no tower
    /// at all — see [`CoordinateType::fold_constructed`].
    pub allocate_coordinates: CoordinateType,
    /// Field: e028_AllocateNode.sliceViewCoordinates_
    ///
    /// Field: e037_AllocateNode.sliceViewCoordinates_
    ///
    /// Field: e003_AllocateNode.sliceViewCoordinates_
    ///
    /// How the SLICE VIEW of this region's dims are folded (`dsc/dsc2.h:1009`) — the core-local view
    /// `buildFoldFromAllocation` builds beside the full one (`ddc/ddc_fold.cpp:2771-2790`).
    ///
    /// ⛔ THE AUTHORITY'S JSON ROUND TRIP DROPS IT: both halves leave it a "TO DO"
    /// (`dsc/dsc2.cpp:1828`), so a DSC that has been through a file has an EMPTY slice view where the
    /// one the fold pass built was not. Nothing here restores it; the field is carried so the pass
    /// that builds it has somewhere to put it.
    pub slice_view_coordinates: CoordinateType,
}

impl Default for AllocateNode {
    /// The authority's member initialisers (`dsc/dsc2.h:976-1005`).
    ///
    /// ⭐ AND IT IS `AllocateNode()`, BASE TAG INCLUDED: that constructor passes `ALLOCATE` to
    /// `ScheduleNode` (`dsc/dsc2.h:975`).
    fn default() -> Self {
        Self {
            base_class: ScheduleNode::new(NodeType::Allocate),
            lds_idx: None,
            const_idx: None,
            component: SenComponent::NoComponent,
            padding: PaddingFormType::default(),
            layout_dim_order: Vec::new(),
            num_buffers: NumBuffers(1),
            is_start_addr_symbolic: false,
            buffer_offset_core_corelet: BTreeMap::new(),
            back_gap_core: BTreeMap::new(),
            indirect_alloc_type: IndirectAllocType::NoIndirection,
            index_tensor_type: IndexTensorType::Address,
            gap_stick_spread: BTreeMap::new(),
            ignore_symbolic_volume_limits: false,
            non_unified_alloc_in_hbm: false,
            allocate_coordinates: CoordinateType::default(),
            slice_view_coordinates: CoordinateType::default(),
        }
    }
}

impl AllocateNode {
    /// `getPageSize()` (`dsc/dsc2.h:1011`, defined `dsc/dsc2.cpp:4480-4513`): per dim, how many of
    /// that dim's elements one page holds, and absent for a dim that is not paged at all. ⭐ TOTAL —
    /// both of the authority's runtime refusals are types here.
    ///
    /// ⛔ THE ANSWER IS OFTEN NOT THIS NODE'S OWN LAYOUT: an `INDEX_TENSOR` allocation pages over
    /// the VALUE tensor's, reached through `relatedIndirectAccessAlloc_` (`dsc/dsc2.h:999-1001`) —
    /// schedule-node pointer identity, not carried here — so that link is this method's PARAMETER,
    /// and `DT_CHECK(relatedIndirectAccessAlloc_)` (`dsc/dsc2.cpp:4491`) is its type. It is taken
    /// unconditionally because the authority holds it non-null on EVERY allocation whose type is not
    /// `NO_INDIRECTION`, `DT_CHECK`ing exactly that before exporting one (`dsc/dsc2.cpp:916-921`);
    /// a `VALUE_TENSOR` reads `this` (`:4486-4488`) and a direct allocation reads neither
    /// (`:4483-4485`), so on those two arms the argument is unread and a caller holding only one
    /// allocation passes it as both.
    ///
    /// ⛔ AND AN ANSWER COMPUTED FROM `self` ON THE INDEX ARM WOULD BE SILENTLY WRONG for exactly
    /// the index allocations the paged path mints — which is why the link is a parameter rather than
    /// this method being the two total arms only.
    ///
    /// ⛔ `DT_ERROR("Unhandled indirect alloc type")` (`:4494`) is a fourth arm of a three-value
    /// enum, and an exhaustive `match` is where it goes. Both `DesignSpaceConfig` readers repeat
    /// that same unreachable arm over the ANSWER (`dsc/dsc2.cpp:3575`, `:3920`).
    pub fn page_size(
        &self,
        related_indirect_access_alloc: &Self,
    ) -> BTreeMap<PrimaryDimTypes, PageSize> {
        match self.indirect_alloc_type {
            IndirectAllocType::NoIndirection => BTreeMap::new(),
            IndirectAllocType::ValueTensor => self.page_size_of_layout(),
            IndirectAllocType::IndexTensor => related_indirect_access_alloc.page_size_of_layout(),
        }
    }

    /// `getPageSize`'s walk over the reference allocation's layout (`dsc/dsc2.cpp:4496-4512`), whose
    /// `DT_CHECK` of the two vectors' equal lengths (`:4496`) is gone into
    /// [`layout_dim_order`](Self::layout_dim_order)'s pairing.
    ///
    /// ⛔ ONE ABSENT ENTRY UNBOUNDS ITS DIM IN BOTH DIRECTIONS: it erases what earlier positions of
    /// that dim accumulated ("safe even if key not present", `:4503`) and blocks every later one
    /// (`:4505`), so a dim reaches the answer only when EVERY entry of it is filled — and the layout
    /// may repeat a dim.
    ///
    /// ⛔ ZERO IS NOT ABSENCE HERE: a filled zero multiplies its dim's page size to zero (`:4508`),
    /// and both in-file readers then divide a dim size by it in `float` (`dsc/dsc2.cpp:3569`,
    /// `:3899`). That INF is the authority's, not this port's, and it cannot be typed away from
    /// here: `finalizeAllocateLayouts` reaches zero by integer division on any extent below one
    /// stick (`ddc/ddcv1.cpp:1723-1729`), so [`MaxDimSize`] cannot exclude it.
    fn page_size_of_layout(&self) -> BTreeMap<PrimaryDimTypes, PageSize> {
        let mut page_size = BTreeMap::new();
        let mut unbounded_dims = BTreeSet::new();
        for &(dim, max_size) in &self.layout_dim_order {
            match max_size {
                None => {
                    unbounded_dims.insert(dim);
                    page_size.remove(&dim);
                }
                Some(MaxDimSize(max_size)) if !unbounded_dims.contains(&dim) => {
                    page_size.entry(dim).or_insert(PageSize(1)).0 *= max_size;
                }
                Some(_) => {}
            }
        }
        page_size
    }
}

// crustify:todo: e028_AllocateNode

// crustify:todo: e028_AllocateNode.allocUsers_

// crustify:todo: e028_AllocateNode.startAddressCoreCorelet_

// crustify:todo: e028_AllocateNode.tempStorageForCompute_

/// One constant the program supplies as data rather than reading it out of a tensor —
/// `dsc2::ConstantInfo` (`dsc/dsc2.h:46-61`). It is one entry of
/// `DesignSpaceConfig::constantInfo_`, keyed by the [`ConstantId`] an
/// [`AllocateNode::const_idx`] points back at (`dsc/designSpaceConfig.h:90`).
///
/// ⛔ THIS CARRIES 4 OF CONSTANTINFO'S 5 FIELDS, so the `e009_ConstantInfo`, `e028_ConstantInfo` and
/// `e030_ConstantInfo` anchors below stay open. `data_` (`dsc/dsc2.h:49-50`) is carried now that
/// `util/foldManager/` is ported ([`FoldManager`], `e026_FoldManager`), and the `e009` scan does not
/// name it at all, so only the `e030` anchor did. `allocations_` (`dsc/dsc2.h:52`) is a
/// `std::map<SenComponents, AllocateNode*>` of NON-OWNING aliases into the schedule tree: the DDL
/// conversion hangs the minted node on its parent block and aliases it here in the same breath
/// (`ddc/ddl/ddl_conversion.cpp:826-832`), and the PE/SFP work split clones a node into a second
/// component the same way, refusing a component that already has one
/// (`ddc/ddc_transformation_util.cpp:1407-1417`). What blocks it is that this port has no node
/// IDENTITY to alias WITH: [`ScheduleNode`] carries no parent or child link — its `prev_` anchor is
/// still open — and [`AllocateNode`]'s own pointer fields are open anchors above. An owned
/// `BTreeMap<SenComponent, AllocateNode>` would give the constant a second copy of a node the tree
/// owns, and the placement written through the tree would not be visible here — which is exactly
/// what its readers draw back out of it: `fillDataInfo` (`ddc/ddcv1.cpp:2386-2388`) and the L3
/// scheduler's own copy of that loop (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:5810-5817`) clone
/// the node's PLACED address and its symbolic flag through the alias, `getAllocation` hands the node
/// itself back (`dsc/dsc2.cpp:2632`), and the DDL conversion resolves a data connect's allocation
/// through it (`ddc/ddl/ddl_conversion.cpp:2861`).
///
/// ⛔ THE ALIAS IS DERIVED AND THE ROUND TRIP IS NOT LOSSY: the JSON exporter writes it as the node's
/// NAME and the importer's `constantInfo_` arm has no case for it (`dsc/dsc2.cpp:95-102`,
/// `:1134-1150`), but importing the schedule tree RELINKS it —
/// `constantInfo_[an->constIdx_].allocations_[an->component_] = an` for every ALLOCATE node, beside
/// the labeled DS's own link (`:1831-1837`). `SdscHasher`'s walk of it is commented out
/// (`dsc/superdsc.cpp:1515-1521`), so it is not part of the fingerprint either.
///
/// ⛔ AND ABSENT IS NOT NULL: a constant may legitimately have no entry — "Keep
/// constInfo.allocations_ empty because we do not need to allocate a data structure to store this
/// constant" (`dsc/dsc2.cpp:5301-5302`) — deallocation ERASES the key (`:2518`), and `getAllocation`
/// tests the two states separately, `!count(storage) || !at(storage)` (`:2618-2619`), returning
/// `nullptr` for both under `allowMissingAlloc` and `DT_ERROR`ing otherwise.
///
/// ⛔ ITS ONE METHOD IS [`assign`](Self::assign) AND IT IS NOT [`Clone`]: the copy assignment's four
/// member assignments are `dataFormat_`, `name_`, `allocations_` and `data_.clone(rhs.data_)`
/// (`dsc/dsc2.h:54-60`), so one of the four — the alias map — has no field here, and the fifth,
/// [`is_data_symbolic`](Self::is_data_symbolic), is assigned by neither of the four. See
/// [`assign`](Self::assign) for the two divergences from the copy constructor, both measured against
/// the authority.
///
/// ⛔ NO `PartialEq`: the authority declares none, and its own duplicate test is not an equality —
/// it compares `name_`, `dataFormat_` and the datum's element COUNT, not the datum
/// (`ddc/ddl/ddl_conversion.cpp:706-714`). A derive would compare the whole datum plus
/// [`is_data_symbolic`](Self::is_data_symbolic), and [`FoldManager`]'s own `operator==` ignores the
/// fold dimensions' LABELS (`util/foldManager/foldInfrastructure.h:1101-1169`), so neither reading is
/// the one that test takes.
#[derive(Clone, Debug)]
pub struct ConstantInfo {
    /// Field: e028_ConstantInfo.dataFormat_
    /// Field: e030_ConstantInfo.dataFormat_
    /// Field: e009_ConstantInfo.dataFormat_
    ///
    /// The format the datum's values are encoded in — the field's own comment says so, "values
    /// encoded in the specified format" (`dsc/dsc2.h:47`, `:50`).
    ///
    /// ⛔ [`DataFormats::Invalid`] IS THE INITIALISER AND THREE READERS DIVIDE BY ITS WIDTH:
    /// `replicationFactor_` (`ddc/ddcv1.cpp:455-457`), the external-constant path's `bitsPerElem`
    /// (`ddc/ddl/ddl_conversion.cpp:697-698`), and `getBlockTransferSizePerDim`'s
    /// `CONSTANT_TO_CONSTANT` dummy dim, `1024 / width`, in this very file
    /// (`dsc/dsc2.cpp:3486-3492`); bridge 1 takes it whole as the destination precision
    /// (`dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:2712`) and `SdscHasher` pushes
    /// its raw ORDINAL into the SDSC's identity (`dsc/superdsc.cpp:1514`, compared and hashed at
    /// `dsc/superdsc.h:172`, `:187`).
    ///
    /// ⛔ AND THE TABLE DOES NOT REFUSE THE INITIALISER: `INVALID`'S ENTRY IS `-1`
    /// (`util/sendefs/sendefs.cpp:131`), so `1024 / -1` is -1024 loads and `bitsPerElem * numElems`
    /// goes NEGATIVE, slipping under the very gate that bounds it, "NumElements larger than opconst
    /// size" (`ddc/ddl/ddl_conversion.cpp:702-704`). The one guard that stops `INVALID` is the parse
    /// check, "Invalid type name" (`:654-656`) — which is why [`DataFormats::bit_width`] answers
    /// [`None`] rather than a width no reader can use.
    ///
    /// ⛔ THE OP-CONST UNPACKER IS NOT A READER OF THIS FIELD: its mask comes from the LABELED DS's
    /// same-named field (`dsc/dsc2.cpp:5260-5265` reads `lds.dataFormat_`, and `lds` is
    /// `dsc.labeledDs_.at(ldsIdx)` at `:4773`). That path WRITES this field from it (`:5282`) and
    /// `DT_CHECK`s that the two agree when it reuses the entry (`:5231`), so they hold the same
    /// value — but the lookup is on `LabeledDsInfo`, a type this port does not carry.
    ///
    /// ⛔ ONE WRITER REWRITES IT WHILE CONVERTING THE VALUE: a `SEN169_FP16` constant feeding an
    /// `IEEE_FP32` compute op is stored as `IEEE_FP32` with its datum put through `Fp16BinToFloat`
    /// (`ddc/ddl/ddl_conversion.cpp:667-673`), so this is not simply the DDL's declared type.
    pub data_format: DataFormats,
    /// Field: e028_ConstantInfo.name_
    /// Field: e030_ConstantInfo.name_
    /// Field: e009_ConstantInfo.name_
    ///
    /// The constant's name (`dsc/dsc2.h:48`) — the DDL's own for a defined constant
    /// (`ddc/ddl/ddl_conversion.cpp:694-695`) or the external constant's (`:700`) — which is also
    /// the name of the allocation that holds it, in `Ddc::getLdsOrConstNameOfAllocNode`
    /// (`ddc/ddcv1.cpp:25-26`) and again in the L3 scheduler's own copy of that function
    /// (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:5500`), so stage 2a names allocations by this
    /// field too.
    ///
    /// ⛔ IT IS AN IDENTITY AND A BEHAVIOUR SWITCH, NOT A LABEL. An external constant is matched
    /// against the container BY NAME and its other properties are then required to agree, else
    /// "Constant found in DSC but properties do not match", and a name with no entry is
    /// "Missing external constant in DSC" (`ddc/ddl/ddl_conversion.cpp:706-720`). The padding path
    /// reuses the entry named `padval` and `DT_CHECK`s its format (`dsc/dsc2.cpp:5228-5235`), and a
    /// constant named `useZeroMean` holding 1 turns an `EXX2` compute op into `EXX2_ZEROMEAN`
    /// (`ddc/ddcv1.cpp:2064-2072`).
    ///
    /// ⭐ A `String` AND NOT AN ENUM, because the set is open: it is whatever attribute the DDL
    /// carries (`ddc/ddl/ddl_conversion.cpp:694-695`), and those three literals are compared against
    /// it rather than enumerating it.
    pub name: String,
    /// Field: e030_ConstantInfo.data_
    ///
    /// The constant's values, one payload per point of the core/corelet/SDSC fold space — the field's
    /// own comment is "core/corelet/sdsc folds, values encoded in the specified format"
    /// (`dsc/dsc2.h:49-50`), the format being [`data_format`](Self::data_format).
    ///
    /// ⛔ ONE PAYLOAD IS A WHOLE `Vec<i64>`, NOT ONE VALUE, and the element COUNT is what two readers
    /// check: the external-constant match refuses a DSC entry whose `getSingleData().size()` differs
    /// from the DDL's `numElems` (`ddc/ddl/ddl_conversion.cpp:711`) and the replication factor divides
    /// by it (`ddc/ddcv1.cpp:454`). `Vec<i64>` and not a newtype because it is the payload
    /// [`FoldManager`] is instantiated at (`util/foldManager/foldInfrastructure.h:886`), the one list
    /// payload in scope.
    ///
    /// ⛔ ITS BUILT SHAPE IS `buildAllConstantFoldSpace`'s AND NOTHING ELSE'S: both writers hand it
    /// the SDSC's own fold props and then insert at one coordinate
    /// (`ddc/ddl/ddl_conversion.cpp:687-693`, `dsc/dsc2.cpp:5295-5299`), so every level is a constant
    /// fold and [`FoldManager::build_affine_dim`] is not reachable from here — which is why the
    /// payload has none.
    ///
    /// ⭐ AND `hasZeroFoldDim()` IS THE STATE ITS READERS BRANCH ON, not a value: the PCFG translator
    /// `DT_CHECK`s it false before cloning the datum per core and corelet (`dsc/dsc2Pcfg.cpp:1365`,
    /// `:1944`) and the DSC's own print skips a constant that has it (`dsc/designSpaceConfig.cpp:4787`)
    /// — so the default-constructed zero-dimension space is a distinguishable "no datum built yet".
    pub data: FoldManager<Vec<i64>>,
    /// Field: e028_ConstantInfo.isDataSymbolic_
    /// Field: e030_ConstantInfo.isDataSymbolic_
    /// Field: e009_ConstantInfo.isDataSymbolic_
    ///
    /// Whether the datum holds a [`VariableSymbol`] still to be resolved rather than a value
    /// (`dsc/dsc2.h:51`). `stzJumpAddr` is the worked example: its datum is the reserved dynamic
    /// execution address symbol and this is set beside it
    /// (`dbo/src/Utils/sdsc_bundle/ProgramCorrection.cpp:1031`, `:1151-1154`); the gather path sets
    /// it on each base-address byte from its metadata's `is_base_addr_symbolic`
    /// (`dbo/src/Transforms/sdsc_bundle/GatherIndexConversion.cpp:176-231`).
    ///
    /// ⭐ BRIDGE 1 IS ONE READER: it becomes the `is_symbol` attribute on the emitted
    /// `ConstantBitstreamOp`, on the single-fold path and on every per-fold one
    /// (`dsc-based-utils/DSC2ToDataflowIR/V3/SNDSCLowering.cpp:479-480`, `:515-516`, reached from
    /// `dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:2492-2496`).
    ///
    /// ⛔ AND IT IS A REFUSAL PREDICATE, NOT ONLY AN ATTRIBUTE: the PCFG translator `DT_CHECK`s
    /// `!constInfo.isDataSymbolic_` before cloning the datum, on the transfer-to-constant path and
    /// again on the compute-transfer one (`dsc/dsc2Pcfg.cpp:1366`, `:1945`). Those are the only
    /// other readers in the authority, and they are off our path by the campaign's own decision —
    /// so `stzJumpAddr` is a constant bridge 1 emits and PCFG refuses, which is a property of this
    /// flag and not of either consumer.
    ///
    /// ⛔ THE AUTHORITY'S COPY ASSIGNMENT DROPS IT: `operator=` assigns the other four members and
    /// never touches this one (`dsc/dsc2.h:54-60`), so an assignment leaves the destination's flag
    /// standing. The copy CONSTRUCTOR does carry it, and that is what
    /// `emplace(myId, std::move(myConstInfo))` resolves to, because declaring a copy assignment
    /// operator suppresses the implicit move constructor (`ddc/ddl/ddl_conversion.cpp:725`);
    /// [`Clone`] here is that constructor. The one in-scope assignment writes onto a freshly
    /// default-constructed entry whose flag is already `false` (`dsc/dsc2.cpp:5307`), so nothing on
    /// our path turns on the omission today, and the JSON round trip carries the flag on both sides
    /// (`dsc/dsc2.cpp:93-94`, `:1146-1147`).
    pub is_data_symbolic: bool,
}

impl Default for ConstantInfo {
    /// The authority's member initialisers (`dsc/dsc2.h:47-51`). ⛔ THE FORMAT IS `INVALID`, not
    /// [`DataFormats::default()`] — that is `ComputeNode::dataFormat_`'s initialiser
    /// (`dsc/dsc2.h:934`), a different field's. [`data`](ConstantInfo::data) has no initialiser at
    /// all and is default-constructed, which is a zero-dimension fold space holding one empty vector —
    /// measured: `getNumDims() == 0`, `hasZeroFoldDim()` true, and `getData({})` empty.
    fn default() -> Self {
        Self {
            data_format: DataFormats::Invalid,
            name: String::new(),
            data: FoldManager::<Vec<i64>>::new(),
            is_data_symbolic: false,
        }
    }
}

impl ConstantInfo {
    /// `operator=(const ConstantInfo&)` (`dsc/dsc2.h:54-60`) — the authority's copy ASSIGNMENT, which
    /// is NOT [`Clone`] here: that is its copy CONSTRUCTOR, and declaring this one is what suppresses
    /// the implicit move constructor so `emplace(myId, std::move(myConstInfo))`
    /// (`ddc/ddl/ddl_conversion.cpp:725`) resolves to the constructor while
    /// `dsc.constantInfo_[constId] = constInfo` (`dsc/dsc2.cpp:5307`) resolves to this. Named after
    /// [`FoldManager::assign`], the same operator's port one layer down.
    ///
    /// ⛔ FOUR ASSIGNMENTS FOR FIVE MEMBERS: IT DROPS
    /// [`is_data_symbolic`](Self::is_data_symbolic) (`dsc/dsc2.h:55-58`), so the DESTINATION'S OWN
    /// FLAG STANDS. Measured both ways — a source whose flag is true leaves a fresh destination
    /// `false`, where [`Clone`] carries `true`; a source whose flag is false leaves a destination
    /// whose flag is `true` still `true`.
    ///
    /// ⛔ AND IT CLONES THE DATUM RATHER THAN ASSIGNING IT, because [`FoldManager::assign`] refuses a
    /// fold space of a different dimensionality or cardinality
    /// (`util/foldManager/foldInfrastructure.h:922-933`) and a fresh destination always is one.
    /// `clone` with no ignored dimension clears the destination and rebuilds rhs's space
    /// (`:987-1019`), which is [`FoldManager`]'s own [`Clone`]. Measured: a one-dimension destination
    /// takes a two-dimension source whole.
    ///
    /// ⛔ THE FOURTH ASSIGNMENT HAS NO FIELD HERE: `allocations_ = rhs.allocations_` (`:57`) — see the
    /// type's note and the open `e009_ConstantInfo.allocations_` anchor below.
    pub fn assign(&mut self, rhs: &Self) {
        self.data_format = rhs.data_format;
        self.name = rhs.name.clone();
        self.data = rhs.data.clone();
    }
}

// crustify:todo: e009_ConstantInfo

// crustify:todo: e009_ConstantInfo.allocations_

// crustify:todo: e028_ConstantInfo

// crustify:todo: e028_ConstantInfo.allocations_

// crustify:todo: e030_ConstantInfo

// crustify:todo: e030_ConstantInfo.allocations_

/// A transfer's zero-pad size in elements — the `int` a `FoldManager<int>` level carries
/// (`dsc/dsc2.h:810-811`), and the currency of both readers as well as of the `alphas` and `betas`
/// the builder takes.
///
/// ⛔ SIGNED, AND NEGATIVE VALUES ARE PRODUCED ON PURPOSE: the one production builder passes
/// `-coreOffset` and `-chunkOffset` as alphas and a beta that subtracts both again
/// (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:5388-5451`), which is why both readers clamp at 0
/// rather than trust the fold (`dsc/dsc2.cpp:4707-4708`, `:4729-4731`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PadSize(pub i32);

/// A work-slice index — the coordinate of the OUTER of the two folded dims
/// (`dsc/dsc2.cpp:4669-4671`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WkSliceIdx(pub i64);

/// A chunk index within one work slice — the coordinate of the INNER folded dim (`:4674-4675`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ChunkIdx(pub i64);

/// How many chunks one work-slice pad walk may visit — `numChunks` (`dsc/dsc2.cpp:4685`).
///
/// ⛔ NOT THE STORED `chunk_index` EXTENT, and the walk does not clamp it to one: the caller derives
/// it per dim (`dsc/dsc2.cpp:4827`), and measured, a value past the extent makes the reader throw
/// unless the walk breaks first — immediately on the back end, whose first coordinate is
/// `numChunks - 1`.
///
/// ⛔ A `u32` OVER THE AUTHORITY'S `int`, AND THE TRADE IS NOT SYMMETRIC. The negative half costs
/// nothing: measured, -1 and `INT_MIN` both answer 0 at both ends, exactly as 0 does, so
/// `NumChunks(0)` speaks for all of them. The residual is the other half — counts past `INT_MAX` are
/// spellable here and unspellable there, and unlike [`WkSliceIdx`] there is no wider authority seam
/// to appeal to, because `numChunks` is an `int` the whole way down.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NumChunks(pub u32);

/// The element offset one fully padded chunk contributes to a work slice's pad (`:4830-4831`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ChunkOffset(pub i32);

/// A chunk's padded extent — both the cap a pad size is legal up to and the threshold that ends the
/// work-slice walk (`dsc/dsc2.cpp:4705-4710`, `:4718-4731`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ChunkSizePadded(pub i32);

/// Which end of a transfer's data a pad sits at — the authority's `const bool isPadFront`
/// (`dsc/dsc2.h:784`, `:792`, `:796`, `:800`, `:804`), which selects one of two parallel field
/// pairs at every one of its five declaration sites.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PadEnd {
    /// `isPadFront == true`, and the walk visits chunk 0 first (`dsc/dsc2.cpp:4699-4700`).
    Front,
    /// `isPadFront == false`, and the walk visits `numChunks - 1` first.
    Back,
}

/// `TransferPadInfo::FoldDimPosition` (`dsc/dsc2.h:768-772`) — the two folded dims, outer first.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FoldDimPosition {
    /// `WORK_SLICE_FOLDDIM`, labelled `wkslice_index` (`dsc/dsc2.cpp:4669-4671`).
    WorkSlice = 0,
    /// `CHUNK_FOLDDIM`, labelled `chunk_index` (`:4674-4675`).
    Chunk = 1,
}

impl FoldDimPosition {
    /// `TOTAL_FOLDDIM_NUM` — the arity `buildPadSizes` `DT_CHECK`s on all three of its vectors
    /// (`dsc/dsc2.cpp:4612-4615`), here an array length, so the check is E0308.
    pub const COUNT: usize = 2;

    /// Outer to inner, which is the order `buildFoldSpace` nests the levels in.
    pub const ALL: [Self; Self::COUNT] = [Self::WorkSlice, Self::Chunk];

    /// The label `buildTransferFoldDim` sets on this position's prop (`dsc/dsc2.cpp:4669-4671`,
    /// `:4674-4675`).
    ///
    /// ⛔ NOTHING MATCHES ON IT — see [`FoldDimProp`]'s own note — so it reaches the metadata dump and
    /// nowhere else, and a `FoldManager`'s `operator==` does not compare it
    /// (`util/foldManager/foldInfrastructure.h:1104-1109`).
    pub const fn label(self) -> &'static str {
        match self {
            Self::WorkSlice => "wkslice_index",
            Self::Chunk => "chunk_index",
        }
    }
}

/// ⛔ E0308 IF A POSITION IS EVER ADDED: `COUNT` is the array length every caller's `sizes`,
/// `alphas` and `betas` are checked against.
const _: [(); FoldDimPosition::COUNT] = [(); FoldDimPosition::Chunk as usize + 1];

/// One dim's pad-size fold: the [`FoldManager`] the authority's `MapWithFMHelper` keeps per dim, over
/// the two `FoldDimProp`s stored beside it (`dsc/dsc2.cpp:4655-4682`).
///
/// ⛔ THE PROPS AND THE TREE ARE ONE OWNER HERE BECAUSE IN C++ THEY ALIAS: `FoldManager::dim_prop_`
/// holds `const FoldDimProp*` INTO `transferPadFrontFoldProps`
/// (`util/foldManager/foldInfrastructure.h:2909`, `:888`), so `DT_CHECK_MSG(!foldProps.count(dim))`
/// (`dsc/dsc2.cpp:4662`) is the only thing standing between a rebuild's `resize` and a dangling
/// pointer. [`FoldManager`] co-owns its props for exactly that reason, so this is the manager alone
/// and the two `currDimFoldProps` entries (`:4664-4676`) live inside it.
#[derive(Clone, Debug, PartialEq, Eq)]
struct PadSizeFold(FoldManager<i32>);

impl PadSizeFold {
    /// `buildTransferFoldDim` (`dsc/dsc2.cpp:4655-4682`) and the four `insert*ForKey` calls that
    /// follow it (`:4624-4635`), which are one construction: `addKeyBuildFoldSpace` (`:4678-4681`)
    /// nests two affine levels with `alpha_{}`/`beta_{}` at zero, and each `insertAlphaForKey(..,
    /// pos)` then writes the level `collectFoldFunctionAtLevel(pos)` reaches — 0 the non-leaf, 1 its
    /// leaf.
    ///
    /// ⛔ NEITHER [`None`] BELOW IS REACHABLE, and they are propagated rather than asserted away.
    /// [`FoldManager::build_fold_space_dims`] refuses an empty list or a kind it cannot build, and
    /// this list is [`FoldDimPosition::COUNT`] entries of `Affine` over the `i32` payload that HAS the
    /// affine constructor; [`FoldManager::insert_alpha_beta`] refuses a position past the end or a
    /// level that is not Affine, and every position comes from [`FoldDimPosition::ALL`]. The caller's
    /// own [`None`] is the authority's one real check (`:4662`) and it is the same channel.
    fn new(
        sizes: [FoldDimSize; FoldDimPosition::COUNT],
        alphas: [PadSize; FoldDimPosition::COUNT],
        betas: [PadSize; FoldDimPosition::COUNT],
    ) -> Option<Self> {
        let dim_prop = FoldDimPosition::ALL.map(|pos| {
            (
                FoldDimProp::new(sizes[pos as usize], pos.label()),
                BaseFuncType::Affine,
            )
        });
        let mut fm = FoldManager::<i32>::new();
        fm.build_fold_space_dims(&dim_prop)?;
        for pos in FoldDimPosition::ALL {
            let idx = pos as usize;
            fm.insert_alpha_beta(&alphas[idx].0, &betas[idx].0, FoldDimPos(pos as i32))?;
        }
        Some(Self(fm))
    }

    /// The manager every read goes through, including the `FoldDimProp` readers the authority reaches
    /// via `transferPadFrontFoldProps` and this port reaches via `dim_prop_`.
    const fn manager(&self) -> &FoldManager<i32> {
        &self.0
    }

    /// `MapWithFMHelper::getDataForKey` past the key check — its three-argument callers reach the
    /// variadic overload (`util/foldManager/mapWithFMHelper.h:253-258`), which packs a
    /// `std::deque<int64_t>` and delegates to `:206-209`; that is `DT_CHECK(key_val_.count(key))`
    /// and then [`FoldManager::get_data`], which is `isLegal`
    /// (`util/foldManager/foldInfrastructure.h:1666-1681`) and the walk.
    ///
    /// ⛔ THE RANGE TEST IS SIGNED AND THAT IS NOT A BUG TO FIX: `getSize() <= idx` widens a
    /// `uint32_t` extent to `int64_t` (`:1677`), so a NEGATIVE coordinate is LEGAL and computes.
    /// Measured on the authority: work slice -1 answers 65 where work slice 2 of 2 throws, and a
    /// zero extent refuses coordinate 0 while still answering 65 for -1.
    /// ⛔ AND ITS ARITY HALF CANNOT REFUSE HERE (`:1668-1669`): the coordinate array and the fold
    /// space are both [`FoldDimPosition::COUNT`] long, which is what makes two coordinates the
    /// signature rather than a checked length.
    fn data(&self, wk_slice: WkSliceIdx, chunk: ChunkIdx) -> Option<PadSize> {
        self.manager()
            .get_data(&[FoldDimIndex(wk_slice.0), FoldDimIndex(chunk.0)])
            .map(PadSize)
    }
}

/// Replaces: e024_TransferPadInfo
///
/// Replaces: e037_MapWithFMHelper
///
/// ⛔ TWO UNITS, ONE TYPE: e037 is a facade whose only member is a reference to a map this type owns
/// (`util/foldManager/mapWithFMHelper.h:830-831`), and `dsc/dsc2.h:808-809` is its only in-scope
/// instantiation, so dissolving it into the owner discharges it. `crate::schedule::fold_helper`
/// records the five methods those two instances reach and where each of them landed.
///
/// A transfer's LX zero-pad sizes, one two-level affine fold per padded dim per end
/// (`dsc/dsc2.h:755-812`) — what `L3DlOpsScheduler` writes onto a `TransferNode` so that
/// `dsc/dsc2.cpp:4736-5817` can turn padding into condition and transfer nodes.
///
/// ⛔ NO [`Clone`], AND THE ABSENCE IS THE `DT_CHECK`: the authority's copy constructor is
/// "Do nothing on purpose" (`dsc/dsc2.h:761-764`) — it rebuilds the two helper references and copies
/// NOTHING, so a copy is EMPTY. `dsc/dsc2.cpp:5701-5702` clones a padded transfer node and `:5792`
/// `DT_CHECK_MSG`s the clone `isEmpty()`; measured, source non-empty and copy empty. A `Clone` that
/// silently dropped the folds would be the astonishing one, so the only way to spell that copy here
/// is [`Default`], which makes IBM's runtime check a fact of the type.
/// ⛔ AND ITS DEFAULTED MOVE CONSTRUCTOR (`dsc/dsc2.h:760`) IS A USE-AFTER-FREE, unrepresentable
/// here: `MapWithFMHelper`'s only member is a REFERENCE to the sibling map
/// (`util/foldManager/mapWithFMHelper.h:829-831`), so a move copies a reference that still points
/// into the moved-FROM object. Measured — the moved-to object reported `isEmpty() == 0` with zero
/// keys, answered the moved-FROM object's data after that object was rebuilt, and destroying the
/// source made AddressSanitizer report a use-after-scope inside `getAllKeys`
/// (`util/foldManager/mapWithFMHelper.h:53`). A Rust move carries the
/// storage, so the [`unit_tests`] case for that is a deliberate divergence.
///
/// The control is what makes the `compile_fail` case evidence — stable rustdoc does not check the
/// annotated error code:
/// ```compile_fail,E0599
/// let _ = deeptools::schedule::dsc2::TransferPadInfo::default().clone();
/// ```
/// ```
/// use deeptools::schedule::dsc2::TransferPadInfo;
/// assert!(TransferPadInfo::default().is_empty(), "`TransferPadInfo(const TransferPadInfo&)`");
/// ```
#[derive(Debug, Default, PartialEq, Eq)]
pub struct TransferPadInfo {
    /// Field: e024_TransferPadInfo.transferPadFrontFoldProps
    ///
    /// Field: e024_TransferPadInfo.transferPadFrontSize_
    ///
    /// Field: e024_TransferPadInfo.transferPadFrontSizeHelper
    ///
    /// Field: e037_MapWithFMHelper.key_val_
    ///
    /// ⛔ THREE OF THE AUTHORITY'S FIELDS ARE ONE FIELD HERE, and the helper is not a field at all:
    /// `transferPadFrontSizeHelper` is a `MapWithFMHelper` whose ONLY member is
    /// `std::map<Dkey, FoldManager<Dval>>& key_val_` (`util/foldManager/mapWithFMHelper.h:829-831`)
    /// bound to `transferPadFrontSize_` in every constructor (`dsc/dsc2.h:759-767`) — a facade over
    /// the sibling map, holding no state of its own. `transferPadFrontFoldProps` is then the storage
    /// map's `dim_prop_` pointers point INTO; see [`PadSizeFold`].
    ///
    /// ⛔ AND `key_val_` IS ANCHORED HERE AND ON [`back`](Self::back) BOTH: the authority binds that
    /// ONE member twice, once per end (`dsc/dsc2.h:758-759`), so the two bindings are two distinct
    /// fields of this type and neither may go unnamed.
    front: BTreeMap<PrimaryDimTypes, PadSizeFold>,
    /// Field: e024_TransferPadInfo.transferPadBackFoldProps
    ///
    /// Field: e024_TransferPadInfo.transferPadBackSize_
    ///
    /// Field: e024_TransferPadInfo.transferPadBackSizeHelper
    ///
    /// Field: e037_MapWithFMHelper.key_val_
    ///
    /// The same three fields at the other end, and independent of [`front`](Self::front): measured,
    /// building only the front leaves every back query throwing.
    back: BTreeMap<PrimaryDimTypes, PadSizeFold>,
}

impl TransferPadInfo {
    /// `isEmpty()` (`dsc/dsc2.h:774-776`) — the predicate two schedulers gate the whole
    /// padding-to-schedule-tree transformation on (`ddc/ddcv1.cpp:2860`, `dsc/dsc2.cpp:4761`).
    pub fn is_empty(&self) -> bool {
        self.front.is_empty() && self.back.is_empty()
    }

    /// `buildPadFrontSizes` and `buildPadBackSizes` (`dsc/dsc2.cpp:4638-4650`), which are
    /// `buildPadSizes(.., isPadFront)` (`:4607-4636`) with the flag fixed — one function with
    /// [`PadEnd`] as an argument.
    ///
    /// ⛔ [`None`] IS `DT_CHECK_MSG(!foldProps.count(dim), "Expect empty fold properties.")`
    /// (`:4662`) — a dim can be built once per end, and the authority throws on the second attempt
    /// rather than rebuilding. Its other three checks are one `DT_CHECK_MSG` on the arity of all
    /// three vectors (`:4612-4615`), gone into the array lengths.
    ///
    /// ⛔ AND A NEGATIVE SIZE IS NOT A NEGATIVE EXTENT THERE, IT IS A FOUR-BILLION ONE:
    /// `setSize` takes a `uint32_t` (`util/foldManager/foldInfrastructure.h:129`, field `:153`) and
    /// the caller hands it an `int` (`dsc/dsc2.cpp:4669`, `:4674`), so the conversion happens at the
    /// call. Measured on the authority, `sizes = {-1, 3}` stores an extent of 4294967295 and
    /// `isLegal`'s range test then admits EVERY `int` coordinate — work slice 1000000 answers
    /// -39999975 instead of throwing. [`FoldDimSize`] is a `u32`, so the negative is unspellable
    /// here, but 4294967295 is not: the vacuous guard is reachable and only reachable that way.
    pub fn build_pad_sizes(
        &mut self,
        end: PadEnd,
        dim: PrimaryDimTypes,
        sizes: [FoldDimSize; FoldDimPosition::COUNT],
        alphas: [PadSize; FoldDimPosition::COUNT],
        betas: [PadSize; FoldDimPosition::COUNT],
    ) -> Option<()> {
        let folds = match end {
            PadEnd::Front => &mut self.front,
            PadEnd::Back => &mut self.back,
        };
        match folds.entry(dim) {
            Entry::Occupied(_) => None,
            Entry::Vacant(slot) => {
                slot.insert(PadSizeFold::new(sizes, alphas, betas)?);
                Some(())
            }
        }
    }

    /// `getPadFrontOrBackDimsSet` (`dsc/dsc2.h:783-788`) through `MapWithFMHelper::getAllKeys`
    /// (`util/foldManager/mapWithFMHelper.h:51-55`).
    ///
    /// ⛔ AN ORDERED ITERATOR RATHER THAN A `std::set` BY VALUE, WHICH IS WHAT THE ONE CALLER WANTS:
    /// it `std::set_union`s the two ends into a vector (`dsc/dsc2.cpp:4814-4820`), so it needs the
    /// ascending order a `std::set` gave it and never the container. A [`BTreeMap`]'s keys are
    /// already in that order, so this allocates nothing.
    pub fn pad_dims(&self, end: PadEnd) -> impl Iterator<Item = PrimaryDimTypes> + '_ {
        let folds = match end {
            PadEnd::Front => &self.front,
            PadEnd::Back => &self.back,
        };
        folds.keys().copied()
    }

    /// `getWkSlicePadSizeFrontOrBack` (`dsc/dsc2.cpp:4684-4716`) — one work slice's total pad, walked
    /// chunk by chunk from the padded end until the first chunk that is not fully padded.
    ///
    /// ⛔ `chunkSizePadded` IS THE ONLY THING THAT ENDS THE WALK EARLY, so a 0 or negative one visits
    /// every chunk: measured, `chunkSizePadded = 0` over three chunks answers `3 * chunkOffset`.
    /// ⛔ [`None`] IS THE READER THROWING MID-WALK, on an unknown dim
    /// (`util/foldManager/mapWithFMHelper.h:207`) or a coordinate past its extent — reachable exactly
    /// when [`NumChunks`] exceeds the stored `chunk_index` extent and no chunk breaks the walk first.
    pub fn wk_slice_pad_size(
        &self,
        end: PadEnd,
        dim: PrimaryDimTypes,
        wk_slice: WkSliceIdx,
        num_chunks: NumChunks,
        chunk_offset: ChunkOffset,
        chunk_size_padded: ChunkSizePadded,
    ) -> Option<PadSize> {
        let fold = match end {
            PadEnd::Front => self.front.get(&dim)?,
            PadEnd::Back => self.back.get(&dim)?,
        };
        let mut num_chunks_visited = 0u32;
        let mut partial_pad_size = 0i32;
        while num_chunks_visited < num_chunks.0 {
            let curr_chunk_idx = match end {
                PadEnd::Front => num_chunks_visited,
                PadEnd::Back => num_chunks.0 - num_chunks_visited - 1,
            };
            // "the agreement is that negative pad size is treated as zero" (`:4702-4704`).
            let curr_pad_size = fold
                .data(wk_slice, ChunkIdx(i64::from(curr_chunk_idx)))?
                .0
                .max(0);
            if curr_pad_size < chunk_size_padded.0 {
                partial_pad_size = curr_pad_size;
                break;
            }
            num_chunks_visited += 1;
        }
        // ⛔ [`Wrapping`] BECAUSE `:4715` IS UNDEFINED THERE, NOT BECAUSE IT WRAPS: `chunkOffset *
        // numChunksVisited + partialPadSize` is `int` arithmetic, and UBSan reports both operators
        // on it. Clang wraps at -O0, -O2 and under UBSan alike, so this is the only behaviour the
        // authority has been observed to have — and the only form here that does not panic.
        Some(PadSize(
            (Wrapping(chunk_offset.0) * Wrapping(num_chunks_visited as i32)
                + Wrapping(partial_pad_size))
            .0,
        ))
    }

    /// `getTransferPadSizeFrontOrBack` (`dsc/dsc2.cpp:4718-4732`) — one chunk's pad, clamped into
    /// `[0, chunkSizePadded]`.
    ///
    /// ⛔ NOT [`Ord::clamp`]: it panics when `min > max`, and a negative [`ChunkSizePadded`] reaches
    /// this. `std::min(std::max(v, 0), chunkSizePadded)` (`:4729-4731`) answers the CAP there —
    /// measured, a cap of -5 returns -5, outside the range the comment above it claims.
    pub fn transfer_pad_size(
        &self,
        end: PadEnd,
        dim: PrimaryDimTypes,
        wk_slice: WkSliceIdx,
        chunk: ChunkIdx,
        chunk_size_padded: ChunkSizePadded,
    ) -> Option<PadSize> {
        let fold = match end {
            PadEnd::Front => self.front.get(&dim)?,
            PadEnd::Back => self.back.get(&dim)?,
        };
        Some(PadSize(
            fold.data(wk_slice, chunk)?
                .0
                .max(0)
                .min(chunk_size_padded.0),
        ))
    }
}

#[cfg(test)]
mod equivalence {
    use super::*;

    /// The shape every case below builds — the one production builder's, with its own signs:
    /// `sizes = {numWkSlice, numChunks}`, `alphasPadFront = {-coreOffset, -chunkOffset}`,
    /// `betasPadBack` subtracting both again
    /// (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:5388-5451`).
    const SIZES: [FoldDimSize; FoldDimPosition::COUNT] = [FoldDimSize(2), FoldDimSize(3)];
    const ALPHAS_FRONT: [PadSize; FoldDimPosition::COUNT] = [PadSize(-40), PadSize(-10)];
    const BETAS_FRONT: [PadSize; FoldDimPosition::COUNT] = [PadSize(25), PadSize(0)];
    const ALPHAS_BACK: [PadSize; FoldDimPosition::COUNT] = [PadSize(40), PadSize(10)];
    const BETAS_BACK: [PadSize; FoldDimPosition::COUNT] = [PadSize(-53), PadSize(0)];

    fn both_ends() -> TransferPadInfo {
        let mut info = TransferPadInfo::default();
        assert_eq!(
            info.build_pad_sizes(
                PadEnd::Front,
                PrimaryDimTypes::X,
                SIZES,
                ALPHAS_FRONT,
                BETAS_FRONT
            ),
            Some(())
        );
        assert_eq!(
            info.build_pad_sizes(
                PadEnd::Back,
                PrimaryDimTypes::X,
                SIZES,
                ALPHAS_BACK,
                BETAS_BACK
            ),
            Some(())
        );
        info
    }

    /// `empty.isEmpty = 1`, `empty.frontKeys = 0`, `empty.backKeys = 0`, `built.isEmpty = 0`,
    /// `built.frontKeys = 1`, `built.backKey0 = 4`, `raw.front.w0.{c0,c1,c2} = 25, 15, 5`,
    /// `raw.front.w1.* = 0`, `raw.back.w0.* = 0`, `raw.back.w1.{c0,c1,c2} = 0, 0, 7`,
    /// `neg.wkslice = 65`, `neg.chunk = 35`, `oob.wkslice = THROW`, `oob.chunk = THROW`,
    /// `unknown.dim = THROW`, `rebuild.same_dim = THROW`, `frontOnly.backKeys = 0`,
    /// `frontOnly.backQuery = THROW`.
    ///
    /// ⛔ A NEGATIVE COORDINATE IS LEGAL AND COMPUTES while an out-of-range one throws: `isLegal`'s
    /// range test widens a `uint32_t` extent to signed (`util/foldManager/foldInfrastructure.h:1677`).
    /// ⛔ AND THE TWO ENDS ARE INDEPENDENT: `frontOnly` proves a back query throws on an object whose
    /// front is built, so they are two maps and not one keyed by end.
    #[test]
    fn e024_a_pad_fold_is_two_affine_levels_over_the_wk_slice_and_chunk_coordinates() {
        let empty = TransferPadInfo::default();
        assert!(empty.is_empty());
        assert_eq!(empty.pad_dims(PadEnd::Front).count(), 0);
        assert_eq!(empty.pad_dims(PadEnd::Back).count(), 0);

        let info = both_ends();
        assert!(!info.is_empty());
        assert_eq!(
            info.pad_dims(PadEnd::Front).collect::<Vec<_>>(),
            [PrimaryDimTypes::X]
        );
        assert_eq!(
            info.pad_dims(PadEnd::Back).collect::<Vec<_>>(),
            [PrimaryDimTypes::X]
        );
        assert_eq!(PrimaryDimTypes::X as usize, 4, "`built.frontKey0 = 4`");

        // `max(getDataForKey(..), 0)` alone: the cap is the identity at `INT_MAX`.
        let raw = |end, w: i64, c: i64| {
            info.transfer_pad_size(
                end,
                PrimaryDimTypes::X,
                WkSliceIdx(w),
                ChunkIdx(c),
                ChunkSizePadded(i32::MAX),
            )
        };
        for (chunk, front) in [(0, 25), (1, 15), (2, 5)] {
            assert_eq!(raw(PadEnd::Front, 0, chunk), Some(PadSize(front)));
            assert_eq!(raw(PadEnd::Front, 1, chunk), Some(PadSize(0)));
            assert_eq!(raw(PadEnd::Back, 0, chunk), Some(PadSize(0)));
        }
        assert_eq!(raw(PadEnd::Back, 1, 0), Some(PadSize(0)));
        assert_eq!(raw(PadEnd::Back, 1, 1), Some(PadSize(0)));
        assert_eq!(raw(PadEnd::Back, 1, 2), Some(PadSize(7)));

        assert_eq!(
            raw(PadEnd::Front, -1, 0),
            Some(PadSize(65)),
            "`neg.wkslice`"
        );
        assert_eq!(raw(PadEnd::Front, 0, -1), Some(PadSize(35)), "`neg.chunk`");
        assert_eq!(raw(PadEnd::Front, 2, 0), None, "`oob.wkslice`");
        assert_eq!(raw(PadEnd::Front, 0, 3), None, "`oob.chunk`");

        assert_eq!(
            info.transfer_pad_size(
                PadEnd::Front,
                PrimaryDimTypes::Y,
                WkSliceIdx(0),
                ChunkIdx(0),
                ChunkSizePadded(10)
            ),
            None,
            "`unknown.dim`"
        );

        let mut rebuilt = both_ends();
        assert_eq!(
            rebuilt.build_pad_sizes(
                PadEnd::Front,
                PrimaryDimTypes::X,
                SIZES,
                ALPHAS_FRONT,
                BETAS_FRONT
            ),
            None,
            "`rebuild.same_dim`"
        );

        let mut front_only = TransferPadInfo::default();
        front_only.build_pad_sizes(
            PadEnd::Front,
            PrimaryDimTypes::X,
            SIZES,
            ALPHAS_FRONT,
            BETAS_FRONT,
        );
        assert!(!front_only.is_empty());
        assert_eq!(front_only.pad_dims(PadEnd::Back).count(), 0);
        assert_eq!(
            front_only.transfer_pad_size(
                PadEnd::Back,
                PrimaryDimTypes::X,
                WkSliceIdx(0),
                ChunkIdx(0),
                ChunkSizePadded(10)
            ),
            None,
            "`frontOnly.backQuery`"
        );
    }

    /// `wk.front.w0 = 25`, `wk.front.w1 = 0`, `wk.back.w0 = 0`, `wk.back.w1 = 7`,
    /// `wk.numChunksZero = 0`, `wk.chunkSizeZero = 30`, `wk.negWkSlice = 30`,
    /// `wk.oobWkSlice = THROW`, `wk.beyond_extent_cap10 = 25`,
    /// `wk.beyond_extent_capzero = THROW`, `wk.back_beyond_extent = THROW`,
    /// `wk.capEqualsPad = 205`.
    ///
    /// ⛔ `chunkSizePadded` IS THE ONLY EARLY EXIT: at 0 no chunk is "partial", so the walk visits all
    /// three and answers `3 * chunkOffset` — and past the stored extent it then throws, immediately on
    /// the back end whose first coordinate is `numChunks - 1`.
    #[test]
    fn e024_the_wk_slice_walk_stops_at_the_first_chunk_that_is_not_fully_padded() {
        let info = both_ends();
        let wk = |end, w: i64, num_chunks: u32, cap: i32| {
            info.wk_slice_pad_size(
                end,
                PrimaryDimTypes::X,
                WkSliceIdx(w),
                NumChunks(num_chunks),
                ChunkOffset(10),
                ChunkSizePadded(cap),
            )
        };

        // Front, work slice 0: chunks 0 and 1 are fully padded (25, 15 >= 10), chunk 2 is partial at
        // 5, so `10 * 2 + 5`.
        assert_eq!(wk(PadEnd::Front, 0, 3, 10), Some(PadSize(25)));
        assert_eq!(wk(PadEnd::Front, 1, 3, 10), Some(PadSize(0)));
        assert_eq!(wk(PadEnd::Back, 0, 3, 10), Some(PadSize(0)));
        assert_eq!(wk(PadEnd::Back, 1, 3, 10), Some(PadSize(7)));

        assert_eq!(
            wk(PadEnd::Front, 0, 0, 10),
            Some(PadSize(0)),
            "`numChunksZero`"
        );
        assert_eq!(
            wk(PadEnd::Front, 0, 3, 0),
            Some(PadSize(30)),
            "`chunkSizeZero`"
        );
        assert_eq!(
            wk(PadEnd::Front, -1, 3, 10),
            Some(PadSize(30)),
            "`negWkSlice`"
        );
        assert_eq!(wk(PadEnd::Front, 2, 3, 10), None, "`oobWkSlice`");

        assert_eq!(
            wk(PadEnd::Front, 0, 5, 10),
            Some(PadSize(25)),
            "`beyond_extent_cap10`"
        );
        assert_eq!(wk(PadEnd::Front, 0, 5, 0), None, "`beyond_extent_capzero`");
        assert_eq!(wk(PadEnd::Back, 0, 5, 0), None, "`back_beyond_extent`");

        // `wk.capEqualsPad = 205`: chunk 1's pad is EXACTLY `chunkSizePadded`, and
        // `currPadSize < chunkSizePadded` (`:4707`) calls that FULLY padded — so the walk carries on
        // to chunk 2 and answers `100 * 2 + 5`, not `100 * 1 + 15`.
        assert_eq!(
            info.wk_slice_pad_size(
                PadEnd::Front,
                PrimaryDimTypes::X,
                WkSliceIdx(0),
                NumChunks(3),
                ChunkOffset(100),
                ChunkSizePadded(15)
            ),
            Some(PadSize(205))
        );
    }

    /// `clamp.front.w0.{c0,c1,c2} = 10, 10, 5`, `clamp.front.w1.* = 0`, `clamp.back.w0.* = 0`,
    /// `clamp.back.w1.{c0,c1,c2} = 0, 0, 7`, `clamp.negCap = -5`, `clamp.oobWkSlice = THROW`.
    ///
    /// ⛔ `clamp.negCap = -5` IS WHY THIS IS NOT [`Ord::clamp`]: `min(max(v, 0), cap)` answers the CAP
    /// when the cap is negative, which is outside the `[0, chunk_param_with_zero_pad]` range the
    /// authority's own comment claims (`dsc/dsc2.cpp:4723-4728`), and [`Ord::clamp`] would panic.
    #[test]
    fn e024_a_chunks_pad_size_is_clamped_into_the_padded_chunk_and_the_cap_wins() {
        let info = both_ends();
        let clamp = |end, w: i64, c: i64, cap: i32| {
            info.transfer_pad_size(
                end,
                PrimaryDimTypes::X,
                WkSliceIdx(w),
                ChunkIdx(c),
                ChunkSizePadded(cap),
            )
        };
        for (chunk, front) in [(0, 10), (1, 10), (2, 5)] {
            assert_eq!(clamp(PadEnd::Front, 0, chunk, 10), Some(PadSize(front)));
            assert_eq!(clamp(PadEnd::Front, 1, chunk, 10), Some(PadSize(0)));
            assert_eq!(clamp(PadEnd::Back, 0, chunk, 10), Some(PadSize(0)));
        }
        assert_eq!(clamp(PadEnd::Back, 1, 0, 10), Some(PadSize(0)));
        assert_eq!(clamp(PadEnd::Back, 1, 1, 10), Some(PadSize(0)));
        assert_eq!(clamp(PadEnd::Back, 1, 2, 10), Some(PadSize(7)));

        assert_eq!(
            clamp(PadEnd::Front, 0, 0, -5),
            Some(PadSize(-5)),
            "`negCap`"
        );
        assert_eq!(clamp(PadEnd::Front, 2, 0, 10), None, "`oobWkSlice`");
    }

    /// `C.stored_wkslice_extent = 4294967295`, `C.stored_chunk_extent = 3`,
    /// `C.wk_1000000 = -39999975`, `C.wk_int32max = 65`, `C.wk_4294967295 = THROW`.
    ///
    /// ⛔ THE RANGE GUARD GOES VACUOUS AT A FOUR-BILLION EXTENT, AND THE AUTHORITY'S OWN BUILDER
    /// REACHES IT: an `int` size crosses `setSize(uint32_t)`
    /// (`util/foldManager/foldInfrastructure.h:129`) at `dsc/dsc2.cpp:4669`, so `sizes = {-1, 3}`
    /// stores 4294967295 and every `int` work slice is in range from then on. [`FoldDimSize`] is a
    /// `u32`, so the `-1` cannot be spelled here — but the extent it produces can, and that is what
    /// this pins, because the guard is the only thing between a query and the walk.
    /// ⛔ AND `i32::MAX` IS WHY THE AFFINE WALK IS `Wrapping`: clang wraps `-40 * 2147483647 + 25`
    /// to 65, where a checked multiply would panic in a debug build and diverge.
    #[test]
    fn e024_a_four_billion_extent_makes_the_range_guard_admit_every_int_work_slice() {
        let mut info = TransferPadInfo::default();
        assert_eq!(
            info.build_pad_sizes(
                PadEnd::Front,
                PrimaryDimTypes::X,
                [FoldDimSize(u32::MAX), FoldDimSize(3)],
                ALPHAS_FRONT,
                BETAS_FRONT
            ),
            Some(())
        );
        let fold = &info.front[&PrimaryDimTypes::X];
        assert_eq!(
            fold.manager().fold_dim_props()[0].size(),
            FoldDimSize(4_294_967_295),
            "`C.stored_wkslice_extent`"
        );
        assert_eq!(
            fold.manager().fold_dim_props()[1].size(),
            FoldDimSize(3),
            "`C.stored_chunk_extent`"
        );

        // The unclamped reader, because `transfer_pad_size` would `max(.., 0)` the negative away.
        assert_eq!(
            fold.data(WkSliceIdx(1_000_000), ChunkIdx(0)),
            Some(PadSize(-39_999_975)),
            "`C.wk_1000000`"
        );
        assert_eq!(
            fold.data(WkSliceIdx(i64::from(i32::MAX)), ChunkIdx(0)),
            Some(PadSize(65)),
            "`C.wk_int32max`"
        );
        // The guard is `getSize() <= idx`, so the extent itself is the one value still out of range.
        assert_eq!(
            fold.data(WkSliceIdx(4_294_967_295), ChunkIdx(0)),
            None,
            "`C.wk_4294967295`"
        );
    }

    /// `C.zero_extent_stored = 0`, `C.zero_extent_wk0 = THROW`, `C.zero_extent_wk_minus_1 = 65`.
    ///
    /// ⛔ AN EXTENT OF 0 REFUSES COORDINATE 0 AND STILL ANSWERS FOR -1. That is the signed range
    /// test at its limit (`util/foldManager/foldInfrastructure.h:1677`) and NOT `hasZeroFoldDim`'s
    /// "always legal" shortcut (`:1667`, `:2603`): that one reads the NUMBER of fold dims, never an
    /// extent, so it cannot fire on a fold whose signature is two positions.
    #[test]
    fn e024_a_zero_extent_refuses_coordinate_zero_and_still_answers_for_minus_one() {
        let mut info = TransferPadInfo::default();
        assert_eq!(
            info.build_pad_sizes(
                PadEnd::Front,
                PrimaryDimTypes::X,
                [FoldDimSize(0), FoldDimSize(3)],
                ALPHAS_FRONT,
                BETAS_FRONT
            ),
            Some(())
        );
        let fold = &info.front[&PrimaryDimTypes::X];
        assert_eq!(
            fold.manager().fold_dim_props()[0].size(),
            FoldDimSize(0),
            "`C.zero_extent_stored`"
        );
        assert_eq!(
            fold.data(WkSliceIdx(0), ChunkIdx(0)),
            None,
            "`C.zero_extent_wk0`"
        );
        assert_eq!(
            fold.data(WkSliceIdx(-1), ChunkIdx(0)),
            Some(PadSize(65)),
            "`C.zero_extent_wk_minus_1`"
        );
    }

    /// `F.frontDims = 2 5 9`, `F.rebuild = THROW`, `F.frontDims_after = 2 5 9`.
    ///
    /// ⛔ EVERY OTHER CASE HERE BUILDS ONE DIM, SO NONE OF THEM CAN SEE ORDER — a [`Vec`] would pass
    /// them all. `getAllKeys` walks a `std::map` into a `std::set`
    /// (`util/foldManager/mapWithFMHelper.h:51-55`, `dsc/dsc2.h:783-788`) exactly because the one
    /// caller `std::set_union`s the two ends (`dsc/dsc2.cpp:4814-4820`), which is an ordered merge.
    /// Inserted 9, 2, 5 the authority answers 2 5 9.
    /// ⛔ AND THE REFUSED REBUILD LEAVES THE LEVELS ALONE, not just the key set: the authority throws
    /// at `DT_CHECK_MSG(!foldProps.count(dim))` (`dsc/dsc2.cpp:4662`) before it resizes anything.
    #[test]
    fn e024_pad_dims_is_ascending_and_a_refused_rebuild_leaves_the_fold_alone() {
        let mut info = TransferPadInfo::default();
        for dim in [PrimaryDimTypes::Ki, PrimaryDimTypes::Ij, PrimaryDimTypes::Y] {
            assert_eq!(
                info.build_pad_sizes(PadEnd::Front, dim, SIZES, ALPHAS_FRONT, BETAS_FRONT),
                Some(())
            );
        }
        let ascending = [PrimaryDimTypes::Ij, PrimaryDimTypes::Y, PrimaryDimTypes::Ki];
        assert_eq!(ascending.map(|dim| dim as usize), [2, 5, 9]);
        assert_eq!(
            info.pad_dims(PadEnd::Front).collect::<Vec<_>>(),
            ascending,
            "`F.frontDims`"
        );

        assert_eq!(
            info.build_pad_sizes(
                PadEnd::Front,
                PrimaryDimTypes::Y,
                SIZES,
                ALPHAS_BACK,
                BETAS_BACK
            ),
            None,
            "`F.rebuild`"
        );
        assert_eq!(
            info.pad_dims(PadEnd::Front).collect::<Vec<_>>(),
            ascending,
            "`F.frontDims_after`"
        );
        assert_eq!(
            info.front[&PrimaryDimTypes::Y].data(WkSliceIdx(0), ChunkIdx(0)),
            Some(PadSize(25)),
            "the refusal did not write `BETAS_BACK`'s -53 over the level"
        );
        // `F.after.tp_y_0_0 = 25`, `F.after.tp_y_0_2 = 5`, `F.after.wk_y_3chunk = 25` — the same
        // refusal through the PUBLIC readers, which is where a half-applied rebuild would surface.
        assert_eq!(
            info.transfer_pad_size(
                PadEnd::Front,
                PrimaryDimTypes::Y,
                WkSliceIdx(0),
                ChunkIdx(0),
                ChunkSizePadded(i32::MAX)
            ),
            Some(PadSize(25)),
            "`F.after.tp_y_0_0`"
        );
        assert_eq!(
            info.transfer_pad_size(
                PadEnd::Front,
                PrimaryDimTypes::Y,
                WkSliceIdx(0),
                ChunkIdx(2),
                ChunkSizePadded(i32::MAX)
            ),
            Some(PadSize(5)),
            "`F.after.tp_y_0_2`"
        );
        assert_eq!(
            info.wk_slice_pad_size(
                PadEnd::Front,
                PrimaryDimTypes::Y,
                WkSliceIdx(0),
                NumChunks(3),
                ChunkOffset(10),
                ChunkSizePadded(10)
            ),
            Some(PadSize(25)),
            "`F.after.wk_y_3chunk`"
        );
    }

    /// `N.front_neg1 = 0`, `N.back_neg1 = 0`, `N.front_intmin = 0`, `N.front_zero = 0`.
    ///
    /// ⛔ THE EVIDENCE THAT [`NumChunks`] LOSES NOTHING BY BEING UNSIGNED. The authority's
    /// `numChunks` is an `int` (`dsc/dsc2.cpp:4685`), and measured, -1 and `INT_MIN` each answer 0 at
    /// both ends — identically to 0 — because `numChunksVisited < numChunks` fails before the first
    /// `getDataForKey` (`:4699`), leaving `chunkOffset * 0 + 0`. So every negative the authority
    /// accepts is spelled `NumChunks(0)` here, and the `chunkOffset` is never applied.
    #[test]
    fn e024_a_chunk_count_of_zero_visits_nothing_and_speaks_for_the_authoritys_negatives() {
        let info = both_ends();
        for end in [PadEnd::Front, PadEnd::Back] {
            assert_eq!(
                info.wk_slice_pad_size(
                    end,
                    PrimaryDimTypes::X,
                    WkSliceIdx(0),
                    NumChunks(0),
                    ChunkOffset(10),
                    ChunkSizePadded(10)
                ),
                Some(PadSize(0)),
                "`N.{end:?}_neg1`"
            );
        }
    }

    /// `O.mul_ovf = -1894967296`, `O.add_ovf_back = -2147483640`.
    ///
    /// ⛔ THE AUTHORITY IS UNDEFINED AT `:4715`, NOT WRAPPING, AND THESE ARE THE VALUES IT PRODUCES
    /// ANYWAY. `chunkOffset * numChunksVisited + partialPadSize` is `int` arithmetic on `int`s, and
    /// UndefinedBehaviorSanitizer reports it on the second fixture below ("signed integer overflow:
    /// 10 + 2147483646 cannot be represented in type 'int'"); clang nonetheless wraps identically at
    /// -O0, -O2 and under UBSan. [`Wrapping`] reproduces that and is the only form that does not
    /// panic, which this crate forbids — so what this test pins is a UB-dependent agreement, and a
    /// future reader must not have to rediscover that the authority has no defined answer here.
    #[test]
    fn e024_the_wk_slice_tail_wraps_where_the_authoritys_int_arithmetic_is_undefined() {
        // Three fully padded chunks against a cap of 0, so the walk visits all of them and the
        // MULTIPLY overflows: 800000000 * 3 = 2400000000.
        assert_eq!(
            both_ends().wk_slice_pad_size(
                PadEnd::Front,
                PrimaryDimTypes::X,
                WkSliceIdx(1),
                NumChunks(3),
                ChunkOffset(800_000_000),
                ChunkSizePadded(0)
            ),
            Some(PadSize(-1_894_967_296)),
            "`O.mul_ovf`"
        );

        // And the ADD, which needs a partial pad near `i32::MAX`: at work slice `i32::MIN` the outer
        // level folds to 2147483646 on chunk 0, one below the cap, so the walk breaks having visited
        // the back end's chunk 1 — and 10 * 1 + 2147483646 does not fit.
        let mut info = TransferPadInfo::default();
        assert_eq!(
            info.build_pad_sizes(
                PadEnd::Back,
                PrimaryDimTypes::X,
                SIZES,
                [PadSize(1), PadSize(1)],
                [PadSize(-1), PadSize(-1)]
            ),
            Some(())
        );
        assert_eq!(
            info.wk_slice_pad_size(
                PadEnd::Back,
                PrimaryDimTypes::X,
                WkSliceIdx(i64::from(i32::MIN)),
                NumChunks(2),
                ChunkOffset(10),
                ChunkSizePadded(i32::MAX)
            ),
            Some(PadSize(-2_147_483_640)),
            "`O.add_ovf_back`"
        );
    }

    /// `C.api_tp_1000000 = 0`, `C.api_wk_1000000_cap10 = 0`, `C.api_wk_int32max_cap10 = 30`.
    ///
    /// ⛔ THE VACUOUS RANGE GUARD REACHES THE PUBLIC SURFACE, AND THE CLAMP DISGUISES IT. The
    /// four-billion extent admits work slice 1000000, which folds to -39999975; both readers
    /// `max(.., 0)` that to 0 (`dsc/dsc2.cpp:4707`, `:4729-4731`), so `wk_slice_pad_size` breaks on
    /// chunk 0 and answers 0 — indistinguishable from a slice that is legitimately unpadded, where an
    /// in-range extent would have thrown instead. `i32::MAX` is the coordinate that does NOT
    /// disguise itself: it folds to 65, stays at or above a cap of 10 across all three chunks, and
    /// answers `10 * 3 = 30`.
    #[test]
    fn e024_a_four_billion_extent_answers_a_clamped_zero_through_the_public_readers() {
        let mut info = TransferPadInfo::default();
        assert_eq!(
            info.build_pad_sizes(
                PadEnd::Front,
                PrimaryDimTypes::X,
                [FoldDimSize(u32::MAX), FoldDimSize(3)],
                ALPHAS_FRONT,
                BETAS_FRONT
            ),
            Some(())
        );
        assert_eq!(
            info.transfer_pad_size(
                PadEnd::Front,
                PrimaryDimTypes::X,
                WkSliceIdx(1_000_000),
                ChunkIdx(0),
                ChunkSizePadded(i32::MAX)
            ),
            Some(PadSize(0)),
            "`C.api_tp_1000000`"
        );
        assert_eq!(
            info.wk_slice_pad_size(
                PadEnd::Front,
                PrimaryDimTypes::X,
                WkSliceIdx(1_000_000),
                NumChunks(3),
                ChunkOffset(10),
                ChunkSizePadded(10)
            ),
            Some(PadSize(0)),
            "`C.api_wk_1000000_cap10`"
        );
        assert_eq!(
            info.wk_slice_pad_size(
                PadEnd::Front,
                PrimaryDimTypes::X,
                WkSliceIdx(i64::from(i32::MAX)),
                NumChunks(3),
                ChunkOffset(10),
                ChunkSizePadded(10)
            ),
            Some(PadSize(30)),
            "`C.api_wk_int32max_cap10`"
        );
    }

    /// The one probe [`ComputeNode::operand_formats`] is measured at below. `SENINT8` is chosen
    /// because it is in `MACC`'s integer list, so the `MACC` arm fires rather than falling through.
    const PROBE: DataFormats = DataFormats::Senint8;

    /// The stand-in for `dsc.labeledDs_.at(outputsLdsAndLoopOffsets_.at(0).myLdsIdx_).dataFormat_`
    /// (`dsc/dsc2.cpp:2352-2353`). Deliberately a format NO branch of the method can produce, so a
    /// slot holding it can only have come from the argument.
    const PACKMERGE_ARG: DataFormats = DataFormats::Sen153Fp9;

    fn compute(r#type: ComputeOpType, data_format: DataFormats, inputs: usize) -> Vec<DataFormats> {
        ComputeNode {
            r#type,
            data_format,
            inputs: vec![SenComponent::Ptnorth; inputs],
            ..ComputeNode::default()
        }
        .operand_formats(PACKMERGE_ARG)
    }

    /// 63 of the 71 ops carry `dataFormat_` into every slot, `PACKMERGE` answers from the argument
    /// alone, and exactly SEVEN move a slot — of which `FMA16` moves its ACCUMULATOR ONLY, since the
    /// input chain (`dsc/dsc2.cpp:2360-2371`) names FMA8, FMA4, IMA8, IMA4 and FMA32 and not FMA16
    /// while the accumulator chain (`:2385`) does name it.
    ///
    /// ⭐ AND `MACC` IS THE ONE OP WHOSE ACCUMULATOR COMES FROM `dataFormat_` (`:2377-2391`), for
    /// exactly SIX of the 22 formats — the DDL's two accumulators, four floats onto `%ptsum_fp`'s
    /// `SEN169_FP16` and two ints onto `%ptsum_int`'s `SENINT24` (`ddc/ddl_templates/bmm.ddl:47-48`).
    #[test]
    fn e035_seven_ops_and_six_macc_formats_move_an_operand_format() {
        // Measured: `op\t<type>\t8\t3\t...` for every op, at three inputs.
        let moved = [
            (
                ComputeOpType::Macc,
                [PROBE, PROBE, DataFormats::Senint24, DataFormats::Senint24],
            ),
            (ComputeOpType::Fma32, [DataFormats::IeeeFp32; 4]),
            (
                ComputeOpType::Fma16,
                [
                    PROBE,
                    PROBE,
                    DataFormats::Sen169Fp16,
                    DataFormats::Sen169Fp16,
                ],
            ),
            (
                ComputeOpType::Fma8,
                [
                    DataFormats::Sen143Fp8,
                    DataFormats::Sen143Fp8,
                    DataFormats::Sen169Fp16,
                    DataFormats::Sen169Fp16,
                ],
            ),
            (
                ComputeOpType::Fma4,
                [
                    DataFormats::Sen121Fp4,
                    DataFormats::Sen121Fp4,
                    DataFormats::Sen169Fp16,
                    DataFormats::Sen169Fp16,
                ],
            ),
            (
                ComputeOpType::Ima8,
                [PROBE, PROBE, DataFormats::Senint24, DataFormats::Senint24],
            ),
            (
                ComputeOpType::Ima4,
                [
                    DataFormats::Senint4,
                    DataFormats::Senint4,
                    DataFormats::Senint24,
                    DataFormats::Senint24,
                ],
            ),
        ];
        assert_eq!(moved.len(), 7);
        assert_eq!(ComputeOpType::ALL.len(), 71);

        let mut carried = 0;
        for op in ComputeOpType::ALL {
            let got = compute(op, PROBE, 3);
            if op == ComputeOpType::Packmerge {
                assert_eq!(got, vec![PACKMERGE_ARG; 3], "{op:?}");
                continue;
            }
            match moved.iter().find(|(moved_op, _)| *moved_op == op) {
                Some((_, want)) => assert_eq!(got, want.to_vec(), "{op:?}"),
                None => {
                    assert_eq!(got, vec![PROBE; 4], "{op:?}");
                    carried += 1;
                }
            }
        }
        assert_eq!(carried, 71 - moved.len() - 1);

        // Measured: `macc\t<format>\t...` for every format, at three inputs.
        let moved = [
            (DataFormats::Sen143Fp8, DataFormats::Sen169Fp16),
            (DataFormats::Sen152Fp8, DataFormats::Sen169Fp16),
            (DataFormats::Bfloat16, DataFormats::Sen169Fp16),
            (DataFormats::Sen121Fp4, DataFormats::Sen169Fp16),
            (DataFormats::Senint4, DataFormats::Senint24),
            (DataFormats::Senint8, DataFormats::Senint24),
        ];
        assert_eq!(moved.len(), 6);
        assert_eq!(DataFormats::ALL.len(), 22);

        for format in DataFormats::ALL {
            let accumulator = moved
                .iter()
                .find(|(from, _)| *from == format)
                .map_or(format, |(_, to)| *to);
            assert_eq!(
                compute(ComputeOpType::Macc, format, 3),
                vec![format, format, accumulator, accumulator],
                "{format:?}"
            );
        }
    }

    /// ⛔ THE FORMAT COUNT IS NOT THE OPERAND COUNT, AND FOUR INPUTS YIELD FEWER FORMATS THAN THREE:
    /// `min(2, inputs_.size())` inputs, a third slot only at exactly three, then the output
    /// (`dsc/dsc2.cpp:2372-2373`, `:2392-2393`). `PACKMERGE` answers three at every arity (`:2354`).
    #[test]
    fn e035_the_format_count_is_not_the_operand_count() {
        // FMA8, so the input format (`SEN143_FP8`) is distinguishable from the accumulator's
        // (`SEN169_FP16`) and each slot's origin is visible.
        let input = DataFormats::Sen143Fp8;
        let accumulator = DataFormats::Sen169Fp16;
        let measured = [
            vec![accumulator],
            vec![input, accumulator],
            vec![input, input, accumulator],
            vec![input, input, accumulator, accumulator],
            vec![input, input, accumulator],
        ];
        for (inputs, want) in measured.iter().enumerate() {
            assert_eq!(
                compute(ComputeOpType::Fma8, PROBE, inputs),
                *want,
                "{inputs}"
            );
        }
        assert!(
            measured[4].len() < measured[3].len(),
            "four inputs, three formats"
        );

        for inputs in 0..=4 {
            assert_eq!(
                compute(ComputeOpType::Packmerge, PROBE, inputs),
                vec![PACKMERGE_ARG; 3],
                "PACKMERGE at {inputs}"
            );
        }
    }

    /// `getPageSize`'s loop transcribed as the authority writes it — two independent vectors walked
    /// by one index, with the `DT_CHECK` on their lengths it needs to be safe
    /// (`dsc/dsc2.cpp:4496-4511`).
    fn page_size_over_two_vectors(
        layout_dim_order: &[PrimaryDimTypes],
        max_dim_sizes: &[i32],
    ) -> BTreeMap<PrimaryDimTypes, i32> {
        assert_eq!(layout_dim_order.len(), max_dim_sizes.len(), "`:4496`");
        let mut page_size = BTreeMap::new();
        let mut unbounded_dims = BTreeSet::new();
        for i in 0..layout_dim_order.len() {
            let dim = layout_dim_order[i];
            let max_size = max_dim_sizes[i];
            if max_size < 0 {
                unbounded_dims.insert(dim);
                page_size.remove(&dim);
            } else if !unbounded_dims.contains(&dim) {
                *page_size.entry(dim).or_insert(1) *= max_size;
            }
        }
        page_size
    }

    /// [`AllocateNode::page_size`] over the merged pairs against that transcription, on the cases the
    /// merge has to survive: a dim repeated with two filled entries, an absent entry with filled
    /// entries of the same dim on BOTH sides of it, a zero, and the empty layout. ⭐ THE CURRENCY MAP
    /// IS THE FIXTURE'S OWN CONVERSION — a negative `int` becomes [`None`] through
    /// [`u32::try_from`], which is the boundary `maxSize < 0` draws (`dsc/dsc2.cpp:4501`).
    #[test]
    fn the_merged_layout_answers_the_page_size_the_authoritys_two_vectors_do() {
        const CASES: [(&[PrimaryDimTypes], &[i32]); 4] = [
            (
                &[
                    PrimaryDimTypes::In,
                    PrimaryDimTypes::Out,
                    PrimaryDimTypes::In,
                ],
                &[4, 3, 5],
            ),
            (
                &[
                    PrimaryDimTypes::In,
                    PrimaryDimTypes::In,
                    PrimaryDimTypes::In,
                    PrimaryDimTypes::Out,
                ],
                &[4, -1, 7, 9],
            ),
            (
                &[
                    PrimaryDimTypes::Y,
                    PrimaryDimTypes::Out,
                    PrimaryDimTypes::In,
                ],
                &[-1, 0, 4],
            ),
            (&[], &[]),
        ];

        for (case, (layout_dim_order, max_dim_sizes)) in CASES.into_iter().enumerate() {
            let node = AllocateNode {
                indirect_alloc_type: IndirectAllocType::ValueTensor,
                layout_dim_order: layout_dim_order
                    .iter()
                    .copied()
                    .zip(max_dim_sizes.iter().copied())
                    .map(|(dim, size)| (dim, u32::try_from(size).ok().map(MaxDimSize)))
                    .collect(),
                ..AllocateNode::default()
            };
            let ported = node
                .page_size(&node)
                .into_iter()
                .map(|(dim, PageSize(size))| (dim, i32::try_from(size).expect("`int`")))
                .collect::<BTreeMap<_, _>>();
            assert_eq!(
                ported,
                page_size_over_two_vectors(layout_dim_order, max_dim_sizes),
                "case {case}"
            );
        }
    }

    /// `DscPcfgTranslator::convertCondValToInt` transcribed over the authority's TWO fields
    /// (`dsc/dsc2Pcfg.cpp:788-806`) — the one reader of a `LoopCond` that never follows `loopComp_`,
    /// which is why the value half can land while the loop link cannot. ⭐ ITS `default:` ARM IS
    /// `INT`, and it is the only arm that reads `condValInt_`.
    fn convert_cond_val_to_int_over_two_fields(
        val_type: CondValType,
        val_int: i32,
        loop_count: i32,
    ) -> i32 {
        match val_type {
            CondValType::First => loop_count - 1,
            CondValType::Last => 0,
            CondValType::Int => loop_count - 1 - val_int,
        }
    }

    /// The same switch driven by the fused [`CondVal`], which has no `condValInt_` to read on two of
    /// its three arms.
    fn convert_cond_val_to_int(val: CondVal, loop_count: i32) -> i32 {
        match val {
            CondVal::First => loop_count - 1,
            CondVal::Last => 0,
            CondVal::Iteration(idx) => loop_count - 1 - idx.0,
        }
    }

    /// ⭐ THE FUSION LOSES NOTHING HERE, and the second loop is the evidence: the pair form admits
    /// `(FIRST, n)` and `(LAST, n)` for every `n`, and IBM's switch answers those arms without ever
    /// reading `n` — so the states [`CondVal`] cannot spell are states that had no distinct answer.
    #[test]
    fn the_fused_condition_value_converts_exactly_as_the_authoritys_two_fields_do() {
        for loop_count in [1i32, 2, 8, 64] {
            for val in [
                CondVal::First,
                CondVal::Last,
                CondVal::Iteration(IterationIdx(0)),
                CondVal::Iteration(IterationIdx(1)),
                CondVal::Iteration(IterationIdx(loop_count - 1)),
            ] {
                assert_eq!(
                    convert_cond_val_to_int(val, loop_count),
                    convert_cond_val_to_int_over_two_fields(
                        val.val_type(),
                        val.val_int(),
                        loop_count
                    ),
                    "{val:?} over {loop_count}"
                );
            }

            for val_type in [CondValType::First, CondValType::Last] {
                for val_int in [CondVal::ABSENT_VAL_INT, 0, 3] {
                    assert_eq!(
                        convert_cond_val_to_int_over_two_fields(val_type, val_int, loop_count),
                        convert_cond_val_to_int(CondVal::from_wire(val_type, val_int), loop_count),
                        "{val_type:?} {val_int} over {loop_count}"
                    );
                }
            }
        }
    }

    /// `SNControlFlowLowering.cpp:99-109` — bridge 1's own resolution of the two bound-relative
    /// forms on an `affine.for` with constant bounds, as an iteration of the loop it guards.
    fn resolve_cond_val_on_bridge_one(val: CondVal, lower: i32, upper: i32) -> i32 {
        match val {
            CondVal::First => lower,
            CondVal::Last => upper - 1,
            CondVal::Iteration(idx) => idx.0,
        }
    }

    /// ⛔ THE TWO READERS OF A `FIRST`/`LAST` ANSWER OPPOSITE ENDS OF THE SAME LOOP, so no fused
    /// resolution is possible and nothing in this module offers one: the PCFG counts DOWN from the
    /// last iteration, which makes its `FIRST` bridge 1's `LAST` and its `LAST` bridge 1's `FIRST`
    /// (`dsc/dsc2Pcfg.cpp:792-799` against `SNControlFlowLowering.cpp:99-109`). ⭐ THE SWAP IS THE
    /// CONVENTION, NOT A DEFECT — [`IterationIdx`] carries the same inversion on the `INT` arm — and a
    /// port that resolved either form to a number would have to pick a reader and be wrong on the
    /// other.
    #[test]
    fn the_two_readers_resolve_first_and_last_to_opposite_ends_of_the_loop() {
        for loop_count in [2i32, 8, 64] {
            // A DSC loop lowers to a zero-based `affine.for`, so `loopCount` IS the upper bound.
            let (lower, upper) = (0, loop_count);
            assert_eq!(
                convert_cond_val_to_int(CondVal::First, loop_count),
                resolve_cond_val_on_bridge_one(CondVal::Last, lower, upper),
                "the PCFG's FIRST is bridge 1's LAST over {loop_count}"
            );
            assert_eq!(
                convert_cond_val_to_int(CondVal::Last, loop_count),
                resolve_cond_val_on_bridge_one(CondVal::First, lower, upper),
                "the PCFG's LAST is bridge 1's FIRST over {loop_count}"
            );
            // ⛔ AND THE SAME FORM DISAGREES ACROSS THE TWO, one assertion per form so that a single
            // reader answering both ends cannot pass.
            assert_ne!(
                convert_cond_val_to_int(CondVal::First, loop_count),
                resolve_cond_val_on_bridge_one(CondVal::First, lower, upper),
                "FIRST over {loop_count}"
            );
            assert_ne!(
                convert_cond_val_to_int(CondVal::Last, loop_count),
                resolve_cond_val_on_bridge_one(CondVal::Last, lower, upper),
                "LAST over {loop_count}"
            );
        }
    }

    /// A DDL conditional expression: the four ops `processCondition` accepts, an operand at a time
    /// (`ddc/ddl/ddl_conversion.cpp:317`, `:321`, `:345`). Its `And`/`Or` are binary because the
    /// authority folds an n-ary op operand by operand into the same two guards.
    enum Cond {
        Term(LoopCond),
        Not(Box<Cond>),
        And(Box<Cond>, Box<Cond>),
        Or(Box<Cond>, Box<Cond>),
    }

    fn term(dim: PrimaryDimTypes, cond_op: LoopCondOp, cond_val: CondVal) -> Cond {
        Cond::Term(LoopCond {
            dim,
            cond_op,
            cond_val,
        })
    }

    fn not(inner: Cond) -> Cond {
        Cond::Not(Box::new(inner))
    }

    fn and(lhs: Cond, rhs: Cond) -> Cond {
        Cond::And(Box::new(lhs), Box::new(rhs))
    }

    fn or(lhs: Cond, rhs: Cond) -> Cond {
        Cond::Or(Box::new(lhs), Box::new(rhs))
    }

    /// The authority's composite as it really is: all three grammar levels in ONE flat pair, a
    /// `vector<vector<LoopCond>>` and a `bool` (`dsc/dsc2.h:675-683`).
    type FlatComposite = (Vec<Vec<LoopCond>>, bool);

    /// `ConditionAndOp` transcribed: the four-part shape check (`ddc/ddl/ddl_conversion.cpp:380-383`)
    /// and then the term concatenation it guards (`:390-392`). [`None`] is its
    /// `DT_ERROR("Illegal ddl")` (`:389`).
    fn flat_and(lhs: FlatComposite, rhs: FlatComposite) -> Option<FlatComposite> {
        if lhs.0.len() != 1 || lhs.1 || rhs.0.len() != 1 || rhs.1 {
            return None;
        }
        let mut clauses = lhs.0;
        for clause in rhs.0 {
            if let Some(dest) = clauses.first_mut() {
                dest.extend(clause);
            }
        }
        Some((clauses, false))
    }

    /// `ConditionOrOp` transcribed: its negation check (`:393-399`) and clause concatenation
    /// (`:401-403`).
    fn flat_or(lhs: FlatComposite, rhs: FlatComposite) -> Option<FlatComposite> {
        if lhs.1 || rhs.1 {
            return None;
        }
        let mut clauses = lhs.0;
        clauses.extend(rhs.0);
        Some((clauses, false))
    }

    /// The whole expression over the flat pair. ⭐ THE `condNot` ARM'S OWN `!empty()` GUARD
    /// (`:325-326`) IS UNREACHABLE HERE: no expression below resolves to an empty composite, which is
    /// the state that guard sends to `coreClCond_` instead.
    fn flat_eval(cond: &Cond) -> Option<FlatComposite> {
        match cond {
            Cond::Term(loop_cond) => Some((vec![vec![*loop_cond]], false)),
            Cond::Not(inner) => {
                let (clauses, negated) = flat_eval(inner)?;
                Some((clauses, !negated))
            }
            Cond::And(lhs, rhs) => flat_and(flat_eval(lhs)?, flat_eval(rhs)?),
            Cond::Or(lhs, rhs) => flat_or(flat_eval(lhs)?, flat_eval(rhs)?),
        }
    }

    /// The same expression through the layered types. ⭐ EVERY [`None`] HERE IS A GRAMMAR LEVEL THAT
    /// IS NOT THERE, never a check these functions perform: [`LoopCondConjunction::and`] and
    /// [`LoopCondDisjunction::or`] are total, and all this recursion can fail at is asking a negated
    /// composite or a multi-clause disjunction to be a conjunction.
    fn layered_conjunction(cond: &Cond) -> Option<LoopCondConjunction> {
        match cond {
            Cond::Term(loop_cond) => Some(LoopCondConjunction::new(*loop_cond)),
            Cond::And(lhs, rhs) => Some(layered_conjunction(lhs)?.and(layered_conjunction(rhs)?)),
            Cond::Not(_) | Cond::Or(..) => layered_disjunction(cond)?.into_conjunction(),
        }
    }

    fn layered_disjunction(cond: &Cond) -> Option<LoopCondDisjunction> {
        match cond {
            Cond::Or(lhs, rhs) => Some(layered_disjunction(lhs)?.or(layered_disjunction(rhs)?)),
            Cond::Not(_) => layered_eval(cond)?.without_negation(),
            term_or_and => Some(layered_conjunction(term_or_and)?.into()),
        }
    }

    fn layered_eval(cond: &Cond) -> Option<LoopCondComposite> {
        match cond {
            Cond::Not(inner) => Some(layered_eval(inner)?.negate()),
            other => Some(layered_disjunction(other)?.into()),
        }
    }

    /// [`LoopCond`] carries no `PartialEq` — its identity is the loop pointer it is still missing —
    /// so both forms are compared over the three value fields it does carry, nested exactly as the
    /// two levels are.
    type CondKey = (Vec<Vec<(PrimaryDimTypes, LoopCondOp, CondVal)>>, bool);

    fn flat_key(flat: FlatComposite) -> CondKey {
        (
            flat.0
                .into_iter()
                .map(|clause| {
                    clause
                        .into_iter()
                        .map(|t| (t.dim, t.cond_op, t.cond_val))
                        .collect()
                })
                .collect(),
            flat.1,
        )
    }

    fn layered_key(cond: &LoopCondComposite) -> CondKey {
        (
            cond.or_of_ands
                .clauses()
                .map(|clause| {
                    clause
                        .terms()
                        .map(|t| (t.dim, t.cond_op, t.cond_val))
                        .collect()
                })
                .collect(),
            cond.negated,
        )
    }

    /// ⭐ THE LEVELS COST NOTHING AND GUARD THE TWO SHAPES: over every expression shape the DDL can
    /// hand `processCondition`, the layered types compose exactly where the authority's flat pair
    /// does — same clauses, same terms, same order, same parity — and refuse exactly where it raises
    /// `DT_ERROR("Illegal ddl")`. ⛔ WHAT THE FLAT PAIR ADMITS AND THE TYPES DO NOT IS NOT IN THIS
    /// TABLE, BECAUSE NO EXPRESSION REACHES IT: an empty clause, an empty composite and a negated
    /// empty composite are all unspellable here, and all three are a dropped guard or an
    /// out-of-bounds read in bridge 1 (`SNControlFlowLowering.cpp:180`, `:203-272`).
    #[test]
    fn the_layered_condition_composes_and_refuses_exactly_where_the_authoritys_flat_pair_does() {
        let x = || term(PrimaryDimTypes::X, LoopCondOp::Eq, CondVal::First);
        let y = || term(PrimaryDimTypes::Y, LoopCondOp::Ne, CondVal::Last);
        let it3 = CondVal::Iteration(IterationIdx(3));
        let z = || term(PrimaryDimTypes::Mb, LoopCondOp::Lt, it3);

        let cases = [
            ("term", x()),
            ("not", not(x())),
            ("not not", not(not(x()))),
            ("and", and(x(), y())),
            ("and and", and(and(x(), y()), z())),
            ("or", or(x(), y())),
            ("or or", or(or(x(), y()), z())),
            ("or of and", or(and(x(), y()), z())),
            ("and of or", and(or(x(), y()), z())),
            ("and of not", and(not(x()), y())),
            ("or of not", or(not(x()), y())),
            ("and of not not", and(x(), not(not(y())))),
            ("or of not not", or(not(not(x())), y())),
            ("not of and", not(and(x(), y()))),
            ("not of or", not(or(x(), y()))),
            ("not of or of and", not(or(and(x(), y()), z()))),
        ];

        for (case, cond) in &cases {
            assert_eq!(
                flat_eval(cond).map(flat_key),
                layered_eval(cond).as_ref().map(layered_key),
                "case {case}"
            );
        }

        // ⛔ AND THE TABLE IS NOT ALL-ACCEPTING OR ALL-REFUSING, which is the only way the assertion
        // above means anything: the two shapes the authority's diagnostic names are refused, and
        // every other shape is composed.
        assert_eq!(
            cases
                .iter()
                .filter(|(_, cond)| flat_eval(cond).is_none())
                .count(),
            3,
            "and of or, and of not, or of not"
        );
    }

    /// Which arm of `adjustConditionForSplitLoop`'s dispatch a `(CondOp, CondValType)` pair takes
    /// (`dsc/dsc2.cpp:2082-2137`), in the authority's own order.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum SplitArm {
        /// `DT_ERROR("Unsupported loop condition value.")` (`:2084`).
        UnsupportedValue,
        /// The four always-true/always-false pairings (`:2087-2101`).
        AlwaysConstant,
        /// `twoLevelOrOfAnds_.at(iOr).push_back(newCond)` — a term ANDed into this clause (`:2113`).
        AndTerm,
        /// `twoLevelOrOfAnds_.push_back(newDisjunctiveClause)` — a clause ORed one level up
        /// (`:2126-2134`).
        OrClause,
        /// `DT_ERROR("Unsupported condition operation")` (`:2136`).
        UnsupportedOp,
    }

    fn split_arm(cond_op: LoopCondOp, cond_val: CondVal) -> SplitArm {
        let cond_val_type = cond_val.val_type();
        if cond_val_type != CondValType::First && cond_val_type != CondValType::Last {
            return SplitArm::UnsupportedValue;
        }
        if matches!(
            (cond_op, cond_val_type),
            (LoopCondOp::Gt, CondValType::Last)
                | (LoopCondOp::Lt, CondValType::First)
                | (LoopCondOp::Le, CondValType::Last)
                | (LoopCondOp::Ge, CondValType::First)
        ) {
            return SplitArm::AlwaysConstant;
        }
        if cond_op == LoopCondOp::Eq {
            return SplitArm::AndTerm;
        }
        if cond_op == LoopCondOp::Ne
            || matches!(
                (cond_op, cond_val_type),
                (LoopCondOp::Gt, CondValType::First) | (LoopCondOp::Lt, CondValType::Last)
            )
        {
            return SplitArm::OrClause;
        }
        SplitArm::UnsupportedOp
    }

    /// ⛔ THE `:2136` REFUSAL IS TWO PAIRS, AND BOTH ARE THE EQUALITY ARM'S: over the eighteen
    /// `(LoopCondOp, CondVal)` combinations a ported condition can spell, the split dispatch reaches
    /// `DT_ERROR("Unsupported condition operation")` (`dsc/dsc2.cpp:2136`) on exactly `(LE, FIRST)`
    /// and `(GE, LAST)` — the e044 anchor's FOURTH refusal, not a fifth — and written as the `EQ`
    /// each of them is, both land in the ANDed-term arm `:2113` and not the new-clause arm `:2134`.
    #[test]
    fn the_split_dispatchs_last_refusal_is_two_pairs_and_both_are_the_equality_arm() {
        let vals = [
            CondVal::Iteration(IterationIdx(0)),
            CondVal::First,
            CondVal::Last,
        ];
        let pairs = || {
            LoopCondOp::ALL
                .into_iter()
                .flat_map(|op| vals.into_iter().map(move |val| (op, val)))
        };

        assert_eq!(
            pairs()
                .filter(|&(op, val)| split_arm(op, val) == SplitArm::UnsupportedOp)
                .collect::<Vec<_>>(),
            vec![
                (LoopCondOp::Le, CondVal::First),
                (LoopCondOp::Ge, CondVal::Last),
            ]
        );

        // ⛔ AND THE ARM AN EQUALITY TAKES IS THE OTHER SHAPE: `index <= First` and `index >= Last`
        // AND a term into the clause they already sit in, where `:2134` would OR a whole new clause.
        for val in [CondVal::First, CondVal::Last] {
            assert_eq!(split_arm(LoopCondOp::Eq, val), SplitArm::AndTerm, "{val:?}");
        }

        // ⛔ AND THE SWEEP IS NOT ALL-REFUSING, which is the only way the two assertions above mean
        // anything: the eighteen pairs spread over all five arms.
        let tally = |arm: SplitArm| pairs().filter(|&(op, val)| split_arm(op, val) == arm).count();
        assert_eq!(
            [
                tally(SplitArm::UnsupportedValue),
                tally(SplitArm::AlwaysConstant),
                tally(SplitArm::AndTerm),
                tally(SplitArm::OrClause),
                tally(SplitArm::UnsupportedOp),
            ],
            [6, 4, 2, 4, 2]
        );
    }

    /// `constructConditionalOperation`'s path selector as the authority writes it
    /// (`SNControlFlowLowering.cpp:81`), with `has_else_branch` as its caller computes it —
    /// `children.size() == 2` (`:1052`).
    fn takes_multi_clause_path(cond: &LoopCondComposite, has_else_branch: bool) -> bool {
        cond.or_of_ands.clause_count().get() > 1 || cond.negated || has_else_branch
    }

    /// ⛔ THE CLAUSE COUNT IS NOT BRIDGE 1'S PATH SELECTOR: over the eight
    /// `(clauses, negated, else region)` states, `:81` and a `clause_count() > 1` dispatch disagree on
    /// THREE — every single-clause condition that is negated or carries an else region — and the
    /// single-clause body those three would be sent to creates its `scf.if` with no else region at
    /// all (`SNControlFlowLowering.cpp:258-259`, `:262-263`).
    #[test]
    fn bridge_ones_condition_path_is_not_selected_by_the_clause_count_alone() {
        let last = |dim| LoopCond {
            dim,
            cond_op: LoopCondOp::Eq,
            cond_val: CondVal::Last,
        };
        let one = LoopCondDisjunction::new(LoopCondConjunction::new(last(PrimaryDimTypes::X)));
        let two = one
            .clone()
            .or_clause(LoopCondConjunction::new(last(PrimaryDimTypes::Y)));

        let mut disagreements = Vec::new();
        for or_of_ands in [one, two] {
            for negated in [false, true] {
                for has_else_branch in [false, true] {
                    let cond: LoopCondComposite = or_of_ands.clone().into();
                    let cond = if negated { cond.negate() } else { cond };
                    let count = cond.or_of_ands.clause_count().get();
                    if takes_multi_clause_path(&cond, has_else_branch) != (count > 1) {
                        disagreements.push((count, negated, has_else_branch));
                    }
                }
            }
        }

        assert_eq!(
            disagreements,
            vec![(1, false, true), (1, true, false), (1, true, true)]
        );
    }

    /// ⭐ THE AUTHORITY'S OWN TWO BODIES, COMPILED AND EXECUTED — 303,028 cases, and the one
    /// divergence [`ComputeNode::operand_sizes`] documents is the only place they may disagree.
    /// `dsc/dsc2.cpp:2291-2346` and `:2348-2396` were LINE-EXTRACTED byte-exact — never
    /// transcribed — beside everything they read (`sys-arch-spec/arch_enums.h:13-124`,
    /// `util/sendefs/sendefs.h:30-54`, `dsc/dscdefn.h:134-207`, `sys-arch-spec/isa/isa.hpp:25-32`,
    /// `util/utils.h:39-46`) and the two tables they index (`util/sendefs/sendefs.cpp:129-141`,
    /// `dsc/dscdefn.cpp:142-144`), with `DT_ERROR` throwing as `util/dt_exception.hpp:121` makes
    /// it. 287,408 size cases and 15,620 format cases, reference digest 14470225117110672529:
    ///   * 23 components — `dsc2::memories` plus both PT endpoints, LATCH, CONSTANT, ZERO, LXLU and
    ///     NO_COMPONENT — x 71 ops x 22 formats x all five arches, one input;
    ///   * that grid again at `numFoldsEngaged = 3`, one arch each side of the RCUDD1A split;
    ///   * 23 THREE-input rotations of the same list, so operand ORDER and per-operand independence
    ///     are on the digest too, not just the singleton answers;
    ///   * 71 ops x 22 formats x 0..=4 inputs x 2 packmerge formats for the formats, which is what
    ///     covers the two arities the anchor calls out — 3 inputs yielding FOUR formats and 4
    ///     yielding three.
    /// Each line carries the enums' INTEGER indices, so digest equality also pins
    /// [`ComputeOpType::ALL`] and [`DataFormats::ALL`] to the authority's declaration order and
    /// [`Gen`] to `IsaCoreGen`'s.
    ///
    /// ⛔ THE DIVERGENCE IS NORMALISED, NOT HIDDEN: the C++ side reports `ABSENT` whenever an
    /// element count comes back negative, which is exactly when the port answers `None` — the -1
    /// bit width of `INVALID` reaching the `1024 / bitWidth` default (`:2295`). 11,347 cases take
    /// that route and 14,520 take the two `DT_ERROR` refusals; all 25,867 are `None` here, and the
    /// other 277,161 compare verbatim.
    /// ⛔ WHAT IT CANNOT REACH IS `getComputeOperandFormats`' OWN TWO THROWS, because the port takes
    /// that lookup pre-resolved — see [`ComputeNode::operand_formats`].
    #[test]
    fn e035_operand_sizes_and_formats_agree_with_the_executed_authority() {
        /// FNV-1a 64 over the canonical case lines, with the same seed on the C++ side.
        struct Digest {
            hash: u64,
            cases: u64,
            nones: u64,
        }

        impl Digest {
            fn feed(&mut self, line: &str) {
                for byte in line.bytes() {
                    self.hash ^= u64::from(byte);
                    self.hash = self.hash.wrapping_mul(1_099_511_628_211);
                }
                self.cases += 1;
            }
        }

        fn joined(values: impl IntoIterator<Item = i32>) -> String {
            values
                .into_iter()
                .map(|value| value.to_string())
                .collect::<Vec<_>>()
                .join(",")
        }

        fn size_case(
            digest: &mut Digest,
            inputs: &[SenComponent],
            r#type: ComputeOpType,
            data_format: DataFormats,
            arch: Gen,
            folds: i32,
        ) {
            let node = ComputeNode {
                r#type,
                data_format,
                inputs: inputs.to_vec(),
                num_folds_engaged: NumFoldsEngaged(folds),
                ..ComputeNode::default()
            };
            let mut line = String::from("S|");
            for input in inputs {
                line += &format!("{}.", *input as i32);
            }
            line += &format!(
                "|{}|{}|{}|{folds}|",
                r#type as i32, data_format as i32, arch as i32
            );
            match node.operand_sizes(arch) {
                Some(sizes) => line += &joined(sizes.into_iter().map(|size| size.0)),
                None => {
                    line += "ABSENT";
                    digest.nones += 1;
                }
            }
            digest.feed(&(line + "\n"));
        }

        fn format_case(
            digest: &mut Digest,
            r#type: ComputeOpType,
            data_format: DataFormats,
            inputs: usize,
            packmerge: DataFormats,
        ) {
            let node = ComputeNode {
                r#type,
                data_format,
                inputs: vec![SenComponent::Lx; inputs],
                ..ComputeNode::default()
            };
            digest.feed(&format!(
                "F|{}|{}|{inputs}|{}|{}\n",
                r#type as i32,
                data_format as i32,
                packmerge as i32,
                joined(
                    node.operand_formats(packmerge)
                        .into_iter()
                        .map(|format| format as i32)
                ),
            ));
        }

        // The C++ side iterates a `std::set<SenComponents>`, so both sides feed the digest in
        // ascending discriminant order.
        let mut components: Vec<SenComponent> = MEMORIES.to_vec();
        components.extend([
            SenComponent::Ptwest,
            SenComponent::Ptnorth,
            SenComponent::Latch,
            SenComponent::Constant,
            SenComponent::Zero,
            SenComponent::Lxlu,
            SenComponent::NoComponent,
        ]);
        components.sort_by_key(|component| *component as i32);
        let archs = [Gen::Mpw2, Gen::Mpw3, Gen::Mpw4, Gen::Rcudd1a, Gen::Sen1p5];

        let mut digest = Digest {
            hash: 1_469_598_103_934_665_603,
            cases: 0,
            nones: 0,
        };

        for &component in &components {
            for r#type in ComputeOpType::ALL {
                for data_format in DataFormats::ALL {
                    for arch in archs {
                        size_case(&mut digest, &[component], r#type, data_format, arch, 1);
                    }
                }
            }
        }
        for &component in &components {
            for r#type in ComputeOpType::ALL {
                for data_format in DataFormats::ALL {
                    for arch in [Gen::Rcudd1a, Gen::Sen1p5] {
                        size_case(&mut digest, &[component], r#type, data_format, arch, 3);
                    }
                }
            }
        }
        for (i, &component) in components.iter().enumerate() {
            let triple = [
                component,
                components[(i + 7) % components.len()],
                components[(i + 13) % components.len()],
            ];
            for r#type in ComputeOpType::ALL {
                for data_format in DataFormats::ALL {
                    size_case(&mut digest, &triple, r#type, data_format, Gen::Sen1p5, 3);
                }
            }
        }
        for r#type in ComputeOpType::ALL {
            for data_format in DataFormats::ALL {
                for inputs in 0..=4 {
                    for packmerge in [DataFormats::Senint2, DataFormats::IeeeFp32] {
                        format_case(&mut digest, r#type, data_format, inputs, packmerge);
                    }
                }
            }
        }

        assert_eq!(
            (digest.cases, digest.hash),
            (303_028, 14_470_225_117_110_672_529),
            "executed dsc/dsc2.cpp:2291-2346 and :2348-2396"
        );
        // ⛔ AND THE TWO SIDES ARE NOT MERELY AGREEING ON `ABSENT`: refusal is 8.5% of the sweep,
        // so the digest above is carrying 277,161 answered cases.
        assert_eq!(
            digest.nones, 25_867,
            "11,347 negative defaults and 14,520 DT_ERRORs"
        );
    }

    /// `dsc/dsc2.cpp:2663-2683` — the whole `ConditionNode` arm of
    /// `DesignSpaceConfig::setRelevantCompCoreCl`, executed against this port over 62,500 cases: both
    /// this node's map and the parent's swept over all 125 combinations of three cores × five
    /// per-core states (absent, listed with no corelets, `{0}`, `{1}`, `{0,1}`), for one child and
    /// two, with the guard both a loop condition and a core/corelet one. The C++ side LINE-EXTRACTS
    /// the two arms, `set_intersect`/`set_diff` (`util/utils.h:111-125`) and `hasCoreClCond`
    /// (`dsc/dsc2.h:693-695`) byte-exact and supplies only the declarations they read.
    ///
    /// ⛔ AND IT CORRECTED THE CLAIM IT PINS: of the 37,500 pairs that swap ONE core between absent
    /// and listed-with-no-corelets, 35,625 propagate IDENTICALLY. The 1,875 that do not are the
    /// single bucket `L1 N2 parent=listed-empty` — the core/corelet guard, an else region, and the
    /// PARENT's own set listing that core with no corelets, which is the state no writer but the wire
    /// produces (see [`ConditionNode::core_cl_cond`]). ⭐ AND THE CONTROL SEPARATES THAT FROM A DEAD
    /// COMPARISON: swapping the same core to `{0}` instead moves the propagation in 9,375 of the same
    /// 37,500.
    ///
    /// ⚠️ WHAT IS OUT OF THE SWEPT SPACE IS THE PARENT'S OWN THROW: `:2664` is
    /// `relevantComps_.at(NO_COMPONENT)`, so a parent without that entry throws `std::out_of_range`
    /// before either arm runs, and both sides seed it on every case.
    #[test]
    fn e044_core_cl_cond_propagation_matches_the_executed_authority() {
        /// The two arms as the authority writes them (`dsc/dsc2.cpp:2665-2683`). A child's half is
        /// `None` when the pass skipped this node and `Some` when it ran, because both arms reach
        /// their child through `relevantComps_[NO_COMPONENT]`, an `operator[]` that CREATES the
        /// entry: an empty map and no map at all are different answers downstream (`:2658-2660`).
        fn propagate(
            node: &ConditionNode,
            orig: &BTreeMap<CoreId, BTreeSet<CoreletId>>,
            children: usize,
        ) -> [Option<BTreeMap<CoreId, BTreeSet<CoreletId>>>; 2] {
            if !node.has_core_cl_cond() || children == 0 {
                return [None, None];
            }
            let mut then_core_cl = BTreeMap::new();
            for (core, cls) in &node.core_cl_cond {
                let Some(orig_cls) = orig.get(core) else {
                    continue;
                };
                let new_set: BTreeSet<CoreletId> = orig_cls.intersection(cls).copied().collect();
                if !new_set.is_empty() {
                    then_core_cl.insert(*core, new_set);
                }
            }
            let mut else_core_cl = None;
            if children >= 2 {
                let mut half = orig.clone();
                for (core, cls) in &node.core_cl_cond {
                    let Some(else_cls) = half.get_mut(core) else {
                        continue;
                    };
                    *else_cls = else_cls.difference(cls).copied().collect();
                    if else_cls.is_empty() {
                        half.remove(core);
                    }
                }
                else_core_cl = Some(half);
            }
            [Some(then_core_cl), else_core_cl]
        }

        fn shown(map: &BTreeMap<CoreId, BTreeSet<CoreletId>>) -> String {
            let cores: Vec<String> = map
                .iter()
                .map(|(core, cls)| {
                    let cls: Vec<String> = cls.iter().map(|cl| cl.0.to_string()).collect();
                    format!("{}:[{}]", core.0, cls.join(","))
                })
                .collect();
            format!("{{{}}}", cores.join(";"))
        }

        /// The five per-core states, in the order the C++ sweep spells them.
        fn build(codes: [u8; 3]) -> BTreeMap<CoreId, BTreeSet<CoreletId>> {
            let mut map = BTreeMap::new();
            for (core, code) in codes.into_iter().enumerate() {
                let cls = match code {
                    0 => continue,
                    1 => BTreeSet::new(),
                    2 => BTreeSet::from([CoreletId(0)]),
                    3 => BTreeSet::from([CoreletId(1)]),
                    _ => BTreeSet::from([CoreletId(0), CoreletId(1)]),
                };
                map.insert(CoreId(core as u8), cls);
            }
            map
        }

        fn one_case(
            loop_empty: bool,
            children: usize,
            cond: &BTreeMap<CoreId, BTreeSet<CoreletId>>,
            orig: &BTreeMap<CoreId, BTreeSet<CoreletId>>,
        ) -> String {
            let mut node = ConditionNode {
                core_cl_cond: cond.clone(),
                ..ConditionNode::default()
            };
            if !loop_empty {
                node.loop_cond = Some(
                    LoopCondConjunction::new(LoopCond {
                        dim: PrimaryDimTypes::Y,
                        cond_op: LoopCondOp::Eq,
                        cond_val: CondVal::Last,
                    })
                    .into(),
                );
            }
            let [then_half, else_half] = propagate(&node, orig, children);
            let half = |region: &Option<BTreeMap<CoreId, BTreeSet<CoreletId>>>| match region {
                None => "absent".to_string(),
                Some(map) => shown(map),
            };
            format!(
                "L{}|N{children}|c{}|o{}|t{}|e{}",
                u8::from(loop_empty),
                shown(cond),
                shown(orig),
                half(&then_half),
                if children > 1 {
                    half(&else_half)
                } else {
                    "none".to_string()
                }
            )
        }

        let codes =
            || (0..5).flat_map(|a| (0..5).flat_map(move |b| (0..5).map(move |c| [a, b, c])));
        let mut hash = 14_695_981_039_346_656_037_u64;
        let mut cases = 0_u64;
        let mut collapse_agree = 0_u64;
        let mut collapse_disagree = 0_u64;
        let mut distinguishable = 0_u64;
        let mut buckets: BTreeMap<(u8, usize, u8), u64> = BTreeMap::new();

        for loop_empty in [false, true] {
            for children in 1..=2 {
                for cond in codes() {
                    let cond_map = build(cond);
                    for orig in codes() {
                        let line = one_case(loop_empty, children, &cond_map, &build(orig));
                        for byte in line.bytes() {
                            hash ^= u64::from(byte);
                            hash = hash.wrapping_mul(1_099_511_628_211);
                        }
                        cases += 1;
                    }
                }
                // Does swapping ONE core between "absent" and "listed with no corelets" change the
                // propagated then/else regions?  Everything else held fixed.
                for core in 0..3 {
                    for other0 in 0..5 {
                        for other1 in 0..5 {
                            for orig in codes() {
                                let orig_map = build(orig);
                                let mut swept = [0_u8; 3];
                                let fill = [other0, other1];
                                let mut next = 0;
                                for (position, code) in swept.iter_mut().enumerate() {
                                    if position != core {
                                        *code = fill[next];
                                        next += 1;
                                    }
                                }
                                // The propagated halves only, never the printed condition itself.
                                let halves = |codes: [u8; 3]| {
                                    let line =
                                        one_case(loop_empty, children, &build(codes), &orig_map);
                                    line[line.find("|t").unwrap()..].to_string()
                                };
                                let absent = halves(swept);
                                swept[core] = 1;
                                let listed_empty = halves(swept);
                                swept[core] = 2;
                                let listed_zero = halves(swept);
                                if absent == listed_empty {
                                    collapse_agree += 1;
                                } else {
                                    collapse_disagree += 1;
                                    *buckets
                                        .entry((u8::from(loop_empty), children, orig[core]))
                                        .or_default() += 1;
                                }
                                if listed_zero != absent {
                                    distinguishable += 1;
                                }
                            }
                        }
                    }
                }
            }
        }

        assert_eq!(
            (cases, hash),
            (62_500, 9_967_431_661_323_938_275),
            "executed dsc/dsc2.cpp:2663-2683"
        );
        assert_eq!((collapse_agree, collapse_disagree), (35_625, 1_875));
        assert_eq!(distinguishable, 9_375, "the control");
        assert_eq!(
            buckets.into_iter().collect::<Vec<_>>(),
            vec![((1, 2, 1), 1_875)],
            "every disagreement needs the core/corelet guard, an else region, and the PARENT listing \
             that core with no corelets"
        );
    }

    /// The one production coordinate build: `ddc/ddc_fold.cpp:2755-2769` walks an allocation's fold
    /// params from the last down to the first and always inserts at position 0, which leaves the levels
    /// in fold-param order with the spatial ones outermost. Three params, one level of each category, is
    /// the shape every case below was measured against.
    const FOLD_PARAMS: [(FoldDimSize, &str, Alpha, Beta); 3] = [
        (FoldDimSize(4), "core", Alpha(10), Beta(1)),
        (FoldDimSize(3), "corelet", Alpha(20), Beta(2)),
        (FoldDimSize(5), "elem_arr_0", Alpha(30), Beta(3)),
    ];

    fn canonical_tower(coord: &mut CoordinateType, dim: PrimaryDimTypes) {
        for pos in (0..FOLD_PARAMS.len()).rev() {
            let (cardinality, label, alpha, beta) = FOLD_PARAMS[pos];
            let category = match pos {
                0 => CoordinateCategory::Spatial,
                1 => CoordinateCategory::Temporal,
                _ => CoordinateCategory::ElemArr,
            };
            assert_eq!(
                coord.add_fold(
                    dim,
                    category,
                    cardinality,
                    label,
                    alpha,
                    beta,
                    FoldDimPos(0)
                ),
                Some(())
            );
        }
    }

    /// A dim's levels read back the way the passes outside this class read them — through
    /// `coordinates_.at(dim)` (`ddc/ddc_fold.cpp:1192-1197`, `dsc/dsc2.cpp:3848-3871`).
    fn levels(coord: &CoordinateType, dim: PrimaryDimTypes) -> Vec<(u32, String, i64, i64)> {
        let fold_manager = &coord.coordinates()[&dim];
        (0..fold_manager.num_dims())
            .map(|pos| {
                let pos = FoldDimPos(i32::try_from(pos).unwrap());
                let (alpha, beta) = fold_manager.alpha_beta(pos).unwrap();
                (
                    fold_manager.fold_dim_size(pos).unwrap().0,
                    fold_manager.fold_dim_prop(pos).unwrap().label().to_string(),
                    *alpha,
                    *beta,
                )
            })
            .collect()
    }

    fn counts(coord: &CoordinateType, dim: PrimaryDimTypes) -> (usize, usize, usize) {
        (
            coord.num_of_spatial_folds(dim),
            coord.num_of_temporal_folds(dim),
            coord.num_of_elem_arr_folds(dim),
        )
    }

    /// `dsc/dsc2.h:118-142` and `:154-164` and `:222-234`, executed: the levels come back in fold-param
    /// order carrying the alpha and beta each was given, and the two count thresholds put one level in
    /// each category.
    ///
    /// ⛔ AND THE TWO CATEGORY READERS DISAGREE ON A NEGATIVE POSITION. `getFoldCategory` answers
    /// `UNKNOWN_COORD` off both ends of the tower — this port's [`None`] — while
    /// `getCoordinateCategoryOfPos`, whose `DT_CHECK` is one-sided (`:158`), reports position `-1` as
    /// `SPATIAL_COORD`.
    #[test]
    fn the_canonical_tower_is_one_level_of_each_category() {
        let mut coord = CoordinateType::default();
        canonical_tower(&mut coord, PrimaryDimTypes::X);

        assert_eq!(
            levels(&coord, PrimaryDimTypes::X),
            vec![
                (4, "core".to_string(), 10, 1),
                (3, "corelet".to_string(), 20, 2),
                (5, "elem_arr_0".to_string(), 30, 3),
            ],
            "numDims=3, in fold-param order"
        );
        assert_eq!(
            (0..3)
                .map(|pos| (
                    coord.coordinate_category_of_pos(PrimaryDimTypes::X, FoldDimPos(pos)),
                    coord.fold_category(PrimaryDimTypes::X, FoldDimPos(pos)),
                ))
                .collect::<Vec<_>>(),
            vec![
                (
                    Some(CoordinateCategory::Spatial),
                    Some(CoordinateCategory::Spatial)
                ),
                (
                    Some(CoordinateCategory::Temporal),
                    Some(CoordinateCategory::Temporal)
                ),
                (
                    Some(CoordinateCategory::ElemArr),
                    Some(CoordinateCategory::ElemArr)
                ),
            ],
            "the two readers agree inside the tower"
        );
        assert_eq!(counts(&coord, PrimaryDimTypes::X), (1, 1, 1));

        assert_eq!(
            coord.fold_category(PrimaryDimTypes::X, FoldDimPos(-1)),
            None,
            "foldCat(-1)=UNKNOWN"
        );
        assert_eq!(
            coord.fold_category(PrimaryDimTypes::X, FoldDimPos(3)),
            None,
            "foldCat(3)=UNKNOWN"
        );
        assert_eq!(
            coord.coordinate_category_of_pos(PrimaryDimTypes::X, FoldDimPos(-1)),
            Some(CoordinateCategory::Spatial),
            "catOfPos(-1)=SPATIAL — the disagreement"
        );

        assert!(coord.has_coord_for_dim(PrimaryDimTypes::X));
        assert!(!coord.has_coord_for_dim(PrimaryDimTypes::Y));
        assert_eq!(coord.tensor_dims(), vec![PrimaryDimTypes::X]);
        assert!(!coord.fold_constructed());
        coord.complete_fold_construction();
        assert!(coord.fold_constructed());
    }

    /// `operator=` (`:166-184`) and `clear()` (`:81-96`), executed. The copy is a level-for-level replay
    /// through `addFold`, so the sizes, labels, alphas and betas come back in the source's order, and
    /// the core map, the constructed flag and the padding form all survive it.
    ///
    /// ⛔ `clear()` LEAVES `coreIdToWkSlice_` ALONE, so it is not `*self = Self::default()` — measured:
    /// one core entry written before the clear is still there after it, while the towers, the counts and
    /// the padding are gone.
    #[test]
    fn a_copy_replays_the_tower_and_clear_keeps_the_core_map() {
        let mut coord = CoordinateType::default();
        canonical_tower(&mut coord, PrimaryDimTypes::X);
        coord.complete_fold_construction();
        coord
            .core_id_to_wk_slice
            .entry(CoreId(7))
            .or_default()
            .insert(PrimaryDimTypes::X, WkSliceIdx(2));
        coord.set_padding(PrimaryDimTypes::X, PadType::PaddedNoZeroPad);

        let copy = coord.clone();
        assert_eq!(
            levels(&copy, PrimaryDimTypes::X),
            vec![
                (4, "core".to_string(), 10, 1),
                (3, "corelet".to_string(), 20, 2),
                (5, "elem_arr_0".to_string(), 30, 3),
            ],
            "the replay is an identity on an affine tower"
        );
        assert_eq!(counts(&copy, PrimaryDimTypes::X), (1, 1, 1));
        assert_eq!(
            (
                copy.core_id_to_wk_slice[&CoreId(7)][&PrimaryDimTypes::X],
                copy.fold_constructed(),
                copy.padding(PrimaryDimTypes::X),
            ),
            (WkSliceIdx(2), true, PadType::PaddedNoZeroPad)
        );
        assert_eq!(coord, copy, "equal=1");

        coord.clear();
        assert_eq!(
            (
                coord.coordinates().len(),
                coord.core_id_to_wk_slice.len(),
                coord.num_of_spatial_folds(PrimaryDimTypes::X),
                coord.padding(PrimaryDimTypes::X),
                coord.fold_constructed(),
            ),
            (0, 1, 0, PadType::NoPad, false)
        );
    }

    /// `clearFoldForDim(dim)` (`:98-114`), executed: the dim keeps its key and its padding form and
    /// loses its levels and its counts.
    ///
    /// ⛔ AND A COPY THEN DROPS THE KEY, because the replay has no level to make a call out of — so the
    /// copy is not equal to its source, while the padding form for the dropped dim survives into it.
    #[test]
    fn a_cleared_fold_keeps_its_key_until_it_is_copied() {
        let mut coord = CoordinateType::default();
        canonical_tower(&mut coord, PrimaryDimTypes::X);
        coord.set_padding(PrimaryDimTypes::X, PadType::PaddedNoZeroPad);
        coord.clear_fold_for_dim(PrimaryDimTypes::X);

        assert_eq!(
            (
                coord.coordinates().len(),
                coord.coordinates()[&PrimaryDimTypes::X].num_dims(),
                counts(&coord, PrimaryDimTypes::X),
                coord.padding(PrimaryDimTypes::X),
                coord.has_coord_for_dim(PrimaryDimTypes::X),
            ),
            (1, 0, (0, 0, 0), PadType::PaddedNoZeroPad, true)
        );

        let copy = coord.clone();
        assert_eq!(
            (
                copy.coordinates().len(),
                copy.has_coord_for_dim(PrimaryDimTypes::X),
                copy.padding(PrimaryDimTypes::X),
            ),
            (0, false, PadType::PaddedNoZeroPad)
        );
        assert_ne!(copy, coord, "equalToSource=0");
    }

    /// ⛔ `operator=` RECOMPUTES THE THREE COUNTS RATHER THAN COPYING THEM (`:170-179`), which is
    /// observable the moment they disagree with the tower. Measured: a temporal count raised to 4 on a
    /// three-level tower makes BOTH inner positions read as temporal, and the copy comes back with the
    /// counts those positions imply — 1 spatial, 2 temporal, 0 element-arrangement — so a coordinate
    /// copied out of that state is NOT equal to the one it came from.
    ///
    /// ⭐ THE ELEMENT-ARRANGEMENT COUNT IS INVISIBLE TO BOTH CATEGORY READERS, since every position at
    /// or past `spatial + temporal` is element arrangement whatever it says: raising it to 7 moves no
    /// position, and the copy puts it back to 1.
    #[test]
    fn a_copy_recomputes_the_counts_from_the_sources_own_categories() {
        let mut coord = CoordinateType::default();
        canonical_tower(&mut coord, PrimaryDimTypes::X);
        assert_eq!(
            coord.set_num_of_temporal_fold_per_dim(PrimaryDimTypes::X, 4),
            Some(())
        );
        assert_eq!(counts(&coord, PrimaryDimTypes::X), (1, 4, 1), "the source");

        let copy = coord.clone();
        assert_eq!(
            (
                counts(&copy, PrimaryDimTypes::X),
                copy.coordinates()[&PrimaryDimTypes::X].num_dims(),
            ),
            ((1, 2, 0), 3)
        );
        assert_ne!(coord, copy, "equalAfterCopy=0");

        let mut wide = CoordinateType::default();
        canonical_tower(&mut wide, PrimaryDimTypes::X);
        assert_eq!(
            wide.set_num_of_elem_arr_fold_per_dim(PrimaryDimTypes::X, 7),
            Some(())
        );
        assert_eq!(counts(&wide, PrimaryDimTypes::X), (1, 1, 7));
        assert_eq!(
            (0..3)
                .map(|pos| wide.coordinate_category_of_pos(PrimaryDimTypes::X, FoldDimPos(pos)))
                .collect::<Vec<_>>(),
            vec![
                Some(CoordinateCategory::Spatial),
                Some(CoordinateCategory::Temporal),
                Some(CoordinateCategory::ElemArr),
            ],
            "no position moved"
        );
        assert_eq!(counts(&wide.clone(), PrimaryDimTypes::X), (1, 1, 1));
    }

    /// `setNumOfTemporalFoldPerDim` and `setNumOfElemArrFoldPerDim` (`:215-220`), executed: both go
    /// through `.at(dim)` on their OWN count map, so neither will insert.
    ///
    /// ⛔ WHICH MEANS A DIM THAT HAS A TOWER CAN STILL REFUSE THEM — measured, the temporal setter throws
    /// on a dim whose only level was added as spatial, because that dim has no temporal entry to
    /// overwrite. Having a coordinate is not having a count.
    #[test]
    fn the_count_setters_refuse_a_dim_with_no_count_of_that_category() {
        let mut coord = CoordinateType::default();
        assert_eq!(
            coord.add_fold(
                PrimaryDimTypes::X,
                CoordinateCategory::Spatial,
                FoldDimSize(4),
                "core",
                Alpha(10),
                Beta(1),
                FoldDimPos(0),
            ),
            Some(())
        );

        assert_eq!(
            coord.set_num_of_temporal_fold_per_dim(PrimaryDimTypes::In, 4),
            None,
            "a dim with no tower at all"
        );
        assert_eq!(
            coord.set_num_of_elem_arr_fold_per_dim(PrimaryDimTypes::In, 4),
            None
        );
        assert_eq!(
            coord.set_num_of_temporal_fold_per_dim(PrimaryDimTypes::X, 2),
            None,
            "and a dim with a spatial-only tower"
        );
        assert_eq!(
            counts(&coord, PrimaryDimTypes::X),
            (1, 0, 0),
            "the control: the spatial count it does have"
        );
    }

    /// `:157-158` and `:224`, executed: both category readers refuse a dim with no tower, and
    /// `getCoordinateCategoryOfPos` also refuses a position at or past its depth.
    #[test]
    fn the_category_readers_refuse_an_absent_dim_and_a_position_past_the_end() {
        let mut coord = CoordinateType::default();
        canonical_tower(&mut coord, PrimaryDimTypes::X);

        assert_eq!(
            coord.coordinate_category_of_pos(PrimaryDimTypes::In, FoldDimPos(0)),
            None
        );
        assert_eq!(
            coord.fold_category(PrimaryDimTypes::In, FoldDimPos(0)),
            None
        );
        assert_eq!(
            coord.coordinate_category_of_pos(PrimaryDimTypes::X, FoldDimPos(3)),
            None
        );
        assert_eq!(
            coord.coordinate_category_of_pos(PrimaryDimTypes::X, FoldDimPos(2)),
            Some(CoordinateCategory::ElemArr),
            "the control: the last position it does have"
        );
    }

    /// `operator==` (`:188-213`), executed.
    ///
    /// ⛔ IT NEVER READS `foldConstructed_` — two towers built alike are equal with one of them marked
    /// constructed — AND IT DOES READ `coreIdToWkSlice_`, so one core entry on one side is enough to
    /// separate them.
    #[test]
    fn equality_ignores_the_constructed_flag_and_reads_the_core_map() {
        let mut lhs = CoordinateType::default();
        let mut rhs = CoordinateType::default();
        canonical_tower(&mut lhs, PrimaryDimTypes::X);
        canonical_tower(&mut rhs, PrimaryDimTypes::X);
        lhs.complete_fold_construction();
        assert_eq!(lhs, rhs, "== ignores foldConstructed");

        rhs.complete_fold_construction();
        rhs.core_id_to_wk_slice
            .entry(CoreId(1))
            .or_default()
            .insert(PrimaryDimTypes::X, WkSliceIdx(0));
        assert_ne!(lhs, rhs, "== sees coreIdToWkSlice_");
    }

    /// Two dims, one with the canonical three levels and a padding form and one with a single spatial
    /// level, plus a core map two cores wide — the object `printCoordinates` (`:257-315`) writes and
    /// `dsc_import_json` reads back, transcribed from the executed authority.
    ///
    /// ⭐ THE PUNCTUATION IS THE POINT: `"folds" : ` carries a SECOND space because `FoldManager::print`
    /// opens with one of its own, a non-last dim's closing brace is followed by `", "` with a TRAILING
    /// space before the newline, and `"coreIdToWkSlice_" : { ` and its closing `} ` both end in a space.
    /// Only `PrintContent(false)` is measured here; the value block a `true` adds is
    /// [`FoldManager::print`]'s and is measured there.
    #[test]
    fn print_coordinates_matches_the_authority() {
        let mut coord = CoordinateType::default();
        canonical_tower(&mut coord, PrimaryDimTypes::X);
        coord.set_padding(PrimaryDimTypes::X, PadType::PaddedNoZeroPad);
        assert_eq!(
            coord.add_fold(
                PrimaryDimTypes::Y,
                CoordinateCategory::Spatial,
                FoldDimSize(2),
                "core_y",
                Alpha(7),
                Beta(0),
                FoldDimPos(0),
            ),
            Some(())
        );
        for (core, dim, slice) in [
            (CoreId(3), PrimaryDimTypes::X, WkSliceIdx(1)),
            (CoreId(3), PrimaryDimTypes::Y, WkSliceIdx(0)),
            (CoreId(5), PrimaryDimTypes::X, WkSliceIdx(2)),
        ] {
            coord
                .core_id_to_wk_slice
                .entry(core)
                .or_default()
                .insert(dim, slice);
        }

        let mut out = String::new();
        assert_eq!(
            coord.print_coordinates(&mut out, PrintContent(false), ""),
            Some(())
        );
        assert_eq!(
            out,
            concat!(
                "\n",
                "\"coordinates_\" : {\n",
                "  \"coordInfo\" : {\n",
                "    \"x\" : {\n",
                "      \"spatial\" : 1,\n",
                "      \"temporal\" : 1,\n",
                "      \"elemArr\" : 1,\n",
                "      \"padding\" : \"padded_nozeropad\",\n",
                "      \"folds\" :  {\n",
                "        \"dim_prop_func\" : [\n",
                "          { \"Affine\" : {\"alpha_\" : 10, \"beta_\" : 1} },\n",
                "          { \"Affine\" : {\"alpha_\" : 20, \"beta_\" : 2} },\n",
                "          { \"Affine\" : {\"alpha_\" : 30, \"beta_\" : 3} }\n",
                "        ],\n",
                "        \"dim_prop_attr\" : [\n",
                "          { \"factor_\" : 4, \"label_\" : \"core\" },\n",
                "          { \"factor_\" : 3, \"label_\" : \"corelet\" },\n",
                "          { \"factor_\" : 5, \"label_\" : \"elem_arr_0\" }\n",
                "        ]\n",
                "      }\n",
                "    }, \n",
                "    \"y\" : {\n",
                "      \"spatial\" : 1,\n",
                "      \"temporal\" : 0,\n",
                "      \"elemArr\" : 0,\n",
                "      \"padding\" : \"nopad\",\n",
                "      \"folds\" :  {\n",
                "        \"dim_prop_func\" : [\n",
                "          { \"Affine\" : {\"alpha_\" : 7, \"beta_\" : 0} }\n",
                "        ],\n",
                "        \"dim_prop_attr\" : [\n",
                "          { \"factor_\" : 2, \"label_\" : \"core_y\" }\n",
                "        ]\n",
                "      }\n",
                "    }\n",
                "  },\n",
                "  \"coreIdToWkSlice_\" : { \n",
                "    \"3\" : { \"x\" : 1, \"y\" : 0 }, \n",
                "    \"5\" : { \"x\" : 2 }\n",
                "  } \n",
                "}\n",
            )
        );

        let mut out = String::new();
        assert_eq!(coord.debug_print(&mut out), Some(()));
        assert_eq!(
            out,
            concat!(
                "\n",
                "DDC Coordinates<int64_t>: 2 coordinate entries\n",
                "\n",
                "Primary Dim= x\n",
                "  Fold dimension= \"factor_\" : 4, \"label_\" : \"core\"\n",
                "    Affine:\"alpha_\" : 10, \"beta_\" : 1\n",
                "  Fold dimension= \"factor_\" : 3, \"label_\" : \"corelet\"\n",
                "    Affine:\"alpha_\" : 20, \"beta_\" : 2\n",
                "  Fold dimension= \"factor_\" : 5, \"label_\" : \"elem_arr_0\"\n",
                "    Affine: \"alpha_\" : 30, \"beta_\" : 3\n",
                "  #Spatial  = 1\n",
                "  #Temporal = 1\n",
                "  #ElemArr  = 1\n",
                "  Padding: { (x, padded_nozeropad) }\n",
                "\n",
                "Primary Dim= y\n",
                "  Fold dimension= \"factor_\" : 2, \"label_\" : \"core_y\"\n",
                "    Affine: \"alpha_\" : 7, \"beta_\" : 0\n",
                "  #Spatial  = 1\n",
                "  #Temporal = 0\n",
                "  #ElemArr  = 0\n",
                "  Padding: { (y, nopad) }\n",
                "  coreIdToWkSlice_ : { \n",
                "    3 : { x : 1, y : 0 }, \n",
                "    5 : { x : 2 }\n",
                "  } \n",
            ),
            "debugPrint — and the two levels of one tower that differ only in whether a space follows \
             `Affine:` are the leaf/non-leaf arms (`:379-384`), not a typo"
        );
    }

    /// ⛔ `ps` PREFIXES EVERY LINE OF THE OBJECT INCLUDING THE FIRST AND THE LAST, and nothing inside
    /// the fold manager's own block gets it twice (`:258-314`) — measured with a two-character prefix so
    /// the indentation is distinguishable from it.
    #[test]
    fn print_coordinates_threads_the_prefix_through_every_line() {
        let mut coord = CoordinateType::default();
        assert_eq!(
            coord.add_fold(
                PrimaryDimTypes::Y,
                CoordinateCategory::Spatial,
                FoldDimSize(2),
                "core_y",
                Alpha(7),
                Beta(0),
                FoldDimPos(0),
            ),
            Some(())
        );
        coord
            .core_id_to_wk_slice
            .entry(CoreId(3))
            .or_default()
            .insert(PrimaryDimTypes::Y, WkSliceIdx(0));

        let mut out = String::new();
        assert_eq!(
            coord.print_coordinates(&mut out, PrintContent(false), ".."),
            Some(())
        );
        assert_eq!(
            out,
            concat!(
                "\n",
                "..\"coordinates_\" : {\n",
                "..  \"coordInfo\" : {\n",
                "..    \"y\" : {\n",
                "..      \"spatial\" : 1,\n",
                "..      \"temporal\" : 0,\n",
                "..      \"elemArr\" : 0,\n",
                "..      \"padding\" : \"nopad\",\n",
                "..      \"folds\" :  {\n",
                "..        \"dim_prop_func\" : [\n",
                "..          { \"Affine\" : {\"alpha_\" : 7, \"beta_\" : 0} }\n",
                "..        ],\n",
                "..        \"dim_prop_attr\" : [\n",
                "..          { \"factor_\" : 2, \"label_\" : \"core_y\" }\n",
                "..        ]\n",
                "..      }\n",
                "..    }\n",
                "..  },\n",
                "..  \"coreIdToWkSlice_\" : { \n",
                "..    \"3\" : { \"y\" : 0 }\n",
                "..  } \n",
                "..}\n",
            )
        );
    }

    /// ⛔ AN EMPTY COORDINATE STILL PRINTS ITS TWO EMPTY SUB-OBJECTS, while `debugPrint` prints ONE LINE
    /// with no trailing newline and no core block at all — the `if (!coreIdToWkSlice_.empty())` guard
    /// (`:411`) has no counterpart in `printCoordinates`.
    #[test]
    fn an_empty_coordinate_prints_its_shell() {
        let coord = CoordinateType::default();

        let mut out = String::new();
        assert_eq!(
            coord.print_coordinates(&mut out, PrintContent(false), ""),
            Some(())
        );
        assert_eq!(
            out,
            concat!(
                "\n",
                "\"coordinates_\" : {\n",
                "  \"coordInfo\" : {\n",
                "  },\n",
                "  \"coreIdToWkSlice_\" : { \n",
                "  } \n",
                "}\n",
            )
        );

        let mut out = String::new();
        assert_eq!(coord.debug_print(&mut out), Some(()));
        assert_eq!(out, "\nDDC Coordinates<int64_t>: 0 coordinate entries");
    }

    /// ⛔ A DIM WHOSE TOWER WAS CLEARED PRINTS A STRING WHERE THE OTHERS PRINT AN OBJECT —
    /// `"folds" : "0"` — because a zero-dimension [`FoldManager`] prints just its datum and ignores both
    /// the prefix and `printContent`. The output is therefore not JSON-uniform across dims, and
    /// `debugPrint` drops the dim's `Fold dimension=` lines entirely while still counting it as an
    /// entry and still printing its padding.
    #[test]
    fn a_dim_with_no_levels_prints_its_datum() {
        let mut coord = CoordinateType::default();
        for (dim, cardinality, label, alpha) in [
            (PrimaryDimTypes::X, FoldDimSize(4), "core", Alpha(10)),
            (PrimaryDimTypes::Y, FoldDimSize(2), "core_y", Alpha(7)),
        ] {
            let beta = if dim == PrimaryDimTypes::X {
                Beta(1)
            } else {
                Beta(0)
            };
            assert_eq!(
                coord.add_fold(
                    dim,
                    CoordinateCategory::Spatial,
                    cardinality,
                    label,
                    alpha,
                    beta,
                    FoldDimPos(0),
                ),
                Some(())
            );
        }
        coord.clear_fold_for_dim(PrimaryDimTypes::X);

        assert_eq!(
            coord.tensor_dims(),
            vec![PrimaryDimTypes::X, PrimaryDimTypes::Y],
            "tensorDims: 4 5 — the cleared dim is still one of them"
        );

        let expected = concat!(
            "\n",
            "\"coordinates_\" : {\n",
            "  \"coordInfo\" : {\n",
            "    \"x\" : {\n",
            "      \"spatial\" : 0,\n",
            "      \"temporal\" : 0,\n",
            "      \"elemArr\" : 0,\n",
            "      \"padding\" : \"nopad\",\n",
            "      \"folds\" : \"0\"\n",
            "    }, \n",
            "    \"y\" : {\n",
            "      \"spatial\" : 1,\n",
            "      \"temporal\" : 0,\n",
            "      \"elemArr\" : 0,\n",
            "      \"padding\" : \"nopad\",\n",
            "      \"folds\" :  {\n",
            "        \"dim_prop_func\" : [\n",
            "          { \"Affine\" : {\"alpha_\" : 7, \"beta_\" : 0} }\n",
            "        ],\n",
            "        \"dim_prop_attr\" : [\n",
            "          { \"factor_\" : 2, \"label_\" : \"core_y\" }\n",
            "        ]\n",
            "      }\n",
            "    }\n",
            "  },\n",
            "  \"coreIdToWkSlice_\" : { \n",
            "  } \n",
            "}\n",
        );
        let mut out = String::new();
        assert_eq!(
            coord.print_coordinates(&mut out, PrintContent(false), ""),
            Some(())
        );
        assert_eq!(out, expected);

        let mut with_content = String::new();
        assert_eq!(
            coord.print_coordinates(&mut with_content, PrintContent(true), ""),
            Some(())
        );
        assert_eq!(
            with_content.find("\"folds\" : \"0\"\n"),
            expected.find("\"folds\" : \"0\"\n"),
            "the zero-dim block is the same under either flag"
        );
        assert!(
            with_content.contains(
                "\"data_\" : {\n          \"[0]\" :\"0\",\n          \"[1]\" :\"7\"\n        }\n"
            ),
            "the control: the dim that DOES have a level gains a value block"
        );

        let mut out = String::new();
        assert_eq!(coord.debug_print(&mut out), Some(()));
        assert_eq!(
            out,
            concat!(
                "\n",
                "DDC Coordinates<int64_t>: 2 coordinate entries\n",
                "\n",
                "Primary Dim= x\n",
                "  #Spatial  = 0\n",
                "  #Temporal = 0\n",
                "  #ElemArr  = 0\n",
                "  Padding: { (x, nopad) }\n",
                "\n",
                "Primary Dim= y\n",
                "  Fold dimension= \"factor_\" : 2, \"label_\" : \"core_y\"\n",
                "    Affine: \"alpha_\" : 7, \"beta_\" : 0\n",
                "  #Spatial  = 1\n",
                "  #Temporal = 0\n",
                "  #ElemArr  = 0\n",
                "  Padding: { (y, nopad) }",
            ),
            "no core map, so no core block and no trailing newline"
        );
    }

    /// `ConstantInfo fresh;` — `data_` has no initialiser (`dsc/dsc2.h:49-50`), so the datum is
    /// `FoldManager`'s default. Measured: `fmt=2 name= sym=0 numDims=0 zeroFold=1 data0=[]`.
    #[test]
    fn constant_info_default_datum_is_a_zero_dimension_empty_vector() {
        let fresh = ConstantInfo::default();
        assert_eq!(fresh.data_format, DataFormats::Invalid);
        assert_eq!(fresh.name, "");
        assert!(!fresh.is_data_symbolic);
        assert_eq!(fresh.data.num_dims(), 0);
        assert!(fresh.data.has_zero_fold_dim());
        assert_eq!(fresh.data.get_data(&[]), Some(Vec::new()));
    }

    /// `operator=` (`dsc/dsc2.h:54-60`) against its own two measured cases: onto a fresh destination,
    /// and onto one that already carries a different flag and a one-dimension datum.
    #[test]
    fn constant_info_assign_drops_the_symbolic_flag_and_clones_the_datum() {
        let mut src = ConstantInfo {
            data_format: DataFormats::IeeeFp32,
            name: "padval".to_string(),
            is_data_symbolic: true,
            ..ConstantInfo::default()
        };
        assert_eq!(
            src.data.build_all_constant_fold_space(&[
                FoldDimProp::new(FoldDimSize(2), "core"),
                FoldDimProp::new(FoldDimSize(3), "corelet"),
            ]),
            Some(())
        );
        let at_origin = [FoldDimIndex(0), FoldDimIndex(0)];
        assert_eq!(src.data.insert_data(vec![7, 8, 9], &at_origin), Some(()));

        // `dstFresh = src` — measured `fmt=1 name=padval sym=0 numDims=2 data0=[7,8,9]`, where the
        // copy CONSTRUCTOR of the same source gives `sym=1`.
        let mut dst_fresh = ConstantInfo::default();
        dst_fresh.assign(&src);
        assert_eq!(dst_fresh.data_format, DataFormats::IeeeFp32);
        assert_eq!(dst_fresh.name, "padval");
        assert!(!dst_fresh.is_data_symbolic, "`operator=` drops the flag");
        assert!(
            src.clone().is_data_symbolic,
            "the copy constructor carries it"
        );
        assert_eq!(dst_fresh.data.get_data(&at_origin), Some(vec![7, 8, 9]));

        // `dstUsed = srcNoFlag` — measured `sym=1` before AND after, with `numDims` 1 then 2.
        let mut dst_used = ConstantInfo {
            data_format: DataFormats::Sen169Fp16,
            name: "was-here".to_string(),
            is_data_symbolic: true,
            ..ConstantInfo::default()
        };
        assert_eq!(
            dst_used
                .data
                .build_all_constant_fold_space(&[FoldDimProp::new(FoldDimSize(5), "other")]),
            Some(())
        );
        assert_eq!(
            dst_used.data.insert_data(vec![99], &[FoldDimIndex(0)]),
            Some(())
        );
        let mut src_no_flag = src.clone();
        src_no_flag.is_data_symbolic = false;
        assert_eq!(
            dst_used.data.assign(&src_no_flag.data),
            None,
            "which is why the authority clones the datum instead of assigning it"
        );
        dst_used.assign(&src_no_flag);
        assert!(
            dst_used.is_data_symbolic,
            "the destination's own flag stands"
        );
        assert_eq!(dst_used.data_format, DataFormats::IeeeFp32);
        assert_eq!(dst_used.name, "padval");
        assert_eq!(dst_used.data.num_dims(), 2, "reshaped by the clone");
        assert_eq!(dst_used.data.get_data(&at_origin), Some(vec![7, 8, 9]));
    }

    /// [`CoordPropInfoType`]'s five initialisers, and the two fields the reverse item its two
    /// default-construction sites build does NOT forward (`ddc/ddc_fold.cpp:1222-1226`, `:1924-1928`).
    ///
    /// ⭐ MEASURED against the authority at `a0d29abbed` by a probe over the real header, which
    /// default-constructs the struct, aggregate-initialises `retry`'s six (`ddc/ddc.h:438-441`) and
    /// copies a `scaleDown = true` source the way the reverse sites do:
    /// `default dataConnect=[] refIsProducer=1 propState=0 dims=0 scaleDown=0`,
    /// `retry   dataConnect=[dc0] refIsProducer=0 propState=0 dims=1 scaleDown=0`,
    /// `reverse dataConnect=[dc1] refIsProducer=0 propState=0 dims=0 scaleDown=0`.
    #[test]
    fn the_reverse_item_forwards_neither_scale_down_nor_dims() {
        let fresh = CoordPropInfoType::default();
        assert_eq!(fresh.data_connect, DataConnect(String::new()));
        assert!(fresh.ref_is_producer, "the authority's `= true`");
        assert_eq!(fresh.prop_state, PropStateType::NotProcessed);
        assert!(fresh.dims_to_propagate.is_empty());
        assert!(!fresh.scale_down);

        let retried = CoordPropInfoType {
            data_connect: DataConnect("dc0".to_string()),
            ref_is_producer: false,
            prop_state: PropStateType::NotProcessed,
            dims_to_propagate: vec![PrimaryDimTypes::X],
            ..CoordPropInfoType::default()
        };
        assert!(!retried.scale_down, "`retry` omits the seventh initialiser");

        let source = CoordPropInfoType {
            data_connect: DataConnect("dc1".to_string()),
            dims_to_propagate: vec![PrimaryDimTypes::X, PrimaryDimTypes::Y],
            scale_down: true,
            ..CoordPropInfoType::default()
        };
        let reverse = CoordPropInfoType {
            data_connect: source.data_connect.clone(),
            ref_is_producer: !source.ref_is_producer,
            ..CoordPropInfoType::default()
        };
        assert_eq!(reverse.data_connect, DataConnect("dc1".to_string()));
        assert!(!reverse.ref_is_producer);
        assert!(!reverse.scale_down, "so the reverse walk is unscaled");
        assert!(reverse.dims_to_propagate.is_empty());
    }

    // ======================================================== the schedule tree (e004, e034, e042)

    use crate::schedule::dims::MetaDimKind;

    /// FNV-1a 64 over one canonical, `'\n'`-terminated line per case, with the same seed and the same
    /// line text on the C++ side. One instance per SECTION rather than one for the whole sweep,
    /// because four of the eleven sections the authority was run under are not expressible here — see
    /// [`e042_the_depth_first_walk_agrees_with_the_executed_authority`].
    struct SectionDigest {
        hash: u64,
        cases: u64,
    }

    impl SectionDigest {
        fn new() -> Self {
            Self {
                hash: 1_469_598_103_934_665_603,
                cases: 0,
            }
        }

        fn feed(&mut self, line: &str) {
            for byte in line.bytes().chain(std::iter::once(b'\n')) {
                self.hash ^= u64::from(byte);
                self.hash = self.hash.wrapping_mul(1_099_511_628_211);
            }
            self.cases += 1;
        }
    }

    /// One node's `relevantComps_`, as the C++ harness spells it: component, then core, then that
    /// core's corelets. An EMPTY corelet list is a real state and the fixture uses one.
    type TreeRel = &'static [(SenComponent, &'static [(u8, &'static [u8])])];

    /// `n->name_ = name` followed by `DesignSpaceConfig::setRelevantComps(n, rel)`, which is
    /// `relevantComps_ = r` on the friend class (`dsc/dsc2.h:519`, `dsc/designSpaceConfig.h:262`).
    fn tree_name_rel(base: &mut ScheduleNode, name: &str, rel: TreeRel) {
        base.name = name.to_owned();
        let comps = base.relevant_comps_mut();
        for (comp, cores) in rel {
            let entry = comps.entry(*comp).or_default();
            for (core, corelets) in *cores {
                entry.insert(
                    CoreId(*core),
                    corelets.iter().copied().map(CoreletId).collect(),
                );
            }
        }
    }

    fn tree_block(name: &str, rel: TreeRel) -> BlockNode {
        let mut node = BlockNode::default();
        tree_name_rel(&mut node.base_class, name, rel);
        node
    }

    /// `Fixture::mkLoop(dims, num, den)` — `new LoopNode()` with `dims_`, `numId_` and `denId_`
    /// written afterwards, which is what [`LoopNode::new`] takes up front.
    fn tree_loop(
        name: &str,
        rel: TreeRel,
        dims: &[(PrimaryDimTypes, MetaDimKind)],
        num: i32,
        den: i32,
    ) -> LoopNode {
        let mut node = LoopNode::new(
            Some(DataStageId(num)),
            Some(DataStageId(den)),
            dims.iter()
                .map(|(dim, kind)| PrimaryDimAndKind::new(*dim, *kind))
                .collect(),
            false,
        );
        tree_name_rel(&mut node.base_class.base_class, name, rel);
        node
    }

    fn tree_transfer(name: &str, rel: TreeRel) -> ChildNode {
        let mut node = TransferNode::default();
        tree_name_rel(&mut node.base_class, name, rel);
        ChildNode::Transfer(node)
    }

    fn tree_compute(name: &str, rel: TreeRel) -> ChildNode {
        let mut node = ComputeNode::default();
        tree_name_rel(&mut node.base_class, name, rel);
        ChildNode::Compute(node)
    }

    fn tree_sync(name: &str, rel: TreeRel) -> ChildNode {
        let mut node = SyncNode::default();
        tree_name_rel(&mut node.base_class, name, rel);
        ChildNode::Sync(node)
    }

    fn tree_allocate(name: &str, rel: TreeRel) -> ChildNode {
        let mut node = AllocateNode::default();
        tree_name_rel(&mut node.base_class, name, rel);
        ChildNode::Allocate(node)
    }

    fn tree_stick_mask(name: &str, rel: TreeRel) -> ChildNode {
        let mut node = StickMaskNode::default();
        tree_name_rel(&mut node.base_class, name, rel);
        ChildNode::StickMask(node)
    }

    /// `addChildNode(node)` with both defaults, i.e. `next_.end()` (`dsc/dsc2.cpp:2013-2029`).
    fn tree_push(parent: &mut BlockNode, child: ChildNode) {
        assert!(
            parent.add_child_node(InsertionPoint::Back, child).is_none(),
            "a Back insertion is always in range"
        );
    }

    /// The C++ harness's one fixture, built through `addChildNode` only:
    ///
    /// ```text
    /// head[den=0]
    ///   L1(LOOP {X,Unpadded})        LX{0:{0,1}} L0{0:{0},1:{0}} PE{0:{0,1}}
    ///     t1(TRANSFER)               LX{0:{0,1}} L0{1:{0}}
    ///     B1(BLOCK)                  LX{0:{0}}   PE{0:{0,1}}
    ///       c1(COMPUTE)              PE{0:{0,1}}
    ///       L2(LOOP {Y,Padded})      LX{1:{1}}   L0{0:{}}
    ///         t2(TRANSFER)           LX{1:{1}}
    ///         s1(SYNC)               L0{0:{}}    <- a core with an EMPTY corelet set
    ///     C1(CONDITION)              LX{0:{0,1}}, coreClCond_ {0:{0}}
    ///       Bt(BLOCK)                LX{0:{0}}
    ///         a1(ALLOCATE)           LX{0:{0}}
    ///       Be(BLOCK)                LX{0:{1}}
    ///         m1(STICKMASK)          LX{0:{1}}
    ///   L3(LOOP {X,PadFront} {Y,Unpadded})  PE{1:{0}}
    ///     c2(COMPUTE)                PE{1:{0}}
    /// ```
    ///
    /// ⛔ BUILT BOTTOM-UP WHERE THE C++ BUILDS TOP-DOWN, and that is forced rather than stylistic:
    /// the authority holds a raw `BlockNode*` to a node it has already handed to a parent, which is
    /// exactly the second owner an owning `Vec<ChildNode>` refuses. The tree it produces is the same
    /// one — every case below prints it.
    fn tree_fixture() -> ScheduleTree {
        let mut l2 = tree_loop(
            "L2",
            &[
                (SenComponent::Lx, &[(1, &[1])]),
                (SenComponent::L0, &[(0, &[])]),
            ],
            &[(PrimaryDimTypes::Y, MetaDimKind::Padded)],
            2,
            1,
        );
        tree_push(
            &mut l2.base_class,
            tree_transfer("t2", &[(SenComponent::Lx, &[(1, &[1])])]),
        );
        tree_push(
            &mut l2.base_class,
            tree_sync("s1", &[(SenComponent::L0, &[(0, &[])])]),
        );

        let mut b1 = tree_block(
            "B1",
            &[
                (SenComponent::Lx, &[(0, &[0])]),
                (SenComponent::Pe, &[(0, &[0, 1])]),
            ],
        );
        tree_push(
            &mut b1,
            tree_compute("c1", &[(SenComponent::Pe, &[(0, &[0, 1])])]),
        );
        tree_push(&mut b1, ChildNode::Loop(l2));

        let mut bt = tree_block("Bt", &[(SenComponent::Lx, &[(0, &[0])])]);
        tree_push(
            &mut bt,
            tree_allocate("a1", &[(SenComponent::Lx, &[(0, &[0])])]),
        );
        let mut be = tree_block("Be", &[(SenComponent::Lx, &[(0, &[1])])]);
        tree_push(
            &mut be,
            tree_stick_mask("m1", &[(SenComponent::Lx, &[(0, &[1])])]),
        );

        let mut c1 = ConditionNode::default();
        tree_name_rel(
            &mut c1.base_mut().base_class,
            "C1",
            &[(SenComponent::Lx, &[(0, &[0, 1])])],
        );
        c1.core_cl_cond = BTreeMap::from([(CoreId(0), BTreeSet::from([CoreletId(0)]))]);
        assert!(
            c1.add_then_region(bt).is_none(),
            "C1 has no then region yet"
        );
        assert!(
            c1.add_else_region(be).is_none(),
            "C1 has exactly one region"
        );

        let mut l1 = tree_loop(
            "L1",
            &[
                (SenComponent::Lx, &[(0, &[0, 1])]),
                (SenComponent::L0, &[(0, &[0]), (1, &[0])]),
                (SenComponent::Pe, &[(0, &[0, 1])]),
            ],
            &[(PrimaryDimTypes::X, MetaDimKind::Unpadded)],
            1,
            0,
        );
        tree_push(
            &mut l1.base_class,
            tree_transfer(
                "t1",
                &[
                    (SenComponent::Lx, &[(0, &[0, 1])]),
                    (SenComponent::L0, &[(1, &[0])]),
                ],
            ),
        );
        tree_push(&mut l1.base_class, ChildNode::Block(b1));
        tree_push(&mut l1.base_class, ChildNode::Condition(c1));

        let mut l3 = tree_loop(
            "L3",
            &[(SenComponent::Pe, &[(1, &[0])])],
            &[
                (PrimaryDimTypes::X, MetaDimKind::PadFront),
                (PrimaryDimTypes::Y, MetaDimKind::Unpadded),
            ],
            3,
            0,
        );
        tree_push(
            &mut l3.base_class,
            tree_compute("c2", &[(SenComponent::Pe, &[(1, &[0])])]),
        );

        let mut tree = ScheduleTree::default();
        tree_name_rel(
            &mut tree.head_mut().base_class.base_class,
            "head",
            &[
                (SenComponent::Lx, &[(0, &[0, 1])]),
                (SenComponent::L0, &[(0, &[0]), (1, &[0])]),
                (SenComponent::Pe, &[(0, &[0, 1]), (1, &[0])]),
            ],
        );
        tree_push(&mut tree.head_mut().base_class, ChildNode::Loop(l1));
        tree_push(&mut tree.head_mut().base_class, ChildNode::Loop(l3));
        tree
    }

    /// The fixture's `by_name` map, as a search rather than a table of raw pointers.
    fn tree_find<'tree>(tree: &'tree ScheduleTree, name: &str) -> &'tree ChildNode {
        fn search<'a>(children: &'a [ChildNode], name: &str) -> Option<&'a ChildNode> {
            for child in children {
                if child.base().name == name {
                    return Some(child);
                }
                if let Some(found) = child
                    .as_block()
                    .and_then(|block| search(block.children(), name))
                {
                    return Some(found);
                }
            }
            None
        }
        search(tree.head().base_class.children(), name).expect("the fixture names every node")
    }

    /// ⛔ `ChildNode::as_block_mut` STOPS AT A CONDITION NODE where [`ChildNode::as_block`] does not,
    /// because the condition node's base is private for the reason [`ConditionNode`] records — so this
    /// walk names the three block-bearing variants itself. `head` is not a `ChildNode` at all: it is
    /// the tree's own `LoopNode` member.
    fn tree_find_block_mut<'tree>(
        tree: &'tree mut ScheduleTree,
        name: &str,
    ) -> &'tree mut BlockNode {
        fn block_of(child: &mut ChildNode) -> Option<&mut BlockNode> {
            match child {
                ChildNode::Block(node) => Some(node),
                ChildNode::Loop(node) => Some(&mut node.base_class),
                ChildNode::Condition(node) => Some(node.base_mut()),
                _ => None,
            }
        }
        fn search<'a>(children: &'a mut [ChildNode], name: &str) -> Option<&'a mut BlockNode> {
            for child in children {
                let matched = child.base().name == name;
                let Some(block) = block_of(child) else {
                    continue;
                };
                if matched {
                    return Some(block);
                }
                if let Some(found) = search(block.next.as_mut_slice(), name) {
                    return Some(found);
                }
            }
            None
        }
        if name == "head" {
            return &mut tree.head_mut().base_class;
        }
        search(tree.head_mut().base_class.next.as_mut_slice(), name)
            .expect("the fixture names every block node")
    }

    fn tree_find_condition_mut<'tree>(
        tree: &'tree mut ScheduleTree,
        name: &str,
    ) -> &'tree mut ConditionNode {
        fn search<'a>(children: &'a mut [ChildNode], name: &str) -> Option<&'a mut ConditionNode> {
            for child in children {
                if child.base().name == name {
                    return child.as_condition_mut();
                }
                let Some(kids) = child.children_mut() else {
                    continue;
                };
                if let Some(found) = search(kids.as_mut_slice(), name) {
                    return Some(found);
                }
            }
            None
        }
        search(tree.head_mut().base_class.next.as_mut_slice(), name)
            .expect("the fixture names one condition node")
    }

    /// `joinNames` — the visited names, comma-separated, `-` when nothing was visited.
    fn tree_join(nodes: &[&ChildNode]) -> String {
        if nodes.is_empty() {
            return "-".to_owned();
        }
        nodes
            .iter()
            .map(|node| node.base().name.as_str())
            .collect::<Vec<_>>()
            .join(",")
    }

    /// `shapeSOf` — the whole tree as ` parent>child[kind]` pairs in child order, `-` when empty.
    /// ⛔ NO `prev=` COLUMN, which is the reason the sections are digested separately: the C++
    /// harness's whole-tree print carries the parent pointer and this type has none.
    fn tree_shape(tree: &ScheduleTree) -> String {
        fn walk(parent: &BlockNode, out: &mut String) {
            for child in parent.children() {
                out.push_str(&format!(
                    " {}>{}[{}]",
                    parent.base_class.name,
                    child.base().name,
                    child.node_type().name()
                ));
                if let Some(block) = child.as_block() {
                    walk(block, out);
                }
            }
        }
        let mut out = String::new();
        walk(&tree.head().base_class, &mut out);
        if out.is_empty() { " -".to_owned() } else { out }
    }

    /// `coreClStr` — `<core>:{<corelets>}` per core, `{}` when the map is empty.
    fn tree_core_cl(core_cl: &BTreeMap<CoreId, BTreeSet<CoreletId>>) -> String {
        if core_cl.is_empty() {
            return "{}".to_owned();
        }
        let mut out = String::new();
        for (core, corelets) in core_cl {
            out.push_str(&format!("{}:{{", core.0));
            out.push_str(
                &corelets
                    .iter()
                    .map(|corelet| corelet.0.to_string())
                    .collect::<Vec<_>>()
                    .join(","),
            );
            out.push('}');
        }
        out
    }

    /// `getThenCoreCl`/`getElseCoreCl` are `next_.at(0)`/`.at(1)`, so the absent branch is
    /// `std::out_of_range` there and [`Option::None`] here.
    fn tree_opt_core_cl(core_cl: Option<BTreeMap<CoreId, BTreeSet<CoreletId>>>) -> String {
        match core_cl {
            Some(core_cl) => tree_core_cl(&core_cl),
            None => "OUT_OF_RANGE".to_owned(),
        }
    }

    /// `getThenBranchNode`/`getElseBranchNode` answer `nullptr` for the very state the two above
    /// throw on (`dsc/dsc2.h:707-718`), which `nameOf` prints as `-`.
    fn tree_branch_name(branch: Option<&ChildNode>) -> String {
        match branch {
            Some(node) => node.base().name.clone(),
            None => "-".to_owned(),
        }
    }

    /// The `cond` section's seed: a fresh condition node with `existing` BLOCK children already in
    /// place, inserted through the override itself because that is what the C++ harness's
    /// `static_cast<BlockNode&>(c).addChildNode(r)` reaches — `addChildNode` IS virtual
    /// (`dsc/dsc2.h:535`, overridden `:697`), so the cast changes the overload set and not the body.
    fn tree_condition(existing: usize) -> ConditionNode {
        const REGION_RELS: [TreeRel; 2] = [
            &[(SenComponent::Lx, &[(0, &[0])])],
            &[
                (SenComponent::Lx, &[(0, &[1])]),
                (SenComponent::Pe, &[(1, &[0, 1])]),
            ],
        ];
        let mut cond = ConditionNode::default();
        cond.base_mut().base_class.name = "C".to_owned();
        cond.core_cl_cond = BTreeMap::from([(CoreId(0), BTreeSet::from([CoreletId(0)]))]);
        for (index, rel) in REGION_RELS.into_iter().take(existing).enumerate() {
            let mut region = BlockNode::default();
            tree_name_rel(&mut region.base_class, &format!("R{index}"), rel);
            assert!(
                cond.add_child_node(InsertionPoint::Back, region).is_none(),
                "the seed never exceeds two regions"
            );
        }
        cond
    }

    /// One loop-guarded condition, which is all `getNextView`'s second dispatch reads: the C++ side
    /// pushes one clause into `loopCond_.twoLevelOrOfAnds_` and the printed view does not name it.
    fn tree_loop_guard() -> LoopCondComposite {
        LoopCondComposite::from(LoopCondDisjunction::new(LoopCondConjunction::new(
            LoopCond {
                dim: PrimaryDimTypes::X,
                cond_op: LoopCondOp::Eq,
                cond_val: CondVal::First,
            },
        )))
    }

    /// The four type sets and six components every walk below is swept over.
    const TREE_TYPE_SETS: [(&str, &[NodeType]); 4] = [
        ("any", &[]),
        ("loop", &[NodeType::Loop]),
        (
            "blocky",
            &[NodeType::Block, NodeType::Loop, NodeType::Condition],
        ),
        (
            "leaf",
            &[
                NodeType::Transfer,
                NodeType::Compute,
                NodeType::Sync,
                NodeType::Allocate,
                NodeType::StickMask,
            ],
        ),
    ];
    const TREE_COMPS: [(&str, SenComponent); 6] = [
        ("ALL", SenComponent::All),
        ("LX", SenComponent::Lx),
        ("L0", SenComponent::L0),
        ("PE", SenComponent::Pe),
        ("HBM", SenComponent::Hbm),
        ("NONE", SenComponent::NoComponent),
    ];
    /// The six block nodes any child can be added to, plus `Bt` and `Be` for the readings. `C1` is a
    /// parent in the C++ harness too, but only through a `BlockNode*` — see
    /// [`e004_a_child_insertion_lands_where_the_authority_puts_it`].
    const TREE_PARENTS: [&str; 6] = ["head", "L1", "B1", "L2", "Bt", "L3"];

    /// `dsc/dsc2.cpp:2222-2255` — `ScheduleTree::traverseTreeDFS`, executed against this port over
    /// 1,392 cases: four node-type sets × six components × sixteen start nodes × the exclude list,
    /// reference digest 12546984765533884190.
    ///
    /// ⭐ `clId`, `coreId` AND `maxLoopDepth` ARE ALL -1 IN EVERY CASE, AND THAT IS THE WHOLE TREE'S
    /// CALL GRAPH, not a narrowing of the sweep. No in-scope caller passes a core or a corelet —
    /// `dsc/dsc2Pcfg.cpp` is the only one that does and DCG/PCFG is off our path — and
    /// `maxLoopDepth` has NO caller at all tree-wide. That matters because `maxLoopDepth` is the only
    /// consumer of the `loopDepth` stack (`:2233`, `:2246-2250`), and that stack is the only reader of
    /// `prev_` inside this body (`:2239`): with it dead the traversal is a plain pre-order walk over
    /// an owning tree, which is what [`ChildNode::visit_dfs`] is.
    ///
    /// ⛔ AND `prev_` IS REDUNDANT ON A WELL-FORMED TREE ANYWAY: a C++-only `prevparent` section
    /// compares every node's `prev_` against the parent found by walking the child lists, and they
    /// agree 15/15 on this fixture. The disagreement it was written to catch is real but unreachable
    /// from here — `insertLoopAbove` leaves BOTH links stale (11/11 cases) and has zero callers.
    ///
    /// ⚠️ THE START NODE IS NOT A `ScheduleNode*` HERE. The authority takes one and branches on
    /// `startNode->prev_ == nullptr` (`:2228`), so `nullptr` and the head are the SAME case — which
    /// is why `s-` and `shead` are the tree-wide walk in both columns. Every other start is a
    /// subtree, i.e. [`ChildNode::traverse_dfs`], and only those carry an exclude list: the one live
    /// caller passes the loop it is scanning (`ddc/ddc_transformation.cpp:1525-1527`).
    #[test]
    fn e042_the_depth_first_walk_agrees_with_the_executed_authority() {
        const STARTS: [&str; 16] = [
            "-", "head", "L1", "t1", "B1", "c1", "L2", "t2", "s1", "C1", "Bt", "a1", "Be", "m1",
            "L3", "c2",
        ];

        let tree = tree_fixture();
        let mut digest = SectionDigest::new();
        for (set_name, node_types) in TREE_TYPE_SETS {
            for (comp_name, comp) in TREE_COMPS {
                for start in STARTS {
                    let whole_tree = start == "-" || start == "head";
                    let excludes: &[&str] = if whole_tree {
                        &["-"]
                    } else {
                        &["-", "B1", "L2", "C1"]
                    };
                    for exclude in excludes {
                        let order = if whole_tree {
                            tree.traverse_dfs(node_types, comp)
                        } else if *exclude == "-" {
                            tree_find(&tree, start).traverse_dfs(node_types, comp)
                        } else {
                            tree_find(&tree, start).traverse_dfs_excluding(
                                node_types,
                                comp,
                                &[tree_find(&tree, exclude)],
                            )
                        };
                        digest.feed(&format!(
                            "dfs {set_name} {comp_name} s{start} x{exclude} = {}",
                            tree_join(&order)
                        ));
                    }
                }
            }
        }
        assert_eq!(
            (digest.cases, digest.hash),
            (1_392, 12_546_984_765_533_884_190),
            "executed dsc/dsc2.cpp:2222-2255"
        );
    }

    /// `dsc/dsc2.cpp:1984-2002` — both `getNextView` bodies, executed over 32 cases: four components
    /// × eight block nodes, reference digest 1180193418606023406.
    ///
    /// ⛔ THE TWO BODIES ARE SELECTED BY THE HANDLE'S STATIC TYPE, NOT BY THE NODE. `getNextView` is
    /// NOT virtual (`dsc/dsc2.h:541`, `:701`), so the `cond_static` column — the same condition node
    /// reached as a `ConditionNode` rather than as its base — is a DIFFERENT function from the
    /// `nextview C1` column beside it, and `cond_loopguard` is that function on a node whose
    /// `loopCond_` is set, where it widens the view to `ALL, -1, -1` (`:1996-2001`). Rust method
    /// resolution reproduces the split exactly, which is why [`ConditionNode::next_view`] shadows
    /// [`BlockNode::next_view`] instead of overriding it.
    #[test]
    fn e034_the_child_view_keeps_the_conditions_own_static_dispatch() {
        const BLOCKS: [&str; 8] = ["head", "L1", "B1", "L2", "C1", "Bt", "Be", "L3"];

        let mut digest = SectionDigest::new();
        for (comp_name, comp) in &TREE_COMPS[..4] {
            for block in BLOCKS {
                let mut tree = tree_fixture();
                let mut line = format!(
                    "nextview {block} {comp_name} = {}",
                    tree_join(&tree_find_block_mut(&mut tree, block).next_view(*comp))
                );
                if block == "C1" {
                    let cond = tree_find_condition_mut(&mut tree, block);
                    line += &format!(" | cond_static = {}", tree_join(&cond.next_view(*comp)));
                    cond.loop_cond = Some(tree_loop_guard());
                    line += &format!(" | cond_loopguard = {}", tree_join(&cond.next_view(*comp)));
                }
                digest.feed(&line);
            }
        }
        assert_eq!(
            (digest.cases, digest.hash),
            (32, 1_180_193_418_606_023_406),
            "executed dsc/dsc2.cpp:1984-2002"
        );
    }

    /// `dsc/dsc2.cpp:2013-2029` — `BlockNode::addChildNode`, executed over 44 cases: six parents ×
    /// `addBefore` × the four sibling choices that exist on each, reference digest
    /// 15424645884529233517.
    ///
    /// ⭐ THE FOUR INSERTION POINTS ARE THE WHOLE FUNCTION. A null sibling with `addBefore` is
    /// `begin()` and without it `end()`; a sibling resolves to the iterator found by scanning `next_`
    /// and then to it or one past it (`:2017-2027`) — which is [`InsertionPoint`]'s four variants.
    ///
    /// ⛔ AND THE AUTHORITY'S OWN SIBLING SCAN IS UNDEFINED BEHAVIOUR, WHICH IS WHY NO CASE HERE
    /// REFUSES: `for (; it != next_.end() && it->get() != siblingRefNode; it++)` is not what it
    /// writes — the dereference happens BEFORE the end test, so a sibling that is not a child of this
    /// node runs off the vector instead of reaching the `DT_ERROR` one line below. Compiled and run
    /// under ASAN over the extracted body that is a heap-buffer-overflow read followed by a member
    /// call on the null it loads, never the diagnostic. An out-of-range index cannot be built here,
    /// and [`BlockNode::add_child_node`] hands the node back for one that is.
    #[test]
    fn e004_a_child_insertion_lands_where_the_authority_puts_it() {
        let mut digest = SectionDigest::new();
        for parent in TREE_PARENTS {
            for add_before in [0, 1] {
                for sibling in ["-", "SELF0", "SELF1", "SELFLAST"] {
                    let mut tree = tree_fixture();
                    let count = tree_find_block_mut(&mut tree, parent).children().len();
                    let index = match sibling {
                        "SELF0" => (count > 0).then_some(0),
                        "SELF1" => (count > 1).then_some(1),
                        "SELFLAST" => (count > 0).then_some(count - 1),
                        _ => None,
                    };
                    if sibling != "-" && index.is_none() {
                        continue;
                    }
                    let (at, reference) = match index {
                        None if add_before == 1 => (InsertionPoint::Front, "-".to_owned()),
                        None => (InsertionPoint::Back, "-".to_owned()),
                        Some(index) => {
                            let name = tree_find_block_mut(&mut tree, parent).children()[index]
                                .base()
                                .name
                                .clone();
                            let at = if add_before == 1 {
                                InsertionPoint::Before(index)
                            } else {
                                InsertionPoint::After(index)
                            };
                            (at, name)
                        }
                    };
                    let refused = tree_find_block_mut(&mut tree, parent)
                        .add_child_node(at, tree_compute("NEW", &[]))
                        .is_some();
                    let line = format!(
                        "addchild {parent} before{add_before} sib{sibling} ref{reference} ="
                    );
                    digest.feed(&if refused {
                        format!("{line} REFUSED")
                    } else {
                        format!("{line}{}", tree_shape(&tree))
                    });
                }
            }
        }
        assert_eq!(
            (digest.cases, digest.hash),
            (44, 15_424_645_884_529_233_517),
            "executed dsc/dsc2.cpp:2013-2029"
        );
    }

    /// `dsc/dsc2.cpp:2041-2051` and `:2053-2056` — `insertPerfectlyNestedBlockNode` and
    /// `moveChildren`, executed over 48 cases: six parents × the two node kinds a caller passes × a
    /// pre-filled node × both routes, reference digest 581752260481533989 with 12 refusals.
    ///
    /// ⭐ THE TWO OPERATIONS SHARE A BODY AND DIFFER ONLY IN THE GUARD. `moveChildren` is
    /// `toNode->next_ = std::move(next_)` with no test at all, so the 12 refusals are exactly the
    /// `prefill mc0` cases: nesting refuses a node that already has children (`:2043-2045`) and
    /// moving OVERWRITES them — `nest.kids` on a `prefill1 mc1` case is the parent's count, never the
    /// filler's, and the filler is gone.
    ///
    /// ⛔ AND A MOVE DOES NOT PUT THE DESTINATION IN THE TREE, which the shape column shows: after
    /// `mc1` the parent has no children and the node holding them is not reachable from the head. The
    /// authority leaks it unless the caller inserts it; here it is a local that drops.
    ///
    /// ⚠️ TWO NESTING METHODS, NOT A WIDER PARAMETER: the authority takes `BlockNode*` and its ONE
    /// caller passes a `LoopNode*` (`ddc/ddc_transformation_util.cpp:287-303`, the call at `:301`),
    /// so `kind1 mc0` is the live case and `kind0 mc0` has no caller in the authority at all. See
    /// [`BlockNode::insert_perfectly_nested_loop_node`].
    ///
    /// ⛔ NEITHER IS REACHABLE ON A CONDITION NODE HERE, where both succeed in C++ on one — a
    /// C++-only `nestcond` section records that, and it puts a third and fourth region into a node
    /// whose own contract caps it at two. All three live call sites hold a `LoopNode`.
    #[test]
    fn e004_nesting_and_moving_children_agree_with_the_executed_authority() {
        const NEST_DIMS: [(PrimaryDimTypes, MetaDimKind); 1] =
            [(PrimaryDimTypes::Mb, MetaDimKind::Unpadded)];

        let mut digest = SectionDigest::new();
        for parent in TREE_PARENTS {
            for kind in [0, 1] {
                for prefill in [0, 1] {
                    for via_move in [0, 1] {
                        let mut tree = tree_fixture();
                        let mut moved_kids = None;
                        let refused = match (kind, via_move) {
                            (0, 0) => {
                                let mut nest = tree_block("NEST", &[]);
                                if prefill == 1 {
                                    tree_push(&mut nest, tree_sync("FILLER", &[]));
                                }
                                tree_find_block_mut(&mut tree, parent)
                                    .insert_perfectly_nested_block_node(nest)
                                    .is_some()
                            }
                            (_, 0) => {
                                let mut nest = tree_loop("NEST", &[], &NEST_DIMS, 7, 7);
                                if prefill == 1 {
                                    tree_push(&mut nest.base_class, tree_sync("FILLER", &[]));
                                }
                                tree_find_block_mut(&mut tree, parent)
                                    .insert_perfectly_nested_loop_node(nest)
                                    .is_some()
                            }
                            (0, _) => {
                                let mut nest = tree_block("NEST", &[]);
                                if prefill == 1 {
                                    tree_push(&mut nest, tree_sync("FILLER", &[]));
                                }
                                tree_find_block_mut(&mut tree, parent).move_children_to(&mut nest);
                                moved_kids = Some(nest.children().len());
                                false
                            }
                            (_, _) => {
                                let mut nest = tree_loop("NEST", &[], &NEST_DIMS, 7, 7);
                                if prefill == 1 {
                                    tree_push(&mut nest.base_class, tree_sync("FILLER", &[]));
                                }
                                tree_find_block_mut(&mut tree, parent)
                                    .move_children_to(&mut nest.base_class);
                                moved_kids = Some(nest.base_class.children().len());
                                false
                            }
                        };
                        let mut line =
                            format!("nest {parent} kind{kind} prefill{prefill} mc{via_move} =");
                        if refused {
                            line += " REFUSED";
                        } else {
                            line += &tree_shape(&tree);
                            let parent_kids =
                                tree_find_block_mut(&mut tree, parent).children().len();
                            let nest_kids = match moved_kids {
                                Some(kids) => kids,
                                None => tree_find_block_mut(&mut tree, "NEST").children().len(),
                            };
                            line += &format!(" | parent.kids={parent_kids} nest.kids={nest_kids}");
                        }
                        digest.feed(&line);
                    }
                }
            }
        }
        assert_eq!(
            (digest.cases, digest.hash),
            (48, 581_752_260_481_533_989),
            "executed dsc/dsc2.cpp:2041-2051 and :2053-2056"
        );
    }

    /// `dsc/dsc2.cpp:2143-2167` and `:2003-2011` — `ConditionNode::addChildNode`, `addThenRegion`,
    /// `addElseRegion` and the two branch readings, executed over 9 cases against conditions seeded
    /// with zero, one and two regions, reference digest 6850261760959776999 with 5 refusals.
    ///
    /// ⭐ THE THREE GUARDS ARE DIFFERENT PREDICATES, and that is what the 5 refusals separate:
    /// `addChildNode` refuses only the THIRD child (`:2145-2147`), `addThenRegion` refuses a
    /// non-empty `next_` (`:2152-2155`) and `addElseRegion` refuses anything but exactly one
    /// (`:2160-2163`) — so `h1 existing0` refuses while `h0 existing0` does not, and both refuse at
    /// two. All three also refuse a non-BLOCK child, which is an `E0308` here rather than a case:
    /// [`ConditionNode::add_child_node`] takes a [`BlockNode`].
    ///
    /// ⛔ THE TWO READING PAIRS DISAGREE ABOUT THE SAME STATE. `getThenCoreCl`/`getElseCoreCl` are
    /// `next_.at(0)`/`.at(1)` and throw where the region is absent, while
    /// `getThenBranchNode`/`getElseBranchNode` answer `nullptr` (`dsc/dsc2.h:707-718`) — the
    /// `OUT_OF_RANGE` and `-` columns on the same line. Bridge 1 has one UNGUARDED caller of the
    /// throwing pair (`DSC2ToDataflowIR/DSC2ToDataflowIR.cpp:146`) against a guarded one
    /// (`SNControlFlowLowering.cpp:1116`), so both are `Option` here.
    #[test]
    fn e034_the_condition_nodes_two_region_contract_agrees_with_the_executed_authority() {
        const NEW_REL: TreeRel = &[(SenComponent::Pe, &[(0, &[0])])];
        const REG_REL: TreeRel = &[
            (SenComponent::Pe, &[(0, &[0])]),
            (SenComponent::Lx, &[(1, &[1])]),
        ];

        let mut digest = SectionDigest::new();
        for existing in [0, 1, 2] {
            let mut cond = tree_condition(existing);
            let mut line = format!("condadd existing{existing} = ");
            match cond.add_child_node(InsertionPoint::Back, tree_block("NEW", NEW_REL)) {
                Some(_) => line += "REFUSED",
                None => line += &format!("kids={}", cond.base().children().len()),
            }
            line += &format!(
                " then={} else={} thenb={} elseb={}",
                tree_opt_core_cl(cond.then_core_cl(SenComponent::All)),
                tree_opt_core_cl(cond.else_core_cl(SenComponent::All)),
                tree_branch_name(cond.then_branch()),
                tree_branch_name(cond.else_branch()),
            );
            digest.feed(&line);

            for helper in [0, 1] {
                let mut cond = tree_condition(existing);
                let region = tree_block("REG", REG_REL);
                let mut line = format!("condregion h{helper} existing{existing} = ");
                let handed_back = if helper == 1 {
                    cond.add_else_region(region)
                } else {
                    cond.add_then_region(region)
                };
                match handed_back {
                    Some(_) => line += "REFUSED",
                    None => line += &format!("kids={}", cond.base().children().len()),
                }
                line += &format!(
                    " then={} else={} thenb={} elseb={}",
                    tree_opt_core_cl(cond.then_core_cl(SenComponent::All)),
                    tree_opt_core_cl(cond.else_core_cl(SenComponent::All)),
                    tree_branch_name(cond.then_branch()),
                    tree_branch_name(cond.else_branch()),
                );
                digest.feed(&line);
            }
        }
        assert_eq!(
            (digest.cases, digest.hash),
            (9, 6_850_261_760_959_776_999),
            "executed dsc/dsc2.cpp:2143-2167 and :2003-2011"
        );
    }

    /// `dsc/dsc2.h:621-652` — `ScheduleTree`'s default constructor, `clear` and `empty`, executed
    /// over 3 cases, reference digest 7012364341469060330.
    ///
    /// ⭐ THE DEFAULT CONSTRUCTOR'S ONE STATEMENT IS `head_.denId_ = 0` (`:625`), the core datastage,
    /// and it is the only field of the head that is not a `LoopNode` default — which is what
    /// `head.den=0` beside `head.num=-1` pins.
    ///
    /// ⛔ AND THE COPY CONSTRUCTOR LOSES IT, which is why this type has no [`Clone`] AND why the
    /// "harmless" case is not harmless. `ScheduleTree(const ScheduleTree& old) { copyFrom(old); }`
    /// (`:628`) never runs that statement, and `copyFrom` `DT_ERROR`s on any non-empty tree
    /// (`dsc/dsc2.cpp:2287-2291`) — its deep-copy body is commented out with a `TODO` above it. A
    /// C++-only `treecopy` section records both halves: copying an EMPTY tree yields
    /// `copy.head.den=-1` against the original's `0`, and copying the fixture refuses.
    #[test]
    fn e042_a_fresh_tree_is_one_loop_head_holding_the_core_datastage() {
        let mut digest = SectionDigest::new();
        let fresh = ScheduleTree::default();
        digest.feed(&format!(
            "tree fresh empty={} head.den={} head.num={} head.kind={} head.dims={} head.parametric={} head.ldsidx={} shape={}",
            u8::from(fresh.is_empty()),
            fresh.head().den_id.map_or(-1, |id| id.0),
            fresh.head().num_id.map_or(-1, |id| id.0),
            fresh.head().base_class.base_class.node_type().name(),
            fresh.head().dims.len(),
            u8::from(fresh.head().is_parametric_loop()),
            fresh.head().parametric_lds_idx().map_or(-1, |idx| idx.0),
            tree_shape(&fresh),
        ));

        let mut tree = tree_fixture();
        digest.feed(&format!(
            "tree built empty={} shape={}",
            u8::from(tree.is_empty()),
            tree_shape(&tree)
        ));
        tree.clear();
        digest.feed(&format!(
            "tree cleared empty={} shape={}",
            u8::from(tree.is_empty()),
            tree_shape(&tree)
        ));

        assert_eq!(
            (digest.cases, digest.hash),
            (3, 7_012_364_341_469_060_330),
            "executed dsc/dsc2.h:621-652"
        );
    }

    /// `util/utils.h:100-107` over `dsc/dsc2.h:529-537` — `clone()` on a loop, a block and a
    /// condition node, executed over 3 cases, reference digest 3999731503567925404.
    ///
    /// ⛔ A CLONED BLOCK NODE HAS NO CHILDREN AND THAT IS DELIBERATE: `kids=0` against `origkids=3`
    /// is `VectorOfChildren(const VectorOfChildren&) {}`, "do nothing on purpose", with the
    /// authority's own note that the caller inserts the copies (`:533-536`). Both clone sites in DDC
    /// depend on it — `ddc/ddc_transformation.cpp:984-986` clones a loop and then adds one cloned
    /// child, and `ddc/ddc_transformation_util.cpp:580-588` clones a condition node and then calls
    /// `addThenRegion`, which refuses outright on a non-empty `next_`. A deep [`Clone`] would turn a
    /// working path into a fatal error.
    ///
    /// ⭐ EVERYTHING ELSE SURVIVES, which is what the remaining columns are for: the name, both
    /// datastage ids, the dim list, the whole `relevantComps_` map, the `const` kind, the loop's two
    /// private parametric fields and the condition's `coreClCond_`.
    #[test]
    fn e034_a_cloned_node_keeps_every_field_but_its_children() {
        let mut digest = SectionDigest::new();
        let tree = tree_fixture();

        let original_loop = tree_find(&tree, "L1").as_loop().expect("L1 is a loop");
        let cloned_loop = original_loop.clone();
        digest.feed(&format!(
            "clone loop kids={} origkids={} name={} num={} den={} dims={} rel={} kind={} parametric={} ldsidx={}",
            cloned_loop.base_class.children().len(),
            original_loop.base_class.children().len(),
            cloned_loop.base_class.base_class.name,
            cloned_loop.num_id.map_or(-1, |id| id.0),
            cloned_loop.den_id.map_or(-1, |id| id.0),
            cloned_loop.dims.len(),
            cloned_loop.base_class.base_class.relevant_comps().len(),
            cloned_loop.base_class.base_class.node_type().name(),
            u8::from(cloned_loop.is_parametric_loop()),
            cloned_loop.parametric_lds_idx().map_or(-1, |idx| idx.0),
        ));

        let original_block = tree_find(&tree, "B1").as_block().expect("B1 is a block");
        let cloned_block = original_block.clone();
        digest.feed(&format!(
            "clone block kids={} origkids={} name={} rel={} kind={}",
            cloned_block.children().len(),
            original_block.children().len(),
            cloned_block.base_class.name,
            cloned_block.base_class.relevant_comps().len(),
            cloned_block.base_class.node_type().name(),
        ));

        let original_cond = tree_find(&tree, "C1")
            .as_condition()
            .expect("C1 is a condition");
        let cloned_cond = original_cond.clone();
        digest.feed(&format!(
            "clone cond kids={} origkids={} name={} corecl={} hascorecl={} rel={} kind={}",
            cloned_cond.base().children().len(),
            original_cond.base().children().len(),
            cloned_cond.base().base_class.name,
            tree_core_cl(&cloned_cond.core_cl_cond),
            u8::from(cloned_cond.has_core_cl_cond()),
            cloned_cond.base().base_class.relevant_comps().len(),
            cloned_cond.base().base_class.node_type().name(),
        ));

        assert_eq!(
            (digest.cases, digest.hash),
            (3, 3_999_731_503_567_925_404),
            "executed util/utils.h:100-107 over dsc/dsc2.h:529-537"
        );
    }
}

// ⛔ THE TWO BLOCKNODE AND TWO LOOPNODE TYPE ANCHORS STAY OPEN ON e027_DesignSpaceConfig, AND NINE OF
// THEIR ELEVEN REMAINING FIELD ANCHORS NAME NO FIELD. Both classes are ported above with their base
// subobject and their child list, and both keep an open type anchor for the methods that are still
// unreachable:
//  * `BlockNode::deleteChildNode`'s DESTRUCTIVE arm and `moveChildNode`, which forwards to it, both
//    call `ownerDsc->cleanupAllocation(nodeToDelete)` (`dsc/dsc2.cpp:2031-2039`, `:2188-2190`) —
//    `DesignSpaceConfig::cleanupAllocation` (`dsc/designSpaceConfig.h:262`) is unported. The
//    NON-destructive arm is [`BlockNode::take_child_node`] and has landed.
//  * `LoopNode::parametricIterCount` (`dsc/dsc2.cpp:4126`) and `parametricStride` (`:4197`) read
//    `DesignSpaceConfig::dataStageParam_` and `labeledDs_` and climb `getOwnerLoop()`.
//  * `LoopNode::print` (`:4284`) prints `this`, a raw address (`:4288`) — as all three node `print`s
//    do (`:4387`, `:4445`, `:4517`).
//
// ⚠️ AND THE FIELD ANCHORS THAT REMAIN ARE `friend class` LINES AND PARAMETER DEFAULTS, so filling
// them is impossible rather than pending. `.Ddc`, `.DesignSpaceConfig`, `.L3DlOpsScheduler`,
// `.ScheduleNode` and `.ScheduleTree` are `BlockNode`'s five `friend class` declarations
// (`dsc/dsc2.h:556-560`) and four of them are `LoopNode`'s (`:611-614`); `.coreId` is
// `BlockNode::getNextView`'s third parameter, `int coreId = -1) const;` (`:543`), and `.rowId` is
// `parametricIterCount`'s fourth, `int rowId = -1) const;` (`:602`). That is 12 of the 43 anchors the
// e042_LoopDistributionInfo note below tallies as naming no field.
//
// ⭐ WHAT IS FILLED IS `.BaseClass` AND `.next_` UNDER BOTH BLOCKNODE IDS, AND `.head_` UNDER BOTH
// SCHEDULETREE IDS — `e032_ScheduleTree` and `e042_ScheduleTree` are closed entirely, type anchors
// included, because every one of `ScheduleTree`'s methods landed. ⚠️ `e042` NAMES TWO ENTITIES IN THIS
// CAMPAIGN, `e042_LoopDistributionInfo` and `e042_ScheduleTree`, and `e034` names two,
// `e034_TransferNode` and `e034_LoopNode`; only the name disambiguates them.

// crustify:todo: e030_BlockNode

// crustify:todo: e030_BlockNode.Ddc

// crustify:todo: e030_BlockNode.DesignSpaceConfig

// crustify:todo: e030_BlockNode.L3DlOpsScheduler

// crustify:todo: e030_BlockNode.ScheduleNode

// crustify:todo: e030_BlockNode.ScheduleTree

// crustify:todo: e030_BlockNode.coreId

// crustify:todo: e031_LoopNode

// crustify:todo: e031_LoopNode.Ddc

// crustify:todo: e031_LoopNode.DesignSpaceConfig

// crustify:todo: e031_LoopNode.ScheduleNode

// crustify:todo: e031_LoopNode.ScheduleTree

// crustify:todo: e031_LoopNode.rowId

// crustify:todo: e033_DataInfo.bufferSwitchPosition_

// crustify:todo: e033_DataInfo.loopEleOffsets_

// crustify:todo: e033_DataInfo.startAddr_

// crustify:todo: e037_AllocateNode

// crustify:todo: e037_AllocateNode.allocUsers_

// crustify:todo: e037_AllocateNode.tempStorageForCompute_

// ⛔ THREE `CoordPropInfoType` ANCHORS STAY OPEN IN EVERY GENERATION, ON SCHEDULE-NODE IDENTITY:
// `refNode` and `nodeToFold` (`dsc/dsc2.h:1089-1090`) are `ScheduleNode*` used as `refsAdded_` keys
// and `static_cast` to the concrete node to clear its coordinates (`ddc/ddc.h:410-418`, `:487-503`),
// and each generation's TYPE anchor stays with them because `print` (`:1098-1107`) dereferences both.
// The other five fields are carried above, and `e016_CoordPropInfoType`'s type anchor stays open for
// the same reason.

// crustify:todo: e038_CoordPropInfoType

// crustify:todo: e038_CoordPropInfoType.nodeToFold

// crustify:todo: e038_CoordPropInfoType.refNode

// crustify:todo: e039_LoopCond

// crustify:todo: e039_LoopCond.loopComp_

// ⛔ e041_DistributionStatusInfo IS A DEAD DECLARATION: the four anchors below stay open because
// there is nothing to port, not because the work is pending. `DistributionStatusInfo`,
// `DistributionStatusType`, `NEED_LOOP_SPLIT`, `loopToSplit`, `loopSplitDim` and `loopSplitDimSizes`
// occur in the authority tree — any file type — ONLY in their own declaration
// (`dsc/dsc2.h:1129-1135`); the CodeQL oracle lists nothing behind them but compiler-generated
// members.
//
// ⛔ AND ITS NEIGHBOUR IS NOT EVIDENCE FOR THAT, EITHER WAY. It sits under a block-commented
// `loopDistributionParamInfo` extern (`dsc/dsc2.h:1124-1127`), but that declaration was RELOCATED,
// not abandoned: the map is a live `Ddc` member (`ddc/ddc.h:553`) read throughout
// `ddc/ddc_fold.cpp` and `ddc/ddcv1.cpp`. This type's deadness rests on its own symbol census.

// crustify:todo: e041_DistributionStatusInfo

// crustify:todo: e041_DistributionStatusInfo.loopSplitDim

// crustify:todo: e041_DistributionStatusInfo.loopSplitDimSizes

// crustify:todo: e041_DistributionStatusInfo.loopToSplit

// ⛔ e042_LoopDistributionInfo IS BLOCKED ON SCHEDULE-NODE IDENTITY, NOT ON ITS VALUE HALF: `cat` is
// already ported as `LoopDistributionCat` above. What stays open is `loopNode`, a `LoopNode*` held as
// pointer identity — and ALL EIGHT writers supply one, so there is no single producer to port it
// behind. Two climb an owner chain (`dsc/dsc2.cpp:6593-6605`, and its CORELET_SLICE push at
// `:6620-6622`), two are in the fold (`ddc/ddc_fold.cpp:3513-3517`, `:3549-3551`), and four are in
// the L3 scheduler — including two in `findAndStoreLoopWithDim`, which takes the loop as a PARAMETER
// and never climbs (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:7316-7318`, `:7325-7327`, then
// `:7586-7588`, `:7608-7610`).
//
// ⚠️ AND `.break` BELOW IS NOT A FIELD. The type declares three (`dsc/dsc2.h:1142-1144`); the fourth
// anchor is a `break;` from `print`'s own switch (`:1151-1167`), minted once the field scan stopped
// requiring a trailing underscore. Tree-wide that costs 8 anchors of 453, beside 14 from `friend
// class` lines, 16 from method-signature continuations ending `) const;` and 5 from `using`/`typedef`
// declarations: 43 anchors that name no field.

// crustify:todo: e042_LoopDistributionInfo

// crustify:todo: e042_LoopDistributionInfo.break

// crustify:todo: e042_LoopDistributionInfo.cat

// crustify:todo: e042_LoopDistributionInfo.dimAndKind

// crustify:todo: e042_LoopDistributionInfo.loopNode

// crustify:todo: e043_LoopCondComposite

// ⛔ e003_AllocateNode'S TWO OPEN FIELD ANCHORS ARE e028_'S AND e037_'S: `allocUsers_`
// (`dsc/dsc2.h:1007`) and `tempStorageForCompute_` (`:978`) are schedule-node pointer identity. The
// TYPE anchor stays open with them, with the five methods that operate on that list, and with the two
// uncarried members this generation's scan does not name at all — `startAddressCoreCorelet_`
// (`:985-986`) and `relatedIndirectAccessAlloc_` (`:999-1001`).
//
// ⚠️ THAT SCAN NAMES 15 OF THE CLASS'S 21 MEMBERS, plus its 4 static string maps. All six it omits —
// the two above and `numBuffers_` (`:984`), `backGapCore_` (`:989`), `ignoreSymbolicVolumeLimits_`
// (`:1002-1003`) and `nonUnifiedAllocInHBM_` (`:1004-1005`), the last four CARRIED above — either
// carry a trailing `//` comment or wrap onto a second line; e028_'s own scan did name two of them.

// crustify:todo: e003_AllocateNode

// crustify:todo: e003_AllocateNode.allocUsers_

// crustify:todo: e003_AllocateNode.tempStorageForCompute_

// crustify:todo: e010_CoordPropInfoType

// crustify:todo: e010_CoordPropInfoType.nodeToFold

// crustify:todo: e010_CoordPropInfoType.refNode

// crustify:todo: e004_BlockNode

// crustify:todo: e004_BlockNode.Ddc

// crustify:todo: e004_BlockNode.DesignSpaceConfig

// crustify:todo: e004_BlockNode.L3DlOpsScheduler

// crustify:todo: e004_BlockNode.ScheduleNode

// crustify:todo: e004_BlockNode.ScheduleTree

// crustify:todo: e004_BlockNode.coreId

// crustify:todo: e034_LoopNode

// crustify:todo: e034_LoopNode.Ddc

// crustify:todo: e034_LoopNode.DesignSpaceConfig

// crustify:todo: e034_LoopNode.ScheduleNode

// crustify:todo: e034_LoopNode.ScheduleTree

// crustify:todo: e034_LoopNode.rowId

// crustify:todo: e030_LoopCond

// crustify:todo: e030_LoopCond.loopComp_

// ⛔ e022_DistributionStatusInfo IS THE THIRD SCHEDULING OF THE DEAD DECLARATION ABOVE, and the
// census still holds at this revision: `loopToSplit`, `loopSplitDim`, `loopSplitDimSizes`,
// `DistributionStatusType` and `NEED_LOOP_SPLIT` occur tree-wide — any file type — ONLY in
// `dsc/dsc2.h:1129-1135`.
//
// ⭐ AND THAT IS WHY A TWO-FIELD STRUCT WOULD NOT CLOSE THESE FOUR ANCHORS: a ported
// `DistributionStatusInfo` cannot meet the second condition for done — a real non-test caller —
// because the authority has none to port. Filling them needs the declaration to acquire a use
// upstream, not a Rust type to be written down here.

// crustify:todo: e022_DistributionStatusInfo

// crustify:todo: e022_DistributionStatusInfo.loopSplitDim

// crustify:todo: e022_DistributionStatusInfo.loopSplitDimSizes

// crustify:todo: e022_DistributionStatusInfo.loopToSplit

// ⛔ e032_LoopDistributionInfo IS THE THIRD SCHEDULING OF e042_ ABOVE, whose `loopNode` note
// stands and whose `.break` is still not a field. What this generation adds is WHY A BORROW CANNOT
// SERVE even though the chain is NOT itself in the tree: `VectorOfLoopAndDim` holds `LoopNode*`,
// non-const (`dsc/dsc2.h:1142`), into the tree that the chain's own subject lives in — and the
// fold reads the chain at `ddc/ddc_fold.cpp:2308` while mutating
// `allocNode->allocateCoordinates_` at `:2268` and `:2328`, a node those same loops enclose (the
// chain is built from it at `:2254-2256`). That is two `&mut` into one owned tree, Rule 4's case,
// not a lifetime to be annotated harder.
//
// ⛔ AND `nullptr` IS A VALUE HERE, NOT AN ABSENCE: `distributeElemArrToTemporalLoops` keys
// `loopParamsAfterDistribution` on a NULL loop for every CORELET_SLICE entry
// (`dsc/dsc2.cpp:6028-6031`), so whatever carries this field carries that key with it. ⭐ WHICH
// MAKES THE KEY ONE CAMPAIGN-WIDE CHOICE, NOT THIS TYPE'S: the same spelling has to serve that
// call-local map and the `Ddc`-member `loopDistributionParamInfo` (`ddc/ddc.h:553`,
// `ddc/ddc_fold.cpp:2312`), which outlives the call. `print` stays open with the field — it
// dereferences `loopNode->name_` (`dsc/dsc2.h:1147`).

// crustify:todo: e032_LoopDistributionInfo

// crustify:todo: e032_LoopDistributionInfo.break

// crustify:todo: e032_LoopDistributionInfo.cat

// crustify:todo: e032_LoopDistributionInfo.dimAndKind

// crustify:todo: e032_LoopDistributionInfo.loopNode

