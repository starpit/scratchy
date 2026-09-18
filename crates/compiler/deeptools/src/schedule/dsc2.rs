//! Re-ported from the C++ authority. See crustify-scheduler/AGENT-BRIEF.md.

use crate::schedule::dims::{DataStructDims, PaddingFormType, PrimaryDimAndKind, PrimaryDimTypes};
use crate::schedule::fold::{
    AffineFoldFunctionLeaf, AffineFoldFunctionNonLeaf, FoldDimIndex, FoldDimProp, FoldDimSize,
    FoldFunc,
};
use core::num::Wrapping;
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
/// `DT_CHECK`ed `<= maxGroupID` (`:4711`).
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
/// ⛔ NOT A FOLD INDEX, AND NOT OUTERMOST-FIRST. IBM's own diagram gives the conversion —
/// `loop_elem_arr_level = foldNumDims - i - 1` over the absolute fold index `i`
/// (`ddc/ddc_fold.cpp:2656-2660`) — and `dsc/dsc2.cpp:6728` inverts it the same way, so level 0 is
/// the INNERMOST fold. The walk also starts at `origNumElemArrFoldsOfRefNode`, not 0 (`:2667`).
///
/// ⭐ UNSIGNED IS SOUND: the writers are a `foldIdx` from a `foldIdx >= 0` loop
/// (`dsc/dsc2.cpp:6016-6017`) and a `currElemArrLevel` that cannot go negative because
/// `elemArrParamsAfterDistribution.at(nextFoldIdx)` throws first (`:6441-6443`).
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
/// lambda `checkAndResetUnitTimeTransfer` (`:1557-1645`) whose only callsite is commented out
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
/// [`LoopInfo::size_idx`] (`dsc/dsc2.cpp:2731-2746`).
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
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GroupTagRegInfo {
    /// Field: e006_GroupTagRegInfo.groupId_
    pub group_id: Option<GroupId>,
    /// Field: e006_GroupTagRegInfo.numSharers_
    pub num_sharers: Option<NumSharers>,
}

// crustify:todo: e007_ScheduleTree

// crustify:todo: e007_ScheduleTree.head_

/// Replaces: e009_FoldParamInfoType
///
/// One fold level's affine parameters, trip count and label (`dsc/dsc2.h:1081-1085`).
/// TRAP: the declared default is the identity in NEITHER field, and `cardinality`'s 0 is not an
/// empty fold — every reader takes a 0 as 1 (`dsc/dsc2.cpp:6249-6253`, `:6359-6361`; see
/// [`Cardinality`]). The identity a caller actually wants is the one `getDefaultRowSplitFold`
/// writes, `alpha = 0` with `cardinality = 1` (`ddc/ddc_fold.cpp:2154-2159`), which overrides BOTH
/// declared initialisers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FoldParamInfoType {
    /// Field: e009_FoldParamInfoType.alpha
    pub alpha: Alpha,
    /// Field: e009_FoldParamInfoType.beta
    pub beta: Beta,
    /// Field: e009_FoldParamInfoType.cardinality
    pub cardinality: Cardinality,
    /// Field: e009_FoldParamInfoType.foldDimLabel
    ///
    /// Open set, not a closed one: `"elem_arr_" + std::to_string(c)` is built per level
    /// (`ddc/ddc_fold.cpp:2698`) and `FoldDimProp::importFromJson` reads it from JSON.
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

/// Replaces: e010_LoopDistributionParamType
///
/// One loop's affine parameters for one dimension after loop distribution (`dsc/dsc2.h:1110-1114`).
/// TRAP: the authority's `= -1` on the last two fields is UNSET, not a value — nothing tests for
/// -1. An unset `relatedElemArrLevel` is LOUD, not silent: `allocFm.getNumDims() - level - 1`
/// (`dsc/dsc2.cpp:6728`) makes the position `getNumDims()`, which `FoldManager::getAlpha`
/// `DT_CHECK`s (`util/foldManager/foldInfrastructure.h:2325-2329`). What IS silent is the opposite
/// direction: `getAlpha` remaps a negative position to `size() + pos`, so a level past the last
/// fold reads a DIFFERENT fold without complaint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LoopDistributionParamType {
    /// Field: e010_LoopDistributionParamType.alpha
    pub alpha: Alpha,
    /// Field: e010_LoopDistributionParamType.beta
    pub beta: Beta,
    /// Field: e010_LoopDistributionParamType.temporalStridePostDistribution
    pub temporal_stride_post_distribution: Option<TemporalStride>,
    /// Field: e010_LoopDistributionParamType.relatedElemArrLevel
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

    /// `dsc/dsc2.h:35-36`: both fields start absent, and a group id is present exactly when there
    /// is more than one sharer — both producers verbatim
    /// (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:4828-4832`, `:5252-5256`).
    #[test]
    fn group_tag_reg_info_carries_a_group_id_exactly_when_it_is_shared() {
        assert_eq!(
            GroupTagRegInfo::default(),
            GroupTagRegInfo {
                group_id: None,
                num_sharers: None
            }
        );

        // `numSharers_ = shares; groupId_ = shares > 1 ? groupName : -1;` with `shares >= 1`
        // guaranteed at `:4697` and `groupName <= maxGroupID` at `:4711`.
        let produce = |shares: u32, group_name: u32| GroupTagRegInfo {
            group_id: (shares > 1).then_some(GroupId(group_name)),
            num_sharers: Some(NumSharers(shares)),
        };
        assert_eq!(produce(1, 7).group_id, None, "one sharer means no share");
        assert_eq!(produce(4, 0).group_id, Some(GroupId(0)), "0 is a legal id");
        for shares in 1..=4 {
            assert_eq!(
                produce(shares, 7).group_id.is_some(),
                shares > 1,
                "a group id is present iff numSharers_ > 1"
            );
        }
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

        // `ddc/ddc_fold.cpp:2687-2689` fills alpha, beta and the level from the fold params;
        // `dsc/dsc2.cpp:6748-6751` fills the stride afterwards, and 0 is a legal stride there.
        let distributed = LoopDistributionParamType {
            alpha: Alpha(64),
            beta: Beta(0),
            temporal_stride_post_distribution: Some(TemporalStride(0)),
            related_elem_arr_level: Some(ElemArrLevel(0)),
        };
        assert_ne!(
            distributed.temporal_stride_post_distribution,
            LoopDistributionParamType::default().temporal_stride_post_distribution
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

    /// `dsc/dsc2.h:43` against BOTH naming conventions: the `+ "el"` suffix the DDC gives the stages
    /// it mints by number (`ddc/ddc_transformation_util.cpp:121-122`, `:131-132`,
    /// `ddc/ddc_transformation.cpp:1018-1019`, `ddc/ddcv1.cpp:1236-1237`) and the shared name all
    /// three NAMED stages carry in both halves — `"core"` (`fillLoopLatchSdsc`,
    /// `dbo/src/Utils/sdsc_bundle/ProgramCorrection.cpp:1074-1075`) and `"chunk"`
    /// (`addOrUpdateDataStageParam` called with one name for both,
    /// `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:1419-1420`, `:1477-1480`).
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
        // climbed parent (`ddc/ddcv1.cpp:634`, `:645`), and the shape that makes the unguarded
        // `dataStageParam_.at(numId_)` reads (`dsc/dsc2.cpp:2995`) a throw rather than a branch.
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

    /// `dsc/dsc2.cpp:2667-2683` INTERSECTS `coreClCond_` into the "then" region's components, so a
    /// core listed with no corelets and a core not listed at all are different conditions — the
    /// distinction a flattened set of core/corelet pairs would lose.
    #[test]
    fn a_condition_nodes_listed_core_with_no_corelets_is_not_an_absent_core() {
        let mut node = ConditionNode::default();
        assert!(node.core_cl_cond.is_empty());
        node.core_cl_cond.insert(CoreId(0), BTreeSet::new());
        node.core_cl_cond
            .insert(CoreId(1), BTreeSet::from([CoreletId(0)]));
        assert_eq!(
            node.core_cl_cond.get(&CoreId(0)).map(BTreeSet::len),
            Some(0)
        );
        assert_eq!(node.core_cl_cond.get(&CoreId(2)), None);
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
        assert!(node.layout_dim_order.is_empty());
        assert!(node.max_dim_sizes.is_empty());
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

    /// `ForceInnermostDimensionsOp` inserting at `begin()` on both vectors
    /// (`ddc/ddl/ddl_conversion.cpp:1900-1901`, `:1904-1905`), over an allocation the DDL conversion
    /// sized with `-1`s (`:803`). The pass refuses to run twice by testing `any_of(maxDimSizes_,
    /// >= 0)` (`:1879-1885`), which is [`Option::is_some`] here — and the entry that separates that
    /// predicate from `buildUnitView`'s cap is pinned below, in
    /// `a_zero_max_dim_size_is_filled_to_every_writer_and_absent_to_the_only_cap`.
    #[test]
    fn forcing_inner_dims_prepends_to_both_vectors_and_is_refused_twice() {
        let mut node = AllocateNode {
            layout_dim_order: vec![PrimaryDimTypes::Y, PrimaryDimTypes::Out],
            ..AllocateNode::default()
        };
        node.max_dim_sizes.resize(node.layout_dim_order.len(), None);
        assert!(!node.max_dim_sizes.iter().any(Option::is_some));

        // `:1900-1905`: the forced dim becomes the innermost, and it carries a data stage index —
        // not an extent — until `finalizeAllocateLayouts` overwrites it (`ddc/ddcv1.cpp:1710-1732`).
        node.layout_dim_order.insert(0, PrimaryDimTypes::In);
        node.max_dim_sizes.insert(0, Some(MaxDimSize(2)));
        assert_eq!(node.layout_dim_order[0], PrimaryDimTypes::In);
        assert_eq!(node.max_dim_sizes.len(), node.layout_dim_order.len());
        assert!(node.max_dim_sizes.iter().any(Option::is_some));
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
                PrimaryDimTypes::Y,
                PrimaryDimTypes::Out,
                PrimaryDimTypes::In,
            ],
            max_dim_sizes: vec![None, Some(MaxDimSize(0)), Some(MaxDimSize(4))],
            ..AllocateNode::default()
        };
        assert_eq!(node.max_dim_sizes.len(), node.layout_dim_order.len());

        // `getPageSize`'s `maxSize < 0`: only the absent entry leaves its dim unbounded.
        let unbounded = node
            .layout_dim_order
            .iter()
            .zip(&node.max_dim_sizes)
            .filter(|(_, max)| max.is_none())
            .map(|(dim, _)| *dim)
            .collect::<Vec<_>>();
        assert_eq!(unbounded, [PrimaryDimTypes::Y]);

        // The writers' `>= 0`: the zero counts as ALREADY WRITTEN, so the DDL pass refuses to run a
        // second time over it and `finalizeAllocateLayouts` overwrites it in place.
        let filled = node
            .max_dim_sizes
            .iter()
            .filter(|max| max.is_some())
            .count();
        assert!(node.max_dim_sizes.iter().any(Option::is_some));
        assert_eq!(filled, 2);

        // `buildUnitView`'s `> 0`: the zero is NOT a cap, and it is `is_some` all the same.
        let caps = node
            .max_dim_sizes
            .iter()
            .filter(|max| max.is_some_and(|MaxDimSize(size)| size > 0))
            .copied()
            .collect::<Vec<_>>();
        assert_eq!(caps, [Some(MaxDimSize(4))]);
        assert_eq!(caps.len(), 1);

        // And the entry that cap ignored is a bound of ZERO downstream, not an absence: one dim's
        // page size is the product of its own entries (`dsc/dsc2.cpp:4507-4508`).
        let page_size_of_out = node
            .layout_dim_order
            .iter()
            .zip(&node.max_dim_sizes)
            .filter(|(dim, max)| **dim == PrimaryDimTypes::Out && max.is_some())
            .map(|(_, max)| max.unwrap().0)
            .product::<i32>();
        assert_eq!(page_size_of_out, 0);
    }

    /// `allocAllMem`'s buffer arithmetic end to end (`ddc/ddcv1.cpp:224-226`, `:244`, `:317-328`,
    /// `:340`, `:352-356`) for a streaming allocation — the one case where the REQUEST and the
    /// RESERVATION differ. ⛔ THE STRIDE IS THE REQUEST OVER THE COUNT, NOT THE RESERVATION OVER THE
    /// COUNT: `kv.second` at `:356` is what was pushed at `:244`, while the widening at `:327` only
    /// reached the local `mySize` that `checkAndAddDs` places at `:340`.
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
    /// (`dsc-based-utils/DSC2ToDataflowIR/V3/SNDSCLowering.cpp:479-480`, `:514-516`) and the PCFG
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
    /// second one also tests the datum, which is `data_`, unported.
    #[test]
    fn a_constants_name_is_the_key_both_in_scope_readers_match_on() {
        let constants = BTreeMap::from([
            (
                ConstantId(0),
                ConstantInfo {
                    data_format: DataFormats::Sen143Fp8,
                    name: "useZeroMean".to_string(),
                    is_data_symbolic: false,
                },
            ),
            (
                ConstantId(1),
                ConstantInfo {
                    data_format: DataFormats::Sen169Fp16,
                    name: "padval".to_string(),
                    is_data_symbolic: false,
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
    /// `TransferPadInfo(TransferPadInfo&&) = default` (`dsc/dsc2.h:763`) moves the two maps but
    /// bitwise-copies `MapWithFMHelper`, whose only member is a REFERENCE to the sibling map
    /// (`util/mapWithFMHelper.h:36-38`) — so the moved-TO object's helper still refers to the
    /// moved-FROM object's storage. Measured on the authority: `dst.isEmpty = 0` with
    /// `dst.frontKeys = 0` and every helper-routed query throwing, `dst.frontKeys = 1` again after
    /// rebuilding on the SOURCE, and `heap-use-after-free` under AddressSanitizer inside `getAllKeys`
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
        // inner (`dsc/dsc2.cpp:4662-4670`).
        let props = &moved.front[&PrimaryDimTypes::X].props;
        assert_eq!(props[0].size(), FoldDimSize(2));
        assert_eq!(props[0].label(), "wkslice_index");
        assert_eq!(props[1].size(), FoldDimSize(3));
        assert_eq!(props[1].label(), "chunk_index");
    }

    /// `isNodeRelevant`'s component reading against the map the authority's own writer builds
    /// (`dsc/dsc2.cpp:2689-2694`), and the divergence between it and `getRelevantComps`: a core
    /// present with an EMPTY corelet set is relevant to the first and invisible to the second
    /// (`dsc/dsc2.cpp:1929-1931` against `:1964-1967`).
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
}

// crustify:todo: e012_CoordinateType

// crustify:todo: e012_CoordinateType.coordinates_

// crustify:todo: e012_CoordinateType.coreIdToWkSlice_

// crustify:todo: e012_CoordinateType.foldConstructed_

// crustify:todo: e012_CoordinateType.numOfElemArrFolds_

// crustify:todo: e012_CoordinateType.numOfSpatialFolds_

// crustify:todo: e012_CoordinateType.numOfTemporalFolds_

// crustify:todo: e012_CoordinateType.padding_

/// `dsc2::ScheduleNode` (`dsc/dsc2.h:444-524`) — the base every node in a schedule tree derives
/// from. `BlockNode` (`:526`) derives from it and owns the children; `LoopNode` (`:563`) and
/// `ConditionNode` (`:685`) derive from `BlockNode`; `TransferNode` (`:814`), `ComputeNode`
/// (`:900`), `SyncNode` (`:964`), `AllocateNode` (`:974`) and `StickMaskNode` (`:1059`) derive from
/// it directly. Those eight are the whole hierarchy, and [`NodeType`] is its discriminant — the
/// importer's `new`-per-kind chain is the exhaustive list (`dsc/dsc2.cpp:1337-1358`).
///
/// ⛔ THIS CARRIES 3 OF SCHEDULENODE'S 4 FIELDS, so the `e029_ScheduleNode`/`e013_ScheduleNode`
/// anchors below stay open. `prev_` (`dsc/dsc2.h:515`) is a `BlockNode*` pointing back at the parent
/// that OWNS this node, through `BlockNode::next_`, a `VectorOfChildren` of `unique_ptr`s (`:538`).
/// Every reader of it is a tree operation, not a question about one node: `getPrev` and
/// `getMutableParent` hand it straight out (`:463-464`), `getOwnerLoop` climbs it to the nearest
/// `LOOP` (`dsc/dsc2.cpp:1896-1900`), `getParentDimLoop` climbs on from there to the nearest loop
/// carrying a dim (`:1906-1914`), `insertLoopAbove` finds `this` in `prev_->next_` by ADDRESS,
/// `nodePtr.get() == this`, and splices a loop into its slot (`:2169-2186`), and `moveNode` forwards
/// the whole job to `prev_->moveChildNode` (`:1977-1982`). So the identity a Rust parent link would
/// need is the one `ScheduleTree::head_` (`dsc/dsc2.h:623`) and `BlockNode::next_` have to define,
/// and both of those anchors are open — `e030_BlockNode`/`e015_BlockNode` and
/// `e032_ScheduleTree`/`e007_ScheduleTree`.
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
    ///
    /// Which kind of node this is (`dsc/dsc2.h:460`). PRIVATE because the authority's is `const`:
    /// it is fixed by the constructor and there is no path that rewrites it, which is what lets the
    /// JSON exporter and importer dispatch on it (`dsc/dsc2.cpp:376-377`, `:1337-1358`).
    node_type: NodeType,
    /// Field: e029_ScheduleNode.name_
    /// Field: e013_ScheduleNode.name_
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
    /// ⛔ AN EMPTY CORELET SET IS A REAL STATE AND THE TWO READERS DISAGREE ABOUT IT. The writer
    /// reaches it through `operator[]`: `relCoreCls[core].insert(cls.begin(), cls.end())` creates the
    /// core's entry before it inserts anything, so a core the `NO_COMPONENT` map carries with no
    /// corelets is created empty under the real component too (`dsc/dsc2.cpp:2692-2694`). Then
    /// `isNodeRelevant(comp, -1, coreId)` returns true on nothing but that core's PRESENCE
    /// (`:1929-1931`), while `getRelevantComps(coreId)` requires `!clSet.empty()` (`:1964-1967`) and
    /// `getRelevantCoreCl` drops such a core from its result altogether (`:1940-1942`). So neither
    /// reader can be composed out of the other, and both are ported.
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
    /// (`:1919-1921`) fires for one argument combination — a core or corelet filter beside a
    /// component filter of `ALL` — and no function here takes both a component and a core, so that
    /// combination cannot be spelled. `isNodeRelevant(comp, -1, coreId)` is
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
    /// by key (`:1373-1382`), and the work split copies a node's whole map onto its clone (`:5355`).
    ///
    /// ⛔ THE FIELD IS UNWRITEABLE WITHOUT IT, and a carried field with no writer is the defect this
    /// campaign already booked once: every one of those four writers is a method of a type whose own
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
    /// has passed (`dsc/dsc2.cpp:2731-2746`). Innermost-first is what the gap writer relies on when
    /// it takes the HIGHEST index of a repeated dim as the outermost one (`:2905-2924`).
    pub sizes_no_gaps: Vec<Size>,
    /// Field: e029_ScheduleNode.compositeLoops_
    /// Field: e013_ScheduleNode.compositeLoops_
    ///
    /// The enclosing loops up to and including the last fusable parent (`dsc/dsc2.h:507`). Unlike
    /// [`Self::outer_loops`] this one also carries loops that touch NONE of the data structure's
    /// dims, pushed with no extent and no offset at all (`dsc/dsc2.cpp:2869-2873`).
    pub composite_loops: Vec<LoopInfo>,
    /// Field: e029_ScheduleNode.outerLoops_
    /// Field: e013_ScheduleNode.outerLoops_
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
    /// `:2076`, `:2356`, `:2447`, `SNComputeLowering.cpp:575`, `:843`).
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
    ///
    /// Which dimension of the data structure this entry is about (`dsc/dsc2.h:502`). Its initialiser
    /// is `PrimaryDimTypesCount`, [`PrimaryDimTypes::Undefined`] here, which is a live value and not
    /// an absent one — the exporter spells it through `primaryDimToString` like any other dim
    /// (`dsc/dsc2.cpp:211-212`), where that enumerator maps to `"undefined"` (`dsc/dims.cpp:23`).
    pub dim: PrimaryDimTypes,
    /// Field: e029_ScheduleNode.sizeIdx_
    /// Field: e013_ScheduleNode.sizeIdx_
    ///
    /// Which entry of [`UnitView::sizes_no_gaps`] this loop steps (`dsc/dsc2.h:503`), as
    /// `calculateSizeIdxAndOffset` resolved it (`dsc/dsc2.cpp:2731-2746`).
    ///
    /// ⛔ [`None`] IS THE AUTHORITY'S `-1` AND IT IS REACHABLE IN A STORED ENTRY: a loop that touches
    /// none of the data structure's dims is pushed as `{currLoop, dim, -1, -1}`
    /// (`dsc/dsc2.cpp:2872`). The gap rescale then multiplies the offset of every entry whose
    /// `sizeIdx_ == i` for a gapped layout entry `i` (`dsc/dsc2.cpp:2880-2897`), a test the `-1`
    /// silently fails and a [`None`] cannot be mistaken for a position.
    pub size_idx: Option<SizeIdx>,
    /// Field: e029_ScheduleNode.elemOffset_
    /// Field: e013_ScheduleNode.elemOffset_
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

/// Replaces: e014_DataStage
///
/// `dsc/dsc2.h:39-44`. One data stage's two halves — the steady-state dims and the epilogue dims of
/// the same data structure. `DesignSpaceConfig::dataStageParam_` keys them by id
/// (`dsc/designSpaceConfig.h:105`); id 0 is the core stage, whose name `getSizeDataStageForNode`
/// `DT_CHECK`s to be `"core"` (`dsc/dsc2.cpp:3638-3639`).
///
/// ⛔ [`name`](Self::name) IS THE STEADY STATE'S NAME ALONE, and WHETHER THE EPILOGUE CARRIES A
/// DIFFERENT ONE DEPENDS ON WHO MINTED THE STAGE. The `+ "el"` suffix belongs to the stages the DDC
/// mints by NUMBER — `constructDatastage` writes `to_string(id)` and `to_string(id) + "el"`
/// (`ddc/ddc_transformation_util.cpp:121-122`, `:131-132`, and the L3 scheduler's own copy at
/// `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:7730-7731`), the chunk split does the same from `ss_`
/// (`ddc/ddc_transformation.cpp:1018-1019`), and `calculateEpilogues` copies `ss_` and appends to
/// the copy's name (`ddc/ddcv1.cpp:1236-1237`). ⛔ ALL THREE **NAMED** STAGES CARRY THE SAME NAME IN
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
    /// Field: e014_DataStage.ss_
    ///
    /// The steady state: the dims of every trip but the last. It is the half readers reach for by
    /// default (`ddc/ddc_fold.cpp:2113`, `ddc/ddcv1.cpp:1924`,
    /// `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:4848`).
    pub ss: DataStructDims,
    /// Field: e014_DataStage.el_
    ///
    /// The epilogue: the dims of the last, short trip. `calculateEpilogues` seeds it from `ss_` and
    /// then shrinks only the dims the metadata calls relevant (`ddc/ddcv1.cpp:1230-1330`), so an
    /// untouched epilogue equals the steady state rather than being empty.
    pub el: DataStructDims,
}

impl DataStage {
    /// `DataStage::name` (`dsc/dsc2.h:43`) — what `attachToPrefilledSchedule` tests against `"core"`
    /// and `"chunk"` (`ddc/ddcv1.cpp:2283-2285`), what the DDL conversion tests for emptiness
    /// (`ddc/ddl/ddl_conversion.cpp:2974`, `:2998`), and what a loop label is built from
    /// (`dsc-based-utils/DSC2ToDataflowIR/V3/SNControlFlowLowering.cpp:953-954`).
    pub fn name(&self) -> &str {
        &self.ss.name
    }
}

// crustify:todo: e015_BlockNode

// crustify:todo: e015_BlockNode.next_

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

/// `dsc/dsc2.h:563-619`. One loop of the schedule tree: the dims it iterates and the two data
/// stages whose ratio is its trip count (`getTripCount(dsc, dim, numId_, denId_)`,
/// `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:1840`). The minting site is
/// `ddc/ddl/ddl_conversion.cpp:1076-1164`: a DDL `LoopOp` becomes a datastage loop and a
/// `ParametricLoopOp` a parametric one.
///
/// ⛔ THIS CARRIES LOOPNODE'S OWN SIX DECLARED FIELDS AND NOTHING INHERITED, so the
/// `e017_LoopNode` anchor at the end of this file is still open. `nodeType_`, `name_`, `prev_` and
/// `relevantComps_` are `ScheduleNode`'s (`dsc/dsc2.h:460-461`, `:515-516`) and `next_` is
/// `BlockNode`'s (`:538`) — e013 and e015 both. That also keeps three methods out:
/// `parametricIterCount` (`dsc/dsc2.cpp:4126`) and `parametricStride` (`:4197`) read
/// `DesignSpaceConfig::dataStageParam_` and `labeledDs_` and climb `getOwnerLoop()`, and `print`
/// (`:4284`) prints `name_`.
///
/// ⛔ WHEN `next_` LANDS, THE `Clone` DERIVE BELOW BECOMES A DIVERGENCE. IBM's `clone()` is
/// `new Derived(static_cast<Derived const&>(*this))` (`util/utils.h:105-107`), i.e. the copy
/// constructor, and `BlockNode::next_`'s copy constructor is EMPTY ON PURPOSE
/// (`VectorOfChildren(const VectorOfChildren&) {}`, `dsc/dsc2.h:533-536`) — so cloning a loop,
/// block or condition yields a node with NO CHILDREN and the caller re-inserts them. Both
/// BlockNode-derived clone sites rely on it: `ddc/ddc_transformation.cpp:984-986` clones a loop and
/// then `addChildNode`s one cloned child, and `ddc/ddc_transformation_util.cpp:580-588` clones a
/// `ConditionNode` and then `addThenRegion`s a fresh block — which `DT_ERROR`s outright if `next_`
/// is non-empty (`dsc/dsc2.cpp:2152-2155`). A derived deep `Clone` over an owning child list would
/// therefore turn a working DDC path into a fatal error, not merely copy too much.
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
#[derive(Clone, Debug, Default)]
pub struct LoopNode {
    /// Field: e017_LoopNode.numId_
    ///
    /// The numerator stage (`dsc/dsc2.h:573`). ⛔ `-1` IS ABSENT, NOT A STAGE, AND NO READER TESTS
    /// FOR IT: every one indexes straight through, `dataStageParam_.at(numId_)`
    /// (`dsc/dsc2.cpp:2995`, `:6104`, `ddc/ddcv1.cpp:2496`, `:2835`, `ddc/ddc_fold.cpp:2349`,
    /// `:3235`, `:3604`) or `constraints_[loop->numId_]` (`ddc/ddcv1.cpp:610`, `:649`), so an absent
    /// numerator is an out-of-range throw and never a branch. The `>= 0` guards belong to
    /// [`den_id`](Self::den_id) alone.
    pub num_id: Option<DataStageId>,
    /// Field: e017_LoopNode.denId_
    ///
    /// The denominator stage (`dsc/dsc2.h:574`). ⛔ NOT SYMMETRIC WITH [`num_id`](Self::num_id):
    /// this is the id with absence guards, and all three sit where the reader has climbed
    /// `getOwnerLoop()` to a PARENT loop and can therefore reach the head — `exploreAssignDataStages`
    /// at `ddc/ddcv1.cpp:634` and `:645`, and `parametricIterCount` at `dsc/dsc2.cpp:4144-4147`. Its
    /// other uses in that same function are unguarded (`ddc/ddcv1.cpp:607`, `:694`, `:701`,
    /// `:727-729`), so the guard marks the climb, not the field.
    pub den_id: Option<DataStageId>,
    /// Field: e017_LoopNode.dims_
    ///
    /// ⛔ ORDERED INNER TO OUTER (`dsc/dsc2.h:575`, and `dsc/dsc2Pcfg.cpp:517` says `// Inner to
    /// outer` verbatim over a front-to-back walk), and bridge 1 depends on that: it walks `dim_idx`
    /// from `dims_.size() - 1` down to 0, and the loop it opens FIRST is the one it later retrieves
    /// as `.at(0)` = outermost, so the LAST entry becomes the OUTERMOST loop of the emitted nest
    /// (`dsc-based-utils/DSC2ToDataflowIR/V3/SNControlFlowLowering.cpp:893`, `:898-957`).
    pub dims: Vec<PrimaryDimAndKind>,
    /// Field: e017_LoopNode.loopCountSymbolIds_
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
    /// Field: e017_LoopNode.isParametricLoop_
    ///
    /// Private in IBM's declaration (`dsc/dsc2.h:617`) and ONE-WAY: `markAsParametricLoop` is the
    /// only writer tree-wide (`ddc/ddl/ddl_conversion.cpp:1128`, `dsc/dsc2.cpp:1412`) and nothing
    /// clears it. Readers go through `isParametricLoop()` (`ddc/ddcv1.cpp:2830`,
    /// `dsc/dsc2.cpp:2994`, `:4129`) except the JSON exporter, which reaches the field itself
    /// through friendship (`dsc/dsc2.cpp:414`, `:416`) — a read, so the getter below still covers it
    /// and this stays private.
    is_parametric_loop: bool,
    /// Field: e017_LoopNode.parametricLdsIdx_
    ///
    /// The reference tensor whose cumulative stick size along the loop dim IS the parametric loop's
    /// stride (`dsc/dsc2.h:618`, read at `dsc/dsc2.cpp:4198-4210`). ⛔ `-1` IS ABSENT and
    /// `parametricStride` refuses on it (`:4199-4202`); the DDL conversion sets it from the
    /// reference tensor's `ldsIdx_` and rejects a tensor without one
    /// (`ddc/ddl/ddl_conversion.cpp:1155-1161`).
    parametric_lds_idx: Option<LdsIdx>,
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

// crustify:todo: e017_LoopNode

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
/// carries (`ddc/ddl/ddl_conversion.cpp:267`) has to round-trip. A Rust `CondVal` that fused the form
/// and the integer into one enum would still have to hold that `-1` to reproduce the JSON.
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

    /// The spelling `LoopCond::condValTypeToString` gives this form (`dsc/dsc2.h:656`, defined
    /// `dsc/dsc2.cpp:21-23`). ⭐ TOTAL, AND THE AUTHORITY'S MAP IS TOO — all three have an entry.
    pub fn name(self) -> &'static str {
        match self {
            Self::Int => "int",
            Self::First => "first",
            Self::Last => "last",
        }
    }

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

// crustify:todo: e018_LoopCond

// crustify:todo: e018_LoopCond.condOp_

// crustify:todo: e018_LoopCond.condValInt_

// crustify:todo: e018_LoopCond.condValType_

// crustify:todo: e018_LoopCond.dim_

// crustify:todo: e018_LoopCond.loopComp_

// crustify:todo: e019_DataInfo

// crustify:todo: e019_DataInfo.bufferSwitchPosition_

// crustify:todo: e019_DataInfo.constantId_

// crustify:todo: e019_DataInfo.dataConnect_

// crustify:todo: e019_DataInfo.isStartAddrSymbolic_

// crustify:todo: e019_DataInfo.latchDataId_

// crustify:todo: e019_DataInfo.myLdsIdx_

// crustify:todo: e019_DataInfo.startAddr_

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

// crustify:todo: e022_LoopCondComposite.negated_

// crustify:todo: e022_LoopCondComposite.twoLevelOrOfAnds_

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
/// adds `repetition_` and `paddingInfo_`.
///
/// ⛔ THIS CARRIES TWELVE OF TRANSFERNODE'S OWN TWENTY DECLARED FIELDS AND NOTHING INHERITED, so
/// eight field anchors below stay open. Every one is blocked on a type another agent owns, none on
/// this class:
///  * `srcLdsAndLoopOffsets_`, `srcIndirectLdsAndLoopOffsets_` (`:832`) and
///    `dstLdsAndLoopOffsets_`, `dstIndirectLdsAndLoopOffsets_` (`:833`) are `DataInfo` — e019;
///  * `lastFusableParentLoopSrc_` (`:830`) and `lastFusableParentLoopDst_` (`:831`) are
///    `const LoopNode*` held as POINTER IDENTITY, which needs e013's `name_`, exactly as
///    `SyncNode::implicitSyncRefTransfer_` does;
///  * `coreletViews_` (`:851`) is a map of `CoreletView`, four `UnitView`s (`:847-850`) — e013;
///  * `transferCoordinates_` (`:852`) is `CoordinateType<CoordinateBaseType>` — e012.
///
/// ⚠️ AND FOUR OF THOSE EIGHT FIELDS HAVE NO ANCHOR AT ALL, so the open anchors undercount the
/// remaining work: the scheduler's field scan takes one declarator per declaration, so
/// `srcLdsAndLoopOffsets_` and `dstLdsAndLoopOffsets_` (`:832-833`) lost theirs to the `Indirect`
/// twin sharing their line, and `CoreletView`'s `srcLoopsAndSize_` and `dstLoopsAndSizes_`
/// (`:848-849`) lost theirs the same way. They are named here rather than anchored, since an invented
/// anchor is indistinguishable from a scheduled one.
///
/// ⛔ AND SEVEN METHODS STAY OUT WITH THEM: `isSrcLabeledDs`, `isDstLabeledDs`, `isSrcConstant`,
/// `isDstConstant` and `isDstIndirect` are one-line reads of the `DataInfo` fields above
/// (`dsc/dsc2.h:867-878`), `getTransferType` is built from four of those five (`:884-896`), and
/// `print` prints `name_` and each `DataInfo` (`dsc/dsc2.cpp:4385-4441`).
///
/// ⛔ `dstVias_` AND `dstLdsAndLoopOffsets_` ARE PARALLEL VECTORS IN THE AUTHORITY, and this type
/// cannot say so until e019 lands: the DDL conversion emplaces one of each per destination in the
/// same iteration (`ddc/ddl/ddl_conversion.cpp:1176-1177`), and `hoistTransfersUpForReuse` takes
/// [`non_memory_result_index`](Self::non_memory_result_index) — an index into `dstVias_` — and
/// indexes `dstLdsAndLoopOffsets_` with it (`ddc/ddc_transformation.cpp:1570-1575`).
/// [`TransferRepetition::dsts`] is the THIRD vector on that same index and it IS carried.
///
/// ⛔ NO `PartialEq`, as [`LoopNode`] has none: IBM declares no `operator==` and every consumer keys
/// on the POINTER. ⛔ AND NO `Clone` DERIVE either, now that `paddingInfo_` is carried — see
/// [`clone`](Self::clone).
#[derive(Debug)]
pub struct TransferNode {
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
    /// Field: e034_TransferNode.replicationFactor_
    ///
    /// How many times the loaded chunk is splatted, `1` for no splat (`dsc/dsc2.h:834`). Bridge 1
    /// divides the recorded stick and element counts by it, and refuses an LXLU splat it cannot
    /// express (`dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:728-729`, `:963-971`).
    pub replication_factor: i32,
    /// Field: e034_TransferNode.unitTimeTransferChunkSize_
    ///
    /// "Continuous elements within a stick" (`dsc/dsc2.h:835-836`) — the contiguous dims of one
    /// unit-time transfer, ONE PER STICK SIZE AND IN STICK ORDER, which is all the one live producer
    /// emits (`ddc/ddcv1.cpp:524-525`). Bridge 1 multiplies the extents into the element count of
    /// one `agen` access (`SNTransferLowering.cpp:33-38`).
    ///
    /// ⛔ IT IS NEVER "EXTENDED WITH LAYOUT DIMS" ON A LIVE PATH, as recorded before. The append of
    /// an out-of-stick `IN` entry (`:1597-1598`) and the erase that truncates this vector
    /// (`:1642-1643`) both sit inside `checkAndResetUnitTimeTransfer` (`:1557-1645`) — the same dead
    /// lambda that owns [`unit_time_transfer_chunk_stride`](Self::unit_time_transfer_chunk_stride)'s
    /// only writer, whose callsite is commented out at `:1649-1650`. So live, this vector's length
    /// is the stick count, each entry's index equals its own position, and the two invariants the
    /// lambda would break — that length, and `DT_CHECK(uttChunkSize.size() == tensorSizes.size())`
    /// at `:1568` — are unreachable. After minting, the 4B-splat mutation of entry 0 (`:544-546`) is
    /// the only DDC write that reaches it.
    pub unit_time_transfer_chunk_size: Vec<SizeAndIndex>,
    /// Field: e034_TransferNode.unitTimeTransferNumChunks_
    ///
    /// How many chunks one unit-time transfer covers, `1` for a single contiguous chunk
    /// (`dsc/dsc2.h:837`). It is the product of the extents the hole split moved out of
    /// [`unit_time_transfer_chunk_size`](Self::unit_time_transfer_chunk_size)
    /// (`ddc/ddcv1.cpp:1633-1642`), which bridge 1 multiplies back in
    /// (`SNTransferLowering.cpp:33-38`).
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
}

impl Default for TransferNode {
    /// The authority's member initialisers (`dsc/dsc2.h:834`, `:837`, `:839`) over
    /// [`DataLocation`]'s own (`sys-arch-spec/arch_enums.h:390-391`).
    ///
    /// ⛔ IT IS NOT `TransferNode()`: that constructor also passes `TRANSFER` to the base class
    /// (`dsc/dsc2.h:815`), and `nodeType_` is `ScheduleNode`'s, e013's to port.
    ///
    /// ⛔ AND ONE MEMBER HAS NO INITIALISER TO REPRODUCE: `repetition_.srcRep_` (`:827`) is left
    /// uninitialised by that constructor, which is what [`TransferRepetition::src`]'s [`None`] spells.
    fn default() -> Self {
        Self {
            src: DataLocation::UNSET,
            src_indirect: DataLocation::UNSET,
            dst_vias: Vec::new(),
            repetition: TransferRepetition::default(),
            replication_factor: 1,
            unit_time_transfer_chunk_size: Vec::new(),
            unit_time_transfer_num_chunks: 1,
            unit_time_transfer_chunk_stride: Vec::new(),
            rotate_num_elements: 0,
            core_id_to_gtr_info: BTreeMap::new(),
            transfer_size: BTreeMap::new(),
            padding_info: TransferPadInfo::default(),
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
            src: self.src,
            src_indirect: self.src_indirect,
            dst_vias: self.dst_vias.clone(),
            repetition: self.repetition.clone(),
            replication_factor: self.replication_factor,
            unit_time_transfer_chunk_size: self.unit_time_transfer_chunk_size.clone(),
            unit_time_transfer_num_chunks: self.unit_time_transfer_num_chunks,
            unit_time_transfer_chunk_stride: self.unit_time_transfer_chunk_stride.clone(),
            rotate_num_elements: self.rotate_num_elements,
            core_id_to_gtr_info: self.core_id_to_gtr_info.clone(),
            transfer_size: self.transfer_size.clone(),
            padding_info: TransferPadInfo::default(),
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
}

// crustify:todo: e034_TransferNode.coreletViews_

// crustify:todo: e034_TransferNode.dstIndirectLdsAndLoopOffsets_

// crustify:todo: e034_TransferNode.dstIndirectLoopsAndSizes_

// crustify:todo: e034_TransferNode.lastFusableParentLoopDst_

// crustify:todo: e034_TransferNode.lastFusableParentLoopSrc_

// crustify:todo: e034_TransferNode.srcIndirectLdsAndLoopOffsets_

// crustify:todo: e034_TransferNode.srcIndirectLoopsAndSize_

// crustify:todo: e034_TransferNode.transferCoordinates_

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
/// (`dsc/dsc2.h:900-901`), and the base's thirteen fields are e029's. FIVE field anchors stay OPEN,
/// every one blocked on a type another agent owns and none on this class: `inputsLdsAndLoopOffsets_`
/// and `outputsLdsAndLoopOffsets_` are `std::vector<DataInfo>` (`:937-938`), e019; `inputCoordinates_`
/// and `outputCoordinate_` are `CoordinateType<CoordinateBaseType>` (`:948-949`), e012. ⛔ `port.json`
/// names this unit's ONE dep `e023_CoordinateType`, a renumbering artefact that is NOT satisfied.
/// ⭐ `coreletViews_` AND ITS TWO SEPARATELY ANCHORED HALVES ARE PORTED HERE and were not portable
/// when e024 ran: `ScheduleNode::UnitView` (`:943-947`) landed with e029 in `625e761da`.
///
/// ⛔ AND THE FIFTH, `instrAttribute_.computeMaskLoopOffsets_`, IS KEYED BY A `const LoopNode*` WHOSE
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
/// ⛔ `print` IS THE ONE METHOD LEFT OUT, and e019 alone still blocks it — it prints the base's
/// `name_`, which has landed, and then each `DataInfo`'s own `print` (`dsc/dsc2.cpp:4443-4477`).
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
    /// Field: e035_ComputeNode.repetitionWithOffset_
    ///
    /// `dsc/dsc2.h:954`.
    pub repetition_with_offset: RepetitionWithOffset,
}

impl Default for ComputeNode {
    /// The authority's default member initializers (`dsc/dsc2.h:932-941`, `:950-954`). ⛔ WHAT IT
    /// CANNOT SET IS THE BASE'S TAG: `ComputeNode()` passes `COMPUTE` to `ScheduleNode`
    /// (`dsc/dsc2.h:901`), and that field is e013's.
    fn default() -> Self {
        Self {
            ex_unit: SenComponent::NoComponent,
            r#type: ComputeOpType::Count,
            data_format: DataFormats::Sen169Fp16,
            inputs: Vec::new(),
            outputs: Vec::new(),
            instr_attribute: InstrAttribute::default(),
            num_folds_engaged: NumFoldsEngaged(1),
            is_opaque_op: false,
            corelet_views: BTreeMap::new(),
            repetition_with_offset: RepetitionWithOffset::default(),
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
    /// this method WHOLE, since both hops of that lookup are unported: e019's `DataInfo`, and
    /// [`DesignSpaceConfig`](crate::schedule::dsc::DesignSpaceConfig) carries no `labeledDs_`.
    /// ⛔ SO THE ARGUMENT IS IGNORED ON EVERY OTHER OP, exactly as IBM's `dsc` is.
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

// crustify:todo: e035_ComputeNode.inputCoordinates_

// crustify:todo: e035_ComputeNode.inputsLdsAndLoopOffsets_

// crustify:todo: e035_ComputeNode.outputCoordinate_

// crustify:todo: e035_ComputeNode.outputsLdsAndLoopOffsets_

/// Replaces: e025_ConditionNode
///
/// `dsc/dsc2.h:685-719`. A two-way branch in the schedule tree: a `BlockNode` whose at most two
/// children are the "then" and the "else" region (`:688`, `:698-699`) — in THAT order, because
/// `getThenBranchNode` is `next_[0]` and `getElseBranchNode` is `next_[1]` (`:707-718`).
///
/// ⛔ THIS CARRIES ONE OF CONDITIONNODE'S TWO GUARDS, so the `e025_ConditionNode` anchor below stays
/// open. `loopCond_` (`:690`) is a `LoopCondComposite` — e022, blocked behind e018's
/// `const LoopNode* loopComp_`, which is schedule-node pointer identity.
///
/// ⛔ AND THAT BLOCK IS IDENTITY AND NOT EQUALITY, which is what a port must supply before e022 can
/// land: `LoopCondComposite::adjustConditionForSplitLoop` selects a term by
/// `loopComp_ != origLoop` and rewrites it to `newLoops.at(0)` (`dsc/dsc2.cpp:2071-2076`), then
/// rebuilds the enclosing conjunction by comparing every term against `newLoops.at(0)` AGAIN
/// (`:2126-2132`) — so two terms of one conjunction that both named `origLoop` are
/// indistinguishable once the first has been substituted, and the rebuild replaces BOTH. Nothing in
/// that function reads a loop's contents, so an index or a name will not do.
///
/// ⛔ AND E022 HAS TWO CARRIERS, NOT ONE: besides `loopCond_` here, `DdlInterface::CondProp` holds
/// one (`ddc/ddl/ddl_conversion.h:419-424`, the composite at `:421`), and it is that copy the DDL
/// front end toggles and then refuses on (`ddc/ddl/ddl_conversion.cpp:326`, `:381-394`). Both must
/// reach the same Rust type.
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
/// `(LT,LAST)` (`:2136`). The first two are total over closed enums and belong in the type; the
/// always-true/false four are a `(CondOp, CondValType)` PAIR, and the authority's own comment says
/// the DDL parser was supposed to simplify them away (`:2097-2098`), so they are a guard on the
/// pairing and not on either enum alone.
///
/// ⛔ AND THE DISCRIMINATOR IS THAT MISSING FIELD, NOT THIS ONE: `hasCoreClCond()` answers
/// `loopCond_.twoLevelOrOfAnds_.empty()` (`:693-695`) and never looks at `coreClCond_`. A node with
/// both guards empty therefore reads as a core/corelet condition selecting NO core, not as an
/// unconditional region — nothing in the authority enforces the header's "only one is filled"
/// (`:688-689`).
///
/// ⛔ ITS EIGHT METHODS ALL REACH `next_`, e015's field, blocked on e013: `addChildNode` (which
/// refuses anything but a `BLOCK` and any third child, `dsc/dsc2.cpp:2143-2150`), `addThenRegion`,
/// `addElseRegion`, `getThenBranchNode`, `getElseBranchNode`, `getThenCoreCl`, `getElseCoreCl` and
/// `getNextView`, which widens the base view to `ALL, -1, -1` whenever the guard is a loop condition
/// (`dsc/dsc2.cpp:1995-2001`).
///
/// ⛔ NO `PartialEq`: node identity in the authority is the pointer. `Clone` is IBM's own, through
/// `InheritWithClone` (`:685`).
#[derive(Clone, Debug, Default)]
pub struct ConditionNode {
    /// Field: e025_ConditionNode.coreClCond_
    ///
    /// The cores and corelets the "then" region applies to (`dsc/dsc2.h:691-692`).
    ///
    /// ⛔ AN ABSENT CORE IS AN EXCLUDED ONE, NOT AN UNCONSTRAINED ONE: `setRelevantCompCoreCl`
    /// INTERSECTS this map into the "then" region's inherited `relevantComps_` and hands the
    /// complement to the "else" region (`dsc/dsc2.cpp:2647-2685`), so an empty map excludes every
    /// core from the "then" side. That is why the corelets are a set per core rather than the pairs
    /// flattened: a core present with no corelets is a different condition from a core absent.
    pub core_cl_cond: BTreeMap<CoreId, BTreeSet<CoreletId>>,
}

// crustify:todo: e025_ConditionNode

// crustify:todo: e025_ConditionNode.loopCond_

/// Replaces: e026_SyncNode
///
/// `dsc/dsc2.h:964-972`. One end of a signal: which units it signals to or waits on, and which end
/// it is. The DDL conversion mints one per `SyncOp` (`ddc/ddl/ddl_conversion.cpp:1708-1732`) and the
/// L3 scheduler mints them in send/receive pairs
/// (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:640-660`).
///
/// ⛔ THIS CARRIES THREE OF SYNCNODE'S FIVE FIELDS, so the `e026_SyncNode` anchor below stays open.
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
#[derive(Clone, Debug, Default)]
pub struct SyncNode {
    /// Field: e026_SyncNode.units_
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
    /// ⚠️ THE REMAINING READS ARE ORDER-FREE, and only two of them are the membership test this
    /// anchor used to claim for all of them: two more iterate but accumulate into sets
    /// (`dsc/dsc2.cpp:2702-2704`, `dsc/dsc2Pcfg.cpp:2050-2053`), and the last two ask an any-of and a
    /// `count` (`ddc/ddc_transformation.cpp:1531-1535`,
    /// `dcg/dcg_fe/pcfg_gen/dlOpsNew.cpp:2650-2651`).
    pub units: BTreeSet<SenComponent>,
    /// Field: e026_SyncNode.isReceive_
    ///
    /// Which end this is (`dsc/dsc2.h:967`): bridge 1 emits a `sync_send` when it is false and a
    /// `sync_recv` when it is true (`SNSyncLowering.cpp:209`, `:239`).
    ///
    /// ⛔ BOTH ARMS ARE ALSO GATED ON A NULL `implicitSyncRefTransfer_`, so this flag selects NEITHER
    /// when one is set: `is_implicit_sync` is that pointer against `nullptr` (`:208`), both arms carry
    /// `&& !is_implicit_sync`, and the third arm reads this field not at all (`:264-269`).
    pub is_receive: bool,
    /// Field: e026_SyncNode.isSoft_
    ///
    /// Whether the send may run ahead of the transfers it covers (`dsc/dsc2.h:967`): bridge 1 sets
    /// the emitted send's `wait_immediately_for_async_transfers` to its NEGATION
    /// (`SNSyncLowering.cpp:210-211`). TWO sites write it — the L3 scheduler's minter
    /// (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:658`) and the JSON importer
    /// (`dsc/dsc2.cpp:1721-1722`) — and the PCFG translator carries it onto both syncs it mints
    /// (`dsc/dsc2Pcfg.cpp:2056-2058`).
    ///
    /// ⚠️ THE SCHEDULER LISTED NO ANCHOR FOR IT: it is declared on the same line as
    /// [`is_receive`](Self::is_receive), and that bridge-1 read is on this campaign's path.
    pub is_soft: bool,
}

// crustify:todo: e026_SyncNode

// crustify:todo: e026_SyncNode.implicitSyncRefTransfer_

// crustify:todo: e026_SyncNode.otherEndOfTheSignals_

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
    ///
    /// The within-slice (wsl) dim's mask (`dsc/dsc2.h:1068`, `dsc/dsc2.cpp:2463-2472`).
    ///
    /// ⚠️ THE SCHEDULER LISTED NO ANCHOR FOR IT, only for its `maskB_` twin; the two are declared on
    /// one line and bridge 1 reads both halves of both (`SNStickMaskLowering.cpp:25-30`).
    pub mask_a: MaskSplit,
    /// Field: e027_StickMaskNode.maskB_
    ///
    /// The cross-slice (xsl) dim's mask (`dsc/dsc2.h:1068`, `dsc/dsc2.cpp:2473-2489`).
    pub mask_b: MaskSplit,
    /// Field: e027_StickMaskNode.transitionSliceId_
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
/// ⛔ NO `PartialEq`: node identity in the authority is the pointer. `Clone` is IBM's own, through
/// `InheritWithClone` (`:1059`), and the DDC leans on it for the reset copy (`ddc/ddcv1.cpp:3663`).
#[derive(Clone, Debug)]
pub struct StickMaskNode {
    /// Field: e027_StickMaskNode.maskValConstId_
    ///
    /// The constant holding the value written into the masked elements (`dsc/dsc2.h:1061`), taken
    /// from `DesignSpaceConfig::maskingConstId_` (`ddc/ddcv1.cpp:3531`).
    ///
    /// ⛔ THE AUTHORITY'S `-1` IS ABSENT, and both readers refuse it rather than indexing with it:
    /// bridge 1 checks `>= 0` before `constantInfo_.at` (`SNStickMaskLowering.cpp:32-36`) and the
    /// PCFG translator repeats the check (`dsc/dsc2Pcfg.cpp:2202-2203`).
    pub mask_val_const_id: Option<ConstantId>,
    /// Field: e027_StickMaskNode.dataFormat_
    ///
    /// The precision of the masked tensor (`dsc/dsc2.h:1062`), copied from the affected transfer's
    /// labeled data structure (`ddc/ddcv1.cpp:3577`).
    ///
    /// ⛔ ITS INITIALISER IS NOT [`DataFormats`]' OWN DEFAULT: this field starts `INVALID` (`:1062`)
    /// where [`ComputeNode::data_format`] starts at fp16, so a node minted without a transfer
    /// carries no width at all — see [`StickMaskNode::default`].
    pub data_format: DataFormats,
    /// Field: e027_StickMaskNode.stickLayout_
    ///
    /// What one stick is made of: each dim inside it with its extent in elements (`dsc/dsc2.h:1063`),
    /// range-built out of `getStickSizes` (`ddc/ddcv1.cpp:3546-3547`) — the conversion [`Size`]
    /// carries a [`From`] impl for.
    ///
    /// ⛔ ITS LAST ENTRY IS THE CROSS-SLICE DIM (`dsc/dsc2.cpp:2445-2446`), which is the whole reason
    /// this is a `Vec` and not a map: [`view`](Self::view) reads the layout's order and its length.
    pub stick_layout: Vec<Size>,
    /// Field: e027_StickMaskNode.firstStickCoordToMaskPerDim_
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

/// `dsc/dsc2.h:1061-1062`: the mask value is absent and the precision is `INVALID`, unlike
/// [`DataFormats`]' own default.
impl Default for StickMaskNode {
    fn default() -> Self {
        Self {
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

// crustify:todo: e027_StickMaskNode.affectedTransfers_

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

    /// The spelling `indirectAllocTypeToString` gives this role (`dsc/dsc2.h:1049-1050`, filled
    /// `dsc/dsc2.cpp:2423-2427`). ⭐ TOTAL, AND THE AUTHORITY'S MAP IS TOO — all three have an entry.
    pub fn name(self) -> &'static str {
        match self {
            Self::NoIndirection => "no_indirection",
            Self::ValueTensor => "value_tensor",
            Self::IndexTensor => "index_tensor",
        }
    }

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

    /// The spelling `indexTensorTypeToString` gives this form (`dsc/dsc2.h:1053-1054`, filled
    /// `dsc/dsc2.cpp:2432-2435`).
    pub fn name(self) -> &'static str {
        match self {
            Self::Address => "address",
            Self::Index => "index",
        }
    }

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
/// RESERVATION to the whole memory capacity, which no other count gets (`ddc/ddcv1.cpp:317-328`),
/// and `processImplicitSync` refuses an implicit sync on anything else, "Implicit syncs are only
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

/// One entry of `AllocateNode::maxDimSizes_` (`dsc/dsc2.h:983`), positionally paired with
/// [`layout_dim_order`](AllocateNode::layout_dim_order).
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
/// ⛔⛔ AND ITS READERS DO NOT AGREE ON WHERE ABSENCE STOPS. [`None`] here is the authority's
/// NEGATIVE entry, which is the boundary `getPageSize` draws (`maxSize < 0` is the unbounded dim,
/// `dsc/dsc2.cpp:4501`) and the boundary both `>= 0` writers draw (`finalizeAllocateLayouts`,
/// `ddc/ddcv1.cpp:1719`, and `ForceInnermostDimensionsOp`'s already-applied refusal,
/// `ddc/ddl/ddl_conversion.cpp:1879-1881`). `buildUnitView`'s is `> 0` —
/// `if (maxDimSize > 0 && size > maxDimSize)` (`dsc/dsc2.cpp:2806`) — so a ZERO entry takes the
/// `else` branch: it is never capped, and it never reaches the `DT_CHECK` that the remainder divides
/// (`:2810`), which is the one guard that would have refused it.
///
/// ⭐ ZERO IS REACHABLE AND IT IS NOT INERT. The pass that fills these entries divides by the
/// cumulative stick size with INTEGER division, so any extent below one stick lands on zero
/// (`ddc/ddcv1.cpp:1723-1729`), and the JSON importer pushes back whatever the dump held
/// (`dsc/dsc2.cpp:1761-1764`). Downstream, that zero is a bound and not an absence: `getPageSize`
/// multiplies it into the dim's page size (`dsc/dsc2.cpp:4508`), and both in-file readers of that map
/// then divide a per-dim size by it under `INDEX_TENSOR` (`dsc/dsc2.cpp:3569`, `:3899`) or
/// `DT_CHECK` `dimSize <= 0` under `VALUE_TENSOR` (`:3572-3573`). So the three boundaries are three
/// different predicates over the same [`Option`], and [`Option::is_some`] is the writers' test, NEVER
/// `buildUnitView`'s cap test.
///
/// ⭐ NEITHER OTHER CURRENCY CAN REACH THE VECTOR:
///
/// ```compile_fail
/// use deeptools::schedule::dsc2::{AllocateNode, DimSize};
/// let mut node = AllocateNode::default();
/// node.max_dim_sizes.push(Some(DimSize(8)));
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MaxDimSize(pub i32);

/// The distance in bytes between one buffer of an allocation and the next — one value of
/// `AllocateNode::bufferOffsetCoreCorelet_` (`dsc/dsc2.h:988`).
///
/// ⛔ A STRIDE, NOT A BASE ADDRESS: `allocAllMem` writes one buffer's own capacity here while the
/// base goes to `startAddressCoreCorelet_` beside it (`ddc/ddcv1.cpp:351-356`), and the L3 scheduler
/// reads the pair together (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:4942-4944`).
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
/// OUTPUT REPETITION COUNT on the innermost layout dim of the cloned compute's output allocation
/// (`ddc/ddc_transformation.cpp:1356`, written at `:1378-1380`) — so a filled entry is not always a
/// stick count, and reading one as the masked pass's `8` would be wrong for every cloned compute.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StickSpread(pub i32);

/// One region of one memory, reserved for one labeled data structure, one constant or one compute
/// temporary — `dsc/dsc2.h:974-1057`. The DDL conversion mints one per `AllocateOp`
/// (`ddc/ddl/ddl_conversion.cpp:780-840`), `allocAllMem` places it (`ddc/ddcv1.cpp:218-360`), and
/// the L3 scheduler reads its addresses back
/// (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:4926-4950`).
///
/// ⛔ THIS CARRIES 15 OF ALLOCATENODE'S 21 FIELDS, so the `e028_AllocateNode` anchor below stays
/// open. Three are schedule-node pointer identity, e013's `name_` and the tree it hangs on:
/// `tempStorageForCompute_` (`:978`), the `ComputeNode` whose temporary this region is;
/// `relatedIndirectAccessAlloc_` (`:999-1001`), the other half of an indirect access; and
/// `allocUsers_` (`:1007`), the reference-counted list of nodes that read or write the region. All
/// three serialize by node name and re-resolve through `nodeNamePtrMap` (`dsc/dsc2.cpp:840-843`,
/// `:919-920`, `:934-944`, `:1743-1745`, `:1798-1801`, `:1811-1823`). The other three need types
/// this campaign has not scoped: `startAddressCoreCorelet_` (`:985-986`) is a
/// `FoldManager<int64_t>`, and `allocateCoordinates_` and `sliceViewCoordinates_` (`:1008-1009`) are
/// `CoordinateType`, e012, which is built on the same `util/foldManager/` — and the authority's own
/// JSON round trip leaves the slice view a "TO DO" on both sides (`dsc/dsc2.cpp:1828`).
///
/// ⛔ AND ITS SEVEN METHODS STAY OUT WITH THOSE FIELDS — but only ONE OF `getPageSize`'S THREE ARMS
/// IS WHAT HOLDS IT OUT (`:1011`, defined `dsc/dsc2.cpp:4480-4513`). `NO_INDIRECTION` answers the
/// empty map (`:4483-4485`) and `VALUE_TENSOR` reads `this` (`:4486-4488`), so both are total in the
/// fields carried here; it is `INDEX_TENSOR` that takes the page extents out of
/// `relatedIndirectAccessAlloc_`'s layout, through the pointer it `DT_CHECK`s non-null
/// (`:4489-4492`), and an answer computed from this node's own layout instead would be silently
/// wrong for exactly the index allocations the paged path mints.
///
/// ⛔ AND IT IS NOT AN L3-ONLY METHOD: of its eleven callers, TWO ARE IN THIS SAME FILE — both
/// `DesignSpaceConfig` methods, which divide a per-dim size by the page size under `INDEX_TENSOR`
/// and bound-check it under `VALUE_TENSOR` (`dsc/dsc2.cpp:3557` and `:3566-3576` in
/// `getBlockTransferSizePerDimCustomLocation`, `:3806` and `:3893-3902` in
/// `getBufferCapacityForNodePerDimCustomLocation`) — and five more are in the L3 scheduler
/// (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:1647`, `:5893`, `:5994`, `:6692`, `:6721`).
/// `addAllocUser`, `removeAllocUser`, `hasAllocUsers`, `hasAllocUser` and `clearAllocUsers`
/// (`:1012-1046`) are that list's five operations, and `print` (`:1048`) streams both.
///
/// ⛔ NO `PartialEq`: node identity in the authority is the pointer, and `allocUsers_` and
/// `relatedIndirectAccessAlloc_` compare by it. `Clone` is IBM's own, through `InheritWithClone`
/// (`:974`).
#[derive(Clone, Debug)]
pub struct AllocateNode {
    /// Field: e028_AllocateNode.ldsIdx_
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
    /// The constant this region holds, or [`None`] for the authority's `-1` (`dsc/dsc2.h:977`). It
    /// indexes `DesignSpaceConfig::constantInfo_`, whose entry supplies the region's name
    /// (`ddc/ddcv1.cpp:26-27`), and the DDL conversion names such a node
    /// `allocate_const<idx>_<component>` (`ddc/ddl/ddl_conversion.cpp:836-838`).
    pub const_idx: Option<ConstantId>,
    /// Field: e028_AllocateNode.component_
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
    /// The padding form of each dim of the region (`dsc/dsc2.h:981`), written from the `AllocateOp`
    /// (`ddc/ddl/ddl_conversion.cpp:793`). `getSizeDataStageForNode` passes it on to size the
    /// allocation (`dsc/dsc2.cpp:3613`).
    pub padding: PaddingFormType,
    /// Field: e028_AllocateNode.layoutDimOrder_
    ///
    /// The dims the region is laid out over, positionally paired with
    /// [`max_dim_sizes`](Self::max_dim_sizes) (`dsc/dsc2.h:982`).
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
    pub layout_dim_order: Vec<PrimaryDimTypes>,
    /// Field: e028_AllocateNode.maxDimSizes_
    ///
    /// One entry per [`layout_dim_order`](Self::layout_dim_order) dim, [`None`] for the authority's
    /// negative "no limit" (`dsc/dsc2.h:983`). Read [`MaxDimSize`] before touching a filled one: the
    /// integer means a data-stage index before `finalizeAllocateLayouts` and an extent after.
    ///
    /// ⛔ ITS LENGTH IS AN INVARIANT, NOT A COINCIDENCE: every producer `resize`s it to
    /// `layoutDimOrder_.size()` with `-1` (`ddc/ddl/ddl_conversion.cpp:803`, `:1641`,
    /// `ddc/ddc_transformation_util.cpp:52`, `ddc/ddc_transformation.cpp:2116`, `:2347`,
    /// `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:561`), `finalizeAllocateLayouts` raises
    /// `DT_ERROR("Mismatch in allocate layout vectors")` when the two differ
    /// (`ddc/ddcv1.cpp:1715-1717`), and `getPageSize` `DT_CHECK`s the same (`dsc/dsc2.cpp:4496`).
    ///
    /// ⛔ A `Vec` OF PAIRS WOULD NOT DO INSTEAD: `SdscCoreletSplit` finds a dim in the layout and
    /// indexes THIS vector by that distance
    /// (`dbo/src/Utils/sdsc_bundle/SdscCoreletSplit.cpp:79`), and the two are linearized as separate
    /// runs into the SuperDsc fingerprint (`dsc/superdsc.cpp:1451-1452`).
    ///
    /// ⛔ A NEGATIVE ENTRY IS ALSO WHAT MAKES A DIM UNBOUNDED IN `getPageSize`, and it wins over
    /// every other entry of the same dim, erasing what earlier positions accumulated
    /// (`dsc/dsc2.cpp:4498-4509`).
    pub max_dim_sizes: Vec<Option<MaxDimSize>>,
    /// Field: e028_AllocateNode.numBuffers_
    ///
    /// How many buffers the region holds (`dsc/dsc2.h:984`); see [`NumBuffers`] for the encoding and
    /// [`NumBuffers::STREAMING`] for the one value bridge 1 tests.
    pub num_buffers: NumBuffers,
    /// Field: e028_AllocateNode.isStartAddrSymbolic_
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
    /// The buffer stride per core and corelet (`dsc/dsc2.h:988`), written by `allocAllMem` beside
    /// the start address (`ddc/ddcv1.cpp:351-356`) and read as `.at(coord.at(0)).at(corelet0Id)`
    /// (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:4942-4944`). See [`BufferOffset`]: it is a stride
    /// in bytes, not a base.
    ///
    /// ⭐ ORDERED, AND THE ORDER IS EXPORTED: the authority's nested `std::map`s print in key order
    /// in the node's JSON (`dsc/dsc2.cpp:879-892`), which a [`BTreeMap`] reproduces. Both keys are
    /// `int` there and non-negative in every writer — `allocAllMem` iterates real cores and corelets
    /// (`ddc/ddcv1.cpp:351-356`) and the L3 scheduler writes `[coreId][coreletId]`
    /// (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:4991`, `:5008`, `:5035`, `:5130`) — so unlike
    /// [`back_gap_core`](Self::back_gap_core) this map has no `-1` pseudo-key and needs no
    /// [`Option`].
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
    /// Which half of an indirect access this region is (`dsc/dsc2.h:990-994`); see
    /// [`IndirectAllocType`].
    pub indirect_alloc_type: IndirectAllocType,
    /// Field: e028_AllocateNode.indexTensorType_
    ///
    /// What an index tensor's entries hold (`dsc/dsc2.h:995-998`); see [`IndexTensorType`]. It is
    /// only meaningful under [`IndirectAllocType::IndexTensor`].
    pub index_tensor_type: IndexTensorType,
    /// Field: e028_AllocateNode.gapStickSpread_
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
    /// ⚠️ THE SCHEDULER LISTED NO ANCHOR FOR IT, and none for
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
}

impl Default for AllocateNode {
    /// The authority's member initialisers (`dsc/dsc2.h:976-1005`).
    ///
    /// ⛔ IT IS NOT `AllocateNode()`: that constructor also passes `ALLOCATE` to the base class
    /// (`dsc/dsc2.h:975`), and `nodeType_` is `ScheduleNode`'s, e013's to port.
    fn default() -> Self {
        Self {
            lds_idx: None,
            const_idx: None,
            component: SenComponent::NoComponent,
            padding: PaddingFormType::default(),
            layout_dim_order: Vec::new(),
            max_dim_sizes: Vec::new(),
            num_buffers: NumBuffers(1),
            is_start_addr_symbolic: false,
            buffer_offset_core_corelet: BTreeMap::new(),
            back_gap_core: BTreeMap::new(),
            indirect_alloc_type: IndirectAllocType::NoIndirection,
            index_tensor_type: IndexTensorType::Address,
            gap_stick_spread: BTreeMap::new(),
            ignore_symbolic_volume_limits: false,
            non_unified_alloc_in_hbm: false,
        }
    }
}

// crustify:todo: e028_AllocateNode

// crustify:todo: e028_AllocateNode.allocUsers_

// crustify:todo: e028_AllocateNode.allocateCoordinates_

// crustify:todo: e028_AllocateNode.sliceViewCoordinates_

// crustify:todo: e028_AllocateNode.startAddressCoreCorelet_

// crustify:todo: e028_AllocateNode.tempStorageForCompute_

/// One constant the program supplies as data rather than reading it out of a tensor —
/// `dsc2::ConstantInfo` (`dsc/dsc2.h:46-61`). It is one entry of
/// `DesignSpaceConfig::constantInfo_`, keyed by the [`ConstantId`] an
/// [`AllocateNode::const_idx`] points back at (`dsc/designSpaceConfig.h:90`).
///
/// ⛔ THIS CARRIES 3 OF CONSTANTINFO'S 5 FIELDS, so the `e028_ConstantInfo` and `e030_ConstantInfo`
/// anchors below stay open. `data_` (`:49-50`) is a `FoldManager<std::vector<int64_t>>`, and
/// `util/foldManager/` is the blocker e008 and e012 are already held by — the re-scheduled
/// `e028_ConstantInfo` no longer lists that field at all, so only the `e030` anchor still names it.
/// `allocations_` (`:52`) is a
/// `std::map<SenComponents, AllocateNode*>` of NON-OWNING aliases into the schedule tree: the DDL
/// conversion hangs the minted node on its parent block and aliases it here in the same breath
/// (`ddc/ddl/ddl_conversion.cpp:826-832`), and the PE/SFP work split clones a node into a second
/// component the same way, refusing a component that already has one
/// (`ddc/ddc_transformation_util.cpp:1407-1417`). What blocks it is that this port has no node
/// IDENTITY to alias WITH: there is no `ScheduleNode` base, no parent or child link, and
/// [`AllocateNode`]'s own pointer fields are open anchors above. An owned
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
/// ⛔ ITS ONE METHOD STAYS OUT WITH THOSE TWO FIELDS: the copy assignment's four member assignments
/// are `dataFormat_`, `name_`, `allocations_` and `data_.clone(rhs.data_)` (`:54-60`), so two of the
/// four are unported. It calls `clone` rather than `data_ = rhs.data_` because `FoldManager`'s own
/// assignment `DT_ERROR`s unless the two fold spaces already agree in dimensionality and cardinality
/// (`util/foldManager/foldInfrastructure.h:922-933`), which a fresh destination never does; `clone`
/// destroys the destination's fold space and rebuilds it (`:987`). See
/// [`is_data_symbolic`](Self::is_data_symbolic) for the field that assignment drops.
///
/// ⛔ NO `PartialEq`: the authority's own duplicate test compares `name_`, `dataFormat_` AND the
/// datum's element count (`ddc/ddl/ddl_conversion.cpp:706-714`), so an equality over the carried
/// fields alone would answer "the same constant" for two constants holding different values.
#[derive(Clone, Debug)]
pub struct ConstantInfo {
    /// Field: e028_ConstantInfo.dataFormat_
    /// Field: e030_ConstantInfo.dataFormat_
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
    /// Field: e028_ConstantInfo.isDataSymbolic_
    /// Field: e030_ConstantInfo.isDataSymbolic_
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
    /// (`dsc-based-utils/DSC2ToDataflowIR/V3/SNDSCLowering.cpp:479-480`, `:514-516`, reached from
    /// `SNTransferLowering.cpp:2492-2496`).
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
    /// (`dsc/dsc2.h:934`), a different field's.
    fn default() -> Self {
        Self {
            data_format: DataFormats::Invalid,
            name: String::new(),
            is_data_symbolic: false,
        }
    }
}

// crustify:todo: e028_ConstantInfo

// crustify:todo: e028_ConstantInfo.allocations_

// crustify:todo: e030_ConstantInfo

// crustify:todo: e030_ConstantInfo.allocations_

// crustify:todo: e030_ConstantInfo.data_

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
/// (`dsc/dsc2.cpp:4651-4665`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WkSliceIdx(pub i64);

/// A chunk index within one work slice — the coordinate of the INNER folded dim (`:4669-4670`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ChunkIdx(pub i64);

/// How many chunks one work-slice pad walk may visit — `numChunks` (`dsc/dsc2.cpp:4686`).
///
/// ⛔ NOT THE STORED `chunk_index` EXTENT, and the walk does not clamp it to one: the caller derives
/// it per dim (`dsc/dsc2.cpp:4823`), and measured, a value past the extent makes the reader throw
/// unless the walk breaks first — immediately on the back end, whose first coordinate is
/// `numChunks - 1`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NumChunks(pub u32);

/// The element offset one fully padded chunk contributes to a work slice's pad (`:4830-4832`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ChunkOffset(pub i32);

/// A chunk's padded extent — both the cap a pad size is legal up to and the threshold that ends the
/// work-slice walk (`dsc/dsc2.cpp:4705-4710`, `:4718-4731`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ChunkSizePadded(pub i32);

/// Which end of a transfer's data a pad sits at — the authority's `const bool isPadFront`
/// (`dsc/dsc2.h:784`, `:794`, `:806`, `:809`), which selects one of two parallel field pairs at
/// every one of its six sites.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PadEnd {
    /// `isPadFront == true`, and the walk visits chunk 0 first (`dsc/dsc2.cpp:4696-4697`).
    Front,
    /// `isPadFront == false`, and the walk visits `numChunks - 1` first.
    Back,
}

/// `TransferPadInfo::FoldDimPosition` (`dsc/dsc2.h:772`) — the two folded dims, outer first.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FoldDimPosition {
    /// `WORK_SLICE_FOLDDIM`, labelled `wkslice_index` (`dsc/dsc2.cpp:4662-4665`).
    WorkSlice = 0,
    /// `CHUNK_FOLDDIM`, labelled `chunk_index` (`:4669-4670`).
    Chunk = 1,
}

impl FoldDimPosition {
    /// `TOTAL_FOLDDIM_NUM` — the arity `buildPadSizes` `DT_CHECK`s on all three of its vectors
    /// (`dsc/dsc2.cpp:4612-4615`), here an array length, so the check is E0308.
    pub const COUNT: usize = 2;

    /// Outer to inner, which is the order `buildFoldSpace` nests the levels in.
    pub const ALL: [Self; Self::COUNT] = [Self::WorkSlice, Self::Chunk];
}

/// ⛔ E0308 IF A POSITION IS EVER ADDED: `COUNT` is the array length every caller's `sizes`,
/// `alphas` and `betas` are checked against.
const _: [(); FoldDimPosition::COUNT] = [(); FoldDimPosition::Chunk as usize + 1];

/// One dim's pad-size fold: the two `FoldDimProp`s the authority stores plus the two-level affine
/// tree its manager builds over them (`dsc/dsc2.cpp:4651-4681`).
///
/// ⛔ THE PROPS AND THE TREE ARE ONE OWNER HERE BECAUSE IN C++ THEY ALIAS: `FoldManager::dim_prop_`
/// holds `const FoldDimProp*` INTO `transferPadFrontFoldProps`
/// (`util/foldManager/foldInfrastructure.h:888`), so `DT_CHECK_MSG(!foldProps.count(dim))`
/// (`dsc/dsc2.cpp:4658`) is the only thing standing between a rebuild's `resize` and a dangling
/// pointer. Co-owning them makes the pointer unnecessary rather than safe.
#[derive(Clone, Debug, PartialEq, Eq)]
struct PadSizeFold {
    props: [FoldDimProp; FoldDimPosition::COUNT],
    fold: FoldFunc<i32>,
}

impl PadSizeFold {
    /// `buildTransferFoldDim` (`dsc/dsc2.cpp:4651-4681`) and the four `insert*ForKey` calls that
    /// follow it (`:4624-4635`), which are one construction: `buildFoldSpace` nests two affine
    /// levels with `alpha_{}`/`beta_{}` at zero and each `insertAlphaForKey(.., pos)` then writes the
    /// level `collectFoldFunctionAtLevel(pos)` reaches — 0 the non-leaf, 1 its leaf.
    fn new(
        sizes: [FoldDimSize; FoldDimPosition::COUNT],
        alphas: [PadSize; FoldDimPosition::COUNT],
        betas: [PadSize; FoldDimPosition::COUNT],
    ) -> Self {
        let outer = FoldDimPosition::WorkSlice as usize;
        let inner = FoldDimPosition::Chunk as usize;
        Self {
            props: [
                FoldDimProp::new(sizes[outer], "wkslice_index"),
                FoldDimProp::new(sizes[inner], "chunk_index"),
            ],
            fold: FoldFunc::AffineNonLeaf(AffineFoldFunctionNonLeaf::new(
                alphas[outer].0,
                betas[outer].0,
                FoldFunc::AffineLeaf(AffineFoldFunctionLeaf::new(alphas[inner].0, betas[inner].0)),
            )),
        }
    }

    /// `MapWithFMHelper::getDataForKey` past the key check (`util/mapWithFMHelper.h:143-147`) — the
    /// manager's `isLegal` range test (`util/foldManager/foldInfrastructure.h:1666-1681`) and then
    /// the walk.
    ///
    /// ⛔ THE RANGE TEST IS SIGNED AND THAT IS NOT A BUG TO FIX: `getSize() <= idx` widens a
    /// `uint32_t` extent to `int64_t` (`:1677`), so a NEGATIVE coordinate is LEGAL and computes.
    /// Measured on the authority: work slice -1 answers 30 where work slice 2 of 2 throws.
    /// ⛔ The count half of `isLegal` is gone instead of ported — two coordinates is the signature.
    fn data(&self, wk_slice: WkSliceIdx, chunk: ChunkIdx) -> Option<PadSize> {
        let coords = [FoldDimIndex(wk_slice.0), FoldDimIndex(chunk.0)];
        for (prop, coord) in self.props.iter().zip(coords) {
            if i64::from(prop.size().0) <= coord.0 {
                return None;
            }
        }
        self.fold.get_data(&coords).map(PadSize)
    }
}

/// Replaces: e024_TransferPadInfo
///
/// Replaces: e008_TransferPadInfo
///
/// A transfer's LX zero-pad sizes, one two-level affine fold per padded dim per end
/// (`dsc/dsc2.h:755-812`) — what `L3DlOpsScheduler` writes onto a `TransferNode` so that
/// `dsc/dsc2.cpp:4768-4990` can turn padding into condition and transfer nodes. e008 is this same
/// class under the superseded numbering, which listed two of its six fields.
///
/// ⛔ NO [`Clone`], AND THE ABSENCE IS THE `DT_CHECK`: the authority's copy constructor is
/// "Do nothing on purpose" (`dsc/dsc2.h:764-767`) — it rebuilds the two helper references and copies
/// NOTHING, so a copy is EMPTY. `dsc/dsc2.cpp:5700` clones a padded transfer node and `:5792`
/// `DT_CHECK_MSG`s the clone `isEmpty()`; measured, source non-empty and copy empty. A `Clone` that
/// silently dropped the folds would be the astonishing one, so the only way to spell that copy here
/// is [`Default`], which makes IBM's runtime check a fact of the type.
/// ⛔ AND ITS DEFAULTED MOVE CONSTRUCTOR (`dsc/dsc2.h:763`) IS A USE-AFTER-FREE, unrepresentable
/// here: `MapWithFMHelper`'s only member is a REFERENCE to the sibling map
/// (`util/mapWithFMHelper.h:36-38`), so a move copies a reference that still points into the
/// moved-FROM object. Measured — the moved-to object reported `isEmpty() == 0` with zero keys,
/// answered the moved-FROM object's data after that object was rebuilt, and destroying the source
/// gave a `heap-use-after-free` under AddressSanitizer inside `getAllKeys`. A Rust move carries the
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
    /// Field: e008_TransferPadInfo.transferPadFrontSize_
    ///
    /// ⛔ THREE OF THE AUTHORITY'S FIELDS ARE ONE FIELD HERE, and the helper is not a field at all:
    /// `transferPadFrontSizeHelper` is a `MapWithFMHelper` whose ONLY member is
    /// `std::map<Dkey, FoldManager<Dval>>& key_val_` (`util/mapWithFMHelper.h:36-38`) bound to
    /// `transferPadFrontSize_` in every constructor (`dsc/dsc2.h:759-767`) — a facade over the
    /// sibling map, holding no state of its own. `transferPadFrontFoldProps` is then the storage that
    /// map's `dim_prop_` pointers point INTO; see [`PadSizeFold`].
    front: BTreeMap<PrimaryDimTypes, PadSizeFold>,
    /// Field: e024_TransferPadInfo.transferPadBackFoldProps
    ///
    /// Field: e024_TransferPadInfo.transferPadBackSize_
    ///
    /// Field: e024_TransferPadInfo.transferPadBackSizeHelper
    ///
    /// Field: e008_TransferPadInfo.transferPadBackSize_
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
    /// (`:4658`) — a dim can be built once per end, and the authority throws on the second attempt
    /// rather than rebuilding. Its other three checks are gone into the array lengths, and a negative
    /// extent — `setSize(int)` onto a `uint32_t` (`util/foldManager/foldInfrastructure.h:129`, `:153`)
    /// — is unspellable in [`FoldDimSize`].
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
                slot.insert(PadSizeFold::new(sizes, alphas, betas));
                Some(())
            }
        }
    }

    /// `getPadFrontOrBackDimsSet` (`dsc/dsc2.h:783-788`) through `MapWithFMHelper::getAllKeys`
    /// (`util/mapWithFMHelper.h:51-57`).
    ///
    /// ⛔ AN ORDERED ITERATOR RATHER THAN A `std::set` BY VALUE, WHICH IS WHAT THE ONE CALLER WANTS:
    /// it `std::set_union`s the two ends into a vector (`dsc/dsc2.cpp:4813-4819`), so it needs the
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
    /// ⛔ [`None`] IS THE READER THROWING MID-WALK, on an unknown dim (`util/mapWithFMHelper.h:144`)
    /// or a coordinate past its extent — reachable exactly when [`NumChunks`] exceeds the stored
    /// `chunk_index` extent and no chunk breaks the walk first.
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
            // "the agreement is that negative pad size is treated as zero" (`:4705-4708`).
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
        // `int` arithmetic on `int` inputs (`:4715`), so it wraps where the authority's does.
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
        // `currPadSize < chunkSizePadded` (`:4709`) calls that FULLY padded — so the walk carries on
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
    /// authority's own comment claims (`dsc/dsc2.cpp:4721-4728`), and [`Ord::clamp`] would panic.
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
}
