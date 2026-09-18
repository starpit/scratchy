//! Re-ported from the C++ authority. See crustify-scheduler/AGENT-BRIEF.md.

use crate::schedule::dims::{DataStructDims, PaddingFormType, PrimaryDimAndKind, PrimaryDimTypes};
use std::collections::{BTreeMap, BTreeSet};
use sys_arch_spec::arch_enums::{DataLocation, SenComponent};
use sys_arch_spec::fields::Gen;
use sys_arch_spec::{CoreId, CoreletId, SFP_SLICES};

/// A group tag register's group id — `gtrIdsUsed_` holds the set of them (`dsc/dsc2.h:35`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GroupId(pub u32);

/// How many cores share one group tag register (`dsc/dsc2.h:36`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NumSharers(pub u32);

/// A fold level's coordinate stride per step — `CoordinateBaseType` (`dsc/dsc2.h:442`, `int64_t`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Alpha(pub i64);

/// A fold level's coordinate offset — `CoordinateBaseType` (`dsc/dsc2.h:442`, `int64_t`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Beta(pub i64);

/// A fold level's trip count, as `FoldManager::getFoldDimSize` reports it (`dsc/dsc2.h:1083`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Cardinality(pub i64);

/// A loop's element stride after distribution, read straight into `loopEleOffsets_`
/// (`ddc/ddcv1.cpp:2455-2460`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TemporalStride(pub i64);

/// An element-arrangement fold level, counted outermost-first as `currElemArrLevel`
/// (`ddc/ddc_fold.cpp:2689`).
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
/// ⭐ UNSIGNED BECAUSE EVERY PRODUCER IS: the two DDC writers pass the stick-size loop counter
/// (`ddc/ddcv1.cpp:524-525`) or a layout position offset past the stick dims (`:1591-1598`), and
/// bridge 1 passes `i` (`SNTransferLowering.cpp:631`). The authority's `-1` initialiser is [`None`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SizeIdx(pub u32);

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
/// GTR (group tag register) info, one per core on a transfer node (`dsc/dsc2.h:34-37`).
/// TRAP: the authority's `= -1` on both fields is the ABSENT encoding, not a value. `groupId_`
/// stays -1 whenever there is only one sharer, i.e. no share
/// (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:4831-4832` and `:5255-5256`), and the JSON round
/// trip writes and reads that -1 verbatim (`dsc/dsc2.cpp:631-632`, `:1594-1597`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GroupTagRegInfo {
    /// Field: e006_GroupTagRegInfo.groupId_
    pub group_id: Option<GroupId>,
    /// Field: e006_GroupTagRegInfo.numSharers_
    pub num_sharers: Option<NumSharers>,
}

// crustify:todo: e007_ScheduleTree

// crustify:todo: e007_ScheduleTree.head_

// crustify:todo: e008_TransferPadInfo

// crustify:todo: e008_TransferPadInfo.transferPadBackSize_

// crustify:todo: e008_TransferPadInfo.transferPadFrontSize_

/// Replaces: e009_FoldParamInfoType
///
/// One fold level's affine parameters, trip count and label (`dsc/dsc2.h:1081-1085`).
/// TRAP: `cardinality` defaults to 0, not 1, so the default value is an EMPTY fold and not an
/// identity one; `getDefaultRowSplitFold` writes the identity explicitly
/// (`ddc/ddc_fold.cpp:2154-2159`).
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
/// -1, and `relatedElemArrLevel` is used as `allocFm.getNumDims() - level - 1`
/// (`dsc/dsc2.cpp:6728`), so an unset level silently indexes off the end.
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
/// and the whole of one unit-time transfer chunk's `sizeDim_` (`:821`).
///
/// ⛔ NO DEFAULT, DELIBERATELY, AND THE AUTHORITY'S ONE IS A HAZARD. `Size` is the only
/// dim-carrying struct in this header whose `dim_` has no member initialiser (`:487`) — the
/// `LoopInfo` beside it initialises its own to `PrimaryDimTypesCount` (`:502`). So `Size() =
/// default` (`:490`) leaves `dim_` indeterminate under default-initialisation and zero — the `in`
/// dim, *not* `PrimaryDimTypesCount` — under the value-initialisation that
/// `sizesNoGaps_.emplace_back()` performs. Its only uses are that JSON-import placeholder
/// (`dsc/dsc2.cpp:1304`, `:1321`, `:1849`), overwritten field by field by `importSize` on the same
/// call. Every site that means a value calls the two-argument form (`dsc/dsc2.cpp:2780`, `:2816`),
/// so the placeholder is not ported and this cannot be spelled:
///
/// ```compile_fail
/// use deeptools::schedule::dsc2::Size;
/// let _ = Size::default();
/// ```
///
/// ⛔ AND `size_ = -1` IS NOT A SIZE. Every reader multiplies it — `newDimSizeSoFar *= size_`
/// (`dsc/dsc2.cpp:2739`), `loopStride *= size_` (`:2962`), `elements *= dim_size.sizeDim_.size_`
/// (`dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:34`) — so the declared `-1` is
/// only ever the placeholder's, and no live `Size` carries it. That is why it is a plain
/// [`DimSize`] and not an `Option`: unlike `e006`'s and `e010`'s `-1`s there is no absent state to
/// carry, because there is no constructible absent `Size`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Size {
    /// Field: e013_ScheduleNode.dim_
    ///
    /// ⚠️ THE SCHEDULER'S `.dim_` ANCHOR COVERS TWO C++ FIELDS. This is `ScheduleNode::Size::dim_`
    /// (`dsc/dsc2.h:487`). `ScheduleNode::UnitView::LoopInfo::dim_` (`:502`) spells the same name
    /// and is a different field with a different default; it is not ported — see the still-open
    /// `e013_ScheduleNode` anchor below.
    pub dim: PrimaryDimTypes,
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

    /// `dsc/dsc2.h:35-36`: both fields start absent, and the two producer sites leave `groupId_`
    /// absent for a single sharer (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:4831-4832`).
    #[test]
    fn group_tag_reg_info_starts_absent_and_a_lone_sharer_gets_no_group() {
        assert_eq!(
            GroupTagRegInfo::default(),
            GroupTagRegInfo {
                group_id: None,
                num_sharers: None
            }
        );

        // `numSharers_ = first; groupId_ = first > 1 ? second : -1;`
        let lone = GroupTagRegInfo {
            group_id: None,
            num_sharers: Some(NumSharers(1)),
        };
        let shared = GroupTagRegInfo {
            group_id: Some(GroupId(7)),
            num_sharers: Some(NumSharers(4)),
        };
        assert_eq!(lone.group_id, None);
        assert_eq!(shared.group_id, Some(GroupId(7)));
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

    /// `dsc/dsc2.h:43` against the four sites that write `el_.name_ = ss_.name_ + "el"`
    /// (`ddc/ddcv1.cpp:1236-1237`, `ddc/ddc_transformation.cpp:1018-1019`,
    /// `ddc/ddc_transformation_util.cpp:121-122`, `:131-132`) and against `fillLoopLatchSdsc`, which
    /// names both halves `"core"` (`dbo/src/Utils/sdsc_bundle/ProgramCorrection.cpp:1074-1075`).
    #[test]
    fn a_data_stages_name_is_its_steady_states_and_the_epilogue_carries_its_own() {
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
        let core = DataStage {
            ss: DataStructDims {
                name: "core".to_string(),
                ..DataStructDims::default()
            },
            el: DataStructDims {
                name: "core".to_string(),
                ..DataStructDims::default()
            },
        };
        assert_eq!(core.name(), "core");
        assert_eq!(core.el.name, core.name());
    }

    /// `dsc/dsc2.h:1088`: the declared order, the field's `NOT_PROCESSED` initialiser (`:1093`), and
    /// the two states the queue actually writes (`ddc/ddc.h:468-469`, `:485-486`).
    #[test]
    fn the_prop_state_discriminants_are_the_authoritys_and_overridden_is_never_entered() {
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
        assert_ne!(PropStateType::Complete, PropStateType::RolledBack);
    }

    /// The two shapes `ddc/ddl/ddl_conversion.cpp:1076-1164` mints, against the declared defaults
    /// (`dsc/dsc2.h:573-578`, `:617-618`) and the dummy loop at
    /// `dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:734`.
    #[test]
    fn a_minted_loop_is_either_a_datastage_loop_or_a_parametric_one_and_never_both() {
        use crate::schedule::dims::MetaDimKind;

        // `new dsc2::LoopNode()` (`ddl_conversion.cpp:1077`): every id and index absent, no dims.
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

        // The importer feeds the exported `-1` back through the same setter (`dsc/dsc2.cpp:1414`).
        parametric.set_parametric_lds_idx(None);
        assert_eq!(parametric.parametric_lds_idx(), None);

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

    /// `dsc/dsc2.h:575` and `:595-597` against `dsc/dsc2.cpp:4223-4228`, and against the two things
    /// bridge 1 reads them for (`SNControlFlowLowering.cpp:898-923`).
    #[test]
    fn dims_run_inner_to_outer_and_a_dim_with_an_empty_symbol_list_still_reads_symbolic() {
        let mut loop_node = LoopNode::new(
            Some(DataStageId(0)),
            Some(DataStageId(1)),
            vec![PrimaryDimTypes::Ij.into(), PrimaryDimTypes::Mb.into()],
            false,
        );

        // Inner to outer, so the LAST entry is the loop bridge 1 emits OUTERMOST.
        assert_eq!(
            loop_node.dims.first().map(|d| d.dim),
            Some(PrimaryDimTypes::Ij)
        );
        assert_eq!(
            loop_node.dims.last().map(|d| d.dim),
            Some(PrimaryDimTypes::Mb)
        );

        // `hasLoopDim` ignores the kind and answers for the dim alone.
        assert!(loop_node.has_loop_dim(PrimaryDimTypes::Mb));
        assert!(!loop_node.has_loop_dim(PrimaryDimTypes::Ki));

        // Nothing is symbolic until `finalizeScheduleTree` fills the map (`dsc/dsc2.cpp:2992-3010`).
        assert!(!loop_node.is_dim_symbolic(PrimaryDimTypes::Ij));
        loop_node
            .loop_count_symbol_ids
            .insert(PrimaryDimTypes::Ij, vec![VariableSymbol(7)]);
        assert!(loop_node.is_dim_symbolic(PrimaryDimTypes::Ij));
        assert!(!loop_node.is_dim_symbolic(PrimaryDimTypes::Mb));

        // ⛔ The `operator[]` write at `:3001` leaves an EMPTY vector for a dim with no mapping, and
        // `isDimSymbolic` still answers yes — bridge 1's `DT_CHECK(size() == 1)` is what then fails.
        loop_node
            .loop_count_symbol_ids
            .insert(PrimaryDimTypes::Mb, Vec::new());
        assert!(loop_node.is_dim_symbolic(PrimaryDimTypes::Mb));
        assert!(loop_node.loop_count_symbol_ids[&PrimaryDimTypes::Mb].is_empty());
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
    /// (`ddc/ddl/ddl_conversion.cpp:1902-1905`), over an allocation the DDL conversion sized with
    /// `-1`s (`:803`). The pass refuses to run twice by testing `any_of(maxDimSizes_, >= 0)`
    /// (`:1879-1884`), which is [`Option::is_some`] here.
    #[test]
    fn forcing_inner_dims_prepends_to_both_vectors_and_is_refused_twice() {
        let mut node = AllocateNode {
            layout_dim_order: vec![PrimaryDimTypes::Y, PrimaryDimTypes::Out],
            ..AllocateNode::default()
        };
        node.max_dim_sizes.resize(node.layout_dim_order.len(), None);
        assert!(!node.max_dim_sizes.iter().any(Option::is_some));

        // `:1902-1905`: the forced dim becomes the innermost, and it carries a data stage index —
        // not an extent — until `finalizeAllocateLayouts` overwrites it (`ddc/ddcv1.cpp:1710-1732`).
        node.layout_dim_order.insert(0, PrimaryDimTypes::In);
        node.max_dim_sizes.insert(0, Some(MaxDimSize(2)));
        assert_eq!(node.layout_dim_order[0], PrimaryDimTypes::In);
        assert_eq!(node.max_dim_sizes.len(), node.layout_dim_order.len());
        assert!(node.max_dim_sizes.iter().any(Option::is_some));
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

// crustify:todo: e013_ScheduleNode

// crustify:todo: e013_ScheduleNode.compositeLoops_

// crustify:todo: e013_ScheduleNode.elemOffset_

// crustify:todo: e013_ScheduleNode.loop_

// crustify:todo: e013_ScheduleNode.name_

// crustify:todo: e013_ScheduleNode.nodeType_

// crustify:todo: e013_ScheduleNode.outerLoops_

// crustify:todo: e013_ScheduleNode.prev_

// crustify:todo: e013_ScheduleNode.relevantComps_

// crustify:todo: e013_ScheduleNode.sizeIdx_

// crustify:todo: e013_ScheduleNode.sizesNoGaps_

// crustify:todo: e013_ScheduleNode.sizesWithGaps_

/// Replaces: e014_DataStage
///
/// `dsc/dsc2.h:39-44`. One data stage's two halves — the steady-state dims and the epilogue dims of
/// the same data structure. `DesignSpaceConfig::dataStageParam_` keys them by id
/// (`dsc/designSpaceConfig.h:105`); id 0 is the core stage, whose name `getSizeDataStageForNode`
/// `DT_CHECK`s to be `"core"` (`dsc/dsc2.cpp:3638-3639`).
///
/// ⛔ [`name`](Self::name) IS THE STEADY STATE'S NAME ALONE, and the epilogue carries a different
/// one. Four ddc sites write `el_.name_ = ss_.name_ + "el"` (`ddc/ddcv1.cpp:1236-1237`,
/// `ddc/ddc_transformation.cpp:1018-1019`, `ddc/ddc_transformation_util.cpp:121-122`, `:131-132`)
/// while `fillLoopLatchSdsc` writes `"core"` into both halves
/// (`dbo/src/Utils/sdsc_bundle/ProgramCorrection.cpp:1074-1075`), so the suffix is a ddc convention
/// and not an invariant of this type.
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
    /// then shrinks only the dims the metadata calls relevant (`ddc/ddcv1.cpp:1230-1327`), so an
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
/// `COMPLETE` (`:468-469`) and `rollbackToPos` marks `ROLLED_BACK` (`:485-486`).
///
/// ⛔ `OVERRIDDEN` IS A STATE THE SCHEDULER NEVER ENTERS: `dsc/dsc2.h:1088` is its only occurrence
/// tree-wide, with no writer and no reader. It is ported because it holds the discriminant
/// `COMPLETE` sits behind.
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
/// ⛔ THE THREE SHAPES BELOW ARE DISJOINT AND THIS TYPE CANNOT ENFORCE IT — IBM declares four
/// independent fields and the JSON importer writes them one entry at a time
/// (`dsc/dsc2.cpp:1405-1426`), so a tagged enum would have to reject a legal intermediate state:
///  * a datastage loop has both ids and no `parametricLdsIdx_` (`ddc/ddl/ddl_conversion.cpp:1096-1103`);
///  * a parametric loop has NEITHER id, exactly one dim and a `parametricLdsIdx_` (`:1127-1161`),
///    and cannot also be symbolic (`ddc/ddcv1.cpp:2830-2831`);
///  * the tree's head has `denId_` alone — `ScheduleTree()` writes the core stage into it and
///    leaves `numId_` absent (`dsc/dsc2.h:629`).
///
/// ⛔ NO `PartialEq`: IBM declares none, and the node identity every consumer uses is the POINTER —
/// `loop_labels_`, `dsc_loops_to_mlir_loops_map_` and `LoopDistributionParamPerLoopType` are all
/// keyed by `const LoopNode*` (`dsc/dsc2.h:1118`).
#[derive(Clone, Debug, Default)]
pub struct LoopNode {
    /// Field: e017_LoopNode.numId_
    ///
    /// The numerator stage (`dsc/dsc2.h:573`). ⛔ `-1` IS ABSENT, NOT A STAGE, and readers test for
    /// it: `exploreAssignDataStages` guards every use with `parent->denId_ >= 0`
    /// (`ddc/ddcv1.cpp:634`, `:645`) and `parametricIterCount` errors out on a parent whose id is
    /// negative (`dsc/dsc2.cpp:4144-4147`).
    pub num_id: Option<DataStageId>,
    /// Field: e017_LoopNode.denId_
    ///
    /// The denominator stage (`dsc/dsc2.h:574`), absent on the same terms as [`num_id`](Self::num_id).
    pub den_id: Option<DataStageId>,
    /// Field: e017_LoopNode.dims_
    ///
    /// ⛔ ORDERED INNER TO OUTER (`dsc/dsc2.h:575`), and bridge 1 depends on that: it walks
    /// `dim_idx` from `dims_.size() - 1` down to 0 and opens each MLIR loop inside the one before,
    /// so the LAST entry becomes the OUTERMOST loop of the emitted nest
    /// (`dsc-based-utils/DSC2ToDataflowIR/V3/SNControlFlowLowering.cpp:898-957`).
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
    /// Private in IBM's declaration (`dsc/dsc2.h:617`); every friend reaches it through
    /// [`is_parametric_loop`](Self::is_parametric_loop) or
    /// [`mark_as_parametric_loop`](Self::mark_as_parametric_loop), the JSON exporter included
    /// (`dsc/dsc2.cpp:413-414`).
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
    /// from the two stages' dims.
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
    /// that every non-parametric loop exports (`dsc/dsc2.cpp:421-422`, `:1414-1415`).
    pub fn set_parametric_lds_idx(&mut self, idx: Option<LdsIdx>) {
        self.parametric_lds_idx = idx;
    }

    /// `dsc/dsc2.cpp:4223-4228`. Whether `dim` is one of this loop's, the kind ignored. Its one
    /// caller is `ScheduleNode::getParentDimLoop`, which climbs owner loops until one answers yes
    /// (`dsc/dsc2.cpp:1905-1913`).
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
/// ⛔ THE DISCRIMINANTS ARE THE AUTHORITY'S: `condOpToString` is a `std::map` keyed by this enum
/// (`dsc/dscdefn.h:110`), so the declaration order is its iteration order. Neither map is iterated
/// for output — every use is `.at()` or `.find()` (`dsc/dsc2.cpp:462`, `:1442`,
/// `ddc/ddl/ddl_conversion.cpp:254`, `:3273`) — so only the mapping is ported.
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

/// ⛔ E0080 IF AN OPERATOR IS EVER INSERTED, DROPPED, REORDERED OR LEFT OUT OF `ALL`: the
/// discriminants are what `condOpToString`'s `std::map` orders by, and `ALL` is positional against
/// them.
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
/// ⛔ IT IS A SET MEMBERSHIP TEST, NOT A PROPERTY OF THE LOOP. `getEnclosingLoopsAndRelatedDims`
/// tags a chain entry `BELOW_CHUNK` when the loop is in the caller's `loopsBelowChunkBoundary` set
/// and `ABOVE_CHUNK` otherwise (`dsc/dsc2.cpp:6587`, the lambda at `:6593-6605` and its ternary at
/// `:6601-6603`), and `ddc/ddc_fold.cpp:3512-3517` writes the same ternary again. The same loop can
/// therefore be tagged either way in two different chains: the tag belongs to the chain, not the
/// loop.
///
/// ⛔ `CORELET_SLICE` IS A SYNTHETIC CHAIN ENTRY, NOT A REAL LOOP'S TAG. Its one writer inserts an
/// extra entry for the corelet-split dim at the first loop at or above the chunk boundary
/// (`dsc/dsc2.cpp:6616-6623`), and every reader branches on it to take a different path from the one
/// a real loop gets (`dsc/dsc2.cpp:6029`, `:6063`, `:6067`, `:6340`, `ddc/ddc_fold.cpp:2547`,
/// `:3222`).
///
/// ⛔ `UNKNOWN` IS UNREACHABLE BY CONSTRUCTION, so the `cat = UNKNOWN` initialiser at
/// `dsc/dsc2.h:1144` is dead: `LoopDistributionInfo`'s only constructor takes the category
/// (`:1139-1141`) and, being user-declared, suppresses the implicit default constructor — and
/// `dsc/dsc2.h:1144` is the sole occurrence of the name `UNKNOWN` tree-wide. That is why this enum
/// derives no `Default` even though the authority writes one, exactly as [`Size`] does not:
///
/// ```compile_fail
/// use deeptools::schedule::dsc2::LoopDistributionCat;
/// let _ = LoopDistributionCat::default();
/// ```
///
/// It is still ported as an enumerator, because it holds the discriminant the other three sit
/// behind.
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
/// ⛔ `nonCoreletMemories` AND `directAddressableMemories` ARE DIFFERENT, SMALLER SETS declared
/// beside it (`dsc/dscdefn.h:519-520`, filled `dsc/dscdefn.cpp:145-150`) — no unit on this worklist
/// reads either, so neither is ported here and neither may be substituted for this one.
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
    /// Field: e023_TransferNode.loc_
    ///
    /// Where this destination writes (`dsc/dsc2.h:817`). Its `storage` is what decides whether the
    /// destination is a memory at all (`dsc/dsc2.cpp:4372-4383`).
    pub loc: DataLocation,
    /// Field: e023_TransferNode.locIndirect_
    ///
    /// The location holding the address when this destination is indirect, or
    /// [`DataLocation::UNSET`] when it is not —
    /// [`is_dst_indirect_at_index`](TransferNode::is_dst_indirect_at_index) tests the `unit` half
    /// against `NO_COMPONENT` (`dsc/dsc2.h:879-883`).
    pub loc_indirect: DataLocation,
    /// Field: e023_TransferNode.via_
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
    /// Field: e023_TransferNode.sizeDim_
    ///
    /// The dim and how much of it this chunk covers (`dsc/dsc2.h:821`). ⛔ MUTATED IN PLACE by the
    /// DDC: the 4B-splat path multiplies the first entry's extent by 4 (`ddc/ddcv1.cpp:544-546`) and
    /// the hole split sets an entry's extent to 1 before moving it to the strides (`:1637-1640`).
    pub size_dim: Size,
    /// Field: e023_TransferNode.srcSizeIdx_
    ///
    /// This dim's position in the SOURCE's view sizes, absent as the authority's `-1`
    /// (`dsc/dsc2.h:822`). Bridge 1's load path searches for the entry whose index equals the
    /// position it is emitting (`SNTransferLowering.cpp:338-341`), so an absent index simply never
    /// matches — and the miss is a live answer there, not a refusal (`:349-370`).
    pub src_size_idx: Option<SizeIdx>,
    /// Field: e023_TransferNode.dstSizeIdx_
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
/// (`ddc/ddcv1.cpp:448`, `:471-475`, `:1050`, `:2325-2327`,
/// `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:4135`, `:7108`, `:7886`, `dsc/dsc2.cpp:3483-3501`).
///
/// ⛔ `INVALID_TRANSFER_TYPE` IS REACHABLE AND MEANS "NEITHER END IS A TENSOR": it is the
/// fallthrough of the five tests (`dsc/dsc2.h:895`), and
/// `getBlockTransferSizePerDimCustomLocation` answers with an empty size map on it rather than
/// refusing (`dsc/dsc2.cpp:3483-3485`).
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

/// `dsc/dsc2.h:814-898`. A transfer of one data stage from one source to one or more destinations —
/// the node bridge 1 lowers into an `agen` load or store. Its minting site is
/// `ddc/ddl/ddl_conversion.cpp:1166-1195`: a DDL `DataTransferOp` becomes one of these, with one
/// [`DstVia`] per declared destination.
///
/// ⛔ THIS CARRIES TEN OF TRANSFERNODE'S OWN TWENTY DECLARED FIELDS AND NOTHING INHERITED, so the
/// `e023_TransferNode` anchor at the end of this file is still open. The other ten, with the reason:
///  * `srcLdsAndLoopOffsets_`, `srcIndirectLdsAndLoopOffsets_` (`:832`) and
///    `dstLdsAndLoopOffsets_`, `dstIndirectLdsAndLoopOffsets_` (`:833`) are `DataInfo` — e019;
///  * `lastFusableParentLoopSrc_` (`:830`) and `lastFusableParentLoopDst_` (`:831`) are
///    `const LoopNode*` held as POINTER IDENTITY, which needs e013's `name_`, exactly as e018 does;
///  * `paddingInfo_` (`:845`) is `TransferPadInfo` — e008, blocked on the unscoped
///    `util/foldManager/`;
///  * `coreletViews_` (`:851`) is a map of `CoreletView`, four `UnitView`s (`:847-850`) — e013;
///  * `transferCoordinates_` (`:852`) is `CoordinateType<CoordinateBaseType>` — e012;
///  * `repetition_` (`:826-829`) is an UNNAMED struct with no reader tree-wide. Its only writers are
///    `repetition_.srcRep_ =` (`ddc/ddl/ddl_conversion.cpp:1171`) and
///    `repetition_.dstReps_.push_back` (`:1189`); nothing reads either member, the JSON round trip
///    does not carry them (the `"repetition_"` entries at `dsc/dsc2.cpp:132` and `:1172` are
///    `ComputeNode::instrAttribute_.repetition_`, `dsc/dsc2.h:907`), and `srcRep_` has no member
///    initialiser, so a default-constructed node's copy is indeterminate. Carrying it means naming
///    C++'s unnamed struct, so it is named here instead.
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
///
/// ⛔ NO `PartialEq`, as [`LoopNode`] has none: IBM declares no `operator==` and every consumer
/// keys on the POINTER. `Clone` is IBM's own, through `InheritWithClone` (`dsc/dsc2.h:814`).
#[derive(Clone, Debug)]
pub struct TransferNode {
    /// Field: e023_TransferNode.src_
    ///
    /// Where the data comes from (`dsc/dsc2.h:824`), written by `setDataLocAndInfo`
    /// (`ddc/ddl/ddl_conversion.cpp:1169`).
    pub src: DataLocation,
    /// Field: e023_TransferNode.srcIndirect_
    ///
    /// The location holding the source address when the read is indirect, or
    /// [`DataLocation::UNSET`] when it is direct (`dsc/dsc2.h:824`).
    /// [`is_src_indirect`](Self::is_src_indirect) is the test every reader applies before touching
    /// it (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:6302-6305`).
    pub src_indirect: DataLocation,
    /// Field: e023_TransferNode.dstVias_
    ///
    /// One entry per destination, in the DDL's declaration order (`dsc/dsc2.h:825`,
    /// `ddc/ddl/ddl_conversion.cpp:1175-1190`). Several entries is a multicast; the row-expansion
    /// path rejects several destinations at once (`:1183-1187`).
    pub dst_vias: Vec<DstVia>,
    /// Field: e023_TransferNode.replicationFactor_
    ///
    /// How many times the loaded chunk is splatted, `1` for no splat (`dsc/dsc2.h:834`). Bridge 1
    /// divides the recorded stick and element counts by it, and refuses an LXLU splat it cannot
    /// express (`dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:728-729`, `:963-971`).
    pub replication_factor: i32,
    /// Field: e023_TransferNode.unitTimeTransferChunkSize_
    ///
    /// "Continuous elements within a stick" (`dsc/dsc2.h:835-836`) — the contiguous dims of one
    /// unit-time transfer, ordered as the stick sizes are and then extended with layout dims
    /// (`ddc/ddcv1.cpp:524-525`, `:1597-1598`). Bridge 1 multiplies the extents into the element
    /// count of one `agen` access (`SNTransferLowering.cpp:33-38`).
    pub unit_time_transfer_chunk_size: Vec<SizeAndIndex>,
    /// Field: e023_TransferNode.unitTimeTransferNumChunks_
    ///
    /// How many chunks one unit-time transfer covers, `1` for a single contiguous chunk
    /// (`dsc/dsc2.h:837`). It is the product of the extents the hole split moved out of
    /// [`unit_time_transfer_chunk_size`](Self::unit_time_transfer_chunk_size)
    /// (`ddc/ddcv1.cpp:1633-1642`), which bridge 1 multiplies back in
    /// (`SNTransferLowering.cpp:33-38`).
    pub unit_time_transfer_num_chunks: i32,
    /// Field: e023_TransferNode.unitTimeTransferChunkStride_
    ///
    /// The dims the chunks stride over — the entries the hole split removed from
    /// [`unit_time_transfer_chunk_size`](Self::unit_time_transfer_chunk_size), each with its extent
    /// set to 1 (`dsc/dsc2.h:838`, `ddc/ddcv1.cpp:1635-1642`).
    ///
    /// ⛔ NO LIVE C++ PRODUCER FILLS THIS. Its one writer is inside the lambda
    /// `checkAndResetUnitTimeTransfer` (`ddc/ddcv1.cpp:1640`), whose sole callsite is commented out
    /// (`:1648-1650`), so outside the JSON importer (`dsc/dsc2.cpp:1573-1584`) it is always empty —
    /// which is why the `size() <= 1` check at `dsc/dsc2.cpp:3555` never fires and why bridge 1's
    /// own refusal of more than one stride dim (`SNTransferLowering.cpp:324-331`) is never reached.
    /// It is carried rather than dropped because bridge 1 reads it in eight places (`:930`, `:1270`,
    /// `:1713`, `:1723`, `:2227`, `:2256`, `:2454`) and a JSON-imported tree can carry it.
    pub unit_time_transfer_chunk_stride: Vec<SizeAndIndex>,
    /// Field: e023_TransferNode.rotateNumElements_
    ///
    /// How far the LXLU rotates the loaded data, `0` for no rotation (`dsc/dsc2.h:839`). Every
    /// reader guards on `> 0` (`SNTransferLowering.cpp:991`, `:1097`, `:2239`, `:2277`).
    pub rotate_num_elements: i32,
    /// Field: e023_TransferNode.coreIdToGTRInfo_
    ///
    /// The group tag register each core uses for this transfer — L3 only, as the authority's own
    /// comment says (`dsc/dsc2.h:840`). The L3 scheduler writes it one core at a time and refuses to
    /// overwrite an entry (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:4836`, `:5249-5260`).
    pub core_id_to_gtr_info: BTreeMap<CoreId, GroupTagRegInfo>,
    /// Field: e023_TransferNode.transferSize_
    ///
    /// "Explicit transfer size. If filled, use this size rather than derived from data stage"
    /// (`dsc/dsc2.h:841-843`). ⛔ ABSENCE IS THE COMMON CASE AND IS TESTED PER DIM, never for the
    /// whole map: `getBlockTransferSizePerDimCustomLocation` overrides one dim's extent only where
    /// `count(dim)` says so (`dsc/dsc2.cpp:3561-3562`), while two fill sites require the map to be
    /// EMPTY first (`dsc/dsc2.cpp:4776-4777`,
    /// `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:7897-7905`).
    pub transfer_size: BTreeMap<PrimaryDimTypes, DimSize>,
}

impl Default for TransferNode {
    /// The authority's member initialisers (`dsc/dsc2.h:834`, `:837`, `:839`) over
    /// [`DataLocation`]'s own (`sys-arch-spec/arch_enums.h:390-391`).
    ///
    /// ⛔ IT IS NOT `TransferNode()`: that constructor also passes `TRANSFER` to the base class
    /// (`dsc/dsc2.h:815`), and `nodeType_` is `ScheduleNode`'s, e013's to port.
    fn default() -> Self {
        Self {
            src: DataLocation::UNSET,
            src_indirect: DataLocation::UNSET,
            dst_vias: Vec::new(),
            replication_factor: 1,
            unit_time_transfer_chunk_size: Vec::new(),
            unit_time_transfer_num_chunks: 1,
            unit_time_transfer_chunk_stride: Vec::new(),
            rotate_num_elements: 0,
            core_id_to_gtr_info: BTreeMap::new(),
            transfer_size: BTreeMap::new(),
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

// crustify:todo: e023_TransferNode

// crustify:todo: e023_TransferNode.coreletViews_

// crustify:todo: e023_TransferNode.dstIndirectLdsAndLoopOffsets_

// crustify:todo: e023_TransferNode.dstIndirectLoopsAndSizes_

// crustify:todo: e023_TransferNode.dstReps_

// crustify:todo: e023_TransferNode.lastFusableParentLoopDst_

// crustify:todo: e023_TransferNode.lastFusableParentLoopSrc_

// crustify:todo: e023_TransferNode.paddingInfo_

// crustify:todo: e023_TransferNode.repetition_

// crustify:todo: e023_TransferNode.srcIndirectLdsAndLoopOffsets_

// crustify:todo: e023_TransferNode.srcIndirectLoopsAndSize_

// crustify:todo: e023_TransferNode.srcRep_

// crustify:todo: e023_TransferNode.transferCoordinates_

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
/// 11 (`dsc-based-utils/DSC2ToDataflowIR/V3/SNComputeLowering.cpp:1102`) and as a FEST flavour at 0
/// through 9 (`:1317-1385`), and the DDL states it verbatim (`ddc/ddl/ddl_conversion.cpp:1368-1370`).
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
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Repetition(pub i32);

/// One entry of a PACK/MERGE mapping — a source element position within the 128-bit slice
/// (`dsc/dsc2.h:906`), scaled by `expand_indices` as `compact_indices[i] * scale + j`
/// (`ddc/ddc_transformation.cpp:1939-1956`).
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
    /// ⛔ `INVALID`'S ENTRY IS `-1` (`:131`) — not a width, so it is ABSENT here. Its one arithmetic
    /// reader divides by it: `getComputeOperandSizes` computes `1024 / bitWidth`
    /// (`dsc/dsc2.cpp:2295`), which on `INVALID` yields -1024 elements in the authority, and a
    /// negative operand size is not one — see [`ComputeNode::operand_sizes`].
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
    /// Field: e024_ComputeNode.indices_
    ///
    /// The PACK/MERGE mapping (`dsc/dsc2.h:906`). ⛔ `-1` IS NOT A POSITION: the authority's own
    /// comment reads "-1 for zero/sign extend" (`dsc/dsc2.h:903`), so an absent entry is an
    /// extension slot rather than a source element.
    pub indices: Vec<Option<PackMergeIndex>>,
    /// Field: e024_ComputeNode.repetition_
    ///
    /// `dsc/dsc2.h:907` — "default 8 slices works the same".
    pub repetition: Repetition,
    /// Field: e024_ComputeNode.sign_extend_
    ///
    /// Whether a PACK/MERGE extends signed (`dsc/dsc2.h:908`), read as a bool attribute by bridge 1
    /// (`dsc-based-utils/DSC2ToDataflowIR/V3/SNComputeLowering.cpp:1124`).
    pub sign_extend: bool,
    /// Field: e024_ComputeNode.read_write_reg_map_
    ///
    /// An OPAQUE op's read/write register alias map (`dsc/dsc2.h:909-910`); `Ddc::finalizeOps` sizes
    /// its register window from it (`ddc/ddcv1.cpp:3376-3377`).
    pub read_write_reg_map: BTreeMap<String, String>,
    /// Field: e024_ComputeNode.read_only_reg_map_
    ///
    /// The read-only half of the same (`dsc/dsc2.h:911-912`, `ddc/ddcv1.cpp:3391`).
    pub read_only_reg_map: BTreeMap<String, String>,
    /// Field: e024_ComputeNode.param_map_
    ///
    /// An OPAQUE op's parameter alias map (`dsc/dsc2.h:913-914`). ⛔ THE SCHEDULER WRITES INTO IT:
    /// `Ddc::finalizeOps` sets `"unroll"` and `"prec"` (`ddc/ddcv1.cpp:3343`, `:3395-3397`).
    pub param_map: BTreeMap<String, String>,
    /// Field: e024_ComputeNode.mode_
    ///
    /// `dsc/dsc2.h:915`. ⛔ `-1` IS ABSENT — the DDL writes it only when it states one
    /// (`ddc/ddl/ddl_conversion.cpp:1368-1370`), and every bridge-1 reader tests it against a
    /// specific encoding (`dsc-based-utils/DSC2ToDataflowIR/V3/SNComputeLowering.cpp:1102`,
    /// `:1317-1385`), which -1 never is.
    pub mode: Option<Mode>,
    /// Field: e024_ComputeNode.compute_mask_
    ///
    /// `dsc/dsc2.h:916`, all eight slices unless the DDL states a mask
    /// (`ddc/ddl/ddl_conversion.cpp:1371-1373`).
    pub compute_mask: ComputeMask,
    // crustify:todo: e024_ComputeNode.computeMaskLoopOffsets_
    /// Field: e024_ComputeNode.input_data_connects_
    ///
    /// One name per input of an OPAQUE op (`dsc/dsc2.h:926-927`), index-parallel with
    /// [`ComputeNode::inputs`] where the fold pass reads them together
    /// (`ddc/ddc_fold.cpp:1632`, `:1856`).
    pub input_data_connects: Vec<DataConnect>,
    /// Field: e024_ComputeNode.output_data_connects_
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
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RepetitionWithOffset {
    /// Field: e024_ComputeNode.forInputs_
    ///
    /// `dsc/dsc2.h:951`.
    pub for_inputs: Vec<Repetition>,
    /// Field: e024_ComputeNode.forOutputs_
    ///
    /// `dsc/dsc2.h:952`. ⛔ A SPREAD THE TRANSFORMATION CONSUMES: it clones the node
    /// `for_outputs[idx] - 1` further times and writes 1 back into the clone
    /// (`ddc/ddc_transformation.cpp:1358-1379`).
    pub for_outputs: Vec<Repetition>,
}

/// `dsc/dsc2.h:900-962`. One compute instruction in the schedule tree: the unit it issues on, the
/// op, the operand components and the instruction attributes.
///
/// ⛔ THIS CARRIES COMPUTENODE'S OWN DECLARED FIELDS AND NOTHING INHERITED — IBM derives it from
/// `InheritWithClone<ScheduleNode, ComputeNode>` and its constructor tags the base with `COMPUTE`
/// (`dsc/dsc2.h:900-901`), and the base's thirteen fields are e013's. So the `e024_ComputeNode`
/// anchor at the end of this file is still open, and these five fields stay with it:
///  * `inputsLdsAndLoopOffsets_` and `outputsLdsAndLoopOffsets_` are `std::vector<DataInfo>`
///    (`:937-938`) — e019, still unported;
///  * `coreletViews_` is a per-corelet `CoreletView`, and both of its halves are
///    `std::vector<ScheduleNode::UnitView>` (`:943-947`) — e013's nested type;
///  * `inputCoordinates_` and `outputCoordinate_` are `CoordinateType<CoordinateBaseType>`
///    (`:948-949`) — e012, still unported;
///  * `instrAttribute_.computeMaskLoopOffsets_` is keyed by `const LoopNode*` (`:923-925`), the
///    pointer identity that `dsc/dsc2.cpp:1165-1216` round-trips through a node-name map.
///
/// ⛔ AND TWO OF THE THREE METHODS STAY OUT WITH THEM: `getComputeOperandFormats` reads
/// `dsc.labeledDs_.at(outputsLdsAndLoopOffsets_.at(0).myLdsIdx_).dataFormat_` for a PACKMERGE
/// (`dsc/dsc2.cpp:2348-2357`), which needs e019 and `DesignSpaceConfig`; `print` prints the base's
/// `name_` and each `DataInfo` (`dsc/dsc2.cpp:4443-4477`).
///
/// ⛔ NO `PartialEq`: node identity in the authority is the POINTER — `allocUsers_` and the fold
/// pass hold `ScheduleNode*` and compare nodes by address (`ddc/ddc_fold.cpp:1698`).
#[derive(Clone, Debug)]
pub struct ComputeNode {
    /// Field: e024_ComputeNode.exUnit_
    ///
    /// The execution unit the instruction issues on (`dsc/dsc2.h:932`). ⛔ A COMPUTE WHOSE OWN
    /// `exUnit_` APPEARS IN ITS OPERANDS IS ILLEGAL DDL (`ddc/ddl/ddl_conversion.cpp:1478-1487`).
    pub ex_unit: SenComponent,
    /// Field: e024_ComputeNode.type_
    ///
    /// The op (`dsc/dsc2.h:933`). Its `COUNT` initialiser means "not chosen yet"; the DDL conversion
    /// always overwrites it (`ddc/ddl/ddl_conversion.cpp:1410-1437`).
    pub r#type: ComputeOpType,
    /// Field: e024_ComputeNode.dataFormat_
    ///
    /// The precision the op runs at (`dsc/dsc2.h:934`). ⛔ IT IS THE OP'S PRECISION FOR `MACC`
    /// ALONE — every other op names its own width, and `dataFormat_` then only says what the
    /// operands hold (`ddc/ddl/ddl_conversion.cpp:1410-1430`, and see [`Self::operand_sizes`]).
    pub data_format: DataFormats,
    /// Field: e024_ComputeNode.inputs_
    ///
    /// Where each input comes from (`dsc/dsc2.h:935`), pushed in lockstep with
    /// `inputsLdsAndLoopOffsets_` and `repetitionWithOffset_.forInputs_`
    /// (`ddc/ddl/ddl_conversion.cpp:1385-1395`).
    pub inputs: Vec<SenComponent>,
    /// Field: e024_ComputeNode.outputs_
    ///
    /// Where each output goes (`dsc/dsc2.h:936`), on the same terms
    /// (`ddc/ddl/ddl_conversion.cpp:1396-1407`).
    pub outputs: Vec<SenComponent>,
    /// Field: e024_ComputeNode.instrAttribute_
    ///
    /// `dsc/dsc2.h:939`.
    pub instr_attribute: InstrAttribute,
    /// Field: e024_ComputeNode.numFoldsEngaged
    ///
    /// `dsc/dsc2.h:940`. ⛔ IT SCALES EVERY OPERAND SIZE (`dsc/dsc2.cpp:2333`, `:2344`); `Ddc` sets
    /// it from the unit's fold count (`ddc/ddcv1.cpp:1887`).
    pub num_folds_engaged: NumFoldsEngaged,
    /// Field: e024_ComputeNode.isOpaqueOp_
    ///
    /// Whether the DDL supplied the instruction verbatim (`dsc/dsc2.h:941`). ⛔ THE FOLD AND
    /// TRANSFORMATION PASSES BRANCH ON IT before reading the data connects
    /// (`ddc/ddc_fold.cpp:1630`, `:1853`, `ddc/ddc_transformation_util.cpp:1466`).
    pub is_opaque_op: bool,
    /// Field: e024_ComputeNode.repetitionWithOffset_
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
}

// crustify:todo: e024_ComputeNode

// crustify:todo: e024_ComputeNode.coreletViews_

// crustify:todo: e024_ComputeNode.inputCoordinates_

// crustify:todo: e024_ComputeNode.inputsLdsAndLoopOffsets_

// crustify:todo: e024_ComputeNode.inputsLoopsAndSizes_

// crustify:todo: e024_ComputeNode.outputCoordinate_

// crustify:todo: e024_ComputeNode.outputsLdsAndLoopOffsets_

// crustify:todo: e024_ComputeNode.outputsLoopsAndSizes_

/// Replaces: e025_ConditionNode
///
/// `dsc/dsc2.h:685-719`. A two-way branch in the schedule tree: a `BlockNode` whose at most two
/// children are the "then" and the "else" region (`:687-688`, `:697-699`).
///
/// ⛔ THIS CARRIES ONE OF CONDITIONNODE'S TWO GUARDS, so the `e025_ConditionNode` anchor below stays
/// open. `loopCond_` (`:690`) is a `LoopCondComposite` — e022, blocked behind e018's
/// `const LoopNode* loopComp_`, which is schedule-node pointer identity.
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
/// `const TransferNode*` whose reader dereferences it for that transfer's destination
/// (`dsc-based-utils/DSC2ToDataflowIR/V3/SNSyncLowering.cpp:180`) and whose JSON round trip goes
/// through the node's `name_` (`dsc/dsc2.cpp:823-826`), e013's field; `otherEndOfTheSignals_`
/// (`:969`) is the `vector<const SyncNode*>` linking the two ends.
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
    /// ⛔ ORDERED HERE, HASH-ORDERED IN THE AUTHORITY, where it is an `unordered_set` (`:966`). Two
    /// consumers put that order in their output — the node's JSON array (`dsc/dsc2.cpp:814-819`) and
    /// the `SyncOp` unit-name `ArrayAttr` of the DSC-to-DDL export
    /// (`ddc/ddl/ddl_conversion.cpp:3319-3325`) — so their text follows libstdc++ bucket order there
    /// and [`SenComponent`]'s declaration order here. Every other reader asks for membership only
    /// (`dcg/dcg_fe/pcfg_gen/dlOpsNew.cpp:2650-2651`, `ddc/ddc_transformation.cpp:1531`).
    pub units: BTreeSet<SenComponent>,
    /// Field: e026_SyncNode.isReceive_
    ///
    /// Which end this is (`dsc/dsc2.h:967`): bridge 1 emits a `sync_send` when it is false and a
    /// `sync_recv` when it is true (`SNSyncLowering.cpp:208-239`).
    pub is_receive: bool,
    /// Field: e026_SyncNode.isSoft_
    ///
    /// Whether the send may run ahead of the transfers it covers (`dsc/dsc2.h:967`): bridge 1 sets
    /// the emitted send's `wait_immediately_for_async_transfers` to its NEGATION
    /// (`SNSyncLowering.cpp:210-211`). Its only writer is the L3 scheduler's minter
    /// (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:658`).
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
/// authority then names the PREVIOUS slice (`dsc/dsc2.cpp:2481-2483`). Bridge 1 compares against it
/// per slice (`SNStickMaskLowering.cpp:47-63`), so `-1` means every slice takes the masked case; it
/// is not an absent value.
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
/// let _ = deeptools::schedule::dsc2::StickMaskView::default();
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
    /// The slice `mask_b` transitions at (`dsc/dsc2.h:1069`): bridge 1 applies `mask_a` alone before
    /// it, both masks at it, and no masking after (`SNStickMaskLowering.cpp:47-63`).
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
    /// labeled data structure (`ddc/ddcv1.cpp:3578`).
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
/// ⛔ THE STRING MAP LISTS `INDEX` FIRST (`dsc/dsc2.cpp:2431-2435`) while the enum declares
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
    /// `dsc/dsc2.cpp:2431-2435`).
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
/// `:833`), so a template may state any count, and `allocAllMem` divides a reserved size by it
/// (`ddc/ddcv1.cpp:353-355`).
///
/// ⛔ AND NOT AN [`Option`] EITHER: [`STREAMING`](Self::STREAMING) is a live third mode, not the
/// absence of a count, and its readers keep it distinct from `2` even while mapping it to `2` —
/// `allocAllMem` reserves the WHOLE memory capacity for a streaming buffer before dividing
/// (`ddc/ddcv1.cpp:317-329`, `:353-355`) and `processImplicitSync` refuses an implicit sync on
/// anything else, "Implicit syncs are only possible on circular buffers (num_buffers=-1)"
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
/// Every reader after the pass treats it as an extent: `buildUnitView` caps a dim at it and
/// `DT_CHECK`s that the remainder divides (`dsc/dsc2.cpp:2805-2812`), and `getPageSize` multiplies
/// the entries of one dim together (`dsc/dsc2.cpp:4497-4510`).
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
/// ⛔ A STRIDE, NOT A BASE ADDRESS: `allocAllMem` writes the reserved size divided by the buffer
/// count while the base goes to `startAddressCoreCorelet_` beside it (`ddc/ddcv1.cpp:351-356`), and
/// the L3 scheduler reads the pair together (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:4942-4944`).
/// A streaming allocation is divided by 2, not by its `-1` (`ddc/ddcv1.cpp:353-354`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BufferOffset(pub i64);

/// How many sticks one dim's data is spread across when it carries gaps — one value of
/// `AllocateNode::gapStickSpread_` (`dsc/dsc2.h:1006`).
///
/// ⛔ IT IS A MULTIPLIER IN ONE READER AND A DIVISOR IN THE OTHER, over the same dim.
/// `buildUnitView` multiplies the dim's unit-view size and every matching loop's `elemOffset_` by it
/// (`dsc/dsc2.cpp:2882-2899`), while `getBufferCapacityForNodePerDimCustomLocation` divides that
/// dim's capacity by it (`dsc/dsc2.cpp:3958-3961`) — the spread inflates the addresses and deflates
/// the capacity, so it is not a size in either direction.
///
/// Its in-scope writers are the masked-compute pass, which puts `8` on the INNERMOST layout dim
/// (`ddc/ddcv1.cpp:1704`), and the internal-register transformations
/// (`ddc/ddc_transformation.cpp:1132-1135`, `:1380`).
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
/// ⛔ AND ITS SEVEN METHODS STAY OUT WITH THOSE FIELDS. `getPageSize` (`:1011`, defined
/// `dsc/dsc2.cpp:4480-4513`) computes the page extents from the VALUE tensor's layout, and under
/// [`IndirectAllocType::IndexTensor`] that is `relatedIndirectAccessAlloc_`'s layout, reached
/// through the pointer it `DT_CHECK`s non-null (`dsc/dsc2.cpp:4491-4493`) — an answer computed from
/// this node's own layout instead would be silently wrong for exactly the index allocations the
/// paged path mints (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:6709-6729` is the caller).
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
    /// (`ddc/ddl/ddl_conversion.cpp:800`) and `ForceInnermostDimensionsOp` refuses any allocation
    /// without one, "Inner dims can only be applied on tensors"
    /// (`ddc/ddl/ddl_conversion.cpp:1874-1878`).
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
    /// (`ddc/ddl/ddl_conversion.cpp:1902-1905`), the masked-compute pass puts its stick spread on
    /// `at(0)` (`ddc/ddcv1.cpp:1704`), and `buildUnitView` appends these dims to the unit view AFTER
    /// the stick dims (`dsc/dsc2.cpp:2882`, whose walk starts at `getStickSizes(...).size()`).
    ///
    /// ⛔ A DIM MAY REPEAT — `backGapCore_`'s reader says so outright, "sizes may have dimensions
    /// repeated. Add gaps to outermost" (`dsc/dsc2.cpp:2903-2904`), and `getPageSize` multiplies
    /// every entry of one dim together (`dsc/dsc2.cpp:4503-4508`). What the DDL forbids is a repeat
    /// WITHIN one `AllocateOp`'s own dim list (`ddc/ddl/ddl_conversion.cpp:796-799`).
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
