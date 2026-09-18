//! Re-ported from the C++ authority. See crustify-scheduler/AGENT-BRIEF.md.

use crate::schedule::dims::{DataStructDims, PrimaryDimAndKind, PrimaryDimTypes};
use std::collections::BTreeMap;

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
