//! Re-ported from the C++ authority. See crustify-scheduler/AGENT-BRIEF.md.

use crate::schedule::dims::PrimaryDimTypes;

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
