//! Re-ported from the C++ authority. See crustify-scheduler/AGENT-BRIEF.md.

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
}
