//! The DDL conversion's interface types — `ddc/ddl/ddl_conversion.h`.
//!
//! Re-ported from the C++ authority. See crustify-scheduler/AGENT-BRIEF.md.
//!
//! ⭐ THE TEMPLATE TABLE IS DATA AND IS NOT COPIED HERE. `opFuncToDdlTemplate`
//! (`ddc/ddl/ddl_conversion.h:86-271`) is 121 rows of template text; `build.rs` reads the vendored
//! `ddc/ddl_templates/` through `ddl/selection.rs` (`OP_FUNC_TEMPLATES`, `Candidate`, `Serves`) and
//! censuses all 32 templates. [`DdlArch`] is that table's ROW TYPE at run time.

use std::collections::{BTreeMap, BTreeSet};

use sys_arch_spec::fields::Gen;
use sys_arch_spec::{CoreId, CoreletId};

use crate::schedule::dims::{MetaDimKind, PrimaryDimTypes};
use crate::schedule::dsc2::{BitWidth, DataFormats, LdsIdx, LoopCondComposite};

/// Replaces: e017_DdlArch
///
/// One candidate DDL template for an op func: the template's filename and the core generation it is
/// restricted to (`ddc/ddl/ddl_conversion.h:60-64`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DdlArch {
    /// Field: e017_DdlArch.filename
    ///
    /// The template's leaf name (`:61`), which the selection loop appends to
    /// `$DEEPTOOLS_PATH/ddc/ddl_templates/` before parsing (`ddc/ddl/ddl_conversion.cpp:51`, `:79`).
    ///
    /// ⚠️ A STRING WHERE A CLOSED SET EXISTS, AND THE SET CANNOT BE NAMED FROM `src/` YET: `build.rs`
    /// censuses the 32 vendored templates into a `Template` enum in `$OUT_DIR/generated.rs`, and
    /// `src/lib.rs` does not include that file — it still names the deleted `crate::arch` and
    /// `crate::schedule::ddl::conversion`. Wiring it is unscheduled work.
    pub filename: String,
    /// Field: e017_DdlArch.dedicatedArch
    ///
    /// The one generation this row serves, absent meaning every generation (`:62-63`).
    pub dedicated_arch: Option<Gen>,
}

impl DdlArch {
    /// Whether the selection loop may try this row on `core_arch` — the authority's `continue`
    /// filter, `arch.has_value() && arch.value() != dscGlobal_.sysDef.coreArch`
    /// (`ddc/ddl/ddl_conversion.cpp:67-68`).
    pub fn serves(&self, core_arch: Gen) -> bool {
        match self.dedicated_arch {
            Some(arch) => arch == core_arch,
            None => true,
        }
    }
}

/// One DDL dimension's properties — `DdlInterface::DimProp` (`ddc/ddl/ddl_conversion.h:297-340`),
/// the value of the blocked `dim_association_`.
///
/// ⛔ `nonPaddedDim` (`:305`) IS NOT CARRIED: it is an `mlir::Value` naming another entry of that
/// same map, and this port has no value identity to key it by — see the account beside the
/// `e019_DdlInterface` anchor below. `dump` (`:334`, body `ddc/ddl/ddl_conversion.cpp:3509-3531`)
/// prints it, so it is not carried either.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DimProp {
    /// The DSC dim this DDL dim was mapped onto (`:298-300`), [`PrimaryDimTypes::Undefined`] being
    /// the authority's own `PrimaryDimTypesCount` initialiser. Read through [`Self::mapped_dim`].
    pub dim: PrimaryDimTypes,
    /// Which DSC dims are still possible for it (`:301-302`), narrowed by intersection and
    /// difference as each layout op is matched — and reachable EMPTY
    /// (`ddc/ddl/ddl_conversion.cpp:2406`).
    pub dim_candidates: BTreeSet<PrimaryDimTypes>,
    /// How many GLOBAL layouts refer to it (`:303`): incremented only under `isGlobal`
    /// (`ddc/ddl/ddl_conversion.cpp:2394-2395`), never decremented, and the primary key of the
    /// mapping sort, ties broken by `dim_candidates.len()` (`:2454-2461`).
    pub num_refs_in_global_layouts: u32,
    /// True when no DSC dim could be mapped (`:304`); `tryDimMapping` sets it on both of its give-up
    /// paths (`ddc/ddl/ddl_conversion.cpp:2476`, `:2492`).
    pub drop_dim: bool,
    /// `:337-339`, and `private` there as here.
    meta_dim_kind: MetaDimKind,
}

impl Default for DimProp {
    /// `DimProp()` (`ddc/ddl/ddl_conversion.h:307-312`): every primary dim is a candidate except
    /// `IJ` and `KIJ`, so ten of the twelve.
    fn default() -> Self {
        Self {
            dim: PrimaryDimTypes::Undefined,
            dim_candidates: PrimaryDimTypes::ALL
                .into_iter()
                .filter(|dim| !matches!(dim, PrimaryDimTypes::Ij | PrimaryDimTypes::Kij))
                .collect(),
            num_refs_in_global_layouts: 0,
            drop_dim: false,
            meta_dim_kind: MetaDimKind::Unpadded,
        }
    }
}

impl DimProp {
    /// `getMetaDimKind` (`ddc/ddl/ddl_conversion.h:326`).
    pub fn meta_dim_kind(&self) -> MetaDimKind {
        self.meta_dim_kind
    }

    /// `setMetaDimKind(MetaDimKind)` (`ddc/ddl/ddl_conversion.h:325`).
    pub fn set_meta_dim_kind(&mut self, kind: MetaDimKind) {
        self.meta_dim_kind = kind;
    }

    /// `setMetaDimKind(llvm::StringRef)` (`ddc/ddl/ddl_conversion.h:317-323`): the DDL's
    /// `dim_property` attribute, stored. Absent for an unrecognised spelling, which is the
    /// authority's `false` and its one caller's `DT_ERROR` (`ddc/ddl/ddl_conversion.cpp:203-206`).
    pub fn set_meta_dim_kind_from_name(&mut self, name: &str) -> Option<MetaDimKind> {
        let kind = MetaDimKind::from_name(name)?;
        self.meta_dim_kind = kind;
        Some(kind)
    }

    /// `isUnpadded` (`ddc/ddl/ddl_conversion.h:327`).
    pub fn is_unpadded(&self) -> bool {
        self.meta_dim_kind == MetaDimKind::Unpadded
    }

    /// `isPadded` (`ddc/ddl/ddl_conversion.h:328`).
    pub fn is_padded(&self) -> bool {
        self.meta_dim_kind == MetaDimKind::Padded
    }

    /// `isMetaDim` (`ddc/ddl/ddl_conversion.h:329-333`): neither of the two above nor the `Count`
    /// sentinel, which is [`MetaDimKind::Undefined`] here.
    pub fn is_meta_dim(&self) -> bool {
        !matches!(
            self.meta_dim_kind,
            MetaDimKind::Unpadded | MetaDimKind::Padded | MetaDimKind::Undefined
        )
    }

    /// [`Self::dim`] behind the `dropDim_` guard four readers carry
    /// (`ddc/ddl/ddl_conversion.cpp:1085-1087`, `:1140-1149`, `:2514-2518`, `:2715-2716`).
    ///
    /// ⛔ `dim_` IS STALE, NOT RESET, WHEN `dropDim_` IS SET: `tryDimMapping` leaves the last
    /// candidate it tried there (`:2476`, `:2492`). ⛔ AND TWO READERS ON THE REVERSE DSC→DDL PATH
    /// SKIP THE GUARD AND SO READ THAT STALE DIM — `:3439` and `:3744` — so this method reports the
    /// discipline, it does not enforce it.
    pub fn mapped_dim(&self) -> Option<PrimaryDimTypes> {
        if self.drop_dim { None } else { Some(self.dim) }
    }
}

/// One DDL `datatype` op's resolved type — `DdlInterface::TypeDefinition`
/// (`ddc/ddl/ddl_conversion.h:361-365`).
///
/// ⭐ BOTH DECLARED INITIALISERS ARE THE SAME "NOT YET PARSED" STATE AND NEITHER SURVIVES A PARSE:
/// `processTypes` takes `dataFormat_ == INVALID` as that marker and then either fills both fields or
/// `DT_ERROR`s (`ddc/ddl/ddl_conversion.cpp:451-466`). Unparsed is the map entry's absence, so a
/// value of this type is always parsed and `bitSize_`'s -1 (`:364`) is unreachable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TypeDefinition {
    /// The format the DDL's `data_type` names (`:362-363`), never [`DataFormats::Invalid`].
    pub data_format: DataFormats,
    /// Its width (`:364`): the DDL's own `bit_width` where stated, else the format's own
    /// (`ddc/ddl/ddl_conversion.cpp:465-466`).
    pub bit_size: BitWidth,
}

impl TypeDefinition {
    /// `processTypes`' parse of one `datatype` op (`ddc/ddl/ddl_conversion.cpp:451-466`). Absent for
    /// an unrecognised `data_type`, which is the authority's `emitError` + `DT_ERROR` (`:460-463`).
    pub fn parse(data_type: &str, stated_bit_width: Option<BitWidth>) -> Option<Self> {
        let data_format = DataFormats::from_name(data_type);
        if data_format == DataFormats::Invalid {
            return None;
        }
        let bit_size = stated_bit_width.or_else(|| data_format.bit_width())?;
        Some(Self {
            data_format,
            bit_size,
        })
    }
}

/// One DDL tensor value's DSC binding — `DdlInterface::TensorProp`
/// (`ddc/ddl/ddl_conversion.h:373-375`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TensorProp {
    /// Its entry in `DesignSpaceConfig::labeledDs_` (`:374`). Absent is the authority's -1, which the
    /// parametric-loop op rejects (`ddc/ddl/ddl_conversion.cpp:1156-1160`) and which the
    /// interim-tensor check compares against as the default of its own parameter (`:2181`).
    pub lds_idx: Option<LdsIdx>,
}

/// An index into `DesignSpaceConfig::computeOp_` (`dsc/designSpaceConfig.h:89`) — the `computeOpIdx`
/// `matchOperations` settles on (`ddc/ddl/ddl_conversion.cpp:2151-2157`). The container itself is not
/// carried yet: see the open anchor at `src/schedule/dsc.rs:2103`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ComputeOpIdx(pub u32);

/// One bound DDL operation's DSC binding — `DdlInterface::OperationProp`
/// (`ddc/ddl/ddl_conversion.h:384-388`).
///
/// ⭐ `computeOpIdx_`'s -1 IS UNREACHABLE IN THE MAP, so the index is carried unconditionally: the
/// one site that creates an entry writes a matched, non-negative index on the very next line, behind
/// an early return for `< 0` (`ddc/ddl/ddl_conversion.cpp:2151-2157`), and every other access is
/// `find` or `count` (`:218`, `:2578`, `:2628`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OperationProp {
    /// Which compute op of the DSC this DDL op matched (`:385`).
    pub compute_op_idx: ComputeOpIdx,
    /// Which corelets of which cores it runs on, EMPTY MEANING ALL CORES (`:386-387`); filled by
    /// inverting the compute op's `coreExclude`/`coreClExclude`
    /// (`ddc/ddl/ddl_conversion.cpp:2162-2172`).
    pub core_cl_cond: BTreeMap<CoreId, BTreeSet<CoreletId>>,
}

impl OperationProp {
    /// A freshly matched op, before any core/corelet narrowing
    /// (`ddc/ddl/ddl_conversion.cpp:2156-2157`).
    pub fn new(compute_op_idx: ComputeOpIdx) -> Self {
        Self {
            compute_op_idx,
            core_cl_cond: BTreeMap::new(),
        }
    }
}

/// One resolved DDL condition — `DdlInterface::CondProp` (`ddc/ddl/ddl_conversion.h:419-424`), as
/// the three states its four fields spell and no more.
///
/// ⛔ THE FOURTH STATE IS A `DT_ERROR`, SO IT IS NOT A STATE: `processCondition`'s and/or arm refuses
/// a mix of [`Self::Loop`] and [`Self::CoreCl`] twice (`ddc/ddl/ddl_conversion.cpp:371-372`,
/// `:407-408`), and its tail normalises "none of the three" into `resolvedValue_ = false`
/// (`:438-441`). ⭐ `resolvedValue_` HAS NO INITIALISER — only `isResolvedToBool_` does (`h:420`) —
/// and is read only under it, so [`Self::Resolved`] is the pair and the undefined read is gone.
///
/// ⛔ NO `negate`: the `condNot` complement of [`Self::CoreCl`] needs `coreIdsUsed_` and
/// `numCoreletsUsed_DSC2_` off the DSC (`ddc/ddl/ddl_conversion.cpp:328-341`), so it is
/// `processCondition`'s, not this type's. ⭐ ITS OTHER TWO ARMS ALSO SHOW A DEAD WRITE: `:375` and
/// `:411` clear `isResolvedToBool_` immediately before `myCp = operandCp` overwrites it.
#[derive(Clone, Debug)]
pub enum CondProp {
    /// `isResolvedToBool_` together with `resolvedValue_` (`ddc/ddl/ddl_conversion.h:420`).
    Resolved(bool),
    /// `loopCond_` (`:421`), non-empty by [`LoopCondComposite`]'s own shape — which discharges the
    /// `twoLevelOrOfAnds_.empty()` half of the tail normalisation above.
    Loop(LoopCondComposite),
    /// `coreClCond_`, the core/corelet set the "then" region runs on (`:422-423`). Never empty — see
    /// [`Self::core_cl`].
    CoreCl(BTreeMap<CoreId, BTreeSet<CoreletId>>),
}

impl CondProp {
    /// The core/corelet state with the authority's tail normalisation applied: an empty map is
    /// `resolvedValue_ = false` (`ddc/ddl/ddl_conversion.cpp:438-441`).
    pub fn core_cl(cores: BTreeMap<CoreId, BTreeSet<CoreletId>>) -> Self {
        if cores.is_empty() {
            Self::Resolved(false)
        } else {
            Self::CoreCl(cores)
        }
    }

    /// `resolvedValue_` when `isResolvedToBool_` (`ddc/ddl/ddl_conversion.h:420`) — the only state
    /// in which the authority reads it.
    pub fn resolved_value(&self) -> Option<bool> {
        match self {
            Self::Resolved(value) => Some(*value),
            _ => None,
        }
    }
}

// ── `e019_DdlInterface`: what blocks its own fifteen fields ──────────────────────────────────────
//
// The nested element types above are `DdlInterface`'s (`ddc/ddl/ddl_conversion.h:288-462`), but the
// class itself is not carried: FOURTEEN of its fifteen fields are a map keyed or valued by an
// identity this port cannot yet spell, and the fifteenth alone is a shell.
//
//   * keyed by `mlir::Value`, i.e. by the `NameId` declared only inside `$OUT_DIR/generated.rs`,
//     which `src/lib.rs` does not include: `dim_association_` (`:342-343`), `type_definition_`
//     (`:366-367`), `tensor_definition_` (`:376-377`), `operation_definition_` (`:389-390`),
//     `datastage_definition_` (`:392-393`), `ext_constant_definition_` (`:395-396`),
//     `alloc_storage_` (`:398-399`), `operand_constant_tensor_` (`:405-406`), `resolvedConditions_`
//     (`:425-426`).
//   * valued by a non-owning schedule-tree pointer, for which this port has no node identity — the
//     same blocker `ConstantInfo` records in `src/schedule/dsc2.rs`, with `e030_BlockNode.next_` and
//     `e032_ScheduleTree.head_` still open: `alloc_storage_` (`dsc2::AllocateNode*`, `:398`),
//     `transfer_acc_pat_dims_` (`:401-403`, keyed by `const dsc2::TransferNode*`), `loop_labels_`
//     (`:409-410`, `dsc2::LoopNode*`), `region2blocks_` (`:412-413`, keyed by `mlir::Region*` — the
//     deleted `RegionId` — and valued by `dsc2::BlockNode*`), `sync_definitions_` (`:440-441`).
//   * keyed by `mlir::Operation*`: `coreToCore_definitions_` (`:452-453`). Its VALUE type
//     `FoldManager<int64_t>` is no longer the blocker — that is e026_FoldManager, filled in
//     `src/schedule/fold.rs` — so what is left open here is the key alone.
//   * `core_chunk_loop_label_` (`:411`) is the one portable field, a `std::string`. One of fifteen is
//     a shell, and `clear()` (`:458-461`) — a placement-new re-run of the constructor — has nothing
//     to clear on one.
//
// Two of `DdlInterface`'s own methods go with those fields: `getNonPaddedDimProp` (`:351-355`) walks
// `dim_association_` through `DimProp::nonPaddedDim`, and `getTensorProp` (`:378`, body
// `ddc/ddl/ddl_conversion.cpp:2083-2105`) walks `tensor_definition_` through an `AliasOneTensorOfOp`'s
// operands.
//
// `SyncProp` (`:432-439`) and `CoreToCore` (`:447-451`) are not ported for the same reason: one of
// `SyncProp`'s two fields is `syncsPerCl_` (`:437`), whose `SendRecv` is two
// `std::vector<dsc2::SyncNode*>` (`:433-436`), and two of `CoreToCore`'s three are
// `FoldManager<int64_t>` (`:449-450`). Each would carry one field of its own declaration.

// crustify:todo: e019_DdlInterface

// crustify:todo: e019_DdlInterface.Count

// crustify:todo: e019_DdlInterface.transfer_acc_pat_dims_

#[cfg(test)]
mod unit_tests {
    use super::*;

    /// The vendor's own `MATMUL_FWD` rows and an un-tagged one
    /// (`ddc/ddl/ddl_conversion.h:87-90`, `:166`).
    #[test]
    fn ddl_arch_serves_its_tagged_generation_only() {
        let bmm = DdlArch {
            filename: "bmm.ddl".to_string(),
            dedicated_arch: Some(Gen::Rcudd1a),
        };
        let bmm_sen1p5 = DdlArch {
            filename: "bmm_sen1p5.ddl".to_string(),
            dedicated_arch: Some(Gen::Sen1p5),
        };
        let pooling = DdlArch {
            filename: "pooling.ddl".to_string(),
            dedicated_arch: None,
        };
        assert!(bmm.serves(Gen::Rcudd1a));
        assert!(!bmm.serves(Gen::Sen1p5));
        assert!(!bmm.serves(Gen::Mpw4));
        assert!(bmm_sen1p5.serves(Gen::Sen1p5));
        assert!(!bmm_sen1p5.serves(Gen::Rcudd1a));
        for arch in [Gen::Mpw2, Gen::Mpw3, Gen::Mpw4, Gen::Rcudd1a, Gen::Sen1p5] {
            assert!(pooling.serves(arch));
        }
    }

    /// The constructor's candidate seed — every primary dim but `IJ` and `KIJ`
    /// (`ddc/ddl/ddl_conversion.h:307-312`).
    #[test]
    fn dim_prop_default_seeds_every_dim_but_ij_and_kij() {
        let prop = DimProp::default();
        assert_eq!(prop.dim_candidates.len(), PrimaryDimTypes::COUNT - 2);
        assert!(!prop.dim_candidates.contains(&PrimaryDimTypes::Ij));
        assert!(!prop.dim_candidates.contains(&PrimaryDimTypes::Kij));
        assert!(!prop.dim_candidates.contains(&PrimaryDimTypes::Undefined));
        assert!(prop.dim_candidates.contains(&PrimaryDimTypes::I));
        assert!(prop.dim_candidates.contains(&PrimaryDimTypes::X1));
        assert!(prop.is_unpadded());
        assert!(!prop.is_padded());
        assert!(!prop.is_meta_dim());
        assert_eq!(prop.mapped_dim(), Some(PrimaryDimTypes::Undefined));
    }

    /// The stale `dim_` a dropped dim keeps (`ddc/ddl/ddl_conversion.cpp:2476`, `:2492`), and the
    /// `dim_property` spellings `setMetaDimKind` accepts (`ddc/ddl/ddl_conversion.h:317-333`).
    #[test]
    fn dim_prop_hides_the_stale_dim_of_a_dropped_dim() {
        let mut prop = DimProp::default();
        prop.dim = PrimaryDimTypes::Ki;
        assert_eq!(prop.mapped_dim(), Some(PrimaryDimTypes::Ki));
        prop.drop_dim = true;
        assert_eq!(prop.mapped_dim(), None);
        assert_eq!(prop.dim, PrimaryDimTypes::Ki);

        assert_eq!(
            prop.set_meta_dim_kind_from_name("window"),
            Some(MetaDimKind::WindowDim)
        );
        assert!(prop.is_meta_dim());
        assert_eq!(prop.set_meta_dim_kind_from_name("no_such_property"), None);
        assert_eq!(prop.meta_dim_kind(), MetaDimKind::WindowDim);
        prop.set_meta_dim_kind(MetaDimKind::Padded);
        assert!(prop.is_padded());
        assert!(!prop.is_meta_dim());
        assert_eq!(
            prop.set_meta_dim_kind_from_name("undefined"),
            Some(MetaDimKind::Undefined)
        );
        assert!(!prop.is_meta_dim());
    }

    /// The DDL's stated `bit_width` wins over the format's own table, and `SENINT24`'s table entry is
    /// 16 (`ddc/ddl/ddl_conversion.cpp:465-466`, `util/sendefs/sendefs.cpp:135`).
    #[test]
    fn type_definition_prefers_the_ddls_stated_width() {
        assert_eq!(
            TypeDefinition::parse("SEN169_FP16", None),
            Some(TypeDefinition {
                data_format: DataFormats::Sen169Fp16,
                bit_size: BitWidth(16),
            })
        );
        assert_eq!(
            TypeDefinition::parse("SENINT24", None).map(|t| t.bit_size),
            Some(BitWidth(16))
        );
        assert_eq!(
            TypeDefinition::parse("SENINT24", Some(BitWidth(24))).map(|t| t.bit_size),
            Some(BitWidth(24))
        );
    }

    /// The unparsed marker cannot be built, whether or not a width is stated
    /// (`ddc/ddl/ddl_conversion.cpp:459-463`).
    #[test]
    fn type_definition_rejects_an_unrecognised_data_type() {
        assert_eq!(TypeDefinition::parse("INVALID", None), None);
        assert_eq!(TypeDefinition::parse("no_such_format", None), None);
        assert_eq!(
            TypeDefinition::parse("no_such_format", Some(BitWidth(8))),
            None
        );
    }

    /// An unbound tensor carries the authority's -1 as an absence
    /// (`ddc/ddl/ddl_conversion.cpp:1156-1160`).
    #[test]
    fn tensor_prop_starts_without_an_lds() {
        assert_eq!(TensorProp::default().lds_idx, None);
        assert_eq!(
            TensorProp {
                lds_idx: Some(LdsIdx(3)),
            }
            .lds_idx,
            Some(LdsIdx(3))
        );
    }

    /// A freshly matched op runs on all cores, which the empty map spells
    /// (`ddc/ddl/ddl_conversion.h:386-387`, `ddc/ddl/ddl_conversion.cpp:2156-2172`).
    #[test]
    fn operation_prop_starts_on_all_cores() {
        let mut prop = OperationProp::new(ComputeOpIdx(2));
        assert_eq!(prop.compute_op_idx, ComputeOpIdx(2));
        assert!(prop.core_cl_cond.is_empty());
        prop.core_cl_cond
            .entry(CoreId(0))
            .or_default()
            .insert(CoreletId(1));
        assert_eq!(
            prop.core_cl_cond[&CoreId(0)],
            BTreeSet::from([CoreletId(1)])
        );
    }

    /// `processCondition`'s tail: no loop condition and no core/corelet condition is
    /// `resolvedValue_ = false` (`ddc/ddl/ddl_conversion.cpp:438-441`).
    #[test]
    fn cond_prop_normalises_an_empty_core_condition_to_false() {
        assert_eq!(
            CondProp::core_cl(BTreeMap::new()).resolved_value(),
            Some(false)
        );
        let cores = BTreeMap::from([(CoreId(1), BTreeSet::from([CoreletId(0)]))]);
        let cond = CondProp::core_cl(cores.clone());
        assert_eq!(cond.resolved_value(), None);
        assert!(matches!(cond, CondProp::CoreCl(got) if got == cores));
        assert_eq!(CondProp::Resolved(true).resolved_value(), Some(true));
    }
}
