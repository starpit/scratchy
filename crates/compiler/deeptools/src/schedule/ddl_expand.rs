//! The DDL conversion's interface types — `ddc/ddl/ddl_conversion.h`.
//!
//! Re-ported from the C++ authority. See crustify-scheduler/AGENT-BRIEF.md.
//!
//! ⭐ THE TEMPLATE TABLE IS DATA AND IS NOT COPIED HERE. `opFuncToDdlTemplate`
//! (`ddc/ddl/ddl_conversion.h:86-271`) is 121 rows of template text; `build.rs` reads the vendored
//! `ddc/ddl_templates/` through `ddl/selection.rs` (`OP_FUNC_TEMPLATES`, `Candidate`, `Serves`) and
//! censuses all 32 templates. [`DdlArch`] is that table's ROW TYPE at run time.

use std::collections::{BTreeMap, BTreeSet};
use std::ops::RangeInclusive;

use sys_arch_spec::arch_enums::SenComponent;
use sys_arch_spec::fields::Gen;
use sys_arch_spec::{CoreId, CoreletId};

use crate::schedule::ddc::Verbosity;
use crate::schedule::dims::{
    DimDensity, DimVal, MetaDimKind, PadType, PaddingFormType, PrimaryDimTypes,
};
use crate::schedule::dsc::{DesignSpaceConfig, NumCoresUsed};
use crate::schedule::dsc2::{
    BitWidth, CondVal, CondValType, DataFormats, IterationIdx, LdsIdx, LoopCondComposite,
    LoopCondOp,
};
use crate::schedule::fold::FoldManager;
use crate::schedule::metadata::Metadata;

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
    /// censuses the 32 vendored templates into a `Template` enum in `$OUT_DIR/generated.rs`, which no
    /// module includes — and which could not compile if one did, because the text `build.rs` emits
    /// still names the deleted `crate::arch::IsaGen` (`build.rs:2583`) and
    /// `crate::schedule::ddl::conversion` (`:2995`). Wiring it is unscheduled work.
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
            dim_candidates: PrimaryDimTypes::NON_COMPOUND.into_iter().collect(),
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

/// One core-to-core communication op's ring — `DdlInterface::CoreToCore`
/// (`ddc/ddl/ddl_conversion.h:447-451`), the value of the blocked `coreToCore_definitions_`.
///
/// ⭐ ALL THREE FIELDS ARE CARRIED, and [`FoldManager`] is why: it is e026_FoldManager, filled in
/// `src/schedule/fold.rs`, which instantiates exactly this `i64` payload. What is blocked is the map
/// that holds this type (`:452-453`), by its `mlir::Operation*` KEY alone.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CoreToCore {
    /// The dim the ring walks (`:448`): the first mapped, non-dropped dim of the op's `dimensions`,
    /// overridden by the one dim split across cores when there is one
    /// (`ddc/ddl/ddl_conversion.cpp:1915-1926`), with [`PrimaryDimTypes::Undefined`] the authority's
    /// own `PrimaryDimTypesCount` initialiser. Read through [`Self::communication_dim`].
    pub dim: PrimaryDimTypes,
    /// Which core this one sends to, per coordinate of the SuperDsc's fold space (`:449`): Map over
    /// core and corelet, Constant over every further `sdscFoldProps_` level, then one `insertData`
    /// per used core (`ddc/ddl/ddl_conversion.cpp:1941-1949`, `:1977-1997`).
    ///
    /// ⛔ ZERO FOLD DIMS IS THE PRE-BUILD STATE THE AUTHORITY ASSERTS, not an empty ring:
    /// `DT_CHECK(nextCore_.hasZeroFoldDim() && prevCore_.hasZeroFoldDim())` (`:1939-1940`) says the
    /// entry `operator[]` has just default-constructed — which is what [`Default`] gives here.
    pub next_core: FoldManager<i64>,
    /// Which core this one receives from (`:450`), built alongside [`Self::next_core`] — and on
    /// corelet 1 the two swap, because cl0's ring runs up the core ids and cl1's runs down
    /// (`ddc/ddl/ddl_conversion.cpp:1959`, `:1977-1997`).
    pub prev_core: FoldManager<i64>,
}

impl CoreToCore {
    /// The communication dim once the fill loop has mapped one
    /// (`ddc/ddl/ddl_conversion.cpp:1915-1926`).
    ///
    /// ⛔ ABSENT IS THE REFUSAL AND NOT A STATE: the sentinel surviving that loop is
    /// `emitError("None of the dimensions is mapped")` and `DT_ERROR` (`:1929-1932`), so every later
    /// read — the first is `numWkSlicesPerDim_.at(dim_)` (`:1934`) — has a dim.
    pub fn communication_dim(&self) -> Option<PrimaryDimTypes> {
        if self.dim == PrimaryDimTypes::Undefined {
            None
        } else {
            Some(self.dim)
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
//     which no module includes and which could not compile if one did (its text names the deleted
//     `crate::arch::IsaGen` and `crate::schedule::ddl::conversion`, `build.rs:2583`, `:2995`):
//     `dim_association_` (`:342-343`), `type_definition_` (`:366-367`), `tensor_definition_`
//     (`:376-377`), `operation_definition_` (`:389-390`), `datastage_definition_` (`:392-393`),
//     `ext_constant_definition_` (`:395-396`), `alloc_storage_` (`:398-399`),
//     `operand_constant_tensor_` (`:405-406`), `resolvedConditions_` (`:425-426`).
//   * valued by a non-owning schedule-tree pointer, for which this port has no node identity — the
//     same blocker `ConstantInfo` records in `src/schedule/dsc2.rs`, with `e030_BlockNode.next_` and
//     `e032_ScheduleTree.head_` still open: `alloc_storage_` (`dsc2::AllocateNode*`, `:398`),
//     `transfer_acc_pat_dims_` (`:401-403`, keyed by `const dsc2::TransferNode*`), `loop_labels_`
//     (`:409-410`, `dsc2::LoopNode*`), `region2blocks_` (`:412-413`, keyed by `mlir::Region*` — the
//     deleted `RegionId` — and valued by `dsc2::BlockNode*`), `sync_definitions_` (`:440-441`).
//   * keyed by `mlir::Operation*`: `coreToCore_definitions_` (`:452-453`) — ONLY the key. Its value
//     type is [`CoreToCore`] above, `FoldManager<int64_t>` fields and all.
//   * `core_chunk_loop_label_` (`:411`) is the one portable field, a `std::string`. One of fifteen is
//     a shell, and `clear()` (`:458-461`) — a placement-new re-run of the constructor — has nothing
//     to clear on one.
//
// Two of `DdlInterface`'s own methods go with those fields: `getNonPaddedDimProp` (`:351-355`) walks
// `dim_association_` through `DimProp::nonPaddedDim`, and `getTensorProp` (`:378`, body
// `ddc/ddl/ddl_conversion.cpp:2083-2105`) walks `tensor_definition_` through an `AliasOneTensorOfOp`'s
// operands.
//
// `SyncProp` (`:432-439`) is the one nested type that stays unported, and for a blocker of its own:
// one of its two fields is `syncsPerCl_` (`:437`), whose `SendRecv` is two
// `std::vector<dsc2::SyncNode*>` (`:433-436`), so it would carry one field of its own declaration.
//
// ⭐ THE TWO FIELD-LEVEL TODOs BELOW ARE NOT ONE KIND. `transfer_acc_pat_dims_` is a real field,
// blocked twice over — bullet two above. `.Count` IS NOT A FIELD: it is `MetaDimKind::Count` on the
// third line of `DimProp::isMetaDim`'s wrapped return (`ddc/ddl/ddl_conversion.h:332`), swept up by
// the scheduler's end-of-line field regex (`crustify/campaigns/scheduler/plan.py:67`). Nothing can
// fill it; it stands because an anchor is never deleted, and UNITS.tsv is the ledger that counts.

// crustify:todo: e019_DdlInterface

// crustify:todo: e019_DdlInterface.Count

// crustify:todo: e019_DdlInterface.transfer_acc_pat_dims_

/// Replaces: e018_DdlConversion
///
/// The DDL↔DSC conversion (`ddc/ddl/ddl_conversion.h:482-559`): it picks a `.ddl` template for the
/// DSC's op func, parses it, and expands it into schedule-tree nodes.
///
/// ⛔ THREE OF ITS SEVEN FIELDS ARE CARRIED — and UNITS.tsv's "0 declared fields" is a field-scan
/// defect, they are declared at `:511-517`. The four that are not:
///   * `const SuperDsc& sdsc` (`:512`) and `const DesignSpaceConfigGlobal& dscGlobal_` (`:513`):
///     neither is a unit in UNITS.tsv and neither is in `src/`.
///   * `DdlInterface ddlInterface` (`:514`) is `e019_DdlInterface`, whose anchor is open above with
///     fourteen of its fifteen fields blocked.
///   * `DdlModuleOp ddlParser_` (`:515`) is the MLIR module and context; the template parse is
///     build-side, in `build.rs` with `ddl/{ast,parse,selection,smc}.rs`.
///
/// ⛔ AND THOSE FOUR ARE WHAT KEEPS FIFTEEN OF THE EIGHTEEN METHODS OFF THIS TYPE.
/// `ddlInterface`'s `mlir::Value`-keyed maps block `processOp` (`ddc/ddl/ddl_conversion.cpp:582`),
/// `processRegion` (`:2008`), `processTransformations` (`:2036`), `processPaddedDimensionOp`
/// (`:102`), `processDimensionOp` (`:187`), `processTypes` (`:447`), `addInternalTensor` (`:473`),
/// `matchDdl2Dsc` (`:2110`), `parseDdl2Dsc` (`:2770`), `getTensor` (`:2825`),
/// `getTensorAndAllocation` (`:2847`), `convertDsc2Ddl` (`:2881`), `checkMetaDimensions` (`:3537`),
/// `checkAccessPattern` (`:3649`) and `processAccessPatterns` (`:3728`); and `dsc.computeOp_` — an
/// open `Field:` anchor in `src/schedule/dsc.rs` — blocks `selectAndParseDdlTemplate` (`:42`),
/// whose 121-row table is already `ddl/selection.rs`'s build-side `OP_FUNC_TEMPLATES`. What lands
/// here is `processExpression` whole, plus every arm of `processCondition` and
/// `verifyDdlConstraints` that reads only this type's three fields and the DDL op's own attributes.
///
/// ⛔ NO `Default`: [`min_num_cores_met`](Self::min_num_cores_met) cannot answer on a DSC that has
/// not been through DSM, and a defaulted one has not.
/// ⛔ AND NO `Clone` EITHER, because [`Metadata`] has none: its `core_dstgid`/`chunk_dstgid` are
/// `const int`, which deletes the authority's copy-assignment (`ddc/ddc_metadata.h:211-212`).
#[derive(Debug)]
pub struct DdlConversion {
    /// Field: e018_DdlConversion.dsc
    ///
    /// The DSC being built (`ddc/ddl/ddl_conversion.h:511`) — a `DesignSpaceConfig&` there, OWNED
    /// here: the object the conversion mutates is one value, per AGENT-BRIEF rule 4.
    pub dsc: DesignSpaceConfig,
    /// Field: e018_DdlConversion.metadata_
    ///
    /// The conversion metadata (`:516`), owned on the same terms.
    pub metadata: Metadata,
    /// Field: e018_DdlConversion.verbose_
    ///
    /// The verbosity level the `Ddc` hands down (`:517`).
    pub verbose: Verbosity,
}

impl DdlConversion {
    /// The constructor's three carriable initialisers (`ddc/ddl/ddl_conversion.h:551-558`).
    pub fn new(dsc: DesignSpaceConfig, metadata: Metadata, verbose: Verbosity) -> Self {
        Self {
            dsc,
            metadata,
            verbose,
        }
    }

    /// `processExpression` (`ddc/ddl/ddl_conversion.cpp:2072-2079`): a DDL attribute string as a
    /// number. Absent is its `DT_ERROR`, which fires unless `strtof` consumed all of it bar
    /// surrounding whitespace.
    ///
    /// ⚠️ ONE FORM DIVERGES: `strtof` also admits C's hex-float (`0x1p3`) and Rust's parse does not.
    /// No vendored template states one — `build.rs` censuses all 32.
    pub fn process_expression(expr: &str) -> Option<f32> {
        expr.trim().parse::<f32>().ok()
    }

    /// A `ddl.condition`'s `value_expr` (`ddc/ddl/ddl_conversion.cpp:265-273`): one of the named
    /// forms, else a number.
    ///
    /// ⛔ THE CALLER NARROWS A `float` TO AN `int` (`:273`), so `2.7` is iteration 2 — and the three
    /// other callers of [`process_expression`] keep the fraction (`:1840`, `:1843`, `:1847`).
    pub fn parse_cond_val(value_expr: &str) -> Option<CondVal> {
        match CondValType::from_name(value_expr) {
            Some(val_type) => Some(CondVal::from_wire(val_type, CondVal::ABSENT_VAL_INT)),
            None => {
                let val = Self::process_expression(value_expr)?;
                Some(CondVal::Iteration(IterationIdx(val as i32)))
            }
        }
    }

    /// A condition on a DROPPED dim (`ddc/ddl/ddl_conversion.cpp:274-292`): the dim does not exist,
    /// so it is a loop of size one and the guard resolves against iteration zero.
    ///
    /// ⭐ `FIRST`/`LAST` BOTH BECOME ZERO (`:277-280`), which is why they cannot stay a [`CondVal`].
    /// ⛔ AND IBM'S IF-CHAIN HAS NO ELSE, leaving `resolvedValue_` uninitialised on a seventh
    /// operator; [`LoopCondOp`] has only these six, which is also what `:259-263` narrowed to.
    pub fn resolve_dropped_dim_condition(cond_op: LoopCondOp, cond_val: CondVal) -> bool {
        let val = match cond_val {
            CondVal::Iteration(idx) => idx.0,
            CondVal::First | CondVal::Last => 0,
        };
        match cond_op {
            LoopCondOp::Eq => val == 0,
            LoopCondOp::Ne => val != 0,
            LoopCondOp::Lt => val < 0,
            LoopCondOp::Le => val <= 0,
            LoopCondOp::Gt => val > 0,
            LoopCondOp::Ge => val >= 0,
        }
    }

    /// `ConditionNotOp` (`ddc/ddl/ddl_conversion.cpp:321-342`), whose three arms are this enum's
    /// three variants in the authority's own dispatch order.
    pub fn negate_condition(&self, cond: CondProp) -> CondProp {
        match cond {
            CondProp::Resolved(value) => CondProp::Resolved(!value),
            CondProp::Loop(loop_cond) => CondProp::Loop(loop_cond.negate()),
            CondProp::CoreCl(core_cl) => self.complement_core_cl(core_cl),
        }
    }

    /// The `condNot` complement of a core/corelet condition (`ddc/ddl/ddl_conversion.cpp:327-341`):
    /// over every used core, a stated corelet set is toggled member by member and an unstated core
    /// gains corelet 0, and 1 as well when more than one corelet is used.
    ///
    /// ⛔ THE UNSTATED-CORE ARM IS NOT THE COMPLEMENT OF THE EMPTY SET — it hard-codes at most two
    /// corelets whatever `numCoreletsUsed_DSC2_` says (`:338-340`), while the toggle arm spans all
    /// of them. ⭐ AN UNSTATED COUNT TOGGLES NOTHING, faithfully: IBM's bound is its `-1`.
    pub fn complement_core_cl(
        &self,
        mut core_cl: BTreeMap<CoreId, BTreeSet<CoreletId>>,
    ) -> CondProp {
        let count = self.dsc.num_corelets_used_dsc2.map_or(0, |used| used.0);
        for &core in &self.dsc.core_ids_used {
            match core_cl.get_mut(&core) {
                Some(corelets) => {
                    for cl in (0..count)
                        .filter_map(|cl| u8::try_from(cl).ok())
                        .map(CoreletId)
                    {
                        if !corelets.remove(&cl) {
                            corelets.insert(cl);
                        }
                    }
                    if corelets.is_empty() {
                        core_cl.remove(&core);
                    }
                }
                None => {
                    let corelets = core_cl.entry(core).or_default();
                    corelets.insert(CoreletId(0));
                    if count > 1 {
                        corelets.insert(CoreletId(1));
                    }
                }
            }
        }
        CondProp::core_cl(core_cl)
    }

    /// `ConditionAndOp` over already-resolved operands (`ddc/ddl/ddl_conversion.cpp:342-434`).
    /// Absent is its "And/or op is mixing incompatible types" (`:371-372`, `:407-408`) or a loop
    /// composition outside the two-level OR of ANDs (`:380-389`, `:393-399`).
    pub fn and_conditions(operands: impl IntoIterator<Item = CondProp>) -> Option<CondProp> {
        Self::compose_conditions(true, operands)
    }

    /// `ConditionOrOp`, the same fold (`ddc/ddl/ddl_conversion.cpp:342-434`) and the same absences.
    pub fn or_conditions(operands: impl IntoIterator<Item = CondProp>) -> Option<CondProp> {
        Self::compose_conditions(false, operands)
    }

    /// The `isAnd` fold both of the above are (`ddc/ddl/ddl_conversion.cpp:343-434`).
    ///
    /// ⭐ A RESOLVED OPERAND IS THE JUNCTION'S IDENTITY OR ITS ANNIHILATOR and nothing else: the
    /// annihilator returns straight out (`:346-360`) and the identity is dropped, whether it arrived
    /// first (`:373-376`, `:409-412`) or later, where IBM's merge block simply finds nothing to do.
    /// ⛔ THE TAIL NORMALISATION IS NOW ONLY THE EMPTY OPERAND LIST (`:438-441`): the other two
    /// states it caught are unspellable — see [`CondProp`].
    fn compose_conditions(
        is_and: bool,
        operands: impl IntoIterator<Item = CondProp>,
    ) -> Option<CondProp> {
        let mut composed: Option<CondProp> = None;
        for operand in operands {
            if let CondProp::Resolved(value) = operand
                && value != is_and
            {
                return Some(CondProp::Resolved(value));
            }
            let Some(current) = composed.take() else {
                composed = Some(operand);
                continue;
            };
            composed = Some(match (current, operand) {
                (CondProp::Resolved(_), later) => later,
                (earlier, CondProp::Resolved(_)) => earlier,
                (CondProp::Loop(earlier), CondProp::Loop(later)) => {
                    CondProp::Loop(Self::compose_loop_conditions(is_and, earlier, later)?)
                }
                (CondProp::CoreCl(earlier), CondProp::CoreCl(later)) => {
                    CondProp::core_cl(Self::compose_core_cl_conditions(is_and, earlier, later))
                }
                _ => return None,
            });
        }
        Some(composed.unwrap_or(CondProp::Resolved(false)))
    }

    /// The loop half of that fold (`ddc/ddl/ddl_conversion.cpp:380-403`), which is the three
    /// grammar types composing: AND appends terms to one clause, OR appends clauses.
    ///
    /// ⭐ THE TWO REFUSALS ARE THE TWO NARROWINGS: `without_negation` is IBM's `negated_` half and
    /// `into_conjunction` its `size() != 1` half, so nothing here re-tests them.
    fn compose_loop_conditions(
        is_and: bool,
        earlier: LoopCondComposite,
        later: LoopCondComposite,
    ) -> Option<LoopCondComposite> {
        let earlier = earlier.without_negation()?;
        let later = later.without_negation()?;
        if is_and {
            Some(
                earlier
                    .into_conjunction()?
                    .and(later.into_conjunction()?)
                    .into(),
            )
        } else {
            Some(earlier.or(later).into())
        }
    }

    /// The core/corelet half (`ddc/ddl/ddl_conversion.cpp:413-430`): AND intersects per core and
    /// drops a core the operand omits or empties, OR unions.
    ///
    /// ⛔ IBM'S AND ARM IS UNDEFINED BEHAVIOUR — it `erase`s the current element of the `std::map`
    /// it is ranging over and then increments that iterator (`:417-418`, `:422`). `retain` is the
    /// well-defined reading of what it means.
    fn compose_core_cl_conditions(
        is_and: bool,
        mut earlier: BTreeMap<CoreId, BTreeSet<CoreletId>>,
        later: BTreeMap<CoreId, BTreeSet<CoreletId>>,
    ) -> BTreeMap<CoreId, BTreeSet<CoreletId>> {
        if is_and {
            earlier.retain(|core, corelets| match later.get(core) {
                Some(theirs) => {
                    *corelets = corelets.intersection(theirs).copied().collect();
                    !corelets.is_empty()
                }
                None => false,
            });
        } else {
            for (core, corelets) in later {
                earlier.entry(core).or_default().extend(corelets);
            }
        }
        earlier
    }

    /// `ddl.constraint {min_num_cores = N}` (`ddc/ddl/ddl_conversion.cpp:2558-2564`): whether the
    /// DSC uses at least that many cores. Absent where the DSC has not stated a count, which is the
    /// indeterminate `int` the authority compares.
    pub fn min_num_cores_met(&self, min_num_cores: NumCoresUsed) -> Option<bool> {
        Some(self.dsc.num_cores_used? >= min_num_cores)
    }

    /// `ddl.constraint {min_num_valid =, max_num_valid =}` (`ddc/ddl/ddl_conversion.cpp:2603-2605`)
    /// as the range a count of active variables must fall in.
    ///
    /// ⭐ THE UNSTATED MAXIMUM IS 100, NOT UNBOUNDED, and that number is the authority's own
    /// `value_or` (`:2604`). The count itself needs `operation_definition_` and `labeledDs_`
    /// (`:2575-2601`), so it is not answerable here.
    pub fn num_valid_range(
        min_num_valid: Option<u32>,
        max_num_valid: Option<u32>,
    ) -> RangeInclusive<u32> {
        min_num_valid.unwrap_or(0)..=max_num_valid.unwrap_or(100)
    }

    /// `ddl.constraint {relative_op_order = true}` (`ddc/ddl/ddl_conversion.cpp:2621-2637`): the
    /// operands that ARE bound must appear in strictly increasing compute-op order.
    ///
    /// ⛔ `relative_op_order = false` IS NOT THIS FORM at all — the authority tests
    /// `has_value() && value()` and falls through to the `cmp` arm (`:2624-2625`).
    pub fn relative_op_order_met(bound: impl IntoIterator<Item = ComputeOpIdx>) -> bool {
        let mut highest: Option<ComputeOpIdx> = None;
        for idx in bound {
            if highest.is_some_and(|seen| idx <= seen) {
                return false;
            }
            highest = Some(idx);
        }
        true
    }

    /// The size a property-less `ddl.constraint {cmp =, value =}` tests, for one bound DDL dim
    /// (`ddc/ddl/ddl_conversion.cpp:2710-2748`): a padding scalar, or the core stage's steady-state
    /// dim under the padding form its meta-dim kind names.
    ///
    /// ⛔ THREE ABSENCES, TWO OF WHICH LEAVE THE CONSTRAINT SATISFIED: a dropped dim (`:2715`) and a
    /// dim with no padding entry (`:2722`) are the authority's `continue` — and that entry is
    /// required by `Padded` and `PadValid` too, which then do not read it. The third,
    /// [`MetaDimKind::Undefined`], is its "Constraint on unsupported dim kind" (`:2745-2746`).
    pub fn dim_constraint_size(&self, dim_prop: &DimProp) -> Option<DimVal> {
        let dim = dim_prop.mapped_dim()?;
        let ref_ds = &self.dsc.data_stage_param.get(&Metadata::CORE_DSTGID)?.ss;
        let kind = dim_prop.meta_dim_kind();
        if matches!(kind, MetaDimKind::Unpadded | MetaDimKind::WindowDim) {
            return ref_ds.primary_dim_to_val(dim);
        }
        let padded_as = |pad| {
            ref_ds.primary_dim_to_val_for_component(
                dim,
                SenComponent::NoComponent,
                None,
                None,
                &PaddingFormType::new(dim, pad),
                DimDensity::FULL,
                false,
            )
        };
        let pad_info = ref_ds.padding_sizes.get(&dim)?;
        match kind {
            MetaDimKind::PadFront => Some(DimVal(pad_info.pad_front)),
            MetaDimKind::PadBack => Some(DimVal(pad_info.pad_back)),
            MetaDimKind::Stride => Some(DimVal(pad_info.stride)),
            MetaDimKind::Dilation => Some(DimVal(pad_info.dilation)),
            MetaDimKind::Padded => padded_as(PadType::PaddedFullSpanWUnneeded),
            MetaDimKind::PadValid => padded_as(PadType::PaddedNoZeroPad),
            MetaDimKind::Unpadded | MetaDimKind::WindowDim | MetaDimKind::Undefined => None,
        }
    }
}

/// A `ddl.constraint`'s `cmp` attribute (`ddc/ddl/ddl_conversion.cpp:2641`), which
/// `verifyDdlConstraints` admits in exactly two spellings.
///
/// ⛔ A CLOSED SET THE AUTHORITY KEEPS AS A `StringRef`, compared against `"equal"` and `"less"` and
/// aborting on anything else (`:2658`, `:2749-2755`, `:2697`). The variant names are the literals',
/// and `build.rs` already emits this type under this name from its own census (`build.rs:3158-3160`,
/// `:3209`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ConstraintCmp {
    #[default]
    Equal,
    Less,
}

impl ConstraintCmp {
    /// The DDL's spelling. Absent is the authority's `"cmp" type not yet supported`
    /// (`ddc/ddl/ddl_conversion.cpp:2697`, `:2755`).
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "equal" => Some(Self::Equal),
            "less" => Some(Self::Less),
            _ => None,
        }
    }

    /// Whether the measured size satisfies the constraint's stated value
    /// (`ddc/ddl/ddl_conversion.cpp:2749-2752`).
    ///
    /// ⛔ ONLY `Equal` REACHES THE `dim_idx` AND `property` FORMS (`:2658`, `:2687`); a `less` there
    /// is the same abort as an unknown spelling.
    pub fn holds(self, size: DimVal, value: DimVal) -> bool {
        match self {
            Self::Equal => size == value,
            Self::Less => size < value,
        }
    }
}

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

    use crate::schedule::dsc::NumCoreletsUsed;
    use crate::schedule::dsc2::{LoopCond, LoopCondConjunction};

    /// A conversion over a DSC and metadata that have not been filled — every test below states the
    /// fields its own arm reads.
    fn conversion() -> DdlConversion {
        DdlConversion::new(
            DesignSpaceConfig::default(),
            Metadata::default(),
            Verbosity(0),
        )
    }

    /// One condition term, for the loop-composition tests below.
    fn term(dim: PrimaryDimTypes) -> LoopCond {
        LoopCond {
            dim,
            cond_op: LoopCondOp::Eq,
            cond_val: CondVal::Last,
        }
    }

    /// `strtof` must have consumed the whole string bar whitespace, else `DT_ERROR`
    /// (`ddc/ddl/ddl_conversion.cpp:2074-2078`).
    #[test]
    fn process_expression_wants_the_whole_string_and_nothing_else() {
        assert_eq!(DdlConversion::process_expression(" 2.5 "), Some(2.5));
        assert_eq!(DdlConversion::process_expression("-3"), Some(-3.0));
        assert_eq!(DdlConversion::process_expression("2.5 cores"), None);
        assert_eq!(DdlConversion::process_expression(""), None);
    }

    /// The named forms win, and anything else is a number truncated into the caller's `int`
    /// (`ddc/ddl/ddl_conversion.cpp:265-273`).
    #[test]
    fn parse_cond_val_prefers_the_named_forms_and_truncates_the_rest() {
        assert_eq!(DdlConversion::parse_cond_val("first"), Some(CondVal::First));
        assert_eq!(DdlConversion::parse_cond_val("last"), Some(CondVal::Last));
        assert_eq!(
            DdlConversion::parse_cond_val("2.7"),
            Some(CondVal::Iteration(IterationIdx(2)))
        );
        assert_eq!(DdlConversion::parse_cond_val("outermost"), None);
    }

    /// A dropped dim is a loop of size one, so `FIRST`/`LAST` are iteration zero and the operator is
    /// applied against it (`ddc/ddl/ddl_conversion.cpp:274-292`).
    #[test]
    fn a_dropped_dim_resolves_its_condition_against_iteration_zero() {
        for cond_val in [CondVal::First, CondVal::Last] {
            assert!(DdlConversion::resolve_dropped_dim_condition(
                LoopCondOp::Eq,
                cond_val
            ));
            assert!(!DdlConversion::resolve_dropped_dim_condition(
                LoopCondOp::Ne,
                cond_val
            ));
        }
        let two = CondVal::Iteration(IterationIdx(2));
        assert!(DdlConversion::resolve_dropped_dim_condition(
            LoopCondOp::Gt,
            two
        ));
        assert!(!DdlConversion::resolve_dropped_dim_condition(
            LoopCondOp::Le,
            two
        ));
    }

    /// `condNot` over a core/corelet condition: a stated core's corelets toggle and an unstated core
    /// gains 0 and 1 (`ddc/ddl/ddl_conversion.cpp:327-341`) — and a complement that empties every
    /// used core is the tail's `resolvedValue_ = false`.
    #[test]
    fn negating_a_core_condition_toggles_every_used_core() {
        let mut conv = conversion();
        conv.dsc.core_ids_used = vec![CoreId(0), CoreId(1)];
        conv.dsc.num_corelets_used_dsc2 = Some(NumCoreletsUsed(2));
        let stated = BTreeMap::from([(CoreId(0), BTreeSet::from([CoreletId(0)]))]);
        let expected = BTreeMap::from([
            (CoreId(0), BTreeSet::from([CoreletId(1)])),
            (CoreId(1), BTreeSet::from([CoreletId(0), CoreletId(1)])),
        ]);
        assert!(
            matches!(conv.negate_condition(CondProp::CoreCl(stated)), CondProp::CoreCl(got) if got == expected)
        );

        let mut solo = conversion();
        solo.dsc.core_ids_used = vec![CoreId(0)];
        solo.dsc.num_corelets_used_dsc2 = Some(NumCoreletsUsed(1));
        let full = BTreeMap::from([(CoreId(0), BTreeSet::from([CoreletId(0)]))]);
        assert_eq!(
            solo.negate_condition(CondProp::CoreCl(full))
                .resolved_value(),
            Some(false)
        );
    }

    /// `condNot`'s other two arms invert only their own half
    /// (`ddc/ddl/ddl_conversion.cpp:322-326`).
    #[test]
    fn negating_a_resolved_or_loop_condition_flips_only_its_own_half() {
        let conv = conversion();
        assert_eq!(
            conv.negate_condition(CondProp::Resolved(true))
                .resolved_value(),
            Some(false)
        );
        let guard: LoopCondComposite = LoopCondConjunction::new(term(PrimaryDimTypes::Y)).into();
        assert!(
            matches!(conv.negate_condition(CondProp::Loop(guard)), CondProp::Loop(got) if got.negated)
        );
    }

    /// `ConditionAndOp` appends the operand's terms to one clause, and a negated operand is the
    /// "two-level OR of ANDs" refusal (`ddc/ddl/ddl_conversion.cpp:380-392`).
    #[test]
    fn an_and_of_loop_conditions_appends_terms_to_one_clause() {
        let earlier: LoopCondComposite = LoopCondConjunction::new(term(PrimaryDimTypes::Y)).into();
        let later: LoopCondComposite = LoopCondConjunction::new(term(PrimaryDimTypes::X)).into();
        let composed = DdlConversion::and_conditions([
            CondProp::Loop(earlier.clone()),
            CondProp::Loop(later.clone()),
        ]);
        assert!(matches!(composed, Some(CondProp::Loop(got))
            if got.or_of_ands.clause_count().get() == 1
                && got.or_of_ands.clauses().next().is_some_and(|c| c.term_count().get() == 2)));
        assert!(
            DdlConversion::and_conditions([
                CondProp::Loop(earlier.negate()),
                CondProp::Loop(later),
            ])
            .is_none()
        );
    }

    /// `ConditionOrOp` appends clauses (`ddc/ddl/ddl_conversion.cpp:401-403`), and the core/corelet
    /// half unions on OR and intersects per core on AND (`:413-430`).
    #[test]
    fn an_or_appends_clauses_and_a_core_condition_unions_or_intersects() {
        let earlier: LoopCondComposite = LoopCondConjunction::new(term(PrimaryDimTypes::Y)).into();
        let later: LoopCondComposite = LoopCondConjunction::new(term(PrimaryDimTypes::X)).into();
        let composed =
            DdlConversion::or_conditions([CondProp::Loop(earlier), CondProp::Loop(later)]);
        assert!(
            matches!(composed, Some(CondProp::Loop(got)) if got.or_of_ands.clause_count().get() == 2)
        );

        let one = CondProp::CoreCl(BTreeMap::from([
            (CoreId(0), BTreeSet::from([CoreletId(0)])),
            (CoreId(1), BTreeSet::from([CoreletId(0)])),
        ]));
        let two = CondProp::CoreCl(BTreeMap::from([(
            CoreId(0),
            BTreeSet::from([CoreletId(1)]),
        )]));
        assert!(matches!(
            DdlConversion::or_conditions([one.clone(), two.clone()]),
            Some(CondProp::CoreCl(got))
                if got == BTreeMap::from([
                    (CoreId(0), BTreeSet::from([CoreletId(0), CoreletId(1)])),
                    (CoreId(1), BTreeSet::from([CoreletId(0)])),
                ])
        ));
        assert_eq!(
            DdlConversion::and_conditions([one, two]).and_then(|cond| cond.resolved_value()),
            Some(false)
        );
    }

    /// A resolved operand is the junction's identity, whichever side it arrives on, or its
    /// annihilator — and an empty junction is the tail normalisation
    /// (`ddc/ddl/ddl_conversion.cpp:346-360`, `:373-376`, `:438-441`).
    #[test]
    fn a_resolved_operand_is_its_junctions_identity_or_its_annihilator() {
        let core = CondProp::CoreCl(BTreeMap::from([(
            CoreId(0),
            BTreeSet::from([CoreletId(0)]),
        )]));
        assert!(matches!(
            DdlConversion::and_conditions([CondProp::Resolved(true), core.clone()]),
            Some(CondProp::CoreCl(_))
        ));
        assert!(matches!(
            DdlConversion::and_conditions([core.clone(), CondProp::Resolved(true)]),
            Some(CondProp::CoreCl(_))
        ));
        assert_eq!(
            DdlConversion::and_conditions([core.clone(), CondProp::Resolved(false)])
                .and_then(|cond| cond.resolved_value()),
            Some(false)
        );
        assert_eq!(
            DdlConversion::or_conditions([core, CondProp::Resolved(true)])
                .and_then(|cond| cond.resolved_value()),
            Some(true)
        );
        assert_eq!(
            DdlConversion::and_conditions(std::iter::empty())
                .and_then(|cond| cond.resolved_value()),
            Some(false)
        );
    }

    /// "And/or op is mixing incompatible types", both ways round
    /// (`ddc/ddl/ddl_conversion.cpp:371-372`, `:407-408`).
    #[test]
    fn mixing_a_loop_and_a_core_condition_is_refused() {
        let guard: LoopCondComposite = LoopCondConjunction::new(term(PrimaryDimTypes::Y)).into();
        let core = CondProp::CoreCl(BTreeMap::from([(
            CoreId(0),
            BTreeSet::from([CoreletId(0)]),
        )]));
        assert!(
            DdlConversion::and_conditions([CondProp::Loop(guard.clone()), core.clone()]).is_none()
        );
        assert!(DdlConversion::or_conditions([core, CondProp::Loop(guard)]).is_none());
    }

    /// `min_num_cores` compares against `numCoresUsed_`, which DSM has not written on a fresh DSC
    /// (`ddc/ddl/ddl_conversion.cpp:2558-2564`).
    #[test]
    fn min_num_cores_is_unanswerable_until_dsm_writes_the_count() {
        let mut conv = conversion();
        assert_eq!(conv.min_num_cores_met(NumCoresUsed(1)), None);
        conv.dsc.num_cores_used = Some(NumCoresUsed(4));
        assert_eq!(conv.min_num_cores_met(NumCoresUsed(4)), Some(true));
        assert_eq!(conv.min_num_cores_met(NumCoresUsed(5)), Some(false));
    }

    /// The unstated bounds are 0 and 100 (`ddc/ddl/ddl_conversion.cpp:2603-2604`).
    #[test]
    fn an_unstated_num_valid_maximum_is_a_hundred() {
        assert_eq!(DdlConversion::num_valid_range(Some(1), Some(1)), 1..=1);
        assert_eq!(DdlConversion::num_valid_range(Some(1), None), 1..=100);
        assert_eq!(DdlConversion::num_valid_range(None, Some(2)), 0..=2);
    }

    /// Strictly increasing, starting from `INT_MIN` so the first bound operand always passes
    /// (`ddc/ddl/ddl_conversion.cpp:2621-2637`).
    #[test]
    fn relative_op_order_wants_strictly_increasing_indices() {
        assert!(DdlConversion::relative_op_order_met([
            ComputeOpIdx(0),
            ComputeOpIdx(3),
            ComputeOpIdx(7)
        ]));
        assert!(!DdlConversion::relative_op_order_met([
            ComputeOpIdx(3),
            ComputeOpIdx(3)
        ]));
        assert!(!DdlConversion::relative_op_order_met([
            ComputeOpIdx(7),
            ComputeOpIdx(3)
        ]));
        assert!(DdlConversion::relative_op_order_met(std::iter::empty()));
    }

    /// The property-less dim constraint reads the core stage's steady state: the plain dim, a
    /// padding scalar, or the dim under the form its kind names
    /// (`ddc/ddl/ddl_conversion.cpp:2710-2748`).
    #[test]
    fn a_dim_constraint_reads_the_core_stages_steady_state() {
        use crate::schedule::dims::{DataStructDims, DimPaddingSizes, DimSize};
        use crate::schedule::dsc2::DataStage;

        let mut conv = conversion();
        conv.dsc.data_stage_param.insert(
            Metadata::CORE_DSTGID,
            DataStage {
                ss: DataStructDims {
                    mb: DimSize::new(8.0),
                    padding_sizes: BTreeMap::from([(
                        PrimaryDimTypes::Mb,
                        DimPaddingSizes {
                            pad_front: 3,
                            ..DimPaddingSizes::default()
                        },
                    )]),
                    ..DataStructDims::default()
                },
                ..DataStage::default()
            },
        );

        let unpadded = DimProp {
            dim: PrimaryDimTypes::Mb,
            ..DimProp::default()
        };
        assert_eq!(conv.dim_constraint_size(&unpadded), Some(DimVal(8)));

        let mut pad_front = unpadded.clone();
        pad_front.set_meta_dim_kind(MetaDimKind::PadFront);
        assert_eq!(conv.dim_constraint_size(&pad_front), Some(DimVal(3)));

        let mut padded = unpadded.clone();
        padded.set_meta_dim_kind(MetaDimKind::Padded);
        assert_eq!(conv.dim_constraint_size(&padded), Some(DimVal(11)));

        let mut dropped = pad_front.clone();
        dropped.drop_dim = true;
        assert_eq!(conv.dim_constraint_size(&dropped), None);

        let mut undefined = unpadded;
        undefined.set_meta_dim_kind(MetaDimKind::Undefined);
        assert_eq!(conv.dim_constraint_size(&undefined), None);
    }

    /// `verifyDdlConstraints` admits two `cmp` spellings and aborts on the rest
    /// (`ddc/ddl/ddl_conversion.cpp:2749-2755`).
    #[test]
    fn constraint_cmp_admits_only_equal_and_less() {
        assert_eq!(
            ConstraintCmp::from_name("equal"),
            Some(ConstraintCmp::Equal)
        );
        assert_eq!(ConstraintCmp::from_name("less"), Some(ConstraintCmp::Less));
        assert_eq!(ConstraintCmp::from_name("greater"), None);
        assert!(ConstraintCmp::Equal.holds(DimVal(8), DimVal(8)));
        assert!(!ConstraintCmp::Equal.holds(DimVal(8), DimVal(9)));
        assert!(ConstraintCmp::Less.holds(DimVal(8), DimVal(9)));
        assert!(!ConstraintCmp::Less.holds(DimVal(9), DimVal(9)));
    }

    /// One `core_to_core_communication` op's fill in the authority's own order: the `DT_CHECK` on a
    /// fresh entry, the Map/Map fold space over core and corelet, and the cl1 swap
    /// (`ddc/ddl/ddl_conversion.cpp:1939-1949`, `:1959`, `:1977-1988`).
    #[test]
    fn core_to_core_starts_unmapped_and_rings_its_two_corelets_opposite_ways() {
        use crate::schedule::fold::{BaseFuncType, FoldDimIndex, FoldDimProp, FoldDimSize};

        let mut c2c = CoreToCore::default();
        assert_eq!(c2c.dim, PrimaryDimTypes::Undefined);
        assert_eq!(c2c.communication_dim(), None);
        assert!(c2c.next_core.has_zero_fold_dim());
        assert!(c2c.prev_core.has_zero_fold_dim());

        let props = [
            FoldDimProp::new(FoldDimSize(2), "core"),
            FoldDimProp::new(FoldDimSize(2), "corelet"),
        ];
        for fold in [&mut c2c.next_core, &mut c2c.prev_core] {
            assert_eq!(
                fold.build_fold_space(&props, &[BaseFuncType::Map; 2]),
                Some(())
            );
        }
        c2c.dim = PrimaryDimTypes::Ki;
        assert_eq!(c2c.communication_dim(), Some(PrimaryDimTypes::Ki));

        // Core 0 sends to core 1 on corelet 0 and receives from it on corelet 1.
        let (cl0, cl1) = (
            [FoldDimIndex(0), FoldDimIndex(0)],
            [FoldDimIndex(0), FoldDimIndex(1)],
        );
        assert_eq!(c2c.next_core.insert_data(1, &cl0), Some(()));
        assert_eq!(c2c.prev_core.insert_data(1, &cl1), Some(()));
        assert_eq!(c2c.next_core.get_data(&cl0), Some(1));
        assert_eq!(c2c.prev_core.get_data(&cl1), Some(1));
        assert!(!c2c.next_core.has_zero_fold_dim());
    }
}
