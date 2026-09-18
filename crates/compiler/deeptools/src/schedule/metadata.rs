//! Re-ported from the C++ authority. See crustify-scheduler/AGENT-BRIEF.md.
//!
//! `ddc/ddc_metadata.h` is the side table stage 2b carries beside the `SuperDsc`: what each data
//! stage's size is constrained to, what each transfer and each opaque op needs, and which
//! allocations the pass minted.

use crate::schedule::dims::{MetaDimKind, PadType, PrimaryDimTypes};
use crate::schedule::dsc2::{AllocateNode, DataStageId, LdsIdx, TransferNode};
use std::collections::{BTreeMap, BTreeSet};
use sys_arch_spec::arch_enums::{OpFunc, SenComponent};
use sys_arch_spec::{CoreId, CoreletId, RowId};

/// Replaces: e001_FailedAlloc
///
/// The memory-tracker key an allocation would not fit in — `ddc/ddc_metadata.h:24-29`.
///
/// ⭐ THE FOUR FIELDS ARE `getTracker`'S ARGUMENT LIST, in its order
/// (`sys-arch-spec/memtracker/mem_track_bundle.h:34`): `Ddc::allocAllMem` fills one the instant
/// `checkAndAddDs` — or `checkAndAddDsAtAddr` on the PTXRF-scale path (`ddc/ddcv1.cpp:335`) —
/// answers `DOESNT_FIT`, naming the tracker that refused (`ddc/ddcv1.cpp:344-369`).
/// ⛔ NO METHODS AND NO INITIALIZERS TO PORT. `ddc/ddcv1.cpp:363` default-constructs the struct and
/// assigns all four fields on the next four lines; the header declares no member function and no
/// default, and no other file in the authority names the type.
/// ⛔ AND NO READER AT ALL — THE FOUR FIELDS ARE WRITE-ONLY. `ddc/ddcv1.cpp:364-367` are the only
/// mentions of any field tree-wide: nothing reads, prints or compares `comp`, `core`, `corelet` or
/// `row`. What `:378` and `:436` read is the VECTOR's `size()`, never a field.
/// ⛔ ONE SLOT, NOT A LIST — DO NOT PORT `failedAllocs` AS A `Vec`. Its single `push_back` is
/// immediately followed by `return false` (`:368-369`), `tryAlloc` runs exactly once (`:377`) and
/// nothing clears the vector, so it holds AT MOST ONE element and `success` already implies
/// `size() == 0`: the `&&` at `:378` and `:436` is redundant — and with no field reader that makes
/// `failedAllocs` PROVABLY DEAD IN THE AUTHORITY: deleting the vector changes nothing `allocAllMem`
/// returns or commits. The record is a diagnostic nobody wired up, which is why the port carries the
/// key and stops there.
/// ⛔ AND NOT THE CONVERSE. The other false exit (`:264` — an opaque op whose unroll is `0`, over
/// `max_unroll_`, or not a power of two) records NO `FailedAlloc`, so a refused `allocAllMem`
/// names a tracker key only when a tracker is what refused.
/// ⭐ `Copy` because the fill site pushes it by value (`ddc/ddcv1.cpp:368`).
///
/// Transposing any two of the three `int` keys is `E0308`, twice per block below — measured, by
/// compiling each snippet by hand against the built rlibs, not inferred. It is what stops a
/// transposition being a silently wrong tracker; the corelet/row pair is the one the C++ cannot
/// catch even at runtime, because this fill site fixes both at `0` (`:205`, `:211`).
/// ⛔ ONE TRANSPOSITION PER BLOCK. `compile_fail` asserts only that the block AS A WHOLE does not
/// compile, so two wrong literals in one block pin NEITHER: measured 2026-09-18, correcting one of
/// them and leaving the other wrong still reported `compile fail ... ok`, while that same correction
/// alone in its own block FAILED. Three pairs, three blocks.
/// ⛔ AND THE LAST DOCTEST IS THE CONTROL, WITHOUT WHICH NONE OF THEM IS EVIDENCE. Stable rustdoc
/// accepts the `E0308` annotation WITHOUT CHECKING IT — annotating a deliberately wrong code still
/// reports `ok` — so `compile_fail` alone would also pass on a misspelled path, a renamed variant or
/// a field that stopped being `pub`. The control compiles the same literal untransposed through the
/// same public path, so each failure above is attributable to its one transposition.
/// ```compile_fail,E0308
/// use deeptools::schedule::metadata::FailedAlloc;
/// use sys_arch_spec::arch_enums::SenComponent;
/// use sys_arch_spec::{CoreId, CoreletId, RowId};
/// let _ = FailedAlloc {
///     comp: SenComponent::Lx,
///     core: CoreletId(0),
///     corelet: CoreId(3),
///     row: RowId(0),
/// };
/// ```
/// ```compile_fail,E0308
/// use deeptools::schedule::metadata::FailedAlloc;
/// use sys_arch_spec::arch_enums::SenComponent;
/// use sys_arch_spec::{CoreId, CoreletId, RowId};
/// let _ = FailedAlloc {
///     comp: SenComponent::Lx,
///     core: CoreId(3),
///     corelet: RowId(0),
///     row: CoreletId(0),
/// };
/// ```
/// ```compile_fail,E0308
/// use deeptools::schedule::metadata::FailedAlloc;
/// use sys_arch_spec::arch_enums::SenComponent;
/// use sys_arch_spec::{CoreId, CoreletId, RowId};
/// let _ = FailedAlloc {
///     comp: SenComponent::Lx,
///     core: RowId(0),
///     corelet: CoreletId(0),
///     row: CoreId(3),
/// };
/// ```
/// ```
/// use deeptools::schedule::metadata::FailedAlloc;
/// use sys_arch_spec::arch_enums::SenComponent;
/// use sys_arch_spec::{CoreId, CoreletId, RowId};
/// let _ = FailedAlloc {
///     comp: SenComponent::Lx,
///     core: CoreId(3),
///     corelet: CoreletId(0),
///     row: RowId(0),
/// };
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct FailedAlloc {
    /// The memory that refused — a key of `metadata.newAllocations_` (`ddc/ddcv1.cpp:184`, `:364`).
    pub comp: SenComponent,
    /// The core it refused on: an element of `coreIdsUsed_` (`dsc/designSpaceConfig.h:75`) for LX, L0
    /// and L0_SCALE, else that vector's `front()` standing proxy for all of them
    /// (`ddc/ddcv1.cpp:189-198`, `:365`).
    pub core: CoreId,
    /// The corelet it refused on: `0` alone, and the rest up to `numCoreletsUsed_DSC2_` only when the
    /// arch is newer than RCUDD1A and the component is L0 or L0_SCALE (`ddc/ddcv1.cpp:200-210`,
    /// `:366`).
    pub corelet: CoreletId,
    /// The register-file row it refused on. ⛔ ALWAYS `0` FROM THIS FILL SITE — `allocAllMem` fixes
    /// `rows` at the single proxy row (`ddc/ddcv1.cpp:211`, `:367`), so the field carries the tracker
    /// key's third component without this caller ever varying it.
    pub row: RowId,
}

/// One quantity of a data-stage size constraint — what [`Constraints::min`], [`Constraints::max`]
/// and [`Constraints::values`] hold (`ddc/ddc_metadata.h:37-38`, `float`).
///
/// ⛔ TWO CURRENCIES IN ONE SLOT, AND THE KEY IT HANGS UNDER PICKS WHICH: `checkConstraints`
/// compares the candidate `size` against `bound * refSize` with `refSize` fixed at 1 for the
/// absolute key (`ddc/ddcv1.cpp:837`, `:895-899`), so under a reference stage the value is the RATIO
/// `size / refSize` (`:901-902`) and under the absolute key an element count.
/// ⛔ FLOAT IS LOAD-BEARING, NOT A TRANSCRIPTION: the swap step stores `1.0 / bound`
/// (`ddc/ddcv1.cpp:759-765`) and the DDL scales an expression by a cumulative stick size before
/// storing it (`ddc/ddl/ddl_conversion.cpp:1836-1848`).
/// ⭐ NaN IS UNREPRESENTABLE, WHICH IS WHAT MAKES THE ORDERING BELOW TOTAL — the authority keeps
/// these in a `std::set<float>` (`:38`), and an ordered set needs a total order.
#[derive(Clone, Copy, Debug)]
pub struct ConstraintValue(f32);

impl ConstraintValue {
    /// The bound, or absent for a NaN no ordered set could place.
    pub fn new(value: f32) -> Option<Self> {
        if value.is_nan() { None } else { Some(Self(value)) }
    }

    /// The bound itself, as the `size` comparisons read it (`ddc/ddcv1.cpp:895-902`).
    pub const fn get(self) -> f32 {
        self.0
    }
}

impl PartialEq for ConstraintValue {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other).is_eq()
    }
}

impl Eq for ConstraintValue {}

impl PartialOrd for ConstraintValue {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ConstraintValue {
    /// `f32::total_cmp`, total because [`ConstraintValue::new`] excludes NaN.
    ///
    /// ⛔ IT SEPARATES `-0.0` FROM `0.0` where `std::set<float>` merges them. No writer produces a
    /// zero: the bounds are dimension sizes and products of them (`ddc/ddcv1.cpp:620`, `:1224`,
    /// `ddc/ddc_transformation.cpp:968-1035`) or a DDL expression scaled by a stick size
    /// (`ddc/ddl/ddl_conversion.cpp:1840-1848`).
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

/// The bounds one dimension set of one data stage must satisfy — `Metadata::Datastage::Constraints`
/// (`ddc/ddc_metadata.h:33-72`).
///
/// ⭐ ONE FUNCTION READS EVERY FIELD: the `checkConstraints` lambda of
/// `Ddc::exploreAssignDataStages` (`ddc/ddcv1.cpp:792-923`), which no ported function owns yet.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Constraints {
    /// Field: e026_Metadata.mustBeMultiple_
    ///
    /// The size must be a multiple of the reference stage's size if there is one, otherwise of
    /// [`min`](Self::min) (`ddc/ddc_metadata.h:34-35`). Absolute, that is the `fmodf` test, and
    /// min-less it is an error (`ddc/ddcv1.cpp:851-854`); relative, it is the no-epilogue
    /// divisibility test on [`loop_dim_kind`](Self::loop_dim_kind) (`:856-891`).
    pub must_be_multiple: bool,
    /// Field: e026_Metadata.loopDimKind_
    ///
    /// Which quantity of the dim the relative multiple applies to (`ddc/ddc_metadata.h:36`);
    /// `Padded` makes the check use the padded size (`ddc/ddcv1.cpp:867-880`).
    ///
    /// ⛔ NOT AN `Option`: the authority's not-set IS `MetaDimKind::Count`, which
    /// [`MetaDimKind::Undefined`] carries, and both readers compare against it by hand
    /// (`ddc/ddcv1.cpp:768-772`, `:857`). Its writers only ever store `Unpadded`, `Padded` or
    /// `WindowDim` (`ddc/ddcv1.cpp:612-619`, `ddc/ddl/ddl_conversion.cpp:1824-1826`).
    pub loop_dim_kind: MetaDimKind,
    /// Field: e026_Metadata.min_
    ///
    /// The smallest ratio allowed, absent for unbounded below (`ddc/ddcv1.cpp:898-899`) — which the
    /// dump spells `-inf` (`ddc/ddc_metadata.h:56-60`).
    pub min: Option<ConstraintValue>,
    /// Field: e026_Metadata.max_
    ///
    /// The largest ratio allowed, absent for unbounded above (`ddc/ddcv1.cpp:895-896`).
    pub max: Option<ConstraintValue>,
    /// Field: e026_Metadata.values_
    ///
    /// The exact ratios allowed, absent when nothing constrains them (`ddc/ddcv1.cpp:901-902`).
    ///
    /// ⛔ PRESENT AND EMPTY IS NOT ABSENT: it admits no size at all, and the DDL rejects an
    /// intersection that empties (`ddc/ddl/ddl_conversion.cpp:1850-1857`).
    pub values: Option<BTreeSet<ConstraintValue>>,
    /// Field: e026_Metadata.cannotBeSymbolic_
    ///
    /// No dim of the set may be symbolic (`ddc/ddc_metadata.h:39`). Written under the absolute key
    /// only, and `DT_CHECK`ed to be absolute where it is read (`ddc/ddcv1.cpp:689`, `:803-807`).
    pub cannot_be_symbolic: bool,
}

impl Default for Constraints {
    /// The authority's member initialisers (`ddc/ddc_metadata.h:34-39`): unbounded both ways, no
    /// value list, and the kind not set.
    fn default() -> Self {
        Self {
            must_be_multiple: false,
            loop_dim_kind: MetaDimKind::Undefined,
            min: None,
            max: None,
            values: None,
            cannot_be_symbolic: false,
        }
    }
}

impl Constraints {
    /// `ddc/ddc_metadata.h:40-42`. Raises the floor — the tighter of the two bounds wins.
    pub fn update_min(&mut self, new_val: ConstraintValue) {
        self.min = Some(match self.min {
            Some(min) => min.max(new_val),
            None => new_val,
        });
    }

    /// `ddc/ddc_metadata.h:43-45`. Lowers the ceiling.
    pub fn update_max(&mut self, new_val: ConstraintValue) {
        self.max = Some(match self.max {
            Some(max) => max.min(new_val),
            None => new_val,
        });
    }

    /// `ddc/ddc_metadata.h:46-48`. Intersects with what is already there — `set_intersect`
    /// (`util/utils.h:112-116`) — and takes the incoming set whole when nothing is.
    pub fn update_values(&mut self, new_vals: BTreeSet<ConstraintValue>) {
        self.values = Some(match &self.values {
            Some(values) => values.intersection(&new_vals).copied().collect(),
            None => new_vals,
        });
    }

    /// The text `dump` writes to `std::cerr`, the space before each comma included
    /// (`ddc/ddc_metadata.h:49-71`). Returned rather than printed, as
    /// [`DataStructDims::export_json`](crate::schedule::dims::DataStructDims::export_json) is.
    ///
    /// ⛔ FLOATS ARE SPELLED RUST'S WAY: `{}` is shortest-round-trip where C++'s `operator<<` gives
    /// six significant digits, so a third prints `0.33333334` and not `0.333333`. C++ also switches
    /// to exponent form at and above `1e6` and below `1e-4` (`1e+06`, `1e-05`) where `{}` never
    /// does. Nothing parses this text back.
    pub fn dump(&self) -> String {
        let mut text = String::from(if self.must_be_multiple {
            "mustBeMultiple_= T "
        } else {
            "mustBeMultiple_= F "
        });
        let kind = if self.loop_dim_kind == MetaDimKind::Undefined {
            "NOT_SET"
        } else {
            self.loop_dim_kind.name()
        };
        text.push_str(&format!("loopDimKind_= {kind} "));
        match self.min {
            Some(min) => text.push_str(&format!(", min_= {} ", min.get())),
            None => text.push_str(", min_= -inf "),
        }
        match self.max {
            Some(max) => text.push_str(&format!(", max_= {} ", max.get())),
            None => text.push_str(", max_= inf "),
        }
        text.push_str(", values_= {");
        if let Some(values) = &self.values {
            for value in values {
                text.push_str(&format!("{} ", value.get()));
            }
        }
        text.push_str("}\n");
        text
    }
}

/// What the data-stage exploration knows about one INTERNAL data stage — `Metadata::Datastage`
/// (`ddc/ddc_metadata.h:32-81`).
///
/// ⭐ ITS ABSENCE FROM [`Metadata::datastages`] IS WHAT "EXTERNAL" MEANS: every loop over the sorted
/// stages skips the ids the map does not hold, `// external` in the authority's own comment
/// (`ddc/ddcv1.cpp:688`, `:742`, `:751-752`, `:1232-1233`, `:1342`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Datastage {
    /// Field: e026_Metadata.constraints_
    ///
    /// Per reference stage, per dimension set, what that set is constrained to
    /// (`ddc/ddc_metadata.h:73-76`).
    ///
    /// ⭐ AN ABSENT REFERENCE IS THE AUTHORITY'S `-1` KEY — "reference is tensor, not a datastage"
    /// (`ddc/ddl/ddl_conversion.cpp:1800`), written as `constraints_[-1]` (`ddc/ddcv1.cpp:689`) and
    /// read as `refDsId < 0 ? nullptr : ...` (`:796-798`). The relative keys come from the loop that
    /// divides this stage, `constraints_[loop->numId_]` (`:610`).
    /// ⛔ SO [`None`] IS THE ONLY SPELLING OF THAT KEY, never `Some(DataStageId(-1))`: the writer at
    /// `:610` cannot produce one, because `numId_` is `-1` only on a parametric loop
    /// (`ddc/ddl/ddl_conversion.cpp:1129`) and `:599` skips those before reaching it.
    pub constraints: BTreeMap<Option<DataStageId>, BTreeMap<BTreeSet<PrimaryDimTypes>, Constraints>>,
    /// Field: e026_Metadata.strategyMinimize_
    ///
    /// Minimise this stage's size rather than maximise it (`ddc/ddc_metadata.h:77`, the DDL's
    /// `strategy` attribute, `ddc/ddl/ddl_conversion.cpp:1517-1522`). `exploreMaximizeDs` skips
    /// every stage that minimises (`ddc/ddcv1.cpp:1344`).
    pub strategy_minimize: bool,
    /// Field: e026_Metadata.allowEpilogue_
    ///
    /// The stage may leave a remainder, so the no-epilogue divisibility test is skipped
    /// (`ddc/ddc_metadata.h:78`, `ddc/ddcv1.cpp:856`, `:1418`).
    pub allow_epilogue: bool,
    /// Field: e026_Metadata.relevantDimsAndNumerator_
    ///
    /// Which dims this stage is explored over, each with the numerator stage of the loop that
    /// introduced it (`ddc/ddc_metadata.h:79`, filled at `ddc/ddcv1.cpp:615-616`). `calculateEpilogues`
    /// reads the pair to size the epilogue (`:1238-1244`).
    pub relevant_dims_and_numerator: BTreeMap<PrimaryDimTypes, DataStageId>,
    /// Field: e026_Metadata.nearestNumeratorIdx_
    ///
    /// The stage one loop level up, absent for the authority's `-1` (`ddc/ddc_metadata.h:80`). Set to
    /// `loop->numId_` (`ddc/ddcv1.cpp:609`) and followed upward as a chain (`:940`, `:990`).
    pub nearest_numerator_idx: Option<DataStageId>,
}

impl Default for Datastage {
    /// The authority's member initialisers (`ddc/ddc_metadata.h:77-80`): minimise, no epilogue, no
    /// numerator yet.
    fn default() -> Self {
        Self {
            constraints: BTreeMap::new(),
            strategy_minimize: true,
            allow_epilogue: false,
            relevant_dims_and_numerator: BTreeMap::new(),
            nearest_numerator_idx: None,
        }
    }
}

/// Field: e026_Metadata.TransferAccessPatternType
///
/// How one dimension is padded at each end of a transfer — the authority's
/// `Metadata::TransferAccessPatternType`, a `std::pair<PadType, PadType>` (`ddc/ddc_metadata.h:84`).
///
/// ⭐ NAMED HALVES BECAUSE `.first` AND `.second` ARE POSITIONAL: every reader takes the first for
/// the source and the second for the destination (`ddc/ddc_transformation_util.cpp:752-753`,
/// `ddc/ddc_fold.cpp:4487`, `:4530`), and the DDL's wire spelling is `<src>-to-<dst>` in that order
/// (`ddc/ddl/ddl_conversion.cpp:3629-3643`, parsed back at `:3687-3716`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TransferAccessPattern {
    /// `first` — the padding the source is read with (`ddc/ddc_fold.cpp:4487`).
    pub src: PadType,
    /// `second` — the padding the destination is written with (`ddc/ddc_fold.cpp:4530`).
    pub dst: PadType,
}

/// Field: e026_Metadata.TransferAccessPatternPerDimType
///
/// The authority's `Metadata::TransferAccessPatternPerDimType` (`ddc/ddc_metadata.h:85-86`): one
/// [`TransferAccessPattern`] per dimension, and a `std::map`, so ordered and not a hash map.
pub type TransferAccessPatternPerDim = BTreeMap<PrimaryDimTypes, TransferAccessPattern>;

/// A forced element count on a stick-replicated dim — what `force_num_elements_` holds
/// (`ddc/ddc_metadata.h:97`), `8` in the authority's own template
/// (`ddc/ddl_templates/exx2_32.ddl:162`).
///
/// ⛔ `u32`, BECAUSE THE SIGN IS NOT A VALUE HERE: every negative `int` behaves exactly as the `-1`
/// initialiser at all four reader predicates (`ddc/ddcv1.cpp:458`, `:511`, `:515`, `:529`), so
/// absence is spelled by [`Option`] and a negative count must not be a second spelling of it.
///
/// ```
/// use deeptools::schedule::metadata::{DataTransfer, ForcedNumElements};
/// let mut transfer = DataTransfer::default();
/// transfer.force_num_elements = Some(ForcedNumElements(8));
/// assert_eq!(transfer.force_num_elements, Some(ForcedNumElements(8)));
/// ```
/// ```compile_fail,E0600
/// use deeptools::schedule::metadata::ForcedNumElements;
/// let _ = ForcedNumElements(-1);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ForcedNumElements(pub u32);

/// What stage 2b knows about one transfer node — `Metadata::DataTransfer`
/// (`ddc/ddc_metadata.h:88-118`). Its own eight fields plus the access-pattern list behind them.
///
/// ⭐ ITS READER IS `fillLoopOffsetsAndAddresses`, which picks exactly ONE of the offset branches
/// per transfer, and ⛔ NOT IN DECLARATION ORDER: the chain is `apply_row_offset_src_` (`:2881`),
/// `apply_row_offset_dst_` (`:2895`), `replicated_` (`:2913`), `apply_pe_sfp_split_offset_src_`
/// (`:2967`), `apply_pe_sfp_split_offset_dest_` (`:2999`), so `replicated_` outranks both PE/SFP
/// fields declared above it (`ddc/ddcv1.cpp:2879-3040`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DataTransfer {
    /// Field: e026_Metadata.apply_row_offset_src_
    ///
    /// Offset the source by the destination's PT row (`ddc/ddc_metadata.h:90`). Set when a row-split
    /// copy has no source units left (`ddc/ddl/ddl_conversion.cpp:1333`), applied at
    /// `ddc/ddcv1.cpp:2881-2894`.
    pub apply_row_offset_src: bool,
    /// Field: e026_Metadata.apply_row_offset_dst_
    ///
    /// The same for the destination side, by the SOURCE's row (`ddc/ddc_metadata.h:91`,
    /// `ddc/ddl/ddl_conversion.cpp:1334`, `ddc/ddcv1.cpp:2895-2911`).
    pub apply_row_offset_dst: bool,
    /// Field: e026_Metadata.apply_pe_sfp_split_offset_src_
    ///
    /// The clone's source starts one original transfer size further in
    /// (`ddc/ddc_metadata.h:92`, set by the PE/SFP split at
    /// `ddc/ddc_transformation_util.cpp:1635`, applied at `ddc/ddcv1.cpp:2967-2998`).
    pub apply_pe_sfp_split_offset_src: bool,
    /// Field: e026_Metadata.apply_pe_sfp_split_offset_dest_
    ///
    /// The destinations that take that offset, by INDEX into `dstLdsAndLoopOffsets_` — the vector
    /// that bounds the push (`ddc/ddc_transformation_util.cpp:1637`, pushed at `:1642`) and the one
    /// the reader indexes (`dstLdsAndLoopOffsets_.at(destInd)`, `ddc/ddcv1.cpp:3027`). `dstVias_` is
    /// only what the push TESTS (`:1638-1640`). `ddc/ddc_metadata.h:93`; `usize` makes a negative
    /// destination unrepresentable.
    pub apply_pe_sfp_split_offset_dest: Vec<usize>,
    /// Field: e026_Metadata.replicated_
    ///
    /// This transfer is a replication clone (`ddc/ddc_metadata.h:94`).
    ///
    /// ⛔ NO WRITER TREE-WIDE, so the branch it guards is unreachable in the authority as it stands:
    /// `ddc/ddcv1.cpp:2913` is the ONLY occurrence besides the declaration, and it opens the
    /// clone-offset branch that reads the two fields below.
    pub replicated: bool,
    /// Field: e026_Metadata.offset_src_
    ///
    /// Whether the replicated source takes an offset (`ddc/ddc_metadata.h:95`). ⛔ NO WRITER EITHER,
    /// and both readers test it as `> 0` rather than using the magnitude
    /// (`ddc/ddcv1.cpp:2914`, `:2933`).
    pub offset_src: i32,
    /// Field: e026_Metadata.offset_dest_
    ///
    /// The same per destination, keyed by INDEX into `dstLdsAndLoopOffsets_`
    /// (`ddc/ddc_metadata.h:96`, `ddc/ddcv1.cpp:2948-2957`). ⛔ NO WRITER; the value is read as
    /// `> 0` only (`:2950`).
    pub offset_dest: BTreeMap<usize, i32>,
    /// Field: e026_Metadata.force_num_elements_
    ///
    /// The DDL's element cap on a stick-replicated dim, absent for any NEGATIVE authority value
    /// (`ddc/ddc_metadata.h:97`, written at `ddc/ddl/ddl_conversion.cpp:1235-1238`).
    ///
    /// ⛔ NOT ONE `> 0` GATE, AND `0` IS ON THE FORCED SIDE OF IT: `populateUnitTimeTransfers` tests
    /// this one value with `> 0` (`ddc/ddcv1.cpp:458`), `< 0` (`:511`) and `>= 0` (`:515`, `:529`),
    /// so at `0` the chunk loop never runs — `unitTimeTransferChunkSize_` is left EMPTY and `:530`
    /// writes `replicationFactor_ = 0` — where a negative value takes every stick size and computes
    /// the factor from them (`:531-535`). Only the DDL's reader gates on `> 0` alone (`:3217`).
    pub force_num_elements: Option<ForcedNumElements>,
    /// Field: e026_Metadata.accessPatternPerDim_
    ///
    /// The per-dim padding pair (`ddc/ddc_metadata.h:117`), private as the authority declares it.
    /// The DDL fills it through [`access_pattern_list_mut`](DataTransfer::access_pattern_list_mut)
    /// (`ddc/ddl/ddl_conversion.cpp:1217-1227`, `:3728-3745`).
    access_pattern_per_dim: TransferAccessPatternPerDim,
}

impl DataTransfer {
    /// `ddc/ddc_metadata.h:98-101`. States one dimension's pattern.
    ///
    /// ⛔ NO CALLER IN THE AUTHORITY: the one filler goes through
    /// [`access_pattern_list_mut`](Self::access_pattern_list_mut) instead
    /// (`ddc/ddl/ddl_conversion.cpp:1226`). Ported because it is this class's own method over this
    /// class's own field.
    pub fn set_access_pattern(&mut self, dim_val: PrimaryDimTypes, access_pattern: TransferAccessPattern) {
        self.access_pattern_per_dim.insert(dim_val, access_pattern);
    }

    /// `ddc/ddc_metadata.h:102-104`. Whether this dimension has a pattern — the test both fold
    /// readers apply first (`ddc/ddc_fold.cpp:4486`, `:4529`).
    pub fn has_access_pattern(&self, dim_val: PrimaryDimTypes) -> bool {
        self.access_pattern_per_dim.contains_key(&dim_val)
    }

    /// `ddc/ddc_metadata.h:105`, the no-argument overload. Whether ANY dimension has one — read to
    /// keep a transfer out of a fold (`ddc/ddc_fold.cpp:4552`).
    pub fn has_any_access_pattern(&self) -> bool {
        !self.access_pattern_per_dim.is_empty()
    }

    /// This dimension's pattern (`ddc/ddl/ddl_conversion.cpp:3619-3628`).
    ///
    /// ⭐ ABSENT WHERE IBM `DT_ERROR`S, AND NO CALLER CAN REACH IT: the fold guards each call with
    /// [`has_access_pattern`](Self::has_access_pattern) (`ddc/ddc_fold.cpp:4486-4487`, `:4529-4530`).
    pub fn access_pattern(&self, dim_val: PrimaryDimTypes) -> Option<TransferAccessPattern> {
        self.access_pattern_per_dim.get(&dim_val).copied()
    }

    /// The DDL's wire spelling of that pattern, `<src>-to-<dst>`
    /// (`ddc/ddl/ddl_conversion.cpp:3629-3643`), which the DataflowIR attribute carries verbatim
    /// (`:3207-3212`) and `convertAccessPatternStrToDdcType` parses back (`:3687-3716`).
    pub fn access_pattern_as_str(&self, dim_val: PrimaryDimTypes) -> Option<String> {
        self.access_pattern(dim_val)
            .map(|pattern| format!("{}-to-{}", pattern.src.name(), pattern.dst.name()))
    }

    /// `ddc/ddc_metadata.h:108-110`. The whole list, to fill (`ddc/ddl/ddl_conversion.cpp:1226`).
    pub fn access_pattern_list_mut(&mut self) -> &mut TransferAccessPatternPerDim {
        &mut self.access_pattern_per_dim
    }

    /// `ddc/ddc_metadata.h:111-113`. The whole list, to read — `getTransferPadding` overwrites one
    /// padding per entry with it (`ddc/ddc_transformation_util.cpp:749-756`).
    pub fn access_pattern_list(&self) -> &TransferAccessPatternPerDim {
        &self.access_pattern_per_dim
    }

    /// The text `dump` writes to `std::cerr` (`ddc/ddl/ddl_conversion.cpp:3756-3768`), returned
    /// rather than printed.
    pub fn dump(&self) -> String {
        let mut text = String::from(
            "\n[Ddl::DataTransfer]\n--------------------------\n  access-patterns per dimension: ",
        );
        for (dim, pattern) in &self.access_pattern_per_dim {
            text.push_str(&format!(
                "\n    Dim {} access-pattern ({} to {})",
                dim.name(),
                pattern.src.name(),
                pattern.dst.name()
            ));
        }
        text.push('\n');
        text
    }
}

/// What stage 2b knows about one opaque compute op — `Metadata::OpaqueOp`
/// (`ddc/ddc_metadata.h:196-203`), all of it stated by the DDL's `OpaqueOp`
/// (`ddc/ddl/ddl_conversion.cpp:1577-1690`).
///
/// ⛔ THIS CARRIES FOUR OF THE SIX DECLARED FIELDS. `inOutRegAllocs_` (`:197`) and
/// `internalRegAlloc_` (`:200`) are `dsc2::AllocateNode*` held as pointer identity, which needs
/// e013's node identity — the blocker e018 and e023 report; their anchors stay open below.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpaqueOp {
    /// Field: e026_Metadata.internalRegs_
    ///
    /// The op's internal register names, in the DDL's order (`ddc/ddc_metadata.h:198`,
    /// `ddc/ddl/ddl_conversion.cpp:1656-1667`). `finalizeOps` turns each into an `R<n>` entry of the
    /// node's read/write register map (`ddc/ddcv1.cpp:3370-3379`), which is why these stay `String`
    /// as [`ComputeNode::read_write_reg_map`](crate::schedule::dsc2::InstrAttribute) does.
    pub internal_regs: Vec<String>,
    /// Field: e026_Metadata.internalRegsWithUnroll_
    ///
    /// How many of those names end in `_unroll` (`ddc/ddc_metadata.h:199`,
    /// `ddc/ddl/ddl_conversion.cpp:1663-1665`), so each further unroll step costs that many more
    /// registers (`ddc/ddcv1.cpp:267-269`).
    pub internal_regs_with_unroll: i32,
    /// Field: e026_Metadata.max_unroll_
    ///
    /// The largest unroll factor the op accepts (`ddc/ddc_metadata.h:201`, the DDL's
    /// `max_unroll_factor`, `ddc/ddl/ddl_conversion.cpp:1598`); a candidate above it is rejected
    /// (`ddc/ddcv1.cpp:262`).
    pub max_unroll: i32,
    /// Field: e026_Metadata.ldsIdx_
    ///
    /// The labelled data structure the op computes over, absent for the authority's `-1`
    /// (`ddc/ddc_metadata.h:202`, set from the new LDS at `ddc/ddl/ddl_conversion.cpp:1576-1578`).
    pub lds_idx: Option<LdsIdx>,
}

impl Default for OpaqueOp {
    /// The authority's member initialisers (`ddc/ddc_metadata.h:198-202`): no registers, no unroll
    /// beyond one, no LDS.
    fn default() -> Self {
        Self {
            internal_regs: Vec::new(),
            internal_regs_with_unroll: 0,
            max_unroll: 1,
            lds_idx: None,
        }
    }
}

/// The authority's `Metadata::DDCTransformationConfigT` (`ddc/ddc_metadata.h:230-232`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DdcTransformationConfig {
    /// Field: e026_Metadata.enableMovingDataTransfer
    ///
    /// Whether stage 2b may move data transfers (`ddc/ddc_metadata.h:231`). The DDL turns it off
    /// (`ddc/ddl/ddl_conversion.cpp:2044`) and `run_v1` gates the transformation on it
    /// (`ddc/ddcv1.cpp:3758`).
    pub enable_moving_data_transfer: bool,
}

impl Default for DdcTransformationConfig {
    /// `ddc/ddc_metadata.h:231` — on unless the DDL says otherwise.
    fn default() -> Self {
        Self {
            enable_moving_data_transfer: true,
        }
    }
}

/// The node pair one external transfer OWNS — `Metadata::ExternalTransfer`
/// (`ddc/ddc_metadata.h:130-136`), one entry per element of `externalTransfers_` (`:137`).
///
/// ⛔ DEAD IN THE AUTHORITY, AND SO IS THE VECTOR THAT HOLDS IT: those eight lines, plus the same
/// declaration repeated in the L3 near-duplicate (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.h:169-176`),
/// are EVERY occurrence tree-wide — no constructor call, no `push_back`, no reader. Carried because
/// the class declares it, and portable for exactly that reason: with no reader there is no second
/// handle on either node, so ownership is the entire contract and none of the aliasing that blocks
/// this type's other node fields applies.
///
/// ⭐ `Box` IS THE `unique_ptr` AND THE ABSENT `Clone` IS ITS MOVE-ONLY-NESS: the entry is the sole
/// owner of both nodes, copying it would free them twice, and that is why the authority's
/// `std::vector<ExternalTransfer>` cannot be copy-constructed and `Metadata::clear()` has to destroy
/// in place (`:224-228`) instead of assigning a fresh table.
#[derive(Debug)]
pub struct ExternalTransfer {
    /// Field: e026_Metadata.transfer_
    ///
    /// The transfer node this entry owns (`ddc/ddc_metadata.h:131`).
    pub transfer: Box<TransferNode>,
    /// Field: e026_Metadata.allocate_
    ///
    /// The allocate node this entry owns (`ddc/ddc_metadata.h:132`).
    pub allocate: Box<AllocateNode>,
}

impl ExternalTransfer {
    /// `ddc/ddc_metadata.h:133-135`. The authority's constructor ADOPTS two nodes built elsewhere —
    /// each `unique_ptr` is constructed from a raw pointer — so both boxes are taken by value and the
    /// caller gives up its handles.
    pub fn new(transfer: Box<TransferNode>, allocate: Box<AllocateNode>) -> Self {
        Self { transfer, allocate }
    }
}

/// The side table `Ddc` carries beside the `SuperDsc` for one program — `ddc::Metadata`
/// (`ddc/ddc_metadata.h:31-239`). `Ddc::run_v1` reads and writes it throughout
/// (`ddc/ddcv1.cpp:3695`).
///
/// ⛔ THIS CARRIES TWELVE OF METADATA'S OWN TWENTY-FOUR DECLARED FIELDS — ten fields and the two
/// `const int` data-stage ids as associated consts — so the `e026_Metadata` anchor below is still
/// open. The twelve left out all hold, or are keyed by, a SCHEDULE-NODE POINTER, which needs e029's
/// node identity, the same blocker e018 and e023 report:
///  * `datatransfers_` (`:119`) keys [`DataTransfer`] by `const dsc2::TransferNode*`, and
///    `opaqueOps_` (`:204`) keys [`OpaqueOp`] by `dsc2::ComputeNode*` — both values are ported here;
///  * `newAllocations_` (`:127`) and `shadowAllocations_` (`:128`) hold `dsc2::AllocateNode*`, as do
///    all three maps of the `Allocation` struct they use (`:121-125`);
///  * `prefilledExternalTransferToDataConnectToFill_` (`:138-139`) is an INTERIOR pointer into a
///    tree-owned node: it stores `&tn->srcLdsAndLoopOffsets_.dataConnect_`, or the
///    `dstLdsAndLoopOffsets_[0]` one (`ddc/ddcv1.cpp:2312`, `:2322`; `dstVias_` supplies only the
///    `storage` half of the key, `:2320`, and the map is keyed at `:2334-2336`) — the DDL WRITES THROUGH
///    it — `*prefilledIt->second = myExtAlloc.getDataConnect()` (`ddc/ddl/ddl_conversion.cpp:939`),
///    after rekeying the same entries in place (`:573-576`, `:2270-2277`). Naming a FIELD inside
///    another type's node is strictly more than naming the node;
///  * `externalNodes_` (`:140`), `TransferNodesInterSliceTranspose_` (`:141`), `implicitSyncs_`
///    (`:206`), `dimToCoreChunkLoops_` (`:208-209`), `nodeCloningMap_` (`:216-217`) and
///    `belowLxScheduleInsertBlock` (`:219`) are node sets, node-keyed maps and one bare pointer;
///  * `dataConnects_` (`:194`) is keyed by name, but its `DataConnect` (`:143-193`) is two vectors
///    of `dsc2::ScheduleNode*` and its `getLoops` walks `getOwnerLoop` to a set of `LoopNode*`
///    (`:179-192`).
///
/// ⛔ AND THE IDENTITY CANNOT BE THE NODE'S NAME: `ddl_conversion.cpp:1790` names EVERY implicit sync
/// node the literal `"sync_implicit_L0"` and `:1791` then inserts each one as its own
/// `implicitSyncs_` entry, keyed by pointer; `:1556` does the same with `"condition"`. A name key
/// would merge distinct nodes and hand the last allocation to all of them — so what these twelve
/// fields need is the ADDRESS-LIKE identity e029_ScheduleNode and e032_ScheduleTree must define.
///
/// ⛔ `dcg/dcg_fe/scheduler/L3DlOpsScheduler.h:111-198` IS A SMALLER NEAR-DUPLICATE OF THIS TYPE,
/// NOT THIS TYPE: its `core_dstgid`/`chunk_dstgid` are mutable and start at `-1`
/// (`:193-194`, written at `L3DlOpsScheduler.cpp:6420-6421`), it has one `apply_row_offset_` where
/// this has two, and it has no `Constraints::cannotBeSymbolic_`. It is not a scheduled unit.
///
/// ⛔ NOT `Clone`: THE AUTHORITY DELETES THE COPY TWICE OVER — the `const int core_dstgid` and
/// `chunk_dstgid` (`ddc/ddc_metadata.h:211-212`) delete copy- AND move-ASSIGNMENT, and the
/// `std::vector<ExternalTransfer>` of `unique_ptr`s (`:130-137`) deletes copy-CONSTRUCTION. Hence
/// `clear()` destroys in place (`:224-228`) where the near-duplicate, whose ids are mutable, gets
/// away with `*this = {}` (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.h:197`).
/// ```compile_fail
/// let _ = deeptools::schedule::metadata::Metadata::default().clone();
/// ```
/// [`Datastage`] declares neither (`ddc/ddc_metadata.h:32-81`) and does copy — the control that
/// proves the block above fails on the missing `Clone` and not on the doctest's shape:
/// ```
/// let _ = deeptools::schedule::metadata::Datastage::default().clone();
/// ```
///
/// ⛔ AND NOT `PartialEq` EITHER, FOR THE SAME `unique_ptr`: the authority declares no
/// `operator==` on `Metadata`, on `ExternalTransfer` or on either node type, and the only equality
/// available over an owned node is the node's VALUE — which is not what a pointer-identified node
/// compares by. Assert on the fields you mean instead.
/// ```compile_fail
/// use deeptools::schedule::metadata::Metadata;
/// let _ = Metadata::default() == Metadata::default();
/// ```
/// [`Datastage`] does compare (`ddc/ddc_metadata.h:32-81` declares no pointer and no `unique_ptr`) —
/// the control that proves the block above fails on the missing `PartialEq` and not on its shape:
/// ```
/// use deeptools::schedule::metadata::Datastage;
/// let _ = Datastage::default() == Datastage::default();
/// ```
///
/// A data-stage id is not a labelled-DS index, so keying the wrong map is a compile error:
/// ```compile_fail
/// use deeptools::schedule::dsc2::LdsIdx;
/// use deeptools::schedule::metadata::Metadata;
/// let _ = Metadata::default().datastages.get(&LdsIdx(0));
/// ```
#[derive(Debug)]
pub struct Metadata {
    /// Field: e026_Metadata.datastages_
    ///
    /// The INTERNAL data stages, by id (`ddc/ddc_metadata.h:82`). An id the map does not hold is an
    /// external stage and every loop over the sorted stages skips it (`ddc/ddcv1.cpp:742`, `:1342`).
    pub datastages: BTreeMap<DataStageId, Datastage>,
    /// Field: e026_Metadata.externalTransfers_
    ///
    /// The external transfers stage 2b minted, each OWNING its node pair
    /// (`ddc/ddc_metadata.h:137`). ⛔ DEAD, LIKE THE ENTRY TYPE: see [`ExternalTransfer`] — the
    /// declaration is the vector's only occurrence tree-wide, so nothing pushes and nothing reads.
    /// Carried because the class declares it, and it is the field that makes this type non-copyable.
    pub external_transfers: Vec<ExternalTransfer>,
    /// Field: e026_Metadata.rowSplitDim
    ///
    /// The dim split across PT rows (`ddc/ddc_metadata.h:213`), taken from the first stick dim
    /// without a slice layout (`ddc/ddcv1.cpp:2033`) and read by stage 2a as well
    /// (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:6335-6338`).
    ///
    /// ⛔ NOT AN `Option`, BECAUSE THE UNSET SENTINEL IS A LIVE MAP KEY: `fillLoopOffsetsAndAddresses`
    /// indexes `constEleOffsets_` with it unconditionally (`ddc/ddcv1.cpp:2887`), so
    /// [`PrimaryDimTypes::Undefined`] — the authority's `PrimaryDimTypesCount` — stays the value.
    pub row_split_dim: PrimaryDimTypes,
    /// Field: e026_Metadata.clSplitDims_
    ///
    /// The dims split across corelets (`ddc/ddc_metadata.h:214`), copied from the core stage's
    /// `coreletSplit_` (`ddc/ddcv1.cpp:2093-2096`).
    ///
    /// ⛔ IT IS PASSED EMPTY TO `finalizeExternalDataStage`, FROM THE LOOP THAT RUNS TO COMPLETION
    /// BEFORE IT IS FILLED (`ddc/ddcv1.cpp:2085-2091` against `:2093-2096`), so that callee sees no
    /// corelet split.
    pub cl_split_dims: BTreeSet<PrimaryDimTypes>,
    /// Field: e026_Metadata.peSfpSplitDims_
    ///
    /// The dims split between PE and SFP (`ddc/ddc_metadata.h:215`), from `getPeSfpSplitDim`
    /// (`ddc/ddcv1.cpp:2035`) and cleared again when the split is undone (`:3750`).
    pub pe_sfp_split_dims: BTreeSet<PrimaryDimTypes>,
    /// Field: e026_Metadata.discardAboveLxSchedule_
    ///
    /// `ddc/ddc_metadata.h:218`. ⛔ DEAD IN THE AUTHORITY: the declaration is its ONLY occurrence
    /// tree-wide — no writer, no reader — so nothing observes it. Carried because the class declares
    /// it.
    pub discard_above_lx_schedule: bool,
    /// Field: e026_Metadata.opFuncBackup_
    ///
    /// The op func of `computeOp_.at(0)` saved before the fold rewrites it
    /// (`ddc/ddc_metadata.h:221-222`, saved at `ddc/ddcv1.cpp:2068`, `:2075`).
    ///
    /// ⛔ NOT AN `Option`: [`OpFunc::None`] IS the authority's `OpFuncs::NONE` sentinel and the
    /// restore compares against it directly (`ddc/ddcv1.cpp:2273-2274`).
    /// ⛔ AND IT CAN HOLD THE REWRITE: both save sites sit in ONE `if` on `EXX2` and the `break` at
    /// `ddc/ddcv1.cpp:2070` leaves only the inner `for`, so `:2075` runs after `:2069` has already
    /// written `EXX2_ZEROMEAN` and backs THAT up — leaving `restoreDsc` to restore the rewrite.
    pub op_func_backup: OpFunc,
    /// Field: e026_Metadata.transformationConfig_
    ///
    /// The transformation switches (`ddc/ddc_metadata.h:230-232`).
    pub transformation_config: DdcTransformationConfig,
    /// Field: e026_Metadata.ldsIdxAfterDdc
    ///
    /// Where each labelled data structure ended up after stage 2b (`ddc/ddc_metadata.h:234-235`),
    /// seeded as the identity (`ddc/ddcv1.cpp:3669`), repointed when an LDS is replaced
    /// (`ddc/ddc_transformation_util.cpp:1908-1913`) and read back by the DataflowIR emission
    /// (`ddc/ddl/ddl_conversion.cpp:3378-3391`).
    pub lds_idx_after_ddc: BTreeMap<LdsIdx, LdsIdx>,
    /// Field: e026_Metadata.intermLdsIdxToExtLds
    ///
    /// Which external tensor an intermediate LDS stands for (`ddc/ddc_metadata.h:237-238`), written
    /// when a tensor is split into intermediates (`ddc/ddc_transformation.cpp:1085`, `:1090`).
    pub interm_lds_idx_to_ext_lds: BTreeMap<LdsIdx, LdsIdx>,
}

impl Default for Metadata {
    /// The member initialisers the authority states (`ddc/ddc_metadata.h:213`, `:218`, `:222`,
    /// `:231`): no data stages yet, no split dims, the row-split dim unset and the op-func backup at
    /// `OpFuncs::NONE`. ⛔ NOT DERIVED: `OpFunc` has no `Default`, and `PrimaryDimTypesCount` is the
    /// only correct start for `rowSplitDim`.
    fn default() -> Self {
        Self {
            datastages: BTreeMap::new(),
            external_transfers: Vec::new(),
            row_split_dim: PrimaryDimTypes::Undefined,
            cl_split_dims: BTreeSet::new(),
            pe_sfp_split_dims: BTreeSet::new(),
            discard_above_lx_schedule: false,
            op_func_backup: OpFunc::None,
            transformation_config: DdcTransformationConfig::default(),
            lds_idx_after_ddc: BTreeMap::new(),
            interm_lds_idx_to_ext_lds: BTreeMap::new(),
        }
    }
}

impl Metadata {
    /// Field: e026_Metadata.core_dstgid
    ///
    /// The core data stage's id — `const int core_dstgid = 0` (`ddc/ddc_metadata.h:211`).
    /// `attachToPrefilledSchedule`, not `run_v1`, checks that stage 0 is really named `"core"` and
    /// stage 1 `"chunk"` (`ddc/ddcv1.cpp:2280-2286`).
    ///
    /// ⭐ AN ASSOCIATED CONST, NOT A FIELD, BECAUSE `const int` MAKES IT ONE VALUE FOR EVERY
    /// `Metadata`: no writer exists tree-wide, and its `const`ness is half of what deletes this
    /// type's copy-assignment. The L3 near-duplicate's same-named member is NOT this one — it is
    /// mutable and starts at `-1` (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.h:193`).
    pub const CORE_DSTGID: DataStageId = DataStageId(0);

    /// Field: e026_Metadata.chunk_dstgid
    ///
    /// The chunk data stage's id — `const int chunk_dstgid = 1` (`ddc/ddc_metadata.h:212`). A loop
    /// from `CORE_DSTGID` to this one is the core-chunk loop (`ddc/ddl/ddl_conversion.cpp:1114-1115`).
    pub const CHUNK_DSTGID: DataStageId = DataStageId(1);

    /// `ddc/ddc_metadata.h:224-228`. Back to a fresh table: the authority destroys the object in
    /// place and re-constructs it, which is assignment from the default here.
    pub fn clear(&mut self) {
        *self = Self::default();
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::schedule::dsc2::NumBuffers;

    /// `ddc/ddcv1.cpp:362-369` — the LX tracker of a used core refuses, with the corelet and row at
    /// the proxy `0` that `corelets(1, 0)` and `rows(1, 0)` fix for it (`:205`, `:211`).
    ///
    /// ⛔ ALL FOUR SINGLE-FIELD MOVES, NOT JUST THE CORELET. `getTracker` keys on all four
    /// (`mem_track_bundle.h:34`), and the corelet move alone does not pin that: measured against a
    /// four-field stand-in whose hand-written `PartialEq` dropped `row`, the corelet-only assertion
    /// PASSED and the row move caught it. One field per move, never two — a move of two at once
    /// pins neither.
    #[test]
    fn every_field_is_part_of_the_tracker_key() {
        let failed = FailedAlloc {
            comp: SenComponent::Lx,
            core: CoreId(3),
            corelet: CoreletId(0),
            row: RowId(0),
        };
        assert_eq!(failed.comp, SenComponent::Lx);
        assert_eq!(failed.core, CoreId(3));
        assert_eq!(failed.corelet, CoreletId(0));
        assert_eq!(failed.row, RowId(0));
        for moved in [
            FailedAlloc {
                comp: SenComponent::L0,
                ..failed
            },
            FailedAlloc {
                core: CoreId(4),
                ..failed
            },
            FailedAlloc {
                corelet: CoreletId(1),
                ..failed
            },
            FailedAlloc {
                row: RowId(1),
                ..failed
            },
        ] {
            assert_ne!(
                failed, moved,
                "a one-field move must name a different tracker"
            );
        }
    }

    /// A bound that no ordered set could place is not a bound (`ddc/ddc_metadata.h:38`).
    #[test]
    fn a_constraint_value_is_never_nan() {
        assert_eq!(ConstraintValue::new(f32::NAN), None);
        assert_eq!(
            ConstraintValue::new(0.5).map(ConstraintValue::get),
            Some(0.5)
        );
        assert!(ConstraintValue::new(0.5) < ConstraintValue::new(2.0));
    }

    /// `ddc/ddc_metadata.h:40-48`: each update TIGHTENS, and a first update takes the value whole.
    /// The dump spells an absent bound `-inf`/`inf` and a not-set kind `NOT_SET` (`:49-71`).
    #[test]
    fn updating_a_constraint_tightens_it_from_both_sides() {
        let mut constraints = Constraints::default();
        assert_eq!(
            constraints.dump(),
            "mustBeMultiple_= F loopDimKind_= NOT_SET , min_= -inf , max_= inf , values_= {}\n"
        );

        constraints.update_min(ConstraintValue::new(2.0).expect("2 is not NaN"));
        constraints.update_min(ConstraintValue::new(1.0).expect("1 is not NaN"));
        constraints.update_max(ConstraintValue::new(8.0).expect("8 is not NaN"));
        constraints.update_max(ConstraintValue::new(16.0).expect("16 is not NaN"));
        constraints.update_values(
            [1.0, 2.0, 4.0]
                .into_iter()
                .filter_map(ConstraintValue::new)
                .collect(),
        );
        constraints.update_values(
            [2.0, 4.0, 8.0]
                .into_iter()
                .filter_map(ConstraintValue::new)
                .collect(),
        );
        constraints.must_be_multiple = true;
        constraints.loop_dim_kind = MetaDimKind::Unpadded;

        // The floor rose to 2 and the ceiling fell to 8; the values are the intersection.
        assert_eq!(
            constraints.dump(),
            "mustBeMultiple_= T loopDimKind_= unpadded , min_= 2 , max_= 8 , values_= {2 4 }\n"
        );
    }

    /// `ddc/ddcv1.cpp:607-620`, the writer: the loop that divides a stage names its numerator as the
    /// nearest one, keys the constraint under it, and caps the ratio at 1. `ddc/ddcv1.cpp:689` then
    /// adds an ABSOLUTE constraint under the `-1` key, which is a different entry
    /// (`ddc/ddl/ddl_conversion.cpp:1800`).
    #[test]
    fn a_datastage_keys_relative_and_absolute_constraints_apart() {
        let mut datastage = Datastage::default();
        assert!(datastage.strategy_minimize);
        assert_eq!(datastage.nearest_numerator_idx, None);

        let numerator = Metadata::CORE_DSTGID;
        datastage.nearest_numerator_idx = Some(numerator);
        datastage
            .relevant_dims_and_numerator
            .insert(PrimaryDimTypes::Mb, numerator);
        let relative = datastage
            .constraints
            .entry(Some(numerator))
            .or_default()
            .entry(BTreeSet::from([PrimaryDimTypes::Mb]))
            .or_default();
        relative.must_be_multiple = true;
        relative.loop_dim_kind = MetaDimKind::Unpadded;
        relative.update_max(ConstraintValue::new(1.0).expect("1 is not NaN"));

        datastage
            .constraints
            .entry(None)
            .or_default()
            .entry(BTreeSet::from([PrimaryDimTypes::Mb]))
            .or_default()
            .cannot_be_symbolic = true;

        assert_eq!(datastage.constraints.len(), 2);
        assert_eq!(
            datastage.constraints[&Some(numerator)][&BTreeSet::from([PrimaryDimTypes::Mb])].max,
            ConstraintValue::new(1.0)
        );
        assert!(datastage.constraints[&None][&BTreeSet::from([PrimaryDimTypes::Mb])].cannot_be_symbolic);
        // The absolute entry is not a relative one: it carries no bound at all.
        assert_eq!(
            datastage.constraints[&None][&BTreeSet::from([PrimaryDimTypes::Mb])].max,
            None
        );
    }

    /// `ddc/ddcv1.cpp:443-535` — the same forced count is read through THREE predicates, and `0` is
    /// on the forced side of two of them: at `0` the chunk loop at `:510-512` skips every stick, so
    /// `unitTimeTransferChunkSize_` stays empty and `:530` writes `replicationFactor_ = 0`, where
    /// absence takes the whole product instead (`:531-535`). The authority's own templates force `8`
    /// (`ddc/ddl_templates/exx2_32.ddl:162`).
    ///
    /// ⛔ A NEGATIVE COUNT IS NOT A THIRD STATE — it is unrepresentable, and the `compile_fail` block
    /// on [`ForcedNumElements`] is what pins that.
    #[test]
    fn a_forced_element_count_of_zero_is_not_an_absent_one() {
        let mut transfer = DataTransfer::default();
        assert_eq!(transfer.force_num_elements, None);

        transfer.force_num_elements = Some(ForcedNumElements(0));
        assert_eq!(transfer.force_num_elements, Some(ForcedNumElements(0)));

        transfer.force_num_elements = Some(ForcedNumElements(8));
        assert_eq!(transfer.force_num_elements, Some(ForcedNumElements(8)));
    }

    /// `ddc/ddl/ddl_conversion.cpp:1226` fills the list through the mutable handle, and
    /// `ddc/ddc_fold.cpp:4486-4530` reads the source padding from the first half and the destination
    /// padding from the second. An unstated dim has no pattern, where IBM `DT_ERROR`s.
    #[test]
    fn a_transfers_access_pattern_names_the_source_and_the_destination_halves() {
        let mut transfer = DataTransfer::default();
        assert!(!transfer.has_any_access_pattern());
        assert_eq!(transfer.access_pattern(PrimaryDimTypes::Mb), None);
        assert_eq!(transfer.force_num_elements, None);

        transfer.access_pattern_list_mut().insert(
            PrimaryDimTypes::Mb,
            TransferAccessPattern {
                src: PadType::NoPad,
                dst: PadType::PaddedWZeroPad,
            },
        );

        assert!(transfer.has_access_pattern(PrimaryDimTypes::Mb));
        assert!(!transfer.has_access_pattern(PrimaryDimTypes::Y));
        assert_eq!(
            transfer.access_pattern(PrimaryDimTypes::Mb).map(|p| p.src),
            Some(PadType::NoPad)
        );
        assert_eq!(
            transfer.access_pattern_as_str(PrimaryDimTypes::Mb),
            Some(String::from("nopad-to-padded_wzeropad"))
        );
        assert_eq!(transfer.access_pattern_as_str(PrimaryDimTypes::Y), None);
        assert_eq!(transfer.access_pattern_list().len(), 1);
        assert_eq!(
            transfer.dump(),
            "\n[Ddl::DataTransfer]\n--------------------------\n  access-patterns per dimension: \
             \n    Dim mb access-pattern (nopad to padded_wzeropad)\n"
        );
    }

    /// `ddc/ddc_metadata.h:198-199`: the unroll-register count is a SEPARATE field from the register
    /// list, because only the names ending in `_unroll` are charged again per step
    /// (`ddc/ddl/ddl_conversion.cpp:1663-1665`, spent at `ddc/ddcv1.cpp:267-269`). Both counts are
    /// carried; the charge itself belongs to `exploreOpMapping`, which no ported method owns yet, so
    /// nothing here re-derives it.
    #[test]
    fn an_opaque_ops_unroll_register_count_is_not_its_register_count() {
        let mut opaque = OpaqueOp::default();
        assert_eq!(opaque.max_unroll, 1);
        assert_eq!(opaque.lds_idx, None);
        assert!(opaque.internal_regs.is_empty());
        assert_eq!(opaque.internal_regs_with_unroll, 0);

        opaque.internal_regs.push(String::from("acc"));
        opaque.internal_regs.push(String::from("tmp_unroll"));
        opaque.internal_regs_with_unroll = 1;
        opaque.max_unroll = 4;
        opaque.lds_idx = Some(LdsIdx(7));

        assert_eq!(opaque.internal_regs.len(), 2);
        assert_eq!(opaque.internal_regs_with_unroll, 1);
        assert_eq!(opaque.max_unroll, 4);
        assert_eq!(opaque.lds_idx, Some(LdsIdx(7)));
    }

    /// `ddc/ddc_metadata.h:224-228` — `clear()` re-constructs the table, so every carried field goes
    /// back to its member initialiser: minimising stages gone, the row-split dim unset, the moving
    /// transfer switch back on (`:231`) and the op-func backup back to `NONE` (`:222`).
    ///
    /// ⛔ FIELD BY FIELD, NOT `assert_eq!(metadata, Metadata::default())`: [`Metadata`] owns a node
    /// pair per external transfer and so states no equality at all, and a whole-struct comparison
    /// against the default would in any case have passed on a `clear()` that dropped a field whose
    /// default is empty too.
    #[test]
    fn clearing_the_metadata_restores_every_member_initialiser() {
        let mut metadata = Metadata::default();
        assert_eq!(Metadata::CORE_DSTGID, DataStageId(0));
        assert_eq!(Metadata::CHUNK_DSTGID, DataStageId(1));
        assert_eq!(metadata.row_split_dim, PrimaryDimTypes::Undefined);
        assert!(metadata.transformation_config.enable_moving_data_transfer);
        assert_eq!(metadata.op_func_backup, OpFunc::None);

        metadata
            .datastages
            .insert(Metadata::CORE_DSTGID, Datastage::default());
        metadata.external_transfers.push(ExternalTransfer::new(
            Box::new(TransferNode {
                replication_factor: 2,
                ..TransferNode::default()
            }),
            Box::new(AllocateNode {
                num_buffers: NumBuffers::STREAMING,
                ..AllocateNode::default()
            }),
        ));
        metadata.row_split_dim = PrimaryDimTypes::In;
        metadata.cl_split_dims.insert(PrimaryDimTypes::Mb);
        metadata.pe_sfp_split_dims.insert(PrimaryDimTypes::Y);
        metadata.discard_above_lx_schedule = true;
        metadata.op_func_backup = OpFunc::Add;
        metadata.transformation_config.enable_moving_data_transfer = false;
        metadata.lds_idx_after_ddc.insert(LdsIdx(0), LdsIdx(2));
        metadata.interm_lds_idx_to_ext_lds.insert(LdsIdx(2), LdsIdx(0));

        metadata.clear();

        assert!(metadata.datastages.is_empty());
        assert!(metadata.external_transfers.is_empty());
        assert_eq!(metadata.row_split_dim, PrimaryDimTypes::Undefined);
        assert!(metadata.cl_split_dims.is_empty());
        assert!(metadata.pe_sfp_split_dims.is_empty());
        assert!(!metadata.discard_above_lx_schedule);
        assert_eq!(metadata.op_func_backup, OpFunc::None);
        assert!(metadata.transformation_config.enable_moving_data_transfer);
        assert!(metadata.lds_idx_after_ddc.is_empty());
        assert!(metadata.interm_lds_idx_to_ext_lds.is_empty());
    }

    /// `ddc/ddc_metadata.h:130-136` — the entry OWNS both nodes, so a write through the entry reaches
    /// the node the entry holds and no second handle exists to disagree with it. The two boxes are the
    /// `unique_ptr`s and [`ExternalTransfer::new`] is the authority's two-pointer constructor
    /// (`:133-135`).
    ///
    /// ⛔ THIS TEST IS THE ONLY WRITER THE FIELD HAS: nothing in the authority constructs an
    /// `ExternalTransfer` or pushes onto `externalTransfers_` (`:137`), so the field is carried and
    /// its shape checked, never exercised by a real caller — the unit report says so.
    #[test]
    fn an_external_transfer_owns_the_node_pair_it_is_built_from() {
        let transfer = TransferNode {
            replication_factor: 4,
            ..TransferNode::default()
        };
        let allocate = AllocateNode {
            component: SenComponent::Lx,
            ..AllocateNode::default()
        };

        let mut entry = ExternalTransfer::new(Box::new(transfer), Box::new(allocate));
        assert_eq!(entry.transfer.replication_factor, 4);
        assert_eq!(entry.allocate.component, SenComponent::Lx);
        assert_eq!(entry.allocate.num_buffers, NumBuffers(1));

        entry.allocate.num_buffers = NumBuffers::STREAMING;
        let mut metadata = Metadata::default();
        assert!(metadata.external_transfers.is_empty());
        metadata.external_transfers.push(entry);

        assert_eq!(metadata.external_transfers.len(), 1);
        assert_eq!(
            metadata.external_transfers[0].transfer.replication_factor,
            4
        );
        assert_eq!(
            metadata.external_transfers[0].allocate.num_buffers,
            NumBuffers::STREAMING
        );
    }
}

// crustify:todo: e026_Metadata

// crustify:todo: e026_Metadata.TransferNodesInterSliceTranspose_

// crustify:todo: e026_Metadata.belowLxScheduleInsertBlock

// crustify:todo: e026_Metadata.compAndAllocNode

// crustify:todo: e026_Metadata.consIdAndAllocNode

// crustify:todo: e026_Metadata.consumers_

// crustify:todo: e026_Metadata.dataConnects_

// crustify:todo: e026_Metadata.datatransfers_

// crustify:todo: e026_Metadata.dimToCoreChunkLoops_

// crustify:todo: e026_Metadata.externalNodes_

// crustify:todo: e026_Metadata.implicitSyncs_

// crustify:todo: e026_Metadata.inOutRegAllocs_

// crustify:todo: e026_Metadata.internalRegAlloc_

// crustify:todo: e026_Metadata.ldsIdxAndAllocNode

// crustify:todo: e026_Metadata.newAllocations_

// crustify:todo: e026_Metadata.nodeCloningMap_

// crustify:todo: e026_Metadata.opaqueOps_

// crustify:todo: e026_Metadata.prefilledExternalTransferToDataConnectToFill_

// crustify:todo: e026_Metadata.producers_

// crustify:todo: e026_Metadata.shadowAllocations_
