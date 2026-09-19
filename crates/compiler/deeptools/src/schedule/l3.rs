//! Re-ported from the C++ authority. See crustify-scheduler/AGENT-BRIEF.md.
//!
//! Stage 2a of the scheduler — `L3DlOpsScheduler.run(sdsc)`
//! (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.h`). Every bare `:NNN` citation below is a line of that
//! header; a `.cpp:NNN` one is `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp`.

use crate::schedule::ddc::{ExPhase, Verbosity};
use crate::schedule::dims::{DimDensity, PaddingFormType, PrimaryDimTypes};
use crate::schedule::dsc::DesignSpaceConfig;
use crate::schedule::dsc2::{
    AllocateNode, DataStageId, GroupId, IndirectAllocType, LdsIdx, SyncNode, TransferNode,
};
use crate::schedule::metadata::{ConstraintValue, ForcedNumElements};
use std::collections::{BTreeMap, BTreeSet};
use sys_arch_spec::arch_enums::{OpFunc, SenComponent};
use sys_arch_spec::{CoreId, CoreletId};

/// Replaces: e012_CrossCoreReductionGroup.GroupType
///
/// The cores of one cross-core reduction group in slice order — IBM's `std::vector<int>` (`:26`),
/// the declared type of both [`CrossCoreReductionGroup::core_ids`] and what `getCores` returns.
pub type GroupType = Vec<SliceCore>;

/// Which core covers one slice of a group's reduced dims — or that no core does.
///
/// ⛔ IBM'S ELEMENT IS AN `int` WHOSE `-1` IS AN IN-BAND FILLER, NOT A CORE. `addCore` resizes with
/// `-1` (`:30`), so every slice below the one it was handed and above the ones already placed reads
/// back `-1`, and both end accessors return that `-1` to their callers as if it were a core id
/// (`:37`, `:47`). The enum keeps the filler and a core distinguishable where IBM's `int` cannot —
/// and unlike the DSC dims' `-1`, nothing in this class computes with it, so it is an absence
/// rather than a value.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SliceCore {
    /// The `resize` filler (`:30`): a slice of the reduced dims that no `addCore` named.
    #[default]
    Unfilled,
    /// The core `addCore` placed at this slice (`:31`).
    Core(CoreId),
}

/// Which slice of a group's REDUCED dims a core covers — `reduceSlice`, the mixed-radix index
/// `getCrossCoreReductionGroupInfo` packs out of the reduced half of a core's work slice
/// (`.cpp:2758-2764`), and the position `addCore` writes at.
///
/// ⛔ NOT [`WkSliceIdx`](crate::schedule::dsc2::WkSliceIdx), which is one folded dim's own
/// coordinate; this is the mixed-radix product over every reduced dim of the group.
/// ⭐ UNSIGNED WHERE IBM IS `int`, WHICH MAKES BOTH OF ITS THROWS UNSPELLABLE: `addCore(c, -1)`
/// resizes to zero and then reaches `.at(SIZE_MAX)` (`std::out_of_range`), and any slice below `-1`
/// resizes to a `size_t` near its maximum (`std::length_error`) — `:30-31`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ReduceSliceIdx(pub usize);

/// Which of the two corelets a group's start or end core is asked for — the `0` and the `1` that
/// `getStartCoreAtCorelet` and `getEndCoreAtCorelet` test for (`:36-41`, `:45-50`).
///
/// ⛔ THIS ENUM IS `DT_ERROR("Unknown corelet id.")` (`:41`, `:50`) MADE UNSPELLABLE, AND THE
/// AUTHORITY CAN REACH THAT ERROR: `getLdsTransferCoreIds` loops `coreletId` up to
/// `numCoreletsUsed_DSC2_` (`.cpp:2781-2783`), so a DSC on three corelets throws there.
/// [`Self::from_corelet_id`] is that one arm, and it is the only way into this type.
/// ⛔ NOT [`CoreletId`], WHICH IS NOT A CLOSED PAIR: it keys every per-corelet tracker and counts up
/// to `numCoreletsUsed_DSC2_`, while these two methods accept nothing but `0` and `1`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ReductionCorelet {
    /// Corelet `0`, which reads the group forwards: start at the lowest slice, end at the highest.
    Corelet0,
    /// Corelet `1`, which reads it backwards (`.cpp:5828`, `constexpr int corelet1Id = 1`).
    Corelet1,
}

impl ReductionCorelet {
    /// Narrow a tracker's corelet key to the pair these accessors accept.
    ///
    /// `None` is `DT_ERROR("Unknown corelet id.")` (`:41`, `:50`) and nothing else: the corelet is
    /// neither `0` nor `1`.
    pub fn from_corelet_id(corelet: CoreletId) -> Option<Self> {
        match corelet {
            CoreletId(0) => Some(Self::Corelet0),
            CoreletId(1) => Some(Self::Corelet1),
            _ => None,
        }
    }
}

/// Replaces: e012_CrossCoreReductionGroup
///
/// One gang of cores that reduce into each other (`:24-53`). The cores of a cross-core reduction op
/// partition into groups by the slices of the dims that are NOT reduced; within a group the slices
/// of the reduced dims order the cores, and the two ends of that order are which core a corelet's
/// chain starts and finishes at (`.cpp:2753-2768`).
///
/// ⭐ `Default` IS THE AUTHORITY'S CONSTRUCTION: `getCrossCoreReductionGroupInfo` sizes its vector
/// from the non-reduced slice product and default-constructs every group (`.cpp:2755`), so a group
/// no core's work slice lands on stays empty — which is what `.cpp:5830` checks before reading one.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CrossCoreReductionGroup {
    /// Field: e012_CrossCoreReductionGroup.coreIds
    ///
    /// The group's cores indexed by their slice of the reduced dims (`:27`). Public in the
    /// authority too, and [`Self::add_core`] is its only writer tree-wide.
    pub core_ids: GroupType,
}

impl CrossCoreReductionGroup {
    /// Replaces: e066_addCore
    ///
    /// Place `core` at `slice`, filling any lower slice no core has claimed (`:29-32`).
    ///
    /// ⛔ IBM'S `resize` SHRINKS AS WELL AS GROWS, AND THIS PORT KEEPS THAT. A slice below the
    /// current length drops every core above it — `add_core(c0, 3)` then `add_core(c1, 1)` leaves
    /// two slices, not four, and `c0` is gone. Whether the one caller reaches it is a property of
    /// `coreIdToWkSlice_`, not of this method: that map iterates by core id (`.cpp:2756`) while the
    /// position is the reduced dims' mixed-radix index (`.cpp:2758-2764`), so nothing makes the
    /// positions arrive ascending. Not corrected here, because the correction is a choice about
    /// which core wins and the authority states none.
    /// ⭐ THE `push` IS IBM'S `.at(slice) = coreId` (`:31`): a resize to `slice + 1` makes the
    /// written slot the last one, so resize-to-`slice`-then-push is the same two lines with no
    /// index to be out of bounds.
    ///
    /// Transposing the core and the slice is `E0308` — measured 2026-09-19 by compiling the block
    /// below against the built rlibs, one error, "swap these arguments", rather than inferred from
    /// the annotation, which stable rustdoc does not check. The control beside it compiles the same
    /// two literals untransposed through the same public path, so the failure is attributable to
    /// the order and not to a renamed path or a field that stopped being `pub`.
    /// ```compile_fail,E0308
    /// use deeptools::schedule::l3::{CrossCoreReductionGroup, ReduceSliceIdx};
    /// use sys_arch_spec::CoreId;
    /// let mut group = CrossCoreReductionGroup::default();
    /// group.add_core(ReduceSliceIdx(2), CoreId(4));
    /// ```
    /// ```
    /// use deeptools::schedule::l3::{CrossCoreReductionGroup, ReduceSliceIdx};
    /// use sys_arch_spec::CoreId;
    /// let mut group = CrossCoreReductionGroup::default();
    /// group.add_core(CoreId(4), ReduceSliceIdx(2));
    /// ```
    pub fn add_core(&mut self, core: CoreId, slice: ReduceSliceIdx) {
        self.core_ids.resize(slice.0, SliceCore::Unfilled);
        self.core_ids.push(SliceCore::Core(core));
    }

    /// Replaces: e012_CrossCoreReductionGroup.getCores
    ///
    /// Every slice's core, in slice order (`:33`).
    ///
    /// ⛔ NO CALLER IN THE AUTHORITY — `getCores` is named nowhere outside its own declaration, so
    /// the fresh vector IBM returns is a copy nobody takes. Borrowed here; a caller that needs an
    /// independent one writes `.to_vec()`.
    pub fn cores(&self) -> &[SliceCore] {
        &self.core_ids
    }

    /// Replaces: e067_getStartCoreAtCorelet
    ///
    /// Which core this corelet's reduction chain starts at: the lowest slice for corelet `0`, the
    /// highest for corelet `1` (`:34-42`).
    ///
    /// ⛔ `None` IS `DT_CHECK(!coreIds.empty())` (`:35`) AND NOTHING ELSE. A slice that exists but
    /// no core claimed is `Some(SliceCore::Unfilled)` — the `-1` IBM hands back.
    /// ⛔ AND NO CALLER IN THE AUTHORITY: only the END pair is read (`.cpp:2783`, `.cpp:5831`).
    pub fn start_core_at_corelet(&self, corelet: ReductionCorelet) -> Option<SliceCore> {
        match corelet {
            ReductionCorelet::Corelet0 => self.core_ids.first().copied(),
            ReductionCorelet::Corelet1 => self.core_ids.last().copied(),
        }
    }

    /// Replaces: e068_getEndCoreAtCorelet
    ///
    /// Which core this corelet's reduction chain finishes at — the opposite end from
    /// [`Self::start_core_at_corelet`] (`:43-51`). The transfer core of a cross-core reduction's
    /// output, and the core whose HBM address gets the corelet offset (`.cpp:2783`, `.cpp:5831`).
    ///
    /// ⛔ `None` IS `DT_CHECK(!coreIds.empty())` (`:44`), AS ABOVE.
    /// ⭐ CORELET `0`'S END IS ALWAYS A REAL CORE AND CORELET `1`'S NEED NOT BE: `add_core` writes
    /// the last slot every time, so the back of a non-empty group is the core placed most recently,
    /// while the front is slice `0` and reads `Unfilled` until some core claims it.
    pub fn end_core_at_corelet(&self, corelet: ReductionCorelet) -> Option<SliceCore> {
        match corelet {
            ReductionCorelet::Corelet0 => self.core_ids.last().copied(),
            ReductionCorelet::Corelet1 => self.core_ids.first().copied(),
        }
    }

    /// Replaces: e012_CrossCoreReductionGroup.isEmpty
    ///
    /// No core has been placed in this group (`:52`) — the state `.cpp:5830` refuses before reading
    /// an end core, and the state the two end accessors answer `None` in.
    pub fn is_empty(&self) -> bool {
        self.core_ids.is_empty()
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    /// `.cpp:2767` — the one fill site, fed ascending. Slice `1` is claimed by no core, so it holds
    /// the `-1` filler while the two claimed slices hold theirs.
    #[test]
    fn the_fill_places_each_core_at_its_reduced_slice_and_leaves_the_gap_unfilled() {
        let mut group = CrossCoreReductionGroup::default();
        assert!(group.is_empty(), "a default-constructed group has no cores");
        group.add_core(CoreId(4), ReduceSliceIdx(0));
        group.add_core(CoreId(5), ReduceSliceIdx(2));
        assert_eq!(
            group.cores(),
            [
                SliceCore::Core(CoreId(4)),
                SliceCore::Unfilled,
                SliceCore::Core(CoreId(5)),
            ]
        );
        assert!(!group.is_empty());
        assert_eq!(group.core_ids.len(), 3);
    }

    /// ⛔ THE AUTHORITY'S `resize` TRUNCATES (`:30`), AND THIS PINS THAT AS IBM'S BEHAVIOUR, NOT AS
    /// DESIRABLE. Core 4 was placed at slice 3; placing core 5 at slice 1 shortens the vector to
    /// two and core 4 is gone from the group entirely.
    #[test]
    fn a_descending_slice_truncates_and_drops_the_cores_above_it() {
        let mut group = CrossCoreReductionGroup::default();
        group.add_core(CoreId(4), ReduceSliceIdx(3));
        assert_eq!(group.core_ids.len(), 4);
        group.add_core(CoreId(5), ReduceSliceIdx(1));
        assert_eq!(
            group.cores(),
            [SliceCore::Unfilled, SliceCore::Core(CoreId(5))],
            "IBM's resize shrinks, so slice 3's core does not survive"
        );
        assert!(
            !group.cores().contains(&SliceCore::Core(CoreId(4))),
            "the dropped core is dropped, not moved"
        );
    }

    /// `:36-40` against `:45-49` — the two corelets read the same group from opposite ends, so each
    /// one's start is the other's end.
    #[test]
    fn the_two_corelets_read_the_group_from_opposite_ends() {
        let mut group = CrossCoreReductionGroup::default();
        group.add_core(CoreId(4), ReduceSliceIdx(0));
        group.add_core(CoreId(5), ReduceSliceIdx(1));
        let (first, last) = (
            Some(SliceCore::Core(CoreId(4))),
            Some(SliceCore::Core(CoreId(5))),
        );
        assert_eq!(
            group.start_core_at_corelet(ReductionCorelet::Corelet0),
            first
        );
        assert_eq!(group.end_core_at_corelet(ReductionCorelet::Corelet0), last);
        assert_eq!(
            group.start_core_at_corelet(ReductionCorelet::Corelet1),
            last
        );
        assert_eq!(group.end_core_at_corelet(ReductionCorelet::Corelet1), first);
    }

    /// `:35` and `:44` — `DT_CHECK(!coreIds.empty())`. All four reads of an empty group are the
    /// throw, and `None` spells only that: a group WITH a slice no core claimed answers
    /// `Some(Unfilled)` instead.
    #[test]
    fn an_empty_group_is_the_dt_check_throw_and_an_unclaimed_slice_is_not() {
        let empty = CrossCoreReductionGroup::default();
        for corelet in [ReductionCorelet::Corelet0, ReductionCorelet::Corelet1] {
            assert_eq!(empty.start_core_at_corelet(corelet), None);
            assert_eq!(empty.end_core_at_corelet(corelet), None);
        }
        let mut group = CrossCoreReductionGroup::default();
        group.add_core(CoreId(4), ReduceSliceIdx(1));
        assert_eq!(
            group.end_core_at_corelet(ReductionCorelet::Corelet1),
            Some(SliceCore::Unfilled),
            "corelet 1's end is slice 0, which is the -1 filler here"
        );
        assert_eq!(
            group.end_core_at_corelet(ReductionCorelet::Corelet0),
            Some(SliceCore::Core(CoreId(4))),
            "corelet 0's end is the slot add_core just wrote, so never the filler"
        );
    }

    /// `:41` and `:50` — `DT_ERROR("Unknown corelet id.")`. The loop at `.cpp:2781` would reach it
    /// with corelet 2 on a three-corelet DSC; the narrowing is the only entry to
    /// [`ReductionCorelet`], so that arm cannot be spelled past it.
    #[test]
    fn a_corelet_past_the_second_is_the_unknown_corelet_error() {
        assert_eq!(
            ReductionCorelet::from_corelet_id(CoreletId(0)),
            Some(ReductionCorelet::Corelet0)
        );
        assert_eq!(
            ReductionCorelet::from_corelet_id(CoreletId(1)),
            Some(ReductionCorelet::Corelet1)
        );
        for unknown in [2, 3, u8::MAX] {
            assert_eq!(ReductionCorelet::from_corelet_id(CoreletId(unknown)), None);
        }
    }
}

/// Which design space configuration of the `SuperDsc` one [`Metadata`] describes — the `const int
/// dscIdx` every method that touches [`L3DlOpsScheduler::dsc_metadata`] takes, an index into
/// `mySDsc.dscs_` (`.cpp:5510-5511`, `.cpp:6418-6421`).
///
/// ⛔ MINTED BECAUSE NOTHING IN THE CRATE SPELLS A DSC INDEX — `DscIdx`, `DscIndex` and `dsc_idx`
/// have zero hits under `crates/compiler/deeptools/src`. The name is the authority's own parameter
/// name, and the newtype is what keeps it out of [`LdsIdx`]'s and [`DataStageId`]'s slots: all three
/// are a bare `int` there, and `allocAllMem` holds all three at once (`.cpp:5510`, `:5547`, `:6208`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DscIdx(pub i32);

/// How the caller pins the LX buffering strategy — `enum class LxBufferTypeMode` (`:57-61`), the
/// scheduler's last constructor parameter.
///
/// ⭐ A REQUEST, NOT THE ANSWER: [`BufferType`] is what `setLxBufferType` decides from it, and `Auto`
/// is the only mode whose answer depends on the program (`.cpp:6427-6448`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LxBufferTypeMode {
    /// Decide from the heuristics (`:58`) — the constructor's default (`:66`).
    #[default]
    Auto,
    /// Force [`BufferType::Double`] (`:59`), whatever the heuristics say (`.cpp:6431-6433`).
    ForceDouble,
    /// Force [`BufferType::SpatialDouble`] (`:60`), tested first of the three (`.cpp:6427-6430`).
    ForceSpatialDouble,
}

/// Which LX buffering the scheduler settled on — `enum BufferType` (`:105-109`).
///
/// ⛔ `BUFFER_TYPE_COUNT` (`:108`) IS NOT A VARIANT HERE: its only occurrence tree-wide is its own
/// declaration, so it is the C++ enum-size idiom with nothing to port, not a third buffering.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BufferType {
    /// Two buffers in time (`:106`) — the member initialiser (`:221`), and what a core generation at
    /// or below `RCUDD1A_ISA` always gets (`.cpp:6435-6440`).
    #[default]
    Double,
    /// Two buffers in space (`:107`) — what adds the super-chunk data stage (`.cpp:2826`) and what
    /// `Auto` picks at two requests or fewer (`.cpp:6447`).
    SpatialDouble,
}

/// How one dimension of one tensor is scheduled — `enum ScheduleDimTypes : unsigned` (`:88-95`), the
/// key half of [`ScheduleDimMap`].
///
/// ⭐ THE DERIVED `Ord` IS THE AUTHORITY'S ITERATION ORDER: [`ScheduleDimMap`] is a `std::map` keyed
/// by this enum (`:97-98`), which walks it in declaration order — and where the order matters
/// `buildLoopOrder` names the types it wants explicitly instead (`.cpp:4398-4400`).
/// ⛔ `ScheduleDimTypesCount` (`:94`) IS NOT A VARIANT: its only occurrence tree-wide is its own
/// declaration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ScheduleDimTypes {
    /// `:89`, the explicit `= 0`.
    Elementwise,
    /// `:90`.
    Broadcast,
    /// `:91`. Processed before the rest for a HBM-pinned output (`.cpp:4398-4403`).
    Reduction,
    /// `:92`. Dimension I or J used in a strided-window or padded fashion (`.cpp:4231`).
    WindowPadded,
    /// `:93`. A dimension reused across more than one `primaryDsInfo_` (`.cpp:4233`).
    Reuse,
}

/// Which dims of one tensor fall in each schedule type — `ScheduleDimMapType` (`:97-98`).
pub type ScheduleDimMap = BTreeMap<ScheduleDimTypes, Vec<PrimaryDimTypes>>;

/// Field: e029_L3DlOpsScheduler.ScheduleDimTableType
///
/// One [`ScheduleDimMap`] per analysed tensor — `ScheduleDimTableType` (`:99`).
/// `buildScheduleDimensionsTable` returns it and `buildLoopOrder` consumes it (`.cpp:4236-4239`,
/// `.cpp:4378-4381`).
///
/// ⛔ THE `int` KEY IS AN `ldsIdx`, NOT A DSC AND NOT A DATA STAGE: every read indexes it with
/// `lds.ldsIdx_` (`.cpp:4401`, `:4434`, `:4444`), and index tensors are left out of the table
/// altogether (`.cpp:4244-4247`) — so a key it does not hold means "not analysed", which `.at`
/// throws on.
pub type ScheduleDimTable = BTreeMap<LdsIdx, ScheduleDimMap>;

/// Field: e029_L3DlOpsScheduler.TransferAccessPatternType
///
/// `typedef std::pair<PadType, PadType>` (`:137`) — the source pad and the destination pad of one
/// dim of one transfer. Character for character the DDC's own typedef
/// (`ddc/ddc_metadata.h:83-84`), so this is that port re-exported rather than a second pair.
pub use crate::schedule::metadata::TransferAccessPattern;

/// Field: e029_L3DlOpsScheduler.TransferAccessPatternPerDimType
///
/// `typedef std::map<PrimaryDimTypes, TransferAccessPatternType>` (`:138-139`), identical to the
/// DDC's (`ddc/ddc_metadata.h:85-86`) and re-exported for the same reason.
pub use crate::schedule::metadata::TransferAccessPatternPerDim;

/// The bounds one dimension set of one data stage must satisfy — `Metadata::Datastage::Constraints`
/// (`:113-126`).
///
/// ⛔ INERT IN THIS CLASS, AND CARRIED ANYWAY BECAUSE THE FIELDS ARE DECLARED: `constraints_`,
/// `mustBeMultiple_`, `min_`, `max_` and `values_` have ZERO occurrences in
/// `L3DlOpsScheduler.cpp`. The whole [`Datastage`] sub-tree came across with the `allocAllMem` copy
/// the authority flags at `.cpp:5505-5507`, and the one line that would have filled it is COMMENTED
/// OUT — `// metadata.datastages_[id] = metadata.Datastage version of newDs;` (`.cpp:7733`). Reading
/// this type's state back as scheduler output would be reading a default.
/// ⛔ AND IT IS NOT THE DDC'S [`Constraints`](crate::schedule::metadata::Constraints): that one also
/// carries `loopDimKind_` and `cannotBeSymbolic_` (`ddc/ddc_metadata.h:36`, `:39`), which none of the
/// three updaters here touches, and it has a `dump` (`:49-71`) this one does not declare.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Constraints {
    /// Field: e029_L3DlOpsScheduler.mustBeMultiple_
    ///
    /// `:114`. The size must be a multiple of the reference stage's size, or of [`Self::min`] under
    /// the absolute key.
    pub must_be_multiple: bool,

    /// `:115`, the FIRST name of `std::optional<float> min_, max_;` — the smallest value allowed,
    /// absent for unbounded below. ⛔ THE SCHEDULER'S FIELD SCAN CAUGHT ONLY THE SECOND NAME, so
    /// there is no `.min_` anchor to fill; the field is carried here and named by its line.
    pub min: Option<ConstraintValue>,

    /// Field: e029_L3DlOpsScheduler.max_
    ///
    /// `:115`, the second name of that declaration — the largest value allowed, absent for unbounded
    /// above.
    pub max: Option<ConstraintValue>,

    /// Field: e029_L3DlOpsScheduler.values_
    ///
    /// `:116`. The exact values allowed, absent when nothing constrains them.
    ///
    /// ⛔ PRESENT AND EMPTY IS NOT ABSENT: [`Self::update_values`] INTERSECTS, so a set that has
    /// emptied admits no size at all.
    pub values: Option<BTreeSet<ConstraintValue>>,
}

impl Constraints {
    /// `:117-119`. Raises the floor — `std::max`, so the tighter of the two bounds wins.
    pub fn update_min(&mut self, new_val: ConstraintValue) {
        self.min = Some(match self.min {
            Some(min) => min.max(new_val),
            None => new_val,
        });
    }

    /// `:120-122`. Lowers the ceiling — `std::min`, tighter again.
    pub fn update_max(&mut self, new_val: ConstraintValue) {
        self.max = Some(match self.max {
            Some(max) => max.min(new_val),
            None => new_val,
        });
    }

    /// `:123-125`. Intersects with what is already there (`set_intersect`, `util/utils.h:112-116`)
    /// and takes the incoming set whole when nothing is.
    pub fn update_values(&mut self, new_vals: BTreeSet<ConstraintValue>) {
        self.values = Some(match &self.values {
            Some(values) => values.intersection(&new_vals).copied().collect(),
            None => new_vals,
        });
    }
}

/// What the data-stage exploration would know about one data stage — `Metadata::Datastage`
/// (`:112-134`). ⛔ Inert in this class; see [`Constraints`] for the census and the commented-out
/// writer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Datastage {
    /// Field: e029_L3DlOpsScheduler.constraints_
    ///
    /// `:127-130`. The constraints on each dimension set, under the reference stage they are relative
    /// to — `None` is the authority's `-1` key, its own comment's "absolute constraints" (`:127`).
    pub constraints:
        BTreeMap<Option<DataStageId>, BTreeMap<BTreeSet<PrimaryDimTypes>, Constraints>>,

    /// `:131`, `= true`. Minimise the stage's size; `false` means maximise, in the authority's own
    /// trailing comment. ⛔ THE FIELD SCAN PRODUCED NO `.strategyMinimize_` ANCHOR, so it is carried
    /// here and named by its line.
    pub strategy_minimize: bool,

    /// Field: e029_L3DlOpsScheduler.relevantDimsAndNumerator_
    ///
    /// `:132`. Each relevant dim's numerator data stage.
    pub relevant_dims_and_numerator: BTreeMap<PrimaryDimTypes, DataStageId>,

    /// Field: e029_L3DlOpsScheduler.nearestNumeratorIdx_
    ///
    /// `:133`, `int = -1`. The nearest enclosing numerator stage; `None` is that `-1`.
    pub nearest_numerator_idx: Option<DataStageId>,
}

impl Default for Datastage {
    /// The member initialisers (`:131`, `:133`): minimise, and no nearest numerator.
    fn default() -> Self {
        Self {
            constraints: BTreeMap::new(),
            strategy_minimize: true,
            relevant_dims_and_numerator: BTreeMap::new(),
            nearest_numerator_idx: None,
        }
    }
}

/// What the scheduler would record about one transfer node — `Metadata::DataTransfer` (`:141-158`).
///
/// ⛔ THREE OF ITS FIVE MEMBER FUNCTIONS ARE DECLARED WITH NO DEFINITION ANYWHERE, so there is
/// nothing to port and calling one would not even link: `getAccessPattern` (`:149`),
/// `getAccessPatternAsStr` (`:150`) and `dump` (`:154`) have no body in `L3DlOpsScheduler.cpp` and
/// none tree-wide — only the DDC's same-named members are defined
/// (`ddc/ddl/ddl_conversion.cpp:3629`). Read the patterns through
/// [`Self::access_pattern_list_mut`], which is the authority's only defined reader (`:151-153`).
/// ⛔ AND IT IS NOT THE DDC'S
/// [`DataTransfer`](crate::schedule::metadata::DataTransfer): this declares ONE
/// `apply_row_offset_` where that has a src and a dst, and none of its `replicated_`, `offset_src_`
/// or `offset_dest_` (`ddc/ddc_metadata.h:88-118`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DataTransfer {
    /// Field: e029_L3DlOpsScheduler.apply_row_offset_
    ///
    /// `:143`. Its ONE reader adds the row's element offset into the transfer's constant element
    /// offsets when it is set (`.cpp:6329-6338`); nothing in this class ever sets it.
    pub apply_row_offset: bool,

    /// Field: e029_L3DlOpsScheduler.force_num_elements_
    ///
    /// `:144`, `int = -1`. `None` is that `-1`: no count is forced. Zero occurrences in
    /// `L3DlOpsScheduler.cpp`.
    pub force_num_elements: Option<ForcedNumElements>,

    /// Field: e029_L3DlOpsScheduler.accessPatternPerDim_
    ///
    /// `:157`, and PRIVATE as the authority's `private:` at `:156` makes it — reachable only through
    /// [`Self::set_access_pattern`] and [`Self::access_pattern_list_mut`].
    access_pattern_per_dim: TransferAccessPatternPerDim,
}

impl DataTransfer {
    /// `:145-148`. Records one dim's pattern, replacing whatever was there.
    pub fn set_access_pattern(
        &mut self,
        dim_val: PrimaryDimTypes,
        access_pattern: TransferAccessPattern,
    ) {
        self.access_pattern_per_dim.insert(dim_val, access_pattern);
    }

    /// `:151-153`, `getMutableAccessPatternList`. The whole map, to read or to edit in place — the
    /// authority declares no `const` counterpart.
    pub fn access_pattern_list_mut(&mut self) -> &mut TransferAccessPatternPerDim {
        &mut self.access_pattern_per_dim
    }
}

/// One transfer node and its allocate node, owned — `Metadata::ExternalTransfer` (`:169-175`).
///
/// ⭐ ADOPTION, NOT SHARING, WHICH IS WHY THIS NODE-HOLDING MEMBER IS PORTABLE AND EVERY
/// POINTER-KEYED ONE IS NOT: the constructor takes two raw pointers and wraps each in a
/// `unique_ptr` (`:172-174`), so the vector at `:176` owns both nodes outright and no identity
/// beyond ownership is needed.
/// ⛔ A SEPARATE TYPE FROM THE DDC'S [`ExternalTransfer`](crate::schedule::metadata::ExternalTransfer)
/// EVEN THOUGH THE TWO DECLARATIONS AGREE (`ddc/ddc_metadata.h:130-137`): they are members of
/// different classes, and the L3 vector is never touched by `L3DlOpsScheduler.cpp` while the DDC's
/// is.
#[derive(Debug)]
pub struct ExternalTransfer {
    /// Field: e029_L3DlOpsScheduler.transfer_
    ///
    /// `:170`. The adopted transfer node.
    pub transfer: Box<TransferNode>,

    /// Field: e029_L3DlOpsScheduler.allocate_
    ///
    /// `:171`. The adopted allocate node.
    pub allocate: Box<AllocateNode>,
}

impl ExternalTransfer {
    /// `:172-174`. Takes ownership of both nodes.
    pub fn new(transfer: Box<TransferNode>, allocate: Box<AllocateNode>) -> Self {
        Self { transfer, allocate }
    }
}

/// What the scheduler would record about one opaque compute op — `Metadata::OpaqueOp` (`:185-190`).
///
/// ⛔ TWO OF ITS FOUR FIELDS ARE `dsc2::AllocateNode*` AND STAY OPEN BELOW: `inOutRegAllocs_` (`:186`)
/// and `internalRegAlloc_` (`:188`).
/// ⛔ THE TWO CARRIED HERE ARE READ ONLY ON A PATH THE AUTHORITY ITSELF REFUSES: `allocAllMem`'s
/// `compAndAllocNode` loop opens with `DT_ERROR("No support")` and reads `max_unroll_` and
/// `internalRegs_.size()` after it (`.cpp:5566-5584`) — and since nothing in this class ever writes
/// `opaqueOps_`, the `.at` at `.cpp:5571` would throw before either read anyway.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpaqueOp {
    /// Field: e029_L3DlOpsScheduler.internalRegs_
    ///
    /// `:187`. The op's internal register names; `allocAllMem` sizes the allocation by their COUNT,
    /// not by their identity (`.cpp:5584`).
    pub internal_regs: Vec<String>,

    /// Field: e029_L3DlOpsScheduler.max_unroll_
    ///
    /// `:189`, `= 1`. More sticks than this — or a stick count that is zero or not a power of two —
    /// fails the allocation (`.cpp:5578-5581`).
    pub max_unroll: i32,
}

impl Default for OpaqueOp {
    /// The member initialisers (`:187`, `:189`): no registers, and an unroll of one.
    fn default() -> Self {
        Self {
            internal_regs: Vec::new(),
            max_unroll: 1,
        }
    }
}

/// The side table the L3 scheduler carries per DSC — `L3DlOpsScheduler::Metadata` (`:111-198`),
/// minted for every DSC by `prepDsc` (`.cpp:6418-6421`).
///
/// ⛔ FIVE OF ITS TEN OWN DECLARED FIELDS ARE LEFT OUT, so the `.datatransfers_`,
/// `.newAllocations_`, `.opaqueOps_`, `.externalNodes_` and `.dataConnects_` anchors below stay open.
/// Every one of them is keyed by, or holds, a `dsc2` SCHEDULE-NODE POINTER — the blocker
/// `schedule/metadata.rs` reports for the DDC's copy of this same struct, and the crate has no arena,
/// no node id and no `Rc` under `src/schedule/` to answer it with:
///  * `datatransfers_` (`:159`) keys [`DataTransfer`] by `const dsc2::TransferNode*`, and `opaqueOps_`
///    (`:191`) keys [`OpaqueOp`] by `dsc2::ComputeNode*` — both VALUE types are ported above;
///  * `newAllocations_` (`:167`) holds the nested `Allocation` (`:161-166`), whose three maps are all
///    `dsc2::AllocateNode*`, so `.ldsIdxAndAllocNode`, `.consIdAndAllocNode` and `.compAndAllocNode`
///    stay open with it;
///  * `externalNodes_` (`:177`) is a `std::set<const dsc2::ScheduleNode*>`;
///  * `dataConnects_` (`:183`) is keyed by NAME, but its `DataConnect` (`:179-182`) is two
///    `unordered_set<const dsc2::LoopNode*>`, so `.producers_` and `.consumers_` stay open too.
///
/// ⛔ THIS IS NOT [`crate::schedule::metadata::Metadata`], WHICH IS THE DDC'S — IT IS A SMALLER COPY,
/// AND ONE DIVERGENCE DECIDES HOW TWO OF THESE FIELDS ARE SPELLED: there `core_dstgid` and
/// `chunk_dstgid` are `const int` = 0 and 1, hence associated consts on that type; here they are
/// MUTABLE `int = -1` (`:193-194`) that `prepDsc` writes per DSC (`.cpp:6420-6421`), which is why
/// they are fields and why `clear()` gets away with `*this = {}` (`:197`) where the DDC's has to
/// destroy in place.
///
/// ⛔ AND ALMOST ALL OF IT IS INERT — DO NOT READ THIS TABLE AS STAGE-2a OUTPUT. Of the ten fields
/// the only ones `L3DlOpsScheduler.cpp` ever WRITES are `newAllocations_` (`.cpp:583-587`) and those
/// two ids. `datatransfers_` is `DT_CHECK`ed EMPTY (`.cpp:5757-5758`, "Expect empty datatransfers_ in
/// metadata for now."), `datastages_`, `externalTransfers_` and `dataConnects_` have no live
/// occurrence at all, and [`Self::row_split_dim`] and `externalNodes_` are read-only — so both always
/// read back their initialiser.
#[derive(Debug)]
pub struct Metadata {
    /// Field: e029_L3DlOpsScheduler.datastages_
    ///
    /// `:135`. The internal data stages by id — `None` in [`Datastage::constraints`] is the `-1` key,
    /// but this map's own `int` key is a real stage id. No live writer; see [`Constraints`].
    pub datastages: BTreeMap<DataStageId, Datastage>,

    /// Field: e029_L3DlOpsScheduler.externalTransfers_
    ///
    /// `:176`. The owned external transfer nodes. Zero occurrences in `L3DlOpsScheduler.cpp`.
    pub external_transfers: Vec<ExternalTransfer>,

    /// Field: e029_L3DlOpsScheduler.core_dstgid
    ///
    /// `:193`, `int = -1`; `None` is that `-1`. `prepDsc` sets it to
    /// [`L3DlOpsScheduler::DATA_STAGE_CORE_IDX`] for every DSC (`.cpp:6420`), and its one live reader
    /// keys `dataStageParam_` with it (`.cpp:6208`) — which the `-1` would throw on, so the `Option`
    /// is the throw.
    pub core_dstgid: Option<DataStageId>,

    /// Field: e029_L3DlOpsScheduler.chunk_dstgid
    ///
    /// `:194`. As above, set to [`L3DlOpsScheduler::DATA_STAGE_CHUNK_IDX`] (`.cpp:6421`). Its only
    /// other occurrence is commented out (`.cpp:5772`).
    pub chunk_dstgid: Option<DataStageId>,

    /// Field: e029_L3DlOpsScheduler.rowSplitDim
    ///
    /// `:195`, `= PrimaryDimTypesCount`, which [`PrimaryDimTypes::Undefined`] carries. READ-ONLY in
    /// this class — `fillLoopOffsetsAndAddresses` indexes the constant element offsets with it
    /// (`.cpp:6335-6338`) — so it always reads back unset.
    pub row_split_dim: PrimaryDimTypes,
}

impl Default for Metadata {
    /// The member initialisers (`:193-195`): no stages, no external transfers, neither data-stage id
    /// assigned and the row-split dim unset. ⛔ NOT DERIVED: [`PrimaryDimTypes`] has no `Default`
    /// standing for `PrimaryDimTypesCount`.
    fn default() -> Self {
        Self {
            datastages: BTreeMap::new(),
            external_transfers: Vec::new(),
            core_dstgid: None,
            chunk_dstgid: None,
            row_split_dim: PrimaryDimTypes::Undefined,
        }
    }
}

impl Metadata {
    /// `:197`, `void clear() { *this = {}; }`. Back to a fresh table — assignment from the default,
    /// exactly as the authority spells it, because none of this type's members is `const`.
    pub fn clear(&mut self) {
        *self = Self::default();
    }
}

/// The share of the data-ring traffic one request pattern actually uses — the `double`
/// `getBurstEfficiency` returns (`.cpp:1609`, `:1625`), and one entry of the table at `:213`.
///
/// ⚠️ AN APPROXIMATION, NOT A MEASUREMENT: IBM's own note calls the values "approximations based on
/// heuristics" (`dcg/dcg_fe/scheduler/BurstEfficiency.def:7-16`).
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct BurstEfficiency(pub f64);

/// How many sticks one L3 request bursts — `getBurstEfficiency`'s `burstSize` (`.cpp:1609`, and
/// `burstNum` in the declaration at `:344`), ONE-BASED.
///
/// ⭐ THE RANGE IS THE TABLE'S OWN, WHICH IS WHAT MAKES [`L3DlOpsScheduler::burst_efficiency`] TOTAL:
/// `DT_CHECK_MSG(burstSize >= 1 && burstSize <= dscGlobal.sysDef.l3BurstSize)` (`.cpp:1611-1612`) with
/// `l3BurstSize = 32` (`sys-arch-spec/sysdef.cpp:215`), re-checked against the table's row count at
/// `.cpp:1616-1617`. [`Self::new`] is that check and the only way into the type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BurstSize(u32);

impl BurstSize {
    /// The number of rows the table has, and so the largest burst size — `l3BurstSize`
    /// (`sys-arch-spec/sysdef.cpp:215`), the `DT_CHECK` at `.cpp:1616-1617` read as a constant.
    pub const MAX: u32 = 32;

    /// The burst size, or `None` for the `DT_CHECK` at `.cpp:1611-1612`.
    pub fn new(burst_size: u32) -> Option<Self> {
        if burst_size >= 1 && burst_size <= Self::MAX {
            Some(Self(burst_size))
        } else {
            None
        }
    }

    /// The burst size itself, as the authority's `DT_CHECK` compares it.
    pub const fn get(self) -> u32 {
        self.0
    }

    /// The zero-based row — the `burstSize - 1` at `.cpp:1623`.
    const fn row(self) -> usize {
        (self.0 - 1) as usize
    }
}

/// To how many cores one L3 request multicasts — `multicastDegree` (`.cpp:1610`), ONE-BASED.
///
/// ⚠️ THE CEILING HERE IS THE TABLE'S 32, NOT THE SYSDEF'S `numCores`: the authority checks
/// `multicastDegree <= dscGlobal.sysDef.numCores` (`.cpp:1613-1615`) and separately requires every row
/// to be `maxNumCores = 32` wide (`.cpp:1618-1621`). A target with fewer cores makes the authority
/// throw where this type still admits the degree, because `numCores` lives on the unported
/// `DesignSpaceConfigGlobal` — the narrower bound cannot be spelled until that type is scheduled.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MulticastDegree(u32);

impl MulticastDegree {
    /// The number of columns every table row has — the `const int maxNumCores = 32` at `.cpp:1618`.
    pub const MAX: u32 = 32;

    /// The multicast degree, or `None` for the table-width half of `.cpp:1613-1621`.
    pub fn new(multicast_degree: u32) -> Option<Self> {
        if multicast_degree >= 1 && multicast_degree <= Self::MAX {
            Some(Self(multicast_degree))
        } else {
            None
        }
    }

    /// The degree itself, as the authority's `DT_CHECK` compares it.
    pub const fn get(self) -> u32 {
        self.0
    }

    /// The zero-based column — the `multicastDegree - 1` at `.cpp:1624`.
    const fn column(self) -> usize {
        (self.0 - 1) as usize
    }
}

/// Stage 2a of the scheduler — `class L3DlOpsScheduler` (`:55-539`), the pass that turns a
/// `SuperDsc`'s design space configurations into a `dsc2` schedule tree and commits its LX
/// allocations (`dbo/src/Utils/sdsc_bundle/SchedulerStages.cpp:29-42`).
///
/// ⛔ THIS CARRIES FOURTEEN OF THE CLASS'S SIXTEEN DECLARED MEMBERS, so the `e029_L3DlOpsScheduler`
/// anchor below stays open. The two left out each need a type this campaign never scheduled as a
/// unit in `crustify-scheduler/UNITS.tsv` — the same blocker `schedule/ddc.rs` reports for the same
/// two members of `class Ddc`:
///  * `memTrackers` (`:79`) — `MemTrackBundle*`, and it is the ONLY way `allocAllMem` reaches a
///    tracker (`.cpp:5540`);
///  * `dscGlobal` (`:200`) — `const DesignSpaceConfigGlobal&`. The field scan produced no anchor for
///    it at all, and it is what supplies `sysDef.bytesPerStick`, `sysDef.maxGroupID`,
///    `sysDef.numCores` and `sysDef.l3BurstSize` (`.cpp:5556`, `:4710`, `:1613`, `:1611`).
///
/// ⛔ SO [`Self::new`] TAKES THREE OF THE CONSTRUCTOR'S FIVE PARAMETERS (`:63-73`). Nothing is lost
/// about the phase list — both real construction sites pass exactly one phase, `{executionStep}` and
/// `{0}` (`dbo/src/Utils/sdsc_bundle/SchedulerStages.cpp:31`,
/// `dcg/dcg_fe/scheduler/L3DlOpsScheduler_standalone.cpp:192`) — but the two references are lost, and
/// no method that reads them can be ported until their types are.
///
/// ⛔ ITS 130 METHOD BODIES ARE NOT THIS UNIT: `UNITS.tsv` splits them into `e029g1`..`e029g7`. What is
/// ported here is the DECLARATION — the fields, the members defined inline in the header, and
/// `getBurstEfficiency` (`:344-345`), whose name matches no group's prefix and which is the only
/// reader of [`Self::burst_efficiency`]'s table.
#[derive(Debug)]
pub struct L3DlOpsScheduler {
    /// Field: e029_L3DlOpsScheduler.exphases
    ///
    /// `:78`, `const std::vector<int>`. The execution phases every memory-tracker call spans
    /// (`.cpp:5537-5538`, `.cpp:5601`).
    ///
    /// ⭐ PRIVATE BECAUSE NON-EMPTINESS IS THIS TYPE'S INVARIANT, NOT A CHECK EACH READER REPEATS:
    /// `DT_CHECK(!exphases.empty())` (`:72`) holds for the whole lifetime since the authority's member
    /// is `const`, so the guard belongs at construction ([`Self::new`]) and the field must not be
    /// reachable for assignment or for a struct literal. Read it with [`Self::exphases`].
    exphases: Vec<ExPhase>,

    /// Field: e029_L3DlOpsScheduler.dscMetadata
    ///
    /// `:205`. One [`Metadata`] per DSC, minted for every DSC by `prepDsc` (`.cpp:6418-6421`) and
    /// `DT_CHECK`ed present before use (`.cpp:580`).
    pub dsc_metadata: BTreeMap<DscIdx, Metadata>,

    /// Field: e029_L3DlOpsScheduler.verbose
    ///
    /// `:208`, defaulting to `0` (`:65`). Every reader is a `> 0` print gate (`.cpp:5585-5594`).
    pub verbose: Verbosity,

    /// Field: e029_L3DlOpsScheduler.gtrCurrGroupName
    ///
    /// `:214`, `size_t = 0`. The NEXT sharing-group id to hand out, not the last one handed out —
    /// `getSharesAndGroupName` reads it and then increments (`.cpp:4712-4714`).
    ///
    /// ⚠️ ITS CEILING IS UNSPELLABLE HERE: `DT_CHECK_MSG(gtrCurrGroupName <= dscGlobal.sysDef.maxGroupID,
    /// "gtr_->groupName_ exceeds the limit.")` (`.cpp:4710-4711`) reads the unported
    /// `DesignSpaceConfigGlobal`.
    pub gtr_curr_group_name: GroupId,

    /// Field: e029_L3DlOpsScheduler.coresSetToGtrGroupNameMap
    ///
    /// `:219`. Which id each distinct sharing SET of cores was already given, so the same set gets the
    /// same `gtr_->groupName_` twice (`.cpp:4707-4713`).
    pub cores_set_to_gtr_group_name_map: BTreeMap<BTreeSet<CoreId>, GroupId>,

    /// Field: e029_L3DlOpsScheduler.lxBufferTypeMode
    ///
    /// `:220`, `const`. What the caller asked for — [`LxBufferTypeMode::Auto`] unless overridden
    /// (`:66`).
    pub lx_buffer_type_mode: LxBufferTypeMode,

    /// Field: e029_L3DlOpsScheduler.lxBufferType
    ///
    /// `:221`, `= DOUBLE`. What `setLxBufferType` decided from the mode, the core generation and the
    /// request count (`.cpp:6425-6450`), and what nine later sites branch on (`.cpp:450`, `:2794`,
    /// `:2826`, `:3592`, `:3734`, `:4124`, `:4645`, `:4658`, `:6766`).
    pub lx_buffer_type: BufferType,

    /// Field: e029_L3DlOpsScheduler.dataStageIbrIdx
    ///
    /// `:222`, `int = -1`; `None` is that `-1`.
    ///
    /// ⛔ A LAZY-MINT GUARD, NOT AN ABSENCE THE READERS TOLERATE: `if (dataStageIbrIdx == -1)
    /// dataStageIbrIdx = getNewDataStageIndex(..)` (`.cpp:6630-6631`), and afterwards
    /// `dataStageParam_.at(dataStageIbrIdx)` (`.cpp:2860`, `:2903`) would throw on the `-1`. The
    /// `Option` is that throw.
    pub data_stage_ibr_idx: Option<DataStageId>,

    /// Field: e029_L3DlOpsScheduler.dataStageOnePageIdx
    ///
    /// `:223`. The same lazy mint (`.cpp:6680-6681`) — ⛔ and the authority CAN reach it unset:
    /// `.cpp:6634` keys `dataStageParam_.count(dataStageOnePageIdx)` on the IBR path, which
    /// `.cpp:6680` does not precede.
    pub data_stage_one_page_idx: Option<DataStageId>,

    /// Field: e029_L3DlOpsScheduler.dataStageSuperChunkIdx
    ///
    /// `:224`. The same lazy mint (`.cpp:4647-4648`), and the stage the super-chunk loops are named
    /// after (`.cpp:4649-4650`). Only [`BufferType::SpatialDouble`] ever mints it (`.cpp:4645`).
    pub data_stage_super_chunk_idx: Option<DataStageId>,
}

impl L3DlOpsScheduler {
    /// Field: e029_L3DlOpsScheduler.dataStageCoreIdx
    ///
    /// `:210`, `= 0` (`.cpp:275`). The core data stage's id; `prepDsc` copies it into every
    /// [`Metadata::core_dstgid`] (`.cpp:6420`).
    ///
    /// ⭐ AN ASSOCIATED CONST BECAUSE `static const` MAKES IT ONE VALUE FOR EVERY SCHEDULER, and the
    /// mutable [`Metadata::core_dstgid`] is what varies.
    pub const DATA_STAGE_CORE_IDX: DataStageId = DataStageId(0);

    /// Field: e029_L3DlOpsScheduler.dataStageChunkIdx
    ///
    /// `:211`, `= 1` (`.cpp:276`). A loop from [`Self::DATA_STAGE_CORE_IDX`] to this one is the
    /// core-chunk loop.
    pub const DATA_STAGE_CHUNK_IDX: DataStageId = DataStageId(1);

    /// Field: e029_L3DlOpsScheduler.lxBelowBlockNodeName
    ///
    /// `:212`, `= "lx_below_schedule"` (`.cpp:277`). The name `createBlockNode` gives the dummy block
    /// the below-LX schedule hangs under (`.cpp:4203`, `:4656`), the name two `DT_CHECK`s then require
    /// of it (`.cpp:440`, `:500`), and the name the tree walk recognises it by (`.cpp:3473`).
    pub const LX_BELOW_BLOCK_NODE_NAME: &str = "lx_below_schedule";

    /// `:63-73`. The scheduler, or `None` for `DT_CHECK(!exphases.empty())` (`:72`) — the
    /// constructor's only check.
    ///
    /// ⛔ THREE PARAMETERS, NOT FIVE: the `dscGlobal` and `memTrackers` references need types this
    /// campaign has not scheduled. See the type doc.
    ///
    /// Transposing the two remaining scalars is a compile error, which is why neither is a bare `int`:
    /// ```compile_fail
    /// use deeptools::schedule::ddc::{ExPhase, Verbosity};
    /// use deeptools::schedule::l3::{L3DlOpsScheduler, LxBufferTypeMode};
    /// let _ = L3DlOpsScheduler::new(vec![ExPhase(0)], LxBufferTypeMode::Auto, Verbosity(0));
    /// ```
    /// ⛔ AND THAT BLOCK NEEDS THIS CONTROL TO MEAN ANYTHING: `compile_fail` passes on ANY error — a
    /// wrong path or a private item included — and rustdoc checks no error code even when one is
    /// written. The same call with the arguments the right way round compiles, so the block above
    /// fails on the TRANSPOSITION and on nothing else:
    /// ```
    /// use deeptools::schedule::ddc::{ExPhase, Verbosity};
    /// use deeptools::schedule::l3::{L3DlOpsScheduler, LxBufferTypeMode};
    /// let _ = L3DlOpsScheduler::new(vec![ExPhase(0)], Verbosity(0), LxBufferTypeMode::Auto);
    /// ```
    pub fn new(
        phases: Vec<ExPhase>,
        verbose: Verbosity,
        lx_buffer_type_mode: LxBufferTypeMode,
    ) -> Option<Self> {
        if phases.is_empty() {
            return None;
        }
        Some(Self {
            exphases: phases,
            dsc_metadata: BTreeMap::new(),
            verbose,
            gtr_curr_group_name: GroupId(0),
            cores_set_to_gtr_group_name_map: BTreeMap::new(),
            lx_buffer_type_mode,
            lx_buffer_type: BufferType::Double,
            data_stage_ibr_idx: None,
            data_stage_one_page_idx: None,
            data_stage_super_chunk_idx: None,
        })
    }

    /// `:78`. The execution phases, non-empty by construction ([`Self::new`]).
    pub fn exphases(&self) -> &[ExPhase] {
        &self.exphases
    }

    /// `.cpp:1609-1626`, `getBurstEfficiency`. The efficiency of one burst size at one multicast
    /// degree.
    ///
    /// ⭐ TOTAL, AND ALL FOUR OF THE AUTHORITY'S `DT_CHECK`s ARE GONE INTO TYPES: the two argument
    /// checks live in [`BurstSize`] and [`MulticastDegree`], and the two table-shape checks
    /// (`.cpp:1616-1621`) are the fixed array dimensions. What remains is the `- 1` at
    /// `.cpp:1623-1624`.
    ///
    /// Transposing the two is a compile error:
    /// ```compile_fail
    /// use deeptools::schedule::l3::{BurstSize, L3DlOpsScheduler, MulticastDegree};
    /// let (b, m) = (BurstSize::new(1).unwrap(), MulticastDegree::new(1).unwrap());
    /// let _ = L3DlOpsScheduler::burst_efficiency(m, b);
    /// ```
    /// ⛔ WITH THE CONTROL THAT PROVES IT FAILS ON THE TRANSPOSITION AND NOT ON A PATH:
    /// ```
    /// use deeptools::schedule::l3::{BurstSize, L3DlOpsScheduler, MulticastDegree};
    /// let (b, m) = (BurstSize::new(1).unwrap(), MulticastDegree::new(1).unwrap());
    /// let _ = L3DlOpsScheduler::burst_efficiency(b, m);
    /// ```
    pub fn burst_efficiency(
        burst_size: BurstSize,
        multicast_degree: MulticastDegree,
    ) -> BurstEfficiency {
        BurstEfficiency(Self::BURST_EFFICIENCY[burst_size.row()][multicast_degree.column()])
    }

    /// `.cpp:652-663`, `createSyncNode` (`:273-276`). One end of a sync — the units it signals to or
    /// waits on, its name, and two flags whose declared defaults are `false` (`:275-276`), which is
    /// what the authority's two conditional writes leave (`.cpp:657-658`).
    ///
    /// ⭐ BY VALUE AND WITHOUT `self`, where the authority returns `new dsc2::SyncNode` from a `const`
    /// member that reads no field (`.cpp:655`): the tree owns its nodes as
    /// [`ChildNode`](crate::schedule::dsc2::ChildNode)s, so a caller holding `&mut self` inserts it.
    /// ⛔ IT DOES NOT PAIR THE ENDS — neither does the authority; each call site pushes
    /// `otherEndOfTheSignals_` itself (`.cpp:3975-3976`), `e036_SyncNode`'s open anchor.
    pub fn create_sync_node(
        units: BTreeSet<SenComponent>,
        name: String,
        is_receive: bool,
        is_soft: bool,
    ) -> SyncNode {
        let mut node = SyncNode::default();
        node.base_class.name = name;
        node.units = units;
        node.is_receive = is_receive;
        node.is_soft = is_soft;
        node
    }

    /// Replaces: e029g3_L3DlOpsScheduler_coord.isDimensionCoreletSplit
    ///
    /// `:81-82`, defined `.cpp:74-86`. Is one corelet's share of `dim` SMALLER than the whole core's?
    /// ⛔ THE CORE DATA STAGE ANSWERS ALONE when `dataStageParam_[0]` exists: `CoreletD_`/`CoreD_` go
    /// unread (`.cpp:77-85`), and `clId = 0` against `clId = -1` IS the comparison — only the first
    /// takes `coreletSplit_[dim][0]` (`dsc/dims.cpp:631-644`), so a dim absent there reads equal.
    /// ⛔ [`None`] is IBM's `.at()` throw AND the unwritten `numCoreletsUsed_`, indeterminate at
    /// `.cpp:75` (`dsc/designSpaceConfig.h:74`) — neither is `Some(false)`.
    pub fn is_dimension_corelet_split(
        dsc: &DesignSpaceConfig,
        dim: PrimaryDimTypes,
    ) -> Option<bool> {
        // `.cpp:75`: one corelet has nothing to split.
        if dsc.num_corelets_used?.0 <= 1 {
            return Some(false);
        }
        // `.cpp:76-83`.
        if let Some(core_stage) = dsc.data_stage_param.get(&Self::DATA_STAGE_CORE_IDX) {
            let reading = |cl_id| {
                core_stage.ss.primary_dim_to_val_for_component(
                    dim,
                    SenComponent::NoComponent,
                    None,
                    cl_id,
                    &PaddingFormType::default(),
                    DimDensity::FULL,
                    false,
                )
            };
            return Some(reading(Some(CoreletId(0)))? < reading(None)?);
        }
        // `.cpp:84-85`.
        Some(dsc.corelet_d.primary_dim_to_val(dim)? < dsc.core_d.primary_dim_to_val(dim)?)
    }
}

impl L3DlOpsScheduler {
    /// Field: e029_L3DlOpsScheduler.burstEfficiency
    ///
    /// `:213`, defined by `#include "BurstEfficiency.def"` (`.cpp:278-280`). Row = burst size 1..32,
    /// column = multicast degree 1..32; for a fixed burst size a larger multicast degree is worse, and
    /// for a fixed degree a larger burst size is better (`BurstEfficiency.def:9-16`).
    ///
    /// ⭐ THE FIXED SHAPE IS TWO OF `getBurstEfficiency`'S FOUR `DT_CHECK`s MADE COMPILE-TIME: this
    /// array cannot hold the wrong number of burst sizes (`.cpp:1616-1617`) nor a row of the wrong
    /// number of multicast degrees (`.cpp:1618-1621`).
    /// ⛔ PRIVATE AND UNTYPED INSIDE, TYPED AT THE SEAM: reach it only through
    /// [`Self::burst_efficiency`], which is what turns an entry into a [`BurstEfficiency`] and the
    /// one-based arguments into indices.
    /// ⛔ AND IT IS DATA, NOT A FORMULA: the rows happen to be arithmetic today, but IBM calls the
    /// values heuristic approximations, so a future table is free to stop being regular.
    #[rustfmt::skip]
    const BURST_EFFICIENCY: [[f64; MulticastDegree::MAX as usize]; BurstSize::MAX as usize] = [
        [0.1000, 0.0995, 0.0990, 0.0985, 0.0980, 0.0975, 0.0970, 0.0965,
         0.0960, 0.0955, 0.0950, 0.0945, 0.0940, 0.0935, 0.0930, 0.0925,
         0.0920, 0.0915, 0.0910, 0.0905, 0.0900, 0.0895, 0.0890, 0.0885,
         0.0880, 0.0875, 0.0870, 0.0865, 0.0860, 0.0855, 0.0850, 0.0845],
        [0.1250, 0.1245, 0.1240, 0.1235, 0.1230, 0.1225, 0.1220, 0.1215,
         0.1210, 0.1205, 0.1200, 0.1195, 0.1190, 0.1185, 0.1180, 0.1175,
         0.1170, 0.1165, 0.1160, 0.1155, 0.1150, 0.1145, 0.1140, 0.1135,
         0.1130, 0.1125, 0.1120, 0.1115, 0.1110, 0.1105, 0.1100, 0.1095],
        [0.1500, 0.1495, 0.1490, 0.1485, 0.1480, 0.1475, 0.1470, 0.1465,
         0.1460, 0.1455, 0.1450, 0.1445, 0.1440, 0.1435, 0.1430, 0.1425,
         0.1420, 0.1415, 0.1410, 0.1405, 0.1400, 0.1395, 0.1390, 0.1385,
         0.1380, 0.1375, 0.1370, 0.1365, 0.1360, 0.1355, 0.1350, 0.1345],
        [0.1750, 0.1745, 0.1740, 0.1735, 0.1730, 0.1725, 0.1720, 0.1715,
         0.1710, 0.1705, 0.1700, 0.1695, 0.1690, 0.1685, 0.1680, 0.1675,
         0.1670, 0.1665, 0.1660, 0.1655, 0.1650, 0.1645, 0.1640, 0.1635,
         0.1630, 0.1625, 0.1620, 0.1615, 0.1610, 0.1605, 0.1600, 0.1595],
        [0.2000, 0.1995, 0.1990, 0.1985, 0.1980, 0.1975, 0.1970, 0.1965,
         0.1960, 0.1955, 0.1950, 0.1945, 0.1940, 0.1935, 0.1930, 0.1925,
         0.1920, 0.1915, 0.1910, 0.1905, 0.1900, 0.1895, 0.1890, 0.1885,
         0.1880, 0.1875, 0.1870, 0.1865, 0.1860, 0.1855, 0.1850, 0.1845],
        [0.2250, 0.2245, 0.2240, 0.2235, 0.2230, 0.2225, 0.2220, 0.2215,
         0.2210, 0.2205, 0.2200, 0.2195, 0.2190, 0.2185, 0.2180, 0.2175,
         0.2170, 0.2165, 0.2160, 0.2155, 0.2150, 0.2145, 0.2140, 0.2135,
         0.2130, 0.2125, 0.2120, 0.2115, 0.2110, 0.2105, 0.2100, 0.2095],
        [0.2500, 0.2495, 0.2490, 0.2485, 0.2480, 0.2475, 0.2470, 0.2465,
         0.2460, 0.2455, 0.2450, 0.2445, 0.2440, 0.2435, 0.2430, 0.2425,
         0.2420, 0.2415, 0.2410, 0.2405, 0.2400, 0.2395, 0.2390, 0.2385,
         0.2380, 0.2375, 0.2370, 0.2365, 0.2360, 0.2355, 0.2350, 0.2345],
        [0.2750, 0.2745, 0.2740, 0.2735, 0.2730, 0.2725, 0.2720, 0.2715,
         0.2710, 0.2705, 0.2700, 0.2695, 0.2690, 0.2685, 0.2680, 0.2675,
         0.2670, 0.2665, 0.2660, 0.2655, 0.2650, 0.2645, 0.2640, 0.2635,
         0.2630, 0.2625, 0.2620, 0.2615, 0.2610, 0.2605, 0.2600, 0.2595],
        [0.3000, 0.2995, 0.2990, 0.2985, 0.2980, 0.2975, 0.2970, 0.2965,
         0.2960, 0.2955, 0.2950, 0.2945, 0.2940, 0.2935, 0.2930, 0.2925,
         0.2920, 0.2915, 0.2910, 0.2905, 0.2900, 0.2895, 0.2890, 0.2885,
         0.2880, 0.2875, 0.2870, 0.2865, 0.2860, 0.2855, 0.2850, 0.2845],
        [0.3250, 0.3245, 0.3240, 0.3235, 0.3230, 0.3225, 0.3220, 0.3215,
         0.3210, 0.3205, 0.3200, 0.3195, 0.3190, 0.3185, 0.3180, 0.3175,
         0.3170, 0.3165, 0.3160, 0.3155, 0.3150, 0.3145, 0.3140, 0.3135,
         0.3130, 0.3125, 0.3120, 0.3115, 0.3110, 0.3105, 0.3100, 0.3095],
        [0.3500, 0.3495, 0.3490, 0.3485, 0.3480, 0.3475, 0.3470, 0.3465,
         0.3460, 0.3455, 0.3450, 0.3445, 0.3440, 0.3435, 0.3430, 0.3425,
         0.3420, 0.3415, 0.3410, 0.3405, 0.3400, 0.3395, 0.3390, 0.3385,
         0.3380, 0.3375, 0.3370, 0.3365, 0.3360, 0.3355, 0.3350, 0.3345],
        [0.3750, 0.3745, 0.3740, 0.3735, 0.3730, 0.3725, 0.3720, 0.3715,
         0.3710, 0.3705, 0.3700, 0.3695, 0.3690, 0.3685, 0.3680, 0.3675,
         0.3670, 0.3665, 0.3660, 0.3655, 0.3650, 0.3645, 0.3640, 0.3635,
         0.3630, 0.3625, 0.3620, 0.3615, 0.3610, 0.3605, 0.3600, 0.3595],
        [0.4000, 0.3995, 0.3990, 0.3985, 0.3980, 0.3975, 0.3970, 0.3965,
         0.3960, 0.3955, 0.3950, 0.3945, 0.3940, 0.3935, 0.3930, 0.3925,
         0.3920, 0.3915, 0.3910, 0.3905, 0.3900, 0.3895, 0.3890, 0.3885,
         0.3880, 0.3875, 0.3870, 0.3865, 0.3860, 0.3855, 0.3850, 0.3845],
        [0.4250, 0.4245, 0.4240, 0.4235, 0.4230, 0.4225, 0.4220, 0.4215,
         0.4210, 0.4205, 0.4200, 0.4195, 0.4190, 0.4185, 0.4180, 0.4175,
         0.4170, 0.4165, 0.4160, 0.4155, 0.4150, 0.4145, 0.4140, 0.4135,
         0.4130, 0.4125, 0.4120, 0.4115, 0.4110, 0.4105, 0.4100, 0.4095],
        [0.4500, 0.4495, 0.4490, 0.4485, 0.4480, 0.4475, 0.4470, 0.4465,
         0.4460, 0.4455, 0.4450, 0.4445, 0.4440, 0.4435, 0.4430, 0.4425,
         0.4420, 0.4415, 0.4410, 0.4405, 0.4400, 0.4395, 0.4390, 0.4385,
         0.4380, 0.4375, 0.4370, 0.4365, 0.4360, 0.4355, 0.4350, 0.4345],
        [0.4750, 0.4745, 0.4740, 0.4735, 0.4730, 0.4725, 0.4720, 0.4715,
         0.4710, 0.4705, 0.4700, 0.4695, 0.4690, 0.4685, 0.4680, 0.4675,
         0.4670, 0.4665, 0.4660, 0.4655, 0.4650, 0.4645, 0.4640, 0.4635,
         0.4630, 0.4625, 0.4620, 0.4615, 0.4610, 0.4605, 0.4600, 0.4595],
        [0.5000, 0.4995, 0.4990, 0.4985, 0.4980, 0.4975, 0.4970, 0.4965,
         0.4960, 0.4955, 0.4950, 0.4945, 0.4940, 0.4935, 0.4930, 0.4925,
         0.4920, 0.4915, 0.4910, 0.4905, 0.4900, 0.4895, 0.4890, 0.4885,
         0.4880, 0.4875, 0.4870, 0.4865, 0.4860, 0.4855, 0.4850, 0.4845],
        [0.5250, 0.5245, 0.5240, 0.5235, 0.5230, 0.5225, 0.5220, 0.5215,
         0.5210, 0.5205, 0.5200, 0.5195, 0.5190, 0.5185, 0.5180, 0.5175,
         0.5170, 0.5165, 0.5160, 0.5155, 0.5150, 0.5145, 0.5140, 0.5135,
         0.5130, 0.5125, 0.5120, 0.5115, 0.5110, 0.5105, 0.5100, 0.5095],
        [0.5500, 0.5495, 0.5490, 0.5485, 0.5480, 0.5475, 0.5470, 0.5465,
         0.5460, 0.5455, 0.5450, 0.5445, 0.5440, 0.5435, 0.5430, 0.5425,
         0.5420, 0.5415, 0.5410, 0.5405, 0.5400, 0.5395, 0.5390, 0.5385,
         0.5380, 0.5375, 0.5370, 0.5365, 0.5360, 0.5355, 0.5350, 0.5345],
        [0.5750, 0.5745, 0.5740, 0.5735, 0.5730, 0.5725, 0.5720, 0.5715,
         0.5710, 0.5705, 0.5700, 0.5695, 0.5690, 0.5685, 0.5680, 0.5675,
         0.5670, 0.5665, 0.5660, 0.5655, 0.5650, 0.5645, 0.5640, 0.5635,
         0.5630, 0.5625, 0.5620, 0.5615, 0.5610, 0.5605, 0.5600, 0.5595],
        [0.6000, 0.5995, 0.5990, 0.5985, 0.5980, 0.5975, 0.5970, 0.5965,
         0.5960, 0.5955, 0.5950, 0.5945, 0.5940, 0.5935, 0.5930, 0.5925,
         0.5920, 0.5915, 0.5910, 0.5905, 0.5900, 0.5895, 0.5890, 0.5885,
         0.5880, 0.5875, 0.5870, 0.5865, 0.5860, 0.5855, 0.5850, 0.5845],
        [0.6250, 0.6245, 0.6240, 0.6235, 0.6230, 0.6225, 0.6220, 0.6215,
         0.6210, 0.6205, 0.6200, 0.6195, 0.6190, 0.6185, 0.6180, 0.6175,
         0.6170, 0.6165, 0.6160, 0.6155, 0.6150, 0.6145, 0.6140, 0.6135,
         0.6130, 0.6125, 0.6120, 0.6115, 0.6110, 0.6105, 0.6100, 0.6095],
        [0.6500, 0.6495, 0.6490, 0.6485, 0.6480, 0.6475, 0.6470, 0.6465,
         0.6460, 0.6455, 0.6450, 0.6445, 0.6440, 0.6435, 0.6430, 0.6425,
         0.6420, 0.6415, 0.6410, 0.6405, 0.6400, 0.6395, 0.6390, 0.6385,
         0.6380, 0.6375, 0.6370, 0.6365, 0.6360, 0.6355, 0.6350, 0.6345],
        [0.6750, 0.6745, 0.6740, 0.6735, 0.6730, 0.6725, 0.6720, 0.6715,
         0.6710, 0.6705, 0.6700, 0.6695, 0.6690, 0.6685, 0.6680, 0.6675,
         0.6670, 0.6665, 0.6660, 0.6655, 0.6650, 0.6645, 0.6640, 0.6635,
         0.6630, 0.6625, 0.6620, 0.6615, 0.6610, 0.6605, 0.6600, 0.6595],
        [0.7000, 0.6995, 0.6990, 0.6985, 0.6980, 0.6975, 0.6970, 0.6965,
         0.6960, 0.6955, 0.6950, 0.6945, 0.6940, 0.6935, 0.6930, 0.6925,
         0.6920, 0.6915, 0.6910, 0.6905, 0.6900, 0.6895, 0.6890, 0.6885,
         0.6880, 0.6875, 0.6870, 0.6865, 0.6860, 0.6855, 0.6850, 0.6845],
        [0.7250, 0.7245, 0.7240, 0.7235, 0.7230, 0.7225, 0.7220, 0.7215,
         0.7210, 0.7205, 0.7200, 0.7195, 0.7190, 0.7185, 0.7180, 0.7175,
         0.7170, 0.7165, 0.7160, 0.7155, 0.7150, 0.7145, 0.7140, 0.7135,
         0.7130, 0.7125, 0.7120, 0.7115, 0.7110, 0.7105, 0.7100, 0.7095],
        [0.7500, 0.7495, 0.7490, 0.7485, 0.7480, 0.7475, 0.7470, 0.7465,
         0.7460, 0.7455, 0.7450, 0.7445, 0.7440, 0.7435, 0.7430, 0.7425,
         0.7420, 0.7415, 0.7410, 0.7405, 0.7400, 0.7395, 0.7390, 0.7385,
         0.7380, 0.7375, 0.7370, 0.7365, 0.7360, 0.7355, 0.7350, 0.7345],
        [0.7750, 0.7745, 0.7740, 0.7735, 0.7730, 0.7725, 0.7720, 0.7715,
         0.7710, 0.7705, 0.7700, 0.7695, 0.7690, 0.7685, 0.7680, 0.7675,
         0.7670, 0.7665, 0.7660, 0.7655, 0.7650, 0.7645, 0.7640, 0.7635,
         0.7630, 0.7625, 0.7620, 0.7615, 0.7610, 0.7605, 0.7600, 0.7595],
        [0.8000, 0.7995, 0.7990, 0.7985, 0.7980, 0.7975, 0.7970, 0.7965,
         0.7960, 0.7955, 0.7950, 0.7945, 0.7940, 0.7935, 0.7930, 0.7925,
         0.7920, 0.7915, 0.7910, 0.7905, 0.7900, 0.7895, 0.7890, 0.7885,
         0.7880, 0.7875, 0.7870, 0.7865, 0.7860, 0.7855, 0.7850, 0.7845],
        [0.8250, 0.8245, 0.8240, 0.8235, 0.8230, 0.8225, 0.8220, 0.8215,
         0.8210, 0.8205, 0.8200, 0.8195, 0.8190, 0.8185, 0.8180, 0.8175,
         0.8170, 0.8165, 0.8160, 0.8155, 0.8150, 0.8145, 0.8140, 0.8135,
         0.8130, 0.8125, 0.8120, 0.8115, 0.8110, 0.8105, 0.8100, 0.8095],
        [0.8500, 0.8495, 0.8490, 0.8485, 0.8480, 0.8475, 0.8470, 0.8465,
         0.8460, 0.8455, 0.8450, 0.8445, 0.8440, 0.8435, 0.8430, 0.8425,
         0.8420, 0.8415, 0.8410, 0.8405, 0.8400, 0.8395, 0.8390, 0.8385,
         0.8380, 0.8375, 0.8370, 0.8365, 0.8360, 0.8355, 0.8350, 0.8345],
        [0.8750, 0.8745, 0.8740, 0.8735, 0.8730, 0.8725, 0.8720, 0.8715,
         0.8710, 0.8705, 0.8700, 0.8695, 0.8690, 0.8685, 0.8680, 0.8675,
         0.8670, 0.8665, 0.8660, 0.8655, 0.8650, 0.8645, 0.8640, 0.8635,
         0.8630, 0.8625, 0.8620, 0.8615, 0.8610, 0.8605, 0.8600, 0.8595],
    ];
}

/// `.cpp:739-869`, the sixteen `isOpFunc*` predicates (`:288-303`) — which family a compute op's
/// [`OpFunc`] belongs to, asked by the ordered dispatch in `getMinParamForDimFromOpFunc`
/// (`.cpp:1137-1169`).
///
/// ⭐ ASSOCIATED FUNCTIONS, NOT METHODS: every one is a `const` member that reads no field, and each
/// set it tests is a function-local `static const` — one table for all schedulers, so no `self`
/// supplies anything. Same shape as [`Self::burst_efficiency`] and [`Self::create_sync_node`].
///
/// ⛔ THE FOURTEEN LEAF SETS ARE DISJOINT AND NAME ONLY 74 OF [`OpFunc`]'S 176 VARIANTS, and that
/// matters twice over: the dispatch is an `if / else if` chain, so an op in two families would take
/// whichever arm comes first, and each of the other 102 ops falls out of it with a minimum parameter
/// of `1` (`.cpp:1168`). ⛔ THREE OF THE 102 ARE SPELLED `BATCHMATMUL_*` — `BATCHMATMULV2`,
/// `BATCHMATMUL_MXFP4W_FWD` and `BATCHMATMUL_MXFP8_FWD` are NOT bmm to this scheduler, so a matmul
/// flavour added to the vocabulary stays unrecognised until one of these sets is extended by hand.
impl L3DlOpsScheduler {
    /// Replaces: e029g2_L3DlOpsScheduler_opfunc.isOpFunc
    ///
    /// `.cpp:739-744`. The three INT4 convolutions — plain, `GENKG3` and `SPARSEKG3`. Its own arm of
    /// the `IN` minimum, where it alone asks for 128 (`.cpp:905-906`).
    pub fn is_op_func_conv2d_int4(op_func_name: OpFunc) -> bool {
        matches!(
            op_func_name,
            OpFunc::Conv2DInt4Fwd | OpFunc::Conv2DInt4FwdGenkg3 | OpFunc::Conv2DInt4FwdSparsekg3
        )
    }

    /// Replaces: e029g2_L3DlOpsScheduler_opfunc.isOpFunc
    ///
    /// `.cpp:746-751`. The four output-stationary-1 convolutions, which take the whole core extent as
    /// their `IN` minimum rather than a constant (`.cpp:907-908`).
    pub fn is_op_func_conv2d_os1(op_func_name: OpFunc) -> bool {
        matches!(
            op_func_name,
            OpFunc::Conv2DFwdOs1
                | OpFunc::Conv2DXrfInt8FwdOs1
                | OpFunc::Conv2DFwdGenOs1
                | OpFunc::Conv2DInt8FwdOs1
        )
    }

    /// Replaces: e029g2_L3DlOpsScheduler_opfunc.isOpFunc
    ///
    /// `.cpp:753-767`. Any convolution: the INT4 three, the OS1 four, and nine more.
    ///
    /// ⛔ THOSE NINE HAVE NO PREDICATE OF THEIR OWN. `opFuncConv2dOthers` (`.cpp:754-763`) is local to
    /// this body, so this is NOT the disjunction of the two named conv2d predicates and nothing can
    /// ask for the remainder alone.
    pub fn is_op_func_conv2d(op_func_name: OpFunc) -> bool {
        Self::is_op_func_conv2d_int4(op_func_name)
            || Self::is_op_func_conv2d_os1(op_func_name)
            || matches!(
                op_func_name,
                OpFunc::Conv2DFwd
                    | OpFunc::Conv2DFp8Fwd
                    | OpFunc::Conv2DInt8Fwd
                    | OpFunc::Conv2DFwdGenkg3
                    | OpFunc::Conv2DFp8FwdGenkg3
                    | OpFunc::Conv2DInt8FwdGenkg3
                    | OpFunc::Conv2DFwdSparsekg3
                    | OpFunc::Conv2DFp8FwdSparsekg3
                    | OpFunc::Conv2DInt8FwdSparsekg3
            )
    }

    /// Replaces: e029g2_L3DlOpsScheduler_opfunc.isOpFunc
    ///
    /// `.cpp:769-774`. The four INT4 batch matmuls, XRF and non-XRF alike (`.cpp:963-964`).
    pub fn is_op_func_bmm_int4(op_func_name: OpFunc) -> bool {
        matches!(
            op_func_name,
            OpFunc::BatchmatmulInt4Fwd
                | OpFunc::BatchmatmulInt4FwdSparsekg3
                | OpFunc::BatchmatmulXrfInt4Fwd
                | OpFunc::BatchmatmulXrfchInt4Fwd
        )
    }

    /// Replaces: e029g2_L3DlOpsScheduler_opfunc.isOpFunc
    ///
    /// `.cpp:776-782`. The five INT8 batch matmuls, which share the FP8-non-XRF arm of the bmm
    /// minimum (`.cpp:965-966`).
    pub fn is_op_func_bmm_int8(op_func_name: OpFunc) -> bool {
        matches!(
            op_func_name,
            OpFunc::BatchmatmulInt8Fwd
                | OpFunc::BatchmatmulInt8FwdMbkg3
                | OpFunc::BatchmatmulInt8FwdSparsekg3
                | OpFunc::BatchmatmulXrfInt8Fwd
                | OpFunc::BatchmatmulXrfchInt8Fwd
        )
    }

    /// Replaces: e029g2_L3DlOpsScheduler_opfunc.isOpFunc
    ///
    /// `.cpp:784-789`. The three FP8 batch matmuls that do NOT go through the XRF — ⭐ INCLUDING
    /// `BATCHMATMUL_FP8_FWD_MB`, the multi-batch one, which is on our own fp8 decode path.
    pub fn is_op_func_bmm_fp8_non_xrf(op_func_name: OpFunc) -> bool {
        matches!(
            op_func_name,
            OpFunc::BatchmatmulFp8Fwd
                | OpFunc::BatchmatmulFp8FwdMb
                | OpFunc::BatchmatmulFp8FwdSparsekg3
        )
    }

    /// Replaces: e029g2_L3DlOpsScheduler_opfunc.isOpFunc
    ///
    /// `.cpp:791-795`. The two FP8 batch matmuls that DO — the XRF and per-channel-XRF pair, split
    /// out because they get their own arm of the bmm minimum (`.cpp:983-990`).
    pub fn is_op_func_bmm_fp8_xrf(op_func_name: OpFunc) -> bool {
        matches!(
            op_func_name,
            OpFunc::BatchmatmulXrfFp8Fwd | OpFunc::BatchmatmulXrfchFp8Fwd
        )
    }

    /// Replaces: e029g2_L3DlOpsScheduler_opfunc.isOpFunc
    ///
    /// `.cpp:797-802`. The four FP16 batch matmuls; the unsuffixed `BATCHMATMUL_FWD` is one of them
    /// (`.cpp:974-982`).
    pub fn is_op_func_bmm_fp16(op_func_name: OpFunc) -> bool {
        matches!(
            op_func_name,
            OpFunc::BatchmatmulFwd
                | OpFunc::BatchmatmulFwdSparsekg3
                | OpFunc::BatchmatmulXrfFwd
                | OpFunc::BatchmatmulXrfchFwd
        )
    }

    /// Replaces: e029g2_L3DlOpsScheduler_opfunc.isOpFunc
    ///
    /// `.cpp:804-808`. Any batch matmul: the union of the five format-keyed sets, and nothing else.
    ///
    /// ⛔ NOT "ANY OP SPELLED `BATCHMATMUL_*`" — see the family note on this `impl`.
    pub fn is_op_func_bmm(op_func_name: OpFunc) -> bool {
        Self::is_op_func_bmm_fp16(op_func_name)
            || Self::is_op_func_bmm_fp8_xrf(op_func_name)
            || Self::is_op_func_bmm_fp8_non_xrf(op_func_name)
            || Self::is_op_func_bmm_int4(op_func_name)
            || Self::is_op_func_bmm_int8(op_func_name)
    }

    /// Replaces: e029g2_L3DlOpsScheduler_opfunc.isOpFunc
    ///
    /// `.cpp:810-827`. The fourteen ops whose operand may be a broadcast scalar — seven activations,
    /// five arithmetic ops, `BIASADD` and `BATCHNORM_FWD`.
    ///
    /// ⛔ `REALDIV` IS NOT ONE, though `ADD`, `MUL`, `SUB` and `REVSUB` all are.
    pub fn is_op_func_scalar_broadcast(op_func_name: OpFunc) -> bool {
        matches!(
            op_func_name,
            OpFunc::ReluFwd
                | OpFunc::Relu6Fwd
                | OpFunc::LeakyreluFwd
                | OpFunc::GeluFwd
                | OpFunc::TanhFwd
                | OpFunc::SigmoidFwd
                | OpFunc::FastSigmoidFwd
                | OpFunc::Add
                | OpFunc::StridedAdd
                | OpFunc::Mul
                | OpFunc::Sub
                | OpFunc::Revsub
                | OpFunc::Biasadd
                | OpFunc::BatchnormFwd
        )
    }

    /// Replaces: e029g2_L3DlOpsScheduler_opfunc.isOpFunc
    ///
    /// `.cpp:829-835`. Four stick reductions and four non-stick ones.
    ///
    /// ⛔ THE TWO HALVES DO NOT MIRROR EACH OTHER: `PROD_NONSTICK` is here with no stick `PROD`
    /// beside it, and `ABSMAX`, `MIN`, `EXX2_ZEROMEAN` and both `_NONSTICK` spellings of the first
    /// two are reductions the vocabulary has and this predicate rejects.
    pub fn is_op_func_reduction(op_func_name: OpFunc) -> bool {
        matches!(
            op_func_name,
            OpFunc::Sum
                | OpFunc::Max
                | OpFunc::Mean
                | OpFunc::Exx2
                | OpFunc::SumNonstick
                | OpFunc::MaxNonstick
                | OpFunc::MeanNonstick
                | OpFunc::ProdNonstick
        )
    }

    /// Replaces: e029g2_L3DlOpsScheduler_opfunc.isOpFunc
    ///
    /// `.cpp:837-841`. The three pooling ops, which share the depthwise conv's minimum
    /// (`.cpp:1159-1161`).
    pub fn is_op_func_pooling(op_func_name: OpFunc) -> bool {
        matches!(
            op_func_name,
            OpFunc::MaxpoolFwd | OpFunc::AvgpoolFwd | OpFunc::AvgpoolNmapFwd
        )
    }

    /// Replaces: e029g2_L3DlOpsScheduler_opfunc.isOpFunc
    ///
    /// `.cpp:843-847`. The one depthwise convolution — a set of one in the authority too, and ⛔ NOT a
    /// member of [`Self::is_op_func_conv2d`], which is why the dispatch pairs it with pooling.
    pub fn is_op_func_depthwise_conv(op_func_name: OpFunc) -> bool {
        matches!(op_func_name, OpFunc::DepthwiseConvFwd)
    }

    /// Replaces: e029g2_L3DlOpsScheduler_opfunc.isOpFunc
    ///
    /// `.cpp:849-856`. The twelve quantize / compute-scale-and-quantize ops.
    ///
    /// ⛔ THREE SPELLINGS ARE LEFT OUT: `Q_FP8_MB` — though `CSQ_INT8_MB` is in — and the two `_V2`s,
    /// `CSQ_INT8_V2` and `CSQ_INT8_MB_V2`.
    pub fn is_op_func_quantization(op_func_name: OpFunc) -> bool {
        matches!(
            op_func_name,
            OpFunc::QFp8
                | OpFunc::QFp8Ch
                | OpFunc::QFp8Chil
                | OpFunc::QFp8Wt
                | OpFunc::CsqInt8
                | OpFunc::CsqInt8Ch
                | OpFunc::CsqInt8Wt
                | OpFunc::CsqInt8Chil
                | OpFunc::CsqInt8Mb
                | OpFunc::CsqInt4
                | OpFunc::CsqInt4Wt
                | OpFunc::CsqInt4Chil
        )
    }

    /// Replaces: e029g2_L3DlOpsScheduler_opfunc.isOpFunc
    ///
    /// `.cpp:858-863`. The DL16↔FP32 conversion pair — ⛔ and only that pair: `FP8TODL16` and
    /// `DL16TOBF16` are conversions this predicate rejects.
    pub fn is_op_func_conversion_dl16_and_fp32(op_func_name: OpFunc) -> bool {
        matches!(op_func_name, OpFunc::Dl16Tofp32 | OpFunc::Fp32Todl16)
    }

    /// Replaces: e029g2_L3DlOpsScheduler_opfunc.isOpFunc
    ///
    /// `.cpp:865-869`. Any op that walks a window across its input with a stride — every convolution,
    /// every pooling op, and the depthwise conv. Its one call site is the `DT_CHECK_MSG` that guards
    /// `computeMinParamForPaddedDim` (`.cpp:873-874`), so it states a precondition rather than
    /// choosing a minimum.
    pub fn is_op_func_strided_window(op_func_name: OpFunc) -> bool {
        Self::is_op_func_conv2d(op_func_name)
            || Self::is_op_func_pooling(op_func_name)
            || Self::is_op_func_depthwise_conv(op_func_name)
    }
}

/// `.cpp:6596-6604`, `isPagedLds` (`:433`) — whether one labeled data structure is the PAGED VALUE
/// tensor of an indirect access, which is what `getAllPagedLdsIndices` filters a DSC's tensors with
/// (`.cpp:6735`) and what `addOnePageDataStage` then reads a page size out of (`.cpp:6686-6697`). Its
/// sibling `isIndexLds` asks the same question of the ADDRESSES (`.cpp:6582-6594`) and belongs to
/// another group.
///
/// ⛔ THE PARAMETER IS THE HBM ALLOCATION, NOT THE `LabeledDsInfo`, BECAUSE THE LINK BETWEEN THEM IS
/// A POINTER THIS CRATE DOES NOT CARRY: `lds.memOrg_.at(HBM).allocateNode_` is a
/// `dsc2::AllocateNode*` (`dsc/dscdefn.h:313`) inside the `MemOrg` map `LabeledDsInfo::memOrg_`
/// (`dsc/dscdefn.h:337`), and the schedule tree owns its nodes as
/// [`ChildNode`](crate::schedule::dsc2::ChildNode)s, so no type outside the tree holds a node's
/// address. Same technique and the same reason as [`AllocateNode::page_size`], which takes
/// `relatedIndirectAccessAlloc_` as an argument.
///
/// ⭐ SO BOTH OF THE AUTHORITY'S ABSENCES COLLAPSE INTO ONE [`None`], AND NEITHER LOSES ANYTHING: no
/// `HBM` key in `memOrg_` (`.cpp:6597`) and a null `allocateNode_` under one (`.cpp:6599`) reach the
/// same `return false` (`.cpp:6603`).
///
/// ⛔ AND `component_` IS TESTED HERE WHERE THE AUTHORITY TESTS THE MAP KEY, BECAUSE THEY ARE ONE
/// FACT AND THE TEST IS LOAD-BEARING. `createAllocateNode` writes `component_` from the storage it is
/// asked for (`.cpp:550`) and its caller files the node under that same storage (`.cpp:6958-6964`
/// with `.cpp:6975`), so the key and the field agree; and the collector that gathers these very nodes
/// off the schedule tree spells the predicate over the node alone instead —
/// `indirectAllocType_ == VALUE_TENSOR && component_ == HBM` (`.cpp:6786-6789`). ⛔ WITHOUT IT AN LX
/// ALLOCATION COULD ANSWER TRUE: `.cpp:6965-6966` copies an HBM allocation's `indirectAllocType_`
/// onto an LX one for the same tensor, so a non-`NO_INDIRECTION` type is not HBM's alone.
impl L3DlOpsScheduler {
    /// Replaces: e029g5_L3DlOpsScheduler_paged.isPagedLds
    ///
    /// `.cpp:6596-6604`. Whether this is the paged VALUE tensor of an indirect access — the data, not
    /// the addresses into it. The block above has the parameter and both absences.
    pub fn is_paged_lds(hbm_allocate: Option<&AllocateNode>) -> bool {
        hbm_allocate.is_some_and(|allocate| {
            allocate.indirect_alloc_type == IndirectAllocType::ValueTensor
                && allocate.component == SenComponent::Hbm
        })
    }
}

#[cfg(test)]
mod equivalence {
    use super::*;
    use crate::schedule::dims::PadType;
    use crate::schedule::dsc2::TransferNode;

    /// `:63-73` against both real construction sites — `{executionStep}` and `{0}`
    /// (`dbo/src/Utils/sdsc_bundle/SchedulerStages.cpp:31`, `L3DlOpsScheduler_standalone.cpp:192`).
    /// An empty list is `DT_CHECK(!exphases.empty())` (`:72`) and the only way in refuses it, so no
    /// later reader repeats the check. Every other member starts at its initialiser (`:214`, `:221-224`).
    #[test]
    fn the_constructor_refuses_an_empty_phase_list_and_starts_every_member_at_its_initialiser() {
        assert_eq!(
            L3DlOpsScheduler::new(Vec::new(), Verbosity(0), LxBufferTypeMode::Auto).is_none(),
            true,
            "DT_CHECK(!exphases.empty()) is the constructor's only check"
        );

        let scheduler = L3DlOpsScheduler::new(
            vec![ExPhase(7)],
            Verbosity(2),
            LxBufferTypeMode::ForceSpatialDouble,
        )
        .expect("one execution step is what both call sites pass");
        assert_eq!(scheduler.exphases(), [ExPhase(7)]);
        assert_eq!(scheduler.verbose, Verbosity(2));
        assert_eq!(
            scheduler.lx_buffer_type_mode,
            LxBufferTypeMode::ForceSpatialDouble
        );
        assert_eq!(
            scheduler.lx_buffer_type,
            BufferType::Double,
            "the mode is a request; setLxBufferType has not run yet (.cpp:6425)"
        );
        assert_eq!(scheduler.gtr_curr_group_name, GroupId(0));
        assert_eq!(scheduler.data_stage_ibr_idx, None);
        assert_eq!(scheduler.data_stage_one_page_idx, None);
        assert_eq!(scheduler.data_stage_super_chunk_idx, None);
        assert!(scheduler.dsc_metadata.is_empty());
        assert!(scheduler.cores_set_to_gtr_group_name_map.is_empty());

        assert_eq!(L3DlOpsScheduler::DATA_STAGE_CORE_IDX, DataStageId(0));
        assert_eq!(L3DlOpsScheduler::DATA_STAGE_CHUNK_IDX, DataStageId(1));
        assert_eq!(
            L3DlOpsScheduler::LX_BELOW_BLOCK_NODE_NAME,
            "lx_below_schedule"
        );
    }

    /// `:117-122` — `std::max` on the floor and `std::min` on the ceiling, so each update keeps the
    /// TIGHTER bound and the first one takes the incoming value whole.
    #[test]
    fn the_two_scalar_bounds_keep_the_tighter_of_the_two() {
        let val = |v: f32| ConstraintValue::new(v).expect("not NaN");
        let mut constraints = Constraints::default();
        assert_eq!((constraints.min, constraints.max), (None, None));
        assert!(!constraints.must_be_multiple);

        constraints.update_min(val(4.0));
        constraints.update_max(val(64.0));
        assert_eq!(constraints.min, Some(val(4.0)));
        assert_eq!(constraints.max, Some(val(64.0)));

        constraints.update_min(val(2.0));
        constraints.update_max(val(128.0));
        assert_eq!(constraints.min, Some(val(4.0)), "a looser floor is ignored");
        assert_eq!(
            constraints.max,
            Some(val(64.0)),
            "a looser ceiling is ignored"
        );

        constraints.update_min(val(8.0));
        constraints.update_max(val(32.0));
        assert_eq!(constraints.min, Some(val(8.0)));
        assert_eq!(constraints.max, Some(val(32.0)));
    }

    /// `:123-125` — `set_intersect` (`util/utils.h:112-116`). ⛔ AND THE NEGATIVE THE `Option` EXISTS
    /// FOR: a disjoint update leaves the set PRESENT AND EMPTY, which admits no size at all, where
    /// absent admits every size.
    #[test]
    fn the_value_set_intersects_and_a_disjoint_update_empties_it_rather_than_clearing_it() {
        let val = |v: f32| ConstraintValue::new(v).expect("not NaN");
        let mut constraints = Constraints::default();
        assert_eq!(constraints.values, None);

        constraints.update_values(BTreeSet::from([val(8.0), val(16.0), val(32.0)]));
        assert_eq!(
            constraints.values,
            Some(BTreeSet::from([val(8.0), val(16.0), val(32.0)])),
            "nothing was there, so the incoming set is taken whole"
        );

        constraints.update_values(BTreeSet::from([val(16.0), val(32.0), val(64.0)]));
        assert_eq!(
            constraints.values,
            Some(BTreeSet::from([val(16.0), val(32.0)])),
            "the intersection, not the union"
        );

        constraints.update_values(BTreeSet::from([val(8.0)]));
        assert_eq!(
            constraints.values,
            Some(BTreeSet::new()),
            "present and empty is not absent"
        );
    }

    /// `.cpp:1609-1626` against the table's own four corners, plus both monotonicities
    /// `BurstEfficiency.def:13-16` states — which is also what pins all 1024 transcribed values as
    /// the `.def`'s and not a formula's.
    #[test]
    fn the_burst_table_is_the_defs_corners_and_both_monotonicities_it_states() {
        let eff = |burst: u32, degree: u32| {
            L3DlOpsScheduler::burst_efficiency(
                BurstSize::new(burst).expect("in range"),
                MulticastDegree::new(degree).expect("in range"),
            )
            .0
        };
        assert_eq!(eff(1, 1), 0.1000);
        assert_eq!(eff(1, 32), 0.0845);
        assert_eq!(eff(32, 1), 0.8750);
        assert_eq!(eff(32, 32), 0.8595);

        for burst in 1..=BurstSize::MAX {
            for degree in 2..=MulticastDegree::MAX {
                assert!(
                    eff(burst, degree) < eff(burst, degree - 1),
                    "a larger multicast degree is worse at burst size {burst}"
                );
            }
        }
        for degree in 1..=MulticastDegree::MAX {
            for burst in 2..=BurstSize::MAX {
                assert!(
                    eff(burst, degree) > eff(burst - 1, degree),
                    "a larger burst size is better at multicast degree {degree}"
                );
            }
        }
    }

    /// `.cpp:1611-1615` — the two argument `DT_CHECK`s. Both arguments are ONE-BASED, so `0` is out
    /// of range at the bottom and the table's own edge is the top.
    #[test]
    fn both_one_based_arguments_refuse_zero_and_anything_past_the_tables_edge() {
        assert_eq!(BurstSize::new(0), None);
        assert_eq!(BurstSize::new(BurstSize::MAX + 1), None);
        assert_eq!(BurstSize::new(1).map(BurstSize::get), Some(1));
        assert_eq!(BurstSize::new(32).map(BurstSize::get), Some(32));

        assert_eq!(MulticastDegree::new(0), None);
        assert_eq!(MulticastDegree::new(MulticastDegree::MAX + 1), None);
        assert_eq!(MulticastDegree::new(1).map(MulticastDegree::get), Some(1));
        assert_eq!(MulticastDegree::new(32).map(MulticastDegree::get), Some(32));
    }

    /// `:193-197` — `clear()` is `*this = {}`, so every carried field goes back to its initialiser,
    /// including the two data-stage ids `prepDsc` had written (`.cpp:6420-6421`) and the row-split dim.
    ///
    /// ⛔ FIELD BY FIELD, NOT AGAINST `Metadata::default()`: [`Metadata`] owns a node pair per
    /// external transfer and so states no equality at all.
    #[test]
    fn clearing_the_table_restores_the_initialisers_including_the_two_unassigned_stage_ids() {
        let mut metadata = Metadata::default();
        assert_eq!(metadata.core_dstgid, None);
        assert_eq!(metadata.chunk_dstgid, None);
        assert_eq!(metadata.row_split_dim, PrimaryDimTypes::Undefined);
        assert!(metadata.datastages.is_empty());
        assert!(metadata.external_transfers.is_empty());

        metadata.core_dstgid = Some(L3DlOpsScheduler::DATA_STAGE_CORE_IDX);
        metadata.chunk_dstgid = Some(L3DlOpsScheduler::DATA_STAGE_CHUNK_IDX);
        metadata.row_split_dim = PrimaryDimTypes::In;
        metadata
            .datastages
            .insert(DataStageId(3), Datastage::default());
        let stage = &metadata.datastages[&DataStageId(3)];
        assert!(stage.strategy_minimize, "`:131`, minimise by default");
        assert_eq!(stage.nearest_numerator_idx, None, "`:133`, the -1");
        assert!(stage.constraints.is_empty());
        assert!(stage.relevant_dims_and_numerator.is_empty());

        metadata.clear();
        assert_eq!(metadata.core_dstgid, None);
        assert_eq!(metadata.chunk_dstgid, None);
        assert_eq!(metadata.row_split_dim, PrimaryDimTypes::Undefined);
        assert!(metadata.datastages.is_empty());
        assert!(metadata.external_transfers.is_empty());
    }

    /// `:145-153` — the setter inserts under the dim and the mutable list is the only reader, so a
    /// second write to the same dim REPLACES rather than adds. The two initialisers at `:143-144` come
    /// with it.
    #[test]
    fn a_seconds_write_to_one_dim_replaces_that_dims_access_pattern() {
        let mut transfer = DataTransfer::default();
        assert!(!transfer.apply_row_offset);
        assert_eq!(transfer.force_num_elements, None);
        assert!(transfer.access_pattern_list_mut().is_empty());

        let padded = TransferAccessPattern {
            src: PadType::PaddedWZeroPad,
            dst: PadType::NoPad,
        };
        transfer.set_access_pattern(PrimaryDimTypes::In, padded);
        transfer.set_access_pattern(PrimaryDimTypes::Mb, TransferAccessPattern::default());
        assert_eq!(transfer.access_pattern_list_mut().len(), 2);

        let lowered = TransferAccessPattern {
            src: PadType::PaddedFullSpan,
            dst: PadType::LoweredPadded,
        };
        transfer.set_access_pattern(PrimaryDimTypes::In, lowered);
        assert_eq!(transfer.access_pattern_list_mut().len(), 2);
        assert_eq!(
            transfer.access_pattern_list_mut()[&PrimaryDimTypes::In],
            lowered
        );
        assert_eq!(
            transfer.access_pattern_list_mut()[&PrimaryDimTypes::Mb],
            TransferAccessPattern::default()
        );
    }

    /// `:169-176` — the two `unique_ptr`s mean the entry OWNS both nodes, so a write through the entry
    /// reaches the node the entry holds and no second handle can disagree with it.
    ///
    /// ⛔ THIS TEST IS THE FIELD'S ONLY WRITER: nothing in `L3DlOpsScheduler.cpp` constructs an
    /// `ExternalTransfer` or pushes onto `externalTransfers_`, so the shape is checked and never
    /// exercised by a real caller.
    #[test]
    fn an_external_transfer_owns_the_node_pair_it_adopts() {
        let transfer = TransferNode {
            replication_factor: 4,
            ..TransferNode::default()
        };
        let allocate = AllocateNode {
            lds_idx: Some(LdsIdx(9)),
            ..AllocateNode::default()
        };

        let mut entry = ExternalTransfer::new(Box::new(transfer), Box::new(allocate));
        assert_eq!(entry.transfer.replication_factor, 4);
        assert_eq!(entry.allocate.lds_idx, Some(LdsIdx(9)));
        entry.allocate.lds_idx = Some(LdsIdx(11));

        let mut metadata = Metadata::default();
        metadata.external_transfers.push(entry);
        assert_eq!(
            metadata.external_transfers[0].allocate.lds_idx,
            Some(LdsIdx(11)),
            "the write reached the node the vector owns"
        );
    }

    /// `:185-190` — the opaque op's member initialisers. An unroll of one is what `allocAllMem`
    /// compares a stick count against (`.cpp:5578-5581`), and it is reached only past a `DT_ERROR`.
    #[test]
    fn an_opaque_op_starts_with_no_registers_and_an_unroll_of_one() {
        let op = OpaqueOp::default();
        assert!(op.internal_regs.is_empty());
        assert_eq!(op.max_unroll, 1);
    }

    /// `.cpp:3961-3974`, the soft-sync pair, against `createSyncNode` alone. The flags are ordered
    /// `isReceive` then `isSoft` (`:275-276`), so the send — soft and not a receive — is the mint a
    /// transposition breaks, and each end carries the units and the name the sequence hands it.
    #[test]
    fn the_soft_sync_pair_mints_a_soft_send_and_a_soft_receive() {
        let send = L3DlOpsScheduler::create_sync_node(
            BTreeSet::from([SenComponent::L3lu]),
            format!(
                "sync_soft_send_{}_to_{}",
                SenComponent::L3lu.spelling(),
                SenComponent::Lxlu.spelling()
            ),
            false,
            true,
        );
        let receive = L3DlOpsScheduler::create_sync_node(
            BTreeSet::from([SenComponent::Lxlu]),
            format!(
                "sync_soft_receive_{}_from_{}",
                SenComponent::Lxlu.spelling(),
                SenComponent::L3lu.spelling()
            ),
            true,
            true,
        );

        assert_eq!(send.base_class.name, "sync_soft_send_l3lu_to_lxlu");
        assert_eq!(send.units, BTreeSet::from([SenComponent::L3lu]));
        assert!(
            !send.is_receive,
            "the send end is not a receive (`.cpp:3967`)"
        );
        assert!(send.is_soft, "and it is soft (`.cpp:3967`)");

        assert_eq!(receive.base_class.name, "sync_soft_receive_lxlu_from_l3lu");
        assert_eq!(receive.units, BTreeSet::from([SenComponent::Lxlu]));
        assert!(
            receive.is_receive,
            "the other end is the receive (`.cpp:3974`)"
        );
        assert!(receive.is_soft, "and it is soft too (`.cpp:3974`)");
    }

    /// `.cpp:739-869` — the fourteen leaf `isOpFunc*` sets against every one of [`OpFunc`]'s 176
    /// variants. ⛔ WHAT MAKES THIS MORE THAN A SECOND READING OF THE SAME LITERALS: the families
    /// must PARTITION, because `getMinParamForDimFromOpFunc` is an `if / else if` chain
    /// (`.cpp:1143-1166`) and an op in two of them would silently take the earlier arm; and each
    /// cardinality is the authority's own set size, which a dropped or duplicated variant breaks
    /// even when every name present is spelled right.
    #[test]
    fn the_fourteen_leaf_op_func_families_partition_74_of_the_176_op_funcs() {
        // The nine convolutions in `opFuncConv2dOthers` (`.cpp:754-763`) have no predicate of their
        // own, so the fourteenth leaf is the one the union answers for and the other two do not.
        let conv2d_other = |op: OpFunc| {
            L3DlOpsScheduler::is_op_func_conv2d(op)
                && !L3DlOpsScheduler::is_op_func_conv2d_int4(op)
                && !L3DlOpsScheduler::is_op_func_conv2d_os1(op)
        };
        let leaves: [(&str, &dyn Fn(OpFunc) -> bool, usize); 14] = [
            ("conv2dInt4", &L3DlOpsScheduler::is_op_func_conv2d_int4, 3),
            ("conv2dOs1", &L3DlOpsScheduler::is_op_func_conv2d_os1, 4),
            ("conv2dOthers", &conv2d_other, 9),
            ("bmmInt4", &L3DlOpsScheduler::is_op_func_bmm_int4, 4),
            ("bmmInt8", &L3DlOpsScheduler::is_op_func_bmm_int8, 5),
            (
                "bmmFp8NonXrf",
                &L3DlOpsScheduler::is_op_func_bmm_fp8_non_xrf,
                3,
            ),
            ("bmmFp8Xrf", &L3DlOpsScheduler::is_op_func_bmm_fp8_xrf, 2),
            ("bmmFp16", &L3DlOpsScheduler::is_op_func_bmm_fp16, 4),
            (
                "scalarBroadcast",
                &L3DlOpsScheduler::is_op_func_scalar_broadcast,
                14,
            ),
            ("reduction", &L3DlOpsScheduler::is_op_func_reduction, 8),
            ("pooling", &L3DlOpsScheduler::is_op_func_pooling, 3),
            (
                "depthwiseConv",
                &L3DlOpsScheduler::is_op_func_depthwise_conv,
                1,
            ),
            (
                "quantization",
                &L3DlOpsScheduler::is_op_func_quantization,
                12,
            ),
            (
                "conversionDl16AndFp32",
                &L3DlOpsScheduler::is_op_func_conversion_dl16_and_fp32,
                2,
            ),
        ];

        let mut classified = 0usize;
        for op in OpFunc::ALL {
            let hits: Vec<&str> = leaves
                .iter()
                .filter(|(_, holds, _)| holds(op))
                .map(|(name, _, _)| *name)
                .collect();
            assert!(
                hits.len() <= 1,
                "{op:?} is in {hits:?}; the dispatch takes the first arm, so the families must be \
                 disjoint (`.cpp:1143-1166`)"
            );
            classified += hits.len();
        }
        assert_eq!(
            classified, 74,
            "the sixteen predicates name 74 of the 176 op-funcs; every other one falls through to a \
             minimum parameter of 1 (`.cpp:1168`)"
        );

        for (name, holds, count) in &leaves {
            assert_eq!(
                OpFunc::ALL.iter().filter(|&&op| holds(op)).count(),
                *count,
                "{name} does not hold the authority's number of op-funcs"
            );
        }

        // The absences the family notes claim, each an op the vocabulary spells and no set takes.
        for absent in [
            OpFunc::Realdiv,
            OpFunc::Absmax,
            OpFunc::Min,
            OpFunc::Exx2Zeromean,
            OpFunc::QFp8Mb,
            OpFunc::CsqInt8V2,
            OpFunc::CsqInt8MbV2,
            OpFunc::Fp8Todl16,
            OpFunc::Dl16Tobf16,
        ] {
            assert!(
                !leaves.iter().any(|(_, holds, _)| holds(absent)),
                "{absent:?} is in no isOpFunc* set"
            );
        }
    }

    /// `.cpp:753-767`, `:804-808`, `:865-869` — the three predicates that are unions. Each holds
    /// exactly what its parts hold, and its cardinality is the sum of theirs: 16 convolutions, 18
    /// batch matmuls, 20 strided-window ops. ⛔ AND THE NEGATIVE THAT COSTS THE MOST: three ops
    /// SPELLED `BATCHMATMUL_*` are in none of the five format sets, so `isOpFuncBmm` rejects them and
    /// they take the fall-through minimum of 1 rather than `getMinParamBmm`.
    #[test]
    fn the_three_union_predicates_are_their_parts_and_three_batchmatmuls_are_in_none() {
        for op in OpFunc::ALL {
            assert_eq!(
                L3DlOpsScheduler::is_op_func_bmm(op),
                L3DlOpsScheduler::is_op_func_bmm_fp16(op)
                    || L3DlOpsScheduler::is_op_func_bmm_fp8_xrf(op)
                    || L3DlOpsScheduler::is_op_func_bmm_fp8_non_xrf(op)
                    || L3DlOpsScheduler::is_op_func_bmm_int4(op)
                    || L3DlOpsScheduler::is_op_func_bmm_int8(op),
                "{op:?}: isOpFuncBmm is the five format sets (`.cpp:805-807`)"
            );
            assert_eq!(
                L3DlOpsScheduler::is_op_func_strided_window(op),
                L3DlOpsScheduler::is_op_func_conv2d(op)
                    || L3DlOpsScheduler::is_op_func_pooling(op)
                    || L3DlOpsScheduler::is_op_func_depthwise_conv(op),
                "{op:?}: isOpFuncStridedWindow is conv2d, pooling and depthwise (`.cpp:866-867`)"
            );
        }

        let count =
            |holds: &dyn Fn(OpFunc) -> bool| OpFunc::ALL.iter().filter(|&&op| holds(op)).count();
        assert_eq!(count(&L3DlOpsScheduler::is_op_func_conv2d), 3 + 4 + 9);
        assert_eq!(count(&L3DlOpsScheduler::is_op_func_bmm), 4 + 5 + 3 + 2 + 4);
        assert_eq!(
            count(&L3DlOpsScheduler::is_op_func_strided_window),
            16 + 3 + 1
        );

        for spelled_bmm in [
            OpFunc::Batchmatmulv2,
            OpFunc::BatchmatmulMxfp4WFwd,
            OpFunc::BatchmatmulMxfp8Fwd,
        ] {
            assert!(
                !L3DlOpsScheduler::is_op_func_bmm(spelled_bmm),
                "{spelled_bmm:?} is spelled BATCHMATMUL_* and is in none of the five sets"
            );
        }
        assert!(
            !L3DlOpsScheduler::is_op_func_bmm(OpFunc::MatmulFwd),
            "MATMUL_FWD is a matmul, not a batch matmul"
        );
        assert!(
            !L3DlOpsScheduler::is_op_func_conv2d(OpFunc::DepthwiseConvFwd),
            "the depthwise conv is its own family (`.cpp:843-847`)"
        );
    }

    /// `.cpp:74-86` on the branch it chooses and on both answers of each. The two dim objects are set
    /// to say "split" while the core stage says "not", so reading them once a core stage exists would
    /// flip an answer; and `numCoreletsUsed_` absent is neither of the two `bool`s.
    #[test]
    fn the_core_data_stage_answers_alone_and_the_two_dim_objects_only_without_one() {
        use crate::schedule::dims::{DataStructDims, DimSize, DimVal};
        use crate::schedule::dsc::NumCoreletsUsed;
        use crate::schedule::dsc2::DataStage;

        let dim = PrimaryDimTypes::Out;
        let filled = |size: f64| DataStructDims {
            out: DimSize::new(size),
            ..DataStructDims::default()
        };

        let mut dsc = DesignSpaceConfig::default();
        dsc.num_corelets_used = Some(NumCoreletsUsed(2));
        // `.cpp:84-85` answers `true` from these two: 32 per corelet of 64 per core.
        dsc.corelet_d = filled(32.0);
        dsc.core_d = filled(64.0);

        // With no core stage the fallback reads them.
        assert_eq!(
            L3DlOpsScheduler::is_dimension_corelet_split(&dsc, dim),
            Some(true),
            "`CoreletD_` below `CoreD_` is the split (`.cpp:84-85`)"
        );

        // A core stage that does NOT split the dim takes the answer over.
        dsc.data_stage_param.insert(
            L3DlOpsScheduler::DATA_STAGE_CORE_IDX,
            DataStage {
                ss: filled(64.0),
                ..DataStage::default()
            },
        );
        assert_eq!(
            L3DlOpsScheduler::is_dimension_corelet_split(&dsc, dim),
            Some(false),
            "the stage answers alone and `CoreletD_`/`CoreD_` go unread (`.cpp:77-83`)"
        );

        // The same stage with the dim in `coreletSplit_`: `clId = 0` reads its share, `-1` the whole.
        dsc.data_stage_param
            .get_mut(&L3DlOpsScheduler::DATA_STAGE_CORE_IDX)
            .unwrap()
            .ss
            .corelet_split
            .insert(dim, vec![DimVal(32), DimVal(32)]);
        assert_eq!(
            L3DlOpsScheduler::is_dimension_corelet_split(&dsc, dim),
            Some(true),
            "32 against the whole object's 64 (`.cpp:79-81`)"
        );

        // `numCoreletsUsed_` absent is the indeterminate read at `.cpp:75`, not a `false`.
        dsc.num_corelets_used = None;
        assert_eq!(
            L3DlOpsScheduler::is_dimension_corelet_split(&dsc, dim),
            None
        );

        // One corelet is `.cpp:75`'s early `false`, ahead of either branch.
        dsc.num_corelets_used = Some(NumCoreletsUsed(1));
        assert_eq!(
            L3DlOpsScheduler::is_dimension_corelet_split(&dsc, dim),
            Some(false)
        );
    }

    /// `.cpp:6596-6604`. The VALUE half of an indirect access is paged; the index half beside it is
    /// not, a direct allocation is not, the LX allocation that inherits the very same
    /// `indirectAllocType_` (`.cpp:6965-6966`) is not, and an absent HBM allocation — IBM's missing
    /// `memOrg_` key and its null `allocateNode_` alike — is not.
    #[test]
    fn only_a_value_tensor_allocation_in_hbm_is_a_paged_lds() {
        let allocate = |indirect: IndirectAllocType, component: SenComponent| AllocateNode {
            indirect_alloc_type: indirect,
            component,
            ..AllocateNode::default()
        };

        assert!(L3DlOpsScheduler::is_paged_lds(Some(&allocate(
            IndirectAllocType::ValueTensor,
            SenComponent::Hbm
        ))));

        for indirect in [
            IndirectAllocType::NoIndirection,
            IndirectAllocType::IndexTensor,
        ] {
            assert!(
                !L3DlOpsScheduler::is_paged_lds(Some(&allocate(indirect, SenComponent::Hbm))),
                "{indirect:?} is not the paged value tensor (`.cpp:6600`)"
            );
        }

        assert!(
            !L3DlOpsScheduler::is_paged_lds(Some(&allocate(
                IndirectAllocType::ValueTensor,
                SenComponent::Lx
            ))),
            "an LX node carries the same indirectAllocType_ (`.cpp:6965`)"
        );

        assert!(
            !L3DlOpsScheduler::is_paged_lds(None),
            "no HBM memOrg_ entry and a null allocateNode_ both reach `return false` (`.cpp:6603`)"
        );
    }
}

// crustify:todo: e029_L3DlOpsScheduler

// crustify:todo: e029_L3DlOpsScheduler.compAndAllocNode

// crustify:todo: e029_L3DlOpsScheduler.consIdAndAllocNode

// crustify:todo: e029_L3DlOpsScheduler.consumers_

// crustify:todo: e029_L3DlOpsScheduler.dataConnects_

// crustify:todo: e029_L3DlOpsScheduler.datatransfers_

// crustify:todo: e029_L3DlOpsScheduler.externalNodes_

// crustify:todo: e029_L3DlOpsScheduler.inOutRegAllocs_

// crustify:todo: e029_L3DlOpsScheduler.insertBefore

// crustify:todo: e029_L3DlOpsScheduler.internalRegAlloc_

// crustify:todo: e029_L3DlOpsScheduler.isSoft

// crustify:todo: e029_L3DlOpsScheduler.ldsIdxAndAllocNode

// crustify:todo: e029_L3DlOpsScheduler.memTrackers

// crustify:todo: e029_L3DlOpsScheduler.newAllocations_

// crustify:todo: e029_L3DlOpsScheduler.opaqueOps_

// crustify:todo: e029_L3DlOpsScheduler.producers_

// crustify:todo: e029g1_L3DlOpsScheduler_sync

// crustify:todo: e029g1_L3DlOpsScheduler_sync.createSynchronization

// crustify:todo: e029g1_L3DlOpsScheduler_sync.createSynchronizationDSC

// crustify:todo: e029g2_L3DlOpsScheduler_opfunc

// crustify:todo: e029g2_L3DlOpsScheduler_opfunc.isOpCrossCoreReduction

// crustify:todo: e029g3_L3DlOpsScheduler_coord

// crustify:todo: e029g3_L3DlOpsScheduler_coord.propagateCoordinate

// crustify:todo: e029g3_L3DlOpsScheduler_coord.sliceCoordinateForCorelet

// crustify:todo: e029g5_L3DlOpsScheduler_paged

// crustify:todo: e029g5_L3DlOpsScheduler_paged.processPagedTensorTransfers

// crustify:todo: e029g5_L3DlOpsScheduler_paged.processHbmPagedTensors

// crustify:todo: e029g5_L3DlOpsScheduler_paged.processDscHbmPagedTensors

// crustify:todo: e029g5_L3DlOpsScheduler_paged.optimizeHbmTransfers

// crustify:todo: e029g5_L3DlOpsScheduler_paged.optimizeHbmLdsOutputInScheduleTree
