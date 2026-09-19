//! Re-ported from the C++ authority. See crustify-scheduler/AGENT-BRIEF.md.
//!
//! Stage 2a of the scheduler — `L3DlOpsScheduler.run(sdsc)`
//! (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.h`). Every bare `:NNN` citation below is a line of that
//! header; a `.cpp:NNN` one is `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp`.

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
