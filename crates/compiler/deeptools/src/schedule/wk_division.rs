//! Re-ported from the C++ authority. See crustify-scheduler/AGENT-BRIEF.md.
//!
//! How one folded dimension's work is divided across a core gang, and the size and element
//! coordinates each core then sees (`util/foldManager/wkDivisionParams.h:19-456`).
//!
//! ⛔ THE AUTHORITY PATH IS `util/foldManager/wkDivisionParams.h`, NOT the `ddc/wkDivisionParams.h`
//! `crustify-scheduler/UNITS.tsv` cites for this unit: no such file exists in the tree. Every
//! bare `:NNN` citation below is a line of that header.
//!
//! ⭐ FIELDS KEEP THE AUTHORITY'S `int32_t` RANGE; COMPUTED SPANS AND COORDINATES WIDEN TO `i64`.
//! The one consumer reaches this type with an `int64_t` fold dim index and takes
//! `std::vector<std::pair<int64_t, int64_t>>` back (`util/foldManager/foldInfrastructure.h:838`,
//! `:844`), so IBM's implicit narrowing at that call and its `int32_t` span and coordinate
//! arithmetic are widened instead of copied. No real core index or work size comes near either
//! bound.
//!
//! ⛔ `start_cid_offset_` IS THE ONE FIELD THAT ALSO WIDENS, AND BOTH ACCOUNTS THIS FILE GAVE BEFORE
//! THIS REVIEW NAMED THE WRONG ONE. It is a [`Cid`], so it carries the consumer's `int64_t` rather
//! than the `int32_t` IBM declares (`:387`). `real_coordinates_` does NOT widen and must not: it is
//! the one field a public setter can refill after construction
//! ([`set_real_coordinates`](WkSplitParam::set_real_coordinates), IBM's `:372-376`), and an `i64`
//! range there took `pair.second - pair.first` (`:261`) past `i64` — measured, `attempt to subtract
//! with overflow` reached from safe code. [`RealCoordRange`] keeps IBM's stored `int32_t` pair
//! (`:418`) and [`CoordRange`] is what `getCoordVec` hands back (`:250`), which is what IBM's own two
//! vectors are between them.
//!
//! ⭐ WHAT THE WIDENING BUYS, AS A PROPERTY RATHER THAN A HOPE: every quantity computed here is
//! exact throughout the range in which the authority's own `int32_t` arithmetic is defined, and
//! widens where IBM wraps. Since this review it is TOTAL as well, which the two paragraphs it
//! replaces only claimed: no value this type accepts can overflow an `i64` in any method, because
//! [`WkSplitParam::new`] refuses a param whose own span and coordinate products would not fit and
//! both `cid` offsets saturate. Four such overflows were reachable from safe code before that, one
//! of them with three ordinary `int32_t` fields at their maximum; each is pinned by a test below.
//! The one unbounded accumulation left is `cum_running_size` over `real_coordinates_` (`:262`), at
//! most `2^32` per entry, so it needs about `2^31` stored entries — thirty-four gigabytes of table —
//! and the authority's `int64_t` accumulator carries exactly the same bound.

// ⛔ EIGHT OF THE TWENTY-SIX FIELD ANCHORS THE SCHEDULER WROTE FOR THIS UNIT NAME METHOD-BODY
// LOCALS, NOT DECLARED FIELDS. They are removed rather than invented as state, which Rules 2 and 5
// of crustify-scheduler/AGENT-BRIEF.md forbid; the census matched a declaration inside a method
// body — six of the eight carry an initialiser, `coord_vec` (`:252`) and `myRealCoord` (`:266`) do
// not. Named here so the removal is not silent:
//   offset_cid      `:197`          local of getSliceId
//   slid            `:213`          local of getSliceId
//   vsize           `:241`, `:447`  locals of getSize and getCoord
//   coord_vec       `:252`          local of getCoordVec
//   cumRunningSize  `:256`          local of getCoordVec
//   myRealCoord     `:266`          local of getCoordVec
//   start_vcoord    `:434`          local of getCoord
//   end_vcoord      `:435`          local of getCoord
// The other eighteen are this type's fourteen declared fields (`:379-420`) plus the four of its
// nested `StrWinPad` (`:21-26`), which the scheduler flattened onto the owner.

/// One core index — IBM's `cid`, an `int32_t` at every declaration that takes one (`:145`, `:196`,
/// `:235`, `:250`, `:428`).
///
/// ⛔ NOT `sys_arch_spec::CoreId`, which is a `u8`: `adjustCID` is written to be reached with a
/// negative (`:146`), so a `u8` would force a fallible narrowing at the seam.
/// ⛔ WIDER THAN IBM'S OWN `cid`, AND THAT IS THE DIVERGENCE. The sole consumer passes a fold dim
/// index, an `int64_t` (`util/foldManager/foldInfrastructure.h:838`, `:844`), which the `int32_t`
/// parameter silently wraps — so a coordinate past `INT32_MAX` answers for a DIFFERENT core there.
/// Keeping the caller's width leaves it out of range instead; the conversion happens in exactly one
/// place, `WkSplitFoldFunctionLeaf::as_cid`, where it is named.
/// ⛔ AND "OUT OF RANGE" MEANT "ABORT" AT THE TWO EXTREMES UNTIL THIS REVIEW, which is the one
/// outcome the widening exists to rule out. `Cid(i64::MAX)` against `start_cid_offset = Cid(-1)`
/// aborted in [`slice_id`](WkSplitParam::slice_id)'s subtraction and `Cid(i64::MIN)` against a
/// negative `max_cores_` aborted in [`adjust_cid`](WkSplitParam::adjust_cid)'s addition — both
/// measured, both from safe code, both reachable through `WkSplitFoldFunctionLeaf::folded_size`,
/// whose `FoldDimIndex` is an unguarded `i64`. Both offsets now saturate, which is exact for every
/// input: a cid whose true offset leaves `i64` is outside every representable span, and so is the
/// saturated one, so each answers [`None`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Cid(pub i64);

/// Work done by one steady-state core (`:382`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WkSs(pub i32);

/// Work done by one epilogue core (`:383`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WkEpilogue(pub i32);

/// The work one core does, as `getSize` reports it for either kind of slice (`:235-248`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WkSize(pub i64);

/// How many cores the gang has — `seidGangs.at(seGangId).size()` at the producer
/// (`dsm/workOptimizer/baseOptimizer/workdivopt.cpp:1978`, stored at `:386`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MaxCores(pub i32);

/// How many cores get steady-state work (`:391`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NumSsSlices(pub i32);

/// How many cores get epilogue work (`:392`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NumEpilogueSlices(pub i32);

/// Gap cores after all work slices are passed, IBM's `GapAfterFullWkSl` (`:407-412`).
///
/// ⛔ IT DOES DIVIDE, AND THE CLAIM THAT IT NEVER DOES WAS THIS FILE'S OWN: "UNGUARDED WHERE ITS
/// SIBLINGS ARE NOT: it is the only gap that never divides ... a negative can only shrink the covered
/// span ..., never make a divisor zero". It is summed into `getFullInnerLength()` (`:172-173`), which
/// is the `%` divisor at `:204`, and six unit work slices against `gap_after_all_slices_ = -6` take
/// that divisor to EXACTLY ZERO — a param `checkLegality` accepts, measured (`:131-136`). Measured
/// against the header under UBSan, `getSliceId(-9)` on it is `runtime error: division by zero,
/// wkDivisionParams.h:204`, because `adjustCID` cannot bring `-9` back into a gang of eight and the
/// `:200` range test lets a negative offset through.
/// ⛔ SO THE NEWTYPE STAYS UNGUARDED FOR A DIFFERENT REASON THAN IT CLAIMED. Refusing a negative
/// here would refuse a param IBM builds, and whether the sum reaches zero is not this field's to
/// know. What makes the divisor safe is the `span_with_gaps < 1` test in
/// [`slice_id`](WkSplitParam::slice_id), which is equivalence-preserving rather than a refusal: once
/// the span is zero or less, IBM's own `:200` answers `-1` for every non-negative offset too.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GapAfterAllSlices(pub i32);

/// Which work slice a core runs (`:213`, `:226`).
///
/// ⛔ NOT `dsc2::SliceId`, which counts a stick's slices. This one counts a dimension's work
/// slices, and the two are indexed by unrelated things.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WkSliceId(pub i64);

/// Cores spanned by some run of work slices — what `getSingleSliceInnerLength` and
/// `getFullInnerLength` return (`:155`, `:169`).
///
/// ⛔ `i64`, NOT THE AUTHORITY'S `int32_t`, BECAUSE A SPAN IS A PRODUCT OF TWO INDEPENDENT `int32_t`
/// FIELDS AND SO DOES NOT FIT ONE. With `gap_within_inner_repeat_ = INT32_MAX` and
/// `repeat_factor_inner_ = 2` — a pair `checkLegality` (`:131-136`) accepts and every newtype in this
/// file accepts — `getSingleSliceInnerLength(false)` wraps to **0**, measured against the header
/// itself, and this port aborted with `attempt to add with overflow`. Widened, the quantity is exact
/// and the `>= 1` that [`slice_id`](WkSplitParam::slice_id)'s divisors rest on holds for EVERY
/// accepted input, rather than only downstream of the authority's own `:200`/`:208` range tests.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CoreSpan(pub i64);

/// One end of a core's range in the dimension's VIRTUAL, gap-free element space (`:434-435`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VCoord(pub i64);

/// The virtual element range one core covers, inclusive at both ends — `getCoord`'s pair (`:454`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VCoordRange {
    /// First virtual element, `start_vcoord`.
    pub start: VCoord,
    /// Last virtual element, `end_vcoord`.
    pub end: VCoord,
}

/// One end of a range in the dimension's REAL element space, as `getCoordVec` hands it back
/// (`:266`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Coord(pub i64);

/// A real element range, inclusive at both ends — one entry of what `getCoordVec` returns (`:250`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CoordRange {
    /// First real element, `pair.first`.
    pub start: Coord,
    /// Last real element, `pair.second`.
    pub end: Coord,
}

impl CoordRange {
    /// What a gap core's coordinate is — `getCoord`'s `std::make_pair(-1, -1)` (`:432`), pushed as
    /// the one and only entry of the vector a gap core gets back (`:253-254`).
    ///
    /// ⛔ NOT AN ABSENCE THE CONSUMERS CAN IGNORE: the vendor's own goldens assert this pair on
    /// nine of sixteen cores (`util/foldManager/test/test_fold_infrastructure.cpp:245-253`), so it
    /// stays a named value rather than becoming an `Option`.
    pub const GAP: Self = Self {
        start: Coord(-1),
        end: Coord(-1),
    };

    /// The identity conversion IBM performs when there is no real coordinate table: with
    /// `real_coordinates_` empty the "real" coordinates ARE the virtual ones, widened (`:253-254`).
    pub const fn from_virtual(range: VCoordRange) -> Self {
        Self {
            start: Coord(range.start.0),
            end: Coord(range.end.0),
        }
    }
}

/// One end of a STORED entry of `real_coordinates_` — an `int32_t` at IBM's declaration (`:418`) and
/// at the producer that fills it (`dsm/workOptimizer/baseOptimizer/workdivopt.cpp:2069-2074`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RealCoord(pub i32);

/// One entry of IBM's `real_coordinates_` table, inclusive at both ends (`:418`).
///
/// ⛔ NOT [`CoordRange`], AND SPELLING THEM ONE TYPE PUT AN OVERFLOW IN SAFE CODE. IBM stores
/// `int32_t` pairs and widens each to `int64_t` only on the way out (`:250`, `:254`, `:279`); one
/// `i64` type for both let [`WkSplitParam::set_real_coordinates`] store `(i64::MIN, i64::MAX)`,
/// after which `getCoordVec`'s own `pair.second - pair.first` (`:261`) aborted with `attempt to
/// subtract with overflow` — measured. At the authority's width an entry spans at most `2^32`, so
/// the remap arithmetic is total, and the range the old type could hold is now inexpressible:
///
/// ```compile_fail
/// use deeptools::schedule::wk_division::{RealCoord, RealCoordRange};
/// let _ = RealCoordRange {
///     start: RealCoord(i64::MIN),
///     end: RealCoord(i64::MAX),
/// };
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RealCoordRange {
    /// First stored element, `pair.first`.
    pub start: RealCoord,
    /// Last stored element, `pair.second`.
    pub end: RealCoord,
}

impl RealCoordRange {
    /// The widening IBM performs on each entry inside `getCoordVec` (`:257-278`).
    fn widen(self) -> (i64, i64) {
        (i64::from(self.start.0), i64::from(self.end.0))
    }
}

/// Whether a span includes its trailing gap cores — IBM's `with_gap`, which both call sites that
/// omit it default to `true` (`:155`, `:169`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WithGap {
    /// Count the trailing gap cores.
    Yes,
    /// Leave them out.
    No,
}

/// Gap cores after each valid core assignment, IBM's `gap_within_inner_repeat_` (`:395-396`).
///
/// ⛔ CONSTRUCTION REFUSES A NEGATIVE, AND THAT IS WHAT MAKES TWO DIVISIONS SAFE. `-1` sends
/// `offset_cid % (gap_within_inner_repeat_ + 1)` (`:222`) to a zero divisor outright, and takes
/// `(gap_within_inner_repeat_ + 1) * repeat_factor_inner_` (`:157`) to zero with it, which is the
/// divisor at `:213`. IBM checks neither; the producer passes a gap count
/// (`dsm/workOptimizer/baseOptimizer/workdivopt.cpp:2065`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GapWithinInnerRepeat(i32);

impl GapWithinInnerRepeat {
    /// No gap after a valid core — the declared initialiser (`:395`) and what the producer passes
    /// whenever `innerGap` is zero.
    pub const NONE: Self = Self(0);

    /// A gap core count, or absent for the negative that would divide by zero.
    pub fn new(gap: i32) -> Option<Self> {
        (gap >= 0).then_some(Self(gap))
    }

    /// The stored count.
    pub const fn get(self) -> i32 {
        self.0
    }
}

/// Cores sharing one work slice, IBM's `NumInnerSameWkSl` (`:398-400`).
///
/// ⛔ CONSTRUCTION REFUSES A NON-POSITIVE, so IBM's `DT_CHECK(repeat_factor_inner_ > 0)` (`:133`)
/// is unreachable and the `/` at `:213` cannot divide by zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RepeatFactorInner(i32);

impl RepeatFactorInner {
    /// One core per work slice.
    pub const ONE: Self = Self(1);

    /// A repeat factor, or absent for the non-positive IBM aborts on.
    pub fn new(factor: i32) -> Option<Self> {
        (factor > 0).then_some(Self(factor))
    }

    /// The stored factor.
    pub const fn get(self) -> i32 {
        self.0
    }
}

/// Gap cores after every run of cores sharing a work slice, IBM's `GapAfterInnerSameWkSl`
/// (`:401-405`).
///
/// ⛔ CONSTRUCTION REFUSES A NEGATIVE: it is summed into the `/` divisor at `:213` (`:157-158`),
/// where a negative can cancel `(gap_within + 1) * repeat_factor_inner` to zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GapAfterInnerRepeat(i32);

impl GapAfterInnerRepeat {
    /// No gap after a shared work slice — the declared initialiser (`:401`), and all the producer
    /// ever passes (`dsm/workOptimizer/baseOptimizer/workdivopt.cpp:2066`).
    pub const NONE: Self = Self(0);

    /// A gap core count, or absent for a negative.
    pub fn new(gap: i32) -> Option<Self> {
        (gap >= 0).then_some(Self(gap))
    }

    /// The stored count.
    pub const fn get(self) -> i32 {
        self.0
    }
}

/// How many times the full set of work slices repeats across the gang, IBM's `RepeatFullWkSl`
/// (`:414-415`).
///
/// ⛔ CONSTRUCTION REFUSES A NON-POSITIVE, so IBM's `DT_CHECK(outer_repeat_factor_ > 0)` (`:132`)
/// is unreachable — including from `build(const WkSplitParam&)`, which is where it fires.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OuterRepeatFactor(i32);

impl OuterRepeatFactor {
    /// The slices are laid down once.
    pub const ONE: Self = Self(1);

    /// A repeat factor, or absent for the non-positive IBM aborts on.
    pub fn new(factor: i32) -> Option<Self> {
        (factor > 0).then_some(Self(factor))
    }

    /// The stored factor.
    pub const fn get(self) -> i32 {
        self.0
    }
}

/// The step between consecutive windows, `dimToStride_` at the producer
/// (`dsm/workOptimizer/baseOptimizer/workdivopt.cpp:2088`).
///
/// ⛔ CONSTRUCTION REFUSES A NON-POSITIVE. `start_vcoord * stride_` (`:450`) turns a negative into
/// negative coordinates, which `getCoordVec` then reads as its gap sentinel (`:253`), and a zero
/// collapses every core onto element 0. IBM leaves the declared `-1` (`:23`) live whenever
/// `isStrWinPad_` is false; here that state is [`None`] instead.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Stride(i32);

impl Stride {
    /// A stride, or absent for the non-positive that would produce negative or collapsed
    /// coordinates.
    pub fn new(stride: i32) -> Option<Self> {
        (stride > 0).then_some(Self(stride))
    }

    /// The stored stride.
    pub const fn get(self) -> i32 {
        self.0
    }
}

/// How many elements one window covers, `dimToWindowSize_` at the producer
/// (`dsm/workOptimizer/baseOptimizer/workdivopt.cpp:2089`).
///
/// ⛔ CONSTRUCTION REFUSES A NON-POSITIVE: it is the whole of a one-element slice's expanded size
/// (`:243-244`), so a zero gives a core no work while `getSliceId` still claims it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Window(i32);

impl Window {
    /// A window size, or absent for a non-positive.
    pub fn new(window: i32) -> Option<Self> {
        (window > 0).then_some(Self(window))
    }

    /// The stored size.
    pub const fn get(self) -> i32 {
        self.0
    }
}

/// Garbage padding the windowed op reads past its last window
/// (`dsm/workOptimizer/baseOptimizer/workdivopt.cpp:2090-2095`).
///
/// ⛔ CONSTRUCTION REFUSES A NEGATIVE, which is the producer's own clamp: it writes `0` and only
/// then overwrites with `garbagePadding` if that is positive.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ExtraBack(i32);

impl ExtraBack {
    /// Nothing read past the last window — the declared initialiser (`:25`).
    pub const NONE: Self = Self(0);

    /// An extra-padding count, or absent for a negative.
    pub fn new(extra_back: i32) -> Option<Self> {
        (extra_back >= 0).then_some(Self(extra_back))
    }

    /// The stored count.
    pub const fn get(self) -> i32 {
        self.0
    }
}

/// A strided, windowed, padded dimension's expansion parameters — IBM's nested `StrWinPad`
/// (`:21-26`).
///
/// ⛔ NO `Default`, DELIBERATELY: IBM's `StrWinPad swp;` is the OFF state, and the OFF state here is
/// [`None`] in [`WkSplitParam::swp_info`], not a value of this type. A `Default` would spell it
/// `stride_ = -1`, which [`Stride`] refuses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StrWinPad {
    /// Field: e049_WkSplitParam.stride_
    ///
    /// The step between consecutive windows (`:23`).
    pub stride: Stride,
    /// Field: e049_WkSplitParam.window_
    ///
    /// How many elements one window covers (`:24`).
    pub window: Window,
    /// Field: e049_WkSplitParam.extra_back_
    ///
    /// Garbage padding past the last window (`:25`).
    pub extra_back: ExtraBack,
}

/// Replaces: e049_WkSplitParam
///
/// One folded dimension's work division across a core gang (`:19-456`).
///
/// ⛔ `isBuilt_` IS THE TYPE'S EXISTENCE, SO THERE IS NO `Default`. IBM's `build(const
/// WkSplitParam&)` sets `isBuilt_ = true` and then checks the fields it copied (`:123-124`), so
/// merging a default-constructed source aborts on `DT_CHECK(outer_repeat_factor_ > 0)` — reachable
/// from the fold merge sites (`util/foldManager/foldInfrastructure.h:1071`, `:2515`). A holder keeps
/// an `Option<WkSplitParam>`; the unbuilt state is that `None` and the abort is inexpressible.
/// See [`unbuilt_meta_data`](Self::unbuilt_meta_data) for its one surviving observable effect.
///
/// ⭐ EVERY CONSTRUCTOR ARGUMENT HAS A DISTINCT TYPE, so none of IBM's eleven adjacent `int32_t`s
/// can be transposed without an `E0308` — the two vendor fixtures alone pass eleven positional
/// integers each (`util/foldManager/test/test_fold_infrastructure.cpp:161-165`, `:222-226`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WkSplitParam {
    /// Field: e049_WkSplitParam.wk_ss_
    ///
    /// Work done by each steady-state core (`:382`).
    wk_ss: WkSs,
    /// Field: e049_WkSplitParam.wk_epilogue_
    ///
    /// Work done by each epilogue core (`:383`).
    wk_epilogue: WkEpilogue,
    /// Field: e049_WkSplitParam.max_cores_
    ///
    /// The gang's core count (`:386`).
    max_cores: MaxCores,
    /// Field: e049_WkSplitParam.start_cid_offset_
    ///
    /// The core index work starts at (`:387`); always `0` at the sole producer
    /// (`dsm/workOptimizer/baseOptimizer/workdivopt.cpp:1979`). ⛔ THE ONE FIELD WIDER THAN IBM'S
    /// DECLARATION — see [`Cid`].
    start_cid_offset: Cid,
    /// Field: e049_WkSplitParam.num_ss_slices_
    ///
    /// How many cores get steady-state work (`:391`).
    num_ss_slices: NumSsSlices,
    /// Field: e049_WkSplitParam.num_epilogue_slices_
    ///
    /// How many cores get epilogue work (`:392`).
    num_epilogue_slices: NumEpilogueSlices,
    /// Field: e049_WkSplitParam.gap_within_inner_repeat_
    ///
    /// Gap cores after each valid core assignment (`:395-396`).
    gap_within_inner_repeat: GapWithinInnerRepeat,
    /// Field: e049_WkSplitParam.repeat_factor_inner_
    ///
    /// Cores sharing one work slice (`:398-400`).
    repeat_factor_inner: RepeatFactorInner,
    /// Field: e049_WkSplitParam.gap_after_inner_repeat_
    ///
    /// Gap cores after each run of sharing cores (`:401`).
    gap_after_inner_repeat: GapAfterInnerRepeat,
    /// Field: e049_WkSplitParam.gap_after_all_slices_
    ///
    /// Gap cores after all work slices are passed (`:407`).
    gap_after_all_slices: GapAfterAllSlices,
    /// Field: e049_WkSplitParam.outer_repeat_factor_
    ///
    /// How many times the full set of work slices repeats (`:414`).
    outer_repeat_factor: OuterRepeatFactor,
    /// Field: e049_WkSplitParam.real_coordinates_
    ///
    /// The dimension's valid element ranges, in order, when it is not densely covered (`:418`,
    /// filled at `dsm/workOptimizer/baseOptimizer/workdivopt.cpp:2069-2074`). Empty means the
    /// virtual space IS the real one.
    ///
    /// ⛔ [`RealCoordRange`], AT IBM'S STORED `int32_t` WIDTH, AND NOT THE `i64` [`CoordRange`] THIS
    /// FIELD HELD BEFORE THIS REVIEW — that is where the widening account was wrong and where the
    /// overflow was.
    real_coordinates: Vec<RealCoordRange>,
    /// Field: e049_WkSplitParam.swp_info_
    ///
    /// Field: e049_WkSplitParam.isStrWinPad_
    ///
    /// The strided-window expansion, absent when there is none (`:420`).
    ///
    /// ⛔ TWO ANCHORS ON ONE FIELD BECAUSE `isStrWinPad_` (`:22`) IS THIS `Option`'S DISCRIMINANT.
    /// The collapse is lossless against the only producer, which writes all three payload fields
    /// inside the one branch that sets the flag and leaves all four declared
    /// (`dsm/workOptimizer/baseOptimizer/workdivopt.cpp:2085-2096`).
    /// ⚠️ DELIBERATE DIVERGENCE, AND IT IS IN `operator==` AND `printMetaData`: IBM compares and
    /// prints `stride_`, `window_` and `extra_back_` even when the flag is false (`:351-354`,
    /// `:316-319`), so a param carrying a live payload under a false flag is unequal to one
    /// carrying the declared `-1`s. Only `perfdsc`'s JSON importer can build that, reading the four
    /// fields independently (`perfdsc/perfDscImportHelper.h:134-137`); no ported path can, and
    /// [`print_meta_data`](Self::print_meta_data) renders `None` as IBM's declared initialisers.
    swp_info: Option<StrWinPad>,
}

impl WkSplitParam {
    /// The field names `printMetaData` emits, in order (`:296-314`).
    const META_DATA_FIELDS: [&'static str; 12] = [
        "isBuilt_",
        "wk_ss_",
        "wk_epilogue_",
        "max_cores_",
        "start_cid_offset_",
        "num_ss_slices_",
        "num_epilogue_slices_",
        "gap_within_inner_repeat_",
        "repeat_factor_inner_",
        "gap_after_inner_repeat_",
        "gap_after_all_slices_",
        "outer_repeat_factor_",
    ];

    /// IBM's thirteen-argument `build` and the constructor that delegates to it (`:50-61`,
    /// `:85-107`), with `checkLegality` (`:131-136`) as the refusal.
    ///
    /// ⛔ THE THIRD `DT_CHECK` IS THE ONLY ONE OF IBM'S LEFT HERE: `RepeatFactorInner` and
    /// `OuterRepeatFactor` refuse a non-positive at their own construction, so `:132-133` cannot be
    /// reached. This one is cross-field — more cores claimed than the gang has — so it belongs to
    /// the whole value.
    ///
    /// ⛔ AND A SECOND REFUSAL IBM DOES NOT HAVE, ADDED BY THIS REVIEW: a param whose own span or
    /// coordinate products would leave `i64`. Two of those were reachable from safe code with no
    /// field outside its own `int32_t` — `gap_within_inner_repeat = repeat_factor_inner =
    /// num_ss_slices = INT32_MAX` aborted [`full_inner_length`](Self::full_inner_length) with
    /// `attempt to multiply with overflow`, and `wk_ss = num_ss_slices = INT32_MAX` under a stride
    /// of three aborted `coord`. Refusing the value is the only answer that is neither an abort nor
    /// a fabricated span or coordinate, and it costs nothing real: the bound admits `INT32_MAX - 1`
    /// unit slices on a gang of `INT32_MAX`.
    #[expect(
        clippy::too_many_arguments,
        reason = "IBM's own thirteen-argument build (`:85-91`); a params struct would be a type the \
                  authority does not name, which Rule 2 of the brief forbids"
    )]
    pub fn new(
        wk_ss: WkSs,
        wk_epilogue: WkEpilogue,
        max_cores: MaxCores,
        start_cid_offset: Cid,
        num_ss_slices: NumSsSlices,
        num_epilogue_slices: NumEpilogueSlices,
        gap_within_inner_repeat: GapWithinInnerRepeat,
        repeat_factor_inner: RepeatFactorInner,
        gap_after_inner_repeat: GapAfterInnerRepeat,
        gap_after_all_slices: GapAfterAllSlices,
        outer_repeat_factor: OuterRepeatFactor,
        real_coordinates: Vec<RealCoordRange>,
        swp_info: Option<StrWinPad>,
    ) -> Option<Self> {
        // ⛔ BOTH SLICE COUNTS WIDEN BEFORE THE ADD. Summed in `i32` first, this guard aborted on
        // the very input it exists to refuse — `INT32_MAX` steady-state slices plus one epilogue
        // slice — while the authority wraps `getFullInnerLength` to `-2147483648`, passes
        // `checkLegality` and builds the illegal param anyway.
        let claimed = (i64::from(num_ss_slices.0) + i64::from(num_epilogue_slices.0))
            * i64::from(outer_repeat_factor.get());
        let candidate = Self {
            wk_ss,
            wk_epilogue,
            max_cores,
            start_cid_offset,
            num_ss_slices,
            num_epilogue_slices,
            gap_within_inner_repeat,
            repeat_factor_inner,
            gap_after_inner_repeat,
            gap_after_all_slices,
            outer_repeat_factor,
            real_coordinates,
            swp_info,
        };
        (claimed <= i64::from(max_cores.0) && candidate.products_fit()).then_some(candidate)
    }

    /// Whether every product this type's own methods form fits `i64`, taken in `i128` over absolute
    /// magnitudes so ONE bound covers both arms of each span and both branches of `coord`
    /// (`:438-443`).
    ///
    /// ⛔ CONSERVATIVE ON PURPOSE, AND THE SLACK IS WHERE THE FIRST ATTEMPT WAS WRONG: bounding the
    /// coordinate by `|num_ss_slices_ + num_epilogue_slices_|` under-counts, because a NEGATIVE
    /// epilogue count keeps the sum small while `num_ss_slices_ * wk_ss_` (`:442`) stays at its
    /// maximum. The sum of magnitudes, doubled, covers the epilogue branch's second term as well. It
    /// may therefore refuse a param whose own worst-case coordinate would in fact have fitted; that
    /// costs an element count and a work size both within a factor of two of `INT32_MAX`.
    fn products_fit(&self) -> bool {
        let span = (i128::from(self.gap_within_inner_repeat.get()) + 1)
            * i128::from(self.repeat_factor_inner.get());
        let span_with_gap = span + i128::from(self.gap_after_inner_repeat.get());
        let slices = i128::from(self.num_ss_slices.0) + i128::from(self.num_epilogue_slices.0);
        let full = span_with_gap * slices;
        let full_with_gap = full + i128::from(self.gap_after_all_slices.0);

        let wk = i128::from(self.wk_ss.0).abs() + i128::from(self.wk_epilogue.0).abs();
        let slices_abs =
            i128::from(self.num_ss_slices.0).abs() + i128::from(self.num_epilogue_slices.0).abs();
        let start = 2 * slices_abs * wk;
        let end = start + wk + 1;
        let (stride, window, extra_back) = match self.swp_info {
            Some(swp) => (
                i128::from(swp.stride.get()),
                i128::from(swp.window.get()),
                i128::from(swp.extra_back.get()),
            ),
            None => (1, 0, 0),
        };
        // The strided expansion scales the start and then derives the end from it (`:448-451`).
        let vsize = (start + end) * stride + window + extra_back;
        let coord = start * stride + vsize;

        let representable = i128::from(i64::MIN)..=i128::from(i64::MAX);
        [span, span_with_gap, full, full_with_gap, coord]
            .iter()
            .all(|value| representable.contains(value))
    }

    /// IBM's copy-assigning `build(const WkSplitParam&)` (`:109-125`), reached from
    /// `WkSplitFoldFunction_Leaf::insertWkSplitParam` (`util/foldManager/foldInfrastructure.h
    /// :859-861`). It overwrites all thirteen fields and re-checks legality, so it is a whole-value
    /// assignment; the source is already legal, so the re-check cannot fail.
    pub fn overwrite_with(&mut self, source: &Self) {
        *self = source.clone();
    }

    /// IBM's `adjustCID` — wraps a negative offset core index back into the gang (`:145-147`).
    ///
    /// ⛔ SATURATING, BECAUSE [`Cid`] IS WIDER THAN THE GANG. `Cid(i64::MIN)` against a negative
    /// `max_cores_` — a param `checkLegality` accepts, measured (`:131-136`) — aborted this addition
    /// with `attempt to add with overflow`. A cid whose true adjustment leaves `i64` is still far
    /// below every span, so the saturated value answers the same [`None`].
    pub const fn adjust_cid(&self, cid: Cid) -> Cid {
        if cid.0 < 0 {
            // `i64::from` is not const-callable yet (rust-lang/rust#143874); the widening is
            // lossless either way.
            Cid(cid.0.saturating_add(self.max_cores.0 as i64))
        } else {
            cid
        }
    }

    /// Cores spanned by one work slice, `(gap_within + 1) * repeat_factor_inner` plus the trailing
    /// gap when asked (`:155-161`).
    ///
    /// ⭐ ALWAYS AT LEAST ONE, FOR EVERY INPUT THE TYPE ACCEPTS, which is what makes both arms safe
    /// divisors in [`slice_id`](Self::slice_id). [`GapWithinInnerRepeat`] refuses a negative,
    /// [`RepeatFactorInner`] a non-positive and [`GapAfterInnerRepeat`] a negative — but the three
    /// refusals bound only the FACTORS, and claiming they bounded the RESULT was this file's own
    /// error: the product has to be taken in [`CoreSpan`]'s `i64` too, since in `int32_t` it wraps to
    /// 0 for `INT32_MAX` and 2 and this function aborted on the way there. Widened, `WithGap::No`
    /// lies in `[1, 2^62]` and `WithGap::Yes` in `[1, 2^62 + 2^31]`.
    pub const fn single_slice_inner_length(&self, with_gap: WithGap) -> CoreSpan {
        // `i64::from` is not const-callable yet (rust-lang/rust#143874); each widening is lossless,
        // and each is taken BEFORE its operator rather than after.
        let shared =
            (self.gap_within_inner_repeat.get() as i64 + 1) * self.repeat_factor_inner.get() as i64;
        match with_gap {
            WithGap::Yes => CoreSpan(shared + self.gap_after_inner_repeat.get() as i64),
            WithGap::No => CoreSpan(shared),
        }
    }

    /// Cores spanned by one full set of work slices (`:169-177`).
    ///
    /// ⛔ `WithGap::No` DROPS ONLY `gap_after_all_slices_`. IBM's `false` arm still calls
    /// `getSingleSliceInnerLength()` with ITS default `true` (`:175`), so the per-slice trailing gap
    /// is in both arms. That is intended, not a slip: `getSliceId` needs exactly "everything but the
    /// gap after all slices" at `:207-209`.
    ///
    /// ⛔ THIS PRODUCT WAS THE ONE THAT COULD STILL LEAVE `i64`, AND NAMING IT WAS NOT ENOUGH. The
    /// account it replaces said reaching it "needs two core counts thirty-one bits past anything
    /// `seidGangs.at(seGangId).size()` produces"; measured, it needs no field outside its own
    /// `int32_t` at all — `gap_within_inner_repeat = repeat_factor_inner = num_ss_slices = INT32_MAX`
    /// passes every guard this file had and aborted here with `attempt to multiply with overflow`.
    /// [`WkSplitParam::new`] refuses such a param, so this is total. [`slice_id`](Self::slice_id)
    /// inherits this product and adds none of its own — in particular not IBM's further `span *
    /// outer` (`:200`).
    pub const fn full_inner_length(&self, with_gap: WithGap) -> CoreSpan {
        // `i64::from` is not const-callable yet (rust-lang/rust#143874); each widening is lossless.
        let slices = self.num_ss_slices.0 as i64 + self.num_epilogue_slices.0 as i64;
        let full = self.single_slice_inner_length(WithGap::Yes).0 * slices;
        match with_gap {
            WithGap::Yes => CoreSpan(full + self.gap_after_all_slices.0 as i64),
            WithGap::No => CoreSpan(full),
        }
    }

    /// IBM's `getOuterRepeatFactor` (`:179`).
    pub const fn outer_repeat_factor(&self) -> OuterRepeatFactor {
        self.outer_repeat_factor
    }

    /// IBM's `getNumSSslices` (`:181`).
    pub const fn num_ss_slices(&self) -> NumSsSlices {
        self.num_ss_slices
    }

    /// IBM's `getNumElSlices` (`:183`).
    pub const fn num_epilogue_slices(&self) -> NumEpilogueSlices {
        self.num_epilogue_slices
    }

    /// IBM's `getWkSs` (`:185`).
    pub const fn wk_ss(&self) -> WkSs {
        self.wk_ss
    }

    /// IBM's `updateWkSs` (`:186`). Every caller divides or scales the work in place, and all of
    /// them are in out-of-scope `dsm/` (`dsm/workOptimizer/baseOptimizer/dwsrsAct2.cpp:94`,
    /// `dyn_wkset_opt.cpp:11343`, `dyn_wkset_opt_act3.cpp:280`).
    pub fn update_wk_ss(&mut self, new_wk_ss: WkSs) {
        self.wk_ss = new_wk_ss;
    }

    /// IBM's `getWkEl` (`:187`).
    pub const fn wk_epilogue(&self) -> WkEpilogue {
        self.wk_epilogue
    }

    /// IBM's `updateWkEl` (`:188`), the epilogue counterpart of
    /// [`update_wk_ss`](Self::update_wk_ss).
    pub fn update_wk_epilogue(&mut self, new_wk_epilogue: WkEpilogue) {
        self.wk_epilogue = new_wk_epilogue;
    }

    /// Which work slice a core runs, or [`None`] for a gap core — IBM's four `return -1`s
    /// (`:196-227`).
    ///
    /// ⛔ DELIBERATE DIVERGENCE, AND THE CASE THAT JUSTIFIES IT IS AN ALIAS, NOT A NEGATIVE. IBM
    /// tests only `offset_cid >= span * outer` (`:200`), so a NEGATIVE post-adjust offset falls
    /// through to `offset_cid % full_wksl_size_with_after_gaps_` (`:204`). Measured against the
    /// header, gang of eight, three steady-state slices of ten, no gaps, span three: `cid = -11`
    /// adjusts to `-3`, C++'s `-3 % 3` is `0`, and `getSliceId` answers **slice 0** — a core below
    /// the gang reported as doing the first slice's full work, `getSize` 10, and with a
    /// real-coordinate table even mapped to real elements `(100, 109)`. That answer is
    /// indistinguishable from core 0's own. `cid = -10` gives the cruder failure of the same line:
    /// slice `-2`, past IBM's `-1` sentinel, so `getSize` reads it as steady-state work (`:241`) and
    /// `getCoord` returns `(-20, -11)` (`:438`). Range-testing the whole of `0..span * outer` answers
    /// [`None`] for both, and the negatives that wrap back INSIDE the gang still agree with IBM cid
    /// for cid. Unreachable from the sole producer, which passes `start_cid_offset = 0`
    /// (`dsm/workOptimizer/baseOptimizer/workdivopt.cpp:1979`).
    ///
    /// ⛔ THAT SAME `:204` DIVIDES BY ZERO IN THE AUTHORITY, which is the second thing the range test
    /// covers: with a negative `gap_after_all_slices_` cancelling the span, `getSliceId(-9)` is
    /// `runtime error: division by zero` under UBSan — see [`GapAfterAllSlices`]. The
    /// `span_with_gaps < 1` arm below is where that is answered, and it agrees with IBM wherever IBM
    /// is defined.
    ///
    /// ⛔ THE OFFSET SUBTRACTION SATURATES (see [`Cid`]): `Cid(i64::MAX)` against `start_cid_offset =
    /// Cid(-1)` aborted it before this review. Saturating is exact — a true offset outside `i64` is
    /// outside every representable span, and so is `i64::MAX`.
    ///
    /// ⭐ THIS FUNCTION ADDS NO PRODUCT OF ITS OWN, which is why the range test is `offset / span >=
    /// outer` rather than IBM's `offset >= span * outer`. The two are equivalent for `span >= 1` and
    /// `offset >= 0`, and the division form cannot leave `i64`.
    pub fn slice_id(&self, cid: Cid) -> Option<WkSliceId> {
        let offset = self
            .adjust_cid(Cid(cid.0.saturating_sub(self.start_cid_offset.0)))
            .0;

        // The gap cores that come at the end (`:200-201`), widened to also exclude the negative
        // IBM lets through. `span_with_gaps >= 1` is what every divisor below rests on, so it is
        // tested here instead of assumed: a span of zero or less covers no core at all.
        let span_with_gaps = self.full_inner_length(WithGap::Yes).0;
        if offset < 0
            || span_with_gaps < 1
            || offset / span_with_gaps >= i64::from(self.outer_repeat_factor.get())
        {
            return None;
        }

        // Fold cids using the outer repeat factor (`:204`).
        let offset = offset % span_with_gaps;

        // The gap that comes after all work slices are passed (`:207-209`).
        if offset >= self.full_inner_length(WithGap::No).0 {
            return None;
        }

        // The work slice, ignoring inner gaps (`:212-215`). Both spans are `>= 1` by the guarded
        // newtypes AND [`CoreSpan`]'s width, never by a check here.
        let single = self.single_slice_inner_length(WithGap::Yes).0;
        let slid = offset / single;
        let offset = offset % single;

        // The gap that comes after each work slice (`:218-220`).
        if offset >= self.single_slice_inner_length(WithGap::No).0 {
            return None;
        }

        // The gap cores after each valid core assignment (`:222-224`). The `+ 1` is taken after the
        // widening; in `i32` it overflowed at `GapWithinInnerRepeat::new(i32::MAX)`.
        if offset % (i64::from(self.gap_within_inner_repeat.get()) + 1) >= 1 {
            return None;
        }

        Some(WkSliceId(slid))
    }

    /// The work one core does, zero for a gap core — IBM's `getSize` (`:235-248`). Its
    /// `DT_CHECK(isBuilt_)` (`:236`) is discharged by this value existing.
    pub fn size(&self, cid: Cid) -> WkSize {
        let Some(slice_id) = self.slice_id(cid) else {
            return WkSize(0);
        };
        let vsize = if slice_id.0 < i64::from(self.num_ss_slices.0) {
            i64::from(self.wk_ss.0)
        } else {
            i64::from(self.wk_epilogue.0)
        };
        WkSize(match self.swp_info {
            Some(swp) => {
                (vsize - 1) * i64::from(swp.stride.get())
                    + i64::from(swp.window.get())
                    + i64::from(swp.extra_back.get())
            }
            None => vsize,
        })
    }

    /// The virtual element range one core covers, [`None`] for a gap core — IBM's private `getCoord`
    /// (`:428-455`), private here for the same reason: [`coord_vec`](Self::coord_vec) is the reader.
    ///
    /// ⛔ THE STRIDED START IS SCALED BEFORE THE END IS DERIVED FROM IT (`:450-451`), so the window
    /// lands at `start * stride`, not at `start`, and consecutive cores' ranges overlap by
    /// `window - stride`.
    /// ⛔ ITS TWO PRODUCTS ARE TOTAL BY CONSTRUCTION NOW, NOT BY BEING FAR AWAY. The account this
    /// replaces said `start * stride` (`:450`) "needs a work size or a stride within a factor of two
    /// of `INT32_MAX` — element counts, thirty-one bits past where the authority's own `int32_t` has
    /// gone undefined"; measured, `wk_ss = num_ss_slices = INT32_MAX` at a stride of three aborted
    /// here with `attempt to multiply with overflow`, every field inside its own `int32_t`.
    /// [`WkSplitParam::new`] refuses such a param, because a saturated coordinate would have been a
    /// fabricated one.
    fn coord(&self, cid: Cid) -> Option<VCoordRange> {
        let slice_id = self.slice_id(cid)?;
        let num_ss = i64::from(self.num_ss_slices.0);
        let (start, end) = if slice_id.0 < num_ss {
            let start = slice_id.0 * i64::from(self.wk_ss.0);
            (start, start + i64::from(self.wk_ss.0) - 1)
        } else {
            let start = num_ss * i64::from(self.wk_ss.0)
                + (slice_id.0 - num_ss) * i64::from(self.wk_epilogue.0);
            (start, start + i64::from(self.wk_epilogue.0) - 1)
        };
        let (start, end) = match self.swp_info {
            Some(swp) => {
                let vsize = (end - start) * i64::from(swp.stride.get())
                    + i64::from(swp.window.get())
                    + i64::from(swp.extra_back.get());
                let start = start * i64::from(swp.stride.get());
                (start, start + vsize - 1)
            }
            None => (start, end),
        };
        Some(VCoordRange {
            start: VCoord(start),
            end: VCoord(end),
        })
    }

    /// The real element ranges one core covers — IBM's `getCoordVec` (`:250-284`), the one method
    /// the fold infrastructure calls (`util/foldManager/foldInfrastructure.h:838`).
    ///
    /// ⛔ A GAP CORE YIELDS ONE [`CoordRange::GAP`], NOT AN EMPTY VEC (`:253-254`).
    /// ⛔ ONE SLICE CAN SPAN SEVERAL REAL RANGES, so the result is a vector and not one range: a
    /// virtual span that straddles a hole in `real_coordinates_` pushes once per range it overlaps
    /// (`:279`), and one that runs off the end of the table is silently clipped short.
    /// ⛔ AND AN EMPTY VEC IS THE THIRD OUTCOME, REACHABLE AND PREVIOUSLY UNDOCUMENTED. A core with
    /// real work whose virtual span lies entirely PAST the table overlaps no range, so the loop
    /// (`:257-281`) pushes nothing: not a gap, not an absence, but "this core's work has no real
    /// elements". Measured against the header — three slices of ten over a table of fifteen elements
    /// gives core 2 `getSliceId` 2, `getSize` 10 and `getCoordVec` empty.
    /// ⛔ EACH STORED ENTRY IS WIDENED ONE AT A TIME, exactly where IBM widens it, which is the whole
    /// of why [`RealCoordRange`] and [`CoordRange`] are two types.
    pub fn coord_vec(&self, cid: Cid) -> Vec<CoordRange> {
        let Some(v_coord) = self.coord(cid) else {
            return vec![CoordRange::GAP];
        };
        // With no table, or a negative start, IBM pushes the virtual range itself (`:253-254`) —
        // which for a gap core is where its `(-1, -1)` comes from.
        if self.real_coordinates.is_empty() || v_coord.start.0 < 0 {
            return vec![CoordRange::from_virtual(v_coord)];
        }

        let mut coord_vec = Vec::new();
        let mut cum_running_size = 0_i64;
        for stored in &self.real_coordinates {
            let (range_start, range_end) = stored.widen();
            // Where this real range sits in the virtual space (`:258-262`).
            let running_start = cum_running_size;
            let running_end = running_start + (range_end - range_start);
            cum_running_size += range_end - range_start + 1;

            if running_start > v_coord.end.0 || running_end < v_coord.start.0 {
                continue;
            }
            coord_vec.push(CoordRange {
                start: Coord(if v_coord.start.0 <= running_start {
                    range_start
                } else {
                    range_start + (v_coord.start.0 - running_start)
                }),
                end: Coord(if v_coord.end.0 >= running_end {
                    range_end
                } else {
                    range_end - (running_end - v_coord.end.0)
                }),
            });
        }
        coord_vec
    }

    /// IBM's `get_real_coordinates_` (`:363-366`).
    ///
    /// ⚠️ NO CALLER IN THE AUTHORITY TREE: `real_coordinates_` is read only inside `getCoordVec`
    /// (`:257`). Ported because it is part of the class's surface, not because anything wants it.
    pub fn real_coordinates(&self) -> &[RealCoordRange] {
        &self.real_coordinates
    }

    /// IBM's `set_real_coordinates_` (`:372-376`); its `DT_CHECK(isBuilt_)` (`:374`) is discharged
    /// by this value existing. Both callers are in out-of-scope `dsm/`
    /// (`dsm/workOptimizer/baseOptimizer/dwsrsAct2.cpp:148`, `workdivopt.cpp:2728`).
    ///
    /// ⛔ THIS IS THE ONE WAY A FIELD IS REFILLED AFTER [`new`](Self::new) HAS CHECKED THE VALUE, so
    /// its argument type is the whole guard: at [`RealCoordRange`]'s `int32_t` width it cannot carry
    /// an entry whose span leaves `i64`, and nothing here has to re-run `products_fit`, since the
    /// table takes part in no product `new` bounds.
    pub fn set_real_coordinates(&mut self, real_coordinates: Vec<RealCoordRange>) {
        self.real_coordinates = real_coordinates;
    }

    /// The metadata text IBM streams (`:294-330`), returned rather than printed as
    /// [`Constraints::dump`](crate::schedule::metadata::Constraints::dump) is.
    ///
    /// ⛔ `out << bool` PRINTS `1`/`0`, NOT `true`/`false` (`:296`, `:316`), and `ps` is a PREFIX ON
    /// EVERY FIELD (`:297` onwards), not an indent written once. An absent
    /// [`swp_info`](Self::swp_info) renders as IBM's declared initialisers (`:22-25`).
    pub fn print_meta_data(&self, ps: &str) -> String {
        let values: [i64; 12] = [
            1,
            i64::from(self.wk_ss.0),
            i64::from(self.wk_epilogue.0),
            i64::from(self.max_cores.0),
            self.start_cid_offset.0,
            i64::from(self.num_ss_slices.0),
            i64::from(self.num_epilogue_slices.0),
            i64::from(self.gap_within_inner_repeat.get()),
            i64::from(self.repeat_factor_inner.get()),
            i64::from(self.gap_after_inner_repeat.get()),
            i64::from(self.gap_after_all_slices.0),
            i64::from(self.outer_repeat_factor.get()),
        ];
        let (is_str_win_pad, stride, window, extra_back) = match self.swp_info {
            Some(swp) => (1, swp.stride.get(), swp.window.get(), swp.extra_back.get()),
            None => (0, -1, -1, 0),
        };
        let mut out = String::new();
        for (name, value) in Self::META_DATA_FIELDS.iter().zip(values) {
            out.push_str(ps);
            out.push_str(&format!("\"{name}\" : {value},"));
        }
        out.push_str(ps);
        out.push_str(&format!(
            "\"swp_info_\" : {{ \"isStrWinPad_\" : {is_str_win_pad}, \"stride_\" : {stride}, \
             \"window_\" : {window}, \"extra_back_\" : {extra_back} }},"
        ));
        out.push_str(ps);
        out.push_str("\"real_coordinates_\" : [ ");
        let ranges: Vec<String> = self
            .real_coordinates
            .iter()
            .map(|range| format!("[{}, {}]", range.start.0, range.end.0))
            .collect();
        out.push_str(&ranges.join(", "));
        out.push_str(" ]");
        out
    }

    /// Field: e049_WkSplitParam.isBuilt_
    ///
    /// What `printMetaData` prints for a param that was never built — `isBuilt_ = false` (`:379`)
    /// with every other declared initialiser (`:382-420`).
    ///
    /// ⛔ THIS IS THE ONLY OBSERVABLE EFFECT `isBuilt_` HAS LEFT, and it is why the field gets an
    /// anchor at all: `WkSplitFoldFunction_Leaf::printMetaData` delegates unconditionally, with no
    /// `DT_CHECK` (`util/foldManager/foldInfrastructure.h:880-882`), so an unbuilt leaf does print
    /// this. A holder whose `Option<WkSplitParam>` is `None` prints it here rather than leaving a
    /// later port to invent the text. Its two other readers are checks this type discharges
    /// (`:236`, `:374`, plus `foldInfrastructure.h:849`).
    pub fn unbuilt_meta_data(ps: &str) -> String {
        let mut out = String::new();
        for name in Self::META_DATA_FIELDS {
            out.push_str(ps);
            out.push_str(&format!("\"{name}\" : 0,"));
        }
        out.push_str(ps);
        out.push_str(
            "\"swp_info_\" : { \"isStrWinPad_\" : 0, \"stride_\" : -1, \"window_\" : -1, \
             \"extra_back_\" : 0 },",
        );
        out.push_str(ps);
        out.push_str("\"real_coordinates_\" : [  ]");
        out
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    /// The vendor's own `build` argument order (`:85-91`), so a fixture line below can be diffed
    /// against the C++ call it was transcribed from. ⛔ RAW SCALARS ONLY HERE, AND ONLY HERE: the
    /// point of this helper is that the transcription is auditable, and the sixteen asserted golden
    /// values are what catch a transposed argument.
    #[expect(
        clippy::too_many_arguments,
        reason = "mirrors the vendor's own thirteen-argument build call for auditability"
    )]
    fn built(
        wk_ss: i32,
        wk_epilogue: i32,
        max_cores: i32,
        start_cid_offset: i64,
        num_ss_slices: i32,
        num_epilogue_slices: i32,
        gap_within_inner_repeat: i32,
        repeat_factor_inner: i32,
        gap_after_inner_repeat: i32,
        gap_after_all_slices: i32,
        outer_repeat_factor: i32,
        real_coordinates: Vec<RealCoordRange>,
        swp_info: Option<StrWinPad>,
    ) -> WkSplitParam {
        WkSplitParam::new(
            WkSs(wk_ss),
            WkEpilogue(wk_epilogue),
            MaxCores(max_cores),
            Cid(start_cid_offset),
            NumSsSlices(num_ss_slices),
            NumEpilogueSlices(num_epilogue_slices),
            GapWithinInnerRepeat::new(gap_within_inner_repeat).unwrap(),
            RepeatFactorInner::new(repeat_factor_inner).unwrap(),
            GapAfterInnerRepeat::new(gap_after_inner_repeat).unwrap(),
            GapAfterAllSlices(gap_after_all_slices),
            OuterRepeatFactor::new(outer_repeat_factor).unwrap(),
            real_coordinates,
            swp_info,
        )
        .unwrap()
    }

    fn range(start: i64, end: i64) -> CoordRange {
        CoordRange {
            start: Coord(start),
            end: Coord(end),
        }
    }

    /// One STORED entry, at IBM's `int32_t` width (`:418`) — a different type from what `getCoordVec`
    /// hands back, which is the whole point of [`RealCoordRange`].
    fn table_range(start: i32, end: i32) -> RealCoordRange {
        RealCoordRange {
            start: RealCoord(start),
            end: RealCoord(end),
        }
    }

    fn sizes(param: &WkSplitParam, cores: i64) -> Vec<i64> {
        (0..cores).map(|cid| param.size(Cid(cid)).0).collect()
    }

    fn coords(param: &WkSplitParam, cores: i64) -> Vec<Vec<CoordRange>> {
        (0..cores).map(|cid| param.coord_vec(Cid(cid))).collect()
    }

    /// The vendor's `constructor_test_wksplit` fixture, transcribed whole
    /// (`util/foldManager/test/test_fold_infrastructure.cpp:157-189`): eight cores, two dimensions,
    /// no gaps. `ij` shares each slice across four cores; `out` gives each of four slices one core
    /// and repeats the set twice.
    #[test]
    fn the_vendors_eight_core_fixture_reproduces_both_dimensions() {
        let ij = built(10, 0, 8, 0, 2, 0, 0, 4, 0, 0, 1, Vec::new(), None);
        let out = built(4, 0, 8, 0, 4, 0, 0, 1, 0, 0, 2, Vec::new(), None);

        assert_eq!(sizes(&ij, 8), vec![10; 8]);
        assert_eq!(sizes(&out, 8), vec![4; 8]);

        let golden_ij: Vec<Vec<CoordRange>> = (0..8)
            .map(|i| vec![range(10 * (i / 4), 9 + 10 * (i / 4))])
            .collect();
        let golden_out: Vec<Vec<CoordRange>> = (0..8)
            .map(|i| vec![range(4 * (i % 4), 3 + 4 * (i % 4))])
            .collect();
        assert_eq!(coords(&ij, 8), golden_ij);
        assert_eq!(coords(&out, 8), golden_out);
    }

    /// The vendor's `constructor_test_wksplit2` fixture, transcribed whole
    /// (`util/foldManager/test/test_fold_infrastructure.cpp:219-268`): sixteen cores, three gap
    /// cores after all slices, the set laid down twice, and — for `out` — a fourth slice of
    /// epilogue work. ⭐ THE GOLDENS PIN `(-1, -1)` ON NINE OF SIXTEEN CORES, which is why a gap
    /// core's coordinate is a value and not an absence, and pin the last two cores as uncovered
    /// even though the gang has room, which is `outer_repeat_factor` times the span, not `max_cores`.
    #[test]
    fn the_vendors_sixteen_core_fixture_pins_the_gap_coordinate() {
        let ij = built(10, 0, 16, 0, 2, 0, 0, 2, 0, 3, 2, Vec::new(), None);
        let out = built(3, 2, 16, 0, 3, 1, 0, 1, 0, 3, 2, Vec::new(), None);

        assert_eq!(
            sizes(&ij, 16),
            vec![10, 10, 10, 10, 0, 0, 0, 10, 10, 10, 10, 0, 0, 0, 0, 0]
        );
        assert_eq!(
            sizes(&out, 16),
            vec![3, 3, 3, 2, 0, 0, 0, 3, 3, 3, 2, 0, 0, 0, 0, 0]
        );

        let gap = vec![CoordRange::GAP];
        assert_eq!(
            coords(&ij, 16),
            vec![
                vec![range(0, 9)],
                vec![range(0, 9)],
                vec![range(10, 19)],
                vec![range(10, 19)],
                gap.clone(),
                gap.clone(),
                gap.clone(),
                vec![range(0, 9)],
                vec![range(0, 9)],
                vec![range(10, 19)],
                vec![range(10, 19)],
                gap.clone(),
                gap.clone(),
                gap.clone(),
                gap.clone(),
                gap.clone(),
            ]
        );
        assert_eq!(
            coords(&out, 16),
            vec![
                vec![range(0, 2)],
                vec![range(3, 5)],
                vec![range(6, 8)],
                vec![range(9, 10)],
                gap.clone(),
                gap.clone(),
                gap.clone(),
                vec![range(0, 2)],
                vec![range(3, 5)],
                vec![range(6, 8)],
                vec![range(9, 10)],
                gap.clone(),
                gap.clone(),
                gap.clone(),
                gap.clone(),
                gap,
            ]
        );
        assert_eq!(ij.slice_id(Cid(14)), None);
        assert_eq!(ij.slice_id(Cid(3)), Some(WkSliceId(1)));
        assert_eq!(out.slice_id(Cid(10)), Some(WkSliceId(3)));
    }

    /// ⛔ NO FIXTURE IN THE AUTHORITY TREE EXERCISES `real_coordinates_` — both vendor tests pass a
    /// default-constructed empty vector (`util/foldManager/test/test_fold_infrastructure.cpp:157`,
    /// `:219`) and so does `fold_standalone.cpp`. This case is derived from the remap itself
    /// (`:256-281`): a table of `100..=109` then `200..=204` is fifteen virtual elements, and four
    /// cores each claiming four of them means core 2 straddles the hole and core 3 runs off the end.
    #[test]
    fn a_real_coordinate_table_splits_one_slice_and_clips_another() {
        let param = built(
            4,
            0,
            8,
            0,
            4,
            0,
            0,
            1,
            0,
            0,
            1,
            vec![table_range(100, 109), table_range(200, 204)],
            None,
        );

        assert_eq!(param.coord_vec(Cid(0)), vec![range(100, 103)]);
        assert_eq!(param.coord_vec(Cid(1)), vec![range(104, 107)]);
        // Virtual 8..=11 straddles the hole: two of its elements are in each real range.
        assert_eq!(
            param.coord_vec(Cid(2)),
            vec![range(108, 109), range(200, 201)]
        );
        // Virtual 12..=15 runs past the table's fifteen elements and is clipped to three.
        assert_eq!(param.coord_vec(Cid(3)), vec![range(202, 204)]);
        // A gap core still gets the sentinel, table or no table.
        assert_eq!(param.coord_vec(Cid(4)), vec![CoordRange::GAP]);
        assert_eq!(param.real_coordinates().len(), 2);
    }

    /// ⛔ NO FIXTURE EXERCISES `swp_info_` EITHER, so this is derived from the expansion (`:242-245`,
    /// `:446-452`): four elements at stride 2 with a window of 3 and one element of back padding
    /// span `(4 - 1) * 2 + 3 + 1 = 10`, and because the START is scaled before the END is derived
    /// from it, consecutive cores overlap by `window - stride`.
    #[test]
    fn a_strided_window_expands_each_slice_and_overlaps_its_neighbour() {
        let swp = StrWinPad {
            stride: Stride::new(2).unwrap(),
            window: Window::new(3).unwrap(),
            extra_back: ExtraBack::new(1).unwrap(),
        };
        let param = built(4, 0, 8, 0, 4, 0, 0, 1, 0, 0, 1, Vec::new(), Some(swp));

        assert_eq!(sizes(&param, 5), vec![10, 10, 10, 10, 0]);
        assert_eq!(param.coord_vec(Cid(0)), vec![range(0, 9)]);
        assert_eq!(param.coord_vec(Cid(1)), vec![range(8, 17)]);
        assert_eq!(param.coord_vec(Cid(2)), vec![range(16, 25)]);
        assert_eq!(param.coord_vec(Cid(3)), vec![range(24, 33)]);
        assert_eq!(param.coord_vec(Cid(4)), vec![CoordRange::GAP]);

        // Without the expansion the same division is dense and four elements wide.
        let dense = built(4, 0, 8, 0, 4, 0, 0, 1, 0, 0, 1, Vec::new(), None);
        assert_eq!(dense.coord_vec(Cid(1)), vec![range(4, 7)]);
        assert_ne!(dense, param);
    }

    /// All three of IBM's `checkLegality` `DT_CHECK`s (`:131-136`) plus the two divisors it never
    /// checks, each refused at construction instead. ⛔ The cross-field one is the only refusal left
    /// on [`WkSplitParam::new`]: four slices laid down twice needs eight cores, and a gang of seven
    /// is what IBM aborts on.
    #[test]
    fn every_legality_check_is_refused_at_construction() {
        assert_eq!(OuterRepeatFactor::new(0), None);
        assert_eq!(OuterRepeatFactor::new(-1), None);
        assert_eq!(RepeatFactorInner::new(0), None);
        assert_eq!(GapWithinInnerRepeat::new(-1), None);
        assert_eq!(GapAfterInnerRepeat::new(-1), None);
        assert_eq!(Stride::new(0), None);
        assert_eq!(Window::new(0), None);
        assert_eq!(ExtraBack::new(-1), None);

        let too_few_cores = WkSplitParam::new(
            WkSs(4),
            WkEpilogue(0),
            MaxCores(7),
            Cid(0),
            NumSsSlices(4),
            NumEpilogueSlices(0),
            GapWithinInnerRepeat::NONE,
            RepeatFactorInner::ONE,
            GapAfterInnerRepeat::NONE,
            GapAfterAllSlices(0),
            OuterRepeatFactor::new(2).unwrap(),
            Vec::new(),
            None,
        );
        assert_eq!(too_few_cores, None);
        assert!(
            WkSplitParam::new(
                WkSs(4),
                WkEpilogue(0),
                MaxCores(8),
                Cid(0),
                NumSsSlices(4),
                NumEpilogueSlices(0),
                GapWithinInnerRepeat::NONE,
                RepeatFactorInner::ONE,
                GapAfterInnerRepeat::NONE,
                GapAfterAllSlices(0),
                OuterRepeatFactor::new(2).unwrap(),
                Vec::new(),
                None,
            )
            .is_some()
        );
    }

    /// `:294-330`. ⛔ THE BOOLS ARE DIGITS AND `ps` PREFIXES EVERY FIELD, and an absent
    /// `swp_info_` prints IBM's declared `-1`s (`:22-25`) — the `-1` is not dropped just because
    /// this port spells the off state [`None`]. The unbuilt text is what a leaf holding no param
    /// prints (`util/foldManager/foldInfrastructure.h:880-882`).
    #[test]
    fn the_metadata_text_prints_bools_as_digits_and_prefixes_every_field() {
        let param = built(
            10,
            0,
            8,
            0,
            2,
            0,
            0,
            4,
            0,
            0,
            1,
            vec![table_range(0, 9)],
            None,
        );

        assert_eq!(
            param.print_meta_data(""),
            "\"isBuilt_\" : 1,\"wk_ss_\" : 10,\"wk_epilogue_\" : 0,\"max_cores_\" : 8,\
             \"start_cid_offset_\" : 0,\"num_ss_slices_\" : 2,\"num_epilogue_slices_\" : 0,\
             \"gap_within_inner_repeat_\" : 0,\"repeat_factor_inner_\" : 4,\
             \"gap_after_inner_repeat_\" : 0,\"gap_after_all_slices_\" : 0,\
             \"outer_repeat_factor_\" : 1,\"swp_info_\" : { \"isStrWinPad_\" : 0, \"stride_\" : -1, \
             \"window_\" : -1, \"extra_back_\" : 0 },\"real_coordinates_\" : [ [0, 9] ]"
        );
        assert!(
            param
                .print_meta_data("\n  ")
                .starts_with("\n  \"isBuilt_\" : 1,\n  \"wk_ss_\"")
        );

        assert!(
            WkSplitParam::unbuilt_meta_data("").starts_with("\"isBuilt_\" : 0,\"wk_ss_\" : 0,")
        );
        assert!(WkSplitParam::unbuilt_meta_data("").ends_with(
            "\"stride_\" : -1, \"window_\" : -1, \"extra_back_\" : 0 },\
                            \"real_coordinates_\" : [  ]"
        ));
    }

    /// `:339-357` — `operator==` compares twelve scalars, the coordinate table and all four
    /// `swp_info_` fields, and does NOT compare `isBuilt_`; the derive matches it field for field
    /// because this type has no `isBuilt_`. ⛔ `overwrite_with` is IBM's `build(const
    /// WkSplitParam&)` (`:109-125`), a whole-value assignment, so equality is the way to see it
    /// landed.
    #[test]
    fn overwriting_copies_every_compared_field() {
        let source = built(
            3,
            2,
            16,
            0,
            3,
            1,
            0,
            1,
            0,
            3,
            2,
            vec![table_range(4, 8)],
            None,
        );
        let mut target = built(10, 0, 8, 0, 2, 0, 0, 4, 0, 0, 1, Vec::new(), None);
        assert_ne!(target, source);

        target.overwrite_with(&source);
        assert_eq!(target, source);
        assert_eq!(target.wk_ss(), WkSs(3));
        assert_eq!(target.wk_epilogue(), WkEpilogue(2));
        assert_eq!(target.num_ss_slices(), NumSsSlices(3));
        assert_eq!(target.num_epilogue_slices(), NumEpilogueSlices(1));
        assert_eq!(
            target.outer_repeat_factor(),
            OuterRepeatFactor::new(2).unwrap()
        );
        assert_eq!(target.real_coordinates(), [table_range(4, 8)]);

        target.update_wk_ss(WkSs(1));
        target.update_wk_epilogue(WkEpilogue(1));
        target.set_real_coordinates(Vec::new());
        assert_eq!(target.wk_ss(), WkSs(1));
        assert_eq!(target.wk_epilogue(), WkEpilogue(1));
        assert!(target.real_coordinates().is_empty());
        assert_ne!(target, source);
    }

    /// `:145-147`, `:155-177`. ⛔ `WithGap::No` ON THE FULL LENGTH DROPS ONLY `gap_after_all_slices_`
    /// — IBM's `false` arm still takes the per-slice trailing gap, because it calls
    /// `getSingleSliceInnerLength()` with its own default `true` (`:175`). Two slices of two sharing
    /// cores with one trailing gap each is `2 * 3 = 6`, not `2 * 2 = 4`.
    #[test]
    fn the_full_length_without_gaps_still_carries_the_per_slice_gap() {
        let param = built(10, 0, 16, 0, 2, 0, 0, 2, 1, 3, 2, Vec::new(), None);

        assert_eq!(param.single_slice_inner_length(WithGap::Yes), CoreSpan(3));
        assert_eq!(param.single_slice_inner_length(WithGap::No), CoreSpan(2));
        assert_eq!(param.full_inner_length(WithGap::Yes), CoreSpan(9));
        assert_eq!(param.full_inner_length(WithGap::No), CoreSpan(6));

        // The core each work slice's trailing gap eats, and the three after all slices.
        assert_eq!(param.slice_id(Cid(1)), Some(WkSliceId(0)));
        assert_eq!(param.slice_id(Cid(2)), None);
        assert_eq!(param.slice_id(Cid(5)), None);
        assert_eq!(param.slice_id(Cid(6)), None);

        // A negative cid wraps into the gang, IBM's own `adjustCID`.
        assert_eq!(param.adjust_cid(Cid(-2)), Cid(14));
        assert_eq!(param.adjust_cid(Cid(3)), Cid(3));
    }

    /// ⛔ THE GUARD ABORTED ON THE INPUT IT EXISTS TO REFUSE. `(num_ss_slices_ +
    /// num_epilogue_slices_)` was summed in `i32` before widening, so `INT32_MAX` steady-state slices
    /// plus one epilogue slice panicked with `attempt to add with overflow` instead of answering
    /// [`None`]. The authority is worse in kind, not better: measured against the header, its
    /// `getFullInnerLength` wraps to `-2147483648`, `checkLegality` (`:131-136`) passes, and the
    /// illegal param is built — after which every core reads as a gap core.
    #[test]
    fn the_legality_guard_refuses_a_slice_sum_past_i32_instead_of_aborting() {
        let unrepresentable = WkSplitParam::new(
            WkSs(1),
            WkEpilogue(1),
            MaxCores(i32::MAX),
            Cid(0),
            NumSsSlices(i32::MAX),
            NumEpilogueSlices(1),
            GapWithinInnerRepeat::NONE,
            RepeatFactorInner::ONE,
            GapAfterInnerRepeat::NONE,
            GapAfterAllSlices(0),
            OuterRepeatFactor::ONE,
            Vec::new(),
            None,
        );
        assert_eq!(unrepresentable, None);

        // One slice fewer is exactly representable and exactly legal: it claims every core.
        let exact = built(
            1,
            1,
            i32::MAX,
            0,
            i32::MAX - 1,
            1,
            0,
            1,
            0,
            0,
            1,
            Vec::new(),
            None,
        );
        assert_eq!(exact.num_ss_slices(), NumSsSlices(i32::MAX - 1));
    }

    /// ⛔ "ALWAYS AT LEAST ONE" HELD ONLY BECAUSE THE FUNCTION ABORTED FIRST. `(gap_within + 1) *
    /// repeat_factor_inner` was an `i32` product, so `gap_within = INT32_MAX, repeat_factor_inner =
    /// 2` — accepted by every newtype here AND by `checkLegality` (`:131-136`) — panicked with
    /// `attempt to add with overflow`. Measured against the header, the authority instead wraps:
    /// `getSingleSliceInnerLength(false)` is **0**, and with `gap_after_inner_repeat_ = 0` so is the
    /// `true` arm, after which the `:200`/`:208` range tests report every core a gap core.
    /// [`CoreSpan`]'s width is the fix, so the extreme is asserted rather than avoided.
    #[test]
    fn one_work_slice_spans_at_least_one_core_at_the_widest_gap() {
        let param = built(
            1,
            0,
            1,
            0,
            1,
            0,
            i32::MAX,
            2,
            i32::MAX,
            0,
            1,
            Vec::new(),
            None,
        );
        let shared = (i64::from(i32::MAX) + 1) * 2;

        assert_eq!(
            param.single_slice_inner_length(WithGap::No),
            CoreSpan(shared)
        );
        assert_eq!(
            param.single_slice_inner_length(WithGap::Yes),
            CoreSpan(shared + i64::from(i32::MAX))
        );
        assert!(param.single_slice_inner_length(WithGap::No).0 >= 1);

        // The one slice still starts at core 0, and the 2^31 - 1 gap cores still follow it.
        assert_eq!(param.slice_id(Cid(0)), Some(WkSliceId(0)));
        assert_eq!(param.size(Cid(0)), WkSize(1));
        assert_eq!(param.slice_id(Cid(1)), None);
        assert_eq!(param.size(Cid(1)), WkSize(0));
    }

    /// ⚠️ THE DIVERGENCE AT [`WkSplitParam::slice_id`], AND THE VALUE IT SUPPRESSES IS A PLAUSIBLE
    /// ONE. Measured against `util/foldManager/wkDivisionParams.h:196-227`: a gang of eight with
    /// three steady-state slices of ten and no gaps has a span of three, so `cid = -11` adjusts to
    /// `-3`, C++'s `-3 % 3` is `0`, and IBM answers slice 0 — `getSize` 10 and, with the table below,
    /// real elements `(100, 109)`, byte for byte what core 0 answers. `cid = -10` answers slice `-2`,
    /// past the `-1` sentinel, so `getSize` returns 10 there too and `getCoord` gives `(-20, -11)`.
    #[test]
    fn a_core_below_the_gang_does_not_alias_onto_a_real_slice() {
        let table = vec![table_range(100, 109), table_range(200, 204)];
        let param = built(10, 0, 8, 0, 3, 0, 0, 1, 0, 0, 1, table, None);

        // What core 0 answers, and what IBM hands cid -11 as well.
        assert_eq!(param.slice_id(Cid(0)), Some(WkSliceId(0)));
        assert_eq!(param.coord_vec(Cid(0)), vec![range(100, 109)]);

        assert_eq!(param.adjust_cid(Cid(-11)), Cid(-3));
        assert_eq!(param.slice_id(Cid(-11)), None);
        assert_eq!(param.size(Cid(-11)), WkSize(0));
        assert_eq!(param.coord_vec(Cid(-11)), vec![CoordRange::GAP]);

        assert_eq!(param.slice_id(Cid(-10)), None);
        assert_eq!(param.size(Cid(-10)), WkSize(0));

        // ⭐ THE NEGATIVES THAT WRAP BACK INSIDE THE GANG STILL AGREE WITH IBM, CID FOR CID, so the
        // divergence is confined to the offsets `adjustCID` cannot bring back (`:145-147`).
        assert_eq!(param.slice_id(Cid(-8)), Some(WkSliceId(0)));
        assert_eq!(param.slice_id(Cid(-7)), Some(WkSliceId(1)));
        assert_eq!(param.coord_vec(Cid(-7)), vec![range(200, 204)]);
        assert_eq!(param.slice_id(Cid(-5)), None);
    }

    /// ⛔ `getCoordVec`'S THIRD OUTCOME, REACHABLE AND UNDOCUMENTED UNTIL THIS REVIEW: a core with
    /// real work whose virtual span lies entirely past `real_coordinates_` overlaps no range, so the
    /// loop (`:257-281`) pushes nothing and the vector comes back EMPTY — neither a
    /// [`CoordRange::GAP`] nor a clipped range. Measured against the header, which answers `sl=2
    /// sz=10 cv=` for core 2 of this exact param.
    #[test]
    fn a_slice_entirely_past_the_real_table_yields_no_ranges_at_all() {
        let table = vec![table_range(100, 109), table_range(200, 204)];
        let param = built(10, 0, 8, 0, 3, 0, 0, 1, 0, 0, 1, table, None);

        // Core 2 is a real core doing real work ...
        assert_eq!(param.slice_id(Cid(2)), Some(WkSliceId(2)));
        assert_eq!(param.size(Cid(2)), WkSize(10));
        // ... and has nowhere real to do it. Three answers, three distinct shapes.
        assert_eq!(param.coord_vec(Cid(2)), Vec::new());
        assert_eq!(param.coord_vec(Cid(1)), vec![range(200, 204)]);
        assert_eq!(param.coord_vec(Cid(3)), vec![CoordRange::GAP]);
    }

    /// ⛔ THE ONE GAP THIS FILE CLAIMED COULD NEVER MAKE A DIVISOR ZERO MAKES IT ZERO. The claim on
    /// [`GapAfterAllSlices`] was that it "is the only gap that never divides", so it "cannot make a
    /// divisor zero" — but `getSliceId` sums it into `full_wksl_size_with_after_gaps_` and then
    /// divides by that at `:204`. Six unit slices with `gap_after_all_slices_ = -6` sum to exactly
    /// zero, and with `-fsanitize=undefined` the authority reports `division by zero,
    /// wkDivisionParams.h:204` for `getSliceId(-9)` — measured against the header, because IBM has no
    /// `offset_cid < 0` test at all and `adjustCID(-9) = -1` reaches `:204` past the `:200`
    /// comparison. The `span_with_gaps < 1` test here is what stands in for it, and it is load-bearing
    /// for core 0 too: `0 / 0` is a Rust panic where IBM's `0 >= 0` happens to return `-1`.
    #[test]
    fn a_negative_gap_after_all_slices_collapses_the_span_and_every_core_is_a_gap_core() {
        let param = built(10, 0, 8, 0, 6, 0, 0, 1, 0, -6, 1, Vec::new(), None);

        assert_eq!(param.single_slice_inner_length(WithGap::Yes), CoreSpan(1));
        assert_eq!(param.full_inner_length(WithGap::No), CoreSpan(6));
        assert_eq!(param.full_inner_length(WithGap::Yes), CoreSpan(0));

        for cid in 0..8 {
            assert_eq!(param.slice_id(Cid(cid)), None, "cid {cid}");
        }
        assert_eq!(sizes(&param, 8), vec![0; 8]);
        assert_eq!(param.coord_vec(Cid(0)), vec![CoordRange::GAP]);

        // The cid IBM divides by zero on: `adjustCID(-9)` is `-1`, which its `:200` comparison
        // against a span of zero lets through.
        assert_eq!(param.adjust_cid(Cid(-9)), Cid(-1));
        assert_eq!(param.slice_id(Cid(-9)), None);
        assert_eq!(param.size(Cid(-9)), WkSize(0));
    }

    /// ⛔ BOTH EXTREMES OF [`Cid`] ABORTED THE PORT, and neither needed a field outside its own
    /// `int32_t`: `slice_id`'s `cid - start_cid_offset_` (`:197`) panicked with `attempt to subtract
    /// with overflow`, and `adjust_cid`'s `cid + max_cores_` (`:146`) with `attempt to add with
    /// overflow`. Saturating both is what makes the cid space total, and it is the faithful reading:
    /// IBM's `int32_t` cid space has no extremes this far out, so the only question is what a
    /// WIDENED cid does, and every saturated value lands in the region both agree is no core at all.
    #[test]
    fn the_extremes_of_the_cid_space_are_gap_cores_rather_than_aborts() {
        // A positive offset the subtraction cannot represent: `i64::MAX - (-1)`.
        let high = built(4, 0, 8, -1, 4, 0, 0, 1, 0, 0, 1, Vec::new(), None);
        assert_eq!(high.slice_id(Cid(i64::MAX)), None);
        assert_eq!(high.size(Cid(i64::MAX)), WkSize(0));
        assert_eq!(high.coord_vec(Cid(i64::MAX)), vec![CoordRange::GAP]);

        // A negative offset the subtraction cannot represent: `i64::MIN - 1`.
        let low = built(4, 0, 8, 1, 4, 0, 0, 1, 0, 0, 1, Vec::new(), None);
        assert_eq!(low.slice_id(Cid(i64::MIN)), None);
        assert_eq!(low.size(Cid(i64::MIN)), WkSize(0));

        // A gang whose own size is negative, so `adjustCID` drives a negative cid further down
        // rather than back into range: `i64::MIN + (-1)`.
        let shrinking = built(0, 0, -1, 0, -1, 0, 0, 1, 0, 0, 1, Vec::new(), None);
        assert_eq!(shrinking.adjust_cid(Cid(i64::MIN)), Cid(i64::MIN));
        assert_eq!(shrinking.slice_id(Cid(i64::MIN)), None);
    }

    /// ⛔ A SPAN PRODUCT THAT LEAVES `i64` IS REFUSED AT CONSTRUCTION, WHERE IT USED TO ABORT
    /// `full_inner_length`. `gap_within_inner_repeat_ = repeat_factor_inner_ = num_ss_slices_ =
    /// INT32_MAX` is three fields each inside its own `int32_t`, and it panicked with `attempt to
    /// multiply with overflow`. The authority does not refuse it either — with
    /// `-fsanitize=undefined` it reports `signed integer overflow` at `:157` and `:171` and then
    /// answers `getSingleSliceInnerLength() = -2147483648`, `getFullInnerLength() = -2147483648` and
    /// `getSliceId(0) = -1`: a negative span for a param that claims INT32_MAX slices. Refusing the
    /// value is the only answer that is neither that nor an abort.
    #[test]
    fn a_span_product_that_leaves_i64_is_refused_at_construction() {
        let refused = WkSplitParam::new(
            WkSs(1),
            WkEpilogue(0),
            MaxCores(i32::MAX),
            Cid(0),
            NumSsSlices(i32::MAX),
            NumEpilogueSlices(0),
            GapWithinInnerRepeat::new(i32::MAX).unwrap(),
            RepeatFactorInner::new(i32::MAX).unwrap(),
            GapAfterInnerRepeat::new(0).unwrap(),
            GapAfterAllSlices(0),
            OuterRepeatFactor::new(1).unwrap(),
            Vec::new(),
            None,
        );
        assert!(refused.is_none());

        // ⭐ AND THE BOUND COSTS NOTHING REAL: the same gang, filled with INT32_MAX UNIT slices, is
        // accepted and agrees with the authority exactly — measured `getFullInnerLength() =
        // 2147483647`, `getSliceId(0) = 0`, `getSliceId(2147483646) = 2147483646`.
        let widest = built(
            1,
            0,
            i32::MAX,
            0,
            i32::MAX,
            0,
            0,
            1,
            0,
            0,
            1,
            Vec::new(),
            None,
        );
        assert_eq!(
            widest.full_inner_length(WithGap::Yes),
            CoreSpan(i64::from(i32::MAX))
        );
        assert_eq!(widest.slice_id(Cid(0)), Some(WkSliceId(0)));
        assert_eq!(
            widest.slice_id(Cid(i64::from(i32::MAX) - 1)),
            Some(WkSliceId(i64::from(i32::MAX) - 1))
        );
        assert_eq!(widest.slice_id(Cid(i64::from(i32::MAX))), None);
    }

    /// ⛔ A COORDINATE PRODUCT THAT LEAVES `i64` IS REFUSED AT CONSTRUCTION, WHERE IT USED TO ABORT
    /// `coord`. `wk_ss_ = num_ss_slices_ = INT32_MAX` at a stride of three panicked with `attempt to
    /// multiply with overflow`; the authority instead reports `signed integer overflow` at `:438`,
    /// `:439`, `:447`, `:448`, `:450` and `:451` and hands back the INVERTED range `(2147483633,
    /// -21)` for core 5 — measured.
    #[test]
    fn a_coordinate_product_that_leaves_i64_is_refused_at_construction() {
        let swp = StrWinPad {
            stride: Stride::new(3).unwrap(),
            window: Window::new(1).unwrap(),
            extra_back: ExtraBack::new(0).unwrap(),
        };
        let refused = WkSplitParam::new(
            WkSs(i32::MAX),
            WkEpilogue(0),
            MaxCores(i32::MAX),
            Cid(0),
            NumSsSlices(i32::MAX),
            NumEpilogueSlices(0),
            GapWithinInnerRepeat::new(0).unwrap(),
            RepeatFactorInner::new(1).unwrap(),
            GapAfterInnerRepeat::new(0).unwrap(),
            GapAfterAllSlices(0),
            OuterRepeatFactor::new(1).unwrap(),
            Vec::new(),
            Some(swp),
        );
        assert!(refused.is_none());

        // ⭐ THE SAME WORK SIZE ON A GANG OF 1024 IS ACCEPTED, and here the two answers DIVERGE
        // because the authority's is undefined: at `:243` and `:448` its `(vsize - 1) * stride_` is
        // `2147483646 * 3` in `int`, and its wrapped results are `getSize(0) = 2147483643` and
        // `getCoordVec(0) = (0, 2147483642)` — measured. The widened arithmetic is what those two
        // expressions mean, so this port answers with it.
        let gang = built(
            i32::MAX,
            0,
            1024,
            0,
            1024,
            0,
            0,
            1,
            0,
            0,
            1,
            Vec::new(),
            Some(swp),
        );
        assert_eq!(gang.size(Cid(0)), WkSize(6_442_450_939));
        assert_eq!(gang.coord_vec(Cid(0)), vec![range(0, 6_442_450_938)]);
    }

    /// ⛔ THE WIDEST STORED ENTRY THE TYPE CAN NOW HOLD REMAPS TOTALLY, AND THE ENTRY THAT ABORTED
    /// THE PORT IS NO LONGER EXPRESSIBLE. With one `i64` type for both ends,
    /// [`WkSplitParam::set_real_coordinates`] took `(i64::MIN, i64::MAX)` and `getCoordVec`'s own
    /// `pair.second - pair.first` (`:261`) panicked with `attempt to subtract with overflow`.
    ///
    /// ⭐ AT IBM'S WIDTH THE TWO AGREE WHEREVER IBM IS DEFINED: a stored span of exactly `INT32_MAX`
    /// gives `(-1, 2)`, `(3, 6)` and `(7, 10)` for the first three cores of a four-by-four split in
    /// both — measured.
    ///
    /// ⚠️ AND WHERE IBM IS NOT DEFINED THEY DIVERGE, DELIBERATELY: at `(INT32_MIN, INT32_MAX)` IBM's
    /// `:261` subtraction is `2147483647 - -2147483648` in `int`, which UBSan reports and which wraps
    /// to `-1`, so its `runningUnrealCoord` ends BEFORE it starts, `noOverlap` holds for every core
    /// and `getCoordVec` returns EMPTY — measured. This port widens each end at the point IBM's own
    /// `int64_t` destination does, so it clips the entry and returns the range.
    #[test]
    fn the_widest_stored_entry_remaps_without_leaving_i64() {
        // A stored span of exactly `INT32_MAX`, the widest IBM's `:261` can subtract.
        let defined = built(
            4,
            0,
            16,
            0,
            4,
            0,
            0,
            1,
            0,
            0,
            1,
            vec![table_range(-1, 2_147_483_646)],
            None,
        );
        assert_eq!(
            coords(&defined, 3),
            vec![vec![range(-1, 2)], vec![range(3, 6)], vec![range(7, 10)]]
        );

        // The widest entry the type can hold at all, which IBM cannot subtract.
        let mut widest = defined.clone();
        widest.set_real_coordinates(vec![table_range(i32::MIN, i32::MAX)]);
        assert_eq!(widest.real_coordinates(), [table_range(i32::MIN, i32::MAX)]);
        assert_eq!(
            coords(&widest, 2),
            vec![
                vec![range(-2_147_483_648, -2_147_483_645)],
                vec![range(-2_147_483_644, -2_147_483_641)],
            ]
        );
    }
}
