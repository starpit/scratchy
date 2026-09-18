//! Re-ported from the C++ authority. See crustify-scheduler/AGENT-BRIEF.md.

use crate::schedule::dims::PrimaryDimTypes;
use crate::schedule::dsc2::FoldParamInfoType;
use crate::schedule::metadata::Metadata;
use sys_arch_spec::RowId;
use sys_arch_spec::arch_enums::SenComponent;

/// Replaces: Ddc::registerComponents
///
/// Field: e025_Ddc.registerComponents
///
/// The seven components whose storage is a register file (`ddc/ddcv1.cpp:17-18`, declared
/// `ddc/ddc.h:36`). All three readers ask only for membership, to decide whether a transfer result
/// needs a real allocation (`ddc/ddc_transformation_util.cpp:923`, `:1041`,
/// `ddc/ddc_transformation.cpp:624`).
///
/// ⛔ A SUBSET OF `dsc2::memories`, NOT ITS COMPLEMENT: all seven are in that sixteen-component set
/// too (`dsc/dscdefn.h:518`, filled `dsc/dscdefn.cpp:142-146`, ported as
/// [`crate::schedule::dsc2::MEMORIES`]), and `dsc/dsc2.cpp:2315` excludes `PELRF` and `SFPLRF` from a
/// `memories.count(input)` branch BY NAME. Membership here never means "is not a memory".
pub const REGISTER_COMPONENTS: [SenComponent; 7] = [
    SenComponent::Pelrf,
    SenComponent::Sfplrf,
    SenComponent::Ptarf,
    SenComponent::Ptxrf,
    SenComponent::Pestate,
    SenComponent::Sfpstate,
    SenComponent::Lxluscalereg,
];

/// ⛔ E0080 IF A COMPONENT IS EVER LISTED TWICE: IBM's is an `unordered_set`, which collapses a
/// duplicate and keeps its `count()` honest; this array would keep both and make its `len()` a lie.
const _: () = {
    let mut i = 0;
    while i < REGISTER_COMPONENTS.len() {
        let mut j = i + 1;
        while j < REGISTER_COMPONENTS.len() {
            assert!(
                REGISTER_COMPONENTS[i] as i32 != REGISTER_COMPONENTS[j] as i32,
                "REGISTER_COMPONENTS lists one component twice"
            );
            j += 1;
        }
        i += 1;
    }
};

/// The `int verbose` the constructor takes (`ddc/ddc.h:49`). Open-ended: it arrives from `atoi` on
/// the standalone's `-v` and is handed on to `DdlConvertInterface` (`ddc/ddcv1.cpp:3723`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Verbosity(pub i32);

/// `transformationReportLevel_` (`ddc/ddc.h:40`).
///
/// ⛔ NOT A CLOSED LEVEL SET, unlike [`CoordReportLevel`]: `atoi` on the standalone's `-r` reaches
/// the constructor unchecked (`ddc/ddc_standalone.cpp:43`, `:69`), so a 4 or a -1 stays
/// representable here because it is representable there. The other writer DOES assign a literal,
/// and it is the one on our path: `runDdc` passes `0`
/// (`dbo/src/Utils/sdsc_bundle/SchedulerStages.cpp:35`).
///
/// ⛔ AND ITS READERS ARE NOT ALL `>`. Forty-one lines under `ddc/` test `> 0`, `> 1` or `> 2`; the
/// remaining two, `ddc/ddc_transformation.cpp:87` and `:463`, test `!transformationReportLevel_` —
/// a ZERO test, and it selects the ternary arm that EAGERLY concatenates two `getNodeDescription`
/// calls. At a negative level `!x` is false, so the authority BUILDS that message, while a porter
/// who writes the idiomatic `level > TransformationReportLevel(0)` there skips it. The guard below
/// pins the value the two readings disagree on, so narrowing this to a `u32` or to an `Off..` enum
/// cannot land silently.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TransformationReportLevel(pub i32);

/// ⛔ `ddc/ddc_transformation.cpp:87` and `:463` branch on `x != 0`, NOT on `x > 0`. This fails to
/// build the moment the type stops representing a level on which those two readings differ.
const _: () = {
    let level = TransformationReportLevel(-1).0;
    assert!(
        (level != 0) != (level > 0),
        "ddc_transformation.cpp:87 reads `!transformationReportLevel_`, which is not `> 0`"
    );
};

/// One LATCH's producer/consumer link id, as `latchDataId_` carries it (`dsc/dsc2.h:725`). The `-1`
/// that field defaults to is "not latched" and is not one of these: the counter only counts up.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LatchDataId(pub i32);

/// Which execution step's memory-tracker snapshot to work against — `exphase` (`ddc/ddc.h:101`), set
/// from `runDdc`'s `executionStep` (`dbo/src/Utils/sdsc_bundle/SchedulerStages.cpp:38`).
///
/// ⛔ UNSIGNED BECAUSE A NEGATIVE PHASE IS THE THROW, NOT A PHASE. `epsToListIter` is keyed only by
/// `0..exPhases` (`util/memtracker/mem_track.cpp:128-130`), and the unguarded `.at(currEp)` calls
/// [`Ddc::exphase`] enumerates make any other key `std::out_of_range`. Both writers in the authority
/// already supply a member of that domain — `0` (`ddc/ddc_standalone.cpp:72`) and `execStepOf`, a
/// phase index defaulting to `0` (`dbo/src/ProgramAttrs.h:97-104`, via `SchedulerStages.cpp:38`) — so
/// the authority's `-1` is spelled `None` here and NOWHERE ELSE:
/// ```compile_fail
/// let _ = deeptools::schedule::ddc::ExPhase(-1);
/// ```
/// ⛔ AND ITS CONTROL, because `compile_fail` passes on any error at all: constructing a phase that IS
/// in the domain compiles by the same path, so the block above fails on the SIGN and nothing else:
/// `E0600`, "cannot apply unary operator `-` to type `u32`", read from `rustc` against the built rlib
/// because rustdoc checks no error code even when one is written.
/// ```
/// let _ = deeptools::schedule::ddc::ExPhase(0);
/// ```
/// ⚠️ RESIDUAL, NOT CLOSED: `exPhases` is a runtime count, so a phase at or above it is equally
/// `std::out_of_range` and stays representable here. This type rules out the one value the authority
/// actually writes as "unset"; it does not bound the phase count.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ExPhase(pub u32);

/// The type of `coordFoldReportLevel_` and `coordPropReportLevel_` (`ddc/ddc.h:41-42`), named after
/// the authority's own parameter `int coordReportLevel = 0` (`dsc/dsc2.h:1183`, `:1230`, `:1238`).
///
/// ⭐ A CLOSED SET, PROVABLY: tree-wide those two fields are assigned only 0, 1, 2 and 3
/// (`ddc/ddc.h:57`, `:59`, `:61`, `:65`, `:67`, `:69`) and compared only `> 0`, `> 1` and `> 2`.
/// Deriving `Ord` over discriminants 0..3 reproduces every comparison and makes a fourth level
/// unrepresentable.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(i32)]
pub enum CoordReportLevel {
    /// The member initialiser (`ddc/ddc.h:41-42`): print nothing.
    #[default]
    Off = 0,
    Short = 1,
    Med = 2,
    Long = 3,
}

/// ⛔ E0080 IF A LEVEL IS RENUMBERED: the derived `Ord` only means the authority's `> 0`, `> 1` and
/// `> 2` while the discriminants are the literals the constructor assigns.
const _: () = {
    assert!(CoordReportLevel::Off as i32 == 0);
    assert!(CoordReportLevel::Short as i32 == 1);
    assert!(CoordReportLevel::Med as i32 == 2);
    assert!(CoordReportLevel::Long as i32 == 3);
};

impl CoordReportLevel {
    /// The `content_*` chain of the coordinate option string (`ddc/ddc.h:56-62`).
    ///
    /// ⛔ TWO TRAPS, BOTH THE AUTHORITY'S: the chain is `else if` in the order short, med, long, so
    /// the FIRST spelling present wins regardless of level — `"content_long,content_short"` is
    /// [`Self::Short`]; and `find(..) != npos` is a SUBSTRING test, so `"content_medium"` is
    /// [`Self::Med`].
    pub fn content(coord_option_str: &str) -> Self {
        if coord_option_str.contains("content_short") {
            Self::Short
        } else if coord_option_str.contains("content_med") {
            Self::Med
        } else if coord_option_str.contains("content_long") {
            Self::Long
        } else {
            Self::Off
        }
    }

    /// The `prop_*` chain of the SAME string (`ddc/ddc.h:64-70`), with the same first-spelling-wins
    /// and substring behaviour as [`Self::content`]. Two independent chains over one value, so
    /// `"content_med,prop_long"` sets both levels differently.
    pub fn prop(coord_option_str: &str) -> Self {
        if coord_option_str.contains("prop_short") {
            Self::Short
        } else if coord_option_str.contains("prop_med") {
            Self::Med
        } else if coord_option_str.contains("prop_long") {
            Self::Long
        } else {
            Self::Off
        }
    }
}

/// The whitespace a `std::stringstream` extractor skips and stops at — `std::isspace` in the classic
/// locale.
fn is_stream_space(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\n' | '\u{b}' | '\u{c}' | '\r')
}

/// `dtGetEnv<T>`'s parse step for `T = std::string` (`util/dtgetenv.hpp:122-138`), split off from the
/// environment read so it is exercisable without touching the process environment.
///
/// ⛔ ONE WHITESPACE-FREE TOKEN AND NOTHING AFTER IT. The guard is `(ss >> parsed) && ss.eof()`: the
/// extractor skips leading whitespace, stops at the first whitespace AFTER the token, and sets
/// `eofbit` only when it stopped by running out of input. So `" verify_loopelemoff"` parses and
/// `"verify_loopelemoff "` does NOT — one trailing space discards the whole option silently. An empty
/// value never reaches here; `dtGetEnv` rejects it at `'\0' == ptr[0]` (`:126`).
fn parse_env_token(value: &str) -> Option<&str> {
    let rest = value.trim_start_matches(is_stream_space);
    if rest.is_empty() || rest.contains(is_stream_space) {
        return None;
    }
    Some(rest)
}

/// The one `dtGetEnv` call the constructor makes — `dtGetEnv<std::string>("DDCCOORD")`
/// (`ddc/ddc.h:72`).
///
/// ⛔ DIVERGES ON s390x, IN THE ONE DIRECTION WE CAN BUILD. `dtGetEnv` asks `canParseEnvVar` first
/// (`util/dtgetenv.hpp:65-112`), whose `#if PRODUCTION_MODE` branch answers only for its own
/// allowlist plus the `AIU_WORLD_RANK_` prefix; `DDCCOORD` is on neither, so on a production build
/// this variable is invisible and `verify_coordinate_based_loop_elem_off` can never become true. This
/// is the `#else return true` branch (`:109-110`); `:108` is the PRODUCTION_MODE arm's
/// `return canParse;`.
///
/// ⛔ AND DIVERGES ON A NON-UTF-8 VALUE. `std::env::var` answers `Err(NotUnicode)` where `getenv`
/// hands `std::stringstream` the raw bytes, so a whitespace-free token that is not valid UTF-8 and
/// CONTAINS `verify_loopelemoff` sets the flag in the authority and leaves it false here. Closing it
/// wants `std::env::var_os` and a byte search; it is left open because the variable is invisible on
/// the one target that ships (above), so no shipped run reaches the difference.
pub fn ddc_coord_env_option() -> Option<String> {
    let value = std::env::var("DDCCOORD").ok()?;
    parse_env_token(&value).map(str::to_owned)
}

/// The Deep Dataflow Constructor — `class Ddc` (`ddc/ddc.h:34-802`), the pass that turns a
/// `SuperDsc`'s design space configurations into a schedule tree.
///
/// ⛔ THIRTEEN OF THE CLASS'S TWENTY DECLARED FIELDS, plus the static `registerComponents` as
/// [`REGISTER_COMPONENTS`], so the `e025_Ddc` and `e032_Ddc` anchors below stay open. All seven
/// left out are a pointer or a pointer-keyed container, under two blockers.
///
/// Types that are not scheduled units in `crustify-scheduler/UNITS.tsv` at all:
///  * `dscGlobal` (`ddc/ddc.h:37`) — `const DesignSpaceConfigGlobal&`;
///  * `memTrackers` (`ddc/ddc.h:102`) — `MemTrackBundle*`, and note it has NO member initialiser, so
///    a `Ddc` whose caller forgets it holds an indeterminate pointer; both real construction sites
///    assign it at once (`dbo/src/Utils/sdsc_bundle/SchedulerStages.cpp:36`,
///    `ddc/ddc_standalone.cpp:71`);
///  * `sdsc_` (`ddc/ddc.h:106`) — `SuperDsc*`, set to `run_v1`'s argument (`ddc/ddcv1.cpp:3693`);
///  * `currDsc` (`ddc/ddc.h:107`) — `DesignSpaceConfig*`; the target type IS ported
///    ([`crate::schedule::dsc::DesignSpaceConfig`]) but it points INTO `sdsc.dscs_`
///    (`ddc/ddcv1.cpp:3700`), so it cannot be represented without `SuperDsc`.
///
/// Fields needing `dsc2::ScheduleNode` identity, the blocker e013, e016, e018 and e023 also report:
///  * `loopsBelowChunkBoundary` (`ddc/ddc.h:108`) — `unordered_set<const dsc2::LoopNode*>`, filled by
///    a DFS over the tree (`ddc/ddcv1.cpp:3683-3689`);
///  * `coordPropTracker` (`ddc/ddc.h:548`) and all three fields of its nested `CoordPropTracker`
///    (`ddc/ddc.h:536-546`): `itemsToProcess_` is a `deque<dsc2::CoordPropInfoType>`, which holds two
///    `ScheduleNode*` (`dsc/dsc2.h:1089-1090`) and is itself an open anchor in `schedule/dsc2.rs`;
///    `refsAdded_` is keyed by `ScheduleNode*` twice over; and `currItemToProcess_` is a CURSOR INTO
///    `itemsToProcess_`, so its `int` alone would be a field no ported reader could honour;
///  * and `loopDistributionParamInfo` (`ddc/ddc.h:550-553`) — a `ScheduleNode*`-keyed map of
///    `ScheduleNode*`-keyed maps.
///
/// ⛔ AND ONE NESTED TYPE THAT IS NOT A FIELD, WHICH IS WHY THE CENSUS SAYS TWENTY AND NOT
/// TWENTY-ONE: `struct RowGroupInfo` (`ddc/ddc.h:555-609`) declares no member of itself. The class's
/// last data member is `loopDistributionParamInfo` at `:550-553`; everything from `:555` on is a type
/// or a method, and `RowGroupInfo` reaches `Ddc` only as a `RowGroupInfo&` parameter. Its `cat` and
/// `activeRow` are ported as [`RowGroupCategory`]; its other three members are blocked —
/// `commonGroupAncestor` is a `dsc2::BlockNode*`, every `RowGroupNodeInfo` holds a `ScheduleNode*`,
/// and `ascendingOrder` (`ddc/ddc.h:578`) is DEAD, one hit tree-wide, its order re-derived from
/// `nodeInfo` at `ddc/ddc_fold.cpp:2192`.
///
/// Transposing the constructor's two `int` parameters is a compile error, which is why neither is a
/// bare scalar:
/// ```compile_fail
/// use deeptools::schedule::ddc::{Ddc, TransformationReportLevel, Verbosity};
/// let _ = Ddc::new(TransformationReportLevel(0), false, Verbosity(0), "");
/// ```
/// ⛔ AND THAT BLOCK NEEDS THIS CONTROL TO MEAN ANYTHING: `compile_fail` passes on ANY error, a wrong
/// path or a private item included, and rustdoc checks no error code even when one is written. The
/// same call with the arguments the right way round compiles, so the block above fails on the
/// TRANSPOSITION (`error[E0308]`, read from `rustc` against the built rlib) and on nothing else:
/// ```
/// use deeptools::schedule::ddc::{Ddc, TransformationReportLevel, Verbosity};
/// let _ = Ddc::new(Verbosity(0), false, TransformationReportLevel(0), "");
/// ```
///
/// ⛔ NOT `Clone`, AND NEITHER IS `class Ddc`: [`Ddc::metadata`] alone deletes its copy-construction
/// (`ddc/ddc_metadata.h:130-137`) and its copy-assignment (`:211-212`), and the `const
/// DesignSpaceConfigGlobal&` at `ddc/ddc.h:37` deletes copy-assignment a second time over.
///
/// ⛔ AND NOT `PartialEq` EITHER, FOR THE SAME REASON ONE STEP DOWN: [`Metadata`] carries the
/// `std::vector<ExternalTransfer>` of OWNED nodes (`ddc/ddc_metadata.h:130-137`), and neither that
/// entry nor either node type declares an `operator==` — a node's identity in the authority is its
/// address, not its value. Assert on the field you mean.
#[derive(Debug)]
pub struct Ddc {
    /// Field: e025_Ddc.verbose_
    ///
    /// Field: e032_Ddc.verbose_
    ///
    /// `ddc/ddc.h:38`. One of TWO fields with no member initialiser that the constructor does set:
    /// the other is the `dscGlobal` reference (`:37`), bound in the same init list (`:52-53`). A THIRD
    /// has none and the constructor does NOT set it — `memTrackers` (`:102`), which is why the type
    /// doc calls it an indeterminate pointer on a caller that forgets it.
    pub verbose: Verbosity,

    /// Field: e025_Ddc.latchDataIdCounter_
    ///
    /// Field: e032_Ddc.latchDataIdCounter_
    ///
    /// `ddc/ddc.h:39`. Holds the NEXT id to hand out, not the last one handed out. `run_v1` resets it
    /// per DSC (`ddc/ddcv1.cpp:3708`); [`Ddc::next_latch_data_id`] is its only consumer.
    pub latch_data_id_counter: LatchDataId,

    /// Field: e025_Ddc.transformationReportLevel_
    ///
    /// Field: e032_Ddc.transformationReportLevel_
    ///
    /// `ddc/ddc.h:40` — a constructor parameter, never derived from the coordinate option string.
    pub transformation_report_level: TransformationReportLevel,

    /// Field: e025_Ddc.coordFoldReportLevel_
    ///
    /// Field: e032_Ddc.coordFoldReportLevel_
    ///
    /// `ddc/ddc.h:41`, from the option string's `content_*` spelling ([`CoordReportLevel::content`]).
    /// `Ddc::coordinateCapture` (`ddc/ddc_fold.cpp:1538-1623`) is its ONLY reader and all eight reads
    /// are inside it (`:1543`, `:1555`, `:1559`, `:1577`, `:1581`, `:1606`, `:1615`, `:1619`).
    pub coord_fold_report_level: CoordReportLevel,

    /// Field: e025_Ddc.coordPropReportLevel_
    ///
    /// Field: e032_Ddc.coordPropReportLevel_
    ///
    /// `ddc/ddc.h:42`, from the SAME string's `prop_*` spelling ([`CoordReportLevel::prop`]).
    /// SIXTY-FIVE lines of `ddc/ddc_fold.cpp` read it (`:503` through `:4680`) against the eight of
    /// [`Ddc::coord_fold_report_level`], which is why it is a second level and not one knob.
    pub coord_prop_report_level: CoordReportLevel,

    /// Field: e025_Ddc.verifyCoordinateBasedLoopElemOff
    ///
    /// Field: e032_Ddc.verifyCoordinateBasedLoopElemOff
    ///
    /// `ddc/ddc.h:43` — the ONLY field driven by an environment variable rather than a parameter
    /// ([`ddc_coord_env_option`]), with one reader: `ddc/ddcv1.cpp:2465` takes the coordinate-derived
    /// element-offset path when this OR `datastage_based_elem_off` is set.
    pub verify_coordinate_based_loop_elem_off: bool,

    /// Field: e025_Ddc.datastageBasedElemOff
    ///
    /// Field: e032_Ddc.datastageBasedElemOff
    ///
    /// `ddc/ddc.h:44`. NOT a constructor parameter: `run_v1` latches it true per `SuperDsc`, never
    /// back to false, when any compute op is a `ReStickifyOpLx` or `ReStickifyOpHBM`, and MIRRORS it
    /// onto `sdsc.datastageBasedElemOff` in the same breath (`ddc/ddcv1.cpp:3709-3715`) so
    /// `dsc/dsc2.cpp:3034` can read it — and that read is of `SuperDsc`'s own copy
    /// (`dsc/superdsc.h:116`), a DIFFERENT field. EIGHT sites read THIS one, all in `ddc/ddcv1.cpp`:
    /// `:2308`, `:2454`, `:2465`, `:2469`, `:2597`, `:2687` and `:3049` switch element-offset
    /// derivation, and `:3785` switches `coordinateCapture()` off outright, which is a whole pass and
    /// not an offset.
    pub datastage_based_elem_off: bool,

    /// Field: e025_Ddc.dscToDdl_
    ///
    /// Field: e032_Ddc.dscToDdl_
    ///
    /// `ddc/ddc.h:45`. Constructor parameter; its one reader dumps the converted DDL to stdout at the
    /// end of `run_v1` (`ddc/ddcv1.cpp:3796`). `runDdc` passes false (`SchedulerStages.cpp:35`).
    pub dsc_to_ddl: bool,

    /// Field: e025_Ddc.trueLXTracker_
    ///
    /// Field: e032_Ddc.trueLXTracker_
    ///
    /// `ddc/ddc.h:46-47`. Set true only by `runDdc` (`SchedulerStages.cpp:37`), never by the
    /// standalone (`ddc/ddc_standalone.cpp:69-72`), and read once.
    ///
    /// ⛔ THAT READER IS A REFUSAL, NOT A MODE SELECTOR. `trueLXTracker_ || comp != SenComponents::LX`
    /// is the CONDITION OF A `DT_CHECK_MSG` (`ddc/ddcv1.cpp:185-188`), which throws `DtException` when
    /// it is false (`util/dt_exception.hpp:110-118`) and is caught one frame out as a pass failure
    /// (`dbo/src/Transforms/sdsc_bundle/SchedulerPasses.cpp:154-160`). It is the FIRST statement of
    /// `tryAlloc`'s loop over `metadata.newAllocations_` (`ddc/ddcv1.cpp:183-184`), so with this false
    /// the first LX entry aborts allocation with "DDC is currently being passed ephemeral mem
    /// trackers, it can't use those for LX allocations". The standalone does not track LX differently
    /// — it cannot allocate LX at all.
    pub true_lx_tracker: bool,

    /// Field: e025_Ddc.exphase
    ///
    /// Field: e032_Ddc.exphase
    ///
    /// `ddc/ddc.h:101`; `None` is the authority's `-1`.
    ///
    /// ⛔ `-1` IS NOT INERT — IT THROWS, which is why this is an `Option` AND why [`ExPhase`] cannot
    /// spell it. `allocAllMem` widens the field to `std::vector<int> seps(1, exphase)`
    /// (`ddc/ddcv1.cpp:278`) and hands `seps` to SEVEN tracker calls. FIVE ARE UNGUARDED: `removeDs`
    /// (`util/memtracker/mem_track.cpp:441-442`, called `ddc/ddcv1.cpp:280`) and `addDsAtStartAddr`
    /// (`:380-382`, called `ddc/ddcv1.cpp:288`, `:290`, `:307`, `:310`) index
    /// `epsToListIter.at(currEp)` with no lookup guard, and that map only ever holds keys
    /// `0..exPhases` (`:128-130`, grown at `:116-118`), so `-1` raises `std::out_of_range`. The
    /// `removeDs` at `:280` runs FIRST, so the throw is what a caller sees. The other two —
    /// `checkAndAddDsAtAddr` and `checkAndAddDs` (`ddc/ddcv1.cpp:335`, `:340`) — do not throw and do
    /// not rescue it either: `checkIfDsFits`'s miss branch answers `fits = false`
    /// (`util/memtracker/mem_track.cpp:151-154`), so every allocation returns `DOESNT_FIT`, `allDsFit`
    /// goes false (`ddc/ddcv1.cpp:344-349`), and `runDdc` raises "Scheduler failed to find a suitable
    /// op mapping" (`SchedulerStages.cpp:40-42`). Only `backupEps`
    /// (`util/memtracker/mem_track.cpp:566-568`, called `ddc/ddcv1.cpp:218`) and `restoreEps`
    /// (`:580-582`, called `ddc/ddcv1.cpp:433`) are the guarded no-ops an earlier reading of this
    /// field generalised from two calls to nine.
    pub exphase: Option<ExPhase>,

    /// Field: e025_Ddc.metadata
    ///
    /// Field: e032_Ddc.metadata
    ///
    /// `ddc/ddc.h:105`. Per-DSC scratch: `run_v1` clears it at the top of every DSC iteration
    /// (`ddc/ddcv1.cpp:3706`), which is [`Metadata::clear`].
    pub metadata: Metadata,

    /// Field: e025_Ddc.coreletSplitDim
    ///
    /// Field: e032_Ddc.coreletSplitDim
    ///
    /// `ddc/ddc.h:109`. [`PrimaryDimTypes::Undefined`] is the authority's `PrimaryDimTypesCount`
    /// initialiser and means "no corelet split". `initGlobalData` recomputes it per DSC: back to the
    /// sentinel, then the FIRST key of the core stage's `coreletSplit_` if non-empty
    /// (`ddc/ddcv1.cpp:3673-3681`).
    ///
    /// ⛔ SIX MEMBER READS AT FOUR SITES, AND ONLY TWO SITES TEST THE SENTINEL.
    /// `ddc/ddc_fold.cpp:2480` and `:3121` do. `:2543` compares it to a real dim, but is reached
    /// only where `:2480` already excluded the sentinel. The fourth site DOES NOT TEST IT:
    /// `is_any_of(coreletSplitDim, coordPropInfo.dimsToPropagate)` (`:3105`) puts the sentinel
    /// through ordinary comparison against a list of real dims, and a match raises `DT_ERROR`
    /// (`:3109-3114`), whose condition and message read the field twice more at `:3108` and
    /// `:3112` — all of it BEFORE `:3121`. It survives only because no `dimsToPropagate` carries
    /// the sentinel, which is an invariant of the CALLER and not of this field. (`:3112`'s
    /// `primaryDimToString.at` would not itself throw on the sentinel: it IS a key of that map,
    /// spelled `"undefined"`, `dsc/dims.cpp:23`.) ⚠️ `ddc/ddcv1.cpp:1933-1949` is NOT a reader:
    /// `:1933` declares a LOCAL `coreletSplitDim` that shadows the member.
    pub corelet_split_dim: PrimaryDimTypes,

    /// Field: e025_Ddc.dataStageExplorationDone_
    ///
    /// Field: e032_Ddc.dataStageExplorationDone_
    ///
    /// `ddc/ddc.h:112`. A one-way phase latch WITHIN one DSC: false at the top of each
    /// (`ddc/ddcv1.cpp:3707`), true once `exploreAssignDataStages` finishes (`ddc/ddcv1.cpp:556`).
    /// SIX sites branch on it (`ddc/ddc_transformation_util.cpp:294`, `:637`, `:761` and
    /// `ddc/ddc_transformation.cpp:256`, `:296`, `:675`), so the same transformation behaves
    /// differently before and after — a phase, not a flag.
    pub data_stage_exploration_done: bool,
}

impl Ddc {
    /// `Ddc::Ddc` (`ddc/ddc.h:49-77`) minus the `dscGlobal` reference the type doc reports as
    /// blocked. The authority's four defaulted arguments are all supplied here; `runDdc`'s call is
    /// `Ddc(dscGlobal, verbose, false, 0)` (`SchedulerStages.cpp:35`), which is this with `""`.
    ///
    /// ⛔ ONE OPTION STRING SETS TWO LEVELS and the environment sets a third field behind the
    /// caller's back — [`CoordReportLevel::content`], [`CoordReportLevel::prop`] and
    /// [`ddc_coord_env_option`] each lose malformed input silently.
    pub fn new(
        verbose: Verbosity,
        dsc_to_ddl: bool,
        transformation_report_level: TransformationReportLevel,
        coord_option_str: &str,
    ) -> Self {
        Self {
            verbose,
            latch_data_id_counter: LatchDataId(0),
            transformation_report_level,
            coord_fold_report_level: CoordReportLevel::content(coord_option_str),
            coord_prop_report_level: CoordReportLevel::prop(coord_option_str),
            verify_coordinate_based_loop_elem_off: ddc_coord_env_option()
                .is_some_and(|option| option.contains("verify_loopelemoff")),
            datastage_based_elem_off: false,
            dsc_to_ddl,
            true_lx_tracker: false,
            exphase: None,
            metadata: Metadata::default(),
            corelet_split_dim: PrimaryDimTypes::Undefined,
            data_stage_exploration_done: false,
        }
    }

    /// `currLatchDataId = latchDataIdCounter_++` (`ddc/ddc_transformation_util.cpp:967`), the
    /// counter's only consumer.
    ///
    /// ⛔ POST-INCREMENT: the id handed out is the value BEFORE the bump, so a DSC's first latch is 0.
    /// It is then written into the destination's `latchDataId_` and copied to every consumer
    /// (`:969-970`, `:990`, `:1007-1008`), which is what links producer to consumer through a LATCH.
    pub fn next_latch_data_id(&mut self) -> LatchDataId {
        let id = self.latch_data_id_counter;
        self.latch_data_id_counter = LatchDataId(id.0 + 1);
        id
    }
}

/// `Ddc::printFoldParams` (`ddc/ddc.h:611-616`), appending to a `String` rather than writing to a
/// stream, as [`crate::schedule::dims::DataStructDims::print`] does. A free function because the
/// authority's member reads no field of `Ddc` — only its argument and `std::cout`.
///
/// ⛔ ZERO CALLERS IN THE AUTHORITY: declared, defined inline and never invoked anywhere in the tree.
/// It is here because it is a declared member of this unit, not because anything reaches it yet.
pub fn print_fold_params(fold_params: &[FoldParamInfoType], out: &mut String) {
    for fp_info in fold_params {
        out.push('(');
        out.push_str(&fp_info.alpha.0.to_string());
        out.push_str(", ");
        out.push_str(&fp_info.beta.0.to_string());
        out.push_str(", ");
        out.push_str(&fp_info.cardinality.0.to_string());
        out.push_str(", ");
        out.push_str(&fp_info.fold_dim_label);
        out.push_str(") ");
    }
}

/// Field: e025_Ddc.Category
///
/// Field: e025_Ddc.cat
///
/// Field: e025_Ddc.activeRow
///
/// How a coordinate propagation relates the reference node's PT row to the working node's —
/// `RowGroupInfo::Category` (`ddc/ddc.h:556-562`), carrying `activeRow` (`:577`).
///
/// ⭐ TWO C++ FIELDS, ONE TYPE, BECAUSE `activeRow` IS ONLY EVER A `ROW_TO_SAME_ROW` FACT. Both of
/// its writes are the statement after `cat = ROW_TO_SAME_ROW` in the same block
/// (`ddc/ddc_fold.cpp:1189-1190`, `:1211-1212`), and its one read sits inside `if (cat ==
/// ROW_TO_SAME_ROW)` (`:1234-1237`). Neither of those two paths calls `gatherRelatedPTRowsBase`,
/// the only function that rewrites a category after the fact and only ever to `NO_BUNDLING`
/// (`:745`, `:788`, `:804`, `:817`, `:842`, `:850`, `:869`, `:877`), so no path sets a row and then
/// moves off this variant. Both writers take the row from a `getCompRowId` they have already tested
/// `!= -1` (`:1181-1183`, `:1208-1210`), so the `-1` default is never stored either.
///
/// ⛔ THE DEFAULT IS THE LAST ENUMERATOR, NOT THE FIRST. `} cat = NO_BUNDLING;` (`ddc/ddc.h:562`)
/// is a member initialiser, and seven sites construct a `RowGroupInfo` by declaration alone and
/// rely on it (`ddc/ddc_fold.cpp:2428`, `:3750`, `:3789`, `:3837`, `:4610`, `:4647`, `:4676`).
///
/// A row therefore cannot be spelled without the category that owns it:
/// ```compile_fail
/// use deeptools::schedule::ddc::RowGroupCategory;
/// let _ = RowGroupCategory::NoBundling { active_row: sys_arch_spec::RowId(3) };
/// ```
/// ⛔ AND THAT NEEDS THIS CONTROL, because `compile_fail` passes on ANY error and rustdoc checks no
/// error code even when one is written: the same field on the variant that owns it compiles, so the
/// block above fails on the CATEGORY (`error[E0559]`) and on nothing else.
/// ```
/// use deeptools::schedule::ddc::RowGroupCategory;
/// let _ = RowGroupCategory::RowToSameRow { active_row: sys_arch_spec::RowId(3) };
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum RowGroupCategory {
    /// `ROW_TO_SAME_ROW` (`ddc/ddc.h:557`): reference and working node share one PT row, so the
    /// rowsplit fold is that row alone (`ddc/ddc_fold.cpp:2175-2188`).
    RowToSameRow {
        /// The one row the group requires (`ddc/ddc.h:577`).
        active_row: RowId,
    },
    /// `NONROW_TO_ROW` (`ddc/ddc.h:558`): UN-bundling, gathered from the reversed propagation
    /// (`ddc/ddc_fold.cpp:1221-1231`).
    NonRowToRow,
    /// `ROW_TO_NONROW` (`ddc/ddc.h:559`): the rows must be bundled (`ddc/ddc_fold.cpp:1177`).
    RowToNonRow,
    /// `ROW_NORTH_SOUTH` (`ddc/ddc.h:560`): adjacent rows — the `DT_CHECK` at
    /// `ddc/ddc_fold.cpp:1184` is what keeps them adjacent.
    RowNorthSouth,
    /// `NO_BUNDLING` (`ddc/ddc.h:561`), and the member initialiser at `:562`.
    #[default]
    NoBundling,
}

impl RowGroupCategory {
    /// The spelling `RowGroupInfo::print` gives this category (`ddc/ddc.h:583-599`).
    ///
    /// ⛔ ITS `default:` ARM (`ddc/ddc.h:599-600`) PRINTS NOTHING and is unreachable — the switch
    /// already covers all five enumerators — so there is no sixth spelling to reproduce.
    pub fn print(self) -> &'static str {
        match self {
            Self::RowToSameRow { .. } => "Row-to-SameRow",
            Self::NonRowToRow => "NonRow-to-Row",
            Self::RowToNonRow => "Row-to-NonRow",
            Self::RowNorthSouth => "Row-North-South",
            Self::NoBundling => "No-Bundling",
        }
    }
}

// crustify:todo: e032_Ddc

// crustify:todo: e032_Ddc.currItemToProcess_

// crustify:todo: e032_Ddc.itemsToProcess_

// crustify:todo: e032_Ddc.refsAdded_

// crustify:todo: e032_Ddc.sdsc_

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::schedule::dsc2::{Alpha, Beta, Cardinality, MEMORIES};

    /// `runDdc`'s own call (`dbo/src/Utils/sdsc_bundle/SchedulerStages.cpp:35`).
    fn ddc() -> Ddc {
        Ddc::new(Verbosity(0), false, TransformationReportLevel(0), "")
    }

    /// `ddc/ddc.h:52-55`: the constructor names four fields; every other one keeps its member
    /// initialiser, and the default option string leaves both coordinate levels off.
    #[test]
    fn the_constructor_sets_the_four_fields_it_names_and_nothing_else() {
        let ddc = Ddc::new(Verbosity(3), true, TransformationReportLevel(2), "");
        assert_eq!(ddc.verbose, Verbosity(3));
        assert!(ddc.dsc_to_ddl);
        assert_eq!(
            ddc.transformation_report_level,
            TransformationReportLevel(2)
        );
        assert_eq!(ddc.latch_data_id_counter, LatchDataId(0));
        assert_eq!(ddc.coord_fold_report_level, CoordReportLevel::Off);
        assert_eq!(ddc.coord_prop_report_level, CoordReportLevel::Off);
        assert!(!ddc.datastage_based_elem_off);
        assert!(!ddc.true_lx_tracker);
        assert!(!ddc.data_stage_exploration_done);
        assert_eq!(ddc.exphase, None);
        assert_eq!(ddc.corelet_split_dim, PrimaryDimTypes::Undefined);
    }

    /// Each spelling on its own (`ddc/ddc.h:56-62`, `:64-70`), and one string setting both levels
    /// independently through `Ddc::new`.
    #[test]
    fn each_coordinate_spelling_selects_its_own_level_on_its_own_chain() {
        assert_eq!(
            CoordReportLevel::content("content_short"),
            CoordReportLevel::Short
        );
        assert_eq!(
            CoordReportLevel::content("content_med"),
            CoordReportLevel::Med
        );
        assert_eq!(
            CoordReportLevel::content("content_long"),
            CoordReportLevel::Long
        );
        assert_eq!(
            CoordReportLevel::prop("prop_short"),
            CoordReportLevel::Short
        );
        assert_eq!(CoordReportLevel::prop("prop_med"), CoordReportLevel::Med);
        assert_eq!(CoordReportLevel::prop("prop_long"), CoordReportLevel::Long);

        let ddc = Ddc::new(
            Verbosity(0),
            false,
            TransformationReportLevel(0),
            "content_med,prop_long",
        );
        assert_eq!(ddc.coord_fold_report_level, CoordReportLevel::Med);
        assert_eq!(ddc.coord_prop_report_level, CoordReportLevel::Long);
    }

    /// ⛔ THE `else if` ORDER IS SHORT, MED, LONG — the FIRST spelling present wins, not the loudest
    /// — and `find(..) != npos` is a SUBSTRING test, not equality (`ddc/ddc.h:56-70`).
    #[test]
    fn the_first_spelling_wins_and_it_matches_as_a_substring() {
        assert_eq!(
            CoordReportLevel::content("content_long,content_short"),
            CoordReportLevel::Short
        );
        assert_eq!(
            CoordReportLevel::prop("prop_long,prop_med"),
            CoordReportLevel::Med
        );
        assert_eq!(
            CoordReportLevel::content("content_medium"),
            CoordReportLevel::Med
        );
        assert_eq!(
            CoordReportLevel::prop("--prop_long--"),
            CoordReportLevel::Long
        );
        assert_eq!(
            CoordReportLevel::content("prop_short"),
            CoordReportLevel::Off
        );
        assert_eq!(
            CoordReportLevel::prop("content_short"),
            CoordReportLevel::Off
        );
    }

    /// ⭐ THE DERIVED `Ord` IS THE PORT OF THE READERS' `> 0`, `> 1` AND `> 2`, so it is asserted
    /// against the authority's integer comparison rather than against hand-picked pairs.
    #[test]
    fn the_levels_order_as_the_readers_compare_them() {
        let levels = [
            CoordReportLevel::Off,
            CoordReportLevel::Short,
            CoordReportLevel::Med,
            CoordReportLevel::Long,
        ];
        for (i, &low) in levels.iter().enumerate() {
            for &high in &levels[i + 1..] {
                assert!(high > low, "{high:?} should outrank {low:?}");
            }
        }
        for &level in &levels {
            assert_eq!(level > CoordReportLevel::Off, level as i32 > 0, "{level:?}");
            assert_eq!(
                level > CoordReportLevel::Short,
                level as i32 > 1,
                "{level:?}"
            );
            assert_eq!(level > CoordReportLevel::Med, level as i32 > 2, "{level:?}");
        }
    }

    /// One token, leading whitespace skipped, extracted to end of input (`util/dtgetenv.hpp:129`).
    #[test]
    fn one_token_parses_and_leading_whitespace_is_skipped() {
        assert_eq!(
            parse_env_token("verify_loopelemoff"),
            Some("verify_loopelemoff")
        );
        assert_eq!(
            parse_env_token("  \t verify_loopelemoff"),
            Some("verify_loopelemoff")
        );
    }

    /// ⛔ A TRAILING SPACE DISCARDS THE WHOLE OPTION: the extractor stops at it without setting
    /// `eofbit`, so `(ss >> parsed) && ss.eof()` is false (`util/dtgetenv.hpp:129`). An all-blank or
    /// empty value extracts no token at all.
    #[test]
    fn a_second_token_a_trailing_space_or_a_blank_value_yields_nothing() {
        assert_eq!(parse_env_token("verify_loopelemoff "), None);
        assert_eq!(parse_env_token("verify_loopelemoff extra"), None);
        assert_eq!(parse_env_token("verify_loopelemoff\n"), None);
        assert_eq!(parse_env_token(""), None);
        assert_eq!(parse_env_token("   "), None);
    }

    /// `ddc/ddc.h:72-76`: the flag is a SUBSTRING test on the parsed token, so a token that merely
    /// contains the word sets it and a truncated one does not.
    #[test]
    fn the_env_flag_matches_the_word_inside_one_token() {
        assert!(
            parse_env_token("xverify_loopelemoffx")
                .is_some_and(|option| option.contains("verify_loopelemoff"))
        );
        assert!(
            !parse_env_token("verify_loopelemof")
                .is_some_and(|option| option.contains("verify_loopelemoff"))
        );
    }

    /// ⛔ POST-INCREMENT (`ddc/ddc_transformation_util.cpp:967`): the first id is 0 and the counter
    /// leads by one. `run_v1` resets it per DSC (`ddc/ddcv1.cpp:3708`), so ids repeat across DSCs.
    #[test]
    fn the_first_latch_data_id_is_zero_and_the_counter_leads_by_one() {
        let mut ddc = ddc();
        assert_eq!(ddc.next_latch_data_id(), LatchDataId(0));
        assert_eq!(ddc.next_latch_data_id(), LatchDataId(1));
        assert_eq!(ddc.latch_data_id_counter, LatchDataId(2));
        ddc.latch_data_id_counter = LatchDataId(0);
        assert_eq!(ddc.next_latch_data_id(), LatchDataId(0));
    }

    /// `ddc/ddc.h:613-614` — four comma-separated values per level, trailing space and all; an empty
    /// vector prints nothing because the loop body never runs.
    #[test]
    fn fold_params_print_as_the_authority_writes_them() {
        let params = vec![
            FoldParamInfoType {
                alpha: Alpha(4),
                beta: Beta(-1),
                cardinality: Cardinality(8),
                fold_dim_label: "elem_arr_0".to_owned(),
            },
            FoldParamInfoType::default(),
        ];
        let mut out = String::new();
        print_fold_params(&params, &mut out);
        assert_eq!(out, "(4, -1, 8, elem_arr_0) (1, 0, 0, ) ");

        out.clear();
        print_fold_params(&[], &mut out);
        assert_eq!(out, "");
    }

    /// `ddc/ddcv1.cpp:17-18` against the initializer list, and ⛔ EVERY ONE IS ALSO A
    /// `dsc2::memories` COMPONENT (`dsc/dscdefn.cpp:142-146`), so the two sets are not complements
    /// and `!is_any_of(storage, registerComponents)` is not "is a memory".
    #[test]
    fn the_seven_register_components_are_all_also_dsc2_memories() {
        assert_eq!(
            REGISTER_COMPONENTS,
            [
                SenComponent::Pelrf,
                SenComponent::Sfplrf,
                SenComponent::Ptarf,
                SenComponent::Ptxrf,
                SenComponent::Pestate,
                SenComponent::Sfpstate,
                SenComponent::Lxluscalereg,
            ]
        );
        for component in REGISTER_COMPONENTS {
            assert!(MEMORIES.contains(&component), "{component:?}");
        }
    }

    /// `} cat = NO_BUNDLING;` (`ddc/ddc.h:562`) is a member initialiser, so the seven sites that
    /// declare a `RowGroupInfo` and nothing else start at the LAST enumerator, not the first.
    #[test]
    fn the_row_group_category_default_is_no_bundling() {
        assert_eq!(RowGroupCategory::default(), RowGroupCategory::NoBundling);
    }

    /// The spellings `RowGroupInfo::print` gives the five categories (`ddc/ddc.h:583-599`); they
    /// differ from the enumerator names in all five arms.
    #[test]
    fn the_category_print_spellings_are_the_authoritys() {
        let same_row = RowGroupCategory::RowToSameRow {
            active_row: RowId(3),
        };
        assert_eq!(
            [
                same_row.print(),
                RowGroupCategory::NonRowToRow.print(),
                RowGroupCategory::RowToNonRow.print(),
                RowGroupCategory::RowNorthSouth.print(),
                RowGroupCategory::NoBundling.print(),
            ],
            [
                "Row-to-SameRow",
                "NonRow-to-Row",
                "Row-to-NonRow",
                "Row-North-South",
                "No-Bundling"
            ]
        );
    }

    /// ⛔ `ddc/ddc_transformation.cpp:87` and `:463` test `!transformationReportLevel_`; the other
    /// forty-one reader lines under `ddc/` test `> 0`, `> 1` or `> 2`. The two readings agree on
    /// every level `runDdc` can supply (`SchedulerStages.cpp:35` passes `0`) and disagree on every
    /// level only the standalone's `-r` can reach.
    #[test]
    fn the_zero_test_at_transformation_cpp_87_is_not_a_greater_than() {
        let off = TransformationReportLevel(0);
        for raw in 0..4 {
            let level = TransformationReportLevel(raw);
            assert_eq!(level.0 != 0, level > off, "{level:?}");
        }
        for raw in -4..0 {
            let level = TransformationReportLevel(raw);
            assert!(level.0 != 0, "the authority builds the message at {level:?}");
            assert!(!(level > off), "a `> Off` reading skips it at {level:?}");
        }
    }
}

// crustify:todo: e025_Ddc

// crustify:todo: e025_Ddc.COMPLETE

// crustify:todo: e025_Ddc.ROLLED_BACK

// crustify:todo: e025_Ddc.ascendingOrder

// crustify:todo: e025_Ddc.beta

// crustify:todo: e025_Ddc.break

// crustify:todo: e025_Ddc.commonGroupAncestor

// crustify:todo: e025_Ddc.continue

// crustify:todo: e025_Ddc.coordPropTracker

// crustify:todo: e025_Ddc.currDsc

// crustify:todo: e025_Ddc.currItemToProcess_

// crustify:todo: e025_Ddc.itemsToProcess_

// crustify:todo: e025_Ddc.loopDistributionParamInfo

// crustify:todo: e025_Ddc.loopsBelowChunkBoundary

// crustify:todo: e025_Ddc.memTrackers

// crustify:todo: e025_Ddc.node

// crustify:todo: e025_Ddc.nodeInfo

// crustify:todo: e025_Ddc.refsAdded_

// crustify:todo: e025_Ddc.retryCount

// crustify:todo: e025_Ddc.row

// crustify:todo: e025_Ddc.sdsc_

// crustify:todo: e025_Ddc.unseenDims
