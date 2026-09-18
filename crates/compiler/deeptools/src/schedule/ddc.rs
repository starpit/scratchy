//! Re-ported from the C++ authority. See crustify-scheduler/AGENT-BRIEF.md.

use crate::schedule::dims::PrimaryDimTypes;
use crate::schedule::dsc2::FoldParamInfoType;
use crate::schedule::metadata::Metadata;
use sys_arch_spec::arch_enums::SenComponent;

/// Replaces: Ddc::registerComponents
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
/// the standalone's `-v` and is handed on to `DdlConvertInterface` (`ddc/ddcv1.cpp:3722`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Verbosity(pub i32);

/// `transformationReportLevel_` (`ddc/ddc.h:40`).
///
/// ⛔ NOT A CLOSED LEVEL SET, unlike [`CoordReportLevel`]: its readers test `> 0`, `> 1` and `> 2`
/// but nothing ever ASSIGNS it a literal — it comes straight from `atoi` on the standalone's `-r`, so
/// a 4 or a -1 stays representable here because it is representable there.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TransformationReportLevel(pub i32);

/// One LATCH's producer/consumer link id, as `latchDataId_` carries it (`dsc/dsc2.h:725`). The `-1`
/// that field defaults to is "not latched" and is not one of these: the counter only counts up.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LatchDataId(pub i32);

/// Which execution step's memory-tracker snapshot to work against — `exphase` (`ddc/ddc.h:101`), set
/// from `runDdc`'s `executionStep` (`dbo/src/Utils/sdsc_bundle/SchedulerStages.cpp:38`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ExPhase(pub i32);

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
/// value never reaches here; `dtGetEnv` rejects it at `'\0' == ptr[0]` (`:129`).
fn parse_env_token(value: &str) -> Option<&str> {
    let rest = value.trim_start_matches(is_stream_space);
    if rest.is_empty() || rest.contains(is_stream_space) {
        return None;
    }
    Some(rest)
}

/// The one `dtGetEnv` call the constructor makes — `dtGetEnv<std::string>("DDCCOORD")`
/// (`ddc/ddc.h:71`).
///
/// ⛔ DIVERGES ON s390x, IN THE ONE DIRECTION WE CAN BUILD. `dtGetEnv` asks `canParseEnvVar` first
/// (`util/dtgetenv.hpp:65-110`), whose `#if PRODUCTION_MODE` branch answers only for its own
/// allowlist plus the `AIU_WORLD_RANK_` prefix; `DDCCOORD` is on neither, so on a production build
/// this variable is invisible and `verify_coordinate_based_loop_elem_off` can never become true. This
/// is the `#else return true` branch (`:108`).
fn ddc_coord_env_option() -> Option<String> {
    let value = std::env::var("DDCCOORD").ok()?;
    parse_env_token(&value).map(str::to_owned)
}

/// The Deep Dataflow Constructor — `class Ddc` (`ddc/ddc.h:34-802`), the pass that turns a
/// `SuperDsc`'s design space configurations into a schedule tree.
///
/// ⛔ THIRTEEN OF THE CLASS'S TWENTY-ONE DECLARED FIELDS, plus the static `registerComponents` as
/// [`REGISTER_COMPONENTS`], so the `e032_Ddc` anchor below stays open. All eight left out are a
/// pointer or a pointer-keyed container, under two blockers.
///
/// Types that are not scheduled units in `crustify-scheduler/UNITS.tsv` at all:
///  * `dscGlobal` (`ddc/ddc.h:37`) — `const DesignSpaceConfigGlobal&`;
///  * `memTrackers` (`ddc/ddc.h:102`) — `MemTrackBundle*`, and note it has NO member initialiser, so
///    a `Ddc` whose caller forgets it holds an indeterminate pointer; both real construction sites
///    assign it at once (`dbo/src/Utils/sdsc_bundle/SchedulerStages.cpp:36`,
///    `ddc/ddc_standalone.cpp:71`);
///  * `sdsc_` (`ddc/ddc.h:106`) — `SuperDsc*`, set to `run_v1`'s argument (`ddc/ddcv1.cpp:3692`);
///  * `currDsc` (`ddc/ddc.h:107`) — `DesignSpaceConfig*`; the target type IS ported
///    ([`crate::schedule::dsc::DesignSpaceConfig`]) but it points INTO `sdsc.dscs_`
///    (`ddc/ddcv1.cpp:3699`), so it cannot be represented without `SuperDsc`.
///
/// Fields needing `dsc2::ScheduleNode` identity, the blocker e013, e016, e018 and e023 also report:
///  * `loopsBelowChunkBoundary` (`ddc/ddc.h:108`) — `unordered_set<const dsc2::LoopNode*>`, filled by
///    a DFS over the tree (`ddc/ddcv1.cpp:3683-3689`);
///  * `coordPropTracker` (`ddc/ddc.h:548`) and all three fields of its nested `CoordPropTracker`
///    (`ddc/ddc.h:536-546`): `itemsToProcess_` is a `deque<dsc2::CoordPropInfoType>`, which holds two
///    `ScheduleNode*` (`dsc/dsc2.h:1089-1090`) and is itself an open anchor in `schedule/dsc2.rs`;
///    `refsAdded_` is keyed by `ScheduleNode*` twice over; and `currItemToProcess_` is a CURSOR INTO
///    `itemsToProcess_`, so its `int` alone would be a field no ported reader could honour;
///  * `loopDistributionParamInfo` (`ddc/ddc.h:550-553`) — a `ScheduleNode*`-keyed map of
///    `ScheduleNode*`-keyed maps;
///  * and the nested `struct RowGroupInfo` (`ddc/ddc.h:555-609`), whose `commonGroupAncestor` is a
///    `dsc2::BlockNode*` and whose every `RowGroupNodeInfo` holds a `ScheduleNode*`.
///
/// Transposing the constructor's two `int` parameters is a compile error, which is why neither is a
/// bare scalar:
/// ```compile_fail
/// use deeptools::schedule::ddc::{Ddc, TransformationReportLevel, Verbosity};
/// let _ = Ddc::new(TransformationReportLevel(0), false, Verbosity(0), "");
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ddc {
    /// Field: e032_Ddc.verbose_
    ///
    /// `ddc/ddc.h:38` — the one field with no member initialiser that the constructor always sets.
    pub verbose: Verbosity,

    /// Field: e032_Ddc.latchDataIdCounter_
    ///
    /// `ddc/ddc.h:39`. Holds the NEXT id to hand out, not the last one handed out. `run_v1` resets it
    /// per DSC (`ddc/ddcv1.cpp:3708`); [`Ddc::next_latch_data_id`] is its only consumer.
    pub latch_data_id_counter: LatchDataId,

    /// Field: e032_Ddc.transformationReportLevel_
    ///
    /// `ddc/ddc.h:40` — a constructor parameter, never derived from the coordinate option string.
    pub transformation_report_level: TransformationReportLevel,

    /// Field: e032_Ddc.coordFoldReportLevel_
    ///
    /// `ddc/ddc.h:41`, from the option string's `content_*` spelling ([`CoordReportLevel::content`]).
    /// Read by the fold pass; `coordinateCapture` (`ddc/ddc_fold.cpp:1538-1622`) is representative.
    pub coord_fold_report_level: CoordReportLevel,

    /// Field: e032_Ddc.coordPropReportLevel_
    ///
    /// `ddc/ddc.h:42`, from the SAME string's `prop_*` spelling ([`CoordReportLevel::prop`]). Around
    /// forty sites in `ddc/ddc_fold.cpp` read it, which is why it is a second level and not one knob.
    pub coord_prop_report_level: CoordReportLevel,

    /// Field: e032_Ddc.verifyCoordinateBasedLoopElemOff
    ///
    /// `ddc/ddc.h:43` — the ONLY field driven by an environment variable rather than a parameter
    /// ([`ddc_coord_env_option`]), with one reader: `ddc/ddcv1.cpp:2465` takes the coordinate-derived
    /// element-offset path when this OR `datastage_based_elem_off` is set.
    pub verify_coordinate_based_loop_elem_off: bool,

    /// Field: e032_Ddc.datastageBasedElemOff
    ///
    /// `ddc/ddc.h:44`. NOT a constructor parameter: `run_v1` latches it true per `SuperDsc`, never
    /// back to false, when any compute op is a `ReStickifyOpLx` or `ReStickifyOpHBM`, and MIRRORS it
    /// onto `sdsc.datastageBasedElemOff` in the same breath (`ddc/ddcv1.cpp:3709-3715`) so
    /// `dsc/dsc2.cpp:3034` can read it. Ten sites switch element-offset derivation on it.
    pub datastage_based_elem_off: bool,

    /// Field: e032_Ddc.dscToDdl_
    ///
    /// `ddc/ddc.h:45`. Constructor parameter; its one reader dumps the converted DDL to stdout at the
    /// end of `run_v1` (`ddc/ddcv1.cpp:3796`). `runDdc` passes false (`SchedulerStages.cpp:35`).
    pub dsc_to_ddl: bool,

    /// Field: e032_Ddc.trueLXTracker_
    ///
    /// `ddc/ddc.h:46-47`. Set true only by `runDdc` (`SchedulerStages.cpp:37`), never by the
    /// standalone, and read once: `trueLXTracker_ || comp != SenComponents::LX` decides whether LX
    /// participates in memory tracking (`ddc/ddcv1.cpp:186`). So the standalone tracks LX differently
    /// from the real bundle path.
    pub true_lx_tracker: bool,

    /// Field: e032_Ddc.exphase
    ///
    /// `ddc/ddc.h:101`; `None` is the authority's `-1`.
    ///
    /// ⛔ `-1` IS SILENTLY INERT, NOT OUT OF RANGE, WHICH IS WHY THIS IS AN `Option`: every tracker
    /// call keyed by it opens with a lookup-miss guard, so `backupEps`
    /// (`util/memtracker/mem_track.cpp:566`) returns an EMPTY vector and `restoreEps` (`:580`)
    /// returns at once. A `Ddc` left at the default backs up and restores nothing and still
    /// completes, so the unset case has to be one a reader cannot forget.
    pub exphase: Option<ExPhase>,

    /// Field: e032_Ddc.metadata
    ///
    /// `ddc/ddc.h:105`. Per-DSC scratch: `run_v1` clears it at the top of every DSC iteration
    /// (`ddc/ddcv1.cpp:3705`), which is [`Metadata::clear`].
    pub metadata: Metadata,

    /// Field: e032_Ddc.coreletSplitDim
    ///
    /// `ddc/ddc.h:109`. [`PrimaryDimTypes::Undefined`] is the authority's `PrimaryDimTypesCount`
    /// initialiser and means "no corelet split", exactly as its readers test it
    /// (`ddc/ddc_fold.cpp:2480`, `:3121`). `initGlobalData` recomputes it per DSC: back to the
    /// sentinel, then the FIRST key of the core stage's `coreletSplit_` if non-empty
    /// (`ddc/ddcv1.cpp:3673-3681`).
    pub corelet_split_dim: PrimaryDimTypes,

    /// Field: e032_Ddc.dataStageExplorationDone_
    ///
    /// `ddc/ddc.h:112`. A one-way phase latch WITHIN one DSC: false at the top of each
    /// (`ddc/ddcv1.cpp:3707`), true once `exploreAssignDataStages` finishes (`ddc/ddcv1.cpp:556`).
    /// Seven sites in `ddc/ddc_transformation{,_util}.cpp` branch on it, so the same transformation
    /// behaves differently before and after — a phase, not a flag.
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

/// `Ddc::printFoldParams` (`ddc/ddc.h:611-617`), appending to a `String` rather than writing to a
/// stream, as [`crate::schedule::dims::PrimaryDimTypes::print`] does. A free function because the
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

    /// One token, leading whitespace skipped, extracted to end of input (`util/dtgetenv.hpp:132`).
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
    /// `eofbit`, so `(ss >> parsed) && ss.eof()` is false (`util/dtgetenv.hpp:132`). An all-blank or
    /// empty value extracts no token at all.
    #[test]
    fn a_second_token_a_trailing_space_or_a_blank_value_yields_nothing() {
        assert_eq!(parse_env_token("verify_loopelemoff "), None);
        assert_eq!(parse_env_token("verify_loopelemoff extra"), None);
        assert_eq!(parse_env_token("verify_loopelemoff\n"), None);
        assert_eq!(parse_env_token(""), None);
        assert_eq!(parse_env_token("   "), None);
    }

    /// `ddc/ddc.h:71-76`: the flag is a SUBSTRING test on the parsed token, so a token that merely
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

    /// `ddc/ddc.h:613-616` — four comma-separated values per level, trailing space and all; an empty
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
}
