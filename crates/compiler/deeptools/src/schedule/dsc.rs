//! Re-ported from the C++ authority. See crustify-scheduler/AGENT-BRIEF.md.

use crate::schedule::dims::{self, DataStructDims, PrimaryDimTypes};
use crate::schedule::dsc2::{
    self, ConstantId, ConstantInfo, DataStage, DataStageId, GroupId, MaskSplit, SLICES_PER_STICK,
    Size, VariableSymbol,
};
use std::collections::{BTreeMap, BTreeSet};
use sys_arch_spec::{CoreId, arch_enums::OpFunc};

/// How many cores this DSC's work is spread over — `numCoresUsed_` (`dsc/designSpaceConfig.h:73`),
/// which the DDL constraint check compares against an op's minimum (`ddc/ddl/ddl_conversion.cpp:2559`)
/// and the L3 scheduler multiplies flops by (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:2294`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NumCoresUsed(pub u32);

/// How many corelets of each core are used — `numCoreletsUsed_` (`dsc/designSpaceConfig.h:74`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NumCoreletsUsed(pub u32);

/// One loop's trip count, as `getLoopCount` divides one stage's dim by the next's — IBM's
/// `int loopCount` (`dsc/designSpaceConfig.cpp:416-427`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LoopCount(pub i32);

/// One `paramNameToVal` value as its callers read it: the map holds `double*` INTO this object's own
/// `DataStructDims` members (`dsc/designSpaceConfig.h:358-609`), and every dim member is born `-1`
/// (`dsc/dims.h:160-193`).
///
/// ⛔ AN UNFILLED DIM IS THE VALUE `-1`, NOT AN ABSENCE. Measured on a default-constructed
/// `DesignSpaceConfig`, all 240 keys read `-1` and none throws — so `getLoopCount` answers
/// `int(-1 / -1) == 1` for all 29 stage loops and `getInpInHBM` answers `1` for all twelve labels.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct ParamVal(pub f64);

impl ParamVal {
    /// What a dim reads before anyone fills it (`dsc/dims.h:160-193`).
    pub const UNFILLED: Self = Self(-1.0);
}

/// What `getLoopCount` answers (`dsc/designSpaceConfig.cpp:416-427`), keeping IBM's three outcomes
/// apart the way [`LayoutOrderPosition`] does for `getDimIndexInLayoutOrder`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoopTripCount {
    /// The quotient IBM computes and truncates towards zero (`:425-426`).
    Is(LoopCount),
    /// The denominator stage's dim is `0`, where IBM's `int(x / 0.0)` is UNDEFINED — measured as
    /// `2147483647`, which is not a trip count.
    UndefinedZeroDenominator,
    /// Neither key the loop's spelling builds is in `paramNameToVal`, so IBM stopped before
    /// dividing: `INNER` on its own `DT_CHECK` (`:417`), `CONST` and `INVALID` on `.at()`.
    NoSuchLoop,
}

/// One stick dim's extent in ELEMENTS — the `double` of `PrimaryDsInfo::stickSize_`
/// (`dsc/dscdefn.h:478`), which `getStickSizes` truncates to an `int` on the way out
/// (`dsc/dsc2.cpp:4088`).
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct StickSize(pub f64);

/// How many times one stick dim's data is replicated — `PrimaryDsInfo::stickRepl_`
/// (`dsc/dscdefn.h:479`). ⛔ NOT A SIZE: `get_stick_srpdt` multiplies the two together
/// (`dsc/designSpaceConfig.cpp:9483-9486`), so transposing them must not compile.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StickRepl(pub i32);

/// A stick dim's extent times its replication — what `get_stick_srpdt` returns
/// (`dsc/designSpaceConfig.cpp:9483-9486`).
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct StickSizeWithRepl(pub f64);

/// One loop's scale — `LoopProperties::scale_` (`dsc/dscdefn.h:317-319`).
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct LoopScale(pub f64);

/// How many L0 slices one stick is cut into — `getStickSizes`' `numL0Slices` parameter
/// (`dsc/dsc2.cpp:4069`).
///
/// ⛔ CONSTRUCTION REFUSES A NON-POSITIVE, which is IBM's
/// `DT_CHECK_MSG(!l0SliceOnly || numL0Slices > 0, "If l0SliceOnly requested, numL0Sclides must be
/// provided")` (`dsc/dsc2.cpp:4074-4075`) moved to the type: the `-1` default is unrepresentable
/// inside [`StickSizeScope::L0SliceOnly`], so that check cannot be reached.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NumL0Slices(i32);

impl NumL0Slices {
    /// A slice count, or absent for the non-positive IBM aborts on.
    pub fn new(slices: i32) -> Option<Self> {
        (slices > 0).then_some(Self(slices))
    }

    /// The stored count.
    pub const fn get(self) -> i32 {
        self.0
    }
}

/// A tensor's row count in HBM minus its zero padding — `getInpRowInHBM`
/// (`dsc/designSpaceConfig.cpp:930-934`).
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct HbmRows(pub f64);

/// A tensor's column count in HBM minus its zero padding — `getInpColInHBM`
/// (`dsc/designSpaceConfig.cpp:936-940`).
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct HbmCols(pub f64);

/// A tensor's element count in HBM — `getInpInHBM`, rows times columns (or `-1` when both are
/// negative) (`dsc/designSpaceConfig.cpp:942-946`).
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct HbmElements(pub f64);

/// The role one labeled data structure plays in an op (`dsc/dscdefn.h:37-46`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DsTypes {
    Input,
    Output,
    Kernel,
    KernelIdx,
    InputScale,
    KernelScale,
    Internal,
    /// `LabeledDsInfo::dsType_`'s own initialiser (`dsc/dscdefn.h:327`).
    #[default]
    NotSet,
}

impl DsTypes {
    /// The keys of `dsTypeToString`, in the enum's order — which is the order its `std::map`
    /// iterates them in, and NOT its initialiser's: that one lists `INTERNAL` fifth
    /// (`dsc/designSpaceConfig.cpp:9255-9263` against `dsc/dscdefn.h:37-46`).
    pub const ALL: [Self; 8] = [
        Self::Input,
        Self::Output,
        Self::Kernel,
        Self::KernelIdx,
        Self::InputScale,
        Self::KernelScale,
        Self::Internal,
        Self::NotSet,
    ];

    /// Field: e027_DesignSpaceConfig.dsTypeToString
    ///
    /// `dsTypeToString` (`dsc/designSpaceConfig.cpp:9255-9263`).
    pub const fn name(self) -> &'static str {
        match self {
            Self::Input => "INPUT",
            Self::Output => "OUTPUT",
            Self::Kernel => "KERNEL",
            Self::KernelIdx => "KERNEL_IDX",
            Self::InputScale => "INPUT_SCALE",
            Self::KernelScale => "KERNEL_SCALE",
            Self::Internal => "INTERNAL",
            Self::NotSet => "NOT_SET",
        }
    }

    /// Field: e027_DesignSpaceConfig.stringToDsType
    ///
    /// `stringToDsType`, which is `flipMap(dsTypeToString)` (`dsc/designSpaceConfig.cpp:9264-9265`) —
    /// so an unknown spelling is absent, where IBM's `.at()` throws.
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|ds| ds.name() == name)
    }
}

/// How a data transfer is staged through the memory hierarchy — `enum DtType` (`dsc/dscdefn.h:214`).
///
/// ⛔ NO [`Default`]: `DtInfo::type` is declared bare, with no member initialiser
/// (`dsc/dscdefn.h:257`), so the authority cannot produce one either.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DtType {
    DblBuff,
    BlkLoad,
    Streaming,
}

impl DtType {
    /// The keys of `dtTypeToString`, in the enum's order (`dsc/dscdefn.h:214`).
    pub const ALL: [Self; 3] = [Self::DblBuff, Self::BlkLoad, Self::Streaming];

    /// Field: e027_DesignSpaceConfig.dtTypeToString
    ///
    /// `dtTypeToString` (`dsc/designSpaceConfig.cpp:9266-9269`).
    pub const fn name(self) -> &'static str {
        match self {
            Self::DblBuff => "dblbuff",
            Self::BlkLoad => "blkload",
            Self::Streaming => "streaming",
        }
    }

    /// Field: e027_DesignSpaceConfig.stringToDtType
    ///
    /// `stringToDtType`, which is `flipMap(dtTypeToString)` (`dsc/designSpaceConfig.cpp:9270-9271`) —
    /// so an unknown spelling is absent, where IBM's `.at()` throws.
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|dt| dt.name() == name)
    }
}

/// An op a DSC hands to Sen rather than computing itself — `enum class ExternalSenOps`
/// (`dsc/dscdefn.h:216-225`).
///
/// The discriminants are the authority's: a band starting at `-1024`, below every `OpFuncs` value
/// (`sys-arch-spec/arch_enums.h:135-136`, `NONE = 0`). Nothing reads them numerically — tree-wide,
/// `ExternalSenOps` occurs only in that declaration and in the two tables below.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(i32)]
pub enum ExternalSenOps {
    SenCompute = -1024,
    SenConst,
    SenHostSend,
    SenSenSend,
    SenHostRecv,
    SenSenRecv,
    SenDataPrep,
    SenBnPrecompute,
}

/// ⛔ E0080 IF THE BAND MOVES: `SENCOMPUTE` opens the enum at `-1024` and `SENBNPRECOMPUTE` closes
/// it at `-1017` (`dsc/dscdefn.h:217-224`).
const _: [(); 1024] = [(); -(ExternalSenOps::SenCompute as i32) as usize];
const _: [(); 1017] = [(); -(ExternalSenOps::SenBnPrecompute as i32) as usize];

impl ExternalSenOps {
    /// The keys of `exSenOpsToString`, in the enum's order — which is the order its `std::map`
    /// iterates them in, and NOT its initialiser's: that one lists `SENDATAPREP` third
    /// (`dsc/designSpaceConfig.cpp:9272-9281`).
    pub const ALL: [Self; 8] = [
        Self::SenCompute,
        Self::SenConst,
        Self::SenHostSend,
        Self::SenSenSend,
        Self::SenHostRecv,
        Self::SenSenRecv,
        Self::SenDataPrep,
        Self::SenBnPrecompute,
    ];

    /// Field: e027_DesignSpaceConfig.exSenOpsToString
    ///
    /// `exSenOpsToString` (`dsc/designSpaceConfig.cpp:9272-9281`) — the printed name is the Sen op's,
    /// not the enumerator's: `SENCOMPUTE` prints `SenPreparedOp`, `SENDATAPREP` `SenDataConvert` and
    /// `SENBNPRECOMPUTE` `SenFusedBatchNormPrecompute`.
    pub const fn name(self) -> &'static str {
        match self {
            Self::SenCompute => "SenPreparedOp",
            Self::SenConst => "SenConst",
            Self::SenHostSend => "SenHostSend",
            Self::SenSenSend => "SenSenSend",
            Self::SenHostRecv => "SenHostRecv",
            Self::SenSenRecv => "SenSenRecv",
            Self::SenDataPrep => "SenDataConvert",
            Self::SenBnPrecompute => "SenFusedBatchNormPrecompute",
        }
    }

    /// Field: e027_DesignSpaceConfig.stringToExSenOps
    ///
    /// `stringToExSenOps`, which is `flipMap(exSenOpsToString)` (`dsc/designSpaceConfig.cpp:9282-9283`)
    /// — so an unknown spelling is absent, where IBM's `.at()` throws.
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|op| op.name() == name)
    }
}

/// Every loop the DSC language names, in the authority's order (`dsc/dscdefn.h:48-92`).
///
/// ⛔ THE DISCRIMINANTS ARE LOAD-BEARING. `checkLoopStage` answers by RANGE COMPARISON against the
/// `FIRST_*`/`LAST_*` aliases (`dsc/designSpaceConfig.cpp:8302-8315`), so inserting or dropping a
/// loop silently re-stages the ones after it — see [`LoopNames::is_stage`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LoopNames {
    Inner = 0,
    DbIn = 1,
    DbOut,
    DbIj,
    DbI,
    DbJ,
    DbMb,
    DbKij,
    DbKi,
    DbKj,
    DbX,
    DbY,
    BtIn = 12,
    BtOut,
    BtIj,
    BtI,
    BtJ,
    BtMb,
    BtKij,
    BtX,
    BtY,
    TpIn = 21,
    TpOut,
    TpIj,
    TpI,
    TpJ,
    TpMb,
    TpKij,
    TpX,
    TpY,
    /// The special loop for const offsets (`dsc/dscdefn.h:90`).
    Const = 30,
    Invalid = 31,
}

/// ⛔ E0080 IF A LOOP IS INSERTED OR DROPPED: the DB block starts at 1 and `INVALID` closes the enum
/// at 31 (`dsc/dscdefn.h:48-92`), and the stage ranges are read off those values.
const _: [(); 1] = [(); LoopNames::DbIn as usize];
const _: [(); 31] = [(); LoopNames::Invalid as usize];

/// Which of the three loop stages a loop belongs to — `checkLoopStage`'s own `LoopStage` parameter
/// and its comment, "LoopStage 0-tp | 1-bt | 2-db" (`dsc/designSpaceConfig.cpp:8301-8315`).
///
/// ⛔ AN ENUM, SO THE `default:` ARM IS UNREACHABLE: IBM answers `false` for every other `int` and
/// then cannot reach its own `DT_ERROR_FMT("DSC-checkLoopStage, Bad LoopStage: %d")` (`:8313-8315`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LoopStage {
    Tp,
    Bt,
    Db,
}

impl LoopNames {
    /// `FIRST_DB_LOOP` (`dsc/dscdefn.h:62`).
    pub const FIRST_DB_LOOP: Self = Self::DbIn;
    /// `LAST_DB_LOOP` (`dsc/dscdefn.h:63`).
    pub const LAST_DB_LOOP: Self = Self::DbY;
    /// `FIRST_BT_LOOP` (`dsc/dscdefn.h:75`).
    pub const FIRST_BT_LOOP: Self = Self::BtIn;
    /// `LAST_BT_LOOP` (`dsc/dscdefn.h:76`).
    pub const LAST_BT_LOOP: Self = Self::BtY;
    /// `FIRST_TP_LOOP` (`dsc/dscdefn.h:88`).
    pub const FIRST_TP_LOOP: Self = Self::TpIn;
    /// `LAST_TP_LOOP` (`dsc/dscdefn.h:89`).
    pub const LAST_TP_LOOP: Self = Self::TpY;

    /// The keys of `loopNameToString`, in the enum's order (`dsc/designSpaceConfig.cpp:9222-9238`).
    pub const ALL: [Self; 32] = [
        Self::Inner,
        Self::DbIn,
        Self::DbOut,
        Self::DbIj,
        Self::DbI,
        Self::DbJ,
        Self::DbMb,
        Self::DbKij,
        Self::DbKi,
        Self::DbKj,
        Self::DbX,
        Self::DbY,
        Self::BtIn,
        Self::BtOut,
        Self::BtIj,
        Self::BtI,
        Self::BtJ,
        Self::BtMb,
        Self::BtKij,
        Self::BtX,
        Self::BtY,
        Self::TpIn,
        Self::TpOut,
        Self::TpIj,
        Self::TpI,
        Self::TpJ,
        Self::TpMb,
        Self::TpKij,
        Self::TpX,
        Self::TpY,
        Self::Const,
        Self::Invalid,
    ];

    /// Field: e027_DesignSpaceConfig.loopNameToString
    ///
    /// `loopNameToString` (`dsc/designSpaceConfig.cpp:9222-9238`).
    ///
    /// ⛔ THE SPELLING IS PARSED, NOT JUST PRINTED: [`loop_count`](DesignSpaceConfig::loop_count)
    /// cuts this string into a numerator stage, a denominator stage and a dim
    /// (`dsc/designSpaceConfig.cpp:417-424`), which is why `INNER`'s odd capital `"Inner"` matters
    /// and why `DBKI`/`DBKJ` have no BT or TP twin.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Inner => "Inner",
            Self::DbIn => "dbin",
            Self::DbOut => "dbout",
            Self::DbIj => "dbij",
            Self::DbI => "dbi",
            Self::DbJ => "dbj",
            Self::DbMb => "dbmb",
            Self::DbKij => "dbkij",
            Self::DbKi => "dbki",
            Self::DbKj => "dbkj",
            Self::DbX => "dbx",
            Self::DbY => "dby",
            Self::BtIn => "btin",
            Self::BtOut => "btout",
            Self::BtIj => "btij",
            Self::BtI => "bti",
            Self::BtJ => "btj",
            Self::BtMb => "btmb",
            Self::BtKij => "btkij",
            Self::BtX => "btx",
            Self::BtY => "bty",
            Self::TpIn => "tpin",
            Self::TpOut => "tpout",
            Self::TpIj => "tpij",
            Self::TpI => "tpi",
            Self::TpJ => "tpj",
            Self::TpMb => "tpmb",
            Self::TpKij => "tpkij",
            Self::TpX => "tpx",
            Self::TpY => "tpy",
            Self::Const => "const",
            Self::Invalid => "invalid",
        }
    }

    /// Field: e027_DesignSpaceConfig.stringToLoopName
    ///
    /// `stringToLoopName`, which is `flipMap(loopNameToString)`
    /// (`dsc/designSpaceConfig.cpp:9239-9240`).
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|loop_name| loop_name.name() == name)
    }

    /// Field: e027_DesignSpaceConfig.stringToLoopNameDm
    ///
    /// `stringToLoopNameDm`, the DM's own spellings — a SEPARATE table, not the flip of
    /// [`name`](Self::name) (`dsc/designSpaceConfig.cpp:9241-9254`).
    ///
    /// ⛔ IT NAMES NEITHER `DBI`/`DBJ`/`DBKI`/`DBKJ` NOR THEIR BT/TP TWINS, and it maps TWO
    /// spellings — `"pcompute"` and `"Inner"` — onto [`Inner`](Self::Inner), so it is not
    /// invertible.
    pub fn from_name_dm(name: &str) -> Option<Self> {
        Some(match name {
            "pcompute" | "Inner" => Self::Inner,
            "din" => Self::DbIn,
            "dout" => Self::DbOut,
            "dij" => Self::DbIj,
            "dmb" => Self::DbMb,
            "dkij" => Self::DbKij,
            "dx" => Self::DbX,
            "dy" => Self::DbY,
            "bin" => Self::BtIn,
            "bout" => Self::BtOut,
            "bij" => Self::BtIj,
            "bmb" => Self::BtMb,
            "bkij" => Self::BtKij,
            "bx" => Self::BtX,
            "by" => Self::BtY,
            "tin" => Self::TpIn,
            "tout" => Self::TpOut,
            "tij" => Self::TpIj,
            "tmb" => Self::TpMb,
            "tkij" => Self::TpKij,
            "tx" => Self::TpX,
            "ty" => Self::TpY,
            _ => return None,
        })
    }

    /// Whether this loop is one of the given stage's — `DesignSpaceConfig::checkLoopStage`
    /// (`dsc/designSpaceConfig.cpp:8302-8315`), which reads no field and so lives on the loop.
    pub fn is_stage(self, stage: LoopStage) -> bool {
        let (first, last) = match stage {
            LoopStage::Tp => (Self::FIRST_TP_LOOP, Self::LAST_TP_LOOP),
            LoopStage::Bt => (Self::FIRST_BT_LOOP, Self::LAST_BT_LOOP),
            LoopStage::Db => (Self::FIRST_DB_LOOP, Self::LAST_DB_LOOP),
        };
        first <= self && self <= last
    }
}

/// Whether the primary data structures share their reuse (`dsc/dscdefn.h:470-472`). DSI branches on
/// it (`dsi/dsi.cpp:1789`, `:2095`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrimaryDsRelationInfo {
    /// `isPdsReuse` (`dsc/dscdefn.h:471`), whose member initialiser is `true`.
    pub is_pds_reuse: bool,
}

impl Default for PrimaryDsRelationInfo {
    /// `dsc/dscdefn.h:471`: reuse is the default, so this is NOT [`bool::default`].
    fn default() -> Self {
        Self { is_pds_reuse: true }
    }
}

/// One loop's properties (`dsc/dscdefn.h:317-319`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LoopProperties {
    /// `scale_` (`dsc/dscdefn.h:318`). ⛔ ABSENT UNTIL WRITTEN: the authority declares a bare
    /// `double scale_;` with no member initialiser, so a default-constructed one holds an
    /// indeterminate value; DM assigns it per loop (`dm/dm.cpp:1446`).
    pub scale: Option<LoopScale>,
}

/// How one primary data structure is laid out and sticked (`dsc/dscdefn.h:474-480`).
///
/// ⛔ THE FOUR VECTORS ARE INDEX-PARALLEL, and every reader relies on it: `get_stick` and
/// `get_stick_repl` refuse a length mismatch (`dsc/designSpaceConfig.cpp:9452-9455`, `:9469-9473`)
/// and `getStickSizes` reads `stickSize_.at(i)` for each `stickDimOrder_[i]` (`dsc/dsc2.cpp:4088`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PrimaryDsInfo {
    /// `layoutDimOrder_` (`dsc/dscdefn.h:476`) — the dim order of the allocation, which
    /// [`dim_index_in_layout_order`](DesignSpaceConfig::dim_index_in_layout_order) positions a dim in.
    pub layout_dim_order: Vec<PrimaryDimTypes>,
    /// `stickDimOrder_` (`dsc/dscdefn.h:477`) — the dims INSIDE one stick, outermost first.
    pub stick_dim_order: Vec<PrimaryDimTypes>,
    /// `stickSize_` (`dsc/dscdefn.h:478`), one extent per entry of
    /// [`stick_dim_order`](Self::stick_dim_order).
    pub stick_size: Vec<StickSize>,
    /// `stickRepl_` (`dsc/dscdefn.h:479`), one replication factor per entry of
    /// [`stick_dim_order`](Self::stick_dim_order).
    pub stick_repl: Vec<StickRepl>,
}

impl PrimaryDsInfo {
    /// The product of one dim's stick extents — `DesignSpaceConfig::get_stick`
    /// (`dsc/designSpaceConfig.cpp:9451-9464`). A dim that is not in the stick answers `1.0`, as
    /// IBM's empty product does; absent is its length-mismatch `DT_ERROR`.
    pub fn stick(&self, stick_dim: PrimaryDimTypes) -> Option<StickSize> {
        if self.stick_size.len() != self.stick_dim_order.len() {
            return None;
        }
        let mut total = 1.0;
        for (dim, size) in self.stick_dim_order.iter().zip(&self.stick_size) {
            if *dim == stick_dim {
                total *= size.0;
            }
        }
        Some(StickSize(total))
    }

    /// The product of one dim's replication factors — `DesignSpaceConfig::get_stick_repl`
    /// (`dsc/designSpaceConfig.cpp:9466-9481`), absent on its length-mismatch `DT_ERROR`.
    pub fn stick_repl(&self, stick_dim: PrimaryDimTypes) -> Option<StickRepl> {
        if self.stick_repl.len() != self.stick_dim_order.len() {
            return None;
        }
        let mut repl = 1;
        for (dim, r) in self.stick_dim_order.iter().zip(&self.stick_repl) {
            if *dim == stick_dim {
                repl *= r.0;
            }
        }
        Some(StickRepl(repl))
    }

    /// [`stick`](Self::stick) times [`stick_repl`](Self::stick_repl) —
    /// `DesignSpaceConfig::get_stick_srpdt` (`dsc/designSpaceConfig.cpp:9483-9486`).
    pub fn stick_srpdt(&self, stick_dim: PrimaryDimTypes) -> Option<StickSizeWithRepl> {
        let size = self.stick(stick_dim)?;
        let repl = self.stick_repl(stick_dim)?;
        Some(StickSizeWithRepl(size.0 * f64::from(repl.0)))
    }
}

/// Which part of a stick [`stick_sizes`](DesignSpaceConfig::stick_sizes) reports — IBM's three
/// mutually exclusive `bool` parameters plus the slice count the third needs (`dsc/dsc2.cpp:4066-4070`).
///
/// ⛔ THE EXCLUSIVITY IS THE TYPE, so `DT_CHECK_MSG((stickSliceOnly + stickWithoutSlice +
/// l0SliceOnly) < 2, "... can not be true at same time")` (`dsc/dsc2.cpp:4071-4074`) cannot be
/// reached: the four combinations IBM accepts are the four variants and the rest do not exist.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StickSizeScope {
    /// All three flags false: every stick dim with its full extent.
    WholeStick,
    /// `stickSliceOnly`: only the dims that fit inside one of the stick's eight slices.
    SliceOnly,
    /// `stickWithoutSlice`: only what is left once one slice is filled.
    WithoutSlice,
    /// `l0SliceOnly` with its `numL0Slices`, which replaces the eight
    /// (`dsc/dsc2.cpp:4082`).
    L0SliceOnly(NumL0Slices),
}

/// The twelve `DataStructDims` prefixes `paramNameToVal`'s keys are built from, LONGEST FIRST
/// (`dsc/designSpaceConfig.h:358-609`). They are exactly `getDsdFromStr`'s twelve spellings
/// (`:317-346`), which are exactly the names the constructor assigns
/// (`dsc/designSpaceConfig.cpp:16-28`).
const DSD_PREFIXES: [&str; 12] = [
    "chipletd", "coreletd", "unpadn", "chipd", "dscn", "tel", "pel", "b", "d", "n", "p", "t",
];

/// The twenty dim names `paramNameToVal` pairs with every prefix (`dsc/designSpaceConfig.h:358-609`).
///
/// ⛔ `x1` IS NOT ONE OF THEM, though `DataStructDims` has the field and its own
/// `param_name_to_val` accepts the name (`dsc/dims.cpp:437-482`): 12 x 20 = 240 keys, and no
/// `<dsd>x1` key exists. A resolver that fell through to the dim table would answer for `"nx1"`
/// where IBM's `.at()` throws.
const PARAM_DIM_NAMES: [&str; 20] = [
    "in", "out", "mb", "i", "j", "ij", "ki", "kj", "kij", "x", "y", "r", "c", "rc", "si", "sj",
    "sij", "zi", "zj", "zij",
];

/// One `paramNameToVal` key cut into its prefix and its dim (`dsc/designSpaceConfig.h:358-609`).
/// Longest prefix first, and a prefix whose remainder is not a dim name is not the split.
fn split_param_name(name: &str) -> Option<(&'static str, &'static str)> {
    DSD_PREFIXES.into_iter().find_map(|prefix| {
        let rest = name.strip_prefix(prefix)?;
        PARAM_DIM_NAMES
            .into_iter()
            .find(|dim| *dim == rest)
            .map(|dim| (prefix, dim))
    })
}

/// The dim one of [`PARAM_DIM_NAMES`] selects, read (`dsc/designSpaceConfig.h:358-609`) — unfilled
/// or not, because IBM reads it through a `double*` that cannot be absent.
fn dim_val_by_name(dsd: &DataStructDims, dim: &str) -> Option<ParamVal> {
    let filled = match dim {
        "in" => dsd.r#in,
        "out" => dsd.out,
        "mb" => dsd.mb,
        "i" => dsd.i,
        "j" => dsd.j,
        "ij" => dsd.ij,
        "ki" => dsd.ki,
        "kj" => dsd.kj,
        "kij" => dsd.kij,
        "x" => dsd.x,
        "y" => dsd.y,
        "r" => dsd.r,
        "c" => dsd.c,
        "rc" => dsd.rc,
        "si" => dsd.si,
        "sj" => dsd.sj,
        "sij" => dsd.sij,
        "zi" => dsd.zi,
        "zj" => dsd.zj,
        "zij" => dsd.zij,
        _ => return None,
    };
    Some(filled.map_or(ParamVal::UNFILLED, |size| ParamVal(size.get())))
}

/// IBM's `isFractional` (`dsc/designSpaceConfig.cpp:7812`).
///
/// ⛔ THE NAME IS INVERTED FROM ITS MEANING: it is `floor(val) == val`, so it answers TRUE for a
/// whole number, and every caller negates it to mean "has a fraction". An absent dim is IBM's `-1`,
/// whose floor is itself, so it answers true as well.
fn is_fractional(val: Option<dims::DimSize>) -> bool {
    match val {
        None => true,
        Some(val) => val.get().floor() == val.get(),
    }
}

/// What the program frame is filled for (`util/sendefs/sendefs.h:177-188`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SenTargets {
    /// `DesignSpaceConfig::target_`'s own initialiser (`dsc/designSpaceConfig.h:121`).
    #[default]
    Undefined,
    Sentient,
    Senulator,
    SenPcfg,
    SenTf,
    SystemC,
    R5ss,
    Host,
    Invalid,
    Nop,
}

/// Where one dim sits in one role's layout order, as
/// [`dim_index_in_layout_order`](DesignSpaceConfig::dim_index_in_layout_order) answers it
/// (`dsc/designSpaceConfig.cpp:429-438`).
///
/// ⛔ IBM'S ONE `int` CARRIES TWO OUTCOMES AND ITS TWENTY-EIGHT LIVE READERS DO NOT AGREE ON THE
/// `-1`, so collapsing the two into a single absent loses a distinction the tree draws five ways.
/// SIXTEEN substitute A SCALE OF `1` for it and carry straight on — `dimIdx < 0 ? 1 : scale_.at(..)`
/// (`ddc/ddc_fold.cpp:1390-1391`, `:2297-2298`, `:2535-2536`, `:3000-3001`, `:3160-3161`,
/// `:3365-3366`, `:3964-3965`, `:3980-3981`, `:4279-4280`, `:4289-4290`, `ddc/ddcv1.cpp:2450-2451`,
/// `:2631-2632`, `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:6032-6033`, `:6202-6203`, `:7399-7400`,
/// `:7541-7542`). TWO abort on it (`L3DlOpsScheduler.cpp:68-71`, `:4271-4273`). TWO ask only whether
/// it is negative (`ddc/ddl/ddl_conversion.cpp:1302-1305`, `ddc/ddc_transformation.cpp:1792-1793`).
/// ONE adds it to a dim count and stores the sum as a position (`ddc/ddcv1.cpp:1588-1590`). And
/// SEVEN hand it straight to `scale_.at()`, where it wraps to a huge `size_t` and throws exactly as
/// the role lookup would (`ddc/ddcv1.cpp:500-501`, `:1502-1503`, `:1904-1905`,
/// `ddc/ddc_transformation.cpp:929`, `:931`, `dsc/dsc2.cpp:3559`, `:3820`). The commented-out `== -2`
/// compare (`ddc/ddcv1.cpp:1522-1532`) and the six `dvs/setupVariables/` fixtures are not counted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayoutOrderPosition {
    /// The position the search found (`dsc/designSpaceConfig.cpp:433-435`).
    At(usize),
    /// The search ran off the end of `layoutDimOrder_`, which is IBM's `return -1` (`:437`).
    /// ⛔ A VALUE, NOT AN ABSENCE, at the sixteen sites that substitute a scale of `1` for it.
    DimNotInOrder,
    /// `primaryDsInfo_` has no entry for the role at all, so IBM's `.at(dstype)` (`:431`) threw
    /// before the search began.
    RoleHasNoLayout,
}

/// How many labeled data structures an op function consumes and produces — one entry of
/// `opFuncsToInOuts` (`dsc/designSpaceConfig.h:357`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OpFuncInOuts {
    pub inputs: OpFuncInputs,
    pub outputs: OpFuncOutputs,
}

/// An op function's input count, or the table's `-1`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpFuncInputs {
    /// The table's non-negative count.
    Count(u32),
    /// The table's `-1`, which both readers name "unknown # of operands" and stop counting on
    /// (`dsc/designSpaceConfig.cpp:8220`, `:8278`).
    Unknown,
}

/// An op function's output count.
///
/// ⛔ NO `Unknown`: all 161 entries state `0` or `1`, so only the input side is ever negative
/// (`dsc/designSpaceConfig.cpp:9285-9448`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OpFuncOutputs(pub u32);

/// Field: e027_DesignSpaceConfig.opFuncsToInOuts
///
/// `opFuncsToInOuts` (`dsc/designSpaceConfig.cpp:9285-9448`) — 161 of [`OpFunc`]'s 176 variants, so
/// the fifteen in the last arm are absent, where IBM's `.at()` throws (`:8213`, `:8274`).
///
/// ⛔ EXHAUSTIVE ON PURPOSE: a new `OpFunc` variant must be placed by hand rather than fall through
/// to absent.
pub fn op_func_in_outs(op: OpFunc) -> Option<OpFuncInOuts> {
    use OpFuncInputs::{Count, Unknown};
    let (inputs, outputs) = match op {
        OpFunc::None => (Count(0), 0),
        OpFunc::ConstPadOpLx | OpFunc::ConstPadOpHbm => (Count(0), 1),
        OpFunc::QuantScalePerToken
        | OpFunc::QuantScalePerTokenFp8
        | OpFunc::Exx2
        | OpFunc::LayernormScale
        | OpFunc::Rsqrt
        | OpFunc::Reciprocal
        | OpFunc::ReluFwd
        | OpFunc::Relu6Fwd
        | OpFunc::Dl16Tofp32
        | OpFunc::Fp32Todl16
        | OpFunc::Fp8Todl16
        | OpFunc::Dl16Tobf16
        | OpFunc::ExpFwd
        | OpFunc::FastExpFwd
        | OpFunc::SqrtFwd
        | OpFunc::TanhFwd
        | OpFunc::SigmoidFwd
        | OpFunc::FastSigmoidFwd
        | OpFunc::SiluFwd
        | OpFunc::MishFwd
        | OpFunc::ClipFwd
        | OpFunc::LogFwd
        | OpFunc::Softmax
        | OpFunc::Softplus
        | OpFunc::MaxpoolFwd
        | OpFunc::AvgpoolFwd
        | OpFunc::QFp8
        | OpFunc::QFp8Ch
        | OpFunc::QFp8Chil
        | OpFunc::QFp8Wt
        | OpFunc::QFp8Mb
        | OpFunc::StcdpOpLx
        | OpFunc::StcdpOpHbm
        | OpFunc::ResizeNnhbm
        | OpFunc::ResizeNnlx
        | OpFunc::ApeOpLx
        | OpFunc::ApeOpHbm
        | OpFunc::ReStickifyOpLx
        | OpFunc::ReStickifyOpHbm
        | OpFunc::ReStickifyOpWithPtLx
        | OpFunc::ReStickifyOpWithPthbm
        | OpFunc::XrfWriteHbm
        | OpFunc::XrfWriteLx
        | OpFunc::Nop
        | OpFunc::StickifyOpHbm
        | OpFunc::DblBufMni
        | OpFunc::Abs
        | OpFunc::Neg
        | OpFunc::AllGather
        | OpFunc::AllReduce
        | OpFunc::GenericPartialReduction
        | OpFunc::InterslicetransposeFp16
        | OpFunc::Floor => (Count(1), 1),
        OpFunc::Add
        | OpFunc::AddI32ToI32
        | OpFunc::AddI64ToI64
        | OpFunc::StridedAdd
        | OpFunc::Mul
        | OpFunc::MulI32ToI32
        | OpFunc::Sub
        | OpFunc::Revsub
        | OpFunc::Realdiv
        | OpFunc::Mask2Bit
        | OpFunc::MaskByIndex
        | OpFunc::Biasadd
        | OpFunc::GeluBwd
        | OpFunc::TanhBwd
        | OpFunc::AvgpoolNmapFwd
        | OpFunc::Conv2DFwd
        | OpFunc::Conv2DFp8Fwd
        | OpFunc::Conv2DInt8Fwd
        | OpFunc::Conv2DInt4Fwd
        | OpFunc::Conv2DFwdGenkg3
        | OpFunc::Conv2DFp8FwdGenkg3
        | OpFunc::Conv2DInt8FwdGenkg3
        | OpFunc::Conv2DInt4FwdGenkg3
        | OpFunc::MatmulFwd
        | OpFunc::MatmulFp8Fwd
        | OpFunc::MatmulInt8Fwd
        | OpFunc::MatmulInt4Fwd
        | OpFunc::BatchmatmulFwd
        | OpFunc::Batchmatmulv2
        | OpFunc::BatchmatmulFp8Fwd
        | OpFunc::BatchmatmulFp8FwdMb
        | OpFunc::BatchmatmulInt8Fwd
        | OpFunc::BatchmatmulInt8FwdMbkg3
        | OpFunc::BatchmatmulInt4Fwd
        | OpFunc::BatchmatmulXrfFwd
        | OpFunc::BatchmatmulXrfFp8Fwd
        | OpFunc::BatchmatmulXrfInt8Fwd
        | OpFunc::BatchmatmulXrfInt4Fwd
        | OpFunc::BatchmatmulXrfchFwd
        | OpFunc::BatchmatmulXrfchFp8Fwd
        | OpFunc::BatchmatmulXrfchInt8Fwd
        | OpFunc::BatchmatmulXrfchInt4Fwd
        | OpFunc::Conv2DFwdOs1
        | OpFunc::Conv2DFwdGenOs1
        | OpFunc::Conv2DInt8FwdOs1
        | OpFunc::Conv2DXrfInt8FwdOs1
        | OpFunc::GatherOpHbm
        | OpFunc::Equal
        | OpFunc::Notequal
        | OpFunc::Greaterequal
        | OpFunc::Greaterthan
        | OpFunc::Lesserequal
        | OpFunc::Lesserthan
        | OpFunc::Maximum
        | OpFunc::Minimum
        | OpFunc::Sinkcorrectionfactor => (Count(2), 1),
        OpFunc::Fnms
        | OpFunc::Rope64P1Fwd
        | OpFunc::Rope64P2Fwd
        | OpFunc::Where3
        | OpFunc::Lstmactp1Fwd
        | OpFunc::BatchnormFwd
        | OpFunc::Conv2DFwdSparsekg3
        | OpFunc::Conv2DFp8FwdSparsekg3
        | OpFunc::Conv2DInt8FwdSparsekg3
        | OpFunc::Conv2DInt4FwdSparsekg3
        | OpFunc::BatchmatmulFwdSparsekg3
        | OpFunc::BatchmatmulFp8FwdSparsekg3
        | OpFunc::BatchmatmulInt8FwdSparsekg3
        | OpFunc::BatchmatmulInt4FwdSparsekg3
        | OpFunc::Bnpreczeroshft => (Count(3), 1),
        OpFunc::ScaledGroupMatmulFp4Fwd => (Count(4), 1),
        OpFunc::LayernormNorm
        | OpFunc::LayernormBwdnorm
        | OpFunc::Lstmactp2Fwd
        | OpFunc::CsqInt8
        | OpFunc::CsqInt8V2
        | OpFunc::CsqInt8Ch
        | OpFunc::CsqInt8Wt
        | OpFunc::CsqInt8Chil
        | OpFunc::CsqInt8Mb
        | OpFunc::CsqInt8MbV2
        | OpFunc::CsqInt4
        | OpFunc::CsqInt4Wt
        | OpFunc::CsqInt4Chil => (Count(5), 1),
        OpFunc::SumNonstick
        | OpFunc::MeanNonstick
        | OpFunc::MaxNonstick
        | OpFunc::AbsmaxNonstick
        | OpFunc::MinNonstick
        | OpFunc::Sum
        | OpFunc::TopkValue
        | OpFunc::TopkIndex
        | OpFunc::Mean
        | OpFunc::Max
        | OpFunc::Absmax
        | OpFunc::Min
        | OpFunc::LeakyreluFwd
        | OpFunc::GeluFwd
        | OpFunc::ErfFwd
        | OpFunc::Identity
        | OpFunc::Shuffle
        | OpFunc::DepthwiseConvFwd
        | OpFunc::SqdiffFwd
        | OpFunc::Lstmblockcell => (Unknown, 1),
        // Absent from the table, so IBM's `.at()` throws for them.
        OpFunc::ProdNonstick
        | OpFunc::Exx2Zeromean
        | OpFunc::BatchmatmulMxfp4WFwd
        | OpFunc::SfpReadLxTransposeFwdl0
        | OpFunc::PtBlkTransposeLoad
        | OpFunc::PesfpCollate2BWrites
        | OpFunc::ScatterOpHbm
        | OpFunc::Itof
        | OpFunc::Itofhbm
        | OpFunc::AllShuffle
        | OpFunc::InterslicetransposeFp8
        | OpFunc::BatchmatmulMxfp8Fwd
        | OpFunc::Int32Idxtoaddr
        | OpFunc::StzLatch
        | OpFunc::Undef => return None,
    };
    Some(OpFuncInOuts {
        inputs,
        outputs: OpFuncOutputs(outputs),
    })
}

/// Replaces: e027_DesignSpaceConfig
///
/// `dsc/designSpaceConfig.h:51-705`. One op's design space configuration: the data structures it
/// moves, the memory-hierarchy stages it moves them through, the loop nest that drives them and the
/// schedule tree the DDC and the L3 scheduler build into it.
///
/// ⛔ SIX OF THE THIRTY-SIX DECLARED FIELDS (`:72-133`) ARE NOT CARRIED, and EIGHT anchors below stay
/// open for them: the scheduler flattened the nested `ProgramFrame` (`:129-132`) into a `.ptr_` and a
/// `.size_` anchor of its own, so `prog_frame_ptr_` costs three. Thirty carried plus six uncarried
/// closes on thirty-six; the ANCHOR count is the one that does not. Each uncarried field needs a type
/// this campaign has not scheduled:
/// * `labeledDs_` (`:86`) — `std::vector<LabeledDsInfo>`, and `LabeledDsInfo` is a 25-field cluster
///   over `DtInfo`, `MemOrg`, `CoreDsInfo` and `MxInfo` (`dsc/dscdefn.h:321-468`), none of them a
///   unit in `crustify-scheduler/UNITS.tsv`.
/// * `computeOp_` (`:89`) — `std::vector<ComputeOpInfo>`, four of whose members are
///   `std::vector<LabeledDsInfo*>` held as pointer identity INTO `labeledDs_`
///   (`dsc/dscdefn.h:506-511`).
/// * `auxLoopOrder_` (`:114`) — `AuxLoopSetInfo` holds two raw `DataStructDims*` aliasing this
///   object's own members (`dsc/dscdefn.h:126-127`). ⛔ TWELVE IN-SCOPE USES IN THIS CLASS'S OWN TU:
///   the loop walk descends into every aux set of each [`loop_order`](Self::loop_order) entry
///   (`dsc/designSpaceConfig.cpp:1344-1346`) and the JSON importer resolves those two pointers by
///   matching the string against the `name_` of eight named members PLUS every [`sc`](Self::sc)
///   entry (`:6927-6931`, `:7511-7515`).
/// * `scheduleTree_` (`:115`) — `dsc2::ScheduleTree`, still the open `e032_ScheduleTree` anchor.
///   `isDSC2` — `return !scheduleTree_.empty()` (`dsc/designSpaceConfig.cpp:30`) — is the one method
///   blocked on it alone, and is NOT ported here.
/// * `pcfg_` (`:120`) — `std::vector<SenPcfg>`, and DCG/PCFG is off this campaign's path
///   (`crustify-scheduler/AGENT-BRIEF.md`, decided 2026-09-09).
/// * `prog_frame_ptr_` (`:133`) — a `std::map<SenTargets, ProgramFrame>` whose two members
///   (`:130-131`) are a `std::shared_ptr<void>` and its byte count, and whose only writer is
///   `fillPcfgProgramFrame` on the `SENPCFG` key (`dsc/designSpaceConfig.cpp:1020-1027`). ⛔ NOT the
///   `ProgramFrame` `SuperDsc` uses: that one is sendefs' three-member struct with `st_address`
///   (`util/sendefs/sendefs.h:190-194`).
///
/// ⛔ AND `paramNameToVal` (`:358-609`) IS PORTED AS A RESOLVER, NOT A TABLE. IBM's 240 entries are
/// `double*` INTO this object's own `DataStructDims` members, so a copy leaves every pointer aimed at
/// the SOURCE object; `updateParamNameToVal()` (`:614`) exists to re-point them and it re-points only
/// 62 of the 240 — and it has ZERO callers tree-wide. [`param_name_to_val`](Self::param_name_to_val)
/// resolves the name on each call, so there is nothing to go stale and nothing for that method to fix.
///
/// ⛔ AND EIGHT OF THIS UNIT'S STILL-OPEN ANCHORS NAME NO FIELD AT ALL. `addNewLine`,
/// `allowMissingAlloc`, `allowSymbolicVolumeLimit`, `doNotRound`, `includeGaps`, `numL0Slices`, `ps`
/// and `sizeInNumberOfLoads` are trailing default-valued PARAMETERS on method signatures that wrap
/// across lines (`dsc/designSpaceConfig.h:183`, `:292`, `:275`, `:231`, `:274`, `:259`, `:182`,
/// `:227`); the same scan omits four fields that ARE carried and anchored here —
/// `dimToSymbolMapping_`, `dscN_`, `coordinateMasking_` and `maskingConstId_`.
#[derive(Clone, Debug)]
pub struct DesignSpaceConfig {
    /// Field: e027_DesignSpaceConfig.name_
    ///
    /// The op's name (`dsc/designSpaceConfig.h:72`), filled by DSM.
    pub name: String,
    /// Field: e027_DesignSpaceConfig.numCoresUsed_
    ///
    /// How many cores this DSC uses (`dsc/designSpaceConfig.h:73`).
    ///
    /// ⛔ ABSENT UNTIL DSM WRITES IT: the authority declares a bare `int` with no member
    /// initialiser, so a default-constructed DSC's value is indeterminate — and `0` would be a real
    /// count, which is why this is an [`Option`] rather than a zero.
    pub num_cores_used: Option<NumCoresUsed>,
    /// Field: e027_DesignSpaceConfig.numCoreletsUsed_
    ///
    /// How many corelets per core this DSC uses (`dsc/designSpaceConfig.h:74`), absent on the same
    /// terms as [`num_cores_used`](Self::num_cores_used).
    ///
    /// ⛔ NOT THE SAME FIELD AS [`num_corelets_used_dsc2`](Self::num_corelets_used_dsc2): this one is
    /// DSM's and carries no absent encoding of its own, that one is DM's and starts at `-1`.
    pub num_corelets_used: Option<NumCoreletsUsed>,
    /// Field: e027_DesignSpaceConfig.coreIdsUsed_
    ///
    /// Which cores, by id (`dsc/designSpaceConfig.h:75`). The DDC iterates it to place per-core
    /// allocations (`ddc/ddcv1.cpp:193`, `ddc/ddc_transformation.cpp:1310`).
    pub core_ids_used: Vec<CoreId>,
    /// Field: e027_DesignSpaceConfig.dimToSymbolMapping_
    ///
    /// Per dim, the symbols standing in for its extent: one for a pure symbolic or pivot dim, several
    /// (max-pivot) for an irregular one (`dsc/designSpaceConfig.h:76-78`). Round-tripped through JSON
    /// with the DSC2 fields (`dsc/dsc2.cpp:50-52`, `:1120`).
    pub dim_to_symbol_mapping: BTreeMap<PrimaryDimTypes, Vec<VariableSymbol>>,
    /// Field: e027_DesignSpaceConfig.N_
    ///
    /// The whole op's dims, padded (`dsc/designSpaceConfig.h:81`), named `"n"` by the constructor.
    pub n: DataStructDims,
    /// Field: e027_DesignSpaceConfig.unpadN_
    ///
    /// The same dims before padding (`dsc/designSpaceConfig.h:82`), named `"unpadn"`.
    pub unpad_n: DataStructDims,
    /// Field: e027_DesignSpaceConfig.dscN_
    ///
    /// The parameters THIS DSC performs, which is a share of [`n`](Self::n) when an op is split
    /// across DSCs (`dsc/designSpaceConfig.h:83`), named `"dscn"`. It is what the coordinate-masking
    /// writers subtract the valid extent from (`dsm/dsm.cpp:17532`, `:17573`).
    pub dsc_n: DataStructDims,
    /// Field: e027_DesignSpaceConfig.constantInfo_
    ///
    /// Every constant this op needs, by id (`dsc/designSpaceConfig.h:90`).
    pub constant_info: BTreeMap<ConstantId, ConstantInfo>,
    /// Field: e027_DesignSpaceConfig.primaryDsInfo_
    ///
    /// Per data-structure role, its layout and stick order (`dsc/designSpaceConfig.h:93`). It is what
    /// [`stick_sizes`](Self::stick_sizes), [`layout_dim_set`](Self::layout_dim_set) and
    /// [`dim_index_in_layout_order`](Self::dim_index_in_layout_order) all read.
    pub primary_ds_info: BTreeMap<DsTypes, PrimaryDsInfo>,
    /// Field: e027_DesignSpaceConfig.pdsRelation_
    ///
    /// Whether the primary data structures reuse each other (`dsc/designSpaceConfig.h:94`).
    pub pds_relation: PrimaryDsRelationInfo,
    /// Field: e027_DesignSpaceConfig.ChipD_
    ///
    /// The dims one chip handles (`dsc/designSpaceConfig.h:95`), named `"chipd"`.
    pub chip_d: DataStructDims,
    /// Field: e027_DesignSpaceConfig.ChipletD_
    ///
    /// The dims one chiplet handles (`dsc/designSpaceConfig.h:96`), named `"chipletd"`.
    pub chiplet_d: DataStructDims,
    /// Field: e027_DesignSpaceConfig.CoreD_
    ///
    /// The dims one CORE handles (`dsc/designSpaceConfig.h:97`).
    ///
    /// ⛔ ITS NAME IS `"d"`, NOT `"cored"` (`dsc/designSpaceConfig.cpp:22`), and that bare `d` is the
    /// prefix of twenty `paramNameToVal` keys and of `getLoopCount`'s numerator for every DB loop.
    pub core_d: DataStructDims,
    /// Field: e027_DesignSpaceConfig.CoreletD_
    ///
    /// The dims one corelet handles (`dsc/designSpaceConfig.h:98`), named `"coreletd"`.
    pub corelet_d: DataStructDims,
    /// Field: e027_DesignSpaceConfig.coordinateMasking_
    ///
    /// Per dim, each masked stretch as a `<unmasked, masked>` pair of ELEMENT COUNTS
    /// (`dsc/designSpaceConfig.h:99-100`) — DSM writes `(valid, dscN_.j_ - valid)`
    /// (`dsm/dsm.cpp:17532`, `:17573`), and `constructSAMVNodes` turns it into the SAMV
    /// [`StickMaskNode`](crate::schedule::dsc2::StickMaskNode) (`ddc/ddcv1.cpp:3485-3600`).
    /// Round-tripped with the DSC2 fields (`dsc/dsc2.cpp:36-38`, `:1101-1105`).
    pub coordinate_masking: BTreeMap<PrimaryDimTypes, Vec<MaskSplit>>,
    /// Field: e027_DesignSpaceConfig.maskingConstId_
    ///
    /// The constant holding the value masked elements read, shared by every tensor of the op
    /// (`dsc/designSpaceConfig.h:101`). ⛔ ITS `-1` IS ABSENT: the SAMV node copies it and both
    /// readers test `>= 0` before indexing `constantInfo_` (`ddc/ddcv1.cpp:3531`,
    /// `dsc-based-utils/DSC2ToDataflowIR/V3/SNStickMaskLowering.cpp:32-36`).
    pub masking_const_id: Option<ConstantId>,
    /// Field: e027_DesignSpaceConfig.numCoreletsUsed_DSC2_
    ///
    /// DM's corelet count, which is what the DSC2 path iterates (`dsc/designSpaceConfig.h:104`;
    /// `ddc/ddcv1.cpp:207`, `:1755`, `:1769`, `ddc/ddc_transformation_util.cpp:519`). ⛔ ITS `-1` IS
    /// ABSENT, and IBM's `for (cl = 0; cl < -1; cl++)` simply does not run.
    pub num_corelets_used_dsc2: Option<NumCoreletsUsed>,
    /// Field: e027_DesignSpaceConfig.dataStageParam_
    ///
    /// Each data stage by id (`dsc/designSpaceConfig.h:105`) — the stage a loop's numerator and
    /// denominator name, and where `parametricIterCount` reads its padding from
    /// (`dsc/dsc2.cpp:4155`).
    pub data_stage_param: BTreeMap<DataStageId, DataStage>,
    /// Field: e027_DesignSpaceConfig.B_
    ///
    /// The block-transfer stage's dims (`dsc/designSpaceConfig.h:106`), named `"b"`.
    pub b: DataStructDims,
    /// Field: e027_DesignSpaceConfig.T_
    ///
    /// The tile stage's dims (`dsc/designSpaceConfig.h:107`), named `"t"`.
    pub t: DataStructDims,
    /// Field: e027_DesignSpaceConfig.Tel_
    ///
    /// The tile stage's element-level dims (`dsc/designSpaceConfig.h:108`), named `"tel"`.
    pub tel: DataStructDims,
    /// Field: e027_DesignSpaceConfig.P_
    ///
    /// The processing stage's dims (`dsc/designSpaceConfig.h:109`), named `"p"`.
    pub p: DataStructDims,
    /// Field: e027_DesignSpaceConfig.Pel_
    ///
    /// The processing stage's element-level dims (`dsc/designSpaceConfig.h:110`), named `"pel"`.
    pub pel: DataStructDims,
    /// Field: e027_DesignSpaceConfig.sc_
    ///
    /// The auxiliary loop sets' dims (`dsc/designSpaceConfig.h:111`).
    ///
    /// ⛔ EVERY ENTRY IS REACHABLE BY ITS OWN `name_`, AND THAT IS THE ONLY WAY IN. `getDsdFromStr`
    /// does not name `sc_` and the 240 keys have no `sc` prefix, but each entry carries a runtime
    /// name and both writers register `name_ + <dim>` as twenty MORE `paramNameToVal` keys per entry
    /// (`dm/dm.cpp:971-983`, `dsi/test/psum_test.cpp:150-159`), which is why
    /// [`param_name_to_val`](Self::param_name_to_val) resolves them and why a closed twelve-prefix
    /// table alone would answer absent where IBM answers. ⛔ THE LAST MATCHING ENTRY WINS, because
    /// `map[key] = ptr` overwrites and entry 0 is registered before entry 1.
    ///
    /// ⛔ AND IT IS BOTH READ AND WRITTEN IN THIS CLASS'S OWN IN-SCOPE TU, at five sites: the JSON
    /// importer grows the vector from any key beginning `sc_` (`dsc/designSpaceConfig.cpp:6862`,
    /// `:6891-6899`), the aux-loop pointer resolution matches `numerator_`/`denominator_` against
    /// every entry's `name_` (`:6930-6931`, `:7514-7515`), and `printMed` (`:373-375`) and
    /// `exportJson` (`:6235-6237`) both emit it. `AuxLoopSetInfo::scDimOrder_` (`dsc/dscdefn.h:128`)
    /// is a different field.
    pub sc: Vec<DataStructDims>,
    /// Field: e027_DesignSpaceConfig.loopOrder_
    ///
    /// The loop nest, outermost first (`dsc/designSpaceConfig.h:112`). Filled by DM/DSI
    /// (`dsi/test/psum_test.cpp:70`).
    ///
    /// ⛔ EIGHTY-ONE USES IN THIS CLASS'S OWN IN-SCOPE TU, and one of them picks the key
    /// [`param_name_to_val`](Self::param_name_to_val) is then asked for: `getDimPrefix4LxTransfer_lDs`
    /// walks this order for the loop naming the dim and returns that loop's middle letter — `t`, `b`
    /// or `d` (`dsc/designSpaceConfig.cpp:9063-9075`) — which its caller concatenates with the dim
    /// name (`dm/dm.cpp:1120-1125`). So the loop order chooses WHICH stage's extent a
    /// `paramNameToVal` lookup answers with.
    pub loop_order: Vec<LoopNames>,
    /// Field: e027_DesignSpaceConfig.loopProperties_
    ///
    /// Per loop, its properties (`dsc/designSpaceConfig.h:113`). DM copies the map wholesale and then
    /// overwrites single entries (`dm/dm.cpp:470`, `:1446`).
    pub loop_properties: BTreeMap<LoopNames, LoopProperties>,
    /// Field: e027_DesignSpaceConfig.gtrIdsUsed_
    ///
    /// Which group tag registers this DSC occupies (`dsc/designSpaceConfig.h:116`), round-tripped
    /// with the DSC2 fields (`dsc/dsc2.cpp:111-113`, `:1151-1154`).
    pub gtr_ids_used: BTreeSet<GroupId>,
    /// Field: e027_DesignSpaceConfig.l0TetheredMode_
    ///
    /// Whether L0 is tethered (`dsc/designSpaceConfig.h:117`), which the DDC's allocation walk
    /// branches on (`ddc/ddcv1.cpp:299`, `:324`) and JSON carries (`dsc/dsc2.cpp:364`, `:1156`).
    pub l0_tethered_mode: bool,
    /// Field: e027_DesignSpaceConfig.target_
    ///
    /// What this DSC is being compiled for (`dsc/designSpaceConfig.h:121`); it selects which tool
    /// fills the program frame (`:123-128`) and is the key `fillPcfgProgramFrame` writes under
    /// (`dsc/designSpaceConfig.cpp:1020-1027`).
    pub target: SenTargets,
}

impl Default for DesignSpaceConfig {
    /// The constructor, whose whole body is the twelve `DataStructDims` names
    /// (`dsc/designSpaceConfig.cpp:16-28`).
    ///
    /// ⛔ THOSE NAMES ARE THE ONES `getDsdFromStr` MATCHES, so they are not labels: renaming one
    /// silently unhooks [`dsd_from_str`](Self::dsd_from_str) and every `paramNameToVal` key built on
    /// its prefix. And `CoreD_` is `"d"`, not `"cored"` (`:22`).
    fn default() -> Self {
        fn named(name: &str) -> DataStructDims {
            DataStructDims {
                name: name.to_owned(),
                ..DataStructDims::default()
            }
        }

        Self {
            name: String::new(),
            num_cores_used: None,
            num_corelets_used: None,
            core_ids_used: Vec::new(),
            dim_to_symbol_mapping: BTreeMap::new(),
            n: named("n"),
            unpad_n: named("unpadn"),
            dsc_n: named("dscn"),
            constant_info: BTreeMap::new(),
            primary_ds_info: BTreeMap::new(),
            pds_relation: PrimaryDsRelationInfo::default(),
            chip_d: named("chipd"),
            chiplet_d: named("chipletd"),
            core_d: named("d"),
            corelet_d: named("coreletd"),
            coordinate_masking: BTreeMap::new(),
            masking_const_id: None,
            num_corelets_used_dsc2: None,
            data_stage_param: BTreeMap::new(),
            b: named("b"),
            t: named("t"),
            tel: named("tel"),
            p: named("p"),
            pel: named("pel"),
            sc: Vec::new(),
            loop_order: Vec::new(),
            loop_properties: BTreeMap::new(),
            gtr_ids_used: BTreeSet::new(),
            l0_tethered_mode: false,
            target: SenTargets::Undefined,
        }
    }
}

impl DesignSpaceConfig {
    /// The `DataStructDims` one of the twelve spellings names (`dsc/designSpaceConfig.h:317-346`).
    /// The match is case-insensitive, as IBM's `tolower` makes it; an unknown spelling is absent,
    /// where IBM `DT_ERROR`s "Unknow string input to getDsdFromStr()".
    pub fn dsd_from_str(&self, dsdstr: &str) -> Option<&DataStructDims> {
        Some(match dsdstr.to_ascii_lowercase().as_str() {
            "n" => &self.n,
            "d" => &self.core_d,
            "b" => &self.b,
            "t" => &self.t,
            "p" => &self.p,
            "coreletd" => &self.corelet_d,
            "tel" => &self.tel,
            "pel" => &self.pel,
            "unpadn" => &self.unpad_n,
            "chipd" => &self.chip_d,
            "chipletd" => &self.chiplet_d,
            "dscn" => &self.dsc_n,
            _ => return None,
        })
    }

    /// The same dispatch as a handle to assign through — IBM's `getDsdFromStr` returns a
    /// `DataStructDims&` and its callers write through it (`dsc/designSpaceConfig.h:317-346`).
    pub fn dsd_from_str_mut(&mut self, dsdstr: &str) -> Option<&mut DataStructDims> {
        Some(match dsdstr.to_ascii_lowercase().as_str() {
            "n" => &mut self.n,
            "d" => &mut self.core_d,
            "b" => &mut self.b,
            "t" => &mut self.t,
            "p" => &mut self.p,
            "coreletd" => &mut self.corelet_d,
            "tel" => &mut self.tel,
            "pel" => &mut self.pel,
            "unpadn" => &mut self.unpad_n,
            "chipd" => &mut self.chip_d,
            "chipletd" => &mut self.chiplet_d,
            "dscn" => &mut self.dsc_n,
            _ => return None,
        })
    }

    /// The dim one of `paramNameToVal`'s 240 keys names (`dsc/designSpaceConfig.h:358-609`), or one of
    /// the twenty an [`sc`](Self::sc) entry adds under its own `name_` (`dm/dm.cpp:980-983`).
    ///
    /// ⛔ ABSENT MEANS ONLY THAT THE KEY IS NOT IN THE TABLE, which is IBM's `.at()` throw. An
    /// UNFILLED dim is [`ParamVal::UNFILLED`], because IBM reads through a `double*` at a member born
    /// `-1` — measured: all 240 keys answer `-1` on a default-constructed DSC and none throws.
    pub fn param_name_to_val(&self, name: &str) -> Option<ParamVal> {
        if let Some((entry, dim)) = self.sc_param_name(name) {
            return dim_val_by_name(&self.sc[entry], dim);
        }
        let (prefix, dim) = split_param_name(name)?;
        dim_val_by_name(self.dsd_from_str(prefix)?, dim)
    }

    /// The same key as a handle to assign through — the table's values are `double*` for that reason
    /// (`dsc/designSpaceConfig.h:358-609`).
    pub fn param_name_to_val_mut(&mut self, name: &str) -> Option<&mut Option<dims::DimSize>> {
        if let Some((entry, dim)) = self.sc_param_name(name) {
            return self.sc[entry].param_name_to_val_mut(dim);
        }
        let (prefix, dim) = split_param_name(name)?;
        self.dsd_from_str_mut(prefix)?.param_name_to_val_mut(dim)
    }

    /// Which [`sc`](Self::sc) entry a key names, and the dim spelled after that entry's `name_`
    /// (`dm/dm.cpp:980-983`). ⛔ THE LAST MATCHING ENTRY WINS, because both writers register entry 0
    /// before entry 1 and `map[key] = ptr` overwrites.
    fn sc_param_name(&self, name: &str) -> Option<(usize, &'static str)> {
        self.sc.iter().enumerate().rev().find_map(|(entry, dsd)| {
            let rest = name.strip_prefix(dsd.name.as_str())?;
            PARAM_DIM_NAMES
                .into_iter()
                .find(|dim| *dim == rest)
                .map(|dim| (entry, dim))
        })
    }

    /// One loop's trip count: its numerator stage's dim divided by its denominator stage's
    /// (`dsc/designSpaceConfig.cpp:416-427`).
    ///
    /// ⛔ IT IS COMPUTED BY CUTTING UP THE LOOP'S SPELLING, not by any stage field: `dbin` becomes
    /// `din` over `bin`, i.e. [`core_d`](Self::core_d)`.in` over [`b`](Self::b)`.in`.
    ///
    /// ⛔ AN UNFILLED STAGE STILL HAS A COUNT: both keys read [`ParamVal::UNFILLED`], and measured
    /// against the authority `bin = 16` over an unset `tin` is `int(16 / -1) == -16`, not absence.
    pub fn loop_count(&self, loop_name: LoopNames) -> LoopTripCount {
        let name = loop_name.name();
        let (Some(stage_num), Some(stage_den), Some(dim)) =
            (name.get(..1), name.get(1..2), name.get(2..))
        else {
            return LoopTripCount::NoSuchLoop;
        };
        let (Some(numerator), Some(denominator)) = (
            self.param_name_to_val(&format!("{stage_num}{dim}")),
            self.param_name_to_val(&format!("{stage_den}{dim}")),
        ) else {
            return LoopTripCount::NoSuchLoop;
        };
        if denominator.0 == 0.0 {
            return LoopTripCount::UndefinedZeroDenominator;
        }
        LoopTripCount::Is(LoopCount((numerator.0 / denominator.0) as i32))
    }

    /// Where one dim sits in a role's layout order (`dsc/designSpaceConfig.cpp:429-438`), as a
    /// [`LayoutOrderPosition`] — which keeps IBM's in-band `-1` and its `.at()` throw APART, because
    /// its readers do and sixteen of them treat that `-1` as a scale of `1` rather than as absence.
    pub fn dim_index_in_layout_order(
        &self,
        ds_type: DsTypes,
        dim: PrimaryDimTypes,
    ) -> LayoutOrderPosition {
        let Some(pdsi) = self.primary_ds_info.get(&ds_type) else {
            return LayoutOrderPosition::RoleHasNoLayout;
        };
        match pdsi.layout_dim_order.iter().position(|d| *d == dim) {
            Some(index) => LayoutOrderPosition::At(index),
            None => LayoutOrderPosition::DimNotInOrder,
        }
    }

    /// A role's layout dims as a set (`dsc/dsc2.cpp:4027-4031`).
    pub fn layout_dim_set(&self, ds_type: DsTypes) -> Option<BTreeSet<PrimaryDimTypes>> {
        Some(
            self.primary_ds_info
                .get(&ds_type)?
                .layout_dim_order
                .iter()
                .copied()
                .collect(),
        )
    }

    /// A role's stick dims, in order (`dsc/designSpaceConfig.h:241-243`).
    pub fn stick_dims(&self, ds_type: DsTypes) -> Option<&[PrimaryDimTypes]> {
        Some(&self.primary_ds_info.get(&ds_type)?.stick_dim_order)
    }

    /// The same dims as a set (`dsc/dsc2.cpp:4033-4037`).
    pub fn stick_dim_set(&self, ds_type: DsTypes) -> Option<BTreeSet<PrimaryDimTypes>> {
        Some(self.stick_dims(ds_type)?.iter().copied().collect())
    }

    /// Each stick dim with the extent the requested scope leaves it (`dsc/dsc2.cpp:4066-4106`).
    ///
    /// ⛔ THE SCOPE SPLITS ONE STICK AT ITS SLICE BOUNDARY: `elemInSlice` is the product of every
    /// stick extent divided by the slice count, and the walk stops, truncates or skips a dim
    /// depending on whether it fits inside that many elements. A stick whose element count is not a
    /// multiple of the slice count is absent, where IBM `DT_CHECK`s.
    ///
    /// ⭐ THE `int` MULTIPLY IS TRUNCATING AT EVERY STEP: `elemInSlice *= size` with `size` a
    /// `double` truncates back to `int` per dim, so a fractional stick extent is not merely rounded
    /// once at the end.
    pub fn stick_sizes(&self, ds_type: DsTypes, scope: StickSizeScope) -> Option<Vec<Size>> {
        let pdsi = self.primary_ds_info.get(&ds_type)?;
        let mut elem_in_slice: i32 = 1;
        for size in &pdsi.stick_size {
            elem_in_slice = (f64::from(elem_in_slice) * size.0) as i32;
        }
        let num_slices = match scope {
            StickSizeScope::L0SliceOnly(slices) => slices.get(),
            _ => SLICES_PER_STICK,
        };
        if elem_in_slice <= 0 || elem_in_slice % num_slices != 0 {
            return None;
        }
        elem_in_slice /= num_slices;

        let mut result = Vec::new();
        let mut elem_so_far: i32 = 1;
        for (i, dim) in pdsi.stick_dim_order.iter().enumerate() {
            let mut size = pdsi.stick_size.get(i)?.0 as i32;
            match scope {
                StickSizeScope::SliceOnly | StickSizeScope::L0SliceOnly(_) => {
                    // Can not fit more elements into the slice.
                    if elem_so_far >= elem_in_slice {
                        break;
                    }
                    elem_so_far *= size;
                    if elem_so_far > elem_in_slice {
                        size /= elem_so_far / elem_in_slice;
                    }
                }
                StickSizeScope::WithoutSlice => {
                    let new_elem_so_far = elem_so_far * size;
                    if elem_so_far < elem_in_slice && new_elem_so_far > elem_in_slice {
                        size /= elem_in_slice / elem_so_far;
                    }
                    elem_so_far = new_elem_so_far;
                    if elem_so_far <= elem_in_slice {
                        // dim included in slice
                        continue;
                    }
                }
                StickSizeScope::WholeStick => {}
            }
            result.push(Size::new(*dim, dsc2::DimSize(size)));
        }
        Some(result)
    }

    /// [`stick_sizes`](Self::stick_sizes) folded per dim, multiplying a dim that appears twice
    /// (`dsc/dsc2.cpp:4108-4124`).
    pub fn cumulative_stick_sizes(
        &self,
        ds_type: DsTypes,
        scope: StickSizeScope,
    ) -> Option<BTreeMap<PrimaryDimTypes, dsc2::DimSize>> {
        let mut result: BTreeMap<PrimaryDimTypes, dsc2::DimSize> = BTreeMap::new();
        for size in self.stick_sizes(ds_type, scope)? {
            result
                .entry(size.dim)
                .and_modify(|total| total.0 *= size.size.0)
                .or_insert(size.size);
        }
        Some(result)
    }

    /// One labeled input's HBM row count, less the zero padding on both sides
    /// (`dsc/designSpaceConfig.cpp:930-934`). Absent only where `<label>r` is no key at all — the
    /// twelve `DataStructDims` names all are, so `"cored"` is absent and `"chipd"` is not.
    pub fn inp_row_in_hbm(&self, label: &str) -> Option<HbmRows> {
        let r = self.param_name_to_val(&format!("{label}r"))?.0;
        let zi = self.param_name_to_val("nzi")?.0;
        Some(HbmRows(r - zi * 2.0))
    }

    /// One labeled input's HBM column count, less the zero padding on both sides
    /// (`dsc/designSpaceConfig.cpp:936-940`), absent on the same terms as
    /// [`inp_row_in_hbm`](Self::inp_row_in_hbm).
    pub fn inp_col_in_hbm(&self, label: &str) -> Option<HbmCols> {
        let c = self.param_name_to_val(&format!("{label}c"))?.0;
        let zj = self.param_name_to_val("nzj")?.0;
        Some(HbmCols(c - zj * 2.0))
    }

    /// Rows times columns, or `-1` when BOTH are negative
    /// (`dsc/designSpaceConfig.cpp:942-946`). ⛔ IF ONLY ONE IS NEGATIVE THE PRODUCT IS RETURNED
    /// NEGATIVE — the authority's `&&` is not an `||`, and no caller re-checks the sign.
    pub fn inp_in_hbm(&self, label: &str) -> Option<HbmElements> {
        let r = self.inp_row_in_hbm(label)?.0;
        let c = self.inp_col_in_hbm(label)?.0;
        Some(HbmElements(if r < 0.0 && c < 0.0 { -1.0 } else { r * c }))
    }

    /// Whether `d1` covers `d2` dim for dim, with the reason it does not
    /// (`dsc/designSpaceConfig.cpp:7814-7874`). The primary dims must all be at least as large; the
    /// auxiliary ones must be at least as large OR unfilled; and none may carry a fraction.
    ///
    /// ⛔ THE LAST BLOCK SETS `auxiliaryCheck = true` WHERE IT PLAINLY MEANS `false` (`:7869-7871`),
    /// while its primary twin sets `false` (`:7840`). So a fractional auxiliary dim PASSES this
    /// check and merely appends its message — reproduced here, because it is the behaviour every
    /// caller sees.
    ///
    /// ⛔ AND IT ONLY REQUIRES `ij_`/`kij_` WHEN SOME ROLE'S LAYOUT USES THAT DIM (`:7821-7829`), so
    /// the answer depends on [`primary_ds_info`](Self::primary_ds_info), not on the two arguments
    /// alone.
    pub fn check_data_struct_dims(
        &self,
        d1: &DataStructDims,
        d2: &DataStructDims,
    ) -> (bool, String) {
        let mut primary_check = false;
        let mut auxiliary_check = false;
        let mut msg = String::new();

        let mut dim_used = BTreeSet::new();
        for pdsi in self.primary_ds_info.values() {
            dim_used.extend(pdsi.layout_dim_order.iter().copied());
        }
        let check_ij = dim_used.contains(&PrimaryDimTypes::Ij);
        let check_kij = dim_used.contains(&PrimaryDimTypes::Kij);

        if d1.r#in >= d2.r#in
            && d1.out >= d2.out
            && d1.mb >= d2.mb
            && (d1.ij >= d2.ij || !check_ij)
            && (d1.kij >= d2.kij || !check_kij)
            && d1.x >= d2.x
            && d1.y >= d2.y
        {
            primary_check = true;
        } else {
            msg.push_str("all of values in primary fields should be equal or greater");
        }

        let primary = [
            d1.r#in, d1.out, d1.mb, d1.ij, d1.kij, d1.x, d1.y, d2.r#in, d2.out, d2.mb, d2.ij,
            d2.kij, d2.x, d2.y,
        ];
        if !primary.into_iter().all(is_fractional) {
            primary_check = false;
            msg.push_str("all of values in primary fields should have no fraction");
        }

        let auxiliary = [
            (d1.rc, d2.rc),
            (d1.sij, d2.sij),
            (d1.zij, d2.zij),
            (d1.i, d2.i),
            (d1.j, d2.j),
            (d1.r, d2.r),
            (d1.c, d2.c),
            (d1.ki, d2.ki),
            (d1.kj, d2.kj),
            (d1.si, d2.si),
            (d1.sj, d2.sj),
            (d1.zi, d2.zi),
            (d1.zj, d2.zj),
        ];
        if auxiliary
            .into_iter()
            .all(|(one, two)| one.is_none() || one >= two)
        {
            auxiliary_check = true;
        } else {
            msg.push_str(
                "all of values in auxiliary fields should be equal, greater or un-initialized (-1)",
            );
        }

        let auxiliary_values = auxiliary
            .into_iter()
            .flat_map(|(one, two)| [one, two])
            .collect::<Vec<_>>();
        if !auxiliary_values.into_iter().all(is_fractional) {
            auxiliary_check = true;
            msg.push_str("all of values in auxiliary fields should have no fraction");
        }

        (primary_check && auxiliary_check, msg)
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    fn dim(size: f64) -> Option<dims::DimSize> {
        dims::DimSize::new(size)
    }

    /// `dsc/designSpaceConfig.cpp:16-28`: the constructor's whole body is these twelve names, and
    /// `CoreD_`'s is the bare `"d"` (`:22`) that twenty `paramNameToVal` keys are built on.
    #[test]
    fn the_constructor_names_all_twelve_data_struct_dims_and_core_d_is_just_d() {
        let dsc = DesignSpaceConfig::default();
        let names: Vec<&str> = [
            &dsc.n,
            &dsc.unpad_n,
            &dsc.dsc_n,
            &dsc.chip_d,
            &dsc.chiplet_d,
            &dsc.core_d,
            &dsc.corelet_d,
            &dsc.b,
            &dsc.t,
            &dsc.p,
            &dsc.tel,
            &dsc.pel,
        ]
        .iter()
        .map(|dsd| dsd.name.as_str())
        .collect();
        assert_eq!(
            names,
            [
                "n", "unpadn", "dscn", "chipd", "chipletd", "d", "coreletd", "b", "t", "p", "tel",
                "pel"
            ]
        );

        // The other initialisers of `dsc/designSpaceConfig.h:72-121`.
        assert_eq!(dsc.num_cores_used, None);
        assert_eq!(dsc.masking_const_id, None);
        assert_eq!(dsc.num_corelets_used_dsc2, None);
        assert!(!dsc.l0_tethered_mode);
        assert!(dsc.pds_relation.is_pds_reuse);
        assert_eq!(dsc.target, SenTargets::Undefined);
    }

    /// `dsc/designSpaceConfig.h:317-346`: every spelling the twelve-way selector accepts, lowercased
    /// on the way in, and nothing else.
    #[test]
    fn dsd_from_str_selects_the_twelve_named_dims_case_insensitively() {
        let mut dsc = DesignSpaceConfig::default();
        for spelling in [
            "n", "d", "b", "t", "p", "coreletd", "tel", "pel", "unpadn", "chipd", "chipletd",
            "dscn",
        ] {
            assert_eq!(
                dsc.dsd_from_str(spelling).map(|dsd| dsd.name.as_str()),
                Some(if spelling == "d" { "d" } else { spelling })
            );
        }
        assert_eq!(
            dsc.dsd_from_str("ChipletD").map(|dsd| dsd.name.as_str()),
            Some("chipletd")
        );

        // IBM's DT_ERROR arm, and `sc_`, which the selector never named.
        assert!(dsc.dsd_from_str("cored").is_none());
        assert!(dsc.dsd_from_str("sc").is_none());

        dsc.dsd_from_str_mut("t").unwrap().i = dim(4.0);
        assert_eq!(dsc.t.i, dim(4.0));
    }

    /// `dsc/designSpaceConfig.h:358-609`: 12 prefixes x 20 dims, resolved by longest prefix — and no
    /// `<dsd>x1` key exists even though `DataStructDims` has the field (`dsc/dims.cpp:437-482`).
    #[test]
    fn param_name_to_val_resolves_all_240_keys_and_no_x1_key() {
        let mut dsc = DesignSpaceConfig::default();
        dsc.core_d.r#in = dim(8.0);
        dsc.dsc_n.ij = dim(3.0);
        dsc.pel.zij = dim(1.5);

        assert_eq!(dsc.param_name_to_val("din"), Some(ParamVal(8.0)));
        assert_eq!(dsc.param_name_to_val("dscnij"), Some(ParamVal(3.0)));
        assert_eq!(dsc.param_name_to_val("pelzij"), Some(ParamVal(1.5)));

        // Measured against the authority: every one of the 240 keys ANSWERS on a default-constructed
        // DSC, and every one answers the `-1` its `double*` points at. None of them throws.
        let fresh = DesignSpaceConfig::default();
        let mut resolved = 0;
        for prefix in DSD_PREFIXES {
            if dsc.dsd_from_str(prefix).is_none() {
                continue;
            }
            for dim_name in PARAM_DIM_NAMES {
                let key = format!("{prefix}{dim_name}");
                assert!(split_param_name(&key).is_some());
                assert_eq!(
                    fresh.param_name_to_val(&key),
                    Some(ParamVal(-1.0)),
                    "{key}"
                );
                resolved += 1;
            }
        }
        assert_eq!(resolved, 240);

        // IBM's `.at()` throw, which is the ONLY thing absence encodes here.
        assert_eq!(dsc.param_name_to_val("nx1"), None);
        assert_eq!(dsc.param_name_to_val("nq"), None);

        *dsc.param_name_to_val_mut("telmb").unwrap() = dim(6.0);
        assert_eq!(dsc.tel.mb, dim(6.0));
        assert_eq!(dsc.param_name_to_val("telmb"), Some(ParamVal(6.0)));
    }

    /// `dsc/designSpaceConfig.cpp:416-427`: `dbin` is `din` over `bin`, i.e. `CoreD_.in_` over
    /// `B_.in_`, and the three loops whose spelling does not decompose have no count.
    #[test]
    fn loop_count_divides_the_numerator_stage_by_the_denominator_stage() {
        let mut dsc = DesignSpaceConfig::default();
        dsc.core_d.r#in = dim(64.0);
        dsc.b.r#in = dim(16.0);
        dsc.t.ij = dim(12.0);
        dsc.p.ij = dim(5.0);

        assert_eq!(
            dsc.loop_count(LoopNames::DbIn),
            LoopTripCount::Is(LoopCount(4))
        );
        // Truncating, as IBM's `int loopCount = double / double` is.
        assert_eq!(
            dsc.loop_count(LoopNames::TpIj),
            LoopTripCount::Is(LoopCount(2))
        );

        // IBM's `DT_CHECK(loop != INNER)`, and the two spellings that decompose to no key.
        assert_eq!(dsc.loop_count(LoopNames::Inner), LoopTripCount::NoSuchLoop);
        assert_eq!(dsc.loop_count(LoopNames::Const), LoopTripCount::NoSuchLoop);
        assert_eq!(dsc.loop_count(LoopNames::Invalid), LoopTripCount::NoSuchLoop);

        // Measured against the authority on THIS fixture: an unfilled stage still divides, by its
        // `-1`. `BtIn` is `bin` over an unset `tin`, i.e. `int(16 / -1)`; `BtIj` is an unset `bij`
        // over `tij`, i.e. `int(-1 / 12)`; and every loop neither stage filled reads `int(-1 / -1)`.
        assert_eq!(
            dsc.loop_count(LoopNames::BtIn),
            LoopTripCount::Is(LoopCount(-16))
        );
        assert_eq!(
            dsc.loop_count(LoopNames::BtIj),
            LoopTripCount::Is(LoopCount(0))
        );
        assert_eq!(
            dsc.loop_count(LoopNames::DbOut),
            LoopTripCount::Is(LoopCount(1))
        );

        // A FILLED zero denominator is the one case with no answer: IBM's `int(x / 0.0)`.
        dsc.t.r#in = dim(0.0);
        assert_eq!(
            dsc.loop_count(LoopNames::BtIn),
            LoopTripCount::UndefinedZeroDenominator
        );
    }

    /// `dsc/designSpaceConfig.cpp:9222-9254`: the printed spellings, their flip, and the DM table
    /// that is NOT that flip and maps two spellings onto `INNER`.
    #[test]
    fn loop_names_round_trip_and_the_dm_table_is_a_separate_one() {
        for loop_name in LoopNames::ALL {
            assert_eq!(LoopNames::from_name(loop_name.name()), Some(loop_name));
        }
        assert_eq!(LoopNames::ALL.len(), 32);

        assert_eq!(LoopNames::from_name_dm("pcompute"), Some(LoopNames::Inner));
        assert_eq!(LoopNames::from_name_dm("Inner"), Some(LoopNames::Inner));
        assert_eq!(LoopNames::from_name_dm("din"), Some(LoopNames::DbIn));
        // The DM table names no `dbi`/`dbj`/`dbki`/`dbkj`, and no `dbin` either.
        assert_eq!(LoopNames::from_name_dm("dbin"), None);
        assert_eq!(LoopNames::from_name_dm("di"), None);

        for ds_type in DsTypes::ALL {
            assert_eq!(DsTypes::from_name(ds_type.name()), Some(ds_type));
        }
        assert_eq!(DsTypes::from_name("SCRATCH"), None);
        // `ALL` is in the enum's order, which is the order `std::map` iterates the table by key.
        assert_eq!(DsTypes::ALL.map(|ds| ds as usize), [0, 1, 2, 3, 4, 5, 6, 7]);
    }

    /// `dsc/designSpaceConfig.cpp:8302-8315`: the three stages are contiguous discriminant ranges,
    /// and `INNER`, `CONST` and `INVALID` are in none of them.
    #[test]
    fn each_loop_belongs_to_exactly_one_stage_and_the_three_specials_to_none() {
        for loop_name in LoopNames::ALL {
            let stages = [LoopStage::Tp, LoopStage::Bt, LoopStage::Db]
                .into_iter()
                .filter(|stage| loop_name.is_stage(*stage))
                .count();
            let expected = match loop_name {
                LoopNames::Inner | LoopNames::Const | LoopNames::Invalid => 0,
                _ => 1,
            };
            assert_eq!(stages, expected, "{loop_name:?}");
        }
        assert!(LoopNames::DbKj.is_stage(LoopStage::Db));
        assert!(!LoopNames::DbKj.is_stage(LoopStage::Bt));
        assert!(LoopNames::BtY.is_stage(LoopStage::Bt));
        assert!(LoopNames::TpIn.is_stage(LoopStage::Tp));
    }

    /// `dsc/dsc2.cpp:4007-4037` and `dsc/designSpaceConfig.cpp:429-438`: the layout order positions a
    /// dim, and a role with no entry answers a DIFFERENT position from a dim the order does not hold.
    #[test]
    fn layout_and_stick_dims_come_from_the_role_and_the_two_absences_stay_apart() {
        let mut dsc = DesignSpaceConfig::default();
        dsc.primary_ds_info.insert(
            DsTypes::Input,
            PrimaryDsInfo {
                layout_dim_order: vec![PrimaryDimTypes::Mb, PrimaryDimTypes::Ij],
                stick_dim_order: vec![PrimaryDimTypes::Out, PrimaryDimTypes::Ij],
                stick_size: vec![StickSize(32.0), StickSize(8.0)],
                stick_repl: vec![StickRepl(1), StickRepl(2)],
            },
        );

        assert_eq!(
            dsc.dim_index_in_layout_order(DsTypes::Input, PrimaryDimTypes::Ij),
            LayoutOrderPosition::At(1)
        );
        // IBM's in-band `-1`: the dim is not in this layout, which sixteen readers turn into a scale
        // of 1 and carry on with.
        assert_eq!(
            dsc.dim_index_in_layout_order(DsTypes::Input, PrimaryDimTypes::X),
            LayoutOrderPosition::DimNotInOrder
        );
        // IBM's `.at()` throw: `primaryDsInfo_` has no entry for the role at all.
        assert_eq!(
            dsc.dim_index_in_layout_order(DsTypes::Kernel, PrimaryDimTypes::Ij),
            LayoutOrderPosition::RoleHasNoLayout
        );
        // The point of the two variants: one absent would make these two equal.
        assert_ne!(
            dsc.dim_index_in_layout_order(DsTypes::Input, PrimaryDimTypes::X),
            dsc.dim_index_in_layout_order(DsTypes::Kernel, PrimaryDimTypes::Ij)
        );

        assert_eq!(
            dsc.layout_dim_set(DsTypes::Input),
            Some(BTreeSet::from([PrimaryDimTypes::Mb, PrimaryDimTypes::Ij]))
        );
        assert_eq!(
            dsc.stick_dims(DsTypes::Input),
            Some([PrimaryDimTypes::Out, PrimaryDimTypes::Ij].as_slice())
        );
        assert_eq!(
            dsc.stick_dim_set(DsTypes::Input),
            Some(BTreeSet::from([PrimaryDimTypes::Out, PrimaryDimTypes::Ij]))
        );
        assert!(dsc.stick_dims(DsTypes::Output).is_none());
    }

    /// `dm/dm.cpp:971-983` and `dsc/designSpaceConfig.cpp:6930-6931`: an `sc_` entry's own `name_`
    /// prefixes twenty more `paramNameToVal` keys, and that name is the only way into the vector.
    #[test]
    fn an_sc_entrys_own_name_prefixes_twenty_more_param_keys() {
        let mut dsc = DesignSpaceConfig::default();
        let mut entry = DataStructDims {
            name: "sc_0".to_string(),
            ..DataStructDims::default()
        };
        *entry.param_name_to_val_mut("ij").unwrap() = dims::DimSize::new(48.0);
        dsc.sc.push(entry);

        assert_eq!(dsc.param_name_to_val("sc_0ij"), Some(ParamVal(48.0)));
        // The name alone is not a key, and a dim outside the twenty is not one either.
        assert_eq!(dsc.param_name_to_val("sc_0"), None);
        assert_eq!(dsc.param_name_to_val("sc_0x1"), None);
        // An entry's other nineteen keys answer their unfilled `-1`, as the twelve prefixes' do.
        assert_eq!(dsc.param_name_to_val("sc_0mb"), Some(ParamVal(-1.0)));
        // The twelve-prefix table still answers for its own keys.
        *dsc.param_name_to_val_mut("nij").unwrap() = dims::DimSize::new(9.0);
        assert_eq!(dsc.param_name_to_val("nij"), Some(ParamVal(9.0)));
    }

    /// `dm/dm.cpp:980-983`: entry 0 is registered before entry 1 and `map[key] = ptr` overwrites, so
    /// two entries sharing a `name_` leave the LATER one holding the twenty keys.
    #[test]
    fn the_last_sc_entry_sharing_a_name_holds_the_key() {
        let mut dsc = DesignSpaceConfig::default();
        for size in [3.0, 7.0] {
            let mut entry = DataStructDims {
                name: "sc_0".to_string(),
                ..DataStructDims::default()
            };
            *entry.param_name_to_val_mut("mb").unwrap() = dims::DimSize::new(size);
            dsc.sc.push(entry);
        }

        assert_eq!(dsc.param_name_to_val("sc_0mb"), Some(ParamVal(7.0)));
    }

    /// `dsc/designSpaceConfig.cpp:9451-9486`: each is a PRODUCT over the entries naming that dim, a
    /// dim outside the stick is the empty product `1`, and a length mismatch is IBM's `DT_ERROR`.
    #[test]
    fn stick_queries_multiply_every_entry_naming_the_dim() {
        let pdsi = PrimaryDsInfo {
            layout_dim_order: Vec::new(),
            stick_dim_order: vec![
                PrimaryDimTypes::Ij,
                PrimaryDimTypes::Out,
                PrimaryDimTypes::Ij,
            ],
            stick_size: vec![StickSize(4.0), StickSize(32.0), StickSize(2.0)],
            stick_repl: vec![StickRepl(1), StickRepl(1), StickRepl(3)],
        };
        assert_eq!(pdsi.stick(PrimaryDimTypes::Ij), Some(StickSize(8.0)));
        assert_eq!(pdsi.stick_repl(PrimaryDimTypes::Ij), Some(StickRepl(3)));
        assert_eq!(
            pdsi.stick_srpdt(PrimaryDimTypes::Ij),
            Some(StickSizeWithRepl(24.0))
        );
        assert_eq!(pdsi.stick(PrimaryDimTypes::X), Some(StickSize(1.0)));

        let ragged = PrimaryDsInfo {
            stick_size: vec![StickSize(4.0)],
            ..pdsi
        };
        assert_eq!(ragged.stick(PrimaryDimTypes::Ij), None);
    }

    /// `dsc/dsc2.cpp:4066-4106`: 32 x 8 elements over eight slices leaves 32 in a slice, so the slice
    /// scope stops after the first dim and the without-slice scope reports only what is left.
    #[test]
    fn stick_sizes_split_the_stick_at_its_slice_boundary() {
        let mut dsc = DesignSpaceConfig::default();
        dsc.primary_ds_info.insert(
            DsTypes::Input,
            PrimaryDsInfo {
                layout_dim_order: vec![PrimaryDimTypes::Mb],
                stick_dim_order: vec![PrimaryDimTypes::Out, PrimaryDimTypes::Ij],
                stick_size: vec![StickSize(32.0), StickSize(8.0)],
                stick_repl: vec![StickRepl(1), StickRepl(1)],
            },
        );

        let whole = dsc.stick_sizes(DsTypes::Input, StickSizeScope::WholeStick);
        assert_eq!(
            whole,
            Some(vec![
                Size::new(PrimaryDimTypes::Out, dsc2::DimSize(32)),
                Size::new(PrimaryDimTypes::Ij, dsc2::DimSize(8)),
            ])
        );
        assert_eq!(
            dsc.stick_sizes(DsTypes::Input, StickSizeScope::SliceOnly),
            Some(vec![Size::new(PrimaryDimTypes::Out, dsc2::DimSize(32))])
        );
        assert_eq!(
            dsc.stick_sizes(DsTypes::Input, StickSizeScope::WithoutSlice),
            Some(vec![Size::new(PrimaryDimTypes::Ij, dsc2::DimSize(8))])
        );
        // Four L0 slices leave 64 in a slice, which swallows the second dim too.
        assert_eq!(
            dsc.stick_sizes(
                DsTypes::Input,
                StickSizeScope::L0SliceOnly(NumL0Slices::new(4).unwrap())
            ),
            Some(vec![
                Size::new(PrimaryDimTypes::Out, dsc2::DimSize(32)),
                Size::new(PrimaryDimTypes::Ij, dsc2::DimSize(2)),
            ])
        );
        // IBM's `DT_CHECK(elemInSlice > 0 && elemInSlice % numSlices == 0)`.
        assert_eq!(NumL0Slices::new(0), None);
        assert_eq!(NumL0Slices::new(-1), None);
        assert_eq!(
            dsc.stick_sizes(
                DsTypes::Input,
                StickSizeScope::L0SliceOnly(NumL0Slices::new(7).unwrap())
            ),
            None
        );

        assert_eq!(
            dsc.cumulative_stick_sizes(DsTypes::Input, StickSizeScope::WholeStick),
            Some(BTreeMap::from([
                (PrimaryDimTypes::Out, dsc2::DimSize(32)),
                (PrimaryDimTypes::Ij, dsc2::DimSize(8)),
            ]))
        );
    }

    /// `dsc/dsc2.cpp:4108-4124`: a dim appearing twice in the stick has its extents MULTIPLIED, which
    /// is the whole difference between this and `getStickSizes`.
    #[test]
    fn cumulative_stick_sizes_multiply_a_repeated_dim() {
        let mut dsc = DesignSpaceConfig::default();
        dsc.primary_ds_info.insert(
            DsTypes::Kernel,
            PrimaryDsInfo {
                layout_dim_order: Vec::new(),
                stick_dim_order: vec![
                    PrimaryDimTypes::Ij,
                    PrimaryDimTypes::Out,
                    PrimaryDimTypes::Ij,
                ],
                stick_size: vec![StickSize(4.0), StickSize(2.0), StickSize(8.0)],
                stick_repl: vec![StickRepl(1), StickRepl(1), StickRepl(1)],
            },
        );
        assert_eq!(
            dsc.cumulative_stick_sizes(DsTypes::Kernel, StickSizeScope::WholeStick),
            Some(BTreeMap::from([
                (PrimaryDimTypes::Out, dsc2::DimSize(2)),
                (PrimaryDimTypes::Ij, dsc2::DimSize(32)),
            ]))
        );
    }

    /// `dsc/designSpaceConfig.cpp:930-946`: rows and columns each lose twice their zero padding, and
    /// the `-1` answer needs BOTH to be negative.
    #[test]
    fn inp_in_hbm_subtracts_twice_the_zero_padding_from_each_axis() {
        let mut dsc = DesignSpaceConfig::default();
        dsc.n.r = dim(10.0);
        dsc.n.c = dim(20.0);
        dsc.n.zi = dim(1.0);
        dsc.n.zj = dim(2.0);

        assert_eq!(dsc.inp_row_in_hbm("n"), Some(HbmRows(8.0)));
        assert_eq!(dsc.inp_col_in_hbm("n"), Some(HbmCols(16.0)));
        assert_eq!(dsc.inp_in_hbm("n"), Some(HbmElements(128.0)));

        // Only one axis negative: the product is returned, negative.
        dsc.n.r = dim(1.0);
        assert_eq!(dsc.inp_in_hbm("n"), Some(HbmElements(-16.0)));
        // Both negative: IBM's -1.
        dsc.n.c = dim(1.0);
        assert_eq!(dsc.inp_in_hbm("n"), Some(HbmElements(-1.0)));
        // Measured against the authority: `chipdr` IS one of the 240 keys, so an UNFILLED `ChipD_`
        // reads `-1` and the row count is `-1 - 1 * 2 == -3` — not a throw, and not absence.
        assert_eq!(dsc.inp_row_in_hbm("chipd"), Some(HbmRows(-3.0)));
        assert_eq!(dsc.inp_col_in_hbm("chipd"), Some(HbmCols(-5.0)));
        assert_eq!(dsc.inp_in_hbm("chipd"), Some(HbmElements(-1.0)));
        // The real `.at()` throw: `CoreD_` is named `"d"`, so `"coredr"` is no key.
        assert_eq!(dsc.inp_row_in_hbm("cored"), None);
    }

    /// `dsc/designSpaceConfig.cpp:7814-7874`: the primary dims must cover, the auxiliary ones may be
    /// unfilled, `ij_` is only required when a layout uses it, and a fractional auxiliary dim still
    /// PASSES because that block sets `true` where it means `false` (`:7869-7871`).
    #[test]
    fn check_data_struct_dims_covers_primary_dims_and_lets_a_fractional_aux_dim_pass() {
        let mut dsc = DesignSpaceConfig::default();
        let mut big = DataStructDims::default();
        big.r#in = dim(8.0);
        big.out = dim(8.0);
        big.mb = dim(2.0);
        big.x = dim(4.0);
        big.y = dim(4.0);
        let mut small = big.clone();
        small.r#in = dim(4.0);

        let (ok, msg) = dsc.check_data_struct_dims(&big, &small);
        assert!(ok, "{msg}");
        assert_eq!(msg, "");

        let (ok, msg) = dsc.check_data_struct_dims(&small, &big);
        assert!(!ok);
        assert_eq!(
            msg,
            "all of values in primary fields should be equal or greater"
        );

        // `ij_` is unset on both sides, so it only matters once a layout names it.
        let mut with_ij = big.clone();
        with_ij.ij = dim(4.0);
        assert!(dsc.check_data_struct_dims(&big, &with_ij).0);
        dsc.primary_ds_info.insert(
            DsTypes::Input,
            PrimaryDsInfo {
                layout_dim_order: vec![PrimaryDimTypes::Ij],
                ..PrimaryDsInfo::default()
            },
        );
        assert!(!dsc.check_data_struct_dims(&big, &with_ij).0);

        // A fractional PRIMARY dim fails; a fractional AUXILIARY one passes with a message.
        let mut fractional_primary = big.clone();
        fractional_primary.mb = dim(2.5);
        assert!(!dsc.check_data_struct_dims(&fractional_primary, &small).0);
        let mut fractional_aux = big.clone();
        fractional_aux.zi = dim(1.5);
        let (ok, msg) = dsc.check_data_struct_dims(&fractional_aux, &small);
        assert!(ok);
        assert_eq!(
            msg,
            "all of values in auxiliary fields should have no fraction"
        );
    }

    /// `dsc/designSpaceConfig.cpp:9266-9271`: the three staging spellings and their flip.
    #[test]
    fn dt_type_names_round_trip() {
        for dt in DtType::ALL {
            assert_eq!(DtType::from_name(dt.name()), Some(dt));
        }
        assert_eq!(DtType::from_name("dblbuf"), None);
    }

    /// `dsc/designSpaceConfig.cpp:9272-9283`: the printed names are the Sen ops', not the
    /// enumerators', and all eight differ so the flip is total.
    #[test]
    fn ex_sen_ops_names_round_trip_and_are_not_the_enumerators() {
        for op in ExternalSenOps::ALL {
            assert_eq!(ExternalSenOps::from_name(op.name()), Some(op));
        }
        assert_eq!(
            ExternalSenOps::from_name("SenPreparedOp"),
            Some(ExternalSenOps::SenCompute)
        );
        assert_eq!(
            ExternalSenOps::from_name("SenDataConvert"),
            Some(ExternalSenOps::SenDataPrep)
        );
        assert_eq!(ExternalSenOps::from_name("SENCOMPUTE"), None);
    }

    /// `dsc/designSpaceConfig.cpp:9285-9448`: one vendor entry per operand bucket, the `-1` bucket,
    /// and a variant the table omits.
    #[test]
    fn op_func_in_outs_states_every_operand_bucket() {
        let counted = |op: OpFunc, n: u32, out: u32| {
            assert_eq!(
                op_func_in_outs(op),
                Some(OpFuncInOuts {
                    inputs: OpFuncInputs::Count(n),
                    outputs: OpFuncOutputs(out),
                }),
                "{op:?}"
            );
        };
        counted(OpFunc::None, 0, 0);
        counted(OpFunc::ConstPadOpLx, 0, 1);
        counted(OpFunc::Rsqrt, 1, 1);
        counted(OpFunc::Add, 2, 1);
        counted(OpFunc::Fnms, 3, 1);
        counted(OpFunc::ScaledGroupMatmulFp4Fwd, 4, 1);
        counted(OpFunc::LayernormNorm, 5, 1);
        assert_eq!(
            op_func_in_outs(OpFunc::SumNonstick),
            Some(OpFuncInOuts {
                inputs: OpFuncInputs::Unknown,
                outputs: OpFuncOutputs(1),
            })
        );
        assert_eq!(op_func_in_outs(OpFunc::Undef), None);
    }
}

// crustify:todo: e027_DesignSpaceConfig.addNewLine

// crustify:todo: e027_DesignSpaceConfig.allowMissingAlloc

// crustify:todo: e027_DesignSpaceConfig.allowSymbolicVolumeLimit

// crustify:todo: e027_DesignSpaceConfig.auxLoopOrder_

// crustify:todo: e027_DesignSpaceConfig.computeOp_

// crustify:todo: e027_DesignSpaceConfig.doNotRound

// crustify:todo: e027_DesignSpaceConfig.includeGaps

// crustify:todo: e027_DesignSpaceConfig.labeledDs_

// crustify:todo: e027_DesignSpaceConfig.numL0Slices

// crustify:todo: e027_DesignSpaceConfig.pcfg_

// crustify:todo: e027_DesignSpaceConfig.prog_frame_ptr_

// crustify:todo: e027_DesignSpaceConfig.ps

// crustify:todo: e027_DesignSpaceConfig.ptr_

// crustify:todo: e027_DesignSpaceConfig.scheduleTree_

// crustify:todo: e027_DesignSpaceConfig.sizeInNumberOfLoads

// crustify:todo: e027_DesignSpaceConfig.size_
