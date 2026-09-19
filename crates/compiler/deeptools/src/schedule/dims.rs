//! Re-ported from the C++ authority. See crustify-scheduler/AGENT-BRIEF.md.
//!
//! `dsc/dims.h` classifies a data structure's dimensions: which primary dim, which meta-dim kind,
//! how the dim is padded, and the padding scalars that produce its padded size.

use std::collections::{BTreeMap, BTreeSet};
use std::hash::{Hash, Hasher};

use sys_arch_spec::arch_enums::SenComponent;
use sys_arch_spec::{CoreletId, RowId};

/// Replaces: PrimaryDimTypes
///
/// `dsc/dims.h:34-47`. The twelve dimensions a data structure can carry, plus the authority's
/// `PrimaryDimTypesCount`, which every default-initialised dim field holds and which
/// `EnumsConversion::primaryDimToString` spells `"undefined"` (`dsc/dims.cpp:23`).
///
/// ⛔ THE DISCRIMINANTS ARE THE AUTHORITY'S AND ORDERED CONTAINERS DEPEND ON THEM. A `std::map`
/// keyed by this enum iterates in `int` order, and that order reaches IBM's exported JSON
/// (`dsc/dims.cpp:286`), so reordering a variant reorders the reference output.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PrimaryDimTypes {
    In = 0,
    Out = 1,
    Ij = 2,
    Mb = 3,
    X = 4,
    Y = 5,
    Kij = 6,
    I = 7,
    J = 8,
    Ki = 9,
    Kj = 10,
    X1 = 11,
    /// The authority's `PrimaryDimTypesCount` (`dsc/dims.h:47`): no dimension. It is a live map key
    /// as well as a field value — `ddc/ddcv1.cpp:1803` skips it while iterating a keyed map — so it
    /// stays a variant, and stays last, rather than becoming an absent `Option`.
    #[default]
    Undefined = 12,
}

/// ⛔ E0308 IF A DIM IS EVER INSERTED, DROPPED OR LEFT OUT OF `ALL`: the sentinel's discriminant is
/// the count of the real dims, which is the bound of the authority's own dim loop
/// (`ddc/ddl/ddl_conversion.h:309`).
const _: [(); PrimaryDimTypes::COUNT] = [(); PrimaryDimTypes::Undefined as usize];

/// ⛔ E0308 IF `ALL`, `NON_COMPOUND` AND `is_compound` EVER DISAGREE: the dims the authority's dim
/// loops admit plus the ones they drop are all of them (`ddc/ddl/ddl_conversion.h:309-310`).
const _: [(); PrimaryDimTypes::COUNT] = [(); PrimaryDimTypes::NON_COMPOUND.len() + {
    let mut compound = 0;
    let mut i = 0;
    while i < PrimaryDimTypes::COUNT {
        if PrimaryDimTypes::ALL[i].is_compound() {
            compound += 1;
        }
        i += 1;
    }
    compound
}];

impl PrimaryDimTypes {
    /// The real dimensions in the authority's order — every value `0..PrimaryDimTypesCount` takes
    /// (`ddc/ddl/ddl_conversion.h:309`). The sentinel is not one of them.
    ///
    /// ⛔ NOT THE SET THE AUTHORITY'S DIM LOOPS ADMIT. That loop drops `IJ` and `KIJ` from what it
    /// inserts (`ddc/ddl/ddl_conversion.h:310`), so a candidate set built from `ALL` is two dims
    /// too wide. `NON_COMPOUND` is that set.
    pub const ALL: [Self; 12] = [
        Self::In,
        Self::Out,
        Self::Ij,
        Self::Mb,
        Self::X,
        Self::Y,
        Self::Kij,
        Self::I,
        Self::J,
        Self::Ki,
        Self::Kj,
        Self::X1,
    ];

    /// How many real dimensions there are — the authority's `PrimaryDimTypesCount` as a count.
    pub const COUNT: usize = Self::ALL.len();

    /// `ALL` in order, less the compound pair: the dims `DimProp` admits as candidates
    /// (`ddc/ddl/ddl_conversion.h:309-310`) and the dims `getClSplitDim` compares across corelets
    /// (`ddc/ddcv1.cpp:1802-1803`, which drops the sentinel too — it walks a keyed map).
    pub const NON_COMPOUND: [Self; 10] = [
        Self::In,
        Self::Out,
        Self::Mb,
        Self::X,
        Self::Y,
        Self::I,
        Self::J,
        Self::Ki,
        Self::Kj,
        Self::X1,
    ];

    /// Whether the authority computes this dim from two halves rather than storing it — what its
    /// own `DT_ERROR` calls a "compound dim" (`dsc/dims.cpp:571-572`), written by
    /// `DataStructDims::compound` (`dsc/dims.cpp:84-110`).
    ///
    /// ⭐ ONE NAME FOR AN EXCLUSION THREE AUTHORITY SITES SPELL BY HAND: `is_any_of(d, IJ, KIJ)` at
    /// `dsc/dims.cpp:572` and `ddc/ddl/ddl_conversion.h:310`, and with the sentinel at
    /// `ddc/ddcv1.cpp:1803`.
    pub const fn is_compound(self) -> bool {
        matches!(self, Self::Ij | Self::Kij)
    }

    /// The spelling `EnumsConversion::primaryDimToString` gives this dim (`dsc/dims.cpp:21-35`).
    ///
    /// ⭐ TOTAL WHERE THE MAP LOOKUP WAS NOT: IBM's `.at()` throws for a dim the map omits.
    pub fn name(self) -> &'static str {
        match self {
            Self::In => "in",
            Self::Out => "out",
            Self::Ij => "ij",
            Self::Mb => "mb",
            Self::X => "x",
            Self::Y => "y",
            Self::Kij => "kij",
            Self::I => "i",
            Self::J => "j",
            Self::Ki => "ki",
            Self::Kj => "kj",
            Self::X1 => "x1",
            Self::Undefined => "undefined",
        }
    }

    /// `EnumsConversion::stringToPrimaryDim`, the flip of the above (`dsc/dims.cpp:36-37`), as read
    /// by `FromString<PrimaryDimTypes>` (`dsc/dims.h:129-132`). An unknown name is absent, where
    /// IBM's `.at()` throws.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "in" => Some(Self::In),
            "out" => Some(Self::Out),
            "ij" => Some(Self::Ij),
            "mb" => Some(Self::Mb),
            "x" => Some(Self::X),
            "y" => Some(Self::Y),
            "kij" => Some(Self::Kij),
            "i" => Some(Self::I),
            "j" => Some(Self::J),
            "ki" => Some(Self::Ki),
            "kj" => Some(Self::Kj),
            "x1" => Some(Self::X1),
            "undefined" => Some(Self::Undefined),
            _ => None,
        }
    }
}

/// Replaces: PadType
///
/// `dsc/dims.h:50-57`. How one dimension is padded. The window-pad forms apply to conv/pooling
/// dims only and the full-span forms to non-window-pad dims only, per the authority's comments.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PadType {
    /// The absent-entry default of `PaddingFormType::getPadding` (`dsc/dims.cpp:809-810`).
    #[default]
    NoPad,
    LoweredPadded,
    PaddedNoZeroPad,
    PaddedWZeroPad,
    PaddedFullSpan,
    PaddedFullSpanWUnneeded,
}

impl PadType {
    /// Every padding form, in the authority's order (`dsc/dims.h:50-57`).
    pub const ALL: [Self; 6] = [
        Self::NoPad,
        Self::LoweredPadded,
        Self::PaddedNoZeroPad,
        Self::PaddedWZeroPad,
        Self::PaddedFullSpan,
        Self::PaddedFullSpanWUnneeded,
    ];

    /// The spelling `EnumsConversion::padTypeToString` gives this form (`dsc/dims.cpp:39-46`).
    pub fn name(self) -> &'static str {
        match self {
            Self::NoPad => "nopad",
            Self::LoweredPadded => "lowered_padded",
            Self::PaddedNoZeroPad => "padded_nozeropad",
            Self::PaddedWZeroPad => "padded_wzeropad",
            Self::PaddedFullSpan => "padded_fullspan",
            Self::PaddedFullSpanWUnneeded => "padded_fullspan_wunneeded",
        }
    }

    /// `EnumsConversion::stringToPadType`, the flip of the above (`dsc/dims.cpp:47-48`). An unknown
    /// name is absent, where IBM's `.at()` throws.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "nopad" => Some(Self::NoPad),
            "lowered_padded" => Some(Self::LoweredPadded),
            "padded_nozeropad" => Some(Self::PaddedNoZeroPad),
            "padded_wzeropad" => Some(Self::PaddedWZeroPad),
            "padded_fullspan" => Some(Self::PaddedFullSpan),
            "padded_fullspan_wunneeded" => Some(Self::PaddedFullSpanWUnneeded),
            _ => None,
        }
    }
}

/// Replaces: MetaDimKind
///
/// `dsc/dims.h:59-69`. Which quantity of a padded dimension is meant: the dim itself padded or
/// unpadded, one of its padding scalars, or its window/stride/dilation. `Undefined` is the
/// authority's `Count`, which `EnumsConversion::stringToMetaDimKind` spells `"undefined"`
/// (`dsc/dims.cpp:55`) and whose value also sets the hash shift below.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MetaDimKind {
    #[default]
    Unpadded = 0,
    Padded = 1,
    PadFront = 2,
    PadBack = 3,
    PadValid = 4,
    WindowDim = 5,
    Stride = 6,
    Dilation = 7,
    /// The authority's `MetaDimKind::Count` (`dsc/dims.h:68`).
    Undefined = 8,
}

/// ⛔ E0308 IF A KIND IS EVER INSERTED, DROPPED OR LEFT OUT OF `ALL`: the sentinel's discriminant
/// is the count of the real kinds, and that count is what `HASH_SHIFT` is derived from.
const _: [(); MetaDimKind::COUNT] = [(); MetaDimKind::Undefined as usize];

impl MetaDimKind {
    /// The real kinds in the authority's order (`dsc/dims.h:59-69`). The sentinel is not one.
    pub const ALL: [Self; 8] = [
        Self::Unpadded,
        Self::Padded,
        Self::PadFront,
        Self::PadBack,
        Self::PadValid,
        Self::WindowDim,
        Self::Stride,
        Self::Dilation,
    ];

    /// How many real kinds there are — the authority's `MetaDimKind::Count` as a count.
    pub const COUNT: usize = Self::ALL.len();

    /// Bits the dim is shifted by in `std::hash<PrimaryDimAndKind>`: IBM writes
    /// `int(std::log2(int(MetaDimKind::Count)) + 1)` (`dsc/dims.h:90`), which is 4 here. Derived
    /// from `COUNT` rather than written as 4, so the two cannot drift apart.
    const HASH_SHIFT: u32 = Self::COUNT.ilog2() + 1;

    /// The spelling `EnumsConversion::metaDimKindToString` gives this kind (`dsc/dims.cpp:50-57`).
    pub fn name(self) -> &'static str {
        match self {
            Self::Unpadded => "unpadded",
            Self::Padded => "padded",
            Self::PadFront => "pad_front",
            Self::PadBack => "pad_back",
            Self::PadValid => "pad_valid",
            Self::WindowDim => "window",
            Self::Stride => "stride",
            Self::Dilation => "dilation",
            Self::Undefined => "undefined",
        }
    }

    /// `EnumsConversion::stringToMetaDimKind` (`dsc/dims.cpp:50-55`). An unknown name is absent,
    /// where IBM's `.at()` throws.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "unpadded" => Some(Self::Unpadded),
            "padded" => Some(Self::Padded),
            "pad_front" => Some(Self::PadFront),
            "pad_back" => Some(Self::PadBack),
            "pad_valid" => Some(Self::PadValid),
            "window" => Some(Self::WindowDim),
            "stride" => Some(Self::Stride),
            "dilation" => Some(Self::Dilation),
            "undefined" => Some(Self::Undefined),
            _ => None,
        }
    }
}

/// Replaces: e002_PrimaryDimAndKind
/// Replaces: e040_PrimaryDimAndKind
///
/// `dsc/dims.h:76-82`. One dimension together with which of its quantities is meant — the key
/// every loop carries in `dims_` and the element of every split-dim set.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct PrimaryDimAndKind {
    /// Field: e002_PrimaryDimAndKind.dim_
    /// Field: e040_PrimaryDimAndKind.dim_
    pub dim: PrimaryDimTypes,
    /// Field: e002_PrimaryDimAndKind.kind_
    /// Field: e040_PrimaryDimAndKind.kind_
    pub kind: MetaDimKind,
}

impl PrimaryDimAndKind {
    /// `dsc/dims.h:79-81`.
    pub fn new(dim: PrimaryDimTypes, kind: MetaDimKind) -> Self {
        Self { dim, kind }
    }

    /// IBM's `std::hash<PrimaryDimAndKind>` value: `(dim << 4) ^ kind` (`dsc/dims.h:88-92`).
    ///
    /// ⛔ IT IS OBSERVABLE, AND THIS FUNCTION ALONE DOES NOT REPRODUCE IT.
    /// `ddc/ddc_transformation_util.cpp:170` declares an `unordered_map` keyed by this type, `:203-206`
    /// iterates it into the dim list of a loop it is about to mint, and that same list spells the
    /// node's `name_` (`:151-153` for a new loop, `:253-254` for the base one) — so libstdc++'s
    /// bucket order reaches both the minted tree and its exported names. Matching that needs
    /// libstdc++'s bucket policy, not the hash.
    ///
    /// ⭐ AN ORDERED CONTAINER IS THE DETERMINISTIC ANSWER, AND IT LOSES NOTHING OF IBM'S NUMBER:
    /// `MetaDimKind` never reaches `1 << HASH_SHIFT`, so the XOR is an addition and this value is
    /// strictly increasing in `(dim_, kind_)`. The derived `Ord` above therefore visits keys in
    /// increasing IBM-hash order.
    pub fn hash_value(self) -> usize {
        ((self.dim as usize) << MetaDimKind::HASH_SHIFT) ^ (self.kind as usize)
    }
}

/// IBM's hash value, so the one number the authority defines is all that is fed to a hasher.
///
/// ⛔ THIS DOES NOT GIVE A `HashMap` IBM'S BUCKET ORDER. `RandomState` runs SipHash over these
/// bytes under a per-process seed, so the bucket — and the iteration order — is Rust's, and is not
/// stable across runs. `hash_value` is where IBM's number is observable, and it is what the
/// order-sensitive sites reach through a `BTreeMap`/`BTreeSet` instead.
impl Hash for PrimaryDimAndKind {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_usize(self.hash_value());
    }
}

/// A bare dim means that dim unpadded — IBM's one-argument constructor is not `explicit`, and
/// `ddc/ddc_transformation_util.cpp:1147` relies on the implicit conversion to build a set of
/// these from a `vector<PrimaryDimTypes>` (`dsc/dims.h:79-81`).
impl From<PrimaryDimTypes> for PrimaryDimAndKind {
    fn from(dim: PrimaryDimTypes) -> Self {
        Self::new(dim, MetaDimKind::Unpadded)
    }
}

/// Replaces: e003_PaddingFormType
/// Replaces: e039_PaddingFormType
///
/// `dsc/dims.h:94-120`. The padding form of each dimension of one allocation, transfer or
/// coordinate. A dim absent from the map is unpadded, so the empty form is the default one every
/// `const PaddingFormType &padded = {}` parameter takes.
///
/// IBM's three member typedefs (`PerDimPaddingInfoT`, `iterator`, `const_iterator`,
/// `dsc/dims.h:96-98`) are the map type and its two iterators; they name no stored value, so they
/// carry no `Field:` anchor. `iter` below is the pair of them.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PaddingFormType {
    /// Field: e003_PaddingFormType.padding_
    /// Field: e039_PaddingFormType.padding_
    ///
    /// IBM's `PerDimPaddingInfoT`, a `std::map` (`dsc/dims.h:96`): ordered, and the order is
    /// exported (`dsc/dsc2.cpp:846-848`), which is why this is a `BTreeMap` and not a `HashMap`.
    padding: BTreeMap<PrimaryDimTypes, PadType>,
}

impl PaddingFormType {
    /// The one-dim form (`dsc/dims.h:104`).
    pub fn new(dim: PrimaryDimTypes, pad: PadType) -> Self {
        Self {
            padding: BTreeMap::from([(dim, pad)]),
        }
    }

    /// This dim's padding form, `NoPad` when the dim has no entry (`dsc/dims.cpp:806-813`).
    pub fn padding(&self, dim: PrimaryDimTypes) -> PadType {
        self.padding.get(&dim).copied().unwrap_or(PadType::NoPad)
    }

    /// That form's spelling (`dsc/dims.cpp:815-817`).
    pub fn padding_as_str(&self, dim: PrimaryDimTypes) -> &'static str {
        self.padding(dim).name()
    }

    /// Set this dim's form, replacing any previous one (`dsc/dims.cpp:819-821`).
    pub fn set_padding(&mut self, dim: PrimaryDimTypes, pad: PadType) {
        self.padding.insert(dim, pad);
    }

    /// Drop every dim's form (`dsc/dims.h:105`).
    pub fn clear(&mut self) {
        self.padding.clear();
    }

    /// Every dim that carries a form, in dim order — IBM's `begin`/`end` pair (`dsc/dims.h:110-113`)
    /// exists for JSON import/export, and no in-scope caller mutates through it.
    pub fn iter(&self) -> impl Iterator<Item = (PrimaryDimTypes, PadType)> + '_ {
        self.padding.iter().map(|(&dim, &pad)| (dim, pad))
    }

    /// Whether any dim carries a form (`dsc/dims.h:114`).
    pub fn has_padding_info(&self) -> bool {
        !self.padding.is_empty()
    }

    /// Append the human-readable form IBM streams (`dsc/dims.cpp:823-830`), as
    /// `DesignSpaceConfig` does at `dsc/dsc2.cpp:4529`. `level` indents by two spaces each.
    pub fn print(&self, out: &mut String, level: usize) {
        out.push('\n');
        for _ in 0..level * 2 {
            out.push(' ');
        }
        out.push_str("Padding= ");
        for (dim, pad) in self.iter() {
            out.push_str(" (");
            out.push_str(dim.name());
            out.push_str(": ");
            out.push_str(pad.name());
            out.push(')');
        }
    }
}

/// Replaces: e004_DimPaddingSizes
/// Replaces: e021_DimPaddingSizes
///
/// `dsc/dims.h:134-146`. Everything that contributes to one primary dimension's padded size.
///
/// ⛔ SIGNED, AND `-1` IS LIVE: `dsc/dims.cpp:295` tests `padFront_ < 0` together with an absent
/// `windowDim_` to decide that a dim has no total size at all.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DimPaddingSizes {
    /// Field: e004_DimPaddingSizes.padFront_
    /// Field: e021_DimPaddingSizes.padFront_
    pub pad_front: i32,
    /// Field: e004_DimPaddingSizes.padBack_
    /// Field: e021_DimPaddingSizes.padBack_
    pub pad_back: i32,
    /// Field: e004_DimPaddingSizes.unneededPad_
    /// Field: e021_DimPaddingSizes.unneededPad_
    ///
    /// Total unneeded elements.
    pub unneeded_pad: i32,
    /// Field: e004_DimPaddingSizes.unneededPadFront_
    /// Field: e021_DimPaddingSizes.unneededPadFront_
    ///
    /// Unneeded elements that come from `pad_front`.
    pub unneeded_pad_front: i32,
    /// Field: e004_DimPaddingSizes.unneededPadBack_
    /// Field: e021_DimPaddingSizes.unneededPadBack_
    ///
    /// Unneeded elements that come from `pad_back`.
    pub unneeded_pad_back: i32,
    /// Field: e004_DimPaddingSizes.stride_
    /// Field: e021_DimPaddingSizes.stride_
    pub stride: i32,
    /// Field: e004_DimPaddingSizes.dilation_
    /// Field: e021_DimPaddingSizes.dilation_
    pub dilation: i32,
    /// Field: e004_DimPaddingSizes.windowDim_
    /// Field: e021_DimPaddingSizes.windowDim_
    ///
    /// The window dim this padding is associated with; `Undefined` means none, which is the test
    /// every caller writes (`ddc/ddcv1.cpp:1180`, `dcg/dcg_fe/pcfg_gen/dlOpsNew.cpp:315`).
    pub window_dim: PrimaryDimTypes,
}

/// The authority's member initialisers (`dsc/dims.h:135-142`): zero padding, unit stride and
/// dilation, no window dim.
impl Default for DimPaddingSizes {
    fn default() -> Self {
        Self {
            pad_front: 0,
            pad_back: 0,
            unneeded_pad: 0,
            unneeded_pad_front: 0,
            unneeded_pad_back: 0,
            stride: 1,
            dilation: 1,
            window_dim: PrimaryDimTypes::Undefined,
        }
    }
}

impl DimPaddingSizes {
    /// The scalar this kind names, for the four kinds that name one (`dsc/dims.cpp:59-72`).
    ///
    /// ⭐ ABSENT RATHER THAN A THROW, AND ONLY TWO KINDS REACH IT. IBM `DT_ERROR`s on the other
    /// five, but the one caller filters `Padded`, `Unpadded` and `WindowDim` into a different branch
    /// one line earlier (`ddc/ddl/ddl_conversion.cpp:1824-1826`), so what can arrive here unhandled
    /// is `PadValid` and the DDL-settable sentinel alone (`"undefined"` → `MetaDimKind::Count`,
    /// `dsc/dims.cpp:53`).
    ///
    /// ⛔ TRAP — THE CALLER DIVIDES BY THIS AND A DEFAULT `DimPaddingSizes` ANSWERS `0`:
    /// `scale /= ....getMetaDimVal(kind)` (`ddc/ddl/ddl_conversion.cpp:1828-1831`) is a `float`
    /// divide, `padFront_`/`padBack_` initialise to `0` (`dsc/dims.h:135-136`), and the `inf` then
    /// multiplies into a datastage constraint's min and max (`:1838-1844`). The non-zero obligation
    /// is the constraint port's, not this accessor's — IBM returns the `0`.
    pub fn meta_dim_val(self, kind: MetaDimKind) -> Option<i32> {
        match kind {
            MetaDimKind::Dilation => Some(self.dilation),
            MetaDimKind::Stride => Some(self.stride),
            MetaDimKind::PadFront => Some(self.pad_front),
            MetaDimKind::PadBack => Some(self.pad_back),
            MetaDimKind::Unpadded
            | MetaDimKind::Padded
            | MetaDimKind::PadValid
            | MetaDimKind::WindowDim
            | MetaDimKind::Undefined => None,
        }
    }
}

/// Replaces: e005_SymbolicDimInfo
/// Replaces: e044_SymbolicDimInfo
///
/// `dsc/dims.h:148-155`. The max and granularity of one symbolic dimension. Whether a dim is
/// symbolic at all is its presence in `DataStructDims::symbolicDimInfo_`, never these values
/// (`dsc/dims.cpp:734`), so `-1` is "not filled in" and not a state anyone tests.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SymbolicDimInfo {
    /// Field: e005_SymbolicDimInfo.maxSize_
    /// Field: e044_SymbolicDimInfo.maxSize_
    pub max_size: i32,
    /// Field: e005_SymbolicDimInfo.granularity_
    /// Field: e044_SymbolicDimInfo.granularity_
    pub granularity: i32,
}

/// The authority's member initialisers (`dsc/dims.h:149-150`).
impl Default for SymbolicDimInfo {
    fn default() -> Self {
        Self {
            max_size: -1,
            granularity: -1,
        }
    }
}

/// A dim's size — the authority's `double`, where a negative is the unfilled encoding and never a
/// value (`dsc/dims.h:162-193`).
///
/// ⛔ CONSTRUCTION REFUSES A NEGATIVE, so IBM's `>= 0` test (`dsc/dims.cpp:85`) is `is_some` and its
/// `-1` is `None`. NOT `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:111`, which skips on the `int`
/// `primaryDimToVal_st` returns being `== -1` — that one is a value, and it is not this absence.
/// ⛔ AND REFUSES THE NON-FINITE, which IBM's `double` admits and nothing in it guards: a compiled
/// authority stores a NaN dim, calls `empty()` false on it and writes it back out as `nan`.
/// ⭐ `f64` BECAUSE `zi_`/`zj_` ARE HALF-INTEGERS: `x.5` pads `floor(x.5)` at top/left and
/// `ceil(x.5)` at bottom/right (`dsc/dims.h:187-193`).
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct DimSize(f64);

impl DimSize {
    /// A finite size, or absent for the negative IBM stores while a dim is unfilled.
    pub fn new(size: f64) -> Option<Self> {
        (size.is_finite() && size >= 0.0).then_some(Self(size))
    }

    /// The stored `double`.
    pub const fn get(self) -> f64 {
        self.0
    }
}

/// One dim's value as every `primaryDimToVal_st` reports it — an `int`, already split, scaled and
/// padded (`dsc/dims.h:268-273`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DimVal(pub i32);

/// The fraction of a dim one view covers — IBM's `double dimDensity` parameter, which every dim
/// value is multiplied by (`dsc/dims.h:272`, `dsc/dims.cpp:559`).
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct DimDensity(f64);

impl DimDensity {
    /// The whole dim — the default of every `double dimDensity = 1.0` parameter (`dsc/dims.h:272`).
    pub const FULL: Self = Self(1.0);

    /// A density, or absent unless `0.0 < density <= 1.0`.
    ///
    /// ⛔ THIS IS IBM'S `DT_CHECK(dimDensity > 0.0 && dimDensity <= 1.0)` (`dsc/dims.cpp:558`,
    /// `:639`, `:656`) MOVED TO CONSTRUCTION, so the three call sites cannot be reached with a
    /// density that would have aborted and none of them re-checks it.
    pub fn new(density: f64) -> Option<Self> {
        (density > 0.0 && density <= 1.0).then_some(Self(density))
    }

    /// The stored fraction.
    pub const fn get(self) -> f64 {
        self.0
    }
}

/// The largest volume a set of symbolic dims may reach together (`dsc/dims.h:202`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SymbolicVolume(pub i32);

/// Replaces: e012_DataStructDims
///
/// `dsc/dims.h:158-303`. Every dimension of one data structure or op: the primary dims, the derived
/// halves they compound from, which of them are symbolic and at what granularity, how each is split
/// across corelets, PT rows and the PE/SFP pair, and each one's padding scalars.
///
/// ⛔ `name_` IS NOT PART OF EQUALITY. IBM compares `tie()`, which omits it (`dsc/dims.h:221-228`,
/// `:283`), and `empty()` is that comparison against a default (`dsc/dims.cpp:112-115`).
#[derive(Clone, Debug, Default)]
pub struct DataStructDims {
    /// Field: e012_DataStructDims.name_
    pub name: String,
    /// Field: e012_DataStructDims.in_
    ///
    /// Input features (or channels).
    pub r#in: Option<DimSize>,
    /// Field: e012_DataStructDims.out_
    ///
    /// Output features (or channels).
    pub out: Option<DimSize>,
    /// Field: e012_DataStructDims.mb_
    ///
    /// Minibatch size.
    pub mb: Option<DimSize>,
    /// Field: e012_DataStructDims.ij_
    ///
    /// Output image dimensions (rows/cols) — `compound` writes it from `i_` and `j_`.
    pub ij: Option<DimSize>,
    /// Field: e012_DataStructDims.rc_
    ///
    /// Input image (rows/cols) with zero padding — `compound` writes it from `r_` and `c_`.
    pub rc: Option<DimSize>,
    /// Field: e012_DataStructDims.kij_
    ///
    /// Kernel dimensions (rows/cols) — `compound` writes it from `ki_` and `kj_`.
    pub kij: Option<DimSize>,
    /// Field: e012_DataStructDims.y_
    ///
    /// A kernel reuse dimension (e.g. timestep).
    pub y: Option<DimSize>,
    /// Field: e012_DataStructDims.x_
    ///
    /// A repeat dim that does not add reuse (e.g. attention heads).
    pub x: Option<DimSize>,
    /// Field: e012_DataStructDims.x1_
    ///
    /// A second repeat dim that does not add reuse.
    pub x1: Option<DimSize>,
    /// Field: e012_DataStructDims.sij_
    ///
    /// Stride dimensions (rows/cols). To be removed in future (`dsc/dims.h:174-176`).
    pub sij: Option<DimSize>,
    /// Field: e012_DataStructDims.zij_
    ///
    /// Zero pad dimensions (rows/cols). To be removed in future (`dsc/dims.h:174-176`).
    pub zij: Option<DimSize>,
    /// Field: e012_DataStructDims.i_
    ///
    /// Output image rows.
    pub i: Option<DimSize>,
    /// Field: e012_DataStructDims.j_
    ///
    /// Output image cols.
    pub j: Option<DimSize>,
    /// Field: e012_DataStructDims.r_
    ///
    /// Input image rows with zero padding.
    pub r: Option<DimSize>,
    /// Field: e012_DataStructDims.c_
    ///
    /// Input image cols with zero padding.
    pub c: Option<DimSize>,
    /// Field: e012_DataStructDims.ki_
    ///
    /// Kernel rows.
    pub ki: Option<DimSize>,
    /// Field: e012_DataStructDims.kj_
    ///
    /// Kernel cols.
    pub kj: Option<DimSize>,
    /// Field: e012_DataStructDims.si_
    ///
    /// Stride along rows.
    pub si: Option<DimSize>,
    /// Field: e012_DataStructDims.sj_
    ///
    /// Stride along cols.
    pub sj: Option<DimSize>,
    /// Field: e012_DataStructDims.zi_
    ///
    /// Zero pad rows, at each side: top and bottom (`dsc/dims.h:187-193`).
    pub zi: Option<DimSize>,
    /// Field: e012_DataStructDims.zj_
    ///
    /// Zero pad cols, at each side: left and right (`dsc/dims.h:187-193`).
    pub zj: Option<DimSize>,
    /// Field: e012_DataStructDims.symbolicDimInfo_
    ///
    /// The max and granularity of each symbolic dim. Presence here IS symbolic-ness, and when a dim
    /// is symbolic its main size above is set to the max (`dsc/dims.h:195-197`).
    pub symbolic_dim_info: BTreeMap<PrimaryDimTypes, SymbolicDimInfo>,
    /// Field: e012_DataStructDims.maxSymbolicVolume_
    ///
    /// A joint limit over several symbolic dims, usually below the product of their maxes.
    pub max_symbolic_volume: BTreeMap<BTreeSet<PrimaryDimTypes>, SymbolicVolume>,
    /// Field: e012_DataStructDims.coreletSplit_
    ///
    /// For each split dim, the amount of work per corelet, indexed by corelet id
    /// (`dsc/dims.cpp:633`).
    pub corelet_split: BTreeMap<PrimaryDimTypes, Vec<DimVal>>,
    /// Field: e012_DataStructDims.rowSplit_
    ///
    /// For each split dim and corelet, the amount of work per PT row, indexed by row id
    /// (`dsc/dims.cpp:669`).
    pub row_split: BTreeMap<PrimaryDimTypes, BTreeMap<CoreletId, Vec<DimVal>>>,
    /// Field: e012_DataStructDims.peSfpSplit_
    ///
    /// For each split dim and corelet, the amount of work for PE and for SFP.
    ///
    /// ⭐ IBM's inner map is an `unordered_map` (`dsc/dims.h:212-214`); ordered here, which is
    /// observable only as the key order of this one object inside `exportJson`.
    pub pe_sfp_split:
        BTreeMap<PrimaryDimTypes, BTreeMap<CoreletId, BTreeMap<SenComponent, DimVal>>>,
    /// Field: e012_DataStructDims.paddingSizes_
    ///
    /// For each primary dim that has a padded version, everything contributing to that padded size.
    pub padding_sizes: BTreeMap<PrimaryDimTypes, DimPaddingSizes>,
}

/// IBM's `operator==` over `tie()` (`dsc/dims.h:221-228`, `:283`): every dim and every map, and
/// deliberately NOT `name_`.
///
/// ⛔ NOT DERIVED — deriving would compare the name and break `empty()`. Not `Eq`/`Ord`/`Hash`
/// either, because the dims are `f64`.
impl PartialEq for DataStructDims {
    fn eq(&self, other: &Self) -> bool {
        self.r#in == other.r#in
            && self.out == other.out
            && self.mb == other.mb
            && self.ij == other.ij
            && self.rc == other.rc
            && self.kij == other.kij
            && self.y == other.y
            && self.x == other.x
            && self.x1 == other.x1
            && self.sij == other.sij
            && self.zij == other.zij
            && self.i == other.i
            && self.j == other.j
            && self.r == other.r
            && self.c == other.c
            && self.ki == other.ki
            && self.kj == other.kj
            && self.si == other.si
            && self.sj == other.sj
            && self.zi == other.zi
            && self.zj == other.zj
            && self.symbolic_dim_info == other.symbolic_dim_info
            && self.max_symbolic_volume == other.max_symbolic_volume
            && self.corelet_split == other.corelet_split
            && self.row_split == other.row_split
            && self.pe_sfp_split == other.pe_sfp_split
            && self.padding_sizes == other.padding_sizes
    }
}

/// One dim as `std::ostream <<` writes it, `-1` when unfilled (`dsc/dims.cpp:117-124`).
fn dim_text(dim: Option<DimSize>) -> String {
    ostream_double(dim.map_or(-1.0, DimSize::get))
}

/// A `double` at `std::ostream`'s default six significant digits and `defaultfloat` form, i.e.
/// `printf("%g")`: 64 prints `64`, 0.5 prints `0.5` and 16777216 prints `1.67772e+07`.
///
/// ⛔ RUST'S `{}` IS NOT THAT — it would print `16777216`, and this text is the DGP serialization
/// (`dsc/dims.h:245-248`) as well as every print routine's.
///
/// ⛔ THE FORM IS PICKED BY THE **ROUNDED** VALUE'S EXPONENT, NOT THE VALUE'S (C99 7.19.6.1): six
/// significant digits carry 999999.6 to `1e+06` and 0.00009999999 to `0.0001` before the choice.
fn ostream_double(value: f64) -> String {
    const SIGNIFICANT: i32 = 6;
    if value.is_nan() {
        return String::from("nan");
    }
    if value.is_infinite() {
        return String::from(if value > 0.0 { "inf" } else { "-inf" });
    }
    let rounded = format!("{value:.*e}", (SIGNIFICANT - 1) as usize);
    let (mantissa, exponent) = rounded.split_once('e').unwrap_or((rounded.as_str(), "0"));
    let exponent: i32 = exponent.parse().unwrap_or(0);
    if (-4..SIGNIFICANT).contains(&exponent) {
        let precision = (SIGNIFICANT - 1 - exponent) as usize;
        trim_trailing_zeros(&format!("{value:.precision$}"))
    } else {
        let sign = if exponent < 0 { '-' } else { '+' };
        format!("{}e{sign}{:02}", trim_trailing_zeros(mantissa), exponent.abs())
    }
}

/// `%g` drops a fractional part's trailing zeros, and then a bare trailing point.
fn trim_trailing_zeros(text: &str) -> String {
    if text.contains('.') {
        String::from(text.trim_end_matches('0').trim_end_matches('.'))
    } else {
        String::from(text)
    }
}

/// `name=value` pairs joined by single spaces, the form every print routine streams
/// (`dsc/dims.cpp:152-183`).
fn push_dims(out: &mut String, dims: &[(&str, Option<DimSize>)]) {
    for (index, (name, dim)) in dims.iter().enumerate() {
        if index > 0 {
            out.push(' ');
        }
        out.push_str(name);
        out.push('=');
        out.push_str(&dim_text(*dim));
    }
}

impl DataStructDims {
    /// Reset every field, `name_` included — IBM assigns a whole default (`dsc/dims.cpp:82`).
    pub fn clear(&mut self) {
        *self = Self::default();
    }

    /// Compute the compound primary dims from the derived ones (`dsc/dims.cpp:84-110`). An unfilled
    /// half leaves the compound unfilled, which is the `-1` IBM writes in that case.
    pub fn compound(&mut self) {
        self.ij = Self::product(self.i, self.j);
        self.kij = Self::product(self.ki, self.kj);
        self.zij = Self::product(self.zi, self.zj);
        self.sij = Self::product(self.si, self.sj);
        self.rc = Self::product(self.r, self.c);
    }

    /// The two halves multiplied, unfilled unless both are filled (`dsc/dims.cpp:85-89`).
    fn product(row: Option<DimSize>, col: Option<DimSize>) -> Option<DimSize> {
        DimSize::new(row?.get() * col?.get())
    }

    /// Whether the information has not been filled in — IBM's equality against a default, which
    /// ignores `name_` (`dsc/dims.cpp:112-115`).
    pub fn empty(&self) -> bool {
        *self == Self::default()
    }

    /// The DGP serialization line: sixteen dims and a newline, or nothing at all when empty
    /// (`dsc/dims.cpp:117-124`).
    ///
    /// ⛔ NEITHER THE STRUCT'S FIELD ORDER NOR `operator<<`'s, and it omits the compound dims,
    /// which `read` recomputes. Do not modify without DGP approval (`dsc/dims.h:245-248`).
    pub fn write(&self, out: &mut String) {
        if self.empty() {
            return;
        }
        let dims = [
            self.r#in, self.out, self.mb, self.i, self.j, self.r, self.c, self.ki, self.kj, self.x,
            self.x1, self.y, self.si, self.sj, self.zi, self.zj,
        ];
        for (index, dim) in dims.into_iter().enumerate() {
            if index > 0 {
                out.push(' ');
            }
            out.push_str(&dim_text(dim));
        }
        out.push('\n');
    }

    /// The counterpart of `write` (`dsc/dims.cpp:131-150`), consuming sixteen whitespace-separated
    /// tokens from a shared cursor. The two-argument overload (`:126-130`) is the single-token
    /// `std::stod` inlined here.
    ///
    /// ⛔ ONLY `-1` OR A FINITE NON-NEGATIVE SIZE IS TAKEN. A compiled authority stores `-5` and
    /// `nan` as themselves and writes them back, takes `5abc` as 5 and `0x10` as 16, and terminates
    /// on `1e400`; refusing all five is the alternative to rewriting the token as an unfilled dim.
    /// ⛔ NOTHING IS WRITTEN UNLESS EVERY TOKEN PARSES, where IBM clears first and half-reads.
    #[must_use]
    pub fn read(&mut self, tokens: &mut std::str::SplitWhitespace<'_>) -> Option<()> {
        let mut dims = [None; 16];
        for slot in &mut dims {
            let value: f64 = tokens.next()?.parse().ok()?;
            *slot = if value == -1.0 {
                None
            } else {
                Some(DimSize::new(value)?)
            };
        }
        self.clear();
        [
            self.r#in, self.out, self.mb, self.i, self.j, self.r, self.c, self.ki, self.kj, self.x,
            self.x1, self.y, self.si, self.sj, self.zi, self.zj,
        ] = dims;
        self.compound();
        Some(())
    }

    /// IBM's `operator<<`: every dim, compound ones included, or `Empty` (`dsc/dims.cpp:152-164`).
    pub fn print(&self, out: &mut String) {
        if self.empty() {
            out.push_str("Empty");
            return;
        }
        push_dims(
            out,
            &[
                ("in", self.r#in),
                ("out", self.out),
                ("mb", self.mb),
                ("ij", self.ij),
                ("kij", self.kij),
                ("x", self.x),
                ("x1", self.x1),
                ("y", self.y),
                ("rc", self.rc),
                ("i", self.i),
                ("j", self.j),
                ("r", self.r),
                ("c", self.c),
                ("ki", self.ki),
                ("kj", self.kj),
                ("si", self.si),
                ("sj", self.sj),
                ("zi", self.zi),
                ("zj", self.zj),
            ],
        );
    }

    /// The same without the stride and zero-pad halves (`dsc/dims.cpp:166-175`).
    pub fn print_med(&self, out: &mut String) {
        if self.empty() {
            out.push_str("Empty");
            return;
        }
        push_dims(
            out,
            &[
                ("in", self.r#in),
                ("out", self.out),
                ("ij", self.ij),
                ("mb", self.mb),
                ("kij", self.kij),
                ("x", self.x),
                ("x1", self.x1),
                ("y", self.y),
                ("rc", self.rc),
                ("i", self.i),
                ("j", self.j),
                ("r", self.r),
                ("c", self.c),
            ],
        );
    }

    /// The primary dims only (`dsc/dims.cpp:177-184`).
    pub fn print_short(&self, out: &mut String) {
        if self.empty() {
            out.push_str("Empty");
            return;
        }
        push_dims(
            out,
            &[
                ("in", self.r#in),
                ("out", self.out),
                ("ij", self.ij),
                ("mb", self.mb),
                ("kij", self.kij),
                ("x", self.x),
                ("x1", self.x1),
                ("y", self.y),
            ],
        );
    }

    /// IBM's JSON dump, spacing and key order included (`dsc/dims.cpp:186-308`).
    /// `skip_deprecated_fields` drops the derived halves and the two to-be-removed compounds.
    ///
    /// ⛔ EXCEPT THE `peSfpSplit_` INNER ORDER, WHICH NO TOTAL ORDER REPRODUCES: it is an
    /// `unordered_map` (`dsc/dims.h:212-214`), and a compiled authority emits `{"sfp", "pe"}` for
    /// one corelet and `{"pe", "sfp"}` for the next in the same object with the same two keys.
    /// ⛔ AN ABSENT `totalSize_` PRINTS `-1`, which is what IBM prints when the dim is unfilled
    /// (`dsc/dims.cpp:567-568`); where IBM instead `DT_ERROR`s there is no JSON to compare with.
    pub fn export_json(&self, skip_deprecated_fields: bool) -> String {
        let mut json = format!("{{\"name_\" : \"{}\", ", self.name);
        for (key, dim) in [
            ("in_", self.r#in),
            ("out_", self.out),
            ("mb_", self.mb),
            ("i_", self.i),
            ("j_", self.j),
            ("ki_", self.ki),
            ("kj_", self.kj),
            ("x_", self.x),
            ("x1_", self.x1),
            ("y_", self.y),
        ] {
            json.push_str(&format!("\"{key}\" : {}, ", dim_text(dim)));
        }
        if !skip_deprecated_fields {
            for (key, dim) in [
                ("r_", self.r),
                ("c_", self.c),
                ("ij_", self.ij),
                ("rc_", self.rc),
                ("kij_", self.kij),
                ("sij_", self.sij),
                ("zij_", self.zij),
                ("si_", self.si),
                ("sj_", self.sj),
                ("zi_", self.zi),
                ("zj_", self.zj),
            ] {
                json.push_str(&format!("\"{key}\" : {}, ", dim_text(dim)));
            }
        }
        json.push_str("\"symbolicDimInfo_\" : {");
        for (index, (dim, info)) in self.symbolic_dim_info.iter().enumerate() {
            if index > 0 {
                json.push_str(", ");
            }
            json.push_str(&format!(
                "\"{}\" : {{\"maxSize_\" : {}, \"granularity_\" : {}}}",
                dim.name(),
                info.max_size,
                info.granularity
            ));
        }
        json.push_str("}, \"maxSymbolicVolume_\" : {");
        for (index, (dims, volume)) in self.max_symbolic_volume.iter().enumerate() {
            if index > 0 {
                json.push_str(", ");
            }
            let keys: Vec<String> = dims.iter().map(|dim| (*dim as i32).to_string()).collect();
            json.push_str(&format!("[\"{}\"] : {}", keys.join(", "), volume.0));
        }
        json.push_str("}, \"coreletSplit_\" : {");
        for (index, (dim, split)) in self.corelet_split.iter().enumerate() {
            if index > 0 {
                json.push_str(", ");
            }
            json.push_str(&format!("\"{}\" : [{}]", dim.name(), join_vals(split)));
        }
        json.push_str("}, \"rowSplit_\" : {");
        for (index, (dim, split)) in self.row_split.iter().enumerate() {
            if index > 0 {
                json.push_str(", ");
            }
            json.push_str(&format!("\"{}\" : {{", dim.name()));
            for (inner, (corelet, rows)) in split.iter().enumerate() {
                if inner > 0 {
                    json.push_str(", ");
                }
                json.push_str(&format!("\"{}\" : [{}]", corelet.0, join_vals(rows)));
            }
            json.push('}');
        }
        json.push_str("}, \"peSfpSplit_\" : {");
        for (index, (dim, per_corelet)) in self.pe_sfp_split.iter().enumerate() {
            if index > 0 {
                json.push_str(", ");
            }
            json.push_str(&format!("\"{}\" : {{", dim.name()));
            for (inner, (corelet, split)) in per_corelet.iter().enumerate() {
                if inner > 0 {
                    json.push_str(", ");
                }
                json.push_str(&format!(" \"{}\" : {{", corelet.0));
                for (entry, (comp, val)) in split.iter().enumerate() {
                    if entry > 0 {
                        json.push_str(", ");
                    }
                    json.push_str(&format!("\"{}\" : {}", comp.spelling(), val.0));
                }
                json.push('}');
            }
            json.push('}');
        }
        json.push_str("}, \"paddingSizes_\" : {");
        for (index, (dim, pad)) in self.padding_sizes.iter().enumerate() {
            if index > 0 {
                json.push_str(", ");
            }
            let total = if pad.window_dim == PrimaryDimTypes::Undefined && pad.pad_front < 0 {
                String::from("\"N/A\"")
            } else {
                let padded = PaddingFormType::new(*dim, PadType::PaddedFullSpanWUnneeded);
                self.primary_dim_to_val_base(*dim, &padded, DimDensity::FULL, false)
                    .map_or(-1, |val| val.0)
                    .to_string()
            };
            json.push_str(&format!(
                "\"{}\" : {{\"padFront_\" : {}, \"padBack_\" : {}, \"unneededPad_\" : {}, \
                 \"unneededPadFront_\" : {}, \"unneededPadBack_\" : {}, \"totalSize_\" : {total}, \
                 \"stride_\" : {}, \"dilation_\" : {}, \"windowDim_\" : \"{}\"}}",
                dim.name(),
                pad.pad_front,
                pad.pad_back,
                pad.unneeded_pad,
                pad.unneeded_pad_front,
                pad.unneeded_pad_back,
                pad.stride,
                pad.dilation,
                pad.window_dim.name()
            ));
        }
        json.push_str("}}");
        json
    }

    /// The dim one of IBM's twenty-one parameter names selects, as a handle to assign through
    /// (`dsc/dims.cpp:437-482`). An unknown name is absent, where IBM `DT_ERROR`s.
    pub fn param_name_to_val_mut(&mut self, name: &str) -> Option<&mut Option<DimSize>> {
        Some(match name {
            "in" => &mut self.r#in,
            "out" => &mut self.out,
            "mb" => &mut self.mb,
            "i" => &mut self.i,
            "j" => &mut self.j,
            "ij" => &mut self.ij,
            "ki" => &mut self.ki,
            "kj" => &mut self.kj,
            "kij" => &mut self.kij,
            "x" => &mut self.x,
            "x1" => &mut self.x1,
            "y" => &mut self.y,
            "r" => &mut self.r,
            "c" => &mut self.c,
            "rc" => &mut self.rc,
            "si" => &mut self.si,
            "sj" => &mut self.sj,
            "sij" => &mut self.sij,
            "zi" => &mut self.zi,
            "zj" => &mut self.zj,
            "zij" => &mut self.zij,
            _ => return None,
        })
    }

    /// The field one primary dim names, as a handle to assign through (`dsc/dims.cpp:485-514`).
    /// Only `Undefined` has no field, and that is IBM's `DT_ERROR` arm.
    pub fn primary_dim_to_val_handler_mut(
        &mut self,
        d: PrimaryDimTypes,
    ) -> Option<&mut Option<DimSize>> {
        Some(match d {
            PrimaryDimTypes::In => &mut self.r#in,
            PrimaryDimTypes::Out => &mut self.out,
            PrimaryDimTypes::Mb => &mut self.mb,
            PrimaryDimTypes::I => &mut self.i,
            PrimaryDimTypes::J => &mut self.j,
            PrimaryDimTypes::Ij => &mut self.ij,
            PrimaryDimTypes::Ki => &mut self.ki,
            PrimaryDimTypes::Kj => &mut self.kj,
            PrimaryDimTypes::Kij => &mut self.kij,
            PrimaryDimTypes::X => &mut self.x,
            PrimaryDimTypes::X1 => &mut self.x1,
            PrimaryDimTypes::Y => &mut self.y,
            PrimaryDimTypes::Undefined => return None,
        })
    }

    /// The same field read rather than written — the second copy of that dispatch, which
    /// `primaryDimToVal_base_st` writes out again (`dsc/dims.cpp:526-551`).
    ///
    /// ⛔ AN UNFILLED DIM AND `Undefined` ARE BOTH ABSENT HERE AND IBM DISTINGUISHES THEM: its `-1`
    /// is a value `primaryDimToVal_st` reports and its `DT_ERROR` is not. `own_dim_val` is the seam
    /// that splits the two apart again.
    pub fn primary_dim_to_val_handler(&self, d: PrimaryDimTypes) -> Option<DimSize> {
        match d {
            PrimaryDimTypes::In => self.r#in,
            PrimaryDimTypes::Out => self.out,
            PrimaryDimTypes::Mb => self.mb,
            PrimaryDimTypes::I => self.i,
            PrimaryDimTypes::J => self.j,
            PrimaryDimTypes::Ij => self.ij,
            PrimaryDimTypes::Ki => self.ki,
            PrimaryDimTypes::Kj => self.kj,
            PrimaryDimTypes::Kij => self.kij,
            PrimaryDimTypes::X => self.x,
            PrimaryDimTypes::X1 => self.x1,
            PrimaryDimTypes::Y => self.y,
            PrimaryDimTypes::Undefined => None,
        }
    }

    /// The whole-object view of one dim: its symbolic max (or granularity), else the dim itself,
    /// scaled by the density and padded (`dsc/dims.cpp:516-561`).
    fn primary_dim_to_val_base(
        &self,
        d: PrimaryDimTypes,
        padded: &PaddingFormType,
        dim_density: DimDensity,
        get_symbolic_granularity: bool,
    ) -> Option<DimVal> {
        let val = match self.symbolic_dim_info.get(&d) {
            Some(info) => {
                if get_symbolic_granularity {
                    info.granularity
                } else {
                    info.max_size
                }
            }
            None => self.own_dim_val(d)?.0,
        };
        let val = DimVal((f64::from(val) * dim_density.get()) as i32);
        self.calculate_padded(d, val, padded, get_symbolic_granularity)
    }

    /// One dim's own field as IBM's if-chain reads it (`dsc/dims.cpp:530-556`): the stored size
    /// truncated to an `int`, `-1` while the dim is unfilled, and absent only for the sentinel IBM
    /// `DT_ERROR`s on.
    fn own_dim_val(&self, d: PrimaryDimTypes) -> Option<DimVal> {
        match self.primary_dim_to_val_handler(d) {
            Some(size) => Some(DimVal(size.get() as i32)),
            None if matches!(d, PrimaryDimTypes::Undefined) => None,
            None => Some(DimVal(-1)),
        }
    }

    /// One dim's padded size (`dsc/dims.cpp:563-616`): unfilled stays unfilled, an unpadded dim is
    /// itself, and a padded one spans its window, stride and pads.
    ///
    /// ⛔ A NEGATIVE VALUE RETURNS `-1` AND IS NOT AN ABSENCE (`dsc/dims.cpp:567-568`). It
    /// short-circuits every branch below — compound dims and a missing `paddingSizes_` entry
    /// included — and it is the answer every caller asking after an unfilled dim gets.
    /// ⛔ ABSENT IS IBM'S FIVE `DT_ERROR`s: a compound dim, a padded dim with no `paddingSizes_`
    /// entry, a negative pad, a pad type the branch does not support, and a window dim below 1.
    pub fn calculate_padded(
        &self,
        d: PrimaryDimTypes,
        val: DimVal,
        padding: &PaddingFormType,
        get_symbolic_granularity: bool,
    ) -> Option<DimVal> {
        let pad_type = padding.padding(d);
        if val.0 < 0 {
            return Some(DimVal(-1));
        }
        if pad_type == PadType::NoPad {
            return Some(val);
        }
        if d.is_compound() {
            return None;
        }
        let pad = self.padding_sizes.get(&d)?;
        if pad.window_dim == PrimaryDimTypes::Undefined {
            if pad.pad_front < 0 || pad.pad_back < 0 {
                return None;
            }
            return match pad_type {
                PadType::PaddedFullSpanWUnneeded => Some(DimVal(
                    val.0 + pad.pad_front + pad.pad_back + pad.unneeded_pad,
                )),
                PadType::PaddedFullSpan => Some(DimVal(val.0 + pad.pad_front + pad.pad_back)),
                PadType::NoPad
                | PadType::LoweredPadded
                | PadType::PaddedNoZeroPad
                | PadType::PaddedWZeroPad => None,
            };
        }
        let window = PaddingFormType::default();
        let w_size = self.primary_dim_to_val_base(
            pad.window_dim,
            &window,
            DimDensity::FULL,
            get_symbolic_granularity,
        )?;
        if w_size.0 < 1 {
            return None;
        }
        let spanned = w_size.0 + (val.0 - 1) * pad.stride;
        match pad_type {
            PadType::PaddedFullSpanWUnneeded => Some(DimVal(spanned + pad.unneeded_pad)),
            PadType::PaddedWZeroPad => Some(DimVal(spanned)),
            PadType::PaddedNoZeroPad => {
                if pad.pad_front < 0 || pad.pad_back < 0 {
                    return None;
                }
                Some(DimVal(
                    spanned + pad.unneeded_pad
                        - pad.unneeded_pad_front
                        - pad.unneeded_pad_back
                        - pad.pad_front
                        - pad.pad_back,
                ))
            }
            PadType::LoweredPadded => Some(DimVal(w_size.0 * val.0)),
            PadType::NoPad | PadType::PaddedFullSpan => None,
        }
    }

    /// A split size stated as a max, rescaled to the granularity (`dsc/dims.cpp:618-629`). A dim
    /// that is not symbolic keeps its value.
    ///
    /// ⛔ ABSENT IS IBM'S TWO `DT_CHECK`s plus the division it does not guard: the max must be a
    /// whole number of granules, the ratio must be non-zero, and the value must divide by it.
    fn scale_from_max_to_granularity(&self, d: PrimaryDimTypes, val: DimVal) -> Option<DimVal> {
        let Some(info) = self.symbolic_dim_info.get(&d) else {
            return Some(val);
        };
        if info.granularity == 0 || info.max_size % info.granularity != 0 {
            return None;
        }
        let factor = info.max_size / info.granularity;
        if factor == 0 || val.0 % factor != 0 {
            return None;
        }
        Some(DimVal(val.0 / factor))
    }

    /// One corelet's view of a dim, falling back to the whole-object one when the dim is not split
    /// across corelets or no corelet is named (`dsc/dims.cpp:631-644`).
    fn primary_dim_to_val_cl_view(
        &self,
        d: PrimaryDimTypes,
        cl_id: Option<CoreletId>,
        padded: &PaddingFormType,
        dim_density: DimDensity,
        get_symbolic_granularity: bool,
    ) -> Option<DimVal> {
        if let Some(cl_id) = cl_id
            && let Some(split) = self.corelet_split.get(&d)
        {
            let mut size = *split.get(usize::from(cl_id.0))?;
            if get_symbolic_granularity {
                size = self.scale_from_max_to_granularity(d, size)?;
            }
            let size = DimVal((f64::from(size.0) * dim_density.get()) as i32);
            return self.calculate_padded(d, size, padded, get_symbolic_granularity);
        }
        self.primary_dim_to_val_base(d, padded, dim_density, get_symbolic_granularity)
    }

    /// One dim's value for the whole object: no component, no row, no corelet, unpadded, full
    /// density (`dsc/dims.cpp:647-649`).
    pub fn primary_dim_to_val(&self, d: PrimaryDimTypes) -> Option<DimVal> {
        self.primary_dim_to_val_for_component(
            d,
            SenComponent::NoComponent,
            None,
            None,
            &PaddingFormType::default(),
            DimDensity::FULL,
            false,
        )
    }

    /// One dim's value as one row, PE/SFP half and corelet see it (`dsc/dims.cpp:651-706`). A named
    /// row reads `rowSplit_`, else a PE or SFP reads `peSfpSplit_`, else the corelet view; with no
    /// corelet the split sizes are summed when the dim is also split across corelets and otherwise
    /// taken from the first corelet.
    ///
    /// ⛔ `PELRF` AND `SFPLRF` ARE MAPPED TO `PE` AND `SFP` FIRST, so a memory component selects the
    /// compute component's split (`dsc/dims.cpp:659-663`).
    #[expect(
        clippy::too_many_arguments,
        reason = "IBM's parameter list (`dsc/dims.h:269-273`), defaults included"
    )]
    pub fn primary_dim_to_val_for_component(
        &self,
        d: PrimaryDimTypes,
        pe_or_sfp: SenComponent,
        ptrow_id: Option<RowId>,
        cl_id: Option<CoreletId>,
        padded: &PaddingFormType,
        dim_density: DimDensity,
        get_symbolic_granularity: bool,
    ) -> Option<DimVal> {
        let pe_or_sfp = match pe_or_sfp {
            SenComponent::Pelrf => SenComponent::Pe,
            SenComponent::Sfplrf => SenComponent::Sfp,
            other => other,
        };
        if let Some(ptrow_id) = ptrow_id
            && let Some(split) = self.row_split.get(&d)
        {
            let row = usize::from(ptrow_id.0);
            let val = if let Some(cl_id) = cl_id {
                *split.get(&cl_id)?.get(row)?
            } else if self.corelet_split.contains_key(&d) {
                let mut sum = 0;
                for rows in split.values() {
                    sum += rows.get(row)?.0;
                }
                DimVal(sum)
            } else {
                *split.values().next()?.get(row)?
            };
            return self.finish_split_val(d, val, padded, dim_density, get_symbolic_granularity);
        }
        if matches!(pe_or_sfp, SenComponent::Pe | SenComponent::Sfp)
            && let Some(split) = self.pe_sfp_split.get(&d)
        {
            let val = if let Some(cl_id) = cl_id {
                *split.get(&cl_id)?.get(&pe_or_sfp)?
            } else if self.corelet_split.contains_key(&d) {
                let mut sum = 0;
                for per_corelet in split.values() {
                    sum += per_corelet.get(&pe_or_sfp)?.0;
                }
                DimVal(sum)
            } else {
                *split.values().next()?.get(&pe_or_sfp)?
            };
            return self.finish_split_val(d, val, padded, dim_density, get_symbolic_granularity);
        }
        self.primary_dim_to_val_cl_view(d, cl_id, padded, dim_density, get_symbolic_granularity)
    }

    /// Rescale, apply the density and pad a split size — the three lines both split branches end
    /// with (`dsc/dims.cpp:677-682`, `:697-702`).
    fn finish_split_val(
        &self,
        d: PrimaryDimTypes,
        val: DimVal,
        padded: &PaddingFormType,
        dim_density: DimDensity,
        get_symbolic_granularity: bool,
    ) -> Option<DimVal> {
        let val = if get_symbolic_granularity {
            self.scale_from_max_to_granularity(d, val)?
        } else {
            val
        };
        let val = DimVal((f64::from(val.0) * dim_density.get()) as i32);
        self.calculate_padded(d, val, padded, get_symbolic_granularity)
    }

    /// One dim as a named component sees it: the component's own PT row, when it has one
    /// (`dsc/dims.cpp:708-718`).
    pub fn data_stage_dim_to_val_comp_view(
        &self,
        d: PrimaryDimTypes,
        comp: SenComponent,
        cl_id: Option<CoreletId>,
        padded: &PaddingFormType,
        dim_density: DimDensity,
        get_symbolic_granularity: bool,
    ) -> Option<DimVal> {
        self.primary_dim_to_val_for_component(
            d,
            comp,
            comp.row_id(),
            cl_id,
            padded,
            dim_density,
            get_symbolic_granularity,
        )
    }

    /// Reduce every joint symbolic volume limit over the dims that are no longer symbolic here,
    /// dividing the limit by each such dim's granularity in `ref_dstg` and capping it at the product
    /// of the remaining maxes (`dsc/dims.cpp:729-762`, worked example at `:722-728`).
    ///
    /// ⛔ ABSENT IS IBM'S TWO `DT_CHECK`s — the reference must know the dim, and the limit must
    /// divide by its granularity — and NOTHING IS WRITTEN in that case, where IBM has already
    /// erased the entries it visited.
    #[must_use]
    pub fn prune_max_symbolic_volumes(&mut self, ref_dstg: &Self) -> Option<()> {
        let mut pruned = self.max_symbolic_volume.clone();
        for (sym_dims, volume_limit) in &self.max_symbolic_volume {
            if sym_dims
                .iter()
                .all(|dim| self.symbolic_dim_info.contains_key(dim))
            {
                continue;
            }
            let mut my_sym_dims = BTreeSet::new();
            let mut my_volume_limit = volume_limit.0;
            let mut mul_of_maxes = 1;
            for sym_dim in sym_dims {
                if let Some(info) = self.symbolic_dim_info.get(sym_dim) {
                    my_sym_dims.insert(*sym_dim);
                    mul_of_maxes *= info.max_size;
                } else {
                    let granularity = ref_dstg.symbolic_dim_info.get(sym_dim)?.granularity;
                    if granularity == 0 || my_volume_limit % granularity != 0 {
                        return None;
                    }
                    my_volume_limit /= granularity;
                }
            }
            my_volume_limit = my_volume_limit.min(mul_of_maxes);
            if !my_sym_dims.is_empty()
                && pruned
                    .get(&my_sym_dims)
                    .is_none_or(|limit| limit.0 > my_volume_limit)
            {
                pruned.insert(my_sym_dims, SymbolicVolume(my_volume_limit));
            }
            pruned.remove(sym_dims);
        }
        self.max_symbolic_volume = pruned;
        Some(())
    }

    /// Make one dim symbolic here, taking its max, granularity, value and every split from `ref_ds`
    /// (`dsc/dims.cpp:764-779`). A dim that is already symbolic is left alone.
    ///
    /// ⛔ ABSENT IS IBM'S `DT_CHECK` that the reference knows the dim, plus the three `.at()` calls
    /// that throw when this object splits a dim the reference does not. NOTHING IS WRITTEN in that
    /// case, where IBM has already inserted the symbolic entry.
    #[must_use]
    pub fn make_dim_symbolic(&mut self, ref_ds: &Self, dim: PrimaryDimTypes) -> Option<()> {
        let info = *ref_ds.symbolic_dim_info.get(&dim)?;
        if self.symbolic_dim_info.contains_key(&dim) {
            return Some(());
        }
        let value = ref_ds
            .primary_dim_to_val(dim)
            .and_then(|val| DimSize::new(f64::from(val.0)));
        let corelet = if self.corelet_split.contains_key(&dim) {
            Some(ref_ds.corelet_split.get(&dim)?.clone())
        } else {
            None
        };
        let rows = if self.row_split.contains_key(&dim) {
            Some(ref_ds.row_split.get(&dim)?.clone())
        } else {
            None
        };
        let pe_sfp = if self.pe_sfp_split.contains_key(&dim) {
            Some(ref_ds.pe_sfp_split.get(&dim)?.clone())
        } else {
            None
        };
        *self.primary_dim_to_val_handler_mut(dim)? = value;
        self.symbolic_dim_info.insert(dim, info);
        if let Some(corelet) = corelet {
            self.corelet_split.insert(dim, corelet);
        }
        if let Some(rows) = rows {
            self.row_split.insert(dim, rows);
        }
        if let Some(pe_sfp) = pe_sfp {
            self.pe_sfp_split.insert(dim, pe_sfp);
        }
        Some(())
    }

    /// Make one dim concrete again, dividing its value and every split of it by max/granularity
    /// (`dsc/dims.cpp:781-804`). A dim that is not symbolic is left alone.
    ///
    /// ⛔ IBM ERASES THE SYMBOLIC ENTRY BEFORE IT READS THE DIM, so the value it divides is the
    /// plain field and not the max — with no split, no padding and full density that is `-1` while
    /// the dim is unfilled, which a ratio of one carries straight back (`dsc/dims.cpp:787-792`).
    /// ⛔ ABSENT IS IBM'S `DT_CHECK`s: max a whole number of granules, ratio non-zero, and every
    /// value divisible by it. Nothing is written in that case.
    #[must_use]
    pub fn make_dim_not_symbolic(&mut self, dim: PrimaryDimTypes) -> Option<()> {
        let Some(info) = self.symbolic_dim_info.get(&dim).copied() else {
            return Some(());
        };
        if info.granularity == 0 || info.max_size % info.granularity != 0 {
            return None;
        }
        let factor = info.max_size / info.granularity;
        if factor == 0 {
            return None;
        }
        let divide = |val: DimVal| {
            if val.0 % factor == 0 {
                Some(DimVal(val.0 / factor))
            } else {
                None
            }
        };
        let value = divide(self.own_dim_val(dim)?)?;
        let corelet = match self.corelet_split.get(&dim) {
            Some(split) => Some(
                split
                    .iter()
                    .map(|val| divide(*val))
                    .collect::<Option<Vec<_>>>()?,
            ),
            None => None,
        };
        let rows = match self.row_split.get(&dim) {
            Some(split) => {
                let mut divided = BTreeMap::new();
                for (corelet, vals) in split {
                    let vals = vals
                        .iter()
                        .map(|val| divide(*val))
                        .collect::<Option<Vec<_>>>()?;
                    divided.insert(*corelet, vals);
                }
                Some(divided)
            }
            None => None,
        };
        let pe_sfp = match self.pe_sfp_split.get(&dim) {
            Some(split) => {
                let mut divided = BTreeMap::new();
                for (corelet, per_comp) in split {
                    let mut comps = BTreeMap::new();
                    for (comp, val) in per_comp {
                        comps.insert(*comp, divide(*val)?);
                    }
                    divided.insert(*corelet, comps);
                }
                Some(divided)
            }
            None => None,
        };
        self.symbolic_dim_info.remove(&dim);
        if let Some(slot) = self.primary_dim_to_val_handler_mut(dim) {
            *slot = DimSize::new(f64::from(value.0));
        }
        if let Some(corelet) = corelet {
            self.corelet_split.insert(dim, corelet);
        }
        if let Some(rows) = rows {
            self.row_split.insert(dim, rows);
        }
        if let Some(pe_sfp) = pe_sfp {
            self.pe_sfp_split.insert(dim, pe_sfp);
        }
        Some(())
    }
}

/// Every value of one split, comma separated, as `exportJson` streams them (`dsc/dims.cpp:228-233`).
fn join_vals(vals: &[DimVal]) -> String {
    vals.iter()
        .map(|val| val.0.to_string())
        .collect::<Vec<String>>()
        .join(", ")
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    /// Every dim including the sentinel, in authority order.
    const EVERY_DIM: [PrimaryDimTypes; 13] = [
        PrimaryDimTypes::In,
        PrimaryDimTypes::Out,
        PrimaryDimTypes::Ij,
        PrimaryDimTypes::Mb,
        PrimaryDimTypes::X,
        PrimaryDimTypes::Y,
        PrimaryDimTypes::Kij,
        PrimaryDimTypes::I,
        PrimaryDimTypes::J,
        PrimaryDimTypes::Ki,
        PrimaryDimTypes::Kj,
        PrimaryDimTypes::X1,
        PrimaryDimTypes::Undefined,
    ];

    /// Every kind including the sentinel, in authority order.
    const EVERY_KIND: [MetaDimKind; 9] = [
        MetaDimKind::Unpadded,
        MetaDimKind::Padded,
        MetaDimKind::PadFront,
        MetaDimKind::PadBack,
        MetaDimKind::PadValid,
        MetaDimKind::WindowDim,
        MetaDimKind::Stride,
        MetaDimKind::Dilation,
        MetaDimKind::Undefined,
    ];

    /// The discriminants ordered containers and IBM's exported JSON depend on, and `ALL` holding
    /// exactly the real dims and kinds.
    #[test]
    fn the_discriminants_are_the_authoritys() {
        for (i, dim) in EVERY_DIM.into_iter().enumerate() {
            assert_eq!(dim as usize, i, "{dim:?} moved");
        }
        for (i, kind) in EVERY_KIND.into_iter().enumerate() {
            assert_eq!(kind as usize, i, "{kind:?} moved");
        }
        assert_eq!(
            PrimaryDimTypes::ALL[..],
            EVERY_DIM[..PrimaryDimTypes::COUNT]
        );
        assert_eq!(MetaDimKind::ALL[..], EVERY_KIND[..MetaDimKind::COUNT]);

        // A `std::map` keyed by either enum iterates in this order, so `Ord` must agree with it.
        let mut sorted = EVERY_DIM;
        sorted.sort();
        assert_eq!(sorted, EVERY_DIM);
    }

    /// The dims the authority's own dim loops admit are TEN, not the twelve of `ALL`: `DimProp()`
    /// initialises its candidate list with "all dimensions (excluding IJ, KIJ)"
    /// (`ddc/ddl/ddl_conversion.h:307-311`), and `getClSplitDim` drops the same pair plus the
    /// sentinel (`ddc/ddcv1.cpp:1802-1803`). Spelled as the authority's literals here, not derived
    /// from `ALL`, so a set built from the wrong one of the two is a failure and not a tautology.
    #[test]
    fn the_authoritys_dim_loops_admit_ten_of_the_twelve() {
        assert_eq!(PrimaryDimTypes::COUNT, 12);
        assert_eq!(
            PrimaryDimTypes::NON_COMPOUND,
            [
                PrimaryDimTypes::In,
                PrimaryDimTypes::Out,
                PrimaryDimTypes::Mb,
                PrimaryDimTypes::X,
                PrimaryDimTypes::Y,
                PrimaryDimTypes::I,
                PrimaryDimTypes::J,
                PrimaryDimTypes::Ki,
                PrimaryDimTypes::Kj,
                PrimaryDimTypes::X1,
            ]
        );
        for dim in PrimaryDimTypes::ALL {
            assert_eq!(
                dim.is_compound(),
                !PrimaryDimTypes::NON_COMPOUND.contains(&dim),
                "{dim:?}"
            );
        }
        assert!(PrimaryDimTypes::Ij.is_compound());
        assert!(PrimaryDimTypes::Kij.is_compound());
        // `ddc/ddcv1.cpp:1803` skips the sentinel alongside the pair, but for being no dimension.
        assert!(!PrimaryDimTypes::Undefined.is_compound());
    }

    /// Every dim's spelling round-trips, and an unknown one is absent rather than a throw.
    #[test]
    fn every_dim_name_round_trips() {
        for dim in EVERY_DIM {
            assert_eq!(PrimaryDimTypes::from_name(dim.name()), Some(dim));
        }
        assert_eq!(PrimaryDimTypes::Undefined.name(), "undefined");
        assert_eq!(PrimaryDimTypes::from_name("rc"), None);
    }

    /// Every pad form's spelling round-trips, and an unknown one is absent.
    #[test]
    fn every_pad_type_name_round_trips() {
        for pad in PadType::ALL {
            assert_eq!(PadType::from_name(pad.name()), Some(pad));
        }
        assert_eq!(
            PadType::PaddedFullSpanWUnneeded.name(),
            "padded_fullspan_wunneeded"
        );
        assert_eq!(PadType::from_name("padded"), None);
    }

    /// Every kind's spelling round-trips — `WindowDim` is spelled `"window"` — and an unknown one
    /// is absent.
    #[test]
    fn every_meta_dim_kind_name_round_trips() {
        for kind in EVERY_KIND {
            assert_eq!(MetaDimKind::from_name(kind.name()), Some(kind));
        }
        assert_eq!(MetaDimKind::WindowDim.name(), "window");
        assert_eq!(MetaDimKind::from_name("window_dim"), None);
    }

    /// IBM's hash is `(dim << 4) ^ kind`, and it separates every pair it can be asked about.
    #[test]
    fn the_authoritys_hash_shifts_by_four_and_does_not_collide() {
        assert_eq!(MetaDimKind::HASH_SHIFT, 4);
        assert_eq!(
            PrimaryDimAndKind::new(PrimaryDimTypes::Kj, MetaDimKind::Stride).hash_value(),
            (10 << 4) ^ 6
        );

        let mut seen = std::collections::BTreeSet::new();
        for dim in EVERY_DIM {
            for kind in EVERY_KIND {
                assert!(
                    seen.insert(PrimaryDimAndKind::new(dim, kind).hash_value()),
                    "{dim:?}/{kind:?} collided"
                );
            }
        }
        assert_eq!(seen.len(), EVERY_DIM.len() * EVERY_KIND.len());
    }

    /// The formula per pair, not merely an injective function of the two halves, and the ordering
    /// consequence the deterministic-container decision rests on: IBM's hash is strictly increasing
    /// in `(dim_, kind_)`, so `Ord` and a `BTreeSet` visit keys in increasing IBM-hash order.
    #[test]
    fn the_authoritys_hash_is_monotone_so_ord_visits_in_hash_order() {
        let mut dim_major = Vec::new();
        for dim in EVERY_DIM {
            for kind in EVERY_KIND {
                let pair = PrimaryDimAndKind::new(dim, kind);
                // `<< 4` is IBM's own width, spelled out so this does not reuse `HASH_SHIFT`.
                assert_eq!(
                    pair.hash_value(),
                    ((dim as usize) << 4) ^ (kind as usize),
                    "{dim:?}/{kind:?}"
                );
                dim_major.push(pair);
            }
        }

        let mut sorted = dim_major.clone();
        sorted.sort();
        assert_eq!(sorted, dim_major, "Ord is not dim_ then kind_");
        assert!(
            dim_major
                .windows(2)
                .all(|w| w[0].hash_value() < w[1].hash_value()),
            "the hash is not monotone in the pair"
        );

        // Inserted backwards, so the visit order is the container's and not the input's.
        let visited: Vec<_> = dim_major
            .iter()
            .rev()
            .copied()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        assert_eq!(
            visited, dim_major,
            "a BTreeSet does not visit in hash order"
        );
    }

    /// A bare dim is that dim unpadded, and that is also the default pair.
    #[test]
    fn a_bare_dim_is_that_dim_unpadded() {
        assert_eq!(
            PrimaryDimAndKind::from(PrimaryDimTypes::Mb),
            PrimaryDimAndKind::new(PrimaryDimTypes::Mb, MetaDimKind::Unpadded)
        );
        assert_eq!(
            PrimaryDimAndKind::default(),
            PrimaryDimAndKind::new(PrimaryDimTypes::Undefined, MetaDimKind::Unpadded)
        );
    }

    /// An unset dim reads as `NoPad`, a set one replaces, and `clear` empties the form.
    #[test]
    fn a_padding_form_defaults_to_nopad_and_a_set_replaces() {
        let mut form = PaddingFormType::default();
        assert!(!form.has_padding_info());
        assert_eq!(form.padding(PrimaryDimTypes::Ij), PadType::NoPad);
        assert_eq!(form.padding_as_str(PrimaryDimTypes::Ij), "nopad");

        form.set_padding(PrimaryDimTypes::Ij, PadType::PaddedWZeroPad);
        form.set_padding(PrimaryDimTypes::Ij, PadType::PaddedFullSpan);
        assert!(form.has_padding_info());
        assert_eq!(form.padding(PrimaryDimTypes::Ij), PadType::PaddedFullSpan);
        assert_eq!(form.iter().count(), 1);

        form.clear();
        assert!(!form.has_padding_info());
        assert_eq!(form.padding(PrimaryDimTypes::Ij), PadType::NoPad);
    }

    /// The one-dim constructor carries its pair, and traversal is in dim order, not insertion
    /// order — `Y` is dim 5 and `In` is dim 0.
    #[test]
    fn a_padding_form_prints_in_dim_order() {
        let mut form = PaddingFormType::new(PrimaryDimTypes::Y, PadType::LoweredPadded);
        form.set_padding(PrimaryDimTypes::In, PadType::PaddedNoZeroPad);
        assert_eq!(
            form.iter().collect::<Vec<_>>(),
            [
                (PrimaryDimTypes::In, PadType::PaddedNoZeroPad),
                (PrimaryDimTypes::Y, PadType::LoweredPadded),
            ]
        );

        let mut out = String::from("x");
        form.print(&mut out, 2);
        assert_eq!(
            out,
            "x\n    Padding=  (in: padded_nozeropad) (y: lowered_padded)"
        );
    }

    /// Only four kinds name a stored scalar; the other five are absent. And of those five, the sole
    /// caller can only ever arrive with two: it filters `Padded`, `Unpadded` and `WindowDim` into a
    /// different branch one line before the call (`ddc/ddl/ddl_conversion.cpp:1824-1826`).
    #[test]
    fn only_the_four_direct_kinds_read_a_padding_scalar() {
        let sizes = DimPaddingSizes {
            pad_front: 3,
            pad_back: 4,
            stride: 2,
            dilation: 5,
            ..DimPaddingSizes::default()
        };
        assert_eq!(sizes.meta_dim_val(MetaDimKind::PadFront), Some(3));
        assert_eq!(sizes.meta_dim_val(MetaDimKind::PadBack), Some(4));
        assert_eq!(sizes.meta_dim_val(MetaDimKind::Stride), Some(2));
        assert_eq!(sizes.meta_dim_val(MetaDimKind::Dilation), Some(5));
        for kind in [
            MetaDimKind::Unpadded,
            MetaDimKind::Padded,
            MetaDimKind::PadValid,
            MetaDimKind::WindowDim,
            MetaDimKind::Undefined,
        ] {
            assert_eq!(sizes.meta_dim_val(kind), None, "{kind:?}");
        }

        let taken_as_a_dim = |kind| {
            matches!(
                kind,
                MetaDimKind::Padded | MetaDimKind::Unpadded | MetaDimKind::WindowDim
            )
        };
        let unhandled_at_the_caller: Vec<MetaDimKind> = EVERY_KIND
            .into_iter()
            .filter(|kind| !taken_as_a_dim(*kind) && sizes.meta_dim_val(*kind).is_none())
            .collect();
        assert_eq!(
            unhandled_at_the_caller,
            [MetaDimKind::PadValid, MetaDimKind::Undefined]
        );

        // What the caller divides `scale` by (`ddc/ddl/ddl_conversion.cpp:1828-1831`): a default's
        // pads are `0`, so the divisor can be zero and the accessor still owes IBM's answer.
        assert_eq!(
            DimPaddingSizes::default().meta_dim_val(MetaDimKind::PadFront),
            Some(0)
        );
        assert_eq!(
            DimPaddingSizes::default().meta_dim_val(MetaDimKind::PadBack),
            Some(0)
        );
    }

    /// Unpadded, unit stride and dilation, no window dim — and `operator==` (`dsc/dims.cpp:74-80`)
    /// comparing all eight fields, one perturbation at a time: swapping front for back passes even
    /// for an equality blind to `padBack_`, so each field is moved on its own here.
    #[test]
    fn dim_padding_sizes_defaults_to_unit_stride_and_compares_every_field() {
        let base = DimPaddingSizes::default();
        assert_eq!(
            (
                base.pad_front,
                base.pad_back,
                base.unneeded_pad,
                base.unneeded_pad_front,
                base.unneeded_pad_back,
                base.stride,
                base.dilation,
            ),
            (0, 0, 0, 0, 0, 1, 1)
        );
        assert_eq!(base.window_dim, PrimaryDimTypes::Undefined);
        assert_eq!(base, DimPaddingSizes::default());

        let perturbed = [
            (
                "padFront_",
                DimPaddingSizes {
                    pad_front: 7,
                    ..base
                },
            ),
            (
                "padBack_",
                DimPaddingSizes {
                    pad_back: 7,
                    ..base
                },
            ),
            (
                "unneededPad_",
                DimPaddingSizes {
                    unneeded_pad: 7,
                    ..base
                },
            ),
            (
                "unneededPadFront_",
                DimPaddingSizes {
                    unneeded_pad_front: 7,
                    ..base
                },
            ),
            (
                "unneededPadBack_",
                DimPaddingSizes {
                    unneeded_pad_back: 7,
                    ..base
                },
            ),
            ("stride_", DimPaddingSizes { stride: 7, ..base }),
            (
                "dilation_",
                DimPaddingSizes {
                    dilation: 7,
                    ..base
                },
            ),
            (
                "windowDim_",
                DimPaddingSizes {
                    window_dim: PrimaryDimTypes::Ki,
                    ..base
                },
            ),
        ];
        for (field, other) in perturbed {
            assert_ne!(base, other, "{field} is not compared");
        }
    }

    /// Both symbolic values start unfilled, and `operator==` (`dsc/dims.h:152-154`) compares both —
    /// one perturbation at a time, because swapping the two at once passes even for an equality
    /// that reads only `maxSize_`.
    #[test]
    fn symbolic_dim_info_starts_unfilled_and_compares_both_fields() {
        let unset = SymbolicDimInfo::default();
        assert_eq!((unset.max_size, unset.granularity), (-1, -1));

        let filled = SymbolicDimInfo {
            max_size: 64,
            granularity: 16,
        };
        assert_eq!(
            filled,
            SymbolicDimInfo {
                max_size: 64,
                granularity: 16
            }
        );
        assert_ne!(
            filled,
            SymbolicDimInfo {
                granularity: 16,
                ..unset
            },
            "maxSize_ is not compared"
        );
        assert_ne!(
            filled,
            SymbolicDimInfo {
                max_size: 64,
                ..unset
            },
            "granularity_ is not compared"
        );
    }

    // The dim vocabulary is spelled `D::In` below; the tests above predate the alias.
    use PrimaryDimTypes as D;

    /// A filled dim size, for a value the tests state is not negative.
    fn size(value: f64) -> Option<DimSize> {
        DimSize::new(value)
    }

    /// A 7x7 output with a 3x3 kernel over two heads, compounded — the object the print and export
    /// tests share.
    fn filled() -> DataStructDims {
        let mut dims = DataStructDims::default();
        dims.name = String::from("ds");
        dims.r#in = size(64.0);
        dims.out = size(32.0);
        dims.mb = size(1.0);
        dims.i = size(7.0);
        dims.j = size(7.0);
        dims.ki = size(3.0);
        dims.kj = size(3.0);
        dims.x = size(2.0);
        dims.compound();
        dims
    }

    /// A negative dim is the unfilled encoding, never a value, and the non-finite is not a size.
    #[test]
    fn a_dim_size_refuses_the_unfilled_encoding() {
        assert_eq!(size(64.0).map(DimSize::get), Some(64.0));
        assert_eq!(size(0.0).map(DimSize::get), Some(0.0));
        assert_eq!(size(-1.0), None);
        assert_eq!(size(f64::NAN), None);
        assert_eq!(size(f64::INFINITY), None, "an infinite dim is not a size");
        assert_eq!(DimDensity::new(1.0), Some(DimDensity::FULL));
        assert_eq!(DimDensity::new(0.0), None);
        assert_eq!(DimDensity::new(1.5), None);
    }

    /// The five compound dims are the products of their halves, and an unfilled half leaves the
    /// compound unfilled.
    #[test]
    fn compound_multiplies_the_halves_and_an_unfilled_half_leaves_it_unfilled() {
        let mut dims = DataStructDims::default();
        dims.i = size(7.0);
        dims.j = size(5.0);
        dims.ki = size(3.0);
        dims.kj = size(3.0);
        dims.zi = size(1.5);
        dims.zj = size(0.5);
        dims.si = size(2.0);
        dims.sj = size(2.0);
        dims.c = size(9.0);
        dims.compound();
        assert_eq!(
            (dims.ij, dims.kij, dims.zij, dims.sij, dims.rc),
            (size(35.0), size(9.0), size(0.75), size(4.0), None)
        );
    }

    /// `empty` is equality against a default, `clear` restores one, and neither sees `name_`.
    #[test]
    fn empty_and_equality_ignore_the_name() {
        let mut dims = DataStructDims::default();
        assert!(dims.empty());
        dims.name = String::from("ds");
        assert!(dims.empty(), "the name is not part of tie()");
        let mut other = DataStructDims::default();
        other.name = String::from("other");
        assert_eq!(dims, other);
        dims.mb = size(1.0);
        assert!(!dims.empty());
        assert_ne!(dims, other);
        dims.clear();
        assert_eq!(dims.name, "");
        assert!(dims.empty());
    }

    /// The DGP line is sixteen dims at `std::ostream`'s six significant digits, an empty object
    /// writes nothing, and `read` takes them back and recompounds.
    ///
    /// ⛔ THE EIGHT REFUSED TOKENS ARE MEASURED AGAINST A COMPILED AUTHORITY, and it agrees on only
    /// three: `abc`, `-` and `1e400` terminate it, while `-5`, `nan` and `inf` are stored and
    /// written straight back and `std::stod` reads `5abc` as 5 and `0x10` as 16. The first three of
    /// those five are what this port used to take and silently rewrite as an unfilled dim.
    #[test]
    fn write_states_sixteen_dims_and_read_takes_them_back() {
        let mut text = String::new();
        DataStructDims::default().write(&mut text);
        assert!(text.is_empty(), "an empty object writes no line");

        let mut big = DataStructDims::default();
        big.r#in = size(16_777_216.0);
        big.zi = size(1.5);
        big.write(&mut text);
        assert_eq!(
            text,
            format!("1.67772e+07 {} 1.5 -1\n", ["-1"; 13].join(" "))
        );

        let source = filled();
        let mut line = String::new();
        source.write(&mut line);
        let mut read_back = DataStructDims::default();
        assert_eq!(read_back.read(&mut line.split_whitespace()), Some(()));
        assert_eq!(read_back, source, "the compound dims are recomputed");
        const REST: &str = " 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1";
        for token in ["abc", "-", "1e400", "-5", "nan", "inf", "5abc", "0x10"] {
            let line = format!("{token}{REST}");
            let mut refused = DataStructDims::default();
            assert_eq!(refused.read(&mut line.split_whitespace()), None, "{token}");
            assert!(refused.empty(), "{token} left nothing behind");
        }
        assert_eq!(
            DataStructDims::default().read(&mut "1 2 3".split_whitespace()),
            None,
            "a short line is sixteen tokens short of one"
        );
    }

    /// The three print widths, and `Empty` when nothing is filled in.
    #[test]
    fn the_three_print_widths_are_the_authoritys() {
        let dims = filled();
        let mut long = String::new();
        let mut med = String::new();
        let mut short = String::new();
        dims.print(&mut long);
        dims.print_med(&mut med);
        dims.print_short(&mut short);
        assert_eq!(
            long,
            "in=64 out=32 mb=1 ij=49 kij=9 x=2 x1=-1 y=-1 rc=-1 i=7 j=7 r=-1 c=-1 ki=3 kj=3 \
             si=-1 sj=-1 zi=-1 zj=-1"
        );
        assert_eq!(
            med,
            "in=64 out=32 ij=49 mb=1 kij=9 x=2 x1=-1 y=-1 rc=-1 i=7 j=7 r=-1 c=-1"
        );
        assert_eq!(short, "in=64 out=32 ij=49 mb=1 kij=9 x=2 x1=-1 y=-1");
        let mut empty = String::new();
        DataStructDims::default().print_short(&mut empty);
        assert_eq!(empty, "Empty");
    }

    /// `std::ostream <<`'s own text for a `double`, both `%g` form boundaries included. Every pair
    /// below was measured against `std::ostringstream() << d`.
    #[test]
    fn a_double_prints_as_the_streams_own_text() {
        for (value, text) in [
            (64.0, "64"),
            (0.5, "0.5"),
            (-1.0, "-1"),
            (0.0, "0"),
            (123456.0, "123456"),
            (1234567.0, "1.23457e+06"),
            (4194304.0, "4.1943e+06"),
            (16777216.0, "1.67772e+07"),
            (0.00012345678, "0.000123457"),
            (1e-5, "1e-05"),
            (1e300, "1e+300"),
            // Six significant digits first, and only then the choice of form.
            (999999.4999, "999999"),
            (999999.5, "1e+06"),
            (9.999995e-5, "0.0001"),
            (0.00009999999, "0.0001"),
            (f64::INFINITY, "inf"),
            (f64::NEG_INFINITY, "-inf"),
            (f64::NAN, "nan"),
        ] {
            assert_eq!(ostream_double(value), text, "{value}");
        }
    }

    /// IBM's JSON text, spacing and the integer set key included — the `peSfpSplit_` inner order
    /// excepted, which is an `unordered_map`'s and not a function of its keys at all.
    #[test]
    fn export_json_is_the_authoritys_text() {
        let mut dims = filled();
        dims.symbolic_dim_info.insert(
            D::Out,
            SymbolicDimInfo {
                max_size: 64,
                granularity: 16,
            },
        );
        dims.max_symbolic_volume
            .insert(BTreeSet::from([D::In, D::Out]), SymbolicVolume(2048));
        dims.corelet_split
            .insert(D::In, vec![DimVal(32), DimVal(32)]);
        dims.row_split
            .insert(D::Mb, BTreeMap::from([(CoreletId(0), vec![DimVal(1)])]));
        dims.pe_sfp_split.insert(
            D::Y,
            BTreeMap::from([(
                CoreletId(1),
                BTreeMap::from([
                    (SenComponent::Pe, DimVal(4)),
                    (SenComponent::Sfp, DimVal(2)),
                ]),
            )]),
        );
        dims.padding_sizes.insert(
            D::J,
            DimPaddingSizes {
                stride: 2,
                window_dim: D::Kj,
                ..DimPaddingSizes::default()
            },
        );
        assert_eq!(
            dims.export_json(false),
            concat!(
                r#"{"name_" : "ds", "in_" : 64, "out_" : 32, "mb_" : 1, "i_" : 7, "j_" : 7, "#,
                r#""ki_" : 3, "kj_" : 3, "x_" : 2, "x1_" : -1, "y_" : -1, "#,
                r#""r_" : -1, "c_" : -1, "ij_" : 49, "rc_" : -1, "kij_" : 9, "sij_" : -1, "#,
                r#""zij_" : -1, "si_" : -1, "sj_" : -1, "zi_" : -1, "zj_" : -1, "#,
                r#""symbolicDimInfo_" : {"out" : {"maxSize_" : 64, "granularity_" : 16}}, "#,
                r#""maxSymbolicVolume_" : {["0, 1"] : 2048}, "#,
                r#""coreletSplit_" : {"in" : [32, 32]}, "#,
                r#""rowSplit_" : {"mb" : {"0" : [1]}}, "#,
                r#""peSfpSplit_" : {"y" : { "1" : {"sfp" : 2, "pe" : 4}}}, "#,
                r#""paddingSizes_" : {"j" : {"padFront_" : 0, "padBack_" : 0, "#,
                r#""unneededPad_" : 0, "unneededPadFront_" : 0, "unneededPadBack_" : 0, "#,
                r#""totalSize_" : 15, "stride_" : 2, "dilation_" : 1, "windowDim_" : "kj"}}}"#,
            )
        );
        let skipped = dims.export_json(true);
        assert!(!skipped.contains(r#""sij_""#), "deprecated fields dropped");
        assert!(skipped.contains(r#""y_" : -1, "symbolicDimInfo_""#));
    }

    /// All twenty-one parameter names reach their own named field, and nothing else reaches one.
    ///
    /// ⭐ EVERY NAME IS WRITTEN THROUGH THE DISPATCH INTO ONE OBJECT, then the fields are read BY
    /// NAME in IBM's own chain order (`dsc/dims.cpp:437-482`): a name that reaches another name's
    /// field leaves two of them holding the wrong value.
    #[test]
    fn every_param_name_selects_its_own_dim() {
        const NAMES: [&str; 21] = [
            "in", "out", "mb", "i", "j", "ij", "ki", "kj", "kij", "x", "x1", "y", "r", "c", "rc",
            "si", "sj", "sij", "zi", "zj", "zij",
        ];
        let mut dims = DataStructDims::default();
        for (index, name) in NAMES.into_iter().enumerate() {
            *dims
                .param_name_to_val_mut(name)
                .unwrap_or_else(|| panic!("{name} has no dim")) = size(index as f64 + 1.0);
        }
        let expected: [Option<DimSize>; 21] =
            std::array::from_fn(|index| size(index as f64 + 1.0));
        assert_eq!(
            [
                dims.r#in, dims.out, dims.mb, dims.i, dims.j, dims.ij, dims.ki, dims.kj, dims.kij,
                dims.x, dims.x1, dims.y, dims.r, dims.c, dims.rc, dims.si, dims.sj, dims.sij,
                dims.zi, dims.zj, dims.zij,
            ],
            expected
        );
        assert!(
            DataStructDims::default()
                .param_name_to_val_mut("in_")
                .is_none()
        );
    }

    /// Every real dim assigns through to its own named field; only the sentinel has none.
    #[test]
    fn every_dim_but_the_sentinel_has_a_field() {
        let mut dims = DataStructDims::default();
        for dim in PrimaryDimTypes::ALL {
            *dims.primary_dim_to_val_handler_mut(dim).unwrap() = size(dim as usize as f64 + 1.0);
        }
        // The dims in discriminant order beside the field each one names (`dsc/dims.cpp:485-514`),
        // so a swapped pair of arms is two wrong fields here.
        let expected: [Option<DimSize>; 12] =
            std::array::from_fn(|index| size(index as f64 + 1.0));
        assert_eq!(
            [
                dims.r#in, dims.out, dims.ij, dims.mb, dims.x, dims.y, dims.kij, dims.i, dims.j,
                dims.ki, dims.kj, dims.x1,
            ],
            expected
        );
        // The read dispatch is IBM's second copy of that table (`dsc/dims.cpp:526-551`).
        for dim in PrimaryDimTypes::ALL {
            assert_eq!(dims.primary_dim_to_val(dim), Some(DimVal(dim as i32 + 1)));
        }
        let mut unfilled = DataStructDims::default();
        assert!(
            unfilled
                .primary_dim_to_val_handler_mut(D::Undefined)
                .is_none()
        );
        assert_eq!(unfilled.primary_dim_to_val_handler(D::Undefined), None);
        assert_eq!(
            unfilled.primary_dim_to_val(D::Mb),
            Some(DimVal(-1)),
            "`L3DlOpsScheduler.cpp:111` skips on `== -1`, so -1 is a value and not an absence"
        );
    }

    /// A symbolic dim reports its max, or its granularity when that is asked for, and a corelet's
    /// share of it is rescaled from the max to the granularity.
    #[test]
    fn a_symbolic_dim_reports_its_max_or_its_granularity() {
        let mut dims = DataStructDims::default();
        dims.out = size(64.0);
        dims.symbolic_dim_info.insert(
            D::Out,
            SymbolicDimInfo {
                max_size: 64,
                granularity: 16,
            },
        );
        dims.corelet_split
            .insert(D::Out, vec![DimVal(32), DimVal(32)]);
        let view = |cl_id, granularity| {
            dims.primary_dim_to_val_for_component(
                D::Out,
                SenComponent::NoComponent,
                None,
                cl_id,
                &PaddingFormType::default(),
                DimDensity::FULL,
                granularity,
            )
        };
        assert_eq!(view(None, false), Some(DimVal(64)));
        assert_eq!(view(None, true), Some(DimVal(16)));
        assert_eq!(view(Some(CoreletId(1)), false), Some(DimVal(32)));
        assert_eq!(view(Some(CoreletId(1)), true), Some(DimVal(8)));
        assert_eq!(
            dims.primary_dim_to_val_for_component(
                D::Out,
                SenComponent::NoComponent,
                None,
                None,
                &PaddingFormType::default(),
                DimDensity::new(0.5).unwrap(),
                false,
            ),
            Some(DimVal(32)),
            "the density scales the max"
        );
    }

    /// A corelet's share of a symbolic dim that is not a whole number of granules has no
    /// granularity view — IBM's `DT_CHECK(val % factor == 0)`.
    #[test]
    fn a_split_that_is_not_a_whole_number_of_granules_has_no_granularity_view() {
        let mut dims = DataStructDims::default();
        dims.out = size(64.0);
        dims.symbolic_dim_info.insert(
            D::Out,
            SymbolicDimInfo {
                max_size: 64,
                granularity: 16,
            },
        );
        dims.corelet_split
            .insert(D::Out, vec![DimVal(30), DimVal(34)]);
        assert_eq!(
            dims.primary_dim_to_val_for_component(
                D::Out,
                SenComponent::NoComponent,
                None,
                Some(CoreletId(0)),
                &PaddingFormType::default(),
                DimDensity::FULL,
                true,
            ),
            None
        );
    }

    /// The padded size of a windowed dim, of a full-span one, and the four cases that have none.
    ///
    /// ⛔ A NEGATIVE VALUE IS `-1`, NOT AN ABSENCE. The assertion below read `None` and passed,
    /// which is how a green test pinned the divergence: a compiled authority returns `-1` from
    /// `dsc/dims.cpp:567-568` before it has looked at the pad type at all.
    #[test]
    fn calculate_padded_spans_the_window_and_refuses_what_ibm_refuses() {
        let mut dims = DataStructDims::default();
        dims.j = size(7.0);
        dims.kj = size(3.0);
        dims.padding_sizes.insert(
            D::J,
            DimPaddingSizes {
                stride: 2,
                unneeded_pad: 1,
                window_dim: D::Kj,
                ..DimPaddingSizes::default()
            },
        );
        dims.padding_sizes.insert(
            D::In,
            DimPaddingSizes {
                pad_front: 1,
                pad_back: 2,
                unneeded_pad: 3,
                ..DimPaddingSizes::default()
            },
        );
        let padded = |dim, val, pad| {
            dims.calculate_padded(dim, DimVal(val), &PaddingFormType::new(dim, pad), false)
        };
        assert_eq!(padded(D::J, 7, PadType::PaddedWZeroPad), Some(DimVal(15)));
        assert_eq!(
            padded(D::J, 7, PadType::PaddedFullSpanWUnneeded),
            Some(DimVal(16))
        );
        assert_eq!(padded(D::J, 7, PadType::PaddedNoZeroPad), Some(DimVal(16)));
        assert_eq!(padded(D::J, 7, PadType::LoweredPadded), Some(DimVal(21)));
        assert_eq!(padded(D::J, 7, PadType::PaddedFullSpan), None);
        assert_eq!(padded(D::J, 7, PadType::NoPad), Some(DimVal(7)));
        assert_eq!(padded(D::In, 10, PadType::PaddedFullSpan), Some(DimVal(13)));
        assert_eq!(
            padded(D::In, 10, PadType::PaddedFullSpanWUnneeded),
            Some(DimVal(16))
        );
        assert_eq!(padded(D::In, 10, PadType::PaddedWZeroPad), None);
        assert_eq!(
            padded(D::J, -1, PadType::PaddedWZeroPad),
            Some(DimVal(-1)),
            "an unfilled dim stays unfilled"
        );
        assert_eq!(
            padded(D::Ij, 49, PadType::PaddedFullSpan),
            None,
            "a compound dim has no padded version"
        );
        assert_eq!(
            padded(D::Y, 4, PadType::PaddedFullSpan),
            None,
            "a padded dim with no padding sizes has none either"
        );
    }

    /// An unfilled dim answers `-1` and is not an absence: IBM returns it from `calculate_padded`
    /// ahead of every other branch (`dsc/dims.cpp:567-568`), so it reaches `primaryDimToVal_st` and
    /// the ratio-of-one divide in `makeDimNotSymbolic` as a value.
    ///
    /// ⛔ THE SENTINEL IS THE NEGATIVE CONTROL: it is the one absence, because IBM `DT_ERROR`s on it
    /// (`dsc/dims.cpp:554-556`). A compiled authority answers `-1` to every other row here, and
    /// `makeDimNotSymbolic` on an unfilled dim with a ratio of one returns having erased the entry.
    #[test]
    fn an_unfilled_dim_is_minus_one_and_only_the_sentinel_is_absent() {
        const EVERY_PAD: [PadType; 6] = [
            PadType::NoPad,
            PadType::LoweredPadded,
            PadType::PaddedNoZeroPad,
            PadType::PaddedWZeroPad,
            PadType::PaddedFullSpan,
            PadType::PaddedFullSpanWUnneeded,
        ];
        let unfilled = DataStructDims::default();
        assert_eq!(unfilled.primary_dim_to_val(D::Mb), Some(DimVal(-1)));
        assert_eq!(unfilled.primary_dim_to_val(D::Undefined), None);
        for pad in EVERY_PAD {
            assert_eq!(
                unfilled.calculate_padded(
                    D::J,
                    DimVal(-1),
                    &PaddingFormType::new(D::J, pad),
                    false
                ),
                Some(DimVal(-1)),
                "{pad:?} is short-circuited"
            );
        }
        let mut symbolic = DataStructDims::default();
        symbolic.symbolic_dim_info.insert(
            D::In,
            SymbolicDimInfo {
                max_size: 64,
                granularity: 64,
            },
        );
        assert_eq!(symbolic.make_dim_not_symbolic(D::In), Some(()));
        assert_eq!(symbolic.r#in, None, "-1 divided by one is still unfilled");
        assert!(!symbolic.symbolic_dim_info.contains_key(&D::In));
    }

    /// A named row reads `rowSplit_`, a PE or SFP reads `peSfpSplit_`, and with no corelet named the
    /// shares are summed when the dim is split across corelets and taken from the first when it is
    /// not.
    #[test]
    fn the_row_split_wins_and_pelrf_selects_the_pe_share() {
        let mut dims = DataStructDims::default();
        dims.mb = size(8.0);
        dims.i = size(4.0);
        dims.corelet_split.insert(D::Mb, vec![DimVal(4), DimVal(4)]);
        dims.row_split.insert(
            D::Mb,
            BTreeMap::from([
                (CoreletId(0), vec![DimVal(3), DimVal(1)]),
                (CoreletId(1), vec![DimVal(2), DimVal(2)]),
            ]),
        );
        dims.row_split.insert(
            D::I,
            BTreeMap::from([
                (CoreletId(0), vec![DimVal(4)]),
                (CoreletId(1), vec![DimVal(9)]),
            ]),
        );
        dims.pe_sfp_split.insert(
            D::Mb,
            BTreeMap::from([
                (
                    CoreletId(0),
                    BTreeMap::from([
                        (SenComponent::Pe, DimVal(3)),
                        (SenComponent::Sfp, DimVal(1)),
                    ]),
                ),
                (
                    CoreletId(1),
                    BTreeMap::from([
                        (SenComponent::Pe, DimVal(2)),
                        (SenComponent::Sfp, DimVal(2)),
                    ]),
                ),
            ]),
        );
        let view = |dim, comp, row, cl| {
            dims.primary_dim_to_val_for_component(
                dim,
                comp,
                row,
                cl,
                &PaddingFormType::default(),
                DimDensity::FULL,
                false,
            )
        };
        let none = SenComponent::NoComponent;
        assert_eq!(view(D::Mb, none, None, None), Some(DimVal(8)));
        assert_eq!(
            view(D::Mb, none, Some(RowId(0)), Some(CoreletId(0))),
            Some(DimVal(3))
        );
        assert_eq!(
            view(D::Mb, none, Some(RowId(0)), None),
            Some(DimVal(5)),
            "summed over corelets"
        );
        assert_eq!(
            view(D::I, none, Some(RowId(0)), None),
            Some(DimVal(4)),
            "the first corelet, since I is not split across corelets"
        );
        assert_eq!(view(D::Mb, none, Some(RowId(4)), None), None);
        assert_eq!(view(D::Mb, SenComponent::Pe, None, None), Some(DimVal(5)));
        assert_eq!(
            view(D::Mb, SenComponent::Pelrf, None, None),
            Some(DimVal(5)),
            "PELRF is mapped to PE"
        );
        assert_eq!(
            view(D::Mb, SenComponent::Sfplrf, None, Some(CoreletId(1))),
            Some(DimVal(2))
        );
        assert_eq!(
            view(D::Mb, SenComponent::Lx, None, None),
            Some(DimVal(8)),
            "a component that is neither PE nor SFP falls through to the corelet view"
        );
    }

    /// A component's own PT row selects its share, and a component with no row falls through.
    #[test]
    fn a_component_view_uses_the_components_own_row() {
        assert_eq!(SenComponent::Ptrow1_0.row_id(), Some(RowId(1)));
        assert_eq!(SenComponent::L0lurow7_1.row_id(), Some(RowId(7)));
        assert_eq!(SenComponent::Pe.row_id(), None);
        let mut dims = DataStructDims::default();
        dims.mb = size(8.0);
        dims.row_split.insert(
            D::Mb,
            BTreeMap::from([(CoreletId(0), vec![DimVal(3), DimVal(5)])]),
        );
        let view = |comp| {
            dims.data_stage_dim_to_val_comp_view(
                D::Mb,
                comp,
                Some(CoreletId(0)),
                &PaddingFormType::default(),
                DimDensity::FULL,
                false,
            )
        };
        assert_eq!(view(SenComponent::Ptrow1), Some(DimVal(5)));
        assert_eq!(view(SenComponent::Ptrow0_1), Some(DimVal(3)));
        assert_eq!(view(SenComponent::Lx), Some(DimVal(8)));
    }

    /// IBM's own worked example (`dsc/dims.cpp:722-728`): a,b,c limited to 2048 with a no longer
    /// symbolic and its granularity 4 leaves b,c limited to 512.
    #[test]
    fn prune_max_symbolic_volumes_is_the_authoritys_worked_example() {
        let symbolic = |max_size, granularity| SymbolicDimInfo {
            max_size,
            granularity,
        };
        let mut reference = DataStructDims::default();
        reference.symbolic_dim_info.insert(D::Mb, symbolic(64, 4));
        let mut dims = DataStructDims::default();
        dims.symbolic_dim_info.insert(D::In, symbolic(64, 16));
        dims.symbolic_dim_info.insert(D::Out, symbolic(64, 16));
        dims.max_symbolic_volume
            .insert(BTreeSet::from([D::In, D::Out, D::Mb]), SymbolicVolume(2048));
        let before = dims.max_symbolic_volume.clone();
        assert_eq!(dims.prune_max_symbolic_volumes(&reference), Some(()));
        assert_eq!(
            dims.max_symbolic_volume,
            BTreeMap::from([(BTreeSet::from([D::In, D::Out]), SymbolicVolume(512))])
        );

        let mut unknown = DataStructDims::default();
        unknown.symbolic_dim_info = dims.symbolic_dim_info.clone();
        unknown.max_symbolic_volume = before.clone();
        assert_eq!(
            unknown.prune_max_symbolic_volumes(&DataStructDims::default()),
            None,
            "the reference must know the dim being dropped"
        );
        assert_eq!(unknown.max_symbolic_volume, before, "nothing was written");
    }

    /// Making a dim symbolic takes its max, granularity, value and splits from the reference.
    #[test]
    fn make_dim_symbolic_takes_the_reference_max_and_splits() {
        let info = SymbolicDimInfo {
            max_size: 64,
            granularity: 16,
        };
        let mut reference = DataStructDims::default();
        reference.out = size(64.0);
        reference.symbolic_dim_info.insert(D::Out, info);
        reference
            .corelet_split
            .insert(D::Out, vec![DimVal(32), DimVal(32)]);
        let mut dims = DataStructDims::default();
        dims.out = size(16.0);
        dims.corelet_split
            .insert(D::Out, vec![DimVal(8), DimVal(8)]);
        assert_eq!(dims.make_dim_symbolic(&reference, D::Out), Some(()));
        assert_eq!(dims.out, size(64.0));
        assert_eq!(dims.symbolic_dim_info.get(&D::Out), Some(&info));
        assert_eq!(
            dims.corelet_split.get(&D::Out),
            Some(&vec![DimVal(32), DimVal(32)])
        );

        dims.out = size(8.0);
        assert_eq!(dims.make_dim_symbolic(&reference, D::Out), Some(()));
        assert_eq!(dims.out, size(8.0), "already symbolic, so left alone");
        assert_eq!(
            dims.make_dim_symbolic(&DataStructDims::default(), D::In),
            None,
            "the reference must know the dim"
        );
        assert!(!dims.symbolic_dim_info.contains_key(&D::In));
    }

    /// Making a dim concrete divides the dim and every split of it by max/granularity — and it is
    /// the field that is divided, not the max.
    #[test]
    fn make_dim_not_symbolic_divides_the_field_and_every_split() {
        let mut dims = DataStructDims::default();
        dims.out = size(32.0);
        dims.symbolic_dim_info.insert(
            D::Out,
            SymbolicDimInfo {
                max_size: 64,
                granularity: 16,
            },
        );
        dims.corelet_split
            .insert(D::Out, vec![DimVal(32), DimVal(32)]);
        dims.row_split.insert(
            D::Out,
            BTreeMap::from([(CoreletId(0), vec![DimVal(16), DimVal(16)])]),
        );
        dims.pe_sfp_split.insert(
            D::Out,
            BTreeMap::from([(
                CoreletId(0),
                BTreeMap::from([
                    (SenComponent::Pe, DimVal(24)),
                    (SenComponent::Sfp, DimVal(8)),
                ]),
            )]),
        );
        let before = dims.clone();
        assert_eq!(dims.make_dim_not_symbolic(D::Out), Some(()));
        assert_eq!(dims.out, size(8.0), "32/4, not 64/4");
        assert!(!dims.symbolic_dim_info.contains_key(&D::Out));
        assert_eq!(
            dims.corelet_split.get(&D::Out),
            Some(&vec![DimVal(8), DimVal(8)])
        );
        assert_eq!(
            dims.row_split.get(&D::Out),
            Some(&BTreeMap::from([(
                CoreletId(0),
                vec![DimVal(4), DimVal(4)]
            )]))
        );
        assert_eq!(
            dims.pe_sfp_split.get(&D::Out),
            Some(&BTreeMap::from([(
                CoreletId(0),
                BTreeMap::from([
                    (SenComponent::Pe, DimVal(6)),
                    (SenComponent::Sfp, DimVal(2)),
                ])
            )]))
        );
        assert_eq!(
            dims.make_dim_not_symbolic(D::Out),
            Some(()),
            "not symbolic any more, so left alone"
        );

        let mut indivisible = before.clone();
        indivisible
            .corelet_split
            .insert(D::Out, vec![DimVal(30), DimVal(34)]);
        assert_eq!(indivisible.make_dim_not_symbolic(D::Out), None);
        assert!(
            indivisible.symbolic_dim_info.contains_key(&D::Out),
            "nothing was written"
        );
    }

    /// `operator==` (`dsc/dims.cpp:832-835`) compares dim and kind, and `std::hash` keys an
    /// `unordered_set` on the pair (`ddc/ddc_transformation_util.cpp:1147`), so a hash set must
    /// separate pairs differing in either half and merge the two routes to the same pair.
    #[test]
    fn a_dim_and_kind_pair_compares_and_hashes_on_both_halves() {
        let mb_unpadded = PrimaryDimAndKind::new(D::Mb, MetaDimKind::Unpadded);
        assert_ne!(
            mb_unpadded,
            PrimaryDimAndKind::new(D::Mb, MetaDimKind::Padded),
            "kind_ is not compared"
        );
        assert_ne!(
            mb_unpadded,
            PrimaryDimAndKind::new(D::Y, MetaDimKind::Unpadded),
            "dim_ is not compared"
        );

        let mut set = std::collections::HashSet::new();
        for dim in EVERY_DIM {
            for kind in EVERY_KIND {
                assert!(
                    set.insert(PrimaryDimAndKind::new(dim, kind)),
                    "{dim:?}/{kind:?} shares a slot"
                );
            }
        }
        assert_eq!(set.len(), EVERY_DIM.len() * EVERY_KIND.len());
        assert!(
            !set.insert(PrimaryDimAndKind::from(D::Mb)),
            "the bare-dim route must land on the pair already there"
        );
        assert!(set.contains(&mb_unpadded));
    }

    /// The header line is streamed even for a form carrying no dims, level 0 — IBM's default
    /// argument (`dsc/dims.h:116`) — indents by nothing, and the sentinel is a legal key that
    /// `primaryDimToString` spells.
    #[test]
    fn an_empty_padding_form_still_prints_its_header() {
        let mut out = String::new();
        PaddingFormType::default().print(&mut out, 0);
        assert_eq!(out, "\nPadding= ");

        let mut out = String::new();
        PaddingFormType::new(D::Undefined, PadType::PaddedFullSpan).print(&mut out, 1);
        assert_eq!(out, "\n  Padding=  (undefined: padded_fullspan)");
    }

    /// `getPaddingAsStr` (`dsc/dims.cpp:815-817`) spells whatever `getPadding` returned, including
    /// the `NOPAD` a dim with no entry falls back to while another dim does carry one.
    #[test]
    fn every_padding_form_is_spelled_through_the_accessor() {
        for pad in PadType::ALL {
            let form = PaddingFormType::new(D::Ki, pad);
            assert_eq!(form.padding_as_str(D::Ki), pad.name());
            assert_eq!(form.padding(D::Kj), PadType::NoPad);
            assert_eq!(form.padding_as_str(D::Kj), "nopad");
            assert!(form.has_padding_info());
        }
    }
}
