//! Re-ported from the C++ authority. See crustify-scheduler/AGENT-BRIEF.md.
//!
//! `dsc/dims.h` classifies a data structure's dimensions: which primary dim, which meta-dim kind,
//! how the dim is padded, and the padding scalars that produce its padded size.

use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};

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
    /// The authority's `PrimaryDimTypesCount` (`dsc/dims.h:43`): no dimension. It is a live map key
    /// as well as a field value — `ddc/ddcv1.cpp:1803` skips it while iterating a keyed map — so it
    /// stays a variant, and stays last, rather than becoming an absent `Option`.
    #[default]
    Undefined = 12,
}

/// ⛔ E0308 IF A DIM IS EVER INSERTED, DROPPED OR LEFT OUT OF `ALL`: the sentinel's discriminant is
/// the count of the real dims, which is what `ddc/ddl/ddl_conversion.h:309` iterates.
const _: [(); PrimaryDimTypes::COUNT] = [(); PrimaryDimTypes::Undefined as usize];

impl PrimaryDimTypes {
    /// The real dimensions in the authority's order, which is what `0..PrimaryDimTypesCount`
    /// iterates (`ddc/ddl/ddl_conversion.h:309`). The sentinel is not one of them.
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
///
/// `dsc/dims.h:76-82`. One dimension together with which of its quantities is meant — the key
/// every loop carries in `dims_` and the element of every split-dim set.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct PrimaryDimAndKind {
    /// Field: e002_PrimaryDimAndKind.dim_
    pub dim: PrimaryDimTypes,
    /// Field: e002_PrimaryDimAndKind.kind_
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
    /// `ddc/ddc_transformation_util.cpp:203-207` iterates an `unordered_map` keyed by this type and
    /// pushes the result into a loop's dim list, so libstdc++'s bucket order reaches the minted
    /// tree. Matching that needs libstdc++'s bucket policy too; an ordered container is the
    /// deterministic answer, and `Ord` is derived above for it.
    pub fn hash_value(self) -> usize {
        ((self.dim as usize) << MetaDimKind::HASH_SHIFT) ^ (self.kind as usize)
    }
}

/// The authority's hash, so a `HashMap` keyed by this type buckets on IBM's value.
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
///
/// `dsc/dims.h:94-120`. The padding form of each dimension of one allocation, transfer or
/// coordinate. A dim absent from the map is unpadded, so the empty form is the default one every
/// `const PaddingFormType &padded = {}` parameter takes.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PaddingFormType {
    /// Field: e003_PaddingFormType.padding_
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
///
/// `dsc/dims.h:134-146`. Everything that contributes to one primary dimension's padded size.
///
/// ⛔ SIGNED, AND `-1` IS LIVE: `dsc/dims.cpp:295` tests `padFront_ < 0` together with an absent
/// `windowDim_` to decide that a dim has no total size at all.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DimPaddingSizes {
    /// Field: e004_DimPaddingSizes.padFront_
    pub pad_front: i32,
    /// Field: e004_DimPaddingSizes.padBack_
    pub pad_back: i32,
    /// Field: e004_DimPaddingSizes.unneededPad_
    ///
    /// Total unneeded elements.
    pub unneeded_pad: i32,
    /// Field: e004_DimPaddingSizes.unneededPadFront_
    ///
    /// Unneeded elements that come from `pad_front`.
    pub unneeded_pad_front: i32,
    /// Field: e004_DimPaddingSizes.unneededPadBack_
    ///
    /// Unneeded elements that come from `pad_back`.
    pub unneeded_pad_back: i32,
    /// Field: e004_DimPaddingSizes.stride_
    pub stride: i32,
    /// Field: e004_DimPaddingSizes.dilation_
    pub dilation: i32,
    /// Field: e004_DimPaddingSizes.windowDim_
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
    /// ⭐ ABSENT RATHER THAN A THROW. IBM `DT_ERROR`s on the other five kinds, and the one caller
    /// (`ddc/ddl/ddl_conversion.cpp:1831`) reaches it with a kind read from the DDL, so the arm is
    /// live. The kinds that name no stored scalar are the absent case, not a stop.
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
///
/// `dsc/dims.h:148-155`. The max and granularity of one symbolic dimension. Whether a dim is
/// symbolic at all is its presence in `DataStructDims::symbolicDimInfo_`, never these values
/// (`dsc/dims.cpp:734`), so `-1` is "not filled in" and not a state anyone tests.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SymbolicDimInfo {
    /// Field: e005_SymbolicDimInfo.maxSize_
    pub max_size: i32,
    /// Field: e005_SymbolicDimInfo.granularity_
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
        assert_eq!(PrimaryDimTypes::ALL[..], EVERY_DIM[..PrimaryDimTypes::COUNT]);
        assert_eq!(MetaDimKind::ALL[..], EVERY_KIND[..MetaDimKind::COUNT]);

        // A `std::map` keyed by either enum iterates in this order, so `Ord` must agree with it.
        let mut sorted = EVERY_DIM;
        sorted.sort();
        assert_eq!(sorted, EVERY_DIM);
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

    /// Only four kinds name a stored scalar; the other five are absent.
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
    }

    /// Unpadded, unit stride and dilation, no window dim — and equality that sees all eight fields,
    /// so swapping front for back is a difference.
    #[test]
    fn dim_padding_sizes_defaults_to_unit_stride_and_compares_every_field() {
        let base = DimPaddingSizes::default();
        assert_eq!(
            (base.pad_front, base.unneeded_pad, base.stride, base.dilation),
            (0, 0, 1, 1)
        );
        assert_eq!(base.window_dim, PrimaryDimTypes::Undefined);

        let front = DimPaddingSizes {
            pad_front: 1,
            ..base
        };
        let back = DimPaddingSizes {
            pad_back: 1,
            ..base
        };
        assert_ne!(front, back);
        assert_eq!(
            front,
            DimPaddingSizes {
                pad_front: 1,
                ..base
            }
        );
        assert_ne!(
            base,
            DimPaddingSizes {
                window_dim: PrimaryDimTypes::Ki,
                ..base
            }
        );
    }

    /// Both symbolic values start unfilled, and equality sees both.
    #[test]
    fn symbolic_dim_info_starts_unfilled_and_compares_both_fields() {
        let unset = SymbolicDimInfo::default();
        assert_eq!((unset.max_size, unset.granularity), (-1, -1));
        assert_ne!(
            SymbolicDimInfo {
                max_size: 64,
                granularity: 16
            },
            SymbolicDimInfo {
                max_size: 16,
                granularity: 64
            }
        );
    }
}
