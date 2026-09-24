//! THE SCHEDULED-SUPERDSC WIRE FORMAT'S CLOSED SETS.
//!
//! Every enum here is a transcription of the exact C++ string table the importer consults —
//! `EnumsConversion::stringTo*` — with the citation in the type's doc. The wire is JSON and carries
//! strings; the reader converts at the boundary so nothing downstream is a raw `String` standing in
//! for a closed set.
//!
//! ⭐ WHERE A VOCABULARY ALREADY EXISTS IN THIS CRATE IT IS NOT REPEATED HERE:
//! [`sys_arch_spec::arch_enums::SenComponent`] (units, storages, components) and
//! [`crate::generated::DataType`] (`dataFormat_`) carry their own `from_spelling` parse boundaries
//! and are re-used. What remains is the vocabulary that exists nowhere else in the crate.

use crate::generated::DataType;

/// A compute node's `type_` — `EnumsConversion::stringToComputeType` (`dsc/dscdefn.cpp:33-103`).
///
/// ⛔ NOT THE GENERATED [`crate::generated::ComputeType`](crate::generated::ComputeType): that one is
/// censused from the `.ddl` templates and spells `FMA16` in caps, while the wire carries the
/// `computeTypeToString` lower-case spellings (`fma16`). The two vocabularies also disagree on
/// membership — `fma8`/`fma4`/`ima8`/`ima4` are wire-only, and `EQUAL`'s wire spelling is `equal` —
/// so a conversion between them is a mapping to write deliberately in the adapter, not an identity
/// to assume here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WireComputeType {
    /// `macc`.
    Macc,
    /// `fma32`.
    Fma32,
    /// `fma16`.
    Fma16,
    /// `fma8`.
    Fma8,
    /// `fma4`.
    Fma4,
    /// `ima8`.
    Ima8,
    /// `ima4`.
    Ima4,
    /// `fmax`.
    Fmax,
    /// `fmin`.
    Fmin,
    /// `fabsmax`.
    Fabsmax,
    /// `fsignedabsmineq`.
    Fsignedabsmineq,
    /// `fnms`.
    Fnms,
    /// `fsub`.
    Fsub,
    /// `fmul`.
    Fmul,
    /// `and`.
    And,
    /// `or`.
    Or,
    /// `xnorround`.
    Xnorround,
    /// `andnot`.
    Andnot,
    /// `fcmp`.
    Fcmp,
    /// `select`.
    Select,
    /// `packmerge`.
    Packmerge,
    /// `reciprocal`.
    Reciprocal,
    /// `layernormscale`.
    Layernormscale,
    /// `fest`.
    Fest,
    /// `ime`.
    Ime,
    /// `ee`.
    Ee,
    /// `reduce`.
    Reduce,
    /// `splat`.
    Splat,
    /// `shr`.
    Shr,
    /// `icvt`.
    Icvt,
    /// `gcvt`.
    Gcvt,
    /// `sigmoid`.
    Sigmoid,
    /// `exp_p1`.
    ExpP1,
    /// `exp_p2`.
    ExpP2,
    /// `log_p1`.
    LogP1,
    /// `log_p2`.
    LogP2,
    /// `gelu`.
    Gelu,
    /// `gelu_bwd_p1`.
    GeluBwdP1,
    /// `gelu_bwd_p2`.
    GeluBwdP2,
    /// `where3`.
    Where3,
    /// `sqrt`.
    Sqrt,
    /// `rsqrt`.
    Rsqrt,
    /// `mish_p1`.
    MishP1,
    /// `mish_p2`.
    MishP2,
    /// `greaterequal`.
    Greaterequal,
    /// `lesserequal`.
    Lesserequal,
    /// `greaterthan`.
    Greaterthan,
    /// `lesserthan`.
    Lesserthan,
    /// `equal`.
    Equal,
    /// `notequal`.
    Notequal,
    /// `undefined`.
    Undefined,
    /// `exx2_32_p1`.
    Exx2_32P1,
    /// `exx2_32_p2`.
    Exx2_32P2,
    /// `exx2_32_p3`.
    Exx2_32P3,
    /// `exp`.
    Exp,
    /// `shuffle`.
    Shuffle,
    /// `dl16tobf16`.
    Dl16tobf16,
    /// `softplus_p1`.
    SoftplusP1,
    /// `softplus_p2`.
    SoftplusP2,
    /// `automatic_shuffling`.
    AutomaticShuffling,
    /// `assign`.
    Assign,
    /// `floor`.
    Floor,
    /// `idx32toaddr`.
    Idx32toaddr,
    /// `addi32toi32`.
    Addi32toi32,
    /// `addi64toi64`.
    Addi64toi64,
    /// `muli32toi32`.
    Muli32toi32,
    /// `cast`.
    Cast,
    /// `muli64toi64_pe`.
    Muli64toi64Pe,
    /// `muli64toi64_sfp`.
    Muli64toi64Sfp,
}

impl WireComputeType {
    /// `EnumsConversion::computeTypeToString` (`dsc/dscdefn.cpp:33-101`), one arm per entry.
    #[must_use]
    pub const fn spelling(self) -> &'static str {
        match self {
            Self::Macc => "macc",
            Self::Fma32 => "fma32",
            Self::Fma16 => "fma16",
            Self::Fma8 => "fma8",
            Self::Fma4 => "fma4",
            Self::Ima8 => "ima8",
            Self::Ima4 => "ima4",
            Self::Fmax => "fmax",
            Self::Fmin => "fmin",
            Self::Fabsmax => "fabsmax",
            Self::Fsignedabsmineq => "fsignedabsmineq",
            Self::Fnms => "fnms",
            Self::Fsub => "fsub",
            Self::Fmul => "fmul",
            Self::And => "and",
            Self::Or => "or",
            Self::Xnorround => "xnorround",
            Self::Andnot => "andnot",
            Self::Fcmp => "fcmp",
            Self::Select => "select",
            Self::Packmerge => "packmerge",
            Self::Reciprocal => "reciprocal",
            Self::Layernormscale => "layernormscale",
            Self::Fest => "fest",
            Self::Ime => "ime",
            Self::Ee => "ee",
            Self::Reduce => "reduce",
            Self::Splat => "splat",
            Self::Shr => "shr",
            Self::Icvt => "icvt",
            Self::Gcvt => "gcvt",
            Self::Sigmoid => "sigmoid",
            Self::ExpP1 => "exp_p1",
            Self::ExpP2 => "exp_p2",
            Self::LogP1 => "log_p1",
            Self::LogP2 => "log_p2",
            Self::Gelu => "gelu",
            Self::GeluBwdP1 => "gelu_bwd_p1",
            Self::GeluBwdP2 => "gelu_bwd_p2",
            Self::Where3 => "where3",
            Self::Sqrt => "sqrt",
            Self::Rsqrt => "rsqrt",
            Self::MishP1 => "mish_p1",
            Self::MishP2 => "mish_p2",
            Self::Greaterequal => "greaterequal",
            Self::Lesserequal => "lesserequal",
            Self::Greaterthan => "greaterthan",
            Self::Lesserthan => "lesserthan",
            Self::Equal => "equal",
            Self::Notequal => "notequal",
            Self::Undefined => "undefined",
            Self::Exx2_32P1 => "exx2_32_p1",
            Self::Exx2_32P2 => "exx2_32_p2",
            Self::Exx2_32P3 => "exx2_32_p3",
            Self::Exp => "exp",
            Self::Shuffle => "shuffle",
            Self::Dl16tobf16 => "dl16tobf16",
            Self::SoftplusP1 => "softplus_p1",
            Self::SoftplusP2 => "softplus_p2",
            Self::AutomaticShuffling => "automatic_shuffling",
            Self::Assign => "assign",
            Self::Floor => "floor",
            Self::Idx32toaddr => "idx32toaddr",
            Self::Addi32toi32 => "addi32toi32",
            Self::Addi64toi64 => "addi64toi64",
            Self::Muli32toi32 => "muli32toi32",
            Self::Cast => "cast",
            Self::Muli64toi64Pe => "muli64toi64_pe",
            Self::Muli64toi64Sfp => "muli64toi64_sfp",
        }
    }

    /// `EnumsConversion::stringToComputeType` (`dsc/dscdefn.cpp:102-103`) — the parse boundary.
    /// An unknown spelling is `None`; the reader turns that into a refusal.
    #[must_use]
    pub fn from_spelling(text: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|t| t.spelling() == text)
    }

    /// Every compute type, in the order the C++ table declares them.
    pub const ALL: [Self; 69] = [
        Self::Macc,
        Self::Fma32,
        Self::Fma16,
        Self::Fma8,
        Self::Fma4,
        Self::Ima8,
        Self::Ima4,
        Self::Fmax,
        Self::Fmin,
        Self::Fabsmax,
        Self::Fsignedabsmineq,
        Self::Fnms,
        Self::Fsub,
        Self::Fmul,
        Self::And,
        Self::Or,
        Self::Xnorround,
        Self::Andnot,
        Self::Fcmp,
        Self::Select,
        Self::Packmerge,
        Self::Reciprocal,
        Self::Layernormscale,
        Self::Fest,
        Self::Ime,
        Self::Ee,
        Self::Reduce,
        Self::Splat,
        Self::Shr,
        Self::Icvt,
        Self::Gcvt,
        Self::Sigmoid,
        Self::ExpP1,
        Self::ExpP2,
        Self::LogP1,
        Self::LogP2,
        Self::Gelu,
        Self::GeluBwdP1,
        Self::GeluBwdP2,
        Self::Where3,
        Self::Sqrt,
        Self::Rsqrt,
        Self::MishP1,
        Self::MishP2,
        Self::Greaterequal,
        Self::Lesserequal,
        Self::Greaterthan,
        Self::Lesserthan,
        Self::Equal,
        Self::Notequal,
        Self::Undefined,
        Self::Exx2_32P1,
        Self::Exx2_32P2,
        Self::Exx2_32P3,
        Self::Exp,
        Self::Shuffle,
        Self::Dl16tobf16,
        Self::SoftplusP1,
        Self::SoftplusP2,
        Self::AutomaticShuffling,
        Self::Assign,
        Self::Floor,
        Self::Idx32toaddr,
        Self::Addi32toi32,
        Self::Addi64toi64,
        Self::Muli32toi32,
        Self::Cast,
        Self::Muli64toi64Pe,
        Self::Muli64toi64Sfp,
    ];
}

/// HOW A LOOP DIMENSION IS COUNTED — `EnumsConversion::stringToMetaDimKind`
/// (`dsc/dims.cpp:50-55`), the `kind_` of a loop's `dims_` entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MetaDimKind {
    /// `unpadded`.
    Unpadded,
    /// `padded`.
    Padded,
    /// `pad_front`.
    PadFront,
    /// `pad_back`.
    PadBack,
    /// `pad_valid`.
    PadValid,
    /// `window`.
    WindowDim,
    /// `stride`.
    Stride,
    /// `dilation`.
    Dilation,
    /// `undefined`.
    Count,
}

impl MetaDimKind {
    /// `EnumsConversion::metaDimKindToString` (`dsc/dims.cpp:56-57`), one arm per entry.
    #[must_use]
    pub const fn spelling(self) -> &'static str {
        match self {
            Self::Unpadded => "unpadded",
            Self::Padded => "padded",
            Self::PadFront => "pad_front",
            Self::PadBack => "pad_back",
            Self::PadValid => "pad_valid",
            Self::WindowDim => "window",
            Self::Stride => "stride",
            Self::Dilation => "dilation",
            Self::Count => "undefined",
        }
    }

    /// `EnumsConversion::stringToMetaDimKind` (`dsc/dims.cpp:50-55`) — the parse boundary.
    #[must_use]
    pub fn from_spelling(text: &str) -> Option<Self> {
        Some(match text {
            "unpadded" => Self::Unpadded,
            "padded" => Self::Padded,
            "pad_front" => Self::PadFront,
            "pad_back" => Self::PadBack,
            "pad_valid" => Self::PadValid,
            "window" => Self::WindowDim,
            "stride" => Self::Stride,
            "dilation" => Self::Dilation,
            "undefined" => Self::Count,
            _ => return None,
        })
    }
}

/// A dimension's padding kind — `EnumsConversion::stringToPadType` (`dsc/dims.cpp:39-48`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PadType {
    /// `nopad`.
    Nopad,
    /// `lowered_padded`.
    LoweredPadded,
    /// `padded_nozeropad`.
    PaddedNozeropad,
    /// `padded_wzeropad`.
    PaddedWzeropad,
    /// `padded_fullspan`.
    PaddedFullspan,
    /// `padded_fullspan_wunneeded`.
    PaddedFullspanWunneeded,
}

impl PadType {
    /// `EnumsConversion::padTypeToString` (`dsc/dims.cpp:39-46`), one arm per entry.
    #[must_use]
    pub const fn spelling(self) -> &'static str {
        match self {
            Self::Nopad => "nopad",
            Self::LoweredPadded => "lowered_padded",
            Self::PaddedNozeropad => "padded_nozeropad",
            Self::PaddedWzeropad => "padded_wzeropad",
            Self::PaddedFullspan => "padded_fullspan",
            Self::PaddedFullspanWunneeded => "padded_fullspan_wunneeded",
        }
    }

    /// `EnumsConversion::stringToPadType` (`dsc/dims.cpp:47-48`) — the parse boundary.
    #[must_use]
    pub fn from_spelling(text: &str) -> Option<Self> {
        Some(match text {
            "nopad" => Self::Nopad,
            "lowered_padded" => Self::LoweredPadded,
            "padded_nozeropad" => Self::PaddedNozeropad,
            "padded_wzeropad" => Self::PaddedWzeropad,
            "padded_fullspan" => Self::PaddedFullspan,
            "padded_fullspan_wunneeded" => Self::PaddedFullspanWunneeded,
            _ => return None,
        })
    }
}

/// An allocation's indirection kind — `EnumsConversion::stringToIndirectAllocType`
/// (`dsc/dsc2.cpp:2436-2443`), the `indirectAllocType_` of an allocate node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IndirectAllocType {
    /// `no_indirection`.
    NoIndirection,
    /// `value_tensor`.
    ValueTensor,
    /// `index_tensor`.
    IndexTensor,
}

impl IndirectAllocType {
    /// `EnumsConversion::indirectAllocTypeToString` (`dsc/dsc2.cpp:2436-2440`).
    #[must_use]
    pub const fn spelling(self) -> &'static str {
        match self {
            Self::NoIndirection => "no_indirection",
            Self::ValueTensor => "value_tensor",
            Self::IndexTensor => "index_tensor",
        }
    }

    /// `EnumsConversion::stringToIndirectAllocType` (`dsc/dsc2.cpp:2441-2443`) — the parse boundary.
    #[must_use]
    pub fn from_spelling(text: &str) -> Option<Self> {
        Some(match text {
            "no_indirection" => Self::NoIndirection,
            "value_tensor" => Self::ValueTensor,
            "index_tensor" => Self::IndexTensor,
            _ => return None,
        })
    }
}

/// WHICH SEGMENT A LABELED DATASPACE BELONGS TO — `EnumsConversion::stringToLdsSegment`
/// (`dsc/dscdefn.cpp:108-139`), the `segment_` of a `labeledDs_` entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LdsSegment {
    /// `output`.
    Output,
    /// `input`.
    Input,
    /// `stack`.
    Stack,
    /// `model`.
    Model,
    /// `heap`.
    Heap,
    /// `reserve1`.
    Reserve1,
    /// `reserve2`.
    Reserve2,
    /// `const`.
    Const,
}

impl LdsSegment {
    /// `EnumsConversion::ldsSegmentToString` (`dsc/dscdefn.cpp:108-118`) — the eight non-HMI
    /// spellings; the `segNHMIN` HMI spellings are excluded because no fixture carries one and the
    /// port's lowering never reads them.
    #[must_use]
    pub const fn spelling(self) -> &'static str {
        match self {
            Self::Output => "output",
            Self::Input => "input",
            Self::Stack => "stack",
            Self::Model => "model",
            Self::Heap => "heap",
            Self::Reserve1 => "reserve1",
            Self::Reserve2 => "reserve2",
            Self::Const => "const",
        }
    }

    /// `EnumsConversion::stringToLdsSegment` (`dsc/dscdefn.cpp:137-139`) — the parse boundary.
    #[must_use]
    pub fn from_spelling(text: &str) -> Option<Self> {
        Some(match text {
            "output" => Self::Output,
            "input" => Self::Input,
            "stack" => Self::Stack,
            "model" => Self::Model,
            "heap" => Self::Heap,
            "reserve1" => Self::Reserve1,
            "reserve2" => Self::Reserve2,
            "const" => Self::Const,
            _ => return None,
        })
    }
}

/// WHICH SCALED-LDS CATEGORY A LABELED DATASPACE IS —
/// `LabeledDsInfo::stringToScaledLdsCategory` (`dsc/dscdefn.cpp:133-140`), the
/// `scaledLdsCategory_` of a `labeledDs_` entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScaledLdsCategory {
    /// `regular_tensor` — the default (`dsc/dscdefn.h:356`).
    RegularTensor,
    /// `value_tensor`.
    ValueTensor,
    /// `scale_tensor` — for MX scale.
    ScaleTensor,
}

impl ScaledLdsCategory {
    /// `LabeledDsInfo::scaledLdsCategoryToString` (`dsc/dscdefn.cpp:133-137`).
    #[must_use]
    pub const fn spelling(self) -> &'static str {
        match self {
            Self::RegularTensor => "regular_tensor",
            Self::ValueTensor => "value_tensor",
            Self::ScaleTensor => "scale_tensor",
        }
    }

    /// `LabeledDsInfo::stringToScaledLdsCategory` (`dsc/dscdefn.cpp:138-140`) — the parse
    /// boundary.
    #[must_use]
    pub fn from_spelling(text: &str) -> Option<Self> {
        Some(match text {
            "regular_tensor" => Self::RegularTensor,
            "value_tensor" => Self::ValueTensor,
            "scale_tensor" => Self::ScaleTensor,
            _ => return None,
        })
    }
}

/// WHICH TARGET A DSC IS FOR — `EnumsConversion::stringToSenTargets`
/// (`util/sendefs/sendefs.cpp:121-128`), the `target_` at the sdsc level and each op's own
/// `target_`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SenTarget {
    /// `sentient`.
    Sentient,
    /// `senulator`.
    Senulator,
    /// `sentf`.
    Sentf,
    /// `senpcfg`.
    Senpcfg,
    /// `systemc`.
    Systemc,
    /// `r5ss`.
    R5ss,
    /// `host`.
    Host,
    /// `nop`.
    Nop,
    /// `undefined`.
    Undefined,
    /// `INVALID` — what the C++ `FromString<SenTargets>` (`util/sendefs/sendefs.h:304-324`)
    /// returns for an unknown string. The dump/import table pair above has no row for it, so it
    /// never appears on the wire; the reader refuses an unknown spelling instead.
    Invalid,
}

impl SenTarget {
    /// `EnumsConversion::senTargetsToString` (`util/sendefs/sendefs.cpp:121-126`) — the table
    /// BOTH the dumper writes through and the importer reads through (`dsc/superdsc.cpp:383`,
    /// `:803`), so the wire spellings are these lower-case ones, not `FromString`'s uppers.
    #[must_use]
    pub const fn spelling(self) -> &'static str {
        match self {
            Self::Sentient => "sentient",
            Self::Senulator => "senulator",
            Self::Sentf => "sentf",
            Self::Senpcfg => "senpcfg",
            Self::Systemc => "systemc",
            Self::R5ss => "r5ss",
            Self::Host => "host",
            Self::Nop => "nop",
            Self::Undefined => "undefined",
            Self::Invalid => "INVALID",
        }
    }

    /// `EnumsConversion::stringToSenTargets` (`util/sendefs/sendefs.cpp:127-128`, the flipped
    /// table) — the parse boundary. An unknown spelling is `None`: the reader refuses where the
    /// C++ map lookup would throw.
    #[must_use]
    pub fn from_spelling(text: &str) -> Option<Self> {
        Some(match text {
            "sentient" => Self::Sentient,
            "senulator" => Self::Senulator,
            "sentf" => Self::Sentf,
            "senpcfg" => Self::Senpcfg,
            "systemc" => Self::Systemc,
            "r5ss" => Self::R5ss,
            "host" => Self::Host,
            "nop" => Self::Nop,
            "undefined" => Self::Undefined,
            _ => return None,
        })
    }
}

/// THE SCHEDULE TREE'S NODE KINDS — `ScheduleNode::NodeType` (`dsc/dsc2.h:457-458`,
/// `dsc/dsc2.cpp:1879-1890`).
///
/// ⛔ THE IMPORTER REFUSES THE 9TH. `stringToNodeType` covers `invalid` too, but scan 1 of the
/// import (`dsc/dsc2.cpp:1327-1340`) dispatches on the parsed kind and `INVALID` reaches the
/// `else` arm's `DT_ERROR("Missing basic fields")`; a wire file that spelled it would be refused
/// there, so the reader spells the refusal here instead of carrying a kind nothing can act on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NodeType {
    /// `block`.
    Block,
    /// `loop`.
    Loop,
    /// `transfer`.
    Transfer,
    /// `compute`.
    Compute,
    /// `sync`.
    Sync,
    /// `condition`.
    Condition,
    /// `stickmask`.
    StickMask,
    /// `allocate`.
    Allocate,
}

impl NodeType {
    /// `ScheduleNode::nodeTypeToString` (`dsc/dsc2.cpp:1879-1888`), one arm per entry.
    #[must_use]
    pub const fn spelling(self) -> &'static str {
        match self {
            Self::Block => "block",
            Self::Loop => "loop",
            Self::Transfer => "transfer",
            Self::Compute => "compute",
            Self::Sync => "sync",
            Self::Condition => "condition",
            Self::StickMask => "stickmask",
            Self::Allocate => "allocate",
        }
    }

    /// `ScheduleNode::stringToNodeType` (`dsc/dsc2.cpp:1889-1890`) — the parse boundary. `invalid`
    /// and an unknown spelling are both `None`: the importer's scan 1 turns the former into
    /// `DT_ERROR("Missing basic fields")` (`dsc/dsc2.cpp:1338-1340`), which is a refusal.
    #[must_use]
    pub fn from_spelling(text: &str) -> Option<Self> {
        Some(match text {
            "block" => Self::Block,
            "loop" => Self::Loop,
            "transfer" => Self::Transfer,
            "compute" => Self::Compute,
            "sync" => Self::Sync,
            "condition" => Self::Condition,
            "stickmask" => Self::StickMask,
            "allocate" => Self::Allocate,
            _ => return None,
        })
    }
}

/// `dataFormat_` ON THE WIRE parses through the generated [`DataType`]'s vocabulary, which is the
/// same one the C++ `FromString<DataFormats>` reads (`util/sendefs/sendefs.h:254-301`).
///
/// ⚠️ The C++ `FromString` returns `INVALID` for an unknown spelling instead of refusing; the
/// reader refuses, because a data format that is `INVALID` has no width and every consumer of the
/// field divides by one. The generated enum has no `ALL` (the `.ddl` census is scratchy's emission
/// scope, not the wire's), so the wire's own membership is spelled out here and pinned by test
/// against the C++ `FromString` chain's literals.
#[must_use]
pub fn data_format(text: &str) -> Option<DataType> {
    Some(match text {
        "IEEE_FP32" => DataType::IeeeFp32,
        "SEN169_FP16" => DataType::Sen169Fp16,
        "SEN143_FP8" => DataType::Sen143Fp8,
        "SEN152_FP8" => return None, // in the C++ chain, outside the generated census
        "SEN153_FP9" => return None, // in the C++ chain, outside the generated census
        "SENINT2" => return None, // in the C++ chain, outside the generated census
        "SENINT4" => DataType::Senint4,
        "SENINT8" => DataType::Senint8,
        "SENINT16" => return None, // in the C++ chain, outside the generated census
        "SENINT24" => DataType::Senint24,
        "IEEE_INT32" => return None, // in the C++ chain, outside the generated census
        "IEEE_INT64" => return None, // in the C++ chain, outside the generated census
        "SENUINT32" => DataType::Senuint32,
        "SENUINT2" => return None, // in the C++ chain, outside the generated census
        "IEEE_FP16" => return None, // in the C++ chain, outside the generated census
        "BOOL" => DataType::Bool,
        "BFLOAT16" => DataType::Bfloat16,
        "SEN18F_FP24" => return None, // in the C++ chain, outside the generated census
        "SEN080_FP8" => DataType::Sen080Fp8,
        "SEN053_FP8" => DataType::Sen053Fp8,
        "SEN121_FP4" => DataType::Sen121Fp4,
        _ => return None,
    })
}

/// A compute op's fidelity — `OpAttributes::stringtoFidelity` (`dsc/dscdefn.cpp:157-160`), the
/// `fidelity_` of a `computeOp_` entry's `attributes_`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Fidelity {
    /// `regular`.
    Regular,
    /// `fast`.
    Fast,
}

impl Fidelity {
    /// `OpAttributes::fidelityToString` (`dsc/dscdefn.cpp:153-156`).
    #[must_use]
    pub const fn spelling(self) -> &'static str {
        match self {
            Self::Regular => "regular",
            Self::Fast => "fast",
        }
    }

    /// `OpAttributes::stringtoFidelity` (`dsc/dscdefn.cpp:157-160`) — the parse boundary.
    #[must_use]
    pub fn from_spelling(text: &str) -> Option<Self> {
        Some(match text {
            "regular" => Self::Regular,
            "fast" => Self::Fast,
            _ => return None,
        })
    }
}

/// WHERE A COMPUTE OP SITS IN THE LOOP NEST — `DesignSpaceConfig::stringToLoopName`
/// (`dsc/designSpaceConfig.cpp:9239-9241`, the flip of `loopNameToString` at `:9222-9238`), the
/// `location` of a `computeOp_` entry.
///
/// ⚠️ ONLY THE SPELLINGS THE DUMPER WRITES. `location` is dumped through `loopNameToString`
/// itself (`designSpaceConfig.cpp:6669-6670`), so every spelling this reads is one that map can
/// produce — including `"Inner"` (the `INNER` default both fixtures carry) and `"invalid"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WireLoopName {
    /// `Inner` — `INNER`, the innermost loop.
    Inner,
    /// `dbin` … `dby` — the DB (double-buffer) loops.
    Dbin,
    /// `dbout`.
    Dbout,
    /// `dbij`.
    Dbij,
    /// `dbi`.
    Dbi,
    /// `dbj`.
    Dbj,
    /// `dbmb`.
    Dbmb,
    /// `dbkij`.
    Dbkij,
    /// `dbki`.
    Dbki,
    /// `dbkj`.
    Dbkj,
    /// `dbx`.
    Dbx,
    /// `dby`.
    Dby,
    /// `btin` … `bty` — the BT loops.
    Btin,
    /// `btout`.
    Btout,
    /// `btij`.
    Btij,
    /// `bti`.
    Bti,
    /// `btj`.
    Btj,
    /// `btmb`.
    Btmb,
    /// `btkij`.
    Btkij,
    /// `btx`.
    Btx,
    /// `bty`.
    Bty,
    /// `tpin` … `tpy` — the TP loops.
    Tpin,
    /// `tpout`.
    Tpout,
    /// `tpij`.
    Tpij,
    /// `tpi`.
    Tpi,
    /// `tpj`.
    Tpj,
    /// `tpmb`.
    Tpmb,
    /// `tpkij`.
    Tpkij,
    /// `tpx`.
    Tpx,
    /// `tpy`.
    Tpy,
    /// `const` — the const-offset spl loop.
    Const,
    /// `invalid`.
    Invalid,
}

impl WireLoopName {
    /// `DesignSpaceConfig::loopNameToString` (`dsc/designSpaceConfig.cpp:9222-9238`).
    #[must_use]
    pub const fn spelling(self) -> &'static str {
        match self {
            Self::Inner => "Inner",
            Self::Dbin => "dbin",
            Self::Dbout => "dbout",
            Self::Dbij => "dbij",
            Self::Dbi => "dbi",
            Self::Dbj => "dbj",
            Self::Dbmb => "dbmb",
            Self::Dbkij => "dbkij",
            Self::Dbki => "dbki",
            Self::Dbkj => "dbkj",
            Self::Dbx => "dbx",
            Self::Dby => "dby",
            Self::Btin => "btin",
            Self::Btout => "btout",
            Self::Btij => "btij",
            Self::Bti => "bti",
            Self::Btj => "btj",
            Self::Btmb => "btmb",
            Self::Btkij => "btkij",
            Self::Btx => "btx",
            Self::Bty => "bty",
            Self::Tpin => "tpin",
            Self::Tpout => "tpout",
            Self::Tpij => "tpij",
            Self::Tpi => "tpi",
            Self::Tpj => "tpj",
            Self::Tpmb => "tpmb",
            Self::Tpkij => "tpkij",
            Self::Tpx => "tpx",
            Self::Tpy => "tpy",
            Self::Const => "const",
            Self::Invalid => "invalid",
        }
    }

    /// `DesignSpaceConfig::stringToLoopName` (`dsc/designSpaceConfig.cpp:9239-9241`) — the parse
    /// boundary.
    #[must_use]
    pub fn from_spelling(text: &str) -> Option<Self> {
        Some(match text {
            "Inner" => Self::Inner,
            "dbin" => Self::Dbin,
            "dbout" => Self::Dbout,
            "dbij" => Self::Dbij,
            "dbi" => Self::Dbi,
            "dbj" => Self::Dbj,
            "dbmb" => Self::Dbmb,
            "dbkij" => Self::Dbkij,
            "dbki" => Self::Dbki,
            "dbkj" => Self::Dbkj,
            "dbx" => Self::Dbx,
            "dby" => Self::Dby,
            "btin" => Self::Btin,
            "btout" => Self::Btout,
            "btij" => Self::Btij,
            "bti" => Self::Bti,
            "btj" => Self::Btj,
            "btmb" => Self::Btmb,
            "btkij" => Self::Btkij,
            "btx" => Self::Btx,
            "bty" => Self::Bty,
            "tpin" => Self::Tpin,
            "tpout" => Self::Tpout,
            "tpij" => Self::Tpij,
            "tpi" => Self::Tpi,
            "tpj" => Self::Tpj,
            "tpmb" => Self::Tpmb,
            "tpkij" => Self::Tpkij,
            "tpx" => Self::Tpx,
            "tpy" => Self::Tpy,
            "const" => Self::Const,
            "invalid" => Self::Invalid,
            _ => return None,
        })
    }
}
/// A `computeOp_` ENTRY'S FUNCTION — `EnumsConversion::stringToOpFuncs`
/// (`sys-arch-spec/arch_enums.cpp:501-502`, the flip of `opFuncsToString` at `:322-499`),
/// the `opFuncName` of a `computeOp_` entry. All 176 entries, because the importer looks the
/// spelling up in the table and `DT_CHECK`s a miss (`dsc/designSpaceConfig.cpp:7315-7322`),
/// so every spelling the C++ accepts must parse here too.
///
/// ⚠️ NOT THE GENERATED [`crate::generated::OpFunc`]: that is the sealed set *scratchy* can
/// emit (censused from the `.ddl` templates), while this is the vocabulary the *wire*
/// carries. The two overlap but are not equal — the same distinction
/// [`WireComputeType`](self::WireComputeType) makes against the generated `ComputeType`.
///
/// ⚠️ THE DUMPER HAS A FALLBACK SPELLING. Where `opFuncsToString` has no entry for the op
/// func, the dumper writes `"SenPreparedOp"` (`designSpaceConfig.cpp:6659-6662`); that
/// spelling parses to [`Self::SenPreparedOp`], which no `opFuncsToString` entry produces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WireOpFunc {
    /// `none`.
    None,
    /// `abs`.
    Abs,
    /// `absmaxnonstick`.
    AbsmaxNonstick,
    /// `absmax`.
    Absmax,
    /// `add`.
    Add,
    /// `allgather`.
    Allgather,
    /// `allreduce`.
    Allreduce,
    /// `allshuffle`.
    Allshuffle,
    /// `APEOpHBM`.
    Apeophbm,
    /// `APEOpLX`.
    Apeoplx,
    /// `avgpoolfwd`.
    AvgpoolFwd,
    /// `avgpoolnmapfwd`.
    AvgpoolNmapFwd,
    /// `batchmatmulfp8mb`.
    BatchmatmulFp8FwdMb,
    /// `batchmatmulfp8sparsekg3`.
    BatchmatmulFp8FwdSparsekg3,
    /// `batchmatmulfp8`.
    BatchmatmulFp8Fwd,
    /// `batchmatmulsparsekg3`.
    BatchmatmulFwdSparsekg3,
    /// `batchmatmul`.
    BatchmatmulFwd,
    /// `batchmatmulmxfp4w`.
    BatchmatmulMxfp4wFwd,
    /// `batchmatmulint4sparsekg3`.
    BatchmatmulInt4FwdSparsekg3,
    /// `batchmatmulint4`.
    BatchmatmulInt4Fwd,
    /// `batchmatmulint8mbkg3`.
    BatchmatmulInt8FwdMbkg3,
    /// `batchmatmulint8sparsekg3`.
    BatchmatmulInt8FwdSparsekg3,
    /// `batchmatmulint8`.
    BatchmatmulInt8Fwd,
    /// `batchmatmulmxfp8`.
    BatchmatmulMxfp8Fwd,
    /// `batchmatmulxrffp8`.
    BatchmatmulXrfFp8Fwd,
    /// `batchmatmulxrf`.
    BatchmatmulXrfFwd,
    /// `batchmatmulxrfint4`.
    BatchmatmulXrfInt4Fwd,
    /// `batchmatmulxrfint8`.
    BatchmatmulXrfInt8Fwd,
    /// `batchmatmulxrfchfp8`.
    BatchmatmulXrfchFp8Fwd,
    /// `batchmatmulxrfch`.
    BatchmatmulXrfchFwd,
    /// `batchmatmulxrfchint4`.
    BatchmatmulXrfchInt4Fwd,
    /// `batchmatmulxrfchint8`.
    BatchmatmulXrfchInt8Fwd,
    /// `batchmatmulv2`.
    Batchmatmulv2,
    /// `batchnormfwd`.
    BatchnormFwd,
    /// `biasadd`.
    Biasadd,
    /// `bnpreczeroshft`.
    Bnpreczeroshft,
    /// `clip`.
    ClipFwd,
    /// `ConstPadOpHBM`.
    Constpadophbm,
    /// `ConstPadOpLX`.
    Constpadoplx,
    /// `conv2dfp8genkg3`.
    Conv2dFp8FwdGenkg3,
    /// `conv2dfp8sparsekg3`.
    Conv2dFp8FwdSparsekg3,
    /// `conv2dfp8`.
    Conv2dFp8Fwd,
    /// `conv2dgenos1`.
    Conv2dFwdGenOs1,
    /// `conv2dgenkg3`.
    Conv2dFwdGenkg3,
    /// `conv2dos1`.
    Conv2dFwdOs1,
    /// `conv2dsparsekg3`.
    Conv2dFwdSparsekg3,
    /// `conv2d`.
    Conv2dFwd,
    /// `conv2dint4genkg3`.
    Conv2dInt4FwdGenkg3,
    /// `conv2dint4sparsekg3`.
    Conv2dInt4FwdSparsekg3,
    /// `conv2dint4`.
    Conv2dInt4Fwd,
    /// `conv2dint8genkg3`.
    Conv2dInt8FwdGenkg3,
    /// `conv2dint8os1`.
    Conv2dInt8FwdOs1,
    /// `conv2dint8sparsekg3`.
    Conv2dInt8FwdSparsekg3,
    /// `conv2dint8`.
    Conv2dInt8Fwd,
    /// `conv2dxrfint8os1`.
    Conv2dXrfInt8FwdOs1,
    /// `csqint4chil`.
    CsqInt4Chil,
    /// `csqint4wt`.
    CsqInt4Wt,
    /// `csqint4`.
    CsqInt4,
    /// `csqint8ch`.
    CsqInt8Ch,
    /// `csqint8chil`.
    CsqInt8Chil,
    /// `csqint8mbv2`.
    CsqInt8MbV2,
    /// `csqint8mb`.
    CsqInt8Mb,
    /// `csqint8v2`.
    CsqInt8V2,
    /// `csqint8wt`.
    CsqInt8Wt,
    /// `csqint8`.
    CsqInt8,
    /// `double_buffer_mni`.
    DblBufMni,
    /// `depthwiseconv2dnative`.
    DepthwiseConvFwd,
    /// `dl16tobf16`.
    Dl16tobf16,
    /// `dl16tofp32`.
    Dl16tofp32,
    /// `equal`.
    Equal,
    /// `erf`.
    ErfFwd,
    /// `exp`.
    ExpFwd,
    /// `exx2_zeromean`.
    Exx2Zeromean,
    /// `exx2`.
    Exx2,
    /// `fastexp`.
    FastExpFwd,
    /// `fastsigmoid`.
    FastSigmoidFwd,
    /// `floor`.
    Floor,
    /// `fnms`.
    Fnms,
    /// `fp32todl16`.
    Fp32todl16,
    /// `fp8todl16`.
    Fp8todl16,
    /// `GatherOpHBM`.
    Gatherophbm,
    /// `gelubackward`.
    GeluBwd,
    /// `gelufwd`.
    GeluFwd,
    /// `genericpartialreduction`.
    GenericPartialReduction,
    /// `greaterequal`.
    Greaterequal,
    /// `greaterthan`.
    Greaterthan,
    /// `identity`.
    Identity,
    /// `int32idxtoaddr`.
    Int32idxtoaddr,
    /// `shuffle`.
    Shuffle,
    /// `ITOF`.
    Itof,
    /// `ITOFHBM`.
    Itofhbm,
    /// `layernormbackwardnorm`.
    LayernormBwdnorm,
    /// `layernormnorm`.
    LayernormNorm,
    /// `layernormscale`.
    LayernormScale,
    /// `leakyrelufwd`.
    LeakyreluFwd,
    /// `lesserequal`.
    Lesserequal,
    /// `lesserthan`.
    Lesserthan,
    /// `log`.
    LogFwd,
    /// `lstmactp1`.
    Lstmactp1Fwd,
    /// `lstmactp2`.
    Lstmactp2Fwd,
    /// `lstmblockcell`.
    Lstmblockcell,
    /// `mask2bit`.
    Mask2bit,
    /// `maskbyindex`.
    MaskByIndex,
    /// `matmulfp8`.
    MatmulFp8Fwd,
    /// `matmul`.
    MatmulFwd,
    /// `matmulint4`.
    MatmulInt4Fwd,
    /// `matmulint8`.
    MatmulInt8Fwd,
    /// `maxnonstick`.
    MaxNonstick,
    /// `max`.
    Max,
    /// `maximum`.
    Maximum,
    /// `maxpoolfwd`.
    MaxpoolFwd,
    /// `meannonstick`.
    MeanNonstick,
    /// `mean`.
    Mean,
    /// `minnonstick`.
    MinNonstick,
    /// `min`.
    Min,
    /// `minimum`.
    Minimum,
    /// `mish`.
    MishFwd,
    /// `mul`.
    Mul,
    /// `neg`.
    Neg,
    /// `nop`.
    Nop,
    /// `notequal`.
    Notequal,
    /// `PESFP_Collate_2B_Writes`.
    PesfpCollate2bWrites,
    /// `prodnonstick`.
    ProdNonstick,
    /// `pt_blk_transpose_load`.
    PtBlkTransposeLoad,
    /// `qfp8ch`.
    QFp8Ch,
    /// `qfp8chil`.
    QFp8Chil,
    /// `qfp8mb`.
    QFp8Mb,
    /// `qfp8wt`.
    QFp8Wt,
    /// `qfp8`.
    QFp8,
    /// `quantscalepertokenfp8`.
    QuantScalePerTokenFp8,
    /// `quantscalepertoken`.
    QuantScalePerToken,
    /// `realdiv`.
    Realdiv,
    /// `reciprocal`.
    Reciprocal,
    /// `relufwd`.
    ReluFwd,
    /// `relu6fwd`.
    Relu6Fwd,
    /// `ResizeNNHBM`.
    Resizennhbm,
    /// `ResizeNNLX`.
    Resizennlx,
    /// `ReStickifyOpHBM`.
    Restickifyophbm,
    /// `ReStickifyOpLx`.
    Restickifyoplx,
    /// `ReStickifyOpWithPTHBM`.
    Restickifyopwithpthbm,
    /// `ReStickifyOpWithPTLx`.
    Restickifyopwithptlx,
    /// `revsub`.
    Revsub,
    /// `rope64p1`.
    Rope64p1Fwd,
    /// `rope64p2`.
    Rope64p2Fwd,
    /// `rsqrt`.
    Rsqrt,
    /// `scaledgroupmatmulfp4`.
    ScaledGroupMatmulFp4Fwd,
    /// `ScatterOpHBM`.
    Scatterophbm,
    /// `sfp_readlx_transpose_fwdl0`.
    SfpReadlxTransposeFwdl0,
    /// `sigmoid`.
    SigmoidFwd,
    /// `silu`.
    SiluFwd,
    /// `sinkcorrectionfactor`.
    Sinkcorrectionfactor,
    /// `softmax`.
    Softmax,
    /// `softplus`.
    Softplus,
    /// `squareddifference`.
    SqdiffFwd,
    /// `sqrt`.
    SqrtFwd,
    /// `STCDPOpHBM`.
    Stcdpophbm,
    /// `STCDPOpLx`.
    Stcdpoplx,
    /// `StickifyOpHBM`.
    Stickifyophbm,
    /// `stridedadd`.
    StridedAdd,
    /// `StzLatch`.
    StzLatch,
    /// `sub`.
    Sub,
    /// `sumnonstick`.
    SumNonstick,
    /// `sum`.
    Sum,
    /// `tanhbackward`.
    TanhBwd,
    /// `tanh`.
    TanhFwd,
    /// `topkindex`.
    TopkIndex,
    /// `topkvalue`.
    TopkValue,
    /// `undef`.
    Undef,
    /// `where3`.
    Where3,
    /// `XRFWriteHBM`.
    Xrfwritehbm,
    /// `XRFWriteLX`.
    Xrfwritelx,
    /// `interslicetranspose_fp16`.
    InterslicetransposeFp16,
    /// `interslicetranspose_fp8`.
    InterslicetransposeFp8,
    /// `addi32toi32`.
    AddI32ToI32,
    /// `addi64toi64`.
    AddI64ToI64,
    /// `muli32toi32`.
    MulI32ToI32,
    /// `SenPreparedOp` — the dumper's fallback for an op func `opFuncsToString` does not
    /// carry (`designSpaceConfig.cpp:6659-6662`).
    SenPreparedOp,
}

impl WireOpFunc {
    /// `EnumsConversion::opFuncsToString` (`sys-arch-spec/arch_enums.cpp:322-499`) — one arm
    /// per entry, plus the dumper's fallback spelling.
    #[must_use]
    pub const fn spelling(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Abs => "abs",
            Self::AbsmaxNonstick => "absmaxnonstick",
            Self::Absmax => "absmax",
            Self::Add => "add",
            Self::Allgather => "allgather",
            Self::Allreduce => "allreduce",
            Self::Allshuffle => "allshuffle",
            Self::Apeophbm => "APEOpHBM",
            Self::Apeoplx => "APEOpLX",
            Self::AvgpoolFwd => "avgpoolfwd",
            Self::AvgpoolNmapFwd => "avgpoolnmapfwd",
            Self::BatchmatmulFp8FwdMb => "batchmatmulfp8mb",
            Self::BatchmatmulFp8FwdSparsekg3 => "batchmatmulfp8sparsekg3",
            Self::BatchmatmulFp8Fwd => "batchmatmulfp8",
            Self::BatchmatmulFwdSparsekg3 => "batchmatmulsparsekg3",
            Self::BatchmatmulFwd => "batchmatmul",
            Self::BatchmatmulMxfp4wFwd => "batchmatmulmxfp4w",
            Self::BatchmatmulInt4FwdSparsekg3 => "batchmatmulint4sparsekg3",
            Self::BatchmatmulInt4Fwd => "batchmatmulint4",
            Self::BatchmatmulInt8FwdMbkg3 => "batchmatmulint8mbkg3",
            Self::BatchmatmulInt8FwdSparsekg3 => "batchmatmulint8sparsekg3",
            Self::BatchmatmulInt8Fwd => "batchmatmulint8",
            Self::BatchmatmulMxfp8Fwd => "batchmatmulmxfp8",
            Self::BatchmatmulXrfFp8Fwd => "batchmatmulxrffp8",
            Self::BatchmatmulXrfFwd => "batchmatmulxrf",
            Self::BatchmatmulXrfInt4Fwd => "batchmatmulxrfint4",
            Self::BatchmatmulXrfInt8Fwd => "batchmatmulxrfint8",
            Self::BatchmatmulXrfchFp8Fwd => "batchmatmulxrfchfp8",
            Self::BatchmatmulXrfchFwd => "batchmatmulxrfch",
            Self::BatchmatmulXrfchInt4Fwd => "batchmatmulxrfchint4",
            Self::BatchmatmulXrfchInt8Fwd => "batchmatmulxrfchint8",
            Self::Batchmatmulv2 => "batchmatmulv2",
            Self::BatchnormFwd => "batchnormfwd",
            Self::Biasadd => "biasadd",
            Self::Bnpreczeroshft => "bnpreczeroshft",
            Self::ClipFwd => "clip",
            Self::Constpadophbm => "ConstPadOpHBM",
            Self::Constpadoplx => "ConstPadOpLX",
            Self::Conv2dFp8FwdGenkg3 => "conv2dfp8genkg3",
            Self::Conv2dFp8FwdSparsekg3 => "conv2dfp8sparsekg3",
            Self::Conv2dFp8Fwd => "conv2dfp8",
            Self::Conv2dFwdGenOs1 => "conv2dgenos1",
            Self::Conv2dFwdGenkg3 => "conv2dgenkg3",
            Self::Conv2dFwdOs1 => "conv2dos1",
            Self::Conv2dFwdSparsekg3 => "conv2dsparsekg3",
            Self::Conv2dFwd => "conv2d",
            Self::Conv2dInt4FwdGenkg3 => "conv2dint4genkg3",
            Self::Conv2dInt4FwdSparsekg3 => "conv2dint4sparsekg3",
            Self::Conv2dInt4Fwd => "conv2dint4",
            Self::Conv2dInt8FwdGenkg3 => "conv2dint8genkg3",
            Self::Conv2dInt8FwdOs1 => "conv2dint8os1",
            Self::Conv2dInt8FwdSparsekg3 => "conv2dint8sparsekg3",
            Self::Conv2dInt8Fwd => "conv2dint8",
            Self::Conv2dXrfInt8FwdOs1 => "conv2dxrfint8os1",
            Self::CsqInt4Chil => "csqint4chil",
            Self::CsqInt4Wt => "csqint4wt",
            Self::CsqInt4 => "csqint4",
            Self::CsqInt8Ch => "csqint8ch",
            Self::CsqInt8Chil => "csqint8chil",
            Self::CsqInt8MbV2 => "csqint8mbv2",
            Self::CsqInt8Mb => "csqint8mb",
            Self::CsqInt8V2 => "csqint8v2",
            Self::CsqInt8Wt => "csqint8wt",
            Self::CsqInt8 => "csqint8",
            Self::DblBufMni => "double_buffer_mni",
            Self::DepthwiseConvFwd => "depthwiseconv2dnative",
            Self::Dl16tobf16 => "dl16tobf16",
            Self::Dl16tofp32 => "dl16tofp32",
            Self::Equal => "equal",
            Self::ErfFwd => "erf",
            Self::ExpFwd => "exp",
            Self::Exx2Zeromean => "exx2_zeromean",
            Self::Exx2 => "exx2",
            Self::FastExpFwd => "fastexp",
            Self::FastSigmoidFwd => "fastsigmoid",
            Self::Floor => "floor",
            Self::Fnms => "fnms",
            Self::Fp32todl16 => "fp32todl16",
            Self::Fp8todl16 => "fp8todl16",
            Self::Gatherophbm => "GatherOpHBM",
            Self::GeluBwd => "gelubackward",
            Self::GeluFwd => "gelufwd",
            Self::GenericPartialReduction => "genericpartialreduction",
            Self::Greaterequal => "greaterequal",
            Self::Greaterthan => "greaterthan",
            Self::Identity => "identity",
            Self::Int32idxtoaddr => "int32idxtoaddr",
            Self::Shuffle => "shuffle",
            Self::Itof => "ITOF",
            Self::Itofhbm => "ITOFHBM",
            Self::LayernormBwdnorm => "layernormbackwardnorm",
            Self::LayernormNorm => "layernormnorm",
            Self::LayernormScale => "layernormscale",
            Self::LeakyreluFwd => "leakyrelufwd",
            Self::Lesserequal => "lesserequal",
            Self::Lesserthan => "lesserthan",
            Self::LogFwd => "log",
            Self::Lstmactp1Fwd => "lstmactp1",
            Self::Lstmactp2Fwd => "lstmactp2",
            Self::Lstmblockcell => "lstmblockcell",
            Self::Mask2bit => "mask2bit",
            Self::MaskByIndex => "maskbyindex",
            Self::MatmulFp8Fwd => "matmulfp8",
            Self::MatmulFwd => "matmul",
            Self::MatmulInt4Fwd => "matmulint4",
            Self::MatmulInt8Fwd => "matmulint8",
            Self::MaxNonstick => "maxnonstick",
            Self::Max => "max",
            Self::Maximum => "maximum",
            Self::MaxpoolFwd => "maxpoolfwd",
            Self::MeanNonstick => "meannonstick",
            Self::Mean => "mean",
            Self::MinNonstick => "minnonstick",
            Self::Min => "min",
            Self::Minimum => "minimum",
            Self::MishFwd => "mish",
            Self::Mul => "mul",
            Self::Neg => "neg",
            Self::Nop => "nop",
            Self::Notequal => "notequal",
            Self::PesfpCollate2bWrites => "PESFP_Collate_2B_Writes",
            Self::ProdNonstick => "prodnonstick",
            Self::PtBlkTransposeLoad => "pt_blk_transpose_load",
            Self::QFp8Ch => "qfp8ch",
            Self::QFp8Chil => "qfp8chil",
            Self::QFp8Mb => "qfp8mb",
            Self::QFp8Wt => "qfp8wt",
            Self::QFp8 => "qfp8",
            Self::QuantScalePerTokenFp8 => "quantscalepertokenfp8",
            Self::QuantScalePerToken => "quantscalepertoken",
            Self::Realdiv => "realdiv",
            Self::Reciprocal => "reciprocal",
            Self::ReluFwd => "relufwd",
            Self::Relu6Fwd => "relu6fwd",
            Self::Resizennhbm => "ResizeNNHBM",
            Self::Resizennlx => "ResizeNNLX",
            Self::Restickifyophbm => "ReStickifyOpHBM",
            Self::Restickifyoplx => "ReStickifyOpLx",
            Self::Restickifyopwithpthbm => "ReStickifyOpWithPTHBM",
            Self::Restickifyopwithptlx => "ReStickifyOpWithPTLx",
            Self::Revsub => "revsub",
            Self::Rope64p1Fwd => "rope64p1",
            Self::Rope64p2Fwd => "rope64p2",
            Self::Rsqrt => "rsqrt",
            Self::ScaledGroupMatmulFp4Fwd => "scaledgroupmatmulfp4",
            Self::Scatterophbm => "ScatterOpHBM",
            Self::SfpReadlxTransposeFwdl0 => "sfp_readlx_transpose_fwdl0",
            Self::SigmoidFwd => "sigmoid",
            Self::SiluFwd => "silu",
            Self::Sinkcorrectionfactor => "sinkcorrectionfactor",
            Self::Softmax => "softmax",
            Self::Softplus => "softplus",
            Self::SqdiffFwd => "squareddifference",
            Self::SqrtFwd => "sqrt",
            Self::Stcdpophbm => "STCDPOpHBM",
            Self::Stcdpoplx => "STCDPOpLx",
            Self::Stickifyophbm => "StickifyOpHBM",
            Self::StridedAdd => "stridedadd",
            Self::StzLatch => "StzLatch",
            Self::Sub => "sub",
            Self::SumNonstick => "sumnonstick",
            Self::Sum => "sum",
            Self::TanhBwd => "tanhbackward",
            Self::TanhFwd => "tanh",
            Self::TopkIndex => "topkindex",
            Self::TopkValue => "topkvalue",
            Self::Undef => "undef",
            Self::Where3 => "where3",
            Self::Xrfwritehbm => "XRFWriteHBM",
            Self::Xrfwritelx => "XRFWriteLX",
            Self::InterslicetransposeFp16 => "interslicetranspose_fp16",
            Self::InterslicetransposeFp8 => "interslicetranspose_fp8",
            Self::AddI32ToI32 => "addi32toi32",
            Self::AddI64ToI64 => "addi64toi64",
            Self::MulI32ToI32 => "muli32toi32",
            Self::SenPreparedOp => "SenPreparedOp",
        }
    }

    /// `EnumsConversion::stringToOpFuncs` (`sys-arch-spec/arch_enums.cpp:501-502`) — the parse
    /// boundary. `"SenPreparedOp"` is accepted because the dumper can write it; an unknown
    /// spelling is [`None`], which the reader turns into the refusal the importer's
    /// `DT_CHECK` raises.
    #[must_use]
    pub fn from_spelling(text: &str) -> Option<Self> {
        Some(match text {
            "none" => Self::None,
            "abs" => Self::Abs,
            "absmaxnonstick" => Self::AbsmaxNonstick,
            "absmax" => Self::Absmax,
            "add" => Self::Add,
            "allgather" => Self::Allgather,
            "allreduce" => Self::Allreduce,
            "allshuffle" => Self::Allshuffle,
            "APEOpHBM" => Self::Apeophbm,
            "APEOpLX" => Self::Apeoplx,
            "avgpoolfwd" => Self::AvgpoolFwd,
            "avgpoolnmapfwd" => Self::AvgpoolNmapFwd,
            "batchmatmulfp8mb" => Self::BatchmatmulFp8FwdMb,
            "batchmatmulfp8sparsekg3" => Self::BatchmatmulFp8FwdSparsekg3,
            "batchmatmulfp8" => Self::BatchmatmulFp8Fwd,
            "batchmatmulsparsekg3" => Self::BatchmatmulFwdSparsekg3,
            "batchmatmul" => Self::BatchmatmulFwd,
            "batchmatmulmxfp4w" => Self::BatchmatmulMxfp4wFwd,
            "batchmatmulint4sparsekg3" => Self::BatchmatmulInt4FwdSparsekg3,
            "batchmatmulint4" => Self::BatchmatmulInt4Fwd,
            "batchmatmulint8mbkg3" => Self::BatchmatmulInt8FwdMbkg3,
            "batchmatmulint8sparsekg3" => Self::BatchmatmulInt8FwdSparsekg3,
            "batchmatmulint8" => Self::BatchmatmulInt8Fwd,
            "batchmatmulmxfp8" => Self::BatchmatmulMxfp8Fwd,
            "batchmatmulxrffp8" => Self::BatchmatmulXrfFp8Fwd,
            "batchmatmulxrf" => Self::BatchmatmulXrfFwd,
            "batchmatmulxrfint4" => Self::BatchmatmulXrfInt4Fwd,
            "batchmatmulxrfint8" => Self::BatchmatmulXrfInt8Fwd,
            "batchmatmulxrfchfp8" => Self::BatchmatmulXrfchFp8Fwd,
            "batchmatmulxrfch" => Self::BatchmatmulXrfchFwd,
            "batchmatmulxrfchint4" => Self::BatchmatmulXrfchInt4Fwd,
            "batchmatmulxrfchint8" => Self::BatchmatmulXrfchInt8Fwd,
            "batchmatmulv2" => Self::Batchmatmulv2,
            "batchnormfwd" => Self::BatchnormFwd,
            "biasadd" => Self::Biasadd,
            "bnpreczeroshft" => Self::Bnpreczeroshft,
            "clip" => Self::ClipFwd,
            "ConstPadOpHBM" => Self::Constpadophbm,
            "ConstPadOpLX" => Self::Constpadoplx,
            "conv2dfp8genkg3" => Self::Conv2dFp8FwdGenkg3,
            "conv2dfp8sparsekg3" => Self::Conv2dFp8FwdSparsekg3,
            "conv2dfp8" => Self::Conv2dFp8Fwd,
            "conv2dgenos1" => Self::Conv2dFwdGenOs1,
            "conv2dgenkg3" => Self::Conv2dFwdGenkg3,
            "conv2dos1" => Self::Conv2dFwdOs1,
            "conv2dsparsekg3" => Self::Conv2dFwdSparsekg3,
            "conv2d" => Self::Conv2dFwd,
            "conv2dint4genkg3" => Self::Conv2dInt4FwdGenkg3,
            "conv2dint4sparsekg3" => Self::Conv2dInt4FwdSparsekg3,
            "conv2dint4" => Self::Conv2dInt4Fwd,
            "conv2dint8genkg3" => Self::Conv2dInt8FwdGenkg3,
            "conv2dint8os1" => Self::Conv2dInt8FwdOs1,
            "conv2dint8sparsekg3" => Self::Conv2dInt8FwdSparsekg3,
            "conv2dint8" => Self::Conv2dInt8Fwd,
            "conv2dxrfint8os1" => Self::Conv2dXrfInt8FwdOs1,
            "csqint4chil" => Self::CsqInt4Chil,
            "csqint4wt" => Self::CsqInt4Wt,
            "csqint4" => Self::CsqInt4,
            "csqint8ch" => Self::CsqInt8Ch,
            "csqint8chil" => Self::CsqInt8Chil,
            "csqint8mbv2" => Self::CsqInt8MbV2,
            "csqint8mb" => Self::CsqInt8Mb,
            "csqint8v2" => Self::CsqInt8V2,
            "csqint8wt" => Self::CsqInt8Wt,
            "csqint8" => Self::CsqInt8,
            "double_buffer_mni" => Self::DblBufMni,
            "depthwiseconv2dnative" => Self::DepthwiseConvFwd,
            "dl16tobf16" => Self::Dl16tobf16,
            "dl16tofp32" => Self::Dl16tofp32,
            "equal" => Self::Equal,
            "erf" => Self::ErfFwd,
            "exp" => Self::ExpFwd,
            "exx2_zeromean" => Self::Exx2Zeromean,
            "exx2" => Self::Exx2,
            "fastexp" => Self::FastExpFwd,
            "fastsigmoid" => Self::FastSigmoidFwd,
            "floor" => Self::Floor,
            "fnms" => Self::Fnms,
            "fp32todl16" => Self::Fp32todl16,
            "fp8todl16" => Self::Fp8todl16,
            "GatherOpHBM" => Self::Gatherophbm,
            "gelubackward" => Self::GeluBwd,
            "gelufwd" => Self::GeluFwd,
            "genericpartialreduction" => Self::GenericPartialReduction,
            "greaterequal" => Self::Greaterequal,
            "greaterthan" => Self::Greaterthan,
            "identity" => Self::Identity,
            "int32idxtoaddr" => Self::Int32idxtoaddr,
            "shuffle" => Self::Shuffle,
            "ITOF" => Self::Itof,
            "ITOFHBM" => Self::Itofhbm,
            "layernormbackwardnorm" => Self::LayernormBwdnorm,
            "layernormnorm" => Self::LayernormNorm,
            "layernormscale" => Self::LayernormScale,
            "leakyrelufwd" => Self::LeakyreluFwd,
            "lesserequal" => Self::Lesserequal,
            "lesserthan" => Self::Lesserthan,
            "log" => Self::LogFwd,
            "lstmactp1" => Self::Lstmactp1Fwd,
            "lstmactp2" => Self::Lstmactp2Fwd,
            "lstmblockcell" => Self::Lstmblockcell,
            "mask2bit" => Self::Mask2bit,
            "maskbyindex" => Self::MaskByIndex,
            "matmulfp8" => Self::MatmulFp8Fwd,
            "matmul" => Self::MatmulFwd,
            "matmulint4" => Self::MatmulInt4Fwd,
            "matmulint8" => Self::MatmulInt8Fwd,
            "maxnonstick" => Self::MaxNonstick,
            "max" => Self::Max,
            "maximum" => Self::Maximum,
            "maxpoolfwd" => Self::MaxpoolFwd,
            "meannonstick" => Self::MeanNonstick,
            "mean" => Self::Mean,
            "minnonstick" => Self::MinNonstick,
            "min" => Self::Min,
            "minimum" => Self::Minimum,
            "mish" => Self::MishFwd,
            "mul" => Self::Mul,
            "neg" => Self::Neg,
            "nop" => Self::Nop,
            "notequal" => Self::Notequal,
            "PESFP_Collate_2B_Writes" => Self::PesfpCollate2bWrites,
            "prodnonstick" => Self::ProdNonstick,
            "pt_blk_transpose_load" => Self::PtBlkTransposeLoad,
            "qfp8ch" => Self::QFp8Ch,
            "qfp8chil" => Self::QFp8Chil,
            "qfp8mb" => Self::QFp8Mb,
            "qfp8wt" => Self::QFp8Wt,
            "qfp8" => Self::QFp8,
            "quantscalepertokenfp8" => Self::QuantScalePerTokenFp8,
            "quantscalepertoken" => Self::QuantScalePerToken,
            "realdiv" => Self::Realdiv,
            "reciprocal" => Self::Reciprocal,
            "relufwd" => Self::ReluFwd,
            "relu6fwd" => Self::Relu6Fwd,
            "ResizeNNHBM" => Self::Resizennhbm,
            "ResizeNNLX" => Self::Resizennlx,
            "ReStickifyOpHBM" => Self::Restickifyophbm,
            "ReStickifyOpLx" => Self::Restickifyoplx,
            "ReStickifyOpWithPTHBM" => Self::Restickifyopwithpthbm,
            "ReStickifyOpWithPTLx" => Self::Restickifyopwithptlx,
            "revsub" => Self::Revsub,
            "rope64p1" => Self::Rope64p1Fwd,
            "rope64p2" => Self::Rope64p2Fwd,
            "rsqrt" => Self::Rsqrt,
            "scaledgroupmatmulfp4" => Self::ScaledGroupMatmulFp4Fwd,
            "ScatterOpHBM" => Self::Scatterophbm,
            "sfp_readlx_transpose_fwdl0" => Self::SfpReadlxTransposeFwdl0,
            "sigmoid" => Self::SigmoidFwd,
            "silu" => Self::SiluFwd,
            "sinkcorrectionfactor" => Self::Sinkcorrectionfactor,
            "softmax" => Self::Softmax,
            "softplus" => Self::Softplus,
            "squareddifference" => Self::SqdiffFwd,
            "sqrt" => Self::SqrtFwd,
            "STCDPOpHBM" => Self::Stcdpophbm,
            "STCDPOpLx" => Self::Stcdpoplx,
            "StickifyOpHBM" => Self::Stickifyophbm,
            "stridedadd" => Self::StridedAdd,
            "StzLatch" => Self::StzLatch,
            "sub" => Self::Sub,
            "sumnonstick" => Self::SumNonstick,
            "sum" => Self::Sum,
            "tanhbackward" => Self::TanhBwd,
            "tanh" => Self::TanhFwd,
            "topkindex" => Self::TopkIndex,
            "topkvalue" => Self::TopkValue,
            "undef" => Self::Undef,
            "where3" => Self::Where3,
            "XRFWriteHBM" => Self::Xrfwritehbm,
            "XRFWriteLX" => Self::Xrfwritelx,
            "interslicetranspose_fp16" => Self::InterslicetransposeFp16,
            "interslicetranspose_fp8" => Self::InterslicetransposeFp8,
            "addi32toi32" => Self::AddI32ToI32,
            "addi64toi64" => Self::AddI64ToI64,
            "muli32toi32" => Self::MulI32ToI32,
            "SenPreparedOp" => Self::SenPreparedOp,
            _ => return None,
        })
    }
}
