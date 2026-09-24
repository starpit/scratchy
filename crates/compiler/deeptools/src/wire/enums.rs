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
