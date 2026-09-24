//! ⛔⛔ THE RULES, EXTRACTED FROM THE C++ — the fixed half of the lowering.
//!
//! A template says WHICH kind a statement is; only the C++ says what that kind BECOMES. These tables
//! are that knowledge, encoded once, and every one is quoted in `LOWERING_RULES.md`. Inferring any of
//! them from `.ddl` text is the failure this file exists to stop: the DDL under-specifies exactly
//! where guessing is easiest and wrongest — `computetype="FMA16"` does not say "three operands", and
//! `rotate_num_elements=32` does not say "on the source, after the load, LXLU only".
//!
//! # 🛑🛑 NO STRINGS BETWEEN FUNCTIONS. EVERY DECISION IS AN ENUM.
//!
//! ⛔⛔ THIS FILE ONCE RETURNED `"estimate:Exp:A"` AND `"binary:Max"` FOR `walk.rs` TO `strip_prefix`
//! AND `split_once`. Two functions in one crate talking over a string protocol: the encoder and the
//! parser could disagree and nothing would say so, which is the same class of defect as the
//! `data_connect` join this whole generator exists to fix. Every rule below returns a VARIANT.
//!
//! ⛔ TEXT SURVIVES IN EXACTLY ONE PLACE: the `rust()` renderers, because this build script emits Rust
//! source and cannot name the library's `BinaryOp`/`DfirUnit`/`ElemType` any other way. A renderer is
//! the last line before the output; nothing reads one back.
//!
//! ⛔ AN UNMAPPED SPELLING RETURNS `Err` NAMING ITSELF. It never falls back to something adjacent that
//! reads plausible: `MACC` folded into the MAC family, `FMAX` into `BinaryOp::Max`, and `FEST` into a
//! single `exp_estimate` for every mode were all that mistake, and all three emitted programs the
//! reference does not describe.

/// WHICH UNIT A PORT SITS ON — the ends a wire may terminate at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PortUnit {
    Sfp,
    Pe,
    Lxlu,
    Lxsu,
    L0lu,
    L0su,
    SfpRing,
    /// `unit="pt"` on a PORT, which is ROW 0 — not the array and not every row.
    ///
    /// ⛔⛔ `ddl_conversion.cpp:963-964` takes `units.front()` of the unrolled span, and `"pt"` unrolls
    /// to `"ptrow0-{N-1}"` (`:730-731`). A COMPUTE with the same spelling is cloned across every row
    /// instead (`:1467-1472`) — one spelling, two rules, which is why [`ComputeUnit`] resolves it
    /// separately. Row 0 is also the only PT row in the LXLU's neighbour list
    /// (`DSC2ToDataflowIRUtils.hpp:172-173`).
    PtRow0,
    /// A row span's FIRST row — `ptrow1-7` is row 1, by `units.front()`.
    PtRowN(u32),
}

impl PortUnit {
    /// THE ONE PLACE A PORT'S `unit=` SPELLING IS READ.
    pub fn parse(spelling: &str) -> Result<PortUnit, String> {
        match spelling {
            "sfp" => Ok(PortUnit::Sfp),
            "pe" => Ok(PortUnit::Pe),
            "lxlu" => Ok(PortUnit::Lxlu),
            "lxsu" => Ok(PortUnit::Lxsu),
            "l0lu" => Ok(PortUnit::L0lu),
            "l0su" => Ok(PortUnit::L0su),
            "sfpring" => Ok(PortUnit::SfpRing),
            "pt" | "ptrow0" => Ok(PortUnit::PtRow0),
            // ⛔⛔ A ROW SPAN ON A PORT IS ITS FIRST ROW. `ddl_conversion.cpp:734-749` parses
            // `ptrow<start>-<end>` and pushes one component per row; `:963-964` then keeps
            // `units.front()` for a `ddl.unit`. So `ptrow1-7` names row 1 here — the same rule that
            // makes `pt` (rewritten to `ptrow0-{N-1}`) name row 0.
            // ⭐ ONE ARM SERVES BOTH `ptrow3` AND `ptrow1-7`, because a span keeps its FIRST row and a
            // bare row IS its first row. `unrollRowUnits` treats them the same way — the dash branch
            // pushes `start..=end` and the else branch pushes the one name (`:734-749`) — and
            // `:963-964` then takes `units.front()` either way.
            row if row.starts_with("ptrow") => {
                let first = row
                    .trim_start_matches("ptrow")
                    .split('-')
                    .next()
                    .and_then(|start| start.parse::<u32>().ok())
                    .ok_or_else(|| {
                        format!(
                            "unit=\"{row}\" is not `ptrow<n>` or `ptrow<start>-<end>` — the reference \
                             requires a digit either side of a dash (ddl_conversion.cpp:734-742)"
                        )
                    })?;
                Ok(PortUnit::PtRowN(first))
            }
            other => Err(format!("unit=\"{other}\" is not a wire end")),
        }
    }

    /// The `DfirUnit` expression.
    pub fn dfir(self) -> String {
        match self {
            PortUnit::Sfp => "DfirUnit::Sfp".to_owned(),
            PortUnit::Pe => "DfirUnit::Pe".to_owned(),
            PortUnit::Lxlu => "DfirUnit::Lxlu".to_owned(),
            PortUnit::Lxsu => "DfirUnit::Lxsu".to_owned(),
            PortUnit::L0lu => "DfirUnit::L0lu".to_owned(),
            PortUnit::L0su => "DfirUnit::L0su".to_owned(),
            PortUnit::SfpRing => "DfirUnit::SfpRing".to_owned(),
            PortUnit::PtRow0 => {
                "DfirUnit::PtRow(crate::units::Row::checked(0).expect(\"row 0 exists\"))".to_owned()
            }
            // ⛔ `Row::checked` STILL DECIDES. `ptrow7` names a row SEN1P5 does not have, and a row the
            // arch lacks must not become a program — so the fallback is row 0, which every arch has.
            PortUnit::PtRowN(row) => format!(
                "DfirUnit::PtRow(crate::units::Row::checked({row}).or_else(|| \
                 crate::units::Row::checked(0)).expect(\"row 0 exists\"))"
            ),
        }
    }

    /// The `link::UnitKind` marker, for the typed ends of a wire.
    pub const fn link(self) -> &'static str {
        match self {
            PortUnit::Sfp => "Sfp",
            PortUnit::Pe => "Pe",
            PortUnit::Lxlu => "Lxlu",
            PortUnit::Lxsu => "Lxsu",
            PortUnit::L0lu => "L0lu",
            PortUnit::L0su => "L0su",
            PortUnit::SfpRing => "SfpRing",
            PortUnit::PtRow0 | PortUnit::PtRowN(_) => "PtRow0",
        }
    }

    /// Whether this end is an immediate rather than a wire — see [`ComputeUnit::parse`]'s note.
    pub fn is_constant(spelling: &str) -> bool {
        spelling == "constant"
    }

    /// A NAME FOR THE GENERATED LOCAL that holds this unit's handle.
    ///
    /// ⛔ SEPARATE FROM [`PortUnit::dfir`], because that renders a PATH (`DfirUnit::PtRow(..)`) and a
    /// local's name has to be an identifier. Deriving one from the other by string surgery is how a
    /// `u_DfirUnit::PtRow(..)` would have been emitted.
    pub const fn local(self) -> &'static str {
        match self {
            PortUnit::Sfp => "sfp",
            PortUnit::Pe => "pe",
            PortUnit::Lxlu => "lxlu",
            PortUnit::Lxsu => "lxsu",
            PortUnit::L0lu => "l0lu",
            PortUnit::L0su => "l0su",
            PortUnit::SfpRing => "sfpring",
            PortUnit::PtRow0 => "ptrow0",
            PortUnit::PtRowN(_) => "ptrown",
        }
    }
}

/// WHERE A COMPUTE'S BODY GOES — one unit, one named row, or every row of the array.
///
/// ⛔⛔ `"pt"` IS REWRITTEN TO A ROW SPAN BEFORE ANYTHING SEES IT (`ddl_conversion.cpp:730-731`:
/// `if (unitStr == "pt") unitStr = "ptrow0-" + (numPTRows - 1)`), `unrollRowUnits` pushes one component
/// per row (`:734-749`), and `:1467-1472` CLONES the compute node for each. So a compute on the array
/// is N computes, and that is why `buildNeighborUnits` only ever sees `PTROW0..7`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeUnit {
    /// A unit that is also a wire end.
    Port(PortUnit),
    /// One row of the PT, by index.
    PtRow(u32),
    /// Every row the arch has — what `unit="pt"` unrolls to.
    EveryPtRow,
}

impl ComputeUnit {
    /// THE ONE PLACE A COMPUTE'S `unit=` SPELLING IS READ.
    pub fn parse(spelling: &str) -> Result<ComputeUnit, String> {
        match spelling {
            "pt" => Ok(ComputeUnit::EveryPtRow),
            "ptrow0" => Ok(ComputeUnit::PtRow(0)),
            "ptrow1" => Ok(ComputeUnit::PtRow(1)),
            "ptrow2" => Ok(ComputeUnit::PtRow(2)),
            "ptrow3" => Ok(ComputeUnit::PtRow(3)),
            "ptrow4" => Ok(ComputeUnit::PtRow(4)),
            "ptrow5" => Ok(ComputeUnit::PtRow(5)),
            "ptrow6" => Ok(ComputeUnit::PtRow(6)),
            "ptrow7" => Ok(ComputeUnit::PtRow(7)),
            other => PortUnit::parse(other).map(ComputeUnit::Port),
        }
    }

    /// The scope this unit's body needs, the `DfirUnit` expression, and the scope's close.
    ///
    /// ⛔ THE ROW COUNT IS THE ARCH'S, so a span is a LOOP in the generated code rather than a list
    /// unrolled here — `Target::PT_ROWS` belongs to the feature the LIBRARY is built with.
    pub fn render(self) -> (String, String, String) {
        match self {
            ComputeUnit::Port(port) => (String::new(), port.dfir(), String::new()),
            // ⛔ A ROW THIS ARCH DOES NOT HAVE IS NOT AN ERROR HERE. `ptrow7` names a row SEN1P5 lacks,
            // and `Row::checked` is what says so — at the library's compile time.
            ComputeUnit::PtRow(index) => (
                format!(
                    "        let Some(pt_row) = crate::units::Row::checked({index}) else {{ return }};\n"
                ),
                "DfirUnit::PtRow(pt_row)".to_owned(),
                String::new(),
            ),
            // ⛔⛔ NOT A RUST `for`, AND THAT WAS A REAL BUG. Emitting
            // `for pt_row in rows_of(Unit::Pt) { .. }` around each compute opened a fresh SCOPE per
            // statement, so a `p_<connect>` bound in one was invisible to the next — which broke the
            // systolic accumulator, whose MAC reads the connect it writes
            // (`let p_arf_ptsum = e.mac(.., p_arf_ptsum)`).
            //
            // ⭐ AND THE REFERENCE DOES NOT LOOP EITHER: `ddl_conversion.cpp:1467-1472` CLONES the
            // compute node once per row, which is N program units, not one unit with a loop. Our
            // `Program` is already per-node, so the clone belongs at that level.
            //
            // ⛔ SO THIS EMITS ROW 0 AND THE FAN-OUT IS OWED. Row 0 is the row the LXLU feeds
            // (`DSC2ToDataflowIRUtils.hpp:172-173`) and the one a port's `unit="pt"` resolves to, so it
            // is the right single row — but a bmm that must run on all eight is not yet all eight, and
            // that is a debt to discharge with the clone, not a loop to bolt on here.
            ComputeUnit::EveryPtRow => (
                String::new(),
                "DfirUnit::PtRow(crate::units::Row::checked(0).expect(\"row 0 exists\"))".to_owned(),
                String::new(),
            ),
        }
    }
}

/// WHICH `ElemType` A `data_type=` IS.
///
/// ⛔ THE SPELLING IS THE `DataType` IDENT, NOT THE TEMPLATE'S TEXT. `Program::formats` holds the idents
/// `ident_of` already made — `Sen169Fp16`, not `SEN169_FP16`. Matching the raw text deferred all 218.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Elem {
    F16,
    F32,
    Bf16,
    F8E4M3Fn,
    F4E2M1Fn,
    Int(u32),
}

impl Elem {
    /// THE ONE PLACE A `data_type=` IDENT IS READ.
    pub fn parse(spelling: &str) -> Result<Elem, String> {
        match spelling {
            "Sen169Fp16" => Ok(Elem::F16),
            "IeeeFp32" => Ok(Elem::F32),
            "Bfloat16" => Ok(Elem::Bf16),
            "Sen143Fp8" | "Sen053Fp8" | "Sen080Fp8" => Ok(Elem::F8E4M3Fn),
            "Sen121Fp4" => Ok(Elem::F4E2M1Fn),
            "Senint8" => Ok(Elem::Int(8)),
            "Senint4" => Ok(Elem::Int(4)),
            "Senint24" => Ok(Elem::Int(24)),
            "Senuint32" => Ok(Elem::Int(32)),
            "Bool" => Ok(Elem::Int(1)),
            other => Err(format!("data_type=\"{other}\" has no ElemType")),
        }
    }

    /// The `ElemType` expression.
    pub fn rust(self) -> String {
        match self {
            Elem::F16 => "ElemType::F16".to_owned(),
            Elem::F32 => "ElemType::F32".to_owned(),
            Elem::Bf16 => "ElemType::Bf16".to_owned(),
            Elem::F8E4M3Fn => "ElemType::F8E4M3Fn".to_owned(),
            Elem::F4E2M1Fn => "ElemType::F4E2M1Fn".to_owned(),
            Elem::Int(bits) => format!("ElemType::Int({bits})"),
        }
    }
}

/// A TWO-OPERAND VALUE OP — `vectorchain.binary`'s `binary_op`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    MulDiv2,
    Min,
    Max,
    AbsMin,
    AbsMax,
    And,
    Or,
    Xnor,
    AndNot,
}

impl BinOp {
    /// The `BinaryOp` variant.
    pub const fn rust(self) -> &'static str {
        match self {
            BinOp::Add => "Add",
            BinOp::Sub => "Sub",
            BinOp::Mul => "Mul",
            BinOp::MulDiv2 => "MulDiv2",
            BinOp::Min => "Min",
            BinOp::Max => "Max",
            BinOp::AbsMin => "AbsMin",
            BinOp::AbsMax => "AbsMax",
            BinOp::And => "And",
            BinOp::Or => "Or",
            BinOp::Xnor => "Xnor",
            BinOp::AndNot => "AndNot",
        }
    }
}

/// A COMPARISON — `vectorchain.element_wise_compare`'s `compare_op`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cmp {
    Eq,
    Neq,
    Lt,
    Le,
    Gt,
    Ge,
}

impl Cmp {
    /// The `CompareOp` variant.
    pub const fn rust(self) -> &'static str {
        match self {
            Cmp::Eq => "Eq",
            Cmp::Neq => "Neq",
            Cmp::Lt => "Lt",
            Cmp::Le => "Le",
            Cmp::Gt => "Gt",
            Cmp::Ge => "Ge",
        }
    }
}

/// WHICH SFP ESTIMATE, and its version where it takes one.
///
/// ⛔⛔ THE MODE SELECTS THIS, AND ONE ARM FOR ALL MODES WAS A LIVE WRONG ANSWER. `compute_kind` read
/// `Exp` for every `FEST`, so a template asking for a reciprocal, a log, an rsqrt, a sigmoid or a tanh
/// got an EXPONENTIAL — a different function dbo-opt compiles happily. The table is
/// `constructUnaryOperation`'s own (`SNComputeLowering.cpp:1316-1388`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Est {
    Exp,
    Rec,
    Ln,
    Rsqrt,
    Sigmoid,
    Tanh,
}

/// ⛔ `rec` AND `ln` TAKE NO VERSION while the others do, which is why this is an `Option` on the
/// estimate rather than a defaulted argument at the constructor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ver {
    A,
    B,
    Slope,
    Offset,
}

impl Est {
    /// The `EstimateKind` variant.
    pub const fn rust(self) -> &'static str {
        match self {
            Est::Exp => "Exp",
            Est::Rec => "Rec",
            Est::Ln => "Ln",
            Est::Rsqrt => "Rsqrt",
            Est::Sigmoid => "Sigmoid",
            Est::Tanh => "Tanh",
        }
    }
}

impl Ver {
    /// The `EstimateVersion` variant.
    pub const fn rust(self) -> &'static str {
        match self {
            Ver::A => "A",
            Ver::B => "B",
            Ver::Slope => "Slope",
            Ver::Offset => "Offset",
        }
    }
}

/// A ONE-OPERAND SHAPE — `constructUnaryOperation`'s family (`SNComputeLowering.cpp:1633-1640`), which
/// refuses anything but exactly one input (`:1276-1278`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unary {
    Floor,
    /// The lane mapping, whose `indices=`/`repetition=` the template states.
    Shuffle,
    /// ⛔ `ICVT` mode 7 — and there is no other ICVT. Despite the name it converts nothing: the result
    /// type is the input's, because `unary_op_result_type` is only reassigned by `SHUFFLE` and `CAST`
    /// (`:1309`, `:1493`, `:1529`). A type conversion is `computetype="cast"`.
    FastExp,
    /// An SFP estimate and its version.
    Estimate(Est, Option<Ver>),
    /// ⛔ `SPLAT` IS A SHUFFLE WITH A GENERATED INDEX PATTERN, not an op of its own.
    /// `SNComputeLowering.cpp:1406-1451`: `sign_extend_ == 0` builds `vectorchain.shuffle` with
    /// all-zero indices — EIGHT of them for f16, FOUR for f32 — and no pad operand; `sign_extend_ == 1`
    /// builds `{0, -1, 0, -1, ..}` (a `-1` writing zero) WITH an `arith.constant 0 : index` pad. Any
    /// other value is `emitError("sign_extend has to be either 0 or 1.")`.
    ///
    /// ⛔ AND THE LANE COUNT IS THE ELEMENT TYPE'S, which is why it is carried here: an f32 splat with
    /// eight indices describes a pattern twice the width of the register.
    Splat {
        /// Whether the pattern interleaves the zero-writing `-1`s.
        sign_extend: bool,
    },
    /// ⛔ `REDUCE` IS `vectorchain.scan_with_gap`, AND ITS MODE PICKS THE BINOP —
    /// `SNComputeLowering.cpp:1453-1490`: 1 add, 8 max, 10 abs_max, 12 min, 14 abs_min, anything else
    /// `emitError("Unknown binary operation for reduction.")`. It carries `gap = 8` and a left-to-right
    /// evaluation order, and takes NO mask operand — the only unary in the family that does not.
    Reduce(BinOp),
}

/// WHAT A `computetype=` BUILDS — the five families of `constructComputeOperation`
/// (`SNComputeLowering.cpp:1591-1673`), whose `else` is `"Unknown compute operation"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeShape {
    /// `a * b + acc`, THREE operands — `IMA8 IMA4 FMA4 FMA8 FMA16 FMA32 FNMS` (`:1594-1598`).
    Mac,
    /// A two-operand value op.
    Binary(BinOp),
    /// A comparison, whose result is an `i1` vector and not a value one.
    Compare(Cmp),
    /// ⛔ FMAX/FMIN — a COMPARE PLUS A SELECTION, two ops (`:1240-1251`), the compare being
    /// `compare_gt` for FMAX and `compare_le` for FMIN. Which result leaves depends on whether the
    /// destination is `PESTATE`/`SFPSTATE` (`:1253-1269`).
    MinMax(Cmp),
    /// A ternary whose FIRST operand is the condition (`:1170-1174`).
    Select,
    /// `vectorchain.pack` — two data operands, result type from the OUTPUT format (`:1077-1082`).
    Pack,
    /// A one-operand shape.
    Unary(Unary),
}

/// WHAT A `computetype=` BUILDS, or the reason it cannot be built here.
///
/// ⛔ CASE-INSENSITIVE, BECAUSE THE REFERENCE IS. `ddl_conversion.cpp:1397` reads
/// `compute_op.getComputetype().lower()` and looks THAT up in `stringToComputeType`
/// (`dscdefn.cpp:34-106`, flipped), so the template's spelling case carries no meaning. The vendored
/// set is mixed (`FMA16`, `MACC`, `EQUAL` upper; `assign` lower).
///
/// ⛔⛔ `mode=` IS AN `I64Attr`, NOT A STRING, AND READING IT WITH `attr_str` GAVE `None` FOR EVERY
/// TEMPLATE. `ddl.compute` carries `OptionalAttr<I64Attr>:$mode` (`DdlOps.td:625-631`) and `build.rs`
/// renders it through `opt_int`. This is the SECOND time an attribute was read with the wrong accessor
/// here — `rotate_num_elements` was the first — and both times the symptom was a fact silently absent
/// rather than a mismatch, which is why the accessor's type is the guard and the value never is.
///
/// ⛔ `elem` IS THE BIND'S PRECISION, and it is load-bearing rather than decoration: `MACC` is a
/// PSEUDO-OP the reference monomorphises against it (`ddl_conversion.cpp:1399-1419`).
pub fn compute_kind(
    computetype: &str,
    mode: Option<i64>,
    sign_extend: Option<i64>,
    elem: Elem,
) -> Result<ComputeShape, String> {
    let spelling = computetype.to_ascii_uppercase();
    match spelling.as_str() {
        // ⛔ `MACC` IS NOT IN THE MAC LIST — see its own arm below for what it is.
        "FMA16" | "FMA32" | "FMA8" | "FMA4" | "IMA8" | "IMA4" | "FNMS" => Ok(ComputeShape::Mac),

        "FMAX" => Ok(ComputeShape::MinMax(Cmp::Gt)),
        "FMIN" => Ok(ComputeShape::MinMax(Cmp::Le)),

        // ⭐ THE BINARY-OR-TERNARY FAMILY (`:1618-1625`). ONE constructor serves all of these, and it
        // is binary OR TERNARY — the operand count is the TEMPLATE's, not the opcode's.
        "FABSMAX" => Ok(ComputeShape::Binary(BinOp::AbsMax)),
        "FABSMIN" => Ok(ComputeShape::Binary(BinOp::AbsMin)),
        "AND" => Ok(ComputeShape::Binary(BinOp::And)),
        "OR" => Ok(ComputeShape::Binary(BinOp::Or)),
        "XNOR" => Ok(ComputeShape::Binary(BinOp::Xnor)),
        "ANDNOT" => Ok(ComputeShape::Binary(BinOp::AndNot)),
        "FADD" => Ok(ComputeShape::Binary(BinOp::Add)),
        "FSUB" => Ok(ComputeShape::Binary(BinOp::Sub)),
        // ⛔ `FMUL` WITH `mode=11` IS `mul_div2` — the operation is `X*Y/2` (`:1101-1108`), so emitting
        // `mul` would silently double every result the mode asks to be halved.
        "FMUL" if mode == Some(11) => Ok(ComputeShape::Binary(BinOp::MulDiv2)),
        "FMUL" => Ok(ComputeShape::Binary(BinOp::Mul)),

        // ⛔ THE ENUM SPELLS IT `EQUALTO`, not `EQUAL` — the template's text and the `ComputeOpType`
        // name differ here, so both are matched.
        "EQUAL" | "EQUALTO" => Ok(ComputeShape::Compare(Cmp::Eq)),
        "NOTEQUAL" => Ok(ComputeShape::Compare(Cmp::Neq)),
        "LESSERTHAN" => Ok(ComputeShape::Compare(Cmp::Lt)),
        "LESSEREQUAL" => Ok(ComputeShape::Compare(Cmp::Le)),
        "GREATERTHAN" => Ok(ComputeShape::Compare(Cmp::Gt)),
        "GREATEREQUAL" => Ok(ComputeShape::Compare(Cmp::Ge)),

        "SELECT" => Ok(ComputeShape::Select),
        "PACKMERGE" => Ok(ComputeShape::Pack),

        "FLOOR" => Ok(ComputeShape::Unary(Unary::Floor)),
        "SHUFFLE" => Ok(ComputeShape::Unary(Unary::Shuffle)),
        // ⛔ `sign_extend=` DECIDES THE PATTERN, and anything but 0 or 1 is the reference's own error
        // (`SNComputeLowering.cpp:1406-1451`). Absent means 0 — `dsc2.h:908` defaults it false.
        "REDUCE" => match mode {
            Some(1) => Ok(ComputeShape::Unary(Unary::Reduce(BinOp::Add))),
            Some(8) => Ok(ComputeShape::Unary(Unary::Reduce(BinOp::Max))),
            Some(10) => Ok(ComputeShape::Unary(Unary::Reduce(BinOp::AbsMax))),
            Some(12) => Ok(ComputeShape::Unary(Unary::Reduce(BinOp::Min))),
            Some(14) => Ok(ComputeShape::Unary(Unary::Reduce(BinOp::AbsMin))),
            other => Err(format!(
                "computetype=\"REDUCE\" with mode={other:?} has no binop — the reference admits \
                 1/8/10/12/14 and errors otherwise (SNComputeLowering.cpp:1453-1490)"
            )),
        },
        "SPLAT" => match sign_extend {
            None | Some(0) => Ok(ComputeShape::Unary(Unary::Splat { sign_extend: false })),
            Some(1) => Ok(ComputeShape::Unary(Unary::Splat { sign_extend: true })),
            Some(other) => Err(format!(
                "computetype=\"SPLAT\" with sign_extend={other} — the reference admits only 0 or 1 \
                 (SNComputeLowering.cpp:1406-1451)"
            )),
        },
        "ICVT" if mode == Some(7) => Ok(ComputeShape::Unary(Unary::FastExp)),
        "ICVT" => Err(format!(
            "computetype=\"ICVT\" with mode={mode:?} — only mode 7 exists, and it is \
             `vectorchain.fast_exp`, not a conversion (SNComputeLowering.cpp:1389-1399)"
        )),
        "FEST" => match mode {
            Some(0) => Ok(ComputeShape::Unary(Unary::Estimate(Est::Exp, Some(Ver::A)))),
            Some(1) => Ok(ComputeShape::Unary(Unary::Estimate(Est::Exp, Some(Ver::B)))),
            Some(2) => Ok(ComputeShape::Unary(Unary::Estimate(Est::Rec, None))),
            Some(3) => Ok(ComputeShape::Unary(Unary::Estimate(Est::Ln, None))),
            Some(5) => Ok(ComputeShape::Unary(Unary::Estimate(Est::Rsqrt, None))),
            Some(6) => Ok(ComputeShape::Unary(Unary::Estimate(
                Est::Sigmoid,
                Some(Ver::Slope),
            ))),
            Some(7) => Ok(ComputeShape::Unary(Unary::Estimate(
                Est::Sigmoid,
                Some(Ver::Offset),
            ))),
            Some(8) => Ok(ComputeShape::Unary(Unary::Estimate(
                Est::Tanh,
                Some(Ver::Slope),
            ))),
            Some(9) => Ok(ComputeShape::Unary(Unary::Estimate(
                Est::Tanh,
                Some(Ver::Offset),
            ))),
            other => Err(format!(
                "computetype=\"FEST\" with mode={other:?} has no estimate — the reference's own \
                 dispatch errors on it (SNComputeLowering.cpp:1316-1388), and 4 is absent"
            )),
        },

        // ⭐⭐⭐ `MACC` IS A PRECISION-POLYMORPHIC PSEUDO-OP, MONOMORPHISED AT DDL-PARSE TIME.
        // `ddl_conversion.cpp:1399-1419` intercepts it BEFORE the `stringToComputeType` lookup at
        // `:1421` and picks the concrete opcode from the bind's own precision:
        //
        //   SENINT4 -> IMA4 · SENINT8 -> IMA8 · SEN143/152_FP8 -> FMA8
        //   SEN169_FP16 / BFLOAT16 -> FMA16 · IEEE_FP32 -> FMA32 · SEN121_FP4 -> FMA4
        //
        // and anything else is `emitError("Unexpected input precision")`. `dscdefn.h:135` says so in
        // one line: `MACC,  // precision independent fma/ima`. All six land in the MAC family, so the
        // SHAPE is always a MAC and the precision only has to be admissible.
        //
        // ⛔ AND THIS IS WHY `ComputeOpType::MACC` APPEARS NOWHERE ELSE IN THE C++ — the code never
        // ASSIGNS MACC, it assigns the resolved type. Grepping the enum name and finding nothing led me
        // to conclude 129 templates were unlowerable; the interception is one branch above the lookup I
        // was reading.
        "MACC" => match elem {
            Elem::Int(4 | 8) | Elem::F8E4M3Fn | Elem::F16 | Elem::Bf16 | Elem::F32
            | Elem::F4E2M1Fn => Ok(ComputeShape::Mac),
            other => Err(format!(
                "computetype=\"MACC\" at {other:?} — the reference monomorphises it against the \
                 bind's precision and errors on anything but int4/int8/fp8/fp16/bf16/fp32/fp4 \
                 (ddl_conversion.cpp:1399-1418, \"Unexpected input precision\")"
            )),
        },

        // ⛔⛔ `ASSIGN` IS ELIMINATED BEFORE ANY OTHER TRANSFORMATION RUNS, and not by this dispatch.
        // `Ddc::performAutomaticShuffling` (`ddc_transformation.cpp:1865`) is the FIRST pass after DDL
        // parse (`ddcv1.cpp:3733`), and `:2011-2028` matches every PE/SFP compute whose `type_` is
        // `ASSIGN` and hands it to `AutoShuffler::replace_assign`, which emits a straight-line sequence
        // of `PACKMERGE` nodes and DELETES the assign (`shuffle.cpp:788-908`).
        //
        // ⭐ SO THE OUTPUT SHAPE IS ALREADY BUILT HERE — `ComputeShape::Pack` — and what is missing is
        // the SEARCH that decides the sequence: a Dijkstra over layouts (`shuffle.cpp:1161-1222`) whose
        // nodes are `AbstractLayout`s, edges are eight `ShuffleAction`s, and cost is instruction count.
        // ~1,600 lines with no MLIR dependency, and no test coverage in the reference.
        "ASSIGN" => Err(
            "computetype=\"assign\" is eliminated by the AutoShuffler BEFORE any other pass \
             (ddc_transformation.cpp:2011-2028, ddcv1.cpp:3733), which replaces it with a Dijkstra-chosen \
             sequence of PACKMERGE ops (shuffle.cpp:1161-1222). The emitted op is `vectorchain.pack`, \
             which this generator already builds; the missing piece is the layout search"
                .to_owned(),
        ),

        // ⛔⛔ THE OPAQUE FAMILY IS NOT `vectorchain` AT ALL, AND IT CANNOT BE EMITTED HERE.
        // `constructOpaqueOperation` (`:1645-1667`) builds ONE attributes-only `dataflow::OpaqueOp`
        // whose register dictionaries carry FINAL LITERAL ADDRESSES — `"R0"`, `"R5"` — produced by
        // `Ddc::allocAllMem` -> `finalizeOps` (`ddcv1.cpp:132`, `:3332`) BEFORE any DataflowIR exists,
        // and `DataflowToSentient.cpp:2005-2010` copies them verbatim. There is no symbolic form to
        // fill in later, so this needs a register allocator and not an arm.
        "RECIPROCAL" | "LAYERNORMSCALE" | "GELU" | "SIGMOID" | "EXP" | "EXP_P1" | "EXP_P2"
        | "LOG_P1" | "LOG_P2" | "GELU_BWD_P1" | "GELU_BWD_P2" | "SQRT" | "RSQRT" | "MISH_P1"
        | "MISH_P2" | "EXX2_32_P1" | "EXX2_32_P2" | "EXX2_32_P3" | "DL16TOFP32" | "FP32TODL16"
        | "DL16TOBF16" | "SOFTPLUS_P1" | "SOFTPLUS_P2" | "IDX32TOADDR" | "MUL_I32_TO_I32"
        | "ADD_I32_TO_I32" | "ADD_I64_TO_I64" | "MUL_I64_TO_I64_PE" | "MUL_I64_TO_I64_SFP" => {
            Err(format!(
                "computetype=\"{computetype}\" is an OPAQUE body, and its register dictionaries carry \
                 FINAL addresses assigned before DataflowIR exists (ddcv1.cpp:3347-3395) — it needs a \
                 register allocator, not an emitter arm"
            ))
        }

        other => Err(format!(
            "computetype=\"{other}\" has no arm in SNComputeLowering's dispatch \
             (:1591-1673 is exhaustive over the reference's own set, so a spelling missing here is \
             either an unported family or a name the templates and the enum spell differently)"
        )),
    }
}
