//! The `.td` front end: island IRs are TRANSCRIBED, not hand-written.
//!
//! The four islands are not four hand-authored Rust IRs. Two of them are already declared, in
//! TableGen, by the C++ that owns them — so the Rust types are a mechanical projection of an
//! existing table rather than a second record of it:
//!
//! | island     | source                                          | here |
//! |------------|-------------------------------------------------|------|
//! | DataflowIR | `Dataflow.td`                                   | ✅   |
//! | SentientIR | `SentientOps.td` + `SentientTypes.td`           | ✅   |
//! | ProgIR     | `sys-arch-spec/progir/progir.h` — plain C++     | ⛔   |
//! | init bytes | none — `init.bin` names an artifact, see below  | ⛔   |
//!
//! ⛔ **`InitOps.td` does not describe the packet.** `init.bin { name, size } : tensor<Nxi8>`
//! holds a NAME into an external context map — "the actual artifact is stored externally in a
//! context map keyed by this name". The byte layout lives in DIP's encoders (`dip.cpp:3392`,
//! `initpacket.cpp:215`), which is what `packet.dsl`'s `wire` relation exists to become. Until
//! `wire` is written there is no schema to serde against, so the final lowering cannot yet be
//! "just serde" — that is a missing table, not a missing mechanism.
//!
//! This module parses the TableGen subset MLIR ODS actually uses: `def`s with `arguments`,
//! `results`, traits and a mnemonic. It deliberately does not implement TableGen — no `multiclass`,
//! no `foreach`, no `!cast`. If a dialect starts using those, this reports it rather than guessing.

use std::fmt::Write as _;

#[derive(Debug, Clone)]
pub struct TdOp {
    /// The record name, e.g. `Dataflow_GetUnitOp`.
    pub def_name: String,
    /// The base class, e.g. `Dataflow_Op`.
    pub base: String,
    /// The assembly mnemonic, e.g. `get_unit`.
    pub mnemonic: String,
    pub traits: Vec<String>,
    pub arguments: Vec<TdField>,
    pub results: Vec<TdField>,
    /// `let assemblyFormat = [{ … }]` — THE OP'S SYNTAX, as the dialect states it.
    ///
    /// ⭐ THIS IS WHAT MAKES A PRINTER A PROJECTION RATHER THAN A TRANSCRIPTION. Empty means the op
    /// declares none, and then there are exactly two cases, told apart by [`Self::custom_asm`]:
    /// a hand-written parser/printer in C++ we cannot see, or no format at all — in which case
    /// MLIR's GENERIC form (`"dialect.op"(%a) {attrs} : (T) -> T`) is the only legal spelling.
    /// `dataflow.return` is the second kind.
    pub asm_format: String,
    /// `let hasCustomAssemblyFormat = 1` — the syntax lives in C++, not in the `.td`.
    pub custom_asm: bool,
    pub summary: String,
    pub file: String,
    pub line: usize,
}

#[derive(Debug, Clone)]
pub struct TdField {
    /// The ODS type as written: `StrAttr`, `Index`, `Variadic<Index>`, `AnyRankedTensor`.
    pub ty: String,
    /// The `$name`, without the sigil. Empty when ODS omitted it.
    pub name: String,
    pub variadic: bool,
    pub optional: bool,
}

#[derive(Debug, Default)]
pub struct TdSpec {
    pub ops: Vec<TdOp>,
    /// Constructs this parser deliberately does not implement, so their presence is visible
    /// rather than silently dropped.
    pub unsupported: Vec<String>,
}

/// Blank out `[{ … }]` code blocks. They hold prose and MLIR examples with unbalanced braces, and
/// a brace matcher that walked into one would swallow the rest of the file.
///
/// ⭐ WITH ONE EXCEPTION: `let assemblyFormat = [{ … }]`. THE OP'S SYNTAX IS IN THERE, and blanking
/// it is why nothing downstream could print an op — the format that says `$unitId attr-dict \`:\`
/// type(results)` was erased by the first pass over the file, so every consumer had to hand-write
/// what the `.td` already stated. That block is kept, escaped onto the `let` line so [`let_str`]
/// reads it; its newlines are pushed AFTER the closing quote so every later [`TdOp::line`] is
/// still the line it was.
fn blank_code_blocks(src: &str) -> String {
    let b = src.as_bytes();
    let mut out = String::with_capacity(src.len());
    let mut i = 0;
    while i < b.len() {
        if i + 1 < b.len() && b[i] == b'[' && b[i + 1] == b'{' {
            let mut j = i + 2;
            let mut newlines = 0usize;
            let mut body = String::new();
            while j + 1 < b.len() && !(b[j] == b'}' && b[j + 1] == b']') {
                if b[j] == b'\n' {
                    newlines += 1;
                }
                body.push(b[j] as char);
                j += 1;
            }
            // Whether this block is the one construct worth keeping. The `let` and the `[{` are
            // separated only by `=` and whitespace, so the tail of what we have already written
            // decides it.
            let tail = &out[out.len().saturating_sub(48)..];
            if tail.contains("assemblyFormat") {
                out.push('"');
                for c in body.split_whitespace().collect::<Vec<_>>().join(" ").chars() {
                    if c == '"' || c == '\\' {
                        out.push('\\');
                    }
                    out.push(c);
                }
                out.push('"');
            } else {
                out.push_str("\"\"");
            }
            for _ in 0..newlines {
                out.push('\n');
            }
            i = (j + 2).min(b.len());
            continue;
        }
        out.push(b[i] as char);
        i += 1;
    }
    out
}

fn strip_line_comments(src: &str) -> String {
    src.lines()
        .map(|l| match l.find("//") {
            Some(i) => &l[..i],
            None => l,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Balanced-delimiter scan from `open` at `start`, returning the index of its match.
fn match_delim(b: &[u8], start: usize, open: u8, close: u8) -> Option<usize> {
    let mut depth = 0i32;
    let mut i = start;
    while i < b.len() {
        if b[i] == open {
            depth += 1;
        } else if b[i] == close {
            depth -= 1;
            if depth == 0 {
                return Some(i);
            }
        }
        i += 1;
    }
    None
}

fn split_top(s: &str, sep: char) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut cur = String::new();
    for c in s.chars() {
        match c {
            '<' | '(' | '[' | '{' => {
                depth += 1;
                cur.push(c);
            }
            '>' | ')' | ']' | '}' => {
                depth -= 1;
                cur.push(c);
            }
            c if c == sep && depth == 0 => {
                out.push(cur.trim().to_string());
                cur = String::new();
            }
            _ => cur.push(c),
        }
    }
    if !cur.trim().is_empty() {
        out.push(cur.trim().to_string());
    }
    out
}

/// `StrAttr:$name` / `Variadic<Index>:$units` / `Optional<F64Attr>:$x` / bare `Index`.
fn parse_field(item: &str) -> Option<TdField> {
    let item = item.trim();
    if item.is_empty() {
        return None;
    }
    let (ty, name) = match item.split_once(":$") {
        Some((t, n)) => (t.trim(), n.trim()),
        None => (item, ""),
    };
    Some(TdField {
        variadic: ty.starts_with("Variadic<"),
        optional: ty.starts_with("Optional<") || ty.starts_with("DefaultValuedAttr<"),
        ty: ty.to_string(),
        name: name.to_string(),
    })
}

/// `let <field> = (<kind> …);` — returns the parenthesised list, minus its leading `ins`/`outs`.
fn let_list(body: &str, field: &str) -> Option<String> {
    let needle = format!("let {field}");
    let at = body.find(&needle)?;
    let b = body.as_bytes();
    let open = body[at..].find('(').map(|i| at + i)?;
    let close = match_delim(b, open, b'(', b')')?;
    let inner = body[open + 1..close].trim();
    Some(
        inner
            .strip_prefix("ins")
            .or_else(|| inner.strip_prefix("outs"))
            .unwrap_or(inner)
            .trim()
            .to_string(),
    )
}

/// `let <field> = "…";` — the string, with `\"` and `\\` unescaped.
///
/// ⛔ THE ESCAPES ARE NOT DECORATION. `blank_code_blocks` re-emits an `assemblyFormat` as a quoted
/// string, and a format holding a literal quote would otherwise end the value early and truncate
/// the op's syntax to whatever preceded it — a shorter format that still parses.
fn let_str(body: &str, field: &str) -> String {
    let needle = format!("let {field}");
    let Some(at) = body.find(&needle) else { return String::new() };
    let rest = &body[at + needle.len()..];
    let Some(eq) = rest.find('=') else { return String::new() };
    let tail = rest[eq + 1..].trim_start();
    let Some(t) = tail.strip_prefix('"') else { return String::new() };
    let mut out = String::new();
    let mut chars = t.chars();
    while let Some(c) = chars.next() {
        match c {
            '\\' => match chars.next() {
                Some(e) => out.push(e),
                None => break,
            },
            '"' => return out,
            _ => out.push(c),
        }
    }
    out
}

/// `let <field> = 1;` — an ODS boolean, which TableGen writes as a bare int.
fn let_flag(body: &str, field: &str) -> bool {
    let needle = format!("let {field}");
    let Some(at) = body.find(&needle) else { return false };
    let rest = &body[at + needle.len()..];
    let Some(eq) = rest.find('=') else { return false };
    rest[eq + 1..].trim_start().starts_with('1')
}

pub fn parse_td(spec: &mut TdSpec, file: &str, text: &str) {
    let cleaned = strip_line_comments(&blank_code_blocks(text));

    for kw in ["multiclass ", "foreach ", "!cast<"] {
        if cleaned.contains(kw) {
            spec.unsupported.push(format!("{file}: uses `{}`", kw.trim()));
        }
    }

    let b = cleaned.as_bytes();
    let mut cursor = 0usize;
    while let Some(rel) = cleaned[cursor..].find("\ndef ") {
        let at = cursor + rel + 1;
        let line = cleaned[..at].bytes().filter(|c| *c == b'\n').count() + 1;

        // `def NAME : BASE<"mnemonic", [traits]> { … }` — the body is optional.
        let header_end = cleaned[at..].find('{').map(|i| at + i);
        let stmt_end = cleaned[at..].find(';').map(|i| at + i);
        let (body, next) = match (header_end, stmt_end) {
            (Some(h), s) if s.is_none_or(|s| h < s) => match match_delim(b, h, b'{', b'}') {
                Some(e) => (cleaned[h + 1..e].to_string(), e + 1),
                None => (String::new(), at + 4),
            },
            (_, Some(s)) => (String::new(), s + 1),
            _ => (String::new(), at + 4),
        };
        let header = &cleaned[at..header_end.unwrap_or(next).min(next)];

        cursor = next;

        let after_def = header.trim_start_matches("def ").trim();
        let Some((def_name, after_colon)) = after_def.split_once(':') else { continue };
        let def_name = def_name.trim().to_string();
        let after_colon = after_colon.trim();

        // Only ops carry a mnemonic in `<"…">`. Types, attrs and enums are skipped here: they are
        // a separate projection and pretending they are ops would fabricate structure.
        let base = after_colon.split('<').next().unwrap_or("").trim().to_string();
        let Some(lt) = after_colon.find('<') else { continue };
        let Some(gt) = match_delim(after_colon.as_bytes(), lt, b'<', b'>') else { continue };
        let args = split_top(&after_colon[lt + 1..gt], ',');
        let Some(first) = args.first() else { continue };
        if !first.trim().starts_with('"') {
            continue;
        }
        let mnemonic = first.trim().trim_matches('"').to_string();
        let traits = args
            .get(1)
            .map(|t| {
                split_top(t.trim().trim_start_matches('[').trim_end_matches(']'), ',')
                    .into_iter()
                    .filter(|s| !s.is_empty())
                    .collect()
            })
            .unwrap_or_default();

        let arguments = let_list(&body, "arguments")
            .map(|l| split_top(&l, ',').iter().filter_map(|f| parse_field(f)).collect())
            .unwrap_or_default();
        let results = let_list(&body, "results")
            .map(|l| split_top(&l, ',').iter().filter_map(|f| parse_field(f)).collect())
            .unwrap_or_default();

        spec.ops.push(TdOp {
            def_name,
            base,
            mnemonic,
            traits,
            arguments,
            results,
            asm_format: let_str(&body, "assemblyFormat"),
            custom_asm: let_flag(&body, "hasCustomAssemblyFormat"),
            summary: let_str(&body, "summary"),
            file: file.to_string(),
            line,
        });
    }
}

/// An ODS type as the generated Rust sees it.
///
/// ⛔ `Value` IS AN SSA HANDLE, NOT A VALUE. An operand or result is a reference to something
/// another op produced; only attributes carry data. Collapsing the two is how an emitter ends up
/// inventing operands it cannot have.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RustTy {
    /// `StrAttr` — owned, because the emitter builds these.
    Str,
    /// `I64Attr`, `IndexAttr`, `SI64Attr`.
    I64,
    /// `BoolAttr`.
    Bool,
    /// `F64Attr`.
    F64,
    /// `I64ArrayAttr`, `ArrayAttr`.
    I64Array,
    /// `AffineMapAttr` — a map, not a number. Two of these hang off every memory view.
    AffineMap,
    /// An SSA operand or result: `Index`, `AnyType`, `AnyRankedTensor`, a dialect type.
    Value,
    /// `Variadic<T>`.
    Many(Box<RustTy>),
    /// `Optional<T>`, `DefaultValuedAttr<T, _>`.
    Opt(Box<RustTy>),
    /// Anything this mapping does not know. Carried as its ODS spelling so it is VISIBLE rather
    /// than silently becoming a String.
    Unknown(String),
}

impl RustTy {
    pub fn render(&self) -> String {
        match self {
            RustTy::Str => "String".into(),
            RustTy::I64 => "i64".into(),
            RustTy::Bool => "bool".into(),
            RustTy::F64 => "f64".into(),
            RustTy::I64Array => "Vec<i64>".into(),
            RustTy::AffineMap => "AffineMap".into(),
            RustTy::Value => "Value".into(),
            RustTy::Many(t) => format!("Vec<{}>", t.render()),
            RustTy::Opt(t) => format!("Option<{}>", t.render()),
            RustTy::Unknown(_) => "Value".into(),
        }
    }
    pub fn is_unknown(&self) -> bool {
        matches!(self, RustTy::Unknown(_))
            || match self {
                RustTy::Many(t) | RustTy::Opt(t) => t.is_unknown(),
                _ => false,
            }
    }
}

/// Map an ODS type spelling to the Rust the emitter builds with.
pub fn ods_to_rust(ods: &str) -> RustTy {
    let t = ods.trim();
    if let Some(inner) = t.strip_prefix("Variadic<").and_then(|s| s.strip_suffix('>')) {
        return RustTy::Many(Box::new(ods_to_rust(inner)));
    }
    if let Some(inner) = t.strip_prefix("Optional<").and_then(|s| s.strip_suffix('>')) {
        return RustTy::Opt(Box::new(ods_to_rust(inner)));
    }
    // `DefaultValuedAttr<I64ArrayAttr, "{}">` — take the first type argument.
    if let Some(rest) = t.strip_prefix("DefaultValuedAttr<") {
        let inner = rest.split(',').next().unwrap_or("").trim();
        return RustTy::Opt(Box::new(ods_to_rust(inner)));
    }
    if let Some(inner) = t.strip_prefix("OptionalAttr<").and_then(|s| s.strip_suffix('>')) {
        return RustTy::Opt(Box::new(ods_to_rust(inner)));
    }
    // `Arg<T, "doc", [MemRead]>` / `Res<T, "doc">` — the constraint is T; the rest is
    // documentation and side-effect annotation, neither of which the emitter builds.
    for w in ["Arg<", "Res<"] {
        if let Some(rest) = t.strip_prefix(w) {
            let inner = rest.split(',').next().unwrap_or("").trim();
            return ods_to_rust(inner);
        }
    }
    match t {
        "StrAttr" | "SymbolNameAttr" | "StrArrayAttr" => RustTy::Str,
        "I64Attr" | "IndexAttr" | "SI64Attr" | "I32Attr" | "UI64Attr" => RustTy::I64,
        "BoolAttr" | "UnitAttr" => RustTy::Bool,
        "F64Attr" | "F32Attr" => RustTy::F64,
        "I64ArrayAttr" | "ArrayAttr" | "I32ArrayAttr" => RustTy::I64Array,
        "AffineMapAttr" | "AffineMapArrayAttr" => RustTy::AffineMap,
        "Index" | "AnyType" | "AnyRankedTensor" | "AnyMemRef" | "AnyTensor" => RustTy::Value,
        other => RustTy::Unknown(other.to_string()),
    }
}

/// A Rust identifier for a mnemonic: `get_unit` -> `GetUnit`.
pub fn camel(mnemonic: &str) -> String {
    mnemonic
        .split(['_', '.'])
        .filter(|s| !s.is_empty())
        .map(|s| {
            let mut c = s.chars();
            match c.next() {
                Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
                None => String::new(),
            }
        })
        .collect()
}

/// Emit the parsed dialects as const tables plus one enum per dialect.
///
/// The enum is the island's op vocabulary, projected from the `.td` rather than retyped. A
/// mechanical lowering between two islands is then a match from one of these enums to another,
/// which is a table — not a pass.
pub fn emit(spec: &TdSpec) -> String {
    let mut o = String::new();
    let _ = writeln!(o, "// @generated by build.rs from deeptools *.td — do not edit.\n");

    let _ = writeln!(o, "pub struct TdFieldDef {{ pub ty: &'static str, pub name: &'static str, pub variadic: bool, pub optional: bool }}");
    let _ = writeln!(o, "pub struct TdOpDef {{");
    let _ = writeln!(o, "    pub def_name: &'static str,");
    let _ = writeln!(o, "    pub base: &'static str,");
    let _ = writeln!(o, "    pub mnemonic: &'static str,");
    let _ = writeln!(o, "    pub traits: &'static [&'static str],");
    let _ = writeln!(o, "    pub arguments: &'static [TdFieldDef],");
    let _ = writeln!(o, "    pub results: &'static [TdFieldDef],");
    let _ = writeln!(o, "    pub asm_format: &'static str,");
    let _ = writeln!(o, "    pub custom_asm: bool,");
    let _ = writeln!(o, "    pub summary: &'static str,");
    let _ = writeln!(o, "    pub origin: &'static str,");
    let _ = writeln!(o, "}}\n");

    let _ = writeln!(o, "pub static TD_OPS: &[TdOpDef] = &[");
    for op in &spec.ops {
        let fields = |fs: &[TdField]| {
            fs.iter()
                .map(|f| {
                    format!(
                        "TdFieldDef {{ ty: {:?}, name: {:?}, variadic: {}, optional: {} }}",
                        f.ty, f.name, f.variadic, f.optional
                    )
                })
                .collect::<Vec<_>>()
                .join(", ")
        };
        let traits: Vec<String> = op.traits.iter().map(|t| format!("{t:?}")).collect();
        let _ = writeln!(o, "    TdOpDef {{");
        let _ = writeln!(o, "        def_name: {:?}, base: {:?}, mnemonic: {:?},", op.def_name, op.base, op.mnemonic);
        let _ = writeln!(o, "        traits: &[{}],", traits.join(", "));
        let _ = writeln!(o, "        arguments: &[{}],", fields(&op.arguments));
        let _ = writeln!(o, "        results: &[{}],", fields(&op.results));
        let _ = writeln!(o, "        asm_format: {:?}, custom_asm: {},", op.asm_format, op.custom_asm);
        let _ = writeln!(o, "        summary: {:?},", op.summary);
        let _ = writeln!(o, "        origin: {:?},", format!("{}:{}", op.file, op.line));
        let _ = writeln!(o, "    }},");
    }
    let _ = writeln!(o, "];\n");

    // One enum per DIALECT op base — `Dataflow_Op`, `Sentient_Op`, `Init_Op`, …
    //
    // ⛔ Filter to `*_Op`. MLIR builtins match the same shape — `I32EnumAttrCase<"None", 0>` and
    // `SingleBlockImplicitTerminator<"YieldOp">` both lead with a string literal — and projecting
    // them would invent an "island" out of an attribute-case helper.
    let bases_with_ops: Vec<&str> = dialect_bases(spec);
    for base in bases_with_ops.iter().copied() {
        // Dedup by projected identifier: two `.td` records may share a mnemonic (an op and its
        // arch-specialised twin), and both would emit the same variant.
        let ops: Vec<&TdOp> = dedup_ops(spec, base);
        if ops.len() < 2 {
            continue;
        }
        let ident = camel(&base.replace("_Op", "").replace('_', "_"));
        let _ = writeln!(o, "/// `{base}` — {} ops, projected from TableGen.", ops.len());
        let _ = writeln!(o, "#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]");
        let _ = writeln!(o, "pub enum {ident}Op {{");
        for op in &ops {
            let _ = writeln!(o, "    /// `{}`", op.mnemonic);
            let _ = writeln!(o, "    {},", camel(&op.mnemonic));
        }
        let _ = writeln!(o, "}}\n");
        let _ = writeln!(o, "impl {ident}Op {{");
        let _ = writeln!(o, "    pub const ALL: &'static [{ident}Op] = &[");
        for op in &ops {
            let _ = writeln!(o, "        {ident}Op::{},", camel(&op.mnemonic));
        }
        let _ = writeln!(o, "    ];");
        let _ = writeln!(o, "    pub fn mnemonic(self) -> &'static str {{");
        let _ = writeln!(o, "        match self {{");
        for op in &ops {
            let _ = writeln!(o, "            {ident}Op::{} => {:?},", camel(&op.mnemonic), op.mnemonic);
        }
        let _ = writeln!(o, "        }}");
        let _ = writeln!(o, "    }}");
        // The row in TD_OPS this variant projects — its operands, results, and `assemblyFormat`.
        // Indexed, so the link cannot be a name lookup that silently finds nothing.
        let _ = writeln!(o, "    /// The `.td` row this variant projects: arity, types, syntax.");
        let _ = writeln!(o, "    pub fn def(self) -> &'static TdOpDef {{");
        let _ = writeln!(o, "        match self {{");
        for op in &ops {
            let idx = spec.ops.iter().position(|x| std::ptr::eq(x, *op)).unwrap_or(0);
            let _ = writeln!(o, "            {ident}Op::{} => &TD_OPS[{idx}],", camel(&op.mnemonic));
        }
        let _ = writeln!(o, "        }}");
        let _ = writeln!(o, "    }}");
        let _ = writeln!(o, "    /// The one place a mnemonic STRING becomes this type.");
        let _ = writeln!(o, "    pub fn from_mnemonic(m: &str) -> Option<Self> {{");
        let _ = writeln!(o, "        Self::ALL.iter().copied().find(|o| o.mnemonic() == m)");
        let _ = writeln!(o, "    }}");
        let _ = writeln!(o, "}}\n");
    }

    // ── ANY OP IN ANY ISLAND, AS ONE TYPE ────────────────────────────────────
    //
    // ⛔⛔ THE PAIR `(dialect: String, mnemonic: String)` IS NOT AN OP, IT IS TWO STRINGS THAT MIGHT
    // BE ONE. It admits `("dataflow", "get_unitt")`, `("Dataflow", "get_unit")` and
    // `("dataflow", "")` — all of which compile, and the first of which we shipped: the bake's very
    // first refusal was *"custom op 'get_unit' is unknown"*, an op name no island declares, caught
    // by MLIR rather than by rustc. As an enum that state cannot be written down.
    let _ = writeln!(o, "/// ANY op of ANY island, as one type — the enum a placed op carries.");
    let _ = writeln!(o, "#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]");
    let _ = writeln!(o, "pub enum IslandOp {{");
    for base in bases_with_ops.iter().copied() {
        if dedup_ops(spec, base).len() < 2 {
            continue;
        }
        let ident = camel(&base.replace("_Op", ""));
        let _ = writeln!(o, "    {ident}({ident}Op),");
    }
    let _ = writeln!(o, "}}\n");
    let _ = writeln!(o, "impl IslandOp {{");
    let _ = writeln!(o, "    /// The dialect prefix this op prints under.");
    let _ = writeln!(o, "    pub fn dialect(self) -> &'static str {{");
    let _ = writeln!(o, "        match self {{");
    for base in bases_with_ops.iter().copied() {
        if dedup_ops(spec, base).len() < 2 {
            continue;
        }
        let ident = camel(&base.replace("_Op", ""));
        let _ = writeln!(
            o,
            "            IslandOp::{ident}(_) => {:?},",
            base.replace("_Op", "").to_ascii_lowercase()
        );
    }
    let _ = writeln!(o, "        }}");
    let _ = writeln!(o, "    }}");
    let _ = writeln!(o, "    pub fn mnemonic(self) -> &'static str {{");
    let _ = writeln!(o, "        match self {{");
    for base in bases_with_ops.iter().copied() {
        if dedup_ops(spec, base).len() < 2 {
            continue;
        }
        let ident = camel(&base.replace("_Op", ""));
        let _ = writeln!(o, "            IslandOp::{ident}(o) => o.mnemonic(),");
    }
    let _ = writeln!(o, "        }}");
    let _ = writeln!(o, "    }}");
    let _ = writeln!(o, "    pub fn def(self) -> &'static TdOpDef {{");
    let _ = writeln!(o, "        match self {{");
    for base in bases_with_ops.iter().copied() {
        if dedup_ops(spec, base).len() < 2 {
            continue;
        }
        let ident = camel(&base.replace("_Op", ""));
        let _ = writeln!(o, "            IslandOp::{ident}(o) => o.def(),");
    }
    let _ = writeln!(o, "        }}");
    let _ = writeln!(o, "    }}");
    // ⭐ THE ONE BOUNDARY. The DSL front end produces text, so text becomes a type exactly here and
    // nowhere after — a `None` is a rule naming an op no island declares, which is a build-time
    // answer rather than an MLIR diagnostic.
    let _ = writeln!(o, "    /// THE ONE PLACE A STRING BECOMES AN OP. `None` = no island declares it.");
    let _ = writeln!(o, "    pub fn parse(dialect: &str, mnemonic: &str) -> Option<Self> {{");
    let _ = writeln!(o, "        match dialect {{");
    for base in bases_with_ops.iter().copied() {
        if dedup_ops(spec, base).len() < 2 {
            continue;
        }
        let ident = camel(&base.replace("_Op", ""));
        let d = base.replace("_Op", "").to_ascii_lowercase();
        let _ = writeln!(
            o,
            "            {d:?} => {ident}Op::from_mnemonic(mnemonic).map(IslandOp::{ident}),"
        );
    }
    let _ = writeln!(o, "            _ => None,");
    let _ = writeln!(o, "        }}");
    let _ = writeln!(o, "    }}");
    let _ = writeln!(o, "}}\n");

    // ── the op STRUCTS ───────────────────────────────────────────────────────
    //
    // ⛔ THE ODS ARGUMENT LIST IS NOT THE WHOLE OP. `Dataflow.td:57` declares `get_unit` with two
    // `StrAttr`s; the C++ attaches `core`, `corelet` and `num_folds` with `setAttr` AFTER
    // construction (DSC2ToDataflowIRUtils.hpp:638-643). A struct generated from the `.td` alone
    // DROPS THEM. So each struct carries its ODS fields AND an `extra` map, and the bridge DSL is
    // what says which extras an op really has.
    let _ = writeln!(o, "/// An SSA handle — what an operand or result refers to. NOT data.");
    let _ = writeln!(o, "pub type Value = u32;");
    let _ = writeln!(o, "/// An affine map, as the two every memory view carries.");
    let _ = writeln!(o, "pub type AffineMap = &'static str;\n");

    for base in &bases_with_ops {
        let ops: Vec<&TdOp> = dedup_ops(spec, base);
        // Prefix with the dialect: `yield` exists in more than one, and bare `YieldOp` collides.
        let dialect = camel(&base.replace("_Op", ""));
        for op in &ops {
            let ident = dialect.clone() + &camel(&op.mnemonic) + "Op";
            // The summary is free prose from the `.td` — one line, no comment terminators.
            let sum: String = op
                .summary
                .chars()
                .filter(|c| *c != '\n' && *c != '\r')
                .collect::<String>()
                .replace("*/", "* /");
            let _ = writeln!(o, "/// `{}` - {}", op.mnemonic, sum.trim());
            let _ = writeln!(o, "/// Projected from {}. ODS fields only; see `extra`.", op.file);
            let _ = writeln!(o, "#[derive(Debug, Clone, Default, PartialEq, Eq)]");
            let _ = writeln!(o, "pub struct {ident} {{");
            for f in op.arguments.iter().chain(op.results.iter()) {
                if f.name.is_empty() {
                    continue;
                }
                let ty = ods_to_rust(&f.ty);
                let name = if f.name == "type" { "ty".to_string() } else { f.name.clone() };
                if ty.is_unknown() {
                    // The ODS spelling may span lines (`Arg<AnyMemRef, "...",\n [MemRead]>`);
                    // a raw newline here breaks the doc comment and the struct with it.
                    let one: String = f.ty.split_whitespace().collect::<Vec<_>>().join(" ");
                    let _ = writeln!(o, "    /// ODS {} - unmapped, carried as a handle.", one);
                }
                let _ = writeln!(o, "    pub {}: {},", name, ty.render());
            }
            let _ = writeln!(o, "    /// Attributes the C++ sets AFTER create, absent from the `.td`.");
            let _ = writeln!(o, "    pub extra: Vec<(&'static str, i64)>,");
            let _ = writeln!(o, "}}\n");
        }
    }

    let un: Vec<String> = spec.unsupported.iter().map(|u| format!("{u:?}")).collect();
    let _ = writeln!(o, "pub static TD_UNSUPPORTED: &[&str] = &[{}];", un.join(", "));
    o
}

/// Dialect op bases, deduped. Shared by the enum and struct emitters so they cannot disagree.
fn dialect_bases(spec: &TdSpec) -> Vec<&str> {
    let mut b: Vec<&str> = spec
        .ops
        .iter()
        .map(|o| o.base.as_str())
        .filter(|b| b.ends_with("_Op"))
        .collect();
    b.sort_unstable();
    b.dedup();
    b
}

/// Ops of one base, deduped by projected identifier.
fn dedup_ops<'a>(spec: &'a TdSpec, base: &str) -> Vec<&'a TdOp> {
    let mut ops: Vec<&TdOp> = Vec::new();
    for op in spec.ops.iter().filter(|o| o.base == base) {
        let id = camel(&op.mnemonic);
        if !ops.iter().any(|k| camel(&k.mnemonic) == id) {
            ops.push(op);
        }
    }
    ops
}
