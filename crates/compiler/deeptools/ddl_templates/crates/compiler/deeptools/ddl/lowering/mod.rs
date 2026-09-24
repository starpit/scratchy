//! ⭐⭐⭐ THE LOWERING GENERATOR — one function per (template, bind), no interpreter.
//!
//! # WHY THIS IS A GENERATOR AND NOT A TABLE
//!
//! The universe of subtile ops is closed and the universe of `ddl` ops is closed, so every question
//! the lowering asks has an answer HERE, at expansion, where the template's text still exists: which
//! unit a statement runs on, which `data_connect=` each port is, which of them a compute reads,
//! whether a `computetype=` is a MAC. Emitting a table and asking those questions again at run time
//! is an INTERPRETER, and an interpreter's misses are silent — `_ => {}` over statement kinds and
//! `else { continue }` over port lookups is how every `data_connect` join in 32 templates vanished
//! without a word and how an SFP unit came out with two receives and no compute.
//!
//! # THE `.ddl` IS THE DATA, THE C++ IS THE RULE SET
//!
//! ⛔⛔ 30,880 statements across 218 programs are the SCHEDULE, and it varies per op. The ten-odd
//! rules saying what each statement kind BECOMES are FIXED and live only in the C++. So the split of
//! this module is that split:
//!
//! - [`rules`] — the C++ knowledge, encoded once, every entry quoted in `LOWERING_RULES.md`.
//! - [`walk`] — one program's statements turned into calls, using those rules.
//! - this file — the driver and the census of what could not be emitted.
//!
//! ⛔ A PROGRAM THIS CANNOT YET EMIT GETS NO FUNCTION AT ALL, its reason frozen in the generated
//! `DEFERRED`. Not a stub, not an `Identity` stand-in: the absence is a compile error at whatever
//! tries to call it, which is the loudest possible form of "not built yet".
//!
//! # ⭐⭐ AN UNUSED `p_<connect>` IN THE GENERATED CODE IS A DANGLING COMPUTE, AND MUST STAY LOUD
//!
//! Because a producer binds a named local and a consumer reads it, rustc's own unused-variable
//! warning reports a `data_connect` that has a producer and NO consumer — and that is a program
//! dbo-opt refuses: `BinaryOpLowering` resolves the destination in `fillOpInfo` and fails before
//! `setReuseInformation` (`VectorChainToSentientPESFP.cpp:332-338`), resurfacing as *"Dangling
//! non-compute op has no use"*. The reference checks the other direction — an empty PRODUCER set
//! (`ddcv1.cpp:3323-25`) — so this catches the half it does not.
//!
//! ⛔ SO NEVER SILENCE ONE WITH `let _` OR AN UNDERSCORE PREFIX. Either the consuming statement kind
//! is still deferred (the warning clears when it lands) or the template's own graph is being read
//! wrong, and both are things to see.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;

use crate::{OperandRef, Program, attr_bool, attr_int, attr_str, attr_strs, dataflow, ints};

mod rules;
mod walk;

use walk::lowering_body;
/// ⭐⭐⭐ THE LOWERING, GENERATED — one function per (template, bind), no interpreter.
///
/// # WHY THIS IS CODE AND NOT A TABLE
///
/// The universe of subtile ops is closed and the universe of `ddl` ops is closed, so every question
/// this lowering asks has an answer HERE, where the template's text still exists: which unit a
/// statement runs on, which `data_connect=` each port is, which of them a compute reads, whether a
/// `computetype=` is a MAC. Emitting a table and asking those questions again at run time is an
/// INTERPRETER, and an interpreter's misses are silent — `_ => {}` over statement kinds and
/// `else { continue }` over port lookups is how every `data_connect` join in 32 templates vanished
/// without a word and how an SFP unit came out with two receives and no compute.
///
/// ⛔⛔ SO THE PORTS BECOME NAMED LOCALS AND THE JOIN HAPPENS HERE. A producer emits
/// `let p_<connect> = ...`; a consumer reads that local. `createDataConnectMetadata` requires every
/// `data_connect` to have a non-empty producer set (`ddcv1.cpp:3323-25`) — as generated code, that is
/// Rust's own name resolution, and it cannot be faked or logged and carried on.
///
/// ⛔ A PROGRAM THIS CANNOT YET EMIT GETS NO FUNCTION AT ALL, and the reason is printed and frozen in
/// [`DEFERRED`]. Not a stub, not an `Identity` stand-in: the absence is a compile error at whatever
/// tries to call it, which is the loudest possible form of "not built yet".
pub fn codegen_lowering(out: &mut String, programs: &[Program]) {
    out.push_str(
        r#"
/// THE GENERATED LOWERING — one function per (template, bind).
///
/// ⭐ EVERY FUNCTION HERE IS STRAIGHT-LINE. No loop over statements, no match on a kind, no port
/// lookup: `build.rs` resolved all of it and wrote the calls out.
pub mod lower {
    use crate::arch::Arch;
    use crate::bridges::subtile_to_dataflow_ir::emit::{Emit, Unresolved};
    use crate::bridges::subtile_to_dataflow_ir::shape::Tile;
    use crate::bridges::subtile_to_dataflow_ir::emit::{Arm, Place, Pred};
    use crate::islands::dataflow_ir::dialects::vectorchain::{
        BinaryOp, CompareOp, EstimateKind, EstimateVersion,
    };
    use crate::islands::dataflow_ir::link;
    use crate::islands::dataflow_ir::memory;
    use crate::generated::SyncSignal;
    use crate::islands::dataflow_ir::ty::ElemType;
    use crate::units::DfirUnit;

"#,
    );

    let mut deferred: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut emitted = 0usize;
    let mut dispatch: Vec<(String, String, String)> = Vec::new();
    for program in programs {
        let name = lowering_fn_name(program);
        match lowering_body(program) {
            Ok(body) => {
                let _ = writeln!(
                    out,
                    "    /// `{}.ddl` — `{}`, bind `{}`.\n    pub fn {name}<A: Arch>(e: &mut Emit) {{\n{body}    }}\n",
                    program.stem, program.op_func, program.bind
                );
                dispatch.push((
                    crate::ident_of(&program.stem),
                    program.bind.clone(),
                    name.clone(),
                ));
                emitted += 1;
            }
            Err(why) => deferred.entry(why).or_default().push(name),
        }
    }

    // ⭐⭐⭐ THE DISPATCH, GENERATED TOO — how a `Program` reaches its own lowering.
    //
    // ⛔ KEYED ON (template, bind), WHICH IS WHAT A `Program` CARRIES. `OpFunc::program(gen, format)`
    // already resolves an op-func AT A PRECISION to one `&'static Program`, so keying on that
    // program's own two identifying fields makes this agree with the selection BY CONSTRUCTION rather
    // than by a second copy of the format-matching rules.
    //
    // ⛔ AND `None` IS THE DEFERRED CASE, NOT A FALLBACK. A program with no generated body has no row
    // here, so the caller learns "not built" and cannot be handed something adjacent that happens to
    // typecheck. `bind` distinguishes two binds sharing an op-func — `%stradd_op` and `%stradd2_op`
    // are both `stridedadd` and read different operands (`bmm.ddl:79-80`).
    out.push_str(
        "\n    /// WHICH GENERATED LOWERING SERVES A `Program` — keyed on the two fields that identify\n    \
         /// it. `None` means the generator deferred it, and the reason is in [`super::DEFERRED`].\n    \
         pub fn lower_for<A: Arch>(\n        \
             template: super::Template,\n        \
             bind: &str,\n    \
         ) -> Option<fn(&mut Emit)> {\n        \
             match (template, bind) {\n",
    );
    for (template, bind, name) in &dispatch {
        let _ = writeln!(
            out,
            "            (super::Template::{template}, {bind:?}) => Some({name}::<A>),"
        );
    }
    out.push_str("            _ => None,\n        }\n    }\n");
    out.push_str("}\n\n");

    // 🛑 THE LEDGER OF WHAT IS NOT GENERATED, so "step 1 is done" is a number and not a claim.
    out.push_str(
        "/// WHAT THE GENERATOR COULD NOT EMIT, AND WHY — ratcheted DOWN, never up.\n\
         ///\n\
         /// ⛔ EACH ROW IS A PROGRAM WITH NO LOWERING FUNCTION. Nothing stands in for it: calling it\n\
         /// is a compile error, which is what makes the absence impossible to mistake for a lowering\n\
         /// that ran.\n\
         pub const DEFERRED: &[(&str, &[&str])] = &[\n",
    );
    for (why, names) in &deferred {
        let _ = writeln!(
            out,
            "    ({:?}, &[{}]),",
            why,
            names
                .iter()
                .map(|n| format!("{n:?}"))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    out.push_str("];\n");

    let blocked: usize = deferred.values().map(Vec::len).sum();
    println!(
        "cargo:warning=lowering generated for {emitted} of {} programs; {blocked} deferred across {} reasons",
        programs.len(),
        deferred.len()
    );
    for (why, names) in &deferred {
        println!("cargo:warning=  deferred ({}): {why}", names.len());
    }
}

/// A FUNCTION NAME THAT SAYS WHICH TEMPLATE AND WHICH BIND.
fn lowering_fn_name(program: &Program) -> String {
    let scrub = |text: &str| -> String {
        text.chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() {
                    c.to_ascii_lowercase()
                } else {
                    '_'
                }
            })
            .collect()
    };
    format!("{}__{}", scrub(&program.stem), scrub(&program.bind))
}
