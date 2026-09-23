//! IBM Spyre deeptools. The C++ authority is `/Users/nickm/git/deeptools-src` at `a0d29abbed`.
//!
//! ⛔⛔ THE PORT WAS DELETED A SECOND TIME ON 2026-09-23 (tag `scheduler-before-nuke-2026-09-23`),
//! and the cause is ONE decision, not a list of defects: **the scope was hand-built instead of
//! derived from the CodeQL dependency oracle.** A hand-picked scope cannot be CLOSED, so it leaks,
//! and every leak costs a wave to discover. It leaked at least six times — 36 of L3's 128 methods
//! scheduled; `superdsc.h`/`dscdefn.h`, 717 lines the scheduler cannot be written without, never
//! vendored; the translator sized at 1,739 lines when it is 9,300; V3 scoped 4 files of 6; the
//! DataflowIR vocabulary absent, so every lowering blocked on RULE 2; and a regex field census blind
//! to three declaration forms. Each is a question an oracle answers in one pass.
//!
//! ⭐ WHAT SURVIVES IS DATA AND THE GENERATORS THAT READ IT — no ported C++ at all:
//!   - `ddl_templates/` (32 `.ddl`) + `ddl/` — the templates ARE the schedule
//!   - `td/` (22 `.td`) + [`td`] — the MLIR dialects, 452 ops, generated and never hand-written
//!   - `build.rs` — emits both into `$OUT_DIR`
//!   - [`arch`] and `schedule::ddl::conversion` — the eight declarations `build.rs`'s own output
//!     names, without which the generated code does not compile
//!
//! The next port starts from an oracle-derived scope or it does not start.

pub mod arch;
pub mod schedule;
pub mod td;
