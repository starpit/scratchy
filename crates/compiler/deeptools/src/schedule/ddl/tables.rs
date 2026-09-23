//! The censused `.ddl` templates, as `build.rs` writes them.
//!
//! `build.rs` walks the 32 vendored templates and writes `$OUT_DIR/generated.rs`: `PROGRAMS` (the
//! statement trees), `DDL_TEMPLATES` (op-func × generation → candidates) and `MODULES` (each
//! template's binds, padded dims, constraints and regions), plus the enums those are stated in —
//! `StmtKind`, `Unit`, `Memory`, `ComputeType`, `OpFunc`, `DataConnect`, `NameId`, `Operand`,
//! `Stmt`, `Program` and the rest.
//!
//! ⛔ THIS FILE IS THE ONLY READER OF THAT OUTPUT. Before it existed, `build.rs` regenerated
//! 44,306 lines on every build that nothing included, so the compiler never saw them and every
//! gate was green over data with no consumer.

#![allow(clippy::unreadable_literal)]

include!(concat!(env!("OUT_DIR"), "/generated.rs"));
