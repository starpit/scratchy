//! The `.ddl` templates as data, and the types that data is stated in.
//!
//! ⭐ The 32 vendored templates ARE the schedule: expanding them yields 11,218 of the reference's
//! 14,711 schedule nodes. `build.rs` parses them at build time into [`tables`]; nothing here
//! re-derives what a template states.

pub mod conversion;
pub mod tables;
