//! The arch facts `build.rs` names when it censuses the vendored `.ddl` templates.
//!
//! `build.rs` writes `$OUT_DIR/generated.rs` against `crate::arch::IsaGen` and
//! `crate::arch::Elements`; both are declared here because that output is their only author.

/// A GENERATION A `.ddl` TEMPLATE CAN BE SELECTED FOR — the two the crate has features for.
///
/// ⛔ NOT `sys_arch_spec::fields::Gen` (five) nor `ddl/selection.rs:38` (three, because
/// `opFuncToDdlTemplate` also tags candidates `MPW4_ISA`). These are the two a core arch can be
/// *compared against*, which is how `build.rs:2567-2571` resolves the arch skip. Widening it turns
/// "never modelled" into "matched".
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum IsaGen {
    /// `RCUDD1A_ISA`, the crate's default.
    Rcudd1a,
    /// `SEN1P5_ISA`.
    Sen1p5,
}

impl IsaGen {
    /// The modelled generation a core arch selects templates as, if it is one of the two.
    ///
    /// [`None`] is `Gen::Mpw2`/`Mpw3`/`Mpw4` — a core arch no vendored candidate is comparable
    /// against, which is the *"DDL found but not suitable"* answer and not *"no DDL available"*.
    #[must_use]
    pub fn from_core_arch(core_arch: sys_arch_spec::fields::Gen) -> Option<Self> {
        match core_arch {
            sys_arch_spec::fields::Gen::Rcudd1a => Some(Self::Rcudd1a),
            sys_arch_spec::fields::Gen::Sen1p5 => Some(Self::Sen1p5),
            _ => None,
        }
    }
}

/// A count of elements — the unit a `ddl.constraint`'s `value=` states a stick size in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Elements(pub u64);
