//! The types `$OUT_DIR/generated.rs` is written against.
//!
//! `build.rs` parses the 32 vendored `.ddl` templates into `PROGRAMS`, `DDL_TEMPLATES` and
//! `MODULES` — 44,306 lines that expand to 11,218 of the reference's 14,711 schedule nodes. Its
//! output names eight types from outside itself; these are them. Every field and variant here is
//! pinned by that output's own use of it, so `build.rs` is their authority.

use crate::arch::Elements;

/// One `mlir::Region` of a parsed `.ddl` module.
///
/// `ddc/ddl/ddl_conversion.h:412-413` keys `region2blocks_` by `mlir::Region*`; a region is
/// identified, never owned, so the port spells the identity and not the pointer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RegionId(pub u32);

/// A `ddl.constraint`'s `cmp=`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstraintCmp {
    /// `"equal"`.
    Equal,
    /// `"less"`.
    Less,
}

/// An extent stated by a `ddl.constraint`'s `value=`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Extent(pub i64);

/// The `min_num_cores=` a template demands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CoreCount(pub u32);

/// One `ddl.constraint` as a module states it — `verifyDdlConstraints`
/// (`ddc/ddl/ddl_conversion.cpp:2553`) walks these in pre-order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DdlConstraint {
    /// `min_num_cores=`.
    MinNumCores(CoreCount),
    /// `min_num_valid=` / `max_num_valid=`, already at their `value_or` defaults.
    NumValid {
        /// `min_num_valid=`, defaulting to ZERO.
        min: u32,
        /// `max_num_valid=`, defaulting to ONE HUNDRED and not to unbounded.
        max: u32,
    },
    /// `relative_op_order=true`.
    RelativeOpOrder,
    /// `property=` + `dim_idx=` + `cmp="equal"` + `value=`.
    StickSizeAt {
        /// `property=="slice"`.
        slice: bool,
        /// `dim_idx=`.
        dim_idx: usize,
        /// `value=`.
        value: Elements,
    },
    /// `property=` + `cmp="equal"` with no `dim_idx=`.
    StickSizesAgree {
        /// `property=="slice"`.
        slice: bool,
    },
    /// `cmp=` + `value=` with no `property=`.
    DimSize {
        /// `cmp=`.
        cmp: ConstraintCmp,
        /// `value=`.
        value: Extent,
    },
}

/// A `ddl.padded_dimension` — a padded dim, the unpadded dim it pads, and its meta dims.
///
/// The two lists are borrowed because `generated.rs` states them as `&'static` slices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PaddedDimension<'d> {
    /// `primary=` — the unpadded dim this one pads.
    pub primary: super::tables::NameId,
    /// `padding=`, admitting only `PadFront`, `PadValid` and `PadBack`.
    pub padding: &'d [super::tables::NameId],
    /// `window=`, admitting only `WindowDim`, `Stride` and `Dilation`.
    pub window: &'d [super::tables::NameId],
}
