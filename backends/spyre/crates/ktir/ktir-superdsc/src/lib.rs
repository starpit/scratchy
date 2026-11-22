// SPDX-License-Identifier: Apache-2.0
//! KTIR → SuperDSC: one lowering of any KTIR function to the descriptors `dxp` compiles.
//!
//! The lowering reads everything it needs from the function itself: each op's iteration space and
//! operands from the views the function constructs, each value's format from its type, each
//! constant from the splat that states it. Nothing here knows which kernel or which model a
//! function came from, so a new kernel is lowered by the same code as every other.
//!
//! A function's structure is kept, never flattened: a top-level `scf.for` with a constant trip
//! count is a [`program::Segment::Walk`], lowered once and relaunched by the card once per trip,
//! with every address that moves between trips stated as a [`program::Address::Stepped`]. There is
//! no representation of a trip written out twice.

pub mod eval;
pub mod lift;
pub mod lin;
pub mod program;
pub mod qty;
pub mod view;
