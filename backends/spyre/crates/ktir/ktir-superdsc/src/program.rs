// SPDX-License-Identifier: Apache-2.0
//! A KTIR function as the device runs it: device ops over strided views of three kinds of storage,
//! in segments that keep the function's own structure.
//!
//! A [`Segment::Straight`] runs once. A [`Segment::Walk`] is a top-level `scf.for` with a constant
//! trip count: its body is lowered once and the card relaunches it once per trip. What differs
//! between trips is an address, never an op: an operand whose origin moves with the trip is
//! [`Address::Stepped`], and the type of a straight segment's operands has no way to say so.
//! The values a walk carries from one trip to the next are buffers ([`Carried`]): each trip reads
//! the buffer and writes its next value back into it.

use std::num::{NonZeroU32, NonZeroU64};

use ktir_core::ddl_binds::DeviceOp;

use crate::qty::{Elems, Extent, Stride};

/// The `i`-th parameter of the function: a buffer the launch binds an address to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ParamId(u32);

/// An intermediate buffer the lowering owns.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LocalId(u32);

/// A constant stick: one value in every lane.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ConstId(u32);

macro_rules! id {
    ($t:ident) => {
        impl $t {
            pub const fn new(i: u32) -> $t {
                $t(i)
            }
            pub const fn get(self) -> u32 {
                self.0
            }
        }
    };
}
id!(ParamId);
id!(LocalId);
id!(ConstId);

/// Where an operand's elements live.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Store {
    Param(ParamId),
    Local(LocalId),
    Const(ConstId),
}

/// The format a buffer's elements are held in on the device.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Format {
    Fp16,
    Fp32,
    Fp8,
    Int32,
}

impl Format {
    pub const fn bytes_per_elem(self) -> NonZeroU64 {
        match self {
            Format::Fp8 => NonZeroU64::MIN,
            Format::Fp16 => NonZeroU64::new(2).unwrap(),
            Format::Fp32 | Format::Int32 => NonZeroU64::new(4).unwrap(),
        }
    }
}

/// One axis of a view: `extent` positions, `stride` elements apart.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Axis {
    pub extent: Extent,
    pub stride: Stride,
}

/// Where a walk-body operand's view starts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Address {
    /// The same element on every trip.
    Fixed(Elems),
    /// `at_trip0 + trip · per_trip` on trip `trip`.
    Stepped { at_trip0: Elems, per_trip: NonZeroU64 },
}

/// A strided view over a store's elements, axes outermost first: logical index `i` is element
/// `origin + Σ i[a]·stride[a]`. `O` is how the origin is stated: an [`Elems`] in a straight segment,
/// an [`Address`] in a walk's body.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct View<O> {
    pub origin: O,
    pub axes: Vec<Axis>,
}

/// One operand of a device op: a view over a store, its axes aligned to the op's space.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Operand<O> {
    pub store: Store,
    pub view: View<O>,
    pub format: Format,
    /// `Some` for an operand reached through an index tensor.
    pub gather: Option<Gather<O>>,
}

/// An access whose address the device reads from an index tensor.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Gather<O> {
    /// Along view axis `axis`, position `p` is row `index[p]` (a row is one step of the axis's
    /// stride), not row `p`; the indexed buffer holds `rows` rows along the axis.
    Rows {
        axis: usize,
        index: Box<Operand<O>>,
        rows: Extent,
    },
    /// The whole view starts `index[0] · step` elements further on; the buffer holds `bound`
    /// such steps.
    Offset {
        index: Box<Operand<O>>,
        step: Stride,
        bound: Extent,
    },
}

/// What a device op computes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OpFunc {
    /// A pointwise, convert or reduce op of the generation's device-facts table. A reduction folds
    /// every space axis its output does not spread over.
    Device(DeviceOp),
    /// The output's elements are the input's.
    Copy,
    /// `out[b…, m, n] = Σ_k a[b…, m, k] · b[b…, k, n]` over the space `(b…, m, n, k)`.
    Contraction,
}

/// One device op over its iteration space, outermost axis first. A pointwise or reduce op's
/// operands have one view axis per space axis; a contraction's are `a: (b…, m, k)`,
/// `b: (b…, k, n)`, `out: (b…, m, n)`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Node<O> {
    pub func: OpFunc,
    pub space: Vec<Extent>,
    pub inputs: Vec<Operand<O>>,
    pub output: Operand<O>,
}

/// A top-level constant-trip `scf.for`: `body` relaunched `trips` times (at least two; a loop of one
/// trip is straight code). Each value the loop carries is one buffer in `carried`: the straight
/// segment before the walk writes its initial value into it, every trip reads it, and the body's
/// last ops write the trip's next value back into it.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Walk {
    pub trips: NonZeroU32,
    pub carried: Vec<LocalId>,
    pub body: Vec<Node<Address>>,
}

/// A run of the function's ops.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Segment {
    Straight(Vec<Node<Elems>>),
    Walk(Walk),
}

/// An intermediate's storage: `elements` of `format`, contiguous.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Local {
    pub elements: Elems,
    pub format: Format,
}

/// A constant stick's value, stated in f32 bits (an integer constant exactly so), held in `format`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ConstStick {
    pub format: Format,
    pub bits: u32,
}

/// A lowered function.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Program {
    /// Each parameter's format, by [`ParamId`]; `None` for a parameter the function never reads
    /// or writes.
    pub params: Vec<Option<Format>>,
    pub locals: Vec<Local>,
    pub consts: Vec<ConstStick>,
    pub segments: Vec<Segment>,
}
