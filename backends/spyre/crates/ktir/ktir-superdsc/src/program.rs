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

/// A constant table: a whole tensor of values stated at compile time.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TableId(u32);

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
id!(TableId);

/// Where an operand's elements live.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Store {
    Param(ParamId),
    Local(LocalId),
    Const(ConstId),
    /// A tensor whose every element the program states at compile time (`tensor.generate`):
    /// data the bake stages, never computed on the device.
    Table(TableId),
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
    /// `x` as this format holds it: fp32 to nearest, fp16 and fp8 e4m3 to nearest even (e4m3
    /// with 3 mantissa bits, normals down to 2^-6, subnormals in steps of 2^-9, saturating at
    /// ±448, its largest finite value), int32 wrapping.
    pub fn round(self, x: f64) -> f64 {
        match self {
            Format::Fp32 => f64::from(x as f32),
            Format::Fp16 => f64::from(half::f16::from_f32(x as f32).to_f32()),
            Format::Int32 => f64::from(x as i64 as i32),
            Format::Fp8 => {
                let x = x as f32;
                if x == 0.0 || x.is_nan() {
                    return f64::from(x);
                }
                let a = f64::from(x.abs()).min(448.0);
                let quantum = if a >= 2f64.powi(-6) {
                    2f64.powi(a.log2().floor() as i32 - 3)
                } else {
                    2f64.powi(-9)
                };
                let q = a / quantum;
                let r = q.floor();
                let up = q - r > 0.5 || (q - r == 0.5 && r % 2.0 != 0.0);
                ((if up { r + 1.0 } else { r }) * quantum).min(448.0).copysign(f64::from(x))
            }
        }
    }

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

/// One value as a store of some format holds it: an integer format's exactly, a float format's
/// as its f32 bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Scalar {
    Float(u32),
    Int(i32),
}

impl Scalar {
    /// `x` as `format` holds it: an integer format takes it only when it is an i32 exactly; a
    /// float format rounds it ([`Format::round`]).
    pub fn new(x: f64, format: Format) -> Option<Scalar> {
        match format {
            Format::Int32 => {
                let i = x as i32;
                (f64::from(i) == x).then_some(Scalar::Int(i))
            }
            Format::Fp16 | Format::Fp32 | Format::Fp8 => Some(Scalar::Float((format.round(x) as f32).to_bits())),
        }
    }

    pub fn as_f64(self) -> f64 {
        match self {
            Scalar::Float(b) => f64::from(f32::from_bits(b)),
            Scalar::Int(i) => f64::from(i),
        }
    }
}

/// A constant stick: `value` in every lane, held in `format`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ConstStick {
    pub format: Format,
    pub value: Scalar,
}

/// A lowered function.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Program {
    /// Each parameter's format, by [`ParamId`]; `None` for a parameter the function never reads
    /// or writes.
    pub params: Vec<Option<Format>>,
    pub locals: Vec<Local>,
    pub consts: Vec<ConstStick>,
    pub tables: Vec<Table>,
    pub segments: Vec<Segment>,
}

/// A constant table of `shape`, held in `format`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Table {
    pub format: Format,
    pub shape: Vec<Extent>,
    pub data: TableData,
}

/// What a constant table holds.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum TableData {
    /// `base + Σ index[d]·steps[d]` at index `index`: an affine table, stated by its form (every
    /// value exact in the table's format, checked where it is made).
    Ramp { base: i64, steps: Vec<i64> },
    /// Each element, row-major.
    Values(Vec<Scalar>),
}

impl Table {
    /// The element at row-major position `flat`.
    pub fn get(&self, flat: u64) -> f64 {
        match &self.data {
            TableData::Values(v) => v[flat as usize].as_f64(),
            TableData::Ramp { base, steps } => {
                let mut rest = flat;
                let mut v = *base;
                for d in (0..self.shape.len()).rev() {
                    let n = self.shape[d].get();
                    v += (rest % n) as i64 * steps[d];
                    rest /= n;
                }
                v as f64
            }
        }
    }

    /// The least and greatest value the table holds.
    pub fn range(&self) -> (f64, f64) {
        match &self.data {
            TableData::Values(v) => v.iter().map(|s| s.as_f64()).fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), x| {
                (lo.min(x), hi.max(x))
            }),
            TableData::Ramp { base, steps } => {
                let (mut lo, mut hi) = (*base, *base);
                for (e, &s) in self.shape.iter().zip(steps) {
                    let span = (e.get() as i64 - 1) * s;
                    lo += span.min(0);
                    hi += span.max(0);
                }
                (lo as f64, hi as f64)
            }
        }
    }
}
