// SPDX-License-Identifier: Apache-2.0
//! A lowered [`Program`], run on host buffers: what its device ops compute, under one stated
//! numeric model, so a lowering can be checked against a reference of the kernel's own math.
//!
//! THE MODEL. Every float op computes in f32 and rounds its result to the output's format
//! (fp16 round-to-nearest-even; fp8 e4m3 is not modelled and is refused by name); an integer op
//! wraps in i32; a compare writes 1 or 0; a reduction folds in f32 from its identity; a
//! contraction accumulates in f32. The device's dl16 format and its transcendental units are
//! not bit-modelled: what this checks is that the lowering addresses, orders and combines the
//! values the KTIR states — views, gathers, parallel axes, walks, reductions, casts.

use half::f16;
use ktir_core::ddl_binds::DeviceOp;

use crate::program::{Address, Format, Gather, Node, OpFunc, Operand, Program, Segment, Store};
use crate::qty::Elems;

/// Why a program did not run.
#[derive(Clone, Debug, PartialEq)]
pub enum EvalError {
    /// A buffer the program reads was not given, or is the wrong length.
    Param { param: usize, want: u64, got: usize },
    /// An address past the end of its buffer.
    OutOfBounds { node: usize, store: Store, at: i64, len: usize },
    /// An op this model does not state.
    Unmodelled(OpFunc),
    /// A format this model does not state.
    Format(Format),
}

/// Round `x` to what `format` holds.
fn round(x: f64, format: Format) -> Result<f64, EvalError> {
    Ok(match format {
        Format::Fp32 => f64::from(x as f32),
        Format::Fp16 => f64::from(f16::from_f32(x as f32).to_f32()),
        Format::Int32 => f64::from(x as i64 as i32),
        Format::Fp8 => return Err(EvalError::Format(Format::Fp8)),
    })
}

struct Run<'p> {
    p: &'p Program,
    params: Vec<Vec<f64>>,
    locals: Vec<Vec<f64>>,
    tables: Vec<Vec<f64>>,
}

/// The element offset an origin names on `trip`.
trait Origin {
    fn at(&self, trip: u64) -> i64;
}

impl Origin for Elems {
    fn at(&self, _: u64) -> i64 {
        self.get() as i64
    }
}

impl Origin for Address {
    fn at(&self, trip: u64) -> i64 {
        match *self {
            Address::Fixed(e) => e.get() as i64,
            Address::Stepped { at_trip0, per_trip } => (at_trip0.get() + trip * per_trip.get()) as i64,
        }
    }
}

impl Run<'_> {
    fn buffer(&self, s: Store) -> &[f64] {
        match s {
            Store::Param(p) => &self.params[p.get() as usize],
            Store::Local(l) => &self.locals[l.get() as usize],
            Store::Const(_) => &[],
            Store::Table(t) => &self.tables[t.get() as usize],
        }
    }

    /// The element of `o` at view position `pos`, on `trip`: its address, the device-read
    /// index folded in.
    fn address<O: Origin>(&self, ni: usize, o: &Operand<O>, pos: &[u64], trip: u64) -> Result<i64, EvalError> {
        let mut at = o.view.origin.at(trip);
        for (a, &i) in o.view.axes.iter().zip(pos) {
            at += (i * a.stride.get()) as i64;
        }
        match &o.gather {
            None => {}
            Some(Gather::Rows { axis, index, .. }) => {
                // Position `p` along `axis` is row `index[p]`: the index is read at the same
                // leading (parallel) positions and `p` along its own last axis.
                let stride = o.view.axes[*axis].stride.get() as i64;
                at -= pos[*axis] as i64 * stride;
                let lead = index.view.axes.len() - 1;
                let mut ipos: Vec<u64> = pos[..lead].to_vec();
                ipos.push(pos[*axis]);
                at += self.read(ni, index, &ipos, trip)? as i64 * stride;
            }
            Some(Gather::Offset { index, step, .. }) => {
                let lead = index.view.axes.len() - 1;
                let mut ipos: Vec<u64> = pos[..lead].to_vec();
                ipos.push(0);
                at += self.read(ni, index, &ipos, trip)? as i64 * step.get() as i64;
            }
        }
        Ok(at)
    }

    fn read<O: Origin>(&self, ni: usize, o: &Operand<O>, pos: &[u64], trip: u64) -> Result<f64, EvalError> {
        if let Store::Const(c) = o.store {
            return Ok(f64::from(self.p.consts[c.get() as usize].value()));
        }
        let at = self.address(ni, o, pos, trip)?;
        let buf = self.buffer(o.store);
        usize::try_from(at)
            .ok()
            .and_then(|i| buf.get(i))
            .copied()
            .ok_or(EvalError::OutOfBounds { node: ni, store: o.store, at, len: buf.len() })
    }

    fn write<O: Origin>(&mut self, ni: usize, o: &Operand<O>, pos: &[u64], trip: u64, v: f64) -> Result<(), EvalError> {
        let at = self.address(ni, o, pos, trip)?;
        let v = round(v, o.format)?;
        let buf = match o.store {
            Store::Param(p) => &mut self.params[p.get() as usize],
            Store::Local(l) => &mut self.locals[l.get() as usize],
            Store::Const(_) | Store::Table(_) => return Err(EvalError::Unmodelled(OpFunc::Copy)),
        };
        let len = buf.len();
        let slot = usize::try_from(at)
            .ok()
            .and_then(|i| buf.get_mut(i))
            .ok_or(EvalError::OutOfBounds { node: ni, store: o.store, at, len })?;
        *slot = v;
        Ok(())
    }

    fn node<O: Origin>(&mut self, ni: usize, n: &Node<O>, trip: u64) -> Result<(), EvalError> {
        let space: Vec<u64> = n.space.iter().map(|e| e.get()).collect();
        let total: u64 = space.iter().product();
        let positions = (0..total).map(|mut flat| {
            let mut pos = vec![0u64; space.len()];
            for a in (0..space.len()).rev() {
                pos[a] = flat % space[a];
                flat /= space[a];
            }
            pos
        });
        match n.func {
            OpFunc::Contraction => {
                let r = space.len();
                let k = space[r - 1];
                for pos in positions.filter(|p| p[r - 1] == 0) {
                    let (lead, m, nn) = (&pos[..r - 3], pos[r - 3], pos[r - 2]);
                    let mut acc = 0f32;
                    for kk in 0..k {
                        let a = [lead, &[m, kk]].concat();
                        let b = [lead, &[kk, nn]].concat();
                        acc += self.read(ni, &n.inputs[0], &a, trip)? as f32 * self.read(ni, &n.inputs[1], &b, trip)? as f32;
                    }
                    let out = [lead, &[m, nn]].concat();
                    self.write(ni, &n.output, &out, trip, f64::from(acc))?;
                }
            }
            OpFunc::Device(op @ (DeviceOp::Sum | DeviceOp::Max | DeviceOp::Min)) => {
                let identity = match op {
                    DeviceOp::Sum => 0.0,
                    DeviceOp::Max => f64::NEG_INFINITY,
                    _ => f64::INFINITY,
                };
                let positions: Vec<Vec<u64>> = positions.collect();
                let mut acc: std::collections::HashMap<i64, f32> = std::collections::HashMap::new();
                for pos in &positions {
                    let at = self.address(ni, &n.output, pos, trip)?;
                    let x = self.read(ni, &n.inputs[0], pos, trip)? as f32;
                    let a = acc.entry(at).or_insert(identity as f32);
                    *a = match op {
                        DeviceOp::Sum => *a + x,
                        DeviceOp::Max => a.max(x),
                        _ => a.min(x),
                    };
                }
                for pos in &positions {
                    let at = self.address(ni, &n.output, pos, trip)?;
                    self.write(ni, &n.output, pos, trip, f64::from(acc[&at]))?;
                }
            }
            func => {
                for pos in positions {
                    let ins: Vec<f64> = n
                        .inputs
                        .iter()
                        .map(|o| self.read(ni, o, &pos, trip))
                        .collect::<Result<_, _>>()?;
                    let v = pointwise(func, &ins)?;
                    self.write(ni, &n.output, &pos, trip, v)?;
                }
            }
        }
        Ok(())
    }
}

/// One pointwise op on its operands' values.
fn pointwise(func: OpFunc, x: &[f64]) -> Result<f64, EvalError> {
    let f = |i: usize| x[i] as f32;
    let b = |c: bool| if c { 1.0 } else { 0.0 };
    let i32w = |i: usize| x[i] as i64 as i32;
    Ok(match func {
        OpFunc::Copy => x[0],
        OpFunc::Device(op) => match op {
            DeviceOp::Add => f64::from(f(0) + f(1)),
            DeviceOp::Sub => f64::from(f(0) - f(1)),
            DeviceOp::Multiply => f64::from(f(0) * f(1)),
            DeviceOp::RealDiv => f64::from(f(0) / f(1)),
            DeviceOp::Maximum => f64::from(f(0).max(f(1))),
            DeviceOp::Minimum => f64::from(f(0).min(f(1))),
            DeviceOp::Equal => b(f(0) == f(1)),
            DeviceOp::NotEqual => b(f(0) != f(1)),
            DeviceOp::GreaterThan => b(f(0) > f(1)),
            DeviceOp::GreaterEqual => b(f(0) >= f(1)),
            DeviceOp::LesserThan => b(f(0) < f(1)),
            DeviceOp::LesserEqual => b(f(0) <= f(1)),
            DeviceOp::Where3 => {
                if x[0] != 0.0 {
                    x[1]
                } else {
                    x[2]
                }
            }
            DeviceOp::Neg => f64::from(-f(0)),
            DeviceOp::Abs => f64::from(f(0).abs()),
            DeviceOp::Exp => f64::from(f(0).exp()),
            DeviceOp::Log => f64::from(f(0).ln()),
            DeviceOp::Sqrt => f64::from(f(0).sqrt()),
            DeviceOp::Rsqrt => f64::from(1.0 / f(0).sqrt()),
            DeviceOp::Reciprocal => f64::from(1.0 / f(0)),
            DeviceOp::Tanh => f64::from(f(0).tanh()),
            DeviceOp::Sigmoid => f64::from(1.0 / (1.0 + (-f(0)).exp())),
            DeviceOp::Silu => f64::from(f(0) / (1.0 + (-f(0)).exp())),
            DeviceOp::Floor => f64::from(f(0).floor()),
            DeviceOp::Identity | DeviceOp::Dl16ToFp32 | DeviceOp::Fp32ToDl16 => x[0],
            DeviceOp::AddI32ToI32 => f64::from(i32w(0).wrapping_add(i32w(1))),
            DeviceOp::MulI32ToI32 => f64::from(i32w(0).wrapping_mul(i32w(1))),
            DeviceOp::Int32ToFp32 => f64::from(i32w(0) as f32),
            DeviceOp::Fp32ToInt32 => f64::from(f(0) as i32),
            other => return Err(EvalError::Unmodelled(OpFunc::Device(other))),
        },
        OpFunc::Contraction => return Err(EvalError::Unmodelled(func)),
    })
}

/// Run `p` on `params` (each parameter's buffer as its element values, in its own format;
/// an unused parameter's buffer may be empty) and answer every parameter's buffer afterwards.
pub fn run(p: &Program, params: Vec<Vec<f64>>) -> Result<Vec<Vec<f64>>, EvalError> {
    let mut r = Run {
        p,
        params,
        locals: p.locals.iter().map(|l| vec![0.0; l.elements.get() as usize]).collect(),
        tables: p
            .tables
            .iter()
            .map(|t| t.bits.iter().map(|&b| f64::from(f32::from_bits(b))).collect())
            .collect(),
    };
    for s in &p.segments {
        match s {
            Segment::Straight(nodes) => {
                for (ni, n) in nodes.iter().enumerate() {
                    r.node(ni, n, 0)?;
                }
            }
            Segment::Walk(w) => {
                for trip in 0..u64::from(w.trips.get()) {
                    for (ni, n) in w.body.iter().enumerate() {
                        r.node(ni, n, trip)?;
                    }
                }
            }
        }
    }
    Ok(r.params)
}
