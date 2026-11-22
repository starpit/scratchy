// SPDX-License-Identifier: Apache-2.0
//! KTIR → [`Program`]: every op lifted once.
//!
//! Integers stay affine in the walk's trip and the grid's indices ([`Lin`]); memory views and
//! access tiles are views over parameters ([`SymView`]); a load is the view itself, read in place;
//! every tensor op is one device node over the views of its operands, its result a fresh
//! intermediate. Nothing is evaluated per trip or per grid point:
//!
//! * a top-level `scf.for` of two or more trips becomes one [`Segment::Walk`]. Its induction
//!   variable is `lb + trip·step`, so every address computed from it is [`Address::Stepped`]; each
//!   value it carries is one buffer the straight code before it initialises and each trip
//!   overwrites. This arm is the only place an `scf.for` is consumed;
//! * the grid is parallel axes: `program_id(d)` is the symbol `pid[d]`, each intermediate is held
//!   once per grid point, and every node gains one outer space axis per grid dimension wider than
//!   one, each operand's stride along it read off the operand's origin.

use std::collections::HashMap;
use std::num::{NonZeroU32, NonZeroU64};

use ktir_core::attrkey::AttrKey;
use ktir_core::ddl_binds::DeviceOp;
use ktir_core::dtypes::DType;
use ktir_core::ir::{Attr, IRFunction, Operation, Ssa};
use ktir_core::irtype::IrType;
use ktir_core::opkind::OpKind;

use crate::lin::{Lin, NotAffine, Sym};
use crate::program::{
    Address, Axis, ConstId, ConstStick, Format, Gather, Local, LocalId, Node, OpFunc, Operand,
    ParamId, Program, Segment, Store, View, Walk,
};
use crate::qty::{Elems, Extent, Stride};
use crate::view::{SymView, ViewError, extent};

/// Why a function did not lift.
#[derive(Clone, Debug, PartialEq)]
pub enum LiftError {
    /// An op this lowering has no device form for yet.
    Op(OpKind),
    /// An op whose operands or attributes are not what KTIR states for it.
    Malformed { op: OpKind, what: &'static str },
    /// An integer op over the trip or the grid whose result is not affine in them.
    NotAffine { op: OpKind },
    /// A shape op the view cannot express.
    View { op: OpKind, error: ViewError },
    /// An element type with no device format.
    Format { op: OpKind, dtype: DType },
    /// A loop inside a walk's body.
    NestedWalk,
    /// An address that falls before its buffer, or moves backwards, or moves with the trip in
    /// straight code.
    Address { op: OpKind },
}

/// A tensor during the lift: a view over a store whose origin is still affine.
#[derive(Clone, Debug, PartialEq)]
struct T {
    store: Store,
    view: SymView,
    format: Format,
    gather: Option<G>,
}

#[derive(Clone, Debug, PartialEq)]
enum G {
    Rows { axis: usize, index: Box<T>, rows: Extent },
    Offset { index: Box<T>, step: Stride, bound: Extent },
}

/// One node during the lift.
#[derive(Clone, Debug)]
struct Raw {
    func: OpFunc,
    space: Vec<Extent>,
    inputs: Vec<T>,
    output: T,
}

#[derive(Clone, Debug, PartialEq)]
enum Val {
    Int(Lin),
    Float(f64),
    /// A parameter's address `at` elements in.
    Ptr { param: ParamId, at: Lin },
    /// A memory view or an access tile, not yet read.
    Mem(T),
    /// A tensor value.
    Tensor(T),
    /// One value in every position of `shape`.
    Splat { value: f64, format: Format, shape: Vec<Extent> },
    /// A scalar the device reads: `index[0]·scale + add`.
    Dyn { index: T, scale: i64, add: i64 },
    /// `tensor.empty`.
    Empty(Vec<Extent>),
}

struct Cx<'f> {
    f: &'f IRFunction<'f>,
    env: HashMap<Ssa, Val>,
    params: Vec<Option<Format>>,
    locals: Vec<Local>,
    consts: Vec<ConstStick>,
    const_ids: HashMap<ConstStick, ConstId>,
    segments: Vec<Segment>,
    /// The nodes of the run being lifted: straight code, or a walk's body.
    run: Vec<Raw>,
    /// `Some(trips, carried)` while a walk's body is being lifted.
    walking: Option<(NonZeroU32, Vec<LocalId>)>,
    /// The grid's extents, x first.
    grid: [u64; 3],
}

fn int_list<'a>(op: &Operation<'a>, k: AttrKey) -> Option<&'a [i64]> {
    match op.attr(k) {
        Some(Attr::IntList(v)) => Some(*v),
        _ => None,
    }
}

fn format_of(op: OpKind, dtype: DType) -> Result<Format, LiftError> {
    match dtype {
        DType::F16 => Ok(Format::Fp16),
        DType::F32 => Ok(Format::Fp32),
        DType::I32 => Ok(Format::Int32),
        DType::Fp8E4m3 => Ok(Format::Fp8),
        DType::Bool | DType::I64 => Err(LiftError::Format { op, dtype }),
    }
}

fn extents(op: OpKind, dims: &[i64]) -> Result<Vec<Extent>, LiftError> {
    dims.iter()
        .map(|&d| extent(d).ok_or(LiftError::Malformed { op, what: "a dimension of one position or more" }))
        .collect()
}

impl<'f> Cx<'f> {
    fn get(&self, s: Ssa) -> Result<Val, LiftError> {
        self.env.get(&s).cloned().ok_or(LiftError::Malformed {
            op: OpKind::FuncReturn,
            what: "every value is defined before it is read",
        })
    }

    fn bind(&mut self, op: &Operation<'_>, v: Val) {
        if let Some(r) = op.result {
            self.env.insert(r, v);
        }
    }

    fn int(&self, op: &Operation<'_>, s: Ssa) -> Result<Lin, LiftError> {
        match self.get(s)? {
            Val::Int(l) => Ok(l),
            _ => Err(LiftError::Malformed { op: op.op_type, what: "an integer operand" }),
        }
    }

    fn result_format(op: &Operation<'_>) -> Result<Format, LiftError> {
        let dtype = op
            .result_type
            .and_then(IrType::elem)
            .ok_or(LiftError::Malformed { op: op.op_type, what: "a result with an element type" })?;
        format_of(op.op_type, dtype)
    }

    fn result_shape(op: &Operation<'_>) -> Result<Vec<Extent>, LiftError> {
        let dims = op
            .result_type
            .and_then(IrType::dims)
            .ok_or(LiftError::Malformed { op: op.op_type, what: "a shaped result" })?;
        extents(op.op_type, dims)
    }

    /// The constant stick holding `value` in `format`.
    fn constant(&mut self, value: f64, format: Format) -> ConstId {
        let c = ConstStick {
            format,
            bits: (value as f32).to_bits(),
        };
        let next = ConstId::new(self.consts.len() as u32);
        *self.const_ids.entry(c).or_insert_with(|| {
            self.consts.push(c);
            next
        })
    }

    /// A fresh intermediate of `shape`, row-major, held once per grid point.
    fn local(&mut self, shape: &[Extent], format: Format) -> T {
        let n: u64 = shape.iter().map(|e| e.get()).product();
        let mut origin = Lin::constant(0);
        let mut copies = 1u64;
        for d in 0..3 {
            if self.grid[d] > 1 {
                origin.pid[d] = (n * copies) as i64;
                copies *= self.grid[d];
            }
        }
        self.locals.push(Local {
            elements: Elems::new(n * copies),
            format,
        });
        T {
            store: Store::Local(LocalId::new(self.locals.len() as u32 - 1)),
            view: SymView::row_major(origin, shape),
            format,
            gather: None,
        }
    }

    /// `v` as an operand over `shape`: a tensor broadcast to it, a scalar or splat as a constant
    /// stick read at every position.
    fn operand(&mut self, op: &Operation<'_>, v: &Val, shape: &[Extent], format: Format) -> Result<T, LiftError> {
        let view_err = |error| LiftError::View { op: op.op_type, error };
        match v {
            Val::Tensor(t) => Ok(T {
                view: t.view.broadcast(shape).map_err(view_err)?,
                ..t.clone()
            }),
            Val::Splat { value, format, .. } => Ok(self.splat(*value, *format, shape)),
            Val::Float(x) => Ok(self.splat(*x, format, shape)),
            Val::Int(l) => match l.as_const() {
                Some(i) => Ok(self.splat(i as f64, format, shape)),
                None => Err(LiftError::Op(op.op_type)),
            },
            Val::Dyn { index, scale: 1, add: 0 } => Ok(T {
                view: SymView {
                    origin: index.view.origin,
                    axes: shape.iter().map(|&e| Axis { extent: e, stride: Stride::BROADCAST }).collect(),
                },
                ..index.clone()
            }),
            _ => Err(LiftError::Malformed { op: op.op_type, what: "a tensor or a scalar operand" }),
        }
    }

    fn splat(&mut self, value: f64, format: Format, shape: &[Extent]) -> T {
        let c = self.constant(value, format);
        T {
            store: Store::Const(c),
            view: SymView {
                origin: Lin::constant(0),
                axes: shape.iter().map(|&e| Axis { extent: e, stride: Stride::BROADCAST }).collect(),
            },
            format,
            gather: None,
        }
    }

    fn note_param(&mut self, op: OpKind, s: Store, format: Format) -> Result<(), LiftError> {
        if let Store::Param(p) = s {
            let slot = &mut self.params[p.get() as usize];
            match slot {
                Some(f) if *f != format => {
                    return Err(LiftError::Malformed { op, what: "one format per parameter" });
                }
                _ => *slot = Some(format),
            }
        }
        Ok(())
    }

    /// One pointwise device op over the result's shape.
    fn pointwise(&mut self, op: &Operation<'_>, func: OpFunc, format: Format) -> Result<(), LiftError> {
        let shape = Self::result_shape(op)?;
        let ins: Vec<Val> = op.operands.iter().map(|&s| self.get(s)).collect::<Result<_, _>>()?;
        let in_format = ins
            .iter()
            .find_map(|v| match v {
                Val::Tensor(t) => Some(t.format),
                _ => None,
            })
            .unwrap_or(format);
        let inputs = ins
            .iter()
            .map(|v| self.operand(op, v, &shape, in_format))
            .collect::<Result<Vec<_>, _>>()?;
        let output = self.local(&shape, format);
        self.run.push(Raw {
            func,
            space: shape,
            inputs,
            output: output.clone(),
        });
        self.bind(op, Val::Tensor(output));
        Ok(())
    }

    fn device(&mut self, op: &Operation<'_>, d: DeviceOp) -> Result<(), LiftError> {
        let format = Self::result_format(op)?;
        self.pointwise(op, OpFunc::Device(d), format)
    }

    fn view_op(
        &mut self,
        op: &Operation<'_>,
        f: impl Fn(&SymView) -> Result<SymView, ViewError>,
    ) -> Result<(), LiftError> {
        let v = match self.get(op.operands[0])? {
            Val::Tensor(t) => Val::Tensor(T {
                view: f(&t.view).map_err(|error| LiftError::View { op: op.op_type, error })?,
                ..t
            }),
            Val::Splat { value, format, .. } => Val::Splat {
                value,
                format,
                shape: Self::result_shape(op)?,
            },
            _ => return Err(LiftError::Malformed { op: op.op_type, what: "a shape op over a tensor" }),
        };
        self.bind(op, v);
        Ok(())
    }

    /// `linalg.reduce`: the combiner over the stated dimensions; a non-identity init folds in after.
    fn reduce(&mut self, op: &Operation<'_>) -> Result<(), LiftError> {
        let (d, combine, identity) = match op.attr(AttrKey::ReduceFn) {
            Some(Attr::Op(OpKind::ArithAddf)) => (DeviceOp::Sum, DeviceOp::Add, 0.0),
            Some(Attr::Op(OpKind::ArithMaxnumf | OpKind::ArithMaximumf | OpKind::ArithMaxf)) => {
                (DeviceOp::Max, DeviceOp::Maximum, f64::NEG_INFINITY)
            }
            Some(Attr::Op(OpKind::ArithMinnumf | OpKind::ArithMinimumf | OpKind::ArithMinf)) => {
                (DeviceOp::Min, DeviceOp::Minimum, f64::INFINITY)
            }
            Some(Attr::Op(k)) => return Err(LiftError::Op(*k)),
            _ => return Err(LiftError::Malformed { op: op.op_type, what: "linalg.reduce states its combiner" }),
        };
        let dims = int_list(op, AttrKey::Dimensions)
            .ok_or(LiftError::Malformed { op: op.op_type, what: "linalg.reduce states its dimensions" })?;
        let format = Self::result_format(op)?;
        let Val::Tensor(input) = self.get(op.operands[0])? else {
            return Err(LiftError::Malformed { op: op.op_type, what: "linalg.reduce reads a tensor" });
        };
        let space = input.view.shape();
        let folded: Vec<usize> = dims.iter().map(|&d| d as usize).collect();
        let out_shape: Vec<Extent> = (0..space.len()).filter(|a| !folded.contains(a)).map(|a| space[a]).collect();
        let result = self.local(&out_shape, format);
        let mut out = result.clone();
        for &a in &folded {
            out.view.axes.insert(a, Axis { extent: space[a], stride: Stride::BROADCAST });
        }
        self.run.push(Raw {
            func: OpFunc::Device(d),
            space,
            inputs: vec![input],
            output: out,
        });
        let init = self.get(op.operands[1])?;
        let value = match init {
            Val::Splat { value, .. } if value == identity => result,
            init => {
                let rhs = self.operand(op, &init, &out_shape, format)?;
                let o = self.local(&out_shape, format);
                self.run.push(Raw {
                    func: OpFunc::Device(combine),
                    space: out_shape,
                    inputs: vec![result, rhs],
                    output: o.clone(),
                });
                o
            }
        };
        self.bind(op, Val::Tensor(value));
        Ok(())
    }

    /// `linalg.matmul` / `linalg.batch_matmul`: one contraction; a non-zero init adds after.
    fn contraction(&mut self, op: &Operation<'_>) -> Result<(), LiftError> {
        let malformed = |what| LiftError::Malformed { op: op.op_type, what };
        let (Val::Tensor(a), Val::Tensor(b)) = (self.get(op.operands[0])?, self.get(op.operands[1])?) else {
            return Err(malformed("a contraction reads two tensors"));
        };
        let rank = a.view.axes.len();
        if rank < 2 || b.view.axes.len() != rank {
            return Err(malformed("a contraction's operands share a rank of two or more"));
        }
        let transpose_b = match op.attr(AttrKey::IndexingMaps) {
            Some(Attr::AffineMapList(maps)) if maps.len() == 3 => match maps[1].result_dims().as_deref() {
                Some([2, 1]) | Some([0, 3, 2]) => false,
                Some([1, 2]) | Some([0, 2, 3]) => true,
                _ => return Err(malformed("the B map is (k, n) or (n, k)")),
            },
            None => false,
            Some(_) => return Err(malformed("indexing_maps lists A, B and C")),
        };
        let b = if transpose_b {
            let mut perm: Vec<usize> = (0..rank).collect();
            perm.swap(rank - 2, rank - 1);
            T {
                view: b.view.transpose(&perm).map_err(|error| LiftError::View { op: op.op_type, error })?,
                ..b
            }
        } else {
            b
        };
        let (m, k, n) = (a.view.axes[rank - 2].extent, a.view.axes[rank - 1].extent, b.view.axes[rank - 1].extent);
        if b.view.axes[rank - 2].extent != k {
            return Err(malformed("A's k is B's k"));
        }
        let format = Self::result_format(op)?;
        let mut out_shape: Vec<Extent> = a.view.shape()[..rank - 2].to_vec();
        out_shape.extend([m, n]);
        let mut space = out_shape.clone();
        space.push(k);
        let result = self.local(&out_shape, format);
        self.run.push(Raw {
            func: OpFunc::Contraction,
            space,
            inputs: vec![a, b],
            output: result.clone(),
        });
        let value = match self.get(op.operands[2])? {
            Val::Splat { value: 0.0, .. } | Val::Empty(_) => result,
            init => {
                let rhs = self.operand(op, &init, &out_shape, format)?;
                let o = self.local(&out_shape, format);
                self.run.push(Raw {
                    func: OpFunc::Device(DeviceOp::Add),
                    space: out_shape,
                    inputs: vec![result, rhs],
                    output: o.clone(),
                });
                o
            }
        };
        self.bind(op, Val::Tensor(value));
        Ok(())
    }

    /// `ktdp.construct_access_tile`: a window at a corner that is affine, or one device-read
    /// offset (a page the table names).
    fn access_tile(&mut self, op: &Operation<'_>) -> Result<(), LiftError> {
        let Val::Mem(mem) = self.get(op.operands[0])? else {
            return Err(LiftError::Malformed { op: op.op_type, what: "an access tile windows a memory view" });
        };
        let view_err = |error| LiftError::View { op: op.op_type, error };
        let stated = Self::result_shape(op)?;
        // A tile read in another order states its shape in that order: the window is the
        // memory-order block, read permuted.
        let order: Option<Vec<usize>> =
            int_list(op, AttrKey::CoordinateOrder).map(|o| o.iter().map(|&a| a as usize).collect());
        let shape = match &order {
            Some(o) => {
                let mut block = stated.clone();
                for (i, &a) in o.iter().enumerate() {
                    block[a] = stated[i];
                }
                block
            }
            None => stated,
        };
        let mut corner = Vec::with_capacity(op.operands.len() - 1);
        let mut offset: Option<(usize, T, i64, i64)> = None;
        for (a, &s) in op.operands[1..].iter().enumerate() {
            match self.get(s)? {
                Val::Int(l) => corner.push(l),
                Val::Dyn { index, scale, add } if offset.is_none() => {
                    offset = Some((a, index, scale, add));
                    corner.push(Lin::constant(add));
                }
                _ => return Err(LiftError::Malformed { op: op.op_type, what: "a corner of integers" }),
            }
        }
        let mut t = T {
            view: mem.view.window(&corner, &shape).map_err(view_err)?,
            ..mem.clone()
        };
        if let Some((a, index, scale, add)) = offset {
            // Row `index·scale + add` starts the window, so the index may name every row from
            // which the window still fits: `(held − add − window) / scale + 1` values, a whole
            // count by construction (the last index whose window fits, plus one).
            let bad = LiftError::Address { op: op.op_type };
            let scale = u64::try_from(scale).ok().and_then(NonZeroU64::new).ok_or(bad.clone())?;
            let add = u64::try_from(add).map_err(|_| bad.clone())?;
            let (held, window) = (mem.view.axes[a].extent.get(), shape[a].get());
            let room = held.checked_sub(add + window).ok_or(bad.clone())?;
            t.gather = Some(G::Offset {
                index: Box::new(index),
                step: Stride::new(mem.view.axes[a].stride.get() * scale.get()),
                bound: Extent::new(NonZeroU64::new(room / scale.get() + 1).ok_or(bad)?),
            });
        }
        if let Some(o) = order {
            t.view = t.view.transpose(&o).map_err(view_err)?;
        }
        self.bind(op, Val::Mem(t));
        Ok(())
    }

    /// `ktdp.construct_indirect_access_tile`: one dimension's rows named by an index view.
    fn indirect_tile(&mut self, op: &Operation<'_>) -> Result<(), LiftError> {
        let malformed = |what| LiftError::Malformed { op: op.op_type, what };
        let Val::Mem(mem) = self.get(op.operands[0])? else {
            return Err(malformed("an indirect tile indexes a memory view"));
        };
        let shape = Self::result_shape(op)?;
        let kinds = match op.attr(AttrKey::DimKinds) {
            Some(Attr::StrList(k)) => *k,
            _ => return Err(malformed("an indirect tile states its dimension kinds")),
        };
        let data = int_list(op, AttrKey::DimData).ok_or(malformed("an indirect tile states its index views"))?;
        let offsets = match op.attr(AttrKey::IntermediateVars) {
            Some(Attr::Ssas(v)) => v.to_vec(),
            _ => return Err(malformed("an indirect tile states its offsets")),
        };
        let mut corner = vec![Lin::constant(0); shape.len()];
        let mut rows = None;
        for (d, kind) in kinds.iter().enumerate() {
            match *kind {
                "direct_sub" => corner[d] = self.int(op, offsets[d])?,
                "indirect" if rows.is_none() => {
                    let Val::Mem(iview) = self.get(op.operands[1 + data[d] as usize])? else {
                        return Err(malformed("an index view is a memory view"));
                    };
                    let at = self.int(op, offsets[d])?;
                    let index = T {
                        view: iview
                            .view
                            .window(&[at], &[shape[d]])
                            .map_err(|error| LiftError::View { op: op.op_type, error })?,
                        ..iview
                    };
                    rows = Some((d, index));
                }
                _ => return Err(malformed("an indirect tile has one indirect dimension")),
            }
        }
        let (axis, index) = rows.ok_or(malformed("an indirect tile has an indirect dimension"))?;
        let held = mem.view.axes[axis].extent;
        let mut view = mem.view.window(&corner, &shape).map_err(|error| LiftError::View { op: op.op_type, error })?;
        view.axes[axis].extent = shape[axis];
        self.bind(
            op,
            Val::Mem(T {
                view,
                gather: Some(G::Rows { axis, index: Box::new(index), rows: held }),
                ..mem
            }),
        );
        Ok(())
    }

    fn store(&mut self, op: &Operation<'_>) -> Result<(), LiftError> {
        let value = self.get(op.operands[0])?;
        let Val::Mem(dest) = self.get(op.operands[1])? else {
            return Err(LiftError::Malformed { op: op.op_type, what: "a store writes an access tile" });
        };
        self.note_param(op.op_type, dest.store, dest.format)?;
        let shape = dest.view.shape();
        let input = self.operand(op, &value, &shape, dest.format)?;
        self.run.push(Raw {
            func: OpFunc::Copy,
            space: shape,
            inputs: vec![input],
            output: dest,
        });
        Ok(())
    }

    /// A top-level `scf.for`: straight code at one trip, a [`Walk`] at two or more.
    fn walk(&mut self, op: &Operation<'_>) -> Result<(), LiftError> {
        let malformed = |what| LiftError::Malformed { op: op.op_type, what };
        let bound = |s: Ssa| self.int(op, s).and_then(|l| l.as_const().ok_or(malformed("a constant loop bound")));
        let (lb, ub, step) = (bound(op.operands[0])?, bound(op.operands[1])?, bound(op.operands[2])?);
        if step <= 0 {
            return Err(malformed("a loop that advances"));
        }
        let trips = if ub > lb { (ub - lb + step - 1) / step } else { 0 };
        let iv = match op.attr(AttrKey::IterVar) {
            Some(Attr::Ssas(v)) if !v.is_empty() => v[0],
            _ => return Err(malformed("scf.for states its induction variable")),
        };
        let args: Vec<Ssa> = match op.attr(AttrKey::IterArgs) {
            Some(Attr::Ssas(v)) => v.to_vec(),
            _ => Vec::new(),
        };
        let results: Vec<Ssa> = match op.attr(AttrKey::ResultNames) {
            Some(Attr::Ssas(v)) => v.to_vec(),
            _ => op.result.into_iter().collect(),
        };
        let inits: Vec<Val> = op.operands[3..].iter().map(|&s| self.get(s)).collect::<Result<_, _>>()?;
        let body = op.regions.first().copied().unwrap_or(&[]);
        if trips <= 1 {
            // One trip (or none) is straight code at the lower bound.
            self.env.insert(iv, Val::Int(Lin::constant(lb)));
            for (a, v) in args.iter().zip(&inits) {
                self.env.insert(*a, v.clone());
            }
            let out = if trips == 1 { self.ops(body)?.unwrap_or_default() } else { inits };
            for (r, v) in results.iter().zip(out) {
                self.env.insert(*r, v);
            }
            return Ok(());
        }
        if self.walking.is_some() {
            return Err(LiftError::NestedWalk);
        }
        let trips = NonZeroU32::new(u32::try_from(trips).map_err(|_| malformed("a trip count that fits u32"))?)
            .ok_or(malformed("a walk of two or more trips"))?;
        // Each carried value is one buffer, initialised by the straight code before the walk.
        let mut carried = Vec::with_capacity(inits.len());
        let mut buffers = Vec::with_capacity(inits.len());
        for init in &inits {
            let (shape, format) = match init {
                Val::Tensor(t) => (t.view.shape(), t.format),
                Val::Splat { shape, format, .. } => (shape.clone(), *format),
                _ => return Err(malformed("a walk carries tensors")),
            };
            let buffer = self.local(&shape, format);
            let input = self.operand(op, init, &shape, format)?;
            self.run.push(Raw {
                func: OpFunc::Copy,
                space: shape,
                inputs: vec![input],
                output: buffer.clone(),
            });
            let Store::Local(l) = buffer.store else {
                return Err(malformed("a carried buffer is an intermediate"));
            };
            carried.push(l);
            buffers.push(buffer);
        }
        self.close_straight()?;
        self.walking = Some((trips, carried));
        self.env.insert(iv, Val::Int(Lin::constant(lb).add(Lin::sym(Sym::Trip).scale(step))));
        for (a, b) in args.iter().zip(&buffers) {
            self.env.insert(*a, Val::Tensor(b.clone()));
        }
        let yields = self.ops(body)?.unwrap_or_default();
        // Each trip's next value, written back into its buffer once the trip has read the last.
        for (y, b) in yields.iter().zip(&buffers) {
            if *y == Val::Tensor(b.clone()) {
                continue;
            }
            let shape = b.view.shape();
            let input = self.operand(op, y, &shape, b.format)?;
            self.run.push(Raw {
                func: OpFunc::Copy,
                space: shape,
                inputs: vec![input],
                output: b.clone(),
            });
        }
        self.close_walk()?;
        for (r, b) in results.iter().zip(buffers) {
            self.env.insert(*r, Val::Tensor(b));
        }
        Ok(())
    }

    fn step(&mut self, op: &Operation<'_>) -> Result<Option<Vec<Val>>, LiftError> {
        use OpKind as K;
        let malformed = |what| LiftError::Malformed { op: op.op_type, what };
        match op.op_type {
            K::ArithConstant => {
                let v = match op.attr(AttrKey::Value) {
                    Some(Attr::Int(i)) => Val::Int(Lin::constant(*i)),
                    Some(Attr::Float(f)) => Val::Float(*f),
                    Some(Attr::Bool(b)) => Val::Int(Lin::constant(i64::from(*b))),
                    _ => return Err(malformed("arith.constant states its value")),
                };
                self.bind(op, v);
            }
            K::KtdpGetComputeTileId => {
                let names: Vec<Ssa> = match op.attr(AttrKey::ResultNames) {
                    Some(Attr::Ssas(v)) => v.to_vec(),
                    _ => op.result.into_iter().collect(),
                };
                for (d, r) in names.iter().enumerate().take(3) {
                    let l = if self.grid[d] > 1 { Lin::sym(Sym::Pid(d)) } else { Lin::constant(0) };
                    self.env.insert(*r, Val::Int(l));
                }
            }
            K::ArithIndexCast | K::ArithIndexCastui | K::ArithExtsi | K::ArithExtui | K::ArithTrunci => {
                let v = self.get(op.operands[0])?;
                self.bind(op, v);
            }
            K::ArithAddi | K::ArithSubi | K::ArithMuli | K::ArithDivsi | K::ArithDivui | K::ArithRemsi
            | K::ArithRemui | K::ArithMinsi | K::ArithMaxsi | K::ArithFloordivsi | K::ArithCeildivsi => {
                let (a, b) = (self.get(op.operands[0])?, self.get(op.operands[1])?);
                let tensor = |v: &Val| matches!(v, Val::Tensor(_) | Val::Splat { .. });
                if tensor(&a) || tensor(&b) {
                    return match op.op_type {
                        K::ArithAddi => self.device(op, DeviceOp::AddI32ToI32).map(|()| None),
                        K::ArithMuli => self.device(op, DeviceOp::MulI32ToI32).map(|()| None),
                        _ => Err(LiftError::Op(op.op_type)),
                    };
                }
                let na = |_: NotAffine| LiftError::NotAffine { op: op.op_type };
                let konst = |l: Lin| l.as_const().ok_or(LiftError::NotAffine { op: op.op_type });
                let v = match (a, b) {
                    (Val::Int(x), Val::Int(y)) => Val::Int(match op.op_type {
                        K::ArithAddi => x.add(y),
                        K::ArithSubi => x.sub(y),
                        K::ArithMuli => x.mul(y).map_err(na)?,
                        K::ArithDivsi | K::ArithDivui | K::ArithFloordivsi => x.div_floor(konst(y)?).map_err(na)?,
                        K::ArithCeildivsi => x.div_ceil(konst(y)?).map_err(na)?,
                        K::ArithRemsi | K::ArithRemui => x.rem(konst(y)?).map_err(na)?,
                        K::ArithMinsi => Lin::constant(konst(x)?.min(konst(y)?)),
                        _ => Lin::constant(konst(x)?.max(konst(y)?)),
                    }),
                    (Val::Ptr { param, at }, Val::Int(y)) | (Val::Int(y), Val::Ptr { param, at })
                        if op.op_type == K::ArithAddi =>
                    {
                        Val::Ptr { param, at: at.add(y) }
                    }
                    (Val::Dyn { index, scale, add }, Val::Int(y)) | (Val::Int(y), Val::Dyn { index, scale, add }) => {
                        let y = konst(y)?;
                        match op.op_type {
                            K::ArithAddi => Val::Dyn { index, scale, add: add + y },
                            K::ArithMuli => Val::Dyn { index, scale: scale * y, add: add * y },
                            _ => return Err(LiftError::NotAffine { op: op.op_type }),
                        }
                    }
                    _ => return Err(LiftError::NotAffine { op: op.op_type }),
                };
                self.bind(op, v);
            }
            K::KtdpConstructMemoryView => {
                let shape = extents(op.op_type, int_list(op, AttrKey::Shape).ok_or(malformed("a view states its shape"))?)?;
                let strides: Vec<Stride> = match int_list(op, AttrKey::Strides) {
                    Some(s) => s
                        .iter()
                        .map(|&x| u64::try_from(x).map(Stride::new).map_err(|_| malformed("non-negative strides")))
                        .collect::<Result<_, _>>()?,
                    None => SymView::row_major(Lin::constant(0), &shape).axes.iter().map(|a| a.stride).collect(),
                };
                let format = Self::result_format(op)?;
                let Val::Ptr { param, at } = self.get(op.operands[0])? else {
                    return Err(malformed("a memory view's base is a parameter"));
                };
                let t = T {
                    store: Store::Param(param),
                    view: SymView::of_memory(at, &shape, &strides),
                    format,
                    gather: None,
                };
                self.note_param(op.op_type, t.store, format)?;
                self.bind(op, Val::Mem(t));
            }
            K::KtdpConstructAccessTile => self.access_tile(op)?,
            K::KtdpConstructIndirectAccessTile => self.indirect_tile(op)?,
            K::KtdpLoad => match self.get(op.operands[0])? {
                // A load reads the tile in place: the view is the value.
                Val::Mem(t) => self.bind(op, Val::Tensor(t)),
                _ => return Err(malformed("a load reads an access tile")),
            },
            K::KtdpStore => self.store(op)?,
            K::TensorExtract => {
                let Val::Tensor(t) = self.get(op.operands[0])? else {
                    return Err(malformed("tensor.extract reads a tensor"));
                };
                let mut origin = t.view.origin;
                for (a, &s) in op.operands[1..].iter().enumerate() {
                    let i = self.int(op, s)?;
                    origin = origin.add(
                        i.mul(Lin::constant(t.view.axes[a].stride.get() as i64))
                            .map_err(|_| LiftError::NotAffine { op: op.op_type })?,
                    );
                }
                let index = T {
                    view: SymView {
                        origin,
                        axes: vec![Axis { extent: Extent::ONE, stride: Stride::UNIT }],
                    },
                    ..t
                };
                self.bind(op, Val::Dyn { index, scale: 1, add: 0 });
            }
            K::TensorSplat => {
                let shape = Self::result_shape(op)?;
                let format = Self::result_format(op)?;
                let v = match self.get(op.operands[0])? {
                    Val::Float(value) => Val::Splat { value, format, shape },
                    Val::Int(l) => Val::Splat {
                        value: l.as_const().ok_or(LiftError::NotAffine { op: op.op_type })? as f64,
                        format,
                        shape,
                    },
                    dynamic @ Val::Dyn { scale: 1, add: 0, .. } => Val::Tensor(self.operand(op, &dynamic, &shape, format)?),
                    _ => return Err(malformed("a splat of a scalar")),
                };
                self.bind(op, v);
            }
            K::TensorEmpty => {
                let shape = Self::result_shape(op)?;
                self.bind(op, Val::Empty(shape));
            }
            K::LinalgFill => {
                let value = match self.get(op.operands[0])? {
                    Val::Float(f) => f,
                    Val::Int(l) => l.as_const().ok_or(LiftError::NotAffine { op: op.op_type })? as f64,
                    _ => return Err(malformed("a fill of a constant")),
                };
                let v = Val::Splat {
                    value,
                    format: Self::result_format(op)?,
                    shape: Self::result_shape(op)?,
                };
                self.bind(op, v);
            }
            K::TensorExpandShape | K::TensorCollapseShape | K::TensorReshape => {
                let to = Self::result_shape(op)?;
                self.view_op(op, |v| v.reshape(&to))?;
            }
            K::LinalgBroadcast => {
                let shape = Self::result_shape(op)?;
                let added: Vec<usize> = int_list(op, AttrKey::Dimensions)
                    .ok_or(malformed("linalg.broadcast states its dimensions"))?
                    .iter()
                    .map(|&d| d as usize)
                    .collect();
                self.view_op(op, |v| {
                    let mut w = v.clone();
                    for &d in &added {
                        w.axes.insert(d, Axis { extent: shape[d], stride: Stride::BROADCAST });
                    }
                    Ok(w)
                })?;
            }
            K::LinalgTranspose => {
                let perm: Vec<usize> = int_list(op, AttrKey::Permutation)
                    .ok_or(malformed("linalg.transpose states its permutation"))?
                    .iter()
                    .map(|&d| d as usize)
                    .collect();
                self.view_op(op, |v| v.transpose(&perm))?;
            }
            K::LinalgReduce => self.reduce(op)?,
            K::LinalgMatmul | K::LinalgBatchMatmul => self.contraction(op)?,
            K::ArithAddf => self.device(op, DeviceOp::Add)?,
            K::ArithSubf => self.device(op, DeviceOp::Sub)?,
            K::ArithMulf => self.device(op, DeviceOp::Multiply)?,
            K::ArithDivf => self.device(op, DeviceOp::RealDiv)?,
            K::ArithMaximumf | K::ArithMaxnumf | K::ArithMaxf => self.device(op, DeviceOp::Maximum)?,
            K::ArithMinimumf | K::ArithMinnumf | K::ArithMinf => self.device(op, DeviceOp::Minimum)?,
            K::ArithNegf => self.device(op, DeviceOp::Neg)?,
            K::ArithAbsf | K::MathAbsf => self.device(op, DeviceOp::Abs)?,
            K::MathExp => self.device(op, DeviceOp::Exp)?,
            K::MathLog => self.device(op, DeviceOp::Log)?,
            K::MathSqrt => self.device(op, DeviceOp::Sqrt)?,
            K::MathRsqrt => self.device(op, DeviceOp::Rsqrt)?,
            K::MathTanh => self.device(op, DeviceOp::Tanh)?,
            K::MathFloor => self.device(op, DeviceOp::Floor)?,
            K::ArithCmpf => {
                let d = match op.attr(AttrKey::Predicate) {
                    Some(Attr::Str("ogt" | "ugt")) => DeviceOp::GreaterThan,
                    Some(Attr::Str("oge" | "uge")) => DeviceOp::GreaterEqual,
                    Some(Attr::Str("olt" | "ult")) => DeviceOp::LesserThan,
                    Some(Attr::Str("ole" | "ule")) => DeviceOp::LesserEqual,
                    Some(Attr::Str("oeq" | "ueq")) => DeviceOp::Equal,
                    Some(Attr::Str("one" | "une")) => DeviceOp::NotEqual,
                    _ => return Err(malformed("arith.cmpf states an ordered or unordered compare")),
                };
                // The predicate is 0 or 1, held in the format its operands compare in.
                let format = op
                    .operands
                    .iter()
                    .find_map(|&s| match self.get(s) {
                        Ok(Val::Tensor(t)) => Some(t.format),
                        _ => None,
                    })
                    .ok_or(malformed("arith.cmpf compares a tensor"))?;
                self.pointwise(op, OpFunc::Device(d), format)?;
            }
            K::ArithSelect => self.device(op, DeviceOp::Where3)?,
            K::ArithExtf | K::ArithTruncf | K::ArithConvertf => {
                let to = Self::result_format(op)?;
                let v = self.get(op.operands[0])?;
                match v {
                    Val::Float(_) => self.bind(op, v),
                    Val::Splat { value, shape, .. } => self.bind(op, Val::Splat { value, format: to, shape }),
                    Val::Tensor(ref t) if t.format == to => self.bind(op, v),
                    Val::Tensor(ref t) => match (t.format, to) {
                        (Format::Fp16, Format::Fp32) => self.pointwise(op, OpFunc::Device(DeviceOp::Dl16ToFp32), to)?,
                        (Format::Fp32, Format::Fp16) => self.pointwise(op, OpFunc::Device(DeviceOp::Fp32ToDl16), to)?,
                        _ => return Err(LiftError::Op(op.op_type)),
                    },
                    _ => return Err(malformed("a conversion of a value")),
                }
            }
            K::ArithSitofp | K::ArithUitofp => {
                let to = Self::result_format(op)?;
                match self.get(op.operands[0])? {
                    Val::Int(l) => self.bind(
                        op,
                        Val::Float(l.as_const().ok_or(LiftError::NotAffine { op: op.op_type })? as f64),
                    ),
                    Val::Splat { value, shape, .. } => self.bind(op, Val::Splat { value, format: to, shape }),
                    Val::Tensor(t) if t.format == Format::Int32 && to == Format::Fp32 => {
                        self.pointwise(op, OpFunc::Device(DeviceOp::Int32ToFp32), to)?
                    }
                    _ => return Err(LiftError::Op(op.op_type)),
                }
            }
            K::ArithFptosi => match self.get(op.operands[0])? {
                Val::Float(x) => self.bind(op, Val::Int(Lin::constant(x as i64))),
                Val::Tensor(t) if t.format == Format::Fp32 => {
                    self.pointwise(op, OpFunc::Device(DeviceOp::Fp32ToInt32), Format::Int32)?
                }
                _ => return Err(LiftError::Op(op.op_type)),
            },
            K::ScfFor => self.walk(op)?,
            K::ScfYield => {
                let vs = op.operands.iter().map(|&s| self.get(s)).collect::<Result<Vec<_>, _>>()?;
                return Ok(Some(vs));
            }
            K::FuncReturn | K::RegionBb0Args => {}
            other => return Err(LiftError::Op(other)),
        }
        Ok(None)
    }

    fn ops(&mut self, ops: &[Operation<'_>]) -> Result<Option<Vec<Val>>, LiftError> {
        for op in ops {
            if let Some(y) = self.step(op)? {
                return Ok(Some(y));
            }
        }
        Ok(None)
    }

    /// The grid axes a node gains, outermost first: each grid dimension wider than one.
    fn grid_axes(&self) -> Vec<usize> {
        (0..3).rev().filter(|&d| self.grid[d] > 1).collect()
    }

    fn finish<O: Clone>(
        &self,
        raw: Raw,
        origin: &impl Fn(Lin) -> Result<O, LiftError>,
    ) -> Result<Node<O>, LiftError> {
        let grid = self.grid_axes();
        let mut space: Vec<Extent> = grid
            .iter()
            .map(|&d| Extent::new(NonZeroU64::new(self.grid[d]).unwrap_or(NonZeroU64::MIN)))
            .collect();
        space.extend(raw.space);
        let operand = |t: T| self.operand_of(t, &grid, origin);
        Ok(Node {
            func: raw.func,
            space,
            inputs: raw.inputs.into_iter().map(operand).collect::<Result<_, _>>()?,
            output: operand(raw.output)?,
        })
    }

    fn operand_of<O: Clone>(
        &self,
        t: T,
        grid: &[usize],
        origin: &impl Fn(Lin) -> Result<O, LiftError>,
    ) -> Result<Operand<O>, LiftError> {
        let mut axes: Vec<Axis> = Vec::with_capacity(grid.len() + t.view.axes.len());
        for &d in grid {
            let s = u64::try_from(t.view.origin.pid[d]).map_err(|_| LiftError::Address { op: OpKind::KtdpGetComputeTileId })?;
            axes.push(Axis {
                extent: Extent::new(NonZeroU64::new(self.grid[d]).unwrap_or(NonZeroU64::MIN)),
                stride: Stride::new(s),
            });
        }
        axes.extend(t.view.axes);
        let mut base = t.view.origin;
        base.pid = [0; 3];
        let gather = match t.gather {
            None => None,
            Some(G::Rows { axis, index, rows }) => Some(Gather::Rows {
                axis: axis + grid.len(),
                index: Box::new(self.operand_of(*index, grid, origin)?),
                rows,
            }),
            Some(G::Offset { index, step, bound }) => Some(Gather::Offset {
                index: Box::new(self.operand_of(*index, grid, origin)?),
                step,
                bound,
            }),
        };
        Ok(Operand {
            store: t.store,
            view: View { origin: origin(base)?, axes },
            format: t.format,
            gather,
        })
    }

    fn close_straight(&mut self) -> Result<(), LiftError> {
        if self.run.is_empty() {
            return Ok(());
        }
        let fixed = |l: Lin| -> Result<Elems, LiftError> {
            match (l.trip, u64::try_from(l.c)) {
                (0, Ok(c)) => Ok(Elems::new(c)),
                _ => Err(LiftError::Address { op: OpKind::ScfFor }),
            }
        };
        let run = std::mem::take(&mut self.run);
        let nodes = run.into_iter().map(|r| self.finish(r, &fixed)).collect::<Result<_, _>>()?;
        self.segments.push(Segment::Straight(nodes));
        Ok(())
    }

    fn close_walk(&mut self) -> Result<(), LiftError> {
        let (trips, carried) = self.walking.take().ok_or(LiftError::NestedWalk)?;
        let address = |l: Lin| -> Result<Address, LiftError> {
            let at = u64::try_from(l.c).map_err(|_| LiftError::Address { op: OpKind::ScfFor })?;
            match l.trip {
                0 => Ok(Address::Fixed(Elems::new(at))),
                t => Ok(Address::Stepped {
                    at_trip0: Elems::new(at),
                    per_trip: u64::try_from(t)
                        .ok()
                        .and_then(NonZeroU64::new)
                        .ok_or(LiftError::Address { op: OpKind::ScfFor })?,
                }),
            }
        };
        let run = std::mem::take(&mut self.run);
        let body = run.into_iter().map(|r| self.finish(r, &address)).collect::<Result<_, _>>()?;
        self.segments.push(Segment::Walk(Walk { trips, carried, body }));
        Ok(())
    }
}

/// Lift `f`.
pub fn lift(f: &IRFunction<'_>) -> Result<Program, LiftError> {
    let (gx, gy, gz) = f.grid;
    let mut cx = Cx {
        f,
        env: HashMap::new(),
        params: vec![None; f.arguments.len()],
        locals: Vec::new(),
        consts: Vec::new(),
        const_ids: HashMap::new(),
        segments: Vec::new(),
        run: Vec::new(),
        walking: None,
        grid: [gx.max(1) as u64, gy.max(1) as u64, gz.max(1) as u64],
    };
    for (i, (s, _)) in f.arguments.iter().enumerate() {
        cx.env.insert(
            *s,
            Val::Ptr {
                param: ParamId::new(i as u32),
                at: Lin::constant(0),
            },
        );
    }
    let ops = cx.f.operations;
    cx.ops(ops)?;
    cx.close_straight()?;
    Ok(Program {
        params: cx.params,
        locals: cx.locals,
        consts: cx.consts,
        segments: cx.segments,
    })
}
