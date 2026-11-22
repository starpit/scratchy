// SPDX-License-Identifier: Apache-2.0
//! KTIR → [`Program`]: every op lifted once.
//!
//! Integers stay affine in the walk's trip and the parallel axes ([`Lin`]); memory views and
//! access tiles are views over parameters ([`SymView`]); a load is the view itself, read in place;
//! every tensor op is one device node over the views of its operands, its result a fresh
//! intermediate. Nothing is evaluated per trip, per grid point or per loop iteration:
//!
//! * a parallel axis is a grid dimension wider than one, or an `scf.for` whose trips are
//!   independent (it carries no value, and writes no parameter it also reads). Its position is a
//!   symbol; each intermediate made under it is held once per position; every node made under it
//!   gains one outer space axis, each operand's stride along it read off the operand's origin;
//! * an `scf.for` whose trips depend on each other becomes one [`Segment::Walk`]. Its induction
//!   variable is `lb + trip·step`, so every address computed from it is [`Address::Stepped`]; each
//!   value it carries is one buffer the straight code before it initialises and each trip
//!   overwrites.
//!
//! [`Cx::for_loop`] is the only place an `scf.for` is consumed, and it has no path that lifts a
//! body more than once into the program.

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

impl T {
    /// Every store this operand addresses, with whether it is the operand's own (as opposed to an
    /// index it reads through).
    fn stores(&self) -> Vec<(Store, bool)> {
        let mut v = vec![(self.store, true)];
        match &self.gather {
            Some(G::Rows { index, .. }) | Some(G::Offset { index, .. }) => {
                v.extend(index.stores().into_iter().map(|(s, _)| (s, false)));
            }
            None => {}
        }
        v
    }
}

#[derive(Clone, Debug, PartialEq)]
enum G {
    Rows { axis: usize, index: Box<T>, rows: Extent },
    Offset { index: Box<T>, step: Stride, bound: Extent },
}

/// One node during the lift, with the parallel axes it was made under, outermost first.
#[derive(Clone, Debug)]
struct Raw {
    func: OpFunc,
    par: Vec<Par>,
    space: Vec<Extent>,
    inputs: Vec<T>,
    output: T,
}

/// How far a lift had grown: what [`Cx::restore`] takes it back to.
struct Mark {
    run: usize,
    locals: usize,
    consts: usize,
    params: Vec<Option<Format>>,
}

/// A parallel axis: its symbol slot and its extent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Par {
    slot: usize,
    extent: Extent,
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
    /// A compare's result: 1 or 0 in the format its operands compared in. Read only by a select
    /// or a cast.
    Pred(T),
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
    /// The parallel axes open now, outermost first.
    par: Vec<Par>,
    /// The symbol slot of each grid dimension wider than one, x first.
    grid_slot: [Option<usize>; 3],
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

    /// A fresh intermediate of `shape`, row-major, held once per position of every open parallel
    /// axis (the innermost axis's copies adjacent).
    fn local(&mut self, shape: &[Extent], format: Format) -> T {
        let n: u64 = shape.iter().map(|e| e.get()).product();
        let mut origin = Lin::constant(0);
        let mut copies = 1u64;
        for p in self.par.iter().rev() {
            origin.axis[p.slot] = (n * copies) as i64;
            copies *= p.extent.get();
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

    /// Push a node made now, under the parallel axes open now.
    fn push(&mut self, func: OpFunc, space: Vec<Extent>, inputs: Vec<T>, output: T) {
        let par = self.par.clone();
        self.run.push(Raw { func, par, space, inputs, output });
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
        self.push(func, shape, inputs, output.clone());
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
            Val::Pred(t) => Val::Pred(T {
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
        self.push(OpFunc::Device(d), space, vec![input], out);
        let init = self.get(op.operands[1])?;
        let value = match init {
            Val::Splat { value, .. } if value == identity => result,
            init => {
                let rhs = self.operand(op, &init, &out_shape, format)?;
                let o = self.local(&out_shape, format);
                self.push(OpFunc::Device(combine), out_shape, vec![result, rhs], o.clone());
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
        // The batch: each axis the extent the two operands share, an operand of extent one there
        // read at stride 0 (one matrix shared across the batch).
        let mut batch = Vec::with_capacity(rank - 2);
        for i in 0..rank - 2 {
            let (ea, eb) = (a.view.axes[i].extent, b.view.axes[i].extent);
            batch.push(match (ea, eb) {
                _ if ea == eb => ea,
                (Extent::ONE, e) | (e, Extent::ONE) => e,
                _ => return Err(malformed("A's and B's batch extents agree, or one of them is one")),
            });
        }
        let view_err = |error| LiftError::View { op: op.op_type, error };
        let with_batch = |t: T, last: [Extent; 2]| -> Result<T, LiftError> {
            let mut to = batch.clone();
            to.extend(last);
            Ok(T { view: t.view.broadcast(&to).map_err(view_err)?, ..t })
        };
        let a = with_batch(a, [m, k])?;
        let b = with_batch(b, [k, n])?;
        let format = Self::result_format(op)?;
        let mut out_shape: Vec<Extent> = batch.clone();
        out_shape.extend([m, n]);
        let mut space = out_shape.clone();
        space.push(k);
        let result = self.local(&out_shape, format);
        self.push(OpFunc::Contraction, space, vec![a, b], result.clone());
        let value = match self.get(op.operands[2])? {
            Val::Splat { value: 0.0, .. } | Val::Empty(_) => result,
            init => {
                let rhs = self.operand(op, &init, &out_shape, format)?;
                let o = self.local(&out_shape, format);
                self.push(OpFunc::Device(DeviceOp::Add), out_shape, vec![result, rhs], o.clone());
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
            t.gather = Some(Self::offset_gather(op.op_type, &mem.view, a, shape[a], index, scale, add)?);
        }
        if let Some(o) = order {
            t.view = t.view.transpose(&o).map_err(view_err)?;
        }
        self.bind(op, Val::Mem(t));
        Ok(())
    }

    /// The gather of a `window`-wide window along axis `a` of `held` that starts at row
    /// `index·scale + add`, the index read by the device. The index may name every row from which
    /// the window still fits: `(rows − add − window) / scale + 1` values, a whole count by
    /// construction (the last index whose window fits, plus one).
    fn offset_gather(
        op: OpKind,
        held: &SymView,
        a: usize,
        window: Extent,
        index: T,
        scale: i64,
        add: i64,
    ) -> Result<G, LiftError> {
        let bad = LiftError::Address { op };
        let scale = u64::try_from(scale).ok().and_then(NonZeroU64::new).ok_or(bad.clone())?;
        let add = u64::try_from(add).map_err(|_| bad.clone())?;
        let room = held.axes[a].extent.get().checked_sub(add + window.get()).ok_or(bad.clone())?;
        Ok(G::Offset {
            index: Box::new(index),
            step: Stride::new(held.axes[a].stride.get() * scale.get()),
            bound: Extent::new(NonZeroU64::new(room / scale.get() + 1).ok_or(bad)?),
        })
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
                    let view_err = |error| LiftError::View { op: op.op_type, error };
                    let index = match self.get(offsets[d])? {
                        Val::Int(at) => T {
                            view: iview.view.window(&[at], &[shape[d]]).map_err(view_err)?,
                            ..iview
                        },
                        // The index rows themselves sit at a device-read offset: one access
                        // cannot follow two indirections, so the rows are gathered into an
                        // intermediate first and the tile indexes through that.
                        Val::Dyn { index, scale, add } => {
                            let block = T {
                                view: iview.view.window(&[Lin::constant(add)], &[shape[d]]).map_err(view_err)?,
                                gather: Some(Self::offset_gather(
                                    op.op_type,
                                    &iview.view,
                                    0,
                                    shape[d],
                                    index,
                                    scale,
                                    add,
                                )?),
                                ..iview
                            };
                            let rows = self.local(&[shape[d]], block.format);
                            self.push(OpFunc::Copy, vec![shape[d]], vec![block], rows.clone());
                            rows
                        }
                        _ => return Err(malformed("an indirect tile's offset is an integer")),
                    };
                    rows = Some((d, index));
                }
                _ => return Err(malformed("an indirect tile has one indirect dimension")),
            }
        }
        let (axis, index) = rows.ok_or(malformed("an indirect tile has an indirect dimension"))?;
        // The index names the rows along `axis`, so the tile may be wider there than the buffer:
        // the window takes one row along it, and the axis is then the index's length.
        let held = mem.view.axes[axis].extent;
        let mut one_row = shape.clone();
        one_row[axis] = Extent::ONE;
        let mut view = mem.view.window(&corner, &one_row).map_err(|error| LiftError::View { op: op.op_type, error })?;
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
        self.push(OpFunc::Copy, shape, vec![input], dest);
        Ok(())
    }

    /// An `scf.for`: straight code at one trip; at two or more, a parallel axis when its trips are
    /// independent, else a [`Walk`].
    fn for_loop(&mut self, op: &Operation<'_>) -> Result<(), LiftError> {
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
        let trips_ext = Extent::new(
            u64::try_from(trips).ok().and_then(NonZeroU64::new).ok_or(malformed("a positive trip count"))?,
        );
        // A loop that carries nothing is a parallel axis, unless a trip writes a parameter the
        // loop also reads (one trip may then read what another wrote). The body is lifted once
        // under the axis; a body that writes what it reads is taken back and lifted as a walk.
        if args.is_empty() {
            let slot = (0..crate::lin::AXES)
                .find(|s| !self.par.iter().any(|p| p.slot == *s))
                .ok_or(malformed("at most eight parallel axes open at once"))?;
            let mark = self.mark();
            self.par.push(Par { slot, extent: trips_ext });
            self.env.insert(iv, Val::Int(Lin::constant(lb).add(Lin::sym(Sym::Axis(slot)).scale(step))));
            let lifted = self.ops(body);
            self.par.pop();
            lifted?;
            if self.independent(&mark, slot) {
                return Ok(());
            }
            self.restore(mark);
        }
        if self.walking.is_some() || self.par.len() > self.grid_slot.iter().flatten().count() {
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
            self.push(OpFunc::Copy, shape, vec![input], buffer.clone());
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
            self.push(OpFunc::Copy, shape, vec![input], b.clone());
        }
        self.close_walk()?;
        for (r, b) in results.iter().zip(buffers) {
            self.env.insert(*r, Val::Tensor(b));
        }
        Ok(())
    }

    /// Everything a lift grows, as it stands now.
    fn mark(&self) -> Mark {
        Mark {
            run: self.run.len(),
            locals: self.locals.len(),
            consts: self.consts.len(),
            params: self.params.clone(),
        }
    }

    /// Take back everything lifted since `m`.
    fn restore(&mut self, m: Mark) {
        self.run.truncate(m.run);
        self.locals.truncate(m.locals);
        for c in self.consts.drain(m.consts..) {
            self.const_ids.remove(&c);
        }
        self.params = m.params;
    }

    /// Whether the positions of parallel axis `slot`, as lifted since `m`, are independent: no
    /// two positions write one element, and nothing the loop writes outside its own
    /// intermediates is also read by it (one position could read what another wrote).
    fn independent(&self, m: &Mark, slot: usize) -> bool {
        // A store made before the loop: a parameter, or an intermediate the loop did not make.
        let outer = |s: Store| match s {
            Store::Param(_) => true,
            Store::Local(l) => (l.get() as usize) < m.locals,
            Store::Const(_) => false,
        };
        let mut written = std::collections::HashSet::new();
        let mut read = std::collections::HashSet::new();
        for r in &self.run[m.run..] {
            // Each position writes its own elements: the output's stride along the axis is at
            // least the span one position writes, and no index decides where it writes.
            let o = &r.output;
            let stride = o.view.origin.axis[slot];
            if o.gather.is_some() || stride <= 0 || (stride as u64) < o.view.span() {
                return false;
            }
            for (s, own) in o.stores() {
                if outer(s) {
                    if own { written.insert(s) } else { read.insert(s) };
                }
            }
            for (s, _) in r.inputs.iter().flat_map(T::stores) {
                if outer(s) {
                    read.insert(s);
                }
            }
        }
        written.is_disjoint(&read)
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
                    let l = match self.grid_slot[d] {
                        Some(slot) => Lin::sym(Sym::Axis(slot)),
                        None => Lin::constant(0),
                    };
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
                        K::ArithSubi => x.minus(y),
                        K::ArithMuli => x.times(y).map_err(na)?,
                        K::ArithDivsi | K::ArithDivui | K::ArithFloordivsi => x.div_floor(konst(y)?).map_err(na)?,
                        K::ArithCeildivsi => x.div_ceil(konst(y)?).map_err(na)?,
                        K::ArithRemsi | K::ArithRemui => x.modulo(konst(y)?).map_err(na)?,
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
                        i.times(Lin::constant(t.view.axes[a].stride.get() as i64))
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
                    // A one-element tensor repeats its element over every position.
                    Val::Tensor(t) if t.view.axes.iter().all(|a| a.extent == Extent::ONE) => Val::Tensor(T {
                        view: SymView {
                            origin: t.view.origin,
                            axes: shape.iter().map(|&e| Axis { extent: e, stride: Stride::BROADCAST }).collect(),
                        },
                        ..t
                    }),
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
                // A reshape is a view of the same elements where the merged axes are one strided
                // run; elsewhere the value is copied row-major first, which every reshape views.
                if let Val::Tensor(t) = self.get(op.operands[0])?
                    && let Err(ViewError::NotContiguous { .. }) = t.view.reshape(&to)
                {
                    let shape = t.view.shape();
                    let dense = self.local(&shape, t.format);
                    self.push(OpFunc::Copy, shape, vec![t], dense.clone());
                    let view = dense.view.reshape(&to).map_err(|error| LiftError::View { op: op.op_type, error })?;
                    self.bind(op, Val::Tensor(T { view, ..dense }));
                } else {
                    self.view_op(op, |v| v.reshape(&to))?;
                }
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
                // The device compares with C++'s float comparators (`senulator/pcfg2compute.cpp`
                // :460-466): `equal` and the orderings are ORDERED (false on NaN) and `notequal`
                // is UNORDERED (true on NaN). Exactly those predicates are device ops.
                let d = match op.attr(AttrKey::Predicate) {
                    Some(Attr::Str("ogt")) => DeviceOp::GreaterThan,
                    Some(Attr::Str("oge")) => DeviceOp::GreaterEqual,
                    Some(Attr::Str("olt")) => DeviceOp::LesserThan,
                    Some(Attr::Str("ole")) => DeviceOp::LesserEqual,
                    Some(Attr::Str("oeq")) => DeviceOp::Equal,
                    Some(Attr::Str("une")) => DeviceOp::NotEqual,
                    _ => return Err(LiftError::Op(op.op_type)),
                };
                // The device writes 1 or 0 in the format its inputs compare in
                // (`senulator/pcfg2compute.cpp:760-781`, `fcmpOp`: `out.setType(type)`).
                let format = op
                    .operands
                    .iter()
                    .find_map(|&s| match self.get(s) {
                        Ok(Val::Tensor(t)) => Some(t.format),
                        _ => None,
                    })
                    .ok_or(malformed("arith.cmpf compares a tensor"))?;
                self.pointwise(op, OpFunc::Device(d), format)?;
                if let Some(r) = op.result
                    && let Some(Val::Tensor(t)) = self.env.remove(&r)
                {
                    self.env.insert(r, Val::Pred(t));
                }
            }
            K::ArithSelect => {
                // `cond ? a : b`, in the select's own value format; the condition is a
                // predicate.
                let shape = Self::result_shape(op)?;
                let format = Self::result_format(op)?;
                let Val::Pred(cond) = self.get(op.operands[0])? else {
                    return Err(malformed("arith.select chooses by a predicate"));
                };
                let cond = T {
                    view: cond.view.broadcast(&shape).map_err(|error| LiftError::View { op: op.op_type, error })?,
                    ..cond
                };
                let a = self.get(op.operands[1])?;
                let b = self.get(op.operands[2])?;
                let a = self.operand(op, &a, &shape, format)?;
                let b = self.operand(op, &b, &shape, format)?;
                let output = self.local(&shape, format);
                self.push(OpFunc::Device(DeviceOp::Where3), shape, vec![cond, a, b], output.clone());
                self.bind(op, Val::Tensor(output));
            }
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
                        (Format::Fp16, Format::Fp8) => self.pointwise(op, OpFunc::Device(DeviceOp::Qfp8Ch), to)?,
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
                    // A predicate is 0 or 1 in the format its operands compared in: as that
                    // format, it is itself.
                    Val::Pred(t) if t.format == to => self.bind(op, Val::Tensor(t)),
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
            K::ScfFor => self.for_loop(op)?,
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

    /// The node as the program states it: one outer space axis per parallel axis it was made
    /// under, each operand's stride along it read off the operand's origin.
    fn finish<O: Clone>(&self, raw: Raw, origin: &impl Fn(Lin) -> Result<O, LiftError>) -> Result<Node<O>, LiftError> {
        let mut space: Vec<Extent> = raw.par.iter().map(|p| p.extent).collect();
        space.extend(raw.space);
        let operand = |t: T| Self::operand_of(t, &raw.par, origin);
        Ok(Node {
            func: raw.func,
            space,
            inputs: raw.inputs.into_iter().map(operand).collect::<Result<_, _>>()?,
            output: operand(raw.output)?,
        })
    }

    fn operand_of<O: Clone>(
        t: T,
        par: &[Par],
        origin: &impl Fn(Lin) -> Result<O, LiftError>,
    ) -> Result<Operand<O>, LiftError> {
        let bad = LiftError::Address { op: OpKind::ScfFor };
        let mut axes: Vec<Axis> = Vec::with_capacity(par.len() + t.view.axes.len());
        let mut base = t.view.origin;
        for p in par {
            let s = u64::try_from(base.axis[p.slot]).map_err(|_| bad.clone())?;
            axes.push(Axis { extent: p.extent, stride: Stride::new(s) });
            base.axis[p.slot] = 0;
        }
        // An origin that still moves along an axis the node is not under names no one element.
        if base.axis.iter().any(|&c| c != 0) {
            return Err(bad);
        }
        axes.extend(t.view.axes);
        let shift = par.len();
        let gather = match t.gather {
            None => None,
            Some(G::Rows { axis, index, rows }) => Some(Gather::Rows {
                axis: axis + shift,
                index: Box::new(Self::operand_of(*index, par, origin)?),
                rows,
            }),
            Some(G::Offset { index, step, bound }) => Some(Gather::Offset {
                index: Box::new(Self::operand_of(*index, par, origin)?),
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
        par: Vec::new(),
        grid_slot: [None; 3],
    };
    // The grid's dimensions wider than one are the first parallel axes, z outermost.
    for (d, g) in [(2, gz), (1, gy), (0, gx)] {
        if let Some(extent) = u64::try_from(g).ok().and_then(NonZeroU64::new).filter(|e| e.get() > 1) {
            let slot = cx.par.len();
            cx.par.push(Par { slot, extent: Extent::new(extent) });
            cx.grid_slot[d] = Some(slot);
        }
    }
    for (i, (s, ty)) in f.arguments.iter().enumerate() {
        let param = ParamId::new(i as u32);
        let v = match ty {
            // A runtime scalar is one element of its own buffer, read by the device.
            IrType::Scalar(dtype) => {
                let format = format_of(OpKind::FuncReturn, *dtype)?;
                cx.params[i] = Some(format);
                Val::Dyn {
                    index: T {
                        store: Store::Param(param),
                        view: SymView::row_major(Lin::constant(0), &[Extent::ONE]),
                        format,
                        gather: None,
                    },
                    scale: 1,
                    add: 0,
                }
            }
            _ => Val::Ptr { param, at: Lin::constant(0) },
        };
        cx.env.insert(*s, v);
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
