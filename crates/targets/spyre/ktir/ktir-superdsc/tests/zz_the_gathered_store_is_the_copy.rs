// SPDX-License-Identifier: Apache-2.0

//! ⭐⭐⭐⭐⭐ THE DEVICE-SIDE GATHER-COPY — a `ktdp.store` whose VALUE is a gathered load, and
//! these are its proofs.
//!
//! The whole-function door's gather join fires only on classified compute ops, so a program that
//! merely MOVES gathered rows — `table[ids] → out`, the store and nothing else — stated a gather
//! no op carried, and the final `gather_carried` check refused the whole function. That refusal
//! is right about every body that CONSUMES the gathered rows and wrong about the one that moves
//! them: the copy is the vendor's own `test_gather_1core` shape (an `identity` pointwise
//! carrying the index operand), which the door already emits twice — rung 4 mints it for a
//! gathered matmul-B ([`zz_the_gathered_matmul_materializes`]), and the scalarmul's gathered arm
//! reaches the same assembler for a gathered `multiply`. `gathered_store_copy` is the third call
//! site: no mint (the store's own output parameter is the destination), no scale, the store's
//! own position in the program order.
//!
//! # WHAT THESE TESTS PIN
//!
//! * the POSITIVE: the door emits the copy — an `identity` opFunc carrying the
//!   indirect-access index, gathering from the TABLE's address, writing the OUTPUT parameter's
//!   address — the two facts whose absence is the silent direct read the guard exists to refuse;
//! * CONTROL (fail-closed survival, the window seal): a store whose window is NOT the gathered
//!   tile's shape still refuses by name — the arm lowers whole gathered rows and nothing else;
//! * CONTROL (fail-closed survival, the column-corner seal): a store at a nonzero column corner
//!   still refuses by name — the gather addresses its table from the base;
//! * CONTROL (fail-closed survival, the one-consumer seal): a gathered load two stores read
//!   still refuses by name — the copy stands for exactly one.

use ktir_core::affine::{AffineExpr, AffineMap};
use ktir_core::arena::Arena;
use ktir_core::attrkey::AttrKey;
use ktir_core::dtypes::DType;
use ktir_core::ir::{Attr, IRFunction, Operation, Ssa};
use ktir_core::irtype::IrType;
use ktir_core::opkind::OpKind;
use ktir_superdsc::emit::whole_function::lower_function;
use ktir_superdsc::ktir_node::{BufferId, KtirNode, Program};
use ktir_superdsc::place::{PlaceId, act_name};
use ktir_superdsc::placement::{BundleLayout, SegRole, TensorPlacement};

/// The argmax row feedback's own geometry, at the smallest whole-stick shape: a `[V+1, 64]` f16
/// table (one row per token plus the pad row), a `[64]` i32 index stick (the clamped-index stick
/// the slab spelling reads word 0 of), a `[1, 64]` gathered tile (ONE row), and a `[1, 64]` f16
/// output slot.
const V: i64 = 128;
const W: i64 = 64;

/// What one control varies — the accepting shape plus the three seals' near misses. Each case
/// moves ONE fact alone, so a refusal names the seal that fired and a pass cannot be vacuous.
#[derive(Clone, Copy)]
enum Shape {
    /// The fixture's own shape: the store's window is the gathered tile, at corner 0, with the
    /// gathered load consumed by exactly this store.
    Accepting,
    /// The store's OUTPUT VIEW is `[2, 64]` — one row wider than the `[1, 64]` gathered tile.
    /// The window seal must refuse: the copy writes whole gathered rows, never a wider window.
    WideWindow,
    /// The store's access tile sits at COLUMN CORNER 1. The corner seal must refuse: the gather
    /// addresses its table from the base, so a column-blocked destination cannot be gathered.
    ColumnCorner,
}

/// One gathered-copy program:
///
/// `out[0, 0] = table[ids[0], :]` — the indirect tile over the table view indexed by the ids
/// view, its load stored whole into the output parameter. NO compute op exists in the function;
/// the only consumer of the gathered load is the store.
fn gathered_store_program(shape: Shape) -> KtirNode {
    let a: &'static Arena = Arena::global();
    let shape_attr = |op: Operation<'static>, r: i64, c: i64| {
        op.with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(vec![r, c])))
            .with_attr(a, AttrKey::Dtype, Attr::Dtype(DType::F16))
    };

    // Parameters: %0 = table, %1 = ids, %2 = out.
    let (p_t, p_i, p_o) = (Ssa(0), Ssa(1), Ssa(2));
    // The index constant 0.
    let zero = Ssa(3);
    // The table chain.
    let (v_t, t_t, val_t) = (Ssa(4), Ssa(5), Ssa(6));
    // The ids chain.
    let (v_i, t_i, val_i) = (Ssa(7), Ssa(8), Ssa(9));
    // The INDIRECT tile over the table view and its load — the gathered row.
    let (ind, row) = (Ssa(10), Ssa(11));
    // The output view/tile and the store.
    let (v_o, t_o) = (Ssa(12), Ssa(13));

    // The indirect dim's subscript — `Sym(0) + Dim(0)`, the one form `gather_subscripts` builds
    // and the one `gathers_of` folds into an entry base. Interned before the ops vec because its
    // children are borrows.
    let sub0 = a.expr(AffineExpr::Add(
        a.expr(AffineExpr::Sym(0)),
        a.expr(AffineExpr::Dim(0)),
    ));

    // The output view's row extent — the window seal's one variable. The gathered tile stays
    // `[1, 64]`; only the STORE's window moves.
    let out_rows = match shape {
        Shape::Accepting | Shape::ColumnCorner => 1,
        Shape::WideWindow => 2,
    };

    let mut ops = vec![
        Operation::new(a, Some(zero), OpKind::ArithConstant, &[]).with_attr(
            a,
            AttrKey::Value,
            Attr::Int(0),
        ),
        // TABLE: [V+1, 64] f16 — one row per token plus the pad row.
        shape_attr(
            Operation::new(a, Some(v_t), OpKind::KtdpConstructMemoryView, &[p_t]),
            V + 1,
            W,
        ),
        shape_attr(
            Operation::new(
                a,
                Some(t_t),
                OpKind::KtdpConstructAccessTile,
                &[v_t, zero, zero],
            ),
            V + 1,
            W,
        ),
        shape_attr(
            Operation::new(a, Some(val_t), OpKind::KtdpLoad, &[t_t]),
            V + 1,
            W,
        ),
        // ids: [64] i32 — the clamped-index stick.
        Operation::new(a, Some(v_i), OpKind::KtdpConstructMemoryView, &[p_i])
            .with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(vec![64])))
            .with_attr(a, AttrKey::Dtype, Attr::Dtype(DType::I32)),
        shape_attr(
            Operation::new(
                a,
                Some(t_i),
                OpKind::KtdpConstructAccessTile,
                &[v_i, zero, zero],
            ),
            64,
            1,
        ),
        shape_attr(
            Operation::new(a, Some(val_i), OpKind::KtdpLoad, &[t_i]),
            64,
            1,
        ),
        // THE INDIRECT TILE: data view first, index view second — `gather_of`'s own reading. Its
        // `Shape` is the GATHERED tile `[1, 64]`: ONE row, whole-stick. The anchor triple is
        // what a REAL program carries after `to_ktir_emit`'s re-base: dim 0 indirect with the
        // `Sym(0) + Dim(0)` subscript, `Sym(0)` named by `intermediate_vars[0]` = the zero
        // constant — a one-block gather anchored at index word 0.
        shape_attr(
            Operation::new(
                a,
                Some(ind),
                OpKind::KtdpConstructIndirectAccessTile,
                &[v_t, v_i],
            ),
            1,
            W,
        )
        .with_attr(
            a,
            AttrKey::DimKinds,
            Attr::StrList(a.names(vec!["indirect", "direct_sub"])),
        )
        .with_attr(a, AttrKey::DimData, Attr::IntList(a.ints(vec![0, 0])))
        .with_attr(
            a,
            AttrKey::IntermediateVars,
            Attr::Ssas(a.ssa(vec![zero, zero])),
        )
        .with_attr(
            a,
            AttrKey::DimSubs,
            Attr::AffineMapList(a.maps(vec![
                AffineMap {
                    num_dims: 2,
                    num_syms: 2,
                    exprs: std::slice::from_ref(sub0),
                },
                AffineMap {
                    num_dims: 2,
                    num_syms: 2,
                    exprs: a.exprs(vec![AffineExpr::Dim(1)]),
                },
            ])),
        ),
        // THE GATHERED ROW — the load the store consumes.
        shape_attr(Operation::new(a, Some(row), OpKind::KtdpLoad, &[ind]), 1, W),
        // OUT: [1, 64] (the accepting shape) or [2, 64] (the window control).
        shape_attr(
            Operation::new(a, Some(v_o), OpKind::KtdpConstructMemoryView, &[p_o]),
            out_rows,
            W,
        ),
        shape_attr(
            Operation::new(
                a,
                Some(t_o),
                OpKind::KtdpConstructAccessTile,
                &[v_o, zero, zero],
            ),
            out_rows,
            W,
        ),
        // The store — the gathered load's one consumer.
        Operation::new(a, None, OpKind::KtdpStore, &[row, t_o]),
    ];
    if let Shape::ColumnCorner = shape {
        // The corner control REPLACES the store's tile with one at column corner 1 — the same
        // view, the same shape, one corner moved. BOTH the store and the original tile are
        // popped: leaving the original in place would put two ops at result t_o, and the door's
        // `find(o.result == out_tile)` takes the FIRST — the control would silently test the
        // zero-corner tile and pass vacuously.
        ops.pop(); // the store
        ops.pop(); // the original zero-corner tile
        ops.push(shape_attr(
            Operation::new(
                a,
                Some(t_o),
                OpKind::KtdpConstructAccessTile,
                &[v_o, zero, Ssa(15)],
            ),
            out_rows,
            W,
        ));
        // The constant 1 the corner reads — interned after the tile that names it only because
        // operands are SSA values, resolved at walk time.
        let one = Operation::new(a, Some(Ssa(15)), OpKind::ArithConstant, &[]).with_attr(
            a,
            AttrKey::Value,
            Attr::Int(1),
        );
        ops.insert(1, one);
        ops.push(Operation::new(a, None, OpKind::KtdpStore, &[row, t_o]));
    }

    KtirNode {
        func: IRFunction {
            name: "gathered_store_fwd",
            arguments: a.args(vec![
                (p_t, IrType::Index),
                (p_i, IrType::Index),
                (p_o, IrType::Index),
            ]),
            operations: a.ops(ops),
            grid: (1, 1, 1),
            return_type: None,
        },
        // The program kind is real but never dispatched on: the whole-function walk is the door,
        // and the argmax's own refusal test pins that a per-`Program` arm does not exist.
        program: Program::ArgMax,
        bindings: vec![BufferId::new(301), BufferId::new(302), BufferId::new(303)],
        mask: None,
        node_out_tid: None,
        minted: Default::default(),
    }
}

/// The one-consumer control: the SAME gathered load stored twice — the copy stands for exactly
/// one store, and a second consumer must refuse by name.
fn two_store_program() -> KtirNode {
    let mut k = gathered_store_program(Shape::Accepting);
    let a: &'static Arena = Arena::global();
    // A second output view/tile over the SAME output parameter, and a second store of the SAME
    // gathered row: the gathered load now has two consumers.
    let (v_o2, t_o2) = (Ssa(20), Ssa(21));
    let zero = Ssa(3);
    let row = Ssa(11);
    let p_o = Ssa(2);
    let extra = vec![
        Operation::new(a, Some(v_o2), OpKind::KtdpConstructMemoryView, &[p_o])
            .with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(vec![1, W])))
            .with_attr(a, AttrKey::Dtype, Attr::Dtype(DType::F16)),
        Operation::new(
            a,
            Some(t_o2),
            OpKind::KtdpConstructAccessTile,
            &[v_o2, zero, zero],
        )
        .with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(vec![1, W])))
        .with_attr(a, AttrKey::Dtype, Attr::Dtype(DType::F16)),
        Operation::new(a, None, OpKind::KtdpStore, &[row, t_o2]),
    ];
    let mut ops = k.func.operations.to_vec();
    ops.extend(extra);
    k.func.operations = a.ops(ops);
    k
}

/// A fully-placed layout for the three parameters — `resolve_seg_base` panics on a name that is
/// in neither `ids` nor the synth map, so every operand the emission names must be placed. The
/// table (t301) and the output (t303) sit at DISTINCT nonzero offsets, so address assertions
/// discriminate WHICH buffer an op reads, not a coincidence of two zero offsets.
fn layout_for(k: &KtirNode) -> BundleLayout {
    let mut l = BundleLayout::default();
    for (i, b) in k.bindings.iter().enumerate() {
        let tid = b.get();
        l.ids.borrow_mut().insert(act_name(tid), PlaceId::Act(tid));
        // Bytes per parameter: table [V+1, 64] f16, ids [64] i32, out [1, 64] f16.
        let bytes: u64 = match i {
            0 => ((V + 1) as u64) * (W as u64) * 2,
            1 => 64 * 4,
            _ => W as u64 * 2,
        };
        l.placements.insert(
            tid,
            TensorPlacement {
                tid,
                role: SegRole::Activation,
                segment: 3,
                bank: 0,
                // Distinct offsets inside seg3, so no two parameters share a base.
                offset: (i as u64) * 0x10000,
                size: bytes,
            },
        );
    }
    l
}

/// The emitted ops, descriptors and all — asserted on the TYPED structs rather than re-parsed
/// JSON, so the assertions name the fields the card acts on with no second parser to disagree
/// with the first.
fn emitted(k: &KtirNode) -> Result<Vec<ktir_superdsc::emit::EmittedOp>, String> {
    let layout = layout_for(k);
    let mut sid = 0i64;
    lower_function(k, Some(&layout), &mut sid, None).map_err(|e| e.message)
}

/// The `value_tensor` alloc's base address in a gathered descriptor — the gather's SOURCE buffer
/// (the rung-4 test's own reader, verbatim).
fn value_tensor_address(op: &ktir_superdsc::emit::EmittedOp) -> Option<u64> {
    let dsc = op.dsc();
    let dsc = dsc.dscs_.first()?.values().next()?;
    let node = dsc
        .scheduleTree_
        .iter()
        .find(|n| n.nodeType_ == "allocate" && n.indirectAllocType_ == "value_tensor")?;
    node.startAddressCoreCorelet_
        .data_
        .get("[0, 0, 0]")?
        .parse()
        .ok()
}

/// The op's OWN output address in a gathered descriptor — the `OUTPUT`-tagged labeledDs that is
/// NOT the value tensor. A gathered descriptor tags the value tensor `OUTPUT` too (the vendor's
/// `test_gather_1core` shape, mirrored by `attach_gather_index`), so the first `OUTPUT` match is
/// the gather's SOURCE half the time; excluding the value alloc's `ldsIdx_` is what makes the
/// reader name the destination.
fn output_address(op: &ktir_superdsc::emit::EmittedOp) -> Option<u64> {
    let dsc = op.dsc();
    let dsc = dsc.dscs_.first()?.values().next()?;
    let value_idx = dsc
        .scheduleTree_
        .iter()
        .find(|n| n.nodeType_ == "allocate" && n.indirectAllocType_ == "value_tensor")?
        .ldsIdx_;
    let idx = dsc
        .labeledDs_
        .iter()
        .find(|ds| ds.dsType_ == "OUTPUT" && ds.ldsIdx_ != value_idx)?
        .ldsIdx_;
    let node = dsc
        .scheduleTree_
        .iter()
        .find(|n| n.ldsIdx_ == idx && n.nodeType_ == "allocate")?;
    node.startAddressCoreCorelet_
        .data_
        .get("[0, 0, 0]")?
        .parse()
        .ok()
}

/// The table t301's device address under [`layout_for`]: seg3 + 0x0. The output t303's:
/// seg3 + 0x20000. The only buffers in the test at those addresses.
const TABLE_ADDR: u64 = 0xC_0000_0000;
const OUT_ADDR: u64 = 0xC_0000_0000 + 0x20000;

/// ⭐⭐⭐ THE POSITIVE: the door emits the copy — the vendor's own gathered-`identity` shape,
/// gathering from the TABLE's address and writing the OUTPUT parameter's address. One op (a
/// one-entry run is one index stick, so the cut emits one leg), no compute op before it, and the
/// two facts whose absence is the silent direct read the `gather_carried` guard exists to refuse:
/// the `identity` opFunc CARRIES the indirect-access index, and its value alloc IS the table.
#[test]
fn the_gathered_store_emits_the_identity_copy() {
    let ops = emitted(&gathered_store_program(Shape::Accepting))
        .expect("the gathered-copy program lowers through the whole-function door");
    assert_eq!(
        ops.len(),
        1,
        "a one-row gathered store is ONE copy leg -- got {:?}",
        ops.iter().map(|o| o.op_name.clone()).collect::<Vec<_>>()
    );
    let copy = &ops[0];
    let dsc = copy
        .dsc()
        .dscs_
        .first()
        .and_then(|m| m.values().next())
        .expect("the copy has a descriptor");
    // The copy is the vendor's own gathered fixture shape: an `identity` opFunc with the
    // indirect-access index attached, and the value/index alloc pair.
    let identity = dsc
        .computeOp_
        .iter()
        .any(|c| c.opFuncName == "identity" && !c.indirectAccessIndexLabeledDs.is_empty());
    assert!(
        identity,
        "the copy is an `identity` carrying the indirect-access index (the wrong-rows trap's \
         whole point): {:?}",
        copy.op_name
    );
    let has_pair =
        dsc.scheduleTree_.iter().any(|n| {
            n.indirectAllocType_ == "value_tensor" && n.relatedIndirectAccessAlloc_.is_some()
        }) && dsc
            .scheduleTree_
            .iter()
            .any(|n| n.indirectAllocType_ == "index_tensor");
    assert!(
        has_pair,
        "the copy's value/index alloc pair is the gather itself: {:?}",
        copy.op_name
    );
    // The copy's VALUE alloc IS the TABLE, at the table's own address.
    assert_eq!(
        value_tensor_address(copy),
        Some(TABLE_ADDR),
        "the copy gathers from the table at {TABLE_ADDR:#x}: {:?}",
        copy.op_name
    );
    // And its OUTPUT operand is the STORE's own parameter, at the output's address — the copy
    // writes the row slot, not a minted intermediate (there is no mint: `mint_ew_output` checks
    // the placement the caller already made).
    //
    // A gathered descriptor carries TWO `OUTPUT`-tagged labeledDs — the op's destination AND the
    // value tensor (the vendor's own `test_gather_1core` shape; the rung-4 test reads its table
    // through `indirectAllocType_ == "value_tensor"` for exactly this reason) — so "the OUTPUT
    // address" here means the one the value alloc is NOT.
    assert_eq!(
        output_address(copy),
        Some(OUT_ADDR),
        "the copy writes the output parameter at {OUT_ADDR:#x}: {:?}",
        copy.op_name
    );
}

/// ⛔ CONTROL — THE WINDOW SEAL: a store whose window is `[2, 64]` over a `[1, 64]` gathered tile
/// refuses by name. The copy writes whole gathered rows; a wider window would write a row the
/// gather never took, and a narrower one would truncate — both are new producer forms the arm
/// must not guess at.
#[test]
fn a_window_wider_than_the_gathered_tile_refuses() {
    let e = match emitted(&gathered_store_program(Shape::WideWindow)) {
        Err(e) => e,
        Ok(ops) => panic!(
            "a [2, 64] store over a [1, 64] gathered tile must refuse -- got {} op(s)",
            ops.len()
        ),
    };
    assert!(
        e.contains("the store's window must state exactly the tile's shape"),
        "the WINDOW seal's own refusal must fire -- got {e}"
    );
}

/// ⛔ CONTROL — THE COLUMN-CORNER SEAL: a store at column corner 1 refuses by name. The gather
/// addresses its table from the base, so a column-blocked destination cannot be gathered — the
/// scalarmul's own seal, stated for the copy.
#[test]
fn a_column_blocked_store_refuses() {
    let e = match emitted(&gathered_store_program(Shape::ColumnCorner)) {
        Err(e) => e,
        Ok(ops) => panic!(
            "a gathered store at column corner 1 must refuse -- got {} op(s)",
            ops.len()
        ),
    };
    assert!(
        e.contains("a column-blocked destination cannot be gathered"),
        "the CORNER seal's own refusal must fire -- got {e}"
    );
}

/// ⛔ CONTROL — THE ONE-CONSUMER SEAL: a gathered load two stores read refuses by name. The copy
/// stands for exactly one store; a second consumer needs its own op, and emitting one of the two
/// would silently drop the other.
#[test]
fn a_gathered_load_two_stores_read_refuses() {
    let e = match emitted(&two_store_program()) {
        Err(e) => e,
        Ok(ops) => panic!(
            "a gathered load with two store consumers must refuse -- got {} op(s)",
            ops.len()
        ),
    };
    assert!(
        e.contains("stands for exactly one"),
        "the ONE-CONSUMER seal's own refusal must fire -- got {e}"
    );
}
