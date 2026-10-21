// SPDX-License-Identifier: Apache-2.0

//! ⭐⭐⭐⭐⭐ RUNG 4 — THE GATHERED MATMUL'S **B** MATERIALIZES, AND THESE ARE ITS PROOFS.
//!
//! The whole-function door's gather guard refuses a gathered operand on anything but a
//! `ScalarMul` — fail-closed, because every other assembler would read the table at its base and
//! ignore the ids: the right shape and the wrong rows, from a clean bake. RUNG 4's exception is a
//! `linalg.matmul` whose B is the gathered table, and the materialization is
//! `whole_function::gathered_matmul_materializes`: a KERNEL-less gathered `identity` copy into a
//! minted intermediate, the matmul reading the intermediate as plain-B `[k, n]`.
//!
//! # WHY THAT SHAPE AND NOT AN INDEX ON THE MATMUL ITSELF
//!
//! A matmul operand goes to `Stk::<KernelTag>::kernel(k, n)`, and a KERNEL-stick operand makes
//! the dxp reuse explorer run (`L3DlOpsScheduler.cpp:1550` gates it on `isReuse`;
//! `hasDimensionReuse` `:303-321` is `primaryDsInfo_.size() > 1 && count(KERNEL)`), which
//! demands an LX allocate node for every HBM-pinned labeledDs (`calculateFlopPerByte`
//! `:2334`/`:2337`) — a build-time `Err` from this crate's own gather-on-KERNEL guard. The
//! vendor's own gathered fixture (`dxp/test/test_gather_1core`) and IBM's paged attention both
//! spell the way around it: the gather rides a KERNEL-less elementwise copy (`identity` /
//! `AddZero`), and the matmul reads the copy's result.
//!
//! # WHAT THESE TESTS PIN
//!
//! * the POSITIVE: the door emits the materialized copy (an `identity` opFunc carrying an
//!   indirect-access index, over the TABLE's tid) and then the matmul, whose B is the minted
//!   intermediate and NOT the table;
//! * CONTROL (the wrong-rows trap): the copy declares the INDEX operand and points its value
//!   operand at the TABLE's tid — the two facts whose absence is the silent direct read the guard
//!   exists to refuse, and the matmul does not name the table at all;
//! * THE A SLOT (restated for the walk the WIP left behind): a gathered read materializes for
//!   ANY consumer — the copy is the gather, and what reads its result reads an ordinary
//!   activation — so a gather feeding the matmul's **A** materializes exactly as the B-slot one
//!   does, on a shape-valid fixture (every extent square at K), and the matmul reads the copy.
//!   The old control pinned the dead law (a ScalarMul-only refusal for the A slot) and is
//!   replaced, not weakened: the fail-closed rule survives as its own case below — a gathered
//!   load with TWO consumers is refused by name, because the materialized copy stands for
//!   exactly one;
//! * THE POOLED TABLE (b1): a gathered source whose layout declares a `gather_reach` is a POOL
//!   SLICE — its placement covers one entry while its index names values across the whole pool.
//!   The door checks the read's ENTRY UNIT against the reach's own (the unit the host's
//!   `GatherEntry` composition counts in — a descriptor striding at any other width reads every
//!   entry at the wrong address), and the VALUES' bound is the ceiling the stage-side
//!   composition makes unconstructible (pinned in `zz_the_gather_index_reproduces_the_host_page_
//!   address`). The one-entry placement the swept-window law refuses is admitted under a reach,
//!   a reach whose entry unit DISAGREES with the read is refused by name, and the same placement
//!   with NO reach still refuses: the reach widens the indexed read and nothing else.

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

/// `[m=8, k=64] · [64, 64] -> [8, 64]` over a `[128, 64]` V table gathered by a `[64]` i32 index —
/// the rung-4 fixture's own geometry (M=8, K=64, V=128, HEAD_DIM=64), the smallest shape every
/// door admits (the `tl.dot` K floor, the matmul's whole-stick K, the gather's index-vector and
/// f16-column floors).
const M: i64 = 8;
const K: i64 = 64;
const V: i64 = 128;
const HEAD: i64 = 64;

/// Which matmul slot the gathered load feeds.
#[derive(Clone, Copy)]
enum GatheredSlot {
    /// The fixture's own shape: the gathered load is the matmul's B (operand 1), `o = matmul(p,
    /// gather(v, ids))` with P `[8, 64]`.
    B,
    /// The gathered load feeds the matmul's **A** (operand 0) with the plain operand `[64, 64]`
    /// and the result `[64, 64]` — every extent square at K, so the fixture is SHAPE-VALID in
    /// this arrangement and the test pins the materialization itself, not a shape refusal.
    A,
}

/// One rung-4 program with a zero `outs`:
///
/// * `GatheredSlot::B` — `o = matmul(p, gather(v, ids))`, the fixture's own shape;
/// * `GatheredSlot::A` — `o = matmul(gather(v, ids), p)`, the same gather feeding the A slot with
///   B a plain load of the table.
///
/// Built from the op kinds `gather_of`, `region_for_operand` and `lower_function` actually read —
// `construct_memory_view`, `construct_access_tile`, `construct_indirect_access_tile`,
// `arith.constant`, `ktdp.load`, `linalg.matmul`, `ktdp.store` — so the door sees a real program
// and not a stub. The ONLY difference between the two cases is which matmul slot consumes the
// gathered load.
fn paged_program(slot: GatheredSlot) -> KtirNode {
    let a: &'static Arena = Arena::global();
    let shape = |op: Operation<'static>, r: i64, c: i64| {
        op.with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(vec![r, c])))
            .with_attr(a, AttrKey::Dtype, Attr::Dtype(DType::F16))
    };

    // Parameters: %0 = P, %1 = V table, %2 = ids, %3 = output.
    let (p_p, p_v, p_ids, p_out) = (Ssa(0), Ssa(1), Ssa(2), Ssa(3));
    // The index constant 0 and the A chain.
    let (zero, v_a, t_a, val_a) = (Ssa(4), Ssa(5), Ssa(6), Ssa(7));
    // The V chain — the table's PLAIN view, for the non-gathered proof that V is a real tensor.
    let (v_v, t_v, val_v) = (Ssa(8), Ssa(9), Ssa(10));
    // The ids chain.
    let (v_i, t_i, val_i) = (Ssa(11), Ssa(12), Ssa(13));
    // The INDIRECT tile over the (possibly swapped) table view, and its load.
    let (ind, prod) = (Ssa(14), Ssa(15));
    // The output view/tile, the zero outs, and the matmul result.
    let (v_o, t_o, outs, matmul) = (Ssa(16), Ssa(17), Ssa(18), Ssa(19));

    // Which matmul slot consumes the gathered load — the one variable. The B case is the
    // fixture's own `[8, 64] · [64, 64] -> [8, 64]`; the A case keeps every extent square at K
    // (`[64, 64] · [64, 64] -> [64, 64]`), because the gathered load `[K, HEAD]` is the
    // contraction's A there and the plain operand must carry K = 64 as its own K extent — the
    // first draft left P at `[8, 64]` and the control died at the matmul's own K-extent check,
    // which pins nothing about the gather.
    let (p_rows, p_cols, out_rows, out_cols) = match slot {
        GatheredSlot::B => (M, K, M, HEAD),
        GatheredSlot::A => (K, K, K, K),
    };

    // The swap itself: same table, same ids, only the POSITION of the gathered value changes.
    // That is also exactly the shape a TID-comparing guard would have mis-lowered (both operands
    // resolve to the same tids), which is why the emission checks VALUE IDENTITY
    // (`in_values[gi] == g.load`) and not tids.
    let (a_operand, b_operand) = match slot {
        GatheredSlot::B => (val_a, prod),
        GatheredSlot::A => (prod, val_a),
    };

    // The indirect dim's subscript — `Sym(0) + Dim(0)`, the one form `gather_subscripts` builds
    // and the one `gathers_of` folds into an entry base. Interned before the ops vec because its
    // children are borrows.
    let sub0 = a.expr(AffineExpr::Add(
        a.expr(AffineExpr::Sym(0)),
        a.expr(AffineExpr::Dim(0)),
    ));
    let ops = vec![
        Operation::new(a, Some(zero), OpKind::ArithConstant, &[]).with_attr(
            a,
            AttrKey::Value,
            Attr::Int(0),
        ),
        // P: the plain operand — [8, 64] as the B case's A, [64, 64] as the A case's B.
        shape(
            Operation::new(a, Some(v_a), OpKind::KtdpConstructMemoryView, &[p_p]),
            p_rows,
            p_cols,
        ),
        shape(
            Operation::new(
                a,
                Some(t_a),
                OpKind::KtdpConstructAccessTile,
                &[v_a, zero, zero],
            ),
            p_rows,
            p_cols,
        ),
        shape(
            Operation::new(a, Some(val_a), OpKind::KtdpLoad, &[t_a]),
            p_rows,
            p_cols,
        ),
        // V: [128, 64] — the table.
        shape(
            Operation::new(a, Some(v_v), OpKind::KtdpConstructMemoryView, &[p_v]),
            V,
            HEAD,
        ),
        shape(
            Operation::new(
                a,
                Some(t_v),
                OpKind::KtdpConstructAccessTile,
                &[v_v, zero, zero],
            ),
            V,
            HEAD,
        ),
        shape(
            Operation::new(a, Some(val_v), OpKind::KtdpLoad, &[t_v]),
            V,
            HEAD,
        ),
        // ids: [64] i32 — the index vector.
        Operation::new(a, Some(v_i), OpKind::KtdpConstructMemoryView, &[p_ids])
            .with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(vec![K])))
            .with_attr(a, AttrKey::Dtype, Attr::Dtype(DType::I32)),
        shape(
            Operation::new(
                a,
                Some(t_i),
                OpKind::KtdpConstructAccessTile,
                &[v_i, zero, zero],
            ),
            K,
            1,
        ),
        shape(
            Operation::new(a, Some(val_i), OpKind::KtdpLoad, &[t_i]),
            K,
            1,
        ),
        // THE INDIRECT TILE: data view first, index view second — `gather_of`'s own reading. Its
        // `Shape` is the GATHERED tile `[64, 64]`, over the V table in BOTH cases (the control
        // swaps the MATMUL SLOT, not the table — same table, same ids, same extents, only the
        // position of the gathered value changes). The anchor triple is what a REAL program
        // carries after `to_ktir_emit`'s re-base: dim 0 indirect with the `Sym(0) + Dim(0)`
        // subscript `gather_subscripts` builds, `Sym(0)` named by `intermediate_vars[0]` = the
        // zero constant — a one-block gather anchored at index word 0.
        shape(
            Operation::new(
                a,
                Some(ind),
                OpKind::KtdpConstructIndirectAccessTile,
                &[v_v, v_i],
            ),
            K,
            HEAD,
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
        shape(
            Operation::new(a, Some(prod), OpKind::KtdpLoad, &[ind]),
            K,
            HEAD,
        ),
        // OUT: [8, 64] in the B case, [64, 64] in the A case.
        shape(
            Operation::new(a, Some(v_o), OpKind::KtdpConstructMemoryView, &[p_out]),
            out_rows,
            out_cols,
        ),
        shape(
            Operation::new(
                a,
                Some(t_o),
                OpKind::KtdpConstructAccessTile,
                &[v_o, zero, zero],
            ),
            out_rows,
            out_cols,
        ),
        // The zero outs.
        shape(
            Operation::new(a, Some(outs), OpKind::TensorEmpty, &[]),
            out_rows,
            out_cols,
        ),
        // THE CONTRACTION — plain-B, NO indexing maps: the gathered rows contracted where they lie.
        shape(
            Operation::new(
                a,
                Some(matmul),
                OpKind::LinalgMatmul,
                &[a_operand, b_operand, outs],
            ),
            out_rows,
            out_cols,
        ),
        Operation::new(a, None, OpKind::KtdpStore, &[matmul, t_o]),
    ];

    KtirNode {
        func: IRFunction {
            name: "paged_vmatmul_fwd",
            arguments: a.args(vec![
                (p_p, IrType::Index),
                (p_v, IrType::Index),
                (p_ids, IrType::Index),
                (p_out, IrType::Index),
            ]),
            operations: a.ops(ops),
            grid: (1, 1, 1),
            return_type: None,
        },
        program: Program::Matmul,
        bindings: vec![
            BufferId::new(201),
            BufferId::new(202),
            BufferId::new(203),
            BufferId::new(204),
        ],
        mask: None,
        node_out_tid: None,
        minted: Default::default(),
    }
}

/// A fully-placed layout for the four parameters — `resolve_seg_base` panics on a name that is in
/// neither `ids` nor the synth map, so every operand the emission names must be placed.
///
/// The table (t202) is placed ALONE in the Activation segment, at a NON-ZERO offset — so its
/// device address (`SEGMENT_OFFSETS[3] + 0x8000`) is shared by no other operand in the test, and
/// an assertion on that number is an assertion on WHICH buffer an op reads, not on a coincidence
/// of two zero offsets.
fn layout_for(k: &KtirNode, slot: GatheredSlot) -> BundleLayout {
    let mut l = BundleLayout::default();
    for (i, b) in k.bindings.iter().enumerate() {
        let tid = b.get();
        l.ids.borrow_mut().insert(act_name(tid), PlaceId::Act(tid));
        // Bytes per parameter: P (the plain operand, slot-shaped), V [128,64] f16, ids [64] i32,
        // OUT (the result, slot-shaped).
        let bytes: u64 = match (slot, i) {
            (_, 1) => (V as u64) * (HEAD as u64) * 2,
            (_, 2) => (K as u64) * 4,
            // P: [8, 64] in the B case, [64, 64] in the A case.
            (GatheredSlot::B, 0) => (M as u64) * (K as u64) * 2,
            (GatheredSlot::A, 0) => (K as u64) * (K as u64) * 2,
            // OUT: [8, 64] in the B case, [64, 64] in the A case.
            (GatheredSlot::B, _) => (M as u64) * (HEAD as u64) * 2,
            (GatheredSlot::A, _) => (K as u64) * (K as u64) * 2,
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
/// JSON, so the assertions name the fields the card acts on (`labeledDs_`, `scheduleTree_`) with
/// no second parser to disagree with the first.
fn emitted(
    k: &KtirNode,
    slot: GatheredSlot,
) -> Result<Vec<ktir_superdsc::emit::EmittedOp>, String> {
    let layout = layout_for(k, slot);
    let mut sid = 0i64;
    lower_function(k, Some(&layout), &mut sid, None).map_err(|e| e.message)
}

/// The device address the `dsType_`-tagged operand of `op` starts at — the alloc's per-core
/// `data_` value at core 0, joined to `labeledDs_` by `ldsIdx_`. This is the fact the card acts
/// on: two descriptors reading the same buffer name it through the same number, and the
/// descriptor names operands only positionally (`Tensor0/1/2`), so the address is the
/// discriminator.
fn alloc_address(op: &ktir_superdsc::emit::EmittedOp, tag: &str) -> Option<u64> {
    let dsc = op.dsc();
    let dsc = dsc.dscs_.first()?.values().next()?;
    let idx = dsc.labeledDs_.iter().find(|ds| ds.dsType_ == tag)?.ldsIdx_;
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

/// The `value_tensor` alloc's base address in a gathered descriptor — the gather's SOURCE buffer.
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

/// The table t202's device address under [`layout_for`]: `SEGMENT_OFFSETS[3] + 0x10000` (parameter
/// 1's slot). The only buffer in the test at this address.
const TABLE_ADDR: u64 = 0xC_0000_0000 + 0x10000;

/// ⭐⭐⭐ THE POSITIVE: the door emits the materialized copy legs AND the matmul, in that order.
///
/// The copy is the KERNEL-less identity the vendor's own gathered fixture spells, split into one
/// op per index stick (`CopyDims::ENTRIES_PER_OP` = 32, so a 64-entry gather is TWO legs —
/// `t205_g0`/`t205_g1`, row-window writes via the leg's `first_row`), with the index operand the
/// direct-read trap would drop; and the matmul reads the MINTED INTERMEDIATE — proven by address,
/// because the descriptor names operands only positionally: the table sits at `TABLE_ADDR`
/// (seg3 + 0x10000) and the intermediate in the synth segment (seg0), so "the matmul's KERNEL
/// operand is not at `TABLE_ADDR`" is exactly "the matmul does not read the table".
#[test]
fn the_gathered_matmul_emits_the_copy_then_the_matmul() {
    let ops = emitted(&paged_program(GatheredSlot::B), GatheredSlot::B)
        .expect("the rung-4 program lowers through the whole-function door");
    // 64 entries / 32 per op = 2 copy legs, then the contraction.
    assert_eq!(
        ops.len(),
        3,
        "a gathered matmul emits two materialized copy legs (one per 32-entry index stick) and \
         the contraction -- got {:?}",
        ops.iter().map(|o| o.op_name.clone()).collect::<Vec<_>>()
    );
    let (copy0, copy1, mm) = (&ops[0], &ops[1], &ops[2]);
    for copy in [copy0, copy1] {
        let dsc = copy.dsc();
        let dsc = dsc
            .dscs_
            .first()
            .and_then(|m| m.values().next())
            .expect("each copy leg has a descriptor");
        // The copy is the vendor's own gathered fixture shape: an `identity` opFunc with the
        // indirect-access index attached, and the value/index alloc pair.
        let identity = dsc
            .computeOp_
            .iter()
            .any(|c| c.opFuncName == "identity" && !c.indirectAccessIndexLabeledDs.is_empty());
        assert!(
            identity,
            "each copy leg is an `identity` carrying the indirect-access index (the wrong-rows \
             trap's whole point): {:?}",
            copy.op_name
        );
        let has_pair = dsc.scheduleTree_.iter().any(|n| {
            n.indirectAllocType_ == "value_tensor" && n.relatedIndirectAccessAlloc_.is_some()
        }) && dsc
            .scheduleTree_
            .iter()
            .any(|n| n.indirectAllocType_ == "index_tensor");
        assert!(
            has_pair,
            "each copy leg's value/index alloc pair is the gather itself: {:?}",
            copy.op_name
        );
        // The copy's VALUE alloc IS the TABLE, at the table's own address.
        assert_eq!(
            value_tensor_address(copy),
            Some(TABLE_ADDR),
            "each copy leg gathers from the table at {TABLE_ADDR:#x}: {:?}",
            copy.op_name
        );
    }
    // The contraction is a matmul.
    let mm_dsc = mm
        .dsc()
        .dscs_
        .first()
        .and_then(|m| m.values().next())
        .expect("the matmul has a descriptor");
    assert!(
        mm_dsc
            .computeOp_
            .iter()
            .any(|c| c.opFuncName == "batchmatmul"),
        "the last op is the contraction: {:?}",
        mm.op_name
    );
    // ⭐ THE MATMUL READS THE INTERMEDIATE, NOT THE TABLE — by address. A direct-read descriptor
    // (the trap this guard exists to refuse) would put the KERNEL operand at `TABLE_ADDR`.
    let b_addr = alloc_address(mm, "KERNEL");
    assert!(
        b_addr.is_some(),
        "the matmul declares a KERNEL (B) operand: {:?}",
        mm.op_name
    );
    assert_ne!(
        b_addr,
        Some(TABLE_ADDR),
        "the matmul's B is at {b_addr:#?}, NOT the table's {TABLE_ADDR:#x} -- a B at the table's \
         address is the direct-read trap (right shape, wrong rows): {:?}",
        mm.op_name
    );
}

/// ⭐⭐⭐ THE A SLOT, RESTATED FOR THE LAW THE WALK NOW RUNS: a gathered read materializes for ANY
/// consumer — the copy is the gather, and what reads its result reads an ordinary activation —
/// so a gather feeding the matmul's **A** emits the same two copy legs and a contraction whose
/// A is the minted intermediate, NOT the table.
///
/// ⛔ THE OLD CONTROL PINNED A DEAD LAW, AND THIS IS ITS REPLACEMENT, NOT A WEAKENING. The old
/// test expected the walk's ScalarMul-only refusal to fire for the A slot; the walk it inherited
/// (38b5c73c9) materializes every gathered read instead, so the refusal never fires and the
/// fixture died at the matmul's own K-extent check — P was left `[8, 64]`, which is not a valid
/// B against a `[64, 64]` A. This fixture is SHAPE-VALID in the A arrangement (every extent
/// square at K), so what it pins is the materialization itself. The fail-closed rule survives
/// as its own case in the test below this one.
///
/// The A operand is proven by ADDRESS, the same discriminator the B-slot positive uses: the
/// table sits at `TABLE_ADDR` (seg3 + 0x10000) and the intermediate in the synth segment, so
/// "the matmul's INPUT operand is not at `TABLE_ADDR`" is exactly "the matmul does not read the
/// table".
#[test]
fn an_a_side_gathered_operand_materializes_and_the_matmul_reads_the_copy() {
    let ops = emitted(&paged_program(GatheredSlot::A), GatheredSlot::A)
        .expect("an A-side gathered operand materializes like any other gathered read");
    assert_eq!(
        ops.len(),
        3,
        "the A-slot gather emits the same two materialized copy legs and the contraction -- got \
         {:?}",
        ops.iter().map(|o| o.op_name.clone()).collect::<Vec<_>>()
    );
    let (copy0, _copy1, mm) = (&ops[0], &ops[1], &ops[2]);
    // The copy legs are the gather itself — the same identity-with-index shape as the B case.
    let dsc = copy0
        .dsc()
        .dscs_
        .first()
        .and_then(|m| m.values().next())
        .expect("each copy leg has a descriptor");
    assert!(
        dsc.computeOp_
            .iter()
            .any(|c| c.opFuncName == "identity" && !c.indirectAccessIndexLabeledDs.is_empty()),
        "the A-slot copy leg is an `identity` carrying the indirect-access index: {:?}",
        copy0.op_name
    );
    assert_eq!(
        value_tensor_address(copy0),
        Some(TABLE_ADDR),
        "the A-slot copy leg gathers from the table at {TABLE_ADDR:#x}: {:?}",
        copy0.op_name
    );
    // ⭐ THE MATMUL READS THE INTERMEDIATE IN ITS **A** SLOT, NOT THE TABLE — by address. A
    // direct-read descriptor (the trap the guard exists to refuse) would put the INPUT operand
    // at `TABLE_ADDR`.
    let mm_dsc = mm
        .dsc()
        .dscs_
        .first()
        .and_then(|m| m.values().next())
        .expect("the matmul has a descriptor");
    assert!(
        mm_dsc.computeOp_
            .iter()
            .any(|c| c.opFuncName == "batchmatmul"),
        "the last op is the contraction: {:?}",
        mm.op_name
    );
    let a_addr = alloc_address(mm, "INPUT");
    assert!(
        a_addr.is_some(),
        "the matmul declares an INPUT (A) operand: {:?}",
        mm.op_name
    );
    assert_ne!(
        a_addr,
        Some(TABLE_ADDR),
        "the matmul's A is at {a_addr:#?}, NOT the table's {TABLE_ADDR:#x} -- an A at the table's \
         address is the direct-read trap (right shape, wrong rows): {:?}",
        mm.op_name
    );
}

/// ⛔ THE FAIL-CLOSED RULE SURVIVES THE MATERIALIZE-ANY-GATHERED-READ LAW, AS ITS OWN CASE: a
/// gathered load with **TWO** consumers is refused by name, because the materialized copy this
/// door emits stands for exactly one — a second consumer needs its own node, and silently
/// sharing the copy would read the first consumer's rows. This is the refusal half the old
/// A-slot control stood for, restated for the law that replaced it.
#[test]
fn a_gathered_load_with_two_consumers_is_refused_by_name() {
    // The B fixture with ONE extra op: the gathered load `prod` feeds BOTH the matmul and a
    // second store of its own — two consumers of one gathered read.
    let mut k = paged_program(GatheredSlot::B);
    let a: &'static Arena = Arena::global();
    let zero = Ssa(4);
    // A second output parameter for the gathered rows themselves.
    let (p_g, v_g, t_g) = (Ssa(20), Ssa(21), Ssa(22));
    let store_g = Ssa(23);
    k.func.arguments = a.args(vec![
        (Ssa(0), IrType::Index),
        (Ssa(1), IrType::Index),
        (Ssa(2), IrType::Index),
        (Ssa(3), IrType::Index),
        (p_g, IrType::Index),
    ]);
    k.bindings.push(BufferId::new(205));
    let shape = |op: Operation<'static>, r: i64, c: i64| {
        op.with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(vec![r, c])))
            .with_attr(a, AttrKey::Dtype, Attr::Dtype(DType::F16))
    };
    // `prod` (Ssa 15) is the gathered load; store it raw beside the contraction.
    let extra = vec![
        shape(
            Operation::new(a, Some(v_g), OpKind::KtdpConstructMemoryView, &[p_g]),
            K,
            HEAD,
        ),
        shape(
            Operation::new(a, Some(t_g), OpKind::KtdpConstructAccessTile, &[v_g, zero, zero]),
            K,
            HEAD,
        ),
        Operation::new(a, Some(store_g), OpKind::KtdpStore, &[Ssa(15), t_g]),
    ];
    let mut ops = k.func.operations.to_vec();
    ops.extend(extra);
    k.func.operations = a.ops(ops);
    let mut l = layout_for(&k, GatheredSlot::B);
    let tid = 205;
    l.ids.borrow_mut().insert(act_name(tid), PlaceId::Act(tid));
    l.placements.insert(
        tid,
        TensorPlacement {
            tid,
            role: SegRole::Activation,
            segment: 3,
            bank: 0,
            offset: 4 * 0x10000,
            size: (K as u64) * (HEAD as u64) * 2,
        },
    );
    let mut sid = 0i64;
    let e = match lower_function(&k, Some(&l), &mut sid, None) {
        Err(e) => e.message,
        Ok(_) => panic!("a gathered load with two consumers must refuse, not share one copy"),
    };
    assert!(
        e.contains("2 consumer(s)"),
        "the refusal names the consumer count and the law (one copy stands for exactly one \
         consumer) -- got {e}"
    );
}

// ── THE POOLED TABLE (b1) ──────────────────────────────────────────────────────────────────
//
// The fixture is `paged_program(GatheredSlot::B)` unchanged; only the LAYOUT changes. The table
// (t202) is placed at ONE ENTRY's footprint (`[1, HEAD]` = 128 B) instead of the whole
// `[V, HEAD]`, and the bundle declares the table's `gather_reach` as the TYPED pair — the entry
// ceiling and the entry unit — minted through the pool's own door:
//
//   * the SWEPT window is 64 × 64 = 4096 elements = 8192 B — 64× the one-entry placement, which
//     the swept-window law (the placement footprint) refuses;
//   * the reach's entry unit must be the width the read itself strides at (HEAD = 64), which a
//     pool at hd=1 states (a stick block is `hd · 64`) and a pool at hd=2 does not — that
//     disagreement is the door's refusal, and the test that pins it FAILS WITHOUT the check.

/// The pool the fixture's table models, at head dim `hd`: one reach minted through the pool's
/// own door (`GatherReach::of_plane`), with a 4-page budget over a 128-entry page (four whole
/// planes — a page smaller than one of its own planes is not a pool, and the door refuses it).
/// At `hd = 1` the pool's stick block IS the fixture's HEAD (64 elements); at `hd = 2` it is
/// twice that, which is exactly the disagreement the door must refuse.
fn fixture_reach(hd: usize) -> Option<ktir_superdsc::sdsc_abstract::GatherReach> {
    let pool = ktir_superdsc::sdsc_abstract::PagedKvPool::new(8, hd);
    let page_stride = 128 * pool.stick_block_bytes();
    ktir_superdsc::sdsc_abstract::GatherReach::of_plane(4, page_stride, pool)
}

/// [`layout_for`] with the table (t202) placed at ONE ENTRY's footprint and a declared
/// `gather_reach` — the pooled-table statement: a slice placement plus the typed bound on the
/// index values that address the pool behind it.
fn pooled_layout_for(
    k: &KtirNode,
    reach: ktir_superdsc::sdsc_abstract::GatherReach,
) -> BundleLayout {
    let mut l = layout_for(k, GatheredSlot::B);
    let tid = k.bindings[1].get();
    l.placements.get_mut(&tid).expect("the table is placed").size = HEAD as u64 * 2;
    l.gather_reach.insert(tid, reach);
    l
}

/// [`emitted`] under a caller-built layout — the pooled cases' door, so the reach and the
/// placement it bounds are stated by the test and not re-derived.
fn emitted_with(k: &KtirNode, layout: &BundleLayout) -> Result<Vec<ktir_superdsc::emit::EmittedOp>, String> {
    let mut sid = 0i64;
    lower_function(k, Some(layout), &mut sid, None).map_err(|e| e.message)
}

/// ⭐ THE POSITIVE: a one-entry placement the swept-window law refuses is ADMITTED under a
/// declared reach — the door checks the read's entry unit against the reach's own (64 = 64) and
/// the mint checks ONE entry's coverage against the slice (64 × 2 B = 128 B ≤ 128 B). The
/// emission is the rung-4 shape itself: two copy legs and the contraction, the legs still
/// gathering FROM the table's own address (the slice placement's base — the pool's base).
#[test]
fn a_pooled_table_gathers_past_its_one_slice_placement() {
    let k = paged_program(GatheredSlot::B);
    let reach = fixture_reach(1).expect("hd=1 states the fixture's own entry unit");
    assert_eq!(
        reach.entry_elems(),
        HEAD as u64,
        "the hd=1 pool's stick block is the fixture's HEAD — the unit the read strides at"
    );
    let ops = emitted_with(&k, &pooled_layout_for(&k, reach))
        .expect("a pooled table's gather is bounded by its reach, not its slice placement");
    assert_eq!(
        ops.len(),
        3,
        "the pooled gather emits the same two copy legs and the contraction -- got {:?}",
        ops.iter().map(|o| o.op_name.clone()).collect::<Vec<_>>()
    );
    for copy in &ops[..2] {
        assert_eq!(
            value_tensor_address(copy),
            Some(TABLE_ADDR),
            "each copy leg gathers from the table's own (slice) placement base: {:?}",
            copy.op_name
        );
    }
}

/// ⛔ THE REFUSAL, ON THE FACT THE DOOR ALONE CAN SEE: the read's ENTRY UNIT. This node gathers
/// 64-element rows, but the declared pool's entries are 128-element stick blocks (an hd=2 pool)
/// — two units for one address. dxp turns a value into an address as `idx * skip_addr + base`,
/// and the host composes those values in the pool's own unit, so a descriptor striding at 64
/// reads every entry at half its address — from a clean bake. The refusal names the op, the
/// table, both widths and the ceiling.
///
/// ⭐ THIS IS THE TEST THAT FAILS WITHOUT THE CHECK: with no unit law (the state this commit
/// fixes), the hd=2 reach is admitted and the emission succeeds — the very defect the audit
/// named, a bound that looks like a law and pins nothing.
#[test]
fn a_pooled_gather_whose_entry_unit_disagrees_with_the_pool_is_refused_by_name() {
    let k = paged_program(GatheredSlot::B);
    let reach = fixture_reach(2).expect("hd=2 states a wider entry unit");
    assert_eq!(
        reach.entry_elems(),
        2 * HEAD as u64,
        "the hd=2 pool's stick block is twice the fixture's HEAD -- the disagreement under test"
    );
    let e = match emitted_with(&k, &pooled_layout_for(&k, reach)) {
        Err(e) => e,
        Ok(_) => panic!(
            "a read striding 64-element entries over a pool of 128-element stick blocks must \
             refuse -- two units for one address"
        ),
    };
    assert!(
        e.contains("strides one index entry at 64"),
        "the refusal names the read's own unit -- got {e}"
    );
    assert!(
        e.contains("128-element stick blocks"),
        "the refusal names the pool's own unit -- got {e}"
    );
}

/// ⛔ CONTROL — THE REACH WIDENS THE INDEXED READ AND NOTHING ELSE: the SAME one-entry
/// placement with NO declared reach still refuses at the placement footprint. Without this, the
/// positive above proves nothing — a door that stopped checking placements would pass it too.
#[test]
fn a_one_slice_table_without_a_reach_still_refuses_at_the_placement() {
    let k = paged_program(GatheredSlot::B);
    let reach = fixture_reach(1).expect("hd=1 states the fixture's own entry unit");
    let mut l = pooled_layout_for(&k, reach);
    let tid = k.bindings[1].get();
    l.gather_reach.remove(&tid);
    let e = match emitted_with(&k, &l) {
        Err(e) => e,
        Ok(_) => panic!("a swept window of 8192B past a 128B placement must refuse"),
    };
    assert!(
        e.contains("exceeds its placement footprint"),
        "the un-reached table keeps the swept-window law -- got {e}"
    );
}
