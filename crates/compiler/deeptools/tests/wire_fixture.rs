//! THE WIRE READER AGAINST A REAL DUMPED PROGRAM — `~/tmp/phase0/pre/sdsc_0.json`, the fixture
//! `DBO_DEBUG=1` dxp wrote for `0_rmsq_o728`.
//!
//! This test is the parse boundary's first contact with a real scheduled SuperDSC: every field
//! the C++ dumper writes for that program's 30 nodes (8 sync, 7 loop, 6 allocate, 6 transfer,
//! 2 block, 1 compute) must land in a typed value or name a refusal. It pins nothing about the
//! lowering — that is the golden-fixture harness's job — only that the reader accepts what the
//! dumper actually writes.

use deeptools::wire;

const SDSC: &str = include_str!(concat!(
    env!("HOME"),
    "/tmp/phase0/pre/sdsc_0.json"
));

#[test]
fn the_rmsq_fixture_parses() {
    let file = wire::read_file(SDSC).expect("the fixture must parse");
    assert_eq!(file.programs.len(), 1, "one program in the file");
    let program = &file.programs[0];
    assert_eq!(program.name, "0_rmsq_o728");
    assert_eq!(program.num_cores_used, 32);
    assert_eq!(program.target, wire::SenTarget::Sentient);
    assert_eq!(program.ops.len(), 1, "one op (dscs_ entry)");

    let op = &program.ops[0];
    assert_eq!(op.schedule_tree.len(), 30);

    let mut counts = std::collections::BTreeMap::new();
    for node in &op.schedule_tree {
        let key = match &node.kind {
            wire::NodeKind::Block => "block",
            wire::NodeKind::Loop(_) => "loop",
            wire::NodeKind::Transfer(_) => "transfer",
            wire::NodeKind::Compute(_) => "compute",
            wire::NodeKind::Sync(_) => "sync",
            wire::NodeKind::Allocate(_) => "allocate",
        };
        *counts.entry(key).or_insert(0) += 1;
    }
    assert_eq!(
        counts,
        [
            ("allocate", 6),
            ("block", 2),
            ("compute", 1),
            ("loop", 7),
            ("sync", 8),
            ("transfer", 6),
        ]
        .into_iter()
        .collect(),
        "the fixture's node-kind census"
    );
}

/// The deep fields the lowering actually reads — pinned so a parse that silently defaults
/// everything cannot pass the census test above.
#[test]
fn the_rmsq_fixture_carries_its_deep_fields() {
    let file = wire::read_file(SDSC).expect("the fixture must parse");
    let op = &file.programs[0].ops[0];

    // The head allocation's `startAddressCoreCorelet_` is a three-dim fold (core × corelet ×
    // time) whose `data_` keys are JSON-stringified coordinates.
    let alloc = op
        .schedule_tree
        .iter()
        .find(|n| n.name == "allocate-Tensor0_hbm")
        .expect("the fixture's first allocation");
    let wire::NodeKind::Allocate(alloc) = &alloc.kind else {
        panic!("named node is not an allocation");
    };
    assert_eq!(alloc.lds_idx, 0);
    let wire::FoldData::Folded(start) = &alloc.start_address_core_corelet else {
        panic!("the fixture's start address is folded");
    };
    assert_eq!(start.props.len(), 3, "core × corelet × time");
    assert_eq!(start.props[0].factor, 32, "the core fold");
    assert_eq!(start.props[0].label, "core");
    match &start.funcs[0] {
        wire::FoldDimFunc::Map => {}
        other => panic!("first fold func is {other:?}, want Map"),
    }
    // `[1, 0, 0]` → 128 — the second core's slice of the HBM tensor.
    assert_eq!(start.data.get("[1, 0, 0]"), Some(&"128".to_owned()));

    // Its `allocUsers_` names a real node — the dangling-ref refusal would have fired otherwise.
    assert_eq!(
        alloc.alloc_users.get("transfer_lds0_src:hbm_dst:lx"),
        Some(&1)
    );

    // The transfer's source DataInfo carries the same fold shape for `startAddr_`.
    let transfer = op
        .schedule_tree
        .iter()
        .find(|n| n.name == "transfer_lds0_src:hbm_dst:lx")
        .expect("the fixture's hbm→lx transfer");
    let wire::NodeKind::Transfer(transfer) = &transfer.kind else {
        panic!("named node is not a transfer");
    };
    assert_eq!(transfer.src.unit, SenComponent::L3lu);
    assert_eq!(transfer.src.storage, SenComponent::Hbm);
    assert_eq!(
        transfer.dst_vias.len(),
        1,
        "the transfer routes through one destination"
    );
    assert_eq!(transfer.dst_vias[0].loc.unit, SenComponent::L3lu);
    assert_eq!(transfer.dst_vias[0].loc.storage, SenComponent::Lx);
    let wire::FoldData::Folded(start) = &transfer.src_lds_and_loop_offsets.start_addr else {
        panic!("the transfer's source start address is folded");
    };
    assert_eq!(start.data.get("[0, 0, 0]"), Some(&"0".to_owned()));

    // The loop the whole tree hangs from: `loop_ds0_ds1_y`, one unpadded `y` dim.
    let loop_node = op
        .schedule_tree
        .iter()
        .find(|n| n.name == "loop_ds0_ds1_y")
        .expect("the fixture's outer loop");
    let wire::NodeKind::Loop(loop_node) = &loop_node.kind else {
        panic!("named node is not a loop");
    };
    assert_eq!(loop_node.num_id, 0);
    assert_eq!(loop_node.den_id, 1);
    assert_eq!(loop_node.dims.len(), 1);
    assert_eq!(loop_node.dims[0].dim, PrimaryDim::Y);
    assert_eq!(loop_node.dims[0].kind, MetaDimKind::Unpadded);

    // `labeledDs_`'s `memOrg_` back-pointers name the allocate nodes.
    assert_eq!(op.labeled_ds.len(), 3);
    let lds0 = &op.labeled_ds[0];
    assert_eq!(lds0.ds_name, "Tensor0");
    assert_eq!(lds0.segment, Some(LdsSegment::Output));
    assert_eq!(
        lds0.mem_org
            .get(&SenComponent::Hbm)
            .and_then(|m| m.allocate_node.as_deref()),
        Some("allocate-Tensor0_hbm")
    );
    assert_eq!(
        lds0.mem_org
            .get(&SenComponent::Lx)
            .and_then(|m| m.allocate_node.as_deref()),
        Some("allocate_lds0_lx")
    );

    // The one compute node is an RMSQ on the l3lu unit in fp16.
    let compute = op
        .schedule_tree
        .iter()
        .find(|n| matches!(n.kind, wire::NodeKind::Compute(_)))
        .expect("the fixture's compute node");
    let wire::NodeKind::Compute(compute) = &compute.kind else {
        unreachable!("matched above");
    };
    assert_eq!(compute.ex_unit, SenComponent::Sfp);
    assert_eq!(compute.ty, WireComputeType::Fma16);
    assert_eq!(compute.data_format, DataType::Sen169Fp16);
    assert_eq!(compute.num_folds_engaged, 1);
}

use deeptools::generated::DataType;
use deeptools::wire::{LdsSegment, MetaDimKind, PrimaryDim, WireComputeType};
use sys_arch_spec::arch_enums::SenComponent;
