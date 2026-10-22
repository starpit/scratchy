//! The minted ArgMax node's splice seat, pinned end to end: the SAME `lower`
//! every other spliced Triton kernel takes, over the SAME mint the codegen seam
//! performs — the op_name by the builder's naming law, the program, the
//! parameter count, the grid, and the buffer pairing (the widened row first,
//! the ten tables in operand order, the row-feedback table, the
//! clamped-index stick, the embed row slot, the idx stick LAST). The golden's
//! own discipline: what the card launches is what this test pins.

use ktir_superdsc::ktir_node::Program;
use scratchy_subtile::lower::{GemmWeight, InputRef, OpDesc};
use scratchy_subtile::subtile_ir::{
    SourceShape, SubOp, TensorId, argmax_pad_vocab, lower_region, mint_argmax,
};
use scratchy_triton_splice::lower as splice_lower;

#[test]
fn the_minted_argmax_lowers_through_the_splice_like_every_other_kernel() {
    let nz = |k| std::num::NonZeroU32::new(k).unwrap();
    // K=1, vocab 5: the one-request lm-head shape the mint admits, at the
    // smallest vocab that still pads (5 -> PAD_V 8). The third source is the
    // embed row slot the caller names — hidden 3, DISTINCT from every minted
    // width so a wrong-source binding fails loudly.
    let input = scratchy_subtile::lower::LoweringInput {
        exported: None,
        sources: vec![
            SourceShape { rows: 1, cols: 1 },
            SourceShape { rows: 1, cols: 5 },
            SourceShape { rows: 1, cols: 3 },
        ],
        ops: vec![OpDesc {
            op: SubOp::MatmulTile {
                n: 5,
                weight: GemmWeight::Dense,
            },
            m: 1,
            inputs: vec![InputRef::Ext(0), InputRef::Ext(1)],
        }],
        result: 0,
    };
    let mut g = lower_region(&input, nz(64));
    let minted = mint_argmax(&mut g, TensorId::from_index(2));
    assert_eq!((minted.vocab, minted.pad_vocab), (5, argmax_pad_vocab(5)));
    assert_eq!(
        (
            g.tensors[minted.table.index()].rows,
            g.tensors[minted.table.index()].cols
        ),
        (6, 3),
        "the row-feedback table is [vocab+1, hidden]"
    );

    let node = g.nodes.last().unwrap();
    let op = splice_lower(node, &g, false)
        .expect("the minted argmax lowers through the splice, like every other kernel");

    // The naming law: `{program_stem}_s{node id}` — the builder's own law, what
    // the golden and the emulator's function key both read.
    assert_eq!(op.op_name, format!("argmax_s{}", node.id.index()));

    // A KTIR op carries its PROGRAM; the descriptor is None — that IS the answer.
    assert!(op.op.is_none(), "a spliced op has no SuperDSC descriptor");
    let k = op.ktir.as_ref().expect("a spliced op carries its program");
    assert_eq!(k.program, Program::ArgMax);
    // Fifteen kernel parameters: the row, the ten tables, the row-feedback
    // table, the clamped-index stick, the embed row slot, the idx stick out —
    // the constexprs (V, PAD_V, NB, BLOCK, W) fold into the program, not the
    // signature.
    assert_eq!(
        k.func.arguments.len(),
        15,
        "the compiled program takes the fifteen pointer parameters"
    );
    assert_eq!(
        k.func.grid,
        (1, 1, 1),
        "one work item: the whole row is one tile"
    );

    // The buffer pairing: the tape's real tensor indices, the operand prefix,
    // the output LAST — the widened row (renumbered to 15), the ten minted
    // table sources [3, 13), the row-feedback table 13, the clamped-index
    // stick 14, the embed row slot 2, the idx stick 16.
    let bound: Vec<u32> = k.bindings.iter().map(|b| b.get()).collect();
    assert_eq!(
        bound,
        vec![15, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 2, 16],
        "the row first, the ten tables, the table, the idx stick, the row \
         slot, the stick LAST"
    );
    assert!(k.mask.is_none(), "the argmax reads no runtime length mask");
}
