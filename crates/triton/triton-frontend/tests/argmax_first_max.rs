//! `argmax_first_max` through the WHOLE frontend value path, gated on structure.
//!
//! The real-Triton goldens (`.ttir_raw.mlir` / `.ttir.mlir`) for this fixture do not exist
//! yet -- producing them needs the triton toolchain, and until that run happens this test
//! is the gate: `codegen::compile` from Python source, `opt::make_ttir` on the value, and
//! the OP CENSUS asserted -- the algorithm's every link must be present as the generic op
//! that carries it, and nothing argmax-specific may exist below the kernel. When the
//! goldens land, this file grows the `diff::compare` halves and `make_ttir.rs`/`
//! `fixture_status.rs` take the case into their tables; until then a refusal here fails
//! loudly instead of a golden diff quietly never running.
//!
//! THE ALGORITHM'S OP FOOTPRINT, pinned (see the fixture's docstring for the arithmetic):
//!
//!   * `tt.reduce` × 7 — the whole-row maxima: the global max `m`, the first-group
//!     combine `jG`, the first-block combine `jA`, the first-lane combine `jB`, and
//!     the three read-out combines `g*`, `b*` and `l*`. Each is the golden rank-2 →
//!     rank-1 reduce over the [1, PAD_V] tile's second axis, and the WHOLE ROW IS
//!     ONE TILE (the rmsnorm whole-row block), so there is NO per-stick loop to
//!     unroll — the program the door receives is STRAIGHT-LINE, ~25 ops, which is
//!     the door-lowerable form (a per-stick-loop kernel would splice to ~2300
//!     trips);
//!   * `arith.maxnumf`/`arith.truncf` — each f16 reduce runs in f32 (upstream's own
//!     promotion) and narrows straight back, moe.py's spelling; the max of f16 values
//!     is exactly representable, so the round trip is exact;
//!   * `arith.cmpf` × 4 — the equality mask `(x == m)`, the group one-hot
//!     `(gramp == jG)`, the block one-hot `(bramp == jA)`, and the lane one-hot
//!     `(lramp == jB)`;
//!   * `arith.uitofp` × 4 — the i1 masks' → f16 widening (the 0/1 selectands);
//!   * `arith.mulf` — `mask · gramp`, `mask · selG · bramp`, `mask · selG · sel ·
//!     lramp`, the one-hot `selG · sel · selL`, the three read-outs, and the three
//!     `ones`-table widens of the [1] answers to the [1, BLOCK] output stick;
//!   * ⛔ NO `arith.subf` ANYWHERE — the v5 form inverted the ramps by subtraction
//!     (`NB - jA`, `BLOCK - jB`), but those `[1]`-shaped `splat - tensor` subs have no
//!     whole-stick door emission, so the inversion rides the EQUALITY selection
//!     instead (the increasing 0-based tables read the numbers back out through the
//!     one-hot). A subf sneaking back in is the unlowerable form returning;
//!   * `arith.extf`/`arith.addf`/`arith.fptosi` — the pad law's `x + padoff`
//!     (the f16 offset-row add before the first max) and the ONE fp32 island:
//!     the final `g*·4096 + b*·64 + l*` reaches past fp16's exact range at
//!     vocab width, so it widens, multiply-adds in f32 (exact below 2^24), and
//!     casts to i32 — moe.py's gelu island and the KSPLIT pair's own device
//!     recipe;
//!   * `tt.descriptor_load` × 12 / `tt.descriptor_store` × 3 — the row, the six
//!     staged index tables (the two-level ramps and increasing read-outs),
//!     the two widen tables, the clamp's V-bound table, the pad law's offset
//!     row and the ROW-FEEDBACK TABLE's gathered row in; the i32 stick, the
//!     clamped index stick and the input-row slot out;
//!   * `arith.minnumf` + a second `arith.fptosi` + `tt.load` — the #603 PR3b row
//!     feedback's clamp and hop: the index is clamped ON THE FP32 ISLAND
//!     (`minnumf` against the STAGED [1, BLOCK] f32 V-bound table — the
//!     no-splat law the ones/sixtyfour tables already state; a `tl.full`
//!     constant tile is a `tensor.splat` operand the whole-function door
//!     refuses by design — and the card's gather reads its index from memory,
//!     so arithmetic between a read and a gather offset has no index view to
//!     point at), stored as a twin
//!     i32 stick, and read back through the RAW POINTER (`tt.load`, moe.py's
//!     scratch spelling, `expert_slab`'s indirect-offset shape) to offset the
//!     table load directly. Pinned as EXACTLY ONE `tt.load` and EXACTLY ONE
//!     `minnumf`: the hop is clamp → store → read-back → gather, and a second
//!     `tt.load` or a missing `minnumf` is a different (or dropped) hop. ⛔ NO
//!     `arith.cmpi`/`arith.select`: the clamp-between-read-and-offset form
//!     (`where(tok > V, V, tok)`) is unlowerable — the slab gather's index
//!     view points at memory, never at a computed value;
//!   * `tt.expand_dims`/`tt.broadcast` — the [1] → [1,1] → [1, BLOCK] seams (the golden
//!     shapes: rank-2 tiles, rank-1 reduce results, no scalar anywhere, so no
//!     reduce-to-scalar and no splat of an interior value exists in the kernel).
//!
//! WHAT MUST NOT BE PRESENT: any op whose existence is argmax-specific, any `scf.for`
//! (the straight-line law above), and any `tt.splat`. The whole point of the ladder
//! form is that every op above is the same op `rmsnorm`/`attention`/`moe` use.

mod common;

use triton_frontend::target::Target;
use triton_frontend::{codegen, opt};

/// Compiles the fixture and returns (module, op-name census).
fn compile_argmax() -> (triton_frontend::ttir::Module, Vec<String>) {
    let case = common::argmax_first_max();
    let src = common::fixture_src(&case.fixture);
    let mut m = codegen::compile(&src, &case.spec, Target::spyre())
        .unwrap_or_else(|e| panic!("argmax_first_max did not compile: {e}"));
    let stats = opt::make_ttir_measured(&mut m)
        .unwrap_or_else(|e| panic!("argmax_first_max: make_ttir refused: {e}"));
    // The bridge-two contract, same as every `make_ttir.rs` case: no calls survive, and
    // exactly one function -- `triton_ktir::ir::Module::kernel()` refuses anything else.
    assert_eq!(stats.calls_after, 0, "tt.call survived make_ttir");
    assert_eq!(
        stats.funcs_after, 1,
        "symbol-dce left {} functions",
        stats.funcs_after
    );

    // THE WALK DESCENDS INTO REGIONS: the two passes live inside `scf.for` bodies, so a
    // top-level-only walk would count the loop shells and miss every op that matters.
    // (This exact mistake is why this walk exists as a named helper rather than an inline
    // loop -- an earlier version asserted on a region-blind census and "found" empty
    // bodies that were full.)
    let mut ops = Vec::new();
    fn walk_ops(ops: &mut Vec<String>, blocks: &[triton_frontend::ttir::Block]) {
        for b in blocks {
            for op in &b.ops {
                ops.push(op.name.clone());
                for r in &op.regions {
                    walk_ops(ops, &r.blocks);
                }
            }
        }
    }
    for f in &m.funcs {
        walk_ops(&mut ops, &f.body.blocks);
    }
    (m, ops)
}

fn count(ops: &[String], name: &str) -> usize {
    ops.iter().filter(|o| o.as_str() == name).count()
}

#[test]
fn argmax_first_max_compiles_and_states_the_algorithm_in_generic_ops() {
    let (_m, ops) = compile_argmax();
    // The census, printed for `--nocapture` runs: one line, sorted, so a changed lowering
    // shows exactly which op moved rather than failing an opaque count.
    let mut census = ops.clone();
    census.sort();
    census.dedup();
    eprintln!("argmax_first_max op census: {census:?}");
    eprintln!("argmax_first_max op sequence: {ops:?}");

    // Every link of the algorithm, as the generic op that carries it. SEVEN
    // whole-row reduces: the global max, the first-group combine, the
    // first-block combine, the first-lane combine, and the three read-out
    // combines.
    assert!(count(&ops, "tt.reduce") >= 7, "the seven whole-row maxima");
    assert!(
        count(&ops, "arith.maxnumf") >= 7,
        "the reduces' f32 combine (the promotion)"
    );
    assert!(
        count(&ops, "arith.truncf") >= 7,
        "the reduces' narrow back to f16"
    );
    assert!(
        count(&ops, "arith.extf") >= 2,
        "the fp32 island's two widens"
    );
    assert!(count(&ops, "arith.cmpf") >= 4, "the four equality masks");
    assert!(
        count(&ops, "arith.uitofp") >= 4,
        "the masks' i1 -> f16 widening"
    );
    assert!(
        count(&ops, "arith.mulf") >= 11,
        "mask * ramp, the one-hot, the read-outs, and the ones-table widens"
    );
    // ⛔ AND NO SUBTRACTION: the inversion rides the equality selection, so a subf
    // reappearing is the v5 form (an unlowerable [1]-shaped splat-minus-tensor) back.
    assert_eq!(
        count(&ops, "arith.subf"),
        0,
        "an arith.subf appeared: the kernel inverts through the equality selection, so a \
         subtraction is the v5 `NB - jA` form returning -- a [1]-shaped splat-minus-tensor \
         with no whole-stick door emission"
    );
    assert!(
        count(&ops, "arith.addf") >= 2,
        "the pad law's x + padoff and the island's g*·4096 + b*·64 + l*"
    );
    assert!(
        count(&ops, "arith.fptosi") >= 2,
        "the island's f32 -> i32 cast and the clamped index stick's"
    );
    assert!(
        count(&ops, "tt.descriptor_load") >= 12,
        "the row loads (x, gramp, bramp, lramp, ginc, binc, linc, ones, sixtyfour, \
         the clamp's V-bound table, padoff, and the row-feedback table's gathered row)"
    );
    assert!(
        count(&ops, "tt.descriptor_store") >= 3,
        "the i32 stick store, the clamped index stick store, and the row-feedback \
         input-row slot store"
    );
    // THE ROW FEEDBACK'S CLAMP AND HOP, exactly once each: the index clamped on the
    // fp32 island (minnumf against the staged V-bound table, the no-splat law), the clamped twin stored as
    // an i32 stick, and that stick read back through the raw pointer to offset the
    // table's descriptor load directly. A missing `tt.load` is a dropped feedback
    // (the row slot stays zero and the numeric leg's byte-exact row pin diverges);
    // a second one is a different hop.
    assert_eq!(
        count(&ops, "arith.minnumf"),
        1,
        "exactly one fp minimum (the row-feedback clamp on the fp32 island)"
    );
    assert_eq!(
        count(&ops, "tt.load"),
        1,
        "exactly one scalar read-back of the clamped stick (the row feedback's index hop)"
    );
    // ⛔ AND NO CLAMP IN THE INDEX PATH: the slab gather's index view points at
    // MEMORY, never at a computed value, so a cmpi/select between the read-back and
    // the gather offset is the unlowerable form returning.
    assert_eq!(
        count(&ops, "arith.cmpi"),
        0,
        "an arith.cmpi appeared: the row-feedback clamp rides the fp32 island (minnumf) \
         onto a stored stick -- a compare in the gather's index path is the \
         unlowerable between-read-and-offset form"
    );
    assert_eq!(
        count(&ops, "arith.select"),
        0,
        "an arith.select appeared: the row-feedback clamp rides the fp32 island (minnumf) \
         onto a stored stick -- a select in the gather's index path is the \
         unlowerable between-read-and-offset form"
    );
    assert!(count(&ops, "tt.expand_dims") >= 3, "the [1] -> [1,1] seams");
    assert!(
        count(&ops, "tt.broadcast") >= 3,
        "the [1,1] -> [1,BLOCK]/[1,PAD_V] seams"
    );
    // AND NO SCALAR REDUCE / INTERIOR SPLAT: the shapes are the golden forms, so a scalar
    // sneaking back in (a reduce-to-scalar, a splat of an interior value) is a lowering
    // regression the numeric legs would refuse -- name it here, one rung earlier.
    assert!(
        !ops.iter().any(|o| o == "tt.splat"),
        "a tt.splat survived: the kernel states no runtime-argument splat, so this is a \
         splat of an interior value -- the form to_ktir refuses by design"
    );

    // AND STRAIGHT-LINE, NO LOOPS: the whole row is one tile, so no scf.for exists to
    // unroll -- the door receives ~20 generic ops, not ~2300 spliced trips. A loop
    // sneaking back in is the unroll bomb the door cannot afford.
    assert!(
        !ops.iter().any(|o| o == "scf.for"),
        "an scf.for survived: the kernel states whole-row tiles, so a loop is a lowering \
         regression -- and the trip-unroll would splice it into thousands of descriptors"
    );
}
