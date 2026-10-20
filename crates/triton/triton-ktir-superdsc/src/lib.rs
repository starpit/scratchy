// SPDX-License-Identifier: Apache-2.0

//! ADAPTER: our KTIR value -> `ktir-superdsc`'s request type and per-kind entry points.
//!
//! # WHAT THIS IS AND IS NOT
//!
//! It is NOT a lowering. Every descriptor decision belongs to `ktir-superdsc`, whose output is
//! validated by byte-identity against a bake proven on silicon. This crate's whole job is to
//! present our KTIR in the shape that crate's door takes:
//!
//! ```text
//!   our Module --(triton_ktir::handoff)--> ktir_core::IRFunction
//!                                              |
//!                                              +--> KtirNode { func, program, bindings, ... }
//!                                                        |
//!                                                        +--> emit::{rmsnorm, attn_at, ...}
//! ```
//!
//! # THE FIVE FIELDS, AND WHICH ARE HARD
//!
//! `KtirNode` is `{ func, program, bindings, mask, node_out_tid }`. Four are mechanical:
//!
//! * `func` — [`triton_ktir::handoff::lower`] already builds it. Their type wants `'static`, so
//!   the arena is [`ktir_core::arena::Arena::global`].
//! * `bindings` — a `BufferId` per parameter, in order. Their crate treats these as opaque keys
//!   and never asks what a buffer is, so parameter position IS the numbering.
//! * `mask` / `node_out_tid` — `None` for everything we emit today.
//!
//! **`program` is the one that is not mechanical**, and it is the whole open question:
//! [`Program`] names what a program COMPUTES (`RmsNorm`, `Attn`, `Matmul`,
//! `Elementwise(Silu)`, …). Their producer knows it because it built the node from a tape.
//! We have to say it about a Triton kernel.
//!
//! # WHY THAT IS NOT AUTOMATICALLY A "RECOGNIZER"
//!
//! Their `Program` doc says the kind "is NOT derivable … a pointwise KTIR body is the same
//! op-DAG shape whichever function it applies". That is true of THEIR producer's KTIR and
//! false of ours: we emit `arith.addf` where an add is meant and `arith.mulf` where a multiply
//! is, so `Elementwise(Add)` is a 1:1 op mapping, not pattern matching.
//!
//! The fused kinds are the real question, and there are two honest answers — the choice is the
//! owner's, not this crate's:
//!
//! 1. **Source provenance.** Triton names these things: a `@triton.jit` helper called `rmsnorm`,
//!    a `tl.sigmoid` call the frontend sees BEFORE inlining, the kernel's own name. Carried as
//!    an attribute, the kind is a fact the PROGRAMMER stated -- not one the compiler inferred.
//! 2. **Structural matching.** Recognise the silu longhand, the rmsnorm shape, the online
//!    softmax. This is what "no kernel recognizers, no algorithm names in the compiler" rules
//!    out, and it is fragile in exactly the way that constraint anticipates.
//!
//! Until that is settled, [`drive`] takes the `Program` as an ARGUMENT. The driver states it per
//! fixture, so bring-up is not blocked on the design decision and no classifier is smuggled in
//! by accident.

use ktir_core::arena::Arena;
use ktir_superdsc::emit::{self, EmittedOp};
use ktir_superdsc::ktir_node::{BufferId, KtirNode, Program};
use ktir_superdsc::placement::BundleLayout;

pub mod bake;
// THE FOURTEEN CONFIGURATIONS. `examples/bake_py` (the dxp bake) and `triton-numeric` (the
// emulator's NUMERIC check) both need "what is `rope_q32`?" answered, and a configuration restated
// in two places is one that drifts in one of them -- see the module's own `rope_q32` note, where a
// second statement of the grid left 252 of 256 token positions unwritten and still baked.
pub mod cases;
pub mod layout;

/// What went wrong, and at which stage.
#[derive(Debug)]
pub struct Error {
    pub stage: &'static str,
    pub message: String,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}] {}", self.stage, self.message)
    }
}

/// Our Triton-free module as their request type.
///
/// Run [`triton_ktir::passes::to_ktir`] first — [`triton_ktir::handoff::lower`] refuses every
/// `tt.*` op by name, which is what keeps a half-converted module from reaching a device program.
pub fn node_for(m: &triton_ktir::ir::Module, program: Program) -> Result<KtirNode, Error> {
    let func = triton_ktir::passes::to_ktir_emit::lower(m, Arena::global()).map_err(|e| Error {
        stage: "handoff",
        message: e.to_string(),
    })?;
    // PARAMETER POSITION IS THE BUFFER NUMBER. Their crate documents `BufferId` as opaque —
    // it renders an operand name for it and looks it up in a `BundleLayout`, never asking what
    // the buffer holds, because extents/strides/format all come from the program's own
    // `ktdp.construct_memory_view`. So there is nothing to invent here.
    let bindings: Vec<BufferId> = (0..func.arguments.len())
        .map(|i| BufferId::new(i as u32))
        .collect();
    Ok(KtirNode {
        func,
        program,
        bindings,
        mask: None,
        node_out_tid: None,
    })
}

/// Hand one node to the entry point its `Program` names, under a layout DERIVED FROM THE PROGRAM.
///
/// The layout comes from [`layout::for_regions`] over the node's own
/// [`regions`](emit::lower_ktir_to_superdsc::regions) — the same parameter walk the chosen body
/// reads, so the memory plan and the descriptors that address through it are built from ONE reading
/// of the program. See [`layout`] for why `None` is not an option for two of the bodies.
///
/// [`emit_node_with`] is the same call with the layout stated explicitly; it exists so the
/// no-layout arm stays MEASURABLE from the same binary, and so a caller that wants to INSPECT the
/// plan can build it itself.
pub fn emit_node(node: &KtirNode) -> Result<Vec<EmittedOp>, Error> {
    let regions = emit::lower_ktir_to_superdsc::regions(node).map_err(|e| Error {
        stage: "regions",
        message: format!("{e:?}"),
    })?;
    let l = layout::for_regions(node, &regions)?;
    emit_regions(node, &regions, Some(&l))
}

/// [`emit_node`] with the layout stated by the caller — `None` selects `ktir-superdsc`'s unit-test
/// arm (per-op `segment_base(arg_index)`, no scale registry). The DRIVER's `KTIR_NO_LAYOUT=1` path,
/// kept so "did adding a layout change this verdict?" is a question one binary can answer.
pub fn emit_node_with(
    node: &KtirNode,
    layout: Option<&BundleLayout>,
) -> Result<Vec<EmittedOp>, Error> {
    let regions = emit::lower_ktir_to_superdsc::regions(node).map_err(|e| Error {
        stage: "regions",
        message: format!("{e:?}"),
    })?;
    emit_regions(node, &regions, layout)
}

fn emit_regions(
    node: &KtirNode,
    regions: &[emit::lower_ktir_to_superdsc::Region],
    layout: Option<&BundleLayout>,
) -> Result<Vec<EmittedOp>, Error> {
    let mut sym = 0i64;
    let name = node.func.name;

    let out = match node.program {
        Program::RmsNorm => {
            emit::lower_ktir_to_superdsc::rmsnorm(name, node, regions, &mut sym, layout)
        }
        Program::ScalarMul => {
            emit::lower_ktir_to_superdsc::scalarmul(name, node, regions, &mut sym, layout)
        }
        Program::Elementwise(kind) => {
            // NO BROADCAST FLAGS: this per-NODE door takes regions the caller already resolved and
            // has no program to prove a broadcast chain from. The whole-function walk is what
            // recognises `m[:, None]` and states the axes; here an empty slice says "every operand is
            // dense", which is what a `Program::Elementwise` node with resolved regions is.
            emit::lower_ktir_to_superdsc::elementwise(name, kind, regions, &[], &mut sym, layout)
        }
        Program::SiluMul => emit::lower_ktir_to_superdsc::silumul(name, regions, &mut sym, layout),
        Program::LmLast => {
            emit::lower_ktir_to_superdsc::lmlast(name, node, regions, &mut sym, layout)
        }
        Program::Transpose => {
            emit::lower_ktir_to_superdsc::transpose(name, regions, &mut sym, layout)
        }
        // `matmul` threads a per-BUNDLE fp8 activation-quantize dedup set. One node at a time
        // here, so a fresh set per call is correct; a real bundle walk threads one across nodes.
        //
        // ⛔ AND IT NEEDS THE B ORIENTATION, WHICH IS READ FROM THE NODE'S OWN `indexing_maps` — never
        // defaulted. The framing of the W REGION is `[n, k]` for the transpose-B form and `[k, n]` for
        // the plain one, the extent guards inside `matmul` read that framing, and at `k == n` neither
        // is recoverable from the extents (granite's `[4096, 4096]` output projection). This door
        // lowers ONE program per node, so it reads the node's `linalg.matmul` ops and requires them to
        // agree; the whole-function door proves it per op, which is what a multi-matmul kernel needs.
        Program::Matmul => {
            let mms: Vec<_> = node
                .func
                .operations
                .iter()
                .filter(|o| o.op_type == ktir_core::opkind::OpKind::LinalgMatmul)
                .collect();
            let mut orient = None;
            let mut disagree = false;
            for op in &mms {
                match emit::whole_function::matmul_b_orientation(&node.func, op) {
                    Err(e) => {
                        return Err(Error {
                            stage: "b-orientation",
                            message: e.message,
                        });
                    }
                    Ok(b) => match orient {
                        None => orient = Some(b),
                        Some(prev) if prev != b => disagree = true,
                        Some(_) => {}
                    },
                }
            }
            let Some(b) = orient.filter(|_| !disagree) else {
                return Err(Error {
                    stage: "b-orientation",
                    message: format!(
                        "{name}: this node states `Program::Matmul` but its function holds {}                          `linalg.matmul` op(s){} — so there is no ONE proven B orientation for the                          single descriptor this door emits. The orientation decides which of the W                          region's two extents is K and cannot be recovered from the extents at                          `k == n`, so it is refused rather than assumed. The whole-function door                          (`emit_whole`) proves it per op.",
                        mms.len(),
                        if disagree {
                            ", and they do not agree"
                        } else {
                            ""
                        }
                    ),
                });
            };
            let mut quantized = std::collections::HashSet::new();
            emit::lower_ktir_to_superdsc::matmul_oriented(
                name,
                regions,
                &mut sym,
                layout,
                &mut quantized,
                emit::lower_ktir_to_superdsc::MatmulProof {
                    b,
                    // The per-`Program` door's operands are whole staged tensors, not windows of
                    // the caller's parameters — the fact the spurious-pad drop discriminates on.
                    origin: emit::lower_ktir_to_superdsc::OperandOrigin::Staged,
                    // ⭐ NO HEAD-MAJOR o_proj HANDOFF HERE. That fact is minted from a spyre
                    // BUNDLE's attention-output row count (mq == 1 over a multi-slab head dim) and
                    // lives on the spyre door's `BundleAttnParams`; this Triton producer states no
                    // bundle facts, so it keeps the pre-restructure emission.
                    headmajor: None,
                },
            )
        }
        // THE NOT-YET-LOWERED VOCABULARY, REFUSED BY NAME — enumerated, never `_`, so a new
        // `Program` variant is E0004 here rather than a surprise at emission. None of these has a
        // row on this ladder: the splice's registry produces none of them (its by-name refusals
        // stop their SubOps at the producer), so a node carrying one here is a ROUTING CHANGE, not
        // a missing arm — the day the splice learns to produce one, its arm moves out of this
        // group and takes a real entry point with it. (The MoE programs are no longer variants at
        // all — the MoE lowers only from `kernels/moe.py` through KTIR.)
        Program::RmsNormUnit
        | Program::TanhSoftCap
        | Program::ScalarWeightMul
        | Program::Reshape
        | Program::WindowedRmsNorm { .. } => {
            return Err(Error {
                stage: "program",
                message: format!(
                    "{:?} has no entry point on the Triton ladder — the splice produces none of \
                     these, so a node carrying one here is a routing change, not a missing arm",
                    node.program
                ),
            });
        }
        // ⛔ `Program::Attn` AND `Program::Rope` ARE REFUSED FOR THE SAME REASON: their
        // pattern-matched doors (`drive_attn` → `attn_at::<NQH, NKVH, HD>`, `drive_rope` →
        // `rope_at::<HD>` — scratchy's prebuilt fragment assemblies selected by constexpr
        // geometry match) are REMOVED. Both lower from their compiled ops through the
        // whole-function door (`emit_whole`), and a per-`Program` dispatch would be a second,
        // hand-rolled path. Do not re-add one.
        Program::Attn | Program::Rope => {
            return Err(Error {
                stage: "program",
                message: format!(
                    "{:?} has no per-`Program` entry point — its hand-rolled geometry door is \
                     deleted, and it lowers from its compiled ops through the whole-function \
                     door (`emit_whole`) instead",
                    node.program
                ),
            });
        }
    };
    // ⛔ Every body here is held to the column windows its program states, the spyre door's law
    // (`column_windows_are_honoured`); a whole-tensor node passes without looking.
    out.and_then(|ops| {
        emit::lower_ktir_to_superdsc::column_windows_are_honoured(name, regions, &ops)?;
        Ok(ops)
    })
    .map_err(|e| Error {
        stage: "ktir-superdsc",
        message: format!("{e:?}"),
    })
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
//  THE MODEL-GEOMETRY CONST DOOR
//
//  `rope_at::<HD>` and `attn_at::<NQH, NKVH, HD>` are const-generic because the head geometry
//  parameterises the DEVICE layout — GQA grouping, slabs, head strides — so a branch on it is a
//  branch on a const the compiler can hold onto (`rope_at`'s own doc: "the collapsed RoPE form and
//  the slab RoPE form become TWO INSTANTIATIONS rather than two arms of one function"). A runtime
//  `Program` cannot reach them; something has to name the triple.
//
//  ⛔ THE VALUES ARE THE FIXTURE'S OWN `tl.constexpr`, READ, NEVER DEFAULTED. `HEAD_DIM`, `H` and
//  `GQA` are constexpr parameters of the kernels themselves (`attention_flash.py` line 339-343,
//  `rope.py` line 132-135) and are recorded per configuration in
//  `test/experiment1/index.json`'s `constexprs`, which `run_experiment1.py` fills from the
//  fixture's own `constexprs()` (line 174/223, `constexprs=ce`). A DEFAULT here would be a wrong
//  GQA grouping — every query head reading the wrong kv plane — with nothing to catch it, so the
//  driver refuses a configuration whose keys it cannot read.
//
//  ⛔ AND `GQA` IS THE GROUPING, NOT THE KV HEAD COUNT. `attention_flash.py`: "a query head reads
//  the kv plane it shares, `off_h // GQA`" and "the kv planes are nkvh-deduped, so their descriptor
//  extent is `H // GQA` planes". So NQH = H and NKVH = H / GQA — the division is the whole point,
//  and inverting it is exactly the silent mis-grouping the const door exists to make visible.
// ══════════════════════════════════════════════════════════════════════════════════════════════

// ⛔ `drive_rope` IS DELETED (2026-10-19). It was the model-geometry door: read `HEAD_DIM`/`H`/
// `N_TOK` off a `Program::Rope` kernel's constexprs, cross-check them against the output view, and
// emit `rope_at::<HD>`'s hand-rolled rope body — substituting a prebuilt implementation for the
// given Triton code, exactly the shape `drive_attn` had. Rope goes through the whole-function door
// (`bake_py` `KTIR_WHOLE=1`): the compiled fixture's ops, lowered one at a time, the per-core
// uniform corners resolved as bands and the rotate-half pairs' unequal column windows carried
// per-operand by the pointwise emission (`rope_q32`/`rope_kv8` bake this way, 48 ops each).
// `rope_at`/`RopeAt` in the vendored `ktir-superdsc` crate are deleted with it, and so is
// `RopeRows` — the row order it stated is a fact of the ops the whole door now reads, not a
// parameter anyone has to name. Do not re-add this door.

// ⛔ `drive_attn` IS DELETED (2026-09-29). It was the pattern-match door: read `H`/`GQA`/
// `HEAD_DIM` off a `Program::Attn` kernel's constexprs, pick the matching
// `attn_at::<NQH, NKVH, HD>` instantiation, and emit scratchy's prebuilt attention fragment
// assembly — substituting a hand-rolled implementation for the given Triton code. Attention
// goes through the whole-function door (`bake_py` `KTIR_WHOLE=1`): the compiled fixture's ops,
// lowered one at a time. `attn_at` itself stays in the vendored `ktir-superdsc` crate (scratchy
// still uses it); it is simply no longer reachable from our layer. Do not re-add this door.

/// THE HANDED-OFF FUNCTION AS THEIR MATCHERS READ IT, one line per op.
///
/// An instrument, not part of the compile path. Their structural readings (`program_rmsnorm_eps`,
/// `program_score_scale`, …) walk `func.operations` / `ops_deep()` looking for exact op-kind chains,
/// so when one refuses the only way to see WHY is to read the same list it read. `Debug` on
/// `IRFunction` prints the arena's whole graph and is unusable for that.
pub fn dump(node: &KtirNode) -> String {
    use std::fmt::Write as _;
    let f = &node.func;
    let mut s = String::new();
    let _ = writeln!(
        s,
        "@{} args={} grid={:?} top={} deep={}",
        f.name,
        f.arguments.len(),
        f.grid,
        f.operations.len(),
        f.ops_deep().len()
    );
    fn walk(ops: &[ktir_core::ir::Operation<'static>], depth: usize, s: &mut String) {
        use std::fmt::Write as _;
        for o in ops {
            let pad = "  ".repeat(depth + 1);
            let res = o.result.map(|r| format!("%{} = ", r.0)).unwrap_or_default();
            let ins: Vec<String> = o.operands.iter().map(|v| format!("%{}", v.0)).collect();
            let attrs: Vec<String> = o
                .attributes
                .iter()
                .map(|(k, v)| format!("{k:?}={v:?}"))
                .collect();
            let _ = writeln!(
                s,
                "{pad}{res}{:?}({}) {}",
                o.op_type,
                ins.join(", "),
                attrs.join(" ")
            );
            for r in o.regions {
                walk(r, depth + 1, s);
            }
        }
    }
    walk(f.operations, 0, &mut s);
    s
}

/// Lower a WHOLE-KERNEL function through `ktir-superdsc`'s whole-function door.
///
/// The per-`Program` entry points read a function's parameter list as one op's operands, which is
/// exact for scratchy's one-node-per-function producer and wrong for ours -- and for IBM's C++
/// reference producer, which puts 3 `linalg.matmul` in one function for a SwiGLU MLP and 12 for a
/// decoder layer. `emit::whole_function::lower_function` walks the ops instead, handing each body
/// only its own operands' regions. It recognises nothing: an op with no 1:1 `Program` is refused by
/// name rather than assigned a node kind on the producer's behalf.
pub fn emit_whole(node: &KtirNode, layout: Option<&BundleLayout>) -> Result<Vec<EmittedOp>, Error> {
    let mut sym = 0i64;
    ktir_superdsc::emit::whole_function::lower_function(node, layout, &mut sym, None).map_err(|e| {
        Error {
            stage: "whole-function",
            message: format!("{e:?}"),
        }
    })
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
//  THE RECOGNIZER BRIDGE — one module, both `uniform_corners`, one contract.
//
//  The uniform-corner family is stated TWICE by design (this crate's leaf, `ktir-superdsc`, cannot
//  depend on the pass crate): `triton_ktir::passes::uniform_grid_corners::uniform_corners` decides
//  what the pass mints, and `ktir-superdsc`'s twin decides what the door resolves. Two statements
//  of one family are free to drift, and the one drift that already happened is why this test
//  exists: the twin unpacked `arith.index_cast` as a two-operand op, so it skipped every cast —
//  and the pass's minted spelling is `index_cast(muli(index_cast(landmark), S))`, casts all the
//  way down. The twin resolved NOTHING the pass shipped, and silently: a corner it cannot claim
//  is a refusal at the door, not a wrong answer, so nothing downstream ever compared the two.
//
//  So: both recognizers over one module, and the per-tile corner forms must AGREE. Two witnesses,
//  because the family has two spellings: the MINTED one (the pass's own `AFFINE` fixture, a
//  delinearized corner the pass rewrites) and the SHIPPED one (`mul_scale_small`, whose corner
//  arrives already canonical — the fire gate's exemption shape). The comparison is per tile and
//  slot, never per SSA — the handoff (`to_ktir_emit::lower`) owns its own numbering.
// ══════════════════════════════════════════════════════════════════════════════════════════════
#[cfg(test)]
mod uniform_corners_bridge {
    use super::*;
    use ktir_superdsc::ktir_node::Elementwise;
    use triton_frontend::target::Target;
    use triton_frontend::{codegen, opt};

    /// The pass's own minting witness, from `uniform_grid_corners`'s tests: a sibling crate cannot
    /// import another's test fixture, and the assertions below re-derive the pass's own, so if
    /// either statement of the text moves, a test here fails rather than the two drifting apart.
    /// TWO deliberate differences from the pass's copy, both demanded by the road to the door: the
    /// pointer argument is stated as `index` — the form `to_ktir`'s own pointer rewrite leaves it
    /// in, because the handoff refuses a surviving pointer — and the affine sets/maps are stated
    /// as the parser's `#set`/`#map` aliases, its documented spelling for an attribute dictionary
    /// (an inline `affine_set` carries `>=`, which the dictionary's depth counter reads as a
    /// closing bracket).
    const AFFINE: &str = "\
#set0 = affine_set<(d0, d1) : (d0 >= 0, d1 >= 0)>
#map0 = affine_map<(d0, d1) -> (d0, d1)>
#set1 = affine_set<(d0, d1) : (d0 >= 0, d1 >= 0)>
module {
  func.func @k(%q: index) attributes {grid = [16 : index]} {
    %tid = ktdp.get_compute_tile_id : index
    %c4 = arith.constant 4 : index
    %xm = arith.remui %tid, %c4 : index
    %x = arith.index_cast %xm : index to i32
    %yd = arith.divui %tid, %c4 : index
    %y = arith.index_cast %yd : index to i32
    %c64 = arith.constant 64 : i32
    %c256 = arith.constant 256 : i32
    %m = arith.muli %x, %c64 : i32
    %h = arith.muli %y, %c256 : i32
    %corner = arith.addi %h, %m : i32
    %c0 = arith.constant 0 : i32
    %v = ktdp.construct_memory_view %q, sizes: [1024, 64], strides: [64, 1] {coordinate_set = #set0, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<1024x64xf16>
    %t = ktdp.construct_access_tile %v[%corner, %c0] {access_tile_order = #map0, access_tile_set = #set1} : memref<1024x64xf16> -> !ktdp.access_tile<64x64xindex>
    return
  }
}
";

    /// Both recognizers over one module; the per-tile corner forms must AGREE. Returns the
    /// pass-side forms, per tile and per corner slot (slot 0 is the view), so each witness can pin
    /// its own specifics. `None` on both sides is a constant corner — agreed absence is agreement.
    /// The comparison is per tile and slot, never per SSA — the handoff (`to_ktir_emit::lower`)
    /// owns its own numbering.
    fn assert_the_forms_agree(
        m: &triton_ktir::ir::Module,
        program: Program,
    ) -> Vec<Option<(i64, i64)>> {
        // ── THE PASS'S HALF — over the module as the pass left it ──
        let kernel = m.kernel().expect("one kernel");
        let body = kernel.regions[0].ops.clone();
        let landmark = body
            .iter()
            .find(|o| o.kind == triton_ktir::ir::OpKind::KtdpGetComputeTileId)
            .and_then(|o| o.result())
            .expect("the launch states its position");
        let index = m.def_index();
        let shipped =
            triton_ktir::passes::uniform_grid_corners::uniform_corners(&body, &index, landmark);

        // ── THE DOOR'S HALF — over the node the door actually receives ──
        let node = node_for(m, program).expect("handoff");
        let consts = ktir_superdsc::emit::lower_ktir_to_superdsc::index_constants(&node.func);
        let door =
            ktir_superdsc::emit::lower_ktir_to_superdsc::uniform_corners(&node.func, &consts);

        let mut pass_corners: Vec<Option<(i64, i64)>> = Vec::new();
        for o in body.iter() {
            if o.kind == triton_ktir::ir::OpKind::KtdpConstructAccessTile {
                for slot in 1..o.operands.len() {
                    pass_corners.push(shipped.get(&o.operands[slot]).map(|u| (u.stride, u.base)));
                }
            }
        }
        let mut door_corners: Vec<Option<(i64, i64)>> = Vec::new();
        for o in node.func.operations.iter() {
            if o.op_type == ktir_core::opkind::OpKind::KtdpConstructAccessTile {
                for slot in 1..o.operands.len() {
                    door_corners.push(door.get(&o.operands[slot]).map(|u| (u.stride, u.base)));
                }
            }
        }
        assert_eq!(
            pass_corners, door_corners,
            "the pass's recognizer and the door's twin disagree about this module's tile corners"
        );
        // NON-VACUOUS: something is claimed — else the equality above is `None == None` over an
        // empty contract.
        assert!(
            pass_corners.iter().any(|c| c.is_some()),
            "no tile corner is uniform — the recognizers agreed over nothing"
        );
        pass_corners
    }

    /// THE MINTED SPELLING. The `AFFINE` witness: the [4,4] Q/O corner arithmetic
    /// (`start_m = ctid % 4`, `off_hz = ctid / 4`, corner `= off_hz·256 + start_m·64`) lands on
    /// `ctid·64`, so the pass rewrites it to the canonical `muli(index_cast(landmark), 64)`. The
    /// module goes through `to_ktir` — the pipeline that owns the pointer rewrite and runs the
    /// corner pass — because that is the only road to the handoff, and the handoff is the road to
    /// the door. The door's twin must claim exactly what the pass minted — and the mint's
    /// `index_cast` is UNARY, the operand the twin once unpacked as two-operand and skipped, which
    /// is the drift this bridge exists to hold shut. `Elementwise(Mul)` is a TAG ONLY: nothing is
    /// emitted here, and neither recognizer reads it.
    #[test]
    fn the_door_reads_what_the_pass_mints() {
        let mut m = triton_ktir::text::parse::parse(AFFINE).expect("parse the witness");
        triton_ktir::passes::to_ktir::run(&mut m, &[4, 4]).expect("to_ktir mints the corner");

        // The pass's own chain assertion, restated: the tile's row corner IS the mint — a `muli`
        // over a UNARY `index_cast` of the landmark.
        let body = m.kernel().unwrap().regions[0].ops.clone();
        let tile = body
            .iter()
            .find(|o| o.kind == triton_ktir::ir::OpKind::KtdpConstructAccessTile)
            .expect("one tile");
        let row = tile.operands[1];
        let index = m.def_index();
        let head = index.def_of(row).expect("the corner is computed");
        assert_eq!(head.kind, triton_ktir::ir::OpKind::ArithMuli);
        let cast = index
            .def_of(head.operands[0])
            .expect("the stride's operand");
        assert_eq!(cast.kind, triton_ktir::ir::OpKind::ArithIndexCast);
        assert_eq!(cast.operands.len(), 1, "the mint's cast is unary");

        let forms = assert_the_forms_agree(&m, Program::Elementwise(Elementwise::Mul));
        assert_eq!(
            forms,
            vec![Some((64, 0)), None],
            "row corner minted as ctid·64, column corner constant"
        );
    }

    /// THE SHIPPED SPELLING. `mul_scale_small` compiles from the real fixture with its corner
    /// ALREADY canonical (`index_cast(muli(index_cast(landmark), BLOCK))` — the fire gate's own
    /// exemption shape, casts all the way down): no rewrite happens, the launch stays, and the
    /// door still has to resolve what shipped. This is the spelling production modules carry and
    /// the whole-function door actually bakes, so the bridge runs on the chain as the door
    /// receives it, not only as the pass writes it.
    #[test]
    fn the_door_reads_the_shipped_spelling() {
        let (kspec, grid, program) =
            cases::case("mul_scale_small").expect("the case table states the configuration");
        let src = std::fs::read_to_string(&kspec.file).expect("read the fixture");
        let mut tt = codegen::compile(&src, &kspec, Target::spyre()).expect("codegen");
        opt::make_ttir(&mut tt).expect("make_ttir");
        let mut m = triton_ktir::from_ttir::convert(&tt).expect("from_ttir");
        triton_ktir::make_ktir(&mut m, &grid).expect("make_ktir");
        triton_ktir::passes::to_ktir::run(&mut m, &grid).expect("to_ktir");

        let forms = assert_the_forms_agree(&m, program);
        // The module's tiles: two with constant corners (the `None`s — agreed absence on both
        // sides) and two whose corner is the per-core uniform one. Whatever is CLAIMED is the
        // shipped form, and something is.
        assert!(
            !forms.is_empty()
                && forms.iter().flatten().all(|f| *f == (64, 0))
                && forms.contains(&Some((64, 0))),
            "mul_scale's claimed corners are pid·BLOCK = pid·64 — got {forms:?}"
        );
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
//  ROPE THROUGH THE WHOLE DOOR, DRIVEN FROM THE FIXTURE — the contract the geometry arm's
//  deletion leaves behind.
//
//  Rope used to stop at a model-geometry const door here ([`drive_rope`] over the deleted
//  `RopeAt`); that door is deleted, and with it the layout's rotate-matrix placement — no op of
//  the compiled body names `ROPE_P_TID`. What remains to hold: the fixture still compiles to a
//  `Program::Rope` node, the layout still serves it, and the whole-function door lowers it — the
//  band lift resolving the per-core uniform corners, the pointwise emission carrying the
//  rotate-half pairs' unequal per-operand column windows.
// ══════════════════════════════════════════════════════════════════════════════════════════════
#[cfg(test)]
mod rope_from_the_fixture {
    use super::*;

    /// `test/fixtures/rope.py` at `rope_kv8`'s configuration (H=8, N_TOK=256, HEAD_DIM=128,
    /// BLOCK_M=64, HALF=64, grid [4, 8]) — the same values `examples/bake_py`'s `rope_kv8` case and
    /// `pure_rust_ktir.rs`'s `rope_ce` state, so the three cannot drift about what the configuration
    /// is.
    fn rope_kv8() -> ktir_superdsc::ktir_node::KtirNode {
        use triton_frontend::codegen::{ArgSpec, KernelSpec};
        use triton_frontend::semantic::Val;
        use triton_frontend::target::Target;
        let mut signature = std::collections::HashMap::new();
        for p in ["desc_x", "desc_cos", "desc_sin", "desc_o"] {
            signature.insert(p.to_string(), ArgSpec::parse("*fp16").expect("signature"));
        }
        let mut constexprs = std::collections::HashMap::new();
        for (k, v) in [
            ("H", Val::Int(8)),
            ("N_TOK", Val::Int(256)),
            ("HEAD_DIM", Val::Int(128)),
            ("BLOCK_M", Val::Int(64)),
            ("HALF", Val::Int(64)),
        ] {
            signature.insert(
                k.to_string(),
                ArgSpec::parse("constexpr").expect("constexpr"),
            );
            constexprs.insert(k.to_string(), v);
        }
        let file = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("crate sits one level under crates/triton")
            .join("test-fixtures/rope.py");
        let kspec = KernelSpec {
            kernel: "rope_fwd".to_string(),
            signature,
            constexprs,
            file: file.to_string_lossy().to_string(),
        };
        let src = std::fs::read_to_string(&file).expect("read rope.py");
        let mut tt =
            triton_frontend::codegen::compile(&src, &kspec, Target::spyre()).expect("codegen");
        triton_frontend::opt::make_ttir(&mut tt).expect("make_ttir");
        let mut m = triton_ktir::from_ttir::convert(&tt).expect("from_ttir");
        let grid = vec![4, 8];
        triton_ktir::make_ktir(&mut m, &grid).expect("make_ktir");
        triton_ktir::passes::to_ktir::run(&mut m, &grid).expect("to_ktir");
        node_for(&m, Program::Rope).expect("node_for")
    }

    /// ⭐ THE POST-DELETION CONTRACT, IN ONE TEST.
    ///
    /// (1) The layout places NO rotate matrix. `P` existed for `rope_at`'s `matmul(x, P)` and for
    /// nothing else — no op of the compiled body names `ROPE_P_TID` (measured: the whole-door rope
    /// bakes' descriptors reference no reserved tid) — so a placement that survives the body's
    /// deletion is a dead activation entry every rope bundle would allocate, fill and never read.
    ///
    /// (2) The whole-function door lowers the compiled body — the fixture's own ops, not a
    /// hand-rolled geometry — and emits more than nothing. The door's internals (the band lift, the
    /// per-operand column windows) carry their own tests below; this one pins that rope reaches
    /// them at all.
    #[test]
    fn rope_lowers_whole_and_places_no_rotate_matrix() {
        let node = rope_kv8();
        let l = layout::for_node(&node).expect("the layout must serve a rope");
        assert!(
            !l.placements
                .contains_key(&ktir_superdsc::reserved_tids::ROPE_P_TID),
            "the rotate matrix is placed, but the geometry body that read it is deleted"
        );
        let ops = emit_whole(&node, Some(&l)).expect("rope lowers through the whole door");
        assert!(
            !ops.is_empty(),
            "the whole door emitted nothing for a rope the fixture compiles"
        );
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
//  THE 7-SEGMENT WALL, GONE — a 14-parameter function is not a 14-segment demand.
//
//  This is the test for [`layout::for_regions`]'s packing, and it does not depend on any door past
//  the layout: `decoder_layer_one_flat` used to be refused at `layout` for wanting 14 of 7 HBM
//  segments, which put the whole decoder block behind an arithmetic of ours rather than a limit of
//  the device's (`wire::SEGMENT_OFFSETS` states its 7 PER OP — per emitted descriptor's operand
//  list, three or four tensors here).
// ══════════════════════════════════════════════════════════════════════════════════════════════
#[cfg(test)]
mod decoder_from_the_fixture {
    use super::*;

    /// `test/fixtures/decoder_block.py`'s `decoder_layer_fwd` at `decoder_layer_one_flat`'s
    /// configuration — `examples/bake_py`'s own `dec_ptrs_one()` order and `dec_ce(256)` values.
    fn decoder_layer_one_flat() -> ktir_superdsc::ktir_node::KtirNode {
        use triton_frontend::codegen::{ArgSpec, KernelSpec};
        use triton_frontend::semantic::Val;
        use triton_frontend::target::Target;
        let mut signature = std::collections::HashMap::new();
        for p in [
            "desc_x",
            "desc_o",
            "desc_n1",
            "desc_wq",
            "desc_wk",
            "desc_wv",
            "desc_wo",
            "desc_mask",
            "desc_cos",
            "desc_sin",
            "desc_n2",
            "desc_wg",
            "desc_wu",
            "desc_wd",
        ] {
            signature.insert(p.to_string(), ArgSpec::parse("*fp16").expect("signature"));
        }
        let mut constexprs = std::collections::HashMap::new();
        for (k, v) in [
            ("M", Val::Int(64)),
            ("D_MODEL", Val::Int(128)),
            ("D_FF", Val::Int(256)),
            ("BLOCK_N", Val::Int(256)),
            ("HALF", Val::Int(64)),
            ("EPS", Val::Float(1e-05)),
            ("INV_D", Val::Float(1.0 / 128.0)),
            ("QK_SCALE", Val::Float(0.011271055)), // 0.0078125 * 1.44269504, folded (Python's literal, bit-identical; do NOT substitute std's LOG2_E — its f64 spelling differs)
            ("RM", Val::Float(0.22)),
        ] {
            signature.insert(
                k.to_string(),
                ArgSpec::parse("constexpr").expect("constexpr"),
            );
            constexprs.insert(k.to_string(), v);
        }
        let file = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("crate sits one level under crates/triton")
            .join("test-fixtures/decoder_block.py");
        let kspec = KernelSpec {
            kernel: "decoder_layer_fwd".to_string(),
            signature,
            constexprs,
            file: file.to_string_lossy().to_string(),
        };
        let src = std::fs::read_to_string(&file).expect("read decoder_block.py");
        let mut tt =
            triton_frontend::codegen::compile(&src, &kspec, Target::spyre()).expect("codegen");
        triton_frontend::opt::make_ttir(&mut tt).expect("make_ttir");
        let mut m = triton_ktir::from_ttir::convert(&tt).expect("from_ttir");
        let grid = vec![1];
        triton_ktir::make_ktir(&mut m, &grid).expect("make_ktir");
        triton_ktir::passes::to_ktir::run(&mut m, &grid).expect("to_ktir");
        // The kind is stated because the signature demands one; the whole-function door never reads
        // it, and neither does the layout.
        node_for(&m, Program::Matmul).expect("node_for")
    }

    #[test]
    fn fourteen_parameters_pack_into_two_segments_without_overlapping() {
        let node = decoder_layer_one_flat();
        let l = layout::for_node(&node).expect("14 parameters must not be a refusal");
        // ONE PLACEMENT PER PARAMETER **PLUS ONE PER REGISTRY SCALE**, and the second half is new:
        // this used to be a bare `== 14` because a decoder's `scalarmul_scales` was EMPTY — `scales_for`
        // is keyed on the stated `Program` and returns nothing for `Matmul`, which is what a
        // whole-kernel bake states. The whole-function door now populates the registry from the
        // program's own shape (its epsilons and its genuine scalar multipliers, NOT the rmsnorm
        // divisors), and each of those needs a placement or `resolve_seg_base` panics on it. So the
        // invariant is stated as the two parts it has, and the PARAMETER half is checked by tid rather
        // than by count — a count alone would pass if a parameter were dropped and a scale added.
        for (i, tid) in node.bindings.iter().enumerate() {
            assert!(
                l.placements.contains_key(&tid.get()),
                "parameter {i} (t{}) has no placement",
                tid.get()
            );
        }
        assert_eq!(
            l.placements.len(),
            14 + l.scalarmul_scales.len(),
            "one placement per parameter plus one per registry scale (14 parameters, {} scale(s))",
            l.scalarmul_scales.len()
        );
        // TWO segments, whatever the parameter count: reads in `Activation`, stores in `Logits`.
        let mut segs: Vec<usize> = l.placements.values().map(|p| p.segment).collect();
        segs.sort();
        segs.dedup();
        let mut want = vec![
            ktir_superdsc::placement::SegRole::Activation.segment(),
            ktir_superdsc::placement::SegRole::Logits.segment(),
        ];
        want.sort();
        assert_eq!(
            segs, want,
            "reads pack into one segment, stores into another"
        );
        // AND THE PROOF THAT MAKES PACKING SAFE, re-stated as a test rather than trusted of the
        // arithmetic: no two placements share a 128-B granule of one segment. `for_regions` runs
        // `overlaps_none` itself, so this asserts the same fact the other way round — from the
        // placements it produced.
        let mut spans: Vec<(usize, u64, u64, u32)> = l
            .placements
            .values()
            .map(|p| {
                (
                    p.segment,
                    p.offset,
                    p.offset + ktir_superdsc::placement::align128(p.size).max(128),
                    p.tid,
                )
            })
            .collect();
        spans.sort();
        for w in spans.windows(2) {
            let (s0, _, e0, t0) = w[0];
            let (s1, b1, _, t1) = w[1];
            assert!(
                s0 != s1 || b1 >= e0,
                "t{t1} starts at {b1}B of seg{s1} while t{t0} still occupies it through {e0}B"
            );
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
//  THE SEGMENT-TAIL FAULT — a gather's index buffer may not end flush at its segment's
//  high-water, because the leg's HBM FETCH over-reads the ids by up to one 128-B granule and a
//  segment boundary is not a place that over-read can land.
//
//  MEASURED ON CARD (`paged_score_small`, rung 4): the ids were the segment's last placement,
//  ended exactly at `segment_bytes[3]`, and the fetch of the final 128-B stick faulted
//  (`HMI=FETCH addr=0xc00004480`, `run_prefix_only sync rc=-1`). ONE GRANULE of slack past the
//  high-water (17664→17792) made the identical bundle run clean with byte-identical output, and
//  the fault is scoped, not general: the embedding configs run card-verified with their ids at
//  offset 0 (never flush — a neighbour above absorbs the over-read), and attention's flush-at-end
//  OUTPUT writes clean — it is the gather's fetch path alone that pays this.
// ══════════════════════════════════════════════════════════════════════════════════════════════
#[cfg(test)]
mod paged_score_from_the_fixture {
    use super::*;

    /// `test/fixtures/paged_score.py` at `paged_score_small`'s configuration — the same values
    /// `examples/bake_py`'s own case table states, so the two cannot drift about what the
    /// configuration is.
    fn paged_score_small() -> ktir_superdsc::ktir_node::KtirNode {
        use triton_frontend::codegen::{ArgSpec, KernelSpec};
        use triton_frontend::semantic::Val;
        use triton_frontend::target::Target;
        let mut signature = std::collections::HashMap::new();
        for (p, ty) in [
            ("desc_p", "*fp16"),
            ("desc_v", "*fp16"),
            ("desc_ids", "*i32"),
            ("desc_o", "*fp16"),
        ] {
            signature.insert(p.to_string(), ArgSpec::parse(ty).expect("signature"));
        }
        let mut constexprs = std::collections::HashMap::new();
        for (k, v) in [
            ("M", Val::Int(8)),
            ("K", Val::Int(64)),
            ("V", Val::Int(128)),
            ("BLOCK_M", Val::Int(8)),
            ("HEAD_DIM", Val::Int(64)),
        ] {
            signature.insert(
                k.to_string(),
                ArgSpec::parse("constexpr").expect("constexpr"),
            );
            constexprs.insert(k.to_string(), v);
        }
        let file = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("crate sits one level under crates/triton")
            .join("test-fixtures/paged_score.py");
        let kspec = KernelSpec {
            kernel: "paged_vmatmul_fwd".to_string(),
            signature,
            constexprs,
            file: file.to_string_lossy().to_string(),
        };
        let src = std::fs::read_to_string(&file).expect("read paged_score.py");
        let mut tt =
            triton_frontend::codegen::compile(&src, &kspec, Target::spyre()).expect("codegen");
        triton_frontend::opt::make_ttir(&mut tt).expect("make_ttir");
        let mut m = triton_ktir::from_ttir::convert(&tt).expect("from_ttir");
        let grid = vec![1];
        triton_ktir::make_ktir(&mut m, &grid).expect("make_ktir");
        triton_ktir::passes::to_ktir::run(&mut m, &grid).expect("to_ktir");
        node_for(&m, Program::Matmul).expect("node_for")
    }

    /// ⛔ THE FAULT'S OWN PREMISE, ASSERTED FIRST: the fixture's ids really are the segment's
    /// last placement, ending flush at the pre-fix high-water. Without this the slack test below
    /// could pass against any layout at all — which is how the card run came to fault while every
    /// value-level test stayed green.
    #[test]
    fn the_ids_end_flush_at_their_segment_high_water() {
        let node = paged_score_small();
        let l = layout::for_node(&node).expect("the layout must serve a gather");
        let ids = l.placements.get(&2).expect("the ids parameter is t2");
        // 64 i32 entries = 256 B, and `pack`'s alignment keeps 256 a whole number of granules.
        assert_eq!(ids.size, 256, "64 i32 ids, or this test is not the defect");
        // THE PREMISE: the ids' granule span IS the segment's high-water — p [8,64] at 0, table
        // [128,64] at 1024, ids at 17408, so the pre-fix `segment_bytes[3]` was 17664 exactly.
        assert_eq!(
            ids.offset + ktir_superdsc::placement::align128(ids.size).max(128),
            17664,
            "the ids' span must end at the pre-fix high-water 17664 B, or the slack test is \
             not measuring this defect"
        );
        assert_eq!(
            ids.segment,
            ktir_superdsc::placement::SegRole::Activation.segment()
        );
    }

    /// ⭐ THE FIX: the same layout now advances the high-water ONE GRANULE past the ids' span —
    /// the measured slack that made the identical bundle run clean on card with byte-identical
    /// output. The slack is added ONLY for a flush-at-high-water ids (an ids packed anywhere
    /// else has a neighbour above it to absorb the over-read), so the twelve card-verified
    /// configs' `segment_bytes` do not move.
    #[test]
    fn the_layout_leaves_one_granule_of_slack_past_a_flush_ids() {
        let node = paged_score_small();
        let l = layout::for_node(&node).expect("the layout must serve a gather");
        let ids = l.placements.get(&2).expect("the ids parameter is t2");
        assert_eq!(
            l.segment_bytes[ids.segment],
            17664 + 128,
            "seg{} must carry one 128-B granule of slack past the ids' span ({} B), because the \
             gather leg's HBM fetch over-reads the ids by up to one granule and faults on a \
             flush segment boundary — measured on card, rung 4",
            ids.segment,
            ids.offset + ids.size
        );
    }
}
