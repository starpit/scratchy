# 11l — the body split: mechanism notes (auditor working file, pre-ruling)

Wall: the two top rungs refuse ON TRIPS ONLY — 72,397 (384 rows, T=96) and 96,429
(512 rows, T=128) over MAX_BUNDLE_TRIPS=65,536. Descriptors are FINE post-11k
(7,693 / 10,157 = 77·T+301). The ladder must top out at 512 (wall-16) — 11l is REQUIRED.

## The decisive facts (read 10-25, post-attempt-9)

1. **`launches` is a GENERAL true-order sequence — interleaving IS expressible.**
   `RolledSuperDsc.launches: Vec<(u8, u32)>` (lower_subtile_tape_to_superdsc.rs:7179):
   "THE LAYER SEQUENCE, IN TRUE ORDER — one entry per layer of the whole model, naming
   its body (index into `bodies`) and its class-relative iteration. The executor runs
   exactly this; a single-class roll degenerates to [(0,0), (0,1), …]". So a roll whose
   launches are [(0,i),(1,i)] per layer i — body A then body B per layer, interleaved —
   is within the type's own stated contract, not an extension.
2. **The wire and loader already carry N body classes.** superdsc_exec.rs:1232
   ("RE-ROLLED — {} body class(es) + prefix({}) + suffix({})") and :1312 ("bundle {}: {}
   body ({} class(es)) + {} prefix + {} suffix"); the bake success line (codegen.rs:12230)
   prints "body {bfp} + {} class(es)". `bodies: Vec<RolledBody>` (:7168), one entry per
   class, each with `.iters` and `.ops`.
3. **Per-launch addressing is fully general.** `launch_w: Vec<(u8, u64)>` (per-launch
   weight bank+offset, "launches' addressing half, proven against the placements at
   cargo-build", :7180-7182) and `launch_kv: Vec<u64>` (:7183) — one entry per LAUNCH,
   so an A/B interleave carries each half's own weight/KV base per layer.
4. **The residual threads in place across EVERY body already** (:7160-7163): "the layout
   aliases each body's hidden_out_tid and the next body's hidden_in_tid onto the prefix's
   hidden buffer". With interleaved launches, "next body" is the next launch — the A→B
   seam and the B→next-layer-A seam are the SAME mechanism, one more instance.
5. **`per_layer: BTreeMap<u32, Vec<u32>>`** (:7171-7174): tables keyed by each class's
   representative tid, "a table's length is ITS class's iters" — split bodies carry
   split tables (each class's referenced tids only), both length 26 here.
6. **The class re-roller exists and is tested**: `reroll_layer_classes` at
   subtile_tape.rs:2260 (fallback `reroll_subtile_tape`); tests `splits_the_loops`
   (:3612 — S(4)+G(2) → TWO sibling Const-bound loops, per-layer tables carry the
   class's layers only) and `refuses_a_single_class` (:3756). The per-layer class mask
   is read off the unrolled graph (codegen.rs:8567). NOTE: the tape-level loops are
   SIBLING (class-contiguous) — the true-order generality lives in `launches`, not in
   the loop instrs.

## The ruled shape (candidate, to be verified before the exact-code ruling)

A POST-ROLL transform on `RolledSuperDsc` at the bake site (codegen.rs, the 3-bundle
write region :12305+): when a body class's TRIPS exceed MAX_BUNDLE_TRIPS, split that
class's ops at THE SEAM into two classes and interleave the launches:
- bodies = [A, B] (A = the attention half, B = the MLP half — the seam at the residual
  boundary between blocks, so every body's hidden_in/hidden_out threading stays the
  in-place aliasing law already stated);
- launches = [(0,i),(1,i)] for each layer i (true order preserved — program order =
  call order);
- per_layer split by referencing body; launch_w/launch_kv split per half (each half's
  weight operands re-based to its own per-layer segment base — MUST verify how the
  single body's weight operand offsets relate to the launch base before ruling this);
- the bake writes the classes as separate bundles (the wire/loader shape already
  carries N classes; the WRITE path at :12305+ needs reading — "3-bundle write" may
  hard-code 3).
- Effect at granite-2b: 384 → A/B ≈ 36.2k trips each ✓; 512 → ≈ 48.2k each ✓ — both
  under 65,536; descriptors unchanged.

## Open items before the ruling
- READ the 3-bundle write (codegen.rs:12305-12318) + the bake-success region (:12200+):
  does the write loop over bodies (N-general) or hard-code prefix/body/suffix?
- READ how a body op's weight placement relates to launch_w's base (the re-base law for
  a split class) — the layout places every tensor; launch_w is "proven against the
  placements"; find the proof site (bundle::LayerLaunch) and what a split must re-prove.
- The SEAM LAW: where exactly (op index / node boundary) a granite-2b layer splits into
  attention-half / MLP-half — from the body's op order (the 7 stagings t729..t742:
  q/k/v/o are the attention half's GEMMs, gate/up/down the MLP half's). The seam must be
  the residual boundary the layout already threads.
- Whether the fold-fused TWIN path also needs the split (the twin refuses identically —
  yes, same transform, same site).
- Tests: the split's own test (a body over the wall splits; launches interleave true
  order; per-launch addressing proven) + the existing roll tests stay green.

Nothing here is ruled yet — this file is the read. The ruling lands as
11l-body-split.md only after the open items are read and the exact code is written.
