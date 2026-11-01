# PR #999 depth regression — port torch-spyre's lowering of the paged walk

## Ground truth (card agent, PR #999)
- A/B (6542440196): 2b fp8 ITL 33.5/68.6/396.8 vs main 26.8/27.8/42.3; TTFT@10k 94.9 s vs 8.5 s.
  8b 140.6/193.4/678.2 vs main 132.0/136.1/155.4 vs torch-spyre 143.5/148.9/164.0.
- Slope probe (6542567276): launches flat (~42/step), walk body = 90% device exec; prefill trip
  ~0.93 ms (HBM-traffic bound); decode ~34-39 µs/page-layer = 43 ops (2b) × ~0.8 µs.
- torch-spyre program (6542702075, JSONs 6542730465/6542731446 → .scratch/torch_ref/):
  one bundle/layer, device `scf.for` (affine.apply s0+128*d0 on page table + mask), 22 ops/trip,
  trip = 128-token block, kv heads = ONE op (`x` batch split over cores), all trip intermediates +
  carried max/sum/acc in LX (ops 8, 12-25 bind no HBM symbol). KV block = 128 (platform.py:146).

## Steps (each its own commit, card-timed before the next)
Partial all-HBM steps are OUT: a 3-D per-batch kernel in HBM faulted on the card
(`job_bin_ptr + numCoresUsed_*128`, matmul/opspec.rs `matmul_opspec_batched_off` doc). torch-spyre
keeps the bmm kernel + output in LX (no per-core HBM patches) — so the port is their WHOLE trip:
1. [ ] Design: an LX-planning PASS over the walk body (torch-spyre's own: one kv head per core for
       every body op, core-local LX pins for every body intermediate + carried max/sum/acc, the K/V
       page gathered into LX, x-batched contractions with LX kernel/output, Q read from HBM).
       The lowering only reads the pass's annotations (no transforms at lowering time).
2. [ ] Decode trip at hd64 / 64-slot (2b) in that form — card: coherence + 2b cells.
3. [ ] hd128 forms from torch-spyre's 8b program (asked, comment 6542910184).
4. [ ] 128-slot blocks (their per-core gather, sdsc_6) — spyre-inference's real block size.
5. [ ] Prefill trip in the same form (TTFT).
6. [ ] device scf.for (their bundle.mlir) if per-trip records still cost.

## Rules
- Transforms are PASSES, not lowering code; kernel stays spyre-inference verbatim.
- Port THEIR descriptor fields; no invented forms. Card validates each step (coherence + cells).
- Run ALL CI test steps for touched crates (splice!) before pushing.

## Access
- ~/git/torch-spyre and ~/git/spyre-inference exist locally but are OFF LIMITS (home-dir rule) unless the user grants an exception. I ls-ed ~/git once by mistake (2026-10-31 ~11:45), reported it. Work from card-agent JSONs / quoted source.

## torch-spyre 8b decode program (bound-2 bucket) ↔ our KTIR body — op by op
(.scratch/torch_ref8b/; every op: mb = kv split 8 ⇒ core c = kv head c; LX = core-local)
| tsp | what | operands (tsp) | ours today (8b, 61 ops) |
|---|---|---|---|
| 0 | q = query.index_select(row) | HBM→HBM pool | query permute (prologue) |
| 1,2 | tile_max/-sum init [8,4,64] | HBM const (scale -1 bcast) → LX | fill (HBM) |
| 3 | tile_output init [8,4,128] | HBM const → LX | fill (HBM) |
| 4 | kv_row_pool copy [513,8,32] | HBM→HBM | — |
| 5 | page's kv rows (affine page table) | index gather → HBM pool | gather_run (addi32toi32) |
| 6,7 | K,V page gather [8,128,128] | index HBM, pages HBM → LX per core | gather + 8 permutes (each) |
| 8 | restickify K→Kᵀ | LX→LX | 8 transposes |
| 9 | QKᵀ x-batched [x8,mb4,out128,in128] | Q HBM, Kᵀ LX, out LX | 8 bmm (+2 q block copies, hoisted) |
| 10 | ×scale | LX, HBM const | multiply |
| 11 | +mask (affine mask tile, bcast kv,G) | LX, HBM | add |
| 12 | rowmax → [8,4] as stick | LX→LX | max |
| 13-17 | sub/relu/neg/exp/maximum [8,4,64] | LX (carried max @32768) | sub,maximum,neg,exp,maximum |
| 18,19 | scores-new_max, exp [8,4,128] | LX | sub, exp |
| 20-22 | sum*rescale + rowsum → carried sum | LX (@33280) | multiply,sum,add |
| 23 | acc*rescale | LX (carried acc @99328) | multiply |
| 24 | P·V x-batched | all LX | 8 bmm |
| 25 | acc += PV | LX | add |
| 26 | acc/sum | LX → HBM | (epilogue div) |
| 27 | out.index_copy | HBM | scatter |

Design choice for the user: (A) generic — teach the KTIR→SDSC lowering an LX-planning PASS
(uniform kv-per-core division + core-local pins for body values) + the 3 new forms (x-batched bmm
with LX kernel/out, gather→LX, restickify LX→LX); (B) a PagedAttn template emitter reproducing
this program from the node's geometry (like main's attn.rs). (A) fits the triton-path architecture.

## The port (user: generic pass; "do what torch-spyre does")
torch-spyre pipeline (passes.py): splice_while_loops → named dims → stickify/restickify →
work distribution → scratchpad_planning (LX) → codegen. Default LX planner = CP-SAT joint
core-division + placement (co_optimizing_lx_planning=1); deterministic sibling = ExhaustiveSearchSolver
(DFS over division candidates, inner first-fit/greedy placement, objective = Σ HBM bytes of unpinned
buffers). Residency rules (allocator.py _BufferResidencyChecks): spill if mutation target, advancing
(tiled read), indirect/index, partial read, core-div mismatch with ANY consumer, extern in range;
loop carry stays resident when division is free.

Rust port in ktir-superdsc (a PASS, lowering only reads its result):
A. `scratchpad::plan(node, fold, layout) -> LxPlan`: buffers = every value the function computes
   (body + carried + prologue), candidate divisions per op (its own `WorkPlan` candidates), edges
   with matching division pairs, residency verdicts (port the rules), exhaustive search, first-fit
   placement in the LX pin window (`lx_window_fits`). Output: per-op division, per-value
   `Residency::{Hbm, Lx{pin}}`.
B. emission reads LxPlan: each op's splitter = its planned division; each operand's
   `Allocation::LxCoreLocal` at its pin (per_core_addr already handles core-local).
C. forms (copy torch_ref JSONs): gather → LX (GatherDest::CoreLocalLx exists), restickify LX→LX,
   x-batched bmm (LX kernel/out, x split), elementwise/reduce on LX with per-row scalars as one stick.
Reference: decode 2b .scratch/torch_ref/, decode 8b .scratch/torch_ref8b/, prefill 8b
.scratch/torch_ref_prefill/ (prefill splits query rows 16-way; K via LX → restickify → HBM pool).

## Progress (2026-10-31)
- 9289e7da6 (branch pr999-block128): PAGE_SLOTS 64 → 128 (spyre-inference's block). Card gate asked
  (comment 6543914385): coherence 2b fp8 + 8b bf16, rung-1 cells vs b82250b17.
- LX planner (uncommitted → pr999-lxplan): emit/lx_plan.rs + probe (BundleLayout::lx_probe/lx_planned,
  note_lx_touches, core_slices, pointwise_core_local), synth_df honours the plan + records full footprint
  (LxAlloc::full_sizes; held_of reads it), mint replay accepts core-local, lower_walk = probe → plan →
  real → minted-equal + check. Verdicts ported: Indirect, TimeTiled, Partial (offset), slice mismatch.
  Decode hd64 keeps 11 tensors in LX (acc, sum, rescale chain, consts); per-kv bmm tensors HBM.
  All CI tests + clippy pass. 2b card bake (dxp) running.
- NEXT: x-batched contractions (sdsc_9/24 forms) + all-heads restickify (sdsc_8) + gather→LX (sdsc_6),
  with kv-per-core division so the scores/probs/K/V join the LX set (the co-optimizing half: division
  candidates per op → exhaustive search, torch-spyre exhaustive_search.py).

## Next step design (kv-per-core trip, hd64 first)
- e388178d3 = LX planner (local; push as pr999-lxplan after 8b bake).
- Forms, all dividing the walk body ONE kv head per core (derive the group count from the
  BatchMatmul's batch — the contraction's own x):
  1. BatchMatmul → ONE `batchmatmul` N_ {x: batch, mb: m, out: n, in: k}, x split over cores,
     INPUT ['mb','x','in'] (flat q, one stick at hd64), KERNEL per-core block, OUTPUT flat.
     ⛔ its kernel MUST be core-local LX (torch-spyre's form; a per-core-distinct HBM 3-D kernel
     faulted on card) — refuse at build otherwise.
  2. BatchTranspose → ONE restickify, x split (sdsc_8 form), LX→LX.
  3. Body elementwise/reduce → EwDivision::Groups(batch) (mb split into `batch` cores).
  4. Gather: already one kv entry per core (GatherSplit) → CoreLocalLx dest when planned.
  Then lx_plan keeps scores/probs/K/V/Kᵀ in LX automatically (slices align).
- hd128: q flat is stick-major across x — the emitter's rank-3 corner model is flat row-major;
  use Blocks (420a5cc56 copies) or teach the corner model. After hd64 lands.

## 2026-10-31 14:35 — block128 gate 1 FAILED (blind) → root cause + fix
- card td (9289e7da6, 2b fp8): every op consistent with its inputs (td_walk_check.py); P·V rows of odd
  prompt positions zero → the host MASK: card stagers passed the row-major stack, kernels read it
  stick-major (one stick a row at 64 slots hid it). Fix d8d185fae (`card_mask`, stick_scatter) on
  origin pr999-block128 (ff of 9289e7da6). Gate 2 asked (comment 6544746311).
- local pr999: b82250b17 → 9289e7da6 → e388178d3 (LX plan) → 0d9cd13da (mask fix, cherry-picked).
  After gate 2: push block128+fix to PR, then gate the LX plan as its own candidate on top.
- tools: .scratch/card/td_walk_check.py (+fake), td_zero_rows.py, find_ops_by_shape.py; group_ops.py
  now prints starts in CORE order (it used to sort by value — misleading).

## 2026-10-31 ~15:00 state
- gate 2 (d8d185fae): both models COHERENT (8b = main's opening; 2b = mlx-lm's "France, a country
  renowned…"). Rung-1 A/B vs b82250b17 running on the card.
- local pr999: b82250b17 → 9289e7da6 → e388178d3 (LX plan) → 0d9cd13da (mask fix) → 7641c422b
  (division probed with residency: EwDivision/ReduceDivision::Groups, BundleLayout::lx_row_groups /
  ew_law / reduce_rows; plan probes {own, Groups(batch)} and keeps max LX bytes). Behavior-neutral today.
- NEXT (kv-per-core trip): (1) ViewFrame::FoldedRows — rank-3 [x, mb, stick] view of a FLAT stick-major
  [x·mb, stick] tensor: view_stick_layout → row_blocked_df(x·mb, stick); per_core_addr folds the corner
  (non-stick comps outermost-first) before off_view; declares the same arrangement as the elementwise
  ops. (2) kernel blocks [x][k][n] (each stick-major): rank-3 Own Flat corner c·k·n is right per x; the
  producer (batched restickify) must declare the same view. (3) x-batched bmm opspec (dims x,mb,out,in;
  split {x: batch}; dxp orders INPUT ['mb','x','in'], KERNEL ['in','out','x'], OUTPUT ['mb','x','out']),
  kernel MUST be core-local. (4) batched restickify one op split x. (5) BatchMatmul/BatchTranspose take
  these forms under the Groups(batch) plan.

## SHORT CONTEXT regresses too (user, 2026-10-31)
- ~123 tok: 2b fp8 ITL 33.5 vs main 26.8 (TTFT 104 vs 76); 8b 140.6 vs 132.0 (TTFT 210 vs 162).
  ≈170 µs/layer fixed cost at 2b — NOT the page slope. torch-spyre 8b is 143.5 there (slower than
  main), so the port alone won't close it. Asked the card agent (comment 6545270600) for per-group
  decode timing (SCRATCHY_SDSC_GROUP_TIME) at ~123 tok, d8d185fae vs main, queued after the A/B.
- Expected from the port: 128-slot = 1 trip instead of 2; LX kv-per-core trip ~34 µs → few µs/layer.
  Remainder = prologue/epilogue relays, fills, index ops, scatter, the separate KV-write program.
- kv-per-core trip (uncommitted on pr999): x-batched bmm (matmul_opspec_x_batched), batched
  restickify, FoldedRows view frame, LxShare::Required + planner candidate skip/mint-record restore.
  Splice tests: hd64 decode takes kv-per-core (467 KB in LX vs 53 KB). 2b card bake (dxp) running.

## 2026-10-31 ~15:25 branches
- origin pr999-block128 = d8d185fae (block128 + mask fix): coherent; rung-1 A/B vs b82250b17 RUNNING.
- origin pr999-lxplan = f2f470f5c = d8d185fae + 705347d5f (LX plan) + 35cc3910c (division probe) +
  f2f470f5c (kv-per-core trip). Gate queued (comment 6545323479). Local working branch: pr999-lx
  (same tree as old local pr999 905a30239).
- 2b decode trip: 23 ops (was 46), gathers→LX, 1 restickify, 2 all-heads bmm, LX softmax.
- NEXT: short-context fixed costs (await GROUP_TIME); hd128 (page permutes block the V kernel's LX);
  prefill (torch-spyre splits rows 16, K via LX→restickify→HBM pool).

## dxp cache (CORRECTION, user 10-31): there IS a persistent dxp cache
- `/Volumes/Builds/cargo/Users/moosevan/git/scratchy/.claude/worktrees/spyre-attn-triton.build/debug/build/superdsc-dxp-cache`
  (repo .cargo/config.toml `build-dir` puts OUT_DIR on the Builds volume; 1.6 GB, 1801 groups).
  Unchanged groups are served from it → the capture wrapper sees only changed groups. Full capture =
  SCRATCHY_SUPERDSC_CACHE=<empty dir> + touch crates/models/arch/src/lib.rs (cargo doesn't track the var).
  Running for 2b into /Volumes/Builds/pr999-fresh-dxp-cache (DELETE after).

## 2026-10-31 ~16:10
- d8d185fae (block128 + mask fix) PASSED rung-1 A/B vs b82250b17 on every cell (comment 6545689478):
  2b ITL −13/−27/−37%, 8b −4/−13/−30%. PUSHED to feat/spyre-paged-attention (ff). Body patched
  (new table under "Not done", Pools bullet 64→128). Auditer told.
- d37654e1c (local pr999-lx, on f2f470f5c): hd128 page gather in torch-spyre's sdsc_6 form
  (page_gather_blocks_opspec: N_ {mb: entries, x: rows, out: cols, y: 1}, mb split, value
  ['out','x','mb','y'], landing ['x','out','mb','y']). 8b decode walk 74 → 33 ops (trip 64 → 23),
  takes kv-per-core (K/V/Kᵀ/scores in LX, x-batched bmms). All CI test/clippy scopes pass.
  8b light bake (b03sdqqus, -j 2, default dxp cache) running → then push as pr999-lxplan-hd128, gate
  8b vs f2f470f5c after the f2f470f5c gate.
- Temp op-dump test snippet: .scratch/zz_dump_walk_ops.rs.txt (re-insert at end of splice tests mod).

## PREFILL (TTFT) — findings 2026-10-31 ~12:20 (2b capture, splice dumps)
- Prompt rungs up to P=512 rows (bmm mb = G·P = 2048 at 2b). Walk body at hd64: 110 ops/trip:
  66 mask adds (one flat [P,64] add per head block × page stick — the 6777c3c4e workaround), 16 per-kv bmm,
  8 per-kv transposes, gathers in pieces, softmax chain — every [8·G·P, 128] intermediate in HBM
  (4 MB at P=512 → trip HBM-bound, card agent's ~0.93 ms/trip).
- Prologue/epilogue: query row gather = 1 op per stick column × per 32 entries (P=512 hd64: 512 ops);
  output scatter the same (512). P=128 (short prompt): 128 + 128 ops/layer ≈ 0.3-0.5 ms/layer →
  ~15 ms of the ~21 ms short-context TTFT gap (estimate, not measured).
- torch-spyre prefill (8b, .scratch/torch_ref_prefill): query gather ONE op (N_ {mb 512, x 32 heads,
  out 128}, mb split 16, per-core index starts = 32 entries a core, value row-major table, dest LX
  head-major stick-major); body 22 ops: K/V gathers 1 core, Kᵀ restickify, QKᵀ bmm split y(=P) 16 with
  x=KV whole, kernel Kᵀ HBM SHARED by all cores, scores LX; softmax ops 5-dim (P, KV, G, slots) split P 16,
  all LX; mask add one op (mask broadcast over KV, G); V relayout; PV bmm kernel V HBM shared; output
  scatter ONE op split 16.
- Port order (each card-gated): (P1) query gather/scatter: per-core index runs (≤32 entries a core) so
  one op per stick column serves all P rows (our query table is stick-major; theirs row-major);
  (P2) body in the P-split form: ops over (KV, G, P, slots) split P 16, intermediates core-local LX,
  kernels shared from HBM (no per-core-distinct HBM kernel = no known fault), mask add one op.

## P1 done locally (12:45): per-core index sticks (emit::index_stick_starts, work::index_core_split,
   lk::index_runs) — prompt 512-row walk 1140 → 180 ops. Commit msg .scratch/commit_msg_index_sticks.txt.
   Needs: CI scopes (running), commit, light bake after the peer's window (~13:10), card gate (TTFT cells).

## P2 design (torch-spyre's prefill program, .scratch/torch_ref_prefill):
- Their rows are KV-INNERMOST: scores ['x','out','y','mb'] = (g, p, slot-stick, kv, lanes); the P split
  16 is then a CONTIGUOUS row chunk per core. Kernels (Kᵀ, V) in HBM, x unsplit ⇒ one shared start for
  every core (the per-core-distinct HBM 3-D kernel is what faulted for us; theirs is never distinct).
- Our port: a row FRAME for the prompt walk's intermediates with kv innermost — rows (g·P + p)·KV + kv:
  * elementwise/reduce ops stay rank-2 [rows, ·] under EwDivision::Groups(n) (contiguous) ✓ existing;
  * QKᵀ / P·V: bmm [x=KV, mb=G·P, ·] with x INNERMOST in activation/output (device order [x, mb, ·]),
    split mb n, kernel [x, in, out] HBM shared;
  * mask: staged over G·P (their mask has G active, KV broadcast) and sprayed over kv innermost
    (inner row spray [mb, y, out]) — or expanded;
  * prologue/epilogue relayout q ↔ (g, p, kv) rows: StickRuns copies (dst_pitch KV), once per chunk.
- Open risks: the inner spray form on the card; dxp with x innermost bmm operands; mask staging size.

## STEP 6 (device scf.for) IS NOT OPTIONAL (user, 2026-10-31): it matters on gen 2 Spyre (coming soon).
   Keep it on the ladder after P2; loop bound = the rung's baked trip count (compile-time constant).

## 2026-10-31 ~13:15 queue: f2f470f5c (gate running) → d37654e1c (8b trip) → 9e4430bfc (P1, pr999-idx) →
   7659b54cd (columns in LX, pr999-cols; bake running, ask .scratch/pr999_ask_gate_cols.md incl. request
   for the prefill reference JSONs sdsc_0..26).
## P2 = torch-spyre's prefill program (comment 6543834406 listing): every trip tensor 4-D [KV, G, P, ·],
   P split 16 (P on mb or y), K/V gathered on ONE core (K → LX → restickify → HBM pool; V → HBM),
   QKᵀ {y P/16, x KV, mb G}, kernel Kᵀ HBM shared (no per-core start), scores LX; mask add 5-D
   {mb P, x KV(bcast), y G, i slots} with per-core mask starts (P chunk); columns [KV,G,P] with P on the
   stick ('out' 512, y 64); V relayout 22_identity HBM→HBM split x4·y4; PV {y KV, x G, mb P/16}, kernel
   V HBM shared; realdiv + scatter in the epilogue (scatter: per-core index sticks = P1).
   Needs: a 4-D/5-D elementwise+reduce op family with explicit device orders (R4/R5 TensorArgs),
   the bmm forms above, and the planner's P-split division. Port op for op from their JSONs.

## 2026-10-31 ~13:45 f2f470f5c GATE PASSED (comment 6546787591) → PUSHED to PR (ff from d8d185fae), body patched.
   2b ITL 24.1/30.9/92.6 (−16/−38/−63%; 0.90×/1.11×/2.19× main), 8b flat 133.4/165.8/456.1, TTFT −15..−21% at
   1k/10k; 2b TTFT@123 +4.1% = variance (prompt rung fingerprints identical at both heads).
   Trip (ITL slope / 3120 trips): 2b 70 → 22 µs; torch-spyre ~4.5 µs. Remaining suspects: per-trip relaunch
   (no device loop — step 6), mask add split (2 HBM tensors), 3 extra ops.
   Local stack above: d37654e1c (8b trip) → 9e4430bfc (P1) → 7659b54cd (columns) → 440ba2693 (sliding
   window refusal; auditer item 2) → d4d2c385f (gather bound pin; auditer 4b).
   User ruling 10-31: do NOT split #999 (auditer item 3 dropped).

## 2026-10-31 ~18:10 PR head = 9e4430bfc (f2f470f5c, d37654e1c, 9e4430bfc all gate-passed + pushed).
   2b fp8: ITL 25.0/32.1/93.6, TTFT 87.3/1252.5/52340. 8b: ITL 127.5/137.4/216.0, TTFT 185.0/1725.3/61728.
   (main earlier-session: 2b ITL 26.8/27.8/42.3 TTFT 76.4/623/8533; 8b ITL 132/136/155 TTFT 162/998/11455.)
   Queued on card: 7659b54cd (columns LX), the 10-rep TTFT@123 repeat (f2f vs d8d vs main).
   Local above 7659b54cd: 440ba2693 (sliding window refusal), d4d2c385f (gather bound pin), 9043de302
   (fingerprint by descriptors). Uncommitted: matmul_opspec_rows_split (+tests) for P2.
   NEXT: P2 design A (trip-major mask staging → decode single-op mask add) then the (p,h) prompt frame.

## 2026-11-01 ~06:45 — the B=1 DEPTH fix (the big one), three branches out
Origin branches (PR branch still pre-rebase until the user force-pushes):
- `pr999-fixq` = rebased on upstream/main + `6fcd27dbb` (chunk a decode batch at
  spyre-inference's own block slots: a chunk under the 8-page gather floor folds by kv head;
  8+ pages keep the whole-page gather) + `74987efb6` (docs: the fold is OURS, their batched
  kernel never folds and measured folding 1.5-2x slower).
- `pr999-relayout` = `5cc98e429` on top: v5's HeadMajorCopies split, <=32 runs an op, so no core
  copies two runs. CONTINGENT — build only if two concurrent mt256 is garbage while mt4/mt64 are
  coherent. Auditer's condition if it lands: add the refusal in `stick_runs_copy_opspec`
  (runs > MAX_CORES) with a 32-builds/33-refused test.
- `pr999-b1batched` = `966bcf71f` (store-through-reshape output reframe) + `2c6c3f1b2`
  (ONE REQUEST DECODES WITH THE BATCHED KERNEL) + `ca6cb33a0` (hd128 executed, pool-bound
  refusal pinned). AUDIT CLEAR.

⭐ THE FINDING: spyre-inference decodes a SINGLE request with `batched_decode_head_major`, not
`page_attn_head_major_decode`. Its seq buckets are powers of two FROM 1 (bucketer :210-218),
`_tiled_batched_decode_supported` is True for head-major (:271), SPYRE_BATCHED_DECODE defaults on
(envs :116), the dispatch gates on `num_decode_seqs > 0` (spyre_attn :892), uniformity floor is 0.0
(:106). The per-sequence program is the WARMUP variant (`build_for_variant(.., per_seq_only=True)`,
:997-1022). So their B=1 decode walks `min(32, blocks)` pages a trip, one page a core; ours walked
ONE block a trip. 4k ctx = 32 trips vs 1; 16k = 128 vs 4; 64k = 512 vs 16. THIS is the depth ITL row.

Card asks posted: #999 comments 6557063013 (chunking gate + fold timing), 6557158896 (corrections:
the folded chunk emits NO probability relayout), 6557427835 (the B=1 A/B: depth ITL at 4k/16k/64k,
short ITL, TTFT, coherence), 6557481210 (hd128 + pool bound).
NOT DONE: their per-sequence loop (refused by name; needs a pool <= 32 pages to matter, which no
in-scope config reaches — tinyllama-1.1b dies earlier on an untied head).
NEXT after the card: batch scaling (ITL level as batch grows), then P2 prefill/TTFT, then device
scf.for (gen 2).

## 2026-11-01 ~07:05 — ⛔ THE B=1 DEPTH CLAIM WAS WRONG; the real row is the x-batched contraction
Measured off the lowering (probe, not committed): granite-2b geom, hd64.
- SEQ B=1 over 32 blocks: **31 ops**, relaunched 32x; its contractions are batched over the 8 kv
  heads (a00165342).
- BATCHED B=1, J=32, C=1: **979 ops** one launch — `bmm×512`, `transpose×256` = ONE op per
  (page, kv head). ~24 contraction ops a page against 3. ~8x the device ops.
The break is between J=4 and J=32: bmms emitted are 2 / 2 / 512 at entries·kv = 8 / 32 / 256.
**At <=32 batches the contraction x-batches; past 32 (more batches than cores) it emits one op a
batch.** So the batched lowering is sound for every shape it has run (B=2-8, short walks,
entries·kv <= 32) and falls apart exactly where B=1's deep walk lands.
Retracted on the PR (6557522807). pr999-b1batched holds — both commits are right — but it does NOT
go to a perf gate until the contraction x-batches past 32.
▶️ NEXT = the sdsc_9/sdsc_24 x-batched forms past 32 batches: 256 transposes -> 32, 512 bmms -> 64.
THEN re-ask for the B=1 depth A/B with the op counts posted first.

## 2026-11-01 ~07:20 — the x-batched hunks RESTORE the depth win (claim re-made WITH numbers)
`08007b8dd` on pr999-b1batched = ae3904914's three remaining source hunks cherry-picked (its gather
half was already here as 3b60f8cc1): `matmul_opspec_x_batched` + `restickify_transpose_opspec_batched`
take `core_split` past MAX_CORES (consecutive blocks a core, 8 at 256), `work::row_groups` likewise.
Auditer found it; it was shelved only because its binary predated the KV zero-fill (fef674680) and
the gather collision fix (3b60f8cc1).

Ops the walk emits, one sequence over 128 blocks (~16k ctx):
| form | emitted | per page |
| per-sequence (B=1 today) | 23-op body x 128 trips | 23 |
| batched, pre-hunks | 979 straight-line at J=32 | ~30 |
| batched, 08007b8dd | **22-op body x 4 chunks** | **0.69** |
Contractions 512->2, Kt restickifies 256->1, both head dims. 32-block walk = one chunk, 212
straight-line, still ~3.5x fewer/page than 23x32.
⭐ EMULATOR NUMERICS UNCHANGED TO THE DIGIT at all 8 shapes (B=1/2/4, hd 64/128, deep+short) — so
the hunks move work division only. 2b fp8 bake passes.
Gate restored on #999 (6557629587), coherence first.

## 2026-11-01 ~07:30 — ⛔ EVIDENCE CORRECTION (auditer): the emulator can't see SDSC-only changes
`SpyreSession::new_multi` (runner.rs ~250-266) hands IRFunctions — KTIR — to the emulator Session.
`lower_walk`'s SDSC descriptors are only ever INSPECTED locally (the splice tests count ops and read
`numWkSlicesPerDim_`); the dxp bake COMPILES them, nothing local RUNS them. So for 08007b8dd (work
division only) "emulator unchanged to the digit" is CIRCULAR — identical either way. Corrected on
#999 (6557643020). Saved as [[reference_emulator_runs_ktir_not_sdsc]].
⇒ Standing rule for this port: for an SDSC-only change the local claim is "the descriptors say X",
never "the numbers are unchanged", and the first coherent card run IS the first evidence.

## 2026-11-01 ~07:40 — follow-up noted: the ONE-CHUNK arm is the weaker half
With 08007b8dd, per-page device ops at B=1 (granite, hd64):
  J=1  C=1 (128 tok)   27 ops /1 page   = 27
  J=4  C=1 (512 tok)   43 ops /4 pages  = 10.8
  J=32 C=1 (4k)       212 ops /32 pages = 6.6
  J=32 C=4 (16k)       22-op body x4    = 0.69
Every one of these beats the per-sequence walk's 23/page, so none is a regression — but the
ONE-CHUNK arm (C=1) is ~10x worse per page than the looped body. Cause: the kernel's `reduce_chunk`
arm folds over the J block slots INSIDE the straight-line program (`tl.max(tl.max(sc,4),0)`), and
that leading-axis reduction lowers to one op per slot (maximum x32, sub x65, add x66). The
multi-chunk arm keeps per-slot state and merges ONCE after the walk, so its body stays 22.
⇒ FOLLOW-UP (after the card gate): lower a leading-axis fold over J as one strided reduction, or
let a one-chunk walk take the entry-local arm. Worth ~6.6 -> ~1 per page at the 4k rung.
LX headroom checked for the hunks: restickify resident = 2*(blocks/cores)*rows*cols*2; granite's
worst case is blocks=256 (entries<=32 x nkvh 8), cores=32, so 8 blocks a core = 512 KB at hd128
against USABLE_LX_BYTES 1_677_721. Fits with room; nkvh 32 would refuse by name.
   CORRECTION to that follow-up: spyre-inference sets `use_entry_local = num_chunks > 1`
   (batched_decode_head_major.py:131) — identical to ours. So "let a one-chunk walk take the
   entry-local arm" would DEVIATE from their kernel and is OUT (port, never reinvent). The only
   legitimate fix is on OUR side: lower a leading-axis fold over J as ONE strided reduction instead
   of one op a slot. Kernel stays verbatim.

## 2026-11-01 ~07:15 — 08007b8dd: BOTH bakes pass; branch HELD for the card gate
granite-2b fp8 1m13s, granite-8b fp8 2m11s, both full-model, both exit 0. hd128 was the risk (the
batched restickify's LX residency) and it bakes. A passing bake = the descriptors COMPILE and the LX
plan FITS; it does NOT mean they compute right (see the SDSC/KTIR note above). Card only.
⛔ DELIBERATELY NOT LANDING the one-chunk reduction fix yet: it would be a FOURTH unvalidated SDSC
change on a branch where none has run on the card, which is exactly what makes a garbage result
un-attributable. Gate first, then land it.

## 2026-11-01 ~07:25 — PREFILL/TTFT: two axes RULED OUT by probe (negative evidence for P2)
Measured on 08007b8dd, off the lowering:
- the prompt trip's body is **23 ops, one trip a block, contractions already batched** (bmm x2 a
  trip, NOT one per kv head) at mq 64/128/512 and hd 64/128 — identical shape to the decode trip's.
  So the prefill walk was NEVER the per-(block,head) explosion the decode batch had.
- its body is **already LX-resident**: 20 LX buffers against 9 HBM, and the 9 are the unavoidable
  operands (query, the K/V pools, mask, out) — the same 20/9 split the decode trip shows.
⇒ The TTFT regression (2b 52.3s @10k vs main 8.5s) is NOT op count and NOT LX residency. It is the
trip's intrinsic HBM traffic (card: ~0.93 ms/trip) times the trip COUNT: a 10k prefill's last chunk
walks ~80 pages, x 40 layers, x ~20 chunks. So the levers are the ones P2 already names — bigger/
fewer trips, the 4-D [KV,G,P,.] trip tensors, and the device scf.for to cut per-trip records — not
a work-division fix like the decode one. Do NOT go looking at op counts or LX again for TTFT.

## 2026-11-01 ~07:30 — the one-chunk fold: HALVING, max only (cc542eacf, branch pr999-foldtree)
CORRECTS my earlier guess: the 32 extra ops are NOT a frontend decomposition. The KTIR is 92 ops at
J=4 AND J=32, one `LinalgReduce` a fold — OUR SDSC lowering unrolls it. And the unroll is necessary
in kind (whole_function.rs: the card's reduce folds along the STICK, so a leading-axis fold has no
single-op form). What was wrong: it WALKED the J windows (J-1 ops of one window) where it can HALVE
(log2 J levels of half the windows each). Same elements, same work, 5 ops not 31 at J=32.
Measured B=1: slot maximum 32 -> 6 ops; chunk 212 -> 186 (hd64), 277 -> 251 (hd128).
⛔ MAX ONLY — a numerics argument, not caution: max reassociates EXACTLY (same bits), fp16 add does
NOT, and nothing local executes these descriptors, so a sum tree is unverifiable here. The two sum
folds keep walking and the TEST ASSERTS THEY DO (`count("add") >= 2*(J-1)`), so the remaining ~52
ops cannot be taken without deliberately deleting that assertion. Sum tree = card-gated follow-up.
Details: the wide levels get their OWN scratch (widening the result made it present as its buffer,
[256,1] vs a [128,1] consumer — refused by the extent law, correctly); mint key discriminator 10
(MintKey::Scratch is (ssa,n) and this ssa is an OPERAND, i.e. another op's result, so 0 could alias).
🚨 NEW CARD-ONLY RISK: intermediate levels take BOTH operands from one buffer at disjoint offsets,
writing back over the left one. Reasoned safe (per-core lhs/dst geometry identical, rhs rows never
written; the old chain already accumulated in place) but new on this path.

## 2026-11-01 ~07:45 — audit round on the fold tree; pr999-foldtree head = 57aee51cd
- SUM ORDER: I could NOT establish torch-spyre's own summation order. `.scratch/torch_ref` holds 6
  of its 28 descriptors (sdsc_6/8/9/13/24/26) and the reductions are not among them. So neither the
  tree NOR the walk is verified; calling the walk faithful was wrong. 57aee51cd rewords the
  assertion to name its exit condition (a card run holding the new order to spyre-inference's
  output) and to say the walk is the INCUMBENT, not the reference. No emission change.
- ⭐ STRUCTURAL NOTE: their compiled decode trip is ONE 128-token block under a device `scf.for`
  with the softmax state carried across trips — so their program may contain NO leading-axis J-fold
  at all; the fold is an artifact of OUR straight-line one-chunk arm. If so, the right answer for
  the 4096 rung is the DEVICE LOOP (ladder step 6), not a better fold. Not acted on.
- MINT KEY: traced, NO live collision. The two `n=0` sites (whole_function.rs:4291, 4703) are the
  SAME purpose — the mid buffer of a two-permute head-major relayout — in mutually exclusive arms,
  so sharing is intentional. `(src,1)/(src,2+)` key a gather's loaded value and `(r_res,1)/(r_res,2)`
  a head-major pair's result: distinct ops ⇒ distinct SSA; they could collide only if a pair's result
  were gathered, which we never bake.
  ▶️ NEXT LOCKDOWN (not done, deliberately): replace `MintKey::Scratch(Ssa, u8)`'s hand-picked u8
  with an enum of mint PURPOSES, one variant a call site, so uniqueness is compiler-enforced. ~10
  sites over 2 files; deferred because nothing local executes SDSC (a mis-merged key would be
  invisible here and would CAUSE the aliasing it prevents), so it lands as its own gated change.

## 2026-11-01 ~08:55 — pr999-mintkey = a4e3db66c (NO LONGER HELD)
Sequence: fe6a116e6 typed `MintKey::Scratch(Ssa, u8)` -> `(Ssa, Scratch)`, 13 variants. The auditer
caught that for 11 sites it was one-to-one but for the FOUR that had picked `0` it was a SPLIT, and
my walk-level descriptor diff cannot reach two of them (lower_ktir 1195 scatter rows, whole_function
7479 block-relayout middle — lowered by the FULL-MODEL FORWARD, not `lower_walk`). "The bake passes"
does not close it either: compiles != unchanged.
a4e3db66c gives the four `Scratch::RelayoutMid`, the key the u8 gave them ⇒ every key unchanged, and
the two unreachable sites unchanged BY CONSTRUCTION, no bake dumps needed.
EVIDENCE (rebuilt from scratch — I had deleted the dumps for disk, so the earlier "identical" had no
files behind it): checked the 3 sources out at 57aee51cd (u8 era), dumped; restored my edits from
`.scratch/v6/shared_variant.patch` (md5 339f6b7e…), dumped; `cmp` ⇒ 10,582 descriptors /
85,676,902 B byte-identical. + ktir-superdsc full suite, splice, target-spyre, clippy, 2b fp8 bake.
⛔ NEW COMMIT, NEVER AN AMEND — fe6a116e6 was pushed; amending = force-push = rewriting history.
▶️ FUTURE: split the n=0 group, proof = before/after descriptor diff of the 2b AND 8b BAKES (the
four synth DIFFERENT extents and `synth` is first-one-wins, so a collision silently takes the
first's shape). The enum doc says this.
   REFINEMENT (auditer noted n=1/n=2 were split too): that split is a latent-bug FIX, not merely
   diff-covered. Those groups mix an OPERAND ssa with a RESULT ssa — 4769 `src` is the gathered
   value (an operand), 6725 `res` the current op's result, 5160 `r_res` a pair's result — and an
   operand of one op IS the result of another by ordinary dataflow. So under the u8, a batched
   contraction's result consumed by a chained-index gather would have keyed `(ssa, 1)` from BOTH
   sites and been handed one buffer at different synth extents (first-one-wins). My earlier
   "different ops ⇒ different SSA" argument was too glib: true across results, false between an
   operand and a result, which is the pair these groups mix. The empty cmp shows no LIVE instance;
   the split closes it for shapes we have not baked. Only the n=0 four have any case for sharing.

## 2026-11-01 ~09:05 — ⚠️ FLAGGED TO spyre-cb-tp: addr_eq widening needs a ONE-STICK GATE
They found the tp=8 (nkvh=1) blocker in `StickLayout::addr_eq` (sdsc_abstract.rs:1429) — at one kv
head the new-block score's group-0 read covers the WHOLE tensor, so it declares an arrangement:
Flat [G, m·L] against the q producer's stick-blocked [m, G·L]. Their claim that those are the same
bytes is TRUE AT ONE STICK A HEAD and FALSE ABOVE IT:
  one stick  (c = h·L+d):      sb (c/L)(mL)+rL+(c%L) = h·m·L + r·L + d ; flat g·mL + rL + d  ⇒ EQUAL
  two sticks (c = h·2L+sL+d):  sb (2h+s)·m·L + r·L + d ; flat 2h·mL + 2rL + sL + d
                               equal iff s(m-1) == r — at m=32 only (0,0) and (1,31) ⇒ NOT EQUAL
So granite-2b tp=8 (4,1,64) is fine and granite-8b tp=8 (4,1,128) would bake CLEAN WITH WRONG
ADDRESSES unless the acceptance is gated on `cols == G * lanes()` per group.
⭐ AND: A GREEN 2b BAKE CANNOT DETECT THE MISSING GATE — 2b is hd=64, where gated and ungated behave
identically. The discriminating test is the hd=128 NEGATIVE case (Flat [4, 32·128] vs stick-blocked
[32, 4·128] must stay a CONFLICT); an accepting test at hd=64 passes either way.
Precedents given them: `is_row_expansion_of` already demands a whole number of sticks and addr_eq's
own comment says a Flat side "really is row-major and would differ"; `same_card_bytes`
(emit/whole_function.rs) computes this per element and returns FALSE at hd=128 — the same boundary my
store-through-reshape reframe hit this morning.
   ⛔ CORRECTION (same session, ~09:10): MY OBJECTION ABOVE IS WRONG — withdraw it. In their law
   `is_stick_major_frame_of`, G is the STICK count, not the head count:
     `flat.rows == sb.cols / lanes && flat.cols == sb.rows * lanes`
   Checked exhaustively over all 32x512 elements at hd=128, nqh=4:
     per-STICK Flat [8, 2048] vs RowBlocked [32, 512] → IDENTICAL every element
     per-HEAD  Flat [4, 4096] vs RowBlocked [32, 512] → NOT identical
   So their law is correct at ANY head dim and REJECTS the per-head frame I feared. The gate I
   asked for was already there.
   ⭐ WHY I GOT IT WRONG, which is the same trap: their example was hd=64, where the head count and
   the stick count are THE SAME NUMBER (hd == one stick), so "Flat [4, 2048]" fitted both readings
   and I derived against my guess instead of asking which G. A non-discriminating example cannot
   tell a correct law from a broken one — I had just warned them of that shape one message earlier.
   Their added tests are the right pair: hd=128 NEGATIVE both directions, and an hd=128 per-stick
   POSITIVE byte-checked through `dev_off` before addr_eq (tests the thing, not my arithmetic).
   STILL OPEN (theirs): whether the 8b at tp=8 reaches that addr_eq call at all — their reading is
   that a slab-split score reads below the tensor's footprint and so does not declare; they will
   confirm from the eight-rank bake.
   RESOLVED (theirs, ~09:20): the 2b bake FAILED and found the case neither of us had — at the mq=4
   rung gqa == mq, so the head-major read's layout IS the tensor's own shape (Flat [4, 256] vs
   RowBlocked [4, 256]), which is the same-shape scramble rule, and NO 2-D StickLayout can carry the
   frame. They dropped the layout-level law for an explicit `ArrangementClaim`: `Rows(StickLayout)`
   or `HeadMajor { groups, rows, df }` for a rank-3 view whose stick axis is exactly one stick, and
   only a HeadMajor claim may frame a stick-blocked [rows, groups·L]. `view_stick_layout` became
   `view_arrangement(..).layout()` so addressing and declaration stay on one classifier. 12 tests,
   both orders, incl. the hd=128 per-head (Rows ⇒ conflict) / per-stick (HeadMajor ⇒ accepted,
   byte-checked) pair.
   ⭐ SAME REMEDY AS THE MINT KEY: carry the fact in a TYPE instead of inferring it from integers
   that happen to coincide (there, four sites had picked `0`; here, gqa == mq collapses two frames
   onto one shape). Second instance today.
   I suggested the discriminating identity check for their "no emitted byte moves": diff the 2b
   descriptors before/after — worlds 1/2/4 must be byte-IDENTICAL (they baked before), world 8 is
   the ONLY new acceptance. A whole-bake pass/fail cannot separate "identical" from "changed but
   still bakes".

## 2026-11-01 ~15:15 — ▶️ NEXT FINDING (not yet fixed): the carried MAX spills to HBM on a planner bug
Per-op HBM/LX audit of the batched chunk body (B=1, J=32, 128 blocks): **15 ops all-LX, 7 touch HBM.**
Four of the seven are the reference's own (the K/V pool gathers and the mask add — torch-spyre reads
those from HBM too, tsp 6/7/11). The other three all touch **t7, the carried maximum**:
| carried value | ours | torch-spyre |
| sum  (add_o8) | LX   | LX @33280 |
| acc  (add_o9) | LX   | LX @99328 |
| **max** (sub_o23, maximum_o7, sub_o27) | **HBM** | **LX @32768** |
`ae3904914`'s own message already admitted it ("every value it computes but the carried max in LX").

⭐ CAUSE, LOCALIZED to `emit::mod.rs::core_slices` (~1976). The planner keeps a tensor only when
every op's `corners` are EQUAL (lx_plan.rs ~347, `c == corners`), taking the max `bytes`. For t7 the
four touches have IDENTICAL splits `{mb:32}` and differ only in REACH — `sub_o27` reads it at
`out_:128`, the others at `out_:64`. The law explicitly allows that ("the slice reaches as far as
any op reads it"). But:
- `slice_bytes` NORMALIZES a broadcast/reduced stick dim to one stick, and its comment says so
  outright: "a per-row column a reduction writes and an elementwise op reads broadcast is the same
  slice to both";
- `corners` does NOT — they come from `per_core_addr` over `view_stick_layout(&as_hbm, plan)`, whose
  layout for a broadcast column depends on the CONSUMER's width.
So the same column read at two consumer widths yields two corner sets and spills, contradicting the
comment two lines above. Fix shape: normalize the view the same way before computing corners.
⚠️ NOT DONE, and deliberately: it changes LX PLACEMENT (i.e. addresses) for every broadcast-read
value, not just t7, so it is a card-gated change and there are already 9 commits awaiting a gate.
Queue it after the gate. Payoff: 3 of 22 trip ops stop touching HBM.
