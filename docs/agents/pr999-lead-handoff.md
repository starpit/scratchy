# PR #999 — lead handoff

Written 2026-11-01 ~16:00Z by the outgoing lead, for a lead picking this up on another machine.
Everything here is checkable: branch SHAs, PR comment ids, file:line. Where something is a
hypothesis rather than a measurement, it says so.

---

## 1. The goal

`AI-native-Systems-Research/scratchy` PR **#999** ports torch-spyre / spyre-inference's **paged
attention** to Triton and lowers it to the Spyre card. The port is functionally done and the
emulator agrees with spyre-inference's own kernels. **The PR is not done because it is slower than
`main` and it breaks under concurrency.**

Three gaps, from the last full card A/B (granite-3.1-2b fp8):

| | this PR | `main` | |
|---|---|---|---|
| ITL short / mid / **deep** | 24.1 / 30.9 / **92.6** ms | 26.8 / 27.8 / **42.3** ms | **2.19× worse at depth** |
| TTFT @1k / **@10k** | 1252 ms / **52.3 s** | 623 ms / **8.5 s** | **6.2× worse at 10k** |
| two concurrent decode | **incoherent** | coherent | **functional break** |

The standing directive from the user: **do what torch-spyre/spyre-inference does** — port their
compiled program, do not invent. Burn down every deviation from it. B=1 performance first, then
batch scaling (ITL should stay level as batch grows until saturation).

---

## 2. Where things stand

### PR branch
`feat/spyre-paged-attention` is at **`74987efb6`** ("A"). CI green. **It is NOT card-clean** — see
the open defect in §4.1.

### Work branches on `origin` (= `github.com/starpit/scratchy`)

| branch | head | contains | status |
|---|---|---|---|
| `pr999-fixq` | `74987efb6` | = the PR head | the baseline all A/Bs use |
| `pr999-relayout` | `5cc98e429` | **the concurrent-decode fix** | ⭐ **card run in flight** |
| `pr999-mintkey` | `95a42fe4c` | the depth work + housekeeping | local-green, uncarded |
| `pr999-b1batched` | `08007b8dd` | subset of the above, superseded | keep for bisects |
| `pr999-foldtree` | `57aee51cd` | subset, superseded | keep for bisects |

`pr999-mintkey` is the live stack. `74987efb6..95a42fe4c` is, oldest first:

```
966bcf71f  fix   output stored through a byte-identical reshape uses the op's own frame
2c6c3f1b2  feat  decode ONE request with the batched kernel, as spyre-inference does
ca6cb33a0  test  B=1 executed at head dim 128; the pool-bound refusal pinned
08007b8dd  feat  a decode batch's chunk in the decode trip's form   <-- the depth lever
cc542eacf  perf  leading-axis maximum folds by halving, not walking
57aee51cd  test  say what lifts the sum-fold assertion
fe6a116e6  refac name each minted scratch instead of picking a u8
a4e3db66c  refac share one variant for the relayout middles
ce6f15bc2  fix   stage the KV pool's tables in every prefill session
95a42fe4c  rev   take the prompt trip's page gather back out       <-- removed, see §5.1
```

### ⚠️ Nothing on `pr999-mintkey` has ever produced a correct card run.
`08007b8dd`'s predecessor decoded garbage on the 8b; the branch has not been re-tested since.
Treat every performance figure attached to it as a **compiler op count, not a measurement**.

---

## 3. In flight right now

### ⛔⛔ THE ONE BLOCKER: the PR head cannot serve a single request, on EITHER model.

Card-proven as of PR comment **6560275506**. Every cell — 2b and 8b, solo and concurrent, seven each
— 500s at `mq=1` with the §4.1 bind failure. Nothing on this PR can be measured until that is fixed.
**Fixing it is the whole job right now.**

1. **`E` = `pr999-relayout` (`5cc98e429`) is BLOCKED, not pending.** It builds **byte-identical to A
   on both models** (2b `cb6dbf53…`, 8b `1ebf3b00…`; trees verified to differ on the pod), so its
   diff is emission-neutral on A's lineage — and every E cell 500s with A's failure anyway. See
   §4.2 for why emission-neutral is actually the *expected* result, and what it implies.
2. **The bisect is running on `nickm2`.** `fef674680` built **byte-identical to A on 8b** (so
   `6fcd27dbb` is emission-neutral there and its cells will 500 the same way). **`797df4104`
   (pre-rebase) is the deciding cell** — it says whether the bind failure came in with the rebase
   onto `upstream/main` or predates it. **ANSWERED (PR comment 6560404488): bare `797df4104`
   fails identically, through `scr chat` AND `scr serve`, on 2b AND 8b — so the bind failure
   PREDATES the rebase.** A same-hour control of this morning's v4 (`797`-plus-patch) binary was on
   the card when I handed over; its verdict closes whether the patch stack was what made `797`
   serve. **Start at §4.1a — it names the one hypothesis left and how to test it without a card.**
3. **The 2b `C` leg** (= B's 2b binary `4829f00e…`) is the only unmeasured 2b point, queued behind.
4. **A PR-thread watcher must be running.** See §6.1 — if it is not, you are not receiving card
   results.

---

## 4. The two live defects

### 4.1 The PR head hard-500s on granite-8b — **unexplained, and it is the PR head**

Every request fails with:

```
bind: raw tensor 't4292967123' (1024 B) has no placement in this bundle …
```

`reserved_region("paged_step").at(idx)` is `base − idx`, **descending**
(`crates/targets/spyre/ktir/ktir-superdsc/src/reserved_tids.rs:264`), and
`PAGED_STEP_BASE = u32::MAX − 2_000_171`. So **`t4292967123` is `at(1)` — the page index table**,
and 1024 B = 8 blocks × the 32-word int32 stick = a walk of 8 pages.

So a decode step at `mq=1` staged the page table into a bundle with no placement for it.
`PagedStep::card_placements` (`crates/targets/spyre/src/wiring.rs`) places `page_table`, `mask` and
`query_rows` **only** in the one-sequence arm, and `rep_rows`/`chunk_pages`/`chunk_masks` only in
the batch arm — so this is a Sequence/Batch mismatch between the seat and the baked bundle.

Ruled out by inspection: `6fcd27dbb`'s entire `wiring.rs` diff is in the **batch** arm; the
one-sequence arm is byte-identical. What that commit *does* change is the batch rungs' layout
(`rep_rows` grows up to 8× at the 8b under the gather floor) — look there if the bisect clears the
rebase. **Evidence: PR comment 6560054671 (the failure), 6560131764 (the tid identification).**

#### ⭐ 4.1a What the card and the tid arithmetic now say together (2026-11-01 16:0x)

Two facts arrived after the section above was written. Together they narrow this to one hypothesis
the next lead can test **without a card**.

**Fact A — the card agent's discriminator (PR comment 6560404488).** Bare `797df4104` fails
identically through `scr chat` *and* `scr serve`, on 2b *and* 8b, with prefix caching on *and* off:
`superdsc decode stage (mq=1, at=20): bind: raw tensor 't4292967123'`. So it is not the serve
wrapper, not a flag, not the port — it is the paged decode path at the **mq=1 seat**, on the first
decode step. Their boot logs show the batch rungs (`seqs=2/4/8/16/32 ready`) and the
`try_load (decode m=1)` session both build; only the *bind* fails. They also noticed a second,
`paged=false` standalone dxp session (fp `0e128e36e297afa9`) alongside the paged engine and flagged
that they do not know whether that duality is expected — **it is worth confirming.**

**Fact B — what `t4292967123` can only be.** It is `at(1)` = `PAGED_PAGE_TABLE_TID`, and
`PagedStep::card_placements` (`crates/targets/spyre/src/wiring.rs:750`) places `page_table` /
`mask` / `query_rows` **only** in the `None` (Sequence) arm, `rep_rows` / `chunk_pages` /
`chunk_masks` **only** in the `Some` (Batch) arm. A missing placement for `page_table` therefore
means exactly one thing: **the step was staged as Sequence into a bundle that was baked as Batch.**

**The contradiction.** On `74987efb6`/`797df4104`, `rows_are_requests = !is_prefill && decode_rows > 1`
(`crates/compiler/macros/src/codegen.rs`), so the **W=1 decode bundle bakes Sequence** — and a local
`SCRATCHY_PROBE_PAGED` bake of the 2b confirmed it prints `paged_rows=Sequence rows=1`. A Sequence
bundle *does* place `page_table`. So the bundle the mq=1 seat **binds into** is not the W=1 bundle
whose bake we just inspected.

**The hypothesis to test first — the owner session may be built on the wrong bundle.**
`SpyreWirings::decode_seat` (`crates/serving/worker/src/spyre_types.rs:422`) hands `RungId::Owner`
`&mut self.decode` and *asserts in a comment* "the owner is the m=1 bake". That session is built at
`crates/serving/worker/src/spyre_load.rs:1773` from `code`, and `code` comes from the fingerprint
stamped into `groups[0].decode.graph_json` (`spyre_load.rs:1077-1092`). **Nobody has checked that
that slot carries the m=1 bake rather than the widest decode rung.** If codegen stamps the decode
slot with the last/widest decode bake, the owner session's `places` map is a *Batch* bundle's, the
mq=1 Sequence staging can never find `page_table`, and every symptom follows: both models, both
entry paths, first decode step, batch rungs fine.

**Why this also predicts the ladder we measured.** `2c6c3f1b2` changes `rows_are_requests` to
`!is_prefill`, making the W=1 bundle **Batch** — which would make both sides agree by accident and
let the request run. That is exactly what the card saw: **A 500s; B/C/D run and return garbage.**
If this hypothesis holds, `2c6c3f1b2` is masking the bug, not fixing it, and the real fix is in the
decode-slot stamp (or in `decode_seat` selecting by width).

**How to test it, no card needed.** Bake the 2b with the `SCRATCHY_PROBE_PAGED` probe described in
§6.2, but print the *fingerprint* alongside `paged_rows`/`rows` in `compute_bundle_layout`, and
separately print the fp codegen stamps into the decode slot. If the stamped fp is a `rows>1` bake,
the hypothesis is confirmed. (The probe needs `-vv` to surface build-script stderr and a `touch` of
the model crate to defeat caching.)

### 4.2 Concurrent decode is incoherent — **root cause CONFIRMED on the card, fix written**

The card agent's stage checker over a batch-of-2 tensordump found the whole math chain clean
(Kᵀ, scores, scaling, mask add with correct live-key counts, max chain, exponentials, row sums) and
the **first mismatch at `exb = blocks(ex)`** — the probability relayout — writing **zeros where `ex`
holds 0.03–0.09**. A *partial write*, not a wrong computation; `pv`, `cout`, `attn` inherit it.
**Evidence: PR comment 6559989746.**

That is the predicted signature of the relayout copying **64 runs over 32 cores — two per core** at
B=2, where a core's second run is placed by the destination's device-extent pitch. Every
card-proven use of that copy gives each core **one** run, and at 64-slot pages the relayout does not
exist at all — which is exactly where the bisect put the first bad commit.

**`5cc98e429` is the fix** (cap the copy at 32 runs an op). Test
`a_decode_batch_lays_its_probabilities_out_one_run_a_core` fails without it, passes with it.

⭐ **But it is emission-neutral on A — and that is probably good news.** The card agent measured E
byte-identical to A on both models (6560275506). The reason is `6fcd27dbb`, already in A: once a
chunk under the gather floor folds by kv head, **the folded chunk emits no probability relayout at
all** (the correction in PR comment 6557158896). The broken multi-run copy only survives in the
whole-page form, which is in B's batched-decode stack, not A's. So **gap 3 may already be closed on
A by the chunking fix**, with `5cc98e429` as belt-and-braces for the shapes that still use the
relayout. None of that is measurable until §4.1 is fixed — do not claim it until a card run says so.

---

## 5. Findings worth acting on

### 5.1 TTFT: the prompt trip's page gather runs on ONE core — **attempt 1 failed, redo it differently**

A 10k prompt walks ~33,500 trips (20 chunks × 40 layers); the card measured the trip at ~0.93 ms and
attributed it **HBM-traffic bound**; the trip's only large HBM mover is the page gather — and it runs
on **one core**. An unfolded `[1, nkvh·block·hd]` row is a single `mb` position with the kv heads on
a `y` axis, and a gather divides over cores by `mb` **alone** (splitting any other axis divides the
index entry and breaks `skip_addr` — `gather_copy_cores`). torch-spyre splits it eight ways
(`sdsc_6`, `numWkSlicesPerDim_ {mb: 8}`).

`ab179e09a` made the kernel read `kv_row_pool` to get 8 index entries. It **failed on the card**
(garbage from token 1, both models) **and broke four local laws**, because the ops the pool read adds
to the prologue displace the relayout that lets the chunk's output scatter in **one** op — at mq=64
it became 32, one per stick column. Reverted in `95a42fe4c`.

**Do it on the HOST instead:** stage `page · nkvh` in the page index table and have the kernel read
`[nkvh, block·hd]` at that row. The trip then gains **no ops at all**. The page table is read by the
prompt trip alone once a decode bundle's rows are its requests', so its contents are free to change.
⛔ It ships only when these four still pass:
`a_prompt_chunks_rows_move_in_one_op_each`,
`a_two_stick_head_prompt_walk_relays_its_heads_and_gathers_each_page_in_one_op`,
`the_prompt_trip_splits_its_query_rows_across_the_cores_in_lx`,
`the_walk_relaunches_its_body_once_per_block`.

### 5.2 The carried maximum spills to HBM on a planner bug — **localized, not fixed**

Per-op audit of the batched chunk body: 15 ops all-LX, 7 touch HBM. Four are the reference's own
(K/V pool gathers, mask add). The other three all touch the **carried maximum**, where the carried
sum and accumulator are both in LX (torch-spyre has all three in LX: `@32768`, `@33280`, `@99328`).

Cause, in `emit/mod.rs::core_slices` (~1976): the planner keeps a tensor only when every op's
`corners` are **equal** (`emit/lx_plan.rs` ~347), sizing by the furthest reach. The carried max's
four touches have **identical splits** and differ only in reach (`out:128` vs `out:64`), which the
law explicitly allows. But `slice_bytes` **normalizes** a broadcast stick dim to one stick — its own
comment says "a per-row column a reduction writes and an elementwise op reads broadcast is the same
slice to both" — while `corners` do **not**, because they come from `per_core_addr` over
`view_stick_layout(&as_hbm, plan)`, whose layout for a broadcast column depends on the *consumer's*
width. Fix shape: normalize the view the same way before computing corners.
⚠️ This moves **LX placement — i.e. addresses — for every broadcast-read value**, so it is a
card-gated change. Payoff: 3 of 22 trip ops stop touching HBM.

### 5.3 Ruled out for TTFT — do not re-measure these

The prompt trip is **already** a 23-op body with contractions batched (2 bmms a trip, not one per kv
head) and **already LX-resident** (20 LX buffers against 9 HBM, the 9 being the unavoidable
operands). Identical shape to the decode trip. So TTFT is **neither an op-count nor a residency
problem**. Negative evidence, recorded so nobody pays for it twice.

### 5.4 Depth: what `08007b8dd` is claimed to do (uncarded)

B=1 decode at 128 blocks (~16k context):

| | body | relaunches | cores |
|---|---|---|---|
| per-sequence trip (the PR head) | 23 ops | **128** | **8 of 32** |
| batched chunk (`08007b8dd`) | 22 ops | **4** | **32 of 32, every op** |

33× fewer op-executions and 4× the machine on each. **Compiler counts. Unverified.**

---

## 6. Methodology — read this before trusting any check

### 6.1 The card loop (this is how work actually gets validated)

There is **no Spyre card on the dev machine**. A separate **card agent (pod `nickm` / `nickm2`)**
runs cells and posts results **as PR #999 comments**. The loop is:

1. push a branch to `origin`;
2. post a request to #999 naming exact SHAs, features, and cells;
3. **watch the thread** and answer its results and questions.

Step 3 is the part that gets dropped. Arm it:

```bash
SINCE=$(date -u +%Y-%m-%dT%H:%M:%SZ) bash .scratch/watch_pr999.sh   # under Monitor, re-arm on expiry
```

(`.scratch/` is untracked — the script is 15 lines of `gh api` polling and is trivial to rewrite:
poll `issues/999/comments`, `pulls/999/comments`, `pulls/999/reviews`, emit one line per new id.)

⛔ **Gate first, then push to the PR branch.** The established pattern is side branch → card gate →
push (`f2f470f5c GATE PASSED → PUSHED to PR`). The outgoing lead inverted it once and had to rewind.

### 6.2 ⛔⛔ What each local check CAN and CANNOT see

**This is the single most important section. Five retractions on this PR in one day came from
getting it wrong.**

| check | sees | **does NOT see** |
|---|---|---|
| `triton-numeric`, `triton_splice_golden` (emulator) | the **KTIR** — kernel algebra, chunking, host staging | ❌ **anything about SDSC**: work division, core splits, `numWkSlicesPerDim_`, per-core addresses. The emulator runs `IRFunction`s; `lower_walk`'s descriptors are only ever *inspected*. |
| a passing bake | descriptors **compile**, LX plan **fits** | ❌ whether they compute the right answer |
| splice `--lib` tests | the emitted descriptors' **shape** (op counts, splits, extents) — these pin the torch-spyre laws | ❌ what the card does with them |
| the card | everything that matters | — |

Consequences, learned the hard way:
- For an **SDSC-only** change (work division, core splits), the honest local claim is *"the
  descriptors say X"* — **never** *"the numbers are unchanged"*. Identical emulator errors are what
  you get whether the change is right or wrong.
- Conversely, before deferring something as "card-only", ask whether a **before/after identity**
  argument is available — identity *is* provable locally even where behaviour is not. That is how
  `fe6a116e6` was validated (10,582 descriptors, `cmp` byte-identical).
- **Run the splice crate's own lib tests after touching a kernel or the lowering.** Four laws were
  broken and pushed because only four *other* suites were run.

### 6.3 The full local suite

```bash
cargo test -p scratchy-triton-splice --lib          # the torch-spyre shape laws — DO NOT SKIP
cargo test -p ktir-superdsc
cargo test -p scratchy-target-spyre --lib
cargo test -p triton-numeric --test paged_attn      # 8 cases vs spyre-inference's own kernels
cargo test -p scratchy-target-spyre --features spyre-emu --test triton_splice_golden
cargo clippy -p <crate> --all-targets -- -D warnings
bash .scratch/capture_tc3/bake_2b_fp8.sh            # and bake_8b_fp8.sh — full-model type-check
```

The bake scripts need the stub SDK (`.scratch/stubsdk`, `FLEX_RS_SPYRE_ROOT`,
`SCRATCHY_SKIP_SENDNN_CXX=1`, `DXP_STANDALONE`, `DEEPTOOLS_PATH`). Rebuild from
`.scratch/capture_tc3/bake_fp8.sh` if `.scratch` is lost.

### 6.4 The reference

`~/git/spyre-inference` and `~/git/torch-spyre` are readable **for this port only** (a standing
exception to the no-home-directory rule). Cite `file:line` when claiming "theirs does X" — several
claims this session were wrong until checked against the source.
`.scratch/torch_ref/` holds a **partial** capture of their compiled decode program (6 of 28
descriptors: `sdsc_6/8/9/13/24/26`); the reductions are **not** among them, so their summation order
is unknown. `.scratch/perf_plan.md` is the running log — ~400 lines, read the last 150.

### 6.5 Working rules that bit us

- Never rewrite history; build forward. (One force-push happened, user-ordered, to rewind the PR.)
- Sign every commit (`git commit -S`), Conventional Commits, scoped.
- Never push to `upstream`; `origin` is `starpit/scratchy`.
- Every change goes to the auditing session for review before it is called done.
- Never use `/tmp`; use `.scratch/`.
- One plain command per Bash call — the worktree guard rejects compound commands.

---

## 7. What I would do first

1. **Fix §4.1. Nothing else matters.** The PR head 500s on every request on both models. Performance
   work, the relayout fix, the depth claim — all of it is unmeasurable until one request completes.
   The deciding bisect cell (`797df4104`, pre-rebase) may already have landed; if it serves, the
   bind failure came in with the rebase onto `upstream/main` and that is where to look. If it also
   500s, the defect is older than this PR's recent history and the search widens.
   Local angle worth trying in parallel: bake the 2b and inspect whether the W=1 decode bundle's
   layout actually places `PAGED_PAGE_TABLE_TID` — the wiring says it should (one-sequence arm of
   `card_placements`), the executor says it does not, and that contradiction is resolvable on this
   machine without a card.
2. Then re-request E's coherence cells (§4.2) on a base that serves.
3. Only then: gate `pr999-mintkey` for the depth claim (§5.4), and redo TTFT per §5.1.

Do not add further uncarded commits to the stack. There are already nine, and attribution is the
thing that has been scarce all day.
