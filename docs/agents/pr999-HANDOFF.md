# PR #999 — handoff to the new lead

## The job

Port torch-spyre / spyre-inference's **paged attention** to Triton and make it run on the Spyre
card. `AI-native-Systems-Research/scratchy` PR **#999**. The port works on the emulator. It is not
done until three gaps vs `main` close:

| gap | now | `main` |
|---|---|---|
| ITL at depth | 92.6 ms | 42.3 ms |
| TTFT @10k | 52.3 s | 8.5 s |
| two concurrent decodes | incoherent | correct |

## Where it stands: **the PR head cannot serve one request, on either model**

Every request 500s on the first decode step:

```
superdsc decode stage (mq=1, at=20): bind: raw tensor 't4292967123' … has no placement
```

Nothing on this PR can be measured until that is fixed. **This is the whole job right now.**

## What we know about it — this is the lead you want

The card ran a clean bisect today (PR comments 6560404488, 6560413795), with a no-drift control in
the same hour:

| head | serves? | output |
|---|---|---|
| **v4** = pre-rebase lineage + the `fixgather` patch stack | **yes** | **coherent**, 2b and 8b |
| bare `797df4104` (pre-rebase, unpatched) | no | the 500 |
| `fef674680` / **A** `74987efb6` (the PR head, post-rebase) | no | the 500 |
| **B** `08007b8dd` and above | yes | garbage |

So the fix **already existed** in the pre-rebase patched lineage, and the rebase lost it. It is in
the delta between the v4 tree and bare `797df4104`.

That patch is `docs/agents/pr999/pr999_fixgather_v4.patch` (6 files). Three cannot be the fix:
`spyre_load.rs` is a comment, `splice/src/lib.rs` is a test, `superdsc_exec.rs` zeroes the KV pool
(needed for coherence, but not a *placement* change). **The fix is in the other three —
`lower_ktir_to_superdsc.rs`, `whole_function.rs`, `pointwise.rs` — or in pre-rebase commits the
patch sat on.**

Independently, from the tid: `t4292967123` is the **page index table**, which
`PagedStep::card_placements` (`crates/targets/spyre/src/wiring.rs:750`) places *only* in the
Sequence arm. So the mq=1 seat stages a Sequence source into a bundle with no Sequence placements.
An emission change can cause exactly that — consistent with the above.

## What to do first

1. **Recover the bind fix** from that delta. Everything else is blocked on it.
2. Then gate the three branches that are already written and never validated: `pr999-relayout`
   (`5cc98e429`, the concurrent-decode fix), `pr999-b1batched` (`08007b8dd`, the depth lever),
   `pr999-foldtree` (`57aee51cd`).
3. Then TTFT. 33,560 trips × 0.93 ms = 31.2 s of the 52.3 s; the prompt page gather runs on **one**
   core where torch-spyre uses 8. One attempt failed (it broke four prompt-walk tests); the redo is
   via a host-folded page table.

## How work actually gets validated

**You cannot validate any of this locally.** The emulator runs KTIR, not SDSC — every emission
change (work division, core splits, placements) is invisible to every local test, and "the numbers
didn't change" is circular there. A full-model bake only proves it compiles.

Real validation is the **card agent on pods `nickm` / `nickm2`**, driven through **PR #999
comments**: you post the heads and cells you want run, it posts back measurements. Both pods were
free at handover. Keep a poller on the thread or you will not see results
(`docs/agents/pr999/watch_pr999.sh`).

## Standing rules that bit us

- Push to **`origin`** (starpit) only — never `upstream`. Never rewrite history. Sign every commit.
- **Gate on the card first, then push to the PR branch.** Nine untested commits went onto it once;
  that was wrong and they were rewound.
- Never claim a card result from an emulator run or a passing bake. That mistake was made twice and
  publicly retracted both times.

## Everything else

Detail — all branches and SHAs, both defects in full, the ruled-out axes, the carried-max planner
bug, the full local suite, the reference capture — is in
**`docs/agents/pr999-lead-handoff.md`**, same branch. Supporting artifacts are in
`docs/agents/pr999/` (the v4 patch, the running log, torch-spyre's descriptor capture, the bake
scripts). Branch: `origin/pr999-mintkey`. PR head: `74987efb6`.
