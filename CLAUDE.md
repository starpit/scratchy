# THE INVIOLABLE RULES

1. **The ONLY test is `scr chat` on the card.** No emulator, no goldens, no
   scratch drivers, no standalone bundle runs. Done = a coherent reply that
   matches main's.
2. **No PR until it works.** Nothing is pushed or opened until `scr chat` on
   the card is right.
3. **We are a compiler. Everything is a constant.** Const generics, no raw
   scalars. Every shape, extent, offset and stride is a typed compile-time
   constant.
4. **No raw addressing math.** Every address goes through a typed helper. If the
   output is garbage or dxp_standalone rejects a bundle, the addressing is
   wrong: the types are not locking the invariant down. Fix the type, never the
   arithmetic.
5. **No hard-coded anything for one model**, and **the lowering knows ops, never
   kernels.** No recognizer of a kernel's op chain, no attention- or
   MoE-specific arm in the lowering. If a general op cannot express what a
   kernel states, fix the general op (or change the kernel), and the build
   fails naming the op until then.
6. **Every iteration takes 2 minutes MAX** (warm build < 1 min, chat ≤ 12 s).
   If one takes longer, STOP immediately and find out why before doing anything
   else.

# Build and test on the pod

We have **4 cards**, one per pod in namespace `a1-vllm-spyre`:
`nickm`, `nickm2`, `nickm3`, `nickm4`. Use all of them in parallel. Find the
live pod names with `oc get pods -n a1-vllm-spyre | grep nickm`.

Sync the tree (never the target dir), then build warm and chat, on the pod:

```bash
oc rsync -n a1-vllm-spyre --delete --exclude target --exclude .git ./ <pod>:/work/<tree>/

export PATH=~/.cargo/bin:$PATH HF_HOME=/work/.cache/huggingface LD_LIBRARY_PATH=/opt/ibm/spyre/senlib/lib:/opt/ibm/spyre/runtime/lib:/opt/ibm/spyre/deeptools/lib DEEPTOOLS_PATH=/opt/ibm/spyre/deeptools/share
cd /work/<tree>
cargo build -Fspyre-hw,model/granite-3.1-2b-instruct,quant/fp8-dynamic-per-channel --release
./target/release/scr chat RedHatAI/granite-3.1-2b-instruct-FP8-dynamic --system-prompt "" -q "write about the history of france" --bench --max-tokens 64
```

Keep one warm `target/` per tree per pod; a cold build is not an iteration.

## Commits
[Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/), scoped
to the crate or area touched (`fix(spyre): ...`). Signed (`-S`), new commits
only.
