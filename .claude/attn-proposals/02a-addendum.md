# 02a — addendum to 02 (P1 census / P2 / P4 per your verdict). 02 lands only after your
P1 branch ruling; P2/P4 are mechanical.

## P1 — THE CENSUS (narrow single-tile-N tiled-path GEMMs in scope, and their m ranges)

**The class condition** (from `of()`): a GEMM is narrow iff `max(k, n) < 4096` AND `n`
decomposes over `FP8_GEMM_N_SET = [4096, 1024, 128]` into exactly ONE tile — i.e.
**n ∈ {1024, 128}** (any other 128-multiple splits into ≥2 tiles ⇒ wide; anything ≥4096
is wide outright). So the census question per model: does any fp8 projection have output
width exactly 1024 or 128, with k < 4096?

**The m values a GEMM can see**: decode m=1 (dense by small-M) and the prefill ladder
rungs `[7, 11, 15, 17, 19, 21, 23, 25, 27, 29, 31, 35, 39, 43, 47, 55, 63, 71, 80, 88,
96, 112, 128, 192, 256, 384, 512]` (run-45-judgment-chat.log line 44; a chunk runs on the
smallest rung that holds it). Unaligned rungs above 128: exactly **{192}** — it pads to
**256**. Every other unaligned rung is ≤ 112 and pads to 128.

**Per model** (widths read from `crates/models/arch/configs/`, text_config):

| model | projections (k→n) | class |
|---|---|---|
| granite-3.1-2b (hidden 2048, inter 11008, kv 512) | qkv 2048→3072 (3 tiles); q 2048 (2 tiles); k/v 512 (4 tiles); o 2048 (2 tiles); gate/up 11008; down 11008→2048 | **all WIDE — zero narrow sites** |
| granite-3.1-8b (hidden 4096) | every input projection has k=4096 ⇒ max(k,n)≥4096; gate/up→12800; down 12800→4096 | **all WIDE** |
| gemma-3-12b (hidden 3840, inter 15360) | qkv 3840→8192; o →4096; gate/up →15360; down 15360→3840 | **all WIDE** |
| gemma-4-26b-a4b (hidden 2816, moe_inter 704, experts 128, global kv 2×512) | local qkv →8192; q 4096; local k/v 2048 (2 tiles); o 4096; dense gate/up →2112 (3 tiles); expert w1/w3 →704 (6 tiles); w2 →2816 (8 tiles) | all wide — **except two CONTINGENT candidates below** |

**The two 26b candidates (narrow by geometry, contingent on path membership):**
1. **global-kv projections**: global kv = 2 heads × global_head_dim 512 = **n=1024**,
   k=2816 < 4096 ⇒ single-tile NARROW — *if* the global-kv linears are separate fp8 GEMMs
   on the card path and not folded into the wide local qkv. Whether gemma-4's global and
   local projections emit as one fused GEMM is a lowering fact I could not settle from the
   configs alone; the gemma-4 attention lowering is the open Triton-source question.
2. **the MoE router**: hidden 2816 → **n=128** (num_experts), k=2816 ⇒ single-tile NARROW —
   *if* the router is an fp8 KTIR matmul through this door. The MoE lowering is open
   (fold vs fused `SubOp::Moe`), and route_topk's per-op path was deleted (#58/#712), so
   this site's membership is genuinely unresolved.

**Census verdict**: NO in-scope GEMM is CONFIRMED narrow — granite 2b/8b and gemma-3-12b
have zero narrow sites; the 26b has two candidates whose path membership is an open
lowering question, and if either is real, the **192 rung pads to 256** — above 128.

**The branch I recommend: your branch 1 (honest refusal), because the generalization is
unforced.** No confirmed site needs it, and `Narrow(RowCount)` drags the arm's whole
const-generic lock (`Fp8TileEmit::<ROWS>`, `Fp8TileStaging::<ROWS>::STAGED_ROWS`) into
runtime-rows territory for a site nobody has confirmed — the class would be minted, not
typed. The refusal names exactly what to do the day a real site lands. Amended
`row_class()` (replacing 02's Err arm; the doc carries the census):

```rust
    /// ⭐ THE TILED PLAN'S ROW CLASS — the emission's dispatch fact. A dense plan (rows run
    /// whole at `m`) never reaches this door: the emission dispatches it first, and a plan
    /// whose rows are neither ABI class arriving here is a plan invariant break — refused
    /// BY NAME, fail-closed, exactly as `n_tile_count` refuses a width. The refusal is not
    /// a dispatch arm; the dispatch itself is exhaustive over the enum.
    ///
    /// ⛔ THE NARROW ARM EXISTS AT EXTENT 128 ONLY — narrow padded extents above 128 rows
    /// are not emitted. Census (02a, `crates/models/arch/configs/`): no in-scope GEMM is
    /// confirmed narrow at all (granite 2b/8b and gemma-3-12b: every projection wide by
    /// split-N or ≥4096); gemma-4-26b has two UNRESOLVED candidates (global-kv n=1024,
    /// MoE router n=128) whose door membership rides the open gemma-4-attention and MoE
    /// lowerings — if either lands, the 192 prefill rung pads to 256 and needs the class
    /// generalized in `work` first. A model that does needs exactly that: the class
    /// generalized in `work`, never a second spelling at the emission.
    pub fn row_class(&self) -> Result<Fp8RowClass, String> {
        match self.rows.get() {
            FP8_GEMM_WIDE_M => Ok(Fp8RowClass::Wide),
            FP8_GEMM_NARROW_M_ALIGN => Ok(Fp8RowClass::Narrow128),
            rows => Err(format!(
                "fp8 GEMM row class {rows}: not a tiled ABI class — whole-row plans dispatch \
                 to the dense arm before the tiling door, and narrow padded extents above 128 \
                 rows are not emitted (no in-scope GEMM produces one; see row_class's doc for \
                 the census and the 26b candidates). Generalize the class in `work` first, \
                 never a second spelling at the emission"
            )),
        }
    }
```

If you rule branch 2 anyway (generalize now despite no confirmed site), say so and I will
re-propose the arm fresh — the shape is `Fp8RowClass::Narrow(RowCount)` with the narrow
emission context going runtime-rows (a separate struct from the const-generic
`Fp8TileEmit<ROWS>`, since `Fp8TileStaging::<ROWS>` cannot take a runtime extent), and
you rule on that whole design. 04's F8 test then pins the generalized form instead of
extent-128.

## P2 — 01's R1 folded into 02's door

02's door line, amended (same fix as 01a's R1):
OLD:
```rust
    if plan.tile_count().get() == 1 && plan.total_rows() == RowCount::of_token_rows(m) {
```
NEW:
```rust
    if plan.tile_count() == crate::work::Fp8GemmTileCount::new(1)
        && plan.total_rows() == RowCount::of_token_rows(m)
    {
```

## P4 — the geometry-guard message

OLD:
```rust
                        "fp8 matmul {out}: tile {t} of t{} was already reserved at {have} bytes but \
```
NEW:
```rust
                        "fp8 matmul {out}: tile {t} of activation t{} was already reserved at {have} bytes but \
```

## Connection to 04's F8 test

Under branch 1 the test pins extent-128 as the in-scope law and cites this census in its
doc (per your P1 note). If you rule branch 2, the test pins the generalized form. The
test's exact code follows your ruling — I will file the amended F8 hunk as part of 04's
application either way.
