# 11a — addendum to 11 (R2, per the verdict): the narrow sites join 03's caller lists

No code change now — this is the inventory 03's re-audit consumes when it unholds. When
03 lands, every new narrow site from 11 is retyped by 03's existing hunks, exactly as the
wide sites are; the standing question ("what does torch-spyre do here?") applies to each
retyping unchanged — the types wrap the reference's laws.

- **(i) B1 (`MatM::of_token_rows` takes `RowCount`)** — three new call sites join the list:
  `fp8_narrow_mm_op`'s mm (`MatM::of_token_rows(rows)`, from the staging's extent) and
  `Fp8NarrowEmit::emit`'s clone and out-splice (`MatM::of_token_rows(m)` each). At 03's
  final form these take the minted counts (`staging.rows()` / `plan.real_rows()`) whole.
- **(ii) A1 (the `fq_role` one-mint)** — the hand-built `SynthRole::FqTiled(_, 0)` wraps
  in `fp8_narrow_mm_op` (FqRaw) and `Fp8NarrowEmit::emit` (FqTileRow/FqAfp8/FqAscale/
  FqRaw/FqDqA) join the one-mint's caller list; the narrow part's index is always 0 (one
  part), so the mint specializes cleanly.
- **(iii) D1's `Fp8QuantSource` extent enum** — the narrow literal (`tile: Some(0)`,
  `rows` = the padded extent) is the enum's `Tiled { rows, idx }`-shaped third member
  beside the dense and wide-tiled forms.
- **(iv) D2's `k_pad` threading** — `Fp8NarrowEmit::emit`'s
  `fp8_quantize_descriptors(source, k, 1)` call joins the threading list (the context's
  single mint feeds it, pairs=1, same as the wide context).
- **(v) A3 (the stick-group windows)** — `Fp8NarrowEmit::emit`'s raw `for g in 0..k_groups`
  and `for c in 0..n_groups` loops join the windows' caller list.
- **(vi) D4 (typed params)** — `fp8_narrow_mm_op`'s `n`/`k_pad` u32 params join the typed
  door params (they read 03's minted `BlockCols` locals at the same call shape as
  `fp8_tile_mm_op`).
