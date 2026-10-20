# SPDX-License-Identifier: Apache-2.0
"""THE ROPE SPLICE'S KERNEL — Granite's split-half NeoX `rotate_half`, adapted from the
numerically validated fixture `crates/triton/test-fixtures/rope.py` (rope_q32/rope_kv8
in `triton-numeric`'s executed configs).

    q_embed = q * cos + rotate_half(q) * sin
    rotate_half(x) = cat(-x[..., d//2:], x[..., :d//2])

restated FULL-WIDTH through the rotate-half permutation matrix `P` — the
worker-staged `[HEAD_DIM, HEAD_DIM]` table whose values are the Kani-proven
`rope_p_entry` (the single source of truth the reserved-tid fill already uses):

    out = x * cos + tl.dot(x, P) * sin
    P[inn][o] = -1  if o < half and inn == o + half    (rot[o] = -x[o+half])
                +1  if o >= half and inn == o - half   (rot[o] = +x[o-half])
                0    otherwise

`tl.dot(x, P)` IS `rotate_half(x)` — the same math with NO half-width column
window. Every load and store is `[.., HEAD_DIM]` at column 0 (per head: at the
head's own column base), so every column window the door sees is WHOLE STICKS at
every head dim. The half-width form this kernel used to state — two `[H, HALF]`
tiles per trip at column offsets 0 and HALF — put a SUB-STICK window (HALF = 32
against the 64-fp16 stick) on the door at hd=64, which its stick law refuses:
that wall is the reason this kernel is full-width.

THE SPLICE'S CONTRACT — the door (`ktir_superdsc_door`'s whole-function walk)
mints each op's geometry from the program's own views and access tiles, so the
kernel states them:

* the x/out views are the WORKER's own staging, one of its two same-bytes
  spellings per form: the per-position form re-tiles the `[mq, heads·hd]` plane
  as the tall `[Y_TOTAL, HEAD_DIM]` view (row = position·H + head), and the
  per-head form reads the plane as it is, `[MQ, H·HEAD_DIM]`.
* ONE ACCESS TILE PER TRIP, FULL WIDTH — the per-position form's `[H, HEAD_DIM]`
  and the per-head form's `[MQ, HEAD_DIM]` (every whole-region form, decode
  included; see the routing comment) — every tile at ROW CORNER 0, the only row
  the door's matmul arm can address its activation at, and every column window
  WHOLE STICKS at every head dim.
* ONE WORK ITEM (grid `[1]`) with the taken body's trip loop UNROLLED by the
  ladder (`to_ktir::unroll_constant_trip_loops`): a constant-trip `tl.range`,
  each trip a copy of the body. The spliced program is straight-line, like the
  builder's.
* cos/sin: the worker stages per-position head-tilled rows (`spyre_forward.rs`'s
  `tile` over `rope_cos_sin`), so element (p, h*hd+d) sits at `p*total + h*hd +
  d` — the `[MQ*H, HEAD_DIM]` head-replicated view, read at the position's OWN
  row base; the per-head form reads the same bytes as its own `[MQ, total]`
  plane (element (p, h·hd+d) is tall row p·H+h, column d). The fixture's
  `stage_table` (delta 8) is the same bytes; the walk mints each op's geometry
  from its own views, not from the cos parameter's view.
* P: ONE whole-tensor `[HEAD_DIM, HEAD_DIM]` load at the origin — the descriptor's own
  view IS the whole table (strides `[HEAD_DIM, 1]`, row-major, the staging the worker
  binds), and the load sits OUTSIDE the trip loop so the unrolled body shares one
  permutation tile. The door lowers the dot as the plain `tt.dot` it is (no `.T`), the
  same form `decoder_block.py`'s score matmuls already carry.

THE ROW ORDER, kept from the fixture (delta 7, measured): row = position * H + head,
TOKEN-MAJOR, the per-position form's tall re-tile of the worker's `[mq, heads*hd]`
row-major staging — the arrangement the attention that consumes this tensor reads.
That re-tile is `addr_eq`'s same-bytes reshape only while `HEAD_DIM` is one stick
wide (hd=64); the routing law above carries what happens past that.

⭐ THE ROW WINDOW (per-row rope): the real front end's per-position decomposition
produces rope nodes whose regions are ONE row's `[1, qw]` slice of the shared q
plane, and a row slice is expressed by the ACCESS TILE's row corner, never by a
taller view (main's `view_rows` law: the offset rides the tile's row index, which
is the only form legible to both the emulator and the GEMM offload). The kernel
therefore states `Y_TOTAL` — the x TENSOR's full re-tiled height `rows·H`, so the
tile corner is in bounds — and `R_START` — the region's row corner `ri·H`, baked
into every load/store row. `MQ` is then the REGION's row count: a whole-region
node states `R_START = 0`, `Y_TOTAL = MQ·H`, which is byte-identical to the one
form this kernel used to state; a per-row node states `MQ = 1` at `R_START =
ri·H`. cos/sin take the same corner: position `p`'s angles are rows
`[p·H, (p+1)·H)` of the head-replicated table, so the region's rows are exactly
`[R_START, R_START + MQ·H)`. P takes NO corner — it is the whole table at every
trip, one region per class of head dim.

⛔ BUT THE DOT'S OWN A-LAW: a per-row node's corner reaches the door's matmul
arm, which reads its activation AT THE BUFFER BASE and REFUSES a stated row
corner BY NAME (the seal on the lm-head tail's once-discarded corner —
`base_addressed`). The half-width kernel walked its trips through the pointwise
arm, whose `band_corner` carries a one-stick row-cornered window (hd=128 only —
at hd=64 the sub-stick window was the wall this kernel closes); the dot has no
such arm. The sanctioned spelling for a row-cornered operand is `lmlast`'s
MATERIALIZATION — copy the row into a synthetic at row 0 and read that — and
until the rope grows one, a per-row rope node is a loud build refusal, never a
silently mis-addressed read. Production's nodes are whole-region (the plane the
worker stages), so the delivery scope never states the corner.
"""

import triton
import triton.language as tl


@triton.jit
def rope_fwd(desc_x, desc_cos, desc_sin, desc_p, desc_o,
             H: tl.constexpr, MQ: tl.constexpr,
             HEAD_DIM: tl.constexpr,
             Y_TOTAL: tl.constexpr, R_START: tl.constexpr):
    tl.static_assert(HEAD_DIM % 2 == 0)
    # ⭐⭐⭐ THE ROUTING IS THE ARRANGEMENT TRUTH, not a body-count pick. Two laws
    # meet here:
    #   * THE DOOR'S MATMUL A-LAW: the rotation's dot lowers through the door's
    #     matmul arm, which reads its activation AT THE BUFFER BASE — an A tile at
    #     a nonzero ROW corner has no address (`base_addressed`, the seal on the
    #     lm-head tail's once-discarded corner), while a stick-aligned COLUMN
    #     corner is addressable.
    #   * THE ARRANGEMENT AUTHORITY: the worker's plane `[mq, heads·hd]` and its
    #     tall `[heads·mq, hd]` re-tile are the SAME BYTES only while `hd` is one
    #     stick wide (`addr_eq`'s row expansion demands `cols == lanes`) — at
    #     hd=128 the tall view is a TWO-STICK SCRAMBLE of the plane's bytes, a
    #     producer/consumer disagreement `declare_arrangement` refuses by name.
    # So EVERY whole-region node (`H > 1`, `R_START == 0`) takes `_rope_per_head`,
    # whose x/out views are the PLANE ITSELF — the producer's own arrangement at
    # every head dim and every row count, tiles `[MQ, HEAD_DIM]` at row 0 and
    # stick-aligned column corners. `_rope_per_position` — the tall view, tiles
    # `[H, HEAD_DIM]` at row `R_START + pos·H` — is kept for the forms that are
    # not whole-region: the tall one-head node (`H == 1`, not production's shape)
    # and the per-row node (`R_START > 0`), where the door's own laws speak (the
    # matmul A-corner refusal; at hd>64 the arrangement refusal) rather than a
    # kernel silently mis-addressing bytes.
    #
    # The trip count is constexpr either way, so `to_ktir` unrolls the taken body
    # into a straight-line program. Body count: the per-head form holds `H` bodies
    # and the per-position one `MQ`, so decode (`MQ == 1`) pays `H` bodies for the
    # one view whose bytes are the producer's own — the price of arrangement truth.
    #
    # Nested constexpr ifs, not `and`: the parser refuses a Python `BoolOp` in a @triton.jit kernel.
    if H > 1:
        if R_START == 0:
            _rope_per_head(desc_x, desc_cos, desc_sin, desc_p, desc_o, H, MQ, HEAD_DIM)
        else:
            _rope_per_position(desc_x, desc_cos, desc_sin, desc_p, desc_o, H, MQ, HEAD_DIM,
                               Y_TOTAL, R_START)
    else:
        _rope_per_position(desc_x, desc_cos, desc_sin, desc_p, desc_o, H, MQ, HEAD_DIM,
                           Y_TOTAL, R_START)


@triton.jit
def _rope_per_position(desc_x, desc_cos, desc_sin, desc_p, desc_o,
                       H: tl.constexpr, MQ: tl.constexpr,
                       HEAD_DIM: tl.constexpr,
                       Y_TOTAL: tl.constexpr, R_START: tl.constexpr):
    # The x/out views over the worker-staged plane, `[rows*heads, hd]` — the full
    # TENSOR re-tiled (Y_TOTAL), so the row-windowed tile corners stay in bounds.
    # The cos/sin views are the same-bytes head-replicated tables, `[tbl_rows*H,
    # hd]` — their extents agree with x's when the tables are per-position rows.
    # FULL WIDTH: one `[H, HEAD_DIM]` tile per load — the rotate-half pairing is
    # the dot's job now, not a second half-width column window.
    x_desc = tl.make_tensor_descriptor(desc_x, shape=[Y_TOTAL, HEAD_DIM],
                                       strides=[HEAD_DIM, 1],
                                       block_shape=[H, HEAD_DIM])
    o_desc = tl.make_tensor_descriptor(desc_o, shape=[Y_TOTAL, HEAD_DIM],
                                       strides=[HEAD_DIM, 1],
                                       block_shape=[H, HEAD_DIM])
    # THE WORKER-STAGED cos/sin, `[mq*heads, hd]` head-replicated (spyre_forward.rs's
    # `tile`): position p's angles are H consecutive rows starting at p*H — read at x's
    # own row base, so cos rows line up with x's H head-rows row for row.
    cos_desc = tl.make_tensor_descriptor(desc_cos, shape=[Y_TOTAL, HEAD_DIM],
                                         strides=[HEAD_DIM, 1],
                                         block_shape=[H, HEAD_DIM])
    sin_desc = tl.make_tensor_descriptor(desc_sin, shape=[Y_TOTAL, HEAD_DIM],
                                         strides=[HEAD_DIM, 1],
                                         block_shape=[H, HEAD_DIM])
    # THE PERMUTATION, one whole-tensor load OUTSIDE the trip loop: every unrolled
    # trip's dot shares this tile. The view IS the whole `[HEAD_DIM, HEAD_DIM]`
    # table (strides `[HEAD_DIM, 1]`, the row-major bytes the worker binds).
    p_desc = tl.make_tensor_descriptor(desc_p, shape=[HEAD_DIM, HEAD_DIM],
                                       strides=[HEAD_DIM, 1],
                                       block_shape=[HEAD_DIM, HEAD_DIM])
    p = p_desc.load([0, 0])

    # ONE ACCESS TILE PER POSITION (`[H, HEAD_DIM]` at row base `R_START + pos * H`,
    # column 0). The trip count is constexpr, so `to_ktir` unrolls it and the spliced
    # program is straight-line with one tile per position. A whole-region node states
    # `R_START = 0`, so its tiles sit at `pos * H` exactly as before; a per-row node
    # states `MQ = 1` and its single trip's corner is the region's own row.
    for pos in tl.range(0, MQ, 1):
        offs_m = R_START + pos * H
        x = x_desc.load([offs_m, 0])
        c = cos_desc.load([offs_m, 0])
        s = sin_desc.load([offs_m, 0])
        # rotate_half as a GEMM: `dot(x, P)` is `cat(-x2, x1)` full-width, so the
        # store is the ONE NeoX line `x·cos + rotate_half(x)·sin`. The f16
        # accumulator is the device's own dot law (`attn.py`'s scores spell it
        # the same way): the door refuses an f32-accumulator dot.
        o_desc.store([offs_m, 0], x * c + tl.dot(x, p, out_dtype=tl.float16) * s)


@triton.jit
def _rope_per_head(desc_x, desc_cos, desc_sin, desc_p, desc_o,
                   H: tl.constexpr, MQ: tl.constexpr,
                   HEAD_DIM: tl.constexpr):
    # PER HEAD, over the plane as it is: `[MQ, H * HEAD_DIM]`, one `[MQ, HEAD_DIM]` tile
    # per head — full width, the whole head at its own column base. The logical identity
    # with the tall view is exact (row p, column h*HEAD_DIM + d is tall row p*H + h,
    # column d), and the BYTE identity is the arrangement authority's row expansion —
    # same bytes only while HEAD_DIM is one stick wide, which is why the whole-region
    # routing takes this form: its view IS the producer's plane. The cos/sin tables are
    # the same head-replicated bytes, so each head's tiles read its own angles.
    width = H * HEAD_DIM
    x_desc = tl.make_tensor_descriptor(desc_x, shape=[MQ, width], strides=[width, 1],
                                       block_shape=[MQ, HEAD_DIM])
    o_desc = tl.make_tensor_descriptor(desc_o, shape=[MQ, width], strides=[width, 1],
                                       block_shape=[MQ, HEAD_DIM])
    cos_desc = tl.make_tensor_descriptor(desc_cos, shape=[MQ, width], strides=[width, 1],
                                         block_shape=[MQ, HEAD_DIM])
    sin_desc = tl.make_tensor_descriptor(desc_sin, shape=[MQ, width], strides=[width, 1],
                                         block_shape=[MQ, HEAD_DIM])
    # The same whole-tensor permutation, loaded once for every head's dot.
    p_desc = tl.make_tensor_descriptor(desc_p, shape=[HEAD_DIM, HEAD_DIM],
                                       strides=[HEAD_DIM, 1],
                                       block_shape=[HEAD_DIM, HEAD_DIM])
    p = p_desc.load([0, 0])
    for h in tl.range(0, H, 1):
        col = h * HEAD_DIM
        x = x_desc.load([0, col])
        c = cos_desc.load([0, col])
        s = sin_desc.load([0, col])
        o_desc.store([0, col], x * c + tl.dot(x, p, out_dtype=tl.float16) * s)
