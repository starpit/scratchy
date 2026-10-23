# SPDX-License-Identifier: Apache-2.0
"""THE ROPE SPLICE'S KERNEL — Granite's split-half NeoX `rotate_half`, adapted from the
numerically validated fixture `crates/triton/test-fixtures/rope.py` (rope_q32/rope_kv8
in `triton-numeric`'s executed configs).

    q_embed = q * cos + rotate_half(q) * sin
    rotate_half(x) = cat(-x[..., d//2:], x[..., :d//2])

which, written per half (x1 = x[..., :d//2], x2 = x[..., d//2:]), is

    out1 = x1 * cos - x2 * sin
    out2 = x2 * cos + x1 * sin

THE SPLICE'S CONTRACT — the door (`ktir_superdsc_door::rope`) derives the node's facts
from the program's own views and access tiles, so the kernel states them:

* the x/out descriptors are `[MQ * H, HEAD_DIM]` — the WORKER's staging. Main's
  `KtirFunc::rope` views x as `[rows*heads, hd]` over the same bytes, so the view extents
  agree.
* ONE ACCESS TILE PER POSITION, `[H, HALF]` — the door reads `heads` off the FIRST access
  tile's row extent (`Region::r_len`) and `mq` as `v_rows / heads`, so a whole-tensor
  tile would state `heads = MQ*H` and mis-derive everything. Main's builder program
  takes exactly these tiles (main's `KtirFunc::rope`'s `tile(x_view, ri*heads, 0, heads, half)`).
* ONE WORK ITEM (grid `[1]`) with the position loop UNROLLED by the ladder
  (`to_ktir::unroll_constant_trip_loops`): constant-trip `tl.range(0, MQ, 1)`, each trip
  a copy of the per-position body. The spliced program is straight-line, like the
  builder's.
* cos/sin: the worker stages per-position head-tilled rows (`spyre_forward.rs`'s `tile`
  over `rope_cos_sin`), so element (p, h*hd+d) sits at `p*total + h*hd + d` — the
  `[MQ*H, HEAD_DIM]` head-replicated view, read at the position's OWN row base. The
  fixture's `stage_table` (delta 8) is the same bytes; the door's "NOT THE COS VIEW"
  trap does not bite here because `rope_at` derives its extents from `mq`/`total`, not
  from the cos parameter's view.

THE ROW ORDER, kept from the fixture (delta 7, measured): row = position * H + head,
TOKEN-MAJOR, which is `addr_eq`'s same-bytes reshape of the worker's `[mq, heads*hd]`
row-major staging — the arrangement the attention that consumes this tensor reads.

⭐ THE APPEND FORM (`rope_append_fwd`, below) is `RopeAppend`'s OWN contract — the
front-end arity-6 `[K, cos, sin, V, K_cache, V_cache]` the arity-3 port collapsed.
It rotates K and writes it, with V, straight into the pool planes; its own doc states
the store laws. `rope_fwd` above stays `RopeRotate`'s (the q stream's, no cache).

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
`[R_START, R_START + MQ·H)`.
"""

import triton
import triton.language as tl


@triton.jit
def rope_fwd(desc_x, desc_cos, desc_sin, desc_o,
             H: tl.constexpr, MQ: tl.constexpr,
             HEAD_DIM: tl.constexpr, HALF: tl.constexpr,
             Y_TOTAL: tl.constexpr, R_START: tl.constexpr):
    tl.static_assert(HALF + HALF == HEAD_DIM)
    # ⭐ THE LOOP RUNS OVER THE SHORTER AXIS — positions or heads — so the program holds
    # `min(MQ, H)` bodies and stops growing with the chunk. The trip count is constexpr either
    # way, so `to_ktir` unrolls it into a straight-line program.
    #
    # ⛔ IT WAS ALWAYS THE POSITION LOOP, AND THAT GREW WITH THE CHUNK: one body per position, so a
    # 512-row prompt chunk's q rope was 512 bodies, every one walked by every pass of the lowering.
    #
    # Nested constexpr ifs, not `and`: the parser refuses a Python `BoolOp` in a @triton.jit kernel.
    # One head (H == 1) always takes the position loop: the door tells the forms apart by the view's
    # width, and at one head both views are one head dim wide. A ROW-WINDOWED rope (`R_START > 0`,
    # the per-row form — one row, so never `MQ > H`) keeps the position loop too: its window rides
    # the tall view's tile corner, which the per-head form's whole-plane view does not state.
    if MQ > H:
        if H > 1:
            if R_START == 0:
                _rope_per_head(desc_x, desc_cos, desc_sin, desc_o, H, MQ, HEAD_DIM, HALF)
            else:
                _rope_per_position(desc_x, desc_cos, desc_sin, desc_o, H, MQ, HEAD_DIM,
                                   HALF, Y_TOTAL, R_START)
        else:
            _rope_per_position(desc_x, desc_cos, desc_sin, desc_o, H, MQ, HEAD_DIM, HALF,
                               Y_TOTAL, R_START)
    else:
        _rope_per_position(desc_x, desc_cos, desc_sin, desc_o, H, MQ, HEAD_DIM, HALF,
                           Y_TOTAL, R_START)


@triton.jit
def _rope_per_position(desc_x, desc_cos, desc_sin, desc_o,
                       H: tl.constexpr, MQ: tl.constexpr,
                       HEAD_DIM: tl.constexpr, HALF: tl.constexpr,
                       Y_TOTAL: tl.constexpr, R_START: tl.constexpr):
    # The x/out views over the worker-staged plane, `[rows*heads, hd]` — the full
    # TENSOR re-tiled (Y_TOTAL), so the row-windowed tile corners stay in bounds.
    # The cos/sin views are the same-bytes head-replicated tables, `[tbl_rows*H,
    # hd]` — their extents agree with x's when the tables are per-position rows.
    x_desc = tl.make_tensor_descriptor(desc_x, shape=[Y_TOTAL, HEAD_DIM],
                                       strides=[HEAD_DIM, 1],
                                       block_shape=[H, HALF])
    o_desc = tl.make_tensor_descriptor(desc_o, shape=[Y_TOTAL, HEAD_DIM],
                                       strides=[HEAD_DIM, 1],
                                       block_shape=[H, HALF])
    # THE WORKER-STAGED cos/sin, `[mq*heads, hd]` head-replicated (spyre_forward.rs's
    # `tile`): position p's angles are H consecutive rows starting at p*H — read at x's
    # own row base, so cos rows line up with x's H head-rows row for row.
    cos_desc = tl.make_tensor_descriptor(desc_cos, shape=[Y_TOTAL, HEAD_DIM],
                                         strides=[HEAD_DIM, 1],
                                         block_shape=[H, HALF])
    sin_desc = tl.make_tensor_descriptor(desc_sin, shape=[Y_TOTAL, HEAD_DIM],
                                         strides=[HEAD_DIM, 1],
                                         block_shape=[H, HALF])

    # ONE ACCESS TILE PER POSITION (`[H, HALF]` at row base `R_START + pos * H`,
    # column offsets 0 and HALF). The trip count is constexpr, so `to_ktir`
    # unrolls it and the spliced program is straight-line with one tile per
    # position. A whole-region node states `R_START = 0`, so its tiles sit at
    # `pos * H` exactly as before; a per-row node states `MQ = 1` and its single
    # trip's corner is the region's own row.
    for pos in tl.range(0, MQ, 1):
        offs_m = R_START + pos * H
        x1 = x_desc.load([offs_m, 0])
        x2 = x_desc.load([offs_m, HALF])
        c = cos_desc.load([offs_m, 0])
        s = sin_desc.load([offs_m, 0])
        # rotate_half, per half: the MINUS is on the second half's contribution to
        # the first, which is `cat(-x2, x1)` written out.
        o_desc.store([offs_m, 0], x1 * c - x2 * s)
        o_desc.store([offs_m, HALF], x2 * c + x1 * s)


@triton.jit
def _rope_per_head(desc_x, desc_cos, desc_sin, desc_o,
                   H: tl.constexpr, MQ: tl.constexpr,
                   HEAD_DIM: tl.constexpr, HALF: tl.constexpr):
    # PER HEAD, over the plane as it is: `[MQ, H * HEAD_DIM]`, one `[MQ, HALF]` tile per head
    # half. Same bytes as the tall view (row p, column h*HEAD_DIM + d is tall row p*H + h,
    # column d), and the cos/sin tables are the same head-replicated bytes, so each head's
    # tiles read its own angles. The door reads this form as `mq = v_rows`, `hd = 2 * c_len`,
    # `heads = v_cols / hd` off x's view and first tile.
    width = H * HEAD_DIM
    x_desc = tl.make_tensor_descriptor(desc_x, shape=[MQ, width], strides=[width, 1],
                                       block_shape=[MQ, HALF])
    o_desc = tl.make_tensor_descriptor(desc_o, shape=[MQ, width], strides=[width, 1],
                                       block_shape=[MQ, HALF])
    cos_desc = tl.make_tensor_descriptor(desc_cos, shape=[MQ, width], strides=[width, 1],
                                         block_shape=[MQ, HALF])
    sin_desc = tl.make_tensor_descriptor(desc_sin, shape=[MQ, width], strides=[width, 1],
                                         block_shape=[MQ, HALF])
    for h in tl.range(0, H, 1):
        col = h * HEAD_DIM
        x1 = x_desc.load([0, col])
        x2 = x_desc.load([0, col + HALF])
        c = cos_desc.load([0, col])
        s = sin_desc.load([0, col])
        o_desc.store([0, col], x1 * c - x2 * s)
        o_desc.store([0, col + HALF], x2 * c + x1 * s)


@triton.jit
def rope_append_fwd(desc_x, desc_cos, desc_sin, desc_v, desc_kc, desc_vc,
                    H: tl.constexpr, HEAD_DIM: tl.constexpr, HALF: tl.constexpr,
                    X_ROWS: tl.constexpr, ROW_START: tl.constexpr,
                    PART_ROWS: tl.constexpr, SLOT_START: tl.constexpr,
                    PLANE_SLOTS: tl.constexpr):
    """THE APPEND FORM — `RopeAppend`'s own contract, the front-end arity-6 shape the
    arity-3 port dropped: rotate K AND write it, with the un-rotated V, into the paged
    KV pool. The rotation is `_rope_per_head`'s math; the STORES ARE THE CACHE WRITES —
    there is no `desc_o`, because nothing on the device path reads the rotated
    intermediate (the splice never binds `AttnDecode`'s `new_k`), and the reference's
    shape — the outputs ARE the cache stores — wins.

    ⭐ THE SCALE LAW, STATED AT THE WRITE. The cached K is the rope's PLAIN output:
    `attn_scale_weight_folds` REFUSES the fold for an fp8 k_proj (the weight is
    `Fp8Dynamic`, arity-3), so no `√scale` rides the projection, and every pool READER
    states its own scale (`qk * (SCALE * LOG2E)` in the batched kernels). A model whose
    fold fired composes at the other end (`√s · √s = s` at scale 1.0). Do not fold a
    scale into this store.

    THE POOL PLANES, at `PagedKvPool`'s own law: one plane per kv head, `[PLANE_SLOTS,
    HEAD_DIM]` row-major, head `h`'s plane at row `h * PLANE_SLOTS`. A part's rows land
    at slots `[SLOT_START, SLOT_START + PART_ROWS)` of its page — the part is cut so it
    stays inside one page (`PagedKvPool::write_parts`, the shared cutting law), the page
    itself is named by the RUNTIME through the block table (`kv_request` /
    `kv_slot_offset`), and the decode slot by the launch's `seq_pos` shift
    (`slot_stride_bytes`). The stores therefore bake `SLOT_START` and nothing else.

    THE ROW WINDOW, per head: the x/cos/sin/v views are the worker-staged plane
    `[X_ROWS, H * HEAD_DIM]`, and this part's rows are `[ROW_START, ROW_START +
    PART_ROWS)` — the same window law `rope_fwd`'s tall form states, at the plane view
    where one row IS one position. cos/sin are the head-replicated tables, read at the
    part's own row base so each head's tiles line up row for row.
    """
    tl.static_assert(HALF + HALF == HEAD_DIM)
    tl.static_assert(
        PART_ROWS >= 1,
        "rope_append_fwd: a part covers at least one row")
    tl.static_assert(
        SLOT_START + PART_ROWS <= PLANE_SLOTS,
        "rope_append_fwd: a part must stay inside one page of the pool plane")
    width = H * HEAD_DIM
    x_desc = tl.make_tensor_descriptor(desc_x, shape=[X_ROWS, width], strides=[width, 1],
                                       block_shape=[PART_ROWS, HALF])
    cos_desc = tl.make_tensor_descriptor(desc_cos, shape=[X_ROWS, width],
                                         strides=[width, 1],
                                         block_shape=[PART_ROWS, HALF])
    sin_desc = tl.make_tensor_descriptor(desc_sin, shape=[X_ROWS, width],
                                         strides=[width, 1],
                                         block_shape=[PART_ROWS, HALF])
    v_desc = tl.make_tensor_descriptor(desc_v, shape=[X_ROWS, width], strides=[width, 1],
                                       block_shape=[PART_ROWS, HEAD_DIM])
    # The two pool planes: `[H * PLANE_SLOTS, HEAD_DIM]` — head h's plane is rows
    # `[h * PLANE_SLOTS, (h + 1) * PLANE_SLOTS)`, and this part writes slots
    # `[SLOT_START, SLOT_START + PART_ROWS)` of it.
    kc_desc = tl.make_tensor_descriptor(desc_kc, shape=[H * PLANE_SLOTS, HEAD_DIM],
                                        strides=[HEAD_DIM, 1],
                                        block_shape=[PART_ROWS, HALF])
    vc_desc = tl.make_tensor_descriptor(desc_vc, shape=[H * PLANE_SLOTS, HEAD_DIM],
                                        strides=[HEAD_DIM, 1],
                                        block_shape=[PART_ROWS, HEAD_DIM])
    for h in tl.range(0, H, 1):
        col = h * HEAD_DIM
        x1 = x_desc.load([ROW_START, col])
        x2 = x_desc.load([ROW_START, col + HALF])
        c = cos_desc.load([ROW_START, col])
        s = sin_desc.load([ROW_START, col])
        # rotate_half, per half — `_rope_per_head`'s own math, stored straight into the
        # K plane instead of an intermediate.
        kc_desc.store([h * PLANE_SLOTS + SLOT_START, 0], x1 * c - x2 * s)
        kc_desc.store([h * PLANE_SLOTS + SLOT_START, HALF], x2 * c + x1 * s)
        # V passes through un-rotated, head for head, slot for slot.
        v = v_desc.load([ROW_START, col])
        vc_desc.store([h * PLANE_SLOTS + SLOT_START, 0], v)
