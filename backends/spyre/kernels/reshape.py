# SPDX-License-Identifier: Apache-2.0
"""THE RESHAPE SPLICE'S KERNEL — the pure relayout `out = reshape(x)`: the same
element sequence, different `[rows, cols]` extents.

The whole input is one tile, reshaped to the output's extents and stored as one
tile: a single copy over the same row-major element sequence.

THE SPLICE'S OWN CONTRACT (what `scratchy-triton-splice` states about this kernel):

* PARAMETERS, IN ORDER: `desc_src`, `desc_dst` — the node's one input, then the
  output.
* CONSTEXPRS: `R_IN`, `C_IN`, `R_OUT`, `C_OUT`, `N_STICKS` — stated by the splice
  from the node's own regions and tensors; the producer refuses unless
  `R_IN·C_IN == R_OUT·C_OUT` (the element-count law) and both regions are whole
  tensors. `N_STICKS` (the buffer's `R·C/64`) is part of the signature the splice
  states; the copy does not read it.
* GRID: `[1]`.
"""

import triton
import triton.language as tl


@triton.jit
def reshape_fwd(desc_src, desc_dst,  #
                R_IN: tl.constexpr, C_IN: tl.constexpr,  #
                R_OUT: tl.constexpr, C_OUT: tl.constexpr,  #
                N_STICKS: tl.constexpr,  #
                ):
    src_desc = tl.make_tensor_descriptor(desc_src, shape=[R_IN, C_IN],
                                         strides=[C_IN, 1],
                                         block_shape=[R_IN, C_IN])
    dst_desc = tl.make_tensor_descriptor(desc_dst, shape=[R_OUT, C_OUT],
                                         strides=[C_OUT, 1],
                                         block_shape=[R_OUT, C_OUT])
    dst_desc.store([0, 0], tl.reshape(src_desc.load([0, 0]), [R_OUT, C_OUT]))
