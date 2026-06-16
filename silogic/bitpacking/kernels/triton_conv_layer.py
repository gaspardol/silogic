"""Triton GPU kernel for bitpacked conv gate-tree layer inference.

Extends the FC gate-grouped kernel to the spatial B-packed domain:
``[n_rows, L, nw_B]`` int64 (B-packing: batch packed, spatial explicit).

For each tree level the outputs are dispatched by gate type — one
``_bitpacked_conv_gate_kernel`` launch per active gate type.

Grid: ``(ceil(n_g / BLOCK_G), L, ceil(nw_B / BLOCK_W))``.

The extra ``L`` axis in the grid means all spatial positions are processed
in parallel, turning the gate-tree evaluation into the same scatter-store
pattern as :mod:`.triton_layer` but over a 3-D activation cube.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
import triton
import triton.language as tl

from ..packed_layer import BitpackedConvTreeLayer


# ── block size constants ──────────────────────────────────────────────────────

BLOCK_G = 64   # gate-group dimension (# of gate outputs per block)
BLOCK_W = 16   # word dimension (# of int64 words per block)


# ── Triton kernel ─────────────────────────────────────────────────────────────

@triton.jit
def _bitpacked_conv_gate_kernel(
    x_ptr,          # [n_rows_in,  L, nw_B] int64 — current level activations
    out_ptr,        # [n_rows_out, L, nw_B] int64 — next level activations
    a_idx_ptr,      # [n_g] int32 — row indices into x for left  operand
    b_idx_ptr,      # [n_g] int32 — row indices into x for right operand
    out_idx_ptr,    # [n_g] int32 — row indices into out for result
    n_g,            # int — number of active gate outputs in this group
    L,              # int — number of spatial positions
    nw_B,           # int — int64 words per (row, spatial) pair = ceil(B/64)
    gate: tl.constexpr,       # 0-15, JIT-specialised — no warp divergence
    BLOCK_G: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """One specialised kernel per gate type per tree level."""
    pid_g = tl.program_id(0)   # gate-group block
    pid_l = tl.program_id(1)   # spatial position (grid dim = L, no tiling)
    pid_w = tl.program_id(2)   # word block

    g_offs = pid_g * BLOCK_G + tl.arange(0, BLOCK_G)
    g_mask = g_offs < n_g
    w_offs = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    w_mask = w_offs < nw_B

    # Load row indices (int32 in memory → promote to int64 for address arithmetic)
    ai      = tl.load(a_idx_ptr   + g_offs, mask=g_mask, other=0).to(tl.int64)
    bi      = tl.load(b_idx_ptr   + g_offs, mask=g_mask, other=0).to(tl.int64)
    out_row = tl.load(out_idx_ptr + g_offs, mask=g_mask, other=0).to(tl.int64)

    # Element (row, l, w) in [n_rows, L, nw_B] has flat offset:
    #   row * (L * nw_B) + l * nw_B + w
    LnwB  = tl.cast(L,     tl.int64) * tl.cast(nw_B, tl.int64)
    l_off = tl.cast(pid_l, tl.int64) * tl.cast(nw_B, tl.int64)

    mask = g_mask[:, None] & w_mask[None, :]

    a = tl.load(
        x_ptr + ai[:, None] * LnwB + l_off + w_offs[None, :],
        mask=mask, other=0,
    )
    b = tl.load(
        x_ptr + bi[:, None] * LnwB + l_off + w_offs[None, :],
        mask=mask, other=0,
    )

    # Gate dispatch — gate is tl.constexpr; dead branches eliminated at JIT time.
    if gate == 0:
        r = tl.zeros([BLOCK_G, BLOCK_W], dtype=tl.int64)
    elif gate == 1:
        r = a & b
    elif gate == 2:
        r = a & ~b
    elif gate == 3:
        r = a
    elif gate == 4:
        r = ~a & b
    elif gate == 5:
        r = b
    elif gate == 6:
        r = a ^ b
    elif gate == 7:
        r = a | b
    elif gate == 8:
        r = ~(a | b)
    elif gate == 9:
        r = ~(a ^ b)
    elif gate == 10:
        r = ~b
    elif gate == 11:
        r = a | ~b
    elif gate == 12:
        r = ~a
    elif gate == 13:
        r = ~a | b
    elif gate == 14:
        r = ~(a & b)
    else:   # gate == 15: True — all bits set
        r = tl.full([BLOCK_G, BLOCK_W], value=-1, dtype=tl.int64)

    tl.store(
        out_ptr + out_row[:, None] * LnwB + l_off + w_offs[None, :],
        r,
        mask=mask,
    )


# ── GPU conv tree layer ───────────────────────────────────────────────────────

class BitpackedConvGPULayer:
    """Triton GPU bitpacked gate-tree conv layer.

    Mirrors :class:`~.packed_layer.BitpackedConvTreeLayer` but dispatches one
    :func:`_bitpacked_conv_gate_kernel` launch per active gate type at each
    tree level.  Activation layout: ``[n_rows, L, nw_B]`` int64, where
    ``n_rows = n_out * n_nodes_at_level``.

    Build from a converted CPU layer via :meth:`from_cpu_layer`.

    Parameters
    ----------
    level_groups:
        One list per tree level.  Each entry is a tuple
        ``(gate, a_idx, b_idx, out_idx)`` where the three index tensors are
        ``[n_g]`` int32 on device — flat row indices into the current- and
        next-level activation cubes.
    leaf_idx:
        ``[n_out * n_leaves]`` int64 on device — flat P-dimension row indices
        that select which unfolded patches feed each leaf of each tree.
    n_out, n_leaves, kh, kw, stride, padding, cin:
        Convolution / tree geometry (mirrors
        :class:`~.packed_layer.BitpackedConvTreeLayer`).
    device:
        CUDA device string, e.g. ``"cuda:0"``.
    """

    def __init__(
        self,
        level_groups: List[List[Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]]],
        leaf_idx: torch.Tensor,
        n_out: int,
        n_leaves: int,
        kh: int,
        kw: int,
        stride: int,
        padding: int,
        cin: int,
        device: str,
    ) -> None:
        self.n_out    = n_out
        self.n_leaves = n_leaves
        self.kh       = kh
        self.kw       = kw
        self.stride   = stride
        self.padding  = padding
        self.cin      = cin
        self.device   = device
        self._level_groups = level_groups   # per-level, per-gate: (g, ai, bi, oi)
        self.leaf_idx      = leaf_idx       # [n_out * n_leaves] int64

    @classmethod
    def from_cpu_layer(
        cls,
        cpu_layer: BitpackedConvTreeLayer,
        device: str,
    ) -> "BitpackedConvGPULayer":
        """Build from an already-converted :class:`~.packed_layer.BitpackedConvTreeLayer`.

        Translates CPU ``(out_i, node_i)`` gate groups to the flat row indices
        used by :func:`_bitpacked_conv_gate_kernel`.

        The mapping is:

        * ``a_row[k]  = out_i[k] * n_nodes_current + 2 * node_i[k]``
        * ``b_row[k]  = out_i[k] * n_nodes_current + 2 * node_i[k] + 1``
        * ``out_row[k] = out_i[k] * n_nodes_next    + node_i[k]``

        where ``n_nodes_current = n_leaves >> level_idx`` (the number of nodes
        at the level currently being consumed) and ``n_nodes_next = n_nodes_current >> 1``.
        """
        n_out    = cpu_layer.n_out
        n_leaves = cpu_layer.n_leaves

        leaf_idx = torch.from_numpy(
            cpu_layer.leaf_idx.reshape(-1).astype(np.int64)
        ).to(device)

        level_groups: List[List[Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]]] = []

        for level_idx, groups in enumerate(cpu_layer._level_groups):
            n_nodes_current = n_leaves >> level_idx   # nodes consumed at this level
            n_nodes_next    = n_nodes_current >> 1    # nodes produced (output of this level)

            gpu_groups: List[Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]] = []
            for g, out_i, node_i in groups:
                # Flat row indices into [n_out * n_nodes, L, nw_B] tensors
                a_row   = (out_i * n_nodes_current + 2 * node_i    ).astype(np.int32)
                b_row   = (out_i * n_nodes_current + 2 * node_i + 1).astype(np.int32)
                out_row = (out_i * n_nodes_next    +     node_i    ).astype(np.int32)

                gpu_groups.append((
                    g,
                    torch.from_numpy(a_row).to(device),
                    torch.from_numpy(b_row).to(device),
                    torch.from_numpy(out_row).to(device),
                ))
            level_groups.append(gpu_groups)

        return cls(
            level_groups=level_groups,
            leaf_idx=leaf_idx,
            n_out=n_out,
            n_leaves=n_leaves,
            kh=cpu_layer.kh,
            kw=cpu_layer.kw,
            stride=cpu_layer.stride,
            padding=cpu_layer.padding,
            cin=cpu_layer.cin,
            device=device,
        )

    def forward_packed(self, x_3d: torch.Tensor) -> torch.Tensor:
        """Evaluate the gate tree on B-packed unrolled input.

        Parameters
        ----------
        x_3d:
            ``[P, L, nw_B]`` int64 on device — produced by
            :func:`~.packed_model.packed_unfold_gpu`.

        Returns
        -------
        ``[n_out, L, nw_B]`` int64 on device.
        """
        _P, L, nw_B = x_3d.shape

        # Gather leaf activations: [n_out * n_leaves, L, nw_B]
        current = x_3d[self.leaf_idx]

        for level_idx, gpu_groups in enumerate(self._level_groups):
            n_nodes_current = self.n_leaves >> level_idx
            n_nodes_next    = n_nodes_current >> 1
            n_out_rows      = self.n_out * n_nodes_next

            result = torch.empty(n_out_rows, L, nw_B, dtype=torch.int64, device=self.device)

            for g, ai, bi, oi in gpu_groups:
                n_g  = len(oi)
                grid = (
                    triton.cdiv(n_g,   BLOCK_G),
                    L,
                    triton.cdiv(nw_B,  BLOCK_W),
                )
                _bitpacked_conv_gate_kernel[grid](
                    current, result, ai, bi, oi,
                    n_g, L, nw_B,
                    gate=g,
                    BLOCK_G=BLOCK_G,
                    BLOCK_W=BLOCK_W,
                )

            current = result

        # After the final level: [n_out * 1, L, nw_B] == [n_out, L, nw_B]
        return current
