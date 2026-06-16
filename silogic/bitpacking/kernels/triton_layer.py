"""Triton GPU kernel for bitpacked FC-layer inference.

Design: gate-grouped dispatch.  All outputs sharing the same gate type are
processed in one kernel launch.  Because the gate is a ``tl.constexpr``,
Triton JIT-compiles a specialised kernel for each of the 16 gate types and
the dead branches are eliminated at compile time — no warp divergence.

Memory layout: ``[dim, n_words]`` int64 stored as CUDA int64 tensors.
Grid: ``(ceil(n_group / BLOCK_OUT), ceil(n_words / BLOCK_WORDS))``.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
import triton
import triton.language as tl


# ── kernel ────────────────────────────────────────────────────────────────────

@triton.jit
def _bitpacked_gate_kernel(
    x_ptr,          # [in_dim, n_words] int64
    out_ptr,        # [out_dim, n_words] int64 — full output buffer
    a_idx_ptr,      # [n_g] int32
    b_idx_ptr,      # [n_g] int32
    out_idx_ptr,    # [n_g] int32 — row indices to scatter-write into out_ptr
    n_g,            # int — number of outputs in this gate group
    n_words,        # int — number of int64 words per activation
    gate: tl.constexpr,         # 0-15 — JIT-specialised (no warp divergence)
    BLOCK_OUT: tl.constexpr,
    BLOCK_WORDS: tl.constexpr,
):
    pid_out  = tl.program_id(0)
    pid_word = tl.program_id(1)

    out_offs  = pid_out  * BLOCK_OUT   + tl.arange(0, BLOCK_OUT)
    word_offs = pid_word * BLOCK_WORDS + tl.arange(0, BLOCK_WORDS)

    out_mask  = out_offs  < n_g
    word_mask = word_offs < n_words

    # Load gate input indices and the scatter-target row indices
    ai      = tl.load(a_idx_ptr   + out_offs, mask=out_mask, other=0).to(tl.int64)
    bi      = tl.load(b_idx_ptr   + out_offs, mask=out_mask, other=0).to(tl.int64)
    out_row = tl.load(out_idx_ptr + out_offs, mask=out_mask, other=0).to(tl.int64)

    a = tl.load(
        x_ptr + ai[:, None] * n_words + word_offs[None, :],
        mask=out_mask[:, None] & word_mask[None, :], other=0,
    )
    b = tl.load(
        x_ptr + bi[:, None] * n_words + word_offs[None, :],
        mask=out_mask[:, None] & word_mask[None, :], other=0,
    )

    # Gate dispatch — compile-time constant; dead branches removed by JIT.
    if gate == 0:
        r = tl.zeros([BLOCK_OUT, BLOCK_WORDS], dtype=tl.int64)
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
    else:
        # gate == 15: True — all bits set; int64(-1) = 0xFFFFFFFFFFFFFFFF
        r = tl.full([BLOCK_OUT, BLOCK_WORDS], value=-1, dtype=tl.int64)

    # Scatter-store directly into the correct rows of the output buffer
    tl.store(
        out_ptr + out_row[:, None] * n_words + word_offs[None, :],
        r,
        mask=out_mask[:, None] & word_mask[None, :],
    )


# ── Python-side layer ─────────────────────────────────────────────────────────

BLOCK_OUT   = 64
BLOCK_WORDS = 16


class BitpackedGPULayer:
    """GPU bitpacked inference for one logic layer.

    Wraps :class:`~silogic.bitpacking.packed_layer.BitpackedFCLayer` gate
    groups and dispatches one Triton kernel per gate type (gate-grouped SIMD).

    Parameters
    ----------
    gate_types:
        ``[out_dim]`` int32 — gate index 0-15 for each output neuron.
    a_idx, b_idx:
        ``[out_dim]`` int32 — input indices.
    device:
        CUDA device string, e.g. ``"cuda:0"``.
    """

    def __init__(
        self,
        gate_types: np.ndarray,
        a_idx: np.ndarray,
        b_idx: np.ndarray,
        device: str = "cuda",
    ) -> None:
        self.out_dim = len(gate_types)
        self.device = device

        # Build one CUDA tensor group per active gate type.
        # Within each group, sort by (ai, bi) for sequential reads (cache-friendly).
        self._groups: List[Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for g in range(16):
            mask = gate_types == g
            if not mask.any():
                continue
            out_idx_np = np.where(mask)[0].astype(np.int32)
            ai_np = a_idx[mask].astype(np.int32)
            bi_np = b_idx[mask].astype(np.int32)
            order = np.lexsort((bi_np, ai_np))
            out_idx = torch.from_numpy(out_idx_np[order]).to(device)
            ai = torch.from_numpy(ai_np[order]).to(device)
            bi = torch.from_numpy(bi_np[order]).to(device)
            self._groups.append((g, out_idx, ai, bi))

    @classmethod
    def from_fc_layer(cls, fc_layer, device: str = "cuda") -> "BitpackedGPULayer":
        """Build from an already-converted :class:`~.packed_layer.BitpackedFCLayer`."""
        return cls(fc_layer.gate_types, fc_layer.a_idx, fc_layer.b_idx, device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x:
            ``[in_dim, n_words]`` int64 CUDA tensor.

        Returns
        -------
        out:
            ``[out_dim, n_words]`` int64 CUDA tensor.
        """
        in_dim, nw = x.shape
        out = torch.empty(self.out_dim, nw, dtype=torch.int64, device=self.device)

        for g, out_idx, ai, bi in self._groups:
            n_g = len(out_idx)
            grid = (triton.cdiv(n_g, BLOCK_OUT), triton.cdiv(nw, BLOCK_WORDS))
            # Kernel scatter-stores directly to the correct rows of `out`
            _bitpacked_gate_kernel[grid](
                x, out, ai, bi, out_idx,
                n_g, nw,
                gate=g,
                BLOCK_OUT=BLOCK_OUT,
                BLOCK_WORDS=BLOCK_WORDS,
            )

        return out

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)
