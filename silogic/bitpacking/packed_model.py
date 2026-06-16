"""End-to-end bitpacked inference pipelines.

:class:`BitpackedNet` mirrors :class:`~silogic.models.LogicNet` and accepts
``[B, in_dim]`` uint8 inputs.

:class:`BitpackedConvNet` mirrors :class:`~silogic.models.LogicConvNet` and
accepts ``[B, C, H, W]`` uint8 inputs.  Gate16 conv blocks run in the B-packed
``[dim, L, nw_B]`` domain (see :mod:`.packed_layer`) end-to-end — only the
raw image is ever expanded; no inter-layer unpack/repack is needed.

Both classes return ``[B, num_classes]`` float32 logits from :meth:`forward`.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Union

import numpy as np
import torch

from .ops import pack_bits, unpack_bits, mask_last_word, n_words
from .packed_layer import (
    BitpackedFCLayer,
    BitpackedLUTLayer,
    BitpackedConvTreeLayer,
    pack_conv_input,
    unpack_conv_output,
    packed_unfold,
    packed_or_pool,
)


Layer = Union[BitpackedFCLayer, BitpackedLUTLayer, None]


class BitpackedNet:
    """Bitpacked inference pipeline for a :class:`~silogic.models.LogicNet`.

    Layers whose connectome is unsupported (``SumThresholdConnectome``) fall
    back to PyTorch ``forward_hard``; their uint8 output is re-packed before
    the next bitpacked layer.

    Parameters
    ----------
    packed_layers:
        List of :class:`~.packed_layer.BitpackedFCLayer` /
        :class:`~.packed_layer.BitpackedLUTLayer` or ``None`` (for fallback
        layers).
    head:
        A :class:`~.packed_layer.BitpackedGroupSumHead` or
        :class:`~.packed_layer.BitpackedLearnedHead`.
    wire_r:
        Number of wire-residual pass-through neurons per layer (matches the
        ``wire_r`` attribute of the source :class:`~silogic.models.LogicNet`).
    fallback_torch_layers:
        ``{layer_index: torch_layer}`` for layers that could not be converted.
    device:
        PyTorch device used for fallback layers.
    """

    def __init__(
        self,
        packed_layers: List[Layer],
        head,
        wire_r: int = 0,
        fallback_torch_layers: Optional[Dict[int, object]] = None,
        device: str = "cpu",
    ) -> None:
        self.packed_layers = packed_layers
        self.head = head
        self.wire_r = wire_r
        self.fallback = fallback_torch_layers or {}
        self.device = device
        self.num_layers = len(packed_layers)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Run bitpacked inference.

        Parameters
        ----------
        x:
            ``[B, in_dim]`` uint8 (values in {0, 1}).

        Returns
        -------
        logits:
            ``[B, num_classes]`` float32.
        """
        B = x.shape[0]
        packed = pack_bits(x)              # [in_dim, n_words] int64

        for i, layer in enumerate(self.packed_layers):
            if layer is None:
                # Fallback: unpack → torch uint8 → forward_hard → re-pack
                packed = self._fallback_layer(i, packed, B)
            else:
                prev_packed = packed
                packed = layer.forward(packed)
                if self.wire_r and prev_packed.shape[0] == packed.shape[0]:
                    packed[:self.wire_r] = prev_packed[:self.wire_r]

        mask_last_word(packed, B)
        return self.head.forward(packed, B)

    def _fallback_layer(self, idx: int, packed: np.ndarray, B: int) -> np.ndarray:
        bits = unpack_bits(packed, B)          # [B, dim] uint8
        x_t = torch.from_numpy(bits).to(self.device)
        with torch.no_grad():
            out = self.fallback[idx].forward_hard(x_t)   # [B, out] uint8
        return pack_bits(out.cpu().numpy())

    # Convenience: accept a torch tensor as well
    def __call__(self, x) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        return self.forward(x)

    # Summary ----------------------------------------------------------------
    def layer_summary(self) -> list:
        rows = []
        for i, layer in enumerate(self.packed_layers):
            if layer is None:
                rows.append({"layer": i, "mode": "uint8-fallback"})
            elif isinstance(layer, BitpackedFCLayer):
                hist = layer.gate_histogram
                top = sorted(hist.items(), key=lambda kv: -kv[1])[:3]
                rows.append({"layer": i, "mode": "gate16", "out_dim": layer.out_dim,
                             "n_gate_types": len(hist), "top_gates": top})
            else:
                rows.append({"layer": i, "mode": "lut",
                             "out_dim": layer.out_dim, "arity": layer.arity})
        return rows


class BitpackedConvNet:
    """Bitpacked inference for :class:`~silogic.models.LogicConvNet`.

    Gate16 ``ConvLogicTree`` blocks run entirely in the B-packed
    ``[dim, L, nw_B]`` domain (see :mod:`.packed_layer`):

    1. Input is packed **once**: ``[B, C, H, W] → [C, L, nw_B]``.
    2. Each block: packed-unfold → gate-tree → OR-pool, all without touching B.
    3. ``[n_out, L_final, nw_B]`` is reshaped to ``[feat_dim, nw_B]`` and fed
       directly to the bitpacked FC head — **no additional repack**.

    Non-gate16 conv blocks fall back to PyTorch uint8 ``forward_hard`` and
    re-enter the packed domain afterwards.

    Parameters
    ----------
    conv_blocks:
        Original ``model.blocks`` list (``[conv0, pool0, conv1, pool1, …]``).
    bp_conv_layers:
        One :class:`~.packed_layer.BitpackedConvTreeLayer` (or ``None``) per
        ``(conv, pool)`` pair — ``None`` means uint8 fallback for that block.
    packed_head_layers:
        Bitpacked FC layers for the dense head (``None`` → uint8 fallback).
    head:
        :class:`~.packed_layer.BitpackedGroupSumHead` or
        :class:`~.packed_layer.BitpackedLearnedHead`.
    fallback_head_torch:
        ``{head_layer_index: torch_layer}`` for FC head layers that could not
        be converted.
    wire_residual:
        Fraction of channels to wire-residual through each conv block (matches
        ``model.wire_residual``).
    """

    def __init__(
        self,
        conv_blocks,
        bp_conv_layers: Optional[List],
        packed_head_layers: List[Layer],
        head,
        fallback_head_torch: Optional[Dict[int, object]] = None,
        wire_residual: float = 0.0,
        device: str = "cpu",
    ) -> None:
        self.conv_blocks      = conv_blocks
        self.bp_conv_layers   = bp_conv_layers or []
        self.packed_head_layers = packed_head_layers
        self.head             = head
        self.fallback_head    = fallback_head_torch or {}
        self.wire_residual    = wire_residual
        self.device           = device

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        x:
            ``[B, C, H, W]`` uint8 for image input, or ``[B, in_dim]`` uint8
            for already-flattened features.

        Returns
        -------
        logits:
            ``[B, num_classes]`` float32.
        """
        B = x.shape[0]
        nw_B = n_words(B)

        if x.ndim == 4:
            # ── spatial conv pipeline (B-packing: [dim, L, nw_B] int64) ────
            _, C_in, H, W = x.shape

            # Pack the raw image once: [C, H*W, nw_B]
            x_3d  = pack_conv_input(x, nw_B)
            cur_H = H
            cur_W = W

            for blk_idx, bp_conv in enumerate(self.bp_conv_layers):
                conv = self.conv_blocks[blk_idx * 2]
                pool = self.conv_blocks[blk_idx * 2 + 1]
                x_3d_in = x_3d

                if bp_conv is not None:
                    # Packed unfold: [C, L, nw_B] → [P, L_out, nw_B]
                    x_unfolded = packed_unfold(
                        x_3d, bp_conv.kh, bp_conv.kw,
                        bp_conv.stride, bp_conv.padding, cur_H, cur_W,
                    )
                    Ho = (cur_H + 2 * bp_conv.padding - bp_conv.kh) // bp_conv.stride + 1
                    Wo = (cur_W + 2 * bp_conv.padding - bp_conv.kw) // bp_conv.stride + 1

                    # Gate tree: [P, L_out, nw_B] → [n_out, L_out, nw_B]
                    y_3d = bp_conv.forward_packed(x_unfolded)

                    # Wire residual (same-spatial, channel-wise copy)
                    if self.wire_residual > 0:
                        r = min(int(conv.n * self.wire_residual), x_3d_in.shape[0])
                        if r > 0:
                            y_3d[:r] = x_3d_in[:r, : Ho * Wo, :]

                    # OR-pool in packed domain: [n_out, L_out, nw_B] → [n_out, L_pool, nw_B]
                    ps   = pool.size
                    x_3d = packed_or_pool(y_3d, Ho, Wo, ps)
                    cur_H = Ho // ps
                    cur_W = Wo // ps

                else:
                    # Fallback: unpack → torch uint8 forward → repack
                    x_uint8 = unpack_conv_output(x_3d, B, cur_H, cur_W)
                    x_t     = torch.from_numpy(x_uint8).to(self.device)
                    y       = conv.forward_hard(x_t)

                    if self.wire_residual > 0:
                        r = min(int(conv.n * self.wire_residual), x_t.shape[1])
                        if r > 0:
                            y = torch.cat([x_t[:, :r], y[:, r:]], dim=1)

                    pooled  = pool.forward_hard(y).cpu().numpy()  # [B, n_out, Ho, Wo]
                    cur_H   = pooled.shape[2]
                    cur_W   = pooled.shape[3]
                    x_3d    = pack_conv_input(pooled, nw_B)       # [n_out, L, nw_B]

            # Flatten: [n_out, L_final, nw_B] → [feat_dim, nw_B]
            n_out_final, L_final, _ = x_3d.shape
            packed = np.ascontiguousarray(
                x_3d.reshape(n_out_final * L_final, nw_B)
            )

        else:
            # Already-flat input: standard FC bitpacked path
            packed = pack_bits(x)

        for j, layer in enumerate(self.packed_head_layers):
            if layer is None:
                bits = unpack_bits(packed, B)
                inp  = torch.from_numpy(bits).to(self.device)
                with torch.no_grad():
                    out = self.fallback_head[j].forward_hard(inp)
                packed = pack_bits(out.cpu().numpy())
            else:
                packed = layer.forward(packed)

        mask_last_word(packed, B)
        return self.head.forward(packed, B)

    def __call__(self, x) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        return self.forward(x)


# ── GPU spatial helpers (torch ops on int64, no Triton required) ──────────────

def packed_unfold_gpu(
    x_3d: torch.Tensor,
    kh: int,
    kw: int,
    stride: int,
    padding: int,
    H_in: int,
    W_in: int,
) -> torch.Tensor:
    """Spatial unfold of B-packed activations on GPU (stays in int64 domain).

    Parameters
    ----------
    x_3d:
        ``[C, L_in, nw_B]`` int64 on CUDA.
    kh, kw, stride, padding:
        Convolution kernel parameters.
    H_in, W_in:
        Input spatial dimensions.

    Returns
    -------
    ``[C*kh*kw, L_out, nw_B]`` int64 on the same device.
    """
    C, _L_in, nw_B = x_3d.shape
    device = x_3d.device
    L_in  = H_in * W_in
    H_out = (H_in + 2 * padding - kh) // stride + 1
    W_out = (W_in + 2 * padding - kw) // stride + 1
    L_out = H_out * W_out
    P     = C * kh * kw

    # Build index arrays on CPU
    hy_out = torch.arange(L_out, dtype=torch.int32) // W_out
    hx_out = torch.arange(L_out, dtype=torch.int32) % W_out

    c_arr  = torch.arange(C,  dtype=torch.int32).repeat_interleave(kh * kw)   # [P]
    dy_arr = torch.arange(kh, dtype=torch.int32).repeat_interleave(kw).repeat(C)
    dx_arr = torch.arange(kw, dtype=torch.int32).repeat(C * kh)

    hy_in = hy_out.unsqueeze(0) * stride - padding + dy_arr.unsqueeze(1)  # [P, L_out]
    hx_in = hx_out.unsqueeze(0) * stride - padding + dx_arr.unsqueeze(1)

    valid = (hy_in >= 0) & (hy_in < H_in) & (hx_in >= 0) & (hx_in < W_in)
    # Sentinel L_in → zero-padding row appended to x_3d
    l_in = torch.where(valid, hy_in * W_in + hx_in, torch.tensor(L_in, dtype=torch.int32))

    c_gpu = c_arr.to(device)           # [P]
    l_gpu = l_in.long().to(device)     # [P, L_out]

    # Append one all-zero row for out-of-bounds positions
    x_pad = torch.cat(
        [x_3d, x_3d.new_zeros(C, 1, nw_B)], dim=1
    )                                  # [C, L_in+1, nw_B]

    return x_pad[c_gpu[:, None], l_gpu, :]  # [P, L_out, nw_B]


def packed_or_pool_gpu(
    x_3d: torch.Tensor,
    H: int,
    W: int,
    pool_size: int = 2,
) -> torch.Tensor:
    """OR-pool on B-packed int64 activations, staying on GPU.

    Parameters
    ----------
    x_3d:
        ``[C, H*W, nw_B]`` int64 on CUDA.
    H, W:
        Spatial dimensions of the input.
    pool_size:
        Pooling window side length (square; default 2).

    Returns
    -------
    ``[C, (H//ps)*(W//ps), nw_B]`` int64 on the same device.
    """
    C, _L, nw_B = x_3d.shape
    ps    = pool_size
    H_out = H // ps
    W_out = W // ps
    L_out = H_out * W_out

    hy_out = torch.arange(L_out, dtype=torch.int64)
    hx_out = hy_out % W_out
    hy_out = hy_out // W_out

    result = x_3d.new_zeros(C, L_out, nw_B)
    for dy in range(ps):
        for dx in range(ps):
            l_in = (hy_out * ps + dy) * W + (hx_out * ps + dx)
            result |= x_3d[:, l_in.to(x_3d.device), :]

    return result


# ── GPU conv inference pipeline ───────────────────────────────────────────────

class BitpackedConvGPUNet:
    """Triton-accelerated bitpacked inference for :class:`~silogic.models.LogicConvNet`.

    Runs the full B-packed conv pipeline on GPU:

    1. Pack the batch dimension on CPU (once).
    2. Transfer to GPU.
    3. For each conv block:

       a. :func:`packed_unfold_gpu` — spatial unfold without expanding B.
       b. :class:`~.kernels.triton_conv_layer.BitpackedConvGPULayer` — Triton gate tree.
       c. :func:`packed_or_pool_gpu` — OR-pool.

    4. FC head layers via :class:`~.kernels.triton_layer.BitpackedGPULayer`.
    5. Final head computation on CPU.

    Build from a CPU :class:`BitpackedConvNet` via :meth:`from_conv_net`.

    Parameters
    ----------
    gpu_conv_layers:
        One :class:`~.kernels.triton_conv_layer.BitpackedConvGPULayer` (or
        ``None`` for LUT-tree / unsupported fallback) per ``(conv, pool)`` pair.
    gpu_head_layers:
        One :class:`~.kernels.triton_layer.BitpackedGPULayer` (or ``None``)
        per FC head layer.
    bp_conv_net:
        Source CPU :class:`BitpackedConvNet` (provides fallback layers and
        wire-residual ratio).
    head:
        :class:`~.packed_layer.BitpackedGroupSumHead` or
        :class:`~.packed_layer.BitpackedLearnedHead`.
    block_spatial:
        ``[(cur_H, cur_W, Ho, Wo, ps), …]`` — geometry per block, precomputed
        by :meth:`from_conv_net`.
    device:
        CUDA device string.
    """

    def __init__(
        self,
        gpu_conv_layers: list,
        gpu_head_layers: list,
        bp_conv_net: "BitpackedConvNet",
        head,
        block_spatial: list,
        device: str = "cuda",
    ) -> None:
        self.gpu_conv_layers = gpu_conv_layers
        self.gpu_head_layers = gpu_head_layers
        self.bp_conv_net     = bp_conv_net
        self.head            = head
        self.block_spatial   = block_spatial
        self.device          = device

    @classmethod
    def from_conv_net(
        cls,
        bp_conv_net: "BitpackedConvNet",
        H_in: int,
        W_in: int,
        device: str = "cuda",
    ) -> "BitpackedConvGPUNet":
        """Build a GPU inference pipeline from a CPU :class:`BitpackedConvNet`.

        Parameters
        ----------
        bp_conv_net:
            Already-converted CPU net (:func:`~.convert.convert_logic_conv_net`).
        H_in, W_in:
            Input image spatial dimensions (e.g. 32, 32 for CIFAR-10).
        device:
            CUDA device string.

        Raises
        ------
        RuntimeError
            If Triton is not installed.
        """
        from .kernels import HAS_TRITON, BitpackedConvGPULayer, BitpackedGPULayer
        from .packed_layer import BitpackedConvTreeLayer, BitpackedFCLayer

        if not HAS_TRITON:
            raise RuntimeError(
                "Triton is required for BitpackedConvGPUNet. "
                "Install with: pip install triton"
            )

        # ── conv blocks ───────────────────────────────────────────────────────
        gpu_conv_layers: list = []
        block_spatial: list   = []
        cur_H, cur_W = H_in, W_in

        for blk_idx, bp_conv in enumerate(bp_conv_net.bp_conv_layers):
            pool = bp_conv_net.conv_blocks[blk_idx * 2 + 1]
            ps   = pool.size

            if bp_conv is not None:
                Ho = (cur_H + 2 * bp_conv.padding - bp_conv.kh) // bp_conv.stride + 1
                Wo = (cur_W + 2 * bp_conv.padding - bp_conv.kw) // bp_conv.stride + 1
            else:
                conv = bp_conv_net.conv_blocks[blk_idx * 2]
                Ho   = (cur_H + 2 * conv.padding - conv.kh) // conv.stride + 1
                Wo   = (cur_W + 2 * conv.padding - conv.kw) // conv.stride + 1

            block_spatial.append((cur_H, cur_W, Ho, Wo, ps))

            if isinstance(bp_conv, BitpackedConvTreeLayer):
                gpu_conv_layers.append(
                    BitpackedConvGPULayer.from_cpu_layer(bp_conv, device)
                )
            else:
                gpu_conv_layers.append(None)

            cur_H = Ho // ps
            cur_W = Wo // ps

        # ── FC head layers ────────────────────────────────────────────────────
        gpu_head_layers: list = []
        for layer in bp_conv_net.packed_head_layers:
            if isinstance(layer, BitpackedFCLayer):
                gpu_head_layers.append(
                    BitpackedGPULayer.from_fc_layer(layer, device)
                )
            else:
                gpu_head_layers.append(None)

        return cls(
            gpu_conv_layers=gpu_conv_layers,
            gpu_head_layers=gpu_head_layers,
            bp_conv_net=bp_conv_net,
            head=bp_conv_net.head,
            block_spatial=block_spatial,
            device=device,
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Run GPU bitpacked inference.

        Parameters
        ----------
        x:
            ``[B, C, H, W]`` uint8 (values in {0, 1}).

        Returns
        -------
        logits:
            ``[B, num_classes]`` float32.
        """
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()

        B    = x.shape[0]
        nw_B = n_words(B)

        # Pack batch on CPU, transfer to GPU once
        x_3d_np = pack_conv_input(x, nw_B)                      # [C, L, nw_B]
        x_3d    = torch.from_numpy(x_3d_np).to(self.device)

        for blk_idx, gpu_conv in enumerate(self.gpu_conv_layers):
            cur_H, cur_W, Ho, Wo, ps = self.block_spatial[blk_idx]
            x_3d_in = x_3d

            if gpu_conv is not None:
                # ── fully GPU path ────────────────────────────────────────
                # Unfold: [C, L_in, nw_B] → [P, L_out, nw_B]
                x_unfolded = packed_unfold_gpu(
                    x_3d, gpu_conv.kh, gpu_conv.kw,
                    gpu_conv.stride, gpu_conv.padding, cur_H, cur_W,
                )
                # Gate tree: [P, L_out, nw_B] → [n_out, L_out, nw_B]
                y_3d = gpu_conv.forward_packed(x_unfolded)

                wire_r = self.bp_conv_net.wire_residual
                if wire_r > 0:
                    conv_block = self.bp_conv_net.conv_blocks[blk_idx * 2]
                    r = min(int(conv_block.n * wire_r), x_3d_in.shape[0])
                    if r > 0:
                        y_3d[:r] = x_3d_in[:r, :Ho * Wo, :]

                # OR-pool: [n_out, L_out, nw_B] → [n_out, L_pool, nw_B]
                x_3d = packed_or_pool_gpu(y_3d, Ho, Wo, ps)

            else:
                # ── CPU fallback (LUT-tree / unsupported blocks) ──────────
                x_3d_np_in  = x_3d.cpu().numpy()
                bp_conv_cpu = self.bp_conv_net.bp_conv_layers[blk_idx]
                conv_block  = self.bp_conv_net.conv_blocks[blk_idx * 2]
                pool_block  = self.bp_conv_net.conv_blocks[blk_idx * 2 + 1]

                if bp_conv_cpu is not None:
                    x_uf_np = packed_unfold(
                        x_3d_np_in, bp_conv_cpu.kh, bp_conv_cpu.kw,
                        bp_conv_cpu.stride, bp_conv_cpu.padding, cur_H, cur_W,
                    )
                    y_np = bp_conv_cpu.forward_packed(x_uf_np)

                    wire_r = self.bp_conv_net.wire_residual
                    if wire_r > 0:
                        r = min(int(conv_block.n * wire_r), x_3d_np_in.shape[0])
                        if r > 0:
                            y_np[:r] = x_3d_np_in[:r, :Ho * Wo, :]

                    x_np_pool = packed_or_pool(y_np, Ho, Wo, ps)
                else:
                    x_uint8 = unpack_conv_output(x_3d_np_in, B, cur_H, cur_W)
                    x_t     = torch.from_numpy(x_uint8).to(self.device)
                    y       = conv_block.forward_hard(x_t)

                    wire_r = self.bp_conv_net.wire_residual
                    if wire_r > 0:
                        r = min(int(conv_block.n * wire_r), x_t.shape[1])
                        if r > 0:
                            y = torch.cat([x_t[:, :r], y[:, r:]], dim=1)

                    pooled    = pool_block.forward_hard(y).cpu().numpy()
                    x_np_pool = pack_conv_input(pooled, nw_B)

                x_3d = torch.from_numpy(x_np_pool).to(self.device)

        # Flatten: [n_out, L_final, nw_B] → [feat_dim, nw_B]
        n_out_f, L_f, _ = x_3d.shape
        packed_gpu = x_3d.reshape(n_out_f * L_f, nw_B).contiguous()

        # FC head layers on GPU
        for j, gl in enumerate(self.gpu_head_layers):
            if gl is not None:
                packed_gpu = gl.forward(packed_gpu)
            else:
                packed_np_j = packed_gpu.cpu().numpy()
                bits        = unpack_bits(packed_np_j, B)
                inp         = torch.from_numpy(bits).to(self.device)
                with torch.no_grad():
                    out = self.bp_conv_net.fallback_head[j].forward_hard(inp)
                packed_gpu = torch.from_numpy(
                    pack_bits(out.cpu().numpy())
                ).to(self.device)

        # Final head on CPU
        packed_np = packed_gpu.cpu().numpy()
        mask_last_word(packed_np, B)
        return self.head.forward(packed_np, B)

    def __call__(self, x) -> np.ndarray:
        return self.forward(x)
