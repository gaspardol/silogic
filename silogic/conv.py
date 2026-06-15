"""Convolutional logic-gate-tree layers + OR pooling + residual init.

Reimplements the core ideas of "Convolutional Differentiable Logic Gate
Networks" (Petersen et al., arXiv:2411.04732):
  * Logic-gate *tree* convolution kernels (complete binary tree of depth d,
    2^d leaves selected from the receptive field, 2^d-1 learnable gates,
    parameters shared across spatial placements).
  * Logical OR pooling (max t-conorm = spatial max-pool).
  * Residual initialization (bias each gate toward pass-through 'A').

THE HYBRID (this work): the tree leaves are wired with LILogic's *learnable
Top-K connectivity* instead of fixed random connections, and gates are
evaluated with BasisProj. `connect="fixed"` recovers the Petersen baseline;
`connect="topk"` is the combined model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .model import BASIS_COEFFS, TRUTH_TABLES, gate_probs

try:
    from .triton_conv import tree_conv
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False

try:
    from .triton_conv_topk import tree_conv_topk
    _HAS_TRITON_TOPK = True
except Exception:
    _HAS_TRITON_TOPK = False

GATE_A = 3  # index of the 'A' (pass-through) gate in BASIS_COEFFS ordering


class ConvLogicTree(nn.Module):
    """Logic-gate-tree convolution layer (Petersen-style trees, LILogic wiring).

    Each output channel is a complete binary tree of depth ``tree_depth`` with
    ``2**tree_depth`` leaves drawn from the receptive field and
    ``2**tree_depth - 1`` learnable gates, shared across all spatial placements.

    Args:
        in_channels (int): Number of input channels ``C``.
        out_channels (int): Number of output channels ``n`` (one tree each).
        kernel (int): Square kernel side length (``kernel x kernel`` receptive
            field). Default ``3``.
        tree_depth (int): Depth ``d`` of each gate tree; gives ``2**d`` leaves
            and ``2**d - 1`` gates. Default ``2``.
        stride (int): Convolution stride. Default ``1``.
        padding (int): Zero-padding on each side. Default ``1``.
        connect (str): Leaf wiring mode. ``"topk"`` (default) uses learnable
            Top-K connectivity (softmax over ``k`` candidates per leaf);
            ``"fixed"`` uses one fixed random leaf index per position (Petersen
            baseline). Both have fused Triton kernels on CUDA (``tree_conv`` for
            fixed, ``tree_conv_topk`` for Top-K).
        k (int): Top-K candidate pool size per leaf when ``connect="topk"``
            (clamped to the per-tree pool ``n_chan * kernel * kernel``).
            Unused for ``"fixed"``. Default ``4``.
        n_chan (int): Number of distinct input channels each tree may observe;
            leaves are drawn only from these channels' receptive-field positions
            (paper uses ``2``). Default ``2``.
        residual_init (bool): If ``True`` (default), bias every gate toward the
            ``"A"`` pass-through gate (~90%) for residual initialization; else
            initialize gate logits with small Gaussian noise.
        seed (int): RNG seed for the random channel/leaf selection. Default ``0``.
        residual (bool): If ``True``, add a fixed structural skip at each tree
            node, ``out = XOR(gate(a, b), a)``; disables the Triton kernel.
            Default ``False``.
    """
    def __init__(self, in_channels, out_channels, kernel=3, tree_depth=2,
                 stride=1, padding=1, connect="topk", k=4, n_chan=2,
                 residual_init=True, seed=0, residual=False):
        super().__init__()
        self.cin = in_channels
        self.n = out_channels
        self.kh = self.kw = kernel
        self.d = tree_depth
        self.leaves = 2 ** tree_depth
        self.stride = stride
        self.padding = padding
        self.connect = connect
        self.k = k
        # structural residual: each tree node out = gate(a,b) XOR a (fixed skip)
        self.residual = residual
        gen = torch.Generator().manual_seed(seed)

        P = in_channels * kernel * kernel        # receptive-field size
        # Restrict each tree to observe only n_chan random input channels
        # (paper: trees see 2 channels). Candidate pool per leaf is drawn
        # from those channels' positions in the receptive field.
        cand_pool = []  # [n, pool] candidate flat indices into [0,P)
        pool_sz = n_chan * kernel * kernel
        for _ in range(out_channels):
            chans = torch.randperm(in_channels, generator=gen)[:n_chan]
            idxs = []
            for c in chans:
                base = int(c) * kernel * kernel
                idxs.extend(range(base, base + kernel * kernel))
            cand_pool.append(torch.tensor(idxs))
        cand_pool = torch.stack(cand_pool)        # [n, pool_sz]

        if connect == "fixed":
            # one fixed leaf index per (channel, leaf)
            sel = torch.stack([
                cand_pool[i][torch.randint(0, pool_sz, (self.leaves,),
                                           generator=gen)]
                for i in range(out_channels)])     # [n, leaves]
            self.register_buffer("leaf_idx", sel)
            # decompose flat receptive-field index -> (channel, dy, dx) for Triton
            kk2 = kernel * kernel
            self.register_buffer("cm_idx", (sel // kk2).to(torch.int32))
            rem = sel % kk2
            self.register_buffer("ch_idx", (rem // kernel).to(torch.int32))
            self.register_buffer("cw_idx", (rem % kernel).to(torch.int32))
            self.use_triton = _HAS_TRITON
        else:  # topk learnable
            kk = min(k, pool_sz)
            self.k = kk
            cand = torch.stack([
                torch.stack([cand_pool[i][torch.randperm(pool_sz, generator=gen)[:kk]]
                             for _ in range(self.leaves)])
                for i in range(out_channels)])     # [n, leaves, k]
            self.register_buffer("leaf_cand", cand)
            self.conn = nn.Parameter(torch.randn(out_channels, self.leaves, kk))
            # decompose candidate flat indices -> (channel, dy, dx) for the
            # fused Top-K Triton kernel (same layout as the fixed path)
            kk2 = kernel * kernel
            self.register_buffer("lc_cm", (cand // kk2).to(torch.int32))
            rem = cand % kk2
            self.register_buffer("lc_ch", (rem // kernel).to(torch.int32))
            self.register_buffer("lc_cw", (rem % kernel).to(torch.int32))
            self.use_triton_topk = _HAS_TRITON_TOPK

        # Gate logits per tree level. Level i combines 2^(d-i) -> 2^(d-i-1).
        self.gate_logits = nn.ParameterList()
        for i in range(tree_depth):
            nodes = 2 ** (tree_depth - 1 - i)
            w = torch.zeros(out_channels, nodes, 16)
            if residual_init:
                w[:, :, GATE_A] = 5.0              # ~90% 'A'
            else:
                w.normal_(0, 0.1)
            self.gate_logits.append(nn.Parameter(w))

        self.register_buffer("basis", BASIS_COEFFS.clone())
        self.register_buffer("truth", TRUTH_TABLES.clone())

    def _gather_leaves(self, patches):
        """patches [B, P, L] -> leaves [B, n, leaves, L]."""
        B, P, L = patches.shape
        if self.connect == "fixed":
            idx = self.leaf_idx.reshape(-1)                  # [n*leaves]
            g = patches[:, idx, :]                            # [B, n*leaves, L]
            return g.view(B, self.n, self.leaves, L)
        else:
            w = F.softmax(self.conn, dim=2)                  # [n, leaves, k]
            idx = self.leaf_cand.reshape(-1)                 # [n*leaves*k]
            g = patches[:, idx, :].view(B, self.n, self.leaves, self.k, L)
            return torch.einsum("bnlkL,nlk->bnlL", g, w)

    def _tree_soft(self, leaves):
        """leaves [B, n, leaves, L] -> [B, n, L] via BasisProj per level."""
        vals = leaves
        for i, gl in enumerate(self.gate_logits):
            p = gate_probs(gl, self.training, dim=2)          # [n, nodes, 16]
            c = p @ self.basis                               # [n, nodes, 4]
            a = vals[:, :, 0::2, :]
            b = vals[:, :, 1::2, :]
            c = c.unsqueeze(0).unsqueeze(-1)                 # [1,n,nodes,4,1]
            node = (c[:, :, :, 0] + c[:, :, :, 1] * a +
                    c[:, :, :, 2] * b + c[:, :, :, 3] * (a * b))
            if self.residual:                                # XOR(node, a) skip
                node = node + a - 2.0 * node * a
            vals = node
        return vals[:, :, 0, :]

    def _hw_out(self, H, W):
        Ho = (H + 2 * self.padding - self.kh) // self.stride + 1
        Wo = (W + 2 * self.padding - self.kw) // self.stride + 1
        return Ho, Wo

    def _basis_coef(self):
        """Concatenate per-level gate logits -> BasisProj coeffs [n, nodes, 4].

        Uses Gumbel straight-through gate selection when enabled+training
        (hard gate in forward, soft gradient), else plain softmax.
        """
        gl = torch.cat([g for g in self.gate_logits], dim=1)  # [n, nodes, 16]
        p = gate_probs(gl, self.training, dim=2)
        return p @ self.basis                                 # [n, nodes, 4]

    def forward(self, x):
        B, C, H, W = x.shape
        Ho, Wo = self._hw_out(H, W)
        if (self.connect == "fixed" and getattr(self, "use_triton", False)
                and x.is_cuda and not self.residual):
            coef = self._basis_coef().contiguous()
            return tree_conv(x, coef, self.cm_idx, self.ch_idx, self.cw_idx,
                             self.d, self.stride, self.padding, Ho, Wo)
        if (self.connect == "topk" and getattr(self, "use_triton_topk", False)
                and x.is_cuda and not self.residual):
            coef = self._basis_coef().contiguous()
            w = F.softmax(self.conn, dim=2).contiguous()
            return tree_conv_topk(x, coef, w, self.lc_cm, self.lc_ch, self.lc_cw,
                                  self.d, self.k, self.stride, self.padding, Ho, Wo)
        patches = F.unfold(x, (self.kh, self.kw), stride=self.stride,
                           padding=self.padding)             # [B, C*kh*kw, L]
        leaves = self._gather_leaves(patches)
        out = self._tree_soft(leaves)                        # [B, n, L]
        return out.view(B, self.n, Ho, Wo)

    @torch.no_grad()
    def forward_hard(self, x):
        B, C, H, W = x.shape
        x = x.to(torch.uint8)
        patches = F.unfold(x.float(), (self.kh, self.kw), stride=self.stride,
                           padding=self.padding).to(torch.uint8)
        # hard leaf selection
        if self.connect == "fixed":
            idx = self.leaf_idx.reshape(-1)
            g = patches[:, idx, :].view(B, self.n, self.leaves, -1)
        else:
            sel = self.conn.argmax(dim=2)                    # [n, leaves]
            idx = torch.gather(self.leaf_cand, 2,
                               sel.unsqueeze(-1)).squeeze(-1)  # [n, leaves]
            g = patches[:, idx.reshape(-1), :].view(B, self.n, self.leaves, -1)
        vals = g
        for gl in self.gate_logits:
            gate = gl.argmax(dim=2)                          # [n, nodes]
            a = vals[:, :, 0::2, :]; b = vals[:, :, 1::2, :]
            t_idx = (a.long() << 1) | b.long()               # [B,n,nodes,L]
            tt = self.truth.to(x.device)[gate]               # [n, nodes, 4]
            tt = tt.unsqueeze(0).expand(B, -1, -1, -1)       # [B,n,nodes,4]
            node = torch.gather(tt, 3, t_idx)                # [B,n,nodes,L]
            if self.residual:                                # XOR(node, a) skip
                node = node ^ a
            vals = node
        out = vals[:, :, 0, :]
        Ho, Wo = self._hw_out(H, W)
        return out.view(B, self.n, Ho, Wo).to(torch.uint8)


class OrPool(nn.Module):
    """Logical OR pooling = max t-conorm = spatial max-pool.

    Args:
        size (int): Pooling window and stride. Default ``2`` (halves H and W).
    """
    def __init__(self, size=2):
        super().__init__()
        self.size = size

    def forward(self, x):
        return F.max_pool2d(x, self.size, self.size)

    @torch.no_grad()
    def forward_hard(self, x):
        return F.max_pool2d(x.float(), self.size, self.size).to(torch.uint8)
