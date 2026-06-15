"""Attention-like logic layer (optionally multi-head).

Standard attention scores a query against every key with a dot product, weights,
and sums. Here we replace the dot-product interaction with a *learnable logic
gate*: each output o has a query input q_o (soft-selected from the inputs) that
is combined with EVERY input x_j through a 2-input logic gate G_o, the pairwise
results weighted by a learned (ternary) attention pattern w_oj and summed,
then thresholded:

    G_o(q_o, x_j) = c0 + c1 q_o + c2 x_j + c3 q_o x_j     (BasisProj of 16-gate mix)
    s_o = (1/sqrt n) sum_j  w_oj * G_o(q_o, x_j)
    y_o = threshold(s_o - theta_o)

Because the gate is MULTILINEAR, the O(in*out) pairwise tensor never has to be
materialised -- the sum factorises into two reductions:

    s_o ~ (c0 + c1 q_o) * (sum_j w_oj)  +  (c2 + c3 q_o) * (W x)_o

so a forward is a ternary matmul + reductions. With n_heads>1 each output gets H
independent (query, gate, weight-pattern) triples summed before the threshold,
restoring per-key gate diversity (one gate per output is shared across keys):

    s_o = (1/sqrt n) sum_h sum_j  w^h_oj * G^h_o(q^h_o, x_j)

each head factorises independently -> H matmuls, still no pairwise tensor.

The 1/sqrt(fan-in) scale is the logic analogue of attention's 1/sqrt(d_k): the
score sums ~in_dim ternary terms, so without it the magnitude grows with fan-in
and saturates the threshold's STE gradient (wide/deep nets collapsed). Static ->
folds into theta, free at inference.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .model import (BASIS_COEFFS, TRUTH_TABLES, ternary_ste, ste_threshold,
                    GroupSum)


class PairLogicLayer(nn.Module):
    def __init__(self, in_dim, out_dim, n_heads=1, cand_q=8, seed=None):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.H = n_heads
        gen = torch.Generator().manual_seed(seed) if seed is not None else None
        cq = min(cand_q, in_dim)
        self.cand_q = cq
        # query candidates shared across heads; per-head selection weights
        cand = torch.stack([torch.randperm(in_dim, generator=gen)[:cq]
                            for _ in range(out_dim)])              # [out, cq]
        self.register_buffer("cand", cand)
        self.sel_q = nn.Parameter(torch.randn(out_dim, n_heads, cq) * 0.1)
        # per-(output,head) logic gate (16-way mixture)
        self.gate = nn.Parameter(torch.randn(out_dim, n_heads, 16) * 0.1)
        # per-(output,head) attention pattern over ALL inputs, ternary {-1,0,+1}
        self.w = nn.Parameter(torch.randn(out_dim, n_heads, in_dim))
        self.theta = nn.Parameter(torch.zeros(out_dim))
        self.scale = 1.0 / (in_dim ** 0.5)
        self.register_buffer("basis", BASIS_COEFFS.clone())
        self.register_buffer("truth", TRUTH_TABLES.clone())

    def _coef(self):
        p = F.softmax(self.gate, dim=-1)                          # [out,H,16]
        return p @ self.basis                                    # [out,H,4]

    def _query_soft(self, x):
        a = F.softmax(self.sel_q, dim=-1)                        # [out,H,cq]
        g = x[:, self.cand]                                      # [B,out,cq]
        return torch.einsum("boc,ohc->boh", g, a)               # [B,out,H]

    def forward(self, x):
        B = x.shape[0]
        q = self._query_soft(x)                                  # [B,out,H]
        c = self._coef()                                         # [out,H,4]
        c0, c1, c2, c3 = c[..., 0], c[..., 1], c[..., 2], c[..., 3]  # [out,H]
        wt = ternary_ste(self.w)                                 # [out,H,in]
        wsum = wt.sum(dim=2)                                     # [out,H]
        wx = F.linear(x, wt.reshape(self.out_dim * self.H, self.in_dim))
        wx = wx.view(B, self.out_dim, self.H)                    # [B,out,H]
        term = (c0 + c1 * q) * wsum + (c2 + c3 * q) * wx         # [B,out,H]
        s = self.scale * term.sum(dim=2)                        # [B,out]
        return ste_threshold(s - self.theta)

    @torch.no_grad()
    def forward_hard(self, x):
        B = x.shape[0]
        sel = self.sel_q.argmax(dim=2)                           # [out,H]
        qidx = torch.gather(self.cand.unsqueeze(1).expand(-1, self.H, -1),
                            2, sel.unsqueeze(2)).squeeze(2)       # [out,H]
        q = x[:, qidx]                                           # [B,out,H]
        gsel = self.gate.argmax(dim=2)                           # [out,H]
        c = self.basis[gsel]                                     # [out,H,4]
        c0, c1, c2, c3 = c[..., 0], c[..., 1], c[..., 2], c[..., 3]
        wt = torch.where(self.w > 0.5, 1.0,
                         torch.where(self.w < -0.5, -1.0, 0.0))   # [out,H,in]
        wsum = wt.sum(dim=2)                                     # [out,H]
        wx = F.linear(x.float(), wt.reshape(self.out_dim * self.H, self.in_dim))
        wx = wx.view(B, self.out_dim, self.H)
        term = (c0 + c1 * q) * wsum + (c2 + c3 * q) * wx
        s = self.scale * term.sum(dim=2)
        return (s > self.theta).float()

    def fpga_cost(self):
        # per (output,head): in_dim-wide ternary popcount + 1 gate; + threshold
        return self.out_dim * self.H * self.in_dim


class PairLogicNet(nn.Module):
    """Stack of attention-like pairwise-logic layers + GroupSum head."""

    def __init__(self, in_dim, width, depth, num_classes=10, tau=10.0,
                 n_heads=1, cand_q=8, seed=0):
        super().__init__()
        assert width % num_classes == 0
        layers = []
        d = in_dim
        for i in range(depth):
            layers.append(PairLogicLayer(d, width, n_heads=n_heads,
                                         cand_q=cand_q, seed=seed * 1000 + i))
            d = width
        self.layers = nn.ModuleList(layers)
        self.head = GroupSum(num_classes, tau=tau)
        self.width = width; self.depth = depth; self.n_heads = n_heads

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return self.head(x)

    @torch.no_grad()
    def forward_hard(self, x):
        for l in self.layers:
            x = l.forward_hard(x)
        return self.head.forward_hard(x)

    def fpga_cost(self):
        return sum(l.fpga_cost() for l in self.layers)
