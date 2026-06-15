"""Unified differentiable LUT-node layer with multiple relaxations.

Following BitLogic (Bührer et al., arXiv:2602.07400), a *LUT node* is an n-input
Boolean function trained through a differentiable surrogate ``f`` and discretized
to an exact truth table for inference. The same node can be parameterized many
ways; this layer implements a registry of **boundary-consistent** relaxations
(``f`` recovers the hard LUT at the cube corners), selectable by ``relaxation``:

  * ``"probabilistic"`` — multilinear expectation over the input hypercube,
    ``f = sum_a sigmoid(theta_a) * prod_j a_j^{a_j}(1-a_j)^{1-a_j}`` (one logit per
    truth-table entry; the same form as :class:`~silogic.LUTkLayer`).
  * ``"hybrid"`` — DWN-style: the **discrete** forward ``sigmoid(theta)[idx(H(a))]``
    (hard-thresholded inputs, so the forward equals inference) with the
    probabilistic surrogate **gradient** (smooth backprop).
  * ``"linear"`` — LogicNets-style perceptron node ``sigmoid(theta0 + sum_j theta_j a_j)``
    (O(n) params; only linearly-separable functions).
  * ``"polynomial"`` — degree-``d`` multilinear polynomial
    ``sigmoid(sum_{|S|<=d} theta_S prod_{j in S} a_j)`` (O(n^d) params, implicit
    regularization for d < n).

All variants share LILogic Top-K input selection (``k`` learnable candidates per
input slot) and a node-family-agnostic residual initialization (``residual_p``).
"""
import math
from itertools import combinations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import GroupSum

RELAXATIONS = ("probabilistic", "hybrid", "linear", "polynomial")


def residual_logit(p, tau=1.0):
    """Logit that makes ``sigmoid`` output probability ``p`` (used for residual init)."""
    p = min(max(float(p), 1e-4), 1 - 1e-4)
    return tau * math.log(p / (1 - p))


class LUTNodeLayer(nn.Module):
    """A layer of n-input LUT nodes with a selectable differentiable relaxation.

    Args:
        in_dim (int): Number of input features.
        out_dim (int): Number of LUT nodes (outputs).
        arity (int): Inputs per node ``n``. Default ``4``.
        relaxation (str): Node parameterization; one of ``"probabilistic"``,
            ``"hybrid"``, ``"linear"``, ``"polynomial"``. Default ``"probabilistic"``.
        k (int): Top-K learnable candidates per input slot. Default ``8``.
        degree (int, optional): Polynomial degree for ``relaxation="polynomial"``
            (``None`` -> ``min(2, arity)``). Ignored otherwise.
        residual_p (float): If ``> 0``, residual-initialize each node toward
            passing through its first input with this probability (a node-agnostic
            noisy identity). ``0.0`` disables it. Default ``0.0``.
        seed (int, optional): RNG seed for the candidate wiring / init.
    """

    def __init__(self, in_dim, out_dim, arity=4, relaxation="probabilistic",
                 k=8, degree=None, residual_p=0.0, seed=None):
        super().__init__()
        if relaxation not in RELAXATIONS:
            raise ValueError(f"unknown relaxation {relaxation!r}; choose from {RELAXATIONS}")
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n = arity
        self.relaxation = relaxation
        gen = torch.Generator().manual_seed(seed) if seed is not None else None

        kk = min(k, in_dim)
        self.k = kk
        cand = torch.stack([torch.stack([torch.randperm(in_dim, generator=gen)[:kk]
                                         for _ in range(arity)]) for _ in range(out_dim)])
        self.register_buffer("cand", cand)               # [out, n, k]
        self.conn = nn.Parameter(torch.randn(out_dim, arity, kk, generator=gen))

        if relaxation in ("probabilistic", "hybrid"):
            P = 2 ** arity
            bits = torch.tensor([[(p >> j) & 1 for j in range(arity)] for p in range(P)],
                                dtype=torch.float32)
            self.register_buffer("corners", bits)        # [P, n]
            self.theta = nn.Parameter(torch.randn(out_dim, P) * 0.1)
        elif relaxation == "linear":
            self.theta = nn.Parameter(torch.randn(out_dim, arity + 1) * 0.1)  # [bias, w1..wn]
        else:  # polynomial
            d = min(2, arity) if degree is None else min(degree, arity)
            self.degree = d
            subs = [()]
            for r in range(1, d + 1):
                subs += list(combinations(range(arity), r))
            self._subs = subs
            mask = torch.zeros(len(subs), arity)
            for i, s in enumerate(subs):
                for j in s:
                    mask[i, j] = 1.0
            self.register_buffer("mono_mask", mask)       # [M, n]
            self.theta = nn.Parameter(torch.randn(out_dim, len(subs)) * 0.1)

        if residual_p > 0:
            self._init_residual(residual_p)

    @torch.no_grad()
    def _init_residual(self, p):
        c = residual_logit(p)
        if self.relaxation in ("probabilistic", "hybrid"):
            # node ~ first input: theta_pattern = +c if bit0==1 else -c, + noise
            bit0 = self.corners[:, 0]                     # [P]
            self.theta.copy_((2 * bit0 - 1) * c + 0.1 * torch.randn_like(self.theta))
        elif self.relaxation == "linear":
            self.theta.zero_()
            self.theta[:, 1] = c                          # weight on first input
            self.theta[:, 0] = -c / 2                     # compensating bias
        else:  # polynomial
            self.theta.zero_()
            self.theta[:, self._subs.index(())] = -c / 2
            self.theta[:, self._subs.index((0,))] = c

    # -- input selection (Top-K) -------------------------------------------
    def _select_soft(self, x):
        w = F.softmax(self.conn, dim=2)                  # [out, n, k]
        g = x[:, self.cand]                              # [B, out, n, k]
        return torch.einsum("bonk,onk->bon", g, w)       # [B, out, n]

    def _select_hard(self, x):
        sel = torch.gather(self.cand, 2, self.conn.argmax(2, keepdim=True)).squeeze(2)
        return x[:, sel]                                 # [B, out, n]

    # -- relaxation forward f(a) -> [B, out] in [0,1] ----------------------
    def _f(self, a):
        if self.relaxation in ("probabilistic", "hybrid"):
            au = a.unsqueeze(2)                          # [B, out, 1, n]
            c = self.corners.view(1, 1, -1, self.n)      # [1, 1, P, n]
            prod = (c * au + (1 - c) * (1 - au)).prod(3)  # [B, out, P]
            return (prod * torch.sigmoid(self.theta)[None]).sum(2)
        if self.relaxation == "linear":
            z = self.theta[None, :, 0] + torch.einsum("bon,on->bo", a, self.theta[:, 1:])
            return torch.sigmoid(z)
        # polynomial
        au = a.unsqueeze(2)                              # [B, out, 1, n]
        mk = self.mono_mask.view(1, 1, -1, self.n)       # [1, 1, M, n]
        mono = (au * mk + (1 - mk)).prod(3)              # [B, out, M]
        return torch.sigmoid((mono * self.theta[None]).sum(2))

    def forward(self, x):
        a = self._select_soft(x)
        f_soft = self._f(a)
        if self.relaxation == "hybrid":
            with torch.no_grad():
                f_disc = self._f((a >= 0.5).float())     # discrete forward
            return f_disc + f_soft - f_soft.detach()     # ...soft gradient (STE)
        return f_soft

    @torch.no_grad()
    def forward_hard(self, x):
        a = self._select_hard(x.float())                 # [B, out, n] in {0,1}
        return (self._f(a) >= 0.5).to(torch.uint8)       # boundary-consistent


class LUTNodeNet(nn.Module):
    """Stack of :class:`LUTNodeLayer`s + a :class:`~silogic.GroupSum` head.

    Args:
        in_dim (int): Number of binary input features.
        width (int): Nodes per layer; must be divisible by ``num_classes``.
        depth (int): Number of stacked layers.
        arity (int): Inputs per node. Default ``4``.
        relaxation (str): Node relaxation (see :class:`LUTNodeLayer`). Default
            ``"probabilistic"``.
        num_classes (int): Output classes. Default ``10``.
        k (int): Top-K candidates per input slot. Default ``8``.
        tau (float): GroupSum temperature. Default ``10.0``.
        residual_p (float): Residual-init probability. Default ``0.0``.
        seed (int): Base RNG seed; layer ``i`` uses ``seed*1000+i``. Default ``0``.
    """

    def __init__(self, in_dim, width, depth, arity=4, relaxation="probabilistic",
                 num_classes=10, k=8, tau=10.0, residual_p=0.0, seed=0):
        super().__init__()
        assert width % num_classes == 0
        layers = []
        d = in_dim
        for i in range(depth):
            layers.append(LUTNodeLayer(d, width, arity=arity, relaxation=relaxation,
                                       k=k, residual_p=residual_p, seed=seed * 1000 + i))
            d = width
        self.layers = nn.ModuleList(layers)
        self.head = GroupSum(num_classes, tau=tau)

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return self.head(x)

    @torch.no_grad()
    def forward_hard(self, x):
        for l in self.layers:
            x = l.forward_hard(x)
        return self.head.forward_hard(x)
