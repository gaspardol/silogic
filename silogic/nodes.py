"""Unified differentiable logic-node parameterizations.

Every "logic node" maps selected operands ``[B, out, arity]`` (relaxed Booleans
in ``[0, 1]``) to one relaxed Boolean per node ``[B, out]``, and discretizes to an
exact truth table for inference (``forward_hard``). The library previously had
six *separate* implementations of this idea scattered across files; this module
collects them into one registry of **boundary-consistent** parameterizations
(``f`` recovers the hard LUT at the cube corners), selectable by name:

  * ``"gate16"``      — softmax mixture over the 16 two-input Boolean functions,
    evaluated via BasisProj (``{1,A,B,A*B}``) or FullEval. Arity 2. (LILogic / DLGN)
  * ``"walsh"``       — Walsh-Hadamard coefficients ``theta in R^(2^arity)``,
    ``f = sigmoid(z/tau)``. Arity 2 (4 params) or n. (WARP, arXiv:2602.03527)
  * ``"multilinear"`` — one logit per truth-table entry, multilinear interpolation
    over the hypercube (the k-input generalization of BasisProj). Arity n.
    Equivalent to an FPGA LUT_k. (was ``LUTkLayer`` / the ``"probabilistic"`` relax.)
  * ``"hybrid"``      — DWN-style: discrete forward (hard-thresholded operands) with
    the multilinear surrogate gradient. Arity n.
  * ``"linear"``      — perceptron node ``sigmoid(theta0 + sum_j theta_j a_j)``. Arity n.
  * ``"polynomial"``  — degree-``d`` multilinear polynomial. Arity n.

All nodes share :meth:`Node.residual_init` (bias toward passing through the first
input), replacing the three ad-hoc residual inits that lived in ``model``/``warp``/
the old ``nodes`` module.
"""
from itertools import combinations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .functional import (BASIS_COEFFS, TRUTH_TABLES, GATE_A, gate_probs,
                         residual_logit, corner_bits, walsh_monomials)

# node registry + legacy relaxation aliases
NODES = ("gate16", "walsh", "multilinear", "hybrid", "linear", "polynomial")
RELAXATIONS = ("probabilistic", "hybrid", "linear", "polynomial")  # LUTNodeLayer names
_NODE_ALIASES = {
    "gate16": "gate16", "gate": "gate16",
    "walsh": "walsh",
    "multilinear": "multilinear", "probabilistic": "multilinear",
    "hybrid": "hybrid", "linear": "linear", "polynomial": "polynomial",
}


def _gumbel(shape, device):
    u = torch.rand(shape, device=device).clamp_(1e-6, 1 - 1e-6)
    return -torch.log(-torch.log(u))


class Node(nn.Module):
    """Base class: ``forward``/``forward_hard`` take operands ``[B, out, arity]``."""

    def __init__(self, out_dim, arity):
        super().__init__()
        self.out_dim = out_dim
        self.arity = arity

    def residual_init(self, p):    # overridden per family
        raise NotImplementedError


# ---------------------------------------------------------------------------
class Gate16Node(Node):
    """Softmax mixture over the 16 two-input Boolean functions (arity 2)."""

    def __init__(self, out_dim, arity=2, gate_eval="basis",
                 gate_select="softmax", gumbel_tau=1.0):
        super().__init__(out_dim, 2)
        assert arity == 2, "gate16 nodes are two-input"
        self.gate_eval = gate_eval
        self.gate_select = gate_select
        self.gumbel_tau = gumbel_tau
        self.gate_logits = nn.Parameter(torch.randn(out_dim, 16) * 0.1)
        self.register_buffer("basis", BASIS_COEFFS.clone())
        self.register_buffer("truth", TRUTH_TABLES.clone())

    def _probs(self):
        return gate_probs(self.gate_logits, self.training, dim=1,
                          gate_select=self.gate_select, gumbel_tau=self.gumbel_tau)

    def coef(self):
        """BasisProj coefficients ``[out, 4]`` (for the fused Triton path)."""
        return self._probs() @ self.basis

    def forward(self, operands):
        a, b = operands[..., 0], operands[..., 1]
        p = self._probs()                                            # [out, 16]
        if self.gate_eval == "basis":
            c = p @ self.basis                                       # [out, 4]
            return c[:, 0] + c[:, 1] * a + c[:, 2] * b + c[:, 3] * (a * b)
        ab = a * b
        funcs = torch.stack([
            torch.zeros_like(a), ab, a - ab, a, b - ab, b, a + b - 2 * ab,
            a + b - ab, 1 - a - b + ab, 1 - a - b + 2 * ab, 1 - b, 1 - b + ab,
            1 - a, 1 - a + ab, 1 - ab, torch.ones_like(a),
        ], dim=0)                                                    # [16, B, out]
        return torch.einsum("ibo,oi->bo", funcs, p)

    @torch.no_grad()
    def forward_hard(self, operands):
        a, b = operands[..., 0], operands[..., 1]
        gate = self.gate_logits.argmax(dim=1)                        # [out]
        idx = (a.long() << 1) | b.long()                             # [B, out] in 0..3
        tt = self.truth.to(operands.device)[gate]                    # [out, 4]
        return torch.gather(tt, 1, idx.t()).t().to(torch.uint8)

    @torch.no_grad()
    def residual_init(self, p):
        self.gate_logits.normal_(0, 0.1)
        self.gate_logits[:, GATE_A] = residual_logit(p)              # bias toward 'A'


# ---------------------------------------------------------------------------
class WalshNode(Node):
    """Walsh-Hadamard node: ``z = sum_i theta_i * monomial_i(2a-1)``,
    ``f = sigmoid(z / tau)``. Arity 2 (4 params) or n (``2**arity``)."""

    def __init__(self, out_dim, arity=2, tau=1.0, gate_select="softmax"):
        super().__init__(out_dim, arity)
        self.tau = tau
        self.gate_select = gate_select          # "gumbel" -> Gumbel-sigmoid smoothing
        self.theta = nn.Parameter(torch.randn(out_dim, 2 ** arity) * 0.1)

    @property
    def inv_tau(self):
        return 1.0 / self.tau

    def _z(self, *operands_ab):
        """z for given per-slot soft values (arity 2 convenience: ``_z(a, b)``)."""
        u = [2 * v - 1 for v in operands_ab]
        if self.arity == 2:
            t = self.theta
            return t[:, 0] + t[:, 1] * u[0] + t[:, 2] * u[1] + t[:, 3] * (u[0] * u[1])
        um = torch.stack(u, dim=2)                                   # [B, out, n]
        return (walsh_monomials(um) * self.theta[None]).sum(2)

    def coef(self):
        """Arity-2 BasisProj coeffs ``[out, 4]`` in ``(a, b)`` space (fused path)."""
        t = self.theta
        return torch.stack([t[:, 0] - t[:, 1] - t[:, 2] + t[:, 3],
                            2 * (t[:, 1] - t[:, 3]), 2 * (t[:, 2] - t[:, 3]),
                            4 * t[:, 3]], dim=1)

    def forward(self, operands):
        u = 2 * operands - 1                                         # [B, out, arity]
        z = (walsh_monomials(u) * self.theta[None]).sum(2)
        if self.gate_select == "gumbel" and self.training:           # Gumbel-sigmoid
            z = z + _gumbel(z.shape, z.device) - _gumbel(z.shape, z.device)
        return torch.sigmoid(z * self.inv_tau)

    @torch.no_grad()
    def forward_hard(self, operands):
        u = 2 * operands.float() - 1                                 # uint8-safe
        z = (walsh_monomials(u) * self.theta[None]).sum(2)
        return (z > 0).to(torch.uint8)

    @torch.no_grad()
    def residual_init(self, p):
        self.theta.zero_()
        self.theta[:, 2 ** (self.arity - 1)] = residual_logit(p, self.tau)  # pass last input


# ---------------------------------------------------------------------------
class MultilinearNode(Node):
    """One logit per truth-table entry; multilinear interpolation (arity n).
    Equivalent to an FPGA LUT_k (was ``LUTkLayer`` / the ``probabilistic`` relax.)."""

    def __init__(self, out_dim, arity=2):
        super().__init__(out_dim, arity)
        self.register_buffer("corners", corner_bits(arity))          # [P, n]
        self.logits = nn.Parameter(torch.randn(out_dim, 2 ** arity) * 0.1)

    def _f(self, a):
        au = a.unsqueeze(2)                                          # [B, out, 1, n]
        c = self.corners.view(1, 1, -1, self.arity)                 # [1, 1, P, n]
        prod = (c * au + (1 - c) * (1 - au)).prod(3)                 # [B, out, P]
        return (prod * torch.sigmoid(self.logits)[None]).sum(2)

    def forward(self, operands):
        return self._f(operands)

    @torch.no_grad()
    def forward_hard(self, operands):
        a = operands.long()                                          # [B, out, n] bits
        shifts = (2 ** torch.arange(self.arity, device=a.device)).view(1, 1, -1)
        addr = (a * shifts).sum(dim=2)                               # [B, out]
        tt = (self.logits > 0).to(torch.uint8)                      # [out, P]
        return torch.gather(tt, 1, addr.t()).t()

    @torch.no_grad()
    def residual_init(self, p):
        c = residual_logit(p)
        bit0 = self.corners[:, 0]                                    # [P]
        self.logits.copy_((2 * bit0 - 1) * c + 0.1 * torch.randn_like(self.logits))


class HybridNode(MultilinearNode):
    """DWN-style: discrete forward (hard-thresholded operands) with the
    multilinear surrogate gradient (straight-through)."""

    def forward(self, operands):
        f_soft = self._f(operands)
        with torch.no_grad():
            f_disc = self._f((operands >= 0.5).float())
        return f_disc + f_soft - f_soft.detach()


# ---------------------------------------------------------------------------
class LinearNode(Node):
    """Perceptron node ``sigmoid(theta0 + sum_j theta_j a_j)`` (arity n)."""

    def __init__(self, out_dim, arity=2):
        super().__init__(out_dim, arity)
        self.theta = nn.Parameter(torch.randn(out_dim, arity + 1) * 0.1)  # [bias, w...]

    def _z(self, a):
        return self.theta[None, :, 0] + torch.einsum("boa,oa->bo", a, self.theta[:, 1:])

    def forward(self, operands):
        return torch.sigmoid(self._z(operands))

    @torch.no_grad()
    def forward_hard(self, operands):
        return (self._z(operands.float()) >= 0).to(torch.uint8)

    @torch.no_grad()
    def residual_init(self, p):
        c = residual_logit(p)
        self.theta.zero_()
        self.theta[:, 1] = c                                         # weight on first input
        self.theta[:, 0] = -c / 2                                    # compensating bias


class PolynomialNode(Node):
    """Degree-``d`` multilinear polynomial node (arity n)."""

    def __init__(self, out_dim, arity=2, degree=None):
        super().__init__(out_dim, arity)
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
        self.register_buffer("mono_mask", mask)                      # [M, n]
        self.theta = nn.Parameter(torch.randn(out_dim, len(subs)) * 0.1)

    def _z(self, a):
        au = a.unsqueeze(2)                                          # [B, out, 1, n]
        mk = self.mono_mask.view(1, 1, -1, self.arity)              # [1, 1, M, n]
        mono = (au * mk + (1 - mk)).prod(3)                          # [B, out, M]
        return (mono * self.theta[None]).sum(2)

    def forward(self, operands):
        return torch.sigmoid(self._z(operands))

    @torch.no_grad()
    def forward_hard(self, operands):
        return (self._z(operands.float()) >= 0).to(torch.uint8)

    @torch.no_grad()
    def residual_init(self, p):
        c = residual_logit(p)
        self.theta.zero_()
        self.theta[:, self._subs.index(())] = -c / 2
        self.theta[:, self._subs.index((0,))] = c


_BUILDERS = {
    "gate16": Gate16Node, "walsh": WalshNode, "multilinear": MultilinearNode,
    "hybrid": HybridNode, "linear": LinearNode, "polynomial": PolynomialNode,
}


def build_node(name, out_dim, arity=2, **kw):
    """Construct a node parameterization by name (see module docstring)."""
    key = _NODE_ALIASES.get(str(name).lower())
    if key is None:
        raise ValueError(f"unknown node {name!r}; choose from {NODES}")
    cls = _BUILDERS[key]
    if key == "gate16":
        return cls(out_dim, arity, gate_eval=kw.get("gate_eval", "basis"),
                   gate_select=kw.get("gate_select", "softmax"),
                   gumbel_tau=kw.get("gumbel_tau", 1.0))
    if key == "walsh":
        return cls(out_dim, arity, tau=kw.get("tau", 1.0),
                   gate_select=kw.get("gate_select", "softmax"))
    if key == "polynomial":
        return cls(out_dim, arity, degree=kw.get("degree"))
    return cls(out_dim, arity)
