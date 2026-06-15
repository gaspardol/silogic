"""WARP Logic Neural Networks (Gerlach et al., arXiv:2602.03527).

Each 2-input node is parameterized by the Walsh-Hadamard coefficients theta in
R^4 (vs the 16 gate-logits of DLGN/BasisProj), with a sigmoid relaxation:

    f(a,b) = sigmoid( (1/tau) * (t0 + t1*u + t2*v + t3*u*v) ),   u=2a-1, v=2b-1

- 4 free params/node (max parameter-efficiency; BasisProj uses 16).
- residual init (Eq 10): bias toward pass-through of the 2nd input.
- stochastic smoothing (Eq 9): Gumbel-sigmoid noise -> shrinks soft->hard gap.
- hard inference: out = 1[z > 0]  (the tau->0 limit; an exact 2-LUT).

Same Top-K input connectome as the BasisProj LogicLayer for a fair comparison.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .model import GroupSum

try:
    from .triton_warp import warp_logic
    _HAS_TRITON_WARP = True
except Exception:
    _HAS_TRITON_WARP = False

WARP_GUMBEL = {"enabled": False}
WARP_USE_TRITON = True


def _gumbel(shape, device):
    u = torch.rand(shape, device=device).clamp_(1e-6, 1 - 1e-6)
    return -torch.log(-torch.log(u))


class WARPLayer(nn.Module):
    def __init__(self, in_dim, out_dim, k=8, tau=1.0, residual_p=0.0, seed=None):
        super().__init__()
        self.in_dim = in_dim; self.out_dim = out_dim; self.tau = tau
        gen = torch.Generator().manual_seed(seed) if seed is not None else None
        kk = min(k, in_dim); self.k = kk
        ca = torch.stack([torch.randperm(in_dim, generator=gen)[:kk] for _ in range(out_dim)])
        cb = torch.stack([torch.randperm(in_dim, generator=gen)[:kk] for _ in range(out_dim)])
        self.register_buffer("cand_a", ca); self.register_buffer("cand_b", cb)
        self.register_buffer("cand_a_i32", ca.to(torch.int32))
        self.register_buffer("cand_b_i32", cb.to(torch.int32))
        self.conn_a = nn.Parameter(torch.randn(out_dim, kk, generator=gen))
        self.conn_b = nn.Parameter(torch.randn(out_dim, kk, generator=gen))
        # theta [out, 4] = (const, u, v, uv) Walsh coeffs
        theta = torch.zeros(out_dim, 4)
        if residual_p > 0:           # residual init: pass-through of 2nd input (v)
            theta[:, 2] = tau * math.log(residual_p / (1 - residual_p))
        else:
            theta = torch.randn(out_dim, 4) * 0.1
        self.theta = nn.Parameter(theta)

    def _select(self, x):
        wa = F.softmax(self.conn_a, dim=1); wb = F.softmax(self.conn_b, dim=1)
        a = torch.einsum("bok,ok->bo", x[:, self.cand_a], wa)
        b = torch.einsum("bok,ok->bo", x[:, self.cand_b], wb)
        return a, b

    def _z(self, a, b):
        u = 2 * a - 1; v = 2 * b - 1
        t = self.theta
        return t[:, 0] + t[:, 1] * u + t[:, 2] * v + t[:, 3] * (u * v)

    def _coef(self):
        # theta (const,u,v,uv) with u=2a-1,v=2b-1  ->  z = c0+c1*a+c2*b+c3*a*b
        t = self.theta
        return torch.stack([t[:, 0]-t[:, 1]-t[:, 2]+t[:, 3], 2*(t[:, 1]-t[:, 3]),
                            2*(t[:, 2]-t[:, 3]), 4*t[:, 3]], dim=1)

    def forward(self, x):
        use_k = (_HAS_TRITON_WARP and WARP_USE_TRITON and x.is_cuda
                 and not (WARP_GUMBEL["enabled"] and self.training))
        if use_k:
            wa = F.softmax(self.conn_a, dim=1); wb = F.softmax(self.conn_b, dim=1)
            return warp_logic(x, self.cand_a_i32, self.cand_b_i32, wa, wb,
                              self._coef(), 1.0 / self.tau)
        a, b = self._select(x)
        z = self._z(a, b)
        if WARP_GUMBEL["enabled"] and self.training:    # Gumbel-sigmoid smoothing
            z = z + _gumbel(z.shape, z.device) - _gumbel(z.shape, z.device)
        return torch.sigmoid(z / self.tau)

    @torch.no_grad()
    def forward_hard(self, x):
        # hard input select (argmax conn), binary inputs -> exact 2-LUT.
        # cast to float first: x may be uint8, but _z does 2*x-1 (would underflow).
        x = x.float()
        sa = torch.gather(self.cand_a, 1, self.conn_a.argmax(1, keepdim=True)).squeeze(1)
        sb = torch.gather(self.cand_b, 1, self.conn_b.argmax(1, keepdim=True)).squeeze(1)
        return (self._z(x[:, sa], x[:, sb]) > 0).float()


class WARPNet(nn.Module):
    def __init__(self, in_dim, width, depth, num_classes=10, k=8, tau=1.0,
                 residual_p=0.0, seed=0):
        super().__init__()
        assert width % num_classes == 0
        layers = []; d = in_dim
        for i in range(depth):
            layers.append(WARPLayer(d, width, k=k, tau=tau, residual_p=residual_p,
                                    seed=seed * 100 + i))
            d = width
        self.layers = nn.ModuleList(layers)
        self.head = GroupSum(num_classes, tau=20.0)

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return self.head(x)

    @torch.no_grad()
    def forward_hard(self, x):
        for l in self.layers:
            x = l.forward_hard(x)
        return self.head.forward_hard(x)


# ---- general-arity WARP (n-input LUT via Walsh monomials) ----
class WARPLayerN(nn.Module):
    """Arity-n WARP node: theta in R^(2^n), z = sum_i theta_i * prod_{k in i} u_k,
    u_k = 2*x_k-1 (Walsh monomials), out = sigmoid(z/tau). n inputs selected
    Top-K-style. PyTorch (no custom kernel for n>2 yet)."""
    def __init__(self, in_dim, out_dim, arity=6, k=8, tau=1.0, residual_p=0.0, seed=None):
        super().__init__()
        self.in_dim = in_dim; self.out_dim = out_dim; self.n = arity; self.tau = tau
        gen = torch.Generator().manual_seed(seed) if seed is not None else None
        kk = min(k, in_dim); self.k = kk
        cand = torch.stack([torch.stack([torch.randperm(in_dim, generator=gen)[:kk]
                                         for _ in range(arity)]) for _ in range(out_dim)])
        self.register_buffer("cand", cand)               # [out, n, k]
        self.conn = nn.Parameter(torch.randn(out_dim, arity, kk, generator=gen))
        P = 2 ** arity
        theta = torch.randn(out_dim, P) * 0.1
        if residual_p > 0:                               # pass-through last input (u_{n-1})
            theta.zero_(); theta[:, 2 ** (arity - 1)] = tau * math.log(residual_p/(1-residual_p))
        self.theta = nn.Parameter(theta)

    def _select(self, x):
        w = F.softmax(self.conn, dim=2)                  # [out,n,k]
        g = x[:, self.cand]                              # [B,out,n,k]
        return torch.einsum("bonk,onk->bon", g, w)       # [B,out,n] in [0,1]

    def _monomials(self, u):                             # u [B,out,n] -> [B,out,2^n]
        phi = torch.ones(u.shape[0], u.shape[1], 1, device=u.device)
        for kk in range(self.n):
            phi = torch.cat([phi, phi * u[:, :, kk:kk+1]], dim=2)
        return phi

    def forward(self, x):
        u = 2 * self._select(x) - 1
        z = (self._monomials(u) * self.theta[None]).sum(2)
        if WARP_GUMBEL["enabled"] and self.training:
            z = z + _gumbel(z.shape, z.device) - _gumbel(z.shape, z.device)
        return torch.sigmoid(z / self.tau)

    @torch.no_grad()
    def forward_hard(self, x):
        x = x.float()                                    # uint8-safe: 2*x-1 below
        sel = torch.gather(self.cand, 2, self.conn.argmax(2, keepdim=True)).squeeze(2)  # [out,n]
        u = 2 * x[:, sel] - 1                            # [B,out,n] in {-1,1}
        z = (self._monomials(u) * self.theta[None]).sum(2)
        return (z > 0).float()


class WARPNetN(nn.Module):
    def __init__(self, in_dim, width, depth, arity=6, num_classes=10, k=8, tau=1.0,
                 residual_p=0.0, seed=0):
        super().__init__()
        assert width % num_classes == 0
        layers = []; d = in_dim
        for i in range(depth):
            layers.append(WARPLayerN(d, width, arity=arity, k=k, tau=tau,
                                     residual_p=residual_p, seed=seed*100+i))
            d = width
        self.layers = nn.ModuleList(layers)
        self.head = GroupSum(num_classes, tau=20.0)

    def forward(self, x):
        for l in self.layers: x = l(x)
        return self.head(x)

    @torch.no_grad()
    def forward_hard(self, x):
        for l in self.layers: x = l.forward_hard(x)
        return self.head.forward_hard(x)
