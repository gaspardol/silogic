"""k-input LUT node: the FPGA-native primitive (one LUT_k = any function of k
inputs). Far more expressive per node than a 2-input gate, AND maps to exactly
one FPGA LUT -> more capacity with fewer/smaller layers, still gate-efficient.

Differentiable via multilinear interpolation over the input hypercube (the
k-input generalization of BasisProj):
    out(a) = sum_{p in {0,1}^k}  sigmoid(logit_p) * prod_i (a_i if p_i else 1-a_i)
At inference each LUT binarizes its 2^k entries -> one hard truth table = 1 LUT.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LUTkLayer(nn.Module):
    """Layer of k-input LUT nodes. Inputs selected Top-K-style (learnable
    softmax over k random candidates per input slot) or fixed-random.

    Args:
        in_dim (int): Number of input features.
        out_dim (int): Number of LUT nodes (one FPGA LUT_k each).
        k (int): Inputs per LUT node; the node stores ``2**k`` learnable
            entries = one FPGA LUT_k. Default ``4``.
        learn_conn (bool): If ``True`` (default) each of the ``k`` input slots
            selects its wire via a learnable softmax over its candidates; if
            ``False`` the selection is fixed to the first candidate wire.
        cand_k (int): Number of candidate wires each of the ``k`` input slots
            chooses among (clamped to ``in_dim``). Default ``4``.
        seed (int | None): RNG seed for the fixed random candidate wiring;
            ``None`` (default) uses the global RNG (non-deterministic wiring).
    """

    def __init__(self, in_dim, out_dim, k=4, learn_conn=True, cand_k=4,
                 seed=None):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.k = k
        self.learn_conn = learn_conn
        gen = torch.Generator().manual_seed(seed) if seed is not None else None
        # each of the k input slots selects from cand_k candidates
        ck = min(cand_k, in_dim)
        self.cand_k = ck
        cand = torch.stack([
            torch.stack([torch.randperm(in_dim, generator=gen)[:ck]
                         for _ in range(k)])
            for _ in range(out_dim)])                     # [out, k, cand_k]
        self.register_buffer("cand", cand)
        if learn_conn:
            self.conn = nn.Parameter(torch.randn(out_dim, k, ck))
        # 2^k learnable LUT entry logits per node
        self.lut = nn.Parameter(torch.randn(out_dim, 2 ** k) * 0.1)
        # corner bit patterns [2^k, k] in {0,1}
        P = 2 ** k
        bits = torch.tensor([[(p >> i) & 1 for i in range(k)] for p in range(P)],
                            dtype=torch.float32)
        self.register_buffer("corners", bits)             # [P, k]

    def _select(self, x):
        """Return the k soft input values per node: [batch, out, k]."""
        if self.learn_conn:
            w = F.softmax(self.conn, dim=2)               # [out, k, cand_k]
            g = x[:, self.cand]                            # [batch, out, k, cand_k]
            return torch.einsum("bokc,okc->bok", g, w)
        return x[:, self.cand[..., 0]]                    # fixed: first candidate

    def forward(self, x):
        a = self._select(x)                               # [batch, out, k]
        # multilinear interpolation over the hypercube corners
        a = a.unsqueeze(2)                                # [batch, out, 1, k]
        c = self.corners.view(1, 1, -1, self.k)           # [1,1,P,k]
        term = c * a + (1 - c) * (1 - a)                  # match prob per bit
        prod = term.prod(dim=3)                           # [batch, out, P]
        e = torch.sigmoid(self.lut)                       # [out, P]
        return (prod * e.unsqueeze(0)).sum(dim=2)         # [batch, out]

    @torch.no_grad()
    def forward_hard(self, x):
        x = x.to(torch.uint8)
        if self.learn_conn:
            sel = self.conn.argmax(dim=2)                 # [out, k]
            idx = torch.gather(self.cand, 2, sel.unsqueeze(-1)).squeeze(-1)
        else:
            idx = self.cand[..., 0]
        a = x[:, idx]                                     # [batch, out, k] bits
        # index into the 2^k truth table: addr = sum_i a_i << i
        shifts = (2 ** torch.arange(self.k, device=x.device)).view(1, 1, -1)
        addr = (a.long() * shifts).sum(dim=2)             # [batch, out] in 0..2^k-1
        tt = (self.lut > 0).to(torch.uint8)               # [out, 2^k]
        return torch.gather(tt, 1, addr.t()).t()          # [batch, out]


class LUTkNet(nn.Module):
    """Stack of LUTk layers + GroupSum head (for capacity comparison).

    Args:
        in_dim (int): Number of input features to the first layer.
        width (int): Number of LUT nodes per layer (output width of every
            layer after the first).
        depth (int): Number of stacked :class:`LUTkLayer` layers.
        k (int): Inputs per LUT node; each node stores ``2**k`` entries.
            Default ``4``.
        num_classes (int): Number of output classes for the GroupSum head.
            Default ``10``.
        tau (float): GroupSum temperature. Default ``4.0``.
        cand_k (int): Candidate wires per input slot, passed to each layer.
            Default ``4``.
        seed (int): Base RNG seed; layer ``i`` is seeded with
            ``seed * 1000 + i`` for distinct wiring. Default ``0``.
    """

    def __init__(self, in_dim, width, depth, k=4, num_classes=10, tau=4.0,
                 cand_k=4, seed=0):
        super().__init__()
        from .model import GroupSum
        layers = []
        d_in = in_dim
        for i in range(depth):
            layers.append(LUTkLayer(d_in, width, k=k, cand_k=cand_k,
                                    seed=seed * 1000 + i))
            d_in = width
        self.layers = nn.ModuleList(layers)
        self.head = GroupSum(num_classes, tau=tau)
        self.width = width; self.depth = depth; self.k = k

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return self.head(x)

    @torch.no_grad()
    def forward_hard(self, x):
        for l in self.layers:
            x = l.forward_hard(x)
        return self.head.forward_hard(x)

    def num_luts(self):
        return self.width * self.depth   # one LUT per node
