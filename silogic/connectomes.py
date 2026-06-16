"""Input-selection strategies ("connectomes") shared by every logic layer.

A connectome decides *which* previous-layer wires feed each node's input slots.
Previously every layer family (gate, WARP, LUT-k, LUT-node, pairwise) re-derived
the same Top-K gather/argmax code; this module factors it into one small set of
reusable modules selectable by name, all producing operands of a uniform shape
``[B, out, arity]`` so any node parameterization can consume them.

Strategies (the ``kind`` argument, case-insensitive; legacy names in parens):

  * ``"fixed"`` (``F``)    — one fixed random wire per input slot (DiffLogicNet).
  * ``"dense"`` (``L``)    — softmax over *all* previous nodes (learnable, dense).
  * ``"topk"``  (``TopK``) — each slot picks among ``k`` random candidates via a
    learnable softmax (LILogic). Exposes ``weights()`` / ``candidates_i32()`` for
    the fused Triton path.
  * ``"blocktopk"`` (``BlockTopK``) — Top-K with candidates from a contiguous
    window for cache-local gathers.
  * ``"st"`` / ``"stw"`` / ``"stt"`` — sum-threshold inputs (BNN-style): each
    slot is ``threshold(BatchNorm(weighted popcount of k candidate wires))`` with
    no / binary / ternary weights respectively.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .functional import ste_threshold, sign_ste, ternary_ste

# canonical kind -> implementation; legacy aliases resolved in build_connectome
_ALIASES = {
    "f": "fixed", "fixed": "fixed",
    "l": "dense", "dense": "dense",
    "topk": "topk", "blocktopk": "blocktopk",
    "st": "st", "stw": "stw", "stt": "stt",
}


def _candidates(in_dim, out_dim, arity, k, gen):
    """Per-slot random candidate indices, shape ``[out, arity, k]`` (draw order
    matches the legacy per-operand ``randperm`` loops)."""
    return torch.stack([
        torch.stack([torch.randperm(in_dim, generator=gen)[:k] for _ in range(arity)])
        for _ in range(out_dim)
    ])


class FixedConnectome(nn.Module):
    """One fixed random wire per (node, slot); no learnable connection params."""

    def __init__(self, in_dim, out_dim, arity, gen=None):
        super().__init__()
        self.in_dim, self.out_dim, self.arity = in_dim, out_dim, arity
        idx = torch.randint(0, in_dim, (out_dim, arity), generator=gen)
        self.register_buffer("idx", idx)                 # [out, arity]
        # K=1 candidate/weight view so fused kernels (which expect [out, arity, K]
        # candidates + post-softmax weights) can consume a fixed connectome too.
        self.register_buffer("cand_i32", idx.unsqueeze(-1).to(torch.int32))

    def select_soft(self, x):
        return x[:, self.idx]                            # [B, out, arity]

    select_hard = select_soft

    def candidates_i32(self):
        return self.cand_i32                             # [out, arity, 1]

    def weights(self):
        return torch.ones(self.out_dim, self.arity, 1, device=self.idx.device)


class DenseConnectome(nn.Module):
    """Softmax over all previous nodes for each slot (fully learnable wiring)."""

    def __init__(self, in_dim, out_dim, arity, gen=None):
        super().__init__()
        self.in_dim, self.out_dim, self.arity = in_dim, out_dim, arity
        self.conn = nn.Parameter(torch.randn(out_dim, arity, in_dim, generator=gen) * 0.1)

    def select_soft(self, x):
        w = F.softmax(self.conn, dim=2)                  # [out, arity, in]
        return torch.einsum("bi,oai->boa", x, w)         # [B, out, arity]

    def select_hard(self, x):
        idx = self.conn.argmax(dim=2)                    # [out, arity]
        return x[:, idx]


class TopKConnectome(nn.Module):
    """Each slot selects among ``k`` random candidates via a learnable softmax.

    ``block=True`` draws candidates from a contiguous per-node window and gives
    consecutive nodes consecutive windows, so the gathers are cache-local.
    """

    def __init__(self, in_dim, out_dim, arity, k, gen=None, block=False, window=0):
        super().__init__()
        self.in_dim, self.out_dim, self.arity = in_dim, out_dim, arity
        kk = min(k, in_dim)
        self.k = kk
        if block:
            W = min(in_dim, window if window else max(4 * kk, 256))
            starts = (torch.arange(out_dim).float() / max(1, out_dim)
                      * max(1, in_dim - W)).long()
            cand = torch.stack([
                torch.stack([starts[o] + torch.randperm(W, generator=gen)[:kk]
                             for _ in range(arity)])
                for o in range(out_dim)])
        else:
            cand = _candidates(in_dim, out_dim, arity, kk, gen)
        self.register_buffer("cand", cand)               # [out, arity, k]
        self.register_buffer("cand_i32", cand.to(torch.int32))
        self.conn = nn.Parameter(torch.randn(out_dim, arity, kk, generator=gen))
        self.ste = False   # straight-through (argmax forward, softmax grad) selection

    def weights(self):
        w = F.softmax(self.conn, dim=2)                  # [out, arity, k]
        if self.ste:        # hard one-hot forward, soft gradient -> soft==hard select
            hard = F.one_hot(self.conn.argmax(dim=2), w.shape[-1]).to(w.dtype)
            w = hard + w - w.detach()
        return w

    def candidates_i32(self):
        return self.cand_i32

    def select_soft(self, x):
        w = self.weights()
        g = x[:, self.cand]                              # [B, out, arity, k]
        return torch.einsum("boak,oak->boa", g, w)       # [B, out, arity]

    def select_hard(self, x):
        sel = self.conn.argmax(dim=2)                    # [out, arity]
        idx = torch.gather(self.cand, 2, sel.unsqueeze(-1)).squeeze(-1)
        return x[:, idx]


class SumThresholdConnectome(nn.Module):
    """Sum-threshold inputs (BNN-style): each slot is
    ``ste_threshold(BatchNorm(sum_k w_k * wire_k))`` over ``k`` fixed-random
    candidate wires. ``mode`` selects the weights:

      * ``"none"`` (ST)  — unweighted popcount.
      * ``"binary"`` (STW) — ``{-1,+1}`` weights (free NOT gates).
      * ``"ternary"`` (STT) — ``{-1,0,+1}`` weights (learnable sparsity).

    Outputs are already binary ``{0,1}`` (straight-through), matching inference.
    BatchNorm stabilizes training and folds into the threshold at deploy time.
    """

    def __init__(self, in_dim, out_dim, arity, k, mode="none", gen=None):
        super().__init__()
        self.in_dim, self.out_dim, self.arity, self.mode = in_dim, out_dim, arity, mode
        kk = min(k, in_dim)
        self.k = kk
        self.register_buffer("cand", _candidates(in_dim, out_dim, arity, kk, gen))
        self.bn = nn.ModuleList([nn.BatchNorm1d(out_dim) for _ in range(arity)])
        if mode != "none":
            scale = 0.1 if mode == "binary" else 1.0     # STT inits ~1/3 each of {-1,0,1}
            self.w = nn.Parameter(torch.randn(out_dim, arity, kk) * scale)

    def _sums(self, x, hard):
        g = x[:, self.cand].float()                      # [B, out, arity, k]
        if self.mode == "none":
            return g.sum(dim=3)                          # [B, out, arity] popcount
        if hard:
            if self.mode == "binary":
                w = torch.where(self.w >= 0, 1.0, -1.0)
            else:
                w = torch.where(self.w > 0.5, 1.0, torch.where(self.w < -0.5, -1.0, 0.0))
        else:
            w = sign_ste(self.w) if self.mode == "binary" else ternary_ste(self.w)
        return ((2.0 * g - 1.0) * w).sum(dim=3)          # [B, out, arity]

    def _threshold(self, s, hard):
        cols = []
        for a in range(self.arity):
            z = self.bn[a](s[:, :, a])
            cols.append((z > 0).to(s.dtype) if hard else ste_threshold(z))
        return torch.stack(cols, dim=2)                  # [B, out, arity]

    def select_soft(self, x):
        return self._threshold(self._sums(x, hard=False), hard=False)

    def select_hard(self, x):
        return self._threshold(self._sums(x.float(), hard=True), hard=True).to(torch.uint8)


def build_connectome(kind, in_dim, out_dim, arity=2, k=8, seed=None,
                     window=0):
    """Construct a connectome by name (see module docstring for the kinds)."""
    key = _ALIASES.get(str(kind).lower())
    if key is None:
        raise ValueError(f"unknown connectome {kind!r}")
    gen = torch.Generator().manual_seed(seed) if seed is not None else None
    if key == "fixed":
        return FixedConnectome(in_dim, out_dim, arity, gen)
    if key == "dense":
        return DenseConnectome(in_dim, out_dim, arity, gen)
    if key == "topk":
        return TopKConnectome(in_dim, out_dim, arity, k, gen)
    if key == "blocktopk":
        return TopKConnectome(in_dim, out_dim, arity, k, gen, block=True, window=window)
    mode = {"st": "none", "stw": "binary", "stt": "ternary"}[key]
    return SumThresholdConnectome(in_dim, out_dim, arity, k, mode=mode, gen=gen)
