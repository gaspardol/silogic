"""LILogic Net: Compact Logic Gate Networks with Learnable Connectivity.

Reimplementation of the method from "LILogic Net" (arXiv:2511.12340).

Core ideas implemented here:
  * 16 two-input Boolean functions, each node is a softmax mixture over them.
  * Three connectome strategies for wiring inputs:
      - F     : Fixed random connections (DiffLogicNet style).
      - L     : Fully learnable dense connectome (softmax over all prev nodes).
      - TopK  : Each input independently chooses among K random candidates,
                with a learnable softmax over those K.
  * BasisProj: instead of evaluating all 16 gates, project the gate
    distribution into the {1, A, B, A*B} basis (4 coeffs) and evaluate once.
  * FullEval: the classic path that evaluates all 16 soft gate functions.
  * GroupSum classification head with a global temperature tau.
  * Hard (binarized) inference path for deterministic Boolean circuits.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .triton_dense import dense_logic
    _HAS_TRITON_DENSE = True
except Exception:
    _HAS_TRITON_DENSE = False


# ---------------------------------------------------------------------------
# The 16 Boolean functions expressed in the basis {1, A, B, A*B}.
# Row i = coefficients [c1, c2, c3, c4] such that
#   gate_i(a,b) = c1*1 + c2*a + c3*b + c4*(a*b).
# Order matches Eq. (4) / Table in the paper (standard DiffLogic ordering).
# ---------------------------------------------------------------------------
BASIS_COEFFS = torch.tensor(
    [
        [0, 0, 0, 0],    # 0  False
        [0, 0, 0, 1],    # 1  A and B
        [0, 1, 0, -1],   # 2  not(A => B)  = A and not B
        [0, 1, 0, 0],    # 3  A
        [0, 0, 1, -1],   # 4  not(A <= B)  = not A and B
        [0, 0, 1, 0],    # 5  B
        [0, 1, 1, -2],   # 6  A xor B
        [0, 1, 1, -1],   # 7  A or B
        [1, -1, -1, 1],  # 8  not(A or B)
        [1, -1, -1, 2],  # 9  not(A xor B)
        [1, 0, -1, 0],   # 10 not B
        [1, 0, -1, 1],   # 11 A <= B  = A or not B
        [1, -1, 0, 0],   # 12 not A
        [1, -1, 0, 1],   # 13 A => B  = not A or B
        [1, 0, 0, -1],   # 14 not(A and B)
        [1, 0, 0, 0],    # 15 True
    ],
    dtype=torch.float32,
)


def _truth_tables():
    """Derive the 16x4 hard truth tables from the basis coefficients.

    Columns correspond to (a,b) = (0,0), (0,1), (1,0), (1,1).
    Returns a {0,1} uint8 tensor of shape [16, 4].
    """
    c = BASIS_COEFFS
    tt = torch.stack(
        [
            c[:, 0],                              # (0,0)
            c[:, 0] + c[:, 2],                    # (0,1)
            c[:, 0] + c[:, 1],                    # (1,0)
            c[:, 0] + c[:, 1] + c[:, 2] + c[:, 3],# (1,1)
        ],
        dim=1,
    )
    return tt.round().to(torch.uint8)


TRUTH_TABLES = _truth_tables()  # [16, 4]


# ---------------------------------------------------------------------------
# Gumbel straight-through gate selection (Yousefi et al., "Mind the Gap",
# NeurIPS 2025). During training, perturb the 16 gate logits with Gumbel noise
# and take a HARD argmax in the forward pass (matching inference), while the
# backward pass uses the soft Gumbel-softmax (straight-through estimator).
# This aligns train/inference (≈0 discretization gap) and acts as implicit
# Hessian regularization -> flatter minima, full gate utilization, faster
# convergence. Toggle via GUMBEL["enabled"]; tau is the goldilocks ~0.25-1.0.
# ---------------------------------------------------------------------------
GUMBEL = {"enabled": False, "tau": 1.0}
# STE-binarize logic activations before the sumlinear decoder (forward = {0,1},
# backward = identity) so its summed input matches inference. Set False to ablate.
DEC_FEATURE_STE = True
# Deterministic hard-gate straight-through (forward = argmax one-hot, backward =
# softmax). Aligns gate training with inference (closes the gate discretization
# gap) without the stochastic noise that destabilizes deep nets under Gumbel.
HARD_GATE = {"enabled": False}


def ste_threshold(s):
    """Straight-through binary threshold: forward = (s>0) in {0,1}, backward =
    gradient of sigmoid(s). Output is a relaxed Boolean in {0,1} (forward).

    Args:
        s (torch.Tensor): Real-valued pre-activation; thresholded at ``0``.

    Returns:
        torch.Tensor: Same shape as ``s``; forward values in ``{0, 1}``
        carrying the ``sigmoid(s)`` gradient (straight-through).
    """
    hard = (s > 0).float()
    soft = torch.sigmoid(s)
    return hard + soft - soft.detach()


def sign_ste(w):
    """Binarize weights to {-1,+1} (forward) with clipped straight-through
    gradient (backward), BNN-style. A +/-1 weight on a binary wire = pass or
    invert = a free NOT gate in hardware.

    Args:
        w (torch.Tensor): Real-valued weights; sign taken at ``0`` (``w >= 0``
            maps to ``+1``, else ``-1``).

    Returns:
        torch.Tensor: Same shape as ``w``; forward values in ``{-1, +1}``
        carrying the gradient of ``clamp(w, -1, 1)`` (straight-through).
    """
    hard = torch.where(w >= 0, 1.0, -1.0)
    clip = torch.clamp(w, -1, 1)
    return clip + (hard - clip).detach()


def ternary_ste(w, delta=0.5):
    """Ternarize weights to {-1,0,+1} (forward), straight-through (backward).
    0 = no connection (drops out of the popcount -> sparser, cheaper circuit);
    +/-1 = pass / invert (free NOT gate). Lets the network learn connectivity.

    Args:
        w (torch.Tensor): Real-valued weights to ternarize.
        delta (float): Dead-zone half-width; ``w > delta`` maps to ``+1``,
            ``w < -delta`` maps to ``-1``, otherwise ``0``. Default ``0.5``.

    Returns:
        torch.Tensor: Same shape as ``w``; forward values in ``{-1, 0, +1}``
        carrying the gradient of ``clamp(w, -1, 1)`` (straight-through).
    """
    hard = torch.where(w > delta, 1.0, torch.where(w < -delta, -1.0, 0.0))
    clip = torch.clamp(w, -1, 1)
    return clip + (hard - clip).detach()


def gate_probs(logits, training, dim=-1):
    """Return per-gate selection weights. With Gumbel enabled + training, this
    is a hard one-hot (forward) carrying soft Gumbel-softmax gradient (ST).
    Otherwise a plain softmax.

    Args:
        logits (torch.Tensor): Gate logits over the 16 Boolean functions
            (reduced along ``dim``).
        training (bool): Whether in training mode. Hard one-hot paths
            (``GUMBEL``/``HARD_GATE``) only activate when ``True``.
        dim (int): Axis to softmax/one-hot over (the 16-gate axis).
            Default ``-1``.

    Returns:
        torch.Tensor: Same shape as ``logits``; a softmax distribution, or a
        straight-through one-hot when Gumbel (``GUMBEL["enabled"]``) or hard
        gating (``HARD_GATE["enabled"]``) is on during training.
    """
    if GUMBEL["enabled"] and training:
        return F.gumbel_softmax(logits, tau=GUMBEL["tau"], hard=True, dim=dim)
    soft = F.softmax(logits, dim=dim)
    if HARD_GATE["enabled"] and training:
        idx = soft.argmax(dim=dim, keepdim=True)
        hard = torch.zeros_like(soft).scatter_(dim, idx, 1.0)
        return soft + (hard - soft).detach()   # forward hard, backward soft (STE)
    return soft


class LogicLayer(nn.Module):
    """A single layer of logic gates with a configurable connectome.

    Args:
        in_dim (int): Number of binary inputs (size of previous layer / input
            vector).
        out_dim (int): Number of logic gates in this layer (output width).
        connectome (str): Input wiring strategy; one of ``"F"`` (fixed random
            wiring, no learnable connections), ``"L"`` (dense learnable
            connectome, softmax over all previous nodes), ``"TopK"`` (default;
            each input picks among ``k`` random candidates via a learnable
            softmax), ``"BlockTopK"`` (TopK with candidates drawn from a
            contiguous window for cache-local gathers), ``"ST"`` (unweighted
            sum-threshold: ``threshold(BN(popcount of k candidate wires))``),
            ``"STW"`` (weighted sum-threshold with binary ``{-1,+1}`` weights),
            ``"STT"`` (weighted sum-threshold with ternary ``{-1,0,+1}``
            weights for learnable sparsity).
        k (int): Number of candidate wires per gate-input for the ``"TopK"``,
            ``"BlockTopK"``, ``"ST"``, ``"STW"``, ``"STT"`` connectomes
            (clamped to ``in_dim``). Default ``8``. Ignored by ``"F"``/``"L"``.
        gate_eval (str): Soft gate evaluation path; ``"basis"`` (default,
            BasisProj: project gate distribution into the ``{1,A,B,A*B}`` basis
            and evaluate once) or ``"full"`` (FullEval: evaluate all 16 soft
            gate functions explicitly).
        seed (int | None): Seed for the RNG that fixes the random wiring
            (candidate/fixed indices). ``None`` (default) leaves wiring
            unseeded.
        residual (bool): If ``True``, add a hard-wired structural XOR skip
            ``out = XOR(gate(a,b), a)`` so a residual highway survives gate
            resampling. Requires matching in/out width. Default ``False``.
        window (int): For ``"BlockTopK"``, width of the contiguous candidate
            window; ``0`` (default) uses ``max(4*k, 256)`` (clamped to
            ``in_dim``). Ignored by other connectomes.
    """

    def __init__(self, in_dim, out_dim, connectome="TopK", k=8,
                 gate_eval="basis", seed=None, residual=False, window=0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.connectome = connectome
        self.k = k
        self.gate_eval = gate_eval
        # Structural residual: out = gate(a,b) XOR a (fixed XOR skip). Unlike
        # residual *init*, the skip is hard-wired so Gumbel can resample gates
        # without breaking the gradient/information highway.
        self.residual = residual

        # Per-gate logits over the 16 Boolean functions.
        self.gate_logits = nn.Parameter(torch.randn(out_dim, 16) * 0.1)

        gen = None
        if seed is not None:
            gen = torch.Generator().manual_seed(seed)

        if connectome == "F":
            # Fixed random wiring; no learnable connection params.
            idx_a = torch.randint(0, in_dim, (out_dim,), generator=gen)
            idx_b = torch.randint(0, in_dim, (out_dim,), generator=gen)
            self.register_buffer("idx_a", idx_a)
            self.register_buffer("idx_b", idx_b)

        elif connectome == "TopK":
            kk = min(k, in_dim)
            self.k = kk
            # Candidate indices per gate-input, chosen once at init.
            cand_a = torch.stack([torch.randperm(in_dim, generator=gen)[:kk]
                                  for _ in range(out_dim)])
            cand_b = torch.stack([torch.randperm(in_dim, generator=gen)[:kk]
                                  for _ in range(out_dim)])
            self.register_buffer("cand_a", cand_a)  # [out, k]
            self.register_buffer("cand_b", cand_b)
            self.register_buffer("cand_a_i32", cand_a.to(torch.int32))
            self.register_buffer("cand_b_i32", cand_b.to(torch.int32))
            # Learnable logits over the K candidates ~ N(0,1).
            self.conn_a = nn.Parameter(torch.randn(out_dim, kk))
            self.conn_b = nn.Parameter(torch.randn(out_dim, kk))
            self.use_triton_dense = _HAS_TRITON_DENSE

        elif connectome == "BlockTopK":
            # Block-structured Top-K: each gate draws K candidates from a
            # CONTIGUOUS window, and consecutive output gates have consecutive
            # windows -> the per-lane gathers/scatters become cache-local
            # (coalesced) instead of random. Same params/kernel as TopK.
            kk = min(k, in_dim)
            self.k = kk
            W = min(in_dim, window if window else max(4 * kk, 256))
            starts = (torch.arange(out_dim).float() / max(1, out_dim) *
                      max(1, in_dim - W)).long()              # monotonic in o
            cand_a = torch.stack([starts[o] + torch.randperm(W, generator=gen)[:kk]
                                  for o in range(out_dim)])
            cand_b = torch.stack([starts[o] + torch.randperm(W, generator=gen)[:kk]
                                  for o in range(out_dim)])
            self.register_buffer("cand_a", cand_a)
            self.register_buffer("cand_b", cand_b)
            self.register_buffer("cand_a_i32", cand_a.to(torch.int32))
            self.register_buffer("cand_b_i32", cand_b.to(torch.int32))
            self.conn_a = nn.Parameter(torch.randn(out_dim, kk))
            self.conn_b = nn.Parameter(torch.randn(out_dim, kk))
            self.use_triton_dense = _HAS_TRITON_DENSE

        elif connectome == "L":
            # Dense learnable connectome: softmax over all prev nodes.
            self.conn_a = nn.Parameter(torch.randn(out_dim, in_dim) * 0.1)
            self.conn_b = nn.Parameter(torch.randn(out_dim, in_dim) * 0.1)

        elif connectome == "ST":
            # Sum-threshold input (BNN-style, no weights): each gate input is
            # threshold(BatchNorm(popcount of k fixed-random candidate wires)).
            # BatchNorm stabilizes training and folds into the threshold at
            # inference (0 extra gates). A straight-through sign keeps the
            # forward pass discrete (train == inference).
            kk = min(k, in_dim)
            self.k = kk
            cand_a = torch.stack([torch.randperm(in_dim, generator=gen)[:kk]
                                  for _ in range(out_dim)])
            cand_b = torch.stack([torch.randperm(in_dim, generator=gen)[:kk]
                                  for _ in range(out_dim)])
            self.register_buffer("cand_a", cand_a)
            self.register_buffer("cand_b", cand_b)
            self.bn_a = nn.BatchNorm1d(out_dim)
            self.bn_b = nn.BatchNorm1d(out_dim)

        elif connectome == "STW":
            # Weighted sum-threshold (BNN binary perceptron): each gate input is
            # threshold(BN( sum_k sign(w_k) * (2*x_k-1) )) over k fixed-random
            # candidates. Binary +/-1 weights add only NOT gates in hardware but
            # restore learnable wire-selectivity (vs unweighted ST).
            kk = min(k, in_dim)
            self.k = kk
            cand_a = torch.stack([torch.randperm(in_dim, generator=gen)[:kk]
                                  for _ in range(out_dim)])
            cand_b = torch.stack([torch.randperm(in_dim, generator=gen)[:kk]
                                  for _ in range(out_dim)])
            self.register_buffer("cand_a", cand_a)
            self.register_buffer("cand_b", cand_b)
            self.w_a = nn.Parameter(torch.randn(out_dim, kk) * 0.1)
            self.w_b = nn.Parameter(torch.randn(out_dim, kk) * 0.1)
            self.bn_a = nn.BatchNorm1d(out_dim)
            self.bn_b = nn.BatchNorm1d(out_dim)

        elif connectome == "STT":
            # Ternary-weighted sum-threshold: weights in {-1,0,+1} (0 = no
            # connection -> learnable sparsity, fewer popcount inputs). Init
            # weights ~N(0,1) so ~1/3 each of {-1,0,+1} at start.
            kk = min(k, in_dim)
            self.k = kk
            cand_a = torch.stack([torch.randperm(in_dim, generator=gen)[:kk]
                                  for _ in range(out_dim)])
            cand_b = torch.stack([torch.randperm(in_dim, generator=gen)[:kk]
                                  for _ in range(out_dim)])
            self.register_buffer("cand_a", cand_a)
            self.register_buffer("cand_b", cand_b)
            self.w_a = nn.Parameter(torch.randn(out_dim, kk))
            self.w_b = nn.Parameter(torch.randn(out_dim, kk))
            self.bn_a = nn.BatchNorm1d(out_dim)
            self.bn_b = nn.BatchNorm1d(out_dim)
        else:
            raise ValueError(f"unknown connectome {connectome}")

        self.register_buffer("basis", BASIS_COEFFS.clone())
        self.register_buffer("truth", TRUTH_TABLES.clone())

    # -- input selection (soft) --------------------------------------------
    def _select_inputs_soft(self, x):
        """Return (a, b), each [batch, out_dim], soft input selections."""
        if self.connectome == "F":
            a = x[:, self.idx_a]
            b = x[:, self.idx_b]
        elif self.connectome in ("TopK", "BlockTopK"):
            wa = F.softmax(self.conn_a, dim=1)          # [out, k]
            wb = F.softmax(self.conn_b, dim=1)
            ga = x[:, self.cand_a]                       # [batch, out, k]
            gb = x[:, self.cand_b]
            a = torch.einsum("bok,ok->bo", ga, wa)
            b = torch.einsum("bok,ok->bo", gb, wb)
        else:  # L
            wa = F.softmax(self.conn_a, dim=1)          # [out, in]
            wb = F.softmax(self.conn_b, dim=1)
            a = x @ wa.t()                               # [batch, out]
            b = x @ wb.t()
        return a, b

    # -- gate evaluation (soft) --------------------------------------------
    def _eval_gates_soft(self, a, b):
        p = gate_probs(self.gate_logits, self.training, dim=1)  # [out, 16]
        if self.gate_eval == "basis":
            c = p @ self.basis                           # [out, 4]
            out = c[:, 0] + c[:, 1] * a + c[:, 2] * b + c[:, 3] * (a * b)
            if self.residual:                            # structural XOR skip
                out = out + a - 2.0 * out * a            # relaxed XOR(out, a)
            return out
        else:  # full evaluation of all 16 soft functions
            ab = a * b
            # f_i(a,b) using the same basis coefficients but evaluated per gate
            # gives the identical 16 functions; we compute each explicitly.
            funcs = torch.stack(
                [
                    torch.zeros_like(a),         # 0 False
                    ab,                          # 1 A and B
                    a - ab,                      # 2 A and not B
                    a,                           # 3 A
                    b - ab,                      # 4 not A and B
                    b,                           # 5 B
                    a + b - 2 * ab,              # 6 xor
                    a + b - ab,                  # 7 or
                    1 - a - b + ab,              # 8 nor
                    1 - a - b + 2 * ab,          # 9 xnor
                    1 - b,                       # 10 not B
                    1 - b + ab,                  # 11 A or not B
                    1 - a,                       # 12 not A
                    1 - a + ab,                  # 13 not A or B
                    1 - ab,                      # 14 nand
                    torch.ones_like(a),          # 15 True
                ],
                dim=0,
            )  # [16, batch, out]
            out = torch.einsum("ibo,oi->bo", funcs, p)
            return out

    def _select_inputs_st(self, x):
        """Sum-threshold inputs: a = threshold(BN(sum of k candidate wires))."""
        a_sum = x[:, self.cand_a].sum(dim=-1)   # [batch, out] popcount (soft)
        b_sum = x[:, self.cand_b].sum(dim=-1)
        a = ste_threshold(self.bn_a(a_sum))
        b = ste_threshold(self.bn_b(b_sum))
        return a, b

    def _select_inputs_stw(self, x, quant):
        """Weighted sum-threshold inputs (binary or ternary BNN perceptron)."""
        wa = quant(self.w_a)                     # [out, k] in {-1,+1} or {-1,0,1}
        wb = quant(self.w_b)
        a_pm = 2.0 * x[:, self.cand_a] - 1.0     # {0,1}->{-1,1}  [batch,out,k]
        b_pm = 2.0 * x[:, self.cand_b] - 1.0
        s_a = (a_pm * wa).sum(dim=-1)            # [batch, out]
        s_b = (b_pm * wb).sum(dim=-1)
        return ste_threshold(self.bn_a(s_a)), ste_threshold(self.bn_b(s_b))

    def forward(self, x):
        if self.connectome == "ST":
            a, b = self._select_inputs_st(x)
            return self._eval_gates_soft(a, b)
        if self.connectome == "STW":
            a, b = self._select_inputs_stw(x, sign_ste)
            return self._eval_gates_soft(a, b)
        if self.connectome == "STT":
            a, b = self._select_inputs_stw(x, ternary_ste)
            return self._eval_gates_soft(a, b)
        if (self.connectome in ("TopK", "BlockTopK")
                and getattr(self, "use_triton_dense", False)
                and x.is_cuda and not self.residual):
            wa = F.softmax(self.conn_a, dim=1)
            wb = F.softmax(self.conn_b, dim=1)
            p = gate_probs(self.gate_logits, self.training, dim=1)
            coef = (p @ self.basis).contiguous()
            return dense_logic(x, self.cand_a_i32, self.cand_b_i32, wa, wb, coef)
        a, b = self._select_inputs_soft(x)
        return self._eval_gates_soft(a, b)

    # -- hard (binarized) inference ----------------------------------------
    @torch.no_grad()
    def forward_hard(self, x):
        """x is a {0,1} (uint8/bool) tensor [batch, in_dim]."""
        x = x.to(torch.uint8)
        if self.connectome in ("ST", "STW", "STT"):
            if self.connectome == "ST":
                a_sum = x[:, self.cand_a].sum(dim=-1).float()
                b_sum = x[:, self.cand_b].sum(dim=-1).float()
            else:  # STW (+/-1) / STT ({-1,0,+1}) weighted sum
                if self.connectome == "STW":
                    wa = torch.where(self.w_a >= 0, 1.0, -1.0)
                    wb = torch.where(self.w_b >= 0, 1.0, -1.0)
                else:
                    wa = torch.where(self.w_a > 0.5, 1.0,
                                     torch.where(self.w_a < -0.5, -1.0, 0.0))
                    wb = torch.where(self.w_b > 0.5, 1.0,
                                     torch.where(self.w_b < -0.5, -1.0, 0.0))
                a_sum = ((2.0 * x[:, self.cand_a].float() - 1.0) * wa).sum(-1)
                b_sum = ((2.0 * x[:, self.cand_b].float() - 1.0) * wb).sum(-1)
            a = (self.bn_a(a_sum) > 0).to(torch.uint8)
            b = (self.bn_b(b_sum) > 0).to(torch.uint8)
            gate = self.gate_logits.argmax(dim=1)
            idx = (a.long() << 1) | b.long()
            tt = self.truth.to(x.device)[gate]
            return torch.gather(tt, 1, idx.t()).t().to(torch.uint8)
        if self.connectome == "F":
            ia, ib = self.idx_a, self.idx_b
        elif self.connectome in ("TopK", "BlockTopK"):
            sel_a = self.conn_a.argmax(dim=1)            # [out]
            sel_b = self.conn_b.argmax(dim=1)
            ia = self.cand_a[torch.arange(self.out_dim), sel_a]
            ib = self.cand_b[torch.arange(self.out_dim), sel_b]
        else:  # L
            ia = self.conn_a.argmax(dim=1)
            ib = self.conn_b.argmax(dim=1)
        a = x[:, ia]                                     # [batch, out]
        b = x[:, ib]
        gate = self.gate_logits.argmax(dim=1)            # [out]
        # truth table index = a*2 + b
        idx = (a.long() << 1) | b.long()                 # [batch, out] in 0..3
        tt = self.truth.to(x.device)[gate]               # [out, 4]
        out = torch.gather(tt, 1, idx.t()).t()           # [batch, out]
        if self.residual:                                # structural XOR skip
            out = out ^ a
        return out.to(torch.uint8)


class GroupSum(nn.Module):
    """Aggregate logic outputs into class logits by group counting.

    Splits the ``width`` logic outputs into ``num_classes`` equal blocks and
    sums each block (divided by ``tau``) to form per-class logits.

    Args:
        num_classes (int): Number of output classes / blocks; the input width
            must be divisible by it. Default ``10``.
        tau (float): Temperature; each class logit is its block sum divided by
            ``tau``. Default ``1.0``.
    """

    def __init__(self, num_classes=10, tau=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.tau = tau

    def forward(self, x):
        # x: [batch, width]; width must be divisible by num_classes.
        b, w = x.shape
        x = x.view(b, self.num_classes, w // self.num_classes)
        return x.sum(dim=2) / self.tau

    @torch.no_grad()
    def forward_hard(self, x):
        b, w = x.shape
        x = x.view(b, self.num_classes, w // self.num_classes)
        return x.sum(dim=2).float()


class LogicNet(nn.Module):
    """Stack of logic layers followed by a GroupSum (or learned) head.

    Args:
        in_dim (int): Number of binary input features.
        width (int): Logic gates per layer (output width of every layer).
        depth (int): Number of stacked :class:`LogicLayer` layers.
        num_classes (int): Number of output classes. Default ``10``.
        connectome (str): Per-layer input wiring; one of ``"F"`` (fixed
            random), ``"L"`` (dense learnable softmax), ``"TopK"`` (sparse,
            default), ``"BlockTopK"``, ``"ST"``, ``"STW"``, ``"STT"``.
            See :class:`LogicLayer` for the meaning of each.
        k (int): Candidates per gate-input for the sparse connectomes
            (``"TopK"``/``"BlockTopK"``/``"ST"``/``"STW"``/``"STT"``).
            Default ``8``.
        tau (float): GroupSum temperature (logits are the block sum / ``tau``).
            Used only by the ``"groupsum"`` decoder. Default ``1.0``.
        gate_eval (str): Soft gate evaluation path; ``"basis"`` (default) or
            ``"full"``. See :class:`LogicLayer`.
        seed (int | None): Base seed for per-layer random wiring; layer ``i``
            uses ``seed*1000 + i``. ``None`` leaves wiring unseeded.
            Default ``0``.
        residual (bool): Enable the structural XOR skip in same-width layers
            (active from the first layer whose input width equals ``width``).
            Default ``False``.
        wire_residual (float): Fraction of each same-width layer's outputs to
            hard-wire as identity copies of its input (a grad-1 highway,
            ``0`` gates); rounded to ``int(width * wire_residual)``.
            Default ``0.0`` (disabled).
        decoder (str): Output head; one of ``"groupsum"`` (default, fixed
            block-sum :class:`GroupSum`), ``"linear"`` (learned
            ``Linear(width -> num_classes)``), ``"linfull"`` (full
            ``Linear`` initialised to exactly GroupSum, then learns
            deviations; requires ``width % num_classes == 0``), ``"sumlinear"``
            (sum features to a 256-d state -> BatchNorm -> linear; requires
            ``width % 256 == 0``), ``"ternary"`` (per-feature, per-class
            ``{-1,0,+1}`` ternarized weights generalizing GroupSum).
    """

    def __init__(self, in_dim, width, depth, num_classes=10,
                 connectome="TopK", k=8, tau=1.0, gate_eval="basis",
                 seed=0, residual=False, wire_residual=0.0, decoder="groupsum"):
        super().__init__()
        layers = []
        d_in = in_dim
        for i in range(depth):
            # residual XOR-skip needs matching in/out width: enable from layer 1
            res = residual and (d_in == width)
            layers.append(
                LogicLayer(d_in, width, connectome=connectome, k=k,
                           gate_eval=gate_eval, residual=res,
                           seed=None if seed is None else seed * 1000 + i)
            )
            d_in = width
        self.layers = nn.ModuleList(layers)
        # decoder: "groupsum" (fixed block-sum) or "linear" (learned FC over the
        # width-dim logic feature state). MNIST showed linear >> groupsum.
        self.decoder_kind = decoder
        if decoder == "linear":
            self.dec = nn.Linear(width, num_classes)
        elif decoder == "ternary":
            # ternary head: each feature -> EVERY class with a learned weight in
            # {-1,0,+1} (ternarized via STE). Generalizes GroupSum (which is the
            # fixed {feature->1 class, +1} case). Deployable as a signed popcount.
            self.dec_w = nn.Parameter(torch.randn(num_classes, width) * 0.1)
        elif decoder == "linfull":
            # full Linear(width->classes) initialised to EXACTLY GroupSum
            # (block-mean per class), so it starts >= GroupSum and learns
            # deviations from there -> can't lose, gains where a learned readout
            # helps. No sum-to-256 bottleneck.
            assert width % num_classes == 0
            self.dec = nn.Linear(width, num_classes)
            g = width // num_classes
            with torch.no_grad():
                self.dec.weight.zero_()
                for cls in range(num_classes):
                    self.dec.weight[cls, cls*g:(cls+1)*g] = 1.0 / g
                self.dec.bias.zero_()
        elif decoder == "sumlinear":
            # sum the binary logic features into a 256-d summed-activity hidden
            # state (group-count, like GroupSum but to 256) -> BatchNorm (scale
            # stability) -> linear decoder. No nonlinearity: a true linear head.
            self.dec_hidden = 256
            assert width % self.dec_hidden == 0, "width must be divisible by 256"
            self.dec_bn = nn.BatchNorm1d(self.dec_hidden)
            self.dec_lin = nn.Linear(self.dec_hidden, num_classes)
        else:
            assert width % num_classes == 0, "width must be divisible by classes"
            self.head = GroupSum(num_classes=num_classes, tau=tau)
        self.width = width
        self.depth = depth
        self.connectome = connectome
        self.k = k
        # wire residual: first `wire_r` output nodes of each same-width layer are
        # hardwired copies of the layer input -> unconditional grad-1 identity
        # highway (0 gates, Gumbel-safe, no discretization gap).
        self.wire_r = int(width * wire_residual)

    def _decode(self, x):
        if self.decoder_kind == "ternary":
            if DEC_FEATURE_STE:        # STE features -> head guides gates toward
                x = (x > 0.5).float() + (x - x.detach())   # binary multi-class feats
            return F.linear(x, ternary_ste(self.dec_w))   # {-1,0,+1} weights
        if self.decoder_kind == "linear":
            return self.dec(x)
        if self.decoder_kind == "linfull":
            if DEC_FEATURE_STE:
                x = (x > 0.5).float() + (x - x.detach())
            return self.dec(x)
        # sumlinear: (optionally STE-binarize the activations so the summed
        # input is binary in TRAINING too -> matches inference), sum to 256 ->
        # BN -> linear. Gumbel hardens the gate CHOICE; this hardens the
        # activation VALUES -> together they close the soft->hard gap.
        if DEC_FEATURE_STE:
            x = (x > 0.5).float() + (x - x.detach())     # straight-through
        x = x.view(x.shape[0], self.dec_hidden, -1).sum(2)
        return self.dec_lin(self.dec_bn(x))

    def forward(self, x):
        for layer in self.layers:
            y = layer(x)
            if self.wire_r and x.shape[1] == y.shape[1]:
                y = torch.cat([x[:, :self.wire_r], y[:, self.wire_r:]], dim=1)
            x = y
        if self.decoder_kind in ("linear", "sumlinear", "linfull", "ternary"):
            return self._decode(x)
        return self.head(x)

    @torch.no_grad()
    def forward_hard(self, x):
        for layer in self.layers:
            y = layer.forward_hard(x)
            if self.wire_r and x.shape[1] == y.shape[1]:
                y = torch.cat([x[:, :self.wire_r], y[:, self.wire_r:]], dim=1)
            x = y
        if self.decoder_kind in ("linear", "sumlinear", "linfull", "ternary"):
            return self._decode(x.float())
        return self.head.forward_hard(x)

    def num_gates(self):
        return self.width * self.depth
