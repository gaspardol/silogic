"""Gate algebra and straight-through primitives shared across every node type.

This is the single home for the stateless building blocks that the logic layers
compose: the 16 two-input Boolean functions in the ``{1, A, B, A*B}`` basis,
their hard truth tables, the straight-through estimators (binary threshold,
sign, ternary, feature binarization), residual-init logits, and the n-input
hypercube/Walsh basis builders used by the LUT/WARP nodes.

Nothing here holds learnable state; the layers in :mod:`silogic.nodes` and
:mod:`silogic.connectomes` import these helpers instead of re-deriving them.
"""
import math

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# The 16 two-input Boolean functions in the basis {1, A, B, A*B}.
# Row i = coefficients [c1, c2, c3, c4] with
#   gate_i(a, b) = c1*1 + c2*a + c3*b + c4*(a*b).
# Order matches the standard DiffLogic / BasisProj ordering (paper Eq. 4).
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

# index of the 'A' (pass-through of the first input) gate, used for residual init
GATE_A = 3


def _truth_tables():
    """Derive the 16x4 hard truth tables from the basis coefficients.

    Columns correspond to ``(a, b) = (0,0), (0,1), (1,0), (1,1)``.
    Returns a ``{0,1}`` uint8 tensor of shape ``[16, 4]``.
    """
    c = BASIS_COEFFS
    tt = torch.stack(
        [
            c[:, 0],                                  # (0,0)
            c[:, 0] + c[:, 2],                        # (0,1)
            c[:, 0] + c[:, 1],                        # (1,0)
            c[:, 0] + c[:, 1] + c[:, 2] + c[:, 3],    # (1,1)
        ],
        dim=1,
    )
    return tt.round().to(torch.uint8)


TRUTH_TABLES = _truth_tables()  # [16, 4]


# ---------------------------------------------------------------------------
# Straight-through estimators
# ---------------------------------------------------------------------------
def ste_threshold(s):
    """Straight-through binary threshold: forward = ``(s > 0)`` in ``{0,1}``,
    backward = gradient of ``sigmoid(s)``.

    Args:
        s (torch.Tensor): Real-valued pre-activation; thresholded at ``0``.

    Returns:
        torch.Tensor: Same shape as ``s``; forward values in ``{0, 1}`` carrying
        the ``sigmoid(s)`` gradient (straight-through).
    """
    hard = (s > 0).float()
    soft = torch.sigmoid(s)
    return hard + soft - soft.detach()


def sign_ste(w):
    """Binarize weights to ``{-1,+1}`` (forward) with clipped straight-through
    gradient (backward), BNN-style. A +/-1 weight on a binary wire = pass or
    invert = a free NOT gate in hardware.

    Args:
        w (torch.Tensor): Real-valued weights; ``w >= 0`` maps to ``+1``, else ``-1``.

    Returns:
        torch.Tensor: Forward values in ``{-1, +1}`` carrying the gradient of
        ``clamp(w, -1, 1)``.
    """
    hard = torch.where(w >= 0, 1.0, -1.0)
    clip = torch.clamp(w, -1, 1)
    return clip + (hard - clip).detach()


def ternary_ste(w, delta=0.5):
    """Ternarize weights to ``{-1,0,+1}`` (forward), straight-through (backward).
    ``0`` = no connection (drops out of the popcount -> sparser circuit);
    ``+/-1`` = pass / invert (free NOT gate).

    Args:
        w (torch.Tensor): Real-valued weights to ternarize.
        delta (float): Dead-zone half-width. Default ``0.5``.

    Returns:
        torch.Tensor: Forward values in ``{-1, 0, +1}`` carrying the gradient of
        ``clamp(w, -1, 1)``.
    """
    hard = torch.where(w > delta, 1.0, torch.where(w < -delta, -1.0, 0.0))
    clip = torch.clamp(w, -1, 1)
    return clip + (hard - clip).detach()


def binarize_ste(x, thresh=0.5):
    """Straight-through binarization of relaxed logic features in ``[0,1]``:
    forward = ``(x > thresh)`` in ``{0,1}``, backward = identity. Lets a learned
    decoder train on the same ``{0,1}`` features it sees at inference (BNN-style),
    closing the soft->hard discretization gap.
    """
    return (x > thresh).float() + (x - x.detach())


def residual_logit(p, tau=1.0):
    """Logit ``z`` such that ``sigmoid(z) == p`` (used for residual init)."""
    p = min(max(float(p), 1e-4), 1 - 1e-4)
    return tau * math.log(p / (1 - p))


# ---------------------------------------------------------------------------
# Gate selection over the 16 functions (softmax / Gumbel-ST / hard-gate-ST)
# ---------------------------------------------------------------------------
def gate_probs(logits, training, dim=-1, gate_select="softmax", gumbel_tau=1.0):
    """Per-gate selection weights over the 16 Boolean functions.

    Args:
        logits (torch.Tensor): Gate logits reduced along ``dim``.
        training (bool): The hard paths only act when ``True``.
        dim (int): Axis to softmax/one-hot over. Default ``-1``.
        gate_select (str): ``"softmax"`` (default), ``"gumbel"`` (hard argmax
            forward + Gumbel-softmax gradient; "Mind the Gap", NeurIPS 2025), or
            ``"hard"`` (deterministic argmax one-hot forward + softmax gradient).
        gumbel_tau (float): Temperature for ``gate_select="gumbel"``. Default ``1.0``.
    """
    if training and gate_select == "gumbel":
        return F.gumbel_softmax(logits, tau=gumbel_tau, hard=True, dim=dim)
    soft = F.softmax(logits, dim=dim)
    if training and gate_select == "hard":
        idx = soft.argmax(dim=dim, keepdim=True)
        hard = torch.zeros_like(soft).scatter_(dim, idx, 1.0)
        return soft + (hard - soft).detach()   # forward hard, backward soft (STE)
    return soft


# ---------------------------------------------------------------------------
# n-input LUT basis builders (shared by the multilinear and Walsh nodes)
# ---------------------------------------------------------------------------
def corner_bits(n, dtype=torch.float32):
    """The ``[2^n, n]`` table of hypercube corner bit-patterns (bit j of row p)."""
    P = 2 ** n
    return torch.tensor([[(p >> j) & 1 for j in range(n)] for p in range(P)],
                        dtype=dtype)


def walsh_monomials(u):
    """Walsh monomials of ``u`` ``[B, out, n]`` -> ``[B, out, 2^n]``.

    Column ``i`` is ``prod_{j in bits(i)} u_j`` (with the empty product = 1),
    built incrementally so column ordering matches :func:`corner_bits`.
    """
    phi = torch.ones(u.shape[0], u.shape[1], 1, device=u.device, dtype=u.dtype)
    for j in range(u.shape[2]):
        phi = torch.cat([phi, phi * u[:, :, j:j + 1]], dim=2)
    return phi
