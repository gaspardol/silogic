"""Bit-packing primitives for batched Boolean inference.

Layout: activations are ``int64`` arrays of shape ``[dim, n_words]`` where
``n_words = ceil(B / 64)``.  Bit ``p`` of word ``w`` at column ``d`` holds the
activation of sample ``b = 64*w + p`` for dimension ``d``.  int64 is used
(not uint64) for PyTorch / Triton compatibility; two's-complement bitwise
semantics are identical to uint64.

Optimisation notes
------------------
* ``pack_bits`` uses ``np.packbits`` (C-level, vectorised over bytes) in
  little-endian bit order so that sample 0 = bit 0 = LSB of the first int64.
* ``unpack_bits`` uses ``np.unpackbits`` (also C-level) in little-endian bit
  order; this is ~8-10× faster than the shift-and-OR loop equivalent.
* ``group_sum_packed`` computes GroupSum class counts entirely in the packed
  domain without materialising the full ``[B, dim]`` boolean array: it unpacks
  per-class slices of height ``group_size`` and reduces them, keeping peak
  memory proportional to ``group_size`` rather than ``dim``.
"""
from __future__ import annotations

import numpy as np


# ── packing helpers ───────────────────────────────────────────────────────────

def n_words(B: int) -> int:
    """Number of 64-bit words needed to hold B samples."""
    return (B + 63) >> 6


def pack_bits(x: np.ndarray) -> np.ndarray:
    """Pack ``[B, dim]`` uint8/bool → ``[dim, ceil(B/64)]`` int64.

    Uses ``np.packbits`` with ``bitorder='little'`` so that sample ``b`` maps
    to bit ``b % 64`` of word ``b // 64`` in the output.  The last word is
    zero-padded when ``B`` is not a multiple of 64.
    """
    B, dim = x.shape
    nw = n_words(B)
    pad = nw * 64 - B
    if pad:
        x = np.concatenate([x, np.zeros((pad, dim), dtype=x.dtype)], axis=0)
    # Transpose to [dim, nw*64] so packbits packs along the sample axis
    x_t = np.ascontiguousarray(x.T.astype(np.uint8))              # [dim, nw*64]
    # Pack 8 samples per byte along axis=1: [dim, nw*8] uint8
    packed_u8 = np.packbits(x_t, axis=1, bitorder='little')        # [dim, nw*8]
    # View as int64: 8 consecutive bytes per column → one int64 word [dim, nw]
    return np.ascontiguousarray(packed_u8).view(np.int64)           # [dim, nw]


def unpack_bits(packed: np.ndarray, B: int) -> np.ndarray:
    """Unpack ``[dim, n_words]`` int64 → ``[B, dim]`` uint8.

    Uses ``np.unpackbits`` (C-level) in little-endian order — ~8× faster than
    the equivalent shift-and-mask loop for large ``dim``.
    """
    dim, nw = packed.shape
    # View int64 as uint8: 8 bytes per word → [dim, nw*8]
    as_u8 = packed.view(np.uint8)                                  # [dim, nw*8]
    # Unpack 8 bits per byte (little-endian: bit0 = LSB of first byte = sample 0)
    bits = np.unpackbits(as_u8, axis=1, bitorder='little')         # [dim, nw*64]
    return np.ascontiguousarray(bits[:, :B].T).astype(np.uint8)    # [B, dim]


def group_sum_packed(
    x_packed: np.ndarray,
    B: int,
    num_classes: int,
    tau: float = 1.0,
) -> np.ndarray:
    """Compute GroupSum class logits directly from packed activations.

    Avoids materialising the full ``[B, dim]`` boolean matrix by unpacking
    one class-group slice at a time (peak memory = ``[B, group_size]``).

    Parameters
    ----------
    x_packed:
        ``[dim, n_words]`` int64 — last-layer packed activations.
    B:
        Actual batch size (strips zero-padding from the last word).
    num_classes:
        Number of output classes (``GroupSum`` block count).
    tau:
        GroupSum temperature (logits are divided by ``tau``).

    Returns
    -------
    logits:
        ``[B, num_classes]`` float32.
    """
    dim, nw = x_packed.shape
    gs = dim // num_classes
    logits = np.empty((B, num_classes), dtype=np.float32)
    for c in range(num_classes):
        # Slice one class's neurons: [gs, nw] int64
        group = x_packed[c * gs:(c + 1) * gs]
        # View as uint8: [gs, nw*8], unpack bits: [gs, nw*64]
        bits_c = np.unpackbits(group.view(np.uint8), axis=1, bitorder='little')
        # Sum neurons (axis=0) for each sample (column) → [nw*64]
        # Then slice to B and cast to float32
        logits[:, c] = bits_c[:, :B].sum(axis=0, dtype=np.int32).astype(np.float32)
    return logits / tau


def mask_last_word(packed: np.ndarray, B: int) -> np.ndarray:
    """Zero out padding bits in the last word (in-place if possible)."""
    nw = packed.shape[1]
    remainder = B % 64
    if remainder == 0:
        return packed
    mask = np.int64((1 << remainder) - 1)
    packed[:, nw - 1] &= mask
    return packed


# ── gate arithmetic ───────────────────────────────────────────────────────────

# The 16 two-input Boolean functions, ordered to match TRUTH_TABLES / BASIS_COEFFS.
# Each op takes (a, b) int64 arrays of shape [n, nw] and returns int64 [n, nw].

_GATE16_OPS = [
    lambda a, b: np.zeros_like(a),             # 0  False
    lambda a, b: a & b,                         # 1  AND
    lambda a, b: a & ~b,                        # 2  A ∧ ¬B
    lambda a, b: a.copy(),                      # 3  A (pass-through)
    lambda a, b: ~a & b,                        # 4  ¬A ∧ B
    lambda a, b: b.copy(),                      # 5  B (pass-through)
    lambda a, b: a ^ b,                         # 6  XOR
    lambda a, b: a | b,                         # 7  OR
    lambda a, b: ~(a | b),                      # 8  NOR
    lambda a, b: ~(a ^ b),                      # 9  XNOR
    lambda a, b: ~b,                            # 10 ¬B
    lambda a, b: a | ~b,                        # 11 A ∨ ¬B
    lambda a, b: ~a,                            # 12 ¬A
    lambda a, b: ~a | b,                        # 13 ¬A ∨ B
    lambda a, b: ~(a & b),                      # 14 NAND
    lambda a, b: np.full_like(a, np.int64(-1)), # 15 True (all bits set)
]

# Pre-built 4-bit-code → gate16-index reverse lookup (CODE_TO_GATE16[code] = gate).
# truth_table_code = bit0 | (bit1<<1) | (bit2<<2) | (bit3<<3) where
# bit_k = gate output for input addr k = (a<<1)|b.
_CODE_TO_GATE16: np.ndarray


def _build_code_table() -> np.ndarray:
    from ..functional import TRUTH_TABLES
    tt = TRUTH_TABLES.numpy()                 # [16, 4] uint8
    lut = np.zeros(16, dtype=np.int32)
    for g in range(16):
        code = int(tt[g, 0]) | (int(tt[g, 1]) << 1) | (int(tt[g, 2]) << 2) | (int(tt[g, 3]) << 3)
        lut[code] = g
    return lut


def code_to_gate16() -> np.ndarray:
    """Return the 16-entry code→gate16-index lookup (built lazily)."""
    global _CODE_TO_GATE16
    try:
        return _CODE_TO_GATE16
    except NameError:
        _CODE_TO_GATE16 = _build_code_table()
        return _CODE_TO_GATE16


def apply_gate(a: np.ndarray, b: np.ndarray, gate: int) -> np.ndarray:
    """Apply one of the 16 two-input Boolean gates to packed int64 arrays."""
    return _GATE16_OPS[gate](a, b)
