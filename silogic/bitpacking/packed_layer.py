"""Bitpacked inference layers for FC logic gates, conv tree gates, and output heads.

Two activation layouts are used:

**FC layers** (``[dim, n_words]`` int64)
    ``n_words = ceil(B / 64)``.  Sample ``b`` lives at bit ``b%64`` of word
    ``b//64`` for every dimension in parallel.  Gates are pre-grouped by type
    so each group fires one numpy vectorised bitwise op over
    ``[n_group, n_words]`` — no Python branching per neuron or per sample.

**Conv layers** (``[dim, L, nw_B]`` int64) — *B-packing*
    ``nw_B = ceil(B / 64)``, ``L = H * W`` spatial positions.  The batch
    dimension is packed but the spatial dimension stays explicit.  This lets
    :func:`packed_unfold`, :class:`BitpackedConvTreeLayer`, and
    :func:`packed_or_pool` all operate without ever expanding the batch —
    only a single pack at the image input and a reshape at the FC boundary.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .ops import apply_gate, n_words as _nwords


# ── two-input gate layer ──────────────────────────────────────────────────────

class BitpackedFCLayer:
    """Hard-committed, bitpacked version of one two-input :class:`~silogic.layers.LogicLayer`.

    Parameters
    ----------
    gate_types:
        ``[out_dim]`` int32, values 0-15 (one of the 16 Boolean functions).
    a_idx:
        ``[out_dim]`` int32 — first operand index into the input activations.
    b_idx:
        ``[out_dim]`` int32 — second operand index.

    Design
    ------
    On construction the outputs are **sorted by gate type** into up to 16
    groups.  During :meth:`forward` each group fires *one* numpy vectorised
    bitwise op over ``[n_group, n_words]``, so the inner loop is fully SIMD
    (no Python-level branching per neuron or per sample).
    """

    def __init__(
        self,
        gate_types: np.ndarray,
        a_idx: np.ndarray,
        b_idx: np.ndarray,
    ) -> None:
        self.out_dim = len(gate_types)
        self.gate_types = gate_types.astype(np.int32)
        self.a_idx = a_idx.astype(np.int32)
        self.b_idx = b_idx.astype(np.int32)

        # Pre-group outputs by gate type (up to 16 groups).
        # Within each group, sort by (a_idx, b_idx) so the numpy gather walks
        # the input array as sequentially as possible → better L1/L2 locality.
        self._groups: List[Tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []
        for g in range(16):
            mask = self.gate_types == g
            if not mask.any():
                continue
            out_idx = np.where(mask)[0].astype(np.int32)
            ai = self.a_idx[mask]
            bi = self.b_idx[mask]
            # Sort by (ai, bi) for cache-friendly sequential gather
            order = np.lexsort((bi, ai))
            self._groups.append((
                g,
                out_idx[order],
                ai[order],
                bi[order],
            ))

    # Statistics helpers -------------------------------------------------
    @property
    def gate_histogram(self) -> dict:
        return {g: len(oi) for g, oi, _, _ in self._groups}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Bitpacked forward pass.

        Parameters
        ----------
        x:
            ``[in_dim, n_words]`` int64 — input activations.

        Returns
        -------
        out:
            ``[out_dim, n_words]`` int64 — output activations.
        """
        nw = x.shape[1]
        out = np.empty((self.out_dim, nw), dtype=np.int64)

        for g, out_idx, ai, bi in self._groups:
            # Special-case trivial gates to avoid unnecessary gathers
            if g == 0:   # False — constant 0, no gather needed
                out[out_idx] = np.int64(0)
            elif g == 3: # A — pass-through, one gather
                out[out_idx] = x[ai]
            elif g == 5: # B — pass-through, one gather
                out[out_idx] = x[bi]
            elif g == 15: # True — constant all-1s, no gather needed
                out[out_idx] = np.int64(-1)
            else:
                a_words = x[ai]       # [n_group, nw] — fancy-index gather
                b_words = x[bi]       # [n_group, nw]
                out[out_idx] = apply_gate(a_words, b_words, g)

        return out


# ── n-input LUT layer ────────────────────────────────────────────────────────

class BitpackedLUTLayer:
    """Bitpacked inference for n-input LUT nodes (arity ≥ 2).

    Uses the *sum-of-products* approach: for each truth-table entry ``e`` that
    is ``1``, computes the mask of samples whose input combination equals ``e``
    and ORs it into the output.  Vectorised over the full output dimension
    inside each of the ``2**arity`` passes.
    """

    def __init__(
        self,
        truth_tables: np.ndarray,   # [out_dim, 2^arity] bool
        input_indices: np.ndarray,  # [out_dim, arity] int32
    ) -> None:
        out_dim, lut_size = truth_tables.shape
        self.out_dim = out_dim
        self.arity = int(np.log2(lut_size))
        self.truth_tables = truth_tables.astype(bool)
        self.input_indices = input_indices.astype(np.int32)

        # Pre-group outputs by which truth-table entries are active (vectorise
        # outer loop over outputs for each entry).
        self._entry_groups: List[Tuple[int, np.ndarray, np.ndarray]] = []
        for e in range(lut_size):
            active = np.where(self.truth_tables[:, e])[0].astype(np.int32)
            if len(active):
                self._entry_groups.append((e, active, self.input_indices[active]))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Bitpacked forward pass.

        Parameters
        ----------
        x:
            ``[in_dim, n_words]`` int64.

        Returns
        -------
        out:
            ``[out_dim, n_words]`` int64.
        """
        nw = x.shape[1]
        out = np.zeros((self.out_dim, nw), dtype=np.int64)

        for e, active_out, active_idx in self._entry_groups:
            # active_idx: [n_active, arity]
            # Compute samples-mask for input combination == e
            # match: [n_active, nw] — samples where ALL arity inputs match entry e
            match = np.full((len(active_out), nw), np.int64(-1), dtype=np.int64)
            for j in range(self.arity):
                inp = x[active_idx[:, j]]         # [n_active, nw]
                if (e >> j) & 1:
                    match &= inp
                else:
                    match &= ~inp
            out[active_out] |= match

        return out


# ── conv spatial helpers ──────────────────────────────────────────────────────

def pack_conv_input(x: np.ndarray, nw_B: int) -> np.ndarray:
    """Pack ``[B, C, H, W]`` uint8 → ``[C, H*W, nw_B]`` int64 (B-packing).

    Packs *only* the batch dimension so spatial positions remain explicit.
    This is the input/output currency of :class:`BitpackedConvTreeLayer` and
    the conv pipeline in :class:`~.packed_model.BitpackedConvNet`.

    ``result[c, l, word]`` holds the packed activations of all B samples
    for channel ``c`` at spatial position ``l = h*W + w``, sample ``b``
    living at bit ``b%64`` of word ``b//64``.

    Parameters
    ----------
    x:
        ``[B, C, H, W]`` uint8 input.
    nw_B:
        Number of int64 words per (channel, spatial) pair — ``ceil(B / 64)``.
    """
    from .ops import pack_bits
    B, C, H, W = x.shape
    L = H * W
    # Rearrange to [B, C*L] so that pack_bits sees the standard [B, dim] layout
    x_CL_B = np.ascontiguousarray(
        x.reshape(B, C, L).transpose(1, 2, 0).reshape(C * L, B)
    )                                               # [C*L, B] — contiguous
    packed   = pack_bits(x_CL_B.T)                 # [B, C*L] → [C*L, nw_B]
    return packed.reshape(C, L, nw_B)              # [C, L, nw_B]


def unpack_conv_output(x_3d: np.ndarray, B: int, H: int, W: int) -> np.ndarray:
    """Unpack ``[C, H*W, nw_B]`` int64 → ``[B, C, H, W]`` uint8.

    Inverse of :func:`pack_conv_input`.  Used for the fallback (uint8) path
    when a conv block cannot be bitpacked.
    """
    from .ops import unpack_bits
    C, L, nw_B = x_3d.shape
    bits = unpack_bits(x_3d.reshape(C * L, nw_B), B)   # [B, C*L]
    return bits.reshape(B, C, H, W)


def packed_unfold(
    x_3d: np.ndarray,
    kh: int,
    kw: int,
    stride: int,
    padding: int,
    H_in: int,
    W_in: int,
) -> np.ndarray:
    """Spatial unfold of B-packed activations without materialising the batch.

    Parameters
    ----------
    x_3d:
        ``[C, L_in, nw_B]`` int64 where ``L_in = H_in * W_in``.
    kh, kw, stride, padding:
        Convolution kernel parameters.
    H_in, W_in:
        Input spatial dimensions.

    Returns
    -------
    unfolded:
        ``[C*kh*kw, L_out, nw_B]`` int64 — the unrolled receptive-field
        packed representation.  Out-of-bounds positions (zero-padding) are
        filled with int64 zero (all samples inactive = 0).

    Notes
    -----
    Equivalent to ``F.unfold(x_uint8, ...)`` but stays in the B-packed domain,
    avoiding any expansion of the B dimension.
    """
    C, L_in, nw_B = x_3d.shape
    H_out = (H_in + 2 * padding - kh) // stride + 1
    W_out = (W_in + 2 * padding - kw) // stride + 1
    L_out = H_out * W_out
    P     = C * kh * kw

    hy_out = np.arange(L_out, dtype=np.int32) // W_out
    hx_out = np.arange(L_out, dtype=np.int32) % W_out

    # For each of the P unrolled slots (channel c, offset dy, dx)
    c_arr  = np.arange(C,  dtype=np.int32).repeat(kh * kw)
    dy_arr = np.tile(np.arange(kh, dtype=np.int32).repeat(kw), C)
    dx_arr = np.tile(np.arange(kw, dtype=np.int32), C * kh)

    hy_in = hy_out[None, :] * stride - padding + dy_arr[:, None]   # [P, L_out]
    hx_in = hx_out[None, :] * stride - padding + dx_arr[:, None]   # [P, L_out]

    valid = (hy_in >= 0) & (hy_in < H_in) & (hx_in >= 0) & (hx_in < W_in)
    # Sentinel L_in → zero-padding row appended below
    l_in  = np.where(valid, hy_in * W_in + hx_in, L_in)            # [P, L_out]

    # Append one all-zero row for out-of-bounds positions
    x_pad = np.concatenate(
        [x_3d, np.zeros((C, 1, nw_B), dtype=np.int64)], axis=1
    )                                                                # [C, L_in+1, nw_B]

    # Gather [P, L_out, nw_B] via broadcasting fancy index
    return x_pad[c_arr[:, None], l_in, :]


def packed_or_pool(x_3d: np.ndarray, H: int, W: int, pool_size: int = 2) -> np.ndarray:
    """OR-pool on B-packed activations, staying in the packed domain.

    Parameters
    ----------
    x_3d:
        ``[C, H*W, nw_B]`` int64.
    H, W:
        Spatial dimensions of the input.
    pool_size:
        Pooling window side length (square; default 2).

    Returns
    -------
    pooled:
        ``[C, (H//ps)*(W//ps), nw_B]`` int64.  All 64 samples are OR'd
        simultaneously via a single int64 bitwise-or per word.
    """
    C, L_in, nw_B = x_3d.shape
    ps    = pool_size
    H_out = H // ps
    W_out = W // ps
    L_out = H_out * W_out

    hy_out = np.arange(L_out, dtype=np.int32) // W_out
    hx_out = np.arange(L_out, dtype=np.int32) % W_out

    result = np.zeros((C, L_out, nw_B), dtype=np.int64)
    for dy in range(ps):
        for dx in range(ps):
            l_in = (hy_out * ps + dy) * W + (hx_out * ps + dx)   # [L_out]
            result |= x_3d[:, l_in, :]                             # [C, L_out, nw_B]

    return result


# ── convolutional gate-tree layer ────────────────────────────────────────────

class BitpackedConvTreeLayer:
    """Bitpacked gate16 tree layer for spatial (conv) inputs — B-packing.

    Activation format: ``[dim, L, nw_B]`` int64 where:

    * ``dim`` = channels (P = C*kh*kw for unrolled input; n_out for output).
    * ``L``   = spatial positions (H_out * W_out).
    * ``nw_B`` = ``ceil(B / 64)`` — packs *only* the batch dimension.

    This avoids inter-layer pack/unpack: after the first pack the data stays
    in the ``[dim, L, nw_B]`` format through all conv blocks, OR-pools, and
    the packed-unfold steps.  The FC head receives a reshape ``[n_out*L, nw_B]``
    directly, with no additional repack.

    Parameters
    ----------
    leaf_idx:
        ``[n_out, n_leaves]`` int32 — flat P-index into the unrolled
        receptive-field dimension of the unfolded input.
    gate_types_per_level:
        List (depth entries) of ``[n_out, nodes_at_level]`` int32.
    kh, kw, stride, padding, cin:
        Convolution kernel parameters (kh/kw/stride/padding used by the
        caller's :func:`packed_unfold` step; cin stored for reference).
    """

    def __init__(
        self,
        leaf_idx: np.ndarray,
        gate_types_per_level: List[np.ndarray],
        kh: int,
        kw: int,
        stride: int,
        padding: int,
        cin: int,
    ) -> None:
        self.n_out   = leaf_idx.shape[0]
        self.n_leaves = leaf_idx.shape[1]
        self.leaf_idx = leaf_idx.astype(np.int32)
        self.depth   = len(gate_types_per_level)
        self.kh      = kh
        self.kw      = kw
        self.stride  = stride
        self.padding = padding
        self.cin     = cin

        # Pre-group gate evaluations for each tree level.
        self._level_groups: List[List[Tuple[int, np.ndarray, np.ndarray]]] = []
        for gt in gate_types_per_level:
            n_out, n_nodes = gt.shape
            flat_gate = gt.reshape(-1)
            flat_out  = np.repeat(np.arange(n_out,   dtype=np.int32), n_nodes)
            flat_node = np.tile  (np.arange(n_nodes, dtype=np.int32), n_out)
            groups: List[Tuple[int, np.ndarray, np.ndarray]] = []
            for g in range(16):
                mask = flat_gate == g
                if not mask.any():
                    continue
                out_i  = flat_out [mask]
                node_i = flat_node[mask]
                order  = np.lexsort((node_i, out_i))
                groups.append((g, out_i[order], node_i[order]))
            self._level_groups.append(groups)

    def forward_packed(self, x_3d: np.ndarray) -> np.ndarray:
        """Evaluate the gate tree on B-packed unrolled input.

        Parameters
        ----------
        x_3d:
            ``[P, L, nw_B]`` int64 — the unrolled input produced by
            :func:`packed_unfold`.  ``P = cin * kh * kw``.

        Returns
        -------
        out:
            ``[n_out, L, nw_B]`` int64 — B-packed output activations.
        """
        _P, L, nw_B = x_3d.shape

        # Gather all leaf values: [n_out, n_leaves, L, nw_B]
        current = x_3d[self.leaf_idx.reshape(-1)].reshape(
            self.n_out, self.n_leaves, L, nw_B
        )

        # Bottom-up tree evaluation
        for groups in self._level_groups:
            n_nodes = current.shape[1] // 2
            a = current[:, 0::2, :, :]   # [n_out, n_nodes, L, nw_B]
            b = current[:, 1::2, :, :]
            result = np.empty((self.n_out, n_nodes, L, nw_B), dtype=np.int64)

            for g, out_i, node_i in groups:
                if g == 0:
                    result[out_i, node_i] = np.int64(0)
                elif g == 3:
                    result[out_i, node_i] = a[out_i, node_i]
                elif g == 5:
                    result[out_i, node_i] = b[out_i, node_i]
                elif g == 15:
                    result[out_i, node_i] = np.int64(-1)
                else:
                    a_g = a[out_i, node_i]  # [n_g, L, nw_B]
                    b_g = b[out_i, node_i]
                    result[out_i, node_i] = apply_gate(a_g, b_g, g)

            current = result

        return current[:, 0, :, :]  # [n_out, L, nw_B]


# ── convolutional LUT-tree layer (arity > 2) ─────────────────────────────────

class BitpackedConvLUTLayer:
    """Bitpacked k-ary LUT tree layer for spatial (conv) inputs — B-packing.

    Handles any ``node_arity > 2`` at any tree depth.  Uses sum-of-products:
    for each truth-table entry ``e`` that is 1, computes the mask of sample
    positions whose ``node_arity`` inputs all match ``e`` and ORs it into the
    output — identical to :class:`BitpackedLUTLayer` but with the spatial ``L``
    dimension kept explicit.

    Activation format: ``[dim, L, nw_B]`` int64 (B-packing, same as
    :class:`BitpackedConvTreeLayer`).

    Parameters
    ----------
    leaf_idx:
        ``[n_out, n_leaves]`` int32 — flat P-index into the unrolled
        receptive-field input where ``n_leaves = node_arity ** tree_depth``.
    truth_tables_per_level:
        List of length ``tree_depth``.  Entry ``i`` is
        ``[n_out * nodes_at_level_i, 2**node_arity]`` bool — the hard truth
        table for every node at that level (nodes from
        :class:`~silogic.layers.ConvLogicTree` ``tree_nodes[i]``).
    node_arity:
        Number of inputs per LUT node.
    kh, kw, stride, padding, cin:
        Convolution kernel parameters.
    """

    def __init__(
        self,
        leaf_idx: np.ndarray,               # [n_out, n_leaves]
        truth_tables_per_level: List[np.ndarray],  # list of [n_groups, 2^k] bool
        node_arity: int,
        kh: int,
        kw: int,
        stride: int,
        padding: int,
        cin: int,
    ) -> None:
        self.n_out    = leaf_idx.shape[0]
        self.n_leaves = leaf_idx.shape[1]
        self.node_arity = node_arity
        self.depth    = len(truth_tables_per_level)
        self.leaf_idx = leaf_idx.astype(np.int32)
        self.kh       = kh
        self.kw       = kw
        self.stride   = stride
        self.padding  = padding
        self.cin      = cin

        lut_size = 2 ** node_arity
        # Pre-group per level: (entry_value, [n_active] int32 group indices)
        self._level_entry_groups: List[List[Tuple[int, np.ndarray]]] = []
        for tt in truth_tables_per_level:
            groups: List[Tuple[int, np.ndarray]] = []
            for e in range(lut_size):
                active = np.where(tt[:, e])[0].astype(np.int32)
                if len(active):
                    groups.append((e, active))
            self._level_entry_groups.append(groups)

    def forward_packed(self, x_3d: np.ndarray) -> np.ndarray:
        """Evaluate the LUT tree on B-packed unrolled input.

        Parameters
        ----------
        x_3d:
            ``[P, L, nw_B]`` int64 — produced by :func:`packed_unfold`.

        Returns
        -------
        out:
            ``[n_out, L, nw_B]`` int64.
        """
        _P, L, nw_B = x_3d.shape
        k = self.node_arity

        # Gather all leaf values: [n_out, n_leaves, L, nw_B]
        current = x_3d[self.leaf_idx.reshape(-1)].reshape(
            self.n_out, self.n_leaves, L, nw_B
        )

        for groups in self._level_entry_groups:
            n_nodes = current.shape[1] // k
            n_groups = self.n_out * n_nodes
            # [n_out, n_nodes, k, L, nw_B] → [n_groups, k, L, nw_B]
            inputs = current.reshape(self.n_out, n_nodes, k, L, nw_B).reshape(
                n_groups, k, L, nw_B
            )
            result = np.zeros((n_groups, L, nw_B), dtype=np.int64)

            for e, active_g in groups:
                match = np.full((len(active_g), L, nw_B), np.int64(-1), dtype=np.int64)
                for j in range(k):
                    inp = inputs[active_g, j, :, :]   # [n_active, L, nw_B]
                    if (e >> j) & 1:
                        match &= inp
                    else:
                        match &= ~inp
                result[active_g] |= match

            current = result.reshape(self.n_out, n_nodes, L, nw_B)

        return current[:, 0, :, :]  # [n_out, L, nw_B]


# ── heads ─────────────────────────────────────────────────────────────────────

class BitpackedGroupSumHead:
    """Bitpacked version of :class:`~silogic.heads.GroupSum`.

    Uses the fused :func:`~.ops.group_sum_packed` path: unpacks one class-group
    at a time without materialising the full ``[B, dim]`` boolean matrix.
    """

    def __init__(self, num_classes: int, tau: float = 1.0) -> None:
        self.num_classes = num_classes
        self.tau = tau

    def forward(self, x: np.ndarray, B: int) -> np.ndarray:
        """
        Parameters
        ----------
        x:
            ``[width, n_words]`` int64.
        B:
            Actual batch size (strips zero-padding from the last word).

        Returns
        -------
        logits:
            ``[B, num_classes]`` float32.
        """
        from .ops import group_sum_packed
        return group_sum_packed(x, B, self.num_classes, self.tau)


class BitpackedLearnedHead:
    """Bitpacked version of :class:`~silogic.heads.LearnedDecoder`.

    Unpacks the final activations then applies the stored float32 linear layer.
    """

    def __init__(self, weight: np.ndarray, bias: np.ndarray | None = None) -> None:
        # weight: [num_classes, width] float32
        # bias:   [num_classes] float32  or None
        self.weight = weight.astype(np.float32)
        self.bias = None if bias is None else bias.astype(np.float32)

    def forward(self, x: np.ndarray, B: int) -> np.ndarray:
        """
        Parameters
        ----------
        x:
            ``[width, n_words]`` int64 packed activations.
        B:
            Actual batch size (used to strip zero-padding bits).

        Returns
        -------
        logits:
            ``[B, num_classes]`` float32.
        """
        from .ops import unpack_bits
        bits = unpack_bits(x, B).astype(np.float32)    # [B, width]
        out = bits @ self.weight.T                      # [B, num_classes]
        if self.bias is not None:
            out += self.bias
        return out
