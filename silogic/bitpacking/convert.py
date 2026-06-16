"""Convert trained silogic models to bitpacked inference format.

Supported paths
---------------
* :class:`~silogic.layers.LogicLayer` with **gate16** node (any supported connectome)
  → :class:`.packed_layer.BitpackedFCLayer`
* :class:`~silogic.layers.LogicLayer` with **walsh (arity=2)** node
  → :class:`.packed_layer.BitpackedFCLayer`  (theta → truth-table → gate16)
* :class:`~silogic.layers.LogicLayer` with **multilinear / hybrid (any arity)** node
  → :class:`.packed_layer.BitpackedFCLayer` (arity=2) or
    :class:`.packed_layer.BitpackedLUTLayer` (arity>2)
* :class:`~silogic.layers.LogicLayer` with **linear / polynomial** node
  → same as multilinear (evaluate truth table over 2^arity corners)
* :class:`~silogic.models.LogicNet`
  → :class:`.packed_model.BitpackedNet`
* :class:`~silogic.models.LogicConvNet`
  → :class:`.packed_model.BitpackedConvNet`

``ConvLogicTree`` conversion handles all node types:

* **gate16** — gate types read directly from ``gate_logits``
  → :class:`.packed_layer.BitpackedConvTreeLayer`
* **arity=2 nodes** (walsh, multilinear, hybrid, linear, polynomial) —
  truth table evaluated at 4 corners → gate16 index
  → :class:`.packed_layer.BitpackedConvTreeLayer`
* **arity>2 nodes** (multilinear/hybrid arity=4, …) — truth table evaluated
  at ``2**arity`` corners, sum-of-products dispatch per level
  → :class:`.packed_layer.BitpackedConvLUTLayer`

All paths use B-packing (``[dim, L, nw_B]`` int64): the batch dimension is
packed once; packed-unfold, gate-tree/LUT-tree, and OR-pool run without
expanding B.

Unsupported FC: ``SumThresholdConnectome`` layers fall back to uint8
``forward_hard`` and are packed after each such layer.
"""
from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from ..connectomes import FixedConnectome, TopKConnectome, DenseConnectome
from ..nodes import Gate16Node, WalshNode, MultilinearNode, HybridNode, LinearNode, PolynomialNode
from ..functional import corner_bits

from .ops import code_to_gate16
from .packed_layer import (
    BitpackedFCLayer,
    BitpackedLUTLayer,
    BitpackedConvTreeLayer,
    BitpackedConvLUTLayer,
    BitpackedGroupSumHead,
    BitpackedLearnedHead,
)


# ── connectome: extract hard input indices ────────────────────────────────────

def _extract_conn_indices(conn) -> Optional[np.ndarray]:
    """Return ``[out_dim, arity]`` int32 of hard-selected input indices, or None
    for unsupported connectomes (SumThresholdConnectome)."""
    with torch.no_grad():
        if isinstance(conn, FixedConnectome):
            return conn.idx.cpu().numpy().astype(np.int32)          # [out, arity]
        if isinstance(conn, TopKConnectome):
            sel = conn.conn.argmax(dim=2)                           # [out, arity]
            idx = torch.gather(conn.cand, 2, sel.unsqueeze(-1)).squeeze(-1)
            return idx.cpu().numpy().astype(np.int32)               # [out, arity]
        if isinstance(conn, DenseConnectome):
            return conn.conn.argmax(dim=2).cpu().numpy().astype(np.int32)
    return None


# ── node: extract hard truth table or gate16 index ───────────────────────────

def _gate16_types(node: Gate16Node) -> np.ndarray:
    """``[out_dim]`` int32 gate16 index from argmax."""
    with torch.no_grad():
        return node.gate_logits.argmax(dim=1).cpu().numpy().astype(np.int32)


def _walsh_arity2_to_gate16(theta: np.ndarray) -> np.ndarray:
    """Convert arity-2 Walsh theta ``[out, 4]`` → gate16 index ``[out]``.

    Evaluates z at all 4 input corners and converts the resulting truth table
    to the matching gate16 index via the precomputed code lookup.
    """
    t0, t1, t2, t3 = theta[:, 0], theta[:, 1], theta[:, 2], theta[:, 3]
    b0 = (t0 - t1 - t2 + t3) > 0   # (a=0, b=0)
    b1 = (t0 - t1 + t2 - t3) > 0   # (a=0, b=1)
    b2 = (t0 + t1 - t2 - t3) > 0   # (a=1, b=0)
    b3 = (t0 + t1 + t2 + t3) > 0   # (a=1, b=1)
    code = b0.astype(np.int32) | (b1.astype(np.int32) << 1) | \
           (b2.astype(np.int32) << 2) | (b3.astype(np.int32) << 3)
    return code_to_gate16()[code]


def _lut_truth_tables(node, arity: int) -> np.ndarray:
    """Evaluate the hard truth table for any arity node.

    Returns ``[out_dim, 2^arity]`` bool — the hard output for each of the
    ``2^arity`` input combinations.
    """
    corners = corner_bits(arity).to(torch.float32)  # [2^n, n]
    P = 2 ** arity
    out_dim = node.out_dim
    with torch.no_grad():
        # Expand corners to [1, P, n] and broadcast over out_dim
        c_in = corners.unsqueeze(0).expand(1, P, arity)
        # node expects [B, out, arity] → use B=1, out=P, arity=arity with each
        # output being one corner configuration.
        # We need shape [P, out_dim, arity] → broadcast per output independently:
        # Evaluate forward_hard for each corner pattern.
        results = []
        for p in range(P):
            operand = corners[p].view(1, 1, arity).expand(1, out_dim, arity)
            out = node.forward_hard(operand)   # [1, out_dim]
            results.append(out.squeeze(0))
        # results: list of P tensors each [out_dim]
        tt = torch.stack(results, dim=1)       # [out_dim, P]
        return tt.cpu().numpy().astype(bool)


def _multilinear_arity2_to_gate16(logits: np.ndarray) -> np.ndarray:
    """``[out, 4]`` float logits → gate16 index ``[out]``.

    MultilinearNode addresses: ``addr = slot0*1 + slot1*2`` (slot0 is LSB).
    Gate16 truth-table address: ``addr = (slot0 << 1) | slot1`` (slot0 is MSB).
    The bit ordering must be swapped: multi-addr 1↔2 map to gate16 addr 2↔1.
    """
    tt = (logits > 0).astype(np.int32)   # [out, 4]
    # multi_addr → gate16_addr: 0→0, 1→2, 2→1, 3→3
    code = tt[:, 0] | (tt[:, 2] << 1) | (tt[:, 1] << 2) | (tt[:, 3] << 3)
    return code_to_gate16()[code]


# ── conv tree node helpers ────────────────────────────────────────────────────

def _tree_node_arity2_to_gate16(nd, n_out: int, n_nodes: int) -> np.ndarray:
    """Extract gate16 types ``[n_out, n_nodes]`` from an arity-2 tree-node module.

    Handles Gate16Node, WalshNode, MultilinearNode/HybridNode, and any other
    node type by evaluating the hard truth table at all 4 input corners.
    """
    with torch.no_grad():
        if isinstance(nd, Gate16Node):
            return (
                nd.gate_logits.argmax(dim=1).cpu().numpy()
                .astype(np.int32).reshape(n_out, n_nodes)
            )
        if isinstance(nd, WalshNode):
            theta = nd.theta.detach().cpu().numpy()        # [n_out*n_nodes, 4]
            return _walsh_arity2_to_gate16(theta).reshape(n_out, n_nodes)
        if isinstance(nd, (MultilinearNode, HybridNode)):
            logits = nd.logits.detach().cpu().numpy()      # [n_out*n_nodes, 4]
            return _multilinear_arity2_to_gate16(logits).reshape(n_out, n_nodes)
        # General fallback: evaluate truth table at all 4 corners.
        # corner_bits uses addr = (p>>j)&1; gate16 addr = (slot0<<1)|slot1 so
        # bits 1 and 2 of the code must be swapped (same fix as convert_layer).
        tt = _lut_truth_tables(nd, 2)                      # [n_out*n_nodes, 4] bool
        codes = tt.astype(np.int32)
        code = codes[:, 0] | (codes[:, 2] << 1) | (codes[:, 1] << 2) | (codes[:, 3] << 3)
        return code_to_gate16()[code].reshape(n_out, n_nodes)


# ── layer conversion ──────────────────────────────────────────────────────────

def convert_layer(
    layer,
) -> Union[BitpackedFCLayer, BitpackedLUTLayer, None]:
    """Convert one :class:`~silogic.layers.LogicLayer` to a bitpacked layer.

    Returns ``None`` if the layer cannot be bitpacked (e.g. SumThreshold
    connectome), signalling the caller to use uint8 ``forward_hard`` instead.
    """
    from ..connectomes import SumThresholdConnectome

    conn = layer._conn
    if isinstance(conn, SumThresholdConnectome):
        return None

    idx = _extract_conn_indices(conn)
    if idx is None:
        return None

    node = layer.node
    arity = layer.arity

    # ── gate16 node ─────────────────────────────────────────
    if isinstance(node, Gate16Node):
        gate_types = _gate16_types(node)
        return BitpackedFCLayer(gate_types, idx[:, 0], idx[:, 1])

    # ── walsh arity=2 → gate16 equivalent ───────────────────
    if isinstance(node, WalshNode) and arity == 2:
        theta = node.theta.detach().cpu().numpy()
        gate_types = _walsh_arity2_to_gate16(theta)
        return BitpackedFCLayer(gate_types, idx[:, 0], idx[:, 1])

    # ── multilinear / hybrid arity=2 → gate16 equivalent ────
    if isinstance(node, (MultilinearNode, HybridNode)) and arity == 2:
        logits = node.logits.detach().cpu().numpy()   # [out, 4]
        gate_types = _multilinear_arity2_to_gate16(logits)
        return BitpackedFCLayer(gate_types, idx[:, 0], idx[:, 1])

    # ── general n-input LUT (any node, any arity) ────────────
    # _lut_truth_tables evaluates via corner_bits: corner p → slot j = (p>>j)&1.
    # Gate16 addr = (slot0<<1)|slot1 while corners use addr = slot0 + 2*slot1,
    # so for arity=2 we must swap the gate16-code bits at positions 1 and 2.
    truth_tables = _lut_truth_tables(node, arity)       # [out, 2^n] bool
    if arity == 2:
        codes = truth_tables.astype(np.int32)
        code = codes[:, 0] | (codes[:, 2] << 1) | (codes[:, 1] << 2) | (codes[:, 3] << 3)
        gate_types = code_to_gate16()[code]
        return BitpackedFCLayer(gate_types, idx[:, 0], idx[:, 1])
    return BitpackedLUTLayer(truth_tables, idx)


# ── conv tree conversion ──────────────────────────────────────────────────────

def convert_conv_tree_layer(
    layer,
) -> Optional[BitpackedConvTreeLayer]:
    """Convert one :class:`~silogic.layers.ConvLogicTree` to a bitpacked layer.

    Supported node types:

    * **gate16** — gate types from ``gate_logits`` argmax
      → :class:`.packed_layer.BitpackedConvTreeLayer`
    * **arity=2** (walsh, multilinear, hybrid, linear, polynomial) — truth
      table at 4 corners → gate16 index via same conversion as FC layers
      → :class:`.packed_layer.BitpackedConvTreeLayer`
    * **arity>2** — hard truth table at ``2**arity`` corners for each level
      node → sum-of-products dispatch
      → :class:`.packed_layer.BitpackedConvLUTLayer`

    Both ``connect="fixed"`` and ``connect="topk"`` are supported (topk path
    commits to the hard-argmax leaf, same as ``forward_hard``).
    """
    # ── extract hard leaf indices (same for all node types) ────────────────
    with torch.no_grad():
        if layer.connect == "fixed":
            leaf_idx = layer.leaf_idx.cpu().numpy().astype(np.int32)   # [n, leaves]
        else:  # topk — hard argmax selection (matches forward_hard)
            sel = layer.conn.argmax(dim=2)                              # [n, leaves]
            leaf_idx = torch.gather(
                layer.leaf_cand, 2, sel.unsqueeze(-1)
            ).squeeze(-1).cpu().numpy().astype(np.int32)                # [n, leaves]

    conv_kw = dict(
        kh=layer.kh, kw=layer.kw,
        stride=layer.stride, padding=layer.padding, cin=layer.cin,
    )
    n_out = layer.n

    # ── gate16 node: read gate types directly ──────────────────────────────
    if layer.node_name == "gate16":
        with torch.no_grad():
            gate_types_per_level = [
                gl.argmax(dim=2).cpu().numpy().astype(np.int32)         # [n, nodes]
                for gl in layer.gate_logits
            ]
        return BitpackedConvTreeLayer(
            leaf_idx=leaf_idx,
            gate_types_per_level=gate_types_per_level,
            **conv_kw,
        )

    # ── arity=2 non-gate16: convert each tree level's node to gate16 ───────
    if layer.node_arity == 2:
        gate_types_per_level = []
        for i, nd in enumerate(layer.tree_nodes):
            n_nodes = layer.node_arity ** (layer.d - 1 - i)
            gate_types_per_level.append(
                _tree_node_arity2_to_gate16(nd, n_out, n_nodes)
            )
        return BitpackedConvTreeLayer(
            leaf_idx=leaf_idx,
            gate_types_per_level=gate_types_per_level,
            **conv_kw,
        )

    # ── arity>2: evaluate truth table per level → sum-of-products LUT tree ─
    truth_tables_per_level = [
        _lut_truth_tables(nd, layer.node_arity)   # [n_out*nodes_at_level, 2^arity]
        for nd in layer.tree_nodes
    ]
    return BitpackedConvLUTLayer(
        leaf_idx=leaf_idx,
        truth_tables_per_level=truth_tables_per_level,
        node_arity=layer.node_arity,
        **conv_kw,
    )


# ── head conversion ───────────────────────────────────────────────────────────

def convert_head(head) -> Union[BitpackedGroupSumHead, BitpackedLearnedHead]:
    from ..heads import GroupSum, LearnedDecoder
    if isinstance(head, GroupSum):
        return BitpackedGroupSumHead(head.num_classes, head.tau)
    if isinstance(head, LearnedDecoder):
        # Extract the effective linear weight matrix (handles all decoder kinds)
        return _convert_learned_head(head)
    raise TypeError(f"Unsupported head type: {type(head)}")


def _convert_learned_head(head) -> BitpackedLearnedHead:
    """Extract a float32 weight+bias from any LearnedDecoder variant."""
    from ..heads import LearnedDecoder
    with torch.no_grad():
        kind = head.kind
        if kind in ("linear", "linfull"):
            w = head.dec.weight.cpu().numpy()
            b = head.dec.bias.cpu().numpy()
            return BitpackedLearnedHead(w, b)
        if kind == "ternary":
            import torch.nn.functional as F
            from ..functional import ternary_ste
            w = ternary_ste(head.dec_w).cpu().numpy()
            return BitpackedLearnedHead(w, None)
        if kind == "sumlinear":
            raise NotImplementedError(
                "sumlinear decoder requires BN folding; use forward_hard for this head"
            )
    raise ValueError(f"Unknown decoder kind: {head.kind!r}")


# ── full model conversion ─────────────────────────────────────────────────────

def convert_logic_net(model) -> "packed_model.BitpackedNet":
    """Convert a :class:`~silogic.models.LogicNet` to a :class:`.packed_model.BitpackedNet`."""
    from .packed_model import BitpackedNet

    packed_layers = []
    fallback_layers = []    # (index, uint8_layer) for SumThreshold layers

    for i, layer in enumerate(model.layers):
        bp = convert_layer(layer)
        if bp is None:
            fallback_layers.append(i)
            packed_layers.append(None)
        else:
            packed_layers.append(bp)

    head = convert_head(model.head)
    wire_r = getattr(model, 'wire_r', 0)

    # Store the original layers for fallback indices
    fallback_torch = {i: model.layers[i] for i in fallback_layers}

    return BitpackedNet(
        packed_layers=packed_layers,
        head=head,
        wire_r=wire_r,
        fallback_torch_layers=fallback_torch,
    )


def convert_logic_conv_net(model) -> "packed_model.BitpackedConvNet":
    """Convert a :class:`~silogic.models.LogicConvNet` to
    :class:`.packed_model.BitpackedConvNet`.

    All ``ConvLogicTree`` node types are bitpacked (gate16 and arity=2 nodes
    via :class:`.packed_layer.BitpackedConvTreeLayer`; arity>2 nodes via
    :class:`.packed_layer.BitpackedConvLUTLayer`).  The FC head is always
    bitpacked.  Only ``SumThresholdConnectome`` FC head layers fall back to
    uint8.
    """
    from ..layers import ConvLogicTree
    from .packed_model import BitpackedConvNet

    # Convert conv blocks (pairs: ConvLogicTree + OrPool).
    bp_conv_layers: List = []
    for i in range(0, len(model.blocks), 2):
        conv = model.blocks[i]
        bp = convert_conv_tree_layer(conv) if isinstance(conv, ConvLogicTree) else None
        bp_conv_layers.append(bp)

    # Convert FC head layers
    packed_head_layers = []
    for layer in model.head_layers:
        bp = convert_layer(layer)
        packed_head_layers.append(bp)  # None → fallback

    fallback_head_torch = {
        i: model.head_layers[i]
        for i, bp in enumerate(packed_head_layers) if bp is None
    }

    head = convert_head(model.head)

    return BitpackedConvNet(
        conv_blocks=model.blocks,
        bp_conv_layers=bp_conv_layers,
        packed_head_layers=packed_head_layers,
        head=head,
        fallback_head_torch=fallback_head_torch,
        wire_residual=getattr(model, 'wire_residual', 0.0),
    )
