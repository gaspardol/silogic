"""Hardware-facing intermediate representation (IR) of a discretized LogicNet.

A trained :class:`~silogic.models.LogicNet` discretizes (``forward_hard``) into a
pure Boolean circuit: every node reads a fixed set of previous-layer wires and
outputs one bit given by a truth table; the ``GroupSum`` head sums blocks of the
final layer's bits into per-class integer scores and the prediction is the
argmax. That is *exactly* a feed-forward network of LUTs followed by popcount
adders — i.e. an FPGA design.

This module lowers a ``LogicNet`` to a backend-independent IR:

  * :class:`LutNode`        — one node: input wire indices + a truth table.
  * :class:`Layer`          — a list of nodes (the layer's output bits).
  * :class:`LogicNetCircuit`— the input width, the stack of layers, and the
    ``GroupSum`` parameters (``num_classes`` / ``tau``).

Every node, *for every node family* (``gate16``/``walsh``/``multilinear``/
``hybrid``/``linear``/``polynomial``), is reduced to the same canonical LUT by
evaluating its own ``forward_hard`` at the ``2**arity`` hypercube corners — so
the IR has a single uniform node form and the Verilog backend never needs to
know which family produced it. The truth-table **address convention** is

    addr = sum_j  operand_j << j        (operand 0 is the LSB)

matching :func:`silogic.functional.corner_bits`; :func:`simulate` reproduces
``forward_hard`` bit-exactly under this convention (the tests assert this).

Only ``GroupSum`` heads and the wire-selecting connectomes
(``fixed``/``topk``/``blocktopk``/``dense``) are supported — the
``SumThreshold`` connectomes have no static wire fan-in and raise a clear error.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch

from ..connectomes import SumThresholdConnectome
from ..heads import GroupSum
from ..bitpacking.convert import _extract_conn_indices, _lut_truth_tables


# ── IR dataclasses ────────────────────────────────────────────────────────────

@dataclass
class LutNode:
    """One logic node = an ``arity``-input look-up table.

    Attributes:
        idx: ``arity`` input wire indices into the *previous* layer's bit vector
            (``operand j == prev[idx[j]]``; ``idx[0]`` is the LSB of the address).
        tt:  Truth table of length ``2**arity``; ``tt[addr]`` is the output bit
            for ``addr = sum_j operand_j << j``. Entries are ``0``/``1``.
    """

    idx: List[int]
    tt: List[int]

    @property
    def arity(self) -> int:
        return len(self.idx)

    @property
    def is_const(self) -> bool:
        return all(v == self.tt[0] for v in self.tt)


@dataclass
class Layer:
    """One logic layer: ``nodes[o]`` produces output bit ``o``."""

    nodes: List[LutNode]

    @property
    def width(self) -> int:
        return len(self.nodes)


@dataclass
class LogicNetCircuit:
    """A fully discretized FC logic network with a ``GroupSum`` head.

    Attributes:
        in_dim:      Number of binary input features.
        layers:      Stack of :class:`Layer` (combinational LUT layers).
        num_classes: ``GroupSum`` blocks (== output classes).
        tau:         ``GroupSum`` temperature. Pure positive scale; it does not
            change the argmax, so the hardware reports the raw integer block
            popcounts as ``class_scores`` and ``tau`` is kept only for metadata.
        name:        Suggested top-level module name.
    """

    in_dim: int
    layers: List[Layer]
    num_classes: int
    tau: float = 1.0
    name: str = "logicnet"
    meta: dict = field(default_factory=dict)

    # -- derived sizes used by every backend --------------------------------
    @property
    def out_width(self) -> int:
        return self.layers[-1].width

    @property
    def group_size(self) -> int:
        return self.out_width // self.num_classes

    @property
    def score_bits(self) -> int:
        """Bits needed to hold a class score (popcount of ``group_size`` bits)."""
        return max(1, int(self.group_size).bit_length())

    @property
    def pred_bits(self) -> int:
        return max(1, int(self.num_classes - 1).bit_length())

    def num_luts(self) -> int:
        """Total LUT nodes (excludes pure pass-through / constant nodes)."""
        return sum(1 for L in self.layers for n in L.nodes
                   if not n.is_const and not _is_passthrough(n))

    def summary(self) -> str:
        widths = " -> ".join(str(L.width) for L in self.layers)
        return (f"LogicNetCircuit({self.name}): in={self.in_dim}  "
                f"layers[{widths}]  classes={self.num_classes} "
                f"(group_size={self.group_size}, score_bits={self.score_bits})  "
                f"luts={self.num_luts()}")


def _is_passthrough(n: LutNode) -> bool:
    """A node that just forwards one wire: ``tt[addr] == addr-bit-0``."""
    if any(i != n.idx[0] for i in n.idx):
        return False
    # all slots read the same wire -> only addresses 0 and (2^a-1) are reachable
    return n.tt[0] == 0 and n.tt[-1] == 1


# ── extraction: LogicNet -> IR ────────────────────────────────────────────────

def _passthrough_node(in_wire: int, arity: int) -> LutNode:
    """An identity node forwarding ``prev[in_wire]`` (used for wire residuals)."""
    a = max(1, arity)
    tt = [(p & 1) for p in range(1 << a)]      # output == operand 0 == the wire
    return LutNode(idx=[in_wire] * a, tt=tt)


def extract_logic_net(model, name: str = "logicnet") -> LogicNetCircuit:
    """Lower a trained :class:`~silogic.models.LogicNet` to a :class:`LogicNetCircuit`.

    Args:
        model: A :class:`~silogic.models.LogicNet` (any node family) whose head is
            a :class:`~silogic.heads.GroupSum`.
        name:  Top-level module name to suggest to the backend.

    Raises:
        TypeError: If the head is not a ``GroupSum``.
        NotImplementedError: If any layer uses a ``SumThreshold`` connectome
            (no static wire fan-in to map to a LUT).
    """
    if not isinstance(model.head, GroupSum):
        raise TypeError(
            "fpga export currently supports GroupSum heads only; got "
            f"{type(model.head).__name__}")

    was_training = model.training
    model.eval()
    wire_r = int(getattr(model, "wire_r", 0) or 0)

    layers: List[Layer] = []
    prev_width = None
    try:
        for li, layer in enumerate(model.layers):
            conn = layer._conn
            if isinstance(conn, SumThresholdConnectome):
                raise NotImplementedError(
                    f"layer {li} uses a SumThreshold connectome, which has no "
                    "static wire fan-in; remove it or use a fixed/topk/dense "
                    "connectome for FPGA export.")
            idx = _extract_conn_indices(conn)            # [out, arity] int32
            if idx is None:
                raise NotImplementedError(
                    f"layer {li}: unsupported connectome {type(conn).__name__}")
            tt = _lut_truth_tables(layer.node, layer.arity)   # [out, 2^arity] bool
            tt = tt.astype(np.uint8)
            out_dim = idx.shape[0]

            nodes = [LutNode(idx=idx[o].tolist(), tt=tt[o].tolist())
                     for o in range(out_dim)]

            # wire residual: the LogicNet replaces the first `wire_r` outputs of a
            # *same-width* layer with identity copies of the layer's input wires.
            if wire_r and prev_width == out_dim:
                for o in range(min(wire_r, out_dim)):
                    nodes[o] = _passthrough_node(o, layer.arity)

            layers.append(Layer(nodes=nodes))
            prev_width = out_dim
    finally:
        if was_training:
            model.train()

    return LogicNetCircuit(
        in_dim=model.layers[0].in_dim,
        layers=layers,
        num_classes=model.head.num_classes,
        tau=float(model.head.tau),
        name=name,
        meta={"node": getattr(model, "node", None),
              "depth": len(layers),
              "width": layers[-1].width if layers else 0},
    )


# ── golden software simulator (bit-exact reference for the HDL) ───────────────

def simulate(circuit: LogicNetCircuit, x: np.ndarray):
    """Evaluate the IR in pure numpy — the golden reference for the generated HDL.

    Args:
        circuit: A :class:`LogicNetCircuit`.
        x: ``[B, in_dim]`` array of ``{0,1}`` (any integer/bool dtype).

    Returns:
        ``(scores, pred)`` with ``scores`` an ``[B, num_classes]`` int array of
        per-class block popcounts and ``pred`` a ``[B]`` int array of argmax
        classes (lowest index wins ties — matching ``torch.argmax``).
    """
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[None]
    cur = (x != 0).astype(np.int64)                      # [B, in_dim]
    for layer in circuit.layers:
        B = cur.shape[0]
        out = np.empty((B, layer.width), dtype=np.int64)
        for o, node in enumerate(layer.nodes):
            addr = np.zeros(B, dtype=np.int64)
            for j, wi in enumerate(node.idx):
                addr |= cur[:, wi] << j
            tt = np.asarray(node.tt, dtype=np.int64)
            out[:, o] = tt[addr]
        cur = out
    g = circuit.group_size
    scores = cur.reshape(cur.shape[0], circuit.num_classes, g).sum(axis=2)
    pred = scores.argmax(axis=1)
    return scores, pred
