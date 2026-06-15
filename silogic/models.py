"""Fully-connected logic networks: a stack of :class:`~silogic.layers.LogicLayer`
plus a head from :mod:`silogic.heads`.

The generic :class:`LogicNet` builds the layer stack for *any* node
parameterization / connectome / decoder; the presets (:class:`WARPNet`,
:class:`WARPNetN`, :class:`LUTkNet`, :class:`LUTNodeNet`) just set the defaults
that distinguish those families (replacing five near-identical net classes).
"""
import torch
import torch.nn as nn

from .layers import LogicLayer, ConvLogicTree, OrPool
from .heads import build_decoder


def _per_layer(v, i):
    """Resolve a possibly-per-layer hyperparameter: a list/tuple indexes by layer
    ``i``; any other value broadcasts to every layer. Lets a network mix settings
    (e.g. ``gate_select=["softmax", "softmax", "hard"]`` for soft+hard gates)."""
    return v[i] if isinstance(v, (list, tuple)) else v


class LogicNet(nn.Module):
    """Stack of logic layers followed by a head/decoder.

    Args:
        in_dim (int): Binary input features.
        width (int): Nodes per layer (output width of every layer).
        depth (int): Number of stacked :class:`~silogic.layers.LogicLayer`s.
        num_classes (int): Output classes. Default ``10``.
        node (str): Node parameterization for every layer (see
            :mod:`silogic.nodes`). Default ``"gate16"``.
        connectome (str): Per-layer input wiring. Default ``"topk"``.
        arity (int): Inputs per node (ignored / forced to 2 for ``gate16``).
            Default ``2``.
        k (int): Candidates per input slot for sparse connectomes. Default ``8``.
        tau (float): Head temperature (``GroupSum`` divisor). Default ``1.0``.
        gate_eval (str): ``gate16`` evaluation path. Default ``"basis"``.
        node_tau (float): Sigmoid temperature for ``node="walsh"``. Default ``1.0``.
        residual_p (float): Per-node residual init probability. Default ``0.0``.
        seed (int | None): Base wiring seed; layer ``i`` uses ``seed*seed_mult+i``.
            Default ``0``.
        wire_residual (float): Fraction of each same-width layer's outputs hard-wired
            as identity copies of its input. Default ``0.0``.
        decoder (str): Head; ``"groupsum"`` (default) or a learned decoder
            (``"linear"``/``"linfull"``/``"sumlinear"``/``"ternary"``).
        degree (int | None): Polynomial degree for ``node="polynomial"``.
        seed_mult (int): Per-layer seed multiplier. Default ``1000``.
        gate_select (str | list): Gate selection per layer (see
            :class:`~silogic.layers.LogicLayer`): ``"softmax"`` (default),
            ``"gumbel"``, or ``"hard"``. A list assigns one mode per layer, so a
            network can mix soft and hard gates. ``node`` likewise accepts a list.
        gumbel_tau (float): Temperature for ``gate_select="gumbel"``. Default ``1.0``.
        decoder_ste (bool): STE-binarize features before the learned decoders.
            Default ``True``.
        use_triton (bool): Use fused Triton kernels when available. Default ``True``.
    """

    def __init__(self, in_dim, width, depth, num_classes=10, node="gate16",
                 connectome="topk", arity=2, k=8, tau=1.0, gate_eval="basis",
                 node_tau=1.0, residual_p=0.0, seed=0,
                 wire_residual=0.0, decoder="groupsum", degree=None, seed_mult=1000,
                 gate_select="softmax", gumbel_tau=1.0, decoder_ste=True,
                 use_triton=True):
        super().__init__()
        layers = []
        d_in = in_dim
        for i in range(depth):
            layers.append(LogicLayer(
                d_in, width, node=_per_layer(node, i), connectome=connectome,
                arity=arity, k=k, gate_eval=gate_eval, residual_p=residual_p,
                degree=degree, tau=node_tau,
                gate_select=_per_layer(gate_select, i), gumbel_tau=gumbel_tau,
                use_triton=use_triton,
                seed=None if seed is None else seed * seed_mult + i))
            d_in = width
        self.layers = nn.ModuleList(layers)
        self.head = build_decoder(decoder, width, num_classes, tau,
                                  feature_ste=decoder_ste)
        self.width = width
        self.depth = depth
        self.node = node
        self.connectome = connectome
        self.k = k
        self.decoder_kind = decoder
        self.wire_r = int(width * wire_residual)

    def _run(self, x, hard):
        for layer in self.layers:
            y = layer.forward_hard(x) if hard else layer(x)
            if self.wire_r and x.shape[1] == y.shape[1]:
                y = torch.cat([x[:, :self.wire_r], y[:, self.wire_r:]], dim=1)
            x = y
        return x

    def forward(self, x):
        return self.head(self._run(x, hard=False))

    @torch.no_grad()
    def forward_hard(self, x):
        return self.head.forward_hard(self._run(x, hard=True))

    def num_gates(self):
        return self.width * self.depth


class WARPNet(LogicNet):
    """Stack of 2-input Walsh (WARP) layers + GroupSum head (preset)."""

    def __init__(self, in_dim, width, depth, num_classes=10, k=8, tau=1.0,
                 residual_p=0.0, seed=0, gate_select="softmax"):
        super().__init__(in_dim, width, depth, num_classes=num_classes, node="walsh",
                         arity=2, k=k, node_tau=tau, tau=20.0, residual_p=residual_p,
                         seed=seed, gate_select=gate_select)


class WARPNetN(LogicNet):
    """Stack of arity-``n`` Walsh (WARP) layers + GroupSum head (preset)."""

    def __init__(self, in_dim, width, depth, arity=6, num_classes=10, k=8, tau=1.0,
                 residual_p=0.0, seed=0, gate_select="softmax"):
        super().__init__(in_dim, width, depth, num_classes=num_classes, node="walsh",
                         arity=arity, k=k, node_tau=tau, tau=20.0,
                         residual_p=residual_p, seed=seed, gate_select=gate_select)


class LUTkNet(LogicNet):
    """Stack of k-input multilinear LUT layers + GroupSum head (preset)."""

    def __init__(self, in_dim, width, depth, k=4, num_classes=10, tau=4.0,
                 cand_k=4, seed=0):
        super().__init__(in_dim, width, depth, num_classes=num_classes,
                         node="multilinear", arity=k, k=cand_k, tau=tau, seed=seed)
        self.lut_inputs = k

    def num_luts(self):
        return self.width * self.depth   # one LUT per node


class LUTNodeNet(LogicNet):
    """Stack of n-input LUT-node layers (selectable relaxation) + GroupSum head."""

    def __init__(self, in_dim, width, depth, arity=4, relaxation="probabilistic",
                 num_classes=10, k=8, tau=10.0, residual_p=0.0, seed=0):
        super().__init__(in_dim, width, depth, num_classes=num_classes,
                         node=relaxation, arity=arity, k=k, tau=tau,
                         residual_p=residual_p, seed=seed)


# ---------------------------------------------------------------------------
# Convolutional logic network — the spatial analogue of LogicNet.
# ---------------------------------------------------------------------------
class LogicConvNet(nn.Module):
    """Convolutional logic-gate-tree network with a logic head.

    Mirrors :class:`LogicNet`: it stacks layers and adds a head from
    :mod:`silogic.heads`, exposing the same ``forward`` / ``forward_hard``
    interface and the same ``decoder=`` choices. The body is a stack of
    :class:`~silogic.layers.ConvLogicTree` + :class:`~silogic.layers.OrPool`
    blocks (each block halves H and W); the head is a stack of dense
    :class:`~silogic.layers.LogicLayer`s followed by the decoder.

    Args:
        in_channels (int): Input image channels (e.g. ``3`` for RGB).
        in_hw (int): Input spatial side length ``H == W``.
        channels (list[int]): Per-block output channel counts; one
            ``ConvLogicTree`` + ``OrPool`` block per entry.
        head_width (int, optional): Width of each dense-head layer when
            ``head_widths`` is not given. Default ``None``.
        num_classes (int): Output classes. Default ``10``.
        tree_depth (int, optional): Node-tree depth for every conv block. ``None``
            (default) uses ``2`` for the 2-input gate families and ``1`` for the
            n-input LUT families.
        kernel (int): Conv kernel side length (padding ``kernel // 2``). Default ``3``.
        connect (str): Conv leaf wiring, ``"topk"`` (default) or ``"fixed"``.
        k (int): Top-K candidate pool size for conv leaves. Default ``4``.
        head_connect (str): Head connectivity — ``"topk"`` (default),
            ``"fixed"``/``"f"``, ``"l"``/``"dense"``, or any
            :mod:`silogic.connectomes` name.
        head_k (int): Top-K fan-in per head neuron. Default ``8``.
        head_depth (int): Head layers when ``head_widths`` is not given. Default ``2``.
        n_chan (int): Input channels each conv tree may observe. Default ``2``.
        residual_init (bool): Residual gate init for conv blocks. Default ``True``.
        tau (float): ``GroupSum`` temperature divisor. Default ``100.0``.
        seed (int): Base RNG seed (offset per block/layer). Default ``0``.
        wire_residual (float): Fraction of each conv block's output channels
            hard-wired as OR-pooled identity copies of its input. Default ``0.0``.
        head_widths (list[int], optional): Explicit per-layer head widths
            (overrides ``head_width`` x ``head_depth``). Default ``None``.
        decoder (str): Head — ``"groupsum"`` (default) or a learned decoder
            (``"linear"``/``"linfull"``/``"sumlinear"``/``"ternary"``), exactly
            as in :class:`LogicNet`.
        node (str): Conv node parameterization (see
            :class:`~silogic.layers.ConvLogicTree`): ``"gate16"`` (default) /
            ``"walsh"`` build a 2-input gate tree; ``"multilinear"`` / ``"hybrid"``
            / ``"linear"`` / ``"polynomial"`` build an ``arity``-input LUT tree.
        arity (int): Fan-in per node for the n-input conv node families. Default ``4``.
        node_tau (float): Sigmoid temperature for ``node="walsh"``. Default ``1.0``.
        degree (int | None): Polynomial degree for ``node="polynomial"``.
        gate_select (str): Gate selection for the conv blocks. Default ``"softmax"``.
        gumbel_tau (float): Temperature for ``gate_select="gumbel"``. Default ``1.0``.
        decoder_ste (bool): STE-binarize features before the learned decoders.
            Default ``True``.
        use_triton (bool): Use fused Triton kernels (``gate16`` tree, ``hybrid``
            LUT-tree of any depth) when available. Default ``True``.
    """

    def __init__(self, in_channels, in_hw, channels, head_width=None,
                 num_classes=10, tree_depth=None, kernel=3, connect="topk",
                 k=4, head_connect="topk", head_k=8, head_depth=2,
                 n_chan=2, residual_init=True, tau=100.0, seed=0,
                 wire_residual=0.0, head_widths=None,
                 decoder="groupsum", node="gate16", arity=2, node_tau=1.0,
                 degree=None, gate_select="softmax", gumbel_tau=1.0,
                 decoder_ste=True, use_triton=True):
        super().__init__()
        self.blocks = nn.ModuleList()
        c = in_channels
        for i, n in enumerate(channels):
            self.blocks.append(ConvLogicTree(
                c, n, kernel=kernel, tree_depth=tree_depth, stride=1,
                padding=kernel // 2, connect=connect, k=k, n_chan=n_chan,
                residual_init=residual_init, seed=seed * 17 + i,
                node=node, arity=arity, tau=node_tau, degree=degree,
                gate_select=gate_select, gumbel_tau=gumbel_tau, use_triton=use_triton))
            self.blocks.append(OrPool(2))
            c = n
            in_hw = in_hw // 2
        self.feat_dim = c * in_hw * in_hw
        # dense logic head (same LogicLayer connectome names as the FC net)
        if head_widths is None:
            head_widths = [head_width] * head_depth
        layers = []
        d_in = self.feat_dim
        for j, w in enumerate(head_widths):
            layers.append(LogicLayer(d_in, w, connectome=head_connect, k=head_k, node = node,
                                     use_triton=use_triton, seed=seed * 31 + j))
            d_in = w
        self.head_layers = nn.ModuleList(layers)
        # shared decoder registry (mirrors LogicNet) over the final head width
        self.head = build_decoder(decoder, head_widths[-1], num_classes, tau,
                                  feature_ste=decoder_ste)
        self.decoder_kind = decoder
        self._head_widths = head_widths
        self.wire_residual = wire_residual

    def _apply_blocks(self, x, hard=False):
        # blocks are [conv, pool, conv, pool, ...]; the wire residual is inserted
        # between conv and pool of each pair.
        i = 0
        while i < len(self.blocks):
            conv, pool = self.blocks[i], self.blocks[i + 1]
            x_in = x
            y = conv.forward_hard(x) if hard else conv(x)
            if self.wire_residual > 0:
                r = min(int(conv.n * self.wire_residual), x_in.shape[1])
                if r > 0:
                    y = torch.cat([x_in[:, :r], y[:, r:]], dim=1)
            x = pool.forward_hard(y) if hard else pool(y)
            i += 2
        return x

    def _run(self, x, hard):
        x = self._apply_blocks(x, hard=hard).flatten(1)
        for layer in self.head_layers:
            x = layer.forward_hard(x) if hard else layer(x)
        return x

    def forward(self, x):
        return self.head(self._run(x, hard=False))

    @torch.no_grad()
    def forward_hard(self, x):
        return self.head.forward_hard(self._run(x, hard=True))

    def gate_count(self, in_hw):
        """Approx number of binary logic gates (hardware cost). A conv tree is
        instantiated at every spatial output placement (as in the paper)."""
        total = 0
        hw = in_hw
        for blk in self.blocks:
            if isinstance(blk, ConvLogicTree):
                total += blk.n * hw * hw * blk.gates_per_output  # conv gates/output
            elif isinstance(blk, OrPool):
                hw = hw // 2                                 # pooling halves HW
        total += sum(self._head_widths)                      # dense head gates
        return total


class LogicTreeNet(LogicConvNet):
    """Convolutional logic-gate-tree network (Petersen et al., arXiv:2411.04732)
    + LILogic Top-K wiring — the paper-named preset of :class:`LogicConvNet`."""

