"""LogicTreeNet: convolutional logic-gate-tree network with a logic head.

Combines Petersen-style conv logic trees + OR pooling + residual init with
LILogic's learnable Top-K connectivity (conv leaves AND the dense head).
"""
import torch
import torch.nn as nn
from .conv import ConvLogicTree, OrPool
from .model import LogicLayer, GroupSum


# STE binarization before the linear decoder is on by default; set
# silogic.treenet._STE_ON = False to ablate it (reproduces the broken gap).
_STE_ON = True


def _binarize_ste(x):
    """Straight-through binarization of relaxed logic features in [0,1]:
    forward = (x > 0.5) in {0,1}, backward = identity. Lets a learned decoder
    train on the SAME {0,1} features it will see at inference -> no soft->hard
    discretization gap (BNN-style)."""
    if not _STE_ON:
        return x
    return (x > 0.5).float() + (x - x.detach())


class LogicTreeNet(nn.Module):
    """Convolutional logic-gate-tree network with a logic head.

    Stacks ``ConvLogicTree`` + ``OrPool`` blocks (each block halves H and W),
    then a dense logic head and a decoder mapping to class logits.

    Args:
        in_channels (int): Number of input image channels (e.g. ``3`` for RGB).
        in_hw (int): Input spatial side length ``H == W``.
        channels (list[int]): Per-block output channel counts; one
            ``ConvLogicTree`` + ``OrPool`` block per entry, each halving H, W.
        head_width (int, optional): Width of each dense-head logic layer when
            ``head_widths`` is not given. Default ``None``.
        num_classes (int): Number of output classes. Default ``10``.
        tree_depth (int): Gate-tree depth passed to every conv block. Default ``2``.
        kernel (int): Conv kernel side length; padding is ``kernel // 2``.
            Default ``3``.
        connect (str): Conv leaf wiring, ``"topk"`` (default) or ``"fixed"``
            (see :class:`~silogic.conv.ConvLogicTree`).
        k (int): Top-K candidate pool size for conv leaves. Default ``4``.
        head_connect (str): Head connectivity. ``"topk"`` (default, learnable
            Top-K), ``"fixed"``/``"f"`` (fixed random), or ``"l"``/``"dense"``
            (fully connected); other values are passed through verbatim.
        head_k (int): Top-K fan-in per neuron in each head logic layer. Default ``8``.
        head_depth (int): Number of head logic layers when ``head_widths`` is not
            given (uniform ``head_width`` each). Default ``2``.
        n_chan (int): Input channels each conv tree may observe. Default ``2``.
        residual_init (bool): Residual gate initialization for conv blocks
            (bias toward pass-through ``"A"``). Default ``True``.
        tau (float): Temperature divisor for the ``GroupSum`` decoder. Default ``100.0``.
        seed (int): Base RNG seed (offset per block/layer). Default ``0``.
        residual (bool): Per-node XOR skip inside each conv tree. Default ``False``.
        wire_residual (float): Fraction of each conv block's output channels
            replaced by hardwired (OR-pooled) copies of input channels, forming
            an identity backbone (Gumbel-safe, no discretization gap). ``0.0``
            disables it. Default ``0.0``.
        head_widths (list[int], optional): Explicit per-layer dense-head widths
            (e.g. tapering ``1280k -> 640k -> 320k``); overrides
            ``head_width`` x ``head_depth``. Default ``None``.
        decoder (str): Output decoder. ``"groupsum"`` (default, fixed block-sum;
            requires ``head_widths[-1] % num_classes == 0``) or ``"linear"``
            (learned FC over the final logic features; deviates from the paper).
    """
    def __init__(self, in_channels, in_hw, channels, head_width=None,
                 num_classes=10, tree_depth=2, kernel=3, connect="topk",
                 k=4, head_connect="topk", head_k=8, head_depth=2,
                 n_chan=2, residual_init=True, tau=100.0, seed=0,
                 residual=False, wire_residual=0.0, head_widths=None,
                 decoder="groupsum"):
        super().__init__()
        self.blocks = nn.ModuleList()
        c = in_channels
        hw = in_hw
        for i, n in enumerate(channels):
            self.blocks.append(ConvLogicTree(
                c, n, kernel=kernel, tree_depth=tree_depth, stride=1,
                padding=kernel // 2, connect=connect, k=k, n_chan=n_chan,
                residual_init=residual_init, seed=seed * 17 + i,
                residual=residual))
            self.blocks.append(OrPool(2))
            c = n
            hw = hw // 2
        self.feat_dim = c * hw * hw
        # logic head (dense / Top-K) + group sum
        hc = {"topk": "TopK", "fixed": "F", "f": "F", "l": "L",
              "dense": "L"}.get(head_connect.lower(), head_connect)
        # Paper head (A.1.1) is a *tapering* stack 1280k -> 640k -> 320k; pass
        # head_widths for that. Fall back to a uniform head_width x head_depth.
        if head_widths is None:
            head_widths = [head_width] * head_depth
        layers = []
        d_in = self.feat_dim
        for j, hw_out in enumerate(head_widths):
            layers.append(LogicLayer(d_in, hw_out, connectome=hc,
                                     k=head_k, seed=seed * 31 + j))
            d_in = hw_out
        self.head_layers = nn.ModuleList(layers)
        # decoder: "groupsum" (paper, fixed block-sum) or "linear" (learned FC
        # over the n-dim logic feature state -> class logits). The linear decoder
        # treats the last logic layer's outputs as an n-dim hidden representation
        # and stacks a dense classifier on top (deviates from the paper).
        self.decoder_kind = decoder
        if decoder == "linear":
            self.decoder = nn.Linear(head_widths[-1], num_classes)
        else:
            assert head_widths[-1] % num_classes == 0
            self.group = GroupSum(num_classes, tau=tau)
        self._channels = channels
        self._head_widths = head_widths
        self._head_width = head_widths[-1]
        self._head_depth = len(head_widths)
        self._tree_depth = tree_depth
        # wire residual: fraction of each conv block's output channels are
        # hardwired (OR-pooled) copies of input channels -> structural identity
        # backbone (0 gates, Gumbel-safe, no discretization gap).
        self.wire_residual = wire_residual

    def _apply_blocks(self, x, hard=False):
        # blocks are [conv, pool, conv, pool, ...]; treat each (conv,pool) pair
        # so the wire residual is inserted between conv and pool.
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

    def forward(self, x):
        x = self._apply_blocks(x, hard=False).flatten(1)
        for layer in self.head_layers:
            x = layer(x)
        if self.decoder_kind == "linear":
            return self.decoder(_binarize_ste(x))   # STE -> decoder sees {0,1}
        return self.group(x)

    @torch.no_grad()
    def forward_hard(self, x):
        x = self._apply_blocks(x, hard=True).flatten(1)
        for layer in self.head_layers:
            x = layer.forward_hard(x)
        if self.decoder_kind == "linear":
            return self.decoder(x.float())      # binary features -> linear decode
        return self.group.forward_hard(x)

    def gate_count(self, in_hw):
        """Approx number of binary logic gates (hardware cost). Convolution
        shares parameters but instantiates a gate tree at every spatial output
        placement, so we count per placement (as in the LogicTreeNet paper)."""
        total = 0
        hw = in_hw
        tree_gates = 2 ** self._tree_depth - 1
        for blk in self.blocks:
            if isinstance(blk, ConvLogicTree):
                total += blk.n * hw * hw * tree_gates       # conv tree gates
            elif isinstance(blk, OrPool):
                hw = hw // 2                                 # pooling halves HW
        total += sum(self._head_widths)                      # dense head gates
        return total
