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
