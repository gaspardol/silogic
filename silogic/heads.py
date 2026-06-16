"""Classification heads / decoders for logic networks.

Maps a layer of logic features to class logits. ``GroupSum`` is the parameter-free
block-sum head (Petersen / LILogic); the learned decoders (``linear``, ``linfull``,
``sumlinear``, ``ternary``) treat the final logic activations as a feature vector.
Both the FC :class:`~silogic.models.LogicNet` and the convolutional
:class:`~silogic.models.LogicConvNet` build their head through :func:`build_decoder`.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .functional import binarize_ste, ternary_ste


class GroupSum(nn.Module):
    """Aggregate logic outputs into class logits by group counting.

    Splits the ``width`` logic outputs into ``num_classes`` equal blocks and sums
    each block (divided by ``tau``) to form per-class logits.

    Args:
        num_classes (int): Number of output classes / blocks. Default ``10``.
        tau (float): Temperature; each class logit is its block sum / ``tau``.
            Default ``1.0``.
    """

    def __init__(self, num_classes=10, tau=1.0, feature_ste=False):
        super().__init__()
        self.num_classes = num_classes
        self.tau = tau
        self.feature_ste = feature_ste

    def forward(self, x):
        b, w = x.shape
        if self.feature_ste:          # sum the SAME {0,1} bits seen at inference
            x = binarize_ste(x)       # (closes the soft/hard GroupSum gap)
        x = x.view(b, self.num_classes, w // self.num_classes)
        return x.sum(dim=2) / self.tau

    @torch.no_grad()
    def forward_hard(self, x):
        b, w = x.shape
        x = x.view(b, self.num_classes, w // self.num_classes)
        return x.sum(dim=2).float()


class LearnedDecoder(nn.Module):
    """Learned readout over the final logic features.

    ``kind``:
      * ``"linear"``   — plain ``Linear(width -> num_classes)``.
      * ``"linfull"``  — full ``Linear`` initialised to exactly GroupSum (block
        mean per class), then learns deviations. Requires divisibility.
      * ``"sumlinear"``— sum features into a 256-d state -> BatchNorm -> Linear.
      * ``"ternary"``  — per-feature, per-class ``{-1,0,+1}`` ternarized weights
        (a learnable generalization of GroupSum; deployable as a signed popcount).

    For ``linfull``/``sumlinear``/``ternary`` the features are straight-through
    binarized first (when ``feature_ste=True``, the default) so the readout trains
    on the same ``{0,1}`` features it sees at inference.

    Args:
        feature_ste (bool): Straight-through binarize the features before the
            readout. Default ``True``.
    """

    def __init__(self, kind, width, num_classes=10, feature_ste=True):
        super().__init__()
        self.kind = kind
        self.feature_ste = feature_ste
        if kind == "linear":
            self.dec = nn.Linear(width, num_classes)
        elif kind == "linfull":
            assert width % num_classes == 0
            self.dec = nn.Linear(width, num_classes)
            g = width // num_classes
            with torch.no_grad():
                self.dec.weight.zero_()
                for cls in range(num_classes):
                    self.dec.weight[cls, cls * g:(cls + 1) * g] = 1.0 / g
                self.dec.bias.zero_()
        elif kind == "ternary":
            assert width % num_classes == 0
            g = width // num_classes
            w = torch.zeros(num_classes, width)
            for cls in range(num_classes):
                w[cls, cls * g:(cls + 1) * g] = 1.0
            self.dec_w = nn.Parameter(w)
        elif kind == "sumlinear":
            self.hidden = 256
            assert width % self.hidden == 0, "width must be divisible by 256"
            self.dec_bn = nn.BatchNorm1d(self.hidden)
            self.dec_lin = nn.Linear(self.hidden, num_classes)
        else:
            raise ValueError(f"unknown decoder {kind!r}")

    def _decode(self, x):
        if self.kind == "linear":
            return self.dec(x)
        if self.kind == "ternary":
            if self.feature_ste:
                x = binarize_ste(x)
            return F.linear(x, ternary_ste(self.dec_w))
        if self.kind == "linfull":
            if self.feature_ste:
                x = binarize_ste(x)
            return self.dec(x)
        # sumlinear
        if self.feature_ste:
            x = binarize_ste(x)
        x = x.view(x.shape[0], self.hidden, -1).sum(2)
        return self.dec_lin(self.dec_bn(x))

    def forward(self, x):
        return self._decode(x)

    @torch.no_grad()
    def forward_hard(self, x):
        return self._decode(x.float())


def build_decoder(kind, width, num_classes=10, tau=1.0, feature_ste=True):
    """Construct a head by name: ``"groupsum"`` (default) or a :class:`LearnedDecoder`."""
    if kind == "groupsum":
        assert width % num_classes == 0, "width must be divisible by classes"
        return GroupSum(num_classes=num_classes, tau=tau, feature_ste=feature_ste)
    return LearnedDecoder(kind, width, num_classes, feature_ste=feature_ste)
