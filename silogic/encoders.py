"""Learned input encoders (real-valued features -> binary), for LUT-node nets.

LUT nodes consume bits, so real inputs must be binarized. The library's
:func:`~silogic.binarize` uses fixed thermometer thresholds; this module adds a
**learnable** thermometer encoder (BitLogic, arXiv:2602.07400) whose thresholds
are trained jointly with the network and frozen for deployment.
"""
import torch
import torch.nn as nn


class LearnedThermometerEncoder(nn.Module):
    """Thermometer encoder with learnable per-feature thresholds.

    Each real feature ``x_f`` is encoded as ``bits`` bits ``1[x_f > t_{f,i}]``;
    the thresholds ``t`` are trainable. A straight-through estimator gives a hard
    ``{0,1}`` forward value with a smooth (sigmoid) gradient, so the encoder can be
    trained end-to-end and then discretized exactly at inference.

    Args:
        num_features (int): Number of real input features ``F``.
        bits (int): Thresholds (output bits) per feature; output width is ``F*bits``.
        init_lo (float): Low end of the initial threshold spread. Default ``0.0``.
        init_hi (float): High end of the initial threshold spread. Default ``1.0``.
        scale (float): Sigmoid sharpness of the straight-through soft pass. Default ``10.0``.

    Shape:
        input ``[B, F]`` -> output ``[B, F*bits]``.
    """

    def __init__(self, num_features, bits, init_lo=0.0, init_hi=1.0, scale=10.0):
        super().__init__()
        self.num_features = num_features
        self.bits = bits
        self.scale = scale
        t = torch.linspace(init_lo, init_hi, bits + 2)[1:-1]      # [bits], evenly spread
        self.thresholds = nn.Parameter(t.repeat(num_features, 1))  # [F, bits]

    def forward(self, x):
        d = x.unsqueeze(-1) - self.thresholds[None]              # [B, F, bits]
        hard = (d > 0).float()
        soft = torch.sigmoid(d * self.scale)
        out = hard + soft - soft.detach()                       # straight-through
        return out.reshape(x.shape[0], -1)

    @torch.no_grad()
    def forward_hard(self, x):
        d = x.unsqueeze(-1) - self.thresholds[None]
        return (d > 0).to(torch.uint8).reshape(x.shape[0], -1)
