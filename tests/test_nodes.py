"""Alternative node types: k-input LUTs and attention-like pairwise logic."""
import pytest
import torch

from silogic import LUTkLayer, LUTkNet
from conftest import random_bits


# ---- LUTk ----------------------------------------------------------------
@pytest.mark.parametrize("k", [2, 3, 4])
def test_lutk_forward_range_and_hard(k):
    layer = LUTkLayer(20, 12, k=k, cand_k=4, seed=0)
    x = random_bits(6, 20)
    y = layer(x)
    assert y.shape == (6, 12)
    assert y.min() >= -1e-4 and y.max() <= 1 + 1e-4
    h = layer.forward_hard(x)
    assert set(h.unique().tolist()).issubset({0, 1})


def test_lutk_hardened_soft_matches_hard():
    """Saturate the LUT logits and connection selectors so the multilinear
    interpolation collapses to the hard truth-table lookup."""
    layer = LUTkLayer(16, 10, k=3, cand_k=4, seed=1)
    with torch.no_grad():
        layer.lut.copy_(torch.where(layer.lut > 0, 30.0, -30.0))  # saturate -> {0,1}
        sel = layer.conn.argmax(2)
        layer.conn.zero_()
        for o in range(layer.out_dim):
            for j in range(layer.k):
                layer.conn[o, j, sel[o, j]] = 30.0
    x = random_bits(5, 16)
    err = (layer(x) - layer.forward_hard(x).float()).abs().max()
    assert err < 1e-3


def test_lutk_net():
    net = LUTkNet(40, 20, 2, k=3, seed=0)
    x = random_bits(8, 40)
    assert net(x).shape == (8, 10)
    assert net.forward_hard(x).shape == (8, 10)
    assert net.num_luts() == 20 * 2
