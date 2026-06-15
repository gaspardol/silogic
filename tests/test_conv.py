"""Convolutional logic-gate trees, OR pooling, and the LogicTreeNet stack."""
import pytest
import torch

from silogic import ConvLogicTree, OrPool, LogicTreeNet
from conftest import random_bits


@pytest.mark.parametrize("connect", ["fixed", "topk"])
def test_convtree_forward_shape_and_range(connect):
    conv = ConvLogicTree(3, 8, kernel=3, tree_depth=2, connect=connect, k=4,
                         n_chan=2, seed=0)
    x = random_bits(2, 3, 10, 10)
    y = conv(x)
    assert y.shape == (2, 8, 10, 10)            # padding=1 keeps H,W
    assert y.min() >= -1e-4 and y.max() <= 1 + 1e-4


@pytest.mark.parametrize("connect", ["fixed", "topk"])
def test_convtree_hard_is_binary(connect):
    conv = ConvLogicTree(3, 8, kernel=3, tree_depth=2, connect=connect, k=4,
                         n_chan=2, seed=0)
    x = random_bits(2, 3, 10, 10)
    h = conv.forward_hard(x)
    assert h.dtype == torch.uint8
    assert h.shape == (2, 8, 10, 10)
    assert set(h.unique().tolist()).issubset({0, 1})


def test_orpool_halves_spatial():
    pool = OrPool(2)
    x = random_bits(2, 4, 8, 8)
    assert pool(x).shape == (2, 4, 4, 4)
    assert pool.forward_hard(x).shape == (2, 4, 4, 4)
    # OR pooling = spatial max
    assert torch.equal(pool(x), torch.nn.functional.max_pool2d(x, 2, 2))


def test_convtree_gradient_flows():
    conv = ConvLogicTree(3, 8, kernel=3, tree_depth=2, connect="topk", k=4,
                         n_chan=2, seed=1)
    x = random_bits(2, 3, 8, 8)
    conv(x).sum().backward()
    assert conv.conn.grad is not None and conv.conn.grad.abs().sum() > 0


@pytest.mark.parametrize("decoder", ["groupsum", "linear"])
def test_logictreenet_end_to_end(decoder):
    net = LogicTreeNet(in_channels=3, in_hw=16, channels=[8, 8],
                       head_widths=[40], tree_depth=2, k=4, head_k=4,
                       n_chan=2, seed=0, decoder=decoder)
    x = random_bits(2, 3, 16, 16)
    out = net(x)
    assert out.shape == (2, 10)
    out.sum().backward()
    assert net.forward_hard(x).shape == (2, 10)
    assert net.gate_count(16) > 0
