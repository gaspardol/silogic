"""Convolutional logic-gate trees, OR pooling, and the LogicTreeNet stack."""
import pytest
import torch

from silogic import (ConvLogicTree, ConvLogicLayer, OrPool, LogicTreeNet,
                     LogicConvNet)
from conftest import random_bits

# gate16/walsh compose as a 2-input gate tree; the rest are flat n-input LUTs.
CONV_NODES = ["gate16", "walsh", "multilinear", "hybrid", "linear", "polynomial"]
FLAT_NODES = ["multilinear", "hybrid", "linear", "polynomial"]


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


# ---- node-type compatibility (conv layer works with every node type) -----
@pytest.mark.parametrize("node", CONV_NODES)
@pytest.mark.parametrize("connect", ["topk", "fixed"])
def test_convlayer_node_types(node, connect):
    conv = ConvLogicLayer(6, 8, node=node, connect=connect, k=4, n_chan=2,
                          tree_depth=2, arity=4, seed=1)
    x = random_bits(2, 6, 10, 10)
    y = conv(x)
    assert y.shape == (2, 8, 10, 10)
    assert y.min() >= -1e-4 and y.max() <= 1 + 1e-4
    h = conv.forward_hard(x)
    assert h.dtype == torch.uint8 and h.shape == (2, 8, 10, 10)
    assert set(h.unique().tolist()).issubset({0, 1})
    # gradients reach the node parameters
    conv(x).sum().backward()
    grads = [p.grad for p in conv.parameters() if p.requires_grad]
    assert grads and any(g is not None and g.abs().sum() > 0 for g in grads)


@pytest.mark.parametrize("node", FLAT_NODES)
def test_convlayer_flat_boundary_consistency(node):
    """With one-hot leaf selection, (soft >= 0.5) == forward_hard for the flat
    n-input LUT nodes — the relaxation discretizes to the deployed truth table."""
    conv = ConvLogicLayer(6, 8, node=node, connect="topk", k=4, n_chan=2,
                          arity=3, residual_init=False, seed=3).eval()
    with torch.no_grad():
        sel = conv.conn.argmax(2)
        conv.conn.zero_()
        for o in range(conv.n):
            for leaf in range(conv.leaves):
                conv.conn[o, leaf, sel[o, leaf]] = 30.0
    x = random_bits(2, 6, 10, 10)
    soft_disc = (conv(x) >= 0.5).to(torch.uint8)
    assert torch.equal(soft_disc, conv.forward_hard(x))


@pytest.mark.parametrize("node", CONV_NODES)
def test_logicconvnet_node_types(node):
    net = LogicConvNet(in_channels=6, in_hw=16, channels=[8, 16],
                       head_widths=[40], tree_depth=2, node=node, arity=4,
                       node_tau=0.5, k=4, head_k=4, n_chan=2, seed=0)
    x = random_bits(2, 6, 16, 16)
    assert net(x).shape == (2, 10)
    assert net.forward_hard(x).shape == (2, 10)
    assert net.gate_count(16) > 0


@pytest.mark.parametrize("node", CONV_NODES)
def test_convlayer_depth_tree(node):
    """A depth-2 node tree: gate16/walsh use 2-ary (4 leaves), n-input use
    `arity`-ary; gates_per_output follows the k-ary tree node count."""
    conv = ConvLogicLayer(6, 8, node=node, connect="topk", tree_depth=2, arity=3,
                          k=4, n_chan=2, seed=4)
    k = 2 if node in ("gate16", "walsh") else 3
    assert conv.leaves == k ** 2
    assert conv.gates_per_output == (k ** 2 - 1) // (k - 1)
    x = random_bits(2, 6, 10, 10)
    assert conv(x).shape == (2, 8, 10, 10)
    assert conv.forward_hard(x).shape == (2, 8, 10, 10)


def test_convlayer_default_depth():
    """Default depth: 2 for the gate families, 1 for the n-input LUT families."""
    assert ConvLogicLayer(6, 8, node="gate16").d == 2
    assert ConvLogicLayer(6, 8, node="walsh").d == 2
    assert ConvLogicLayer(6, 8, node="multilinear").d == 1
    assert ConvLogicLayer(6, 8, node="linear").d == 1


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
