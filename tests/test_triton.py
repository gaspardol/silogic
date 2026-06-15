"""Fused Triton kernels must match their pure-PyTorch reference paths.

These tests require a CUDA GPU with Triton; they are skipped otherwise.
"""
import pytest
import torch

from silogic import LogicLayer, ConvLogicTree, WARPLayer
from conftest import requires_cuda, random_bits


@requires_cuda
def test_dense_logic_matches_pytorch():
    layer = LogicLayer(256, 512, connectome="TopK", k=8, seed=0).cuda().eval()
    x = random_bits(64, 256).cuda()
    assert layer._fused_ok(x)
    y_triton = layer(x)
    layer.use_triton = False
    y_torch = layer(x)
    assert (y_triton - y_torch).abs().max() < 1e-4


@requires_cuda
def test_tree_conv_matches_pytorch():
    conv = ConvLogicTree(6, 16, kernel=3, tree_depth=2, connect="fixed",
                         n_chan=2, seed=0).cuda().eval()
    x = random_bits(4, 6, 16, 16).cuda()
    assert getattr(conv, "use_triton", False)
    y_triton = conv(x)
    conv.use_triton = False
    y_torch = conv(x)
    assert (y_triton - y_torch).abs().max() < 1e-4


@requires_cuda
def test_tree_conv_topk_matches_pytorch_fwd_and_grad():
    conv = ConvLogicTree(6, 16, kernel=3, tree_depth=3, connect="topk", k=4,
                         n_chan=2, seed=1).cuda().eval()
    x0 = random_bits(4, 6, 16, 16).cuda()
    assert conv.use_triton

    def run(flag):
        conv.use_triton = flag
        x = x0.clone().requires_grad_(True)
        for p in conv.parameters():
            p.grad = None
        y = conv(x)
        y.pow(2).sum().backward()
        return y.detach(), x.grad.clone(), conv.conn.grad.clone(), \
            conv.gate_logits[0].grad.clone()

    yt, gxt, gct, ggt = run(True)
    yp, gxp, gcp, ggp = run(False)
    assert (yt - yp).abs().max() < 1e-4          # forward
    assert (gxt - gxp).abs().max() < 1e-3        # grad input
    assert (gct - gcp).abs().max() < 1e-3        # grad connections
    assert (ggt - ggp).abs().max() < 1e-3        # grad gate logits


@requires_cuda
@pytest.mark.parametrize("connect", ["topk", "fixed"])
@pytest.mark.parametrize("arity,depth", [(4, 1), (2, 2), (2, 3)])
def test_conv_hybrid_matches_pytorch_fwd_and_grad(arity, depth, connect):
    # image-reading hybrid LUT-tree kernel (no unfold), any depth
    conv = ConvLogicTree(6, 12, node="hybrid", arity=arity, tree_depth=depth,
                         connect=connect, k=6, n_chan=3, seed=1).cuda()
    x0 = random_bits(4, 6, 16, 16).cuda()
    assert conv.use_triton and conv.d == depth
    learn = hasattr(conv, "conn")           # topk learns leaf connections; fixed doesn't

    def run(flag):
        conv.use_triton = flag
        x = x0.clone().requires_grad_(True)
        for p in conv.parameters():
            p.grad = None
        y = conv(x)
        y.pow(2).sum().backward()
        out = [y.detach(), x.grad.clone()]
        out += [nd.logits.grad.clone() for nd in conv.tree_nodes]  # every level's LUT
        if learn:
            out.append(conv.conn.grad.clone())                     # leaf connections
        return out

    k_, p_ = run(True), run(False)
    assert (k_[0] - p_[0]).abs().max() < 1e-4    # forward (discrete value)
    for a, b in zip(k_[1:], p_[1:]):
        assert (a - b).abs().max() < 1e-2        # grads: image, per-level logits, conn


@requires_cuda
@pytest.mark.parametrize("connect", ["topk", "fixed"])
@pytest.mark.parametrize("arity", [2, 3, 4])
def test_multilinear_hybrid_matches_pytorch_fwd_and_grad(arity, connect):
    layer = LogicLayer(256, 512, node="hybrid", arity=arity, connectome=connect,
                       k=8, seed=1).cuda()
    x0 = random_bits(64, 256).cuda()
    assert layer._fused_ok(x0)
    learn = hasattr(layer._conn, "conn")    # topk learns connections; fixed doesn't

    def run(flag):
        layer.use_triton = flag
        x = x0.clone().requires_grad_(True)
        layer.node.logits.grad = None
        if learn:
            layer._conn.conn.grad = None
        y = layer(x)
        y.pow(2).sum().backward()
        out = [y.detach(), x.grad.clone(), layer.node.logits.grad.clone()]
        if learn:
            out.append(layer._conn.conn.grad.clone())
        return out

    k_, p_ = run(True), run(False)
    assert (k_[0] - p_[0]).abs().max() < 1e-4    # forward (discrete value)
    assert (k_[1] - p_[1]).abs().max() < 1e-3    # grad input
    assert (k_[2] - p_[2]).abs().max() < 1e-3    # grad LUT logits
    if learn:
        assert (k_[3] - p_[3]).abs().max() < 1e-3  # grad connections


@requires_cuda
def test_warp_logic_matches_pytorch_fwd_and_grad():
    layer = WARPLayer(256, 512, k=8, tau=0.5, seed=1).cuda()
    x = random_bits(64, 256).cuda().requires_grad_(True)

    layer.use_triton = True
    yk = layer(x); yk.sum().backward()
    gk_t, gk_x = layer.theta.grad.clone(), x.grad.clone()
    layer.theta.grad = None; x.grad = None

    layer.use_triton = False
    yp = layer(x); yp.sum().backward()
    layer.use_triton = True

    assert (yk - yp).abs().max() < 1e-5
    assert (gk_t - layer.theta.grad).abs().max() < 1e-3
    assert (gk_x - x.grad).abs().max() < 1e-3
