"""Fused Triton kernels must match their pure-PyTorch reference paths.

These tests require a CUDA GPU with Triton; they are skipped otherwise.
"""
import torch

import silogic.warp as W
from silogic import LogicLayer, ConvLogicTree, WARPLayer
from conftest import requires_cuda, random_bits


@requires_cuda
def test_dense_logic_matches_pytorch():
    layer = LogicLayer(256, 512, connectome="TopK", k=8, seed=0).cuda().eval()
    x = random_bits(64, 256).cuda()
    assert getattr(layer, "use_triton_dense", False)
    y_triton = layer(x)
    layer.use_triton_dense = False
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
    assert getattr(conv, "use_triton_topk", False)

    def run(flag):
        conv.use_triton_topk = flag
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
def test_warp_logic_matches_pytorch_fwd_and_grad():
    layer = WARPLayer(256, 512, k=8, tau=0.5, seed=1).cuda()
    x = random_bits(64, 256).cuda().requires_grad_(True)

    W.WARP_USE_TRITON = True
    yk = layer(x); yk.sum().backward()
    gk_t, gk_x = layer.theta.grad.clone(), x.grad.clone()
    layer.theta.grad = None; x.grad = None

    W.WARP_USE_TRITON = False
    yp = layer(x); yp.sum().backward()
    W.WARP_USE_TRITON = True

    assert (yk - yp).abs().max() < 1e-5
    assert (gk_t - layer.theta.grad).abs().max() < 1e-3
    assert (gk_x - x.grad).abs().max() < 1e-3
