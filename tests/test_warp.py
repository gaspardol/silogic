"""WARP (Walsh–Hadamard) logic layers: gate algebra, arity, hard inference."""
import math
import pytest
import torch

import silogic.warp as W
from silogic import WARPLayer, WARPNet, WARPLayerN, WARPNetN
from conftest import random_bits


def test_theta_to_coef_change_of_variables():
    """_coef() must give c with z = c0 + c1 a + c2 b + c3 ab equal to the
    theta form z = t0 + t1 u + t2 v + t3 uv where u=2a-1, v=2b-1."""
    layer = WARPLayer(16, 12, k=4, tau=0.5, seed=0)
    with torch.no_grad():
        layer.theta.normal_()
    a = torch.rand(5, 12)
    b = torch.rand(5, 12)
    z_theta = layer._z(a, b)
    c = layer._coef()                              # [out,4]
    z_coef = c[:, 0] + c[:, 1] * a + c[:, 2] * b + c[:, 3] * (a * b)
    assert (z_theta - z_coef).abs().max() < 1e-5


def test_residual_init_biases_passthrough():
    layer = WARPLayer(16, 12, k=4, tau=0.5, residual_p=0.9, seed=0)
    # the v (2nd-input) Walsh coefficient should carry the residual logit
    assert layer.theta[:, 2].abs().min() > 0
    assert torch.allclose(layer.theta[:, 0], torch.zeros(12))


def test_warp_forward_range_and_hard():
    layer = WARPLayer(16, 12, k=4, tau=0.5, seed=0)
    x = random_bits(6, 16)
    y = layer(x)
    assert y.shape == (6, 12)
    assert y.min() >= 0 and y.max() <= 1            # sigmoid output
    h = layer.forward_hard(x)
    assert set(h.unique().tolist()).issubset({0., 1.})


def test_warp_gradient_to_theta():
    layer = WARPLayer(16, 12, k=4, tau=0.5, seed=1)
    x = random_bits(8, 16)
    layer(x).sum().backward()
    assert layer.theta.grad is not None and layer.theta.grad.abs().sum() > 0


def test_warpnet_end_to_end():
    net = WARPNet(64, 40, 3, tau=0.5, seed=0)
    x = random_bits(8, 64)
    assert net(x).shape == (8, 10)
    assert net.forward_hard(x).shape == (8, 10)


@pytest.mark.parametrize("arity", [2, 3, 4])
def test_warpnet_arity(arity):
    net = WARPNetN(64, 40, 2, arity=arity, tau=0.5, residual_p=0.9, seed=0)
    x = random_bits(8, 64)
    assert net(x).shape == (8, 10)
    assert net.forward_hard(x).shape == (8, 10)
    # theta has 2**arity entries per node
    assert net.layers[0].theta.shape == (40, 2 ** arity)


def test_warp_gumbel_smoothing_runs():
    W.WARP_GUMBEL["enabled"] = True
    try:
        net = WARPNet(64, 40, 2, tau=0.5, seed=0)
        net.train()
        net((random_bits(8, 64))).sum().backward()
    finally:
        W.WARP_GUMBEL["enabled"] = False
