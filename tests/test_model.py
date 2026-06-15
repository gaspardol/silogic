"""Core FC logic network: gate algebra, connectomes, decoders, hard inference."""
import pytest
import torch

from silogic import (LogicLayer, LogicNet, GroupSum, BASIS_COEFFS, TRUTH_TABLES,
                     ste_threshold, sign_ste, ternary_ste)
from conftest import random_bits, harden_logic_layer

CONNECTOMES = ["F", "L", "TopK", "BlockTopK", "ST", "STW", "STT"]


def test_truth_tables_match_basis():
    """Each hard truth table is the basis polynomial evaluated at the corners."""
    corners = torch.tensor([[0., 0], [0, 1], [1, 0], [1, 1]])
    for i in range(16):
        c = BASIS_COEFFS[i]
        vals = (c[0] + c[1] * corners[:, 0] + c[2] * corners[:, 1]
                + c[3] * corners[:, 0] * corners[:, 1])
        assert torch.allclose(vals, TRUTH_TABLES[i].float()), f"gate {i}"


@pytest.mark.parametrize("conn", ["F", "L", "TopK"])
def test_basisproj_equals_fulleval(conn):
    """BasisProj and FullEval are numerically identical for shared params."""
    lb = LogicLayer(20, 30, connectome=conn, k=5, gate_eval="basis", seed=1)
    lf = LogicLayer(20, 30, connectome=conn, k=5, gate_eval="full", seed=1)
    lf.load_state_dict(lb.state_dict())
    x = torch.rand(8, 20)
    assert (lb(x) - lf(x)).abs().max() < 1e-5


@pytest.mark.parametrize("conn", CONNECTOMES)
def test_layer_forward_shape_and_range(conn):
    layer = LogicLayer(24, 16, connectome=conn, k=4, seed=2)
    layer.train()
    x = random_bits(5, 24)
    y = layer(x)
    assert y.shape == (5, 16)
    assert y.min() >= -1e-4 and y.max() <= 1 + 1e-4


@pytest.mark.parametrize("conn", CONNECTOMES)
def test_forward_hard_is_binary(conn):
    layer = LogicLayer(24, 16, connectome=conn, k=4, seed=3)
    layer.eval()
    x = random_bits(7, 24)
    h = layer.forward_hard(x)
    assert h.dtype == torch.uint8
    assert h.shape == (7, 16)
    assert set(h.unique().tolist()).issubset({0, 1})


@pytest.mark.parametrize("conn", ["F", "L", "TopK", "BlockTopK"])
def test_hardened_soft_matches_hard(conn):
    # ST/STW/STT use BatchNorm-thresholded inputs that don't reduce to a clean
    # one-hot, so they are exercised by the shape/range tests instead.
    """With one-hot gate/conn params, relaxed forward == hard forward on bits."""
    layer = LogicLayer(12, 16, connectome=conn, k=4, seed=2)
    harden_logic_layer(layer)
    layer.eval()
    x = random_bits(6, 12)
    err = (layer(x) - layer.forward_hard(x).float()).abs().max()
    assert err < 1e-3


@pytest.mark.parametrize("conn", CONNECTOMES)
def test_gradient_flows(conn):
    layer = LogicLayer(24, 16, connectome=conn, k=4, seed=4)
    layer.train()
    x = random_bits(8, 24)
    layer(x).sum().backward()
    grads = [p.grad for p in layer.parameters() if p.requires_grad]
    assert grads and all(g is not None for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)


def test_groupsum_head():
    head = GroupSum(num_classes=10, tau=4.0)
    x = random_bits(3, 50)
    assert head(x).shape == (3, 10)
    assert head.forward_hard(x).shape == (3, 10)


def test_logicnet_end_to_end():
    net = LogicNet(in_dim=64, width=40, depth=3, connectome="TopK", k=8, seed=0)
    x = random_bits(16, 64)
    assert net(x).shape == (16, 10)
    assert net.forward_hard(x).shape == (16, 10)
    assert net.num_gates() == 40 * 3


@pytest.mark.parametrize("decoder,width", [
    ("groupsum", 40), ("linear", 40), ("linfull", 40),
    ("ternary", 40), ("sumlinear", 512),
])
def test_decoders(decoder, width):
    net = LogicNet(in_dim=64, width=width, depth=2, connectome="TopK", k=8,
                   seed=0, decoder=decoder)
    x = random_bits(8, 64)
    out = net(x)
    assert out.shape == (8, 10)
    out.sum().backward()                      # trains end to end
    assert net.forward_hard(x).shape == (8, 10)


def test_wire_residual():
    net = LogicNet(in_dim=40, width=40, depth=3, connectome="TopK", k=8,
                   seed=0, wire_residual=0.25)
    assert net.wire_r == 10
    x = random_bits(4, 40)
    assert net(x).shape == (4, 10)
    assert net.forward_hard(x).shape == (4, 10)


def test_ste_helpers():
    w = torch.tensor([-0.9, -0.2, 0.2, 0.9])
    assert torch.equal(sign_ste(w).detach().sign(), torch.tensor([-1., -1, 1, 1]))
    assert torch.equal(ternary_ste(w).detach(), torch.tensor([-1., 0, 0, 1]))
    s = torch.tensor([-1.0, 1.0])
    assert torch.equal(ste_threshold(s).detach(), torch.tensor([0., 1.]))
    # straight-through: gradient passes despite the hard threshold
    s = torch.tensor([0.3], requires_grad=True)
    ste_threshold(s).backward()
    assert s.grad is not None and s.grad.item() > 0


@pytest.mark.parametrize("gate_select", ["gumbel", "hard"])
def test_gate_select_modes(gate_select):
    """Stochastic Gumbel-ST and deterministic hard-gate ST run (as a per-layer
    constructor arg) and stay binary at inference."""
    net = LogicNet(in_dim=64, width=40, depth=2, connectome="TopK", k=8, seed=0,
                   gate_select=gate_select)
    x = random_bits(8, 64)
    net.train()
    net(x).sum().backward()
    net.eval()
    assert net.forward_hard(x).shape == (8, 10)


def test_per_layer_gate_select():
    """A network can mix soft and hard gates: gate_select as a per-layer list."""
    net = LogicNet(in_dim=64, width=40, depth=3, connectome="TopK", k=8, seed=0,
                   gate_select=["softmax", "hard", "gumbel"])
    assert [l.node.gate_select for l in net.layers] == ["softmax", "hard", "gumbel"]
    x = random_bits(8, 64)
    net.train()
    net(x).sum().backward()
    assert net.forward_hard(x).shape == (8, 10)
