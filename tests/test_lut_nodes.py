"""Unified LUT-node relaxations (probabilistic/hybrid/linear/polynomial),
node-agnostic residual init, and the learned thermometer encoder."""
import pytest
import torch

from silogic import (LUTNodeLayer, LUTNodeNet, LearnedThermometerEncoder,
                     RELAXATIONS)
from conftest import random_bits


@pytest.mark.parametrize("relaxation", RELAXATIONS)
def test_layer_forward_hard_grad(relaxation):
    lay = LUTNodeLayer(40, 20, arity=4, relaxation=relaxation, k=4, seed=1)
    x = random_bits(8, 40)
    y = lay(x)
    assert y.shape == (8, 20)
    assert y.min() >= -1e-4 and y.max() <= 1 + 1e-4
    h = lay.forward_hard(x)
    assert h.dtype == torch.uint8 and set(h.unique().tolist()).issubset({0, 1})
    lay(x).sum().backward()
    grads = [p.grad for p in lay.parameters() if p.requires_grad]
    assert grads and all(g is not None and g.abs().sum() > 0 for g in grads)


@pytest.mark.parametrize("relaxation", RELAXATIONS)
def test_boundary_consistency(relaxation):
    """With one-hot connections, (soft forward >= 0.5) == forward_hard on bits —
    i.e. the relaxation discretizes to exactly the deployed truth table."""
    lay = LUTNodeLayer(16, 12, arity=3, relaxation=relaxation, k=4, seed=2)
    with torch.no_grad():                         # harden the input selection
        sel = lay.conn.argmax(2)
        lay.conn.zero_()
        for o in range(lay.out_dim):
            for s in range(lay.n):
                lay.conn[o, s, sel[o, s]] = 30.0
    lay.eval()
    x = random_bits(6, 16)
    soft_disc = (lay(x) >= 0.5).to(torch.uint8)
    assert torch.equal(soft_disc, lay.forward_hard(x))


@pytest.mark.parametrize("relaxation", RELAXATIONS)
def test_residual_init_passes_through_first_input(relaxation):
    """residual_p ~ 1 makes each node a (near) identity on its first input."""
    lay = LUTNodeLayer(32, 24, arity=4, relaxation=relaxation, k=4,
                       residual_p=0.99, seed=3)
    x = random_bits(16, 32)
    first_idx = torch.gather(lay.cand[:, 0, :], 1,
                             lay.conn[:, 0, :].argmax(1, keepdim=True)).squeeze(1)
    first_input = x[:, first_idx]                 # [B, out] the node's 1st input
    match = (lay.forward_hard(x).float() == first_input).float().mean()
    assert match > 0.9


def test_net_end_to_end():
    net = LUTNodeNet(40, 20, 2, arity=4, relaxation="hybrid", k=4, seed=0)
    x = random_bits(8, 40)
    assert net(x).shape == (8, 10)
    assert net.forward_hard(x).shape == (8, 10)


def test_unknown_relaxation_raises():
    with pytest.raises(ValueError):
        LUTNodeLayer(10, 4, relaxation="nope")


def test_learned_thermometer_encoder():
    enc = LearnedThermometerEncoder(16, bits=4)
    x = torch.rand(8, 16, requires_grad=True)
    e = enc(x)
    assert e.shape == (8, 64)
    assert (e.detach() - e.detach().round()).abs().max() < 1e-5   # STE -> {0,1}
    e.sum().backward()
    assert enc.thresholds.grad is not None and enc.thresholds.grad.abs().sum() > 0
    h = enc.forward_hard(x)
    assert h.dtype == torch.uint8 and h.shape == (8, 64)
