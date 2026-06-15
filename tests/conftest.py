"""Shared pytest fixtures and helpers for the silogic test suite."""
import pytest
import torch

CUDA = torch.cuda.is_available()
# Skip marker for tests that need a GPU (the fused Triton kernels).
requires_cuda = pytest.mark.skipif(not CUDA, reason="requires CUDA + Triton")


@pytest.fixture(autouse=True)
def _seed():
    """Deterministic RNG for every test."""
    torch.manual_seed(0)


@pytest.fixture
def cpu():
    return "cpu"


def random_bits(*shape):
    """A {0,1} float tensor — the natural input to a logic network."""
    return (torch.rand(*shape) > 0.5).float()


def harden_logic_layer(layer):
    """Make a LogicLayer's gate + connection params one-hot at their argmax so
    its relaxed ``forward`` (on binary inputs) coincides with ``forward_hard``."""
    with torch.no_grad():
        g = layer.gate_logits.argmax(1)
        layer.gate_logits.zero_()
        layer.gate_logits[torch.arange(layer.out_dim), g] = 30.0
        if hasattr(layer, "conn_a") and layer.connectome in ("TopK", "BlockTopK", "L"):
            sa = layer.conn_a.argmax(1)
            sb = layer.conn_b.argmax(1)
            layer.conn_a.zero_(); layer.conn_a[torch.arange(layer.out_dim), sa] = 30.0
            layer.conn_b.zero_(); layer.conn_b[torch.arange(layer.out_dim), sb] = 30.0
