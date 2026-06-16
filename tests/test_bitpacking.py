"""Correctness tests for silogic.bitpacking.

Each test verifies bit-exact agreement with torch forward_hard so we can
detect regressions in the pack/unpack math or the gate algebra.
"""
import numpy as np
import pytest
import torch

from silogic.models import LogicNet, WARPNet, LUTkNet, LogicConvNet
from silogic.bitpacking import (
    convert_logic_net, convert_logic_conv_net, pack_bits, unpack_bits, n_words
)
from silogic.bitpacking.ops import group_sum_packed


# ── helpers ────────────────────────────────────────────────────────────────────

def _random_bits(shape, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, shape, dtype=np.uint8)


def _assert_exact(name, model, x_np):
    """Check bitpacked logit argmax matches forward_hard argmax."""
    model.eval()
    x_t = torch.from_numpy(x_np).to(torch.uint8)
    with torch.no_grad():
        ref_logits = model.forward_hard(x_t).cpu().numpy()
    bp = convert_logic_net(model)
    bp_logits = bp(x_np)
    assert ref_logits.shape == bp_logits.shape, f"{name}: shape mismatch"
    assert (ref_logits.argmax(1) == bp_logits.argmax(1)).all(), \
        f"{name}: argmax mismatch on {int((ref_logits.argmax(1) != bp_logits.argmax(1)).sum())} samples"


# ── pack / unpack round-trip ──────────────────────────────────────────────────

@pytest.mark.parametrize("B,dim", [(1, 1), (63, 8), (64, 64), (65, 256), (200, 784)])
def test_pack_roundtrip(B, dim):
    x = _random_bits((B, dim))
    assert (unpack_bits(pack_bits(x), B) == x).all()


def test_n_words():
    assert n_words(1)  == 1
    assert n_words(64) == 1
    assert n_words(65) == 2
    assert n_words(128) == 2
    assert n_words(129) == 3


# ── gate16 / topk ─────────────────────────────────────────────────────────────

def test_gate16_topk():
    model = LogicNet(128, 200, 4, num_classes=10, connectome="topk", k=8)
    _assert_exact("gate16/topk", model, _random_bits((200, 128)))


def test_gate16_fixed():
    model = LogicNet(128, 200, 4, num_classes=10, connectome="fixed")
    _assert_exact("gate16/fixed", model, _random_bits((200, 128)))


def test_gate16_dense():
    model = LogicNet(64, 200, 2, num_classes=10, connectome="dense")
    _assert_exact("gate16/dense", model, _random_bits((100, 64)))


# ── walsh / warp ──────────────────────────────────────────────────────────────

def test_walsh_topk():
    model = WARPNet(128, 200, 4, num_classes=10, k=8)
    _assert_exact("walsh/topk", model, _random_bits((200, 128)))


# ── multilinear LUT ───────────────────────────────────────────────────────────

def test_multilinear_arity2():
    """arity=2 multilinear → fast gate16 equivalent path."""
    model = LogicNet(64, 100, 2, num_classes=10, node="multilinear", arity=2, k=4)
    _assert_exact("multilinear/arity=2", model, _random_bits((100, 64)))


def test_lut_arity4():
    """arity=4 LUT → BitpackedLUTLayer path."""
    model = LUTkNet(64, 100, 2, k=4, num_classes=10, cand_k=4)
    _assert_exact("LUTkNet/arity=4", model, _random_bits((100, 64)))


# ── wire residual ─────────────────────────────────────────────────────────────

def test_wire_residual():
    model = LogicNet(128, 200, 4, num_classes=10, k=8, wire_residual=0.1)
    _assert_exact("gate16/wire_residual=0.1", model, _random_bits((100, 128)))


# ── group_sum_packed ──────────────────────────────────────────────────────────

def test_group_sum_matches_reference():
    """group_sum_packed must match naive unpack+sum."""
    B, width, nc = 137, 100, 10
    x = _random_bits((B, width))
    packed = pack_bits(x)
    ref = x.reshape(B, nc, width // nc).sum(2).astype(np.float32)
    fast = group_sum_packed(packed, B, nc, tau=1.0)
    np.testing.assert_allclose(fast, ref, atol=1e-5)


# ── large batch ───────────────────────────────────────────────────────────────

def test_large_batch():
    model = LogicNet(256, 500, 4, num_classes=10, k=8)
    _assert_exact("large_batch B=1024", model, _random_bits((1024, 256)))


# ── conv tree (BitpackedConvTreeLayer) ────────────────────────────────────────

def _assert_conv_exact(name, model, x_np):
    """Check bitpacked LogicConvNet matches forward_hard argmax."""
    model.eval()
    x_t = torch.from_numpy(x_np).to(torch.uint8)
    with torch.no_grad():
        ref_logits = model.forward_hard(x_t).cpu().numpy()
    bp = convert_logic_conv_net(model)
    bp_logits = bp(x_np)
    assert ref_logits.shape == bp_logits.shape, f"{name}: shape mismatch"
    assert (ref_logits.argmax(1) == bp_logits.argmax(1)).all(), (
        f"{name}: argmax mismatch on "
        f"{int((ref_logits.argmax(1) != bp_logits.argmax(1)).sum())} samples"
    )


def test_conv_tree_fixed():
    """ConvLogicTree with fixed connectivity (gate16) → BitpackedConvTreeLayer."""
    # 8×8 image: block1 → 4×4, block2 → 2×2; feat_dim=16*4=64; head→80 (÷10)
    model = LogicConvNet(
        in_channels=3, in_hw=8, channels=[8, 16],
        head_widths=[80], num_classes=10,
        connect="fixed", kernel=3, tree_depth=2, n_chan=2,
        use_triton=False,
    )
    x = _random_bits((20, 3, 8, 8))
    _assert_conv_exact("conv/gate16/fixed", model, x)


def test_conv_tree_topk():
    """ConvLogicTree with topk connectivity (gate16) → BitpackedConvTreeLayer."""
    model = LogicConvNet(
        in_channels=3, in_hw=8, channels=[8, 16],
        head_widths=[80], num_classes=10,
        connect="topk", k=4, kernel=3, tree_depth=2, n_chan=2,
        use_triton=False,
    )
    x = _random_bits((20, 3, 8, 8))
    _assert_conv_exact("conv/gate16/topk", model, x)


def test_conv_tree_large_batch():
    """Conv bitpacking: verify across a larger batch (B=64, 16×16 image)."""
    # 16×16 → 8×8 → 4×4; feat_dim=8*16=128; head→80 (÷10)
    model = LogicConvNet(
        in_channels=3, in_hw=16, channels=[4, 8],
        head_widths=[80], num_classes=10,
        connect="fixed", kernel=3, tree_depth=2, n_chan=2,
        use_triton=False,
    )
    x = _random_bits((64, 3, 16, 16))
    _assert_conv_exact("conv/gate16/fixed/B=64/16x16", model, x)


def test_conv_tree_walsh():
    """Walsh (arity=2) ConvLogicTree → converted to gate16 → BitpackedConvTreeLayer."""
    model = LogicConvNet(
        in_channels=3, in_hw=8, channels=[8, 16],
        head_widths=[80], num_classes=10,
        node="walsh", connect="fixed", kernel=3, tree_depth=2, n_chan=2,
        use_triton=False,
    )
    x = _random_bits((20, 3, 8, 8))
    _assert_conv_exact("conv/walsh/fixed", model, x)


def test_conv_tree_multilinear_arity2():
    """Multilinear arity=2 ConvLogicTree → gate16 conversion → BitpackedConvTreeLayer."""
    model = LogicConvNet(
        in_channels=3, in_hw=8, channels=[8, 16],
        head_widths=[80], num_classes=10,
        node="multilinear", arity=2, connect="fixed",
        kernel=3, tree_depth=2, n_chan=2,
        use_triton=False,
    )
    x = _random_bits((20, 3, 8, 8))
    _assert_conv_exact("conv/multilinear/arity=2/fixed", model, x)


def test_conv_tree_multilinear_lut():
    """Multilinear arity=4 ConvLogicTree → LUT truth tables → BitpackedConvLUTLayer."""
    model = LogicConvNet(
        in_channels=3, in_hw=8, channels=[8, 16],
        head_widths=[80], num_classes=10,
        node="multilinear", arity=4, connect="fixed",
        kernel=3, n_chan=2,
        use_triton=False,
    )
    x = _random_bits((20, 3, 8, 8))
    _assert_conv_exact("conv/multilinear/arity=4/fixed", model, x)


def test_conv_tree_hybrid_lut():
    """Hybrid arity=4 ConvLogicTree → LUT truth tables → BitpackedConvLUTLayer."""
    model = LogicConvNet(
        in_channels=3, in_hw=8, channels=[8, 16],
        head_widths=[80], num_classes=10,
        node="hybrid", arity=4, connect="fixed",
        kernel=3, n_chan=2,
        use_triton=False,
    )
    x = _random_bits((20, 3, 8, 8))
    _assert_conv_exact("conv/hybrid/arity=4/fixed", model, x)
