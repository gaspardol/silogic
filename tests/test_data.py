"""Data pipeline: thermometer binarization, spatial encoding, edge detectors."""
import os
import pytest
import torch

from silogic import (binarize, binarize_spatial, edge_bits, get_dataset,
                     MNIST_THRESH, SEVEN_THRESH, FOUR_THRESH, EDGE_THRESH)
from silogic.data import DATA_ROOT


def test_binarize_thermometer():
    x = torch.tensor([[[[0.1, 0.3], [0.5, 0.9]]]])     # [1,1,2,2]
    bits = binarize(x, [0.2, 0.6])
    assert bits.dtype == torch.uint8
    # block 1: >=0.2, block 2: >=0.6 (row-major flatten)
    expected = torch.tensor([[0, 1, 1, 1, 0, 0, 0, 1]], dtype=torch.uint8)
    assert torch.equal(bits, expected)


def test_binarize_spatial_shape():
    x = torch.rand(2, 3, 8, 8)
    out = binarize_spatial(x, SEVEN_THRESH)
    assert out.shape == (2, 3 * len(SEVEN_THRESH), 8, 8)
    assert out.dtype == torch.uint8


def test_edge_bits_shape():
    x = torch.rand(2, 3, 8, 8)
    out = edge_bits(x)
    # 3 colours x 3 detectors x 2 signs x len(thr)
    assert out.shape == (2, 3 * 3 * 2 * len(EDGE_THRESH), 8, 8)
    assert out.dtype == torch.uint8


def test_threshold_constants():
    assert MNIST_THRESH == [0.25]
    assert len(SEVEN_THRESH) == 7
    assert len(FOUR_THRESH) == 4


@pytest.mark.skipif(not os.path.isdir(os.path.join(DATA_ROOT, "MNIST")),
                    reason="MNIST not downloaded")
def test_get_dataset_mnist_unaugmented():
    Xtr, ytr, Xte, yte, in_dim = get_dataset("mnist", MNIST_THRESH,
                                             augment=False, device="cpu")
    assert in_dim == 28 * 28               # single threshold
    assert Xtr.shape == (60000, 784) and Xte.shape == (10000, 784)
    assert Xtr.dtype == torch.uint8
    assert set(Xtr[:100].unique().tolist()).issubset({0, 1})
