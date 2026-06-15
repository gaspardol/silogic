"""ImageNet as spatial binary channels for convolutional logic nets.

Mirrors the CIFAR spatial encoding (silogic/data.py::get_cifar_spatial) but
streams from the extracted ImageFolder dirs (1.28M train / 50k val, 1000
classes) instead of caching the whole set in memory/GPU.

Pipeline per image:
  * Resize to a small resolution (32x32) suitable for a logic net.
      train: Resize(36) + RandomCrop(32) + RandomHorizontalFlip
      val:   Resize(32) + CenterCrop(32)
  * Binarize into spatial binary channels with the SAME thermometer + optional
    edge-detector encoding as get_cifar_spatial (reuse binarize_spatial /
    edge_bits). Output: uint8 [C, 32, 32].

The DataLoader yields (binary_uint8 [B, C, 32, 32], label) batches on the fly.
ImageFolder maps the 1000 wnid folders to integer labels 0..999.
"""

import torch
import torchvision as tv
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader

from .data import binarize_spatial, edge_bits, CIFAR3_THRESH
from .paths import IMAGENET_TRAIN, IMAGENET_VAL


def imagenet_channels(thresholds=None, edges=False):
    """Number of binary input channels produced by the encoding (3 colours)."""
    if thresholds is None:
        thresholds = CIFAR3_THRESH
    ch = 3 * len(thresholds)
    if edges:
        # edge_bits: C colours x 3 detectors x 2 signs x len(EDGE_THRESH)
        from .data import EDGE_THRESH
        ch += 3 * 3 * 2 * len(EDGE_THRESH)
    return ch


class _SpatialBinarize:
    """Picklable transform: float CHW image in [0,1] -> uint8 binary CHW."""

    def __init__(self, thresholds, edges):
        self.thresholds = thresholds
        self.edges = edges

    def __call__(self, img):
        x = img.unsqueeze(0)                       # [1,3,H,W]
        b = binarize_spatial(x, self.thresholds)
        if self.edges:
            b = torch.cat([b, edge_bits(x)], dim=1)
        return b.squeeze(0)                        # uint8 [C,H,W]


def _build_transform(train, size, thresholds, edges):
    base = [T.ToImage(), T.ToDtype(torch.float32, scale=True)]
    if train:
        geo = [T.Resize(size + 4, antialias=True),
               T.RandomCrop(size),
               T.RandomHorizontalFlip(p=0.5)]
    else:
        geo = [T.Resize(size, antialias=True), T.CenterCrop(size)]
    return T.Compose(base + geo + [_SpatialBinarize(thresholds, edges)])


def get_imagenet_loaders(batch_size=128, size=32, thresholds=None, edges=False,
                         num_workers=8, train_subset=None, val_subset=None,
                         shuffle_train=True, pin_memory=True):
    """Return (train_loader, val_loader, channels).

    Each batch is (binary_uint8 [B, C, size, size], label[long]).

    train_subset / val_subset: optional int to use only the first N samples
    (deterministic, for fast sanity checks); the full loaders are unaffected
    when left None.
    """
    if thresholds is None:
        thresholds = CIFAR3_THRESH
    ch = imagenet_channels(thresholds, edges)

    tr_tf = _build_transform(True, size, thresholds, edges)
    va_tf = _build_transform(False, size, thresholds, edges)

    train_ds = tv.datasets.ImageFolder(IMAGENET_TRAIN, transform=tr_tf)
    val_ds = tv.datasets.ImageFolder(IMAGENET_VAL, transform=va_tf)

    if train_subset is not None:
        train_ds = torch.utils.data.Subset(
            train_ds, torch.arange(min(train_subset, len(train_ds))).tolist())
    if val_subset is not None:
        val_ds = torch.utils.data.Subset(
            val_ds, torch.arange(min(val_subset, len(val_ds))).tolist())

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=shuffle_train,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None)
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None)
    return train_loader, val_loader, ch
