"""Data loading, threshold binarization, and augmentation.

Faithful to arXiv:2511.12340 Section 3.3:
  * MNIST: single threshold 0.25 -> 784-dim. Augment x10
           (rotation +/-10, shear +/-10, scale +/-10%, elastic alpha=64 sigma=6).
  * FashionMNIST: 7 thresholds [0.125..0.875] -> 5488-dim. Augment x10.
  * CIFAR-10: per-channel thresholds (7 -> 21504, or 4 -> 12288).
              Augment x8 (random crop 32 pad 4, horizontal flip).
"""

import os
import torch
import torchvision as tv
import torchvision.transforms.v2 as T

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")

MNIST_THRESH = [0.25]
SEVEN_THRESH = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]
FOUR_THRESH = [0.2, 0.4, 0.6, 0.8]


def binarize(x, thresholds):
    """x: float tensor [N, C, H, W] in [0,1]. Returns uint8 [N, C*H*W*len(thr)].

    Thermometer encoding: one bit per threshold per pixel, concatenated.

    Args:
        x (torch.Tensor): Float image tensor ``[N, C, H, W]`` in ``[0, 1]``.
        thresholds (list[float]): Thermometer levels in ``[0, 1]``; one bit
            ``x >= t`` is emitted per threshold.

    Returns:
        torch.Tensor: uint8 tensor ``[N, C*H*W*len(thresholds)]`` (flattened,
        per-threshold bit planes concatenated).
    """
    n = x.shape[0]
    flat = x.reshape(n, -1)
    bits = [(flat >= t).to(torch.uint8) for t in thresholds]
    return torch.cat(bits, dim=1)


def _raw(dataset, train):
    if dataset == "mnist":
        ds = tv.datasets.MNIST(DATA_ROOT, train=train, download=False)
        imgs = ds.data.unsqueeze(1).float() / 255.0  # [N,1,28,28]
        labels = ds.targets.clone()
    elif dataset == "fmnist":
        ds = tv.datasets.FashionMNIST(DATA_ROOT, train=train, download=False)
        imgs = ds.data.unsqueeze(1).float() / 255.0
        labels = ds.targets.clone()
    elif dataset == "cifar10":
        ds = tv.datasets.CIFAR10(DATA_ROOT, train=train, download=False)
        imgs = torch.from_numpy(ds.data).permute(0, 3, 1, 2).float() / 255.0
        labels = torch.tensor(ds.targets)
    else:
        raise ValueError(dataset)
    return imgs, labels


def _augment_batches(imgs, transform, device, chunk=4096):
    out = []
    for i in range(0, imgs.shape[0], chunk):
        c = imgs[i:i + chunk].to(device)
        c = transform(c)
        out.append(c.cpu())
    return torch.cat(out, dim=0)


CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "cache")
CIFAR3_THRESH = [0.25, 0.5, 0.75]  # 2-bit precision as in LogicTreeNet S/M
FIVE_THRESH = [1/6, 2/6, 3/6, 4/6, 5/6]  # 5-bit precision as in LogicTreeNet B/L/G


def binarize_spatial(x, thresholds):
    """x: [N,C,H,W] in [0,1] -> [N, C*len(thr), H, W] uint8 (thermometer).

    Args:
        x (torch.Tensor): Float image tensor ``[N, C, H, W]`` in ``[0, 1]``.
        thresholds (list[float]): Thermometer levels in ``[0, 1]``; one bit
            plane ``x >= t`` per threshold.

    Returns:
        torch.Tensor: uint8 tensor ``[N, C*len(thresholds), H, W]`` with the
        spatial dimensions preserved (per-threshold planes concatenated on the
        channel axis).
    """
    bits = [(x >= t).to(torch.uint8) for t in thresholds]
    return torch.cat(bits, dim=1)


import torch.nn.functional as _F

# Fixed edge/curvature detector kernels (approximating the LogicTreeNet B/L/G
# input preprocessing: edge + curvature detectors with thresholds).
_SOBEL_X = torch.tensor([[-1., 0, 1], [-2, 0, 2], [-1, 0, 1]])
_SOBEL_Y = _SOBEL_X.t().contiguous()
_LAPLACE = torch.tensor([[0., 1, 0], [1, -4, 1], [0, 1, 0]])
EDGE_THRESH = [0.12, 0.25]  # signed -> 2 bits each (>t, <-t)


def edge_bits(x, edge_thr=None):
    """x: [N,3,H,W] in [0,1] -> uint8 edge-detector bits, same H,W.

    For each colour channel, apply Sobel-x, Sobel-y, Laplacian; threshold each
    signed response at +/- each level -> 2 bits. 3 colours x 3 detectors x
    2 x len(thr) channels.

    Args:
        x (torch.Tensor): Float image tensor ``[N, C, H, W]`` in ``[0, 1]``
            (``C`` colour channels, typically ``3``).
        edge_thr (list[float], optional): Magnitude levels for the signed
            responses; each emits two bits (``> t`` and ``< -t``). Defaults to
            ``EDGE_THRESH`` (``[0.12, 0.25]``) when ``None``.

    Returns:
        torch.Tensor: uint8 tensor ``[N, C*3*2*len(edge_thr), H, W]`` of
        edge/curvature detector bits at the original spatial resolution.
    """
    if edge_thr is None:
        edge_thr = EDGE_THRESH
    C = x.shape[1]
    ker = torch.stack([_SOBEL_X, _SOBEL_Y, _LAPLACE]).to(x.dtype).to(x.device)  # [3,3,3]
    # grouped conv: each colour -> 3 detector maps
    w = ker.unsqueeze(1).repeat(C, 1, 1, 1)          # [C*3,1,3,3]
    resp = _F.conv2d(x, w, padding=1, groups=C)       # [N, C*3, H, W]
    bits = []
    for t in edge_thr:
        bits.append((resp > t).to(torch.uint8))
        bits.append((resp < -t).to(torch.uint8))
    return torch.cat(bits, dim=1)


def get_cifar_spatial(thresholds=None, n_aug=8, device="cuda", seed=0,
                      edges=False):
    """CIFAR-10 as spatial binary channels for convolutional logic nets.

    Returns (Xtr [N,Ck,32,32] uint8, ytr, Xte, yte, channels). Results are
    cached to disk keyed by all arguments.

    Args:
        thresholds (list[float], optional): Thermometer levels in ``[0, 1]``.
            Defaults to ``CIFAR3_THRESH`` (``[0.25, 0.5, 0.75]``) when ``None``.
        n_aug (int): Number of augmented train copies (random crop + horizontal
            flip); ``max(1, n_aug)`` copies are generated. Default ``8``.
        device (str): Device used to run augmentation. Default ``"cuda"``.
        seed (int): RNG seed for augmentation; part of the cache key.
            Default ``0``.
        edges (bool): If ``True`` append Sobel/Laplacian edge channels via
            :func:`edge_bits`. Default ``False``.

    Returns:
        tuple: ``(Xtr, ytr, Xte, yte, channels)`` where ``Xtr`` is uint8
        ``[N, channels, 32, 32]``, ``ytr``/``yte`` are integer labels, ``Xte``
        is the encoded test set and ``channels`` (int) is the channel count.
    """
    if thresholds is None:
        thresholds = CIFAR3_THRESH
    os.makedirs(CACHE_DIR, exist_ok=True)
    tag = f"cifar_spatial_thr{len(thresholds)}_n{n_aug}_s{seed}_e{int(edges)}"
    path = os.path.join(CACHE_DIR, tag + ".pt")
    if os.path.exists(path):
        d = torch.load(path)
        return d["Xtr"], d["ytr"], d["Xte"], d["yte"], d["ch"]
    torch.manual_seed(seed)
    imgs, labels = _raw("cifar10", train=True)
    te_imgs, te_labels = _raw("cifar10", train=False)
    transform = T.Compose([
        T.RandomCrop(32, padding=4, padding_mode="reflect"),
        T.RandomHorizontalFlip(p=0.5),
    ])

    def encode(im):
        b = binarize_spatial(im, thresholds)
        if edges:
            b = torch.cat([b, edge_bits(im)], dim=1)
        return b

    copies, ys = [], []
    for _ in range(max(1, n_aug)):
        aug = _augment_batches(imgs, transform, device).clamp(0, 1)
        copies.append(encode(aug))
        ys.append(labels)
    Xtr = torch.cat(copies, 0)
    ytr = torch.cat(ys, 0)
    Xte = encode(te_imgs)
    ch = Xtr.shape[1]
    torch.save({"Xtr": Xtr, "ytr": ytr, "Xte": Xte, "yte": te_labels,
                "ch": ch}, path)
    return Xtr, ytr, Xte, te_labels, ch


def get_fmnist_spatial(thresholds=None, n_aug=2, device="cuda", seed=0,
                       edges=True):
    """FashionMNIST as spatial binary channels for convolutional logic nets.

    Thermometer-encodes each 28x28 image (default 5 levels) and, with
    ``edges=True``, appends Sobel/Laplacian edge-detector channels — the input
    richness that lifts a ``LogicTreeNet`` to ~87.5% hard accuracy (vs ~85% for
    the flat FC ``LogicNet``). Light affine-only augmentation (no elastic, which
    distorts garments and caps accuracy). Returns
    (Xtr [N,Ck,28,28] uint8, ytr, Xte, yte, channels). Results are cached to
    disk keyed by all arguments.

    Args:
        thresholds (list[float], optional): Thermometer levels in ``[0, 1]``.
            Defaults to ``FIVE_THRESH`` (5 levels) when ``None``.
        n_aug (int): Number of augmented train copies (random affine);
            ``max(1, n_aug)`` copies are generated. Default ``2``.
        device (str): Device used to run augmentation. Default ``"cuda"``.
        seed (int): RNG seed for augmentation; part of the cache key.
            Default ``0``.
        edges (bool): If ``True`` append Sobel/Laplacian edge channels via
            :func:`edge_bits`. Default ``True``.

    Returns:
        tuple: ``(Xtr, ytr, Xte, yte, channels)`` where ``Xtr`` is uint8
        ``[N, channels, 28, 28]``, ``ytr``/``yte`` are integer labels, ``Xte``
        is the encoded test set and ``channels`` (int) is the channel count.
    """
    if thresholds is None:
        thresholds = FIVE_THRESH
    os.makedirs(CACHE_DIR, exist_ok=True)
    tag = f"fmnist_spatial_thr{len(thresholds)}_n{n_aug}_s{seed}_e{int(edges)}"
    path = os.path.join(CACHE_DIR, tag + ".pt")
    if os.path.exists(path):
        d = torch.load(path)
        return d["Xtr"], d["ytr"], d["Xte"], d["yte"], d["ch"]
    torch.manual_seed(seed)
    imgs, labels = _raw("fmnist", train=True)
    te_imgs, te_labels = _raw("fmnist", train=False)
    transform = T.RandomAffine(degrees=8, translate=(0.07, 0.07), scale=(0.9, 1.1))

    def encode(im):
        b = binarize_spatial(im, thresholds)
        if edges:
            b = torch.cat([b, edge_bits(im)], dim=1)
        return b

    copies, ys = [], []
    for _ in range(max(1, n_aug)):
        aug = _augment_batches(imgs, transform, device).clamp(0, 1)
        copies.append(encode(aug)); ys.append(labels)
    Xtr = torch.cat(copies, 0); ytr = torch.cat(ys, 0)
    Xte = encode(te_imgs); ch = Xtr.shape[1]
    torch.save({"Xtr": Xtr, "ytr": ytr, "Xte": Xte, "yte": te_labels,
                "ch": ch}, path)
    return Xtr, ytr, Xte, te_labels, ch


def get_dataset_cached(dataset, thresholds=None, augment=True, n_aug=None,
                       device="cuda", seed=0):
    """Cache the (expensive) augmented binarized dataset to disk.

    Thin disk-caching wrapper around :func:`get_dataset`; on a cache miss it
    computes and stores the result keyed by all arguments.

    Args:
        dataset (str): One of ``"mnist"``, ``"fmnist"``, ``"cifar10"``.
        thresholds (list[float], optional): Thermometer levels in ``[0, 1]``;
            dataset-specific defaults are used when ``None``. Default ``None``.
        augment (bool): If ``True`` generate augmented copies. Default ``True``.
        n_aug (int, optional): Number of augmented copies; a dataset-specific
            default is used when ``None``. Default ``None``.
        device (str): Device used to run augmentation. Default ``"cuda"``.
        seed (int): RNG seed for augmentation; part of the cache key.
            Default ``0``.

    Returns:
        tuple: ``(Xtr, ytr, Xte, yte, in_dim)`` as returned by
        :func:`get_dataset`.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    thr = thresholds
    tag = f"{dataset}_thr{len(thr) if thr else 'def'}_aug{augment}_n{n_aug}_s{seed}"
    path = os.path.join(CACHE_DIR, tag + ".pt")
    if os.path.exists(path):
        d = torch.load(path)
        return d["Xtr"], d["ytr"], d["Xte"], d["yte"], d["in_dim"]
    torch.manual_seed(seed)
    out = get_dataset(dataset, thresholds, augment, n_aug, device)
    torch.save({"Xtr": out[0], "ytr": out[1], "Xte": out[2],
                "yte": out[3], "in_dim": out[4]}, path)
    return out


def get_dataset(dataset, thresholds=None, augment=True, n_aug=None,
                device="cuda"):
    """Return (Xtrain_uint8, ytrain, Xtest_uint8, ytest, in_dim).

    Augmented training copies are pre-generated once and binarized (flat
    thermometer encoding for the FC ``LogicNet``).

    Args:
        dataset (str): One of ``"mnist"``, ``"fmnist"``, ``"cifar10"``.
        thresholds (list[float], optional): Thermometer levels in ``[0, 1]``.
            Defaults per dataset when ``None`` (``MNIST_THRESH`` for
            ``"mnist"``, ``SEVEN_THRESH`` for ``"fmnist"``/``"cifar10"``).
        augment (bool): If ``True`` (and ``n_aug > 1``) generate augmented
            train copies; otherwise binarize the raw train set. Default
            ``True``.
        n_aug (int, optional): Number of augmented copies; defaults per dataset
            when ``None`` (``10`` for MNIST/FMNIST, ``8`` for CIFAR-10).
            Default ``None``.
        device (str): Device used to run augmentation. Default ``"cuda"``.

    Returns:
        tuple: ``(Xtr, ytr, Xte, yte, in_dim)`` where ``Xtr``/``Xte`` are uint8
        feature tensors ``[N, in_dim]``, ``ytr``/``yte`` are integer labels and
        ``in_dim`` (int) is the flattened input dimension.
    """
    if thresholds is None:
        thresholds = {"mnist": MNIST_THRESH, "fmnist": SEVEN_THRESH,
                      "cifar10": SEVEN_THRESH}[dataset]
    if n_aug is None:
        n_aug = {"mnist": 10, "fmnist": 10, "cifar10": 8}[dataset]

    imgs, labels = _raw(dataset, train=True)
    test_imgs, test_labels = _raw(dataset, train=False)

    if dataset in ("mnist", "fmnist"):
        transform = T.Compose([
            T.RandomAffine(degrees=10, shear=10, scale=(0.9, 1.1)),
            T.ElasticTransform(alpha=64.0, sigma=6.0),
        ])
    else:  # cifar10
        transform = T.Compose([
            T.RandomCrop(32, padding=4, padding_mode="reflect"),
            T.RandomHorizontalFlip(p=0.5),
        ])

    if augment and n_aug > 1:
        copies, ys = [], []
        for _ in range(n_aug):
            aug = _augment_batches(imgs, transform, device)
            copies.append(binarize(aug.clamp(0, 1), thresholds))
            ys.append(labels)
        Xtr = torch.cat(copies, dim=0)
        ytr = torch.cat(ys, dim=0)
    else:
        Xtr = binarize(imgs, thresholds)
        ytr = labels

    Xte = binarize(test_imgs, thresholds)
    in_dim = Xtr.shape[1]
    return Xtr, ytr, Xte, test_labels, in_dim
