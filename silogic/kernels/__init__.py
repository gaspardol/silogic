"""Fused Triton kernels for the logic layers (optional, CUDA-only).

Each kernel is guarded: if Triton (or a CUDA build) is unavailable the import
fails softly and the corresponding ``HAS_*`` flag is ``False``, so the pure
PyTorch reference paths in the layers are used instead. This keeps CPU-only
``import silogic`` working without Triton installed.

Public kernels:
  * :func:`dense_logic` — fused Top-K dense logic head (``dense.py``).
  * :func:`tree_conv` / :func:`tree_conv_af` — fused fixed-connectome gate-tree
    convolution (``conv.py``).
  * :func:`tree_conv_topk` — fused learnable Top-K gate-tree convolution
    (``conv_topk.py``).
  * :func:`warp_logic` — fused Walsh (WARP) logic layer (``warp.py``).
  * :func:`multilinear_logic` — fused n-input multilinear LUT layer with the
    hybrid (DWN) straight-through estimator (``multilinear.py``).
  * :func:`conv_hybrid` — fused convolutional hybrid LUT-tree of any depth,
    reading the image directly (no unfold) (``conv_hybrid.py``).
"""

try:
    from .dense import dense_logic
    HAS_DENSE = True
except Exception:
    HAS_DENSE = False

try:
    from .conv import tree_conv, tree_conv_af, build_inverse_map
    HAS_CONV = True
except Exception:
    HAS_CONV = False

try:
    from .conv_topk import tree_conv_topk
    HAS_CONV_TOPK = True
except Exception:
    HAS_CONV_TOPK = False

try:
    from .warp import warp_logic
    HAS_WARP = True
except Exception:
    HAS_WARP = False

try:
    from .multilinear import multilinear_logic
    HAS_MULTILINEAR = True
except Exception:
    HAS_MULTILINEAR = False

try:
    from .conv_hybrid import conv_hybrid
    HAS_CONV_HYBRID = True
except Exception:
    HAS_CONV_HYBRID = False
