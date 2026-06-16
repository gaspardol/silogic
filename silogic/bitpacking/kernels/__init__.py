"""Triton GPU kernels for bitpacked inference (optional)."""
try:
    import triton  # noqa: F401
    from .triton_layer import BitpackedGPULayer
    from .triton_conv_layer import BitpackedConvGPULayer
    HAS_TRITON = True
except (ImportError, Exception):
    HAS_TRITON = False
    BitpackedGPULayer = None      # type: ignore[assignment,misc]
    BitpackedConvGPULayer = None  # type: ignore[assignment,misc]

__all__ = ["HAS_TRITON", "BitpackedGPULayer", "BitpackedConvGPULayer"]
