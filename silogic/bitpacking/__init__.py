"""Bitpacked inference for silogic models.

Converts any trained :class:`~silogic.models.LogicNet` /
:class:`~silogic.models.LogicConvNet` into a bitpacked representation that
packs **64 samples per int64 word** and evaluates whole batches with single
bitwise instructions.

**FC layers** — ``[dim, n_words]`` int64, ``n_words = ceil(B/64)``
    Gates are pre-sorted by type so all AND-gates, all XOR-gates, etc.
    fire one vectorised numpy call — maximum SIMD throughput, zero per-neuron
    Python branching.  A Triton GPU kernel is also provided.

**Conv layers** — B-packing ``[dim, L, nw_B]`` int64, ``nw_B = ceil(B/64)``
    The spatial dimension ``L = H*W`` remains explicit.  The raw image is
    packed once; packed-unfold, gate-tree, and OR-pool all stay in the packed
    domain without ever expanding ``B``.  The final ``[n_out, L, nw_B]``
    reshapes directly to ``[feat_dim, nw_B]`` for the FC head.

Typical speedups vs torch ``forward_hard`` (CPU):

* FC models (LogicNet):  5–35× at B=512, 50–100× at B=4096
* Conv models (LogicConvNet, all node types):  9–48× at B=64–1024

Quick start — FC model::

    from silogic.bitpacking import convert_logic_net, run_benchmark, print_report

    bp = convert_logic_net(trained_logic_net)
    logits = bp(x_uint8)               # [B, num_classes] float32

    results = run_benchmark(trained_logic_net, x_test, y_test)
    print_report(results)

Quick start — conv model::

    from silogic.bitpacking import convert_logic_conv_net

    bp = convert_logic_conv_net(trained_conv_net)
    logits = bp(x_uint8_BCHW)         # [B, num_classes] float32
"""
from .convert import convert_logic_net, convert_logic_conv_net, convert_layer, convert_head
from .packed_model import (
    BitpackedNet,
    BitpackedConvNet,
    BitpackedConvGPUNet,
    packed_unfold_gpu,
    packed_or_pool_gpu,
)
from .ops import pack_bits, unpack_bits, n_words
from .benchmark import run_benchmark, run_conv_benchmark, print_report, speedup_summary
from .kernels import HAS_TRITON

__all__ = [
    "convert_logic_net",
    "convert_logic_conv_net",
    "convert_layer",
    "convert_head",
    "BitpackedNet",
    "BitpackedConvNet",
    "BitpackedConvGPUNet",
    "packed_unfold_gpu",
    "packed_or_pool_gpu",
    "pack_bits",
    "unpack_bits",
    "n_words",
    "run_benchmark",
    "run_conv_benchmark",
    "print_report",
    "speedup_summary",
    "HAS_TRITON",
]
