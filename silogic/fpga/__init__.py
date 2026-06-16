"""FPGA / HDL export for trained logic networks.

A discretized :class:`~silogic.models.LogicNet` *is* a feed-forward network of
LUTs followed by ``GroupSum`` popcount adders — i.e. a combinational FPGA
design. This subpackage lowers a trained ``LogicNet`` (any node family:
``gate16``/``walsh``/``multilinear``/``hybrid``/``linear``/``polynomial``, with a
``GroupSum`` head) to synthesizable Verilog and a self-checking testbench.

Quickstart
----------
>>> import silogic
>>> from silogic.fpga import export_logic_net
>>> net = silogic.LogicNet(64, 80, depth=3, num_classes=8, connectome="topk")
>>> # ... train net ...
>>> circuit = export_logic_net(net, "build/mynet", name="mynet")
>>> print(circuit.summary())

That writes ``mynet.v`` (the module), ``mynet_tb.v`` (a self-checking
testbench), ``x_vectors.mem`` / ``expected.mem`` (random vectors + golden
predictions), and ``run_sim.sh`` (an Icarus Verilog runner) under ``build/mynet``.

Lower-level entry points:

  * :func:`extract_logic_net` — ``LogicNet`` -> backend-independent IR
    (:class:`LogicNetCircuit`).
  * :func:`to_verilog`        — IR -> Verilog string (``pipeline=`` for a clocked
    pipeline).
  * :func:`simulate`          — bit-exact numpy reference for the HDL.
  * :func:`make_testbench`    — IR -> self-checking testbench string.
"""
from .circuit import (LogicNetCircuit, Layer, LutNode,
                      extract_logic_net, simulate)
from .verilog import to_verilog
from .testbench import make_testbench, make_vectors, export_logic_net
from .estimate import (benchmark_fpga, print_fpga_report, estimate_resources,
                       ResourceEstimate, FpgaBenchmark, has_yosys)

__all__ = [
    "LogicNetCircuit", "Layer", "LutNode",
    "extract_logic_net", "simulate", "to_verilog",
    "make_testbench", "make_vectors", "export_logic_net",
    "benchmark_fpga", "print_fpga_report", "estimate_resources",
    "ResourceEstimate", "FpgaBenchmark", "has_yosys",
]
