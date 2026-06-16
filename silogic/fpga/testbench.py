"""Test-vector + self-checking Verilog testbench generation, and the one-call
:func:`export_logic_net` that writes a complete, simulatable project.

The flow mirrors how you would bring the design up on an FPGA: feed binary
feature vectors in, read the predicted class out, and check it against the
golden software model. The golden model here is :func:`silogic.fpga.circuit.simulate`,
which the tests assert is bit-exact with ``LogicNet.forward_hard``.

``export_logic_net(model, outdir)`` writes:

  * ``<name>.v``        — the synthesizable module (:func:`~silogic.fpga.verilog.to_verilog`).
  * ``<name>_tb.v``     — a self-checking testbench (``$readmemh`` vectors, prints
    ``PASS``/``FAIL`` and an error count, ``$finish``).
  * ``x_vectors.mem``   — ``N`` input vectors, one hex word per line.
  * ``expected.mem``    — the golden predicted class per vector.
  * ``run_sim.sh``      — an Icarus Verilog (``iverilog``/``vvp``) convenience runner.

Simulate with any Verilog tool, e.g. ``bash run_sim.sh`` (needs ``iverilog``).
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

from .circuit import LogicNetCircuit, extract_logic_net, simulate
from .verilog import to_verilog


def _hexwords(bits: np.ndarray) -> list[str]:
    """``[N, W]`` ``{0,1}`` -> list of N hex strings (bit 0 -> LSB)."""
    N, W = bits.shape
    ndig = (W + 3) // 4
    out = []
    for r in range(N):
        v = 0
        for j in range(W):
            if bits[r, j]:
                v |= 1 << j
        out.append(f"{v:0{ndig}x}")
    return out


def make_vectors(circuit: LogicNetCircuit, n: int = 256, seed: int = 0):
    """Random ``{0,1}`` input vectors and their golden predictions.

    Returns ``(x, scores, pred)`` with ``x`` ``[n, in_dim]`` uint8.
    """
    rng = np.random.default_rng(seed)
    x = (rng.random((n, circuit.in_dim)) > 0.5).astype(np.uint8)
    scores, pred = simulate(circuit, x)
    return x, scores, pred


def make_testbench(circuit: LogicNetCircuit, n_vectors: int,
                   module_name: Optional[str] = None,
                   pipeline: bool = False,
                   vec_file: str = "x_vectors.mem",
                   exp_file: str = "expected.mem") -> str:
    """A self-checking testbench that drives ``n_vectors`` and checks ``pred``."""
    name = module_name or circuit.name
    IN, PW = circuit.in_dim, circuit.pred_bits
    cls_w = circuit.num_classes * circuit.score_bits
    # 0-indexed iteration at which the first registered result appears: the
    # path has (1 input + depth layer + 1 score) register stages, so x[0] fed
    # before posedge #0 emerges after posedge #(depth+1).
    latency = (len(circuit.layers) + 1) if pipeline else 0

    T = []
    T.append("// Auto-generated self-checking testbench (silogic.fpga).")
    T.append("`timescale 1ns/1ps")
    T.append("`default_nettype none")
    T.append("module tb;")
    T.append(f"  localparam integer N  = {n_vectors};")
    T.append(f"  localparam integer IN = {IN};")
    T.append(f"  localparam integer PW = {PW};")
    T.append(f"  reg  [IN-1:0] x;")
    T.append(f"  wire [{cls_w - 1}:0] class_scores;")
    T.append(f"  wire [PW-1:0] pred;")
    T.append(f"  reg  [IN-1:0] xmem  [0:N-1];")
    T.append(f"  reg  [PW-1:0] emem  [0:N-1];")
    T.append("  integer i, errors;")
    if pipeline:
        T.append("  reg clk, rst_n;")
        T.append("  reg [PW-1:0] eq [0:1023];   // expected-pred delay line")
        T.append("  integer fed, got;")

    # DUT instance
    inst = [".x(x)", ".class_scores(class_scores)", ".pred(pred)"]
    if pipeline:
        inst = [".clk(clk)", ".rst_n(rst_n)"] + inst
    T.append(f"  {name} dut (")
    T.append(",\n".join("    " + s for s in inst))
    T.append("  );")
    T.append("")
    T.append("  initial begin")
    T.append(f'    $readmemh("{vec_file}", xmem);')
    T.append(f'    $readmemh("{exp_file}", emem);')
    T.append("    errors = 0;")

    if not pipeline:
        T.append("    for (i = 0; i < N; i = i + 1) begin")
        T.append("      x = xmem[i];")
        T.append("      #1;")  # settle combinational logic
        T.append("      if (pred !== emem[i]) begin")
        T.append("        errors = errors + 1;")
        T.append('        if (errors <= 20) $display("MISMATCH vec %0d: got %0d exp %0d",'
                 " i, pred, emem[i]);")
        T.append("      end")
        T.append("    end")
    else:
        T.append("    clk = 0; rst_n = 0; x = 0; fed = 0; got = 0;")
        T.append("    #3 rst_n = 1;")
        T.append("    // feed N vectors, account for the fixed pipeline latency")
        T.append(f"    for (i = 0; i < N + {latency}; i = i + 1) begin")
        T.append("      if (fed < N) begin x = xmem[fed]; eq[fed % 1024] = emem[fed]; fed = fed + 1; end")
        T.append("      @(posedge clk);")
        T.append("      #1;   // let the registered outputs settle (post-NBA)")
        T.append(f"      if (i >= {latency}) begin")
        T.append("        if (pred !== eq[got % 1024]) begin")
        T.append("          errors = errors + 1;")
        T.append('          if (errors <= 20) $display("MISMATCH vec %0d: got %0d exp %0d",'
                 " got, pred, eq[got % 1024]);")
        T.append("        end")
        T.append("        got = got + 1;")
        T.append("      end")
        T.append("    end")

    T.append("    if (errors == 0)")
    T.append('      $display("PASS: all %0d vectors match", N);')
    T.append("    else")
    T.append('      $display("FAIL: %0d / %0d mismatches", errors, N);')
    T.append("    $finish;")
    T.append("  end")
    if pipeline:
        T.append("  always #5 clk = ~clk;")
        T.append("  initial begin #100000000 $display(\"TIMEOUT\"); $finish; end")
    T.append("endmodule")
    T.append("`default_nettype wire")
    return "\n".join(T)


_RUN_SH = """#!/usr/bin/env bash
# Simulate the generated design with Icarus Verilog.
set -e
cd "$(dirname "$0")"
iverilog -g2012 -o {name}_sim {name}.v {name}_tb.v
vvp {name}_sim
"""


def export_logic_net(model, outdir: str, name: str = "logicnet",
                     n_vectors: int = 256, pipeline: bool = False,
                     seed: int = 0) -> LogicNetCircuit:
    """Export a trained :class:`~silogic.models.LogicNet` to a simulatable Verilog
    project under ``outdir``.

    Args:
        model: A ``LogicNet`` with a ``GroupSum`` head (any node family).
        outdir: Output directory (created if missing).
        name: Module / file base name.
        n_vectors: Number of random test vectors to generate.
        pipeline: Emit the registered (clocked) pipeline form. Default ``False``.
        seed: RNG seed for the test vectors.

    Returns:
        The :class:`LogicNetCircuit` IR (also re-usable for further backends).
    """
    os.makedirs(outdir, exist_ok=True)
    circuit = extract_logic_net(model, name=name)

    verilog = to_verilog(circuit, module_name=name, pipeline=pipeline)
    tb = make_testbench(circuit, n_vectors, module_name=name, pipeline=pipeline)
    x, scores, pred = make_vectors(circuit, n=n_vectors, seed=seed)

    def _write(fn, text):
        with open(os.path.join(outdir, fn), "w") as f:
            f.write(text if text.endswith("\n") else text + "\n")

    _write(f"{name}.v", verilog)
    _write(f"{name}_tb.v", tb)
    _write("x_vectors.mem", "\n".join(_hexwords(x)))
    _write("expected.mem", "\n".join(f"{int(p):x}" for p in pred))
    _write("run_sim.sh", _RUN_SH.format(name=name))
    os.chmod(os.path.join(outdir, "run_sim.sh"), 0o755)

    return circuit
