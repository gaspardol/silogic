"""Correctness tests for silogic.fpga (Verilog / FPGA export).

Two layers of checking:

  * The pure-numpy golden simulator (:func:`silogic.fpga.simulate`) must agree
    bit-exactly with ``LogicNet.forward_hard`` for every node family — this is
    the contract the generated HDL relies on, and needs no external tools.
  * If Icarus Verilog (``iverilog``/``vvp``) is on ``PATH``, the generated
    Verilog (combinational *and* pipelined) is actually simulated against
    golden test vectors and must report ``PASS``. These tests skip cleanly when
    no simulator is installed.
"""
import os
import shutil
import subprocess

import numpy as np
import pytest
import torch

from silogic.models import LogicNet
from silogic.fpga import (extract_logic_net, simulate, to_verilog,
                          export_logic_net, benchmark_fpga, estimate_resources,
                          has_yosys)
from silogic.fpga.estimate import fmax_mhz, batch_time_s

HAS_IVERILOG = shutil.which("iverilog") is not None and shutil.which("vvp") is not None

# (node, arity, connectome) combinations spanning every node family + wiring.
CONFIGS = [
    ("gate16", 2, "topk"),
    ("gate16", 2, "fixed"),
    ("gate16", 2, "dense"),
    ("walsh", 2, "topk"),
    ("walsh", 6, "topk"),
    ("multilinear", 3, "topk"),
    ("hybrid", 4, "topk"),
    ("linear", 5, "topk"),
    ("polynomial", 3, "topk"),
]


def _make(node, arity, connectome, seed=1, **kw):
    torch.manual_seed(seed)
    net = LogicNet(36, 32, depth=3, num_classes=4, node=node, arity=arity,
                   connectome=connectome, seed=seed, **kw)
    net.eval()
    return net


# ── golden simulator vs forward_hard (no external tools) ──────────────────────

@pytest.mark.parametrize("node,arity,connectome", CONFIGS)
def test_simulate_matches_forward_hard(node, arity, connectome):
    net = _make(node, arity, connectome)
    rng = np.random.default_rng(0)
    x = (rng.random((64, 36)) > 0.5).astype(np.uint8)
    with torch.no_grad():
        ref = net.forward_hard(torch.from_numpy(x).float()).numpy()   # block sums
    circuit = extract_logic_net(net)
    scores, pred = simulate(circuit, x)
    assert np.array_equal(scores, ref), f"{node}: score mismatch"
    assert np.array_equal(pred, ref.argmax(1)), f"{node}: pred mismatch"


def test_wire_residual_simulate():
    net = _make("gate16", 2, "topk", wire_residual=0.25)
    rng = np.random.default_rng(3)
    x = (rng.random((32, 36)) > 0.5).astype(np.uint8)
    with torch.no_grad():
        ref = net.forward_hard(torch.from_numpy(x).float()).numpy()
    scores, pred = simulate(extract_logic_net(net), x)
    assert np.array_equal(pred, ref.argmax(1))


def test_groupsum_only_head():
    net = _make("gate16", 2, "topk")
    net.head = __import__("silogic").build_decoder("linear", 32, 4)  # non-GroupSum
    with pytest.raises(TypeError):
        extract_logic_net(net)


def test_sumthreshold_unsupported():
    net = _make("gate16", 2, "st")  # SumThreshold connectome
    with pytest.raises(NotImplementedError):
        extract_logic_net(net)


def test_verilog_renders():
    net = _make("multilinear", 4, "topk")
    circuit = extract_logic_net(net)
    v = to_verilog(circuit, pipeline=False)
    vp = to_verilog(circuit, pipeline=True)
    assert "module logicnet" in v and "endmodule" in v
    assert "posedge clk" in vp and "posedge clk" not in v


# ── end-to-end Verilog simulation (needs iverilog) ────────────────────────────

def _run_iverilog(outdir, name):
    subprocess.run(["iverilog", "-g2012", "-o", "sim", f"{name}.v", f"{name}_tb.v"],
                   cwd=outdir, check=True, capture_output=True)
    out = subprocess.run(["vvp", "sim"], cwd=outdir, check=True,
                         capture_output=True, text=True).stdout
    return out


@pytest.mark.skipif(not HAS_IVERILOG, reason="iverilog/vvp not installed")
@pytest.mark.parametrize("pipeline", [False, True])
@pytest.mark.parametrize("node,arity,connectome", CONFIGS)
def test_iverilog_simulation(tmp_path, node, arity, connectome, pipeline):
    net = _make(node, arity, connectome)
    outdir = str(tmp_path / "dut")
    export_logic_net(net, outdir, name="dut", n_vectors=128, pipeline=pipeline)
    out = _run_iverilog(outdir, "dut")
    assert "PASS" in out, out


# ── resource / speed estimation ───────────────────────────────────────────────

def test_estimate_analytic_no_yosys():
    """Analytic fallback works with no external tools and is self-consistent."""
    net = _make("gate16", 2, "topk")
    comb = estimate_resources(net, pipeline=False, use_yosys=False)
    pipe = estimate_resources(net, pipeline=True, use_yosys=False)
    assert comb.source == "analytic" and pipe.source == "analytic"
    assert comb.luts > 0 and comb.logic_levels > 0
    assert pipe.ffs > 0 and pipe.pipeline_latency_cycles == len(net.layers) + 2
    # pipelined per-stage path is no deeper than the whole combinational path
    assert pipe.logic_levels <= comb.logic_levels


def test_fmax_and_batch_time_math():
    # deeper logic -> lower Fmax; pipelined amortizes latency over the batch
    assert fmax_mhz(10, 0.45, False) > fmax_mhz(30, 0.45, False)
    lat, t64 = batch_time_s("pipelined", 64, 8, 0.45, 6)
    _, t256 = batch_time_s("pipelined", 256, 8, 0.45, 6)
    assert t256 > t64 > 0
    # combinational: time scales linearly with batch (1 sample/clock, no fill)
    _, c64 = batch_time_s("combinational", 64, 30, 0.45, 0)
    _, c128 = batch_time_s("combinational", 128, 30, 0.45, 0)
    assert abs(c128 - 2 * c64) < 1e-15


def test_benchmark_analytic_struct():
    net = _make("multilinear", 4, "topk")
    bench = benchmark_fpga(net, batch_sizes=(64, 256), use_yosys=False)
    assert bench.combinational.luts > 0
    assert bench.batch_sizes == (64, 256)


@pytest.mark.skipif(not has_yosys(), reason="yosys not installed")
def test_benchmark_yosys_real_luts():
    net = _make("gate16", 2, "topk")
    bench = benchmark_fpga(net, use_yosys=True)
    assert bench.combinational.source == "yosys"
    assert bench.combinational.luts > 0 and bench.combinational.logic_levels > 0
    # the gate fabric is far shallower than the popcount/argmax head
    assert 0 < bench.fabric_levels < bench.combinational.logic_levels
