"""Resource + inference-speed estimation for exported logic networks.

The bitpacking benchmark times real CPU/GPU runs; an FPGA design has no host to
time, so "speed" here means **throughput and latency derived from the synthesized
circuit**. The honest pipeline is:

  1. Synthesize the generated Verilog to 6-input LUTs with **yosys** (open source,
     ``abc -lut 6``) to get the *real* LUT6 count and the *real* logic-level depth
     of the critical path (``ltp`` = longest topological path, in LUT levels).
  2. Turn logic depth into a clock period with a per-LUT-level delay
     (``ns_per_level``: LUT + local routing; a few representative device/speed-grade
     values are provided). This is the one *estimated* step — a true Fmax needs
     place-and-route (Vivado/Quartus) — so timing is reported as a small band, not
     a single number.
  3. Combine with the two execution modes the exporter emits:
       * **combinational** — one result per clock at ``Fmax = 1/(depth_comb·t)``;
         the whole net (all layers + GroupSum + argmax) is the critical path.
       * **pipelined** — one result per clock at ``Fmax = 1/(depth_stage·t)`` where
         ``depth_stage`` is the deepest *register-to-register* path; latency is
         ``depth+2`` cycles.

If yosys is not on ``PATH`` the LUT count falls back to the IR node count and the
depth to a coarse analytic model (clearly flagged ``source="analytic"``).

>>> from silogic.fpga import benchmark_fpga, print_fpga_report
>>> res = benchmark_fpga(trained_logic_net)     # synthesizes both modes
>>> print_fpga_report(res)
"""
from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

from .circuit import LogicNetCircuit, extract_logic_net
from .verilog import to_verilog, _node_expr


# Representative LUT6 + local-routing delay per logic level (ns). These are
# order-of-magnitude figures for modern FPGAs at mid speed grade; a true number
# needs place-and-route. Used to bracket Fmax, never as an exact value.
NS_PER_LEVEL = {
    "conservative": 0.60,   # congested / slow grade / older part
    "typical": 0.45,        # e.g. UltraScale+ -2, moderate routing
    "aggressive": 0.30,     # fast grade, light routing
}
# Fixed register overhead (clk->Q + setup) added to a registered period, ns.
FF_OVERHEAD_NS = 0.5
# Assumed depth of one stage of a registered popcount tree (projection only).
_HEAD_STAGE_LEVELS = 6


def has_yosys() -> bool:
    return shutil.which("yosys") is not None


# ── synthesis (yosys) ─────────────────────────────────────────────────────────

_YS_COMB = """read_verilog -sv {v}
hierarchy -top {top}
proc; flatten; opt
techmap; opt
abc -lut 6
opt
stat
ltp -noff
"""

_YS_PIPE = """read_verilog -sv {v}
hierarchy -top {top}
proc; flatten; opt
techmap; opt
dffunmap
abc -lut 6
opt
stat
ltp -noff
"""


def _fabric_verilog(circuit: LogicNetCircuit) -> str:
    """Verilog for the gate fabric ONLY (all logic layers, no GroupSum head).

    Synthesizing this isolates the logic-fabric critical path from the integer
    popcount/argmax head, which on these nets dominates the whole-net path.
    """
    L = [f"module {circuit.name}_fabric (input [{circuit.in_dim - 1}:0] x, "
         f"output [{circuit.out_width - 1}:0] y);"]
    src = "x"
    for li, layer in enumerate(circuit.layers):
        L.append(f"  wire [{layer.width - 1}:0] l{li};")
        for o, nd in enumerate(layer.nodes):
            L.append(f"  assign l{li}[{o}] = {_node_expr(src, nd)};")
        src = f"l{li}"
    L.append(f"  assign y = {src};")
    L.append("endmodule")
    return "\n".join(L)


def _run_yosys(verilog: str, top: str, pipeline: bool,
               timeout: float = 600.0) -> Optional[Tuple[int, int, int]]:
    """Synthesize ``verilog`` to LUT6 and return ``(luts, ffs, logic_levels)``.

    ``logic_levels`` is the longest topological path in LUT levels (the
    register-to-register critical path for the pipelined form, the whole-net
    path for the combinational form). Returns ``None`` on any failure.
    """
    if not has_yosys():
        return None
    with tempfile.TemporaryDirectory() as d:
        vpath = os.path.join(d, f"{top}.v")
        with open(vpath, "w") as f:
            f.write(verilog)
        script = (_YS_PIPE if pipeline else _YS_COMB).format(v=vpath, top=top)
        spath = os.path.join(d, "synth.ys")
        with open(spath, "w") as f:
            f.write(script)
        try:
            out = subprocess.run([shutil.which("yosys"), "-s", spath],
                                 capture_output=True, text=True,
                                 timeout=timeout).stdout
        except (subprocess.TimeoutExpired, OSError):
            return None
    luts = ffs = 0
    for m in re.finditer(r"^\s*(\d+)\s+\$lut\s*$", out, re.M):
        luts = int(m.group(1))                       # last stat wins (post-opt)
    for m in re.finditer(r"^\s*(\d+)\s+\$_DFF", out, re.M):
        ffs = int(m.group(1))
    lvl = 0
    m = re.search(r"Longest topological path .*?\(length=(\d+)\)", out)
    if m:
        lvl = int(m.group(1))
    if luts == 0 and lvl == 0:
        return None
    return luts, ffs, lvl


# ── analytic fallback (no yosys) ──────────────────────────────────────────────

def _levels_node(arity: int) -> int:
    """LUT6 levels to realize one ``arity``-input function (Shannon tree)."""
    return 1 if arity <= 6 else math.ceil((arity - 6) / 5) + 1


def _analytic_depth(circuit: LogicNetCircuit, pipeline: bool) -> int:
    """Coarse critical-path depth (LUT levels) when yosys is unavailable."""
    g = circuit.group_size
    # GroupSum popcount of g bits: balanced adder tree, ~1.5 LUT levels per
    # halving once carry chains are mapped (calibrated against yosys).
    popcount = max(1, math.ceil(1.6 * math.log2(max(2, g))))
    argmax = max(1, math.ceil(math.log2(max(2, circuit.num_classes)))) * 2
    if pipeline:
        # deepest single stage: usually the popcount/argmax head stage
        layer = max(_levels_node(L.nodes[0].arity) for L in circuit.layers)
        return max(layer, popcount + argmax)
    layers = sum(_levels_node(L.nodes[0].arity) for L in circuit.layers)
    return layers + popcount + argmax


# ── estimate dataclasses ──────────────────────────────────────────────────────

@dataclass
class ResourceEstimate:
    mode: str                 # "combinational" | "pipelined"
    luts: int                 # LUT6 cells
    ffs: int                  # registers
    logic_levels: int         # critical path in LUT6 levels
    source: str               # "yosys" | "analytic"
    pipeline_latency_cycles: int = 0   # cycles from input to valid output


@dataclass
class FpgaBenchmark:
    circuit: LogicNetCircuit
    combinational: ResourceEstimate
    pipelined: ResourceEstimate
    fabric_levels: int = 0     # gate-fabric-only critical path (no head), 0 if unknown
    ns_per_level: Dict[str, float] = field(default_factory=lambda: dict(NS_PER_LEVEL))
    batch_sizes: Tuple[int, ...] = (64, 256, 1024, 4096)


def estimate_resources(model_or_circuit, pipeline: bool,
                       use_yosys: bool = True,
                       yosys_timeout: float = 600.0) -> ResourceEstimate:
    """Estimate LUTs / FFs / critical-path depth for one execution mode."""
    circuit = (model_or_circuit if isinstance(model_or_circuit, LogicNetCircuit)
               else extract_logic_net(model_or_circuit))
    mode = "pipelined" if pipeline else "combinational"
    lat = (len(circuit.layers) + 2) if pipeline else 0
    res = None
    if use_yosys and has_yosys():
        verilog = to_verilog(circuit, module_name=circuit.name, pipeline=pipeline)
        res = _run_yosys(verilog, circuit.name, pipeline, timeout=yosys_timeout)
    if res is not None:
        luts, ffs, lvl = res
        return ResourceEstimate(mode, luts, ffs, lvl, "yosys", lat)
    # analytic fallback
    luts = circuit.num_luts()
    lvl = _analytic_depth(circuit, pipeline)
    ffs = (circuit.in_dim + sum(L.width for L in circuit.layers)
           + circuit.num_classes * circuit.score_bits) if pipeline else 0
    return ResourceEstimate(mode, luts, ffs, lvl, "analytic", lat)


def fmax_mhz(levels: int, ns_per_level: float, registered: bool) -> float:
    """Clock ceiling (MHz) for a path of ``levels`` LUT levels."""
    period = levels * ns_per_level + (FF_OVERHEAD_NS if registered else 0.0)
    return 1e3 / period if period > 0 else float("inf")


def batch_time_s(mode: str, B: int, levels: int, ns_per_level: float,
                 latency_cycles: int) -> Tuple[float, float]:
    """Return ``(latency_s, batch_time_s)`` for a batch of ``B`` inputs.

    Both modes accept one new input per clock; the pipelined form pays a fixed
    ``latency_cycles`` fill before the first result. The combinational form is
    clocked through input/output registers (one sample/clock) at the slower,
    whole-net Fmax.
    """
    registered = mode == "pipelined"
    f = fmax_mhz(levels, ns_per_level, registered) * 1e6
    period = 1.0 / f
    if mode == "pipelined":
        latency = latency_cycles * period
        total = (B + latency_cycles) * period
    else:
        latency = period                      # 1 clock
        total = B * period
    return latency, total


def benchmark_fpga(model_or_circuit,
                   batch_sizes: Tuple[int, ...] = (64, 256, 1024, 4096),
                   use_yosys: bool = True,
                   yosys_timeout: float = 600.0) -> FpgaBenchmark:
    """Synthesize both execution modes and assemble a speed/resource benchmark.

    Args:
        model_or_circuit: A trained ``LogicNet`` or a :class:`LogicNetCircuit`.
        batch_sizes: Batch sizes to tabulate in :func:`print_fpga_report`.
        use_yosys: Use yosys for real LUT/depth numbers when available.
        yosys_timeout: Per-synthesis timeout (s).
    """
    circuit = (model_or_circuit if isinstance(model_or_circuit, LogicNetCircuit)
               else extract_logic_net(model_or_circuit))
    comb = estimate_resources(circuit, pipeline=False, use_yosys=use_yosys,
                              yosys_timeout=yosys_timeout)
    pipe = estimate_resources(circuit, pipeline=True, use_yosys=use_yosys,
                              yosys_timeout=yosys_timeout)
    fabric = 0
    if use_yosys and has_yosys():
        r = _run_yosys(_fabric_verilog(circuit), f"{circuit.name}_fabric",
                       pipeline=False, timeout=yosys_timeout)
        if r is not None:
            fabric = r[2]
    return FpgaBenchmark(circuit, comb, pipe, fabric, dict(NS_PER_LEVEL), batch_sizes)


# ── reporting ─────────────────────────────────────────────────────────────────

def _fmt_time(s: float) -> str:
    if s < 1e-6:
        return f"{s * 1e9:6.1f} ns"
    if s < 1e-3:
        return f"{s * 1e6:6.2f} us"
    if s < 1.0:
        return f"{s * 1e3:6.2f} ms"
    return f"{s:6.3f} s"


def print_fpga_report(bench: FpgaBenchmark) -> None:
    """Pretty-print the resource + throughput/latency benchmark."""
    c = bench.circuit
    print("=" * 74)
    print(f"FPGA estimate — {c.summary()}")
    src = bench.combinational.source
    print(f"synthesis source: {src}" + ("  (yosys abc -lut 6)" if src == "yosys"
          else "  (yosys not found — coarse analytic model)"))
    print("=" * 74)

    for est in (bench.combinational, bench.pipelined):
        line = (f"{est.mode:14s} LUT6={est.luts:>8,}  "
                f"critical-path={est.logic_levels:>3} LUT levels")
        if est.mode == "pipelined":
            line += (f"  FF={est.ffs:>8,}  latency={est.pipeline_latency_cycles} cyc")
        print(line)
    if bench.fabric_levels:
        head = bench.combinational.logic_levels - bench.fabric_levels
        print(f"  ├ gate fabric (logic layers only): {bench.fabric_levels} LUT levels")
        print(f"  └ GroupSum popcount + argmax head:  ~{head} LUT levels  "
              f"← dominates the path")
    print("-" * 74)

    # Fmax band across the device-speed presets
    print("Fmax estimate (1 result / clock):")
    for est in (bench.combinational, bench.pipelined):
        reg = est.mode == "pipelined"
        fs = {k: fmax_mhz(est.logic_levels, v, reg) for k, v in bench.ns_per_level.items()}
        print(f"  {est.mode:14s} "
              f"{fs['conservative']:6.0f} – {fs['aggressive']:6.0f} MHz "
              f"(typical {fs['typical']:6.0f} MHz)")
    if bench.fabric_levels:
        # ceiling if the popcount/argmax head were registered into shallow stages
        stage = max(bench.fabric_levels, _HEAD_STAGE_LEVELS)
        fp = {k: fmax_mhz(stage, v, True) for k, v in bench.ns_per_level.items()}
        print(f"  {'pipelined+head':14s} "
              f"{fp['conservative']:6.0f} – {fp['aggressive']:6.0f} MHz "
              f"(typical {fp['typical']:6.0f} MHz)  ← projected w/ registered popcount")
    print("-" * 74)

    # Throughput + batch latency at the TYPICAL preset
    t = bench.ns_per_level["typical"]
    print(f"At typical {t} ns/level — peak throughput & batch wall-time:")
    for est in (bench.combinational, bench.pipelined):
        reg = est.mode == "pipelined"
        f = fmax_mhz(est.logic_levels, t, reg)
        thru = f * 1e6                          # 1 sample/clock
        print(f"  {est.mode:14s} {thru / 1e6:8.1f} Msamples/s")
        row = "      batch:"
        for B in bench.batch_sizes:
            _, total = batch_time_s(est.mode, B, est.logic_levels, t,
                                    est.pipeline_latency_cycles)
            row += f"  B={B}:{_fmt_time(total)}"
        print(row)
    print("=" * 74)
    print("note: Fmax is an estimate from logic depth; a guaranteed number needs "
          "place-and-route\n      (Vivado/Quartus). LUT6 count & logic depth are "
          "real (yosys)." if src == "yosys" else
          "note: install yosys for real LUT/depth; numbers above are analytic.")
