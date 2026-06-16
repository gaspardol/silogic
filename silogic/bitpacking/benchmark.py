"""Inference speed benchmarks: soft vs hard vs bitpacked (CPU) vs bitpacked (GPU).

Usage
-----
Run as a module::

    python -m silogic.bitpacking.benchmark

Or call :func:`run_benchmark` / :func:`print_report` programmatically.
"""
from __future__ import annotations

import gc
import sys
import textwrap
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..models import LogicNet, WARPNet
from .convert import convert_logic_net, convert_logic_conv_net
from .ops import pack_bits, unpack_bits


# ── timing helpers ────────────────────────────────────────────────────────────

def _time_fn(fn, n_warmup: int = 3, n_runs: int = 20) -> Tuple[float, float]:
    """Return (mean_ms, std_ms) over n_runs iterations."""
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e3)
    arr = np.array(times)
    return float(arr.mean()), float(arr.std())


def _time_cuda(fn, n_warmup: int = 3, n_runs: int = 20) -> Tuple[float, float]:
    """GPU timing using CUDA events."""
    for _ in range(n_warmup):
        fn(); torch.cuda.synchronize()
    times = []
    for _ in range(n_runs):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    arr = np.array(times)
    return float(arr.mean()), float(arr.std())


# ── result container ──────────────────────────────────────────────────────────

@dataclass
class BenchResult:
    name: str
    batch_size: int
    mean_ms: float
    std_ms: float
    throughput_kps: float     # kilo-samples / second
    accuracy: Optional[float] = None
    extra: dict = field(default_factory=dict)

    def as_row(self) -> str:
        acc = f"{self.accuracy * 100:.2f}%" if self.accuracy is not None else "  n/a  "
        return (
            f"  {self.name:<28s}  {self.batch_size:>6d}  "
            f"{self.mean_ms:>9.3f}ms ± {self.std_ms:>6.3f}  "
            f"{self.throughput_kps:>10.1f}  {acc}"
        )


# ── per-model benchmark ───────────────────────────────────────────────────────

def run_benchmark(
    model: nn.Module,
    x_test: np.ndarray,             # [N, in_dim] uint8 / bool
    y_test: Optional[np.ndarray],   # [N] int labels or None
    batch_sizes: Tuple[int, ...] = (64, 512, 2048, 8192),
    device: str = "cpu",
    include_gpu: bool = False,
    n_warmup: int = 3,
    n_runs: int = 20,
) -> List[BenchResult]:
    """Benchmark soft, hard (uint8), and bitpacked (CPU ± GPU) inference.

    Parameters
    ----------
    model:
        Trained :class:`~silogic.models.LogicNet` (or compatible model with
        ``forward`` / ``forward_hard`` methods).
    x_test:
        Test inputs as uint8 ``[N, in_dim]``.
    y_test:
        Ground-truth labels ``[N]`` (used to compute accuracy for the hard /
        bitpacked paths).  Pass ``None`` to skip accuracy.
    batch_sizes:
        Which batch sizes to benchmark.
    device:
        ``"cpu"`` or ``"cuda"``.
    include_gpu:
        Also benchmark the Triton GPU bitpacked path when ``device="cuda"``.
    """
    results: List[BenchResult] = []
    model.eval()
    torch_device = torch.device(device)
    model.to(torch_device)

    # Pre-convert to bitpacked once
    bp_model = convert_logic_net(model)

    for B in batch_sizes:
        # Take a slice of the test set; repeat if too short
        idx = np.arange(B) % len(x_test)
        x_b = x_test[idx]                              # [B, in_dim] uint8
        y_b = y_test[idx] if y_test is not None else None

        x_float = torch.from_numpy(x_b).float().to(torch_device)
        x_uint8 = torch.from_numpy(x_b).to(torch.uint8).to(torch_device)
        x_packed = pack_bits(x_b)                      # [dim, n_words] int64

        # ── soft (float32) ────────────────────────────────────────────────
        def _soft():
            with torch.no_grad():
                return model(x_float)

        timer = _time_cuda if (device == "cuda") else _time_fn
        m, s = timer(_soft, n_warmup, n_runs)
        results.append(BenchResult(
            name="soft (float32)",
            batch_size=B, mean_ms=m, std_ms=s,
            throughput_kps=B / (m / 1e3) / 1e3,
        ))

        # ── hard (uint8 + truth-table) ────────────────────────────────────
        def _hard():
            with torch.no_grad():
                return model.forward_hard(x_uint8)

        m, s = timer(_hard, n_warmup, n_runs)
        logits_hard = model.forward_hard(x_uint8).cpu().numpy()
        acc_hard = None
        if y_b is not None:
            acc_hard = float((logits_hard.argmax(1) == y_b).mean())
        results.append(BenchResult(
            name="hard uint8 (torch)",
            batch_size=B, mean_ms=m, std_ms=s,
            throughput_kps=B / (m / 1e3) / 1e3,
            accuracy=acc_hard,
        ))

        # ── bitpacked CPU (numpy, gate-grouped) ──────────────────────────
        def _bp_cpu():
            return bp_model.forward(x_b)

        m_bp, s_bp = _time_fn(_bp_cpu, n_warmup, n_runs)
        logits_bp = bp_model.forward(x_b)
        acc_bp = None
        if y_b is not None:
            acc_bp = float((logits_bp.argmax(1) == y_b).mean())
        results.append(BenchResult(
            name="bitpacked CPU (gate-grouped)",
            batch_size=B, mean_ms=m_bp, std_ms=s_bp,
            throughput_kps=B / (m_bp / 1e3) / 1e3,
            accuracy=acc_bp,
        ))

        # Break down: packing vs layer forward vs head
        m_pack, _ = _time_fn(lambda: pack_bits(x_b), n_warmup, n_runs)
        def _bp_layers():
            return _run_layers_only(bp_model, x_packed)
        m_layers, _ = _time_fn(_bp_layers, n_warmup, n_runs)
        def _bp_head():
            return bp_model.head.forward(x_packed, B)
        m_head, _ = _time_fn(_bp_head, n_warmup, n_runs)
        results[-1].extra["breakdown"] = (
            f"pack={m_pack:.2f}ms  layers={m_layers:.2f}ms  head={m_head:.2f}ms"
        )

        # ── bitpacked GPU (Triton) ────────────────────────────────────────
        if include_gpu and device == "cuda":
            from .kernels import HAS_TRITON, BitpackedGPULayer
            if HAS_TRITON:
                gpu_layers = _build_gpu_model(bp_model, device)
                x_packed_cuda = torch.from_numpy(x_packed).to(torch_device)

                def _bp_gpu():
                    return _run_gpu_layers(gpu_layers, x_packed_cuda, bp_model, B)

                m_g, s_g = _time_cuda(_bp_gpu, n_warmup, n_runs)
                logits_gpu = _bp_gpu().cpu().numpy()
                acc_gpu = float((logits_gpu.argmax(1) == y_b).mean()) if y_b is not None else None
                results.append(BenchResult(
                    name="bitpacked GPU (Triton gate-grouped)",
                    batch_size=B, mean_ms=m_g, std_ms=s_g,
                    throughput_kps=B / (m_g / 1e3) / 1e3,
                    accuracy=acc_gpu,
                ))

        gc.collect()

    return results


def _run_layers_only(bp_model, packed: np.ndarray) -> np.ndarray:
    """Run only the packed layers (no pack/unpack overhead)."""
    for layer in bp_model.packed_layers:
        if layer is not None:
            packed = layer.forward(packed)
    return packed


def _build_gpu_model(bp_model, device: str):
    from .kernels import BitpackedGPULayer
    from .packed_layer import BitpackedFCLayer
    gpu_layers = []
    for layer in bp_model.packed_layers:
        if isinstance(layer, BitpackedFCLayer):
            gpu_layers.append(BitpackedGPULayer.from_fc_layer(layer, device))
        else:
            gpu_layers.append(None)
    return gpu_layers


def _run_gpu_layers(gpu_layers, x_packed: torch.Tensor, bp_model, B: int):
    """Run bitpacked layers on GPU, head on CPU."""
    packed = x_packed
    for gl in gpu_layers:
        if gl is not None:
            packed = gl.forward(packed)
    # head: unpack on CPU
    packed_np = packed.cpu().numpy()
    from .ops import mask_last_word
    mask_last_word(packed_np, B)
    return torch.from_numpy(bp_model.head.forward(packed_np, B))


# ── conv model benchmark ──────────────────────────────────────────────────────

def run_conv_benchmark(
    model: nn.Module,
    x_test: np.ndarray,             # [N, C, H, W] uint8 / bool
    y_test: Optional[np.ndarray],   # [N] int labels or None
    H_in: Optional[int] = None,     # image height (inferred from x_test if None)
    W_in: Optional[int] = None,     # image width  (inferred from x_test if None)
    batch_sizes: Tuple[int, ...] = (64, 256, 1024, 4096),
    device: str = "cpu",
    include_gpu: bool = False,
    n_warmup: int = 3,
    n_runs: int = 20,
) -> List[BenchResult]:
    """Benchmark a :class:`~silogic.models.LogicConvNet` across inference modes.

    Measured paths:

    * **soft (float32)** — full PyTorch forward.
    * **hard uint8 (torch)** — truth-table lookup in PyTorch.
    * **bitpacked CPU (B-packing)** — gate-grouped numpy B-packed pipeline.
    * **bitpacked GPU (Triton conv)** — Triton gate-tree + OR-pool on GPU
      (only when ``include_gpu=True`` and ``device="cuda"``).

    Parameters
    ----------
    model:
        Trained :class:`~silogic.models.LogicConvNet`.
    x_test:
        ``[N, C, H, W]`` uint8 test inputs.
    y_test:
        ``[N]`` int labels or ``None`` to skip accuracy.
    H_in, W_in:
        Spatial dimensions (inferred from ``x_test`` when ``None``).
    batch_sizes, device, include_gpu, n_warmup, n_runs:
        Same as :func:`run_benchmark`.
    """
    results: List[BenchResult] = []
    model.eval()
    torch_device = torch.device(device)
    model.to(torch_device)

    H_in = H_in or x_test.shape[2]
    W_in = W_in or x_test.shape[3]

    bp_cpu = convert_logic_conv_net(model)

    for B in batch_sizes:
        idx  = np.arange(B) % len(x_test)
        x_b  = x_test[idx]                          # [B, C, H, W] uint8
        y_b  = y_test[idx] if y_test is not None else None

        x_float  = torch.from_numpy(x_b).float().to(torch_device)
        x_uint8  = torch.from_numpy(x_b).to(torch.uint8).to(torch_device)

        timer = _time_cuda if (device == "cuda") else _time_fn

        # ── soft ──────────────────────────────────────────────────────────
        def _soft():
            with torch.no_grad():
                return model(x_float)

        m, s = timer(_soft, n_warmup, n_runs)
        results.append(BenchResult(
            name="soft (float32)",
            batch_size=B, mean_ms=m, std_ms=s,
            throughput_kps=B / (m / 1e3) / 1e3,
        ))

        # ── hard uint8 ────────────────────────────────────────────────────
        def _hard():
            with torch.no_grad():
                return model.forward_hard(x_uint8)

        m, s = timer(_hard, n_warmup, n_runs)
        logits_hard = model.forward_hard(x_uint8).cpu().numpy()
        acc_hard = float((logits_hard.argmax(1) == y_b).mean()) if y_b is not None else None
        results.append(BenchResult(
            name="hard uint8 (torch)",
            batch_size=B, mean_ms=m, std_ms=s,
            throughput_kps=B / (m / 1e3) / 1e3,
            accuracy=acc_hard,
        ))

        # ── bitpacked CPU ─────────────────────────────────────────────────
        def _bp_cpu():
            return bp_cpu(x_b)

        m_c, s_c = _time_fn(_bp_cpu, n_warmup, n_runs)
        logits_cpu = bp_cpu(x_b)
        acc_cpu = float((logits_cpu.argmax(1) == y_b).mean()) if y_b is not None else None
        results.append(BenchResult(
            name="bitpacked CPU (B-packing)",
            batch_size=B, mean_ms=m_c, std_ms=s_c,
            throughput_kps=B / (m_c / 1e3) / 1e3,
            accuracy=acc_cpu,
        ))

        # ── bitpacked GPU (Triton conv) ───────────────────────────────────
        if include_gpu and device == "cuda":
            from .kernels import HAS_TRITON
            from .packed_model import BitpackedConvGPUNet
            if HAS_TRITON:
                bp_gpu = BitpackedConvGPUNet.from_conv_net(bp_cpu, H_in, W_in, device)

                def _bp_gpu():
                    return bp_gpu(x_b)

                m_g, s_g = _time_cuda(_bp_gpu, n_warmup, n_runs)
                logits_gpu = bp_gpu(x_b)
                acc_gpu = float((logits_gpu.argmax(1) == y_b).mean()) if y_b is not None else None
                results.append(BenchResult(
                    name="bitpacked GPU (Triton conv)",
                    batch_size=B, mean_ms=m_g, std_ms=s_g,
                    throughput_kps=B / (m_g / 1e3) / 1e3,
                    accuracy=acc_gpu,
                ))

        gc.collect()

    return results


# ── report printing ───────────────────────────────────────────────────────────

_HEADER = (
    "  {:<28s}  {:>6s}  {:>16s}  {:>10s}  {:>7s}"
    .format("Method", "Batch", "Latency (ms)", "Throughput", "Accuracy")
)
_DIVIDER = "  " + "-" * 80


def print_report(results: List[BenchResult], title: str = "Inference Speed Report") -> None:
    print()
    print("=" * 84)
    print(f"  {title}")
    print("=" * 84)
    last_B = None
    for r in results:
        if r.batch_size != last_B:
            if last_B is not None:
                print(_DIVIDER)
            print(f"\n  Batch size = {r.batch_size}")
            print(_HEADER)
            print(_DIVIDER)
            last_B = r.batch_size
        print(r.as_row())
        if r.extra:
            for k, v in r.extra.items():
                print(f"      └─ {k}: {v}")
    print("=" * 84)
    print()


def speedup_summary(results: List[BenchResult]) -> Dict[int, Dict[str, float]]:
    """Return speedup of bitpacked over hard-uint8 per batch size."""
    from collections import defaultdict
    by_batch: Dict[int, Dict[str, float]] = defaultdict(dict)
    for r in results:
        by_batch[r.batch_size][r.name] = r.throughput_kps
    summary = {}
    for B, methods in by_batch.items():
        baseline = methods.get("hard uint8 (torch)", 1.0)
        bp = methods.get("bitpacked CPU (gate-grouped)", 0.0)
        summary[B] = {
            "hard_kps": round(baseline, 1),
            "bitpacked_kps": round(bp, 1),
            "speedup_x": round(bp / baseline, 2) if baseline > 0 else 0.0,
        }
    return summary


# ── CLI entry point ───────────────────────────────────────────────────────────

def _demo_benchmark():
    """Run a quick self-contained benchmark with a randomly-initialised model."""
    print("\nBuilding demo LogicNet (gate16, topk)…")
    in_dim, width, depth, num_classes = 784, 4000, 8, 10
    model = LogicNet(in_dim, width, depth, num_classes=num_classes, k=8)
    model.eval()

    rng = np.random.default_rng(42)
    N = 4096
    x_test = rng.integers(0, 2, size=(N, in_dim), dtype=np.uint8)
    y_test = rng.integers(0, num_classes, size=N, dtype=np.int64)

    results = run_benchmark(
        model, x_test, y_test,
        batch_sizes=(64, 512, 2048, 8192),
        device="cpu",
        n_warmup=2, n_runs=10,
    )
    print_report(results, title="Demo: LogicNet gate16/topk — gate-grouped bitpacking")

    su = speedup_summary(results)
    print("Speedup summary (bitpacked vs hard-uint8):")
    for B, info in su.items():
        print(f"  B={B:>5d}: hard={info['hard_kps']:.1f} k/s  "
              f"bitpacked={info['bitpacked_kps']:.1f} k/s  "
              f"speedup={info['speedup_x']:.2f}x")

    # Second benchmark: WARPNet (walsh arity=2)
    print("\nBuilding demo WARPNet (walsh, topk)…")
    warp = WARPNet(in_dim, width, depth, num_classes=num_classes, k=8)
    width = 4000
    warp.eval()
    results2 = run_benchmark(
        warp, x_test, y_test,
        batch_sizes=(512, 4096),
        device="cpu", n_warmup=2, n_runs=10,
    )
    print_report(results2, title="Demo: WARPNet walsh/topk — gate-grouped bitpacking")

    su2 = speedup_summary(results2)
    print("Speedup summary (WARPNet):")
    for B, info in su2.items():
        print(f"  B={B:>5d}: hard={info['hard_kps']:.1f} k/s  "
              f"bitpacked={info['bitpacked_kps']:.1f} k/s  "
              f"speedup={info['speedup_x']:.2f}x")


if __name__ == "__main__":
    _demo_benchmark()
