"""Train a small FC LogicNet on MNIST and export it to synthesizable Verilog.

End-to-end FPGA flow with the silogic public API:

  1. ``get_dataset_cached`` — thermometer-binarized MNIST (binary features in,
     ready for a Boolean circuit),
  2. ``LogicNet`` (any node family) + a ``GroupSum`` head — trained with
     ``train_model``,
  3. ``silogic.fpga.export_logic_net`` — lower the *discretized* network
     (``forward_hard``) to a Verilog module + a self-checking testbench + golden
     test vectors.

The exported circuit is purely combinational LUTs + popcount adders (one LUT per
logic node) — exactly what an FPGA implements natively, no multiplies. Pass
``--pipeline`` for a registered pipeline (one result/cycle, higher Fmax).

    python examples/export_fpga_mnist.py --width 800 --depth 3 --epochs 5

Then simulate the result (needs Icarus Verilog, ``iverilog``):

    bash build/mnist_logicnet/run_sim.sh        # prints PASS / FAIL

The script also runs the pure-Python golden simulator and reports its agreement
with ``forward_hard`` on the real MNIST test set, so you get a correctness number
even without any HDL tools installed.
"""
import argparse

import numpy as np
import torch

from silogic import LogicNet, get_dataset_cached, train_model, MNIST_THRESH
from silogic.fpga import export_logic_net, simulate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--width", type=int, default=800,
                   help="layer width (must be divisible by num_classes for GroupSum)")
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--node", default="gate16",
                   help="node family: gate16/walsh/multilinear/hybrid/linear/polynomial")
    p.add_argument("--arity", type=int, default=2, help="fan-in for n-input nodes")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--tau", type=float, default=10.0)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--outdir", default="build/mnist_logicnet")
    p.add_argument("--pipeline", action="store_true",
                   help="emit the registered (clocked) pipeline form")
    p.add_argument("--n-vectors", type=int, default=512,
                   help="number of random test vectors to write for the testbench")
    p.add_argument("--benchmark", action="store_true",
                   help="estimate LUT count / Fmax / throughput (uses yosys if installed)")
    p.add_argument("--download", action="store_true")
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    # width must be divisible by classes (GroupSum), round up if needed
    if args.width % 10:
        args.width += 10 - args.width % 10
        print(f"rounded width up to {args.width} (divisible by 10 classes)")

    print(f"Loading MNIST ...", flush=True)
    Xtr, ytr, Xte, yte, in_dim = get_dataset_cached(
        "mnist", MNIST_THRESH, augment=False, device=args.device,
        download=args.download)
    print(f"  in_dim={in_dim}  train={tuple(Xtr.shape)}  test={tuple(Xte.shape)}")

    net = LogicNet(in_dim, args.width, args.depth, num_classes=10,
                   node=args.node, arity=args.arity, connectome="topk",
                   k=args.k, tau=args.tau, seed=0)
    print(f"Model: LogicNet {args.node} arity={args.arity} "
          f"width={args.width} depth={args.depth} gates={net.num_gates():,}")

    out = train_model(net, Xtr, ytr, Xte, yte, args.device,
                      epochs=args.epochs, bs=args.batch_size, lr=args.lr,
                      val_every=max(1, args.epochs // 5),
                      compile_=not args.no_compile)
    print(f"trained: soft {out['test_soft']:.2f}%  hard {out['test_hard']:.2f}%")

    # ── export to Verilog ──────────────────────────────────────────────────
    net = net.to("cpu").eval()
    circuit = export_logic_net(net, args.outdir, name="mnist_logicnet",
                               n_vectors=args.n_vectors, pipeline=args.pipeline)
    print("\n" + circuit.summary())
    print(f"wrote Verilog + testbench + vectors -> {args.outdir}/")
    print(f"  simulate with:  bash {args.outdir}/run_sim.sh")

    # ── golden check on the real test set (no HDL tools needed) ─────────────
    xte = (Xte.cpu().numpy() != 0).astype(np.uint8)
    with torch.no_grad():
        ref = net.forward_hard(Xte.cpu().float()).numpy().argmax(1)
    _, pred = simulate(circuit, xte)
    agree = float((pred == ref).mean()) * 100
    acc = float((pred == yte.cpu().numpy()).mean()) * 100
    print(f"\ngolden HDL-IR simulator vs forward_hard on MNIST test: "
          f"{agree:.2f}% agreement (should be 100.00%)")
    print(f"exported-circuit MNIST test accuracy: {acc:.2f}%")

    # ── inference-speed estimate (synthesizes to LUT6 if yosys is installed) ─
    if args.benchmark:
        from silogic.fpga import benchmark_fpga, print_fpga_report
        print()
        print_fpga_report(benchmark_fpga(circuit, batch_sizes=(64, 256, 1024, 4096)))


if __name__ == "__main__":
    main()
