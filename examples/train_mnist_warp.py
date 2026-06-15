"""Train an arity-n WARP logic network on MNIST (default n = 4).

WARP (Walsh–Hadamard, arXiv:2602.03527) parameterizes each node by Walsh
coefficients instead of a 16-way gate mixture. `WARPNetN` generalizes this to
arity-`n` lookup-table nodes: each node sees `n` inputs and is parameterized by
`2^n` Walsh coefficients (a full n-input LUT), trained as
`sigmoid((1/tau) * Σ_i theta_i · monomial_i(2x-1))` and hardened to an exact
n-input truth table at inference.

Higher arity = more expressive nodes (fewer needed) but a wider soft→hard
discretization gap, so this example turns on **stochastic Gumbel-sigmoid
smoothing** (`WARP_GUMBEL`) and **residual init** (`residual_p`), which together
shrink that gap. At n=4, width 1280, depth 4 this reaches ~86-87% hard test
accuracy unaugmented (soft ~95%); standard 2-input WARP (`--arity 2`) reaches
~90% hard with a smaller gap. `--augment` lifts both a few points.

    python examples/train_mnist_warp.py                 # n=4
    python examples/train_mnist_warp.py --arity 2       # standard 2-input WARP
    python examples/train_mnist_warp.py --arity 6 --tau 0.3
"""
import argparse
import torch

import silogic
from silogic import WARPNetN, get_dataset_cached, train_model, MNIST_THRESH


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arity", type=int, default=4, help="inputs per node (2^arity LUT)")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--k", type=int, default=8, help="Top-K candidates per input")
    p.add_argument("--tau", type=float, default=0.5, help="sigmoid temperature")
    p.add_argument("--residual-p", type=float, default=0.9,
                   help="residual-init pass-through probability (0 disables)")
    p.add_argument("--no-gumbel", dest="gumbel", action="store_false",
                   help="disable Gumbel-sigmoid smoothing (widens the soft->hard gap)")
    p.set_defaults(gumbel=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--augment", action="store_true",
                   help="affine+elastic x10 (slower, higher acc)")
    p.add_argument("--download", action="store_true",
                   help="download the dataset via torchvision if not already present")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    print(f"Loading MNIST (augment={args.augment}) ...", flush=True)
    Xtr, ytr, Xte, yte, in_dim = get_dataset_cached(
        "mnist", MNIST_THRESH, augment=args.augment,
        n_aug=10 if args.augment else 1, device=args.device,
        download=args.download)
    print(f"  in_dim={in_dim}  train={tuple(Xtr.shape)}  test={tuple(Xte.shape)}")

    net = WARPNetN(in_dim, args.width, args.depth, arity=args.arity, num_classes=10,
                   k=args.k, tau=args.tau, residual_p=args.residual_p, seed=0)
    npar = sum(p_.numel() for p_ in net.parameters() if p_.requires_grad)
    print(f"Model: WARPNetN arity={args.arity} (2^{args.arity}={2**args.arity} "
          f"theta/node)  width={args.width} depth={args.depth}  params={npar:,}")

    # Gumbel-sigmoid smoothing applies only in training mode; eval / forward_hard
    # are unaffected. (WARPNetN has no n>2 Triton kernel, so run eager.)
    silogic.WARP_GUMBEL["enabled"] = args.gumbel
    try:
        out = train_model(net, Xtr, ytr, Xte, yte, args.device,
                          epochs=args.epochs, bs=args.batch_size, lr=args.lr,
                          optimizer="adamw", cosine=True,
                          val_every=max(1, args.epochs // 10), compile_=False)
    finally:
        silogic.WARP_GUMBEL["enabled"] = False
    print(f"\nMNIST WARP (n={args.arity}) done in {out['train_min']:.1f} min  "
          f"soft {out['test_soft']:.2f}%  hard {out['test_hard']:.2f}%")


if __name__ == "__main__":
    main()
