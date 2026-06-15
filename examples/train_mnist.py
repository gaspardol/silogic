"""Train a fully-connected logic-gate network on MNIST.

Uses the silogic public API end-to-end:
  * ``get_dataset_cached`` — thermometer-binarized + (optionally) augmented MNIST,
  * ``LogicNet`` — a stack of Top-K logic layers + a GroupSum head,
  * ``train_model`` — Adam training with soft + hard (Boolean-circuit) evaluation.

The reported ``hard`` accuracy is the deployable Boolean circuit (argmax gates +
connections), i.e. what runs on an FPGA/ASIC with zero multiplies.

Default run reaches ~98% HARD test accuracy (the deployable Boolean circuit) in
a handful of epochs on a GPU — measured 98.1% hard / 98.5% soft by epoch 5 with
this config (width 10000, depth 6, x10 augmentation).

    python examples/train_mnist.py

Fast smoke run (no augmentation, ~96% hard, ~1 min):
    python examples/train_mnist.py --no-augment --width 2000 --depth 4
"""
import argparse
import torch

from silogic import LogicNet, get_dataset_cached, train_model, MNIST_THRESH


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--width", type=int, default=10000)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--k", type=int, default=8, help="Top-K candidates per input")
    p.add_argument("--tau", type=float, default=10.0, help="GroupSum temperature")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--no-augment", dest="augment", action="store_false",
                   help="disable affine+elastic x10 augmentation (faster, ~96%% hard)")
    p.set_defaults(augment=True)
    p.add_argument("--no-compile", action="store_true")
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

    net = LogicNet(in_dim, args.width, args.depth, num_classes=10,
                   connectome="F", k=args.k, tau=args.tau, seed=0)
    print(f"Model: LogicNet width={args.width} depth={args.depth} "
          f"gates={net.num_gates():,}")

    out = train_model(net, Xtr, ytr, Xte, yte, args.device,
                      epochs=args.epochs, bs=args.batch_size, lr=args.lr,
                      val_every=max(1, args.epochs // 10),
                      compile_=not args.no_compile)
    print(f"\nMNIST done in {out['train_min']:.1f} min  "
          f"soft {out['test_soft']:.2f}%  hard {out['test_hard']:.2f}%")


if __name__ == "__main__":
    main()
