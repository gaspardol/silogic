"""Train a CONVOLUTIONAL logic-gate network (LogicTreeNet) on CIFAR-10.

CIFAR-10 needs spatial structure, so this example uses the convolutional part of
the library rather than the flat FC ``LogicNet``:

  * ``get_cifar_spatial`` — CIFAR-10 as binary spatial channels (a 3-level
    thermometer code per colour -> 9 input channels), with crop+flip augmentation.
  * ``LogicTreeNet`` — Petersen-style logic-gate-tree convolutions
    (``ConvLogicTree``) + logical OR pooling (``OrPool``) + residual init, wired
    with LILogic's learnable Top-K connectivity, finished by a dense logic head
    + GroupSum.
  * ``train_model`` — generic Adam training with soft + hard evaluation.

This is heavier than the MNIST/FashionMNIST examples; a GPU is recommended.

Quick run:
    python examples/train_cifar10.py

Bigger (closer to the LogicTreeNet-S/M regime):
    python examples/train_cifar10.py --channels 256 512 512 --head-width 4000 \
        --n-aug 8 --epochs 200
"""
import argparse
import torch

from silogic import LogicTreeNet, get_cifar_spatial, train_model, CIFAR3_THRESH


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--channels", type=int, nargs="+", default=[128, 256, 256],
                   help="conv logic-tree channels per block (each halves H,W)")
    p.add_argument("--tree-depth", type=int, default=2)
    p.add_argument("--k", type=int, default=4, help="Top-K leaf candidates")
    p.add_argument("--n-chan", type=int, default=2,
                   help="input channels each tree may observe")
    p.add_argument("--head-width", type=int, default=2000)
    p.add_argument("--head-k", type=int, default=8)
    p.add_argument("--tau", type=float, default=100.0)
    p.add_argument("--n-aug", type=int, default=2, help="augmented train copies")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--download", action="store_true",
                   help="download the dataset via torchvision if not already present")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    print(f"Loading CIFAR-10 spatial (3-level thermometer, n_aug={args.n_aug}) ...",
          flush=True)
    Xtr, ytr, Xte, yte, ch = get_cifar_spatial(
        CIFAR3_THRESH, n_aug=args.n_aug, device=args.device,
        download=args.download)
    print(f"  channels={ch}  train={tuple(Xtr.shape)}  test={tuple(Xte.shape)}")

    net = LogicTreeNet(in_channels=ch, in_hw=32, channels=args.channels,
                       head_widths=[args.head_width], num_classes=10,
                       tree_depth=args.tree_depth, k=args.k, n_chan=args.n_chan,
                       head_k=args.head_k, tau=args.tau, seed=0)
    print(f"Model: LogicTreeNet channels={args.channels} "
          f"head_width={args.head_width} gates={net.gate_count(32):,}")

    # torch.compile is disabled: the conv path mixes unfold + custom Triton
    # kernels that don't always trace cleanly; eager is robust here.
    out = train_model(net, Xtr, ytr, Xte, yte, args.device,
                      epochs=args.epochs, bs=args.batch_size, lr=args.lr,
                      optimizer="adamw", val_every=max(1, args.epochs // 10),
                      compile_=False, eval_bs=200)
    print(f"\nCIFAR-10 done in {out['train_min']:.1f} min  "
          f"soft {out['test_soft']:.2f}%  hard {out['test_hard']:.2f}%")


if __name__ == "__main__":
    main()
