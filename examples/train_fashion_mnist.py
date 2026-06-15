"""Train a convolutional LogicTreeNet on FashionMNIST.

FashionMNIST has spatial structure (garment shapes/textures), so the flat
fully-connected ``LogicNet`` ceilings around 85% hard accuracy. A convolutional
``LogicTreeNet`` over a rich binary input — a 5-level thermometer code PLUS
Sobel/Laplacian edge-detector channels (17 channels total) — reaches **~87.5%
hard** test accuracy (the deployable Boolean circuit), the best we found for a
logic-gate net at this scale.

Key recipe details (each matters, found empirically):
  * edge channels: the single biggest input lever (85% -> 87.5%);
  * light affine-only augmentation: elastic distortion caps accuracy on garments;
  * GroupSum tau=10: higher tau starves the conv gates of gradient (1/tau scale)
    and the net underfits at ~72%.

    python examples/train_fashion_mnist.py

For a simpler (lower-accuracy) fully-connected baseline, see how MNIST is done in
train_mnist.py and swap the dataset to "fmnist" with SEVEN_THRESH.
"""
import argparse
import torch

from silogic import LogicTreeNet, get_fmnist_spatial, train_model, FIVE_THRESH


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--channels", type=int, nargs="+", default=[192, 384],
                   help="conv logic-tree channels per block (each halves H,W)")
    p.add_argument("--tree-depth", type=int, default=3)
    p.add_argument("--k", type=int, default=4, help="Top-K leaf candidates")
    p.add_argument("--n-chan", type=int, default=3,
                   help="input channels each tree may observe")
    p.add_argument("--head-width", type=int, default=2560)
    p.add_argument("--head-k", type=int, default=8)
    p.add_argument("--tau", type=float, default=10.0, help="GroupSum temperature")
    p.add_argument("--no-edges", dest="edges", action="store_false",
                   help="drop the Sobel/Laplacian edge input channels")
    p.set_defaults(edges=True)
    p.add_argument("--n-aug", type=int, default=2, help="augmented train copies")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--download", action="store_true",
                   help="download the dataset via torchvision if not already present")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    print(f"Loading FashionMNIST spatial (5-level thermometer + "
          f"{'edges' if args.edges else 'no edges'}, n_aug={args.n_aug}) ...",
          flush=True)
    Xtr, ytr, Xte, yte, ch = get_fmnist_spatial(
        FIVE_THRESH, n_aug=args.n_aug, device=args.device, edges=args.edges,
        download=args.download)
    print(f"  channels={ch}  train={tuple(Xtr.shape)}  test={tuple(Xte.shape)}")

    net = LogicTreeNet(in_channels=ch, in_hw=28, channels=args.channels,
                       head_widths=[args.head_width], num_classes=10,
                       tree_depth=args.tree_depth, k=args.k, n_chan=args.n_chan,
                       head_k=args.head_k, tau=args.tau, seed=0)
    print(f"Model: LogicTreeNet channels={args.channels} "
          f"head_width={args.head_width} gates={net.gate_count(28):,}")

    # cosine LR + AdamW; eager (the conv path mixes unfold + custom kernels).
    out = train_model(net, Xtr, ytr, Xte, yte, args.device,
                      epochs=args.epochs, bs=args.batch_size, lr=args.lr,
                      optimizer="adamw", cosine=True,
                      val_every=max(1, args.epochs // 10),
                      compile_=False, eval_bs=500)
    print(f"\nFashionMNIST done in {out['train_min']:.1f} min  "
          f"soft {out['test_soft']:.2f}%  hard {out['test_hard']:.2f}%")


if __name__ == "__main__":
    main()
