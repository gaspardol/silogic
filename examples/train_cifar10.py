"""Train the LogicTreeNet-G architecture on CIFAR-10 (Petersen et al., 2411.04732).

This is the paper's *largest* convolutional logic-gate network — the one that
reaches ~86.3% on CIFAR-10 in the paper. The architecture (Appendix A.1.1) is:

  * conv logic-gate-tree blocks with channels ``[s, 4s, 16s, 32s]`` (G uses
    ``s = 2048`` -> ``[2048, 8192, 32768, 65536]``, ~56M gates), 3x3 depth-3 trees,
    fixed random leaf wiring (only the gates are learned);
  * a tapering fixed dense logic head ``[1280s, 640s, 320s]`` + GroupSum (tau 450);
  * 5-level thermometer + Sobel/Laplacian edge input channels.

WARNING — this default (``--scale 2048``) is the full ~61M-gate G model:
  * ~17 GB VRAM at batch 128, and ~10+ hours for 50 epochs on a single GPU
    (the conv uses the fixed-wiring ``tree_conv`` Triton kernel);
  * trained here with **plain cross-entropy and no knowledge distillation**, so it
    lands well *below* the paper's 86% — that result needs the CNN-teacher KD
    pipeline in ``experiments/train_cifar_g.py``.

Use ``--scale`` to run a lighter version (e.g. ``--scale 256`` is ~1/8 the width):

    python examples/train_cifar10.py --scale 256          # much lighter
    python examples/train_cifar10.py                       # full G (heavy!)
"""
import argparse
import torch

from silogic import LogicTreeNet, get_cifar_spatial, train_model, FIVE_THRESH


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scale", type=int, default=2048,
                   help="G width unit s: channels [s,4s,16s,32s], head [1280s,640s,320s]")
    p.add_argument("--tree-depth", type=int, default=3)
    p.add_argument("--n-chan", type=int, default=2,
                   help="input channels each tree may observe")
    p.add_argument("--tau", type=float, default=450.0, help="GroupSum temperature (G: 450)")
    p.add_argument("--n-aug", type=int, default=2, help="augmented train copies")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--weight-decay", type=float, default=0.001)
    p.add_argument("--eval-bs", type=int, default=16,
                   help="small to avoid OOM in the int64 hard-inference eval")
    p.add_argument("--download", action="store_true",
                   help="download the dataset via torchvision if not already present")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    s = args.scale
    channels = [s, 4 * s, 16 * s, 32 * s]            # G conv blocks
    head_widths = [1280 * s, 640 * s, 320 * s]       # G tapering dense head

    print(f"Loading CIFAR-10 spatial (5-level thermometer + edges, n_aug={args.n_aug}) ...",
          flush=True)
    Xtr, ytr, Xte, yte, ch = get_cifar_spatial(
        FIVE_THRESH, n_aug=args.n_aug, device=args.device,
        edges=True, download=args.download)
    print(f"  channels={ch}  train={tuple(Xtr.shape)}  test={tuple(Xte.shape)}")

    print(f"Building LogicTreeNet-G (scale={s}): conv {channels}, head {head_widths} ...",
          flush=True)
    net = LogicTreeNet(in_channels=ch, in_hw=32, channels=channels,
                       head_widths=head_widths, num_classes=10,
                       tree_depth=args.tree_depth, kernel=3, connect="fixed",
                       head_connect="F", n_chan=args.n_chan, residual_init=True,
                       tau=args.tau, seed=0)
    print(f"  gates={net.gate_count(32):,}")

    out = train_model(net, Xtr, ytr, Xte, yte, args.device,
                      epochs=args.epochs, bs=args.batch_size, lr=args.lr,
                      optimizer="adamw", weight_decay=args.weight_decay, cosine=True,
                      val_every=max(1, args.epochs // 5), compile_=False,
                      eval_bs=args.eval_bs)
    print(f"\nCIFAR-10 (LogicTreeNet-G, no KD) done in {out['train_min']:.1f} min  "
          f"soft {out['test_soft']:.2f}%  hard {out['test_hard']:.2f}%")


if __name__ == "__main__":
    main()
