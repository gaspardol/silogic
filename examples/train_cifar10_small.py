"""Train a logic-gate network on CIFAR-10 under a tight gate budget.

Goal: high **hard** (deployable Boolean circuit) test accuracy with **<= 250,000
logic gates**, where gates are counted as the library reports them — one LUT
*node* == one gate (the ``gate_count`` value printed below). Inspired by the
BitLogic paper (arXiv:2602.07400, n-input LUT nodes + thermometer encoding).

The default config reaches **~67% hard** (and ~67% *soft* — the two are made
**equal**, see below) in ~1 hour on a GPU, using **228,608 gates**:

  * ``conv`` (default) — a convolutional :class:`silogic.LogicTreeNet`: three
    ``ConvLogicTree`` + OR-pool blocks of arity-6 ``hybrid`` LUT nodes
    (channels ``[64, 256, 1024]``), then a dense logic head + GroupSum. CIFAR is
    thermometer-binarized (``--bits`` levels) with Sobel/Laplacian edge channels —
    edges and a low GroupSum ``tau`` are the biggest accuracy levers.
  * ``ffn`` / ``ffnenc`` — flat :class:`silogic.LogicNet` alternatives (a plain
    flattened-pixel FFN, or one fronted by a learned thermometer encoder). On
    CIFAR the convolutional inductive bias wins, so ``conv`` is the default.

**Closing the soft/hard discretization gap.** Two straight-through tricks (both on
by default) make the *soft* training forward use exactly the operations the
deployed Boolean circuit uses, so ``soft acc == hard acc`` every epoch:
  * ``--leaf-ste`` — conv/head Top-K input *selection* uses a hard argmax in the
    forward (softmax gradient), so it picks the same wire as inference;
  * the GroupSum head straight-through *binarizes* its features before summing, so
    it counts the same bits the hard circuit does.
Bit-flip input augmentation (``--flip-p``) + weight decay regularize the net.

    python examples/train_cifar10_small.py                 # default conv, ~67% hard
    python examples/train_cifar10_small.py --arch ffn      # flat FFN alternative
    python examples/train_cifar10_small.py --cutout 8      # + spatial cutout aug
"""
import argparse
import os
import torch
import torch.nn as nn
import torchvision.transforms.v2 as T

from silogic import (LogicNet, LogicTreeNet, get_cifar_spatial, train_model,
                     LearnedThermometerEncoder)
from silogic.data import _raw, _augment_batches, CACHE_DIR


def get_cifar_flat_float(n_aug, device, seed=0, download=False):
    """CIFAR-10 as flattened float pixels ``[N, 3072]`` in [0,1] with crop+flip
    augmentation (no binarization — the learned encoder does that). Cached."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"cifar_flatfloat_n{n_aug}_s{seed}.pt")
    if os.path.exists(path):
        d = torch.load(path)
        return d["Xtr"], d["ytr"], d["Xte"], d["yte"]
    torch.manual_seed(seed)
    imgs, labels = _raw("cifar10", train=True, download=download)
    te_imgs, te_labels = _raw("cifar10", train=False, download=download)
    transform = T.Compose([
        T.RandomCrop(32, padding=4, padding_mode="reflect"),
        T.RandomHorizontalFlip(p=0.5),
    ])
    copies, ys = [], []
    for _ in range(max(1, n_aug)):
        aug = _augment_batches(imgs, transform, device).clamp(0, 1)
        copies.append(aug.reshape(aug.shape[0], -1))
        ys.append(labels)
    Xtr = torch.cat(copies, 0)
    ytr = torch.cat(ys, 0)
    Xte = te_imgs.reshape(te_imgs.shape[0], -1)
    torch.save({"Xtr": Xtr, "ytr": ytr, "Xte": Xte, "yte": te_labels}, path)
    return Xtr, ytr, Xte, te_labels


class EncodedNet(nn.Module):
    """Learned thermometer encoder (per-feature thresholds, BitLogic) -> LUT FFN.
    Float pixels in, class logits out. The encoder discretizes exactly at
    inference; gate count is the LUT-node count of the inner net only."""

    def __init__(self, num_features, bits, net, flip_p=0.0):
        super().__init__()
        self.enc = LearnedThermometerEncoder(num_features, bits)
        self.net = net
        self.flip_p = flip_p

    def forward(self, x):
        b = self.enc(x)
        if self.training and self.flip_p > 0.0:
            m = torch.rand_like(b) < self.flip_p
            b = torch.where(m, 1.0 - b, b)
        return self.net(b)

    @torch.no_grad()
    def forward_hard(self, x):
        return self.net.forward_hard(self.enc.forward_hard(x).float())

    def num_gates(self):
        return self.net.num_gates()


def thermometer_levels(bits):
    """``bits`` thresholds evenly spread in (0,1). Reproduces the library's
    standard lists (3->CIFAR3_THRESH, 5->FIVE_THRESH, 7->SEVEN_THRESH), and since
    ``get_cifar_spatial`` caches by threshold *count*, a given ``bits`` reuses any
    existing cache of that size."""
    return [(i + 1) / (bits + 1) for i in range(bits)]


def build_ffn(in_dim, args):
    net = LogicNet(in_dim=in_dim, width=args.width, depth=args.depth,
                   num_classes=10, node="hybrid", connectome="topk",
                   arity=args.arity, k=args.k, tau=args.tau,
                   decoder="groupsum", seed=0)
    return net, net.num_gates()


def build_conv(ch, args):
    # channels sized so conv nodes + head nodes <= budget at 32x32 input:
    #   128*32^2 + 256*16^2 = 196,608 conv nodes; + 2*head_width head nodes.
    net = LogicTreeNet(in_channels=ch, in_hw=32, channels=args.channels,
                       head_widths=[args.head_width, args.head_width], num_classes=10,
                       node="hybrid", arity=args.arity, connect="topk",
                       tree_depth=1, k=args.k, n_chan=args.n_chan,
                       head_k=8, tau=args.tau, decoder=args.decoder,
                       gate_select="hard" if args.leaf_ste else "softmax", seed=0)
    return net, net.gate_count(32)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arch", choices=["ffn", "ffnenc", "conv"], default="conv")
    p.add_argument("--bits", type=int, default=5,
                   help="thermometer threshold levels per colour channel")
    p.add_argument("--gate-budget", type=int, default=250_000)
    # FFN knobs
    p.add_argument("--width", type=int, default=16000)
    p.add_argument("--depth", type=int, default=2)
    # conv knobs
    p.add_argument("--channels", type=int, nargs="+", default=[64, 256, 1024])
    p.add_argument("--head-width", type=int, default=16000)
    p.add_argument("--n-chan", type=int, default=4)
    # shared
    p.add_argument("--arity", type=int, default=6, help="LUT node fan-in (max 6)")
    p.add_argument("--k", type=int, default=8, help="Top-K candidates per leaf")
    p.add_argument("--tau", type=float, default=12.0, help="GroupSum temperature")
    p.add_argument("--no-edges", dest="edges", action="store_false",
                   help="drop the Sobel/Laplacian edge input channels")
    p.set_defaults(edges=True)
    p.add_argument("--n-aug", type=int, default=4, help="augmented train copies")
    p.add_argument("--epochs", type=int, default=70)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--weight-decay", type=float, default=2e-3)
    p.add_argument("--flip-p", type=float, default=0.05,
                   help="bit-flip input augmentation prob (BitLogic regularizer)")
    p.add_argument("--decoder", default="groupsum",
                   choices=["groupsum", "ternary", "linfull", "sumlinear"],
                   help="conv head readout: groupsum (popcount) or a learned, still "
                        "Boolean-deployable decoder (ternary = signed popcount)")
    p.add_argument("--cutout", type=int, default=0,
                   help="spatial cutout square size (px) per training batch (conv only)")
    p.add_argument("--no-leaf-ste", dest="leaf_ste", action="store_false",
                   help="disable straight-through conv leaf selection (on by "
                        "default: soft forward picks the same candidate as hard, "
                        "closing the discretization gap so soft acc == hard acc)")
    p.set_defaults(leaf_ste=True)
    p.add_argument("--download", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    # bit-flip aug lives inside EncodedNet for ffnenc (inputs are float pixels);
    # for ffn/conv the binary inputs are flipped by train_model directly.
    train_flip = 0.0 if args.arch == "ffnenc" else args.flip_p

    if args.arch == "ffnenc":
        print(f"Loading CIFAR-10 flat float pixels (learned {args.bits}-bit "
              f"thermometer, n_aug={args.n_aug}) ...", flush=True)
        Xtr, ytr, Xte, yte = get_cifar_flat_float(
            args.n_aug, device=args.device, download=args.download)
        print(f"  train={tuple(Xtr.shape)}  test={tuple(Xte.shape)}")
        inner = LogicNet(in_dim=Xtr.shape[1] * args.bits, width=args.width,
                         depth=args.depth, num_classes=10, node="hybrid",
                         connectome="topk", arity=args.arity, k=args.k,
                         tau=args.tau, decoder="groupsum", seed=0)
        net = EncodedNet(Xtr.shape[1], args.bits, inner, flip_p=args.flip_p)
        gates = net.num_gates()
        desc = (f"EncodedNet learned-therm {args.bits}b -> FFN "
                f"width={args.width} depth={args.depth}")
    else:
        thresh = thermometer_levels(args.bits)
        print(f"Loading CIFAR-10 spatial ({args.bits}-level thermometer + "
              f"{'edges' if args.edges else 'no edges'}, n_aug={args.n_aug}) ...",
              flush=True)
        Xtr, ytr, Xte, yte, ch = get_cifar_spatial(
            thresh, n_aug=args.n_aug, device=args.device,
            edges=args.edges, download=args.download)
        print(f"  channels={ch}  train={tuple(Xtr.shape)}  test={tuple(Xte.shape)}")
        if args.arch == "ffn":
            Xtr = Xtr.reshape(Xtr.shape[0], -1)
            Xte = Xte.reshape(Xte.shape[0], -1)
            net, gates = build_ffn(Xtr.shape[1], args)
            desc = f"LogicNet FFN width={args.width} depth={args.depth}"
        else:
            net, gates = build_conv(ch, args)
            desc = f"LogicTreeNet channels={args.channels} head={args.head_width}"

    print(f"Model: {desc}  node=hybrid arity={args.arity} k={args.k} tau={args.tau}  "
          f"gates={gates:,}  (budget {args.gate_budget:,})", flush=True)
    assert gates <= args.gate_budget, (
        f"gate count {gates:,} exceeds budget {args.gate_budget:,}")

    out = train_model(net, Xtr, ytr, Xte, yte, args.device,
                      epochs=args.epochs, bs=args.batch_size, lr=args.lr,
                      optimizer="adamw", weight_decay=args.weight_decay, cosine=True,
                      val_every=max(1, args.epochs // 12), compile_=False,
                      eval_bs=200, flip_p=train_flip, cutout=args.cutout)
    print(f"\nCIFAR-10 [{args.arch}] done in {out['train_min']:.1f} min  "
          f"gates {gates:,}  soft {out['test_soft']:.2f}%  hard {out['test_hard']:.2f}%")


if __name__ == "__main__":
    main()
