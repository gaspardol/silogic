# Silogic

[![CI](https://github.com/gaspardol/silogic/actions/workflows/ci.yml/badge.svg)](https://github.com/gaspardol/silogic/actions/workflows/ci.yml)
[![Docs](https://github.com/gaspardol/silogic/actions/workflows/docs.yml/badge.svg)](https://github.com/gaspardol/silogic/actions/workflows/docs.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Differentiable logic-gate networks in PyTorch.**

Silogic trains neural networks whose neurons are *Boolean logic gates*. During
training each gate is a smooth, differentiable relaxation; at inference the
network discretizes into a pure Boolean circuit — argmax over gates and
connections, integer truth-table lookups, **no multiplies** — which maps
directly onto FPGA/ASIC LUTs for extremely cheap, low-latency inference.

This package is a compact, hackable reimplementation (plus several extensions)
of three lines of work:

| Paper | What it contributes here |
|---|---|
| **LILogic Net** ([arXiv:2511.12340](https://arxiv.org/abs/2511.12340)) | Learnable Top-K connectivity + `BasisProj` 4-coefficient gate evaluation → `LogicLayer` / `LogicNet` |
| **Convolutional Logic Gate Networks** ([arXiv:2411.04732](https://arxiv.org/abs/2411.04732)) | Logic-gate-tree convolutions + OR pooling + residual init → `ConvLogicTree` / `OrPool` / `LogicTreeNet` |
| **WARP** ([arXiv:2602.03527](https://arxiv.org/abs/2602.03527)) | Walsh–Hadamard gate parameterization (4 params/node vs 16) + stochastic smoothing → `WARPLayer` / `WARPNet` / `WARPNetN` |
| **BitLogic** ([arXiv:2602.07400](https://arxiv.org/abs/2602.07400)) | n-input LUT nodes with selectable boundary-consistent relaxations (`multilinear`/`hybrid`/`linear`/`polynomial`) + learned thermometer encoding → `LUTNodeLayer` / `LUTkLayer` / `LearnedThermometerEncoder` |

Optional **fused Triton CUDA kernels** accelerate the FC, convolutional, and
WARP layers; the library transparently falls back to pure PyTorch when
Triton/CUDA are unavailable.

---

## Install

```bash
pip install -e .                # core (torch + torchvision)
pip install -e ".[triton,dev]"  # + fused CUDA kernels + pytest
```

Python ≥ 3.9, PyTorch ≥ 2.1. A CUDA GPU is optional but recommended (the Triton
kernels and `gpu_data` training path need one).

## Quickstart

Every logic module exposes two forward methods:

- `forward(x)` — the **soft**, differentiable circuit (relaxed Booleans in `[0,1]`),
- `forward_hard(x)` — the **hard**, deployable Boolean circuit (the inference path).

The gap between their accuracies is the *discretization gap*.

```python
import torch
import silogic

net = silogic.LogicNet(in_dim=784, width=2000, depth=4,
                       connectome="TopK", k=8, tau=10.0)

x = (torch.rand(16, 784) > 0.5).float()   # logic nets consume binary features
soft_logits = net(x)                       # differentiable — train against this
hard_logits = net.forward_hard(x)          # Boolean circuit — deploy this
print(net.num_gates())                     # hardware cost (gate count)
```

## Examples

Runnable training scripts (`--help` on each for the full config):

```bash
python examples/train_mnist.py            # FC LogicNet on MNIST
python examples/train_fashion_mnist.py    # conv LogicTreeNet on FashionMNIST
python examples/train_cifar10_small.py    # fast conv LogicTreeNet on CIFAR-10 (~60%)
python examples/train_cifar10_large.py    # the paper's large LogicTreeNet-G (heavy)
```

Each loads a (thermometer-binarized) dataset via the library's data utilities,
builds a model from the public API, trains with `silogic.train_model`, and
reports both **soft** and **hard** test accuracy (the hard number is the
deployable Boolean circuit). Measured with the default configs:

| example | model | hard test acc |
|---|---|---|
| `train_mnist.py` | FC `LogicNet` (width 10000 × 6) | **~98.1%** |
| `train_fashion_mnist.py` | conv `LogicTreeNet` (5-thr + edges) | **~87.5%** |
| `train_cifar10_small.py` | conv `LogicTreeNet` (edges) | **~60%** (~5 min) |
| `train_cifar10_large.py` | LogicTreeNet-G architecture (`--scale` to lighten) | TBD |

> MNIST is easy enough for the flat fully-connected `LogicNet`. FashionMNIST and
> CIFAR-10 have spatial structure, so they use the **convolutional**
> `LogicTreeNet` (logic-gate-tree convs + OR pooling). For small conv nets like
> the FashionMNIST one the two biggest levers are **edge-detector input channels**
> and a **low GroupSum `tau`** (high `tau` starves the conv gates of gradient).

## Architecture

Every model composes three small registries, so the node families that used to
be separate classes are now one parametrizable layer:

| Registry | Module | Choices |
|---|---|---|
| **node** (gate/LUT parameterization) | `silogic.nodes` | `gate16`, `walsh`, `multilinear`, `hybrid`, `linear`, `polynomial` |
| **connectome** (input wiring) | `silogic.connectomes` | `topk`, `blocktopk`, `fixed`, `dense`, `st`, `stw`, `stt` |
| **decoder** (head) | `silogic.heads` | `groupsum`, `linear`, `linfull`, `sumlinear`, `ternary` |

```python
import silogic

# One layer, any parameterization + wiring:
layer = silogic.LogicLayer(256, 512, node="walsh", connectome="topk", arity=2)

# One network builder over any node/connectome/decoder:
net = silogic.LogicNet(784, 2000, depth=4, node="multilinear", arity=4,
                       connectome="topk", decoder="groupsum")
```

The **convolutional** side takes the same `node=` registry. Each output channel
is a `node`-arity *tree* of depth `tree_depth`: the 2-input families (`gate16`,
`walsh`) use 2-input gates (default depth 2), the n-input families
(`multilinear`/`hybrid`/`linear`/`polynomial`) use `arity`-input LUT nodes
(default depth 1 = one LUT per channel, or deeper for a tree of LUTs):

```python
# A convolutional net with multilinear (LUT-k) conv nodes:
cnet = silogic.LogicConvNet(3, 32, channels=[128, 256], head_widths=[1280],
                            node="multilinear", arity=4, connect="topk")
```

Fused Triton kernels live in `silogic.kernels` (optional; CPU import works without
them) — the dense `gate16` head, the `gate16` conv tree, the Walsh (`walsh`) layer,
the n-input `hybrid` LUT layer (DWN straight-through), and the convolutional
`hybrid` LUT-tree of any depth (image-reading, no unfold). Shared gate algebra and
straight-through estimators are in `silogic.functional`. Train/inference-alignment
tricks are **constructor arguments**, not globals — `gate_select`
(`"softmax"`/`"gumbel"`/`"hard"`, and a per-layer list on `LogicNet` so a network
can mix soft and hard gates), `gumbel_tau`, `decoder_ste`, `use_triton`.

## Public API

```python
import silogic

# Fully-connected logic network (generic builder + the unified layer)
silogic.LogicLayer, silogic.LogicNet, silogic.GroupSum

# Convolutional logic-gate trees (the spatial mirror of LogicLayer / LogicNet)
silogic.ConvLogicTree, silogic.ConvLogicLayer, silogic.OrPool   # layer
silogic.LogicConvNet, silogic.LogicTreeNet                      # network

# Named presets (thin wrappers over LogicLayer / LogicNet)
silogic.WARPLayer, silogic.WARPNet, silogic.WARPLayerN, silogic.WARPNetN
silogic.LUTkLayer, silogic.LUTkNet, silogic.LUTNodeLayer, silogic.LUTNodeNet

# Registries (build any variant by name)
silogic.build_node, silogic.build_connectome, silogic.build_decoder

# Training / evaluation
silogic.train_model, silogic.eval_soft, silogic.eval_hard

# Data: thermometer binarization + augmentation + caching
silogic.get_dataset, silogic.get_dataset_cached
silogic.get_cifar_spatial, silogic.get_fmnist_spatial
silogic.binarize, silogic.binarize_spatial, silogic.edge_bits

# Gate algebra + straight-through helpers
silogic.BASIS_COEFFS, silogic.TRUTH_TABLES
silogic.ste_threshold, silogic.sign_ste, silogic.ternary_ste, silogic.gate_probs
```

Full documentation — a user guide and an autogenerated API reference — is at
**<https://gaspardol.github.io/silogic/>**.

## Contributing

Contributions are very welcome — bug fixes, new node types, kernels, examples,
benchmarks, and docs. See **[`CONTRIBUTING.md`](CONTRIBUTING.md)** for the dev
setup, how to run the tests/docs, and the project conventions.

- New here? Start with a [**`good first issue`**](https://github.com/gaspardol/silogic/labels/good%20first%20issue).
- Bigger ideas (faster kernels, bit-packed inference, Verilog export) are tagged
  [**`help wanted`**](https://github.com/gaspardol/silogic/labels/help%20wanted).
- Questions and ideas → [**Discussions**](https://github.com/gaspardol/silogic/discussions).

## License

[MIT](LICENSE).
