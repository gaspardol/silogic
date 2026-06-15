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

Plus two extra node types: k-input LUTs (`LUTkLayer`) and an attention-like
pairwise-logic layer (`PairLogicLayer`).

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
python examples/train_mnist_warp.py       # arity-4 WARP network on MNIST
```

Each loads a (thermometer-binarized) dataset via the library's data utilities,
builds a model from the public API, trains with `silogic.train_model`, and
reports both **soft** and **hard** test accuracy (the hard number is the
deployable Boolean circuit). Measured with the default configs:

| example | model | hard test acc |
|---|---|---|
| `train_mnist.py` | FC `LogicNet` (width 10000 × 6) | **~98.1%** |
| `train_fashion_mnist.py` | conv `LogicTreeNet` (5-thr + edges) | **~87.5%** |
| `train_cifar10_small.py` | conv `LogicTreeNet` (edges) | ~55% (~10 min) |
| `train_cifar10_large.py` | LogicTreeNet-G architecture (`--scale` to lighten) | heavy — ~86% needs KD |
| `train_mnist_warp.py` | arity-4 WARP (`WARPNetN`, 2⁴-LUT nodes) | **~86%** |

> MNIST is easy enough for the flat fully-connected `LogicNet`. FashionMNIST and
> CIFAR-10 have spatial structure, so they use the **convolutional**
> `LogicTreeNet` (logic-gate-tree convs + OR pooling). The CIFAR example is the
> paper's large **LogicTreeNet-G** architecture (heavy — see its `--scale` flag
> and the KD reproduction in `experiments/`); for small conv nets like the
> FashionMNIST one the two biggest levers are **edge-detector input channels**
> and a **low GroupSum `tau`** (high `tau` starves the conv gates of gradient).

## Public API

```python
import silogic

# Fully-connected logic network
silogic.LogicLayer, silogic.LogicNet, silogic.GroupSum

# Convolutional logic-gate trees
silogic.ConvLogicTree, silogic.OrPool, silogic.LogicTreeNet

# WARP (Walsh–Hadamard) networks, arity 2 and general n
silogic.WARPLayer, silogic.WARPNet, silogic.WARPLayerN, silogic.WARPNetN

# Alternative node types
silogic.LUTkLayer, silogic.LUTkNet, silogic.PairLogicLayer, silogic.PairLogicNet

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
