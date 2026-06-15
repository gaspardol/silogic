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
python examples/train_cifar10.py          # conv LogicTreeNet on CIFAR-10
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
| `train_cifar10.py` | conv `LogicTreeNet` | scales with config |
| `train_mnist_warp.py` | arity-4 WARP (`WARPNetN`, 2⁴-LUT nodes) | **~86%** |

> MNIST is easy enough for the flat fully-connected `LogicNet`. FashionMNIST and
> CIFAR-10 have spatial structure, so they use the **convolutional**
> `LogicTreeNet` (logic-gate-tree convs + OR pooling); the FC net ceilings ~3pp
> lower on FashionMNIST. For both conv examples the two biggest levers are
> **edge-detector input channels** and a **low GroupSum `tau`** (high `tau`
> starves the conv gates of gradient).

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

Full documentation (user guide + autogenerated API reference) is at
**<https://gaspardol.github.io/silogic/>**.

## Tests

```bash
pytest                 # 70+ tests; Triton/CUDA tests auto-skip without a GPU
```

The suite covers gate algebra (`BasisProj == FullEval`, truth tables), all
connectomes and decoders, `forward`/`forward_hard` shapes and ranges, gradient
flow, hardened `soft == hard` equivalence, every model family, the data
pipeline, and Triton-vs-PyTorch kernel correctness.

## Repository layout

```
silogic/            the installable package (docs: gaspardol.github.io/silogic)
  model.py          LogicLayer, LogicNet, GroupSum, gate algebra, STE helpers
  conv.py           ConvLogicTree, OrPool
  treenet.py        LogicTreeNet (conv backbone + logic head)
  warp.py           WARP layers/nets (arity 2 and general n)
  lutk.py           k-input LUT nodes
  pairlogic.py      attention-like pairwise-logic layer
  data.py           binarization, augmentation, caching
  train.py          train_model + soft/hard evaluation
  triton_*.py       fused CUDA kernels (dense_logic, tree_conv, warp_logic)
tests/              pytest suite
examples/           MNIST / FashionMNIST / CIFAR-10 training scripts
docs/               Sphinx docs (user guide + autogenerated API reference)
experiments/        research scripts + reproduction notes (RESEARCH.md)
```

The research/experiment scripts (`exp_*.py`, `train_cifar*.py`, `bench_*.py`,
the `driver_*.sh` scripts) live in [`experiments/`](experiments/); their findings
are documented in [`experiments/RESEARCH.md`](experiments/RESEARCH.md).

## Continuous integration & docs

Three GitHub Actions workflows are included (`.github/workflows/`):

- **`ci.yml`** — runs the pytest suite on every push / PR across Python
  3.10–3.12 on CPU runners (the GPU/Triton tests auto-skip).
- **`docs.yml`** — builds the Sphinx site (`docs/`, pydata-sphinx-theme) and
  deploys it to **GitHub Pages** on every push to `main`.
- **`release.yml`** — on a version tag (`v*`), builds the sdist + wheel,
  `twine check`s them, and publishes to **PyPI** via Trusted Publishing (OIDC,
  no API token needed).

To turn these on after pushing to GitHub:

1. **Settings → Pages → Build and deployment → Source: GitHub Actions** (docs).
2. For releases, register a **PyPI Trusted Publisher** for this repo +
   `release.yml` at <https://pypi.org/manage/account/publishing/>, then cut a
   release with `git tag v0.1.0 && git push origin v0.1.0`.
3. Push to `main` — CI runs immediately; the docs deploy to
   <https://gaspardol.github.io/silogic/>.

The CI / Docs status badges are at the top of this README. Build the docs
locally with `pip install -e ".[docs]" && sphinx-build -b html docs docs/_build/html`.

## License

[MIT](LICENSE).
