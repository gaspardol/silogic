---
sd_hide_title: true
---

# Silogic

```{div} sd-text-center sd-fs-2 sd-font-weight-bold
Silogic
```

```{div} sd-text-center sd-fs-5 sd-text-muted
Differentiable logic-gate networks in PyTorch
```

Silogic trains neural networks whose neurons are *Boolean logic gates*. During
training each gate is a smooth, differentiable relaxation; at inference the
network discretizes into a pure Boolean circuit — argmax over gates and
connections, integer truth-table lookups, **no multiplies** — which maps directly
onto FPGA/ASIC LUTs for cheap, low-latency inference.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`book` User guide
Connectomes, decoders, WARP, and the global toggles.
+++
```{button-ref} guide
:color: primary
:expand:
Read the guide
```
:::
:::{grid-item-card} {octicon}`code` API reference
Every public class and function, generated from the docstrings.
+++
```{button-ref} api
:color: primary
:expand:
Browse the API
```
:::
::::

## Install

```bash
pip install -e .                # core (torch + torchvision)
pip install -e ".[triton,dev]"  # + fused CUDA kernels + pytest
```

## Quickstart

Every logic module exposes two forward methods — `forward(x)` (the **soft**,
differentiable circuit) and `forward_hard(x)` (the **hard**, deployable Boolean
circuit). The gap between their accuracies is the *discretization gap*.

```python
import torch
import silogic

net = silogic.LogicNet(in_dim=784, width=2000, depth=4,
                       connectome="TopK", k=8, tau=10.0)

x = (torch.rand(16, 784) > 0.5).float()   # logic nets consume binary features
soft_logits = net(x)                       # differentiable — train against this
hard_logits = net.forward_hard(x)          # Boolean circuit — deploy this
```

## What's implemented

| Paper | Contribution |
|---|---|
| **LILogic Net** ([arXiv:2511.12340](https://arxiv.org/abs/2511.12340)) | Learnable Top-K connectivity + `BasisProj` → {class}`~silogic.LogicLayer` / {class}`~silogic.LogicNet` |
| **Convolutional Logic Gate Networks** ([arXiv:2411.04732](https://arxiv.org/abs/2411.04732)) | Logic-gate-tree convolutions + OR pooling → {class}`~silogic.ConvLogicTree` / {class}`~silogic.LogicTreeNet` |
| **WARP** ([arXiv:2602.03527](https://arxiv.org/abs/2602.03527)) | Walsh–Hadamard gate parameterization → {class}`~silogic.WARPNet` / {class}`~silogic.WARPNetN` |

Plus k-input LUT nodes ({class}`~silogic.LUTkLayer`) and the BitLogic
relaxations ({class}`~silogic.LUTNodeLayer`).

```{toctree}
:hidden:
:maxdepth: 2

guide
api
```
