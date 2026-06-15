# Silogic

**Differentiable logic-gate networks in PyTorch.**

Silogic trains neural networks whose neurons are *Boolean logic gates*. During
training each gate is a smooth, differentiable relaxation; at inference the
network discretizes into a pure Boolean circuit — argmax over gates and
connections, integer truth-table lookups, **no multiplies** — which maps
directly onto FPGA/ASIC LUTs for extremely cheap, low-latency inference.

This package is a compact, hackable reimplementation (plus several extensions)
of three lines of work:

| Paper | Contribution here |
|---|---|
| **LILogic Net** ([arXiv:2511.12340](https://arxiv.org/abs/2511.12340)) | Learnable Top-K connectivity + `BasisProj` gate evaluation → `LogicLayer` / `LogicNet` |
| **Convolutional Logic Gate Networks** ([arXiv:2411.04732](https://arxiv.org/abs/2411.04732)) | Logic-gate-tree convolutions + OR pooling → `ConvLogicTree` / `LogicTreeNet` |
| **WARP** ([arXiv:2602.03527](https://arxiv.org/abs/2602.03527)) | Walsh–Hadamard gate parameterization (4 params/node) → `WARPNet` / `WARPNetN` |

## Install

```bash
pip install -e .                # core (torch + torchvision)
pip install -e ".[triton,dev]"  # + fused CUDA kernels + pytest
```

## Quickstart

Every logic module exposes two forward methods:

- `forward(x)` — the **soft**, differentiable circuit (relaxed Booleans in `[0,1]`),
- `forward_hard(x)` — the **hard**, deployable Boolean circuit (the inference path).

```python
import torch
import silogic

net = silogic.LogicNet(in_dim=784, width=2000, depth=4,
                       connectome="TopK", k=8, tau=10.0)

x = (torch.rand(16, 784) > 0.5).float()   # logic nets consume binary features
soft_logits = net(x)                       # differentiable — train against this
hard_logits = net.forward_hard(x)          # Boolean circuit — deploy this
```

## Examples

Three runnable training scripts ship in `examples/`:

| script | model | hard test acc |
|---|---|---|
| `train_mnist.py` | FC `LogicNet` | ~98.1% |
| `train_fashion_mnist.py` | conv `LogicTreeNet` (thermometer + edges) | ~87.5% |
| `train_cifar10.py` | conv `LogicTreeNet` | scales with config |

## More

See the [API Reference](api.md) for the full surface — layers, networks,
decoders, connectomes, training, data utilities, and global toggles.
