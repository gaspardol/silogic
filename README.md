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
python examples/train_cifar10_small.py    # conv LogicTreeNet on CIFAR-10 (~67% hard, <=250k gates)
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
| `train_cifar10_small.py` | conv `LogicTreeNet` (arity-6 LUT, edges, STE) | **~67%** (228k gates, ~1 hr) |
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

## Bitpacked inference

After training, any `LogicNet` or `LogicConvNet` can be converted to a
**bitpacked** inference object that packs 64 samples into a single `int64`
word and evaluates the whole batch with one bitwise instruction per gate:

```python
from silogic.bitpacking import convert_logic_net, convert_logic_conv_net

# FC model
bp = convert_logic_net(trained_logic_net)
logits = bp(x_uint8)           # [B, num_classes] float32, numpy input

# Conv model
bp = convert_logic_conv_net(trained_conv_net)
logits = bp(x_uint8_BCHW)     # [B, num_classes] float32, [B, C, H, W] uint8 input
```

### How it works

**FC layers** pack the batch into `[dim, ceil(B/64)]` int64 words.  Gates are
pre-sorted by type (16 buckets) so all AND-gates fire one `a & b` over
`[n_group, n_words]`, all XOR-gates fire one `a ^ b`, etc. — zero per-neuron
Python branching.

**Conv layers** use *B-packing*: the batch is packed but spatial positions stay
explicit as a `[dim, L, nw_B]` tensor where `L = H×W` and `nw_B = ceil(B/64)`.
The image is packed **once**; packed-unfold, gate-tree, and OR-pool all run
without expanding `B`.  The final `[n_out, L_final, nw_B]` reshapes directly
to `[feat_dim, nw_B]` for the FC head — no inter-layer repack.

### CPU speedups

Benchmarked on Intel Xeon w5-3423, single socket (gate16/fixed):

| Model | Mode | B=64 | B=256 | B=1024 |
|---|---|---|---|---|
| **LogicNet** (w=4000, d=8) | hard uint8 (torch) | 5 ms | 12 ms | 111 ms |
| | **bitpacked CPU** | 1.0 ms | 2.3 ms | 3.8 ms |
| | *speedup* | **~5×** | **~5×** | **~29×** |
| **LogicConvNet** (3 blocks, 32×32) | hard uint8 (torch) | 188 ms | 872 ms | 3500 ms |
| | **bitpacked CPU** | 6.4 ms | 23 ms | 110 ms |
| | *speedup* | **~29×** | **~38×** | **~32×** |

Speedup grows with batch size because the per-batch overhead (one pack) is
amortized over more samples, and SIMD utilization improves.

### GPU (Triton kernels)

Triton GPU kernels are provided for both **FC layers** and **conv gate-tree layers**.
Each gate type gets its own specialised kernel launch (16 variants, dead branches
eliminated at JIT time — no warp divergence).

Benchmarked on NVIDIA RTX A6000 (gate16/topk).  The GPU hard baseline is
`forward_hard` on CUDA; bitpacked CPU is the B-packing numpy pipeline;
bitpacked Triton is the GPU kernel with all layers on-device:

| Model | Mode | B=64 | B=256 | B=1024 | B=4096 |
|---|---|---|---|---|---|
| **LogicNet** (w=4000, d=8) | hard uint8 (GPU torch) | 0.86 ms | 1.6 ms | 4.6 ms | 18.6 ms |
| | bitpacked CPU | 1.0 ms | 2.2 ms | 3.9 ms | 13.3 ms |
| | *CPU vs GPU hard* | *0.9×* | *0.7×* | *1.2×* | *1.4×* |
| | **bitpacked Triton FC (GPU)** | 3.4 ms | 3.6 ms | 4.7 ms | 8.7 ms |
| | *Triton FC vs GPU hard* | *0.3×* | *0.4×* | *1.0×* | **2.1×** |
| **LogicConvNet** (3 blocks, 32×32) | hard uint8 (GPU torch) | 6.4 ms | 24 ms | 95 ms | 379 ms |
| | bitpacked CPU (B-packing) | 6.8 ms | 22 ms | 110 ms | 855 ms |
| | *CPU vs GPU hard* | *1.0×* | *1.1×* | *0.9×* | *0.4×* |
| | **bitpacked Triton conv (GPU)** | 3.9 ms | 4.7 ms | 9.9 ms | 87 ms |
| | *Triton conv vs GPU hard* | **1.6×** | **5.1×** | **9.6×** | **4.4×** |


### What is supported

| Component | Status |
|---|---|
| `gate16` node (all FC connectomes) | fully bitpacked |
| `walsh` arity=2 | fully bitpacked (converted to gate16) |
| `multilinear` / `hybrid` / `linear` / `polynomial` arity=2 | fully bitpacked (converted to gate16) |
| any node, arity>2 | bitpacked LUT (sum-of-products) |
| `SumThresholdConnectome` FC layers | uint8 fallback + repack |
| `ConvLogicTree` `node="gate16"` | fully bitpacked (B-packing) + Triton GPU |
| `ConvLogicTree` arity=2 nodes (walsh, multilinear, …) | fully bitpacked (converted to gate16, B-packing) + Triton GPU |
| `ConvLogicTree` arity>2 nodes (multilinear/hybrid arity=4, …) | fully bitpacked (LUT sum-of-products, B-packing); GPU falls back to CPU |

To benchmark and print a full report:

```python
from silogic.bitpacking import run_benchmark, run_conv_benchmark, print_report

# FC model
results = run_benchmark(model, x_test, y_test,
                        batch_sizes=(64, 512, 2048, 8192))
print_report(results)

# Conv model (add include_gpu=True + device="cuda" for Triton conv numbers)
results = run_conv_benchmark(conv_model, x_test_BCHW, y_test,
                             batch_sizes=(64, 256, 1024, 4096))
print_report(results)
```

## FPGA / Verilog export

A discretized `LogicNet` *is* a feed-forward network of LUTs followed by
`GroupSum` popcount adders — i.e. a combinational FPGA design. `silogic.fpga`
lowers a trained FC `LogicNet` (**any** node family — `gate16` / `walsh` /
`multilinear` / `hybrid` / `linear` / `polynomial`; **GroupSum head**) to
synthesizable Verilog plus a self-checking testbench:

```python
import silogic
from silogic.fpga import export_logic_net

net = silogic.LogicNet(784, 800, depth=3, num_classes=10, connectome="topk")
# ... train net ...

circuit = export_logic_net(net.eval(), "build/mynet", name="mynet")
print(circuit.summary())     # in/out widths, group size, LUT count
```

That writes a complete, simulatable project under `build/mynet/`:

| file | what |
|---|---|
| `mynet.v` | the synthesizable module (`x` → `class_scores`, `pred`) |
| `mynet_tb.v` | self-checking testbench (prints `PASS` / `FAIL`) |
| `x_vectors.mem` / `expected.mem` | random vectors + golden predictions |
| `run_sim.sh` | Icarus Verilog runner (`bash run_sim.sh`) |

Each logic node becomes one combinational `y = (TT >> {operands}) & 1'b1` —
a constant truth table indexed by the node's selected wires, which synthesis
maps to **one device LUT** (arity ≤ the LUT size, e.g. LUT6). The `GroupSum`
head becomes `num_classes` popcount adder trees (`class_scores`) and a
combinational argmax (`pred`). `tau` is an argmax-invariant positive scale and
is not applied in hardware.

```python
from silogic.fpga import extract_logic_net, to_verilog, simulate

circuit = extract_logic_net(net)                 # backend-independent LUT IR
verilog = to_verilog(circuit, pipeline=True)     # registered pipeline (1 result/cycle)
scores, pred = simulate(circuit, x_uint8)        # bit-exact numpy golden model
```

* **`pipeline=False`** (default) — purely combinational, lowest area, single
  cycle.
* **`pipeline=True`** — input / every layer / head registered (adds `clk` +
  `rst_n`); latency `depth + 2` cycles, one result per cycle, much higher Fmax.

The numpy `simulate` is bit-exact with `forward_hard` (the test suite asserts
this for every node family), and — when Icarus Verilog is installed — the
generated Verilog is simulated against golden vectors and must report `PASS`
(`tests/test_fpga.py`, both forms, every node family). Runnable example:

```bash
python examples/export_fpga_mnist.py --width 800 --depth 3 --epochs 5
bash build/mnist_logicnet/run_sim.sh        # -> PASS
```

> Supported: FC `LogicNet`, all node families, `GroupSum` head, and the
> wire-selecting connectomes (`fixed` / `topk` / `blocktopk` / `dense`). The
> `SumThreshold` connectomes (no static wire fan-in) and the convolutional
> `LogicConvNet` are not yet exported.

### Inference-speed estimate

An FPGA design has no host to time, so "speed" means **throughput + latency
derived from the synthesized circuit**. `benchmark_fpga` synthesizes the
generated Verilog to 6-input LUTs with [yosys](https://yosyshq.net) (`abc -lut 6`)
to get the *real* LUT6 count and the *real* logic-level depth of the critical
path, then turns depth into a clock period with a per-LUT-level delay (a band,
since a guaranteed Fmax needs place-and-route):

```python
from silogic.fpga import benchmark_fpga, print_fpga_report
print_fpga_report(benchmark_fpga(trained_logic_net))   # yosys optional
```

Measured for an FC `gate16` `LogicNet` (width 1280 × depth 4, 10 classes,
≈MNIST-scale; yosys to LUT6, typical 0.45 ns/level):

| | LUT6 | critical path | Fmax (typ.) | throughput | batch=64 |
|---|---|---|---|---|---|
| **combinational** | 2.6k | 30 LUT levels | ~74 MHz | 74 Msample/s | **0.86 µs** |
| **pipelined** | 5.8k | 25 LUT levels | ~85 MHz | 85 Msample/s | **0.82 µs** |

The headline result is *what the depth is made of*: the **gate fabric is only
2 LUT levels** (ABC collapses the arity-2 layers almost entirely) — the other
~28 levels are the `GroupSum` popcount + argmax accumulator, which is the
critical path in **both** modes. So the logic is essentially free and a
**registered popcount head would lift the pipelined Fmax to ~300 MHz**
(`benchmark_fpga` reports this projection). Even un-tuned, a batch of 64 runs in
under a microsecond — vs ~0.4 ms for the bitpacked CPU path at the same batch
(same Xeon w5-3423, this width-1280 × depth-4 net).
Without yosys the report falls back to a coarse analytic model.

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

# FPGA / Verilog export (silogic.fpga)
from silogic.fpga import export_logic_net      # LogicNet -> Verilog project
from silogic.fpga import extract_logic_net, to_verilog, simulate, make_testbench
from silogic.fpga import benchmark_fpga, print_fpga_report   # LUT/Fmax/throughput
```

Full documentation — a user guide and an autogenerated API reference — is at
**<https://gaspardol.github.io/silogic/>**.

## Contributing

Contributions are very welcome — bug fixes, new node types, kernels, examples,
benchmarks, and docs. See **[`CONTRIBUTING.md`](CONTRIBUTING.md)** for the dev
setup, how to run the tests/docs, and the project conventions.

- New here? Start with a [**`good first issue`**](https://github.com/gaspardol/silogic/labels/good%20first%20issue).
- Bigger ideas (faster kernels, Verilog export for the convolutional
  `LogicConvNet`, learned-decoder heads in HDL) are tagged
  [**`help wanted`**](https://github.com/gaspardol/silogic/labels/help%20wanted).
  FC `LogicNet` Verilog export already lives in [`silogic.fpga`](#fpga--verilog-export).
- Questions and ideas → [**Discussions**](https://github.com/gaspardol/silogic/discussions).

## License

[MIT](LICENSE).
