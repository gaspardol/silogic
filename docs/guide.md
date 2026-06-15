# User guide

This guide explains the concepts and the *when-to-use-what*; for the exact
signature of every class and function (each argument, its choices, and defaults)
see the {doc}`api`. Runnable end-to-end scripts live in the
[`examples/`](https://github.com/gaspardol/silogic/tree/main/examples) folder.

Everything is importable from the top level, e.g. `from silogic import LogicNet`.

## Core concepts

A *logic network* is a stack of layers whose neurons are **Boolean logic gates**
instead of weighted sums. Three ideas recur throughout the library:

Binary features in
: Logic networks consume `{0,1}` inputs. Real-valued data is turned into bits with
  a **thermometer code** (one bit per threshold) — see [Data](#data-and-encoding).

Soft training, hard deployment
: Every module exposes two forward methods. {meth}`forward <torch.nn.Module.forward>`
  is the **soft** circuit — a smooth, differentiable relaxation with outputs in
  `[0, 1]` that you train against. `forward_hard` is the **hard** circuit — the
  exact Boolean function (argmax over gates and connections, integer truth-table
  lookups, no multiplies) that you deploy. The accuracy difference between them is
  the **discretization gap**; several features below exist to shrink it.

GroupSum head
: {class}`~silogic.GroupSum` is the default readout: it splits the final logic
  features into one contiguous block per class and sums each block (divided by
  `tau`). It is parameter-free and maps to a popcount in hardware.

```{tip}
Compare `net(x)` (soft) against `net.forward_hard(x)` during training to watch the
discretization gap close. {func}`~silogic.train_model` reports both every epoch.
```

## Fully-connected networks

{class}`~silogic.LogicNet` stacks {class}`~silogic.LogicLayer`s of 2-input gates.
Each gate is a softmax mixture over the 16 Boolean functions; the two operands are
selected by the **connectome**:

| `connectome` | Wiring of each gate input |
|---|---|
| `"F"` | Fixed random wire (DiffLogicNet baseline), no connection params |
| `"L"` | Dense learnable softmax over **all** previous nodes |
| `"TopK"` | Learnable softmax over `k` random candidates — sparse, **the default** |
| `"BlockTopK"` | Like `TopK`, but candidates come from a contiguous window → coalesced memory access |
| `"ST"` | Sum-threshold input: `threshold(BatchNorm(popcount of k fixed wires))`, no weights |
| `"STW"` | Weighted sum-threshold with binary `{-1,+1}` weights (BNN perceptron) |
| `"STT"` | Ternary `{-1,0,+1}` weights → learnable sparsity |

`TopK` is the sweet spot for most problems. The gates are evaluated by **BasisProj**
(`gate_eval="basis"`): the 16-way distribution is projected onto the
`{1, A, B, A·B}` basis and evaluated as one 4-coefficient polynomial — numerically
identical to evaluating all 16 functions, but far cheaper. On CUDA the `TopK` /
`BlockTopK` path uses a fused Triton kernel automatically.

The head is chosen with `decoder`:

| `decoder` | Readout |
|---|---|
| `"groupsum"` | {class}`~silogic.GroupSum` (default; needs `width % num_classes == 0`) |
| `"linear"` | `nn.Linear(width, num_classes)` |
| `"linfull"` | `Linear` initialized to *equal* `GroupSum`, then free to deviate |
| `"sumlinear"` | sum features → 256-d → BatchNorm → `Linear` (needs `width % 256 == 0`) |
| `"ternary"` | every feature → every class with a learned `{-1,0,+1}` weight (a signed popcount) |

```python
import torch, silogic

net = silogic.LogicNet(in_dim=784, width=2000, depth=4, connectome="TopK", k=8)
x = (torch.rand(16, 784) > 0.5).float()
net(x)                 # soft logits  [16, 10]
net.forward_hard(x)    # hard logits  [16, 10]
net.num_gates()        # 8000 — the deployed gate count
```

## Convolutional networks

For images, {class}`~silogic.LogicTreeNet` replaces dense layers with
{class}`~silogic.ConvLogicTree` blocks: a convolution whose kernel is a complete
binary **logic-gate tree** (depth `d` → `2^d` leaves, `2^d − 1` gates), with
parameters shared across spatial placements, followed by logical OR pooling
({class}`~silogic.OrPool`, the max t-conorm). A flattened dense logic head finishes
the net.

Two settings dominate accuracy on image data:

```{note}
**Edge-detector input channels** ({func}`~silogic.edge_bits`) are the single
biggest lever — they add Sobel/Laplacian channels and lift CIFAR-10 by tens of
points. A **low GroupSum `tau`** matters too: high `tau` scales the gradient to the
conv gates by `1/tau` and starves them, so the net underfits.
```

`residual_init=True` biases each gate toward pass-through `A`, letting deep trees
train without vanishing gradients. `connect="topk"` learns the leaf connectivity
(the default); `connect="fixed"` recovers Petersen's fixed-wiring baseline. Both
have **fused Triton kernels** on CUDA (`tree_conv` for fixed, `tree_conv_topk`
for Top-K — the latter ~11× faster than the pure-PyTorch path). {meth}`gate_count(in_hw)
<silogic.LogicTreeNet.gate_count>` estimates the deployed binary-gate cost.

## WARP networks

WARP parameterizes each gate by its **Walsh–Hadamard coefficients** rather than a
16-way gate mixture: a 2-input node needs only 4 free reals (vs 16), trained as
`f = sigmoid(z / tau)` and hardened to an exact LUT in the `tau → 0` limit. Use
{class}`~silogic.WARPNet` for the 2-input case and {class}`~silogic.WARPNetN` for
general **arity `n`** — each node is then a full `2^n`-entry LUT.

```{warning}
Higher arity is more expressive but widens the soft→hard gap. Turn on
{data}`~silogic.WARP_GUMBEL` (stochastic Gumbel-sigmoid smoothing) and use
`residual_p > 0` (residual init) to keep the hard circuit aligned. See
`examples/train_mnist_warp.py`.
```

## Other node types

- {class}`~silogic.LUTkLayer` / {class}`~silogic.LUTkNet` — *k*-input lookup-table
  nodes, the FPGA-native primitive (one `LUT_k` per node, `2^k` learnable entries),
  differentiable via multilinear interpolation over the input hypercube.
- {class}`~silogic.PairLogicLayer` / {class}`~silogic.PairLogicNet` — an
  attention-like layer where the query×key dot product is replaced by a learnable
  logic gate; the multilinear gate factorizes, so the `O(in·out)` pairwise tensor
  is never materialized.
- {class}`~silogic.LUTNodeLayer` / {class}`~silogic.LUTNodeNet` — a **unified**
  n-input LUT-node layer (BitLogic-style) with a selectable `relaxation`: a full
  truth table evaluated as the multilinear `"probabilistic"` expectation, a
  `"hybrid"` node (discrete forward = inference, probabilistic surrogate gradient),
  a cheap `"linear"` perceptron node, or a degree-`d` `"polynomial"` node. All
  share Top-K input selection and a node-agnostic `residual_p` identity init.

## Learned encoders

Logic nets need binary inputs (the fixed thermometer encoders are in
[Data](#data-and-encoding)). {class}`~silogic.LearnedThermometerEncoder` instead
**learns** its per-feature thresholds jointly with the network (straight-through
binarization), then freezes them for deployment.

## Training and evaluation

{func}`~silogic.train_model` is a batteries-included loop: Adam/AdamW, optional
cosine decay (`cosine=True`), `torch.compile` (`compile_`), and soft + hard
evaluation each `val_every` epochs. `Xtr`/`Xte` are `uint8` feature tensors of any
shape the model accepts — flat `[N, D]` for {class}`~silogic.LogicNet`, spatial
`[N, C, H, W]` for {class}`~silogic.LogicTreeNet`. With `gpu_data=True` the whole
train set is moved to the GPU once. It returns
`{"test_soft", "test_hard", "train_min", "history"}`.

```python
out = silogic.train_model(net, Xtr, ytr, Xte, yte, "cuda",
                          epochs=30, optimizer="adamw", cosine=True)
print(out["test_hard"])   # deployable Boolean-circuit accuracy
```

{func}`~silogic.eval_soft` and {func}`~silogic.eval_hard` return the top-1 accuracy
(%) of the soft and hard paths on their own.

```{tip}
Disable `compile_` for the convolutional path (it mixes `unfold` with custom
kernels that don't always trace cleanly).
```

## Data and encoding

Logic nets need binary inputs, so real images are thermometer-encoded — one bit per
threshold. Datasets are read from `data/` (download MNIST/FashionMNIST/CIFAR-10 once
with torchvision).

- {func}`~silogic.binarize` / {func}`~silogic.binarize_spatial` — thermometer encode,
  flattened or keeping the spatial dims.
- {func}`~silogic.edge_bits` — Sobel-x/y + Laplacian edge/curvature channels.
- {func}`~silogic.get_dataset` / {func}`~silogic.get_dataset_cached` — flat
  binarized datasets (with augmentation) for {class}`~silogic.LogicNet`.
- {func}`~silogic.get_cifar_spatial` / {func}`~silogic.get_fmnist_spatial` — binary
  **spatial** channels (optionally with edges) for {class}`~silogic.LogicTreeNet`.

Threshold presets are exported for convenience: `MNIST_THRESH` (`[0.25]`),
`SEVEN_THRESH` (7 levels), `FOUR_THRESH`, `FIVE_THRESH`, `CIFAR3_THRESH`,
`EDGE_THRESH`.

## Gate algebra and straight-through helpers

- {data}`~silogic.BASIS_COEFFS` — `[16, 4]` tensor: row `i` is `[c0,c1,c2,c3]` with
  `gate_i(a,b) = c0 + c1·a + c2·b + c3·a·b`.
- {data}`~silogic.TRUTH_TABLES` — `[16, 4]` `uint8` hard truth tables (columns are
  the corners `(0,0),(0,1),(1,0),(1,1)`).
- {func}`~silogic.ste_threshold` — straight-through binary threshold (forward
  `(s>0)∈{0,1}`, backward the sigmoid gradient).
- {func}`~silogic.sign_ste` / {func}`~silogic.ternary_ste` — binarize / ternarize
  weights to `{-1,+1}` / `{-1,0,+1}` with a straight-through gradient.
- {func}`~silogic.gate_probs` — per-gate selection weights, honoring the global
  toggles below.

## Global toggles

These module-level dicts switch train/inference-alignment tricks on globally; set
them before the forward pass (library code restores them afterwards).

```{list-table}
:header-rows: 1
:widths: 30 70

* - Toggle
  - Effect
* - `silogic.GUMBEL`
  - `{"enabled", "tau"}` — Gumbel straight-through gate selection (hard argmax
    forward, soft Gumbel-softmax gradient) for `LogicLayer`/`ConvLogicTree`; aligns
    train and inference. Incompatible with residual *init* on deep conv nets.
* - `silogic.HARD_GATE`
  - `{"enabled"}` — deterministic hard-gate STE (argmax one-hot forward, softmax
    gradient); the noise-free counterpart to `GUMBEL`.
* - `silogic.WARP_GUMBEL`
  - `{"enabled"}` — Gumbel-sigmoid stochastic smoothing for the WARP layers.
* - `silogic.model.DEC_FEATURE_STE`
  - STE-binarize logic activations before the learned decoders so their summed input
    matches inference.
```
