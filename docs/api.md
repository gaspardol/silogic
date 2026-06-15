# Silogic API reference

Everything below is importable from the top-level package, e.g.
`from silogic import LogicNet`. Every logic module subclasses `torch.nn.Module`
and provides, in addition to the standard differentiable `forward`, a
`forward_hard` method that runs the discretized Boolean circuit used at
inference.

- [Conventions](#conventions)
- [Fully-connected logic networks](#fully-connected-logic-networks)
- [Convolutional logic-gate trees](#convolutional-logic-gate-trees)
- [WARP networks](#warp-networks)
- [Alternative node types](#alternative-node-types)
- [Training and evaluation](#training-and-evaluation)
- [Data](#data)
- [Gate algebra and straight-through helpers](#gate-algebra-and-straight-through-helpers)
- [Global toggles](#global-toggles)

---

## Conventions

- **Inputs are binary features.** Logic networks consume `{0,1}` floats (use the
  thermometer encoders in [Data](#data)). `forward` accepts relaxed values in
  `[0,1]`; `forward_hard` expects `{0,1}` (it casts to `uint8`).
- **`forward` vs `forward_hard`.** `forward` is the smooth, trainable relaxation
  (outputs in `[0,1]`). `forward_hard` is the exact Boolean circuit (argmax over
  gates/connections, truth-table lookups). Train on `forward`; deploy
  `forward_hard`. The accuracy difference is the *discretization gap*.
- **Heads.** `GroupSum` partitions the final logic features into one contiguous
  block per class and sums each block (divided by `tau`) — a parameter-free,
  hardware-friendly readout.

---

## Fully-connected logic networks

### `LogicLayer(in_dim, out_dim, connectome="TopK", k=8, gate_eval="basis", seed=None, residual=False, window=0)`

One layer of 2-input logic gates. Each gate is a softmax mixture over the 16
Boolean functions; the two gate inputs are chosen by the **connectome**:

| `connectome` | Wiring of each gate input |
|---|---|
| `"F"` | Fixed random wire (DiffLogicNet baseline), no connection params |
| `"L"` | Dense learnable softmax over **all** previous nodes |
| `"TopK"` | Learnable softmax over `k` random candidates (sparse, the default) |
| `"BlockTopK"` | Like `TopK` but candidates drawn from a contiguous window → coalesced memory access |
| `"ST"` | Sum-threshold input: `threshold(BatchNorm(popcount of k fixed wires))`, no weights |
| `"STW"` | Weighted sum-threshold with binary `{-1,+1}` weights (BNN perceptron) |
| `"STT"` | Ternary `{-1,0,+1}` weights → learnable sparsity |

- `gate_eval="basis"` (default) projects the 16-gate distribution onto the
  `{1, A, B, A·B}` basis and evaluates a single 4-coefficient polynomial
  (`BasisProj`); `"full"` evaluates all 16 relaxed functions. The two are
  numerically identical.
- `residual=True` adds a structural XOR skip (`out = gate(a,b) XOR a`).
- On CUDA with Triton, `TopK`/`BlockTopK` use the fused `dense_logic` kernel
  automatically.

Methods: `forward(x) -> [B, out_dim]` (relaxed), `forward_hard(x) -> uint8 [B, out_dim]`.

### `LogicNet(in_dim, width, depth, num_classes=10, connectome="TopK", k=8, tau=1.0, gate_eval="basis", seed=0, residual=False, wire_residual=0.0, decoder="groupsum")`

A stack of `depth` `LogicLayer`s of width `width`, plus a decoder head.

**Decoders** (`decoder=`):

| value | head |
|---|---|
| `"groupsum"` | `GroupSum` (paper default; needs `width % num_classes == 0`) |
| `"linear"` | `nn.Linear(width, num_classes)` |
| `"linfull"` | `Linear` initialized to exactly equal `GroupSum`, then free to deviate |
| `"sumlinear"` | sum features → 256-d → BatchNorm → `Linear` (needs `width % 256 == 0`) |
| `"ternary"` | each feature → every class with a learned `{-1,0,+1}` weight (deployable as a signed popcount) |

- `wire_residual=f`: the first `f·width` outputs of each same-width layer are
  hardwired copies of the input — a gate-free identity highway (Gumbel-safe).
- `num_gates()` returns `width * depth`.

Methods: `forward`, `forward_hard`, `num_gates()`.

### `GroupSum(num_classes=10, tau=1.0)`

Block-sum readout: reshape `[B, width]` → `[B, num_classes, width//num_classes]`
and sum the last axis (÷ `tau`). `forward_hard` is the same without the `tau`.

---

## Convolutional logic-gate trees

### `ConvLogicTree(in_channels, out_channels, kernel=3, tree_depth=2, stride=1, padding=1, connect="topk", k=4, n_chan=2, residual_init=True, seed=0, residual=False)`

A convolution whose kernel is a complete binary **logic-gate tree** (depth `d` →
`2^d` leaves, `2^d − 1` gates), with parameters shared across spatial placements.

- `connect="fixed"` recovers Petersen's fixed random leaf wiring (and enables
  the fused `tree_conv` Triton kernel on CUDA); `connect="topk"` uses LILogic's
  learnable Top-K leaf connectivity.
- `n_chan`: each tree observes only this many randomly-chosen input channels.
- `residual_init=True` biases every gate toward pass-through `A` so deep trees
  train without vanishing gradients.

Input/output are `[B, C, H, W]`. Methods: `forward`, `forward_hard`.

### `OrPool(size=2)`

Logical OR pooling = max t-conorm = spatial max-pool. Methods: `forward`,
`forward_hard`.

### `LogicTreeNet(in_channels, in_hw, channels, head_width=None, num_classes=10, tree_depth=2, kernel=3, connect="topk", k=4, head_connect="topk", head_k=8, head_depth=2, n_chan=2, residual_init=True, tau=100.0, seed=0, residual=False, wire_residual=0.0, head_widths=None, decoder="groupsum")`

Full convolutional logic network: a stack of (`ConvLogicTree`, `OrPool`) blocks
over `channels` (each block halves H,W), flattened into a dense logic head
(`head_widths`, a tapering stack like the paper's, or `head_width × head_depth`),
finished by `GroupSum` (or `decoder="linear"`).

- `gate_count(in_hw)` estimates the deployed binary-gate count (conv trees are
  counted per spatial placement).

Methods: `forward`, `forward_hard`, `gate_count(in_hw)`.

---

## WARP networks

WARP parameterizes each gate by its Walsh–Hadamard coefficients `theta` (4 free
reals for a 2-input gate vs 16 gate-logits) with a sigmoid relaxation
`f = sigmoid((1/tau)·Σ theta_i · phi_i(s))`. Hard inference is the `tau→0` limit
(an exact LUT).

- **`WARPLayer(in_dim, out_dim, k=8, tau=1.0, residual_p=0.0, seed=None)`** —
  2-input WARP node, Top-K connectome. `residual_p>0` enables residual init
  (pass-through bias). Fused `warp_logic` Triton kernel on CUDA.
- **`WARPNet(in_dim, width, depth, num_classes=10, k=8, tau=1.0, residual_p=0.0, seed=0)`**
  — stack + `GroupSum`.
- **`WARPLayerN(in_dim, out_dim, arity=6, k=8, tau=1.0, residual_p=0.0, seed=None)`**
  — general arity-`n` LUT node: `theta ∈ R^(2^n)` over the Walsh monomials.
- **`WARPNetN(in_dim, width, depth, arity=6, ...)`** — stack of `WARPLayerN`.

Stochastic Gumbel-sigmoid smoothing (shrinks the soft→hard gap) is toggled via
[`WARP_GUMBEL`](#global-toggles).

---

## Alternative node types

- **`LUTkLayer(in_dim, out_dim, k=4, learn_conn=True, cand_k=4, seed=None)`** /
  **`LUTkNet(in_dim, width, depth, k=4, num_classes=10, tau=4.0, cand_k=4, seed=0)`**
  — k-input lookup-table nodes (the FPGA-native primitive). Differentiable via
  multilinear interpolation over the input hypercube; one `LUT_k` = one FPGA LUT.
  `LUTkNet.num_luts()` returns the LUT count.
- **`PairLogicLayer(in_dim, out_dim, n_heads=1, cand_q=8, seed=None)`** /
  **`PairLogicNet(in_dim, width, depth, num_classes=10, tau=10.0, n_heads=1, cand_q=8, seed=0)`**
  — an attention-like layer where the query×key dot product is replaced by a
  learnable logic gate; the multilinear gate factorizes so the `O(in·out)`
  pairwise tensor is never materialized. `fpga_cost()` returns the popcount cost.

---

## Training and evaluation

### `train_model(model, Xtr, ytr, Xte, yte, device, epochs=200, bs=256, lr=0.075, val_every=25, gpu_data=True, log=print, Xval=None, yval=None, compile_=True, eval_bs=1024, optimizer="adam", weight_decay=0.0, cosine=False)`

With `cosine=True` the learning rate follows a cosine decay to 0 over `epochs`
(this is what lifts the conv FashionMNIST example its final ~0.5pp).

Adam/AdamW training loop. `Xtr`/`Xte` are `uint8` feature tensors (any shape the
model accepts — flat `[N, D]` for `LogicNet`, spatial `[N, C, H, W]` for
`LogicTreeNet`). With `gpu_data=True` the whole train set is moved to the GPU
once. `compile_=True` wraps the model in `torch.compile` with static batch shapes
(disable for the conv path). Returns
`{"test_soft", "test_hard", "train_min", "history"}`.

### `eval_soft(model, X, y, device, bs=1024)` / `eval_hard(model, X, y, device, bs=1024)`

Top-1 accuracy (%) of the soft and hard forward paths respectively.

---

## Data

Faithful to arXiv:2511.12340 §3.3. Datasets are loaded from `data/` (MNIST,
FashionMNIST, CIFAR-10 must be present, e.g. downloaded once with torchvision).

- **`binarize(x, thresholds)`** — thermometer encode `[N,C,H,W]∈[0,1]` →
  `uint8 [N, C·H·W·len(thr)]` (one bit per threshold).
- **`binarize_spatial(x, thresholds)`** — thermometer encode keeping spatial dims
  → `[N, C·len(thr), H, W]`.
- **`edge_bits(x, edge_thr=None)`** — Sobel-x/y + Laplacian edge/curvature
  detector bits for CIFAR-style preprocessing.
- **`get_dataset(dataset, thresholds=None, augment=True, n_aug=None, device="cuda")`**
  — returns `(Xtr, ytr, Xte, yte, in_dim)` for `dataset ∈ {"mnist","fmnist","cifar10"}`,
  flat-binarized with the paper's augmentation.
- **`get_dataset_cached(...)`** — same, cached to `cache/` on disk.
- **`get_cifar_spatial(thresholds=None, n_aug=8, device="cuda", seed=0, edges=False)`**
  — CIFAR-10 as binary **spatial** channels for `LogicTreeNet`; returns
  `(Xtr, ytr, Xte, yte, channels)`.
- **`get_fmnist_spatial(thresholds=None, n_aug=2, device="cuda", seed=0, edges=True)`**
  — FashionMNIST as binary **spatial** channels (5-level thermometer + optional
  Sobel/Laplacian edge channels, light affine augmentation) for `LogicTreeNet`;
  returns `(Xtr, ytr, Xte, yte, channels)`. This is the input behind the ~87.5%
  hard FashionMNIST example.

Threshold presets: `MNIST_THRESH` (`[0.25]`), `SEVEN_THRESH` (7 levels),
`FOUR_THRESH`, `FIVE_THRESH`, `CIFAR3_THRESH`, `EDGE_THRESH`.

---

## Gate algebra and straight-through helpers

- **`BASIS_COEFFS`** — `[16, 4]` tensor: row `i` gives `[c0,c1,c2,c3]` with
  `gate_i(a,b) = c0 + c1·a + c2·b + c3·a·b`.
- **`TRUTH_TABLES`** — `[16, 4]` `uint8`: the hard truth tables (columns are the
  corners `(0,0),(0,1),(1,0),(1,1)`).
- **`ste_threshold(s)`** — straight-through binary threshold: forward `(s>0)∈{0,1}`,
  backward the sigmoid gradient.
- **`sign_ste(w)`** — binarize weights to `{-1,+1}` (forward), clipped STE (backward).
- **`ternary_ste(w, delta=0.5)`** — ternarize weights to `{-1,0,+1}`, STE backward.
- **`gate_probs(logits, training, dim=-1)`** — per-gate selection weights:
  Gumbel-softmax-hard (when `GUMBEL` enabled + training), deterministic hard-gate
  STE (when `HARD_GATE` enabled), else plain softmax.

---

## Global toggles

These module-level dicts switch train/inference-alignment tricks on or off
globally (set before the forward pass; restore afterward in library code/tests):

- **`silogic.GUMBEL = {"enabled": False, "tau": 1.0}`** — Gumbel straight-through
  gate selection (hard argmax forward, soft Gumbel-softmax gradient) for
  `LogicLayer`/`ConvLogicTree`. Aligns train and inference → ≈0 discretization
  gap. (Note: incompatible with residual *init* on deep conv nets — it resamples
  gates and breaks the pass-through highway.)
- **`silogic.HARD_GATE = {"enabled": False}`** — deterministic hard-gate STE
  (argmax one-hot forward, softmax gradient); the noise-free counterpart to
  `GUMBEL`.
- **`silogic.WARP_GUMBEL = {"enabled": False}`** — Gumbel-sigmoid stochastic
  smoothing for the WARP layers.
- **`silogic.model.DEC_FEATURE_STE = True`** — STE-binarize logic activations
  before the learned decoders so their summed input matches inference.
