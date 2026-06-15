"""Silogic — differentiable logic-gate networks in PyTorch.

A compact, hackable reimplementation (and several extensions) of differentiable
logic gate networks:

  * **LILogic Net** (arXiv:2511.12340) — learnable Top-K connectivity + BasisProj.
  * **Convolutional Logic Gate Networks** (Petersen et al., arXiv:2411.04732) —
    logic-gate-tree convolutions + OR pooling + residual init (``LogicTreeNet``).
  * **WARP** (arXiv:2602.03527) — Walsh–Hadamard gate parameterization
    (4 params/node vs 16) with stochastic smoothing (``WARPNet``, ``WARPNetN``).
  * Extra node types: k-input LUTs (``LUTkLayer``) and attention-like pairwise
    logic (``PairLogicLayer``).

Every module follows the standard PyTorch ``nn.Module`` convention. In addition
to the usual differentiable ``forward`` (relaxed Boolean values in ``[0, 1]``),
each logic module exposes a ``forward_hard`` that runs the discretized Boolean
circuit actually deployed at inference — argmax over gates/connections, integer
truth-table lookups. Comparing the two measures the discretization gap.

Quickstart
----------
>>> import torch, silogic
>>> net = silogic.LogicNet(in_dim=784, width=2000, depth=4, connectome="TopK")
>>> x = (torch.rand(16, 784) > 0.5).float()
>>> net(x).shape                 # soft, differentiable logits
torch.Size([16, 10])
>>> net.forward_hard(x).shape    # hard Boolean-circuit logits
torch.Size([16, 10])
"""

__version__ = "0.1.0"

# --- core FC logic network -------------------------------------------------
from .model import (
    LogicLayer,
    LogicNet,
    GroupSum,
    BASIS_COEFFS,
    TRUTH_TABLES,
    ste_threshold,
    sign_ste,
    ternary_ste,
    gate_probs,
    GUMBEL,
    HARD_GATE,
)

# --- convolutional logic-gate trees ---------------------------------------
from .conv import ConvLogicTree, OrPool
from .treenet import LogicTreeNet

# --- WARP (Walsh–Hadamard) logic networks ---------------------------------
from .warp import (
    WARPLayer,
    WARPNet,
    WARPLayerN,
    WARPNetN,
    WARP_GUMBEL,
)

# --- alternative node types -----------------------------------------------
from .lutk import LUTkLayer, LUTkNet
from .pairlogic import PairLogicLayer, PairLogicNet

# --- training + data ------------------------------------------------------
from .train import train_model, eval_hard, eval_soft
from .data import (
    get_dataset,
    get_dataset_cached,
    get_cifar_spatial,
    get_fmnist_spatial,
    binarize,
    binarize_spatial,
    edge_bits,
    MNIST_THRESH,
    SEVEN_THRESH,
    FOUR_THRESH,
    FIVE_THRESH,
    CIFAR3_THRESH,
    EDGE_THRESH,
)

__all__ = [
    "__version__",
    # core
    "LogicLayer", "LogicNet", "GroupSum", "BASIS_COEFFS", "TRUTH_TABLES",
    "ste_threshold", "sign_ste", "ternary_ste", "gate_probs",
    "GUMBEL", "HARD_GATE",
    # conv
    "ConvLogicTree", "OrPool", "LogicTreeNet",
    # warp
    "WARPLayer", "WARPNet", "WARPLayerN", "WARPNetN", "WARP_GUMBEL",
    # node types
    "LUTkLayer", "LUTkNet", "PairLogicLayer", "PairLogicNet",
    # training
    "train_model", "eval_hard", "eval_soft",
    # data
    "get_dataset", "get_dataset_cached", "get_cifar_spatial", "get_fmnist_spatial",
    "binarize", "binarize_spatial", "edge_bits",
    "MNIST_THRESH", "SEVEN_THRESH", "FOUR_THRESH", "FIVE_THRESH",
    "CIFAR3_THRESH", "EDGE_THRESH",
]
