"""Silogic — differentiable logic-gate networks in PyTorch.

A compact, hackable reimplementation (and several extensions) of differentiable
logic gate networks. The library is organized around three small registries that
compose into every model:

  * **nodes** (:mod:`silogic.nodes`) — the gate/LUT *parameterization*: the 16
    two-input functions (``"gate16"``), Walsh/WARP coefficients (``"walsh"``),
    multilinear k-LUTs (``"multilinear"``), and the ``"hybrid"``/``"linear"``/
    ``"polynomial"`` relaxations. 2-input and n-input.
  * **connectomes** (:mod:`silogic.connectomes`) — the input *wiring*:
    ``"fixed"`` / ``"dense"`` / ``"topk"`` / ``"blocktopk"`` / ``"st"`` / ``"stw"``
    / ``"stt"``.
  * **heads** (:mod:`silogic.heads`) — the *decoder*: ``GroupSum`` or a learned
    ``"linear"`` / ``"linfull"`` / ``"sumlinear"`` / ``"ternary"`` readout.

A :class:`LogicLayer` is ``connectome + node``; a :class:`LogicNet` stacks layers
and adds a head. The convolutional side mirrors this exactly: a
:class:`ConvLogicLayer` (gate-tree convolution) and a :class:`LogicConvNet` that
stacks conv blocks + a logic head. Named presets (``WARPNet``, ``LUTkNet``,
``LogicTreeNet``, …) just fix those choices. Every logic module exposes a
differentiable ``forward`` (relaxed Booleans in ``[0,1]``) and a discretized
``forward_hard`` (the deployed Boolean circuit); comparing them measures the
discretization gap.

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

# --- gate algebra + straight-through primitives ---------------------------
from .functional import (
    BASIS_COEFFS, TRUTH_TABLES,
    ste_threshold, sign_ste, ternary_ste, binarize_ste, gate_probs,
    residual_logit,
)

# --- registries: input wiring, node parameterization, decoders ------------
from .connectomes import build_connectome
from .nodes import build_node, NODES, RELAXATIONS
from .heads import GroupSum, build_decoder

# --- layers: unified FC layer + conv layer + named presets ----------------
from .layers import (LogicLayer, WARPLayer, WARPLayerN, LUTkLayer, LUTNodeLayer,
                     ConvLogicTree, ConvLogicLayer, OrPool)

# --- networks: generic builders + presets (FC and convolutional) ----------
from .models import (LogicNet, WARPNet, WARPNetN, LUTkNet, LUTNodeNet,
                     LogicConvNet, LogicTreeNet)

# --- learned input encoders -----------------------------------------------
from .encoders import LearnedThermometerEncoder

# --- training + data ------------------------------------------------------
from .train import train_model, eval_hard, eval_soft
from .data import (
    get_dataset, get_dataset_cached, get_cifar_spatial, get_fmnist_spatial,
    binarize, binarize_spatial, edge_bits,
    MNIST_THRESH, SEVEN_THRESH, FOUR_THRESH, FIVE_THRESH, CIFAR3_THRESH,
    EDGE_THRESH,
)

__all__ = [
    "__version__",
    # primitives
    "BASIS_COEFFS", "TRUTH_TABLES", "ste_threshold", "sign_ste", "ternary_ste",
    "binarize_ste", "gate_probs", "residual_logit",
    # registries
    "build_connectome", "build_node", "NODES", "RELAXATIONS",
    "GroupSum", "build_decoder",
    # layers + presets (FC + conv)
    "LogicLayer", "WARPLayer", "WARPLayerN", "LUTkLayer", "LUTNodeLayer",
    "ConvLogicTree", "ConvLogicLayer", "OrPool",
    # networks (FC + conv)
    "LogicNet", "WARPNet", "WARPNetN", "LUTkNet", "LUTNodeNet",
    "LogicConvNet", "LogicTreeNet",
    # encoders
    "LearnedThermometerEncoder",
    # training
    "train_model", "eval_hard", "eval_soft",
    # data
    "get_dataset", "get_dataset_cached", "get_cifar_spatial", "get_fmnist_spatial",
    "binarize", "binarize_spatial", "edge_bits",
    "MNIST_THRESH", "SEVEN_THRESH", "FOUR_THRESH", "FIVE_THRESH",
    "CIFAR3_THRESH", "EDGE_THRESH",
]
