API reference
=============

.. currentmodule:: silogic

This page is generated from the source docstrings. Everything below is importable
from the top level, e.g. ``from silogic import LogicNet``. For a narrative tour of
connectomes, decoders, and the global toggles, see the :doc:`guide`.

Networks
--------

End-to-end models (a stack of logic layers + a classification head). Each exposes
``forward`` (the differentiable soft circuit) and ``forward_hard`` (the deployable
Boolean circuit).

.. autosummary::
   :toctree: generated/
   :nosignatures:

   LogicNet
   LogicConvNet
   LogicTreeNet
   WARPNet
   WARPNetN
   LUTkNet
   LUTNodeNet

Layers
------

The building blocks — individual logic-gate layers and the classification head.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   LogicLayer
   ConvLogicTree
   ConvLogicLayer
   OrPool
   GroupSum
   WARPLayer
   WARPLayerN
   LUTkLayer
   LUTNodeLayer
   LearnedThermometerEncoder

Training and evaluation
-----------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   train_model
   eval_soft
   eval_hard

Data
----

Thermometer binarization, spatial encoding, edge detectors, and cached dataset
loaders.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   get_dataset
   get_dataset_cached
   get_cifar_spatial
   get_fmnist_spatial
   binarize
   binarize_spatial
   edge_bits

Gate algebra and straight-through helpers
-----------------------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ste_threshold
   sign_ste
   ternary_ste
   gate_probs
