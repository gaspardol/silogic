"""The unified logic layers: a connectome (input selection) + a node (gate
parameterization), plus the convolutional gate-tree layer, with fused Triton paths.

One :class:`LogicLayer` covers every FC logic-gate / LUT / WARP variant by
choosing ``node=`` and ``connectome=``; the named presets (:class:`WARPLayer`,
:class:`WARPLayerN`, :class:`LUTkLayer`, :class:`LUTNodeLayer`) are thin wrappers
that just set those defaults and expose family-specific attribute names.
:class:`ConvLogicTree` is the spatial analogue (a ``node``-arity gate tree).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import kernels
from .connectomes import build_connectome, TopKConnectome, FixedConnectome
from .nodes import build_node, _NODE_ALIASES
from .functional import BASIS_COEFFS, TRUTH_TABLES, gate_probs, GATE_A


class LogicLayer(nn.Module):
    """A layer of logic nodes with a configurable connectome and parameterization.

    Args:
        in_dim (int): Number of binary inputs.
        out_dim (int): Number of nodes (output width).
        node (str): Node parameterization (see :mod:`silogic.nodes`): ``"gate16"``
            (default), ``"walsh"``, ``"multilinear"``, ``"hybrid"``, ``"linear"``,
            ``"polynomial"``.
        connectome (str): Input wiring (see :mod:`silogic.connectomes`): ``"topk"``
            (default), ``"fixed"``/``"F"``, ``"dense"``/``"L"``, ``"blocktopk"``,
            ``"st"``, ``"stw"``, ``"stt"``.
        arity (int): Inputs per node. Forced to ``2`` for ``node="gate16"``.
            Default ``2``.
        k (int): Candidates per input slot for the sparse connectomes. Default ``8``.
        gate_eval (str): For ``node="gate16"``, ``"basis"`` (default) or ``"full"``.
        residual_p (float): If ``>0``, residual-initialize each node toward passing
            through its first input with this probability. Default ``0.0``.
        seed (int | None): Seed for the random wiring. Default ``None``.
        window (int): Candidate window for ``connectome="blocktopk"``. Default ``0``.
        tau (float): Sigmoid temperature for ``node="walsh"``. Default ``1.0``.
        degree (int | None): Polynomial degree for ``node="polynomial"``.
        gate_select (str): Gate selection during training: ``"softmax"`` (default),
            ``"gumbel"`` (hard argmax forward + Gumbel-softmax gradient; Gumbel-
            sigmoid smoothing for ``node="walsh"``), or ``"hard"`` (argmax one-hot
            forward + softmax gradient).
        gumbel_tau (float): Temperature for ``gate_select="gumbel"``. Default ``1.0``.
        use_triton (bool): Use the fused Triton kernels when available. Default
            ``True``; can also be toggled on the instance (``layer.use_triton``).
    """

    def __init__(self, in_dim, out_dim, node="gate16", connectome="topk", arity=2,
                 k=8, gate_eval="basis", residual_p=0.0, seed=None,
                 window=0, tau=1.0, degree=None, gate_select="softmax",
                 gumbel_tau=1.0, use_triton=True):
        super().__init__()
        self._node_name = _NODE_ALIASES.get(str(node).lower(), node)
        if self._node_name == "gate16":
            arity = 2
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.arity = arity
        self.k = k
        self.gate_eval = gate_eval
        self.use_triton = use_triton    # per-layer fused-kernel toggle (else PyTorch)
        # legacy attribute: the connectome NAME (the submodule is self._conn)
        self.connectome = connectome
        self._conn = build_connectome(connectome, in_dim, out_dim, arity, k, seed, window)
        self.node = build_node(node, out_dim, arity, gate_eval=gate_eval,
                               tau=tau, degree=degree, gate_select=gate_select,
                               gumbel_tau=gumbel_tau)
        if residual_p > 0:
            self.node.residual_init(residual_p)

    # -- compatibility accessors (raise AttributeError when not applicable) --
    @property
    def gate_logits(self):
        return self.node.gate_logits

    @property
    def theta(self):
        return self.node.theta

    @property
    def conn_a(self):
        return self._conn.conn[:, 0]

    @property
    def conn_b(self):
        return self._conn.conn[:, 1]

    # -- fused Triton fast path --------------------------------------------
    def _fused_ok(self, x):
        if not (x.is_cuda and self.use_triton):
            return False
        topk = isinstance(self._conn, TopKConnectome)
        if self._node_name == "gate16":
            return topk and self.arity == 2 and kernels.HAS_DENSE
        if self._node_name == "walsh":
            if not (topk and self.arity == 2):
                return False
            gumbel = self.node.gate_select == "gumbel" and self.training
            return kernels.HAS_WARP and not gumbel
        if self._node_name == "hybrid":
            # candidate-based connectomes only (topk/blocktopk + fixed as K=1)
            return (kernels.HAS_MULTILINEAR
                    and isinstance(self._conn, (TopKConnectome, FixedConnectome)))
        return False

    def _fused(self, x):
        w = self._conn.weights()                       # [out, arity, k]
        ca = self._conn.candidates_i32()               # [out, arity, k]
        if self._node_name == "hybrid":               # n-input multilinear LUT (DWN-STE)
            s = torch.sigmoid(self.node.logits)        # [out, 2^arity]
            return kernels.multilinear_logic(x, ca.contiguous(), w.contiguous(),
                                             s, self.arity)
        wa, wb = w[:, 0].contiguous(), w[:, 1].contiguous()
        caa, cbb = ca[:, 0].contiguous(), ca[:, 1].contiguous()
        coef = self.node.coef().contiguous()
        if self._node_name == "gate16":
            return kernels.dense_logic(x, caa, cbb, wa, wb, coef)
        return kernels.warp_logic(x, caa, cbb, wa, wb, coef, self.node.inv_tau)

    def forward(self, x):
        if self._fused_ok(x):
            return self._fused(x)
        operands = self._conn.select_soft(x)           # [B, out, arity]
        return self.node(operands)

    @torch.no_grad()
    def forward_hard(self, x):
        x = x.to(torch.uint8)
        operands = self._conn.select_hard(x)           # [B, out, arity] in {0,1}
        return self.node.forward_hard(operands).to(torch.uint8)


# ---------------------------------------------------------------------------
# Named presets — thin wrappers that set node/connectome defaults and expose
# the family-specific attribute names the rest of the code (and tests) use.
# ---------------------------------------------------------------------------
class WARPLayer(LogicLayer):
    """2-input Walsh (WARP) node with Top-K connectivity (preset)."""

    def __init__(self, in_dim, out_dim, k=8, tau=1.0, residual_p=0.0, seed=None,
                 gate_select="softmax"):
        super().__init__(in_dim, out_dim, node="walsh", connectome="topk", arity=2,
                         k=k, tau=tau, residual_p=residual_p, seed=seed,
                         gate_select=gate_select)

    @property
    def theta(self):
        return self.node.theta

    def _z(self, a, b):
        return self.node._z(a, b)

    def _coef(self):
        return self.node.coef()


class WARPLayerN(LogicLayer):
    """Arity-``n`` Walsh (WARP) node with Top-K connectivity (preset)."""

    def __init__(self, in_dim, out_dim, arity=6, k=8, tau=1.0, residual_p=0.0,
                 seed=None, gate_select="softmax"):
        super().__init__(in_dim, out_dim, node="walsh", connectome="topk", arity=arity,
                         k=k, tau=tau, residual_p=residual_p, seed=seed,
                         gate_select=gate_select)

    @property
    def theta(self):
        return self.node.theta


class LUTkLayer(LogicLayer):
    """k-input multilinear LUT node (one FPGA LUT_k) with Top-K input selection."""

    def __init__(self, in_dim, out_dim, k=4, learn_conn=True, cand_k=4, seed=None):
        conn = "topk" if learn_conn else "fixed"
        super().__init__(in_dim, out_dim, node="multilinear", connectome=conn,
                         arity=k, k=cand_k, seed=seed)
        self.k = k                                     # LUT inputs (override fan-in)
        self.learn_conn = learn_conn

    @property
    def lut(self):
        return self.node.logits

    @property
    def conn(self):
        return self._conn.conn

    @property
    def cand(self):
        return self._conn.cand


class LUTNodeLayer(LogicLayer):
    """n-input LUT node with a selectable differentiable relaxation (preset)."""

    def __init__(self, in_dim, out_dim, arity=4, relaxation="probabilistic", k=8,
                 degree=None, residual_p=0.0, seed=None):
        super().__init__(in_dim, out_dim, node=relaxation, connectome="topk",
                         arity=arity, k=k, degree=degree, residual_p=residual_p,
                         seed=seed)
        self.relaxation = relaxation
        self.n = arity
        self.k = self._conn.k

    @property
    def conn(self):
        return self._conn.conn

    @property
    def cand(self):
        return self._conn.cand


# ---------------------------------------------------------------------------
# Convolutional logic layer: the spatial analogue of LogicLayer. Each output
# channel is a gate *tree* over a receptive field (Petersen et al.,
# arXiv:2411.04732) wired with LILogic Top-K leaf selection; gates use the same
# BasisProj evaluation and fused Triton kernels as the FC gate16 node.
# ---------------------------------------------------------------------------
class ConvLogicTree(nn.Module):
    """Logic-gate-tree convolution layer (Petersen-style trees, LILogic wiring).

    Each output channel is a complete binary tree of depth ``tree_depth`` with
    ``2**tree_depth`` leaves drawn from the receptive field and
    ``2**tree_depth - 1`` learnable gates, shared across all spatial placements.

    Args:
        in_channels (int): Number of input channels ``C``.
        out_channels (int): Number of output channels ``n`` (one tree each).
        kernel (int): Square kernel side length (``kernel x kernel`` receptive
            field). Default ``3``.
        tree_depth (int, optional): Depth ``d`` of each node tree (a
            ``node``-arity tree: ``node_arity**d`` leaves). ``None`` (default)
            uses ``2`` for the 2-input gate families and ``1`` for the n-input
            LUT families.
        stride (int): Convolution stride. Default ``1``.
        padding (int): Zero-padding on each side. Default ``1``.
        connect (str): Leaf wiring mode. ``"topk"`` (default) uses learnable
            Top-K connectivity (softmax over ``k`` candidates per leaf);
            ``"fixed"`` uses one fixed random leaf index per position (Petersen
            baseline). Both have fused Triton kernels on CUDA (``tree_conv`` for
            fixed, ``tree_conv_topk`` for Top-K).
        k (int): Top-K candidate pool size per leaf when ``connect="topk"``
            (clamped to the per-tree pool ``n_chan * kernel * kernel``).
            Unused for ``"fixed"``. Default ``4``.
        n_chan (int): Number of distinct input channels each tree may observe;
            leaves are drawn only from these channels' receptive-field positions
            (paper uses ``2``). Default ``2``.
        residual_init (bool): If ``True`` (default), bias every gate toward the
            ``"A"`` pass-through gate (~90%) for residual initialization; else
            initialize gate logits with small Gaussian noise.
        seed (int): RNG seed for the random channel/leaf selection. Default ``0``.
        node (str): Node parameterization (see :mod:`silogic.nodes`). Every output
            channel is a ``node``-arity tree of depth ``tree_depth``: the 2-input
            families ``"gate16"`` (default) and ``"walsh"`` use 2-input gates; the
            n-input families ``"multilinear"``/``"hybrid"``/``"linear"``/
            ``"polynomial"`` use ``arity``-input LUT nodes (depth 1 == a single LUT
            per channel). ``"gate16"`` keeps the fused Triton kernels; the others
            use the PyTorch path.
        arity (int): Fan-in per node for the n-input node families (children per
            tree node); ignored by the 2-input families. Default ``4``.
        tau (float): Sigmoid temperature for ``node="walsh"``. Default ``1.0``.
        degree (int | None): Polynomial degree for ``node="polynomial"``.
    """
    def __init__(self, in_channels, out_channels, kernel=3, tree_depth=None,
                 stride=1, padding=1, connect="topk", k=4, n_chan=2,
                 residual_init=True, seed=0, node="gate16",
                 arity=4, tau=1.0, degree=None, gate_select="softmax",
                 gumbel_tau=1.0, use_triton=True):
        super().__init__()
        self.node_name = _NODE_ALIASES.get(str(node).lower(), node)
        self.gate_select = gate_select
        self.gumbel_tau = gumbel_tau
        # Per-node fan-in: 2 for the gate families, `arity` for the n-input LUTs.
        # Each output channel is a `node_arity`-ary tree of depth `tree_depth`;
        # depth defaults to 2 for the 2-input gate families and 1 for the n-input
        # LUTs (depth 1 == a single node per channel over `node_arity` leaves).
        self.node_arity = 2 if self.node_name in ("gate16", "walsh") else arity
        if tree_depth is None:
            tree_depth = 2 if self.node_arity == 2 else 1
        self.cin = in_channels
        self.n = out_channels
        self.kh = self.kw = kernel
        self.d = tree_depth
        self.leaves = self.node_arity ** tree_depth
        self.stride = stride
        self.padding = padding
        self.connect = connect
        self.k = k
        self.tau = tau
        self.use_triton = use_triton    # master fused-kernel toggle (else PyTorch)
        gen = torch.Generator().manual_seed(seed)

        P = in_channels * kernel * kernel        # receptive-field size
        # Restrict each tree to observe only n_chan random input channels
        # (paper: trees see 2 channels). Candidate pool per leaf is drawn
        # from those channels' positions in the receptive field.
        cand_pool = []  # [n, pool] candidate flat indices into [0,P)
        pool_sz = n_chan * kernel * kernel
        for _ in range(out_channels):
            chans = torch.randperm(in_channels, generator=gen)[:n_chan]
            idxs = []
            for c in chans:
                base = int(c) * kernel * kernel
                idxs.extend(range(base, base + kernel * kernel))
            cand_pool.append(torch.tensor(idxs))
        cand_pool = torch.stack(cand_pool)        # [n, pool_sz]

        if connect == "fixed":
            # one fixed leaf index per (channel, leaf)
            sel = torch.stack([
                cand_pool[i][torch.randint(0, pool_sz, (self.leaves,),
                                           generator=gen)]
                for i in range(out_channels)])     # [n, leaves]
            self.register_buffer("leaf_idx", sel)
            # decompose flat receptive-field index -> (channel, dy, dx) for Triton
            kk2 = kernel * kernel
            self.register_buffer("cm_idx", (sel // kk2).to(torch.int32))
            rem = sel % kk2
            self.register_buffer("ch_idx", (rem // kernel).to(torch.int32))
            self.register_buffer("cw_idx", (rem % kernel).to(torch.int32))
        else:  # topk learnable
            kk = min(k, pool_sz)
            self.k = kk
            cand = torch.stack([
                torch.stack([cand_pool[i][torch.randperm(pool_sz, generator=gen)[:kk]]
                             for _ in range(self.leaves)])
                for i in range(out_channels)])     # [n, leaves, k]
            self.register_buffer("leaf_cand", cand)
            self.conn = nn.Parameter(torch.randn(out_channels, self.leaves, kk))
            # decompose candidate flat indices -> (channel, dy, dx) for the
            # fused Top-K Triton kernel (same layout as the fixed path)
            kk2 = kernel * kernel
            self.register_buffer("lc_cm", (cand // kk2).to(torch.int32))
            rem = cand % kk2
            self.register_buffer("lc_ch", (rem // kernel).to(torch.int32))
            self.register_buffer("lc_cw", (rem % kernel).to(torch.int32))

        # --- node parameterization --------------------------------------
        if self.node_name == "gate16":
            # Gate logits per tree level (level i combines 2^(d-i) -> 2^(d-i-1)).
            # Kept as a flat ParameterList (not a nodes.Node) to preserve the
            # fused-kernel coefficient layout and checkpoint compatibility.
            self.gate_logits = nn.ParameterList()
            for i in range(tree_depth):
                nodes = 2 ** (tree_depth - 1 - i)
                w = torch.zeros(out_channels, nodes, 16)
                if residual_init:
                    w[:, :, GATE_A] = 5.0          # ~90% 'A'
                else:
                    w.normal_(0, 0.1)
                self.gate_logits.append(nn.Parameter(w))
            self.register_buffer("basis", BASIS_COEFFS.clone())
            self.register_buffer("truth", TRUTH_TABLES.clone())
        else:
            # generic `node_arity`-ary tree: one node module per level, sized
            # out_dim = n * (nodes at that level), evaluated by the node's own
            # relaxation (no fused kernel). Depth 1 == a single node per channel.
            self.tree_nodes = nn.ModuleList()
            for i in range(tree_depth):
                m = self.node_arity ** (tree_depth - 1 - i)
                nd = build_node(self.node_name, out_channels * m,
                                arity=self.node_arity, tau=tau, degree=degree,
                                gate_select=gate_select, gumbel_tau=gumbel_tau)
                if residual_init:
                    nd.residual_init(0.9)
                self.tree_nodes.append(nd)

    def _gather_leaves(self, patches):
        """patches [B, P, L] -> leaves [B, n, leaves, L]."""
        B, P, L = patches.shape
        if self.connect == "fixed":
            idx = self.leaf_idx.reshape(-1)                  # [n*leaves]
            g = patches[:, idx, :]                            # [B, n*leaves, L]
            return g.view(B, self.n, self.leaves, L)
        else:
            w = F.softmax(self.conn, dim=2)                  # [n, leaves, k]
            idx = self.leaf_cand.reshape(-1)                 # [n*leaves*k]
            g = patches[:, idx, :].view(B, self.n, self.leaves, self.k, L)
            return torch.einsum("bnlkL,nlk->bnlL", g, w)

    def _tree_soft(self, leaves):
        """leaves [B, n, leaves, L] -> [B, n, L] via BasisProj per level."""
        vals = leaves
        for i, gl in enumerate(self.gate_logits):
            p = gate_probs(gl, self.training, dim=2, gate_select=self.gate_select,
                           gumbel_tau=self.gumbel_tau)        # [n, nodes, 16]
            c = p @ self.basis                               # [n, nodes, 4]
            a = vals[:, :, 0::2, :]
            b = vals[:, :, 1::2, :]
            c = c.unsqueeze(0).unsqueeze(-1)                 # [1,n,nodes,4,1]
            node = (c[:, :, :, 0] + c[:, :, :, 1] * a +
                    c[:, :, :, 2] * b + c[:, :, :, 3] * (a * b))
            vals = node
        return vals[:, :, 0, :]

    def _hw_out(self, H, W):
        Ho = (H + 2 * self.padding - self.kh) // self.stride + 1
        Wo = (W + 2 * self.padding - self.kw) // self.stride + 1
        return Ho, Wo

    def _basis_coef(self):
        """Concatenate per-level gate logits -> BasisProj coeffs [n, nodes, 4].

        Uses Gumbel straight-through gate selection when enabled+training
        (hard gate in forward, soft gradient), else plain softmax.
        """
        gl = torch.cat([g for g in self.gate_logits], dim=1)  # [n, nodes, 16]
        p = gate_probs(gl, self.training, dim=2, gate_select=self.gate_select,
                       gumbel_tau=self.gumbel_tau)
        return p @ self.basis                                 # [n, nodes, 4]

    @property
    def gates_per_output(self):
        """Logic nodes per output element (hardware cost): a ``node_arity``-ary
        tree of depth ``d`` has ``(node_arity**d - 1) / (node_arity - 1)`` nodes
        (``2**d - 1`` for the binary gate families; ``1`` for a depth-1 LUT)."""
        k = self.node_arity
        return (k ** self.d - 1) // (k - 1)

    def _eval_soft(self, leaves):
        if self.node_name == "gate16":
            return self._tree_soft(leaves)            # BasisProj gate tree
        return self._node_tree_eval(leaves, hard=False)

    def _node_tree_eval(self, leaves, hard):
        """Evaluate a ``node_arity``-ary tree of ``nodes.Node`` modules by folding
        spatial positions into the batch so each level's node applies. Depth 1 is
        a single node per output channel over its ``node_arity`` leaves."""
        k = self.node_arity
        vals = leaves                                          # [B, n, k^d, L]
        B = vals.shape[0]
        for nd in self.tree_nodes:
            n, km, L = vals.shape[1], vals.shape[2], vals.shape[3]
            m = km // k
            operands = vals.view(B, n, m, k, L)               # group k children
            operands = operands.permute(0, 4, 1, 2, 3).reshape(B * L, n * m, k)
            out = nd.forward_hard(operands) if hard else nd(operands)  # [B*L, n*m]
            vals = out.view(B, L, n, m).permute(0, 2, 3, 1)   # [B, n, m, L]
        return vals[:, :, 0, :]                                # [B, n, L]

    def forward(self, x):
        B, C, H, W = x.shape
        Ho, Wo = self._hw_out(H, W)
        # fused Triton kernels: gate16 BasisProj tree (reads the image directly)
        if self.node_name == "gate16" and x.is_cuda and self.use_triton:
            if self.connect == "fixed" and kernels.HAS_CONV:
                coef = self._basis_coef().contiguous()
                return kernels.tree_conv(x, coef, self.cm_idx, self.ch_idx,
                                         self.cw_idx, self.d, self.stride,
                                         self.padding, Ho, Wo)
            if self.connect == "topk" and kernels.HAS_CONV_TOPK:
                coef = self._basis_coef().contiguous()
                w = F.softmax(self.conn, dim=2).contiguous()
                return kernels.tree_conv_topk(x, coef, w, self.lc_cm, self.lc_ch,
                                              self.lc_cw, self.d, self.k, self.stride,
                                              self.padding, Ho, Wo)
        # fused hybrid LUT-tree, reading the image directly (any depth, no unfold;
        # eliminates the [B, n, 2^arity, L] corner tensor). Works for topk + fixed.
        if (self.node_name == "hybrid" and x.is_cuda and self.use_triton
                and kernels.HAS_CONV_HYBRID):
            s = self._hybrid_s()                              # [n, total_nodes, 2^arity]
            if self.connect == "fixed":
                cm, ch, cw = (self.cm_idx.unsqueeze(-1), self.ch_idx.unsqueeze(-1),
                              self.cw_idx.unsqueeze(-1))       # [n, leaves, 1]
                w = torch.ones(self.n, self.leaves, 1, device=x.device)
                Kc = 1
            else:
                cm, ch, cw = self.lc_cm, self.lc_ch, self.lc_cw  # [n, leaves, K]
                w = F.softmax(self.conn, dim=2).contiguous()
                Kc = self.k
            return kernels.conv_hybrid(x, s, w, cm.contiguous(), ch.contiguous(),
                                       cw.contiguous(), self.d, self.node_arity, Kc,
                                       self.stride, self.padding, Ho, Wo)
        patches = F.unfold(x, (self.kh, self.kw), stride=self.stride,
                           padding=self.padding)             # [B, C*kh*kw, L]
        leaves = self._gather_leaves(patches)
        out = self._eval_soft(leaves)                        # [B, n, L]
        return out.view(B, self.n, Ho, Wo)

    def _hybrid_s(self):
        """Per-node LUT entries ``sigmoid(logits)`` -> ``[n, total_nodes, 2^arity]``,
        concatenated level 0 .. depth-1 (matching the kernel's node ids)."""
        return torch.cat([torch.sigmoid(nd.logits).view(self.n, -1, 2 ** self.node_arity)
                          for nd in self.tree_nodes], dim=1).contiguous()

    def _gather_leaves_hard(self, patches):
        """Hard (argmax) leaf selection -> [B, n, leaves, L] of bits."""
        B = patches.shape[0]
        if self.connect == "fixed":
            idx = self.leaf_idx.reshape(-1)
            return patches[:, idx, :].view(B, self.n, self.leaves, -1)
        sel = self.conn.argmax(dim=2)                        # [n, leaves]
        idx = torch.gather(self.leaf_cand, 2, sel.unsqueeze(-1)).squeeze(-1)
        return patches[:, idx.reshape(-1), :].view(B, self.n, self.leaves, -1)

    def _tree_hard(self, leaves):
        """Hard gate16 tree: integer truth-table lookups per level."""
        vals = leaves
        B = vals.shape[0]
        for gl in self.gate_logits:
            gate = gl.argmax(dim=2)                          # [n, nodes]
            a = vals[:, :, 0::2, :]; b = vals[:, :, 1::2, :]
            t_idx = (a.long() << 1) | b.long()               # [B,n,nodes,L]
            tt = self.truth.to(leaves.device)[gate]          # [n, nodes, 4]
            tt = tt.unsqueeze(0).expand(B, -1, -1, -1)       # [B,n,nodes,4]
            node = torch.gather(tt, 3, t_idx)                # [B,n,nodes,L]
            vals = node
        return vals[:, :, 0, :]

    @torch.no_grad()
    def forward_hard(self, x):
        B, C, H, W = x.shape
        x = x.to(torch.uint8)
        Ho, Wo = self._hw_out(H, W)
        patches = F.unfold(x.float(), (self.kh, self.kw), stride=self.stride,
                           padding=self.padding).to(torch.uint8)
        leaves = self._gather_leaves_hard(patches)           # [B, n, leaves, L]
        if self.node_name == "gate16":
            out = self._tree_hard(leaves)
        else:
            out = self._node_tree_eval(leaves, hard=True)
        return out.view(B, self.n, Ho, Wo).to(torch.uint8)


# alias mirroring the FC `LogicLayer` naming
ConvLogicLayer = ConvLogicTree


class OrPool(nn.Module):
    """Logical OR pooling = max t-conorm = spatial max-pool.

    Args:
        size (int): Pooling window and stride. Default ``2`` (halves H and W).
    """
    def __init__(self, size=2):
        super().__init__()
        self.size = size

    def forward(self, x):
        return F.max_pool2d(x, self.size, self.size)

    @torch.no_grad()
    def forward_hard(self, x):
        return F.max_pool2d(x.float(), self.size, self.size).to(torch.uint8)
