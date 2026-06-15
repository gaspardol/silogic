"""Fused Triton kernel for the convolutional **hybrid** LUT tree (image-reading,
multi-depth, no ``F.unfold``).

Each output channel is a ``node_arity``-ary tree of depth ``d`` whose nodes are
hybrid (DWN-STE) multilinear LUTs. Like :mod:`conv_topk` it reads the input image
directly via the per-leaf ``(channel, dy, dx)`` candidate indices (with padding
bounds checks) — no unfold, no ``[B, n, 2^arity, L]`` corner tensor.

Per output (channel ``c``, position ``(oy, ox)``):

    leaf_l = sum_kk w[c,l,kk] * img[b, cand(c,l,kk)]         (Top-K weighted gather)
    node:  idx = sum_j (child_j >= 0.5) << j;  value = s[c, node, idx]   (= f_disc)
    grad flows as if each node were its soft multilinear f_soft (tree-structured STE)

The leaf softmax (``w = softmax(conn)``) and per-level LUT sigmoids
(``s = sigmoid(logits)``, concatenated across levels into ``[N, total_nodes,
2^arity]``) stay in PyTorch, so the kernel produces grads w.r.t. ``x``, ``w`` and
``s`` only. Code-generated per ``(depth, node_arity, K)``.
"""
import functools
import importlib.util
import os
import tempfile
import torch
import triton
import triton.language as tl  # noqa: F401  (used inside generated source)

from .conv import _BLOCK, _WARPS

_KERNEL_DIR = os.path.join(tempfile.gettempdir(), "silogic_conv_hybrid")
os.makedirs(_KERNEL_DIR, exist_ok=True)


def _hybrid_tree(depth, kn):
    """k-ary tree wiring matching ``ConvLogicTree._node_tree_eval``.

    Returns ``(nodes, total, root)`` where ``nodes`` is bottom-up order of
    ``(gid, [child refs])``; a child ref is ``("leaf", l)`` or ``("node", gid)``.
    Node global ids run level 0 first (so they line up with the per-level LUT
    logits concatenated level 0 .. d-1).
    """
    nodes, gid, prev = [], 0, None
    for i in range(depth):
        m_i = kn ** (depth - 1 - i)
        cur = []
        for m in range(m_i):
            refs = [("leaf", m * kn + c) if i == 0 else ("node", prev[m * kn + c])
                    for c in range(kn)]
            nodes.append((gid, refs))
            cur.append(gid)
            gid += 1
        prev = cur
    return nodes, gid, nodes[-1][0]


def _gen_src(depth, kn, K, need_gx=True):
    nodes, total, root = _hybrid_tree(depth, kn)
    leaves = kn ** depth
    P = 2 ** kn

    def rn(ref):
        return f"v{ref[1]}" if ref[0] == "leaf" else f"n{ref[1]}"

    hdr = ["    pid = tl.program_id(0); npos = Ho*Wo; nrow = B*npos",
           "    tiles=(nrow+BLOCK-1)//BLOCK; k = pid//tiles; tile = pid%tiles",
           "    row = tile*BLOCK + tl.arange(0,BLOCK); m = row < nrow",
           "    b = row//npos; p = row%npos; i = p//Wo; j = p%Wo; CHW = C*H*W"]

    def accumulate_leaves(buf):
        for l in range(leaves):
            buf.append(f"    v{l}=tl.zeros([BLOCK],tl.float32)")
            for kk in range(K):
                base = f"(k*{leaves}+{l})*{K}+{kk}"
                buf += [
                    f"    cm=tl.load(cm_ptr+{base}); ch=tl.load(ch_ptr+{base}); "
                    f"cw=tl.load(cw_ptr+{base}); wt=tl.load(w_ptr+{base})",
                    "    ii=i*stride-pad+ch; jj=j*stride-pad+cw",
                    "    vd=m&(ii>=0)&(ii<H)&(jj>=0)&(jj<W)",
                    "    xg=tl.load(x_ptr+b*CHW+(cm*H+ii)*W+jj, mask=vd, other=0.0)",
                    f"    v{l}=v{l}+wt*xg"]

    def eval_tree(buf):
        for gid, refs in nodes:
            buf.append("    idx=tl.zeros([BLOCK],tl.int32)")
            for c, ref in enumerate(refs):
                buf.append(f"    idx += tl.where({rn(ref)} >= 0.5, {1 << c}, 0)")
            buf.append(f"    n{gid}=tl.load(s_ptr+(k*{total}+{gid})*{P}+idx, "
                       "mask=m, other=0.0)")

    # ---- forward ----
    f = ["@triton.jit",
         "def _fwd(x_ptr, s_ptr, w_ptr, cm_ptr, ch_ptr, cw_ptr, out_ptr,",
         "         B, C, H, W, N, Ho, Wo, stride, pad, BLOCK: tl.constexpr):"] + hdr
    accumulate_leaves(f)
    eval_tree(f)
    f.append(f"    tl.store(out_ptr+(b*N+k)*npos+p, n{root}, mask=m)")

    # ---- backward ----
    g = ["@triton.jit",
         "def _bwd(go_ptr, x_ptr, s_ptr, w_ptr, cm_ptr, ch_ptr, cw_ptr,",
         "         gx_ptr, gs_ptr, gw_ptr,",
         "         B, C, H, W, N, Ho, Wo, stride, pad, BLOCK: tl.constexpr):"] + hdr
    accumulate_leaves(g)
    eval_tree(g)                                    # recompute node values
    for gid, _ in nodes:                            # load every LUT entry per node
        for pi in range(P):
            g.append(f"    s{gid}_{pi}=tl.load(s_ptr+(k*{total}+{gid})*{P}+{pi})")
    g.append(f"    U{root}=tl.load(go_ptr+(b*N+k)*npos+p, mask=m, other=0.0)")
    for gid, refs in reversed(nodes):
        for c, ref in enumerate(refs):
            g.append(f"    om{gid}_{c}=1.0-{rn(ref)}")
        # grad w.r.t. this node's LUT entries: df_soft/ds_P = M_P(children)
        for pi in range(P):
            mp = "*".join(rn(refs[c]) if (pi >> c) & 1 else f"om{gid}_{c}"
                          for c in range(kn))
            g.append(f"    tl.atomic_add(gs_ptr+(k*{total}+{gid})*{P}+{pi}, "
                     f"tl.sum(U{gid}*({mp}),axis=0), sem='relaxed')")
        # grad w.r.t. each child = U * df_soft/dchild (multilinear partial)
        for c, ref in enumerate(refs):
            terms = []
            for pi in range(P):
                sign = "" if (pi >> c) & 1 else "-"
                excl = [rn(refs[cc]) if (pi >> cc) & 1 else f"om{gid}_{cc}"
                        for cc in range(kn) if cc != c]
                mexcl = "*".join(excl) if excl else "1.0"
                terms.append(f"{sign}s{gid}_{pi}*{mexcl}")
            tgt = f"gxl{ref[1]}" if ref[0] == "leaf" else f"U{ref[1]}"
            g.append(f"    {tgt}=U{gid}*(" + " + ".join(terms) + ")")
    # pass B: reload each candidate -> grad_w (+ grad_x)
    for l in range(leaves):
        for kk in range(K):
            base = f"(k*{leaves}+{l})*{K}+{kk}"
            g += [
                f"    cm=tl.load(cm_ptr+{base}); ch=tl.load(ch_ptr+{base}); "
                f"cw=tl.load(cw_ptr+{base}); wt=tl.load(w_ptr+{base})",
                "    ii=i*stride-pad+ch; jj=j*stride-pad+cw",
                "    vd=m&(ii>=0)&(ii<H)&(jj>=0)&(jj<W)",
                "    ad=b*CHW+(cm*H+ii)*W+jj",
                "    xg=tl.load(x_ptr+ad, mask=vd, other=0.0)",
                f"    tl.atomic_add(gw_ptr+{base}, tl.sum(gxl{l}*xg,axis=0), sem='relaxed')"]
            if need_gx:
                g.append(f"    tl.atomic_add(gx_ptr+ad, gxl{l}*wt, mask=vd, sem='relaxed')")

    return "\n".join(f) + "\n\n" + "\n".join(g)


@functools.lru_cache(maxsize=None)
def _compiled(depth, kn, K, need_gx=True):
    header = "import triton\nimport triton.language as tl\n\n"
    src = header + _gen_src(depth, kn, K, need_gx=need_gx)
    suff = "" if need_gx else "_nogx"
    path = os.path.join(_KERNEL_DIR, f"convhybrid_d{depth}_a{kn}_k{K}{suff}.py")
    with open(path, "w") as fh:
        fh.write(src)
    spec = importlib.util.spec_from_file_location(f"ch_d{depth}_a{kn}_k{K}{suff}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._fwd, mod._bwd


class ConvHybridFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, s, w, cm, ch, cw, depth, node_arity, K, stride, pad, Ho, Wo):
        B, C, H, W = x.shape
        N = s.shape[0]
        npos = Ho * Wo
        fwd, _ = _compiled(depth, node_arity, K, True)
        x = x.contiguous(); s = s.contiguous(); w = w.contiguous()
        out = torch.empty((B, N, npos), device=x.device, dtype=x.dtype)
        grid = (N * ((B * npos + _BLOCK - 1) // _BLOCK),)
        fwd[grid](x, s, w, cm, ch, cw, out, B, C, H, W, N, Ho, Wo, stride, pad,
                  BLOCK=_BLOCK, num_warps=_WARPS)
        ctx.save_for_backward(x, s, w, cm, ch, cw)
        ctx.meta = (depth, node_arity, K, stride, pad, Ho, Wo)
        return out.view(B, N, Ho, Wo)

    @staticmethod
    def backward(ctx, grad_out):
        x, s, w, cm, ch, cw = ctx.saved_tensors
        depth, node_arity, K, stride, pad, Ho, Wo = ctx.meta
        B, C, H, W = x.shape
        N = s.shape[0]
        npos = Ho * Wo
        need_gx = ctx.needs_input_grad[0]
        _, bwd = _compiled(depth, node_arity, K, need_gx)
        gx = torch.zeros(x.shape, device=x.device, dtype=torch.float32) \
            if need_gx else x
        gs = torch.zeros_like(s, dtype=torch.float32)
        gw = torch.zeros_like(w, dtype=torch.float32)
        grid = (N * ((B * npos + _BLOCK - 1) // _BLOCK),)
        bwd[grid](grad_out.contiguous(), x, s, w, cm, ch, cw, gx, gs, gw,
                  B, C, H, W, N, Ho, Wo, stride, pad, BLOCK=_BLOCK, num_warps=_WARPS)
        return ((gx.to(x.dtype) if need_gx else None), gs, gw) + (None,) * 10


def conv_hybrid(x, s, w, cm, ch, cw, depth, node_arity, K, stride, pad, Ho, Wo):
    """Fused convolutional hybrid LUT-tree (image-reading, any depth).

    Args:
        x: input image ``[B, C, H, W]``.
        s: per-node LUT entries ``[N, total_nodes, 2**node_arity]`` =
            ``sigmoid(logits)`` concatenated level 0 .. depth-1.
        w: per-leaf softmax weights ``[N, node_arity**depth, K]`` (K=1, ones for fixed).
        cm, ch, cw: candidate (channel, dy, dx) indices, int32 ``[N, leaves, K]``.
        depth, node_arity, K, stride, pad, Ho, Wo: tree depth, node fan-in,
            candidates/leaf, conv params, output spatial size.

    Returns:
        ``[B, N, Ho, Wo]`` discrete hybrid-tree activations carrying the soft
        multilinear surrogate's gradient.
    """
    return ConvHybridFn.apply(x, s, w, cm, ch, cw, depth, node_arity, K,
                              stride, pad, Ho, Wo)
