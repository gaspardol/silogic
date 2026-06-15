"""Fused Triton kernel for the LEARNABLE Top-K logic-gate-tree convolution.

The sibling `triton_conv.py` only fuses the *fixed* connectome (one integer leaf
index per tree leaf). Here each leaf is a softmax-weighted blend of `K` candidate
receptive-field positions:

    leaf_l = sum_{kk<K}  w[n,l,kk] * x[ candidate(n,l,kk) ]

then the same BasisProj tree as the fixed kernel. The connection softmax stays in
PyTorch (the kernel takes the post-softmax weights `w`), so only grads w.r.t.
`coef`, `w`, and `x` are produced — autograd handles softmax(conn) -> w.

Like `triton_conv.py` we code-generate a fully-unrolled kernel per (depth, K) and
JIT-compile it once. Candidate positions are passed pre-decomposed into
(channel, dy, dx) = `cm/ch/cw`, each `[N, leaves, K]` int32.
"""
import functools
import importlib.util
import os
import tempfile
import torch
import triton
import triton.language as tl  # noqa: F401  (used inside generated source)

from .conv import _tree_structure, _refname, _BLOCK, _WARPS

_KERNEL_DIR = os.path.join(tempfile.gettempdir(), "silogic_treeconv_topk")
os.makedirs(_KERNEL_DIR, exist_ok=True)


def _gen_src(depth, K, need_gx=True):
    """Flat-indexing fwd/bwd kernels for a depth-`d`, `K`-candidate Top-K tree.

    One program == one (out-channel, row-tile) over the flattened B*Ho*Wo rows.
    Backward is two passes over the candidates: pass A recomputes the leaf values
    (for grad_coef), pass B reloads each candidate to scatter grad_w / grad_x —
    trading a re-gather for much lower register pressure than caching every
    candidate.
    """
    leaves = 2 ** depth
    num_nodes, children = _tree_structure(depth)
    root = num_nodes - 1
    NN = num_nodes
    hdr = ["    pid = tl.program_id(0); npos = Ho*Wo; nrow = B*npos",
           "    tiles=(nrow+BLOCK-1)//BLOCK; k = pid//tiles; tile = pid%tiles",
           "    row = tile*BLOCK + tl.arange(0,BLOCK); m = row < nrow",
           "    b = row//npos; p = row%npos; i = p//Wo; j = p%Wo; CHW = C*H*W"]

    def load_coef(buf):
        for nid in range(num_nodes):
            base = f"(k*{NN}+{nid})*4"
            buf.append(f"    a{nid}0=tl.load(coef_ptr+{base}+0); a{nid}1=tl.load(coef_ptr+{base}+1); "
                       f"a{nid}2=tl.load(coef_ptr+{base}+2); a{nid}3=tl.load(coef_ptr+{base}+3)")

    def accumulate_leaves(buf, indent):
        """v{l} = sum_kk w * x[cand]; emits gathers, no caching."""
        for l in range(leaves):
            buf.append(f"{indent}v{l}=tl.zeros([BLOCK],tl.float32)")
            for kk in range(K):
                base = f"(k*{leaves}+{l})*{K}+{kk}"
                buf += [
                    f"{indent}cm=tl.load(cm_ptr+{base}); ch=tl.load(ch_ptr+{base}); "
                    f"cw=tl.load(cw_ptr+{base}); wt=tl.load(w_ptr+{base})",
                    f"{indent}ii=i*stride-pad+ch; jj=j*stride-pad+cw",
                    f"{indent}vd=m&(ii>=0)&(ii<H)&(jj>=0)&(jj<W)",
                    f"{indent}xg=tl.load(x_ptr+b*CHW+(cm*H+ii)*W+jj, mask=vd, other=0.0)",
                    f"{indent}v{l}=v{l}+wt*xg"]

    def eval_tree(buf, indent):
        for nid in range(num_nodes):
            a, bb = children[nid]
            an, bn = _refname(a), _refname(bb)
            buf.append(f"{indent}n{nid}=a{nid}0+a{nid}1*{an}+a{nid}2*{bn}+a{nid}3*{an}*{bn}")

    # ---- forward ----
    f = ["@triton.jit",
         "def _fwd(x_ptr, coef_ptr, w_ptr, cm_ptr, ch_ptr, cw_ptr, out_ptr,",
         "         B, C, H, W, N, Ho, Wo, stride, pad, BLOCK: tl.constexpr):"] + hdr
    load_coef(f)
    accumulate_leaves(f, "    ")
    eval_tree(f, "    ")
    f.append(f"    tl.store(out_ptr+(b*N+k)*npos+p, n{root}, mask=m)")

    # ---- backward ----
    g = ["@triton.jit",
         "def _bwd(go_ptr, x_ptr, coef_ptr, w_ptr, cm_ptr, ch_ptr, cw_ptr,",
         "         gx_ptr, gc_ptr, gw_ptr,",
         "         B, C, H, W, N, Ho, Wo, stride, pad, BLOCK: tl.constexpr):"] + hdr
    load_coef(g)
    accumulate_leaves(g, "    ")          # pass A: leaf values
    eval_tree(g, "    ")
    g.append(f"    U{root}=tl.load(go_ptr+(b*N+k)*npos+p, mask=m, other=0.0)")
    # backprop through the tree: grad_coef (atomics) + per-leaf grads gxl{l}
    for nid in range(num_nodes - 1, -1, -1):
        a, bb = children[nid]
        an, bn = _refname(a), _refname(bb)
        base = f"(k*{NN}+{nid})*4"
        g += [
            f"    tl.atomic_add(gc_ptr+{base}+0, tl.sum(U{nid},axis=0), sem='relaxed')",
            f"    tl.atomic_add(gc_ptr+{base}+1, tl.sum(U{nid}*{an},axis=0), sem='relaxed')",
            f"    tl.atomic_add(gc_ptr+{base}+2, tl.sum(U{nid}*{bn},axis=0), sem='relaxed')",
            f"    tl.atomic_add(gc_ptr+{base}+3, tl.sum(U{nid}*{an}*{bn},axis=0), sem='relaxed')",
            f"    Ua=U{nid}*(a{nid}1+a{nid}3*{bn}); Ub=U{nid}*(a{nid}2+a{nid}3*{an})"]
        ua = (f"gxl{a[1]}" if a[0] == "leaf" else f"U{a[1]}")
        ub = (f"gxl{bb[1]}" if bb[0] == "leaf" else f"U{bb[1]}")
        g += [f"    {ua}=Ua", f"    {ub}=Ub"]
    # pass B: reload each candidate -> grad_w (always) + grad_x (optional)
    for l in range(leaves):
        for kk in range(K):
            base = f"(k*{leaves}+{l})*{K}+{kk}"
            g += [
                f"    cm=tl.load(cm_ptr+{base}); ch=tl.load(ch_ptr+{base}); "
                f"cw=tl.load(cw_ptr+{base}); wt=tl.load(w_ptr+{base})",
                f"    ii=i*stride-pad+ch; jj=j*stride-pad+cw",
                f"    vd=m&(ii>=0)&(ii<H)&(jj>=0)&(jj<W)",
                f"    ad=b*CHW+(cm*H+ii)*W+jj",
                f"    xg=tl.load(x_ptr+ad, mask=vd, other=0.0)",
                f"    tl.atomic_add(gw_ptr+{base}, tl.sum(gxl{l}*xg,axis=0), sem='relaxed')"]
            if need_gx:
                g += [f"    tl.atomic_add(gx_ptr+ad, gxl{l}*wt, mask=vd, sem='relaxed')"]

    return "\n".join(f) + "\n\n" + "\n".join(g)


@functools.lru_cache(maxsize=None)
def _compiled(depth, K, need_gx=True):
    header = "import triton\nimport triton.language as tl\n\n"
    src = header + _gen_src(depth, K, need_gx=need_gx)
    suff = "" if need_gx else "_nogx"
    path = os.path.join(_KERNEL_DIR, f"treeconv_topk_d{depth}_k{K}{suff}.py")
    with open(path, "w") as fh:
        fh.write(src)
    spec = importlib.util.spec_from_file_location(f"tctk_d{depth}_k{K}{suff}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._fwd, mod._bwd


class TreeConvTopkFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, coef, w, cm, ch, cw, depth, K, stride, pad, Ho, Wo):
        B, C, H, W = x.shape
        N = coef.shape[0]
        npos = Ho * Wo
        fwd, _ = _compiled(depth, K, True)
        x = x.contiguous(); coef = coef.contiguous(); w = w.contiguous()
        out = torch.empty((B, N, npos), device=x.device, dtype=x.dtype)
        grid = (N * ((B * npos + _BLOCK - 1) // _BLOCK),)
        fwd[grid](x, coef, w, cm, ch, cw, out, B, C, H, W, N, Ho, Wo, stride, pad,
                  BLOCK=_BLOCK, num_warps=_WARPS)
        ctx.save_for_backward(x, coef, w, cm, ch, cw)
        ctx.meta = (depth, K, stride, pad, Ho, Wo)
        return out.view(B, N, Ho, Wo)

    @staticmethod
    def backward(ctx, grad_out):
        x, coef, w, cm, ch, cw = ctx.saved_tensors
        depth, K, stride, pad, Ho, Wo = ctx.meta
        B, C, H, W = x.shape
        N = coef.shape[0]
        npos = Ho * Wo
        need_gx = ctx.needs_input_grad[0]
        _, bwd = _compiled(depth, K, need_gx)
        gx = torch.zeros(x.shape, device=x.device, dtype=torch.float32) \
            if need_gx else x
        gc = torch.zeros_like(coef, dtype=torch.float32)
        gw = torch.zeros_like(w, dtype=torch.float32)
        grid = (N * ((B * npos + _BLOCK - 1) // _BLOCK),)
        bwd[grid](grad_out.contiguous(), x, coef, w, cm, ch, cw, gx, gc, gw,
                  B, C, H, W, N, Ho, Wo, stride, pad, BLOCK=_BLOCK,
                  num_warps=_WARPS)
        return ((gx.to(x.dtype) if need_gx else None), gc, gw) + (None,) * 9


def tree_conv_topk(x, coef, w, cm, ch, cw, depth, K, stride, pad, Ho, Wo):
    """Fused learnable Top-K logic-gate-tree convolution.

    Args:
        x: input image ``[B, C, H, W]``.
        coef: BasisProj gate coefficients ``[N, 2**depth - 1, 4]``.
        w: per-leaf softmax connection weights ``[N, 2**depth, K]``.
        cm, ch, cw: candidate (channel, dy, dx) indices, each int32 ``[N, 2**depth, K]``.
        depth, K, stride, pad, Ho, Wo: tree depth, candidate count, conv params,
            and the output spatial size.

    Returns:
        ``[B, N, Ho, Wo]`` soft logic-tree activations.
    """
    return TreeConvTopkFn.apply(x, coef, w, cm, ch, cw, depth, K, stride, pad, Ho, Wo)
