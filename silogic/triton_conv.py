"""Triton kernels for the fixed-connectome logic-gate-tree convolution.

The PyTorch reference (`conv.py`) materialises `F.unfold` (replicates the input
kh*kw times) plus several [B, n, leaves, L] intermediates. These kernels read
directly from the input image via the per-leaf connection indices and evaluate
the whole BasisProj tree per output element — one fused pass, no unfold, no big
intermediates.

Triton's AST front-end does not support Python list mutation inside a kernel,
so instead of a generic depth loop we *code-generate* a fully-unrolled kernel
(scalar variables) for each tree depth and JIT-compile it once.

Only the FIXED connectome is fused (integer leaf indices). Gate behaviour is
passed as BasisProj coefficients `coef[n, num_nodes, 4]`, so the softmax over
the 16 gates stays in PyTorch and only grads w.r.t. `coef` and `x` are needed.
Node ordering matches `conv.py::_tree_structure` (level 0 first, then up).
"""
import functools
import importlib.util
import os
import tempfile
import torch
import triton
import triton.language as tl  # noqa: F401  (used inside generated source)

_KERNEL_DIR = os.path.join(tempfile.gettempdir(), "silogic_treeconv")
os.makedirs(_KERNEL_DIR, exist_ok=True)

# Launch tuning (set empirically; see sweep in test). BLOCK = output positions
# per program; warps = threads/program.
_BLOCK = 128
_WARPS = 4


def _tree_structure(depth):
    leaves = 2 ** depth
    refs = [("leaf", l) for l in range(leaves)]
    children = []
    nid = 0
    for _ in range(depth):
        new = []
        for m in range(0, len(refs), 2):
            children.append((refs[m], refs[m + 1]))
            new.append(("node", nid)); nid += 1
        refs = new
    return nid, children  # nid == 2**depth - 1


def _refname(ref):
    return (f"v{ref[1]}" if ref[0] == "leaf" else f"n{ref[1]}")


def _gen_src(depth, need_gx=True):
    """Code-generate batch-looped fwd/bwd kernels.

    One program == one (channel k, position-tile). The gate coefficients and
    leaf connection indices are loaded ONCE and reused across the whole batch
    (inner `for b` loop). In backward, grad_coef is accumulated in registers
    across the batch and flushed with a single atomic per node.
    """
    leaves = 2 ** depth
    num_nodes, children = _tree_structure(depth)
    root = num_nodes - 1
    NN = num_nodes

    # ---- forward ----
    f = [f"@triton.jit",
         f"def _fwd(x_ptr, coef_ptr, cm_ptr, ch_ptr, cw_ptr, out_ptr,",
         f"         B, C, H, W, N, Ho, Wo, stride, pad, BLOCK: tl.constexpr):",
         f"    pid = tl.program_id(0); npos = Ho*Wo; tiles=(npos+BLOCK-1)//BLOCK",
         f"    k = pid//tiles; tile = pid%tiles",
         f"    pos = tile*BLOCK + tl.arange(0,BLOCK); mpos = pos<npos",
         f"    i = pos//Wo; j = pos%Wo; CHW = C*H*W"]
    # connections + per-leaf channel-relative offset (no batch term): once
    for l in range(leaves):
        f += [
            f"    cm=tl.load(cm_ptr+k*{leaves}+{l}); ch=tl.load(ch_ptr+k*{leaves}+{l}); cw=tl.load(cw_ptr+k*{leaves}+{l})",
            f"    ii=i*stride-pad+ch; jj=j*stride-pad+cw",
            f"    vd{l}=mpos&(ii>=0)&(ii<H)&(jj>=0)&(jj<W)",
            f"    off{l}=(cm*H+ii)*W+jj"]
    # coef once
    for nid in range(num_nodes):
        base = f"(k*{NN}+{nid})*4"
        f += [f"    a{nid}0=tl.load(coef_ptr+{base}+0); a{nid}1=tl.load(coef_ptr+{base}+1); a{nid}2=tl.load(coef_ptr+{base}+2); a{nid}3=tl.load(coef_ptr+{base}+3)"]
    # software-pipelined batch loop (overlap next image's loads with compute)
    f += [f"    for b in tl.range(0, B, num_stages=2):", f"        xb=b*CHW"]
    for l in range(leaves):
        f += [f"        v{l}=tl.load(x_ptr+xb+off{l}, mask=vd{l}, other=0.0, eviction_policy='evict_first')"]
    for nid in range(num_nodes):
        a, bb = children[nid]; an, bn = _refname(a), _refname(bb)
        f += [f"        n{nid}=a{nid}0+a{nid}1*{an}+a{nid}2*{bn}+a{nid}3*{an}*{bn}"]
    f += [f"        tl.store(out_ptr+(b*N+k)*npos+pos, n{root}, mask=mpos)"]

    # ---- backward ---- (grad_x scatter per-b; grad_coef accumulated then 1 atomic)
    g = [f"@triton.jit",
         f"def _bwd(go_ptr, x_ptr, coef_ptr, cm_ptr, ch_ptr, cw_ptr, gx_ptr, gc_ptr,",
         f"         B, C, H, W, N, Ho, Wo, stride, pad, BLOCK: tl.constexpr):",
         f"    pid = tl.program_id(0); npos = Ho*Wo; tiles=(npos+BLOCK-1)//BLOCK",
         f"    k = pid//tiles; tile = pid%tiles",
         f"    pos = tile*BLOCK + tl.arange(0,BLOCK); mpos = pos<npos",
         f"    i = pos//Wo; j = pos%Wo; CHW = C*H*W"]
    for l in range(leaves):
        g += [
            f"    cm=tl.load(cm_ptr+k*{leaves}+{l}); ch=tl.load(ch_ptr+k*{leaves}+{l}); cw=tl.load(cw_ptr+k*{leaves}+{l})",
            f"    ii=i*stride-pad+ch; jj=j*stride-pad+cw",
            f"    vd{l}=mpos&(ii>=0)&(ii<H)&(jj>=0)&(jj<W)",
            f"    off{l}=(cm*H+ii)*W+jj"]
    for nid in range(num_nodes):
        base = f"(k*{NN}+{nid})*4"
        g += [f"    a{nid}0=tl.load(coef_ptr+{base}+0); a{nid}1=tl.load(coef_ptr+{base}+1); a{nid}2=tl.load(coef_ptr+{base}+2); a{nid}3=tl.load(coef_ptr+{base}+3)"]
    # grad_coef partial accumulators kept as per-position VECTORS (loop-carried);
    # the expensive cross-thread reduction is deferred to one tl.sum after the loop.
    for nid in range(num_nodes):
        g += [f"    gv{nid}0=tl.zeros([BLOCK],tl.float32); gv{nid}1=tl.zeros([BLOCK],tl.float32); gv{nid}2=tl.zeros([BLOCK],tl.float32); gv{nid}3=tl.zeros([BLOCK],tl.float32)"]
    # software-pipelined batch loop: overlap next image's loads with compute
    g += [f"    for b in tl.range(0, B, num_stages=2):", f"        xb=b*CHW"]
    for l in range(leaves):
        g += [f"        v{l}=tl.load(x_ptr+xb+off{l}, mask=vd{l}, other=0.0, eviction_policy='evict_first')"]
    for nid in range(num_nodes):
        a, bb = children[nid]; an, bn = _refname(a), _refname(bb)
        g += [f"        n{nid}=a{nid}0+a{nid}1*{an}+a{nid}2*{bn}+a{nid}3*{an}*{bn}"]
    g += [f"        U{root}=tl.load(go_ptr+(b*N+k)*npos+pos, mask=mpos, other=0.0, eviction_policy='evict_first')"]
    for nid in range(num_nodes - 1, -1, -1):
        a, bb = children[nid]; an, bn = _refname(a), _refname(bb)
        g += [
            f"        gv{nid}0=gv{nid}0+U{nid}",
            f"        gv{nid}1=gv{nid}1+U{nid}*{an}",
            f"        gv{nid}2=gv{nid}2+U{nid}*{bn}",
            f"        gv{nid}3=gv{nid}3+U{nid}*{an}*{bn}",
            f"        Ua=U{nid}*(a{nid}1+a{nid}3*{bn}); Ub=U{nid}*(a{nid}2+a{nid}3*{an})"]
        ua_name = (f"gxl{a[1]}" if a[0] == "leaf" else f"U{a[1]}")
        ub_name = (f"gxl{bb[1]}" if bb[0] == "leaf" else f"U{bb[1]}")
        if not need_gx and a[0] == "leaf":
            ua_name = "_"
        if not need_gx and bb[0] == "leaf":
            ub_name = "_"
        g += [f"        {ua_name}=Ua", f"        {ub_name}=Ub"]
    if need_gx:
        for l in range(leaves):
            g += [f"        tl.atomic_add(gx_ptr+xb+off{l}, gxl{l}, mask=vd{l}, sem='relaxed')"]
    # flush grad_coef: reduce the per-position partials once, one relaxed atomic per comp
    for nid in range(num_nodes):
        base = f"(k*{NN}+{nid})*4"
        g += [f"    tl.atomic_add(gc_ptr+{base}+0, tl.sum(gv{nid}0,axis=0), sem='relaxed'); tl.atomic_add(gc_ptr+{base}+1, tl.sum(gv{nid}1,axis=0), sem='relaxed'); tl.atomic_add(gc_ptr+{base}+2, tl.sum(gv{nid}2,axis=0), sem='relaxed'); tl.atomic_add(gc_ptr+{base}+3, tl.sum(gv{nid}3,axis=0), sem='relaxed')"]

    return "\n".join(f) + "\n\n" + "\n".join(g)


def _gen_src_flat(depth, need_gx=True, gleaf=False):
    """Flattened-indexing variant: tile over (batch x position) rows so every
    lane is active even when the spatial size is tiny (late blocks). One program
    == one (channel, row-tile); no batch loop.

    gleaf=True: pass-1 of the ATOMIC-FREE backward -- instead of scattering each
    leaf-gradient into grad_x (atomics), STORE it to gleaf[B,N,leaves,npos]
    (each slot written once). A separate gather kernel then builds grad_x."""
    leaves = 2 ** depth
    num_nodes, children = _tree_structure(depth)
    root = num_nodes - 1
    NN = num_nodes
    hdr = ["    pid = tl.program_id(0); npos = Ho*Wo; nrow = B*npos",
           "    tiles=(nrow+BLOCK-1)//BLOCK; k = pid//tiles; tile = pid%tiles",
           "    row = tile*BLOCK + tl.arange(0,BLOCK); m = row < nrow",
           "    b = row//npos; p = row%npos; i = p//Wo; j = p%Wo; CHW = C*H*W"]
    # ---- forward ----
    f = ["@triton.jit",
         "def _fwd(x_ptr, coef_ptr, cm_ptr, ch_ptr, cw_ptr, out_ptr,",
         "         B, C, H, W, N, Ho, Wo, stride, pad, BLOCK: tl.constexpr):"] + hdr
    for nid in range(num_nodes):
        base = f"(k*{NN}+{nid})*4"
        f += [f"    a{nid}0=tl.load(coef_ptr+{base}+0); a{nid}1=tl.load(coef_ptr+{base}+1); a{nid}2=tl.load(coef_ptr+{base}+2); a{nid}3=tl.load(coef_ptr+{base}+3)"]
    for l in range(leaves):
        f += [
            f"    cm=tl.load(cm_ptr+k*{leaves}+{l}); ch=tl.load(ch_ptr+k*{leaves}+{l}); cw=tl.load(cw_ptr+k*{leaves}+{l})",
            f"    ii=i*stride-pad+ch; jj=j*stride-pad+cw",
            f"    vd{l}=m&(ii>=0)&(ii<H)&(jj>=0)&(jj<W)",
            f"    v{l}=tl.load(x_ptr+b*CHW+(cm*H+ii)*W+jj, mask=vd{l}, other=0.0)"]
    for nid in range(num_nodes):
        a, bb = children[nid]; an, bn = _refname(a), _refname(bb)
        f += [f"    n{nid}=a{nid}0+a{nid}1*{an}+a{nid}2*{bn}+a{nid}3*{an}*{bn}"]
    f += [f"    tl.store(out_ptr+(b*N+k)*npos+p, n{root}, mask=m)"]
    # ---- backward ----
    g = ["@triton.jit",
         "def _bwd(go_ptr, x_ptr, coef_ptr, cm_ptr, ch_ptr, cw_ptr, gx_ptr, gc_ptr,",
         "         B, C, H, W, N, Ho, Wo, stride, pad, BLOCK: tl.constexpr):"] + hdr
    for nid in range(num_nodes):
        base = f"(k*{NN}+{nid})*4"
        g += [f"    a{nid}0=tl.load(coef_ptr+{base}+0); a{nid}1=tl.load(coef_ptr+{base}+1); a{nid}2=tl.load(coef_ptr+{base}+2); a{nid}3=tl.load(coef_ptr+{base}+3)"]
    for l in range(leaves):
        g += [
            f"    cm=tl.load(cm_ptr+k*{leaves}+{l}); ch=tl.load(ch_ptr+k*{leaves}+{l}); cw=tl.load(cw_ptr+k*{leaves}+{l})",
            f"    ii=i*stride-pad+ch; jj=j*stride-pad+cw",
            f"    vd{l}=m&(ii>=0)&(ii<H)&(jj>=0)&(jj<W)",
            f"    ad{l}=b*CHW+(cm*H+ii)*W+jj",
            f"    v{l}=tl.load(x_ptr+ad{l}, mask=vd{l}, other=0.0)"]
    for nid in range(num_nodes):
        a, bb = children[nid]; an, bn = _refname(a), _refname(bb)
        g += [f"    n{nid}=a{nid}0+a{nid}1*{an}+a{nid}2*{bn}+a{nid}3*{an}*{bn}"]
    g += [f"    U{root}=tl.load(go_ptr+(b*N+k)*npos+p, mask=m, other=0.0)"]
    for nid in range(num_nodes - 1, -1, -1):
        a, bb = children[nid]; an, bn = _refname(a), _refname(bb)
        base = f"(k*{NN}+{nid})*4"
        g += [
            f"    tl.atomic_add(gc_ptr+{base}+0, tl.sum(U{nid},axis=0), sem='relaxed')",
            f"    tl.atomic_add(gc_ptr+{base}+1, tl.sum(U{nid}*{an},axis=0), sem='relaxed')",
            f"    tl.atomic_add(gc_ptr+{base}+2, tl.sum(U{nid}*{bn},axis=0), sem='relaxed')",
            f"    tl.atomic_add(gc_ptr+{base}+3, tl.sum(U{nid}*{an}*{bn},axis=0), sem='relaxed')",
            f"    Ua=U{nid}*(a{nid}1+a{nid}3*{bn}); Ub=U{nid}*(a{nid}2+a{nid}3*{an})"]
        ua = (f"gxl{a[1]}" if a[0] == "leaf" else f"U{a[1]}")
        ub = (f"gxl{bb[1]}" if bb[0] == "leaf" else f"U{bb[1]}")
        need_leaf = need_gx or gleaf
        if not need_leaf and a[0] == "leaf":
            ua = "_"
        if not need_leaf and bb[0] == "leaf":
            ub = "_"
        g += [f"    {ua}=Ua", f"    {ub}=Ub"]
    if gleaf:
        # pass-1: store each leaf-gradient (no atomics). gx_ptr is gleaf[B,N,leaves,npos]
        for l in range(leaves):
            g += [f"    tl.store(gx_ptr+((b*N+k)*{leaves}+{l})*npos+p, gxl{l}, mask=m)"]
    elif need_gx:
        for l in range(leaves):
            g += [f"    tl.atomic_add(gx_ptr+ad{l}, gxl{l}, mask=vd{l}, sem='relaxed')"]
    return "\n".join(f) + "\n\n" + "\n".join(g)


@functools.lru_cache(maxsize=None)
def _compiled(depth, need_gx=True, mode="batch", gleaf=False):
    # write generated source to a real file so triton.jit / inspect can read it
    header = "import triton\nimport triton.language as tl\n\n"
    if gleaf:
        src = header + _gen_src_flat(depth, need_gx=True, gleaf=True)
    else:
        gen = _gen_src_flat if mode == "flat" else _gen_src
        src = header + gen(depth, need_gx=need_gx)
    suff = ("" if need_gx else "_nogx") + ("_flat" if mode == "flat" else "") \
        + ("_gleaf" if gleaf else "")
    path = os.path.join(_KERNEL_DIR, f"treeconv_d{depth}{suff}.py")
    with open(path, "w") as fh:
        fh.write(src)
    spec = importlib.util.spec_from_file_location(f"tc_d{depth}{suff}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._fwd, mod._bwd


def build_inverse_map(cm, ch, cw, C):
    """CSR inverse map for the atomic-free backward gather: for each input
    channel, the list of (out_channel n, leaf l, dy, dx) that read it.
    cm/ch/cw are [N, leaves] int. Returns (off[C+1], en, el, edy, edx) int32."""
    N, leaves = cm.shape
    cm = cm.cpu(); ch = ch.cpu(); cw = cw.cpu()
    buckets = [[] for _ in range(C)]
    for n in range(N):
        for l in range(leaves):
            buckets[int(cm[n, l])].append((n, l, int(ch[n, l]), int(cw[n, l])))
    off = torch.zeros(C + 1, dtype=torch.int32)
    en, el, edy, edx = [], [], [], []
    for c in range(C):
        for (n, l, dy, dx) in buckets[c]:
            en.append(n); el.append(l); edy.append(dy); edx.append(dx)
        off[c + 1] = len(en)
    t = lambda a: torch.tensor(a, dtype=torch.int32)
    return off, t(en), t(el), t(edy), t(edx)


@triton.jit
def _gather_gx(gleaf_ptr, off_ptr, en_ptr, el_ptr, edy_ptr, edx_ptr, gx_ptr,
               B, C, H, W, N, Ho, Wo, pad, LEAVES: tl.constexpr,
               BLOCK: tl.constexpr):
    """Pass-2 (atomic-free): grad_x[b,c,iy,ix] = sum over gates reading it of
    gleaf. Each (b,c,pos) owned by one program -> no atomics. Assumes stride 1."""
    pid = tl.program_id(0)
    npos_in = H * W
    tiles = (npos_in + BLOCK - 1) // BLOCK
    bc = pid // tiles
    tile = pid % tiles
    b = bc // C
    c = bc % C
    pos = tile * BLOCK + tl.arange(0, BLOCK)
    m = pos < npos_in
    iy = pos // W
    ix = pos % W
    npos_out = Ho * Wo
    start = tl.load(off_ptr + c)
    end = tl.load(off_ptr + c + 1)
    acc = tl.zeros([BLOCK], tl.float32)
    for e in range(start, end):
        n = tl.load(en_ptr + e)
        l = tl.load(el_ptr + e)
        dy = tl.load(edy_ptr + e)
        dx = tl.load(edx_ptr + e)
        oy = iy + pad - dy        # stride 1
        ox = ix + pad - dx
        valid = m & (oy >= 0) & (oy < Ho) & (ox >= 0) & (ox < Wo)
        gidx = ((b * N + n) * LEAVES + l) * npos_out + (oy * Wo + ox)
        acc += tl.load(gleaf_ptr + gidx, mask=valid, other=0.0)
    tl.store(gx_ptr + ((b * C + c) * H + iy) * W + ix, acc, mask=m)


class TreeConvAFFn(torch.autograd.Function):
    """Conv tree with ATOMIC-FREE backward (2-pass: store leaf-grads, gather)."""
    @staticmethod
    def forward(ctx, x, coef, cm, ch, cw, off, en, el, edy, edx,
                depth, stride, pad, Ho, Wo):
        B, C, H, W = x.shape
        N = coef.shape[0]
        npos = Ho * Wo
        fwd, _ = _compiled(depth, True, "flat")
        x = x.contiguous(); coef = coef.contiguous()
        out = torch.empty((B, N, npos), device=x.device, dtype=x.dtype)
        grid = (N * ((B * npos + _BLOCK - 1) // _BLOCK),)
        fwd[grid](x, coef, cm, ch, cw, out, B, C, H, W, N, Ho, Wo, stride, pad,
                  BLOCK=_BLOCK, num_warps=_WARPS)
        ctx.save_for_backward(x, coef, cm, ch, cw, off, en, el, edy, edx)
        ctx.meta = (depth, stride, pad, Ho, Wo)
        return out.view(B, N, Ho, Wo)

    @staticmethod
    def backward(ctx, grad_out):
        x, coef, cm, ch, cw, off, en, el, edy, edx = ctx.saved_tensors
        depth, stride, pad, Ho, Wo = ctx.meta
        B, C, H, W = x.shape
        N = coef.shape[0]
        leaves = 2 ** depth
        npos = Ho * Wo
        # pass 1: store leaf-grads to gleaf (no atomics) + grad_coef
        _, bwd = _compiled(depth, True, "flat", gleaf=True)
        gleaf = torch.empty((B, N, leaves, npos), device=x.device,
                            dtype=torch.float32)
        gc = torch.zeros_like(coef, dtype=torch.float32)
        grid = (N * ((B * npos + _BLOCK - 1) // _BLOCK),)
        bwd[grid](grad_out.contiguous(), x, coef, cm, ch, cw, gleaf, gc,
                  B, C, H, W, N, Ho, Wo, stride, pad, BLOCK=_BLOCK,
                  num_warps=_WARPS)
        # pass 2: atomic-free gather into grad_x
        gx = None
        if ctx.needs_input_grad[0]:
            gx = torch.empty((B, C, H, W), device=x.device, dtype=torch.float32)
            g2 = (B * C * ((H * W + _BLOCK - 1) // _BLOCK),)
            _gather_gx[g2](gleaf, off, en, el, edy, edx, gx, B, C, H, W, N,
                           Ho, Wo, pad, LEAVES=leaves, BLOCK=_BLOCK,
                           num_warps=_WARPS)
            gx = gx.to(x.dtype)
        return (gx, gc) + (None,) * 13


def tree_conv_af(x, coef, cm, ch, cw, inv, depth, stride, pad, Ho, Wo):
    off, en, el, edy, edx = inv
    return TreeConvAFFn.apply(x, coef, cm, ch, cw, off, en, el, edy, edx,
                              depth, stride, pad, Ho, Wo)


class TreeConvFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, coef, cm, ch, cw, depth, stride, pad, Ho, Wo):
        B, C, H, W = x.shape
        N = coef.shape[0]
        npos = Ho * Wo
        # flat indexing fills all lanes when spatial size < BLOCK (late blocks);
        # batch-loop keeps grad_coef atomics fewer for large spatial maps.
        mode = "flat" if npos <= 32 else "batch"
        fwd, _ = _compiled(depth, True, mode)
        x = x.contiguous(); coef = coef.contiguous()
        # bf16 packing: activations (x, out) in x.dtype (bf16 halves the gather
        # + store traffic); coef stays fp32; the kernel promotes in fp32.
        out = torch.empty((B, N, npos), device=x.device, dtype=x.dtype)
        nrow = (B * npos) if mode == "flat" else npos
        grid = (N * ((nrow + _BLOCK - 1) // _BLOCK),)
        fwd[grid](x, coef, cm, ch, cw, out, B, C, H, W, N, Ho, Wo, stride, pad,
                  BLOCK=_BLOCK, num_warps=_WARPS)
        ctx.save_for_backward(x, coef, cm, ch, cw)
        ctx.meta = (depth, stride, pad, Ho, Wo, mode)
        return out.view(B, N, Ho, Wo)

    @staticmethod
    def backward(ctx, grad_out):
        x, coef, cm, ch, cw = ctx.saved_tensors
        depth, stride, pad, Ho, Wo, mode = ctx.meta
        B, C, H, W = x.shape
        N = coef.shape[0]
        npos = Ho * Wo
        need_gx = ctx.needs_input_grad[0]   # skip grad_x scatter if input is data
        _, bwd = _compiled(depth, need_gx, mode)
        # grad_x / grad_coef accumulate in fp32 (atomics); cast grad_x back to
        # the activation dtype on return. grad_out loaded as-is (bf16 = half).
        gx = torch.zeros(x.shape, device=x.device, dtype=torch.float32) \
            if need_gx else x
        gc = torch.zeros_like(coef, dtype=torch.float32)
        nrow = (B * npos) if mode == "flat" else npos
        grid = (N * ((nrow + _BLOCK - 1) // _BLOCK),)
        bwd[grid](grad_out.contiguous(), x, coef, cm, ch, cw, gx, gc,
                  B, C, H, W, N, Ho, Wo, stride, pad, BLOCK=_BLOCK,
                  num_warps=_WARPS)
        return (gx.to(x.dtype) if need_gx else None), gc, None, None, None, \
            None, None, None, None, None


def tree_conv(x, coef, cm, ch, cw, depth, stride, pad, Ho, Wo):
    return TreeConvFn.apply(x, coef, cm, ch, cw, depth, stride, pad, Ho, Wo)
