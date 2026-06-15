"""Fused Triton kernel for the dense logic layer (head).

The PyTorch path materialises a [B, out, k] gather (~1 GB for a 32k-wide head)
plus softmax/einsum in eager fp32 — ~38% of a big model's step time. This
kernel reads directly from the input with the per-gate candidate indices and
fuses weighted input selection (a = sum_k wa_k * x[cand_a_k]) + BasisProj gate
eval in one pass.

The connection softmax (wa = softmax(conn_a)) and gate selection (coef =
gate_probs @ basis) stay in PyTorch, so their gradients flow normally; the
kernel only needs grads w.r.t. x, wa, wb, coef. Top-K connectome, small k.

Triton's front-end disallows Python lists in-kernel, so the k-unrolled kernels
are code-generated with named per-k variables (cf. triton_conv.py).
"""
import functools
import importlib.util
import os
import tempfile
import torch
import triton
import triton.language as tl  # noqa: F401

_KERNEL_DIR = os.path.join(tempfile.gettempdir(), "silogic_dense")
os.makedirs(_KERNEL_DIR, exist_ok=True)


def _gen_src(K):
    f = [
        "@triton.jit",
        "def _fwd(x_ptr, ca_ptr, cb_ptr, wa_ptr, wb_ptr, coef_ptr, out_ptr,",
        "         B, IN, OUT, BLOCK_O: tl.constexpr):",
        "    ot = tl.program_id(0); b = tl.program_id(1)",
        "    o = ot*BLOCK_O + tl.arange(0, BLOCK_O); mo = o < OUT",
        "    c0=tl.load(coef_ptr+o*4+0,mask=mo,other=0.0); c1=tl.load(coef_ptr+o*4+1,mask=mo,other=0.0)",
        "    c2=tl.load(coef_ptr+o*4+2,mask=mo,other=0.0); c3=tl.load(coef_ptr+o*4+3,mask=mo,other=0.0)",
        "    a=tl.zeros([BLOCK_O],tl.float32); bb=tl.zeros([BLOCK_O],tl.float32)"]
    for k in range(K):
        f += [
            f"    ca=tl.load(ca_ptr+o*{K}+{k},mask=mo,other=0); wak=tl.load(wa_ptr+o*{K}+{k},mask=mo,other=0.0)",
            f"    a += wak*tl.load(x_ptr+b*IN+ca,mask=mo,other=0.0)",
            f"    cb=tl.load(cb_ptr+o*{K}+{k},mask=mo,other=0); wbk=tl.load(wb_ptr+o*{K}+{k},mask=mo,other=0.0)",
            f"    bb += wbk*tl.load(x_ptr+b*IN+cb,mask=mo,other=0.0)"]
    f += ["    tl.store(out_ptr+b*OUT+o, c0+c1*a+c2*bb+c3*a*bb, mask=mo)"]

    g = [
        "@triton.jit",
        "def _bwd(go_ptr, x_ptr, ca_ptr, cb_ptr, wa_ptr, wb_ptr, coef_ptr,",
        "         gx_ptr, gwa_ptr, gwb_ptr, gc_ptr, B, IN, OUT, BLOCK_O: tl.constexpr):",
        "    ot = tl.program_id(0)",
        "    o = ot*BLOCK_O + tl.arange(0, BLOCK_O); mo = o < OUT",
        "    c1=tl.load(coef_ptr+o*4+1,mask=mo,other=0.0); c2=tl.load(coef_ptr+o*4+2,mask=mo,other=0.0); c3=tl.load(coef_ptr+o*4+3,mask=mo,other=0.0)",
        "    gc0=tl.zeros([BLOCK_O],tl.float32); gc1=tl.zeros([BLOCK_O],tl.float32); gc2=tl.zeros([BLOCK_O],tl.float32); gc3=tl.zeros([BLOCK_O],tl.float32)"]
    for k in range(K):
        g += [f"    gwa{k}=tl.zeros([BLOCK_O],tl.float32); gwb{k}=tl.zeros([BLOCK_O],tl.float32)"]
    g += ["    for b in range(B):",
          "        a=tl.zeros([BLOCK_O],tl.float32); bb=tl.zeros([BLOCK_O],tl.float32)"]
    for k in range(K):
        g += [
            f"        ca{k}=tl.load(ca_ptr+o*{K}+{k},mask=mo,other=0); wa{k}=tl.load(wa_ptr+o*{K}+{k},mask=mo,other=0.0)",
            f"        xa{k}=tl.load(x_ptr+b*IN+ca{k},mask=mo,other=0.0); a += wa{k}*xa{k}",
            f"        cb{k}=tl.load(cb_ptr+o*{K}+{k},mask=mo,other=0); wb{k}=tl.load(wb_ptr+o*{K}+{k},mask=mo,other=0.0)",
            f"        xb{k}=tl.load(x_ptr+b*IN+cb{k},mask=mo,other=0.0); bb += wb{k}*xb{k}"]
    g += [
        "        gg=tl.load(go_ptr+b*OUT+o,mask=mo,other=0.0)",
        "        gc0+=gg; gc1+=gg*a; gc2+=gg*bb; gc3+=gg*a*bb",
        "        ga=gg*(c1+c3*bb); gb=gg*(c2+c3*a)"]
    for k in range(K):
        g += [
            f"        gwa{k} += ga*xa{k}; tl.atomic_add(gx_ptr+b*IN+ca{k}, ga*wa{k}, mask=mo, sem='relaxed')",
            f"        gwb{k} += gb*xb{k}; tl.atomic_add(gx_ptr+b*IN+cb{k}, gb*wb{k}, mask=mo, sem='relaxed')"]
    g += [
        "    tl.store(gc_ptr+o*4+0,gc0,mask=mo); tl.store(gc_ptr+o*4+1,gc1,mask=mo); tl.store(gc_ptr+o*4+2,gc2,mask=mo); tl.store(gc_ptr+o*4+3,gc3,mask=mo)"]
    for k in range(K):
        g += [f"    tl.store(gwa_ptr+o*{K}+{k},gwa{k},mask=mo); tl.store(gwb_ptr+o*{K}+{k},gwb{k},mask=mo)"]
    return "\n".join(f) + "\n\n" + "\n".join(g)


@functools.lru_cache(maxsize=None)
def _compiled(K):
    src = "import triton\nimport triton.language as tl\n\n" + _gen_src(K)
    path = os.path.join(_KERNEL_DIR, f"dense_k{K}.py")
    with open(path, "w") as fh:
        fh.write(src)
    spec = importlib.util.spec_from_file_location(f"dense_k{K}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._fwd, mod._bwd


class DenseLogicFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, ca, cb, wa, wb, coef):
        B, IN = x.shape
        OUT, K = ca.shape
        fwd, _ = _compiled(K)
        x = x.contiguous(); wa = wa.contiguous(); wb = wb.contiguous()
        coef = coef.contiguous()
        out = torch.empty((B, OUT), device=x.device, dtype=x.dtype)  # bf16 = half
        BLOCK_O = 128
        grid = (triton.cdiv(OUT, BLOCK_O), B)
        fwd[grid](x, ca, cb, wa, wb, coef, out, B, IN, OUT, BLOCK_O=BLOCK_O)
        ctx.save_for_backward(x, ca, cb, wa, wb, coef)
        ctx.dims = (B, IN, OUT, K)
        return out

    @staticmethod
    def backward(ctx, go):
        x, ca, cb, wa, wb, coef = ctx.saved_tensors
        B, IN, OUT, K = ctx.dims
        _, bwd = _compiled(K)
        gx = torch.zeros(x.shape, device=x.device, dtype=torch.float32)  # atomics fp32
        gwa = torch.empty_like(wa, dtype=torch.float32)
        gwb = torch.empty_like(wb, dtype=torch.float32)
        gc = torch.empty_like(coef, dtype=torch.float32)
        BLOCK_O = 128
        grid = (triton.cdiv(OUT, BLOCK_O),)
        bwd[grid](go.contiguous(), x, ca, cb, wa, wb, coef, gx, gwa, gwb, gc,
                  B, IN, OUT, BLOCK_O=BLOCK_O)
        return gx.to(x.dtype), None, None, gwa, gwb, gc


def dense_logic(x, ca, cb, wa, wb, coef):
    """x[B,IN], ca/cb[OUT,K] int32, wa/wb[OUT,K], coef[OUT,4] -> [B,OUT]."""
    return DenseLogicFn.apply(x, ca, cb, wa, wb, coef)
