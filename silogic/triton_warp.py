"""Fused Triton kernel for the WARP logic layer: gather the Top-K weighted
inputs (a,b), evaluate the multilinear form z = c0+c1*a+c2*b+c3*a*b (the Walsh
gate after the theta->coef change of variables), and apply sigmoid(z/tau) -- all
in one kernel. The theta->coef transform stays in PyTorch (cheap), so its grad
flows through coef. Mirrors triton_dense.py (code-generated per K).
"""
import functools, importlib.util, os, tempfile
import torch, triton, triton.language as tl  # noqa: F401

_KDIR = os.path.join(tempfile.gettempdir(), "silogic_warp")
os.makedirs(_KDIR, exist_ok=True)


def _gen(K):
    f = [
        "@triton.jit",
        "def _fwd(x_ptr, ca_ptr, cb_ptr, wa_ptr, wb_ptr, coef_ptr, out_ptr,",
        "         inv_tau, B, IN, OUT, BLOCK_O: tl.constexpr):",
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
    f += ["    z = (c0+c1*a+c2*bb+c3*a*bb)*inv_tau",
          "    tl.store(out_ptr+b*OUT+o, tl.sigmoid(z), mask=mo)"]

    g = [
        "@triton.jit",
        "def _bwd(go_ptr, x_ptr, ca_ptr, cb_ptr, wa_ptr, wb_ptr, coef_ptr,",
        "         gx_ptr, gwa_ptr, gwb_ptr, gc_ptr, inv_tau, B, IN, OUT, BLOCK_O: tl.constexpr):",
        "    ot = tl.program_id(0)",
        "    o = ot*BLOCK_O + tl.arange(0, BLOCK_O); mo = o < OUT",
        "    c0=tl.load(coef_ptr+o*4+0,mask=mo,other=0.0); c1=tl.load(coef_ptr+o*4+1,mask=mo,other=0.0)",
        "    c2=tl.load(coef_ptr+o*4+2,mask=mo,other=0.0); c3=tl.load(coef_ptr+o*4+3,mask=mo,other=0.0)",
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
        "        out=tl.sigmoid((c0+c1*a+c2*bb+c3*a*bb)*inv_tau)",
        "        gg=tl.load(go_ptr+b*OUT+o,mask=mo,other=0.0)",
        "        gz=gg*out*(1.0-out)*inv_tau",                    # sigmoid' * d(z*inv_tau)/dz
        "        gc0+=gz; gc1+=gz*a; gc2+=gz*bb; gc3+=gz*a*bb",
        "        ga=gz*(c1+c3*bb); gb=gz*(c2+c3*a)"]
    for k in range(K):
        g += [
            f"        gwa{k} += ga*xa{k}; tl.atomic_add(gx_ptr+b*IN+ca{k}, ga*wa{k}, mask=mo, sem='relaxed')",
            f"        gwb{k} += gb*xb{k}; tl.atomic_add(gx_ptr+b*IN+cb{k}, gb*wb{k}, mask=mo, sem='relaxed')"]
    g += ["    tl.store(gc_ptr+o*4+0,gc0,mask=mo); tl.store(gc_ptr+o*4+1,gc1,mask=mo); tl.store(gc_ptr+o*4+2,gc2,mask=mo); tl.store(gc_ptr+o*4+3,gc3,mask=mo)"]
    for k in range(K):
        g += [f"    tl.store(gwa_ptr+o*{K}+{k},gwa{k},mask=mo); tl.store(gwb_ptr+o*{K}+{k},gwb{k},mask=mo)"]
    return "\n".join(f) + "\n\n" + "\n".join(g)


@functools.lru_cache(maxsize=None)
def _compiled(K):
    src = "import triton\nimport triton.language as tl\n\n" + _gen(K)
    path = os.path.join(_KDIR, f"warp_k{K}.py")
    with open(path, "w") as fh:
        fh.write(src)
    spec = importlib.util.spec_from_file_location(f"warp_k{K}", path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod._fwd, mod._bwd


class WarpFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, ca, cb, wa, wb, coef, inv_tau):
        B, IN = x.shape; OUT, K = ca.shape
        fwd, _ = _compiled(K)
        x = x.contiguous(); wa = wa.contiguous(); wb = wb.contiguous(); coef = coef.contiguous()
        out = torch.empty((B, OUT), device=x.device, dtype=x.dtype)
        BLK = 128; grid = (triton.cdiv(OUT, BLK), B)
        fwd[grid](x, ca, cb, wa, wb, coef, out, float(inv_tau), B, IN, OUT, BLOCK_O=BLK)
        ctx.save_for_backward(x, ca, cb, wa, wb, coef); ctx.dims = (B, IN, OUT, K); ctx.it = float(inv_tau)
        return out

    @staticmethod
    def backward(ctx, go):
        x, ca, cb, wa, wb, coef = ctx.saved_tensors
        B, IN, OUT, K = ctx.dims; _, bwd = _compiled(K)
        gx = torch.zeros(x.shape, device=x.device, dtype=torch.float32)
        gwa = torch.empty_like(wa, dtype=torch.float32); gwb = torch.empty_like(wb, dtype=torch.float32)
        gc = torch.empty_like(coef, dtype=torch.float32)
        BLK = 128; grid = (triton.cdiv(OUT, BLK),)
        bwd[grid](go.contiguous(), x, ca, cb, wa, wb, coef, gx, gwa, gwb, gc,
                  ctx.it, B, IN, OUT, BLOCK_O=BLK)
        return gx.to(x.dtype), None, None, gwa, gwb, gc, None


def warp_logic(x, ca, cb, wa, wb, coef, inv_tau):
    return WarpFn.apply(x, ca, cb, wa, wb, coef, inv_tau)
