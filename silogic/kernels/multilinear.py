"""Fused Triton kernel for the n-input multilinear LUT layer with the DWN-style
hybrid straight-through estimator (``node="hybrid"``).

A hybrid node combines a *discrete* forward — ``sigmoid(logits[idx])`` where
``idx`` is the binary address of the hard-thresholded operands — with the *soft*
multilinear surrogate's gradient. Per output ``o``:

    a_j   = sum_k w[o,j,k] * x[cand[o,j,k]]           (Top-K weighted gather, n slots)
    idx   = sum_j (a_j >= 0.5) << j
    fwd   = s[o, idx]                                  (s = sigmoid(logits), 2^n entries)
    f_soft= sum_P s[o,P] * prod_j (a_j if bit_j(P) else 1-a_j)
    grad  flows as if the output were f_soft (STE)

This fuses the ``[B, out, n, k]`` gather + the ``2^n`` multilinear corners that the
PyTorch path materialises. The connection softmax (``w = softmax(conn)``) and the
LUT sigmoid (``s = sigmoid(logits)``) stay in PyTorch, so the kernel only needs
grads w.r.t. ``x``, ``w`` and ``s``. Like the sibling kernels, the (n, K)-unrolled
kernels are code-generated (Triton's front-end disallows in-kernel Python lists).
"""
import functools
import importlib.util
import os
import tempfile
import torch
import triton
import triton.language as tl  # noqa: F401

_KERNEL_DIR = os.path.join(tempfile.gettempdir(), "silogic_multilinear")
os.makedirs(_KERNEL_DIR, exist_ok=True)


def _gen_src(n, K):
    P = 2 ** n
    NK = n * K

    def fac(p, i):                       # multilinear factor of corner p, slot i
        return f"a{i}" if (p >> i) & 1 else f"oma{i}"

    # ---- forward: f_disc = s[idx], idx from hard-thresholded soft operands ----
    fl = [
        "@triton.jit",
        "def _fwd(x_ptr, cand_ptr, w_ptr, s_ptr, out_ptr, B, IN, OUT, "
        "BLOCK_O: tl.constexpr):",
        "    ot = tl.program_id(0); b = tl.program_id(1)",
        "    o = ot*BLOCK_O + tl.arange(0, BLOCK_O); mo = o < OUT",
        "    idx = tl.zeros([BLOCK_O], tl.int32)",
    ]
    for j in range(n):
        fl.append("    a = tl.zeros([BLOCK_O], tl.float32)")
        for k in range(K):
            off = j * K + k
            fl += [
                f"    c = tl.load(cand_ptr + o*{NK} + {off}, mask=mo, other=0)",
                f"    wk = tl.load(w_ptr + o*{NK} + {off}, mask=mo, other=0.0)",
                "    a += wk * tl.load(x_ptr + b*IN + c, mask=mo, other=0.0)",
            ]
        fl.append(f"    idx += tl.where(a >= 0.5, {1 << j}, 0)")
    fl += [
        f"    fv = tl.load(s_ptr + o*{P} + idx, mask=mo, other=0.0)",
        "    tl.store(out_ptr + b*OUT + o, fv, mask=mo)",
    ]

    # ---- backward: gradients of f_soft w.r.t. s, w, x -------------------------
    gl = [
        "@triton.jit",
        "def _bwd(go_ptr, x_ptr, cand_ptr, w_ptr, s_ptr, gx_ptr, gw_ptr, gs_ptr,",
        "         B, IN, OUT, BLOCK_O: tl.constexpr):",
        "    ot = tl.program_id(0)",
        "    o = ot*BLOCK_O + tl.arange(0, BLOCK_O); mo = o < OUT",
    ]
    for p in range(P):
        gl.append(f"    s{p} = tl.load(s_ptr + o*{P} + {p}, mask=mo, other=0.0)")
    for p in range(P):
        gl.append(f"    gs{p} = tl.zeros([BLOCK_O], tl.float32)")
    for j in range(n):
        for k in range(K):
            gl.append(f"    gw{j}_{k} = tl.zeros([BLOCK_O], tl.float32)")
    gl.append("    for b in range(B):")
    for j in range(n):
        gl.append(f"        a{j} = tl.zeros([BLOCK_O], tl.float32)")
        for k in range(K):
            off = j * K + k
            gl += [
                f"        cc{j}_{k} = tl.load(cand_ptr + o*{NK} + {off}, mask=mo, other=0)",
                f"        ww{j}_{k} = tl.load(w_ptr + o*{NK} + {off}, mask=mo, other=0.0)",
                f"        xx{j}_{k} = tl.load(x_ptr + b*IN + cc{j}_{k}, mask=mo, other=0.0)",
                f"        a{j} += ww{j}_{k} * xx{j}_{k}",
            ]
        gl.append(f"        oma{j} = 1.0 - a{j}")
    gl.append("        gg = tl.load(go_ptr + b*OUT + o, mask=mo, other=0.0)")
    # M_P and grad w.r.t. s
    for p in range(P):
        prod = "*".join(fac(p, i) for i in range(n)) if n else "1.0"
        gl.append(f"        gs{p} += gg * ({prod})")
    # grad w.r.t. each soft operand a_j (multilinear partial), then w / x
    for j in range(n):
        terms = []
        for p in range(P):
            sign = "" if (p >> j) & 1 else "-"
            excl = [fac(p, i) for i in range(n) if i != j]
            mexcl = "*".join(excl) if excl else "1.0"
            terms.append(f"{sign}s{p}*{mexcl}")
        gl.append(f"        ga{j} = gg * (" + " + ".join(terms) + ")")
        for k in range(K):
            gl += [
                f"        gw{j}_{k} += ga{j} * xx{j}_{k}",
                f"        tl.atomic_add(gx_ptr + b*IN + cc{j}_{k}, ga{j} * ww{j}_{k}, "
                "mask=mo, sem='relaxed')",
            ]
    for p in range(P):
        gl.append(f"    tl.store(gs_ptr + o*{P} + {p}, gs{p}, mask=mo)")
    for j in range(n):
        for k in range(K):
            gl.append(f"    tl.store(gw_ptr + o*{NK} + {j*K+k}, gw{j}_{k}, mask=mo)")
    return "\n".join(fl) + "\n\n" + "\n".join(gl)


@functools.lru_cache(maxsize=None)
def _compiled(n, K):
    src = "import triton\nimport triton.language as tl\n\n" + _gen_src(n, K)
    path = os.path.join(_KERNEL_DIR, f"multilinear_n{n}_k{K}.py")
    with open(path, "w") as fh:
        fh.write(src)
    spec = importlib.util.spec_from_file_location(f"multilinear_n{n}_k{K}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._fwd, mod._bwd


class MultilinearLogicFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cand, w, s, arity):
        B, IN = x.shape
        OUT, n, K = cand.shape
        assert n == arity
        fwd, _ = _compiled(arity, K)
        x = x.contiguous(); w = w.contiguous(); s = s.contiguous()
        cand = cand.contiguous()
        out = torch.empty((B, OUT), device=x.device, dtype=x.dtype)
        BLOCK_O = 128
        grid = (triton.cdiv(OUT, BLOCK_O), B)
        fwd[grid](x, cand, w, s, out, B, IN, OUT, BLOCK_O=BLOCK_O)
        ctx.save_for_backward(x, cand, w, s)
        ctx.dims = (B, IN, OUT, K, arity)
        return out

    @staticmethod
    def backward(ctx, go):
        x, cand, w, s = ctx.saved_tensors
        B, IN, OUT, K, arity = ctx.dims
        _, bwd = _compiled(arity, K)
        gx = torch.zeros(x.shape, device=x.device, dtype=torch.float32)  # atomics fp32
        gw = torch.empty_like(w, dtype=torch.float32)
        gs = torch.empty_like(s, dtype=torch.float32)
        BLOCK_O = 128
        grid = (triton.cdiv(OUT, BLOCK_O),)
        bwd[grid](go.contiguous(), x, cand, w, s, gx, gw, gs,
                  B, IN, OUT, BLOCK_O=BLOCK_O)
        return gx.to(x.dtype), None, gw.to(w.dtype), gs.to(s.dtype), None


def multilinear_logic(x, cand, w, s, arity):
    """Hybrid (DWN-STE) multilinear LUT layer.

    Args:
        x (Tensor): ``[B, IN]`` inputs.
        cand (Tensor): ``[OUT, arity, K]`` int32 candidate wire indices.
        w (Tensor): ``[OUT, arity, K]`` post-softmax connection weights.
        s (Tensor): ``[OUT, 2**arity]`` = ``sigmoid(logits)`` LUT entries.
        arity (int): Inputs per node ``n``.

    Returns:
        Tensor ``[B, OUT]`` — the discrete forward ``sigmoid(logits[idx])`` carrying
        the soft multilinear surrogate's gradient.
    """
    return MultilinearLogicFn.apply(x, cand, w, s, arity)
