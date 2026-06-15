"""Training and evaluation for silogic logic-gate networks."""

import time
import math
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from .model import LogicNet

torch.set_float32_matmul_precision("high")
_print = functools.partial(print, flush=True)


@torch.no_grad()
def eval_hard(model, X, y, device, bs=1024):
    model.eval()
    correct = 0
    for i in range(0, X.shape[0], bs):
        xb = X[i:i + bs].to(device)
        logits = model.forward_hard(xb)
        pred = logits.argmax(dim=1).cpu()
        correct += (pred == y[i:i + bs]).sum().item()
    return 100.0 * correct / X.shape[0]


@torch.no_grad()
def eval_soft(model, X, y, device, bs=1024):
    model.eval()
    correct = 0
    for i in range(0, X.shape[0], bs):
        xb = X[i:i + bs].to(device).float()
        logits = model(xb)
        pred = logits.argmax(dim=1).cpu()
        correct += (pred == y[i:i + bs]).sum().item()
    return 100.0 * correct / X.shape[0]


def train_model(model, Xtr, ytr, Xte, yte, device, epochs=200, bs=256,
                lr=0.075, val_every=25, gpu_data=True, log=_print,
                Xval=None, yval=None, compile_=True, eval_bs=1024,
                optimizer="adam", weight_decay=0.0, cosine=False):
    """Train and return dict with metrics + per-epoch history.

    Xtr/Xte are uint8 tensors. If gpu_data, the full train set is moved to
    the GPU once (fast); otherwise batches are moved on the fly. Uses
    torch.compile with static batch shapes (drop_last) for throughput.
    """
    model.to(device)
    if optimizer == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=lr,
                                weight_decay=weight_decay)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = Xtr.shape[0]

    if gpu_data:
        try:
            Xtr_d = Xtr.to(device)
            ytr_d = ytr.to(device)
        except RuntimeError:
            gpu_data = False
    if not gpu_data:
        Xtr_d, ytr_d = Xtr, ytr

    fwd = torch.compile(model) if compile_ else model
    nb = n // bs  # drop_last -> static shapes for compile

    history = {"val_acc": [], "bin_val_acc": [], "epoch": []}
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        if cosine:                       # cosine LR decay to 0 over `epochs`
            cur_lr = 0.5 * lr * (1 + math.cos(math.pi * ep / epochs))
            for g in opt.param_groups:
                g["lr"] = cur_lr
        perm = torch.randperm(n, device=device if gpu_data else "cpu")
        for b in range(nb):
            bidx = perm[b * bs:(b + 1) * bs]
            if gpu_data:
                xb = Xtr_d[bidx].float()
                yb = ytr_d[bidx]
            else:
                xb = Xtr_d[bidx].to(device).float()
                yb = ytr_d[bidx].to(device)
            logits = fwd(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

        if (ep + 1) % val_every == 0 or ep == epochs - 1:
            Xv = Xval if Xval is not None else Xte
            yv = yval if yval is not None else yte
            va = eval_soft(model, Xv, yv, device, bs=eval_bs)
            ba = eval_hard(model, Xv, yv, device, bs=eval_bs)
            history["epoch"].append(ep + 1)
            history["val_acc"].append(va)
            history["bin_val_acc"].append(ba)
            log(f"  epoch {ep+1:3d}  loss {loss.item():.4f}  "
                f"soft {va:.2f}  hard {ba:.2f}  "
                f"[{(time.time()-t0)/60:.1f} min]")

    train_min = (time.time() - t0) / 60.0
    test_soft = eval_soft(model, Xte, yte, device, bs=eval_bs)
    test_hard = eval_hard(model, Xte, yte, device, bs=eval_bs)
    return {
        "test_soft": test_soft,
        "test_hard": test_hard,
        "train_min": train_min,
        "history": history,
    }
