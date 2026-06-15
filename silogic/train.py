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
    """Top-1 accuracy (%) of the hard Boolean-circuit forward pass.

    Args:
        model: A logic network exposing ``forward_hard``.
        X (torch.Tensor): uint8 feature tensor ``[N, ...]``.
        y (torch.Tensor): Integer labels ``[N]``.
        device (str): Device to run evaluation on, e.g. ``"cuda"``.
        bs (int): Eval batch size. Default ``1024``.

    Returns:
        float: Top-1 accuracy as a percentage.
    """
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
    """Top-1 accuracy (%) of the soft (differentiable) forward pass.

    Args:
        model: A logic network whose ``__call__`` returns class logits.
        X (torch.Tensor): uint8 feature tensor ``[N, ...]`` (cast to float).
        y (torch.Tensor): Integer labels ``[N]``.
        device (str): Device to run evaluation on, e.g. ``"cuda"``.
        bs (int): Eval batch size. Default ``1024``.

    Returns:
        float: Top-1 accuracy as a percentage.
    """
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

    Args:
        model: Logic network to train; returns class logits and exposes
            ``forward_hard`` for hard evaluation.
        Xtr (torch.Tensor): uint8 train feature tensor ``[N, ...]``.
        ytr (torch.Tensor): Integer train labels ``[N]``.
        Xte (torch.Tensor): uint8 test feature tensor ``[M, ...]``.
        yte (torch.Tensor): Integer test labels ``[M]``.
        device (str): Device to train on, e.g. ``"cuda"`` or ``"cpu"``.
        epochs (int): Number of training epochs. Default ``200``.
        bs (int): Training batch size; ``drop_last`` keeps shapes static for
            ``torch.compile``. Default ``256``.
        lr (float): Initial learning rate. Default ``0.075``.
        val_every (int): Validate every this many epochs (plus the final
            epoch). Default ``25``.
        gpu_data (bool): If ``True`` move the whole train set to the GPU once
            (falls back to per-batch transfer on OOM). Default ``True``.
        log (callable): Logging function called with a status string per
            validation. Default the module's flushing ``print``.
        Xval (torch.Tensor, optional): uint8 validation features; if ``None``
            the test set ``Xte`` is used for validation. Default ``None``.
        yval (torch.Tensor, optional): Validation labels paired with ``Xval``.
            Default ``None``.
        compile_ (bool): If ``True`` wrap the model in ``torch.compile``.
            Default ``True``.
        eval_bs (int): Batch size used by the eval passes. Default ``1024``.
        optimizer (str): Optimizer to use, ``"adam"`` (default) or
            ``"adamw"``.
        weight_decay (float): Weight decay, only applied when
            ``optimizer="adamw"``. Default ``0.0``.
        cosine (bool): If ``True`` apply cosine LR decay from ``lr`` to ``0``
            over ``epochs``. Default ``False``.

    Returns:
        dict: Metrics and history with keys ``"test_soft"`` (float, soft
        test accuracy %), ``"test_hard"`` (float, hard test accuracy %),
        ``"train_min"`` (float, wall-clock training time in minutes) and
        ``"history"`` (dict of per-validation ``"epoch"``, ``"val_acc"`` and
        ``"bin_val_acc"`` lists).
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
