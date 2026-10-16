"""Training loop shared by all four models (identical conditions)."""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .data import Vocab, WindowBatcher
from .models import build, count_params


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def sync(dev: torch.device):
    if dev.type == "mps":
        torch.mps.synchronize()
    elif dev.type == "cuda":
        torch.cuda.synchronize()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_split(root: Path, split: str) -> torch.Tensor:
    return torch.load(root / "data" / "processed" / f"{split}.pt")


@torch.no_grad()
def evaluate(model: nn.Module, batcher: WindowBatcher) -> dict[str, float]:
    """Mean cross-entropy, perplexity and top-1 / top-5 next-word accuracy."""
    model.eval()
    loss_sum = torch.zeros((), device=batcher.ids.device)
    top1 = torch.zeros((), device=batcher.ids.device)
    top5 = torch.zeros((), device=batcher.ids.device)
    n = 0
    for i, (x, y) in enumerate(batcher):
        if i % 20 == 0:
            sync(batcher.ids.device)  # keep the async MPS queue (and its memory) bounded
        logits = model(x)
        loss_sum += nn.functional.cross_entropy(logits, y, reduction="sum")
        k5 = logits.topk(5, dim=-1).indices
        top1 += (k5[:, 0] == y).sum()
        top5 += (k5 == y[:, None]).any(-1).sum()
        n += len(y)
    loss = loss_sum.item() / n
    return {"loss": loss, "perplexity": math.exp(loss), "top1_acc": top1.item() / n, "top5_acc": top5.item() / n}


def train_one(kind: str, seed: int, cfg: dict, root: Path, run_name: str | None = None,
              model_overrides: dict | None = None, log=print) -> dict:
    """Train one model, keep the best-validation checkpoint, report test metrics."""
    dev = device()
    set_seed(seed)
    vocab = Vocab.load(root / "data" / "processed" / "vocab.json")
    mcfg = {**cfg["model"], **(model_overrides or {})}
    tcfg, W = cfg["train"], cfg["data"]["window"]

    model = build(kind, len(vocab), mcfg).to(dev)
    params = count_params(model)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])

    train_b = WindowBatcher(load_split(root, "train"), W, tcfg["batch_size"], dev, shuffle=True, seed=seed,
                            samples_per_epoch=tcfg.get("samples_per_epoch"))
    val_b = WindowBatcher(load_split(root, "val"), W, 512, dev, shuffle=False)
    test_b = WindowBatcher(load_split(root, "test"), W, 512, dev, shuffle=False)

    total_steps = tcfg["max_epochs"] * len(train_b)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=tcfg["lr"], total_steps=total_steps, pct_start=0.05)

    run_dir = root / "results" / "runs" / (run_name or f"{kind}_s{seed}")
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt = run_dir / "checkpoint.pt"

    history, best_val, bad_epochs, best_epoch = [], float("inf"), 0, 0
    train_time = 0.0
    log(f"[{run_dir.name}] params={params['total']:,} (core {params['recurrent_core']:,}) device={dev}")

    for epoch in range(1, tcfg["max_epochs"] + 1):
        model.train()
        sync(dev)
        t0 = time.perf_counter()
        run_loss = torch.zeros((), device=dev)
        grad_norms = []
        for step, (x, y) in enumerate(train_b):
            loss = nn.functional.cross_entropy(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
            opt.step()
            sched.step()
            run_loss += loss.detach()
            if step % 50 == 0:
                grad_norms.append(gn.item())  # .item() also syncs, bounding the MPS queue
        sync(dev)
        epoch_time = time.perf_counter() - t0
        train_time += epoch_time

        train_loss = run_loss.item() / len(train_b)
        val = evaluate(model, val_b)
        row = {"epoch": epoch, "train_loss": train_loss, "train_perplexity": math.exp(train_loss),
               **{f"val_{k}": v for k, v in val.items()}, "epoch_time_s": epoch_time,
               "lr": sched.get_last_lr()[0], "grad_norm_mean": float(np.mean(grad_norms))}
        history.append(row)
        log(f"[{run_dir.name}] ep {epoch:2d}  train {train_loss:.3f}  val {val['loss']:.3f}  "
            f"ppl {val['perplexity']:.1f}  top1 {val['top1_acc']:.3f}  {epoch_time:.0f}s")

        if val["loss"] < best_val - 1e-4:
            best_val, best_epoch, bad_epochs = val["loss"], epoch, 0
            torch.save({"kind": kind, "model_cfg": mcfg, "state_dict": model.state_dict(),
                        "vocab_size": len(vocab), "window": W}, ckpt)
        else:
            bad_epochs += 1
            if bad_epochs >= tcfg["patience"]:
                log(f"[{run_dir.name}] early stop at epoch {epoch} (best {best_epoch})")
                break

    model.load_state_dict(torch.load(ckpt, map_location=dev)["state_dict"])
    test = evaluate(model, test_b)

    # Generation speed: time for 200 single next-word predictions.
    model.eval()
    x = next(iter(test_b))[0][:1]
    with torch.no_grad():
        for _ in range(10):
            model(x)
        sync(dev)
        t0 = time.perf_counter()
        for _ in range(200):
            model(x)
        sync(dev)
    ms_per_token = (time.perf_counter() - t0) / 200 * 1000

    summary = {
        "run": run_dir.name, "kind": kind, "seed": seed, "params": params, "model_cfg": mcfg,
        "epochs_run": len(history), "best_epoch": best_epoch,
        "train_time_s": train_time, "time_per_epoch_s": train_time / len(history),
        "time_to_best_s": sum(h["epoch_time_s"] for h in history[:best_epoch]),
        "best_val": next(h for h in history if h["epoch"] == best_epoch),
        "test": test, "ms_per_token": ms_per_token, "device": str(dev),
    }
    del model, opt, train_b, val_b, test_b
    import gc; gc.collect()
    if dev.type == "mps":
        torch.mps.empty_cache()
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log(f"[{run_dir.name}] TEST loss {test['loss']:.3f} ppl {test['perplexity']:.1f} "
        f"top1 {test['top1_acc']:.3f} top5 {test['top5_acc']:.3f}  train {train_time/60:.1f} min")
    return summary
