"""Step 3b — controlled speed benchmark (run on an otherwise idle machine).

Wall-clock epoch times logged during training can be disturbed by anything else
running on the computer. This script times every architecture back-to-back
under identical conditions:

  train_ms_per_batch   forward + backward + optimiser step, batch 512, window 24
  infer_ms_per_word    one next-word prediction (batch 1), as in generation
  est_min_per_epoch    train_ms_per_batch x batches in one 2M-sample epoch

Output: results/metrics/speed_benchmark.csv
"""
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from echo.data import WindowBatcher  # noqa: E402
from echo.models import BONUS, DISPLAY, KINDS, build, count_params  # noqa: E402
from echo.train import device, load_split, sync  # noqa: E402

WARMUP, STEPS, REPEATS = 20, 100, 3

if __name__ == "__main__":
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())
    dev = device()
    ids = load_split(ROOT, "train")
    W, B = cfg["data"]["window"], cfg["train"]["batch_size"]
    rows = []
    for kind in (*KINDS, *BONUS):
        torch.manual_seed(0)
        model = build(kind, 8000, cfg["model"]).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        batches = iter(WindowBatcher(ids, W, B, dev, shuffle=True, seed=0))
        train_ms = []
        for _ in range(REPEATS):
            for i in range(WARMUP + STEPS):
                if i == WARMUP:
                    sync(dev)
                    t0 = time.perf_counter()
                x, y = next(batches)
                loss = torch.nn.functional.cross_entropy(model(x), y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                if i % 20 == 0:
                    sync(dev)
            sync(dev)
            train_ms.append((time.perf_counter() - t0) / STEPS * 1000)

        model.eval()
        x1 = x[:1]
        infer_ms = []
        with torch.no_grad():
            for _ in range(REPEATS):
                for _ in range(20):
                    model(x1)
                sync(dev)
                t0 = time.perf_counter()
                for _ in range(300):
                    model(x1)
                sync(dev)
                infer_ms.append((time.perf_counter() - t0) / 300 * 1000)

        t = sorted(train_ms)[REPEATS // 2]  # median of repeats
        batches_per_epoch = cfg["train"]["samples_per_epoch"] / B
        rows.append({"kind": kind, "model": DISPLAY[kind], **count_params(model),
                     "train_ms_per_batch": t, "infer_ms_per_word": sorted(infer_ms)[REPEATS // 2],
                     "est_min_per_epoch": t * batches_per_epoch / 60000, "device": str(dev)})
        print(rows[-1], flush=True)
        del model, opt
        if dev.type == "mps":
            torch.mps.empty_cache()

    out = ROOT / "results" / "metrics" / "speed_benchmark.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out)
