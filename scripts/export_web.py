"""Step 5 — ship the trained models + results to the ECHO web lab (web/public)."""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from echo.data import EOS, PAD, UNK, Vocab, tokenize  # noqa: E402
from echo.export import export_all, verify  # noqa: E402
from echo.generate import load_model  # noqa: E402
from echo.models import BONUS, DISPLAY, KINDS  # noqa: E402

PUB = ROOT / "web" / "public"
M = ROOT / "results" / "metrics"


def records(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.to_json(orient="records"))


@torch.no_grad()
def walkthrough(cfg: dict, n_steps: int = 30) -> dict:
    """Greedy LSTM generation with the top-5 candidates at each step (for the explainer)."""
    vocab = Vocab.load(ROOT / "data" / "processed" / "vocab.json")
    proper = set(json.loads((ROOT / "data" / "processed" / "proper_nouns.json").read_text()))
    def show(t, prev):
        if t == EOS:
            return "¶"
        if t == "i" or t in proper or prev in (".", "!", "?", EOS, None):
            return t.capitalize()
        return t
    model = load_model(ROOT / "models" / "lstm.pt", "cpu")
    seed_toks = tokenize(cfg["generate"]["seeds_text"][0])
    ids = [vocab.stoi[EOS]] + vocab.encode(seed_toks)
    steps = []
    prev = seed_toks[-1]
    for _ in range(n_steps):
        logits = model(torch.tensor([ids[-cfg["data"]["window"]:]]))[0]
        logits[[vocab.stoi[PAD], vocab.stoi[UNK]]] = float("-inf")
        p = torch.softmax(logits, -1)
        top = p.topk(5)
        nxt = int(top.indices[0])
        steps.append({"token": show(vocab.itos[nxt], prev),
                      "top": [[show(vocab.itos[int(i)], prev), round(float(v), 4)]
                              for v, i in zip(top.values, top.indices)]})
        ids.append(nxt)
        prev = vocab.itos[nxt]
    seed_disp = [show(t, p) for t, p in zip(seed_toks, [None] + seed_toks[:-1])]
    return {"model": "LSTM", "seed": seed_disp, "steps": steps}


def build_payload(cfg: dict) -> dict:
    stats = json.loads((M / "dataset_stats.json").read_text())
    stories = json.loads((ROOT / "data" / "raw" / "stories.json").read_text())
    lengths = [len(tokenize(s)) for s in stories["train"]]
    hist, edges = np.histogram(lengths, bins=40, range=(0, 600))

    runs = pd.read_csv(M / "runs.csv")
    summary = pd.read_csv(M / "summary_by_model.csv", index_col=0)
    best = json.loads((M / "best_runs.json").read_text())
    matched = pd.read_csv(M / "matched.csv") if (M / "matched.csv").stat().st_size > 10 else pd.DataFrame()
    tq = pd.read_csv(M / "text_quality.csv")
    gens = json.loads((ROOT / "results" / "samples" / "generations.json").read_text())

    bonus = pd.read_csv(M / "bonus.csv") if (M / "bonus.csv").stat().st_size > 10 else pd.DataFrame()
    allruns = pd.concat([runs, bonus])
    speed = pd.read_csv(M / "speed_benchmark.csv") if (M / "speed_benchmark.csv").exists() else pd.DataFrame()
    models = {}
    for kind in [k for k in (*KINDS, *BONUS) if k in set(allruns.kind)]:
        hist_by_seed = {}
        for run in allruns[allruns.kind == kind].run:
            h = json.loads((ROOT / "results" / "runs" / run / "history.json").read_text())
            hist_by_seed[run] = [{k: r[k] for k in ("epoch", "train_perplexity", "val_perplexity", "val_top1_acc",
                                                    "epoch_time_s")} for r in h]
        models[kind] = {
            "name": DISPLAY[kind],
            "summary": json.loads(summary.loc[kind].to_json()) if kind in summary.index else {},
            "runs": records(allruns[allruns.kind == kind]),
            "bonus": kind in BONUS,
            "speed": records(speed[speed.kind == kind])[0] if len(speed) else None,
            "history": hist_by_seed,
            "bestRun": best[kind],
            "matched": records(matched[matched.kind == kind]) if len(matched) else [],
        }

    samples = [{k: g[k] for k in ("kind", "seed_id", "temperature", "sample", "text", "gpt2_ppl", "repetition",
                                   "consistency")} for g in gens]
    seeds = sorted({(g["seed_id"], g["seed_display"]) for g in gens})
    return {
        "dataset": {
            "name": "TinyStories", "source": "https://huggingface.co/datasets/roneneldan/TinyStories",
            "paper": "https://arxiv.org/abs/2305.07759", "stats": stats,
            "lengthHistogram": {"counts": hist.tolist(), "edges": edges.tolist()},
            "example": stories["train"][3],
        },
        "config": {"window": cfg["data"]["window"], "model": cfg["model"], "train": cfg["train"],
                   "temperatures": cfg["generate"]["temperatures"]},
        "models": models,
        "textQuality": records(tq),
        "seeds": [{"id": i, "text": t} for i, t in seeds],
        "samples": samples,
        "walkthrough": walkthrough(cfg),
    }


if __name__ == "__main__":
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())
    models = {k: load_model(ROOT / "models" / f"{k}.pt", "cpu") for k in (*KINDS, *BONUS)
              if (ROOT / "models" / f"{k}.pt").exists()}
    sizes = export_all(models, PUB / "models")
    print("onnx sizes (MB):", {k: round(v / 1e6, 1) for k, v in sizes.items()})
    print("max |torch - onnx| logit diff:", verify(models, PUB / "models"))

    (PUB / "data").mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "data" / "processed" / "vocab.json", PUB / "data" / "vocab.json")
    shutil.copy(ROOT / "data" / "processed" / "proper_nouns.json", PUB / "data" / "proper_nouns.json")
    payload = build_payload(cfg)
    (PUB / "data" / "echo.json").write_text(json.dumps(payload, separators=(",", ":")))
    print("wrote echo.json", round((PUB / "data" / "echo.json").stat().st_size / 1e6, 2), "MB")
