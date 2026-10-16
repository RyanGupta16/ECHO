"""Step 3 — aggregate training metrics, generate text, score text quality.

Outputs
  results/metrics/runs.csv               one row per training run
  results/metrics/summary_by_model.csv   mean ± std over seeds
  results/metrics/matched.csv            parameter-matched variant
  results/metrics/text_quality.csv       per model x temperature
  results/samples/generations.json       every generated sample
  results/samples/SAMPLES.md             readable side-by-side samples
  models/<kind>.pt                       best checkpoint per architecture
"""
import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from echo.data import EOS, Vocab, detokenize, tokenize  # noqa: E402
from echo.generate import generate, load_model  # noqa: E402
from echo.metrics import Judge, distinct_n, repetition_rate, self_bleu  # noqa: E402
from echo.models import BONUS, DISPLAY, KINDS  # noqa: E402
from echo.train import device  # noqa: E402

M = ROOT / "results" / "metrics"
S = ROOT / "results" / "samples"


def flatten(summary: dict) -> dict:
    return {
        "run": summary["run"], "model": DISPLAY[summary["kind"]], "kind": summary["kind"], "seed": summary["seed"],
        "hidden_dim": summary["model_cfg"]["hidden_dim"],
        "params_total": summary["params"]["total"], "params_core": summary["params"]["recurrent_core"],
        "epochs_run": summary["epochs_run"], "best_epoch": summary["best_epoch"],
        "val_loss": summary["best_val"]["val_loss"], "val_ppl": summary["best_val"]["val_perplexity"],
        "test_loss": summary["test"]["loss"], "test_ppl": summary["test"]["perplexity"],
        "test_top1": summary["test"]["top1_acc"], "test_top5": summary["test"]["top5_acc"],
        "train_time_min": summary["train_time_s"] / 60, "time_per_epoch_s": summary["time_per_epoch_s"],
        "time_to_best_min": summary["time_to_best_s"] / 60, "ms_per_token": summary["ms_per_token"],
    }


def aggregate_runs() -> pd.DataFrame:
    rows = [flatten(json.loads(p.read_text())) for p in sorted((ROOT / "results" / "runs").glob("*/summary.json"))]
    df = pd.DataFrame(rows)
    plain = df[~df.run.str.startswith(("matched_", "quick_"))]
    main = plain[plain.kind.isin(KINDS)]
    main.to_csv(M / "runs.csv", index=False)
    plain[plain.kind.isin(BONUS)].to_csv(M / "bonus.csv", index=False)  # Transformer: reported separately
    df[df.run.str.startswith("matched_")].to_csv(M / "matched.csv", index=False)

    cols = ["test_loss", "test_ppl", "test_top1", "test_top5", "train_time_min", "time_per_epoch_s",
            "time_to_best_min", "ms_per_token", "best_epoch"]
    agg = main.groupby("kind")[cols].agg(["mean", "std"])
    agg.columns = [f"{c}_{s}" for c, s in agg.columns]
    agg = agg.join(main.groupby("kind")[["params_total", "params_core"]].first())
    agg.insert(0, "model", [DISPLAY[k] for k in agg.index])
    agg = agg.loc[[k for k in KINDS if k in agg.index]]
    agg.to_csv(M / "summary_by_model.csv")
    return main


def pick_best(main: pd.DataFrame) -> dict[str, str]:
    """Best-validation run per architecture -> copied to models/<kind>.pt."""
    best = {}
    (ROOT / "models").mkdir(exist_ok=True)
    for kind, g in main.groupby("kind"):
        run = g.sort_values("val_loss").iloc[0].run
        best[kind] = run
        shutil.copy(ROOT / "results" / "runs" / run / "checkpoint.pt", ROOT / "models" / f"{kind}.pt")
    (M / "best_runs.json").write_text(json.dumps(best, indent=2))
    return best


def available_kinds() -> list[str]:
    return [k for k in (*KINDS, *BONUS) if (ROOT / "models" / f"{k}.pt").exists()]


def run_generation(cfg: dict, vocab: Vocab, dev) -> list[dict]:
    g = cfg["generate"]
    out = []
    for kind in available_kinds():
        model = load_model(ROOT / "models" / f"{kind}.pt", dev)
        for si, seed_text in enumerate(g["seeds_text"]):
            for temp in g["temperatures"]:
                for k in range(g["samples_per_setting"]):
                    gen = torch.Generator().manual_seed(1000 * si + k)  # same RNG stream for every model
                    toks = generate(model, vocab, seed_text, g["length"], temp, cfg["data"]["window"], gen)
                    out.append({"kind": kind, "model": DISPLAY[kind], "seed_id": si, "seed": seed_text,
                                "temperature": temp, "sample": k, "tokens": toks})
        print(f"generated {kind}", flush=True)
    return out


def human_reference(cfg: dict, n: int = 25) -> list[dict]:
    """Real TinyStories continuations scored the same way (a ceiling for comparison)."""
    stories = json.loads((ROOT / "data" / "raw" / "stories.json").read_text())["test"]
    rng = random.Random(0)
    refs = []
    for s in rng.sample(stories, n):
        toks = tokenize(s)
        if len(toks) < 80:
            continue
        refs.append({"seed": " ".join(toks[:14]), "tokens": toks[14 : 14 + cfg["generate"]["length"]]})
    return refs


def score(gens: list[dict], refs: list[dict], judge: Judge, proper: set) -> pd.DataFrame:
    for r in gens + refs:
        r["text"] = detokenize([t for t in r["tokens"] if t != EOS], proper, continuation=True)
        r["seed_display"] = detokenize(tokenize(r["seed"]), proper)
        r["gpt2_ppl"] = judge.gpt2_perplexity(r["seed_display"], r["text"])
        r["repetition"] = repetition_rate(r["tokens"])
    for batch in (gens, refs):
        sims = judge.consistency([r["seed_display"] for r in batch], [r["text"] for r in batch])
        for r, s in zip(batch, sims):
            r["consistency"] = s

    rows = []
    df = pd.DataFrame(gens)
    for (kind, temp), g in df.groupby(["kind", "temperature"], sort=False):
        per_seed = [list(x.tokens) for _, x in g.groupby("seed_id")]
        rows.append({
            "kind": kind, "model": DISPLAY[kind], "temperature": temp,
            "gpt2_ppl": g.gpt2_ppl.median(), "consistency": g.consistency.mean(),
            "distinct_1": distinct_n(list(g.tokens), 1), "distinct_2": distinct_n(list(g.tokens), 2),
            "distinct_3": distinct_n(list(g.tokens), 3),
            "self_bleu": sum(self_bleu(s) for s in per_seed) / len(per_seed),
            "repetition": g.repetition.mean(), "length": g.tokens.map(len).mean(),
        })
    rd = pd.DataFrame(refs)
    rows.append({
        "kind": "human", "model": "Human (TinyStories)", "temperature": float("nan"),
        "gpt2_ppl": rd.gpt2_ppl.median(), "consistency": rd.consistency.mean(),
        "distinct_1": distinct_n(list(rd.tokens), 1), "distinct_2": distinct_n(list(rd.tokens), 2),
        "distinct_3": distinct_n(list(rd.tokens), 3), "self_bleu": float("nan"),
        "repetition": rd.repetition.mean(), "length": rd.tokens.map(len).mean(),
    })
    tq = pd.DataFrame(rows)
    tq.to_csv(M / "text_quality.csv", index=False)
    return tq


def write_samples_md(gens: list[dict], cfg: dict):
    lines = ["# ECHO — generated samples", "",
             "Same seed, same random-number stream, four models. Sample #0 shown for each temperature "
             "(T = 0 is greedy). Full set: `generations.json`.", ""]
    temps = cfg["generate"]["temperatures"]
    by = {(r["kind"], r["seed_id"], r["temperature"], r["sample"]): r for r in gens}
    for si, seed in enumerate(cfg["generate"]["seeds_text"]):
        lines += [f"## Seed {si + 1}: *“{by[('lstm', si, temps[0], 0)]['seed_display']}”*", ""]
        for t in temps:
            lines += [f"### Temperature {t if t > 0 else '0 (greedy)'}", ""]
            for kind in available_kinds():
                r = by[(kind, si, t, 0)]
                tag = " *(bonus)*" if kind in BONUS else ""
                lines += [f"**{DISPLAY[kind]}**{tag} — GPT-2 ppl {r['gpt2_ppl']:.1f}, repetition {r['repetition']:.2f}", "",
                          f"> {r['text']}", ""]
    (S / "SAMPLES.md").write_text("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-device", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())
    M.mkdir(parents=True, exist_ok=True)
    S.mkdir(parents=True, exist_ok=True)

    main = aggregate_runs()
    bonus = pd.read_csv(M / "bonus.csv") if (M / "bonus.csv").stat().st_size > 10 else pd.DataFrame()
    pick_best(pd.concat([main, bonus]))
    vocab = Vocab.load(ROOT / "data" / "processed" / "vocab.json")
    proper = set(json.loads((ROOT / "data" / "processed" / "proper_nouns.json").read_text()))
    dev = device()
    gens = run_generation(cfg, vocab, dev)
    judge = Judge(args.judge_device or str(dev))
    tq = score(gens, human_reference(cfg), judge, proper)
    (S / "generations.json").write_text(json.dumps(gens, indent=1))
    write_samples_md(gens, cfg)
    print(pd.read_csv(M / "summary_by_model.csv").to_string())
    print(tq.round(3).to_string())
