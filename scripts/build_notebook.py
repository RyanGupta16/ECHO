"""Generates notebooks/ECHO_walkthrough.ipynb (then executed with nbconvert)."""
import json
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = json.loads((ROOT / "docs" / "analysis.json").read_text()) if (ROOT / "docs" / "analysis.json").exists() else {}

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md("""
# ECHO — Autoregressive Text Generation with Simple RNN, Bidirectional RNN, LSTM and GRU

**Foundations of Generative AI (ICT4403) · Assignment 2** · Ryan Gupta (230953288)

**Problem.** Build a word-level autoregressive text generator — a model that predicts the next word from the words
before it and writes text by feeding its own predictions back in — and compare four recurrent architectures trained
under identical conditions.

**Objectives**
1. Describe and preprocess a text dataset; build input–target sequences for next-word prediction. *(Task i)*
2. Train Simple RNN, Bidirectional RNN, LSTM and GRU under comparable conditions. *(Task i)*
3. Compare them on loss, perplexity, accuracy and training time, and on the coherence, diversity, contextual
   consistency and repetition of their generated text. *(Task ii)*
4. Generate from the same seeds at several sampling temperatures, analyse the effect, and justify the best model. *(Task iii)*

> This notebook is the narrative front-end. The heavy lifting lives in `src/echo/` and `scripts/`; every cell below
> calls that code, so the numbers here are the numbers in the README.
""")

code("""
import sys, json, math
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

import yaml, torch, pandas as pd
from IPython.display import Image, Markdown, display
from echo.data import Vocab, tokenize, detokenize, EOS
from echo.models import build, count_params, KINDS, DISPLAY
from echo.generate import load_model, generate
from echo.train import device

pd.set_option("display.precision", 3)
cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())
M = ROOT / "results" / "metrics"; P = ROOT / "results" / "plots"
print("device:", device(), "| torch", torch.__version__)
cfg
""")

md("""
## 1 · Dataset — TinyStories *(Task i)*

**Source:** [roneneldan/TinyStories on Hugging Face](https://huggingface.co/datasets/roneneldan/TinyStories) ·
paper: [Eldan & Li, 2023](https://arxiv.org/abs/2305.07759).

TinyStories is a synthetic corpus of short stories written (by GPT-3.5/4) using only words a typical 3–4-year-old
understands. It was designed precisely so that *small* language models can learn grammar, consistency and simple
reasoning — which makes it an ideal benchmark for compact recurrent networks.

We use the first 30,000 stories of the official **train** split for training, and two disjoint slices of 2,500
stories each from the official **validation** split for validation and test.
""")

code("""
stats = json.loads((M / "dataset_stats.json").read_text())
pd.DataFrame({
    "stories": stats["stories"], "tokens": stats["tokens"],
    "(window → next) pairs": stats["examples"], "<unk> rate": stats["unk_rate"],
}).T
""")
code("""
print(f"unique words in train: {stats['unique_words_train']:,}")
print(f"vocabulary: {stats['vocab_size']:,} (covers {stats['train_coverage']:.2%} of training tokens)")
print("words per story:", {k: round(v, 1) for k, v in stats['words_per_story'].items()})
display(Image(P / "01_dataset.png"))
""")

md("""
### Preprocessing
1. **Normalise** unicode quotes/dashes and lowercase everything.
2. **Tokenise** into words (keeping contractions such as *don't*) and the punctuation marks `. , ! ? ; : "`,
   so the models learn sentence structure and dialogue.
3. **Join** stories into one stream separated by an `<eos>` token, so models also learn where stories end.
4. **Vocabulary** = 8,000 most frequent training tokens + `<pad>`, `<unk>`, `<eos>`; rarer words map to `<unk>`.
""")
code("""
raw = json.loads((ROOT / "data" / "raw" / "stories.json").read_text())["train"][3]
print(raw[:300], "...\\n")
toks = tokenize(raw)
print(toks[:40])
vocab = Vocab.load(ROOT / "data" / "processed" / "vocab.json")
print("ids:", vocab.encode(toks[:12]))
""")

md("""
### Input–target sequences for autoregressive modelling
With a context window of **W = 24 tokens**, every position *t* of the token stream gives one training pair:

$$x_t = (w_t,\\dots,w_{t+W-1}) \\quad\\longrightarrow\\quad y_t = w_{t+W}$$

This *many-to-one* framing is used for **all four** models. It matters for the Bidirectional RNN: because the
target lies strictly *after* the window, reading the window backwards never reveals the answer (no future leakage).
A per-timestep "shifted sequence" framing would leak the target to the backward RNN.

Windows are gathered on the GPU from the token stream on the fly (6.2M pairs never need to be materialised).
""")
code("""
from echo.data import WindowBatcher
ids = torch.load(ROOT / "data" / "processed" / "train.pt")
b = WindowBatcher(ids[:5000], cfg["data"]["window"], 3, "cpu", shuffle=False)
x, y = next(iter(b))
for xi, yi in zip(x, y):
    print(" ".join(vocab.decode(xi)), " ──▶ ", vocab.itos[yi])
""")

md("""
## 2 · Models *(Task i)*

All four share one skeleton — only the recurrent cell changes:

`word ids → Embedding(128) → 2-layer recurrent core (hidden 256, dropout 0.25) → last state → Linear → 8,000-way softmax`

| Model | Cell | Context vector |
|---|---|---|
| Simple RNN | $h_t=\\tanh(W_x x_t + W_h h_{t-1} + b)$ | final hidden state |
| Bidirectional RNN | two tanh RNNs, left→right and right→left over the window | $[\\overrightarrow{h}_W ; \\overleftarrow{h}_1]$ |
| LSTM | input/forget/output gates + cell state $c_t$ | final hidden state |
| GRU | update/reset gates, no separate cell | final hidden state |
""")
code("""
rows = []
for k in KINDS:
    m = build(k, len(vocab), cfg["model"])
    rows.append({"model": DISPLAY[k], **count_params(m)})
print(build("lstm", len(vocab), cfg["model"]))
pd.DataFrame(rows).set_index("model")
""")
code("""
# Why many-to-one matters for the BiRNN — an empirical leakage test.
# In a per-timestep ("shifted sequence") setup, the output at position t is trained to predict word t+1.
# Perturb word t+1 and check whether the output at position t changes: if it does, the model can see its answer.
torch.manual_seed(0)
seq = torch.randint(3, len(vocab), (1, 24))
perturbed = seq.clone(); perturbed[0, 11] = (seq[0, 11] + 1) % len(vocab)    # change the 'answer' for position 10
for k in ["rnn", "birnn"]:
    m = build(k, len(vocab), cfg["model"]).eval()
    with torch.no_grad():
        out_a, _ = m.core(m.embed(seq)); out_b, _ = m.core(m.embed(perturbed))
    leak = not torch.allclose(out_a[0, 10], out_b[0, 10])
    print(f"{DISPLAY[k]:>18}: output at position 10 changes when word 11 changes → leakage = {leak}")
print("\\nECHO therefore predicts only the word AFTER the whole window (many-to-one), so no model ever sees its target.")
""")

md("""
## 3 · Training under comparable conditions *(Task i)*

| Setting | Value (identical for every model) |
|---|---|
| Optimiser | AdamW, lr 2e-3, weight decay 0.01, one-cycle schedule |
| Batch / window | 512 pairs / 24 tokens |
| Epoch | a fresh random 2M-pair sample of the 6.2M training pairs |
| Epochs | up to 8, early stopping (patience 2) on validation loss |
| Regularisation | dropout 0.25, gradient-norm clipping at 1.0 |
| Seeds | 13, 42, 2026 — every model trained 3× |
| Hardware | Apple M4 (MPS backend) |

Training all 12 runs takes several hours, so by default this notebook **loads the saved run logs**. Set
`RUN_TRAINING = True` to retrain from scratch (identical code path: `scripts/train_all.py`).
""")
code("""
RUN_TRAINING = False
if RUN_TRAINING:
    from echo.train import train_one
    for seed in cfg["train"]["seeds"]:
        for k in KINDS:
            train_one(k, seed, cfg, ROOT)
runs = pd.read_csv(M / "runs.csv")
runs[["model", "seed", "epochs_run", "best_epoch", "test_loss", "test_ppl", "test_top1", "test_top5", "train_time_min"]]
""")
code("display(Image(P / '02_learning_curves.png'))")

md("## 4 · Quantitative evaluation *(Task ii)*")
code("""
summary = pd.read_csv(M / "summary_by_model.csv", index_col=0)
cols = ["test_loss", "test_ppl", "test_top1", "test_top5", "time_per_epoch_s", "ms_per_token"]
table = pd.DataFrame({c: summary[f"{c}_mean"].map("{:.3f}".format) + " ± " + summary[f"{c}_std"].map("{:.3f}".format) for c in cols})
table.insert(0, "params", summary["params_total"])
table.index = summary["model"]
table
""")
code("display(Image(P / '03_test_metrics.png'))")
code("""
speed = M / "speed_benchmark.csv"
if speed.exists():
    display(Markdown("**Controlled speed benchmark** (all models timed back-to-back on an idle machine):"))
    display(pd.read_csv(speed).set_index("model")[["total", "recurrent_core", "train_ms_per_batch", "infer_ms_per_word", "est_min_per_epoch"]])
""")
code("""
matched = M / "matched.csv"
if matched.exists() and matched.stat().st_size > 10:
    display(Markdown("**Parameter-matched variant** — hidden size re-tuned so every model has ≈ the LSTM's parameter count:"))
    display(pd.read_csv(matched)[["model", "hidden_dim", "params_total", "test_ppl", "test_top1"]])
    display(Image(P / "06_parameter_matched.png"))
""")
md(ANALYSIS.get("quantitative", "*Quantitative analysis — see README.*"))

md("""
## 5 · Generation and text quality *(Tasks ii & iii)*

For each model's best checkpoint we generate 60 tokens from **5 seed sequences** at **5 temperatures**
(greedy, 0.5, 0.8, 1.0, 1.3), 5 samples each. Every model draws from **the same random-number stream**, so differences
are due to the model, not luck. Text quality is then scored automatically:

| Quality | Metric |
|---|---|
| Coherence | perplexity of the continuation under **GPT-2** (an independent, much larger model) — lower = more natural |
| Contextual consistency | cosine similarity of **MiniLM** sentence embeddings of seed vs continuation |
| Diversity | distinct-1/2/3 (unique n-grams ÷ all n-grams); self-BLEU between samples of the same prompt |
| Repetition | share of repeated 3-grams within a sample |
""")
code("""
proper = set(json.loads((ROOT / "data" / "processed" / "proper_nouns.json").read_text()))
dev = device()
models = {k: load_model(ROOT / "models" / f"{k}.pt", dev) for k in KINDS}
seed_text = cfg["generate"]["seeds_text"][0]
print("SEED:", detokenize(tokenize(seed_text), proper), "\\n")
for T in cfg["generate"]["temperatures"]:
    print(f"── temperature {T if T > 0 else '0 (greedy)'} " + "─" * 50)
    for k, m in models.items():
        g = torch.Generator().manual_seed(0)
        out = generate(m, vocab, seed_text, 40, T, cfg["data"]["window"], g)
        print(f"{DISPLAY[k]:>18}: {detokenize(out, proper, continuation=True)}")
    print()
""")
code("""
tq = pd.read_csv(M / "text_quality.csv")
tq.pivot_table(index="model", columns="temperature", values=["gpt2_ppl", "distinct_2", "repetition", "consistency", "self_bleu"]).round(3)
""")
code("display(Image(P / '04_temperature_heatmaps.png')); display(Image(P / '05_temperature_frontier.png'))")
code("""
# Human reference row: real TinyStories continuations scored by the same judges.
tq[tq.kind == "human"]
""")
md(ANALYSIS.get("qualitative", "*Qualitative analysis — see README.*"))
md(ANALYSIS.get("temperature", "*Temperature analysis — see README.*"))

md("## 6 · Conclusion")
md(ANALYSIS.get("conclusion", "*Conclusion — see README.*"))

nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = ROOT / "notebooks" / "ECHO_walkthrough.ipynb"
out.parent.mkdir(exist_ok=True)
nbf.write(nb, out)
print("wrote", out)
