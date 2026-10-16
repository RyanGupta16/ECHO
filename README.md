# ECHO — Autoregressive Text Generation with RNN, BiRNN, LSTM and GRU

**Foundations of Generative AI (ICT4403) · Assignment 2** · Ryan Gupta (230953288)

**Live demo:** https://ryangupta16.github.io/ECHO/ — all five trained models run in your browser (ONNX Runtime Web), no server.

A word-level next-word generator trained on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories),
comparing four recurrent architectures under identical conditions (same data, vocabulary, window, optimiser,
schedule, 3 seeds each), plus a parameter-matched rerun and a small Transformer as a bonus reference.

## Results (test set, mean ± std over 3 seeds)

| Model | Params | Test perplexity ↓ | Top-1 acc | Top-5 acc |
|---|---:|---:|---:|---:|
| Simple RNN | 3.31 M | 23.80 ± 0.11 | 36.6 % | 62.1 % |
| Bidirectional RNN | 5.72 M | 23.87 ± 0.16 | 37.4 % | 62.4 % |
| LSTM | 4.00 M | 17.15 ± 0.04 | 41.5 % | 67.0 % |
| **GRU** | 3.77 M | **15.97 ± 0.06** | **42.4 %** | **68.3 %** |
| *Transformer (bonus, 1 seed)* | 3.34 M | *12.70* | *45.8 %* | *72.1 %* |

With every model matched to ~4.0 M parameters the ranking is unchanged (GRU 15.9 < LSTM 17.2 < RNN 23.5 ≈ BiRNN 23.5),
so GRU's lead is due to the architecture rather than its size. The BiRNN gains nothing because the backward direction cannot see the
future word being predicted; it only re-reads the same past window.

Generated-text quality is scored with GPT-2 perplexity (coherence), MiniLM seed–continuation similarity (contextual
consistency), distinct-n / self-BLEU (diversity) and 3-gram repetition. Temperature 0.5–0.8 gives the best balance:
greedy decoding (T=0) loops, and T≥1.3 breaks down into noise. Full tables: `results/metrics/`, samples:
[`results/samples/SAMPLES.md`](results/samples/SAMPLES.md), figures: `results/plots/`.

**Verdict:** GRU is the best of the four recurrent models: lowest perplexity, highest accuracy, and the best coherence among
the recurrent models at T = 0.5 (GPT-2 perplexity 10.5 vs 13.5 for LSTM).

## Repository layout

```
configs/default.yaml      shared experimental configuration (comparable conditions)
src/echo/                 data, models, training, generation, metrics, ONNX export
scripts/                  pipeline steps (prepare → train → evaluate → benchmark → plots → export → notebook)
tests/                    sanity tests (no target leakage, disjoint splits, all runs complete)
notebooks/                executed walkthrough notebook
results/                  metrics CSVs, generated samples, plots, training logs
models/                   best checkpoint per architecture (.pt)
web/                      Next.js live demo (static export → GitHub Pages)
docs/submission/          report (.docx / .pdf)
```

## Reproduce

```bash
python -m venv .venv && .venv/bin/pip install torch pandas pyyaml datasets transformers sentence-transformers \
    matplotlib onnx onnxruntime nbformat nbconvert ipykernel pytest
.venv/bin/python scripts/prepare_data.py          # download + tokenise TinyStories
scripts/run_pipeline.sh                           # 12 main runs + 4 parameter-matched + Transformer (resumable)
.venv/bin/python scripts/evaluate_all.py          # metrics, generation, text-quality scores
.venv/bin/python scripts/benchmark_speed.py       # controlled speed benchmark
.venv/bin/python scripts/make_plots.py
.venv/bin/python scripts/export_web.py            # ONNX models + data for the web demo (verifies parity)
.venv/bin/python scripts/build_notebook.py && .venv/bin/jupyter nbconvert --execute --inplace notebooks/ECHO_walkthrough.ipynb
.venv/bin/python -m pytest -q tests
```

Web demo locally: `cd web && npm ci && npm run dev`. Publish to GitHub Pages: `scripts/deploy_pages.sh`
(builds the static site and pushes it to the `gh-pages` branch).
