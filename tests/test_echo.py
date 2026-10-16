"""Sanity tests for the ECHO pipeline: data integrity, no target leakage, and complete training.

Run:  .venv/bin/python -m pytest -q tests
"""
import json
import sys
from pathlib import Path

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from echo.data import EOS, Vocab, WindowBatcher, detokenize, tokenize  # noqa: E402
from echo.metrics import distinct_n, repetition_rate, self_bleu  # noqa: E402
from echo.models import BONUS, KINDS, build  # noqa: E402

CFG = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())
PROC, RUNS = ROOT / "data" / "processed", ROOT / "results" / "runs"
needs_data = pytest.mark.skipif(not (PROC / "train.pt").exists(), reason="run scripts/prepare_data.py first")
needs_runs = pytest.mark.skipif(not RUNS.exists(), reason="no training runs")


# ---------- tokenisation & vocabulary ----------

def test_tokenize_roundtrip():
    toks = tokenize("Once upon a time, Lily didn't want to go!")
    assert toks == ["once", "upon", "a", "time", ",", "lily", "didn't", "want", "to", "go", "!"]
    assert detokenize(toks, proper={"lily"}).startswith("Once upon a time, Lily")


@needs_data
def test_vocab_size_and_specials():
    vocab = Vocab.load(PROC / "vocab.json")
    assert len(vocab) == CFG["data"]["vocab_size"]
    assert EOS in vocab.stoi
    assert vocab.decode(vocab.encode(["the", "dog"])) == ["the", "dog"]


@needs_data
def test_splits_are_disjoint():
    raw = ROOT / "data" / "raw" / "stories.json"
    if not raw.exists():
        pytest.skip("raw stories not cached")
    s = json.loads(raw.read_text())
    assert len(s["train"]) == CFG["data"]["train_stories"]
    assert len(s["val"]) == CFG["data"]["val_stories"] and len(s["test"]) == CFG["data"]["test_stories"]
    assert not set(s["val"]) & set(s["test"]), "val/test overlap"
    assert not set(s["train"]) & set(s["test"]), "train/test overlap"


# ---------- input–target construction ----------

def test_window_batcher_targets_follow_window():
    ids = torch.arange(100)
    W = CFG["data"]["window"]
    x, y = next(iter(WindowBatcher(ids, W, 8, "cpu", shuffle=True, seed=0)))
    assert x.shape == (8, W)
    # Target is exactly the word after the window, and never inside it (no leakage).
    assert torch.equal(y, x[:, -1] + 1)
    assert not (x == y[:, None]).any()


# ---------- models ----------

@pytest.mark.parametrize("kind", [*KINDS, *BONUS])
def test_model_output_shape(kind):
    torch.manual_seed(0)
    m = build(kind, 500, CFG["model"]).eval()
    assert m(torch.randint(3, 500, (4, CFG["data"]["window"]))).shape == (4, 500)


@pytest.mark.parametrize("kind", ["rnn", "lstm", "gru", "tfm"])
def test_causal_models_ignore_nothing_after_last_word(kind):
    """Changing the earliest word must be allowed to matter; the model must predict from the window only."""
    torch.manual_seed(0)
    m = build(kind, 500, CFG["model"]).eval()
    x = torch.randint(3, 500, (1, CFG["data"]["window"]))
    with torch.no_grad():
        a = m(x)
        x2 = x.clone()
        x2[0, 0] = (x2[0, 0] + 1) % 500 + 3
        b = m(x2)
    assert not torch.allclose(a, b), "first word has no influence — context not used"


# ---------- text metrics ----------

def test_text_metrics():
    rep = "the dog ran the dog ran the dog ran".split()
    assert repetition_rate(rep, 3) > 0.5
    assert repetition_rate("a b c d e f".split(), 3) == 0
    assert distinct_n([["a", "b"], ["a", "c"]], 1) == pytest.approx(3 / 4)
    assert self_bleu([["a", "b", "c", "d"]] * 3) > 0.99


# ---------- training completeness ----------

EXPECTED = [f"{k}_s{s}" for k in KINDS for s in CFG["train"]["seeds"]] + [f"matched_{k}_s13" for k in KINDS] + ["tfm_s13"]


@needs_runs
@pytest.mark.parametrize("run", EXPECTED)
def test_run_completed(run):
    path = RUNS / run / "summary.json"
    assert path.exists(), f"{run} has not finished training"
    s = json.loads(path.read_text())
    hist = json.loads((RUNS / run / "history.json").read_text())
    assert s["epochs_run"] >= 1 and len(hist) == s["epochs_run"]
    t = s["test"]
    assert 1 < t["perplexity"] < CFG["data"]["vocab_size"]
    assert t["perplexity"] == pytest.approx(torch.tensor(t["loss"]).exp().item(), rel=1e-4)
    assert 0 < t["top1_acc"] <= t["top5_acc"] < 1
