"""Data pipeline: download TinyStories, clean, tokenize, build vocabulary,
and turn the token stream into (window -> next word) training pairs.

Autoregressive framing
----------------------
Given the word stream w_1 ... w_N and a window size W, every position t yields

    input  x_t = (w_t, ..., w_{t+W-1})      # W words of context
    target y_t = w_{t+W}                     # the word that comes next

The target is never part of the input, which is what lets the Bidirectional
RNN read its context both ways without seeing the answer.
"""
from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch

PAD, UNK, EOS = "<pad>", "<unk>", "<eos>"
SPECIALS = [PAD, UNK, EOS]

_TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?|[0-9]+|[.,!?;:\"]")
_QUOTES = {"‘": "'", "’": "'", "“": '"', "”": '"', "—": " , ", "–": " , "}


def clean(text: str) -> str:
    """Normalise unicode quotes/dashes and lowercase."""
    for a, b in _QUOTES.items():
        text = text.replace(a, b)
    text = unicodedata.normalize("NFKC", text)
    return text.lower()


def tokenize(text: str) -> list[str]:
    """Words (keeping contractions like don't) and sentence punctuation."""
    return _TOKEN_RE.findall(clean(text))


def detokenize(tokens: list[str], proper: set[str] | frozenset = frozenset(), continuation: bool = False) -> str:
    """Join tokens back into readable text (``proper`` = words to capitalise, e.g. names)."""
    out, quote_open = "", False
    for tok in tokens:
        if tok == EOS:
            out = out.rstrip() + "\n\n"
            continue
        if tok == '"':
            if quote_open:
                out = out.rstrip() + '"'
            else:
                out += ('' if out.endswith(("\n", " ")) or not out else ' ') + '"'
            quote_open = not quote_open
            continue
        if tok in ".,!?;:":
            out = out.rstrip() + tok
        else:
            out += ("" if not out or out.endswith(("\n", '"')) and quote_open else " ") + tok
    # Capitalise names, "I" and sentence starts for display.
    out = re.sub(r"\b[a-z]+\b", lambda m: m.group(0).capitalize() if m.group(0) in proper else m.group(0), out)
    out = re.sub(r"\bi\b(?!')|\bi(?='[a-z])", "I", out)
    start = "" if continuation else "^|"  # a continuation does not begin a sentence
    pattern = "(" + start + r'[.!?]"?\s+"?|\n\s*"?|[,:]\s")([a-z])'
    return re.sub(pattern, lambda m: m.group(1) + m.group(2).upper(), out.strip())


def find_proper_nouns(stories: list[str], min_count: int = 20) -> list[str]:
    """Words that are almost always capitalised mid-sentence (names like Lily, Tom)."""
    caps, total = Counter(), Counter()
    for s in stories:
        for m in re.finditer(r"(?<=[a-z,] )([A-Za-z]+)", s):
            w = m.group(1)
            total[w.lower()] += 1
            caps[w.lower()] += w[0].isupper()
    return sorted(w for w, n in total.items() if n >= min_count and caps[w] / n > 0.9 and w != "i")


@dataclass
class Vocab:
    itos: list[str]

    def __post_init__(self):
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens: list[str]) -> list[int]:
        unk = self.stoi[UNK]
        return [self.stoi.get(t, unk) for t in tokens]

    def decode(self, ids) -> list[str]:
        return [self.itos[int(i)] for i in ids]

    @classmethod
    def build(cls, streams: list[list[str]], size: int) -> "Vocab":
        counts = Counter(t for s in streams for t in s)
        words = [w for w, _ in counts.most_common(size - len(SPECIALS))]
        return cls(SPECIALS + words)

    def save(self, path: Path):
        path.write_text(json.dumps(self.itos))

    @classmethod
    def load(cls, path: Path) -> "Vocab":
        return cls(json.loads(Path(path).read_text()))


def download_stories(cfg: dict, raw_dir: Path) -> dict[str, list[str]]:
    """Fetch the configured number of stories from the official splits (cached)."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache = raw_dir / "stories.json"
    if cache.exists():
        return json.loads(cache.read_text())

    from datasets import load_dataset

    ds = load_dataset(cfg["dataset"], streaming=True)
    take = lambda split, n: [r["text"] for _, r in zip(range(n), ds[split])]
    train = take("train", cfg["train_stories"])
    held = take("validation", cfg["val_stories"] + cfg["test_stories"])
    splits = {"train": train, "val": held[: cfg["val_stories"]], "test": held[cfg["val_stories"]:]}
    cache.write_text(json.dumps(splits))
    return splits


def stories_to_tokens(stories: list[str]) -> list[str]:
    """Concatenate stories into one stream, separated by <eos>."""
    out: list[str] = []
    for s in stories:
        toks = tokenize(s)
        if toks:
            out.extend(toks)
            out.append(EOS)
    return out


def prepare(cfg: dict, root: Path) -> dict:
    """Run the full pipeline and save tensors + vocab + statistics."""
    splits = download_stories(cfg, root / "data" / "raw")
    streams = {k: stories_to_tokens(v) for k, v in splits.items()}
    vocab = Vocab.build([streams["train"]], cfg["vocab_size"])

    out_dir = root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    vocab.save(out_dir / "vocab.json")
    (out_dir / "proper_nouns.json").write_text(json.dumps(find_proper_nouns(splits["train"])))
    for k, toks in streams.items():
        torch.save(torch.tensor(vocab.encode(toks), dtype=torch.long), out_dir / f"{k}.pt")

    counts = Counter(streams["train"])
    covered = sum(c for w, c in counts.items() if w in vocab.stoi)
    lengths = [len(tokenize(s)) for s in splits["train"]]
    stats = {
        "stories": {k: len(v) for k, v in splits.items()},
        "tokens": {k: len(v) for k, v in streams.items()},
        "unique_words_train": len(counts),
        "vocab_size": len(vocab),
        "train_coverage": covered / len(streams["train"]),
        "unk_rate": {k: sum(t not in vocab.stoi for t in v) / len(v) for k, v in streams.items()},
        "words_per_story": {
            "mean": sum(lengths) / len(lengths),
            "min": min(lengths),
            "max": max(lengths),
        },
        "top_words": counts.most_common(30),
        "window": cfg["window"],
        "examples": {k: len(v) - cfg["window"] for k, v in streams.items()},
    }
    (root / "results" / "metrics").mkdir(parents=True, exist_ok=True)
    (root / "results" / "metrics" / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    return stats


class WindowBatcher:
    """Fast, GPU-side batching of (window -> next word) pairs.

    Rather than materialising millions of windows, we keep the token stream on
    the device and gather windows by index each step.
    """

    def __init__(self, ids: torch.Tensor, window: int, batch_size: int, device, shuffle: bool, seed: int = 0,
                 samples_per_epoch: int | None = None):
        self.ids = ids.to(device)
        self.window, self.batch_size, self.shuffle = window, batch_size, shuffle
        self.n_total = len(ids) - window
        # Each epoch draws a fresh random subset of windows (all windows if None).
        self.n = min(samples_per_epoch or self.n_total, self.n_total)
        self.offsets = torch.arange(window, device=device)
        self.gen = torch.Generator().manual_seed(seed)

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        order = torch.randperm(self.n_total, generator=self.gen)[: self.n] if self.shuffle else torch.arange(self.n)
        order = order.to(self.ids.device)
        for b in range(0, self.n, self.batch_size):
            start = order[b : b + self.batch_size]
            x = self.ids[start[:, None] + self.offsets]
            y = self.ids[start + self.window]
            yield x, y
