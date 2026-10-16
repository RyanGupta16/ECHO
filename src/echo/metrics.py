"""Qualitative-text metrics, made quantitative.

  coherence              GPT-2 perplexity of the continuation given the seed
                         (an independent, much larger "judge" model; lower = more natural)
  contextual consistency cosine similarity between sentence embeddings of the
                         seed and the continuation (higher = stays on topic)
  diversity              distinct-n: unique n-grams / all n-grams across samples
                         self-BLEU: how alike samples from the same prompt are (lower = more diverse)
  repetition             share of repeated 3-grams inside a sample (higher = loopier)
"""
from __future__ import annotations

import math
from collections import Counter

import torch


def ngrams(tokens: list[str], n: int) -> list[tuple]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def distinct_n(samples: list[list[str]], n: int) -> float:
    grams = [g for s in samples for g in ngrams(s, n)]
    return len(set(grams)) / max(len(grams), 1)


def repetition_rate(tokens: list[str], n: int = 3) -> float:
    grams = ngrams(tokens, n)
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


def _bleu(hyp: list[str], refs: list[list[str]], max_n: int = 4) -> float:
    """Sentence BLEU-4 with add-one smoothing (Lin & Och, 2004)."""
    if not hyp:
        return 0.0
    log_p = 0.0
    for n in range(1, max_n + 1):
        h = Counter(ngrams(hyp, n))
        max_ref = Counter()
        for r in refs:
            for g, c in Counter(ngrams(r, n)).items():
                max_ref[g] = max(max_ref[g], c)
        match = sum(min(c, max_ref[g]) for g, c in h.items())
        total = max(sum(h.values()), 0)
        log_p += math.log((match + 1) / (total + 1)) / max_n
    ref_len = min((abs(len(r) - len(hyp)), len(r)) for r in refs)[1]
    bp = 1.0 if len(hyp) > ref_len else math.exp(1 - ref_len / max(len(hyp), 1))
    return bp * math.exp(log_p)


def self_bleu(samples: list[list[str]]) -> float:
    if len(samples) < 2:
        return 0.0
    return sum(_bleu(s, samples[:i] + samples[i + 1 :]) for i, s in enumerate(samples)) / len(samples)


class Judge:
    """GPT-2 coherence scorer + MiniLM topic-consistency scorer."""

    def __init__(self, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast

        self.device = device
        self.tok = GPT2TokenizerFast.from_pretrained("gpt2")
        self.lm = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
        self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

    @torch.no_grad()
    def gpt2_perplexity(self, seed: str, continuation: str) -> float:
        ctx = self.tok(seed, return_tensors="pt").input_ids
        cont = self.tok(" " + continuation.strip(), return_tensors="pt").input_ids
        ids = torch.cat([ctx, cont], dim=1)[:, :1024].to(self.device)
        labels = ids.clone()
        labels[:, : ctx.shape[1]] = -100  # score only the continuation
        loss = self.lm(ids, labels=labels).loss
        return math.exp(loss.item())

    def consistency(self, seeds: list[str], continuations: list[str]) -> list[float]:
        a = self.embedder.encode(seeds, convert_to_tensor=True, normalize_embeddings=True)
        b = self.embedder.encode(continuations, convert_to_tensor=True, normalize_embeddings=True)
        return (a * b).sum(-1).tolist()
