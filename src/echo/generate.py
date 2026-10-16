"""Autoregressive text generation with temperature sampling.

At every step the model sees the last W words, produces a probability for
every word in the vocabulary, we pick one, append it, slide the window, repeat.

Temperature T rescales the model's scores before sampling:
    p_i  proportional to  exp(logit_i / T)
  T -> 0   always pick the most likely word (greedy): safe, repetitive
  T = 1    sample from the model's own distribution
  T > 1    flatten the distribution: more surprising, less coherent
"""
from __future__ import annotations

from pathlib import Path

import torch

from .data import EOS, PAD, UNK, Vocab, tokenize
from .models import RecurrentLM, build


def load_model(ckpt_path: Path, device) -> RecurrentLM:
    ck = torch.load(ckpt_path, map_location=device)
    model = build(ck["kind"], ck["vocab_size"], ck["model_cfg"]).to(device)
    model.load_state_dict(ck["state_dict"])
    return model.eval()


@torch.no_grad()
def generate(model: RecurrentLM, vocab: Vocab, seed_text: str, length: int, temperature: float,
             window: int, generator: torch.Generator | None = None) -> list[str]:
    """Return the generated continuation tokens (seed excluded)."""
    dev = next(model.parameters()).device
    # A story start in training always follows an <eos>, so prepend one.
    ids = [vocab.stoi[EOS]] + vocab.encode(tokenize(seed_text))
    banned = torch.tensor([vocab.stoi[PAD], vocab.stoi[UNK]], device=dev)
    out: list[int] = []
    for _ in range(length):
        ctx = ids[-window:]  # recurrent nets accept shorter windows as-is
        logits = model(torch.tensor([ctx], device=dev))[0]
        logits[banned] = float("-inf")
        if temperature <= 0:
            nxt = int(logits.argmax())
        else:
            probs = torch.softmax(logits / temperature, dim=-1).cpu()
            nxt = int(torch.multinomial(probs, 1, generator=generator))
        ids.append(nxt)
        out.append(nxt)
        if nxt == vocab.stoi[EOS]:
            break
    return vocab.decode(out)
