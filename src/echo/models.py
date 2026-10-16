"""The four autoregressive language models compared in ECHO.

All four share the same skeleton so that the recurrent cell is the *only*
thing that changes:

    word ids (B, W) -> Embedding -> recurrent core -> context vector -> Linear -> logits over vocab

  rnn    Simple (Elman) RNN, tanh cell
  birnn  Bidirectional Simple RNN: one RNN reads the window left->right,
         another right->left; their final states are concatenated.
         It only ever sees the W context words, never the target word.
  lstm   Long Short-Term Memory: input / forget / output gates + cell state
  gru    Gated Recurrent Unit: update / reset gates, no separate cell state

Bonus contender (outside the assignment's four, trained under the same data,
window, optimiser and budget):
  tfm    small decoder-only Transformer (GPT-style causal self-attention)
"""
from __future__ import annotations

import torch
from torch import nn

KINDS = ("rnn", "birnn", "lstm", "gru")
BONUS = ("tfm",)
DISPLAY = {"rnn": "Simple RNN", "birnn": "Bidirectional RNN", "lstm": "LSTM", "gru": "GRU", "tfm": "Transformer"}


class RecurrentLM(nn.Module):
    def __init__(self, kind: str, vocab_size: int, embed_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        assert kind in KINDS, kind
        self.kind = kind
        self.bidirectional = kind == "birnn"
        self.hidden_dim = hidden_dim

        self.embed = nn.Embedding(vocab_size, embed_dim)
        rnn_cls = {"rnn": nn.RNN, "birnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[kind]
        kwargs = dict(nonlinearity="tanh") if rnn_cls is nn.RNN else {}
        self.core = rnn_cls(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=self.bidirectional,
            **kwargs,
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim * (2 if self.bidirectional else 1), vocab_size)

    def context(self, x: torch.Tensor) -> torch.Tensor:
        """Summarise a window of word ids into one vector."""
        out, _ = self.core(self.drop(self.embed(x)))
        if not self.bidirectional:
            return out[:, -1]
        h = self.hidden_dim
        # Forward direction has read the whole window at the last step;
        # backward direction has read the whole window at the first step.
        return torch.cat([out[:, -1, :h], out[:, 0, h:]], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.drop(self.context(x)))


class Block(nn.Module):
    """Pre-LayerNorm Transformer block with causal self-attention (plain ops, ONNX-friendly)."""

    def __init__(self, d: int, heads: int, ff: int, dropout: float):
        super().__init__()
        self.heads = heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        att = (q @ k.transpose(-1, -2)) / (D // self.heads) ** 0.5
        att = att.masked_fill(mask[:T, :T], float("-inf")).softmax(-1)
        y = (self.drop(att) @ v).transpose(1, 2).reshape(B, T, D)
        x = x + self.drop(self.proj(y))
        return x + self.drop(self.ff(self.ln2(x)))


class TransformerLM(nn.Module):
    """GPT-style next-word model with the same window -> next-word interface as RecurrentLM."""

    def __init__(self, vocab_size: int, d_model: int, layers: int, heads: int, ff: int, dropout: float, max_len: int = 64):
        super().__init__()
        self.kind = "tfm"
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.core = nn.ModuleList(Block(d_model, heads, ff, dropout) for _ in range(layers))
        self.ln = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, vocab_size)
        self.head.weight = self.embed.weight  # weight tying
        self.register_buffer("mask", torch.triu(torch.ones(max_len, max_len, dtype=torch.bool), 1), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        h = self.drop(self.embed(x) + self.pos(torch.arange(T, device=x.device)))
        for block in self.core:
            h = block(h, self.mask)
        return self.head(self.ln(h[:, -1]))


def build(kind: str, vocab_size: int, mcfg: dict) -> nn.Module:
    if kind == "tfm":
        return TransformerLM(vocab_size, **mcfg["transformer"])
    return RecurrentLM(kind, vocab_size, mcfg["embed_dim"], mcfg["hidden_dim"], mcfg["num_layers"], mcfg["dropout"])


def count_params(model: nn.Module) -> dict[str, int]:
    core = sum(p.numel() for p in model.core.parameters())
    total = sum(p.numel() for p in model.parameters())  # tied weights are counted once
    return {"total": total, "recurrent_core": core}


def matched_hidden(kind: str, vocab_size: int, mcfg: dict, target_total: int) -> int:
    """Hidden size that makes `kind` roughly match `target_total` parameters."""
    best, best_gap = mcfg["hidden_dim"], float("inf")
    for h in range(32, 1025, 4):
        n = count_params(build(kind, vocab_size, {**mcfg, "hidden_dim": h}))["total"]
        if abs(n - target_total) < best_gap:
            best, best_gap = h, abs(n - target_total)
    return best
