"""Export trained models to ONNX so the ECHO web lab can run them in the browser.

  <kind>.onnx      ids (1, T) int64 -> logits (1, V)          for generation
  lstm_probe.onnx  ids (1, T) int64 -> per-step logits (T, V),
                   gates (L, T, 4, H)  [input, forget, cell-candidate, output],
                   hidden (L, T, H)                             for the "inside the brain" view

Large weight tensors are stored as float16 and cast back to float32 inside the
graph: half the download size, identical maths at run time.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import helper, numpy_helper
from torch import nn

from .models import RecurrentLM


def _export(module: nn.Module, path: Path, outputs: list[str], dynamic: dict):
    x = torch.randint(0, 100, (1, 12))
    torch.onnx.export(module, (x,), str(path), input_names=["ids"], output_names=outputs,
                      dynamic_axes={"ids": {1: "T"}, **dynamic}, dynamo=False, opset_version=18)
    _halve_weights(path)


def _halve_weights(path: Path, min_size: int = 4096):
    """Store big float32 initializers as float16 + a Cast node back to float32."""
    model = onnx.load(str(path))
    g = model.graph
    casts, keep = [], []
    for init in g.initializer:
        arr = numpy_helper.to_array(init)
        if arr.dtype == np.float32 and arr.size >= min_size:
            half = numpy_helper.from_array(arr.astype(np.float16), init.name + "_fp16")
            keep.append(half)
            casts.append(helper.make_node("Cast", [half.name], [init.name], to=onnx.TensorProto.FLOAT))
        else:
            keep.append(init)
    del g.initializer[:]
    g.initializer.extend(keep)
    for n in reversed(casts):
        g.node.insert(0, n)
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


class _Generator(nn.Module):
    def __init__(self, lm: RecurrentLM):
        super().__init__()
        self.lm = lm

    def forward(self, ids):
        return self.lm(ids)


class LSTMProbe(nn.Module):
    """Re-runs a trained LSTM layer by layer and exposes its gates.

    Gate pre-activations are recomputed from each layer's input x_t and previous
    hidden state h_{t-1}, so no Python loop is needed and T stays dynamic.
    """

    def __init__(self, lm: RecurrentLM):
        super().__init__()
        assert lm.kind == "lstm"
        self.lm = lm
        core = lm.core
        self.layers = nn.ModuleList()
        for l in range(core.num_layers):
            layer = nn.LSTM(core.input_size if l == 0 else core.hidden_size, core.hidden_size, batch_first=True)
            for name in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
                getattr(layer, f"{name}_l0").data.copy_(getattr(core, f"{name}_l{l}").data)
            self.layers.append(layer)

    def forward(self, ids):
        x = self.lm.embed(ids)
        gates, hidden = [], []
        for layer in self.layers:
            y, _ = layer(x)                                     # (1, T, H)
            h_prev = torch.cat([torch.zeros_like(y[:, :1]), y[:, :-1]], dim=1)
            pre = x @ layer.weight_ih_l0.T + h_prev @ layer.weight_hh_l0.T + layer.bias_ih_l0 + layer.bias_hh_l0
            i, f, g, o = pre.chunk(4, dim=-1)                   # PyTorch gate order
            gates.append(torch.stack([torch.sigmoid(i), torch.sigmoid(f), torch.tanh(g), torch.sigmoid(o)], 2)[0])
            hidden.append(y[0])
            x = y
        step_logits = self.lm.head(x[0])                        # what it expects next, at every step
        return step_logits, torch.stack(gates), torch.stack(hidden)


def export_all(models: dict[str, RecurrentLM], out_dir: Path) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    sizes = {}
    for kind, lm in models.items():
        lm = lm.cpu().eval()
        _export(_Generator(lm), out_dir / f"{kind}.onnx", ["logits"], {})
        sizes[kind] = (out_dir / f"{kind}.onnx").stat().st_size
        if kind == "lstm":
            _export(LSTMProbe(lm).eval(), out_dir / "lstm_probe.onnx", ["step_logits", "gates", "hidden"],
                    {"step_logits": {0: "T"}, "gates": {1: "T"}, "hidden": {1: "T"}})
            sizes["lstm_probe"] = (out_dir / "lstm_probe.onnx").stat().st_size
    return sizes


def verify(models: dict[str, RecurrentLM], out_dir: Path) -> dict[str, float]:
    """Max |torch - onnxruntime| logit difference on a random window."""
    import onnxruntime as ort

    x = torch.randint(3, 8000, (1, 24))
    diffs = {}
    for kind, lm in models.items():
        sess = ort.InferenceSession(str(out_dir / f"{kind}.onnx"))
        with torch.no_grad():
            ref = lm.cpu().eval()(x).numpy()
        diffs[kind] = float(np.abs(sess.run(None, {"ids": x.numpy()})[0] - ref).max())
    return diffs
