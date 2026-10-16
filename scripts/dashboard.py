"""Live training dashboard — http://localhost:4320

Read-only: parses results/runs/train_log.txt and the per-run summary files, so it
never touches the training process or the GPU.

  python scripts/dashboard.py
"""
import json
import re
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "runs"
LOG = RUNS / "train_log.txt"
PAGE = Path(__file__).with_name("dashboard.html")
PORT = 4320

EPOCH_RE = re.compile(r"\[(\S+)\] ep\s+(\d+)\s+train ([\d.]+)\s+val ([\d.]+)\s+ppl ([\d.]+)\s+top1 ([\d.]+)\s+(\d+)s")
TEST_RE = re.compile(r"\[(\S+)\] TEST loss ([\d.]+) ppl ([\d.]+) top1 ([\d.]+) top5 ([\d.]+)\s+train ([\d.]+) min")
START_RE = re.compile(r"\[(\S+)\] params=([\d,]+)")
STOP_RE = re.compile(r"\[(\S+)\] early stop at epoch (\d+)")
# Rough per-epoch seconds used for ETAs before a model has finished a run.
DEFAULT_EPOCH_S = {"rnn": 120, "birnn": 255, "lstm": 270, "gru": 260, "tfm": 240}


def plan(cfg):
    kinds = ["rnn", "birnn", "lstm", "gru"]
    q = [(f"{k}_s{s}", k, "main") for s in cfg["train"]["seeds"] for k in kinds]
    q += [(f"matched_{k}_s{cfg['train']['seeds'][0]}", k, "matched") for k in kinds]
    q += [(f"tfm_s{cfg['train']['seeds'][0]}", "tfm", "bonus")]
    return q


def status():
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())
    max_ep = cfg["train"]["max_epochs"]
    text = LOG.read_text() if LOG.exists() else ""
    runs, last_started = {}, None
    for line in text.splitlines():
        if m := START_RE.search(line):
            last_started = m[1]
            runs.setdefault(m[1], {"epochs": [], "params": int(m[2].replace(",", ""))})
        elif m := EPOCH_RE.search(line):
            runs.setdefault(m[1], {"epochs": []})["epochs"].append({
                "ep": int(m[2]), "train": float(m[3]), "val": float(m[4]), "ppl": float(m[5]),
                "top1": float(m[6]), "sec": int(m[7])})
        elif m := TEST_RE.search(line):
            runs.setdefault(m[1], {"epochs": []})["test"] = {
                "loss": float(m[2]), "ppl": float(m[3]), "top1": float(m[4]), "top5": float(m[5]), "min": float(m[6])}
        elif m := STOP_RE.search(line):
            runs.setdefault(m[1], {"epochs": []})["stopped"] = int(m[2])

    alive = subprocess.run(["pgrep", "-f", "scripts/train_all.py"], capture_output=True).returncode == 0
    last_write = LOG.stat().st_mtime if LOG.exists() else time.time()

    # Mean epoch seconds per model kind, from what has actually run.
    secs = {}
    for name, r in runs.items():
        kind = name.split("_")[-2] if name.startswith("matched_") else name.split("_")[0]
        secs.setdefault(kind, []).extend(e["sec"] for e in r["epochs"])
    ep_s = {k: (sum(v) / len(v) if v else DEFAULT_EPOCH_S[k]) for k, v in secs.items()}

    queue, eta, current = [], 0.0, None
    for name, kind, group in plan(cfg):
        r = runs.get(name)
        done = (RUNS / name / "summary.json").exists()
        state = "done" if done else ("running" if name == last_started and alive else "queued")
        per_ep = ep_s.get(kind, DEFAULT_EPOCH_S[kind]) + 15  # + validation
        if state == "running":
            n = len(r["epochs"])
            elapsed = time.time() - last_write
            frac = min(0.98, elapsed / per_ep)
            current = {"name": name, "kind": kind, "epoch": n + 1, "max": max_ep, "frac": frac,
                       "elapsed": elapsed, "per_epoch": per_ep}
            eta += max(0, (max_ep - n) * per_ep - elapsed)
        elif state == "queued":
            eta += max_ep * per_ep
        queue.append({"name": name, "kind": kind, "group": group, "state": state})

    return {"now": time.time(), "alive": alive, "runs": runs, "queue": queue, "current": current,
            "eta_s": eta, "max_epochs": max_ep, "log_tail": [l for l in text.splitlines() if not l.startswith("{")][-14:]}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/api/status"):
            body, ctype = json.dumps(status()).encode(), "application/json"
        else:
            body, ctype = PAGE.read_bytes(), "text/html; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    print(f"ECHO training dashboard → http://localhost:{PORT}")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
