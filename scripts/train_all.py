"""Step 2 — train every model under identical conditions.

  python scripts/train_all.py                 # 4 models x 3 seeds (main experiment)
  python scripts/train_all.py --matched       # parameter-matched variant (1 seed)
  python scripts/train_all.py --kinds lstm --seeds 13 --quick   # smoke test
"""
import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from echo.data import Vocab  # noqa: E402
from echo.models import KINDS, build, count_params, matched_hidden  # noqa: E402
from echo.train import train_one  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", nargs="*", default=list(KINDS))
    ap.add_argument("--seeds", nargs="*", type=int)
    ap.add_argument("--matched", action="store_true", help="match total parameters to the LSTM")
    ap.add_argument("--quick", action="store_true", help="tiny smoke-test run")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())
    if args.quick:
        cfg["train"].update(samples_per_epoch=100_000, max_epochs=2)
    seeds = args.seeds or (cfg["train"]["seeds"][:1] if args.matched else cfg["train"]["seeds"])
    log_path = ROOT / "results" / "runs" / "train_log.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(msg, flush=True)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    vocab_size = len(Vocab.load(ROOT / "data" / "processed" / "vocab.json"))
    target = count_params(build("lstm", vocab_size, cfg["model"]))["total"]
    for seed in seeds:
        for kind in args.kinds:
            overrides, name = None, None
            if args.matched:
                overrides = {"hidden_dim": matched_hidden(kind, vocab_size, cfg["model"], target)}
                name = f"matched_{kind}_s{seed}"
            if args.quick:
                name = f"quick_{kind}_s{seed}"
            done = ROOT / "results" / "runs" / (name or f"{kind}_s{seed}") / "summary.json"
            if done.exists() and not args.quick:
                log(f"skip {done.parent.name} (already trained)")
                continue
            s = train_one(kind, seed, cfg, ROOT, run_name=name, model_overrides=overrides, log=log)
            log(json.dumps({"run": s["run"], "test": s["test"]}))
