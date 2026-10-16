"""Step 1 — download TinyStories and build the training data."""
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from echo.data import prepare  # noqa: E402

if __name__ == "__main__":
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())
    stats = prepare(cfg["data"], ROOT)
    print(json.dumps({k: v for k, v in stats.items() if k != "top_words"}, indent=2))
