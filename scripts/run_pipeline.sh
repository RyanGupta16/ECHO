#!/bin/zsh
# Full training queue (same order as the dashboard plan), safe to re-run after a crash/shutdown:
# train_all.py skips any run that already has a summary.json.
#   nohup caffeinate -i scripts/run_pipeline.sh >/dev/null 2>&1 &
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
OUT=results/runs/main_stdout.txt
$PY scripts/train_all.py                      >> $OUT 2>&1   # 4 models x 3 seeds
$PY scripts/train_all.py --matched            >> $OUT 2>&1   # parameter-matched, seed 13
$PY scripts/train_all.py --kinds tfm --seeds 13 >> $OUT 2>&1 # bonus Transformer
