#!/usr/bin/env bash
# Reruns the clean-vs-raw experiment with full logs, then plots the results.
# Needs plain-text documents in corpus/ and takes about 2-3 days on two V100s.
# Safe to restart: finished data is kept and runs continue from their checkpoints.
#   bash scripts/rerun_clean_vs_raw.sh
set -euo pipefail
GPUS="${GPUS:-2}"
CONFIGS="model3_123M model4_236M_raw model4_236M"
train() { torchrun --nproc_per_node="$GPUS" training/train.py "$@"; }

[ -d corpus ] || { echo "put your .txt documents in corpus/ first"; exit 1; }
[ -f data/clean/meta.json ] || python training/prepare_data.py --src corpus/ --out data/clean --train-tokenizer
[ -f data/raw/meta.json ] || python training/prepare_data.py --src corpus/ --out data/raw --no-clean --val-from data/clean

echo "== 20-step test of each config (finds crashes and out-of-memory errors early)"
for c in $CONFIGS; do
  train --config "training/configs/$c.yaml" --max-steps 20 --out-dir "runs/smoke/$c"
done

echo "== full runs"
for c in $CONFIGS; do
  train --config "training/configs/$c.yaml" --resume
done

python analysis/plot_logs.py runs/model3_123M runs/model4_236M_raw runs/model4_236M --out docs/images/clean_vs_raw.png
