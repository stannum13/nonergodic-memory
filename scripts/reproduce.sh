#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:src"
seeds=(0 1 2)
if ! python -m nonergodic_memory.checkpoints --config configs/reproduce.yaml --checkpoint-dir checkpoints/reproduce --models gru transformer --seeds "${seeds[@]}"; then
  python src/train.py --config configs/reproduce.yaml --seeds "${seeds[@]}" --output-dir checkpoints/reproduce --results results/training.jsonl
fi
python src/probe.py --config configs/reproduce.yaml --seeds "${seeds[@]}" --checkpoint-dir checkpoints/reproduce --results results/reproduction.jsonl
