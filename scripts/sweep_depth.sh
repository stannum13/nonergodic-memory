#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:src"
seeds=(0 1 2)
if ! python -m nonergodic_memory.checkpoints --config configs/reproduce.yaml --checkpoint-dir checkpoints/reproduce --models transformer --seeds "${seeds[@]}"; then
  python src/train.py --config configs/reproduce.yaml --models transformer --seeds "${seeds[@]}" --output-dir checkpoints/reproduce --results results/training.jsonl
fi
rm -f results/sweep_depth.jsonl
python src/intervene_depth.py --config configs/reproduce.yaml --seeds "${seeds[@]}" --checkpoint-dir checkpoints/reproduce --results results/sweep_depth.jsonl
python -m nonergodic_memory.sweeps --axis depth
