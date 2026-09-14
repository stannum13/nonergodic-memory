#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:src"
seeds=(0 1 2)
missing=0
for seed in "${seeds[@]}"; do
  for model in gru transformer; do
    [[ -f "checkpoints/reproduce/${model}_seed${seed}.pt" ]] || missing=1
  done
done
if [[ "$missing" -eq 1 ]]; then
  python src/train.py --config configs/reproduce.yaml --seeds "${seeds[@]}" --output-dir checkpoints/reproduce --results results/training.jsonl
fi
python src/probe.py --config configs/reproduce.yaml --seeds "${seeds[@]}" --checkpoint-dir checkpoints/reproduce --results results/reproduction.jsonl

