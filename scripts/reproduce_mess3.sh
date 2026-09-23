#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:src"
config=configs/mess3_cpu.yaml
checkpoint_dir=checkpoints/mess3_cpu
training_results=results/mess3_training.jsonl
seeds=(0 1 2)
for seed in "${seeds[@]}"; do
  if ! python -m nonergodic_memory.checkpoints --config "$config" --checkpoint-dir "$checkpoint_dir" --models transformer --seeds "$seed" || \
     ! python -m nonergodic_memory.training_records --config "$config" --results "$training_results" --model transformer --seeds "$seed"; then
    python src/train.py --config "$config" --models transformer --seeds "$seed" --output-dir "$checkpoint_dir" --results "$training_results"
  fi
done
python src/probe.py --config "$config" --models transformer --seeds "${seeds[@]}" --checkpoint-dir "$checkpoint_dir" --results results/mess3_reproduction.jsonl
python -m nonergodic_memory.mess3_figures
