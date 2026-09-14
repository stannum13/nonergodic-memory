#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:src"
seeds=(0 1 2)
rm -f results/sweep_width_training.jsonl results/sweep_width_reproduction.jsonl results/sweep_width_extension.jsonl
for config in configs/sweeps/width_*.yaml; do
  name="$(basename "$config" .yaml)"
  checkpoint_dir="checkpoints/sweeps/$name"
  python src/train.py --config "$config" --seeds "${seeds[@]}" --output-dir "$checkpoint_dir" --results results/sweep_width_training.jsonl
  python src/probe.py --config "$config" --seeds "${seeds[@]}" --checkpoint-dir "$checkpoint_dir" --results results/sweep_width_reproduction.jsonl
  python src/intervene.py --config "$config" --seeds "${seeds[@]}" --checkpoint-dir "$checkpoint_dir" --results results/sweep_width_extension.jsonl
done
python -m nonergodic_memory.sweeps --axis width

