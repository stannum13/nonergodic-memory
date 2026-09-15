#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:src"
seeds=(0 1 2)
configs=(
  configs/sweeps/interaction_o000_l064.yaml
  configs/sweeps/interaction_o035_l064.yaml
)
for config in "${configs[@]}"; do
  name="$(basename "$config" .yaml)"
  checkpoint_dir="checkpoints/sweeps/$name"
  if ! python -m nonergodic_memory.checkpoints --config "$config" --checkpoint-dir "$checkpoint_dir" --models transformer --seeds "${seeds[@]}"; then
    python src/train.py --config "$config" --models transformer --seeds "${seeds[@]}" --output-dir "$checkpoint_dir" --results results/sweep_interaction_training.jsonl
  fi
done
rm -f results/sweep_position_restart.jsonl
python src/context_restart.py --configs "${configs[@]}" --models transformer --seeds "${seeds[@]}" --window 8 --include-absolute-positions --checkpoint-root checkpoints/sweeps --results results/sweep_position_restart.jsonl
python -m nonergodic_memory.position_figures
