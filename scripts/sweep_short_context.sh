#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:src"
seeds=(0 1 2)
eval_configs=(
  configs/sweeps/interaction_o000_l064.yaml
  configs/sweeps/interaction_o035_l064.yaml
)
short_configs=(
  configs/sweeps/short_o000_l009.yaml
  configs/sweeps/short_o035_l009.yaml
)
if [[ ! -f results/sweep_position_restart.jsonl ]]; then
  bash scripts/sweep_position_restart.sh
fi
for config in "${short_configs[@]}"; do
  name="$(basename "$config" .yaml)"
  checkpoint_dir="checkpoints/sweeps/$name"
  if ! python -m nonergodic_memory.checkpoints --config "$config" --checkpoint-dir "$checkpoint_dir" --models transformer --seeds "${seeds[@]}" || \
     ! python -m nonergodic_memory.training_records --config "$config" --results results/sweep_short_context_training.jsonl --model transformer --seeds "${seeds[@]}"; then
    python src/train.py --config "$config" --models transformer --seeds "${seeds[@]}" --output-dir "$checkpoint_dir" --results results/sweep_short_context_training.jsonl
  fi
done
rm -f results/sweep_short_context.jsonl
python src/short_context.py --eval-configs "${eval_configs[@]}" --short-configs "${short_configs[@]}" --seeds "${seeds[@]}" --checkpoint-root checkpoints/sweeps --results results/sweep_short_context.jsonl
python -m nonergodic_memory.short_figures
