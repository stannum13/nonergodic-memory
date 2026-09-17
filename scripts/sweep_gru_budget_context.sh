#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:src"
seeds=(0 1 2)
eval_config=configs/sweeps/interaction_o035_l064.yaml
standard_config=configs/sweeps/short_o035_l009.yaml
budget_config=configs/sweeps/budget_o035_l009.yaml
if [[ ! -f results/sweep_context_restart.jsonl ]]; then
  bash scripts/sweep_context_restart.sh
fi
for protocol in standard budget; do
  if [[ "$protocol" == standard ]]; then
    config="$standard_config"
    training_results=results/sweep_gru_short_context_training.jsonl
    evaluation_results=results/sweep_gru_short_context.jsonl
  else
    config="$budget_config"
    training_results=results/sweep_gru_budget_context_training.jsonl
    evaluation_results=results/sweep_gru_budget_context.jsonl
  fi
  name="$(basename "$config" .yaml)"
  checkpoint_dir="checkpoints/sweeps/$name"
  if ! python -m nonergodic_memory.checkpoints --config "$config" --checkpoint-dir "$checkpoint_dir" --models gru --seeds "${seeds[@]}" || \
     ! python -m nonergodic_memory.training_records --config "$config" --results "$training_results" --model gru --seeds "${seeds[@]}"; then
    python src/train.py --config "$config" --models gru --seeds "${seeds[@]}" --output-dir "$checkpoint_dir" --results "$training_results"
  fi
  rm -f "$evaluation_results"
  python src/short_context.py --eval-configs "$eval_config" --short-configs "$config" --model gru --protocol "$protocol" --seeds "${seeds[@]}" --checkpoint-root checkpoints/sweeps --results "$evaluation_results"
done
python -m nonergodic_memory.gru_budget_figures
