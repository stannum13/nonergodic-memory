#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:src"
python src/train.py --config configs/smoke.yaml --seeds 0 --output-dir checkpoints/smoke --results results/smoke_training.jsonl
python src/probe.py --config configs/smoke.yaml --seeds 0 --checkpoint-dir checkpoints/smoke --results results/smoke_reproduction.jsonl
python src/intervene.py --config configs/smoke.yaml --seeds 0 --checkpoint-dir checkpoints/smoke --results results/smoke_extension.jsonl
python -m nonergodic_memory.figures

