#!/usr/bin/env bash
set -euo pipefail
PREDICTIVE_MEMORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PREDICTIVE_MEMORY_ROOT"
export PYTHONPATH="$PREDICTIVE_MEMORY_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON:-python3}" -m nonergodic_memory.watchdog \
  --seconds 3600 \
  --timeout-summary results/predictive_memory_summary.jsonl \
  -- bash scripts/predictive_memory_inner.sh "$@"
