#!/usr/bin/env bash
set -euo pipefail
PREDICTIVE_MEMORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PREDICTIVE_MEMORY_ROOT"
export PYTHONPATH="$PREDICTIVE_MEMORY_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
if (( $# != 0 )); then
  echo "registered predictive-memory wrapper does not accept path/config overrides" >&2
  exit 2
fi
exec "${PYTHON:-python3}" -m nonergodic_memory.watchdog \
  --seconds 3600 \
  --timeout-summary results/predictive_memory_command.jsonl \
  --reservation results/predictive_memory_watchdog.jsonl \
  -- bash scripts/predictive_memory_inner.sh
