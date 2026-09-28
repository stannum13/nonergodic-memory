#!/usr/bin/env bash
set -euo pipefail
PREDICTIVE_MEMORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PREDICTIVE_MEMORY_ROOT"
export PYTHONPATH="$PREDICTIVE_MEMORY_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
"${PYTHON:-python3}" src/predictive_memory.py "$@"
"${PYTHON:-python3}" -m nonergodic_memory.predictive_memory_figures
