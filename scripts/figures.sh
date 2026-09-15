#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:src"
python -m nonergodic_memory.figures
python -m nonergodic_memory.sweeps --axis overlap
python -m nonergodic_memory.sweeps --axis length
python -m nonergodic_memory.sweeps --axis components
python -m nonergodic_memory.sweeps --axis width
python -m nonergodic_memory.sweeps --axis depth
python -m nonergodic_memory.sweeps --axis interaction
python -m nonergodic_memory.context_figures
