#!/usr/bin/env bash
set -euo pipefail
RATE_AWARE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$RATE_AWARE_ROOT"
export PYTHONPATH="$RATE_AWARE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON:-python3}" src/mess3_rate_aware_clock.py --mode all "$@"
