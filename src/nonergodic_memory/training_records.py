"""Read-only validation of reproducible training result grids."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .experiment import config_digest, load_config


def training_set_matches(
    config_path: str | Path, results_path: str | Path, model: str, seeds: list[int]
) -> bool:
    """Require one valid raw training cell per requested model/seed/config."""
    path = Path(results_path)
    if not path.exists():
        return False
    config = load_config(config_path)
    name = Path(config_path).stem
    digest = config_digest(config)
    expected = set(seeds)
    seen = set()
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("config") != name or row.get("model") != model:
                    continue
                seed = row.get("seed")
                if seed not in expected:
                    continue
                if seed in seen:
                    return False
                if (
                    row.get("record_type") != "training"
                    or row.get("config_sha256") != digest
                    or row.get("sequence_length") != config["data"]["sequence_length"]
                    or row.get("device") != "cpu"
                    or not all(
                        math.isfinite(float(row[key]))
                        for key in ("train_nll", "test_nll", "test_kl_exact", "test_bayes_nll")
                    )
                ):
                    return False
                seen.add(seed)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False
    return seen == expected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    args = parser.parse_args()
    if not training_set_matches(args.config, args.results, args.model, args.seeds):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
