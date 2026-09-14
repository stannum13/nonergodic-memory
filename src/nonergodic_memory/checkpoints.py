"""Validate a complete checkpoint set before cached analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .experiment import load_config, validate_checkpoint


def checkpoint_set_matches(
    config: dict, checkpoint_dir: str | Path, models: list[str], seeds: list[int]
) -> bool:
    directory = Path(checkpoint_dir)
    for seed in seeds:
        for model_name in models:
            path = directory / f"{model_name}_seed{seed}.pt"
            if not path.exists():
                return False
            try:
                payload = torch.load(path, map_location="cpu", weights_only=False)
                validate_checkpoint(payload, config, model_name, seed)
            except (OSError, RuntimeError, ValueError, KeyError):
                return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    args = parser.parse_args()
    if not checkpoint_set_matches(
        load_config(args.config), args.checkpoint_dir, args.models, args.seeds
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
