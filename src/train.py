#!/usr/bin/env python3
"""Train small sequence models on a fixed-component HMM mixture."""

from __future__ import annotations

import argparse
from pathlib import Path

from nonergodic_memory.experiment import (
    config_digest,
    generator_name,
    load_config,
    replace_jsonl_runs,
    runtime_provenance,
    train_one,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--models", nargs="+", choices=["gru", "transformer"], default=["gru", "transformer"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--results", default="results/training.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config_name = Path(args.config).stem
    records = []
    for seed in args.seeds:
        for model_name in args.models:
            record, _ = train_one(config, model_name, seed, args.output_dir)
            record["config"] = config_name
            record["config_sha256"] = config_digest(config)
            record["generator"] = generator_name(config)
            record.update(runtime_provenance())
            records.append(record)
            print(f"{model_name} seed={seed} test_nll={record['test_nll']:.4f}")
    replace_jsonl_runs(args.results, records, config_name, args.models, args.seeds)


if __name__ == "__main__":
    main()
