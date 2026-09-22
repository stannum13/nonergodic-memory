#!/usr/bin/env python3
"""Run paired Mess3 learning curves across learning rates."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from nonergodic_memory.experiment import config_digest, load_config, write_jsonl
from nonergodic_memory.mess3_threshold import (
    _rate_config,
    analyze_threshold,
    replace_threshold_records,
    run_threshold_probes,
    run_threshold_training,
    threshold_checkpoint_path,
    validate_threshold_grid,
)
from nonergodic_memory.mess3_threshold_figures import generate_threshold_figures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mess3_threshold.yaml")
    parser.add_argument(
        "--mode",
        choices=("train", "probe", "analyze", "figures", "all"),
        default="all",
    )
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--learning-rates", nargs="+", type=float)
    parser.add_argument("--checkpoint-dir", default="checkpoints/mess3_threshold")
    parser.add_argument("--training-results", default="results/mess3_threshold_training.jsonl")
    parser.add_argument("--probe-results", default="results/mess3_threshold_probes.jsonl")
    parser.add_argument("--summary-results", default="results/mess3_threshold_summary.jsonl")
    parser.add_argument("--output-dir", default="figures")
    return parser.parse_args()


def _cache_complete(
    config: dict,
    seeds: list[int],
    learning_rates: list[float],
    checkpoint_root: Path,
) -> bool:
    for seed in seeds:
        for learning_rate in learning_rates:
            rate_config = _rate_config(config, learning_rate)
            for step in rate_config["train"]["checkpoint_steps"]:
                path = threshold_checkpoint_path(checkpoint_root, seed, learning_rate, int(step))
                if not path.exists():
                    return False
                try:
                    payload = torch.load(path, map_location="cpu", weights_only=False)
                except Exception:
                    return False
                if (
                    payload.get("config") != rate_config
                    or payload.get("seed") != seed
                    or payload.get("condition") != "fresh"
                    or payload.get("step") != int(step)
                ):
                    return False
    return True


def _training_records_complete(
    path: Path,
    config: dict,
    seeds: list[int],
    learning_rates: list[float],
) -> bool:
    if not path.exists():
        return False
    try:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if any(row.get("base_config_sha256") != config_digest(config) for row in rows):
            return False
        counts = Counter(
            (int(row["seed"]), float(row["learning_rate"]), int(row["step"]))
            for row in rows
        )
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return False
    expected = {
        (seed, rate, int(step))
        for seed in seeds
        for rate in learning_rates
        for step in config["train"]["checkpoint_steps"]
    }
    return all(counts[cell] == 1 for cell in expected) and all(
        count == 1 for count in counts.values()
    )


def _load_rows(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    expected_seeds = [int(seed) for seed in config["threshold"]["seeds"]]
    expected_rates = [float(rate) for rate in config["threshold"]["learning_rates"]]
    seeds = args.seeds or expected_seeds
    learning_rates = args.learning_rates or expected_rates
    if not set(seeds) <= set(expected_seeds) or not set(learning_rates) <= set(expected_rates):
        raise SystemExit("selected seeds and learning rates must belong to the configured grid")
    checkpoint_root = Path(args.checkpoint_dir)
    training_path = Path(args.training_results)

    if args.mode in {"train", "all"}:
        rows = []
        reused = 0
        for seed in seeds:
            for learning_rate in learning_rates:
                cached = _cache_complete(config, [seed], [learning_rate], checkpoint_root)
                recorded = _training_records_complete(
                    training_path, config, [seed], [learning_rate]
                )
                if cached and recorded:
                    reused += 1
                else:
                    rows.extend(
                        run_threshold_training(
                            config, [seed], [learning_rate], checkpoint_root
                        )
                    )
        if rows:
            replace_threshold_records(
                training_path,
                rows,
                ("base_config_sha256", "seed", "learning_rate", "step"),
            )
            print("wrote threshold training records")
        if reused:
            print(f"reused {reused} complete threshold training cells")

    if args.mode in {"probe", "all"}:
        if not _cache_complete(config, seeds, learning_rates, checkpoint_root):
            raise SystemExit("threshold checkpoint set is incomplete")
        rows = run_threshold_probes(config, seeds, learning_rates, checkpoint_root)
        replace_threshold_records(
            args.probe_results,
            rows,
            (
                "base_config_sha256",
                "seed",
                "learning_rate",
                "step",
                "site",
                "control",
            ),
        )
        print("wrote threshold probe records")

    if args.mode in {"analyze", "all"}:
        training = _load_rows(args.training_results)
        probes = _load_rows(args.probe_results)
        validate_threshold_grid(config, training, probes)
        summary = analyze_threshold(config, training, probes)
        write_jsonl(args.summary_results, [summary])
        print("wrote threshold summary")

    if args.mode in {"figures", "all"}:
        training = _load_rows(args.training_results)
        probes = _load_rows(args.probe_results)
        summaries = _load_rows(args.summary_results)
        if len(summaries) != 1 or summaries[0].get("record_type") != "threshold_summary":
            raise SystemExit("threshold summary file must contain exactly one summary")
        for path in generate_threshold_figures(
            config, training, probes, summaries[0], args.output_dir
        ):
            print(f"generated {path}")

    if args.mode == "all":
        print("validated complete threshold grid")


if __name__ == "__main__":
    main()
