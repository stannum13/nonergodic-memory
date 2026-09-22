#!/usr/bin/env python3
"""Run predictive and representation diagnostics for Mess3 training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from nonergodic_memory.experiment import (
    config_digest,
    load_config,
    mixture_from_config,
    runtime_provenance,
    write_jsonl,
)
from nonergodic_memory.mess3_diagnosis import (
    evaluate_checkpoint_geometry,
    predictive_baselines,
    train_diagnostic,
)
from nonergodic_memory.mess3_diagnosis_figures import generate_diagnosis_figures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mess3_diagnosis.yaml")
    parser.add_argument("--mode", choices=("baselines", "train", "probe", "figures", "all"), default="all")
    parser.add_argument("--seeds", nargs="+", type=int, default=[10, 11])
    parser.add_argument("--conditions", nargs="+", choices=("reused", "fresh"), default=["reused", "fresh"])
    parser.add_argument("--checkpoint-dir", default="checkpoints/mess3_diagnosis")
    parser.add_argument("--baseline-results", default="results/mess3_diagnosis_baselines.jsonl")
    parser.add_argument("--training-results", default="results/mess3_diagnosis_training.jsonl")
    parser.add_argument("--probe-results", default="results/mess3_diagnosis_probes.jsonl")
    parser.add_argument("--output-dir", default="figures")
    return parser.parse_args()


def _checkpoint_path(root: Path, seed: int, condition: str, step: int) -> Path:
    return root / f"transformer_seed{seed}_{condition}_step{step}.pt"


def _cache_complete(config: dict, root: Path, seeds: list[int], conditions: list[str]) -> bool:
    for seed in seeds:
        for condition in conditions:
            for step in config["train"]["checkpoint_steps"]:
                path = _checkpoint_path(root, seed, condition, int(step))
                if not path.exists():
                    return False
                try:
                    payload = torch.load(path, map_location="cpu", weights_only=False)
                except Exception:
                    return False
                if (
                    payload.get("config") != config
                    or payload.get("seed") != seed
                    or payload.get("condition") != condition
                    or payload.get("step") != int(step)
                ):
                    return False
    return True


def _training_results_complete(
    path: Path, digest: str, config: dict, seeds: list[int], conditions: list[str]
) -> bool:
    if not path.exists():
        return False
    try:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError):
        return False
    actual = {
        (int(row["seed"]), row["condition"], int(row["step"]))
        for row in rows
        if row.get("record_type") == "diagnostic_training" and row.get("config_sha256") == digest
    }
    expected = {
        (seed, condition, int(step))
        for seed in seeds
        for condition in conditions
        for step in config["train"]["checkpoint_steps"]
    }
    return actual == expected


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    digest = config_digest(config)
    config_name = Path(args.config).stem
    checkpoint_root = Path(args.checkpoint_dir)
    provenance = runtime_provenance()

    if args.mode in {"baselines", "all"}:
        mixture = mixture_from_config(config)
        length = int(config["data"]["sequence_length"])
        fit = mixture.sample(int(config["diagnosis"]["baseline_fit_sequences"]), length, 606)
        test = mixture.sample(int(config["data"]["test_sequences"]), length, 707)
        rows = predictive_baselines(mixture, fit, test, int(config["diagnosis"]["window"]))
        for row in rows:
            row.update(config=config_name, config_sha256=digest, generator="mess3", **provenance)
        write_jsonl(args.baseline_results, rows)
        print("wrote predictive baselines")

    if args.mode in {"train", "all"}:
        training_path = Path(args.training_results)
        cached = _cache_complete(config, checkpoint_root, args.seeds, args.conditions)
        recorded = _training_results_complete(training_path, digest, config, args.seeds, args.conditions)
        if not (cached and recorded):
            rows = []
            for seed in args.seeds:
                for condition in args.conditions:
                    run = train_diagnostic(config, seed, condition, checkpoint_root)
                    for row in run:
                        row.update(config=config_name, config_sha256=digest)
                    rows.extend(run)
            write_jsonl(training_path, rows)
            print("wrote diagnostic training records")
        else:
            print("reused complete diagnostic checkpoints and training records")

    if args.mode in {"probe", "all"}:
        if not _cache_complete(config, checkpoint_root, args.seeds, args.conditions):
            raise SystemExit("diagnostic checkpoint set is incomplete")
        rows = []
        for seed in args.seeds:
            for condition in args.conditions:
                for step in config["train"]["checkpoint_steps"]:
                    path = _checkpoint_path(checkpoint_root, seed, condition, int(step))
                    run = evaluate_checkpoint_geometry(config, path, seed, condition)
                    for row in run:
                        row["config"] = config_name
                    rows.extend(run)
        write_jsonl(args.probe_results, rows)
        print("wrote diagnostic probe records")

    if args.mode in {"figures", "all"}:
        for path in generate_diagnosis_figures(
            args.baseline_results, args.training_results, args.probe_results, args.output_dir
        ):
            print(f"generated {path}")


if __name__ == "__main__":
    main()
