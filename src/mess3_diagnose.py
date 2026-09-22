#!/usr/bin/env python3
"""Run predictive and representation diagnostics for Mess3 training."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

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
    parser.add_argument(
        "--expected-seeds",
        nargs="+",
        type=int,
        default=[10, 11],
        help="authoritative complete artifact grid, independent of --seeds recomputation selection",
    )
    parser.add_argument("--conditions", nargs="+", choices=("reused", "fresh"), default=["reused", "fresh"])
    parser.add_argument("--checkpoint-dir", default="checkpoints/mess3_diagnosis")
    parser.add_argument("--baseline-results", default="results/mess3_diagnosis_baselines.jsonl")
    parser.add_argument("--training-results", default="results/mess3_diagnosis_training.jsonl")
    parser.add_argument("--probe-results", default="results/mess3_diagnosis_probes.jsonl")
    parser.add_argument("--output-dir", default="figures")
    return parser.parse_args()


def _checkpoint_path(root: Path, seed: int, condition: str, step: int) -> Path:
    return root / f"transformer_seed{seed}_{condition}_step{step}.pt"


def _replace_keyed_jsonl(
    path: str | Path, records: Iterable[dict], key_fields: tuple[str, ...]
) -> None:
    """Atomically replace exact result cells while preserving all other cells."""
    destination = Path(path)
    new_rows = list(records)
    if not new_rows:
        raise ValueError("replacement records cannot be empty")
    new_digests = {row.get("config_sha256") for row in new_rows}
    if len(new_digests) != 1 or None in new_digests:
        raise ValueError("replacement records must share one config_sha256")
    new_digest = next(iter(new_digests))
    existing: list[dict] = []
    if destination.exists():
        existing = [json.loads(line) for line in destination.read_text().splitlines() if line.strip()]

    def key(row: dict) -> tuple:
        try:
            return tuple(row[field] for field in key_fields)
        except KeyError as error:
            raise ValueError(f"record is missing key field {error.args[0]}") from error

    for label, rows in (("existing", existing), ("replacement", new_rows)):
        counts = Counter(key(row) for row in rows)
        if any(count != 1 for count in counts.values()):
            raise ValueError(f"duplicate {label} result cells")
    new_keys = {key(row) for row in new_rows}
    retained = [
        row
        for row in existing
        if row.get("config_sha256") == new_digest and key(row) not in new_keys
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    write_jsonl(temporary, [*retained, *new_rows])
    temporary.replace(destination)


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
    diagnostic_rows = [row for row in rows if row.get("record_type") == "diagnostic_training"]
    if any(row.get("config_sha256") != digest for row in diagnostic_rows):
        return False
    try:
        actual = Counter(
            (int(row["seed"]), row["condition"], int(row["step"])) for row in diagnostic_rows
        )
    except (KeyError, TypeError, ValueError):
        return False
    expected = {
        (seed, condition, int(step))
        for seed in seeds
        for condition in conditions
        for step in config["train"]["checkpoint_steps"]
    }
    return all(actual[cell] == 1 for cell in expected) and all(count == 1 for count in actual.values())


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
            _replace_keyed_jsonl(
                training_path,
                rows,
                ("config_sha256", "seed", "condition", "step"),
            )
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
        _replace_keyed_jsonl(
            args.probe_results,
            rows,
            ("config_sha256", "seed", "condition", "step", "site", "control"),
        )
        print("wrote diagnostic probe records")

    if args.mode in {"figures", "all"}:
        for path in generate_diagnosis_figures(
            args.baseline_results,
            args.training_results,
            args.probe_results,
            args.output_dir,
            expected_seeds=args.expected_seeds,
            expected_steps=config["train"]["checkpoint_steps"],
        ):
            print(f"generated {path}")


if __name__ == "__main__":
    main()
