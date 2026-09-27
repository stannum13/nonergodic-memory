#!/usr/bin/env python3
"""Run the preregistered Mess3 competence--time experiment resumably."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable

import torch

from nonergodic_memory.experiment import config_digest, load_config, write_jsonl
from nonergodic_memory.mess3_competence_time import (
    _rate_config,
    _rate_label,
    analyze_competence_time,
    competence_time_checkpoint_path,
    run_competence_time_probes,
    run_competence_time_training,
    validate_competence_time_grid,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mess3_competence_time.yaml")
    parser.add_argument(
        "--mode", choices=("train", "probe", "analyze", "figures", "all"), default="all"
    )
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--learning-rates", nargs="+", type=float)
    parser.add_argument("--checkpoint-dir", default="checkpoints/mess3_competence_time")
    parser.add_argument(
        "--training-results", default="results/mess3_competence_time_training.jsonl"
    )
    parser.add_argument("--probe-results", default="results/mess3_competence_time_probes.jsonl")
    parser.add_argument("--summary-results", default="results/mess3_competence_time_summary.jsonl")
    parser.add_argument("--output-dir", default="figures")
    return parser.parse_args()


def _checkpoint_cache_complete(
    config: dict,
    seeds: Iterable[int],
    learning_rates: Iterable[float],
    checkpoint_root: Path,
) -> bool:
    """Accept only checkpoints carrying the exact rate configuration and identity."""
    for seed in seeds:
        for learning_rate in learning_rates:
            rate_config = _rate_config(config, float(learning_rate))
            for step in rate_config["train"]["checkpoint_steps"]:
                path = competence_time_checkpoint_path(
                    checkpoint_root, int(seed), float(learning_rate), int(step)
                )
                if not path.exists():
                    return False
                try:
                    payload = torch.load(path, map_location="cpu", weights_only=False)
                except Exception:
                    return False
                if not isinstance(payload, dict):
                    return False
                if (
                    payload.get("config") != rate_config
                    or payload.get("seed") != int(seed)
                    or payload.get("condition") != "fresh"
                    or payload.get("step") != int(step)
                ):
                    return False
    return True


def _read_jsonl(path: str | Path) -> list[dict]:
    try:
        rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSONL records: {error}") from error
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("JSONL records must be objects")
    return rows


def _training_record_valid(config: dict, row: dict, checkpoint_root: Path) -> bool:
    try:
        rate = float(row["learning_rate"])
        seed, step = int(row["seed"]), int(row["step"])
        expected_digest = config_digest(_rate_config(config, rate))
        expected_path = competence_time_checkpoint_path(checkpoint_root, seed, rate, step)
        return (
            row.get("record_type") == "competence_time_training"
            and row.get("base_config_sha256") == config_digest(config)
            and row.get("rate_config_sha256") == expected_digest
            and row.get("config_sha256") == expected_digest
            and row.get("sampler") == "vectorized"
            and row.get("learning_rate_label") == _rate_label(rate)
            and row.get("condition") == "fresh"
            and seed == row["seed"]
            and step == row["step"]
            and row.get("checkpoint_path") == str(expected_path)
            and all(
                math.isfinite(float(row[metric]))
                for metric in ("competence", "kl_exact", "nll", "uniform_kl")
            )
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def _training_records_complete(
    path: str | Path,
    config: dict,
    seeds: Iterable[int],
    learning_rates: Iterable[float],
    checkpoint_root: Path,
) -> bool:
    """Require one valid result row for every selected training checkpoint."""
    source = Path(path)
    if not source.exists():
        return False
    try:
        rows = _read_jsonl(source)
    except ValueError:
        return False
    if any(not _training_record_valid(config, row, checkpoint_root) for row in rows):
        return False
    try:
        counts = Counter(
            (int(row["seed"]), float(row["learning_rate"]), int(row["step"])) for row in rows
        )
    except (KeyError, TypeError, ValueError):
        return False
    expected = {
        (int(seed), float(rate), int(step))
        for seed in seeds
        for rate in learning_rates
        for step in config["train"]["checkpoint_steps"]
    }
    return all(counts[cell] == 1 for cell in expected) and all(count == 1 for count in counts.values())


def _replace_records(path: str | Path, records: Iterable[dict], key_fields: tuple[str, ...]) -> None:
    """Atomically replace exact compatible cells, never appending duplicate rows."""
    destination = Path(path)
    new_rows = list(records)
    if not new_rows:
        raise ValueError("replacement records cannot be empty")
    base_digests = {row.get("base_config_sha256") for row in new_rows}
    record_types = {row.get("record_type") for row in new_rows}
    if len(base_digests) != 1 or None in base_digests or len(record_types) != 1 or None in record_types:
        raise ValueError("replacement records must share one identity")
    base_digest = next(iter(base_digests))
    record_type = next(iter(record_types))
    existing = _read_jsonl(destination) if destination.exists() else []

    def key(row: dict) -> tuple:
        try:
            return tuple(row[field] for field in key_fields)
        except KeyError as error:
            raise ValueError(f"record is missing key field {error.args[0]}") from error

    if any(row.get("base_config_sha256") != base_digest for row in existing):
        raise ValueError("existing JSONL contains incompatible base digest")
    if any(row.get("record_type") != record_type for row in existing):
        raise ValueError("existing JSONL contains incompatible record types")
    for label, rows in (("existing", existing), ("replacement", new_rows)):
        if any(count != 1 for count in Counter(key(row) for row in rows).values()):
            raise ValueError(f"duplicate {label} competence-time cells")
    replacement_keys = {key(row) for row in new_rows}
    retained = [row for row in existing if key(row) not in replacement_keys]
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    write_jsonl(temporary, [*retained, *new_rows])
    temporary.replace(destination)


def _load_raw_rows(path: str | Path, label: str) -> list[dict]:
    try:
        return _read_jsonl(path)
    except ValueError as error:
        raise SystemExit(f"{label} result file is invalid: {error}") from error


def _selected_grid(config: dict, args: argparse.Namespace) -> tuple[list[int], list[float]]:
    expected_seeds = [int(seed) for seed in config["competence_time"]["seeds"]]
    expected_rates = [float(rate) for rate in config["competence_time"]["learning_rates"]]
    seeds = args.seeds or expected_seeds
    rates = args.learning_rates or expected_rates
    if not set(seeds) <= set(expected_seeds) or not set(rates) <= set(expected_rates):
        raise SystemExit("selected seeds and learning rates must belong to the configured grid")
    return [int(seed) for seed in seeds], [float(rate) for rate in rates]


def _preflight_outputs(config: dict, args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    """Reject incompatible existing outputs before touching any checkpoint."""
    try:
        training, probes = [
            _read_jsonl(path) if Path(path).exists() else []
            for path in (args.training_results, args.probe_results)
        ]
        validate_competence_time_grid(
            config, training, probes, require_complete=False, validate_science=False
        )
        for row in [*training, *probes]:
            expected = competence_time_checkpoint_path(
                args.checkpoint_dir, int(row["seed"]), float(row["learning_rate"]), int(row["step"])
            )
            if row.get("checkpoint_path") != str(expected):
                raise ValueError("raw checkpoint path has incompatible identity or root")
        if Path(args.summary_results).exists():
            summaries = _read_jsonl(args.summary_results)
            if len(summaries) != 1 or (
                summaries[0].get("record_type") != "competence_time_summary"
                or summaries[0].get("base_config_sha256") != config_digest(config)
            ):
                raise ValueError("summary has incompatible identity or provenance")
    except (ValueError, TypeError, KeyError, OverflowError) as error:
        raise SystemExit(f"competence-time existing outputs are incompatible: {error}") from error
    return training, probes


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seeds, learning_rates = _selected_grid(config, args)
    checkpoint_root = Path(args.checkpoint_dir)

    preserve_invalid_grid = False
    if args.mode in {"train", "probe", "all"}:
        training, probes = _preflight_outputs(config, args)
        if args.mode == "all":
            try:
                validate_competence_time_grid(config, training, probes, validate_science=False)
            except ValueError:
                # Compatible but incomplete work still needs cache recovery.
                pass
            else:
                preserve_invalid_grid = bool(
                    analyze_competence_time(config, training, probes)["validity_failures"]
                )
                if preserve_invalid_grid:
                    print("preserving complete raw grid with scientific validity failures")

    if args.mode in {"train", "all"} and not preserve_invalid_grid:
        training_path = Path(args.training_results)
        written = 0
        reused = 0
        for seed in seeds:
            for rate in learning_rates:
                cached = _checkpoint_cache_complete(config, [seed], [rate], checkpoint_root)
                recorded = _training_records_complete(
                    training_path, config, [seed], [rate], checkpoint_root
                )
                if cached and recorded:
                    reused += 1
                else:
                    rows = run_competence_time_training(config, [seed], [rate], checkpoint_root)
                    _replace_records(
                        training_path, rows,
                        ("base_config_sha256", "seed", "learning_rate", "step"),
                    )
                    written += 1
        if written:
            print("wrote competence-time training records")
        if reused:
            print(f"reused {reused} complete competence-time training cells")

    if args.mode in {"probe", "all"} and not preserve_invalid_grid:
        if not _checkpoint_cache_complete(config, seeds, learning_rates, checkpoint_root):
            raise SystemExit("competence-time checkpoint set is incomplete or incompatible")
        rows = run_competence_time_probes(config, seeds, learning_rates, checkpoint_root)
        _replace_records(
            args.probe_results,
            rows,
            ("base_config_sha256", "seed", "learning_rate", "step", "site", "control"),
        )
        print("wrote competence-time probe records")

    if args.mode in {"analyze", "all"}:
        training = _load_raw_rows(args.training_results, "training")
        probes = _load_raw_rows(args.probe_results, "probe")
        try:
            validate_competence_time_grid(config, training, probes, validate_science=False)
        except (ValueError, TypeError, KeyError, OverflowError) as error:
            raise SystemExit(f"competence-time raw grid is incomplete or incompatible: {error}") from error
        write_jsonl(args.summary_results, [analyze_competence_time(config, training, probes)])
        print("wrote competence-time summary")

    if args.mode in {"figures", "all"}:
        try:
            from nonergodic_memory.mess3_competence_time_figures import (
                generate_competence_time_figures,
            )
        except ImportError as error:
            raise SystemExit("competence-time figures are not available until Task 5") from error
        training = _load_raw_rows(args.training_results, "training")
        probes = _load_raw_rows(args.probe_results, "probe")
        summaries = _load_raw_rows(args.summary_results, "summary")
        if len(summaries) != 1 or summaries[0].get("record_type") != "competence_time_summary":
            raise SystemExit("competence-time summary file must contain exactly one summary")
        for path in generate_competence_time_figures(config, training, probes, summaries[0], args.output_dir):
            print(f"generated {path}")

    if args.mode == "all":
        print("validated complete competence-time grid")


if __name__ == "__main__":
    main()
