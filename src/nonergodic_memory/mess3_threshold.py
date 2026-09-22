"""Paired Mess3 learning curves for competence-versus-step tests."""

from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

from .experiment import config_digest, write_jsonl
from .mess3_diagnosis import evaluate_checkpoint_geometry, train_diagnostic


def _rate_label(learning_rate: float) -> str:
    return format(float(learning_rate), ".8g").replace("-", "m").replace(".", "p")


def _rate_config(config: dict, learning_rate: float) -> dict:
    selected = copy.deepcopy(config)
    selected["train"]["learning_rate"] = float(learning_rate)
    return selected


def threshold_checkpoint_path(
    root: str | Path,
    seed: int,
    learning_rate: float,
    step: int,
) -> Path:
    return (
        Path(root)
        / f"lr_{_rate_label(learning_rate)}"
        / f"transformer_seed{seed}_fresh_step{step}.pt"
    )


def _enrich_record(record: dict, base_digest: str, learning_rate: float) -> dict:
    return {
        **record,
        "learning_rate": float(learning_rate),
        "learning_rate_label": _rate_label(learning_rate),
        "base_config_sha256": base_digest,
        "sampler": "vectorized",
    }


def run_threshold_training(
    config: dict,
    seeds: Iterable[int],
    learning_rates: Iterable[float],
    checkpoint_root: str | Path,
) -> list[dict]:
    """Train a complete selected set of paired fresh-data learning curves."""
    if config["data"].get("sampler") != "vectorized":
        raise ValueError("Mess3 threshold training requires sampler=vectorized")
    base_digest = config_digest(config)
    records: list[dict] = []
    for seed in seeds:
        for learning_rate in learning_rates:
            rate_config = _rate_config(config, learning_rate)
            destination = Path(checkpoint_root) / f"lr_{_rate_label(learning_rate)}"
            run = train_diagnostic(rate_config, int(seed), "fresh", destination)
            for record in run:
                enriched = _enrich_record(record, base_digest, learning_rate)
                enriched["record_type"] = "threshold_training"
                enriched["config_sha256"] = config_digest(rate_config)
                enriched["checkpoint_path"] = str(
                    threshold_checkpoint_path(
                        checkpoint_root, int(seed), learning_rate, int(record["step"])
                    )
                )
                records.append(enriched)
    return records


def run_threshold_probes(
    config: dict,
    seeds: Iterable[int],
    learning_rates: Iterable[float],
    checkpoint_root: str | Path,
) -> list[dict]:
    """Probe every selected rate, seed, checkpoint, layer, and control."""
    base_digest = config_digest(config)
    records: list[dict] = []
    for seed in seeds:
        for learning_rate in learning_rates:
            rate_config = _rate_config(config, learning_rate)
            for step in rate_config["train"]["checkpoint_steps"]:
                checkpoint = threshold_checkpoint_path(
                    checkpoint_root, int(seed), learning_rate, int(step)
                )
                run = evaluate_checkpoint_geometry(
                    rate_config, checkpoint, int(seed), condition="fresh"
                )
                for record in run:
                    enriched = _enrich_record(record, base_digest, learning_rate)
                    enriched["record_type"] = "threshold_probe"
                    records.append(enriched)
    return records


def validate_threshold_grid(config: dict, training: list[dict], probes: list[dict]) -> None:
    """Reject incomplete, duplicated, or incompatible threshold result grids."""
    base_digest = config_digest(config)
    if any(row.get("base_config_sha256") != base_digest for row in [*training, *probes]):
        raise ValueError("threshold records have incompatible provenance")
    seeds = {int(seed) for seed in config["threshold"]["seeds"]}
    rates = {float(rate) for rate in config["threshold"]["learning_rates"]}
    steps = {int(step) for step in config["train"]["checkpoint_steps"]}
    expected_training = {
        (seed, rate, step) for seed in seeds for rate in rates for step in steps
    }
    training_keys = [
        (int(row["seed"]), float(row["learning_rate"]), int(row["step"]))
        for row in training
    ]
    if any(count > 1 for count in Counter(training_keys).values()):
        raise ValueError("duplicate threshold training cells")
    if set(training_keys) != expected_training:
        raise ValueError("threshold training grid is incomplete or contains unexpected cells")
    sites = [
        *(f"block_{depth + 1}" for depth in range(int(config["model"]["layers"]))),
        "final_norm",
    ]
    expected_probes = {
        (*cell, site, control)
        for cell in expected_training
        for site in sites
        for control in ("none", "shuffled_labels")
    }
    probe_keys = [
        (
            int(row["seed"]),
            float(row["learning_rate"]),
            int(row["step"]),
            row["site"],
            row["control"],
        )
        for row in probes
    ]
    if any(count > 1 for count in Counter(probe_keys).values()):
        raise ValueError("duplicate threshold probe cells")
    if set(probe_keys) != expected_probes:
        raise ValueError("threshold probe grid is incomplete or contains unexpected cells")


def replace_threshold_records(
    path: str | Path,
    records: Iterable[dict],
    key_fields: tuple[str, ...],
) -> None:
    """Atomically replace selected same-experiment cells and retain the rest."""
    destination = Path(path)
    new_rows = list(records)
    if not new_rows:
        raise ValueError("replacement records cannot be empty")
    base_digests = {row.get("base_config_sha256") for row in new_rows}
    if len(base_digests) != 1 or None in base_digests:
        raise ValueError("replacement records must share one base_config_sha256")
    base_digest = next(iter(base_digests))
    existing = []
    if destination.exists():
        existing = [json.loads(line) for line in destination.read_text().splitlines() if line.strip()]

    def key(row: dict) -> tuple:
        try:
            return tuple(row[field] for field in key_fields)
        except KeyError as error:
            raise ValueError(f"record is missing key field {error.args[0]}") from error

    for label, rows in (("existing", existing), ("replacement", new_rows)):
        if any(count > 1 for count in Counter(key(row) for row in rows).values()):
            raise ValueError(f"duplicate {label} threshold cells")
    replacement_keys = {key(row) for row in new_rows}
    retained = [
        row
        for row in existing
        if row.get("base_config_sha256") == base_digest and key(row) not in replacement_keys
    ]
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    write_jsonl(temporary, [*retained, *new_rows])
    temporary.replace(destination)
