"""Isolated Mess3 storage for the preregistered competence--time experiment."""

from __future__ import annotations

import copy
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np

from .experiment import config_digest
from .mess3_diagnosis import evaluate_checkpoint_geometry, train_diagnostic


def _rate_label(learning_rate: float) -> str:
    return format(float(learning_rate), ".8g").replace("-", "m").replace(".", "p")


def _rate_config(config: dict, learning_rate: float) -> dict:
    selected = copy.deepcopy(config)
    selected["train"]["learning_rate"] = float(learning_rate)
    # The shared diagnostic utility consumes this explicitly locked value.
    if int(selected["diagnosis"]["window"]) < 1:
        raise ValueError("diagnosis window must be positive")
    return selected


def competence_time_checkpoint_path(
    root: str | Path,
    seed: int,
    learning_rate: float,
    step: int,
) -> Path:
    """Return this experiment's checkpoint path without touching v1.0 paths."""
    return (
        Path(root)
        / f"lr_{_rate_label(learning_rate)}"
        / f"transformer_seed{seed}_fresh_step{step}.pt"
    )


def _enrich_record(
    record: dict,
    base_digest: str,
    rate_config: dict,
    learning_rate: float,
) -> dict:
    rate_digest = config_digest(rate_config)
    return {
        **record,
        "learning_rate": float(learning_rate),
        "learning_rate_label": _rate_label(learning_rate),
        "base_config_sha256": base_digest,
        "rate_config_sha256": rate_digest,
        "config_sha256": rate_digest,
        "sampler": "vectorized",
    }


def run_competence_time_training(
    config: dict,
    seeds: Iterable[int],
    learning_rates: Iterable[float],
    checkpoint_root: str | Path = "checkpoints/mess3_competence_time",
) -> list[dict]:
    """Train paired fresh-data curves in the experiment-specific checkpoint tree."""
    if config["data"].get("sampler") != "vectorized":
        raise ValueError("competence-time training requires sampler=vectorized")
    base_digest = config_digest(config)
    records: list[dict] = []
    for seed in seeds:
        for learning_rate in learning_rates:
            rate_config = _rate_config(config, learning_rate)
            destination = Path(checkpoint_root) / f"lr_{_rate_label(learning_rate)}"
            run = train_diagnostic(rate_config, int(seed), "fresh", destination)
            for record in run:
                enriched = _enrich_record(record, base_digest, rate_config, learning_rate)
                enriched["record_type"] = "competence_time_training"
                enriched["checkpoint_path"] = str(
                    competence_time_checkpoint_path(
                        checkpoint_root, int(seed), learning_rate, int(record["step"])
                    )
                )
                records.append(enriched)
    return records


def run_competence_time_probes(
    config: dict,
    seeds: Iterable[int],
    learning_rates: Iterable[float],
    checkpoint_root: str | Path = "checkpoints/mess3_competence_time",
) -> list[dict]:
    """Probe every stored experiment checkpoint, site, and label control."""
    if config["data"].get("sampler") != "vectorized":
        raise ValueError("competence-time probes require sampler=vectorized")
    base_digest = config_digest(config)
    records: list[dict] = []
    for seed in seeds:
        for learning_rate in learning_rates:
            rate_config = _rate_config(config, learning_rate)
            for step in rate_config["train"]["checkpoint_steps"]:
                checkpoint = competence_time_checkpoint_path(
                    checkpoint_root, int(seed), learning_rate, int(step)
                )
                run = evaluate_checkpoint_geometry(
                    rate_config, checkpoint, int(seed), condition="fresh"
                )
                for record in run:
                    enriched = _enrich_record(record, base_digest, rate_config, learning_rate)
                    enriched["record_type"] = "competence_time_probe"
                    enriched["checkpoint_path"] = str(checkpoint)
                    records.append(enriched)
    return records


def validate_competence_time_grid(
    config: dict,
    training: list[dict],
    probes: list[dict],
) -> None:
    """Reject incomplete or incompatible raw experiment cells before analysis."""
    base_digest = config_digest(config)
    all_rows = [*training, *probes]
    if any(row.get("base_config_sha256") != base_digest for row in all_rows):
        raise ValueError("competence-time records have incompatible provenance")
    seeds = {int(seed) for seed in config["competence_time"]["seeds"]}
    rates = {float(rate) for rate in config["competence_time"]["learning_rates"]}
    steps = {int(step) for step in config["train"]["checkpoint_steps"]}
    rate_digests = {rate: config_digest(_rate_config(config, rate)) for rate in rates}
    required = {
        "seed",
        "learning_rate",
        "step",
        "base_config_sha256",
        "rate_config_sha256",
        "sampler",
        "checkpoint_path",
    }
    for label, rows, record_type in (
        ("training", training, "competence_time_training"),
        ("probe", probes, "competence_time_probe"),
    ):
        if any(not required <= row.keys() for row in rows):
            raise ValueError(f"competence-time {label} records have incomplete raw keys")
        if any(
            row.get("record_type") != record_type or row.get("sampler") != "vectorized"
            for row in rows
        ):
            raise ValueError(f"competence-time {label} records have incompatible identity")
        for row in rows:
            rate = float(row["learning_rate"])
            digest = rate_digests.get(rate)
            if row.get("rate_config_sha256") != digest or row.get("config_sha256") != digest:
                raise ValueError("competence-time records have incompatible rate provenance")

    expected_training = {(seed, rate, step) for seed in seeds for rate in rates for step in steps}
    training_keys = [
        (int(row["seed"]), float(row["learning_rate"]), int(row["step"])) for row in training
    ]
    if any(count > 1 for count in Counter(training_keys).values()):
        raise ValueError("duplicate competence-time training cells")
    if set(training_keys) != expected_training:
        raise ValueError("competence-time training grid is incomplete or contains unexpected cells")
    training_metrics = ("competence", "kl_exact", "nll", "uniform_kl")
    if any(
        not all(np.isfinite(float(row.get(metric, float("nan")))) for metric in training_metrics)
        for row in training
    ):
        raise ValueError("competence-time training records contain non-finite metrics")

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
        raise ValueError("duplicate competence-time probe cells")
    if set(probe_keys) != expected_probes:
        raise ValueError("competence-time probe grid is incomplete or contains unexpected cells")
    if any(int(row.get("probe_sequence_overlap", -1)) != 0 for row in probes):
        raise ValueError("competence-time probe records contain sequence overlap")
    probe_metrics = (
        "component_accuracy",
        "component_posterior_r2",
        "conditional_state_accuracy",
        "state_posterior_r2",
        "joint_belief_mse",
        "joint_belief_r2",
        "joint_distance_r2",
    )
    if any(
        not all(np.isfinite(float(row.get(metric, float("nan")))) for metric in probe_metrics)
        for row in probes
    ):
        raise ValueError("competence-time probe records contain non-finite metrics")
