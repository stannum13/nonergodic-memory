"""Storage and frozen analysis for the competence--time experiment."""

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
    *,
    require_complete: bool = True,
    validate_science: bool = True,
) -> None:
    """Check identity and structure, optionally requiring a complete, valid grid.

    Partial structural validation supports preflight before resumable writes.
    Analysis orchestration leaves scientific failures to the frozen analyzer.
    """
    base_digest = config_digest(config)
    all_rows = [*training, *probes]
    if any(not isinstance(row, dict) for row in all_rows):
        raise ValueError("competence-time raw rows must be objects")
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
            if any(float(row[key]) != int(row[key]) for key in ("seed", "step")):
                raise ValueError("competence-time seed and step must be integers")
            rate = float(row["learning_rate"])
            digest = rate_digests.get(rate)
            if row.get("rate_config_sha256") != digest or row.get("config_sha256") != digest:
                raise ValueError("competence-time records have incompatible rate provenance")
            if label == "training":
                if row.get("condition") != "fresh":
                    raise ValueError("competence-time training condition must be fresh")
                checkpoint = row.get("checkpoint_path")
                if not isinstance(checkpoint, str) or not checkpoint:
                    raise ValueError("competence-time training checkpoint path is missing")
                path = Path(checkpoint)
                expected = competence_time_checkpoint_path(
                    path.parent.parent, int(row["seed"]), rate, int(row["step"])
                )
                if checkpoint != str(expected):
                    raise ValueError("competence-time training checkpoint path has incompatible identity")

    expected_training = {(seed, rate, step) for seed in seeds for rate in rates for step in steps}
    training_keys = [
        (int(row["seed"]), float(row["learning_rate"]), int(row["step"])) for row in training
    ]
    if any(count > 1 for count in Counter(training_keys).values()):
        raise ValueError("duplicate competence-time training cells")
    if not set(training_keys) <= expected_training or (
        require_complete and set(training_keys) != expected_training
    ):
        raise ValueError("competence-time training grid is incomplete or contains unexpected cells")

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
    if not set(probe_keys) <= expected_probes or (
        require_complete and set(probe_keys) != expected_probes
    ):
        raise ValueError("competence-time probe grid is incomplete or contains unexpected cells")
    if not validate_science:
        return
    training_metrics = ("competence", "kl_exact", "nll", "uniform_kl")
    if any(
        not all(np.isfinite(float(row.get(metric, float("nan")))) for metric in training_metrics)
        for row in training
    ):
        raise ValueError("competence-time training records contain non-finite metrics")
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


def _competence_time_quadratic(
    train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Fit the registered quadratic, with scaling learned only from train_x."""
    center = float(train_x.mean())
    scale = float(train_x.std())
    # A constant predictor reduces to an intercept; never divide by zero.
    if scale == 0:
        scale = 1.0
    train_z, test_z = (train_x - center) / scale, (test_x - center) / scale
    design = np.column_stack((np.ones_like(train_z), train_z, train_z**2))
    coefficients, *_ = np.linalg.lstsq(design, train_y, rcond=None)
    predictions = np.column_stack((np.ones_like(test_z), test_z, test_z**2)) @ coefficients
    return predictions, {"center": center, "scale": scale}


def _competence_time_loso(rows: list[dict]) -> dict:
    """Hold out every rate/checkpoint of one seed and pool observation errors."""
    seeds = sorted({row["seed"] for row in rows})
    folds = []
    competence_sse, step_sse, observations = 0.0, 0.0, 0
    for seed in seeds:
        train = [row for row in rows if row["seed"] != seed]
        test = [row for row in rows if row["seed"] == seed]
        train_y = np.asarray([row["component_posterior_r2"] for row in train])
        test_y = np.asarray([row["component_posterior_r2"] for row in test])
        competence_prediction, competence_scaling = _competence_time_quadratic(
            np.asarray([row["competence"] for row in train]), train_y,
            np.asarray([row["competence"] for row in test]),
        )
        step_prediction, step_scaling = _competence_time_quadratic(
            np.log1p([row["step"] for row in train]), train_y,
            np.log1p([row["step"] for row in test]),
        )
        competence_error = float(np.sum((test_y - competence_prediction) ** 2))
        step_error = float(np.sum((test_y - step_prediction) ** 2))
        if not np.isfinite([competence_error, step_error]).all():
            raise ValueError("non-finite LOSO squared errors")
        competence_sse += competence_error
        step_sse += step_error
        observations += len(test)
        folds.append({
            "held_out_seed": seed,
            "training_seeds": [other for other in seeds if other != seed],
            "training_cells": len(train), "test_cells": len(test),
            "competence_mse": competence_error / len(test),
            "step_mse": step_error / len(test),
            "scaling": {"competence": competence_scaling, "log_step": step_scaling},
        })
    competence_mse, step_mse = competence_sse / observations, step_sse / observations
    return {
        "observations": observations,
        "competence_loso_mse": competence_mse,
        "log_step_loso_mse": step_mse,
        # Null denotes an undefined ratio (a perfect clock model cannot lose).
        "competence_to_step_mse_ratio": competence_mse / step_mse if step_mse > 0 else None,
        "competence_fold_wins": sum(fold["competence_mse"] < fold["step_mse"] for fold in folds),
        "folds": folds,
    }


def analyze_competence_time(config: dict, training: list[dict], probes: list[dict]) -> dict:
    """Apply PROTOCOL.md's post-initialization decision without mutating raw data.

    Invalid raw grids return an auditable inconclusive summary. Configuration
    errors raise before raw-data validation, so they cannot masquerade as data.
    """
    experiment = config["competence_time"]
    thresholds = experiment["thresholds"]
    seeds = sorted(int(seed) for seed in experiment["seeds"])
    rates = sorted(float(rate) for rate in experiment["learning_rates"])
    steps = sorted(int(step) for step in experiment["primary_checkpoints"])
    if (
        len(seeds) < 2 or len(rates) < 2 or not steps or min(steps) <= 0
        or set(steps) != {int(step) for step in config["train"]["checkpoint_steps"] if int(step) > 0}
        or experiment["primary_site"] != "block_2"
        or experiment["primary_target"] != "component_posterior"
    ):
        raise ValueError("invalid competence-time primary analysis configuration")
    for key in (
        "max_shuffled_component_posterior_r2", "competence_to_step_mse_ratio",
        "minimum_competence_fold_wins", "minimum_rate_dissociation_seeds",
        "minimum_rate_competence_difference",
    ):
        if not np.isfinite(float(thresholds[key])):
            raise ValueError(f"non-finite competence-time threshold: {key}")
    # Validate rate configuration before catching failures in user-supplied rows.
    for rate in rates:
        _rate_config(config, rate)
    result = {
        "record_type": "competence_time_summary",
        "base_config_sha256": config_digest(config),
        "target": "block_2_component_posterior_r2",
        "validation": "leave_one_seed_out", "selection": "step > 0",
        "primary_checkpoints": steps, "thresholds": copy.deepcopy(thresholds),
        "verdict": "inconclusive", "validity_failures": [],
        "primary": None, "rate_dissociation": None,
        "excluded_initialization_count": None,
    }
    try:
        for row in [*training, *probes]:
            if not isinstance(row, dict):
                raise ValueError("competence-time raw rows must be objects")
            if any(float(row[key]) != int(row[key]) for key in ("seed", "step")):
                raise ValueError("competence-time seed and step must be integers")
        # Exact comparison: int(0.5) must not turn overlap into zero.
        if any(float(row.get("probe_sequence_overlap", -1)) != 0 for row in probes):
            result["validity_failures"].append("probe_leakage")
            raise ValueError("competence-time probe records contain sequence overlap")
        validate_competence_time_grid(config, training, probes)
    except (ValueError, TypeError, KeyError, OverflowError) as error:
        if not result["validity_failures"]:
            result["validity_failures"].append("invalid_grid")
        result["grid_error"] = f"{type(error).__name__}: {error}"
        return result

    cells = {
        (int(row["seed"]), float(row["learning_rate"]), int(row["step"])): row
        for row in training
    }
    result["excluded_initialization_count"] = sum(int(row["step"]) == 0 for row in training)
    primary_probes = [row for row in probes if row["site"] == "block_2" and int(row["step"]) in steps]
    max_shuffled = max(abs(float(row["component_posterior_r2"])) for row in primary_probes
                       if row["control"] == "shuffled_labels")
    result["max_abs_shuffled_component_r2"] = max_shuffled
    result["shuffled_control_valid"] = max_shuffled <= thresholds["max_shuffled_component_posterior_r2"]
    if not result["shuffled_control_valid"]:
        result["validity_failures"].append("shuffled_labels")

    differences = [{
        "seed": seed,
        "max_competence_difference": max(
            abs(float(cells[seed, rates[-1], step]["competence"])
                - float(cells[seed, rates[0], step]["competence"])) for step in steps
        ),
    } for seed in seeds]
    passing = [row["seed"] for row in differences if row["max_competence_difference"]
               >= thresholds["minimum_rate_competence_difference"]]
    result["rate_dissociation"] = {
        "slowest_rate": rates[0], "fastest_rate": rates[-1],
        "per_seed": differences, "passing_seeds": passing,
        "valid": len(passing) >= thresholds["minimum_rate_dissociation_seeds"],
    }
    if not result["rate_dissociation"]["valid"]:
        result["validity_failures"].append("rate_dissociation")
    rows = [{
        "seed": int(row["seed"]), "step": int(row["step"]),
        "competence": float(cells[int(row["seed"]), float(row["learning_rate"]), int(row["step"])]["competence"]),
        "component_posterior_r2": float(row["component_posterior_r2"]),
    } for row in primary_probes if row["control"] == "none"]
    try:
        result["primary"] = _competence_time_loso(rows)
    except (ValueError, np.linalg.LinAlgError) as error:
        result["validity_failures"].append("nonfinite_analysis")
        result["analysis_error"] = str(error)
        return result
    if not result["validity_failures"]:
        primary = result["primary"]
        ratio = primary["competence_to_step_mse_ratio"]
        supported = (ratio is not None and ratio < thresholds["competence_to_step_mse_ratio"]
                     and primary["competence_fold_wins"] >= thresholds["minimum_competence_fold_wins"])
        result["verdict"] = "supported" if supported else "falsified"
    return result
