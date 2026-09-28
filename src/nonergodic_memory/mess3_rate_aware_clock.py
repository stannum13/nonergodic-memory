"""Immutable old-data forecasts and prospective rate-aware-clock analysis."""

from __future__ import annotations

import hashlib
import json
import copy
import re
from collections import Counter
from pathlib import Path

import numpy as np

from .experiment import config_digest


# Independent of the caller's configuration and of the mutable YAML file.
_FROZEN_FORECAST = {
    "provenance": {
        "training_jsonl_sha256": "2c90e72389db3a98e0c1196fffaf6bdf24f3492009460bfbe0c99f417315420f",
        "probe_jsonl_sha256": "9a7aa242f267bddc2064c8ddf93c3163891167630d7dd3f0ea4e603993e3abd3",
        "seeds": list(range(30, 38)),
        "learning_rates": [0.00075, 0.0015, 0.003, 0.006],
        "post_initialization_steps": [384, 768, 1152, 1536, 2048, 2560, 3072, 4096],
        "site": "block_2", "control": "none", "metric": "component_posterior_r2",
    },
    "competence": {
        "center": 0.4428598689138331, "scale": 0.3380605976966924,
        "coefficients": [0.03302589694739769, 0.14699110421172318, 0.09640601399613305],
        "support": [-0.24055542481822356, 0.8940463808722273],
        "in_sample_mse": 0.00761248,
    },
    "rate_aware_clock": {
        "transform": "log1p(learning_rate * step)",
        "center": 1.5596899070965387, "scale": 0.7723241912698406,
        "coefficients": [0.08426303744815684, 0.13373197023956584, 0.045168873495373824],
        "support": [0.2530906276821619, 3.2416544117575405],
        "in_sample_mse": 0.00543398,
    },
}


def forecast_geometry(value: float, spec: dict) -> float:
    """Evaluate the raw frozen quadratic; the caller supplies transformed input."""
    z = (float(value) - spec["center"]) / spec["scale"]
    intercept, linear, quadratic = spec["coefficients"]
    return float(intercept + linear * z + quadratic * z**2)


def verify_forecast_provenance(config: dict, root: Path) -> dict:
    """Verify immutable sources and independently reproduce the 256-row fits.

    Exact JSON equality protects serialized constants, including a one-ULP
    change. Only the independent NumPy refit uses absolute tolerance 1e-12.
    ``root`` is the repository containing the retained ``results/`` files.
    """
    if json.dumps(config.get("forecast"), sort_keys=True, allow_nan=False) != json.dumps(
        _FROZEN_FORECAST, sort_keys=True, allow_nan=False
    ):
        raise ValueError("configuration differs from frozen forecast specification")
    provenance = _FROZEN_FORECAST["provenance"]
    rows, hashes = {}, {}
    for label, suffix in (("training", "training"), ("probe", "probes")):
        data = (Path(root) / "results" / f"mess3_competence_time_{suffix}.jsonl").read_bytes()
        key = f"{label}_jsonl_sha256"
        hashes[key] = hashlib.sha256(data).hexdigest()
        if hashes[key] != provenance[key]:
            raise ValueError(f"frozen {label} source hash mismatch")
        rows[label] = [json.loads(line) for line in data.splitlines() if line.strip()]
    expected = {
        (seed, rate, step)
        for seed in provenance["seeds"]
        for rate in provenance["learning_rates"]
        for step in provenance["post_initialization_steps"]
    }
    selected = {}
    for label in rows:
        cells = {}
        for row in rows[label]:
            cell = (row["seed"], row["learning_rate"], row["step"])
            if cell not in expected:
                continue
            if label == "probe" and (row["site"], row["control"]) != (
                provenance["site"], provenance["control"]
            ):
                continue
            if cell in cells:
                raise ValueError(f"duplicate frozen {label} cell")
            cells[cell] = row
        if set(cells) != expected:
            raise ValueError(f"incomplete frozen {label} grid")
        selected[label] = cells
    ordered = sorted(expected)
    target = np.asarray([selected["probe"][cell][provenance["metric"]] for cell in ordered])
    inputs = {
        "competence": np.asarray([selected["training"][cell]["competence"] for cell in ordered]),
        "rate_aware_clock": np.log1p([rate * step for _, rate, step in ordered]),
    }
    refit = {}
    for name, values in inputs.items():
        center, scale = float(values.mean()), float(values.std(ddof=0))
        z = (values - center) / scale
        design = np.column_stack((np.ones_like(z), z, z**2))
        coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
        refit[name] = {
            "center": center, "scale": scale, "coefficients": coefficients.tolist(),
            "support": [float(values.min()), float(values.max())],
            "in_sample_mse": float(np.mean((target - design @ coefficients)**2)),
        }
        for key in ("center", "scale", "coefficients", "support"):
            if not np.allclose(refit[name][key], _FROZEN_FORECAST[name][key], atol=1e-12, rtol=0):
                raise ValueError(f"frozen {name} numerical refit mismatch: {key}")
    return {**hashes, "rows": len(ordered), "refit": refit}


def _is_sha256(value) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _integer(value) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value != int(value):
        raise ValueError("seed, step and counts must be finite integers")
    return int(value)


def _validate_grid(config: dict, training: list[dict], probes: list[dict]) -> None:
    experiment = config["rate_aware_clock"]
    expected = {(seed, rate, step) for seed in experiment["seeds"]
                for rate in experiment["learning_rates"] for step in config["train"]["checkpoint_steps"]}
    sites = ("block_1", "block_2", "final_norm")
    expected_probes = {(*cell, site, control) for cell in expected
                       for site in sites for control in ("none", "shuffled_labels")}
    rate_digests = {}
    for rate in experiment["learning_rates"]:
        selected = copy.deepcopy(config)
        selected["train"]["learning_rate"] = rate
        rate_digests[rate] = config_digest(selected)
    paths = {}
    for label, rows, metrics, expected_cells in (
        ("training", training, ("competence", "kl_exact", "nll", "uniform_kl"), expected),
        ("probe", probes, ("component_accuracy", "component_posterior_r2", "conditional_state_accuracy",
                           "state_posterior_r2", "joint_belief_mse", "joint_belief_r2", "joint_distance_r2"),
         expected_probes),
    ):
        keys = []
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("raw rows must be objects")
            cell = (_integer(row["seed"]), float(row["learning_rate"]), _integer(row["step"]))
            seed, rate, step = cell
            if cell not in expected:
                raise ValueError("unexpected grid cell")
            if (row.get("record_type") != f"rate_aware_clock_{label}" or row.get("condition") != "fresh"
                    or row.get("sampler") != "vectorized"):
                raise ValueError("raw row identity mismatch")
            if (row.get("base_config_sha256") != config_digest(config)
                    or row.get("rate_config_sha256") != rate_digests[rate]
                    or row.get("config_sha256") != rate_digests[rate]):
                raise ValueError("raw row configuration provenance mismatch")
            checkpoint = row["checkpoint_path"]
            if not isinstance(checkpoint, str) or not checkpoint:
                raise ValueError("missing checkpoint path")
            path = Path(checkpoint)
            label_rate = format(rate, ".8g").replace("-", "m").replace(".", "p")
            canonical = path.parent.parent / f"lr_{label_rate}" / f"transformer_seed{seed}_fresh_step{step}.pt"
            if checkpoint != str(canonical) or ".." in path.parts:
                raise ValueError("noncanonical checkpoint path")
            if cell in paths and paths[cell] != checkpoint:
                raise ValueError("training/probe checkpoint path mismatch")
            paths[cell] = checkpoint
            if not all(isinstance(row[metric], (int, float)) and not isinstance(row[metric], bool)
                       and np.isfinite(row[metric]) for metric in metrics):
                raise ValueError("missing, nonnumeric or nonfinite required metric")
            if label == "training":
                if row.get("parameters_finite") is not True or not _is_sha256(row.get("parameter_sha256")):
                    raise ValueError("nonfinite or unaudited model parameters")
                keys.append(cell)
            else:
                if float(row.get("probe_sequence_overlap", -1)) != 0:
                    raise ValueError("probe sequence overlap")
                keys.append((*cell, row["site"], row["control"]))
        if any(count != 1 for count in Counter(keys).values()) or set(keys) != expected_cells:
            raise ValueError(f"missing, duplicate or unexpected {label} cells")


def _validate_audit(config: dict, audit: list[dict]) -> None:
    expected_counts = {"evaluation": config["data"]["test_sequences"],
                       "probe_fit": config["probe"]["train_sequences"],
                       "probe_test": config["probe"]["test_sequences"]}
    pairs = {"evaluation__probe_fit", "evaluation__probe_test", "probe_fit__probe_test"}
    seeds = []
    for row in audit:
        if not isinstance(row, dict):
            raise ValueError("audit rows must be objects")
        seeds.append(_integer(row["seed"]))
        if row.get("record_type") != "rate_aware_clock_audit" or row.get("base_config_sha256") != config_digest(config):
            raise ValueError("audit identity mismatch")
        if set(row["datasets"]) != set(expected_counts) or set(row["intersections"]) != pairs:
            raise ValueError("incomplete token audit")
        for name, count in expected_counts.items():
            dataset = row["datasets"][name]
            if (not _is_sha256(dataset["sha256"]) or _integer(dataset["n_rows"]) != count
                    or not 0 <= _integer(dataset["n_unique_rows"]) <= count):
                raise ValueError("invalid token dataset hash or count")
        if any(_integer(value) != 0 for value in row["intersections"].values()):
            raise ValueError("token arrays overlap")
    if len(seeds) != len(set(seeds)) or set(seeds) != set(config["rate_aware_clock"]["seeds"]):
        raise ValueError("missing, duplicate or unexpected audit seeds")


def _score_forecasts(config: dict, cells: dict, primary_probes: list[dict]) -> dict:
    per_seed = []
    for seed in config["rate_aware_clock"]["seeds"]:
        errors = {"clock": [], "competence": []}
        for row in primary_probes:
            if row["seed"] != seed or row["control"] != "none":
                continue
            rate, step = row["learning_rate"], row["step"]
            competence = cells[seed, rate, step]["competence"]
            forecasts = {
                "competence": forecast_geometry(competence, config["forecast"]["competence"]),
                "clock": forecast_geometry(np.log1p(rate * step), config["forecast"]["rate_aware_clock"]),
            }
            for name, prediction in forecasts.items():
                errors[name].append((float(row["component_posterior_r2"]) - prediction)**2)
        per_seed.append({"seed": seed, "observations": len(errors["clock"]),
                         "clock_mse": float(np.mean(errors["clock"])),
                         "competence_mse": float(np.mean(errors["competence"]))})
    clock = float(np.mean([row["clock_mse"] for row in per_seed]))
    competence = float(np.mean([row["competence_mse"] for row in per_seed]))
    if not np.isfinite([clock, competence]).all():
        raise ValueError("nonfinite forecast squared errors")
    result = {
        "observations": sum(row["observations"] for row in per_seed), "per_seed": per_seed,
        "clock_mse": clock, "competence_mse": competence,
        "clock_to_competence_mse_ratio": clock / competence if competence > 0 else None,
        "competence_to_clock_mse_ratio": competence / clock if clock > 0 else None,
        "clock_seed_wins": sum(row["clock_mse"] < row["competence_mse"] for row in per_seed),
        "competence_seed_wins": sum(row["competence_mse"] < row["clock_mse"] for row in per_seed),
    }
    if competence == 0:
        result["ratio_reason"] = "zero_competence_mse"
    return result


def analyze_rate_aware_clock(config, training, probes, audit) -> dict:
    """Apply the preregistered validity-first three-way decision, without fitting.

    Only retained old rows enter the provenance refit. All confirmation scoring
    uses the serialized coefficients. Invalid observations produce an auditable
    inconclusive summary; step zero is never forecast or checked for support.
    """
    result = {
        "record_type": "rate_aware_clock_summary", "base_config_sha256": config_digest(config),
        "target": "block_2_component_posterior_r2", "validation": "frozen_external_forecast",
        "selection": "step > 0", "verdict": "inconclusive", "validity_failures": [],
        "primary": None, "rate_dissociation": None, "provenance": None,
        "excluded_initialization_count": None,
    }
    failures = result["validity_failures"]
    try:
        result["provenance"] = verify_forecast_provenance(config, Path(__file__).resolve().parents[2])
    except (OSError, ValueError, TypeError, KeyError, np.linalg.LinAlgError) as error:
        failures.append("forecast_provenance")
        result["provenance_error"] = str(error)
        return result
    # The prospective grid and thresholds are locked as well as the forecasts.
    if config_digest(config) != "59f938bbff48f300":
        failures.append("invalid_configuration")
        return result
    experiment = config["rate_aware_clock"]
    thresholds = experiment["thresholds"]
    result["thresholds"] = copy.deepcopy(thresholds)
    result["primary_checkpoints"] = list(experiment["primary_checkpoints"])
    try:
        _validate_grid(config, training, probes)
    except (ValueError, TypeError, KeyError, OverflowError) as error:
        failures.append("invalid_grid")
        result["grid_error"] = str(error)
        return result
    try:
        _validate_audit(config, audit)
    except (ValueError, TypeError, KeyError, OverflowError) as error:
        failures.append("token_isolation")
        result["audit_error"] = str(error)
    cells = {(row["seed"], row["learning_rate"], row["step"]): row for row in training}
    seeds, rates, steps = experiment["seeds"], experiment["learning_rates"], experiment["primary_checkpoints"]
    result["excluded_initialization_count"] = sum(row["step"] == 0 for row in training)
    if any(cells[seed, rates[0], 0]["parameter_sha256"] != cells[seed, rates[1], 0]["parameter_sha256"]
           for seed in seeds):
        failures.append("initialization_pairing")
    primary = [row for row in probes if row["site"] == "block_2" and row["step"] in steps]
    max_shuffled = max(abs(float(row["component_posterior_r2"])) for row in primary
                       if row["control"] == "shuffled_labels")
    result["max_abs_shuffled_component_r2"] = max_shuffled
    if max_shuffled > thresholds["max_shuffled_component_posterior_r2"]:
        failures.append("shuffled_labels")
    differences = [{"seed": seed, "max_competence_difference": max(
        abs(cells[seed, rates[0], step]["competence"] - cells[seed, rates[1], step]["competence"])
        for step in steps)} for seed in seeds]
    passing = [row["seed"] for row in differences
               if row["max_competence_difference"] >= thresholds["minimum_rate_competence_difference"]]
    result["rate_dissociation"] = {
        "per_seed": differences, "passing_seeds": passing,
        "valid": len(passing) >= thresholds["minimum_rate_dissociation_seeds"],
    }
    if not result["rate_dissociation"]["valid"]:
        failures.append("rate_dissociation")
    unsupported = []
    for (seed, rate, step), row in cells.items():
        if step not in steps:
            continue
        for name, value in (("competence", row["competence"]), ("rate_aware_clock", np.log1p(rate * step))):
            lower, upper = config["forecast"][name]["support"]
            if not lower <= value <= upper:
                unsupported.append({"seed": seed, "learning_rate": rate, "step": step, "predictor": name})
    result["out_of_support"] = unsupported
    if unsupported:
        failures.append("forecast_support")
    try:
        result["primary"] = _score_forecasts(config, cells, primary)
    except (ValueError, OverflowError) as error:
        failures.append("nonfinite_analysis")
        result["analysis_error"] = str(error)
        return result
    if not failures:
        scores = result["primary"]
        supported = (scores["clock_mse"] < thresholds["clock_to_competence_mse_ratio"] * scores["competence_mse"]
                     and scores["clock_seed_wins"] >= thresholds["minimum_clock_seed_wins"])
        result["verdict"] = "supported" if supported else "falsified"
    return result
