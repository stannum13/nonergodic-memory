"""Immutable old-data forecasts and prospective rate-aware-clock analysis."""

from __future__ import annotations

import hashlib
import json
import copy
import re
import os
import tempfile
import shutil
from datetime import datetime, timezone
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

from .experiment import config_digest, load_checkpoint, mixture_from_config
from .mess3_competence_time import _enrich_record, _rate_config, _rate_label
from .mess3_diagnosis import evaluate_checkpoint_geometry, sample_from_config, train_diagnostic


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


def _validate_grid(config: dict, training: list[dict], probes: list[dict], *, require_complete=True) -> None:
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
            rate = row["learning_rate"]
            if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not np.isfinite(rate):
                raise ValueError("learning rate must be a finite real number")
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
        if (any(count != 1 for count in Counter(keys).values()) or not set(keys) <= expected_cells
                or (require_complete and set(keys) != expected_cells)):
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
        if not isinstance(row["datasets"], Mapping) or not isinstance(row["intersections"], Mapping):
            raise ValueError("audit datasets and intersections must be mappings")
        if set(row["datasets"]) != set(expected_counts) or set(row["intersections"]) != pairs:
            raise ValueError("incomplete token audit")
        for name, count in expected_counts.items():
            dataset = row["datasets"][name]
            if (not _is_sha256(dataset["sha256"]) or _integer(dataset["n_rows"]) != count
                    or not 1 <= _integer(dataset["n_unique_rows"]) <= count):
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
    differences = []
    for seed in seeds:
        difference = max(
            abs(float(cells[seed, rates[0], step]["competence"])
                - float(cells[seed, rates[1], step]["competence"]))
            for step in steps
        )
        if np.isfinite(difference):
            differences.append({"seed": seed, "max_competence_difference": difference})
        else:
            differences.append({"seed": seed, "max_competence_difference": None,
                                "reason": "nonfinite_competence_difference"})
            if "nonfinite_analysis" not in failures:
                failures.append("nonfinite_analysis")
    passing = [row["seed"] for row in differences
               if row["max_competence_difference"] is not None
               and row["max_competence_difference"] >= thresholds["minimum_rate_competence_difference"]]
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
        if "nonfinite_analysis" not in failures:
            failures.append("nonfinite_analysis")
        result["analysis_error"] = str(error)
        return result
    if not failures:
        scores = result["primary"]
        supported = (scores["clock_mse"] < thresholds["clock_to_competence_mse_ratio"] * scores["competence_mse"]
                     and scores["clock_seed_wins"] >= thresholds["minimum_clock_seed_wins"])
        result["verdict"] = "supported" if supported else "falsified"
    return result


def rate_aware_checkpoint_path(root, seed, rate, step) -> Path:
    """Canonical path within the rate-aware experiment's separate checkpoint tree."""
    return Path(root) / f"lr_{_rate_label(rate)}" / f"transformer_seed{seed}_fresh_step{step}.pt"


def read_rate_aware_jsonl(path) -> list[dict]:
    """Read existing evidence strictly; absence is handled by the caller."""
    try:
        rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot read rate-aware evidence: {path}") from error
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("rate-aware evidence must contain JSON objects")
    return rows


def atomic_write_rate_aware_jsonl(path, rows) -> None:
    """Publish one fully serialized snapshot with a same-directory atomic rename.

    This low-level writer requires callers to preflight existing evidence first.
    Serialization happens before directories or temporary files are created.
    """
    content = "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=destination.parent, prefix=".rate-aware-",
                                         suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _parameter_identity(state) -> tuple[str, bool]:
    """Hash sorted tensor names, dtype, shape and contiguous little-endian bytes."""
    if not isinstance(state, Mapping) or not state:
        raise ValueError("checkpoint has no state tensors")
    digest = hashlib.sha256()
    finite = True
    for name in sorted(state):
        tensor = state[name]
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError("checkpoint state must contain named tensors")
        array = tensor.detach().cpu().contiguous().numpy()
        array = np.ascontiguousarray(array.astype(array.dtype.newbyteorder("<"), copy=False))
        header = json.dumps([name, array.dtype.str, list(array.shape)], separators=(",", ":")).encode()
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        digest.update(array.tobytes(order="C"))
        finite = finite and bool(np.isfinite(array).all())
    return digest.hexdigest(), finite


def _checked_checkpoint(config, root, seed, rate, step) -> dict:
    path = rate_aware_checkpoint_path(root, seed, rate, step)
    try:
        payload, _ = load_checkpoint(path, _rate_config(config, rate), "transformer", seed)
        if payload.get("condition") != "fresh" or _integer(payload.get("step")) != step:
            raise ValueError("checkpoint condition or step mismatch")
        digest, finite = _parameter_identity(payload["state_dict"])
        if not finite:
            raise ValueError("nonfinite checkpoint parameters")
    except Exception as error:
        raise ValueError(f"incompatible or invalid checkpoint {path}: {error}") from error
    return {"parameter_sha256": digest, "parameters_finite": finite}


def _storage_selection(config, seeds, rates):
    seeds, rates = list(seeds), list(rates)
    experiment = config["rate_aware_clock"]
    steps = config["train"]["checkpoint_steps"]
    if (config["data"].get("sampler") != "vectorized" or config["data"].get("generator") != "mess3"
            or config["model"]["layers"] != 2):
        raise ValueError("rate-aware storage requires vectorized Mess3 and two model blocks")
    for section, names in (("data", ("sequence_length", "train_sequences", "test_sequences")),
                            ("model", ("width", "heads", "max_length")),
                            ("train", ("batch_size",)), ("probe", ("train_sequences", "test_sequences"))):
        if any(_integer(config[section][name]) < 1 for name in names):
            raise ValueError("model dimensions and dataset counts must be positive integers")
    if (config["model"]["width"] % config["model"]["heads"]
            or config["model"]["max_length"] < config["data"]["sequence_length"] - 1):
        raise ValueError("invalid attention dimensions or maximum context length")
    if not steps or steps != sorted(set(steps)) or steps[0] != 0 or any(_integer(s) < 0 for s in steps):
        raise ValueError("checkpoint steps must be distinct sorted nonnegative integers including zero")
    if (not seeds or len(seeds) != len(set(seeds)) or any(_integer(s) < 0 for s in seeds)
            or not set(seeds) <= set(experiment["seeds"])):
        raise ValueError("selected seeds must be distinct configured integers")
    if (not rates or len(rates) != len(set(rates))
            or any(isinstance(r, bool) or not isinstance(r, (int, float)) or not np.isfinite(r) or r <= 0 for r in rates)
            or not set(rates) <= set(experiment["learning_rates"])):
        raise ValueError("selected rates must be distinct configured positive numbers")
    if not 1 <= config["diagnosis"]["window"] < config["data"]["sequence_length"]:
        raise ValueError("invalid diagnosis window")
    if set(seeds) & set(range(40, 48)):
        if config_digest(config) != "59f938bbff48f300":
            raise ValueError("confirmation seeds require the locked configuration")
        verify_forecast_provenance(config, Path(__file__).resolve().parents[2])
    return [int(seed) for seed in seeds], [float(rate) for rate in rates]


def _validate_trajectory_rows(config, rows, kind, *, expected_trajectory=None):
    try:
        _validate_grid(config, rows if kind == "training" else [], rows if kind == "probe" else [],
                       require_complete=False)
    except (KeyError, TypeError, OverflowError) as error:
        raise ValueError(f"invalid {kind} measurements: {error}") from error
    steps = set(config["train"]["checkpoint_steps"])
    groups = {(row["seed"], row["learning_rate"]) for row in rows}
    if expected_trajectory is not None and groups != {expected_trajectory}:
        raise ValueError("producer did not return the complete requested trajectory")
    for seed, rate in groups:
        selected = [row for row in rows if (row["seed"], row["learning_rate"]) == (seed, rate)]
        if len(selected) != len(steps) * (6 if kind == "probe" else 1):
            raise ValueError(f"partial {kind} trajectory; refusing to replace evidence")


def preflight_rate_aware_outputs(config, checkpoint_root, *, training_path=None, probe_path=None,
                                 audit_path=None, summary_path=None) -> dict:
    """Validate every existing trajectory and checkpoint before any producer runs.

    Missing files are permitted. Existing result trajectories must be complete;
    corrupt, mismatched, nonfinite and partial evidence is never overwritten.
    """
    _storage_selection(config, config["rate_aware_clock"]["seeds"], config["rate_aware_clock"]["learning_rates"])
    result = {name: read_rate_aware_jsonl(path) if path is not None and Path(path).exists() else []
              for name, path in (("training", training_path), ("probe", probe_path),
                                 ("audit", audit_path), ("summary", summary_path))}
    log = Path(checkpoint_root) / "rate_aware_clock_attempts.jsonl"
    attempts = read_rate_aware_jsonl(log) if log.exists() else []
    for event in attempts:
        if (event.get("record_type") != "rate_aware_clock_attempt"
                or event.get("base_config_sha256") != config_digest(config)
                or event.get("kind") not in ("training", "probe")
                or event.get("status") not in ("started", "failed", "completed")
                or event.get("seed") not in config["rate_aware_clock"]["seeds"]
                or event.get("learning_rate") not in config["rate_aware_clock"]["learning_rates"]):
            raise ValueError("incompatible attempt log")
        if event.get("scientific_failure"):
            raise ValueError("existing scientific failure must not be retried")
    try:
        for kind in ("training", "probe"):
            _validate_trajectory_rows(config, result[kind], kind)
        identities = {}
        for seed in config["rate_aware_clock"]["seeds"]:
            for rate in config["rate_aware_clock"]["learning_rates"]:
                for step in config["train"]["checkpoint_steps"]:
                    path = rate_aware_checkpoint_path(checkpoint_root, seed, rate, step)
                    if path.exists():
                        identities[seed, rate, step] = _checked_checkpoint(config, checkpoint_root, seed, rate, step)
        for row in result["training"] + result["probe"]:
            cell = row["seed"], row["learning_rate"], row["step"]
            if row["checkpoint_path"] != str(rate_aware_checkpoint_path(checkpoint_root, *cell)):
                raise ValueError("raw checkpoint root mismatch")
            if cell not in identities:
                raise ValueError("raw evidence references missing checkpoint")
            if "parameter_sha256" in row and row["parameter_sha256"] != identities[cell]["parameter_sha256"]:
                raise ValueError("checkpoint parameter digest differs from raw evidence")
        for seed in config["rate_aware_clock"]["seeds"]:
            paired = {value["parameter_sha256"] for (s, _, step), value in identities.items() if s == seed and step == 0}
            if len(paired) > 1:
                raise ValueError("initialization pairing mismatch")
        if result["audit"]:
            _validate_audit(config, result["audit"])
        if result["summary"] and (len(result["summary"]) != 1
                or result["summary"][0].get("record_type") != "rate_aware_clock_summary"
                or result["summary"][0].get("base_config_sha256") != config_digest(config)):
            raise ValueError("summary identity mismatch")
        if result["summary"] and result["summary"][0].get("validity_failures"):
            raise ValueError("existing summary records scientifically invalid evidence")
        if config_digest(config) == "59f938bbff48f300" and result["audit"]:
            try:
                _validate_grid(config, result["training"], result["probe"])
            except ValueError:
                pass  # A set of completed trajectories may be resumed.
            else:
                summary = analyze_rate_aware_clock(config, result["training"], result["probe"], result["audit"])
                if summary["validity_failures"]:
                    raise ValueError("complete evidence fails scientific validity: " + ", ".join(summary["validity_failures"]))
    except (TypeError, KeyError, OverflowError) as error:
        raise ValueError(f"incompatible rate-aware evidence: {error}") from error
    return result


def _attempt_event(config, root, seed, rate, kind, status, **details):
    path = root / "rate_aware_clock_attempts.jsonl"
    rows = read_rate_aware_jsonl(path) if path.exists() else []
    rows.append({"record_type": "rate_aware_clock_attempt", "base_config_sha256": config_digest(config),
                 "seed": seed, "learning_rate": rate, "kind": kind, "status": status,
                 "timestamp": datetime.now(timezone.utc).isoformat(), **details})
    atomic_write_rate_aware_jsonl(path, rows)


def _produce_trajectory(config, root, seed, rate, kind):
    selected = _rate_config(config, rate)
    steps = config["train"]["checkpoint_steps"]
    records = []
    if kind == "training":
        # Failed staging directories are retained for diagnosis. Only a fully
        # validated run is promoted to canonical checkpoint paths.
        staging = Path(tempfile.mkdtemp(prefix=".rate-aware-", dir=root))
        run = train_diagnostic(selected, seed, "fresh", staging / f"lr_{_rate_label(rate)}")
        for row in run:
            row = _enrich_record(row, config_digest(config), selected, rate)
            row.update(_checked_checkpoint(config, staging, seed, rate, row["step"]))
            row.update(record_type="rate_aware_clock_training", checkpoint_path=str(
                rate_aware_checkpoint_path(root, seed, rate, row["step"])))
            records.append(row)
        _validate_trajectory_rows(config, records, kind, expected_trajectory=(seed, rate))
        initial = next(row["parameter_sha256"] for row in records if row["step"] == 0)
        for other_rate in config["rate_aware_clock"]["learning_rates"]:
            if rate_aware_checkpoint_path(root, seed, other_rate, 0).exists():
                if _checked_checkpoint(config, root, seed, other_rate, 0)["parameter_sha256"] != initial:
                    raise ValueError("initialization pairing mismatch")
        for step in steps:
            destination = rate_aware_checkpoint_path(root, seed, rate, step)
            destination.parent.mkdir(parents=True, exist_ok=True)
            rate_aware_checkpoint_path(staging, seed, rate, step).replace(destination)
        shutil.rmtree(staging)
    else:
        for step in steps:
            checkpoint = rate_aware_checkpoint_path(root, seed, rate, step)
            identity = _checked_checkpoint(config, root, seed, rate, step)
            for row in evaluate_checkpoint_geometry(selected, checkpoint, seed, condition="fresh"):
                row = _enrich_record(row, config_digest(config), selected, rate)
                row.update(identity)
                row.update(record_type="rate_aware_clock_probe", checkpoint_path=str(checkpoint))
                records.append(row)
        _validate_trajectory_rows(config, records, kind, expected_trajectory=(seed, rate))
    return records


def _run_rate_aware(config, seeds, learning_rates, checkpoint_root, results_path, kind):
    seeds, rates = _storage_selection(config, seeds, learning_rates)
    root = Path(checkpoint_root)
    existing = preflight_rate_aware_outputs(config, root, **{f"{kind}_path": results_path})[kind]
    steps = config["train"]["checkpoint_steps"]
    for seed in seeds:
        for rate in rates:
            paths = [rate_aware_checkpoint_path(root, seed, rate, step) for step in steps]
            cached = [row for row in existing if (row["seed"], row["learning_rate"]) == (seed, rate)]
            if kind == "probe" and not all(path.exists() for path in paths):
                raise ValueError("probe requires every selected checkpoint before computation")
            if kind == "training" and not cached and any(path.exists() for path in paths):
                raise ValueError("unrecorded checkpoints require explicit recovery; refusing to overwrite evidence")
    for seed in seeds:
        for rate in rates:
            if any((row["seed"], row["learning_rate"]) == (seed, rate) for row in existing):
                continue
            _attempt_event(config, root, seed, rate, kind, "started")
            try:
                records = _produce_trajectory(config, root, seed, rate, kind)
                existing.extend(records)
                if results_path is not None:
                    atomic_write_rate_aware_jsonl(results_path, existing)
            except BaseException as error:
                _attempt_event(config, root, seed, rate, kind, "failed", error=f"{type(error).__name__}: {error}",
                               scientific_failure=isinstance(error, (ValueError, FloatingPointError)))
                raise
            _attempt_event(config, root, seed, rate, kind, "completed")
    return [row for row in existing if row["seed"] in seeds and row["learning_rate"] in rates]


def run_rate_aware_training(config, seeds, learning_rates,
                            checkpoint_root="checkpoints/mess3_rate_aware_clock", *, results_path=None) -> list[dict]:
    """Train paired fresh-data trajectories; persist/reuse each completed run."""
    return _run_rate_aware(config, seeds, learning_rates, checkpoint_root, results_path, "training")


def run_rate_aware_probes(config, seeds, learning_rates,
                         checkpoint_root="checkpoints/mess3_rate_aware_clock", *, results_path=None) -> list[dict]:
    """Probe exact checkpoints with the inherited measurement definitions."""
    return _run_rate_aware(config, seeds, learning_rates, checkpoint_root, results_path, "probe")


def _token_array_hashes(tokens):
    """Canonical signed little-endian int64 rows; dataset hash includes its shape.

    Row hashes identify token sequences, independently of sequence IDs and array
    dtype/layout. The dataset hash additionally preserves row order and count.
    """
    values = np.asarray(tokens)
    if values.ndim != 2 or values.dtype.kind not in "iu":
        raise ValueError("token arrays must be two-dimensional integer arrays")
    values = np.asarray(values, dtype="<i8", order="C")
    row_hashes = {hashlib.sha256(row.tobytes(order="C")).hexdigest() for row in values}
    digest = hashlib.sha256(np.asarray(values.shape, dtype="<u8").tobytes() + values.tobytes(order="C"))
    return {"sha256": digest.hexdigest(), "n_rows": len(values), "n_unique_rows": len(row_hashes)}, row_hashes


def audit_token_isolation(config, seeds) -> list[dict]:
    """Hash the actual deterministic evaluation/probe token arrays for each seed.

    Reports overlaps without hiding them or resampling; the validity checker
    decides whether the resulting evidence permits a confirmatory verdict.
    """
    seeds, _ = _storage_selection(config, seeds, config["rate_aware_clock"]["learning_rates"])
    mixture = mixture_from_config(config)
    rows = []
    for seed in seeds:
        datasets, hashes = {}, {}
        for name, count, offset in (("evaluation", config["data"]["test_sequences"], 202),
                                    ("probe_fit", config["probe"]["train_sequences"], 404),
                                    ("probe_test", config["probe"]["test_sequences"], 505)):
            batch = sample_from_config(mixture, config, count, config["data"]["sequence_length"], seed + offset)
            datasets[name], hashes[name] = _token_array_hashes(batch.tokens)
        pairs = (("evaluation", "probe_fit"), ("evaluation", "probe_test"), ("probe_fit", "probe_test"))
        rows.append({"record_type": "rate_aware_clock_audit", "seed": seed,
                     "base_config_sha256": config_digest(config), "datasets": datasets,
                     "intersections": {f"{a}__{b}": len(hashes[a] & hashes[b]) for a, b in pairs}})
    return rows
