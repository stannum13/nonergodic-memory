#!/usr/bin/env python3
"""Run the preregistered persistent source-belief intervention pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from nonergodic_memory.experiment import (
    config_digest,
    load_checkpoint,
    load_config,
    mixture_from_config,
    runtime_provenance,
    set_seed,
)
from nonergodic_memory.predictive_memory import (
    CalibrationResult,
    analyze_evidence,
    calibrate_memory,
    classify_pilot,
    evaluate_responses,
    passes_feasibility_gates,
    sample_prefix_splits,
    validate_complete_evidence,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "results" / "predictive_memory.jsonl"
SUMMARY_PATH = ROOT / "results" / "predictive_memory_summary.jsonl"
REGISTERED_CONFIG_DIGEST = "7b076ce5a049c176"


def preflight_outputs(paths) -> None:
    """Protect result-bearing artifacts from silent overwrite after inspection."""
    existing = [Path(path) for path in paths if Path(path).exists() or Path(path).is_symlink()]
    if existing:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"refusing to overwrite predictive-memory evidence: {names}")


def validate_registered_config(config: dict) -> None:
    actual = config_digest(config)
    if actual != REGISTERED_CONFIG_DIGEST:
        raise ValueError(
            f"predictive-memory config does not match registered digest: {actual}"
        )


def atomic_write_jsonl(path: Path, records: list[dict]) -> None:
    """Publish a complete JSONL snapshot without truncating prior evidence."""
    encoded = "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            Path(temporary_name).unlink()
        except FileNotFoundError:
            pass
        raise


def reserve_run(path: Path, digest: str) -> None:
    """Atomically reserve the one registered attempt."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "record_type": "attempt",
                    "event": "started",
                    "experiment_config_sha256": digest,
                },
                sort_keys=True,
            )
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def append_attempt(path: Path, record: dict) -> None:
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_hash(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _checkpoint_path(seed: int, step: int, cohort: str) -> Path:
    if cohort == "development":
        return ROOT / "checkpoints" / "mess3_diagnosis" / f"transformer_seed{seed}_fresh_step{step}.pt"
    if cohort == "heldout":
        return (
            ROOT
            / "checkpoints"
            / "mess3_threshold"
            / "lr_0p003"
            / f"transformer_seed{seed}_fresh_step{step}.pt"
        )
    raise ValueError(f"unknown cohort: {cohort}")


def _load_registered_checkpoint(seed: int, step: int, cohort: str):
    path = _checkpoint_path(seed, step, cohort)
    if not path.is_file():
        raise FileNotFoundError(path)
    parent_config_path = (
        ROOT / "configs" / "mess3_diagnosis.yaml"
        if cohort == "development"
        else ROOT / "configs" / "mess3_threshold.yaml"
    )
    expected_config = load_config(parent_config_path)
    payload, model = load_checkpoint(path, expected_config, "transformer", seed)
    if payload.get("model_name") != "transformer":
        raise ValueError(f"not a transformer checkpoint: {path}")
    if int(payload.get("seed", -1)) != seed or int(payload.get("step", -1)) != step:
        raise ValueError(f"checkpoint identity mismatch: {path}")
    if payload.get("condition") != "fresh":
        raise ValueError(f"checkpoint is not fresh-data condition: {path}")
    if not np.isclose(float(payload["config"]["train"]["learning_rate"]), 0.003):
        raise ValueError(f"checkpoint learning rate mismatch: {path}")
    if int(payload["config"]["model"]["layers"]) != 2:
        raise ValueError(f"checkpoint depth mismatch: {path}")
    if payload["config"]["data"].get("generator") != "mess3":
        raise ValueError(f"checkpoint generator mismatch: {path}")
    expected_model = {"width": 32, "layers": 2, "heads": 4, "max_length": 128}
    if payload["config"]["model"] != expected_model:
        raise ValueError(f"checkpoint architecture mismatch: {path}")
    if any(not torch.isfinite(value).all() for value in payload["state_dict"].values()):
        raise ValueError(f"checkpoint contains nonfinite tensors: {path}")
    return path, payload, model


def _split_record(seed: int, cohort: str, splits: dict[str, np.ndarray]) -> dict:
    hashes = {name: _array_hash(tokens) for name, tokens in splits.items()}
    sets = {name: {tuple(row) for row in values} for name, values in splits.items()}
    overlap = 0
    names = list(sets)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            overlap += len(sets[left].intersection(sets[right]))
    if overlap:
        raise ValueError(f"registered prefix splits overlap for seed {seed}: {overlap}")
    return {
        "record_type": "split_audit",
        "seed": seed,
        "cohort": cohort,
        "split_hashes": hashes,
        "split_sizes": {name: len(values) for name, values in splits.items()},
        "overlap_count": overlap,
    }


def _calibrate(
    experiment: dict,
    seed: int,
    step: int,
    cohort: str,
    splits: dict[str, np.ndarray],
) -> tuple[CalibrationResult, dict, object]:
    path, payload, model = _load_registered_checkpoint(seed, step, cohort)
    mixture = mixture_from_config(payload["config"])
    calibration = calibrate_memory(
        model,
        mixture,
        splits,
        alpha=float(experiment["experiment"]["ridge_alpha"]),
        rcond=float(experiment["experiment"]["pseudoinverse_rcond"]),
        shuffle_seed=seed * 100 + step + 17,
    )
    checks = passes_feasibility_gates(calibration.metrics, experiment["gates"])
    row = {
        "record_type": "calibration",
        "seed": seed,
        "step": step,
        "cohort": cohort,
        "checkpoint": str(path.relative_to(ROOT)),
        "checkpoint_sha256": _sha256(path),
        "passed": bool(all(checks.values())),
        "checks": checks,
        **calibration.metrics,
    }
    return calibration, row, (mixture, model)


def _evaluate(
    experiment: dict,
    seed: int,
    step: int,
    cohort: str,
    splits: dict[str, np.ndarray],
    calibration: CalibrationResult,
    mixture,
    model,
) -> list[dict]:
    rows = evaluate_responses(
        model,
        mixture,
        splits["evaluation"],
        calibration,
        doses=tuple(
            float(dose) for dose in experiment["experiment"]["doses"] if float(dose) != 0
        ),
        random_controls=int(experiment["experiment"]["random_controls"]),
        random_seed=seed * 10_000 + step + 31,
    )
    for row in rows:
        row.update({"seed": seed, "step": step, "cohort": cohort})
    return rows


def run_registered(
    config_path: Path,
    results_path: Path,
    summary_path: Path,
    reservation_state: dict[str, bool] | None = None,
) -> dict:
    experiment = load_config(config_path)
    validate_registered_config(experiment)
    attempt_path = results_path.with_name(results_path.stem + "_attempt.jsonl")
    command_path = results_path.with_name(results_path.stem + "_command.jsonl")
    preflight_outputs((results_path, summary_path, attempt_path, command_path))
    reserve_run(attempt_path, REGISTERED_CONFIG_DIGEST)
    if reservation_state is not None:
        reservation_state["owned"] = True
    set_seed(20260928)
    sizes = {name: int(value) for name, value in experiment["data"].items()}
    prefix_length = int(experiment["experiment"]["prefix_length"])
    primary_step = int(experiment["models"]["primary_checkpoint"])
    records: list[dict] = []
    start = time.monotonic()
    calibrations: dict[tuple[int, int], tuple[CalibrationResult, object, object]] = {}
    split_cache: dict[int, dict[str, np.ndarray]] = {}

    def persist() -> None:
        for record in records:
            record.setdefault("experiment_config_sha256", REGISTERED_CONFIG_DIGEST)
        atomic_write_jsonl(results_path, records)

    def finish(summary: dict) -> dict:
        analysis = analyze_evidence(
            records, summary, experiment, REGISTERED_CONFIG_DIGEST
        )
        if not analysis["valid"]:
            raise ValueError(
                "predictive-memory evidence failed terminal analysis: "
                + "; ".join(analysis["validity_errors"])
            )
        atomic_write_jsonl(summary_path, [summary])
        append_attempt(attempt_path, {"event": "finished", "status": summary["status"]})
        return summary

    development = tuple(int(seed) for seed in experiment["models"]["development_seeds"])
    heldout = tuple(int(seed) for seed in experiment["models"]["heldout_seeds"])
    steps = tuple(int(step) for step in experiment["models"]["checkpoints"])

    for seed in development:
        path, payload, _ = _load_registered_checkpoint(seed, primary_step, "development")
        mixture = mixture_from_config(payload["config"])
        splits = sample_prefix_splits(
            mixture, model_seed=seed, sizes=sizes, length=prefix_length
        )
        split_cache[seed] = splits
        records.append(_split_record(seed, "development", splits))
        calibration, row, loaded = _calibrate(
            experiment, seed, primary_step, "development", splits
        )
        row["checkpoint_preflight_sha256"] = _sha256(path)
        records.append(row)
        calibrations[(seed, primary_step)] = (calibration, *loaded)
        persist()

    development_passed = all(
        next(
            row["passed"]
            for row in records
            if row["record_type"] == "calibration"
            and row["seed"] == seed
            and row["step"] == primary_step
        )
        for seed in development
    )
    if not development_passed:
        summary = {
            "record_type": "summary",
            "status": "actuator_infeasible",
            "stage": "development",
            "config_sha256": config_digest(experiment),
            "result_rows": len(records),
            "elapsed_seconds": time.monotonic() - start,
            **runtime_provenance(),
        }
        return finish(summary)

    # Development behavior is descriptive and cannot change the frozen held-out design.
    for seed in development:
        for step in steps:
            if (seed, step) not in calibrations:
                calibration, row, loaded = _calibrate(
                    experiment, seed, step, "development", split_cache[seed]
                )
                records.append(row)
                calibrations[(seed, step)] = (calibration, *loaded)
            calibration, mixture, model = calibrations[(seed, step)]
            records.extend(
                _evaluate(
                    experiment,
                    seed,
                    step,
                    "development",
                    split_cache[seed],
                    calibration,
                    mixture,
                    model,
                )
            )
            persist()

    # All held-out trained calibrations are completed before any held-out evaluation set is opened.
    heldout_calibrations: dict[int, tuple[CalibrationResult, object, object]] = {}
    for seed in heldout:
        path, payload, _ = _load_registered_checkpoint(seed, primary_step, "heldout")
        mixture = mixture_from_config(payload["config"])
        splits = sample_prefix_splits(
            mixture, model_seed=seed, sizes=sizes, length=prefix_length
        )
        split_cache[seed] = splits
        records.append(_split_record(seed, "heldout", splits))
        calibration, row, loaded = _calibrate(
            experiment, seed, primary_step, "heldout", splits
        )
        row["checkpoint_preflight_sha256"] = _sha256(path)
        records.append(row)
        heldout_calibrations[seed] = (calibration, *loaded)
        persist()

    heldout_valid = all(
        next(
            row["passed"]
            for row in records
            if row["record_type"] == "calibration"
            and row["cohort"] == "heldout"
            and row["seed"] == seed
            and row["step"] == primary_step
        )
        for seed in heldout
    )
    if not heldout_valid:
        summary = {
            "record_type": "summary",
            "status": "invalid_pilot",
            "stage": "heldout_calibration",
            "config_sha256": config_digest(experiment),
            "result_rows": len(records),
            "elapsed_seconds": time.monotonic() - start,
            **runtime_provenance(),
        }
        return finish(summary)

    for seed in heldout:
        calibration, mixture, model = heldout_calibrations[seed]
        records.extend(
            _evaluate(
                experiment,
                seed,
                primary_step,
                "heldout",
                split_cache[seed],
                calibration,
                mixture,
                model,
            )
        )
        # The untrained checkpoint is a negative control, not a validity gate.
        zero_calibration, zero_row, loaded = _calibrate(
            experiment, seed, 0, "heldout", split_cache[seed]
        )
        records.append(zero_row)
        zero_mixture, zero_model = loaded
        records.extend(
            _evaluate(
                experiment,
                seed,
                0,
                "heldout",
                split_cache[seed],
                zero_calibration,
                zero_mixture,
                zero_model,
            )
        )
        persist()

    response_rows = [row for row in records if row["record_type"] == "response"]
    doses = tuple(
        float(dose) for dose in experiment["experiment"]["doses"] if float(dose) != 0
    )
    evidence_errors = validate_complete_evidence(
        records,
        development_seeds=development,
        heldout_seeds=heldout,
        steps=steps,
        doses=doses,
        random_controls=int(experiment["experiment"]["random_controls"]),
        examples_per_cell=int(experiment["data"]["evaluation"]),
        split_sizes=sizes,
        gates=experiment["gates"],
        experiment_digest=REGISTERED_CONFIG_DIGEST,
    )
    if evidence_errors:
        raise ValueError("complete evidence invalid: " + "; ".join(evidence_errors))
    decision = classify_pilot(
        response_rows,
        heldout_seeds=heldout,
        primary_step=primary_step,
        doses=doses,
        random_controls=int(experiment["experiment"]["random_controls"]),
        examples_per_cell=int(experiment["data"]["evaluation"]),
        mean_score_min=float(experiment["decision"]["mean_score_min"]),
        control_margin_min=float(experiment["decision"]["control_margin_min"]),
    )
    summary = {
        "record_type": "summary",
        **decision,
        "stage": "complete",
        "config_sha256": config_digest(experiment),
        "result_rows": len(records),
        "elapsed_seconds": time.monotonic() - start,
        **runtime_provenance(),
    }
    return finish(summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/predictive_memory.yaml")
    parser.add_argument("--results", type=Path, default=RESULTS_PATH)
    parser.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    args = parser.parse_args()
    reservation_state = {"owned": False}
    try:
        summary = run_registered(
            args.config, args.results, args.summary, reservation_state=reservation_state
        )
    except Exception as error:
        attempt_path = args.results.with_name(args.results.stem + "_attempt.jsonl")
        if reservation_state["owned"] and attempt_path.exists() and not args.summary.exists():
            failure = {
                "record_type": "summary",
                "status": "inconclusive_execution_error",
                "stage": "execution",
                "error_type": type(error).__name__,
                "experiment_config_sha256": REGISTERED_CONFIG_DIGEST,
            }
            atomic_write_jsonl(args.summary, [failure])
            append_attempt(
                attempt_path, {"event": "finished", "status": failure["status"]}
            )
        raise
    print(summary)


if __name__ == "__main__":
    main()
