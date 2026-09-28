"""Exact targets and calibrated interventions for the predictive-memory pilot."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from torch import Tensor

from .data.hmm import HMMMixture
from .models.sequence import TransformerPredictor


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class LinearActuator:
    scaler: StandardScaler
    decoder: Ridge
    pseudoinverse: FloatArray
    retained_rank: int


@dataclass(frozen=True)
class CalibrationResult:
    actuator: LinearActuator
    evaluator: LinearActuator
    shuffled: LinearActuator
    metrics: dict[str, float]


def passes_feasibility_gates(
    metrics: dict[str, float], gates: dict[str, float]
) -> dict[str, bool]:
    """Evaluate the locked actuator and numerical-validity thresholds."""
    return {
        "component_r2": metrics["component_r2"] >= gates["component_r2_min"],
        "joint_r2": metrics["joint_r2"] >= gates["joint_r2_min"],
        "actuator_rank": metrics["actuator_rank"] >= gates["actuator_rank_min"],
        "actuator_constraint": metrics["actuator_constraint_relative_error"]
        <= gates["actuator_constraint_relative_error_max"],
        "displacement_relative_error": metrics["displacement_relative_error"]
        <= gates["displacement_relative_error_max"],
        "oracle_denominator": metrics["oracle_denominator"]
        >= gates["oracle_denominator_min"],
        "edit_within_natural_scale": metrics["edit_rms_p95"]
        <= metrics["natural_rms_p95"],
        "identity": metrics["identity_max_abs_error"] <= gates["identity_atol"],
        "exact_operator": metrics["exact_max_abs_error"] <= gates["exact_atol"],
        "finite": bool(
            np.all(
                np.isfinite(
                    [
                        metrics[key]
                        for key in (
                            "component_r2",
                            "joint_r2",
                            "actuator_rank",
                            "actuator_constraint_relative_error",
                            "displacement_relative_error",
                            "oracle_denominator",
                            "edit_rms_p95",
                            "natural_rms_p95",
                            "identity_max_abs_error",
                            "exact_max_abs_error",
                        )
                    ]
                )
            )
        ),
    }


def classify_pilot(
    rows: list[dict],
    *,
    heldout_seeds: tuple[int, ...],
    primary_step: int,
    doses: tuple[float, ...],
    random_controls: int,
    examples_per_cell: int,
    mean_score_min: float,
    control_margin_min: float,
) -> dict:
    """Apply the locked equal-model-seed pilot decision rule."""
    relevant = [
        row
        for row in rows
        if row.get("seed") in heldout_seeds and row.get("step") == primary_step
    ]
    expected_keys = set()
    for seed in heldout_seeds:
        for dose in doses:
            expected_keys.add((seed, dose, "learned", None))
            expected_keys.add((seed, dose, "shuffled", None))
            for random_index in range(random_controls):
                expected_keys.add((seed, dose, "random", random_index))
    actual_keys = [
        (row.get("seed"), row.get("dose"), row.get("control"), row.get("random_index"))
        for row in relevant
    ]
    errors: list[str] = []
    if set(actual_keys) != expected_keys or len(actual_keys) != len(expected_keys):
        errors.append("response grid is missing, duplicated, or contains unexpected cells")
    numeric_fields = (
        "score",
        "numerator",
        "denominator",
        "standardized_edit_rms_mean",
        "first_token_response_mse",
    )
    for index, row in enumerate(relevant):
        try:
            values = [float(row[field]) for field in numeric_fields]
            numerator_values = np.asarray(row["example_numerator"], dtype=np.float64)
            denominator_values = np.asarray(row["example_denominator"], dtype=np.float64)
        except (KeyError, TypeError, ValueError):
            errors.append(f"response row {index} lacks required numeric evidence")
            continue
        if not np.all(np.isfinite(values)) or not np.all(np.isfinite(numerator_values)) or not np.all(np.isfinite(denominator_values)):
            errors.append(f"response row {index} contains nonfinite evidence")
        if len(numerator_values) != examples_per_cell or len(denominator_values) != examples_per_cell:
            errors.append(f"response row {index} has wrong contribution count")
        if values[2] <= 0 or np.any(denominator_values <= 0):
            errors.append(f"response row {index} has nonpositive denominator")
        if not np.isclose(numerator_values.sum(), values[1], rtol=1e-10, atol=1e-12):
            errors.append(f"response row {index} numerator total disagrees")
        if not np.isclose(denominator_values.sum(), values[2], rtol=1e-10, atol=1e-12):
            errors.append(f"response row {index} denominator total disagrees")
        if values[2] > 0 and not np.isclose(1.0 - values[1] / values[2], values[0], rtol=1e-10, atol=1e-12):
            errors.append(f"response row {index} score disagrees with totals")
    if errors:
        return {"status": "invalid_pilot", "validity_errors": sorted(set(errors))}

    def pooled(selected: list[dict]) -> float:
        numerator = sum(float(row["numerator"]) for row in selected)
        denominator = sum(float(row["denominator"]) for row in selected)
        return float(1.0 - numerator / denominator)

    seed_scores: dict[int, dict[str, float]] = {}
    for seed in heldout_seeds:
        seed_rows = [
            row for row in rows if row["seed"] == seed and row["step"] == primary_step
        ]
        controls: dict[str, float] = {}
        for control in ("learned", "shuffled"):
            selected = [row for row in seed_rows if row["control"] == control]
            controls[control] = pooled(selected)
        controls["random"] = float(
            np.mean(
                [
                    pooled(
                        [
                            row
                            for row in seed_rows
                            if row["control"] == "random"
                            and row["random_index"] == random_index
                        ]
                    )
                    for random_index in range(random_controls)
                ]
            )
        )
        seed_scores[seed] = controls
    learned = np.array([seed_scores[seed]["learned"] for seed in heldout_seeds])
    shuffled = np.array([seed_scores[seed]["shuffled"] for seed in heldout_seeds])
    random = np.array([seed_scores[seed]["random"] for seed in heldout_seeds])
    learned_mean = float(learned.mean())
    shuffled_margin = float(learned_mean - shuffled.mean())
    random_margin = float(learned_mean - random.mean())
    promising = bool(
        np.all(learned > 0)
        and learned_mean >= mean_score_min
        and shuffled_margin >= control_margin_min
        and random_margin >= control_margin_min
    )
    return {
        "status": "promising_pilot" if promising else "criterion_not_met",
        "seed_scores": {str(seed): seed_scores[seed] for seed in heldout_seeds},
        "mean_learned_score": learned_mean,
        "mean_shuffled_score": float(shuffled.mean()),
        "mean_random_score": float(random.mean()),
        "shuffled_margin": shuffled_margin,
        "random_margin": random_margin,
        "all_heldout_positive": bool(np.all(learned > 0)),
    }


def validate_complete_evidence(
    records: list[dict],
    *,
    development_seeds: tuple[int, ...],
    heldout_seeds: tuple[int, ...],
    steps: tuple[int, ...],
    doses: tuple[float, ...],
    random_controls: int,
    examples_per_cell: int,
    split_sizes: dict[str, int],
    gates: dict[str, float],
    experiment_digest: str,
    require_responses: bool = True,
) -> list[str]:
    """Validate the exact registered raw grid before a scientific verdict."""
    errors: list[str] = []
    if any(
        row.get("record_type") not in {"split_audit", "calibration", "response"}
        for row in records
    ):
        errors.append("raw evidence contains an unknown record type")
    if any(
        row.get("experiment_config_sha256") != experiment_digest for row in records
    ):
        errors.append("raw evidence has a missing or wrong experiment digest")
    cohorts = {"development": development_seeds, "heldout": heldout_seeds}
    split_rows = [row for row in records if row.get("record_type") == "split_audit"]
    expected_splits = {
        (cohort, seed) for cohort, seeds in cohorts.items() for seed in seeds
    }
    actual_splits = [(row.get("cohort"), row.get("seed")) for row in split_rows]
    if set(actual_splits) != expected_splits or len(actual_splits) != len(expected_splits):
        errors.append("split audit grid is incomplete or duplicated")
    if any(row.get("overlap_count") != 0 for row in split_rows):
        errors.append("split audit contains overlapping prefixes")
    for index, row in enumerate(split_rows):
        if row.get("split_sizes") != split_sizes:
            errors.append(f"split row {index} has wrong registered sizes")
        hashes = row.get("split_hashes")
        if not isinstance(hashes, dict) or set(hashes) != set(split_sizes) or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in (hashes or {}).values()
        ):
            errors.append(f"split row {index} has invalid hashes")

    calibration_rows = [
        row for row in records if row.get("record_type") == "calibration"
    ]
    expected_calibrations = {
        (cohort, seed, step)
        for cohort, seeds in cohorts.items()
        for seed in seeds
        for step in steps
    }
    actual_calibrations = [
        (row.get("cohort"), row.get("seed"), row.get("step"))
        for row in calibration_rows
    ]
    if set(actual_calibrations) != expected_calibrations or len(
        actual_calibrations
    ) != len(expected_calibrations):
        errors.append("calibration grid is incomplete or duplicated")
    calibration_metrics = {
        "component_r2",
        "joint_r2",
        "actuator_rank",
        "actuator_constraint_relative_error",
        "displacement_relative_error",
        "oracle_denominator",
        "edit_rms_p95",
        "natural_rms_p95",
        "identity_max_abs_error",
        "exact_max_abs_error",
    }
    for index, row in enumerate(calibration_rows):
        if not calibration_metrics.issubset(row):
            errors.append(f"calibration row {index} lacks required metrics")
            continue
        if not np.all(np.isfinite([row[key] for key in calibration_metrics])):
            errors.append(f"calibration row {index} contains nonfinite metrics")
        checks = passes_feasibility_gates(row, gates)
        if row.get("checks") != checks or row.get("passed") != all(checks.values()):
            errors.append(f"calibration row {index} has forged gate status")
        checkpoint_hash = row.get("checkpoint_sha256")
        if (
            not isinstance(row.get("checkpoint"), str)
            or not isinstance(checkpoint_hash, str)
            or len(checkpoint_hash) != 64
        ):
            errors.append(f"calibration row {index} has invalid checkpoint provenance")

    response_rows = [row for row in records if row.get("record_type") == "response"]
    expected_responses = set()
    if require_responses:
        for cohort, seeds in cohorts.items():
            for seed in seeds:
                for step in steps:
                    for dose in doses:
                        expected_responses.add((cohort, seed, step, dose, "learned", None))
                        expected_responses.add((cohort, seed, step, dose, "shuffled", None))
                        for random_index in range(random_controls):
                            expected_responses.add(
                                (cohort, seed, step, dose, "random", random_index)
                            )
    actual_responses = [
        (
            row.get("cohort"),
            row.get("seed"),
            row.get("step"),
            row.get("dose"),
            row.get("control"),
            row.get("random_index"),
        )
        for row in response_rows
    ]
    if set(actual_responses) != expected_responses or len(actual_responses) != len(
        expected_responses
    ):
        errors.append("response grid is incomplete or duplicated")
    for index, row in enumerate(response_rows):
        try:
            numerator = float(row["numerator"])
            denominator = float(row["denominator"])
            score = float(row["score"])
            per_numerator = np.asarray(row["example_numerator"], dtype=np.float64)
            per_denominator = np.asarray(row["example_denominator"], dtype=np.float64)
            diagnostics = np.asarray(
                [row["standardized_edit_rms_mean"], row["first_token_response_mse"]],
                dtype=np.float64,
            )
        except (KeyError, TypeError, ValueError):
            errors.append(f"response row {index} lacks required evidence")
            continue
        if not np.all(np.isfinite([numerator, denominator, score])) or not np.all(
            np.isfinite(np.concatenate([per_numerator, per_denominator, diagnostics]))
        ):
            errors.append(f"response row {index} contains nonfinite evidence")
        if denominator <= 0 or np.any(per_denominator <= 0):
            errors.append(f"response row {index} has nonpositive denominator")
        if numerator < 0 or np.any(per_numerator < 0):
            errors.append(f"response row {index} has negative squared-error evidence")
        if len(per_numerator) != examples_per_cell or len(per_denominator) != examples_per_cell:
            errors.append(f"response row {index} has wrong contribution count")
        if not np.isclose(per_numerator.sum(), numerator, rtol=1e-10, atol=1e-12):
            errors.append(f"response row {index} numerator total disagrees")
        if not np.isclose(per_denominator.sum(), denominator, rtol=1e-10, atol=1e-12):
            errors.append(f"response row {index} denominator total disagrees")
        if denominator > 0 and not np.isclose(
            1.0 - numerator / denominator, score, rtol=1e-10, atol=1e-12
        ):
            errors.append(f"response row {index} score disagrees")
    return sorted(set(errors))


def analyze_evidence(
    records: list[dict], summary: dict, config: dict, experiment_digest: str
) -> dict:
    """Recompute the allowed stage, validity, and scientific verdict from raw rows."""
    development = tuple(int(seed) for seed in config["models"]["development_seeds"])
    heldout = tuple(int(seed) for seed in config["models"]["heldout_seeds"])
    steps = tuple(int(step) for step in config["models"]["checkpoints"])
    primary_step = int(config["models"]["primary_checkpoint"])
    doses = tuple(float(dose) for dose in config["experiment"]["doses"] if dose)
    common = {
        "doses": doses,
        "random_controls": int(config["experiment"]["random_controls"]),
        "examples_per_cell": int(config["data"]["evaluation"]),
        "split_sizes": {name: int(value) for name, value in config["data"].items()},
        "gates": config["gates"],
        "experiment_digest": experiment_digest,
    }
    status = summary.get("status")
    stage = summary.get("stage")
    errors: list[str] = []
    decision: dict = {}
    if status == "actuator_infeasible" and stage == "development":
        errors.extend(
            validate_complete_evidence(
                records,
                development_seeds=development,
                heldout_seeds=(),
                steps=(primary_step,),
                require_responses=False,
                **common,
            )
        )
        calibration = [row for row in records if row.get("record_type") == "calibration"]
        if calibration and all(row.get("passed") for row in calibration):
            errors.append("actuator-infeasible summary has no failed development gate")
    elif status == "invalid_pilot" and stage == "heldout_calibration":
        development_rows = [row for row in records if row.get("cohort") == "development"]
        heldout_rows = [row for row in records if row.get("cohort") == "heldout"]
        errors.extend(
            validate_complete_evidence(
                development_rows,
                development_seeds=development,
                heldout_seeds=(),
                steps=steps,
                **common,
            )
        )
        errors.extend(
            validate_complete_evidence(
                heldout_rows,
                development_seeds=(),
                heldout_seeds=heldout,
                steps=(primary_step,),
                require_responses=False,
                **common,
            )
        )
        calibrations = [
            row for row in heldout_rows if row.get("record_type") == "calibration"
        ]
        if calibrations and all(row.get("passed") for row in calibrations):
            errors.append("invalid heldout summary has no failed heldout gate")
    elif status in {"criterion_not_met", "promising_pilot"} and stage == "complete":
        errors.extend(
            validate_complete_evidence(
                records,
                development_seeds=development,
                heldout_seeds=heldout,
                steps=steps,
                **common,
            )
        )
        if not errors:
            decision = classify_pilot(
                [row for row in records if row.get("record_type") == "response"],
                heldout_seeds=heldout,
                primary_step=primary_step,
                doses=doses,
                random_controls=common["random_controls"],
                examples_per_cell=common["examples_per_cell"],
                mean_score_min=float(config["decision"]["mean_score_min"]),
                control_margin_min=float(config["decision"]["control_margin_min"]),
            )
            if decision.get("status") != status:
                errors.append("stored status disagrees with recomputed decision")
            for key, value in decision.items():
                if key == "status":
                    continue
                stored = summary.get(key)
                if isinstance(value, float):
                    if stored is None or not np.isclose(stored, value, rtol=1e-12, atol=1e-14):
                        errors.append(f"stored summary field {key} disagrees")
                elif stored != value:
                    errors.append(f"stored summary field {key} disagrees")
    else:
        errors.append("summary stage/status is not an allowed scientific terminal state")
    if summary.get("result_rows") != len(records):
        errors.append("stored result row count disagrees")
    if summary.get("config_sha256") != experiment_digest:
        errors.append("stored summary config digest disagrees")
    return {
        "status": status,
        "stage": stage,
        "valid": not errors,
        "validity_errors": sorted(set(errors)),
        **decision,
    }


def joint_belief(component: ArrayLike, conditional_state: ArrayLike) -> FloatArray:
    """Flatten ``P(component) P(state | component)`` in component-major order."""
    weights = np.asarray(component, dtype=np.float64)
    states = np.asarray(conditional_state, dtype=np.float64)
    if states.shape[0] != len(weights):
        raise ValueError("component and conditional-state dimensions must agree")
    return (weights[:, None] * states).reshape(-1)


def source_odds_tilt(q: ArrayLike, log_odds_delta: float) -> FloatArray:
    """Tilt two-component odds while preserving conditional state beliefs."""
    belief = np.asarray(q, dtype=np.float64)
    if belief.ndim != 1 or len(belief) % 2:
        raise ValueError("q must be a flat two-component belief")
    rows = belief.reshape(2, -1)
    weights = rows.sum(axis=1)
    if np.any(weights <= 0) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("q must give both components positive normalized mass")
    conditional = rows / weights[:, None]
    odds = np.exp(float(log_odds_delta)) * weights[1] / weights[0]
    new_weights = np.array([1.0 / (1.0 + odds), odds / (1.0 + odds)])
    return (new_weights[:, None] * conditional).reshape(-1)


def two_token_probabilities(q: ArrayLike, mixture: HMMMixture) -> FloatArray:
    """Return exact ``P(next=a, following=b | q)`` under a post-emission belief."""
    belief = np.asarray(q, dtype=np.float64)
    expected = sum(component.n_states for component in mixture.components)
    if belief.shape != (expected,):
        raise ValueError("q shape does not match mixture states")
    joint = np.zeros((mixture.vocab_size, mixture.vocab_size), dtype=np.float64)
    offset = 0
    for component in mixture.components:
        component_q = belief[offset : offset + component.n_states]
        for a in range(mixture.vocab_size):
            after_a = (component_q @ component.transition) * component.emission[:, a]
            for b in range(mixture.vocab_size):
                joint[a, b] += float(
                    ((after_a @ component.transition) * component.emission[:, b]).sum()
                )
        offset += component.n_states
    return joint


def enumerate_two_token_probabilities(q: ArrayLike, mixture: HMMMixture) -> FloatArray:
    """Independent latent-path enumeration used to audit the analytic operator."""
    belief = np.asarray(q, dtype=np.float64)
    joint = np.zeros((mixture.vocab_size, mixture.vocab_size), dtype=np.float64)
    offset = 0
    for component in mixture.components:
        for state in range(component.n_states):
            state_mass = belief[offset + state]
            for next_state in range(component.n_states):
                transition_one = component.transition[state, next_state]
                for first_token in range(mixture.vocab_size):
                    first_mass = (
                        state_mass
                        * transition_one
                        * component.emission[next_state, first_token]
                    )
                    for following_state in range(component.n_states):
                        transition_two = component.transition[next_state, following_state]
                        for second_token in range(mixture.vocab_size):
                            joint[first_token, second_token] += (
                                first_mass
                                * transition_two
                                * component.emission[following_state, second_token]
                            )
        offset += component.n_states
    return joint


def exact_conditionals(joint: ArrayLike) -> FloatArray:
    """Convert a two-token joint table into second-token conditionals."""
    probabilities = np.asarray(joint, dtype=np.float64)
    if probabilities.ndim != 2:
        raise ValueError("joint probabilities must be a matrix")
    masses = probabilities.sum(axis=1)
    if np.any(masses <= 0):
        raise ValueError("joint probabilities must have positive row masses")
    return probabilities / masses[:, None]


def response_score(
    exact_baseline_joint: ArrayLike,
    exact_edited_joint: ArrayLike,
    model_baseline_conditional: ArrayLike,
    model_edited_conditional: ArrayLike,
) -> dict[str, float]:
    """Score a model's conditional response against the exact Bayesian response."""
    baseline_joint = np.asarray(exact_baseline_joint, dtype=np.float64)
    edited_joint = np.asarray(exact_edited_joint, dtype=np.float64)
    model_baseline = np.asarray(model_baseline_conditional, dtype=np.float64)
    model_edited = np.asarray(model_edited_conditional, dtype=np.float64)
    if baseline_joint.shape != edited_joint.shape or baseline_joint.ndim != 3:
        raise ValueError("exact joint arrays must have matching [example, token, token] shape")
    if model_baseline.shape != baseline_joint.shape or model_edited.shape != baseline_joint.shape:
        raise ValueError("model conditional arrays must match exact joint arrays")
    weights = baseline_joint.sum(axis=2)
    exact_baseline = baseline_joint / weights[:, :, None]
    edited_weights = edited_joint.sum(axis=2)
    if np.any(weights <= 0) or np.any(edited_weights <= 0):
        raise ValueError("exact joint rows must have positive mass")
    exact_edited = edited_joint / edited_weights[:, :, None]
    target = exact_edited - exact_baseline
    observed = model_edited - model_baseline
    denominator = float(np.sum(weights[:, :, None] * target**2))
    if denominator <= 0:
        raise ValueError("exact response denominator must be positive")
    numerator = float(np.sum(weights[:, :, None] * (observed - target) ** 2))
    return {
        "score": float(1.0 - numerator / denominator),
        "numerator": numerator,
        "denominator": denominator,
    }


def sample_prefix_splits(
    mixture: HMMMixture, *, model_seed: int, sizes: dict[str, int], length: int
) -> dict[str, NDArray[np.int64]]:
    """Sample deterministic, exact-sequence-disjoint prefix sets."""
    used: set[tuple[int, ...]] = set()
    splits: dict[str, NDArray[np.int64]] = {}
    for split_index, (name, requested) in enumerate(sizes.items()):
        rows: list[NDArray[np.int64]] = []
        attempt = 0
        while len(rows) < requested:
            remaining = requested - len(rows)
            seed = model_seed * 1_000_000 + 700_000 + split_index * 10_000 + attempt
            batch = mixture.sample_vectorized(max(remaining * 2, 16), length, seed)
            for row in batch.tokens:
                key = tuple(int(token) for token in row)
                if key not in used:
                    used.add(key)
                    rows.append(row.copy())
                    if len(rows) == requested:
                        break
            attempt += 1
            if attempt > 100:
                raise RuntimeError("could not sample disjoint prefix splits")
        splits[name] = np.stack(rows).astype(np.int64)
    return splits


def belief_matrix(mixture: HMMMixture, tokens: ArrayLike) -> FloatArray:
    """Return component-major joint beliefs after every supplied prefix."""
    observations = np.asarray(tokens, dtype=np.int64)
    filtered = mixture.filter(observations)
    weights = filtered.component_posterior[:, -1]
    states = filtered.state_posterior[:, -1]
    return (weights[:, :, None] * states).reshape(len(observations), -1)


@torch.no_grad()
def block1_features(
    model: TransformerPredictor, tokens: ArrayLike, batch_size: int = 256
) -> FloatArray:
    """Flatten complete block-1 prefix memories into one row per sequence."""
    observations = np.asarray(tokens, dtype=np.int64)
    parts: list[FloatArray] = []
    model.eval()
    for start in range(0, len(observations), batch_size):
        tensor = torch.from_numpy(observations[start : start + batch_size])
        _, activations = model.forward_with_layers(tensor)
        parts.append(
            activations[0].cpu().numpy().reshape(len(tensor), -1).astype(np.float64)
        )
    return np.concatenate(parts)


def _joint_tables(beliefs: FloatArray, mixture: HMMMixture) -> FloatArray:
    return np.stack([two_token_probabilities(q, mixture) for q in beliefs])


def _conditional_signal(baseline: FloatArray, edited: FloatArray) -> float:
    weights = baseline.sum(axis=2)
    baseline_cond = baseline / weights[:, :, None]
    edited_masses = edited.sum(axis=2)
    edited_cond = edited / edited_masses[:, :, None]
    per_example = np.sum(
        weights[:, :, None] * (edited_cond - baseline_cond) ** 2, axis=(1, 2)
    )
    return float(np.mean(per_example))


def calibrate_memory(
    model: TransformerPredictor,
    mixture: HMMMixture,
    splits: dict[str, NDArray[np.int64]],
    *,
    alpha: float,
    rcond: float,
    shuffle_seed: int = 91,
) -> CalibrationResult:
    """Fit disjoint decoders and compute every preregistered feasibility metric."""
    fit_x = block1_features(model, splits["actuator_fit"])
    fit_q = belief_matrix(mixture, splits["actuator_fit"])
    decoder_x = block1_features(model, splits["decoder_fit"])
    decoder_q = belief_matrix(mixture, splits["decoder_fit"])
    calibration_x = block1_features(model, splits["calibration"])
    calibration_q = belief_matrix(mixture, splits["calibration"])

    actuator = fit_actuator(fit_x, fit_q[:, :5], alpha=alpha, rcond=rcond)
    evaluator = fit_actuator(decoder_x, decoder_q[:, :5], alpha=alpha, rcond=rcond)
    permutation = np.random.default_rng(shuffle_seed).permutation(len(fit_q))
    shuffled = fit_actuator(fit_x, fit_q[permutation, :5], alpha=alpha, rcond=rcond)

    predicted = actuator.decoder.predict(actuator.scaler.transform(calibration_x))
    predicted_six = np.column_stack([predicted, 1.0 - predicted.sum(axis=1)])
    component_true = calibration_q[:, 3:].sum(axis=1)
    component_predicted = predicted_six[:, 3:].sum(axis=1)

    relative_errors: list[float] = []
    constraint_errors: list[float] = []
    standardized_rms: list[float] = []
    oracle_signals: list[float] = []
    baseline_joint = _joint_tables(calibration_q, mixture)
    for dose in (-np.log(2.0), np.log(2.0)):
        edited_q = np.stack([source_odds_tilt(q, dose) for q in calibration_q])
        requested = edited_q[:, :5] - calibration_q[:, :5]
        raw_delta = intervention_delta(actuator, requested)
        constraint_errors.append(
            actuator_constraint_diagnostics(actuator, requested)[
                "actuator_constraint_relative_error"
            ]
        )
        evaluator_delta = (raw_delta / evaluator.scaler.scale_) @ evaluator.decoder.coef_.T
        denominator = float(np.sum(requested**2))
        relative_errors.append(float(np.sum((evaluator_delta - requested) ** 2) / denominator))
        standardized_rms.extend(
            np.sqrt(np.mean((raw_delta / actuator.scaler.scale_) ** 2, axis=1))
        )
        oracle_signals.append(
            _conditional_signal(baseline_joint, _joint_tables(edited_q, mixture))
        )

    natural_rms = np.sqrt(np.mean(actuator.scaler.transform(fit_x) ** 2, axis=1))
    tensor = torch.from_numpy(splits["calibration"])
    with torch.no_grad():
        _, layers = model.forward_with_layers(tensor)
        zero_requested = np.zeros((len(tensor), 5), dtype=np.float64)
        zero_delta = intervention_delta(actuator, zero_requested).reshape(layers[0].shape)
        identity_memory = layers[0] + torch.from_numpy(zero_delta).to(layers[0].dtype)
        identity = persistent_conditionals(model, tensor, identity_memory)
        ordinary = []
        for token in range(mixture.vocab_size):
            appended = torch.cat(
                [tensor, torch.full((len(tensor), 1), token, dtype=tensor.dtype)], dim=1
            )
            logits, _ = model(appended)
            ordinary.append(logits[:, -1].softmax(-1).numpy())
    identity_error = float(np.max(np.abs(identity - np.stack(ordinary, axis=1))))
    exact_marginal = baseline_joint.sum(axis=2)
    filter_marginal = mixture.filter(splits["calibration"]).predictive[:, -1]
    enumeration_errors = [
        np.max(np.abs(table - enumerate_two_token_probabilities(q, mixture)))
        for q, table in zip(calibration_q, baseline_joint)
    ]
    for dose in (-np.log(2.0), np.log(2.0)):
        tilted = np.stack([source_odds_tilt(q, dose) for q in calibration_q])
        tilted_tables = _joint_tables(tilted, mixture)
        enumeration_errors.extend(
            np.max(np.abs(table - enumerate_two_token_probabilities(q, mixture)))
            for q, table in zip(tilted, tilted_tables)
        )
    exact_error = float(
        max(np.max(np.abs(exact_marginal - filter_marginal)), max(enumeration_errors))
    )

    metrics = {
        "component_r2": float(r2_score(component_true, component_predicted)),
        "joint_r2": float(r2_score(calibration_q[:, :5], predicted)),
        "actuator_rank": float(actuator.retained_rank),
        "actuator_constraint_relative_error": float(max(constraint_errors)),
        "displacement_relative_error": float(max(relative_errors)),
        "oracle_denominator": float(min(oracle_signals)),
        "edit_rms_p95": float(np.percentile(standardized_rms, 95)),
        "natural_rms_p95": float(np.percentile(natural_rms, 95)),
        "identity_max_abs_error": identity_error,
        "exact_max_abs_error": exact_error,
    }
    return CalibrationResult(actuator, evaluator, shuffled, metrics)


def fit_actuator(
    features: ArrayLike, targets: ArrayLike, *, alpha: float, rcond: float
) -> LinearActuator:
    """Fit a standardized ridge decoder and its minimum-norm inverse."""
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    scaler = StandardScaler().fit(x)
    decoder = Ridge(alpha=alpha, solver="lsqr").fit(scaler.transform(x), y)
    singular_values = np.linalg.svd(decoder.coef_, compute_uv=False)
    retained_rank = int(np.sum(singular_values > rcond * singular_values[0]))
    pseudoinverse = np.linalg.pinv(decoder.coef_, rcond=rcond)
    return LinearActuator(scaler, decoder, pseudoinverse, retained_rank)


def intervention_delta(actuator: LinearActuator, target_delta: ArrayLike) -> FloatArray:
    """Return the raw-feature edit realizing a decoded target displacement."""
    requested = np.asarray(target_delta, dtype=np.float64)
    if requested.ndim == 1:
        standardized_delta = actuator.pseudoinverse @ requested
    elif requested.ndim == 2:
        standardized_delta = requested @ actuator.pseudoinverse.T
    else:
        raise ValueError("target displacement must be a vector or matrix")
    return standardized_delta * actuator.scaler.scale_


def actuator_constraint_diagnostics(
    actuator: LinearActuator, requested_delta: ArrayLike
) -> dict[str, float]:
    """Measure whether the frozen pseudoinverse actually installs requested changes."""
    requested = np.asarray(requested_delta, dtype=np.float64)
    raw = intervention_delta(actuator, requested)
    standardized = raw / actuator.scaler.scale_
    installed = (
        actuator.decoder.coef_ @ standardized
        if standardized.ndim == 1
        else standardized @ actuator.decoder.coef_.T
    )
    denominator = float(np.sum(requested**2))
    error = float(np.sum((installed - requested) ** 2) / denominator) if denominator else 0.0
    return {
        "actuator_rank": float(actuator.retained_rank),
        "actuator_constraint_relative_error": error,
    }


@torch.no_grad()
def persistent_conditionals(
    model: TransformerPredictor, tokens: Tensor, edited_prefix: Tensor
) -> FloatArray:
    """Predict the token after each possible appended token with retained prefix memory."""
    if tokens.ndim != 2 or edited_prefix.shape[:2] != tokens.shape:
        raise ValueError("edited prefix must match token batch and prefix length")
    branches = []
    for token in range(model.output.out_features):
        appended = torch.cat(
            [
                tokens,
                torch.full(
                    (len(tokens), 1), token, dtype=tokens.dtype, device=tokens.device
                ),
            ],
            dim=1,
        )
        _, activations = model.forward_with_layers(appended)
        block_one = activations[0].clone()
        block_one[:, : tokens.shape[1]] = edited_prefix
        logits, _ = model.logits_from_depth(block_one, depth=0)
        branches.append(logits[:, -1].softmax(-1))
    return torch.stack(branches, dim=1).cpu().numpy().astype(np.float64)


def _norm_match(
    candidate_raw: FloatArray, learned_raw: FloatArray, scale: FloatArray
) -> FloatArray:
    candidate = candidate_raw / scale
    learned = learned_raw / scale
    candidate_norm = np.linalg.norm(candidate, axis=1)
    learned_norm = np.linalg.norm(learned, axis=1)
    factors = np.divide(
        learned_norm,
        candidate_norm,
        out=np.zeros_like(learned_norm),
        where=candidate_norm > 0,
    )
    return candidate_raw * factors[:, None]


def signed_random_delta(
    direction: ArrayLike,
    standardized_norms: ArrayLike,
    scale: ArrayLike,
    *,
    dose: float,
) -> FloatArray:
    """Construct a fixed random direction with signed dose and matched norms."""
    vector = np.asarray(direction, dtype=np.float64)
    vector = vector / np.linalg.norm(vector)
    norms = np.asarray(standardized_norms, dtype=np.float64)
    feature_scale = np.asarray(scale, dtype=np.float64)
    return np.sign(dose) * norms[:, None] * vector[None, :] * feature_scale


def _response_contributions(
    baseline_joint: FloatArray,
    edited_joint: FloatArray,
    baseline_model: FloatArray,
    edited_model: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    weights = baseline_joint.sum(axis=2)
    exact_baseline = baseline_joint / weights[:, :, None]
    edited_weights = edited_joint.sum(axis=2)
    exact_edited = edited_joint / edited_weights[:, :, None]
    target = exact_edited - exact_baseline
    observed = edited_model - baseline_model
    numerator = np.sum(weights[:, :, None] * (observed - target) ** 2, axis=(1, 2))
    denominator = np.sum(weights[:, :, None] * target**2, axis=(1, 2))
    return numerator, denominator


@torch.no_grad()
def evaluate_responses(
    model: TransformerPredictor,
    mixture: HMMMixture,
    evaluation_tokens: ArrayLike,
    calibration: CalibrationResult,
    *,
    doses: tuple[float, ...],
    random_controls: int,
    random_seed: int,
) -> list[dict]:
    """Evaluate the learned persistent edit and all fixed norm-matched controls."""
    tokens_array = np.asarray(evaluation_tokens, dtype=np.int64)
    tokens = torch.from_numpy(tokens_array)
    model.eval()
    _, activations = model.forward_with_layers(tokens)
    base_memory = activations[0]
    base_features = base_memory.numpy().reshape(len(tokens), -1).astype(np.float64)
    baseline_model = persistent_conditionals(model, tokens, base_memory)
    baseline_first_logits, _ = model.logits_from_depth(base_memory, depth=0)
    baseline_first = baseline_first_logits[:, -1].softmax(-1).numpy().astype(np.float64)
    beliefs = belief_matrix(mixture, tokens_array)
    baseline_joint = _joint_tables(beliefs, mixture)
    rng = np.random.default_rng(random_seed)
    random_directions = rng.normal(
        size=(random_controls, base_features.shape[1])
    )
    random_directions /= np.linalg.norm(random_directions, axis=1, keepdims=True)
    rows: list[dict] = []

    for dose in doses:
        edited_beliefs = np.stack([source_odds_tilt(q, dose) for q in beliefs])
        requested = edited_beliefs[:, :5] - beliefs[:, :5]
        learned_raw = intervention_delta(calibration.actuator, requested)
        shuffled_raw = intervention_delta(calibration.shuffled, requested)
        shuffled_raw = _norm_match(
            shuffled_raw, learned_raw, calibration.actuator.scaler.scale_
        )
        learned_standard_norm = np.linalg.norm(
            learned_raw / calibration.actuator.scaler.scale_, axis=1
        )
        controls: list[tuple[str, int | None, FloatArray]] = [
            ("learned", None, learned_raw),
            ("shuffled", None, shuffled_raw),
        ]
        for random_index, direction in enumerate(random_directions):
            controls.append(
                (
                    "random",
                    random_index,
                    signed_random_delta(
                        direction,
                        learned_standard_norm,
                        calibration.actuator.scaler.scale_,
                        dose=dose,
                    ),
                )
            )

        edited_joint = _joint_tables(edited_beliefs, mixture)
        for control, random_index, raw_delta in controls:
            edited_memory = base_memory + torch.from_numpy(
                raw_delta.reshape(base_memory.shape)
            ).to(dtype=base_memory.dtype)
            edited_model = persistent_conditionals(model, tokens, edited_memory)
            first_logits, _ = model.logits_from_depth(edited_memory, depth=0)
            edited_first = first_logits[:, -1].softmax(-1).numpy().astype(np.float64)
            numerator, denominator = _response_contributions(
                baseline_joint, edited_joint, baseline_model, edited_model
            )
            score = float(1.0 - numerator.sum() / denominator.sum())
            row = {
                "record_type": "response",
                "control": control,
                "dose": float(dose),
                "score": score,
                "numerator": float(numerator.sum()),
                "denominator": float(denominator.sum()),
                "example_numerator": numerator.tolist(),
                "example_denominator": denominator.tolist(),
                "standardized_edit_rms_mean": float(
                    np.mean(
                        np.sqrt(
                            np.mean(
                                (raw_delta / calibration.actuator.scaler.scale_) ** 2,
                                axis=1,
                            )
                        )
                    )
                ),
                "first_token_response_mse": float(np.mean((edited_first - baseline_first) ** 2)),
            }
            if random_index is not None:
                row["random_index"] = random_index
            rows.append(row)
    return rows
