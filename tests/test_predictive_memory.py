import numpy as np
import pytest
import torch
from pathlib import Path

from nonergodic_memory.data.hmm import make_mess3_mixture
from nonergodic_memory.models.sequence import TransformerPredictor
from nonergodic_memory.predictive_memory import (
    actuator_constraint_diagnostics,
    belief_matrix,
    block1_features,
    calibrate_memory,
    classify_pilot,
    evaluate_responses,
    enumerate_two_token_probabilities,
    fit_actuator,
    intervention_delta,
    exact_conditionals,
    joint_belief,
    persistent_conditionals,
    passes_feasibility_gates,
    response_score,
    signed_random_delta,
    sample_prefix_splits,
    source_odds_tilt,
    two_token_probabilities,
    validate_complete_evidence,
)
from predictive_memory import (
    atomic_write_jsonl,
    preflight_outputs,
    reserve_run,
    validate_registered_config,
    run_registered,
)


def test_source_odds_tilt_preserves_conditional_states_and_changes_odds():
    q = np.array([0.10, 0.15, 0.25, 0.20, 0.18, 0.12])
    tilted = source_odds_tilt(q, np.log(2.0))

    old_weights = q.reshape(2, 3).sum(axis=1)
    new_weights = tilted.reshape(2, 3).sum(axis=1)
    assert new_weights[1] / new_weights[0] == pytest.approx(
        2.0 * old_weights[1] / old_weights[0]
    )
    np.testing.assert_allclose(
        tilted.reshape(2, 3) / new_weights[:, None],
        q.reshape(2, 3) / old_weights[:, None],
    )
    assert tilted.sum() == pytest.approx(1.0)


def test_two_token_operator_matches_direct_bayesian_enumeration():
    mixture = make_mess3_mixture()
    prefix = np.array([[0, 1, 0, 2, 2, 1, 0]])
    filtered = mixture.filter(prefix)
    q = joint_belief(filtered.component_posterior[0, -1], filtered.state_posterior[0, -1])
    joint = two_token_probabilities(q, mixture)

    expected = np.zeros((3, 3))
    for a in range(3):
        branch = np.concatenate([prefix, np.array([[a]])], axis=1)
        p_a = filtered.predictive[0, -1, a]
        expected[a] = p_a * mixture.filter(branch).predictive[0, -1]

    np.testing.assert_allclose(joint, expected, atol=1e-12)
    np.testing.assert_allclose(joint.sum(axis=1), filtered.predictive[0, -1], atol=1e-12)
    assert joint.sum() == pytest.approx(1.0)


def test_independent_latent_enumeration_matches_operator_for_tilted_belief():
    mixture = make_mess3_mixture()
    q = np.array([0.10, 0.15, 0.25, 0.20, 0.18, 0.12])
    tilted = source_odds_tilt(q, -np.log(2.0))
    np.testing.assert_allclose(
        two_token_probabilities(tilted, mixture),
        enumerate_two_token_probabilities(tilted, mixture),
        atol=1e-14,
    )


def test_exact_conditionals_are_normalized_and_reject_zero_rows():
    joint = np.array([[0.10, 0.05, 0.05], [0.15, 0.15, 0.10], [0.20, 0.10, 0.10]])
    conditionals = exact_conditionals(joint)
    np.testing.assert_allclose(conditionals.sum(axis=1), 1.0)
    with pytest.raises(ValueError, match="positive row masses"):
        exact_conditionals(np.zeros((3, 3)))


def test_actuator_installs_requested_decoded_displacement_at_minimum_norm():
    rng = np.random.default_rng(4)
    features = rng.normal(size=(300, 8))
    mapping = rng.normal(size=(8, 5))
    targets = features @ mapping + 0.01 * rng.normal(size=(300, 5))
    actuator = fit_actuator(features, targets, alpha=1.0, rcond=1e-3)
    requested = np.array([0.02, -0.01, 0.03, -0.02, 0.01])
    delta = intervention_delta(actuator, requested)

    decoded_delta = actuator.decoder.coef_ @ (delta / actuator.scaler.scale_)
    np.testing.assert_allclose(decoded_delta, requested, atol=1e-10)
    assert np.all(np.isfinite(delta))


def test_actuator_constraint_diagnostics_detect_truncated_target_direction():
    rng = np.random.default_rng(8)
    features = rng.normal(size=(120, 7))
    targets = np.column_stack([features[:, :4], np.zeros(len(features))])
    actuator = fit_actuator(features, targets, alpha=1.0, rcond=1e-3)
    requested = np.tile(np.array([0.0, 0.0, 0.0, 0.0, 0.1]), (12, 1))
    diagnostics = actuator_constraint_diagnostics(actuator, requested)
    assert diagnostics["actuator_rank"] == 4
    assert diagnostics["actuator_constraint_relative_error"] == pytest.approx(1.0)


def test_persistent_identity_matches_ordinary_extended_forward_pass():
    torch.manual_seed(7)
    model = TransformerPredictor(vocab_size=3, width=8, layers=2, heads=2, max_length=40)
    model.eval()
    tokens = torch.randint(0, 3, (5, 6))
    with torch.no_grad():
        _, layers = model.forward_with_layers(tokens)
        persistent = persistent_conditionals(model, tokens, layers[0])
        expected = []
        for token in range(3):
            appended = torch.cat(
                [tokens, torch.full((len(tokens), 1), token, dtype=tokens.dtype)], dim=1
            )
            logits, _ = model(appended)
            expected.append(logits[:, -1].softmax(-1))
    expected_tensor = torch.stack(expected, dim=1).numpy()
    np.testing.assert_allclose(persistent, expected_tensor, atol=1e-6)


def test_response_score_has_exact_and_no_response_reference_points():
    mixture = make_mess3_mixture()
    q = np.array([0.10, 0.15, 0.25, 0.20, 0.18, 0.12])
    baseline_joint = two_token_probabilities(q, mixture)[None]
    tilted_joint = two_token_probabilities(source_odds_tilt(q, np.log(2)), mixture)[None]
    baseline_conditional = exact_conditionals(baseline_joint[0])[None]
    tilted_conditional = exact_conditionals(tilted_joint[0])[None]

    exact = response_score(
        baseline_joint, tilted_joint, baseline_conditional, tilted_conditional
    )
    no_response = response_score(
        baseline_joint, tilted_joint, baseline_conditional, baseline_conditional
    )
    assert exact["score"] == pytest.approx(1.0)
    assert no_response["score"] == pytest.approx(0.0)
    assert exact["denominator"] > 0


def test_registered_split_sampler_is_deterministic_and_disjoint():
    mixture = make_mess3_mixture()
    sizes = {"actuator_fit": 20, "decoder_fit": 12, "calibration": 8, "evaluation": 7}
    first = sample_prefix_splits(mixture, model_seed=10, sizes=sizes, length=9)
    second = sample_prefix_splits(mixture, model_seed=10, sizes=sizes, length=9)

    assert set(first) == set(sizes)
    for name in sizes:
        np.testing.assert_array_equal(first[name], second[name])
    encoded = {
        name: {tuple(row) for row in values}
        for name, values in first.items()
    }
    for left_index, left in enumerate(sizes):
        for right in list(sizes)[left_index + 1 :]:
            assert encoded[left].isdisjoint(encoded[right])


def test_memory_calibration_uses_heldout_prefixes_and_reports_finite_metrics():
    torch.manual_seed(13)
    mixture = make_mess3_mixture()
    model = TransformerPredictor(vocab_size=3, width=8, layers=2, heads=2, max_length=16)
    splits = sample_prefix_splits(
        mixture,
        model_seed=3,
        sizes={"actuator_fit": 80, "decoder_fit": 60, "calibration": 24, "evaluation": 12},
        length=6,
    )
    result = calibrate_memory(model, mixture, splits, alpha=1.0, rcond=1e-3)

    targets = belief_matrix(mixture, splits["calibration"])
    features = block1_features(model, splits["calibration"])
    assert targets.shape == (24, 6)
    assert features.shape == (24, 6 * 8)
    assert targets.sum(axis=1) == pytest.approx(np.ones(24))
    assert np.isfinite(list(result.metrics.values())).all()
    assert result.actuator.decoder.coef_.shape == (5, 48)
    assert result.evaluator.decoder.coef_.shape == (5, 48)


def test_response_evaluation_covers_learned_shuffled_and_random_controls():
    torch.manual_seed(17)
    mixture = make_mess3_mixture()
    model = TransformerPredictor(vocab_size=3, width=8, layers=2, heads=2, max_length=16)
    splits = sample_prefix_splits(
        mixture,
        model_seed=4,
        sizes={"actuator_fit": 80, "decoder_fit": 60, "calibration": 20, "evaluation": 9},
        length=6,
    )
    calibration = calibrate_memory(model, mixture, splits, alpha=1.0, rcond=1e-3)
    rows = evaluate_responses(
        model,
        mixture,
        splits["evaluation"],
        calibration,
        doses=(-np.log(2.0), np.log(2.0)),
        random_controls=2,
        random_seed=33,
    )

    assert len(rows) == 8
    assert {(row["control"], row.get("random_index")) for row in rows} == {
        ("learned", None),
        ("shuffled", None),
        ("random", 0),
        ("random", 1),
    }
    assert all(np.isfinite(row["score"]) for row in rows)
    assert all(len(row["example_numerator"]) == 9 for row in rows)
    assert all(sum(row["example_denominator"]) > 0 for row in rows)


def test_random_control_reverses_with_dose_and_matches_standardized_norm():
    direction = np.array([3.0, 4.0])
    norms = np.array([1.0, 2.0])
    scale = np.array([2.0, 0.5])
    positive = signed_random_delta(direction, norms, scale, dose=np.log(2.0))
    negative = signed_random_delta(direction, norms, scale, dose=-np.log(2.0))
    np.testing.assert_allclose(negative, -positive)
    np.testing.assert_allclose(np.linalg.norm(positive / scale, axis=1), norms)


def test_feasibility_gate_names_every_registered_check():
    metrics = {
        "component_r2": 0.3,
        "joint_r2": 0.6,
        "actuator_rank": 5.0,
        "actuator_constraint_relative_error": 1e-14,
        "displacement_relative_error": 0.4,
        "oracle_denominator": 2e-6,
        "edit_rms_p95": 0.8,
        "natural_rms_p95": 0.9,
        "identity_max_abs_error": 1e-7,
        "exact_max_abs_error": 1e-12,
    }
    gates = {
        "component_r2_min": 0.2,
        "joint_r2_min": 0.5,
        "actuator_rank_min": 5,
        "actuator_constraint_relative_error_max": 1e-10,
        "displacement_relative_error_max": 0.5,
        "oracle_denominator_min": 1e-6,
        "identity_atol": 1e-6,
        "exact_atol": 1e-10,
    }
    checks = passes_feasibility_gates(metrics, gates)
    assert all(checks.values())
    metrics["component_r2"] = 0.19
    assert passes_feasibility_gates(metrics, gates)["component_r2"] is False


def test_pilot_classification_applies_equal_seed_and_control_margins():
    rows = []
    for seed in range(20, 25):
        for dose in (-0.693, 0.693):
            for control, score, random_index in (
                ("learned", 0.35, None),
                ("shuffled", 0.10, None),
                ("random", 0.05, 0),
                ("random", 0.15, 1),
            ):
                row = {
                    "seed": seed,
                    "step": 3072,
                    "dose": dose,
                    "control": control,
                    "score": score,
                    "numerator": 1.0 - score,
                    "denominator": 1.0,
                    "example_numerator": [1.0 - score],
                    "example_denominator": [1.0],
                    "standardized_edit_rms_mean": 0.1,
                    "first_token_response_mse": 0.01,
                }
                if random_index is not None:
                    row["random_index"] = random_index
                rows.append(row)
    decision = classify_pilot(
        rows,
        heldout_seeds=(20, 21, 22, 23, 24),
        primary_step=3072,
        doses=(-0.693, 0.693),
        random_controls=2,
        examples_per_cell=1,
        mean_score_min=0.20,
        control_margin_min=0.10,
    )
    assert decision["status"] == "promising_pilot"
    for row in rows:
        if row["seed"] == 20 and row["control"] == "learned":
            row["score"] = -0.1
            row["numerator"] = 1.1
            row["example_numerator"] = [1.1]
    assert classify_pilot(
        rows,
        heldout_seeds=(20, 21, 22, 23, 24),
        primary_step=3072,
        doses=(-0.693, 0.693),
        random_controls=2,
        examples_per_cell=1,
        mean_score_min=0.20,
        control_margin_min=0.10,
    )["status"] == "criterion_not_met"


def test_pilot_classification_rejects_incomplete_or_nonfinite_grid():
    row = {
        "seed": 20,
        "step": 3072,
        "dose": -0.693,
        "control": "learned",
        "score": float("nan"),
        "numerator": 1.0,
        "denominator": 1.0,
        "example_numerator": [1.0],
        "example_denominator": [1.0],
        "standardized_edit_rms_mean": 0.1,
        "first_token_response_mse": 0.01,
    }
    decision = classify_pilot(
        [row],
        heldout_seeds=(20,),
        primary_step=3072,
        doses=(-0.693, 0.693),
        random_controls=2,
        examples_per_cell=1,
        mean_score_min=0.20,
        control_margin_min=0.10,
    )
    assert decision["status"] == "invalid_pilot"
    assert decision["validity_errors"]


def test_complete_evidence_validator_rejects_missing_registered_cells():
    errors = validate_complete_evidence(
        [],
        development_seeds=(10, 11),
        heldout_seeds=(20, 21, 22, 23, 24),
        steps=(0, 3072),
        doses=(-0.693, 0.693),
        random_controls=8,
        examples_per_cell=128,
        split_sizes={"actuator_fit": 2048, "decoder_fit": 1024, "calibration": 128, "evaluation": 128},
        gates={
            "component_r2_min": 0.2,
            "joint_r2_min": 0.5,
            "actuator_rank_min": 5,
            "actuator_constraint_relative_error_max": 1e-10,
            "displacement_relative_error_max": 0.5,
            "oracle_denominator_min": 1e-6,
            "identity_atol": 1e-6,
            "exact_atol": 1e-10,
        },
        experiment_digest="digest",
    )
    assert any("split" in error for error in errors)
    assert any("calibration" in error for error in errors)
    assert any("response" in error for error in errors)


def test_result_preflight_refuses_to_overwrite_any_existing_evidence(tmp_path: Path):
    result = tmp_path / "result.jsonl"
    summary = tmp_path / "summary.jsonl"
    preflight_outputs((result, summary))
    result.write_text("evidence\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        preflight_outputs((result, summary))


def test_registered_config_digest_is_locked():
    from nonergodic_memory.experiment import load_config

    config = load_config(Path("configs/predictive_memory.yaml"))
    validate_registered_config(config)
    config["models"]["heldout_seeds"] = [20]
    with pytest.raises(ValueError, match="registered digest"):
        validate_registered_config(config)


def test_atomic_evidence_write_preserves_previous_snapshot_on_encoding_error(tmp_path: Path):
    destination = tmp_path / "evidence.jsonl"
    atomic_write_jsonl(destination, [{"record_type": "old", "value": 1}])
    before = destination.read_bytes()
    with pytest.raises(TypeError):
        atomic_write_jsonl(destination, [{"bad": object()}])
    assert destination.read_bytes() == before


def test_run_reservation_is_exclusive(tmp_path: Path):
    marker = tmp_path / "attempt.jsonl"
    reserve_run(marker, "digest")
    with pytest.raises(FileExistsError):
        reserve_run(marker, "digest")


def test_rejected_invocation_does_not_mutate_active_attempt(tmp_path: Path):
    results = tmp_path / "predictive_memory.jsonl"
    summary = tmp_path / "predictive_memory_summary.jsonl"
    attempt = tmp_path / "predictive_memory_attempt.jsonl"
    attempt.write_text('{"event":"started"}\n', encoding="utf-8")
    before = attempt.read_bytes()
    state = {"owned": False}
    with pytest.raises(FileExistsError):
        run_registered(
            Path("configs/predictive_memory.yaml"),
            results,
            summary,
            reservation_state=state,
        )
    assert state == {"owned": False}
    assert attempt.read_bytes() == before
    assert not summary.exists()
