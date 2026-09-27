from pathlib import Path

import copy

import numpy as np
import pytest
import torch

from nonergodic_memory import mess3_competence_time
from nonergodic_memory.experiment import config_digest, load_config
from nonergodic_memory.mess3_competence_time import (
    _rate_config,
    competence_time_checkpoint_path,
    run_competence_time_probes,
    run_competence_time_training,
    validate_competence_time_grid,
)


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "mess3_competence_time.yaml"


def _tiny_competence_time_config() -> dict:
    return {
        "data": {
            "generator": "mess3",
            "sampler": "vectorized",
            "sequence_length": 8,
            "train_sequences": 16,
            "test_sequences": 8,
        },
        "model": {"width": 8, "layers": 2, "heads": 2, "max_length": 16},
        "train": {
            "batch_size": 4,
            "learning_rate": 0.01,
            "weight_decay": 0.01,
            "checkpoint_steps": [0, 2],
        },
        "diagnosis": {"window": 3},
        "probe": {"train_sequences": 24, "test_sequences": 16},
        "competence_time": {"seeds": [30, 31], "learning_rates": [0.01, 0.005]},
    }


def test_competence_time_config_matches_preregistered_grid_and_lock() -> None:
    config = load_config(CONFIG)

    assert config["data"] == {
        "generator": "mess3",
        "sampler": "vectorized",
        "sequence_length": 64,
        "train_sequences": 2048,
        "test_sequences": 256,
    }
    assert config["model"] == {"width": 32, "layers": 2, "heads": 4, "max_length": 128}
    assert config["train"] == {
        "batch_size": 64,
        "learning_rate": 0.003,
        "weight_decay": 0.01,
        "checkpoint_steps": [0, 384, 768, 1152, 1536, 2048, 2560, 3072, 4096],
    }
    assert config["diagnosis"] == {"window": 8}
    assert config["probe"] == {"train_sequences": 1024, "test_sequences": 512}
    assert config["competence_time"] == {
        "seeds": [30, 31, 32, 33, 34, 35, 36, 37],
        "learning_rates": [0.00075, 0.0015, 0.003, 0.006],
        "primary_site": "block_2",
        "primary_target": "component_posterior",
        "primary_checkpoints": [384, 768, 1152, 1536, 2048, 2560, 3072, 4096],
        "controls": {
            "shuffled_labels": True,
            "untrained_step": 0,
            "seed_grouped_loso": True,
            "require_zero_probe_overlap": True,
        },
        "thresholds": {
            "max_shuffled_component_posterior_r2": 0.02,
            "competence_to_step_mse_ratio": 0.80,
            "minimum_competence_fold_wins": 7,
            "minimum_rate_dissociation_seeds": 6,
            "minimum_rate_competence_difference": 0.10,
        },
    }
    assert not set(config["competence_time"]["seeds"]).intersection(range(20, 25))
    assert config_digest(config) == "f75dd20f93eb8827"


def test_competence_time_rate_config_requires_locked_diagnosis_window() -> None:
    config = _tiny_competence_time_config()
    del config["diagnosis"]

    with pytest.raises(KeyError, match="diagnosis"):
        _rate_config(config, 0.01)


def test_competence_time_training_pairs_same_seed_initialization_and_covers_grid(
    tmp_path: Path,
) -> None:
    config = _tiny_competence_time_config()
    root = tmp_path / "checkpoints" / "mess3_competence_time"

    rows = run_competence_time_training(
        config,
        seeds=config["competence_time"]["seeds"],
        learning_rates=config["competence_time"]["learning_rates"],
        checkpoint_root=root,
    )

    assert {(row["seed"], row["learning_rate"], row["step"]) for row in rows} == {
        (seed, rate, step)
        for seed in (30, 31)
        for rate in (0.01, 0.005)
        for step in (0, 2)
    }
    for seed in (30, 31):
        fast = torch.load(
            competence_time_checkpoint_path(root, seed, 0.01, 0), weights_only=False
        )
        slow = torch.load(
            competence_time_checkpoint_path(root, seed, 0.005, 0), weights_only=False
        )
        for name, value in fast["state_dict"].items():
            torch.testing.assert_close(value, slow["state_dict"][name], rtol=0, atol=0)


def test_competence_time_probe_grid_has_complete_isolated_raw_keys(tmp_path: Path) -> None:
    config = _tiny_competence_time_config()
    root = tmp_path / "checkpoints" / "mess3_competence_time"
    training = run_competence_time_training(
        config,
        seeds=config["competence_time"]["seeds"],
        learning_rates=config["competence_time"]["learning_rates"],
        checkpoint_root=root,
    )

    probes = run_competence_time_probes(
        config,
        seeds=config["competence_time"]["seeds"],
        learning_rates=config["competence_time"]["learning_rates"],
        checkpoint_root=root,
    )

    validate_competence_time_grid(config, training, probes)
    required = {
        "seed",
        "learning_rate",
        "step",
        "base_config_sha256",
        "rate_config_sha256",
        "sampler",
        "checkpoint_path",
    }
    assert all(required <= row.keys() for row in [*training, *probes])
    assert len(probes) == 2 * 2 * 2 * 3 * 2


def test_competence_time_grid_rejects_incompatible_base_and_rate_provenance(
    tmp_path: Path,
) -> None:
    config = _tiny_competence_time_config()
    root = tmp_path / "checkpoints" / "mess3_competence_time"
    training = run_competence_time_training(
        config,
        seeds=config["competence_time"]["seeds"],
        learning_rates=config["competence_time"]["learning_rates"],
        checkpoint_root=root,
    )
    probes = run_competence_time_probes(
        config,
        seeds=config["competence_time"]["seeds"],
        learning_rates=config["competence_time"]["learning_rates"],
        checkpoint_root=root,
    )

    wrong_base = copy.deepcopy(training)
    wrong_base[0]["base_config_sha256"] = "wrong"
    with pytest.raises(ValueError, match="provenance"):
        validate_competence_time_grid(config, wrong_base, probes)

    wrong_rate = copy.deepcopy(probes)
    wrong_rate[0]["rate_config_sha256"] = "wrong"
    with pytest.raises(ValueError, match="rate provenance"):
        validate_competence_time_grid(config, training, wrong_rate)


def _synthetic_analysis_grid(kind: str = "competence") -> tuple[dict, list[dict], list[dict]]:
    """In-memory mathematical fixtures; never train confirmation models."""
    config = load_config(CONFIG)
    training, probes = [], []
    for seed in config["competence_time"]["seeds"]:
        for rate_index, rate in enumerate(config["competence_time"]["learning_rates"]):
            digest = config_digest(_rate_config(config, rate))
            for step in config["train"]["checkpoint_steps"]:
                competence = 0.12 * rate_index + 0.3 * step / 4096 + 0.003 * (seed - 30)
                clock = np.log1p(step)
                geometry = (
                    0.1 + 0.4 * competence + 0.3 * competence**2
                    if kind == "competence" else 0.1 + 0.01 * clock**2
                )
                common = {
                    "seed": seed, "learning_rate": rate, "step": step,
                    "base_config_sha256": config_digest(config),
                    "rate_config_sha256": digest, "config_sha256": digest,
                    "sampler": "vectorized",
                    "checkpoint_path": f"synthetic/lr_{rate}/seed{seed}_step{step}.pt",
                }
                training.append({
                    **common, "record_type": "competence_time_training",
                    "competence": competence, "kl_exact": 1 - competence,
                    "nll": 1.2 - competence, "uniform_kl": 1.0,
                })
                for site in ("block_1", "block_2", "final_norm"):
                    for control in ("none", "shuffled_labels"):
                        probes.append({
                            **common, "record_type": "competence_time_probe",
                            "site": site, "control": control,
                            "component_posterior_r2": geometry if control == "none" else 0.0,
                            "probe_sequence_overlap": 0,
                            "component_accuracy": 0.5, "conditional_state_accuracy": 0.5,
                            "state_posterior_r2": 0.0, "joint_belief_mse": 0.1,
                            "joint_belief_r2": 0.0, "joint_distance_r2": 0.0,
                        })
    return config, training, probes


def _analyze(config: dict, training: list[dict], probes: list[dict]) -> dict:
    analyze = getattr(mess3_competence_time, "analyze_competence_time", None)
    assert callable(analyze), "the preregistered analysis API is missing"
    return analyze(config, training, probes)


def test_analysis_support_and_grouped_training_only_scaling() -> None:
    config, training, probes = _synthetic_analysis_grid()
    result = _analyze(config, training, probes)

    assert result["verdict"] == "supported"
    assert result["validity_failures"] == []
    assert result["primary"]["competence_to_step_mse_ratio"] < 0.80
    assert result["primary"]["competence_fold_wins"] == 8
    assert result["excluded_initialization_count"] == 32
    assert result["rate_dissociation"]["passing_seeds"] == list(range(30, 38))
    folds = result["primary"]["folds"]
    assert len(folds) == 8
    for fold in folds:
        held_out = fold["held_out_seed"]
        rows = [row for row in training if row["seed"] != held_out and row["step"] > 0]
        assert fold["training_seeds"] == [seed for seed in range(30, 38) if seed != held_out]
        assert fold["test_cells"] == 32
        assert fold["training_cells"] == 224
        for name, values in (
            ("competence", [row["competence"] for row in rows]),
            ("log_step", np.log1p([row["step"] for row in rows])),
        ):
            assert fold["scaling"][name]["center"] == pytest.approx(np.mean(values))
            assert fold["scaling"][name]["scale"] == pytest.approx(np.std(values))
    for metric, fold_metric in (("competence_loso_mse", "competence_mse"), ("log_step_loso_mse", "step_mse")):
        assert result["primary"][metric] == pytest.approx(
            sum(fold[fold_metric] * fold["test_cells"] for fold in folds) / 256
        )


def test_analysis_falsifies_valid_step_indexed_geometry() -> None:
    result = _analyze(*_synthetic_analysis_grid("step"))
    assert result["verdict"] == "falsified"
    assert result["validity_failures"] == []
    assert result["primary"]["competence_fold_wins"] == 0


def test_analysis_pooled_advantage_does_not_override_only_six_seed_wins() -> None:
    config, training, probes = _synthetic_analysis_grid()
    for row in probes:
        if row["seed"] >= 36 and row["control"] == "none":
            c = 0.18 + 0.3 * row["step"] / 4096 + 0.003 * (row["seed"] - 30)
            row["component_posterior_r2"] = 0.1 + 0.4 * c + 0.3 * c**2
    result = _analyze(config, training, probes)
    assert result["primary"]["competence_to_step_mse_ratio"] < 0.80
    assert result["primary"]["competence_fold_wins"] == 6
    assert result["verdict"] == "falsified"


def test_analysis_zero_error_ties_are_falsified_and_json_serializable() -> None:
    import json

    config, training, probes = _synthetic_analysis_grid()
    for row in probes:
        row["component_posterior_r2"] = 0.0
    result = _analyze(config, training, probes)
    assert result["verdict"] == "falsified"
    assert result["primary"]["competence_fold_wins"] == 0
    assert result["primary"]["competence_to_step_mse_ratio"] is None
    json.dumps(result, allow_nan=False)


def test_analysis_initialization_and_secondary_geometry_do_not_enter_primary_fit() -> None:
    config, training, probes = _synthetic_analysis_grid()
    original = _analyze(config, training, probes)
    for row in training:
        if row["step"] == 0:
            row["competence"] = -1e6
    for row in probes:
        if row["step"] == 0 or row["site"] != "block_2":
            row["component_posterior_r2"] = -1e6
    result = _analyze(config, training, probes)
    assert result["verdict"] == "supported"
    assert result["primary"] == original["primary"]


@pytest.mark.parametrize("shuffled_r2", [0.020001, -0.020001])
def test_analysis_shuffled_control_failure_overrides_support(shuffled_r2: float) -> None:
    config, training, probes = _synthetic_analysis_grid()
    next(row for row in probes if row["step"] > 0 and row["site"] == "block_2"
         and row["control"] == "shuffled_labels")["component_posterior_r2"] = shuffled_r2
    result = _analyze(config, training, probes)
    assert result["verdict"] == "inconclusive"
    assert "shuffled_labels" in result["validity_failures"]


def test_analysis_shuffled_control_includes_exact_boundary() -> None:
    config, training, probes = _synthetic_analysis_grid()
    for row in probes:
        if row["control"] == "shuffled_labels":
            row["component_posterior_r2"] = -0.02
    assert _analyze(config, training, probes)["verdict"] == "supported"


@pytest.mark.parametrize("failure", ["missing_training", "missing_probe", "duplicate", "loss_nan", "competence_inf", "geometry_nan", "missing_site", "null_metric", "provenance"])
def test_analysis_incomplete_nonfinite_or_malformed_grid_is_inconclusive(failure: str) -> None:
    config, training, probes = _synthetic_analysis_grid()
    if failure == "missing_training":
        training.pop()
    elif failure == "missing_probe":
        probes.pop()
    elif failure == "duplicate":
        training.append(training[-1].copy())
    elif failure == "loss_nan":
        training[-1]["nll"] = float("nan")
    elif failure == "competence_inf":
        training[-1]["competence"] = float("inf")
    elif failure == "geometry_nan":
        probes[-1]["component_posterior_r2"] = float("nan")
    elif failure == "missing_site":
        del probes[-1]["site"]
    elif failure == "null_metric":
        training[-1]["nll"] = None
    elif failure == "provenance":
        probes[-1]["base_config_sha256"] = "wrong"
    result = _analyze(config, training, probes)
    assert result["verdict"] == "inconclusive"
    assert result["validity_failures"]
    assert result["primary"] is None
    assert result["grid_error"]


@pytest.mark.parametrize("overlap", [1, 0.5])
def test_analysis_probe_leakage_is_inconclusive(overlap: float) -> None:
    config, training, probes = _synthetic_analysis_grid()
    probes[-1]["probe_sequence_overlap"] = overlap
    result = _analyze(config, training, probes)
    assert result["verdict"] == "inconclusive"
    assert "probe_leakage" in result["validity_failures"]


@pytest.mark.parametrize("field, value", [("seed", 30.5), ("step", 0.5)])
def test_analysis_fractional_grid_identity_is_inconclusive(field: str, value: float) -> None:
    config, training, probes = _synthetic_analysis_grid()
    training[0][field] = value
    result = _analyze(config, training, probes)
    assert result["verdict"] == "inconclusive"
    assert result["primary"] is None


def test_analysis_null_raw_row_is_inconclusive() -> None:
    config, training, probes = _synthetic_analysis_grid()
    probes[0] = None
    result = _analyze(config, training, probes)
    assert result["verdict"] == "inconclusive"
    assert result["grid_error"]


@pytest.mark.parametrize("dissociated_seeds, expected", [(5, "inconclusive"), (6, "supported")])
def test_analysis_rate_dissociation_requires_at_least_six_seeds(dissociated_seeds: int, expected: str) -> None:
    config, training, probes = _synthetic_analysis_grid()
    competence_by_cell = {}
    for row in training:
        if row["seed"] >= 30 + dissociated_seeds:
            row["competence"] = 0.2 + 0.3 * row["step"] / 4096
        competence_by_cell[row["seed"], row["learning_rate"], row["step"]] = row["competence"]
    for row in probes:
        if row["control"] == "none":
            c = competence_by_cell[row["seed"], row["learning_rate"], row["step"]]
            row["component_posterior_r2"] = 0.1 + 0.4 * c + 0.3 * c**2
    result = _analyze(config, training, probes)
    assert result["verdict"] == expected
    assert len(result["rate_dissociation"]["passing_seeds"]) == dissociated_seeds
    assert ("rate_dissociation" in result["validity_failures"]) == (dissociated_seeds < 6)
