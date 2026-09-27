from pathlib import Path

import copy

import pytest
import torch

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
