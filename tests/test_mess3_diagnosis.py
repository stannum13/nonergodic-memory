from pathlib import Path

import numpy as np
import pytest
import torch

from nonergodic_memory.data import make_mess3_mixture
from nonergodic_memory.mess3_diagnosis import (
    competence,
    predictive_baselines,
    predictive_kl,
    train_diagnostic,
)


def _tiny_diagnostic_config() -> dict:
    return {
        "data": {
            "generator": "mess3",
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
        "probe": {"train_sequences": 8, "test_sequences": 8},
    }


def test_predictive_kl_and_competence_reference_points() -> None:
    exact = np.array([[0.6, 0.3, 0.1], [0.2, 0.2, 0.6]])
    uniform = np.full_like(exact, 1 / 3)
    uniform_kl = predictive_kl(exact, uniform)

    assert predictive_kl(exact, exact) == pytest.approx(0.0, abs=1e-15)
    assert competence(uniform_kl, uniform_kl) == pytest.approx(0.0)
    assert competence(0.0, uniform_kl) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="positive"):
        competence(0.0, 0.0)


def test_predictive_baselines_are_aligned_and_normalized() -> None:
    mixture = make_mess3_mixture()
    fit = mixture.sample(128, 12, seed=8)
    test = mixture.sample(24, 12, seed=9)
    rows = predictive_baselines(mixture, fit, test, window=4)

    by_name = {row["predictor"]: row for row in rows}
    assert set(by_name) == {"uniform", "last_token", "window_4_bayes", "full_bayes"}
    assert {row["positions"] for row in rows} == {24 * (11 - 4 + 1)}
    assert by_name["full_bayes"]["kl_exact"] == pytest.approx(0.0, abs=1e-15)
    assert by_name["uniform"]["competence"] == pytest.approx(0.0)
    assert by_name["full_bayes"]["competence"] == pytest.approx(1.0)
    assert np.isfinite([row["nll"] for row in rows]).all()


def test_predictive_baselines_reject_invalid_window() -> None:
    mixture = make_mess3_mixture()
    batch = mixture.sample(8, 6, seed=4)
    with pytest.raises(ValueError, match="window"):
        predictive_baselines(mixture, batch, batch, window=6)


def test_diagnostic_training_matches_initialization_and_requested_steps(tmp_path: Path) -> None:
    config = _tiny_diagnostic_config()
    reused = train_diagnostic(config, seed=5, condition="reused", output_dir=tmp_path / "reused")
    fresh = train_diagnostic(config, seed=5, condition="fresh", output_dir=tmp_path / "fresh")

    assert [row["step"] for row in reused] == [0, 2]
    assert [row["step"] for row in fresh] == [0, 2]
    reused_zero = torch.load(
        tmp_path / "reused/transformer_seed5_reused_step0.pt", weights_only=False
    )
    fresh_zero = torch.load(
        tmp_path / "fresh/transformer_seed5_fresh_step0.pt", weights_only=False
    )
    assert reused_zero["condition"] == "reused"
    assert fresh_zero["condition"] == "fresh"
    for name, value in reused_zero["state_dict"].items():
        torch.testing.assert_close(value, fresh_zero["state_dict"][name], rtol=0, atol=0)
    assert all(np.isfinite(row["kl_exact"]) for row in reused + fresh)
    assert all(row["updates"] == row["step"] for row in reused + fresh)


def test_diagnostic_training_is_deterministic_and_rejects_unknown_condition(
    tmp_path: Path,
) -> None:
    config = _tiny_diagnostic_config()
    first = train_diagnostic(config, seed=7, condition="fresh", output_dir=tmp_path / "a")
    second = train_diagnostic(config, seed=7, condition="fresh", output_dir=tmp_path / "b")
    comparable = ("step", "nll", "kl_exact", "uniform_kl", "competence")
    assert [[row[key] for key in comparable] for row in first] == [
        [row[key] for key in comparable] for row in second
    ]
    with pytest.raises(ValueError, match="condition"):
        train_diagnostic(config, seed=7, condition="other", output_dir=tmp_path / "bad")
