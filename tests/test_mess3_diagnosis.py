import numpy as np
import pytest

from nonergodic_memory.data import make_mess3_mixture
from nonergodic_memory.mess3_diagnosis import (
    competence,
    predictive_baselines,
    predictive_kl,
)


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
