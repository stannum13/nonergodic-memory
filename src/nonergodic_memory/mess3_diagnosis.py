"""Predictive baselines and matched training diagnostics for Mess3."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .context import oracle_window_beliefs
from .data.hmm import HMMMixture, SequenceBatch


FloatArray = NDArray[np.float64]


def _probability_rows(values: FloatArray, name: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or not np.all(np.isfinite(array)) or np.any(array < 0):
        raise ValueError(f"{name} must be a finite nonnegative probability matrix")
    if not np.allclose(array.sum(axis=1), 1.0):
        raise ValueError(f"{name} rows must sum to one")
    return array


def predictive_kl(exact: FloatArray, candidate: FloatArray) -> float:
    """Mean KL(exact || candidate), in nats."""
    truth = _probability_rows(exact, "exact")
    proposal = _probability_rows(candidate, "candidate")
    if truth.shape != proposal.shape:
        raise ValueError("exact and candidate distributions must have the same shape")
    return float(
        np.mean(
            np.sum(
                truth * (np.log(np.clip(truth, 1e-12, None)) - np.log(np.clip(proposal, 1e-12, None))),
                axis=1,
            )
        )
    )


def competence(kl_exact: float, uniform_kl: float) -> float:
    """Fraction of the uniform-to-Bayes predictive KL gap recovered."""
    if not np.isfinite(uniform_kl) or uniform_kl <= 0:
        raise ValueError("uniform KL must be finite and positive")
    if not np.isfinite(kl_exact) or kl_exact < 0:
        raise ValueError("predictive KL must be finite and nonnegative")
    return float(1.0 - kl_exact / uniform_kl)


def _last_token_table(batch: SequenceBatch, vocab_size: int, smoothing: float = 1.0) -> FloatArray:
    counts = np.full((vocab_size, vocab_size), smoothing, dtype=np.float64)
    current = batch.tokens[:, :-1].reshape(-1)
    following = batch.tokens[:, 1:].reshape(-1)
    np.add.at(counts, (current, following), 1.0)
    return counts / counts.sum(axis=1, keepdims=True)


def _sampled_nll(candidate: FloatArray, targets: NDArray[np.int64]) -> float:
    probabilities = candidate[np.arange(len(targets)), targets]
    return float(-np.log(np.clip(probabilities, 1e-12, None)).mean())


def predictive_baselines(
    mixture: HMMMixture,
    fit_batch: SequenceBatch,
    test_batch: SequenceBatch,
    window: int = 8,
) -> list[dict]:
    """Evaluate simple predictors on positions with a complete context window."""
    inputs = test_batch.tokens[:, :-1]
    if not 1 <= window <= inputs.shape[1]:
        raise ValueError("window outside available input positions")
    full = mixture.filter(inputs).predictive[:, window - 1 :, :].reshape(-1, mixture.vocab_size)
    targets = test_batch.tokens[:, window:].reshape(-1)
    uniform = np.full_like(full, 1.0 / mixture.vocab_size)
    table = _last_token_table(fit_batch, mixture.vocab_size)
    current = inputs[:, window - 1 :].reshape(-1)
    last_token = table[current]
    windowed = oracle_window_beliefs(test_batch, mixture, window).predictive
    if not (len(full) == len(targets) == len(windowed)):
        raise ValueError("baseline predictions are not position-aligned")

    candidates = {
        "uniform": uniform,
        "last_token": last_token,
        f"window_{window}_bayes": windowed,
        "full_bayes": full,
    }
    uniform_kl = predictive_kl(full, uniform)
    records = []
    for name, prediction in candidates.items():
        kl = predictive_kl(full, prediction)
        records.append(
            {
                "record_type": "baseline",
                "predictor": name,
                "window": window,
                "positions": len(targets),
                "kl_exact": kl,
                "nll": _sampled_nll(prediction, targets),
                "uniform_kl": uniform_kl,
                "competence": competence(kl, uniform_kl),
            }
        )
    return records
