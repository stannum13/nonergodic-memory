import numpy as np
import pytest
import torch

from nonergodic_memory.analysis import collect_activations
from nonergodic_memory.context import (
    aligned_full_table,
    collect_restart_activations,
    oracle_window_beliefs,
)
from nonergodic_memory.data.hmm import make_two_source_mixture
from nonergodic_memory.models.sequence import GRUPredictor


def _sample_full_table():
    mixture = make_two_source_mixture(0.35)
    batch = mixture.sample(4, 10, seed=17)
    model = GRUPredictor(4, width=8)
    full = collect_activations(model, batch, mixture)
    return mixture, batch, model, full


def test_aligned_full_table_keeps_same_targets_and_beliefs() -> None:
    _, _, _, full = _sample_full_table()
    aligned = aligned_full_table(full, window=8)
    mask = full.positions >= 7
    assert len(aligned.hidden) == 8
    np.testing.assert_array_equal(aligned.positions, [7, 8] * 4)
    np.testing.assert_array_equal(aligned.sequence_ids, full.sequence_ids[mask])
    np.testing.assert_array_equal(aligned.targets, full.targets[mask])
    np.testing.assert_allclose(aligned.component_posterior, full.component_posterior[mask])
    np.testing.assert_allclose(aligned.predictive, full.predictive[mask])
    with pytest.raises(ValueError, match="window"):
        aligned_full_table(full, window=0)
    with pytest.raises(ValueError, match="window"):
        aligned_full_table(full, window=10)


def test_restart_activations_are_aligned_and_use_only_last_window() -> None:
    _, batch, model, full = _sample_full_table()
    aligned = aligned_full_table(full, window=8)
    restarted = collect_restart_activations(model, batch, full, window=8, batch_windows=3)
    assert restarted.hidden.shape == aligned.hidden.shape
    np.testing.assert_array_equal(restarted.positions, aligned.positions)
    np.testing.assert_array_equal(restarted.targets, aligned.targets)
    np.testing.assert_allclose(restarted.component_posterior, aligned.component_posterior)
    with torch.no_grad():
        manual_logits, _ = model(torch.from_numpy(batch.tokens[0, :8][None, :]))
    np.testing.assert_allclose(restarted.logits[0], manual_logits[0, -1].numpy(), atol=1e-6)


def test_oracle_window_beliefs_match_first_full_prefix() -> None:
    mixture, batch, _, full = _sample_full_table()
    windowed = oracle_window_beliefs(batch, mixture, window=8)
    aligned = aligned_full_table(full, window=8)
    assert windowed.predictive.shape == (8, 4)
    assert windowed.component_posterior.shape == (8, 2)
    manual = mixture.filter(batch.tokens[0, :8][None, :])
    np.testing.assert_allclose(windowed.predictive[0], manual.predictive[0, -1])
    np.testing.assert_allclose(windowed.predictive[0], aligned.predictive[0])
    np.testing.assert_allclose(windowed.component_posterior[0], aligned.component_posterior[0])
