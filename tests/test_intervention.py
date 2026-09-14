import numpy as np
import torch
from torch import nn

from nonergodic_memory.analysis import ActivationTable, ProbeBundle
from nonergodic_memory.intervention import (
    erase_subspace,
    evaluate_hidden,
    evaluate_hidden_with_logits,
    norm_matched_random_erasure,
    orthonormal_basis,
    random_basis,
)


class _ThresholdProbe:
    def predict(self, hidden: np.ndarray) -> np.ndarray:
        return (hidden[:, 0] > 0).astype(np.int64)


def test_erasure_removes_target_direction() -> None:
    hidden = np.array([[1.0, 2.0, 3.0], [-2.0, 4.0, 1.0]])
    basis = orthonormal_basis(np.array([[1.0, 0.0, 0.0]]))
    erased = erase_subspace(hidden, basis, center=np.zeros(3))
    np.testing.assert_allclose(erased @ basis.T, 0.0, atol=1e-12)


def test_erasure_preserves_orthogonal_coordinates() -> None:
    hidden = np.array([[1.0, 2.0, 3.0]])
    basis = orthonormal_basis(np.array([[1.0, 0.0, 0.0]]))
    erased = erase_subspace(hidden, basis, center=np.zeros(3))
    np.testing.assert_allclose(erased[0, 1:], hidden[0, 1:])


def test_random_basis_matches_requested_rank() -> None:
    basis = random_basis(width=9, rank=3, seed=8)
    assert basis.shape == (3, 9)
    np.testing.assert_allclose(basis @ basis.T, np.eye(3), atol=1e-12)


def test_norm_matched_control_matches_per_example_damage() -> None:
    rng = np.random.default_rng(1)
    hidden = rng.normal(size=(20, 6))
    target = random_basis(6, 2, seed=2)
    control = random_basis(6, 2, seed=3)
    center = hidden.mean(0)
    learned = erase_subspace(hidden, target, center)
    matched = norm_matched_random_erasure(hidden, control, target, center)
    np.testing.assert_allclose(
        np.linalg.norm(hidden - matched, axis=1),
        np.linalg.norm(hidden - learned, axis=1),
        atol=1e-10,
    )


def test_logits_aware_evaluation_matches_linear_output_evaluation() -> None:
    hidden = np.array([[1.0, -0.5], [-1.0, 0.5]], dtype=np.float64)
    layer = nn.Linear(2, 2)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[0.7, -0.2], [-0.3, 0.4]]))
        layer.bias.copy_(torch.tensor([0.1, -0.1]))
    logits = hidden @ layer.weight.detach().numpy().T + layer.bias.detach().numpy()
    table = ActivationTable(
        hidden=hidden,
        logits=logits,
        components=np.array([1, 0]),
        states=np.array([1, 0]),
        component_posterior=np.array([[0.2, 0.8], [0.7, 0.3]]),
        state_posterior=np.zeros((2, 2)),
        predictive=np.array([[0.4, 0.6], [0.8, 0.2]]),
        targets=np.array([1, 0]),
        sequence_ids=np.array([0, 1]),
        positions=np.array([0, 0]),
    )
    probe = _ThresholdProbe()
    bundle = ProbeBundle(probe, {0: probe, 1: probe}, object(), object(), {})
    direct = evaluate_hidden(hidden, table, bundle, layer)
    propagated = evaluate_hidden_with_logits(hidden, logits, table, bundle)
    assert direct == propagated
