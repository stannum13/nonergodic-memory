"""Held-out activation extraction, linear probes, regression, and PCA."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

from .data.hmm import HMMMixture, SequenceBatch
from .models.sequence import TransformerPredictor


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class ActivationTable:
    hidden: FloatArray
    logits: FloatArray
    components: IntArray
    states: IntArray
    component_posterior: FloatArray
    state_posterior: FloatArray
    predictive: FloatArray
    targets: IntArray
    sequence_ids: IntArray
    positions: IntArray
    joint_belief: FloatArray | None = None


@dataclass(frozen=True)
class ProbeBundle:
    component: object
    conditional_state: dict[int, object]
    component_regression: object
    state_regression: object
    metrics: dict[str, float]
    joint_regression: object | None = None


@torch.no_grad()
def collect_activations(
    model: nn.Module,
    batch: SequenceBatch,
    mixture: HMMMixture,
    sequence_offset: int = 0,
) -> ActivationTable:
    model.eval()
    inputs = torch.from_numpy(batch.tokens[:, :-1])
    logits, hidden = model(inputs)
    return _make_activation_table(hidden, logits, batch, mixture, sequence_offset)


def _make_activation_table(
    hidden: torch.Tensor,
    logits: torch.Tensor,
    batch: SequenceBatch,
    mixture: HMMMixture,
    sequence_offset: int,
) -> ActivationTable:
    inputs = batch.tokens[:, :-1]
    n_sequences, positions = inputs.shape
    exact = mixture.filter(batch.tokens[:, :-1])
    joint_belief = (
        exact.component_posterior[:, :, :, None] * exact.state_posterior
    ).reshape(-1, len(mixture.components) * mixture.max_states)
    component_grid = np.repeat(batch.components[:, None], positions, axis=1)
    state_grid = batch.states[:, :-1]
    return ActivationTable(
        hidden=hidden.numpy().reshape(-1, hidden.shape[-1]).astype(np.float64),
        logits=logits.numpy().reshape(-1, logits.shape[-1]).astype(np.float64),
        components=component_grid.reshape(-1).astype(np.int64),
        states=state_grid.reshape(-1).astype(np.int64),
        component_posterior=exact.component_posterior.reshape(-1, len(mixture.components)),
        state_posterior=exact.state_posterior.reshape(
            -1, len(mixture.components) * mixture.max_states
        ),
        joint_belief=joint_belief,
        predictive=exact.predictive.reshape(-1, mixture.vocab_size),
        targets=batch.tokens[:, 1:].reshape(-1).astype(np.int64),
        sequence_ids=np.repeat(np.arange(n_sequences) + sequence_offset, positions).astype(np.int64),
        positions=np.tile(np.arange(positions), n_sequences).astype(np.int64),
    )


@torch.no_grad()
def collect_transformer_depth_activations(
    model: TransformerPredictor,
    batch: SequenceBatch,
    mixture: HMMMixture,
    depth: int,
    sequence_offset: int = 0,
) -> ActivationTable:
    model.eval()
    inputs = torch.from_numpy(batch.tokens[:, :-1])
    logits, activations = model.forward_with_layers(inputs)
    if not 0 <= depth < len(activations):
        raise ValueError("depth outside captured activation range")
    return _make_activation_table(
        activations[depth], logits, batch, mixture, sequence_offset
    )


def _classifier() -> object:
    return make_pipeline(
        StandardScaler(), LogisticRegression(C=1.0, max_iter=1000, random_state=0)
    )


def _regressor() -> object:
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0))


def pairwise_distance_r2(
    actual: FloatArray,
    predicted: FloatArray,
    seed: int,
    max_pairs: int = 20_000,
) -> float:
    """Score predicted distances using a deterministic sample of unordered pairs."""
    if actual.ndim != 2 or actual.shape != predicted.shape:
        raise ValueError("actual and predicted must have the same two-dimensional shape")
    n_points = len(actual)
    if n_points < 3:
        raise ValueError("at least three coordinates are required")
    if max_pairs < 2:
        raise ValueError("at least two pairs are required")

    total_pairs = n_points * (n_points - 1) // 2
    n_pairs = min(total_pairs, max_pairs)
    if total_pairs > max_pairs:
        pair_indices = np.random.default_rng(seed).choice(total_pairs, size=n_pairs, replace=False)
    else:
        pair_indices = np.arange(n_pairs)

    # Pair indices enumerate rows of the upper triangle: (0, 1), (0, 2), ... .
    diagonal = 2 * n_points - 1
    first = ((diagonal - np.sqrt(diagonal**2 - 8 * pair_indices)) // 2).astype(np.int64)
    starts = first * (2 * n_points - first - 1) // 2
    first -= starts > pair_indices
    starts = first * (2 * n_points - first - 1) // 2
    second = first + 1 + pair_indices - starts

    actual_distances = np.linalg.norm(actual[first] - actual[second], axis=1)
    predicted_distances = np.linalg.norm(predicted[first] - predicted[second], axis=1)
    return float(r2_score(actual_distances, predicted_distances))


def fit_probes(
    train: ActivationTable,
    test: ActivationTable,
    seed: int,
    shuffle_labels: bool = False,
) -> ProbeBundle:
    overlap = np.intersect1d(np.unique(train.sequence_ids), np.unique(test.sequence_ids))
    if overlap.size:
        raise ValueError("probe train and test sequence IDs overlap")
    rng = np.random.default_rng(seed)
    component_labels = train.components.copy()
    state_labels = train.states.copy()
    if shuffle_labels:
        component_labels = rng.permutation(component_labels)
        state_labels = rng.permutation(state_labels)

    component_probe = _classifier()
    component_probe.fit(train.hidden, component_labels)
    component_accuracy = accuracy_score(test.components, component_probe.predict(test.hidden))

    state_probes: dict[int, object] = {}
    state_predictions = np.empty_like(test.states)
    for component_id in np.unique(train.components):
        train_mask = train.components == component_id
        test_mask = test.components == component_id
        probe = _classifier()
        probe.fit(train.hidden[train_mask], state_labels[train_mask])
        state_probes[int(component_id)] = probe
        state_predictions[test_mask] = probe.predict(test.hidden[test_mask])

    component_regression = _regressor()
    component_targets = train.component_posterior.copy()
    state_targets = train.state_posterior.copy()
    if train.joint_belief is None or test.joint_belief is None:
        raise ValueError("joint belief targets are required for regression")
    joint_targets = train.joint_belief.copy()
    if shuffle_labels:
        component_targets = component_targets[rng.permutation(len(component_targets))]
        state_targets = state_targets[rng.permutation(len(state_targets))]
        joint_targets = joint_targets[rng.permutation(len(joint_targets))]
    component_regression.fit(train.hidden, component_targets)
    state_regression = _regressor()
    state_regression.fit(train.hidden, state_targets)
    joint_regression = _regressor()
    joint_regression.fit(train.hidden, joint_targets)

    joint_prediction = joint_regression.predict(test.hidden)
    joint_mse = float(np.mean((test.joint_belief - joint_prediction) ** 2))
    joint_r2 = float(r2_score(test.joint_belief, joint_prediction))

    metrics = {
        "component_accuracy": float(component_accuracy),
        "conditional_state_accuracy": float(accuracy_score(test.states, state_predictions)),
        "component_posterior_r2": float(
            r2_score(test.component_posterior, component_regression.predict(test.hidden))
        ),
        "state_posterior_r2": float(r2_score(test.state_posterior, state_regression.predict(test.hidden))),
        "joint_belief_r2": joint_r2,
        "joint_belief_mse": joint_mse,
        "joint_distance_r2": pairwise_distance_r2(test.joint_belief, joint_prediction, seed),
    }
    return ProbeBundle(
        component_probe,
        state_probes,
        component_regression,
        state_regression,
        metrics,
        joint_regression,
    )


def pca_records(table: ActivationTable, max_points: int = 500) -> tuple[list[dict], list[float]]:
    pca = PCA(n_components=2)
    coordinates = pca.fit_transform(table.hidden)
    if len(table.hidden) > max_points:
        selected = np.linspace(0, len(table.hidden) - 1, max_points, dtype=int)
    else:
        selected = np.arange(len(table.hidden))
    records = [
        {
            "pc1": float(coordinates[index, 0]),
            "pc2": float(coordinates[index, 1]),
            "component": int(table.components[index]),
            "state": int(table.states[index]),
            "position": int(table.positions[index]),
        }
        for index in selected
    ]
    return records, [float(value) for value in pca.explained_variance_ratio_]


def effective_classifier_directions(probe: object) -> FloatArray:
    """Return classifier coefficient rows in the original activation coordinates."""
    scaler = probe.named_steps["standardscaler"]
    classifier = probe.named_steps["logisticregression"]
    return np.asarray(classifier.coef_ / scaler.scale_[None, :], dtype=np.float64)
