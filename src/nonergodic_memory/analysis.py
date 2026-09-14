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


@dataclass(frozen=True)
class ProbeBundle:
    component: object
    conditional_state: dict[int, object]
    component_regression: object
    state_regression: object
    metrics: dict[str, float]


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
    n_sequences, positions = inputs.shape
    exact = mixture.filter(batch.tokens[:, :-1])
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
        predictive=exact.predictive.reshape(-1, mixture.vocab_size),
        targets=batch.tokens[:, 1:].reshape(-1).astype(np.int64),
        sequence_ids=np.repeat(np.arange(n_sequences) + sequence_offset, positions).astype(np.int64),
        positions=np.tile(np.arange(positions), n_sequences).astype(np.int64),
    )


def _classifier() -> object:
    return make_pipeline(
        StandardScaler(), LogisticRegression(C=1.0, max_iter=1000, random_state=0)
    )


def _regressor() -> object:
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0))


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
    if shuffle_labels:
        component_targets = component_targets[rng.permutation(len(component_targets))]
        state_targets = state_targets[rng.permutation(len(state_targets))]
    component_regression.fit(train.hidden, component_targets)
    state_regression = _regressor()
    state_regression.fit(train.hidden, state_targets)

    metrics = {
        "component_accuracy": float(component_accuracy),
        "conditional_state_accuracy": float(accuracy_score(test.states, state_predictions)),
        "component_posterior_r2": float(
            r2_score(test.component_posterior, component_regression.predict(test.hidden))
        ),
        "state_posterior_r2": float(r2_score(test.state_posterior, state_regression.predict(test.hidden))),
    }
    return ProbeBundle(
        component_probe,
        state_probes,
        component_regression,
        state_regression,
        metrics,
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
