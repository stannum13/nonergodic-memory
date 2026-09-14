"""Linear-subspace activation erasure and causal evaluation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from torch import nn

from .analysis import ActivationTable, ProbeBundle, effective_classifier_directions


FloatArray = NDArray[np.float64]


def orthonormal_basis(directions: FloatArray, tolerance: float = 1e-10) -> FloatArray:
    matrix = np.atleast_2d(np.asarray(directions, dtype=np.float64))
    if matrix.size == 0:
        return np.empty((0, matrix.shape[-1]), dtype=np.float64)
    _, singular_values, right = np.linalg.svd(matrix, full_matrices=False)
    rank = int(np.sum(singular_values > tolerance * max(float(singular_values[0]), 1.0)))
    return right[:rank]


def random_basis(width: int, rank: int, seed: int) -> FloatArray:
    if not 0 <= rank <= width:
        raise ValueError("rank must lie between zero and width")
    if rank == 0:
        return np.empty((0, width), dtype=np.float64)
    random_matrix = np.random.default_rng(seed).normal(size=(rank, width))
    return orthonormal_basis(random_matrix)


def erase_subspace(hidden: FloatArray, basis: FloatArray, center: FloatArray) -> FloatArray:
    centered = hidden - center
    return hidden - (centered @ basis.T) @ basis


def norm_matched_random_erasure(
    hidden: FloatArray,
    random_subspace: FloatArray,
    target_subspace: FloatArray,
    center: FloatArray,
) -> FloatArray:
    centered = hidden - center
    random_delta = (centered @ random_subspace.T) @ random_subspace
    target_delta = (centered @ target_subspace.T) @ target_subspace
    random_norm = np.linalg.norm(random_delta, axis=1, keepdims=True)
    target_norm = np.linalg.norm(target_delta, axis=1, keepdims=True)
    scale = np.divide(target_norm, random_norm, out=np.zeros_like(target_norm), where=random_norm > 1e-15)
    return hidden - random_delta * scale


def probe_basis(bundle: ProbeBundle, target: str) -> FloatArray:
    if target == "component":
        directions = effective_classifier_directions(bundle.component)
    elif target == "state":
        directions = np.concatenate(
            [effective_classifier_directions(probe) for probe in bundle.conditional_state.values()], axis=0
        )
    else:
        raise ValueError(f"unknown intervention target: {target}")
    return orthonormal_basis(directions)


def _log_softmax(logits: FloatArray) -> FloatArray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def evaluate_hidden(
    hidden: FloatArray,
    table: ActivationTable,
    probes: ProbeBundle,
    output_layer: nn.Linear,
) -> dict[str, float]:
    weight = output_layer.weight.detach().numpy().astype(np.float64)
    bias = output_layer.bias.detach().numpy().astype(np.float64)
    logits = hidden @ weight.T + bias
    log_probability = _log_softmax(logits)
    nll = -log_probability[np.arange(len(table.targets)), table.targets].mean()
    kl = (
        table.predictive
        * (np.log(np.clip(table.predictive, 1e-12, None)) - log_probability)
    ).sum(axis=1).mean()
    component_predictions = probes.component.predict(hidden)
    state_predictions = np.empty_like(table.states)
    for component_id, probe in probes.conditional_state.items():
        mask = table.components == component_id
        state_predictions[mask] = probe.predict(hidden[mask])
    return {
        "nll": float(nll),
        "kl_exact": float(kl),
        "component_accuracy": float(np.mean(component_predictions == table.components)),
        "conditional_state_accuracy": float(np.mean(state_predictions == table.states)),
    }


def intervention_record(
    table: ActivationTable,
    probes: ProbeBundle,
    output_layer: nn.Linear,
    altered_hidden: FloatArray,
    baseline: dict[str, float],
) -> dict[str, float]:
    post = evaluate_hidden(altered_hidden, table, probes, output_layer)
    return {
        "nll": post["nll"],
        "delta_nll": post["nll"] - baseline["nll"],
        "kl_exact": post["kl_exact"],
        "delta_kl_exact": post["kl_exact"] - baseline["kl_exact"],
        "component_accuracy": post["component_accuracy"],
        "delta_component_accuracy": post["component_accuracy"] - baseline["component_accuracy"],
        "conditional_state_accuracy": post["conditional_state_accuracy"],
        "delta_conditional_state_accuracy": (
            post["conditional_state_accuracy"] - baseline["conditional_state_accuracy"]
        ),
        "mean_removed_norm": float(np.linalg.norm(table.hidden - altered_hidden, axis=1).mean()),
    }

