"""Aligned full-prefix and limited-context belief/model comparisons."""

from __future__ import annotations

from dataclasses import fields, replace

import numpy as np
import torch
from numpy.lib.stride_tricks import sliding_window_view
from torch import nn

from .analysis import ActivationTable
from .data.hmm import FilterResult, HMM, HMMMixture, SequenceBatch


def aligned_full_table(table: ActivationTable, window: int) -> ActivationTable:
    """Keep positions at which a complete restart window exists."""
    if not 1 <= window <= int(np.max(table.positions)) + 1:
        raise ValueError("window outside observed position range")
    mask = table.positions >= window - 1
    return ActivationTable(
        **{field.name: getattr(table, field.name)[mask] for field in fields(ActivationTable)}
    )


@torch.no_grad()
def collect_restart_activations(
    model: nn.Module,
    batch: SequenceBatch,
    full_table: ActivationTable,
    window: int,
    batch_windows: int = 512,
) -> ActivationTable:
    """Restart the model on each final `window` observations, retaining full targets."""
    if batch_windows < 1:
        raise ValueError("batch_windows must be positive")
    aligned = aligned_full_table(full_table, window)
    inputs = batch.tokens[:, :-1]
    windows = sliding_window_view(inputs, window_shape=window, axis=1)
    flat = windows.reshape(-1, window).copy()
    if len(flat) != len(aligned.hidden):
        raise ValueError("full activation table does not align with restart windows")
    model.eval()
    logits_parts, hidden_parts = [], []
    for start in range(0, len(flat), batch_windows):
        logits, hidden = model(torch.from_numpy(flat[start : start + batch_windows]))
        logits_parts.append(logits[:, -1].numpy())
        hidden_parts.append(hidden[:, -1].numpy())
    return replace(
        aligned,
        hidden=np.concatenate(hidden_parts).astype(np.float64),
        logits=np.concatenate(logits_parts).astype(np.float64),
    )


def oracle_window_beliefs(
    batch: SequenceBatch,
    mixture: HMMMixture,
    window: int,
) -> FilterResult:
    """Exact Bayes from the last window and its known elapsed start position."""
    inputs = batch.tokens[:, :-1]
    if not 1 <= window <= inputs.shape[1]:
        raise ValueError("window outside observed position range")
    n_sequences = inputs.shape[0]
    n_windows = inputs.shape[1] - window + 1
    n_components = len(mixture.components)
    component = np.empty((n_sequences, n_windows, n_components), dtype=np.float64)
    state = np.empty(
        (n_sequences, n_windows, n_components, mixture.max_states), dtype=np.float64
    )
    predictive = np.empty((n_sequences, n_windows, mixture.vocab_size), dtype=np.float64)
    elapsed_priors = [hmm.initial.copy() for hmm in mixture.components]
    for start in range(n_windows):
        shifted = HMMMixture(
            [
                HMM(hmm.transition, hmm.emission, prior)
                for hmm, prior in zip(mixture.components, elapsed_priors)
            ],
            mixture.weights,
        )
        trace = shifted.filter(inputs[:, start : start + window])
        component[:, start] = trace.component_posterior[:, -1]
        state[:, start] = trace.state_posterior[:, -1]
        predictive[:, start] = trace.predictive[:, -1]
        elapsed_priors = [
            prior @ hmm.transition
            for hmm, prior in zip(mixture.components, elapsed_priors)
        ]
    return FilterResult(
        component_posterior=component.reshape(-1, n_components),
        state_posterior=state.reshape(-1, n_components, mixture.max_states),
        predictive=predictive.reshape(-1, mixture.vocab_size),
    )
