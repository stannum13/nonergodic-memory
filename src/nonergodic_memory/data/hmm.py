"""Finite HMM mixtures with exact Bayesian filtering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _probabilities(value: ArrayLike, name: str, axis: int = -1) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if np.any(array < 0) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite nonnegative probabilities")
    if not np.allclose(array.sum(axis=axis), 1.0):
        raise ValueError(f"{name} must sum to one along axis {axis}")
    return array


@dataclass(frozen=True)
class HMM:
    """A discrete HMM using row-stochastic transition and emission matrices."""

    transition: FloatArray
    emission: FloatArray
    initial: FloatArray

    def __init__(self, transition: ArrayLike, emission: ArrayLike, initial: ArrayLike):
        transition_array = _probabilities(transition, "transition")
        emission_array = _probabilities(emission, "emission")
        initial_array = _probabilities(initial, "initial")
        states = initial_array.shape[0]
        if transition_array.shape != (states, states):
            raise ValueError("transition must be square and match initial")
        if emission_array.ndim != 2 or emission_array.shape[0] != states:
            raise ValueError("emission rows must match the number of states")
        object.__setattr__(self, "transition", transition_array)
        object.__setattr__(self, "emission", emission_array)
        object.__setattr__(self, "initial", initial_array)

    @property
    def n_states(self) -> int:
        return int(self.initial.shape[0])

    @property
    def vocab_size(self) -> int:
        return int(self.emission.shape[1])


@dataclass(frozen=True)
class SequenceBatch:
    tokens: IntArray
    components: IntArray
    states: IntArray


@dataclass(frozen=True)
class FilterResult:
    component_posterior: FloatArray
    state_posterior: FloatArray
    predictive: FloatArray


class HMMMixture:
    """A nonergodic source: choose one HMM once, then emit a whole sequence."""

    def __init__(self, components: Sequence[HMM], weights: ArrayLike):
        if not components:
            raise ValueError("components cannot be empty")
        vocab_sizes = {component.vocab_size for component in components}
        if len(vocab_sizes) != 1:
            raise ValueError("all components must share a vocabulary")
        self.components = tuple(components)
        self.weights = _probabilities(weights, "weights")
        if self.weights.shape != (len(self.components),):
            raise ValueError("one mixture weight is required per component")

    @property
    def vocab_size(self) -> int:
        return self.components[0].vocab_size

    @property
    def max_states(self) -> int:
        return max(component.n_states for component in self.components)

    def sample(self, n_sequences: int, length: int, seed: int) -> SequenceBatch:
        if n_sequences < 1 or length < 2:
            raise ValueError("n_sequences must be positive and length must be at least two")
        rng = np.random.default_rng(seed)
        component_ids = rng.choice(len(self.components), size=n_sequences, p=self.weights).astype(np.int64)
        states = np.empty((n_sequences, length), dtype=np.int64)
        tokens = np.empty((n_sequences, length), dtype=np.int64)
        for sequence_index, component_id in enumerate(component_ids):
            hmm = self.components[int(component_id)]
            states[sequence_index, 0] = rng.choice(hmm.n_states, p=hmm.initial)
            tokens[sequence_index, 0] = rng.choice(
                self.vocab_size, p=hmm.emission[states[sequence_index, 0]]
            )
            for position in range(1, length):
                states[sequence_index, position] = rng.choice(
                    hmm.n_states, p=hmm.transition[states[sequence_index, position - 1]]
                )
                tokens[sequence_index, position] = rng.choice(
                    self.vocab_size, p=hmm.emission[states[sequence_index, position]]
                )
        return SequenceBatch(tokens=tokens, components=component_ids, states=states)

    def filter(self, tokens: ArrayLike) -> FilterResult:
        observations = np.asarray(tokens, dtype=np.int64)
        if observations.ndim != 2:
            raise ValueError("tokens must have shape [sequences, positions]")
        if np.any(observations < 0) or np.any(observations >= self.vocab_size):
            raise ValueError("token outside vocabulary")
        n_sequences, length = observations.shape
        n_components = len(self.components)
        component_posterior = np.zeros((n_sequences, length, n_components), dtype=np.float64)
        state_posterior = np.zeros(
            (n_sequences, length, n_components, self.max_states), dtype=np.float64
        )
        predictive = np.zeros((n_sequences, length, self.vocab_size), dtype=np.float64)

        for sequence_index in range(n_sequences):
            conditional_states = [hmm.initial.copy() for hmm in self.components]
            log_component_weights = np.log(self.weights)
            for position, token in enumerate(observations[sequence_index]):
                if position:
                    conditional_states = [
                        state @ hmm.transition
                        for state, hmm in zip(conditional_states, self.components)
                    ]
                for component_index, hmm in enumerate(self.components):
                    emitted = conditional_states[component_index] * hmm.emission[:, token]
                    likelihood = float(emitted.sum())
                    if likelihood > 0:
                        conditional_states[component_index] = emitted / likelihood
                        log_component_weights[component_index] += np.log(likelihood)
                    else:
                        log_component_weights[component_index] = -np.inf
                maximum = float(np.max(log_component_weights))
                if not np.isfinite(maximum):
                    raise ValueError("observed sequence has zero probability under the mixture")
                relative_weights = np.exp(log_component_weights - maximum)
                posterior_weights = relative_weights / relative_weights.sum()
                component_posterior[sequence_index, position] = posterior_weights
                for component_index, (state, hmm) in enumerate(
                    zip(conditional_states, self.components)
                ):
                    state_posterior[
                        sequence_index, position, component_index, : hmm.n_states
                    ] = state
                    next_state = state @ hmm.transition
                    predictive[sequence_index, position] += (
                        posterior_weights[component_index] * (next_state @ hmm.emission)
                    )

        return FilterResult(component_posterior, state_posterior, predictive)


def make_source_mixture(n_components: int = 2, overlap: float = 0.35) -> HMMMixture:
    """Create 2–4 two-state sources; overlap 1 makes all emissions identical."""
    if not 0.0 <= overlap <= 1.0:
        raise ValueError("overlap must lie in [0, 1]")
    if not 2 <= n_components <= 4:
        raise ValueError("components must lie in [2, 4]")
    transition = np.array([[0.88, 0.12], [0.18, 0.82]])
    initial = np.array([0.55, 0.45])
    first = np.array([[0.72, 0.18, 0.06, 0.04], [0.10, 0.68, 0.12, 0.10]])
    distinct = np.array([[0.05, 0.05, 0.72, 0.18], [0.12, 0.08, 0.10, 0.70]])
    templates = [first, distinct, first[:, [1, 2, 3, 0]], first[:, [3, 0, 1, 2]]]
    emissions = [first] + [
        overlap * first + (1.0 - overlap) * templates[index]
        for index in range(1, n_components)
    ]
    return HMMMixture(
        [HMM(transition, emission, initial) for emission in emissions],
        np.full(n_components, 1.0 / n_components),
    )


def make_two_source_mixture(overlap: float = 0.35) -> HMMMixture:
    """Backward-compatible constructor for the central two-source experiment."""
    return make_source_mixture(n_components=2, overlap=overlap)
