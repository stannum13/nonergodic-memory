import numpy as np
import pytest
from itertools import product

from nonergodic_memory.data import (
    HMM,
    HMMMixture,
    make_mess3,
    make_mess3_mixture,
    make_source_mixture,
    make_two_source_mixture,
)
from nonergodic_memory.data.hmm import _sample_categorical_rows


class _FixedDraws:
    def __init__(self, draws: list[float]):
        self.draws = np.asarray(draws, dtype=np.float64)

    def random(self, shape: tuple[int, int]) -> np.ndarray:
        assert shape == (len(self.draws), 1)
        return self.draws[:, None]


def test_categorical_rows_use_right_sided_inverse_cdf_at_boundaries() -> None:
    probabilities = np.array(
        [[0.0, 0.5, 0.5], [0.25, 0.25, 0.5], [0.0, 0.0, 1.0]]
    )
    sampled = _sample_categorical_rows(
        probabilities, _FixedDraws([0.0, 0.5, np.nextafter(1.0, 0.0)])
    )

    np.testing.assert_array_equal(sampled, [1, 2, 2])


def test_mess3_factorization_matches_published_labeled_operators() -> None:
    hmm = make_mess3(alpha=0.6, x=0.15)
    labeled = np.stack([hmm.transition * hmm.emission[:, token][None, :] for token in range(3)])
    expected_a = np.array([[.42, .03, .03], [.09, .14, .03], [.09, .03, .14]])
    expected_b = np.array([[.14, .09, .03], [.03, .42, .03], [.03, .09, .14]])
    expected_c = np.array([[.14, .03, .09], [.03, .14, .09], [.03, .03, .42]])
    np.testing.assert_allclose(labeled, np.stack([expected_a, expected_b, expected_c]))
    np.testing.assert_allclose(hmm.initial, np.full(3, 1 / 3))


def test_second_mess3_factorization_matches_published_labeled_operators() -> None:
    hmm = make_mess3(alpha=0.66, x=0.50)
    labeled = np.stack([hmm.transition * hmm.emission[:, token][None, :] for token in range(3)])
    expected_a = np.array([[0.0, .085, .085], [.33, 0.0, .085], [.33, .085, 0.0]])
    expected_b = np.array([[0.0, .33, .085], [.085, 0.0, .085], [.085, .33, 0.0]])
    expected_c = np.array([[0.0, .085, .33], [.085, 0.0, .33], [.085, .085, 0.0]])
    np.testing.assert_allclose(labeled, np.stack([expected_a, expected_b, expected_c]))


@pytest.mark.parametrize("alpha, x", [(0.60, 0.15), (0.66, 0.50)])
def test_mess3_filter_matches_published_operators_for_a_short_word(alpha: float, x: float) -> None:
    hmm = make_mess3(alpha=alpha, x=x)
    mixture = HMMMixture([hmm], [1.0])
    word = np.array([[0, 1, 2, 1]], dtype=np.int64)
    filtered = mixture.filter(word)
    operators = np.stack(
        [hmm.transition * hmm.emission[:, token][None, :] for token in range(hmm.vocab_size)]
    )

    state = hmm.initial.copy()
    for position, token in enumerate(word[0]):
        state = state @ operators[token]
        state /= state.sum()
        np.testing.assert_allclose(filtered.state_posterior[0, position, 0], state)


def test_published_mess3_mixture_has_two_three_state_components() -> None:
    mixture = make_mess3_mixture()
    assert mixture.vocab_size == 3
    assert [component.n_states for component in mixture.components] == [3, 3]
    np.testing.assert_allclose(mixture.weights, [0.5, 0.5])


def test_one_state_mixture_matches_bayes_rule() -> None:
    mixture = HMMMixture(
        [
            HMM([[1.0]], [[0.8, 0.2]], [1.0]),
            HMM([[1.0]], [[0.2, 0.8]], [1.0]),
        ],
        [0.5, 0.5],
    )
    result = mixture.filter(np.array([[0, 0]], dtype=np.int64))

    np.testing.assert_allclose(result.component_posterior[0, 0], [0.8, 0.2])
    np.testing.assert_allclose(result.component_posterior[0, 1], [16 / 17, 1 / 17])
    np.testing.assert_allclose(result.predictive[0, 0], [0.68, 0.32])


def test_filter_outputs_are_normalized_at_every_position() -> None:
    mixture = make_two_source_mixture(overlap=0.35)
    batch = mixture.sample(n_sequences=7, length=9, seed=4)
    result = mixture.filter(batch.tokens)

    assert result.component_posterior.shape == (7, 9, 2)
    assert result.state_posterior.shape == (7, 9, 2, 2)
    assert result.predictive.shape == (7, 9, 4)
    np.testing.assert_allclose(result.component_posterior.sum(-1), 1.0, atol=1e-10)
    np.testing.assert_allclose(result.state_posterior.sum(-1), 1.0, atol=1e-10)
    np.testing.assert_allclose(result.predictive.sum(-1), 1.0, atol=1e-10)


def test_sampling_is_reproducible_and_records_latents() -> None:
    mixture = make_two_source_mixture(overlap=0.5)
    left = mixture.sample(5, 8, seed=11)
    right = mixture.sample(5, 8, seed=11)

    np.testing.assert_array_equal(left.tokens, right.tokens)
    np.testing.assert_array_equal(left.components, right.components)
    np.testing.assert_array_equal(left.states, right.states)
    assert left.states.shape == left.tokens.shape


def test_vectorized_sampling_is_reproducible_and_records_latents() -> None:
    mixture = make_two_source_mixture(overlap=0.5)
    left = mixture.sample_vectorized(50, 8, seed=11)
    right = mixture.sample_vectorized(50, 8, seed=11)

    np.testing.assert_array_equal(left.tokens, right.tokens)
    np.testing.assert_array_equal(left.components, right.components)
    np.testing.assert_array_equal(left.states, right.states)
    assert left.states.shape == left.tokens.shape == (50, 8)
    assert set(left.components) <= {0, 1}
    assert np.all((left.tokens >= 0) & (left.tokens < mixture.vocab_size))


def test_vectorized_sampling_matches_defining_probabilities() -> None:
    mixture = HMMMixture(
        [
            HMM([[0.8, 0.2], [0.3, 0.7]], [[0.9, 0.1], [0.25, 0.75]], [0.6, 0.4]),
            HMM([[0.4, 0.6], [0.1, 0.9]], [[0.2, 0.8], [0.7, 0.3]], [0.25, 0.75]),
        ],
        [0.35, 0.65],
    )
    batch = mixture.sample_vectorized(100_000, 5, seed=91)

    component_frequency = np.bincount(batch.components, minlength=2) / len(batch.components)
    np.testing.assert_allclose(component_frequency, mixture.weights, atol=0.006)
    for component_id, hmm in enumerate(mixture.components):
        selected = batch.components == component_id
        initial_frequency = np.bincount(batch.states[selected, 0], minlength=2) / selected.sum()
        np.testing.assert_allclose(initial_frequency, hmm.initial, atol=0.012)
        for state in range(hmm.n_states):
            emission_mask = selected[:, None] & (batch.states == state)
            emission_frequency = np.bincount(
                batch.tokens[emission_mask], minlength=mixture.vocab_size
            ) / emission_mask.sum()
            np.testing.assert_allclose(emission_frequency, hmm.emission[state], atol=0.012)
            transition_mask = selected[:, None] & (batch.states[:, :-1] == state)
            transition_frequency = np.bincount(
                batch.states[:, 1:][transition_mask], minlength=hmm.n_states
            ) / transition_mask.sum()
            np.testing.assert_allclose(transition_frequency, hmm.transition[state], atol=0.012)


def test_vectorized_sampling_rejects_invalid_sizes() -> None:
    mixture = make_mess3_mixture()
    with pytest.raises(ValueError, match="positive"):
        mixture.sample_vectorized(0, 8, seed=1)
    with pytest.raises(ValueError, match="at least two"):
        mixture.sample_vectorized(2, 1, seed=1)


def test_invalid_overlap_is_rejected() -> None:
    with pytest.raises(ValueError, match="overlap"):
        make_two_source_mixture(overlap=1.1)


def test_multistate_filter_matches_brute_force_enumeration() -> None:
    mixture = HMMMixture(
        [
            HMM([[0.7, 0.3], [0.2, 0.8]], [[0.9, 0.1], [0.3, 0.7]], [0.6, 0.4]),
            HMM([[0.4, 0.6], [0.5, 0.5]], [[0.2, 0.8], [0.75, 0.25]], [0.3, 0.7]),
        ],
        [0.4, 0.6],
    )
    tokens = np.array([[0, 1, 0]])
    actual = mixture.filter(tokens)
    for position in range(tokens.shape[1]):
        joint = np.zeros((2, 2))
        next_token = np.zeros(2)
        total = 0.0
        for component_id, hmm in enumerate(mixture.components):
            for path in product(range(2), repeat=position + 1):
                probability = mixture.weights[component_id] * hmm.initial[path[0]]
                probability *= hmm.emission[path[0], tokens[0, 0]]
                for t in range(1, position + 1):
                    probability *= hmm.transition[path[t - 1], path[t]]
                    probability *= hmm.emission[path[t], tokens[0, t]]
                total += probability
                joint[component_id, path[-1]] += probability
                next_token += probability * (hmm.transition[path[-1]] @ hmm.emission)
        joint /= total
        np.testing.assert_allclose(actual.component_posterior[0, position], joint.sum(1))
        np.testing.assert_allclose(actual.state_posterior[0, position], joint / joint.sum(1)[:, None])
        np.testing.assert_allclose(actual.predictive[0, position], next_token / total)


def test_log_domain_component_weights_recover_after_extreme_evidence_shift() -> None:
    mixture = HMMMixture(
        [
            HMM([[1.0]], [[0.9, 0.1]], [1.0]),
            HMM([[1.0]], [[0.1, 0.9]], [1.0]),
        ],
        [0.5, 0.5],
    )
    tokens = np.array([[0] * 5000 + [1] * 5000], dtype=np.int64)
    result = mixture.filter(tokens)
    np.testing.assert_allclose(result.component_posterior[0, -1], [0.5, 0.5], atol=1e-9)


def test_general_mixture_supports_two_to_four_components() -> None:
    for n_components in (2, 3, 4):
        mixture = make_source_mixture(n_components=n_components, overlap=0.35)
        batch = mixture.sample(40, 10, seed=20 + n_components)
        exact = mixture.filter(batch.tokens)
        assert len(mixture.components) == n_components
        assert exact.component_posterior.shape == (40, 10, n_components)
        assert set(batch.components) <= set(range(n_components))
        np.testing.assert_allclose(exact.component_posterior.sum(-1), 1.0)


def test_general_mixture_rejects_unsupported_component_count() -> None:
    with pytest.raises(ValueError, match="components"):
        make_source_mixture(n_components=5, overlap=0.35)
