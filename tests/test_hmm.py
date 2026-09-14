import numpy as np
import pytest
from itertools import product

from nonergodic_memory.data.hmm import HMM, HMMMixture, make_source_mixture, make_two_source_mixture


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
