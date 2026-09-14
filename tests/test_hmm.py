import numpy as np
import pytest

from nonergodic_memory.data.hmm import HMM, HMMMixture, make_two_source_mixture


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
