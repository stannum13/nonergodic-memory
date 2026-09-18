import numpy as np
import pytest

from nonergodic_memory.analysis import (
    ActivationTable,
    collect_activations,
    collect_transformer_depth_activations,
    fit_probes,
    pairwise_distance_r2,
    pca_records,
)
from nonergodic_memory.data.hmm import make_mess3_mixture, make_two_source_mixture
from nonergodic_memory.models.sequence import GRUPredictor, TransformerPredictor


def separable_table(seed: int, n_sequences: int = 20, sequence_offset: int = 0) -> ActivationTable:
    rng = np.random.default_rng(seed)
    positions = 3
    component = np.repeat(np.arange(n_sequences) % 2, positions)
    state = np.tile(np.arange(n_sequences * positions) % 2, 1)
    hidden = np.column_stack(
        [2 * component - 1, 2 * state - 1, rng.normal(scale=0.05, size=n_sequences * positions)]
    )
    component_posterior = np.eye(2)[component] * 0.9 + 0.05
    state_one = np.eye(2)[state] * 0.9 + 0.05
    state_posterior = np.concatenate([state_one, state_one], axis=1)
    joint_belief = (component_posterior[:, :, None] * state_posterior.reshape(-1, 2, 2)).reshape(
        -1, 4
    )
    return ActivationTable(
        hidden=hidden,
        logits=np.zeros((len(hidden), 4)),
        components=component,
        states=state,
        component_posterior=component_posterior,
        state_posterior=state_posterior,
        joint_belief=joint_belief,
        predictive=np.full((len(hidden), 4), 0.25),
        targets=np.zeros(len(hidden), dtype=int),
        sequence_ids=np.repeat(np.arange(n_sequences) + sequence_offset, positions),
        positions=np.tile(np.arange(positions), n_sequences),
    )


def test_linear_probes_score_disjoint_heldout_sequences() -> None:
    train = separable_table(1)
    test = separable_table(2, sequence_offset=1000)
    result = fit_probes(train, test, seed=4)

    assert result.metrics["component_accuracy"] > 0.95
    assert result.metrics["conditional_state_accuracy"] > 0.95
    assert result.metrics["component_posterior_r2"] > 0.9
    assert result.metrics["state_posterior_r2"] > 0.9
    assert result.joint_regression is not None
    assert {"joint_belief_r2", "joint_belief_mse", "joint_distance_r2"} <= result.metrics.keys()
    assert train.state_posterior.shape[1] == 4


def test_shuffled_labels_destroy_decoding_signal() -> None:
    train = separable_table(3, n_sequences=80)
    test = separable_table(4, n_sequences=80, sequence_offset=1000)
    shuffled = fit_probes(train, test, seed=5, shuffle_labels=True)

    assert shuffled.metrics["component_accuracy"] < 0.7
    assert shuffled.metrics["conditional_state_accuracy"] < 0.7


def test_pca_records_have_two_coordinates_and_variance() -> None:
    records, variance = pca_records(separable_table(6), max_points=17)
    assert len(records) == 17
    assert {"pc1", "pc2", "component", "state", "position"} <= records[0].keys()
    assert len(variance) == 2
    assert 0 < sum(variance) <= 1.0 + 1e-9


def test_probe_rejects_overlapping_sequence_ids() -> None:
    train = separable_table(7)
    test = separable_table(8)
    with pytest.raises(ValueError, match="overlap"):
        fit_probes(train, test, seed=9)


def test_collect_transformer_depth_activations_uses_requested_layer() -> None:
    mixture = make_two_source_mixture(0.35)
    batch = mixture.sample(5, 7, seed=12)
    model = TransformerPredictor(4, width=8, layers=2, heads=2, max_length=8)
    first = collect_transformer_depth_activations(model, batch, mixture, depth=0)
    normalized = collect_transformer_depth_activations(model, batch, mixture, depth=2)
    assert first.hidden.shape == normalized.hidden.shape == (30, 8)
    assert not np.allclose(first.hidden, normalized.hidden)


def test_activation_table_joint_belief_is_weighted_conditional_state() -> None:
    mixture = make_mess3_mixture()
    batch = mixture.sample(4, 7, seed=10)
    model = GRUPredictor(vocab_size=3, width=8)
    table = collect_activations(model, batch, mixture)

    expected = (
        table.component_posterior[:, :, None]
        * table.state_posterior.reshape(-1, 2, 3)
    ).reshape(-1, 6)
    np.testing.assert_allclose(table.joint_belief, expected)
    np.testing.assert_allclose(table.joint_belief.sum(axis=1), 1.0)


def test_pairwise_distance_r2_is_one_for_identical_coordinates() -> None:
    coordinates = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]])
    assert pairwise_distance_r2(coordinates, coordinates, seed=11) == pytest.approx(1.0)


def test_pairwise_distance_r2_is_nonpositive_for_constant_predictions() -> None:
    actual = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]])
    predicted = np.zeros_like(actual)
    assert pairwise_distance_r2(actual, predicted, seed=12) <= 0.0


def test_pairwise_distance_r2_is_deterministic_for_a_seed() -> None:
    actual = np.arange(630, dtype=float).reshape(210, 3)
    predicted = actual + np.linspace(0.0, 1.0, actual.size).reshape(actual.shape)
    first = pairwise_distance_r2(actual, predicted, seed=13)
    second = pairwise_distance_r2(actual, predicted, seed=13)
    assert first == second


@pytest.mark.parametrize(
    ("coordinates", "max_pairs", "message"),
    [
        (np.array([[0.0, 0.0], [1.0, 0.0]]), 20_000, "three coordinates"),
        (np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]]), 1, "at least two pairs"),
    ],
)
def test_pairwise_distance_r2_rejects_undefined_sample_sizes(
    coordinates: np.ndarray, max_pairs: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        pairwise_distance_r2(coordinates, coordinates, seed=14, max_pairs=max_pairs)
