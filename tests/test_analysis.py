import numpy as np

from nonergodic_memory.analysis import ActivationTable, fit_probes, pca_records


def separable_table(seed: int, n_sequences: int = 20) -> ActivationTable:
    rng = np.random.default_rng(seed)
    positions = 3
    component = np.repeat(np.arange(n_sequences) % 2, positions)
    state = np.tile(np.arange(n_sequences * positions) % 2, 1)
    hidden = np.column_stack(
        [2 * component - 1, 2 * state - 1, rng.normal(scale=0.05, size=n_sequences * positions)]
    )
    component_posterior = np.eye(2)[component] * 0.9 + 0.05
    state_posterior = np.eye(2)[state] * 0.9 + 0.05
    return ActivationTable(
        hidden=hidden,
        logits=np.zeros((len(hidden), 4)),
        components=component,
        states=state,
        component_posterior=component_posterior,
        state_posterior=state_posterior,
        predictive=np.full((len(hidden), 4), 0.25),
        targets=np.zeros(len(hidden), dtype=int),
        sequence_ids=np.repeat(np.arange(n_sequences), positions),
        positions=np.tile(np.arange(positions), n_sequences),
    )


def test_linear_probes_score_disjoint_heldout_sequences() -> None:
    train = separable_table(1)
    test = separable_table(2)
    result = fit_probes(train, test, seed=4)

    assert result.metrics["component_accuracy"] > 0.95
    assert result.metrics["conditional_state_accuracy"] > 0.95
    assert result.metrics["component_posterior_r2"] > 0.9
    assert result.metrics["state_posterior_r2"] > 0.9
    assert set(train.sequence_ids).isdisjoint(set(test.sequence_ids + 1000))


def test_shuffled_labels_destroy_decoding_signal() -> None:
    train = separable_table(3, n_sequences=80)
    test = separable_table(4, n_sequences=80)
    shuffled = fit_probes(train, test, seed=5, shuffle_labels=True)

    assert shuffled.metrics["component_accuracy"] < 0.7
    assert shuffled.metrics["conditional_state_accuracy"] < 0.7


def test_pca_records_have_two_coordinates_and_variance() -> None:
    records, variance = pca_records(separable_table(6), max_points=17)
    assert len(records) == 17
    assert {"pc1", "pc2", "component", "state", "position"} <= records[0].keys()
    assert len(variance) == 2
    assert 0 < sum(variance) <= 1.0 + 1e-9
