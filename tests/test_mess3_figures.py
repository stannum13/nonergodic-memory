import json
from pathlib import Path

import numpy as np
import pytest

from nonergodic_memory.mess3_figures import generate_mess3_figures


def write_raw_results(tmp_path: Path) -> tuple[Path, Path]:
    probes, training = [], []
    for seed in (0, 1, 2):
        identity = dict(model="transformer", seed=seed, generator="mess3",
                        config="mess3_cpu", config_sha256="fixture")
        training.append(dict(identity, record_type="training", test_nll=1.0))
        for condition, control in (("trained", "none"), ("untrained", "none"),
                                   ("trained", "shuffled_labels")):
            probes.append(dict(identity, record_type="probe", training_condition=condition,
                               control=control, joint_belief_r2=.7 + seed * .01,
                               joint_distance_r2=.6 + seed * .01,
                               joint_belief_mse=.02 + seed * .001))
        for position in range(6):
            exact = np.random.default_rng(seed * 10 + position).dirichlet(np.ones(6))
            probes.append(dict(identity, record_type="mess3_geometry", component=position % 2,
                               training_condition="trained", position=position,
                               **{f"exact_b{i}": float(x) for i, x in enumerate(exact)},
                               **{f"pred_b{i}": float(x + .01) for i, x in enumerate(exact)}))
    reproduction = tmp_path / "mess3_reproduction.jsonl"
    training_path = tmp_path / "mess3_training.jsonl"
    reproduction.write_text("".join(json.dumps(row) + "\n" for row in probes))
    training_path.write_text("".join(json.dumps(row) + "\n" for row in training))
    return reproduction, training_path


def test_mess3_figures_are_reconstructed_from_raw_jsonl(tmp_path: Path) -> None:
    reproduction, training = write_raw_results(tmp_path)
    output = tmp_path / "figures"
    output.mkdir()
    existing = output / "probes.png"
    existing.write_bytes(b"existing central figure")
    created = generate_mess3_figures(reproduction, training, output)
    assert {path.name for path in created} == {"mess3_geometry.png", "mess3_metrics.png"}
    assert all(path.stat().st_size > 1000 for path in created)
    assert existing.read_bytes() == b"existing central figure"


def test_mess3_figures_reject_missing_coordinate(tmp_path: Path) -> None:
    reproduction, training = write_raw_results(tmp_path)
    rows = [json.loads(line) for line in reproduction.read_text().splitlines()]
    del next(row for row in rows if row["record_type"] == "mess3_geometry")["pred_b5"]
    reproduction.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="mess3_geometry.*pred_b5"):
        generate_mess3_figures(reproduction, training, tmp_path / "figures")


@pytest.mark.parametrize("defect", ["missing_seed", "mixed_digest", "duplicate_probe"])
def test_mess3_figures_reject_incomplete_or_incompatible_runs(tmp_path: Path, defect: str) -> None:
    reproduction, training = write_raw_results(tmp_path)
    rows = [json.loads(line) for line in reproduction.read_text().splitlines()]
    if defect == "missing_seed":
        rows = [row for row in rows if row["seed"] != 2]
    elif defect == "mixed_digest":
        rows[0]["config_sha256"] = "different"
    else:
        rows.append(rows[0])
    reproduction.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError):
        generate_mess3_figures(reproduction, training, tmp_path / "figures")


def test_geometry_records_use_evenly_spaced_held_out_reconstructions() -> None:
    from types import SimpleNamespace
    from sklearn.linear_model import Ridge
    from probe import mess3_geometry_records

    rng = np.random.default_rng(0)
    hidden = rng.normal(size=(2101, 4))
    exact = rng.dirichlet(np.ones(6), size=len(hidden))
    table = SimpleNamespace(hidden=hidden, joint_belief=exact,
                            components=np.arange(len(hidden)) % 2,
                            positions=np.arange(len(hidden)) % 63,
                            sequence_ids=np.arange(len(hidden)) // 63 + 1_000_000)
    regression = Ridge().fit(hidden[:100], exact[:100])
    rows = mess3_geometry_records(table, regression)
    assert rows == mess3_geometry_records(table, regression)
    assert len(rows) == 2000
    selected = np.linspace(0, len(hidden) - 1, 2000, dtype=int)
    expected_prediction = regression.predict(hidden[selected])
    np.testing.assert_allclose([[row[f"pred_b{i}"] for i in range(6)] for row in rows],
                               expected_prediction)
    np.testing.assert_array_equal([[row[f"exact_b{i}"] for i in range(6)] for row in rows],
                                  exact[selected])
    assert [row["position"] for row in rows] == table.positions[selected].tolist()
    assert [row["sequence_id"] for row in rows] == table.sequence_ids[selected].tolist()
