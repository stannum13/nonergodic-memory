import json
from pathlib import Path

import pytest

from nonergodic_memory.mess3_diagnosis_figures import generate_diagnosis_figures


def _generate(paths: tuple[Path, Path, Path], output_dir: Path) -> list[Path]:
    return generate_diagnosis_figures(
        *paths,
        output_dir,
        expected_seeds=(10, 11),
        expected_steps=(0, 2),
    )


def _write_fixture(root: Path) -> tuple[Path, Path, Path]:
    baselines = []
    for index, name in enumerate(("uniform", "last_token", "window_3_bayes", "full_bayes")):
        baselines.append(
            {
                "record_type": "baseline",
                "predictor": name,
                "kl_exact": 0.03 - index * 0.01,
                "competence": index / 3,
                "config_sha256": "fixture",
            }
        )
    training = []
    probes = []
    for seed in (10, 11):
        for condition in ("reused", "fresh"):
            for step in (0, 2):
                training.append(
                    {
                        "record_type": "diagnostic_training",
                        "seed": seed,
                        "condition": condition,
                        "step": step,
                        "kl_exact": 0.03 - 0.005 * step,
                        "competence": 0.1 + 0.2 * step,
                        "config_sha256": "fixture",
                    }
                )
                for site in ("block_1", "block_2", "final_norm"):
                    for control in ("none", "shuffled_labels"):
                        probes.append(
                            {
                                "record_type": "diagnostic_probe",
                                "seed": seed,
                                "condition": condition,
                                "step": step,
                                "site": site,
                                "control": control,
                                "joint_belief_r2": 0.1 + 0.05 * step,
                                "config_sha256": "fixture",
                            }
                        )
    paths = tuple(root / name for name in ("baselines.jsonl", "training.jsonl", "probes.jsonl"))
    for path, rows in zip(paths, (baselines, training, probes)):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return paths


def test_diagnosis_figures_are_generated_from_raw_records(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    created = _generate(paths, tmp_path / "figures")
    assert {path.name for path in created} == {
        "mess3_predictive_baselines.png",
        "mess3_learning_geometry.png",
    }
    assert all(path.stat().st_size > 0 for path in created)


def test_diagnosis_figures_reject_mixed_provenance(tmp_path: Path) -> None:
    baselines, training, probes = _write_fixture(tmp_path)
    rows = [json.loads(line) for line in probes.read_text().splitlines()]
    rows[0]["config_sha256"] = "other"
    probes.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="provenance"):
        _generate((baselines, training, probes), tmp_path / "figures")


@pytest.mark.parametrize("source_name", ["baselines", "training", "probes"])
def test_diagnosis_figures_reject_duplicate_cells(tmp_path: Path, source_name: str) -> None:
    baselines, training, probes = _write_fixture(tmp_path)
    source = {"baselines": baselines, "training": training, "probes": probes}[source_name]
    rows = source.read_text().splitlines()
    source.write_text("\n".join([*rows, rows[0]]) + "\n")

    with pytest.raises(ValueError, match="duplicated"):
        _generate((baselines, training, probes), tmp_path / "figures")


def test_diagnosis_figures_reject_incomplete_probe_grid(tmp_path: Path) -> None:
    baselines, training, probes = _write_fixture(tmp_path)
    rows = probes.read_text().splitlines()
    probes.write_text("\n".join(rows[1:]) + "\n")

    with pytest.raises(ValueError, match="incomplete"):
        _generate((baselines, training, probes), tmp_path / "figures")


def test_diagnosis_figures_reject_incomplete_training_grid(tmp_path: Path) -> None:
    baselines, training, probes = _write_fixture(tmp_path)
    training_rows = [json.loads(line) for line in training.read_text().splitlines()]
    probe_rows = [json.loads(line) for line in probes.read_text().splitlines()]
    missing = (10, "fresh", 2)
    training_rows = [
        row
        for row in training_rows
        if (row["seed"], row["condition"], row["step"]) != missing
    ]
    probe_rows = [
        row
        for row in probe_rows
        if (row["seed"], row["condition"], row["step"]) != missing
    ]
    training.write_text("".join(json.dumps(row) + "\n" for row in training_rows))
    probes.write_text("".join(json.dumps(row) + "\n" for row in probe_rows))

    with pytest.raises(ValueError, match="training grid is incomplete"):
        _generate((baselines, training, probes), tmp_path / "figures")


@pytest.mark.parametrize(("field", "value"), [("seed", 11), ("step", 2)])
def test_diagnosis_figures_reject_wholly_missing_configured_axis(
    tmp_path: Path, field: str, value: int
) -> None:
    baselines, training, probes = _write_fixture(tmp_path)
    for source in (training, probes):
        rows = [json.loads(line) for line in source.read_text().splitlines()]
        rows = [row for row in rows if row[field] != value]
        source.write_text("".join(json.dumps(row) + "\n" for row in rows))

    with pytest.raises(ValueError, match="training grid is incomplete"):
        _generate((baselines, training, probes), tmp_path / "figures")
