import json
from pathlib import Path

import pytest

from nonergodic_memory.mess3_diagnosis_figures import generate_diagnosis_figures


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
                    probes.append(
                        {
                            "record_type": "diagnostic_probe",
                            "seed": seed,
                            "condition": condition,
                            "step": step,
                            "site": site,
                            "control": "none",
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
    created = generate_diagnosis_figures(*paths, tmp_path / "figures")
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
        generate_diagnosis_figures(baselines, training, probes, tmp_path / "figures")
