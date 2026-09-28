import json
from pathlib import Path

import pytest

from nonergodic_memory.experiment import config_digest, load_config
from nonergodic_memory.predictive_memory import analyze_evidence


ROOT = Path(__file__).resolve().parents[1]


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_retained_predictive_memory_evidence_is_the_registered_feasibility_stop():
    records = _read(ROOT / "results/predictive_memory.jsonl")
    summary = _read(ROOT / "results/predictive_memory_summary.jsonl")
    config = load_config(ROOT / "configs/predictive_memory.yaml")

    assert len(records) == 4
    assert len(summary) == 1
    assert [row["record_type"] for row in records].count("split_audit") == 2
    calibrations = [row for row in records if row["record_type"] == "calibration"]
    assert {row["seed"] for row in calibrations} == {10, 11}
    assert not any(row["record_type"] == "response" for row in records)
    assert not any(row.get("cohort") == "heldout" for row in records)
    assert all(
        [name for name, passed in row["checks"].items() if not passed]
        == ["displacement_relative_error"]
        for row in calibrations
    )
    by_seed = {row["seed"]: row for row in calibrations}
    assert by_seed[10]["displacement_relative_error"] == pytest.approx(
        0.6133913055662932
    )
    assert by_seed[11]["displacement_relative_error"] == pytest.approx(
        0.658562931053291
    )
    analysis = analyze_evidence(records, summary[0], config, config_digest(config))
    assert analysis["valid"] is True
    assert analysis["status"] == "actuator_infeasible"
    assert analysis["validity_errors"] == []
