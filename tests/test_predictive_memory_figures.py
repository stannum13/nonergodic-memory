import json
from pathlib import Path

from nonergodic_memory.predictive_memory_figures import generate_predictive_memory_figure


def test_predictive_memory_figure_handles_gate_only_results(tmp_path: Path):
    results = tmp_path / "results.jsonl"
    rows = [
        {
            "record_type": "calibration",
            "seed": seed,
            "step": 3072,
            "cohort": "development",
            "component_r2": component,
            "joint_r2": joint,
            "displacement_relative_error": error,
            "passed": passed,
        }
        for seed, component, joint, error, passed in (
            (10, 0.25, 0.55, 0.4, True),
            (11, 0.15, 0.60, 0.3, False),
        )
    ]
    results.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    summary = tmp_path / "summary.jsonl"
    summary.write_text(
        json.dumps(
            {
                "record_type": "summary",
                "status": "actuator_infeasible",
                "stage": "development",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "figure.png"

    generated = generate_predictive_memory_figure(results, output, summary)

    assert generated == output
    assert output.stat().st_size > 1_000


def test_predictive_memory_figure_rejects_partial_results_without_summary(tmp_path: Path):
    results = tmp_path / "results.jsonl"
    results.write_text(
        json.dumps(
            {
                "record_type": "calibration",
                "seed": 10,
                "step": 3072,
                "cohort": "development",
                "component_r2": 0.1,
                "joint_r2": 0.2,
                "displacement_relative_error": 0.9,
                "passed": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    import pytest

    with pytest.raises((FileNotFoundError, ValueError)):
        generate_predictive_memory_figure(results, tmp_path / "figure.png", tmp_path / "missing.jsonl")
