import json
from pathlib import Path

import pytest

from nonergodic_memory.experiment import config_digest, load_config
from nonergodic_memory.predictive_memory import passes_feasibility_gates
from nonergodic_memory.predictive_memory_figures import generate_predictive_memory_figure


def test_predictive_memory_figure_handles_gate_only_results(tmp_path: Path):
    results = tmp_path / "results.jsonl"
    config = load_config("configs/predictive_memory.yaml")
    digest = config_digest(config)
    rows = []
    for seed in (10, 11):
        rows.append(
            {
                "record_type": "split_audit",
                "seed": seed,
                "cohort": "development",
                "split_hashes": {name: "a" * 64 for name in config["data"]},
                "split_sizes": config["data"],
                "overlap_count": 0,
                "experiment_config_sha256": digest,
            }
        )
    for seed, component, joint, error, passed in (
        (10, 0.25, 0.55, 0.4, True),
        (11, 0.15, 0.60, 0.3, False),
    ):
        metrics = {
            "record_type": "calibration",
            "seed": seed,
            "step": 3072,
            "cohort": "development",
            "component_r2": component,
            "joint_r2": joint,
            "displacement_relative_error": error,
            "actuator_rank": 5.0,
            "actuator_constraint_relative_error": 0.0,
            "oracle_denominator": 2e-6,
            "edit_rms_p95": 0.5,
            "natural_rms_p95": 1.0,
            "identity_max_abs_error": 0.0,
            "exact_max_abs_error": 0.0,
            "checkpoint": f"checkpoint-{seed}.pt",
            "checkpoint_sha256": "b" * 64,
            "experiment_config_sha256": digest,
        }
        metrics["checks"] = passes_feasibility_gates(metrics, config["gates"])
        metrics["passed"] = all(metrics["checks"].values())
        assert metrics["passed"] is passed
        rows.append(metrics)
    results.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    summary = tmp_path / "summary.jsonl"
    summary.write_text(
        json.dumps(
            {
                "record_type": "summary",
                "status": "actuator_infeasible",
                "stage": "development",
                "result_rows": 4,
                "config_sha256": digest,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "figure.png"

    generated = generate_predictive_memory_figure(results, output, summary)

    assert generated == output
    assert output.stat().st_size > 1_000
    stale = json.loads(summary.read_text(encoding="utf-8"))
    stale["result_rows"] = 5
    summary.write_text(json.dumps(stale) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="row count"):
        generate_predictive_memory_figure(results, output, summary)


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
    with pytest.raises((FileNotFoundError, ValueError)):
        generate_predictive_memory_figure(results, tmp_path / "figure.png", tmp_path / "missing.jsonl")
