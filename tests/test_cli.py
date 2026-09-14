import json
import os
import subprocess
import sys
from pathlib import Path

from nonergodic_memory.figures import generate_figures


ROOT = Path(__file__).parents[1]


def test_entrypoints_have_help() -> None:
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    for script in ("train.py", "probe.py", "intervene.py"):
        completed = subprocess.run(
            [sys.executable, str(ROOT / "src" / script), "--help"],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0
        assert "--config" in completed.stdout


def test_figures_are_generated_only_from_jsonl(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    rows = [
        {"record_type": "training", "model": "gru", "seed": 0, "test_nll": 1.1, "test_bayes_nll": 1.0},
        {"record_type": "probe", "model": "gru", "seed": 0, "training_condition": "trained", "control": "none", "component_accuracy": 0.8, "conditional_state_accuracy": 0.7, "component_posterior_r2": 0.6, "state_posterior_r2": 0.5},
        {"record_type": "pca", "model": "gru", "seed": 0, "component": 0, "state": 1, "pc1": 0.2, "pc2": -0.1},
        {"record_type": "pca", "model": "gru", "seed": 0, "component": 1, "state": 0, "pc1": -0.2, "pc2": 0.1},
        {"record_type": "intervention", "model": "gru", "seed": 0, "training_condition": "trained", "target": "component", "control": "learned", "delta_component_accuracy": -0.3, "delta_conditional_state_accuracy": -0.02, "delta_nll": 0.01, "delta_kl_exact": 0.01},
    ]
    with (results / "fixture.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    output = tmp_path / "figures"
    created = generate_figures(results, output)
    assert {path.name for path in created} == {
        "training.png", "probes.png", "pca.png", "intervention.png"
    }
    assert all(path.stat().st_size > 1000 for path in created)
