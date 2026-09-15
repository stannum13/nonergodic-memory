import json
import os
import subprocess
import sys
from pathlib import Path
import pytest

from nonergodic_memory.figures import _load_records, generate_figures
from nonergodic_memory.experiment import load_config, train_one


ROOT = Path(__file__).parents[1]


def test_entrypoints_have_help() -> None:
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    for script in ("train.py", "probe.py", "intervene.py", "context_restart.py"):
        completed = subprocess.run(
            [sys.executable, str(ROOT / "src" / script), "--help"],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0
        assert "--config" in completed.stdout


def test_context_restart_cli_writes_aligned_raw_records(tmp_path: Path) -> None:
    config_path = ROOT / "configs" / "smoke.yaml"
    checkpoint_root = tmp_path / "checkpoints"
    train_one(load_config(config_path), "gru", seed=0, output_dir=checkpoint_root / "smoke")
    output = tmp_path / "restart.jsonl"
    completed = subprocess.run(
        [
            sys.executable, str(ROOT / "src" / "context_restart.py"),
            "--configs", str(config_path), "--models", "gru", "--seeds", "0",
            "--window", "8", "--checkpoint-root", str(checkpoint_root),
            "--results", str(output),
        ],
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 9
    model_rows = [r for r in rows if r["record_type"] == "context_restart"]
    assert {r["context"] for r in model_rows} == {"full", "restart_8"}
    assert {r["control"] for r in model_rows} == {"none", "shuffled_labels"}
    assert {r["training_condition"] for r in model_rows} == {"trained", "untrained"}
    assert all(r["positions_evaluated"] == 4 and r["window"] == 8 for r in rows)
    assert {r["record_type"] for r in rows} == {"context_restart", "context_oracle"}


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


def test_central_results_take_precedence_over_smoke(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    (results / "smoke_training.jsonl").write_text(
        json.dumps({"record_type": "training", "model": "smoke", "seed": 0, "test_nll": 9.0, "test_bayes_nll": 8.0}) + "\n"
    )
    (results / "training.jsonl").write_text(
        json.dumps({"record_type": "training", "model": "central", "seed": 0, "test_nll": 1.0, "test_bayes_nll": 0.9}) + "\n"
    )
    records = _load_records(results)
    assert [record["model"] for record in records] == ["central"]
    output = tmp_path / "figures"
    output.mkdir()
    stale = output / "probes.png"
    stale.write_bytes(b"stale")
    generate_figures(results, output)
    assert not stale.exists()


def test_figures_reject_mixed_digests_for_same_config_name(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    rows = [
        {"record_type": "training", "config": "central", "config_sha256": digest, "model": "gru", "seed": seed, "test_nll": 1.0, "test_bayes_nll": 0.9}
        for digest, seed in (("old", 0), ("new", 1))
    ]
    (results / "training.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="mixed config digests"):
        generate_figures(results, tmp_path / "figures")
