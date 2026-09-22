import json
import os
import subprocess
import sys
from pathlib import Path
import pytest
import yaml

from nonergodic_memory.figures import _load_records, generate_figures
from nonergodic_memory.experiment import load_config, train_one


ROOT = Path(__file__).parents[1]


def test_entrypoints_have_help() -> None:
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    for script in ("train.py", "probe.py", "intervene.py", "context_restart.py", "mess3_diagnose.py"):
        completed = subprocess.run(
            [sys.executable, str(ROOT / "src" / script), "--help"],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0
        assert "--config" in completed.stdout


def test_mess3_diagnosis_cli_writes_complete_tiny_grid(tmp_path: Path) -> None:
    config = {
        "data": {"generator": "mess3", "sequence_length": 8,
                 "train_sequences": 16, "test_sequences": 8},
        "model": {"width": 8, "layers": 2, "heads": 2, "max_length": 16},
        "train": {"batch_size": 4, "learning_rate": 0.01, "weight_decay": 0.01,
                  "checkpoint_steps": [0, 2]},
        "diagnosis": {"window": 3, "baseline_fit_sequences": 32},
        "probe": {"train_sequences": 32, "test_sequences": 24},
    }
    config_path = tmp_path / "diagnosis.yaml"
    config_path.write_text(yaml.safe_dump(config))
    paths = {
        "baseline": tmp_path / "baseline.jsonl",
        "training": tmp_path / "training.jsonl",
        "probe": tmp_path / "probe.jsonl",
    }
    command = [
        sys.executable, str(ROOT / "src/mess3_diagnose.py"),
        "--config", str(config_path), "--mode", "all", "--seeds", "6",
        "--checkpoint-dir", str(tmp_path / "checkpoints"),
        "--baseline-results", str(paths["baseline"]),
        "--training-results", str(paths["training"]),
        "--probe-results", str(paths["probe"]),
        "--output-dir", str(tmp_path / "figures"),
    ]
    completed = subprocess.run(
        command, env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    baseline_rows = [json.loads(line) for line in paths["baseline"].read_text().splitlines()]
    training_rows = [json.loads(line) for line in paths["training"].read_text().splitlines()]
    probe_rows = [json.loads(line) for line in paths["probe"].read_text().splitlines()]
    assert len(baseline_rows) == 4
    assert {(row["condition"], row["step"]) for row in training_rows} == {
        (condition, step) for condition in ("reused", "fresh") for step in (0, 2)
    }
    assert len(probe_rows) == 2 * 2 * 3 * 2
    assert (tmp_path / "figures/mess3_predictive_baselines.png").exists()
    assert (tmp_path / "figures/mess3_learning_geometry.png").exists()

    rerun = subprocess.run(
        command, env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True, text=True, check=False,
    )
    assert rerun.returncode == 0, rerun.stderr
    assert "reused complete diagnostic checkpoints" in rerun.stdout


def test_mess3_cli_records_generator_without_fabricated_overlap(tmp_path: Path) -> None:
    config = {
        "data": {
            "generator": "mess3",
            "sequence_length": 8,
            "train_sequences": 32,
            "test_sequences": 16,
        },
        "model": {"width": 12, "layers": 1, "heads": 2},
        "train": {"epochs": 1, "batch_size": 16, "learning_rate": 0.02},
        "probe": {"train_sequences": 16, "test_sequences": 16},
    }
    config_path = tmp_path / "mess3.yaml"
    config_path.write_text(yaml.safe_dump(config))
    checkpoint_dir = tmp_path / "checkpoints"
    training_results = tmp_path / "training.jsonl"
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    train = subprocess.run(
        [
            sys.executable, str(ROOT / "src" / "train.py"), "--config", str(config_path),
            "--models", "gru", "--seeds", "0", "--output-dir", str(checkpoint_dir),
            "--results", str(training_results),
        ],
        env=environment, capture_output=True, text=True, check=False,
    )
    assert train.returncode == 0, train.stderr
    probe_results = tmp_path / "probe.jsonl"
    probe = subprocess.run(
        [
            sys.executable, str(ROOT / "src" / "probe.py"), "--config", str(config_path),
            "--models", "gru", "--seeds", "0", "--checkpoint-dir", str(checkpoint_dir),
            "--results", str(probe_results),
        ],
        env=environment, capture_output=True, text=True, check=False,
    )
    assert probe.returncode == 0, probe.stderr
    geometry = [json.loads(line) for line in probe_results.read_text().splitlines()
                if json.loads(line)["record_type"] == "mess3_geometry"]
    assert len(geometry) == 16 * 7
    assert all(row["training_condition"] == "trained" for row in geometry)
    assert all(row["sequence_id"] >= 1_000_000 for row in geometry)
    assert all(row["config_sha256"] and row["python_version"] and row["numpy_version"]
               and row["torch_version"] for row in geometry)
    for row in geometry:
        assert all(f"{prefix}_b{i}" in row for prefix in ("exact", "pred") for i in range(6))
        assert sum(row[f"exact_b{i}"] for i in range(6)) == pytest.approx(1.)
    for path in (training_results, probe_results):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert rows
        assert all(row["generator"] == "mess3" for row in rows)
        assert all("overlap" not in row for row in rows)


@pytest.mark.parametrize("cache", ["complete", "absent", "partial"])
def test_mess3_reproduction_script_reuses_only_complete_runs(tmp_path: Path, cache: str) -> None:
    # Stub the interpreter boundary: exercise real shell orchestration without
    # running the full three-seed experiment in the CLI test suite.
    executable = tmp_path / "python"
    log = tmp_path / "commands.jsonl"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "args = sys.argv[1:]\n"
        "with open(os.environ['COMMAND_LOG'], 'a') as handle:\n"
        "    handle.write(json.dumps(args) + '\\n')\n"
        "if len(args) > 1 and args[1] in ('nonergodic_memory.checkpoints', 'nonergodic_memory.training_records'):\n"
        "    seed = args[args.index('--seeds') + 1]\n"
        "    cache = os.environ['CACHE_STATE']\n"
        "    if cache == 'absent' or (cache == 'partial' and\n"
        "        ((seed == '1' and args[1].endswith('checkpoints')) or\n"
        "         (seed == '2' and args[1].endswith('training_records')))):\n"
        "        sys.exit(1)\n"
    )
    executable.chmod(0o755)
    completed = subprocess.run(
        ["bash", str(ROOT / "scripts" / "reproduce_mess3.sh")], cwd=ROOT,
        env={**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}",
             "COMMAND_LOG": str(log), "CACHE_STATE": cache},
        capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    commands = [json.loads(line) for line in log.read_text().splitlines()]
    trains = [args for args in commands if args[0] == "src/train.py"]
    assert [args[args.index("--seeds") + 1] for args in trains] == {
        "complete": [], "absent": ["0", "1", "2"], "partial": ["1", "2"]
    }[cache]
    assert all(args[args.index("--models") + 1] == "transformer" for args in trains)
    assert all(args[args.index("--results") + 1] == "results/mess3_training.jsonl" for args in trains)
    probe = next(args for args in commands if args[0] == "src/probe.py")
    assert probe[probe.index("--seeds") + 1:probe.index("--seeds") + 4] == ["0", "1", "2"]
    assert probe[probe.index("--models") + 1] == "transformer"
    assert probe[probe.index("--results") + 1] == "results/mess3_reproduction.jsonl"
    assert commands[-1][:2] == ["-m", "nonergodic_memory.mess3_figures"]


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


def test_position_preserving_cli_writes_three_aligned_transformer_contexts(tmp_path: Path) -> None:
    config_path = ROOT / "configs" / "smoke.yaml"
    checkpoint_root = tmp_path / "checkpoints"
    train_one(load_config(config_path), "transformer", seed=0, output_dir=checkpoint_root / "smoke")
    output = tmp_path / "absolute.jsonl"
    completed = subprocess.run(
        [
            sys.executable, str(ROOT / "src" / "context_restart.py"),
            "--configs", str(config_path), "--models", "transformer", "--seeds", "0",
            "--window", "8", "--include-absolute-positions",
            "--checkpoint-root", str(checkpoint_root), "--results", str(output),
        ],
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 13
    model_rows = [r for r in rows if r["record_type"] == "context_restart"]
    assert {r["context"] for r in model_rows} == {"full", "restart_8", "restart_8_absolute"}
    assert all(r["positions_evaluated"] == 4 and r["observations_evaluated"] == 48 * 4 for r in model_rows)


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
