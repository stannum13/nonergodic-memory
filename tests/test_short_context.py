import json
import os
from pathlib import Path
import subprocess
import sys

import yaml
import pytest

from nonergodic_memory.experiment import load_config, train_one
from short_context import _check_matched


ROOT = Path(__file__).parents[1]


def test_short_context_cli_writes_aligned_independent_raw_records(tmp_path: Path) -> None:
    base = load_config(ROOT / "configs" / "smoke.yaml")
    short = {**base, "data": {**base["data"], "sequence_length": 9}}
    long = {**base, "data": {**base["data"], "sequence_length": 64}}
    short_path = tmp_path / "short_smoke.yaml"
    long_path = tmp_path / "long_smoke.yaml"
    short_path.write_text(yaml.safe_dump(short))
    long_path.write_text(yaml.safe_dump(long))
    checkpoint_root = tmp_path / "checkpoints"
    train_one(short, "transformer", seed=0, output_dir=checkpoint_root / "short_smoke")
    output = tmp_path / "short.jsonl"
    completed = subprocess.run(
        [
            sys.executable, str(ROOT / "src" / "short_context.py"),
            "--eval-configs", str(long_path), "--short-configs", str(short_path),
            "--seeds", "0", "--checkpoint-root", str(checkpoint_root),
            "--results", str(output),
        ],
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 2
    assert {row["control"] for row in rows} == {"none", "shuffled_labels"}
    assert all(row["record_type"] == "short_context" for row in rows)
    assert all(row["positions_evaluated"] == 56 and row["observations_evaluated"] == 48 * 56 for row in rows)
    assert all(row["probe_fit_data_seed"] == 909 and row["test_data_seed"] == 1009 for row in rows)
    assert all(row["short_training_sequence_length"] == 9 and row["sequence_length"] == 64 for row in rows)
    assert all(row["probe_fit_independent"] and row["short_config_sha256"] != row["config_sha256"] for row in rows)


def test_budget_protocol_matches_exact_tokens_and_optimizer_steps() -> None:
    eval_config = load_config(ROOT / "configs" / "sweeps" / "interaction_o035_l064.yaml")
    budget_config = load_config(ROOT / "configs" / "sweeps" / "budget_o035_l009.yaml")
    _check_matched(eval_config, budget_config, "budget")
    assert budget_config["data"]["train_sequences"] * 8 == eval_config["data"]["train_sequences"] * 63
    assert budget_config["data"]["train_sequences"] // budget_config["train"]["batch_size"] == 8
    broken = {**budget_config, "train": {**budget_config["train"], "batch_size": 503}}
    with pytest.raises(ValueError, match="budget"):
        _check_matched(eval_config, broken, "budget")
    short_epochs = {**budget_config, "train": {**budget_config["train"], "epochs": 8}}
    long_epochs = {**eval_config, "train": {**eval_config["train"], "epochs": 8}}
    with pytest.raises(ValueError, match="budget"):
        _check_matched(long_epochs, short_epochs, "budget")


def test_short_context_cli_supports_gru(tmp_path: Path) -> None:
    base = load_config(ROOT / "configs" / "smoke.yaml")
    short = {**base, "data": {**base["data"], "sequence_length": 9}}
    long = {**base, "data": {**base["data"], "sequence_length": 64}}
    short_path, long_path = tmp_path / "short.yaml", tmp_path / "long.yaml"
    short_path.write_text(yaml.safe_dump(short))
    long_path.write_text(yaml.safe_dump(long))
    checkpoint_root = tmp_path / "checkpoints"
    train_one(short, "gru", seed=0, output_dir=checkpoint_root / "short")
    output = tmp_path / "gru.jsonl"
    completed = subprocess.run(
        [
            sys.executable, str(ROOT / "src" / "short_context.py"),
            "--eval-configs", str(long_path), "--short-configs", str(short_path),
            "--model", "gru", "--seeds", "0", "--checkpoint-root", str(checkpoint_root),
            "--results", str(output),
        ],
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 2 and {row["model"] for row in rows} == {"gru"}
