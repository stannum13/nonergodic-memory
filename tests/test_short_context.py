import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

from nonergodic_memory.experiment import load_config, train_one


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
