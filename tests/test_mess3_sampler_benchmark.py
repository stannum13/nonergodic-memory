import json
import os
import subprocess
import sys
from pathlib import Path

from nonergodic_memory.data import make_mess3_mixture
from benchmark_mess3_sampler import benchmark_sampler


ROOT = Path(__file__).parents[1]


def test_sampler_benchmark_has_auditable_schema() -> None:
    row = benchmark_sampler(make_mess3_mixture(), sequences=8, length=6, repeats=2)

    assert row["record_type"] == "mess3_sampler_benchmark"
    assert row["samplers"] == ["reference", "vectorized"]
    assert row["sequences"] == 8
    assert row["length"] == 6
    assert row["repeats"] == 2
    assert row["reference_median_seconds"] > 0
    assert row["vectorized_median_seconds"] > 0
    assert row["speedup"] > 0


def test_sampler_benchmark_cli_writes_one_json_record(tmp_path: Path) -> None:
    output = tmp_path / "benchmark.jsonl"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "src/benchmark_mess3_sampler.py"),
            "--sequences",
            "8",
            "--length",
            "6",
            "--repeats",
            "2",
            "--output",
            str(output),
        ],
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["record_type"] == "mess3_sampler_benchmark"
