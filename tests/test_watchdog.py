import json
import sys
from pathlib import Path

from nonergodic_memory.watchdog import run_with_timeout


def test_watchdog_records_timeout_and_kills_child(tmp_path: Path):
    summary = tmp_path / "summary.jsonl"
    code = run_with_timeout(
        [sys.executable, "-c", "import time; time.sleep(5)"],
        seconds=0.1,
        timeout_summary=summary,
    )
    row = json.loads(summary.read_text(encoding="utf-8"))
    assert code == 124
    assert row["status"] == "inconclusive_timeout"
    assert row["stage"] == "external_watchdog"


def test_watchdog_returns_success_without_writing_summary(tmp_path: Path):
    summary = tmp_path / "summary.jsonl"
    code = run_with_timeout(
        [sys.executable, "-c", "pass"], seconds=2, timeout_summary=summary
    )
    assert code == 0
    assert not summary.exists()
