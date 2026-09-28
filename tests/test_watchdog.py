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


def test_watchdog_records_success_after_entire_child_command(tmp_path: Path):
    summary = tmp_path / "command.jsonl"
    code = run_with_timeout(
        [sys.executable, "-c", "pass"], seconds=2, timeout_summary=summary
    )
    assert code == 0
    row = json.loads(summary.read_text(encoding="utf-8"))
    assert row["status"] == "complete"
    assert row["child_exit_code"] == 0


def test_watchdog_rejects_second_owner_without_running_or_writing_status(tmp_path: Path):
    status = tmp_path / "command.jsonl"
    reservation = tmp_path / "watchdog.jsonl"
    side_effect = tmp_path / "child-ran"
    reservation.write_text('{"event":"started"}\n', encoding="utf-8")
    before = reservation.read_bytes()
    code = run_with_timeout(
        [sys.executable, "-c", f"from pathlib import Path; Path({str(side_effect)!r}).touch()"],
        seconds=2,
        timeout_summary=status,
        reservation=reservation,
    )
    assert code == 73
    assert reservation.read_bytes() == before
    assert not side_effect.exists()
    assert not status.exists()
