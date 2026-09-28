"""External wall-time watchdog for the registered predictive-memory command."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
from pathlib import Path


def run_with_timeout(
    command: list[str],
    *,
    seconds: float,
    timeout_summary: str | Path,
    reservation: str | Path | None = None,
) -> int:
    destination = Path(timeout_summary)
    reservation_path = Path(reservation) if reservation is not None else None

    if reservation_path is not None:
        reservation_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with reservation_path.open("x", encoding="utf-8") as handle:
                handle.write(
                    json.dumps({"event": "started", "record_type": "watchdog"}, sort_keys=True)
                    + "\n"
                )
                handle.flush()
                os.fsync(handle.fileno())
        except FileExistsError:
            return 73

    def finish_reservation(status: str) -> None:
        if reservation_path is not None:
            with reservation_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"event": "finished", "status": status}, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())

    def write_status(record: dict) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    process = subprocess.Popen(command, start_new_session=True)
    try:
        code = int(process.wait(timeout=seconds))
        write_status(
            {
                "record_type": "command_status",
                "status": "complete" if code == 0 else "inconclusive_execution_error",
                "child_exit_code": code,
            }
        )
        finish_reservation("complete" if code == 0 else "inconclusive_execution_error")
        return code
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        write_status(
            {
                "record_type": "command_status",
                "status": "inconclusive_timeout",
                "stage": "external_watchdog",
                "wall_time_seconds": seconds,
            }
        )
        attempt = destination.with_name(
            destination.name.replace("_command", "_attempt").replace(
                "_summary", "_attempt"
            )
        )
        if attempt != destination and attempt.exists():
            with attempt.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {"event": "finished", "status": "inconclusive_timeout"},
                        sort_keys=True,
                    )
                    + "\n"
                )
                handle.flush()
                os.fsync(handle.fileno())
        finish_reservation("inconclusive_timeout")
        return 124


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--timeout-summary", required=True)
    parser.add_argument("--reservation")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required after --")
    raise SystemExit(
        run_with_timeout(
            command,
            seconds=args.seconds,
            timeout_summary=args.timeout_summary,
            reservation=args.reservation,
        )
    )


if __name__ == "__main__":
    main()
