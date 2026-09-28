"""External wall-time watchdog for the registered predictive-memory command."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
from pathlib import Path


def run_with_timeout(
    command: list[str], *, seconds: float, timeout_summary: str | Path
) -> int:
    process = subprocess.Popen(command, start_new_session=True)
    try:
        return int(process.wait(timeout=seconds))
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        destination = Path(timeout_summary)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            with destination.open("x", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "record_type": "summary",
                            "status": "inconclusive_timeout",
                            "stage": "external_watchdog",
                            "wall_time_seconds": seconds,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                handle.flush()
                os.fsync(handle.fileno())
        attempt = destination.with_name(destination.name.replace("_summary", "_attempt"))
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
        return 124


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--timeout-summary", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required after --")
    raise SystemExit(
        run_with_timeout(
            command, seconds=args.seconds, timeout_summary=args.timeout_summary
        )
    )


if __name__ == "__main__":
    main()
