#!/usr/bin/env python3
"""Run one command with a portable process-group timeout."""

from __future__ import annotations

import os
import signal
import subprocess
import sys


def stop_group(process: subprocess.Popen[bytes], signal_number: int) -> None:
    try:
        os.killpg(process.pid, signal_number)
    except ProcessLookupError:
        pass


def main() -> int:
    if len(sys.argv) < 4 or sys.argv[2] != "--":
        print("usage: run_with_timeout.py SECONDS -- COMMAND [ARG ...]", file=sys.stderr)
        return 2

    try:
        seconds = int(sys.argv[1])
    except ValueError:
        print("SECONDS must be a positive integer", file=sys.stderr)
        return 2
    if seconds <= 0:
        print("SECONDS must be a positive integer", file=sys.stderr)
        return 2

    process = subprocess.Popen(sys.argv[3:], start_new_session=True)
    try:
        return process.wait(timeout=seconds)
    except subprocess.TimeoutExpired:
        print(f"command timed out after {seconds} seconds", file=sys.stderr)
        stop_group(process, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            stop_group(process, signal.SIGKILL)
            process.wait()
        return 124
    except KeyboardInterrupt:
        stop_group(process, signal.SIGINT)
        process.wait()
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
