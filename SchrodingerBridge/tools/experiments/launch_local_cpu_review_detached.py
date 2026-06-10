from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Launch a local CPU-only review command as a detached background process with stdout/stderr logs."
    )
    parser.add_argument("--log-prefix", required=True, help="Prefix path for .stdout.log and .stderr.log")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to launch after '--'")
    args = parser.parse_args()

    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("missing command after '--'")

    log_prefix = Path(str(args.log_prefix)).resolve()
    log_prefix.parent.mkdir(parents=True, exist_ok=True)
    stdout_path = log_prefix.with_suffix(".stdout.log")
    stderr_path = log_prefix.with_suffix(".stderr.log")

    env = os.environ.copy()
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS

    # Overwrite old logs on each fresh detached launch so stale errors do not
    # get mixed into the current run's status read.
    with stdout_path.open("wb") as stdout_f, stderr_path.open("wb") as stderr_f:
        proc = subprocess.Popen(
            command,
            stdout=stdout_f,
            stderr=stderr_f,
            stdin=subprocess.DEVNULL,
            cwd=str(Path.cwd()),
            env=env,
            creationflags=creationflags,
            close_fds=True,
        )
    print(f"pid={proc.pid}")
    print(stdout_path)
    print(stderr_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
