from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE = SCRIPT_DIR.parents[2]
SEGMENTED_SCRIPT = SCRIPT_DIR / "run_remote_round2_family_segmented.py"


def _find_local_python_pids_for_tokens(*tokens: str) -> list[int]:
    clauses: list[str] = []
    for token in tokens:
        text = str(token).strip()
        if not text:
            continue
        safe = text.replace("'", "''")
        clauses.append(f"($_.CommandLine -like '*{safe}*')")
    if not clauses:
        return []
    ps = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.Name -eq 'python.exe' -and "
        + " -and ".join(clauses)
        + " } | Select-Object -ExpandProperty ProcessId"
    )
    proc = subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        check=False,
    )
    pids: list[int] = []
    for line in proc.stdout.splitlines():
        text = line.strip()
        if text.isdigit():
            pids.append(int(text))
    return pids


def _stop_existing_segmented_controllers(*, family_id: str) -> None:
    pids = _find_local_python_pids_for_tokens("run_remote_round2_family_segmented.py", f"--family-id {family_id}")
    if not pids:
        return
    subprocess.run(
        ["powershell", "-NoProfile", "-Command", "Stop-Process -Id " + ",".join(str(pid) for pid in pids) + " -Force"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        check=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch run_remote_round2_family_segmented.py as a detached local background controller.")
    parser.add_argument("--family-id", required=True)
    parser.add_argument("--stdout-log", required=True)
    parser.add_argument("--stderr-log", required=True)
    args, unknown = parser.parse_known_args()

    family_id = str(args.family_id).strip()
    if not family_id:
        raise ValueError("missing family id")

    extra = list(unknown)
    if extra and extra[0] == "--":
        extra = extra[1:]

    _stop_existing_segmented_controllers(family_id=family_id)

    stdout_path = Path(args.stdout_log).resolve()
    stderr_path = Path(args.stderr_log).resolve()
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, str(SEGMENTED_SCRIPT), "--family-id", family_id, *extra]
    env = os.environ.copy()
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS

    with stdout_path.open("w", encoding="utf-8") as stdout_f, stderr_path.open("w", encoding="utf-8") as stderr_f:
        proc = subprocess.Popen(
            cmd,
            cwd=str(WORKSPACE),
            env=env,
            stdout=stdout_f,
            stderr=stderr_f,
            stdin=subprocess.DEVNULL,
            creationflags=creationflags,
            close_fds=True,
        )

    print(f"pid={proc.pid}")
    print(stdout_path)
    print(stderr_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
