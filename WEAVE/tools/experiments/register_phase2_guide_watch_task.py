from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PS1_PATH = SCRIPT_DIR / "run_phase2_guide_watch.ps1"
DEFAULT_TASK_NAME = "SB-Phase2-GuideWatch"


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Register a Windows scheduled task that refreshes the phase2 guide digest periodically.")
    parser.add_argument("--task-name", default=DEFAULT_TASK_NAME)
    parser.add_argument("--interval-minutes", type=int, default=15)
    parser.add_argument("--run-now", action="store_true")
    args = parser.parse_args()

    interval = max(1, int(args.interval_minutes))
    task_name = str(args.task_name).strip() or DEFAULT_TASK_NAME
    ps1_abs = str(PS1_PATH.resolve())
    task_cmd = f'powershell.exe -NoProfile -ExecutionPolicy Bypass -File "{ps1_abs}"'

    create = _run(
        [
            "schtasks",
            "/Create",
            "/TN",
            task_name,
            "/TR",
            task_cmd,
            "/SC",
            "MINUTE",
            "/MO",
            str(interval),
            "/F",
        ]
    )
    print(create.stdout, end="")
    if create.returncode != 0:
        return create.returncode

    query = _run(["schtasks", "/Query", "/TN", task_name, "/FO", "LIST", "/V"])
    print(query.stdout, end="")
    if query.returncode != 0:
        return query.returncode

    if bool(args.run_now):
        run_now = _run(["schtasks", "/Run", "/TN", task_name])
        print(run_now.stdout, end="")
        if run_now.returncode != 0:
            return run_now.returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
