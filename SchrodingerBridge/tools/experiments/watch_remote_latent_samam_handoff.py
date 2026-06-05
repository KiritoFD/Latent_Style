from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent.parent.parent
DEFAULT_HELPER = WORKSPACE_ROOT / "SchrodingerBridge/tools/experiments/handoff_remote_latent_samam_to_a1.py"


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


def _print_block(title: str, text: str) -> None:
    print(f"=== {title} ===")
    if text:
        print(text.rstrip())
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Poll the remote latent SaMam side quest until the first retained "
            "checkpoint exists, then stop that lane and launch A1 automatically."
        )
    )
    parser.add_argument("--helper", default=str(DEFAULT_HELPER))
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--max-polls", type=int, default=0, help="0 means unlimited polling.")
    parser.add_argument("--max-idle-memory-mib", type=int, default=1500)
    parser.add_argument("--idle-poll-seconds", type=int, default=10)
    parser.add_argument("--idle-timeout-seconds", type=int, default=300)
    args = parser.parse_args()

    helper_path = Path(args.helper).resolve()
    if not helper_path.is_file():
        raise FileNotFoundError(helper_path)

    poll_seconds = max(5, int(args.poll_seconds))
    max_polls = max(0, int(args.max_polls))
    poll_index = 0

    while True:
        poll_index += 1
        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
            f"watch poll {poll_index}"
        )
        dry_run = _run([sys.executable, str(helper_path), "--dry-run"])
        _print_block("dry-run", dry_run.stdout)

        if dry_run.returncode == 0:
            launch = _run(
                [
                    sys.executable,
                    str(helper_path),
                    "--stop-latent-on-retained",
                    "--max-idle-memory-mib",
                    str(int(args.max_idle_memory_mib)),
                    "--idle-poll-seconds",
                    str(int(args.idle_poll_seconds)),
                    "--idle-timeout-seconds",
                    str(int(args.idle_timeout_seconds)),
                ]
            )
            _print_block("handoff", launch.stdout)
            return launch.returncode

        if max_polls and poll_index >= max_polls:
            print(f"Reached max polls without retained checkpoint: {max_polls}")
            return dry_run.returncode

        time.sleep(poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
