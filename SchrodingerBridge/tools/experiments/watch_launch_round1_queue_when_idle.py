from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SB_ROOT = SCRIPT_DIR.parent.parent
WORKSPACE = SB_ROOT.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from csv_utils import read_csv_rows


DEFAULT_MANIFEST = SB_ROOT / "docs" / "experiments" / "round1_full_sweep" / "round1_family_manifest.csv"


def _run(cmd: list[str]) -> int:
    print("[watch_launch_round1_queue_when_idle] " + " ".join(str(x) for x in cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(WORKSPACE), check=False)
    return int(proc.returncode)


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    return read_csv_rows(path)


def _count_status(rows: list[dict[str, str]], status: str) -> int:
    wanted = str(status).strip().lower()
    return sum(1 for row in rows if str(row.get("decision_status", "")).strip().lower() == wanted)


def main() -> int:
    parser = argparse.ArgumentParser(description="Wait for round1 to have no running families, then launch the next queue entry once.")
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--poll-seconds", type=int, default=180)
    parser.add_argument("--queue-script", type=Path, default=SCRIPT_DIR / "run_round1_family_queue.py")
    parser.add_argument("--queue-arg", action="append", default=[])
    parser.add_argument("--max-cycles", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest_csv = Path(args.manifest_csv).expanduser()
    if not manifest_csv.is_absolute():
        manifest_csv = (WORKSPACE / manifest_csv).resolve()
    queue_script = Path(args.queue_script).expanduser()
    if not queue_script.is_absolute():
        queue_script = (WORKSPACE / queue_script).resolve()

    cycles = 0
    while True:
        rows = _read_rows(manifest_csv)
        running_count = _count_status(rows, "running")
        planned_count = _count_status(rows, "planned")
        print(
            f"[watch_launch_round1_queue_when_idle] running={running_count} planned={planned_count} manifest={manifest_csv}",
            flush=True,
        )
        if running_count == 0:
            if planned_count == 0:
                print("[watch_launch_round1_queue_when_idle] no planned families remain; exiting", flush=True)
                return 0
            cmd = [sys.executable, str(queue_script)]
            if manifest_csv:
                cmd.extend(["--manifest-csv", str(manifest_csv)])
            for extra in args.queue_arg:
                text = str(extra).strip()
                if text:
                    cmd.append(text)
            if bool(args.dry_run):
                print("[watch_launch_round1_queue_when_idle] DRY_RUN " + " ".join(cmd), flush=True)
                return 0
            return _run(cmd)
        cycles += 1
        if int(args.max_cycles) > 0 and cycles >= int(args.max_cycles):
            return 0
        time.sleep(max(1, int(args.poll_seconds)))


if __name__ == "__main__":
    raise SystemExit(main())
