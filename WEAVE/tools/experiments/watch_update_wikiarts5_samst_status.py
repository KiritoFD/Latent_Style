from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def _run(cmd: list[str]) -> int:
    print("[watch_update_wikiarts5_samst_status] " + " ".join(str(x) for x in cmd), flush=True)
    proc = subprocess.run(cmd, check=False)
    return int(proc.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description="Periodically refresh the wikiarts5 SaMST live-status note.")
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--max-cycles", type=int, default=0)
    args = parser.parse_args()

    update_script = SCRIPT_DIR / "update_wikiarts5_samst_status.py"
    cycles = 0
    while True:
        _run([sys.executable, str(update_script), "--result-root", str(args.result_root)])
        cycles += 1
        if int(args.max_cycles) > 0 and cycles >= int(args.max_cycles):
            return 0
        time.sleep(max(1, int(args.poll_seconds)))


if __name__ == "__main__":
    raise SystemExit(main())
