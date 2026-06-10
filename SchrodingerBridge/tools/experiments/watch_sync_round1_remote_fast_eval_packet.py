from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def _run(cmd: list[str]) -> int:
    print("[watch_sync_round1_remote_fast_eval_packet] " + " ".join(str(x) for x in cmd), flush=True)
    proc = subprocess.run(cmd, check=False)
    return int(proc.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description="Periodically pull and refresh a round-1 family's remote fast-eval packet.")
    parser.add_argument("--family-id", required=True)
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--max-cycles", type=int, default=0)
    args = parser.parse_args()

    sync_script = SCRIPT_DIR / "sync_round1_remote_fast_eval_packet.py"
    cycles = 0
    while True:
        rc = _run([sys.executable, str(sync_script), "--family-id", str(args.family_id)])
        if rc != 0:
            print(f"[watch_sync_round1_remote_fast_eval_packet] sync rc={rc}; continuing", flush=True)
        cycles += 1
        if int(args.max_cycles) > 0 and cycles >= int(args.max_cycles):
            return 0
        time.sleep(max(1, int(args.poll_seconds)))


if __name__ == "__main__":
    raise SystemExit(main())
