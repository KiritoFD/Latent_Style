from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SB_ROOT = SCRIPT_DIR.parent.parent
WORKSPACE = SB_ROOT.parent


def _run(cmd: list[str]) -> int:
    print("[launch_remote_hold4mid_carriergate_injection] " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, check=False)
    return int(proc.returncode)


def main() -> int:
    launch = WORKSPACE / "SchrodingerBridge" / "tools" / "experiments" / "launch_remote_aaai2027_packet.py"
    command = [
        sys.executable,
        str(launch),
        "--config",
        "SchrodingerBridge/configs/aaai2027/inmortal_hold4mid_e8_carriergate_injection_seed42_b8a2.json",
        "--sync-path",
        "SchrodingerBridge/docs/experiments/2026-06-09-hold4mid-e8-carriergate-next-lane.md",
        "--max-prelaunch-memory-mib",
        "1500",
    ]
    return _run(command)


if __name__ == "__main__":
    raise SystemExit(main())
