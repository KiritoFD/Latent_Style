from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SB_ROOT = SCRIPT_DIR.parent.parent
WORKSPACE = SB_ROOT.parent


def _run(cmd: list[str]) -> int:
    print("[launch_remote_knee_spatial_carriergate_bodydecoder_qedgegated_train] " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, check=False)
    return int(proc.returncode)


def main() -> int:
    launch = WORKSPACE / "SchrodingerBridge" / "tools" / "experiments" / "launch_remote_wsl_command.py"
    command = [
        sys.executable,
        str(launch),
        "--task-name",
        "knee-spatial-carriergate-bodydecoder-qedgegated-train",
        "--remote-log-path",
        "/mnt/i/Github/Latent_Style/exp/inmortal-exp/knee_spatial_carriergate_bodydecoder_qedgegated_train.log",
        "--remote-wsl-cwd",
        "/mnt/i/Github/Latent_Style",
        "--python-bin",
        "/home/xy/venvs/samam312/bin/python",
        "--sync-path",
        "SchrodingerBridge/src",
        "--sync-path",
        "SchrodingerBridge/configs/aaai2027/inmortal_knee_e13_spatial_carriergate_bodydecoder_qedgegated_seed42_b8a2.json",
        "--sync-path",
        "SchrodingerBridge/docs/experiments/2026-06-09-knee-e13-spatial-carriergate-bodydecoder-qedgegated.md",
        "--verify-python-file",
        "SchrodingerBridge/src/run.py",
        "--verify-python-file",
        "SchrodingerBridge/src/losses.py",
        "--verify-python-file",
        "SchrodingerBridge/src/config_schema.py",
        "--max-prelaunch-memory-mib",
        "1500",
        "--",
        "bash",
        "-lc",
        (
            "set -euo pipefail; "
            "export PYTHONPATH=SchrodingerBridge/src; "
            "/home/xy/venvs/samam312/bin/python "
            "SchrodingerBridge/src/run.py "
            "--config /mnt/i/Github/Latent_Style/SchrodingerBridge/configs/aaai2027/inmortal_knee_e13_spatial_carriergate_bodydecoder_qedgegated_seed42_b8a2.json"
        ),
    ]
    return _run(command)


if __name__ == "__main__":
    raise SystemExit(main())
