from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SB_ROOT = SCRIPT_DIR.parent.parent
WORKSPACE = SB_ROOT.parent

PYTHON_BIN = "/home/xy/venvs/samam312/bin/python"
REMOTE_ROOT = "/mnt/i/Github/Latent_Style"
REMOTE_SB = f"{REMOTE_ROOT}/SchrodingerBridge"

SYNC_PATHS = [
    "SchrodingerBridge/src/config_schema.py",
    "SchrodingerBridge/src/model.py",
    "SchrodingerBridge/src/model620.py",
    "SchrodingerBridge/src/blocks620.py",
    "SchrodingerBridge/src/style_encoder620.py",
    "SchrodingerBridge/src/losses620.py",
    "SchrodingerBridge/src/utils/inference.py",
    "SchrodingerBridge/src/utils/run_evaluation.py",
    "SchrodingerBridge/tools/experiments/collect_round2_eval_curve.py",
    "SchrodingerBridge/tools/experiments/run_remote_620_eval_sweep.sh",
    "SchrodingerBridge/tools/experiments/launch_remote_620_eval_sweep.py",
    "SchrodingerBridge/exp/phase616_live_dashboard/sync_phase616_live_dashboard.py",
]

VERIFY_PYTHON_FILES = [
    "SchrodingerBridge/src/config_schema.py",
    "SchrodingerBridge/src/model.py",
    "SchrodingerBridge/src/model620.py",
    "SchrodingerBridge/src/blocks620.py",
    "SchrodingerBridge/src/style_encoder620.py",
    "SchrodingerBridge/src/losses620.py",
    "SchrodingerBridge/src/utils/inference.py",
    "SchrodingerBridge/src/utils/run_evaluation.py",
    "SchrodingerBridge/tools/experiments/collect_round2_eval_curve.py",
    "SchrodingerBridge/tools/experiments/launch_remote_620_eval_sweep.py",
]


def _run(cmd: list[str]) -> int:
    print("[launch_remote_620_eval_sweep] " + " ".join(shlex.quote(str(x)) for x in cmd), flush=True)
    return int(subprocess.run(cmd, cwd=str(WORKSPACE), check=False).returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch eval-only NFE/sigma sweeps for a 620 checkpoint.")
    parser.add_argument(
        "--checkpoint",
        default=f"{REMOTE_ROOT}/exp/620_spatial_bridge/620_swd12_sigma002_nfe8_b80/epoch_0008.pt",
        help="Remote WSL checkpoint path.",
    )
    parser.add_argument("--run-prefix", default="620_swd12_epoch0008")
    parser.add_argument("--nfe-list", default="4 8 16")
    parser.add_argument("--sigma-list", default="0.0 0.02")
    parser.add_argument("--task-name", default="620_eval_sweep_swd12_e8")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    remote_log = f"{REMOTE_SB}/exp/620_spatial_bridge/{args.task_name}.remote.log"
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "launch_remote_wsl_command.py"),
        "--task-name",
        str(args.task_name),
        "--remote-log-path",
        remote_log,
        "--remote-wsl-cwd",
        REMOTE_ROOT,
        "--remote-workspace-root",
        REMOTE_ROOT,
        "--python-bin",
        PYTHON_BIN,
        "--max-prelaunch-memory-mib",
        "1500",
        "--runtime-guard-max-memory-mib",
        "11264",
        "--runtime-guard-poll-seconds",
        "15",
    ]
    for path in SYNC_PATHS:
        cmd.extend(["--sync-path", path])
    for path in VERIFY_PYTHON_FILES:
        cmd.extend(["--verify-python-file", path])
    if args.dry_run:
        cmd.append("--dry-run")
    remote_cmd = [
        "bash",
        "SchrodingerBridge/tools/experiments/run_remote_620_eval_sweep.sh",
        "--checkpoint",
        str(args.checkpoint),
        "--run-prefix",
        str(args.run_prefix),
        "--nfe-list",
        str(args.nfe_list),
        "--sigma-list",
        str(args.sigma_list),
    ]
    cmd.extend(["--", *remote_cmd])
    return _run(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
