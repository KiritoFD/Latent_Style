from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SB_ROOT = SCRIPT_DIR.parent.parent
WORKSPACE = SB_ROOT.parent


def _run(cmd: list[str]) -> int:
    print("[launch_remote_dualpath_spatial_sinkhorn_imagebacked_bestfew_rerun] " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, check=False)
    return int(proc.returncode)


def main() -> int:
    launch = WORKSPACE / "SchrodingerBridge" / "tools" / "experiments" / "launch_remote_wsl_command.py"
    run_root = "/mnt/i/Github/Latent_Style/exp/inmortal-exp/aaai2027_inmortal_knee_e13_spatial_carriergate_bodydecoder_qedgegated_dualpath_spatial_sinkhorn_seed42_b8a2"
    command = [
        sys.executable,
        str(launch),
        "--task-name",
        "dualpath-spatial-sinkhorn-imagebacked-bestfew-rerun",
        "--remote-log-path",
        "/mnt/i/Github/Latent_Style/exp/inmortal-exp/dualpath_spatial_sinkhorn_imagebacked_bestfew_rerun.log",
        "--remote-wsl-cwd",
        "/mnt/i/Github/Latent_Style",
        "--python-bin",
        "/home/xy/venvs/samam312/bin/python",
        "--sync-path",
        "SchrodingerBridge/src",
        "--sync-path",
        "SchrodingerBridge/tools/experiments/rerun_full_eval_for_run.py",
        "--verify-python-file",
        "SchrodingerBridge/src/utils/run_evaluation.py",
        "--verify-python-file",
        "SchrodingerBridge/tools/experiments/rerun_full_eval_for_run.py",
        "--max-prelaunch-memory-mib",
        "7000",
        "--",
        "bash",
        "-lc",
        (
            "set -euo pipefail; "
            "export PYTHONPATH=SchrodingerBridge/src; "
            "/home/xy/venvs/samam312/bin/python "
            "SchrodingerBridge/tools/experiments/rerun_full_eval_for_run.py "
            f"--run-dir {run_root} "
            "--python-bin /home/xy/venvs/samam312/bin/python "
            "--test-dir /mnt/i/wikiart_distinct5_samam_512_classview/test "
            "--cache-dir /mnt/i/Github/Latent_Style/eval_cache "
            "--clip-hf-cache-dir /mnt/i/Github/Latent_Style/eval_cache/hf "
            "--batch-size 4 "
            "--vae-decode-batch-size 4 "
            "--target-chunk-size 2 "
            "--profile-timing "
            "--save-generated-images "
            "--no-save-summary-grid "
            "--code-root mainline "
            "--output-subdir full_eval_imagebacked_bestfew "
            "--epochs 1 9"
        ),
    ]
    return _run(command)


if __name__ == "__main__":
    raise SystemExit(main())
