from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SB_ROOT = SCRIPT_DIR.parent.parent
WORKSPACE = SB_ROOT.parent


def _run(cmd: list[str]) -> int:
    print("[launch_remote_knee_spatial_carriergate_bodydecoder_introstyle_probe] " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, check=False)
    return int(proc.returncode)


def main() -> int:
    launch = WORKSPACE / "SchrodingerBridge" / "tools" / "experiments" / "launch_remote_wsl_command.py"
    run_root = "/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/inmortal-exp/aaai2027_inmortal_knee_e13_spatial_carriergate_bodydecoder_seed42_b8a2"
    eval_root = f"{run_root}/full_eval_fresh_localreview"
    manifest_csv = f"{run_root}/full_eval_fresh_localreview_introstyle_manifest.csv"
    output_csv = f"{run_root}/full_eval_fresh_localreview_introstyle_probe.csv"
    output_json = f"{run_root}/full_eval_fresh_localreview_introstyle_probe.json"
    cached_model = "/mnt/i/Github/Latent_Style/eval_cache/modelscope/stabilityai/stable-diffusion-2-1-base"
    command = [
        sys.executable,
        str(launch),
        "--task-name",
        "knee-spatial-carriergate-bodydecoder-introstyle-probe",
        "--remote-log-path",
        "/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/inmortal-exp/knee_spatial_carriergate_bodydecoder_introstyle_probe.log",
        "--remote-wsl-cwd",
        "/mnt/i/Github/Latent_Style",
        "--python-bin",
        "/home/xy/venvs/samam312/bin/python",
        "--sync-path",
        "SchrodingerBridge/src",
        "--sync-path",
        "SchrodingerBridge/tools/eval_introstyle_probe.py",
        "--sync-path",
        "SchrodingerBridge/tools/experiments/build_epoch_eval_manifest.py",
        "--verify-python-file",
        "SchrodingerBridge/src/utils/introstyle_eval.py",
        "--verify-python-file",
        "SchrodingerBridge/tools/eval_introstyle_probe.py",
        "--verify-python-file",
        "SchrodingerBridge/tools/experiments/build_epoch_eval_manifest.py",
        "--max-prelaunch-memory-mib",
        "12000",
        "--",
        "bash",
        "-lc",
        (
            "set -euo pipefail; "
            "export PYTHONPATH=SchrodingerBridge/src; "
            "/home/xy/venvs/samam312/bin/python "
            "SchrodingerBridge/tools/experiments/build_epoch_eval_manifest.py "
            f"--eval-root {eval_root} "
            f"--output-csv {manifest_csv} "
            "--method LBM "
            "--label-prefix KneeSpatialCarrier "
            "--source-root /mnt/i/wikiart_distinct5_samam_512_classview/test "
            "--require-images; "
            "/home/xy/venvs/samam312/bin/python "
            "SchrodingerBridge/tools/eval_introstyle_probe.py "
            f"--manifest {manifest_csv} "
            "--style-bank-root /mnt/i/wikiart_distinct5_samam_512_classview/test "
            f"--output_csv {output_csv} "
            f"--output_json {output_json} "
            f"--model-id {cached_model} "
            "--batch_size 4 "
            "--bank_limit_per_style 64 "
            "--t 25 "
            "--up_ft_index 1 "
            "--ensemble_size 1"
        ),
    ]
    return _run(command)


if __name__ == "__main__":
    raise SystemExit(main())
