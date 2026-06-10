from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SB_ROOT = SCRIPT_DIR.parent.parent
WORKSPACE = SB_ROOT.parent


def _run(cmd: list[str]) -> int:
    print("[launch_remote_dualpath_spatial_sinkhorn_introstyle_bestfew_probe] " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, check=False)
    return int(proc.returncode)


def main() -> int:
    launch = WORKSPACE / "SchrodingerBridge" / "tools" / "experiments" / "launch_remote_wsl_command.py"
    run_name = "aaai2027_inmortal_knee_e13_spatial_carriergate_bodydecoder_qedgegated_dualpath_spatial_sinkhorn_seed42_b8a2"
    run_root = f"/mnt/i/Github/Latent_Style/exp/inmortal-exp/{run_name}"
    cached_model = "/mnt/i/Github/Latent_Style/eval_cache/modelscope/stabilityai/stable-diffusion-2-1-base"
    curve_csv = f"{run_root}/full_eval_fresh_localreview/clip_lpips_curve.csv"
    eval_root = f"{run_root}/full_eval_fresh_localreview"
    handoff_csv = f"{run_root}/full_eval_fresh_localreview_bestfew_handoff.csv"
    manifest_csv = f"{run_root}/full_eval_fresh_localreview_introstyle_bestfew_manifest.csv"
    output_csv = f"{run_root}/full_eval_fresh_localreview_introstyle_bestfew_probe.csv"
    output_json = f"{run_root}/full_eval_fresh_localreview_introstyle_bestfew_probe.json"
    command = [
        sys.executable,
        str(launch),
        "--task-name",
        "dualpath-spatial-sinkhorn-introstyle-bestfew-probe",
        "--remote-log-path",
        "/mnt/i/Github/Latent_Style/exp/inmortal-exp/dualpath_spatial_sinkhorn_introstyle_bestfew_probe.log",
        "--remote-wsl-cwd",
        "/mnt/i/Github/Latent_Style",
        "--python-bin",
        "/home/xy/venvs/samam312/bin/python",
        "--sync-path",
        "SchrodingerBridge/src",
        "--sync-path",
        "SchrodingerBridge/tools/eval_introstyle_probe.py",
        "--sync-path",
        "SchrodingerBridge/tools/experiments/build_best_few_handoff.py",
        "--sync-path",
        "SchrodingerBridge/tools/experiments/build_introstyle_manifest_from_handoff.py",
        "--verify-python-file",
        "SchrodingerBridge/src/utils/introstyle_eval.py",
        "--verify-python-file",
        "SchrodingerBridge/tools/eval_introstyle_probe.py",
        "--verify-python-file",
        "SchrodingerBridge/tools/experiments/build_best_few_handoff.py",
        "--verify-python-file",
        "SchrodingerBridge/tools/experiments/build_introstyle_manifest_from_handoff.py",
        "--max-prelaunch-memory-mib",
        "12000",
        "--",
        "bash",
        "-lc",
        (
            "set -euo pipefail; "
            "export PYTHONPATH=SchrodingerBridge/src; "
            "/home/xy/venvs/samam312/bin/python "
            "SchrodingerBridge/tools/experiments/build_best_few_handoff.py "
            f"--curve-csv {curve_csv} "
            f"--run-name {run_name} "
            f"--eval-root {eval_root} "
            f"--output-csv {handoff_csv}; "
            "/home/xy/venvs/samam312/bin/python "
            "SchrodingerBridge/tools/experiments/build_introstyle_manifest_from_handoff.py "
            f"--handoff-csv {handoff_csv} "
            f"--output-csv {manifest_csv} "
            "--method LBM "
            "--label-prefix DualPathSpatialSinkhornBestFew "
            "--source-root /mnt/i/wikiart_distinct5_samam_512_classview/test; "
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
