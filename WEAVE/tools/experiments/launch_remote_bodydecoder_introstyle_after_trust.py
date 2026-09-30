from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SB_ROOT = SCRIPT_DIR.parent.parent
WORKSPACE = SB_ROOT.parent


def _run(cmd: list[str]) -> int:
    print("[launch_remote_bodydecoder_introstyle_after_trust] " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, check=False)
    return int(proc.returncode)


def main() -> int:
    launch = WORKSPACE / "SchrodingerBridge" / "tools" / "experiments" / "launch_remote_wsl_command.py"
    command = [
        sys.executable,
        str(launch),
        "--task-name",
        "bodydecoder-introstyle-after-trust",
        "--remote-log-path",
        "/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/inmortal-exp/bodydecoder_introstyle_after_trust.log",
        "--remote-wsl-cwd",
        "/mnt/i/Github/Latent_Style",
        "--python-bin",
        "/home/xy/venvs/samam312/bin/python",
        "--sync-path",
        "SchrodingerBridge/src",
        "--sync-path",
        "SchrodingerBridge/tools/experiments/run_inmortal_posttrain_eval_when_done.py",
        "--sync-path",
        "SchrodingerBridge/tools/experiments/rerun_full_eval_for_run.py",
        "--sync-path",
        "SchrodingerBridge/docs/experiments/2026-06-09-bodydecoder-clean-introstyle-rerun.md",
        "--sync-path",
        "SchrodingerBridge/docs/experiments/2026-06-09-introstyle-runtime-decoupling-fix.md",
        "--verify-python-file",
        "SchrodingerBridge/src/utils/run_evaluation.py",
        "--verify-python-file",
        "SchrodingerBridge/src/utils/inference.py",
        "--verify-python-file",
        "SchrodingerBridge/src/utils/introstyle_eval.py",
        "--verify-python-file",
        "SchrodingerBridge/tools/experiments/run_inmortal_posttrain_eval_when_done.py",
        "--verify-python-file",
        "SchrodingerBridge/tools/experiments/rerun_full_eval_for_run.py",
        "--max-prelaunch-memory-mib",
        "12000",
        "--no-health-check",
        "--",
        "bash",
        "-lc",
        (
            "set -euo pipefail; "
            "export PYTHONPATH=SchrodingerBridge/src; "
            "/home/xy/venvs/samam312/bin/python "
            "SchrodingerBridge/tools/experiments/run_inmortal_posttrain_eval_when_done.py "
            "--python-bin /home/xy/venvs/samam312/bin/python "
            "--run-dir /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/inmortal-exp/aaai2027_inmortal_hold4mid_e8_spatial_carriergate_bodydecoder_seed42_b8a2 "
            "--train-pattern aaai2027_inmortal_xpred_kmanifold_pattn_anisostokes_queue_trust_from_e13_seed42_b8a2 "
            "--test-dir /mnt/i/wikiart_distinct5_samam_512_classview/test "
            "--cache-dir /mnt/i/Github/Latent_Style/eval_cache "
            "--clip-hf-cache-dir /mnt/i/Github/Latent_Style/eval_cache/hf "
            "--batch-size 4 "
            "--vae-decode-batch-size 16 "
            "--target-chunk-size 2 "
            "--code-root mainline "
            "--output-subdir full_eval_introstyle_clean_v3 "
            "--epochs 8 12 "
            "--eval-enable-introstyle "
            "--introstyle-style-bank-root /mnt/i/wikiart_distinct5_samam_512_classview/test "
            "--introstyle-modelscope-id stabilityai/stable-diffusion-2-1-base "
            "--introstyle-modelscope-cache-dir /mnt/i/Github/Latent_Style/eval_cache/modelscope "
            "--introstyle-bank-limit-per-style 64 "
            "--introstyle-batch-size 2 "
            "--introstyle-topk 8 "
            "--introstyle-t 25 "
            "--introstyle-up-ft-index 1 "
            "--introstyle-ensemble-size 1 "
            "--refresh-stage-summary "
            "--refresh-epoch-table "
            "--poll-seconds 30 "
            "--max-wait-seconds 43200"
        ),
    ]
    return _run(command)


if __name__ == "__main__":
    raise SystemExit(main())
