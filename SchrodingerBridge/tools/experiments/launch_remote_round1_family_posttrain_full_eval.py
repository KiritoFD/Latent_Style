from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SB_ROOT = SCRIPT_DIR.parent.parent
WORKSPACE = SB_ROOT.parent


def _run(cmd: list[str]) -> int:
    print("[launch_remote_round1_family_posttrain_full_eval] " + " ".join(str(x) for x in cmd), flush=True)
    proc = subprocess.run(cmd, check=False)
    return int(proc.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch a remote post-train full-eval watcher for a round-1 family run.")
    parser.add_argument("--config", required=True, help="Workspace-relative config path.")
    parser.add_argument("--remote-wsl-cwd", default="/mnt/i/Github/Latent_Style")
    parser.add_argument("--remote-python", default="/home/xy/venvs/samam312/bin/python")
    parser.add_argument("--test-dir", default="/mnt/i/wikiart_distinct5_samam_512_classview/test")
    parser.add_argument("--cache-dir", default="/mnt/i/Github/Latent_Style/eval_cache")
    parser.add_argument("--clip-hf-cache-dir", default="/mnt/i/Github/Latent_Style/eval_cache/hf")
    parser.add_argument("--output-subdir", default="full_eval_fast_snapshot_remote")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--max-wait-seconds", type=int, default=43200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--vae-decode-batch-size", type=int, default=16)
    parser.add_argument("--target-chunk-size", type=int, default=2)
    args = parser.parse_args()

    config_rel = Path(args.config)
    config_abs = (WORKSPACE / config_rel).resolve()
    payload = json.loads(config_abs.read_text(encoding="utf-8"))
    run_name = str((payload.get("ablation") or {}).get("name", config_abs.stem)).strip() or config_abs.stem
    run_dir = str((payload.get("checkpoint") or {}).get("save_dir", "")).strip()
    if run_dir.startswith("./"):
        run_dir = f"{args.remote_wsl_cwd.rstrip('/')}/{run_dir[2:]}"

    launch = WORKSPACE / "SchrodingerBridge" / "tools" / "experiments" / "launch_remote_wsl_command.py"
    command = [
        sys.executable,
        str(launch),
        "--task-name",
        f"round1-{run_name}-posttrain-full-eval",
        "--remote-log-path",
        f"{args.remote_wsl_cwd.rstrip('/')}/exp/inmortal-exp/{run_name}_posttrain_full_eval.log",
        "--remote-wsl-cwd",
        str(args.remote_wsl_cwd),
        "--python-bin",
        str(args.remote_python),
        "--sync-path",
        "SchrodingerBridge/src",
        "--sync-path",
        "SchrodingerBridge/tools/experiments/infer_fresh_epochs_from_latest_training_log.py",
        "--sync-path",
        "SchrodingerBridge/tools/experiments/run_inmortal_posttrain_eval_when_done.py",
        "--sync-path",
        "SchrodingerBridge/tools/experiments/rerun_full_eval_for_run.py",
        "--sync-path",
        "SchrodingerBridge/tools/experiments/report_round1_convergence.py",
        "--sync-path",
        "SchrodingerBridge/tools/experiments/build_clip_lpips_curve_from_eval_root.py",
        "--verify-python-file",
        "SchrodingerBridge/src/utils/run_evaluation.py",
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
            f"{args.remote_python} SchrodingerBridge/tools/experiments/run_inmortal_posttrain_eval_when_done.py "
            f"--python-bin {args.remote_python} "
            f"--run-dir {run_dir} "
            f"--train-pattern {run_name} "
            f"--test-dir {args.test_dir} "
            f"--cache-dir {args.cache_dir} "
            f"--clip-hf-cache-dir {args.clip_hf_cache_dir} "
            f"--batch-size {int(args.batch_size)} "
            f"--vae-decode-batch-size {int(args.vae_decode_batch_size)} "
            f"--target-chunk-size {int(args.target_chunk_size)} "
            "--code-root mainline "
            f"--output-subdir {args.output_subdir} "
            f"--poll-seconds {int(args.poll_seconds)} "
            f"--max-wait-seconds {int(args.max_wait_seconds)} "
            "--no-save-summary-grid "
            "--no-save-generated-images "
            "--no-eval-enable-introstyle"
        ),
    ]
    return _run(command)


if __name__ == "__main__":
    raise SystemExit(main())
