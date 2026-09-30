from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SB_ROOT = SCRIPT_DIR.parent.parent
WORKSPACE = SB_ROOT.parent


def _run(cmd: list[str]) -> int:
    print("[launch_remote_introstyle_page1_probe] " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, check=False)
    return int(proc.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Launch the Distinct5 page-1 IntroStyle shortlist probe on the reviewed remote WSL surface."
    )
    parser.add_argument("--python-bin", default="/home/xy/venvs/samam312/bin/python")
    parser.add_argument("--task-name", default="introstyle-page1-remote-probe")
    parser.add_argument("--remote-workspace-root", default="/mnt/i/Github/Latent_Style")
    parser.add_argument("--remote-style-bank-root", default="/mnt/i/wikiart_distinct5_samam_512_classview/test")
    parser.add_argument(
        "--remote-model-id",
        default="",
    )
    parser.add_argument("--remote-modelscope-id", default="stabilityai/stable-diffusion-2-1-base")
    parser.add_argument("--remote-modelscope-cache-dir", default="/mnt/i/Github/Latent_Style/eval_cache/modelscope")
    parser.add_argument("--sample-rows", type=int, default=20)
    parser.add_argument("--bank-limit-per-style", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--ensemble-size", type=int, default=1)
    parser.add_argument("--t", type=int, default=25)
    parser.add_argument("--up-ft-index", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    remote_root = str(args.remote_workspace_root).rstrip("/")
    remote_sb = f"{remote_root}/SchrodingerBridge"
    remote_out = f"{remote_sb}/aaai2027/introstyle_page1"
    remote_log = f"{remote_out}/remote_introstyle_probe.log"

    command = [
        str(WORKSPACE / "SchrodingerBridge" / "tools" / "experiments" / "launch_remote_wsl_command.py"),
        "--task-name",
        str(args.task_name),
        "--remote-log-path",
        remote_log,
        "--remote-wsl-cwd",
        remote_root,
        "--python-bin",
        str(args.python_bin),
        "--sync-path",
        "SchrodingerBridge/tools/eval_introstyle_probe.py",
        "--sync-path",
        "SchrodingerBridge/src/utils/introstyle_eval.py",
        "--sync-path",
        "SchrodingerBridge/aaai2027/build_introstyle_page1_shortlist.py",
        "--sync-path",
        "SchrodingerBridge/aaai2027/introstyle_page1/staging",
        "--verify-python-file",
        "SchrodingerBridge/tools/eval_introstyle_probe.py",
        "--verify-python-file",
        "SchrodingerBridge/src/utils/introstyle_eval.py",
        "--verify-python-file",
        "SchrodingerBridge/aaai2027/build_introstyle_page1_shortlist.py",
        "--no-health-check",
        "--max-prelaunch-memory-mib",
        "5000",
    ]
    if args.dry_run:
        command.append("--dry-run")
    command.extend(
        [
            "--",
            "bash",
            "-lc",
            (
                "set -euo pipefail; "
                "export PYTHONPATH=SchrodingerBridge/src; "
                f"{args.python_bin} SchrodingerBridge/aaai2027/build_introstyle_page1_shortlist.py --sample-rows {int(args.sample_rows)}; "
                f"{args.python_bin} SchrodingerBridge/tools/eval_introstyle_probe.py "
                f"--manifest {remote_out}/introstyle_page1_manifest.csv "
                f"--style-bank-root {args.remote_style_bank_root} "
                f"--output_csv {remote_out}/introstyle_page1_probe.csv "
                f"--output_json {remote_out}/introstyle_page1_probe.json "
                f"{('--model-id ' + str(args.remote_model_id) + ' ') if str(args.remote_model_id).strip() else ''}"
                f"--modelscope-id {args.remote_modelscope_id} "
                f"--modelscope-cache-dir {args.remote_modelscope_cache_dir} "
                f"--batch_size {int(args.batch_size)} "
                f"--bank_limit_per_style {int(args.bank_limit_per_style)} "
                f"--t {int(args.t)} "
                f"--up_ft_index {int(args.up_ft_index)} "
                f"--ensemble_size {int(args.ensemble_size)}; "
                f"{args.python_bin} SchrodingerBridge/aaai2027/build_introstyle_page1_shortlist.py "
                f"--sample-rows {int(args.sample_rows)} "
                f"--probe-results-csv {remote_out}/introstyle_page1_probe.csv"
            ),
        ]
    )
    return _run([sys.executable, *command])


if __name__ == "__main__":
    raise SystemExit(main())
