from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SB_ROOT = SCRIPT_DIR.parent.parent
WORKSPACE = SB_ROOT.parent
if str(SB_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(SB_ROOT / "src"))

from config_schema import load_config


def _run(cmd: list[str]) -> int:
    print("[launch_remote_round1_family_bestfew_rerun] " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, check=False)
    return int(proc.returncode)


def _epochs_from_handoff(path: Path) -> list[int]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    out: list[int] = []
    for row in rows:
        epoch_name = str(row.get("epoch", "")).strip()
        if epoch_name.startswith("epoch_"):
            try:
                out.append(int(epoch_name.split("_")[-1]))
            except ValueError:
                continue
    return sorted(set(out))


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch remote image-backed bestfew rerun for a round-1 family.")
    parser.add_argument("--config", required=True, help="Workspace-relative config path.")
    parser.add_argument("--handoff-csv", default="", help="Optional local bestfew handoff CSV; used to derive epochs.")
    parser.add_argument("--epochs", type=int, nargs="*", default=None)
    parser.add_argument("--remote-wsl-cwd", default="/mnt/i/Github/Latent_Style")
    parser.add_argument("--remote-python", default="/home/xy/venvs/samam312/bin/python")
    parser.add_argument("--test-dir", default="/mnt/i/wikiart_distinct5_samam_512_classview/test")
    parser.add_argument("--cache-dir", default="/mnt/i/Github/Latent_Style/eval_cache")
    parser.add_argument("--clip-hf-cache-dir", default="/mnt/i/Github/Latent_Style/eval_cache/hf")
    parser.add_argument("--output-subdir", default="full_eval_fresh_localreview")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--vae-decode-batch-size", type=int, default=16)
    parser.add_argument("--target-chunk-size", type=int, default=2)
    args = parser.parse_args()

    epochs = list(args.epochs or [])
    if not epochs and str(args.handoff_csv).strip():
        epochs = _epochs_from_handoff(Path(args.handoff_csv).resolve())
    if not epochs:
        raise ValueError("Provide --epochs or --handoff-csv with at least one epoch.")

    config_rel = Path(args.config)
    cfg = load_config((WORKSPACE / config_rel).resolve())
    run_name = str((cfg.get("ablation") or {}).get("name", config_rel.stem)).strip() or config_rel.stem
    run_dir = str((cfg.get("checkpoint") or {}).get("save_dir", "")).strip()
    if run_dir.startswith("./"):
        run_dir = f"{args.remote_wsl_cwd.rstrip('/')}/{run_dir[2:]}"

    launch = WORKSPACE / "SchrodingerBridge" / "tools" / "experiments" / "launch_remote_wsl_command.py"
    epoch_args = " ".join(str(int(ep)) for ep in epochs)
    command = [
        sys.executable,
        str(launch),
        "--task-name",
        f"round1-{run_name}-bestfew-rerun",
        "--remote-log-path",
        f"{args.remote_wsl_cwd.rstrip('/')}/exp/inmortal-exp/{run_name}_bestfew_rerun.log",
        "--remote-wsl-cwd",
        str(args.remote_wsl_cwd),
        "--python-bin",
        str(args.remote_python),
        "--sync-path",
        "SchrodingerBridge/src",
        "--sync-path",
        "SchrodingerBridge/tools/experiments/rerun_full_eval_for_run.py",
        "--verify-python-file",
        "SchrodingerBridge/src/utils/run_evaluation.py",
        "--max-prelaunch-memory-mib",
        "12000",
        "--no-health-check",
        "--",
        "bash",
        "-lc",
        (
            "set -euo pipefail; "
            "export PYTHONPATH=SchrodingerBridge/src; "
            f"{args.remote_python} SchrodingerBridge/tools/experiments/rerun_full_eval_for_run.py "
            f"--run-dir {run_dir} "
            f"--test-dir {args.test_dir} "
            f"--cache-dir {args.cache_dir} "
            f"--clip-hf-cache-dir {args.clip_hf_cache_dir} "
            f"--batch-size {int(args.batch_size)} "
            f"--vae-decode-batch-size {int(args.vae_decode_batch_size)} "
            f"--target-chunk-size {int(args.target_chunk_size)} "
            f"--output-subdir {args.output_subdir} "
            f"--epochs {epoch_args} "
            "--save-generated-images "
            "--no-save-summary-grid "
            "--skip-existing"
        ),
    ]
    return _run(command)


if __name__ == "__main__":
    raise SystemExit(main())
