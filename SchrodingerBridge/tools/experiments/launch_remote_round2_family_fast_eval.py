from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SB_ROOT = SCRIPT_DIR.parent.parent
WORKSPACE = SB_ROOT.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SB_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(SB_ROOT / "src"))

from round2_registry import ROUND2_PURE_SDE_SPECS


DEFAULT_MANIFEST = SB_ROOT / "docs" / "experiments" / "round2_pure_sde" / "round2_family_manifest.csv"


def _run(cmd: list[str]) -> int:
    print("[launch_remote_round2_family_fast_eval] " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(WORKSPACE), check=False)
    return int(proc.returncode)


def _family_patience(family_id: str) -> int:
    for spec in ROUND2_PURE_SDE_SPECS:
        if str(spec.family_id).strip() == str(family_id).strip():
            return int(spec.patience)
    return 4


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch the remote fast-eval watcher for a round-2 family run.")
    parser.add_argument("--family-id", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--remote-wsl-cwd", default="/mnt/i/Github/Latent_Style")
    parser.add_argument("--remote-python", default="/home/xy/venvs/samam312/bin/python")
    parser.add_argument("--test-dir", default="/mnt/i/wikiart_distinct5_samam_512_classview/test")
    parser.add_argument("--cache-dir", default="/mnt/i/Github/Latent_Style/eval_cache")
    parser.add_argument("--clip-hf-cache-dir", default="/mnt/i/Github/Latent_Style/eval_cache/hf")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--vae-decode-batch-size", type=int, default=4)
    parser.add_argument("--target-chunk-size", type=int, default=1)
    parser.add_argument("--max-live-memory-mib-to-launch", type=int, default=9800)
    parser.add_argument("--poll-seconds", type=int, default=180)
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    family_id = str(args.family_id).strip()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (WORKSPACE / config_path).resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    run_name = str((payload.get("ablation") or {}).get("name", config_path.stem)).strip() or config_path.stem
    run_dir = str((payload.get("checkpoint") or {}).get("save_dir", "")).strip()
    if run_dir.startswith("./"):
        run_dir = f"{args.remote_wsl_cwd.rstrip('/')}/{run_dir[2:]}"
    manifest_csv = Path(args.manifest_csv).expanduser()
    if not manifest_csv.is_absolute():
        manifest_csv = (WORKSPACE / manifest_csv).resolve()
    launch = SCRIPT_DIR / "launch_remote_wsl_command.py"
    manifest_rel = manifest_csv.resolve().relative_to(WORKSPACE.resolve())
    command = [
        sys.executable,
        str(launch),
        "--task-name",
        f"round2-{family_id}-fast-eval",
        "--remote-log-path",
        f"{args.remote_wsl_cwd.rstrip('/')}/exp/inmortal-exp/{run_name}_fast_eval.log",
        "--remote-wsl-cwd",
        str(args.remote_wsl_cwd),
        "--python-bin",
        str(args.remote_python),
        "--sync-path",
        "SchrodingerBridge/src",
        "--sync-path",
        "SchrodingerBridge/tools/experiments/rerun_full_eval_for_run.py",
        "--sync-path",
        "SchrodingerBridge/tools/experiments/report_round1_convergence.py",
        "--sync-path",
        "SchrodingerBridge/tools/experiments/watch_round1_family_fast_eval.py",
        "--sync-path",
        str(manifest_rel),
        "--verify-python-file",
        "SchrodingerBridge/src/utils/run_evaluation.py",
        "--verify-python-file",
        "SchrodingerBridge/tools/experiments/watch_round1_family_fast_eval.py",
        "--max-prelaunch-memory-mib",
        "12000",
        "--no-health-check",
        "--",
        "bash",
        "-lc",
        (
            "set -euo pipefail; "
            "export PYTHONPATH=SchrodingerBridge/src; "
            f"{args.remote_python} SchrodingerBridge/tools/experiments/watch_round1_family_fast_eval.py "
            f"--python-bin {args.remote_python} "
            f"--run-dir {run_dir} "
            f"--test-dir {args.test_dir} "
            f"--cache-dir {args.cache_dir} "
            f"--clip-hf-cache-dir {args.clip_hf_cache_dir} "
            "--output-subdir full_eval_fast_snapshot "
            f"--batch-size {int(args.batch_size)} "
            f"--vae-decode-batch-size {int(args.vae_decode_batch_size)} "
            f"--target-chunk-size {int(args.target_chunk_size)} "
            f"--max-live-memory-mib-to-launch {int(args.max_live_memory_mib_to_launch)} "
            f"--poll-seconds {int(args.poll_seconds)} "
            f"--patience {int(_family_patience(family_id))} "
            f"--manifest-csv {manifest_rel.as_posix()} "
            f"--family-id {family_id} "
            "--allowed-status running"
        ),
    ]
    return _run(command)


if __name__ == "__main__":
    raise SystemExit(main())
