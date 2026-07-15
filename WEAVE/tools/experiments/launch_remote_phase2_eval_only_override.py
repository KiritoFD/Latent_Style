from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SB_ROOT = SCRIPT_DIR.parent.parent
WORKSPACE = SB_ROOT.parent
DEFAULT_REMOTE_WSL_CWD = "/mnt/i/Github/Latent_Style"
DEFAULT_REMOTE_PYTHON = "/home/xy/venvs/samam312/bin/python"
DEFAULT_TEST_DIR = "/mnt/i/wikiart_distinct5_samam_512_classview/test"
DEFAULT_CACHE_DIR = "/mnt/i/Github/Latent_Style/eval_cache"
DEFAULT_CLIP_HF_CACHE_DIR = "/mnt/i/Github/Latent_Style/eval_cache/hf"


def _run(cmd: list[str]) -> int:
    print("[launch_remote_phase2_eval_only_override] " + " ".join(str(x) for x in cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(WORKSPACE), check=False)
    return int(proc.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch remote eval-only phase2 inference on an existing checkpoint with a config override.")
    parser.add_argument("--checkpoint", required=True, help="Remote-visible checkpoint path, usually under /mnt/i/...")
    parser.add_argument("--config-override", required=True)
    parser.add_argument("--output-root", default="./exp/inmortal-exp/phase2_eval_only_override", help="Remote output root for the eval run.")
    parser.add_argument("--remote-wsl-cwd", default=DEFAULT_REMOTE_WSL_CWD)
    parser.add_argument("--remote-python", default=DEFAULT_REMOTE_PYTHON)
    parser.add_argument("--test-dir", default=DEFAULT_TEST_DIR)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--clip-hf-cache-dir", default=DEFAULT_CLIP_HF_CACHE_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--task-prefix", default="phase2-eval-only")
    parser.add_argument("--max-prelaunch-memory-mib", type=int, default=1500)
    parser.add_argument("--min-runtime-memory-mib", type=int, default=0)
    parser.add_argument("--max-runtime-memory-mib", type=int, default=10800)
    parser.add_argument("--health-wait-seconds", type=int, default=30)
    parser.add_argument("--force-regen", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-health-check", action="store_true")
    args = parser.parse_args()

    checkpoint = str(args.checkpoint).strip()
    if not checkpoint:
        raise ValueError("--checkpoint is required")

    override = Path(args.config_override)
    override_abs = override if override.is_absolute() else (WORKSPACE / override).resolve()
    override_rel = override_abs.resolve().relative_to(WORKSPACE.resolve())
    override_stem = override_abs.stem

    checkpoint_stem = Path(checkpoint).stem
    run_name = f"phase2_eval_only_{override_stem}_{checkpoint_stem}"
    output = str(Path(str(args.output_root).strip()) / override_stem / checkpoint_stem).replace("\\", "/")

    remote_cmd = [
        str(args.remote_python),
        "SchrodingerBridge/tools/experiments/run_phase2_eval_only_override.py",
        "--checkpoint",
        checkpoint,
        "--config-override",
        f"{str(args.remote_wsl_cwd).rstrip('/')}/{override_rel.as_posix()}",
        "--output",
        output,
        "--test-dir",
        str(args.test_dir),
        "--cache-dir",
        str(args.cache_dir),
        "--clip-hf-cache-dir",
        str(args.clip_hf_cache_dir),
        "--device",
        str(args.device),
    ]
    if int(args.seed) >= 0:
        remote_cmd += ["--seed", str(int(args.seed))]
    if bool(args.force_regen):
        remote_cmd.append("--force-regen")

    launch = SB_ROOT / "tools" / "experiments" / "launch_remote_wsl_command.py"
    cmd = [
        sys.executable,
        str(launch),
        "--task-name",
        f"{str(args.task_prefix).strip()}-{override_stem}-{checkpoint_stem}",
        "--remote-log-path",
        f"{str(args.remote_wsl_cwd).rstrip('/')}/exp/inmortal-exp/{run_name}.log",
        "--remote-wsl-cwd",
        str(args.remote_wsl_cwd),
        "--python-bin",
        str(args.remote_python),
        "--sync-path",
        "SchrodingerBridge/src",
        "--sync-path",
        "SchrodingerBridge/tools/experiments/run_phase2_eval_only_override.py",
        "--sync-path",
        "SchrodingerBridge/tools/experiments/report_remote_experiment_status.py",
        "--sync-path",
        str(override_rel),
        "--verify-python-file",
        "SchrodingerBridge/src/utils/run_evaluation.py",
        "--verify-python-file",
        "SchrodingerBridge/tools/experiments/run_phase2_eval_only_override.py",
        "--max-prelaunch-memory-mib",
        str(int(args.max_prelaunch_memory_mib)),
        "--min-runtime-memory-mib",
        str(int(args.min_runtime_memory_mib)),
        "--max-runtime-memory-mib",
        str(int(args.max_runtime_memory_mib)),
        "--runtime-guard-max-memory-mib",
        str(int(args.max_runtime_memory_mib)),
        "--health-wait-seconds",
        str(int(args.health_wait_seconds)),
    ]
    if bool(args.no_health_check):
        cmd.append("--no-health-check")
    if bool(args.dry_run):
        cmd.append("--dry-run")
    cmd += ["--", *remote_cmd]
    return _run(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
