from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SB_ROOT = SCRIPT_DIR.parent.parent
WORKSPACE = SB_ROOT.parent
DEFAULT_OVERRIDE = "SchrodingerBridge/configs/aaai2027/phase2_eval_pc_solver_xpred_pattn.json"
DEFAULT_REMOTE_WSL_CWD = "/mnt/i/Github/Latent_Style"
DEFAULT_REMOTE_PYTHON = "/home/xy/venvs/samam312/bin/python"
DEFAULT_TEST_DIR = "/mnt/i/wikiart_distinct5_samam_512_classview/test"
DEFAULT_CACHE_DIR = "/mnt/i/Github/Latent_Style/eval_cache"
DEFAULT_CLIP_HF_CACHE_DIR = "/mnt/i/Github/Latent_Style/eval_cache/hf"


def _run(cmd: list[str]) -> int:
    print("[launch_remote_phase2_eval_only_pc_solver] " + " ".join(str(x) for x in cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(WORKSPACE), check=False)
    return int(proc.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch remote eval-only phase2 solver_pc review on an existing checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Remote-visible checkpoint path, usually under /mnt/i/...")
    parser.add_argument("--output-root", default="./exp/inmortal-exp/phase2_eval_only_pc_solver", help="Remote output root for the eval run.")
    parser.add_argument("--config-override", default=DEFAULT_OVERRIDE)
    parser.add_argument("--remote-wsl-cwd", default=DEFAULT_REMOTE_WSL_CWD)
    parser.add_argument("--remote-python", default=DEFAULT_REMOTE_PYTHON)
    parser.add_argument("--test-dir", default=DEFAULT_TEST_DIR)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--clip-hf-cache-dir", default=DEFAULT_CLIP_HF_CACHE_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--task-prefix", default="phase2-pc-eval")
    parser.add_argument("--max-prelaunch-memory-mib", type=int, default=1500)
    parser.add_argument("--min-runtime-memory-mib", type=int, default=0)
    parser.add_argument("--max-runtime-memory-mib", type=int, default=10800)
    parser.add_argument("--health-wait-seconds", type=int, default=30)
    parser.add_argument("--force-regen", action="store_true")
    args = parser.parse_args()

    checkpoint = str(args.checkpoint).strip()
    if not checkpoint:
        raise ValueError("--checkpoint is required")

    override = Path(args.config_override)
    override_abs = override if override.is_absolute() else (WORKSPACE / override).resolve()
    override_rel = override_abs.resolve().relative_to(WORKSPACE.resolve())

    checkpoint_stem = Path(checkpoint).stem
    run_name = f"phase2_eval_only_pc_solver_{checkpoint_stem}"
    output = str(Path(str(args.output_root).strip()) / checkpoint_stem).replace("\\", "/")

    remote_cmd = [
        str(args.remote_python),
        "SchrodingerBridge/tools/experiments/run_phase2_eval_only_pc_solver.py",
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
    if bool(args.force_regen):
        remote_cmd.append("--force-regen")

    launch = SB_ROOT / "tools" / "experiments" / "launch_remote_wsl_command.py"
    cmd = [
        sys.executable,
        str(launch),
        "--task-name",
        f"{str(args.task_prefix).strip()}-{checkpoint_stem}",
        "--remote-log-path",
        f"{str(args.remote_wsl_cwd).rstrip('/')}/exp/inmortal-exp/{run_name}.log",
        "--remote-wsl-cwd",
        str(args.remote_wsl_cwd),
        "--python-bin",
        str(args.remote_python),
        "--sync-path",
        "SchrodingerBridge/src",
        "--sync-path",
        "SchrodingerBridge/tools/experiments/run_phase2_eval_only_pc_solver.py",
        "--sync-path",
        "SchrodingerBridge/tools/experiments/report_remote_experiment_status.py",
        "--sync-path",
        str(override_rel),
        "--verify-python-file",
        "SchrodingerBridge/src/utils/run_evaluation.py",
        "--verify-python-file",
        "SchrodingerBridge/tools/experiments/run_phase2_eval_only_pc_solver.py",
        "--max-prelaunch-memory-mib",
        str(int(args.max_prelaunch_memory_mib)),
        "--min-runtime-memory-mib",
        str(int(args.min_runtime_memory_mib)),
        "--max-runtime-memory-mib",
        str(int(args.max_runtime_memory_mib)),
        "--health-wait-seconds",
        str(int(args.health_wait_seconds)),
        "--",
        *remote_cmd,
    ]
    return _run(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
