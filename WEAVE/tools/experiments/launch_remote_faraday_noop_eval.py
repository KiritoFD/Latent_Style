from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SB_ROOT = SCRIPT_DIR.parent.parent
WORKSPACE_ROOT = SB_ROOT.parent
GENERIC_LAUNCHER = SCRIPT_DIR / "launch_remote_wsl_command.py"


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Launch a split-local IDT/no-op packet on the remote owner surface by materializing unchanged images and evaluating them with the standard evaluator in reuse_generated mode."
    )
    parser.add_argument("--split-slug", required=True)
    parser.add_argument("--remote-splits-root", default="/mnt/i/wikiart_faraday_splits")
    parser.add_argument("--remote-workspace-root", default="/mnt/i/Github/Latent_Style")
    parser.add_argument("--python-bin", default="/home/xy/venvs/samam312/bin/python")
    parser.add_argument("--host", default="100.115.18.62")
    parser.add_argument("--port", type=int, default=2222)
    parser.add_argument("--user", default="administrator")
    parser.add_argument("--wsl-distro", default="Ubuntu-26.04")
    parser.add_argument("--health-wait-seconds", type=int, default=30)
    parser.add_argument("--max-prelaunch-memory-mib", type=int, default=1500)
    parser.add_argument("--max-runtime-memory-mib", type=int, default=11500)
    parser.add_argument("--max-src-samples", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    remote_split_root = f"{args.remote_splits_root.rstrip('/')}/{args.split_slug}"
    remote_test_dir = f"{remote_split_root}/classview/test"
    remote_output = (
        f"{args.remote_workspace_root.rstrip('/')}/SchrodingerBridge/exp/"
        f"{args.split_slug}_idt_5x5"
    )
    remote_log = (
        f"{args.remote_workspace_root.rstrip('/')}/SchrodingerBridge/_codex_tmp/"
        f"{args.split_slug}_idt_eval.log"
    )

    noop_cmd = [
        args.python_bin,
        "SchrodingerBridge/tools/experiments/materialize_noop_eval_images.py",
        "--test-dir",
        remote_test_dir,
        "--output-dir",
        remote_output,
        "--max-src-samples",
        str(int(args.max_src_samples)),
    ]
    eval_cmd = [
        args.python_bin,
        "SchrodingerBridge/src/utils/run_evaluation.py",
        "--output",
        remote_output,
        "--test_dir",
        remote_test_dir,
        "--cache_dir",
        f"{args.remote_workspace_root.rstrip('/')}/eval_cache",
        "--clip_hf_cache_dir",
        f"{args.remote_workspace_root.rstrip('/')}/eval_cache/hf",
        "--profile_timing",
        "--reuse_generated",
        "--eval_enable_art_fid",
    ]
    chained = " && ".join(
        [
            "cd /mnt/i/Github/Latent_Style",
            " ".join(noop_cmd),
            " ".join(eval_cmd),
        ]
    )

    launcher_cmd = [
        sys.executable,
        str(GENERIC_LAUNCHER),
        "--task-name",
        f"faraday-idt-{args.split_slug}",
        "--remote-log-path",
        remote_log,
        "--remote-wsl-cwd",
        args.remote_workspace_root,
        "--remote-workspace-root",
        args.remote_workspace_root,
        "--python-bin",
        args.python_bin,
        "--host",
        args.host,
        "--port",
        str(int(args.port)),
        "--user",
        args.user,
        "--wsl-distro",
        args.wsl_distro,
        "--max-prelaunch-memory-mib",
        str(int(args.max_prelaunch_memory_mib)),
        "--health-wait-seconds",
        str(int(args.health_wait_seconds)),
        "--max-runtime-memory-mib",
        str(int(args.max_runtime_memory_mib)),
        "--sync-path",
        "SchrodingerBridge/src",
        "--sync-path",
        "SchrodingerBridge/tools/experiments/materialize_noop_eval_images.py",
        "--verify-python-file",
        "SchrodingerBridge/src/config_schema.py",
        "--verify-python-file",
        "SchrodingerBridge/tools/experiments/materialize_noop_eval_images.py",
    ]
    if args.dry_run:
        launcher_cmd.append("--dry-run")
    launcher_cmd.extend(["--", "bash", "-lc", chained])

    result = _run(launcher_cmd)
    sys.stdout.write(result.stdout)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
