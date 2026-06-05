from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SB_ROOT = SCRIPT_DIR.parent.parent
WORKSPACE_ROOT = SB_ROOT.parent
GENERIC_LAUNCHER = SCRIPT_DIR / "launch_remote_wsl_command.py"

SYNC_PATHS = [
    "Related_Works/baseline_pipeline/scripts/prepare_distinct5_img2img_turbo_datasets.py",
    "Related_Works/baseline_pipeline/scripts/run_img2img_turbo_distinct5_smoke.py",
    "Related_Works/repos/cyclegan_turbo/img2img-turbo/src",
    "Related_Works/repos/cyclegan_turbo/img2img-turbo/docs",
    "Related_Works/repos/cyclegan_turbo/img2img-turbo/README.md",
    "Related_Works/repos/cyclegan_turbo/img2img-turbo/requirements.txt",
    "Related_Works/repos/cyclegan_turbo/img2img-turbo/environment.yaml",
]


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
        description="Launch a remote Distinct5 img2img-turbo smoke on the 3060 using the generic host-owned WSL launcher."
    )
    parser.add_argument("--target", default="Early_Renaissance")
    parser.add_argument("--remote-workspace-root", default="/mnt/i/Github/Latent_Style")
    parser.add_argument("--python-bin", default="/home/xy/venvs/samam312/bin/python")
    parser.add_argument("--pretrained-model-name-or-path", default="stabilityai/sd-turbo")
    parser.add_argument("--host", default="100.115.18.62")
    parser.add_argument("--port", type=int, default=2222)
    parser.add_argument("--user", default="administrator")
    parser.add_argument("--wsl-distro", default="Ubuntu-26.04")
    parser.add_argument("--smoke-train-images-per-style", type=int, default=30)
    parser.add_argument("--smoke-test-images-per-style", type=int, default=30)
    parser.add_argument("--max-train-steps", type=int, default=20)
    parser.add_argument("--validation-steps", type=int, default=10)
    parser.add_argument("--checkpointing-steps", type=int, default=10)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--max-prelaunch-memory-mib", type=int, default=1500)
    parser.add_argument("--health-wait-seconds", type=int, default=30)
    parser.add_argument("--max-runtime-memory-mib", type=int, default=11000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    smoke_dataset_root = (
        f"{args.remote_workspace_root.rstrip('/')}/Related_Works/runs/img2img_turbo_distinct5_remote_smoke_datasets_{stamp}"
    )
    smoke_run_root = (
        f"{args.remote_workspace_root.rstrip('/')}/Related_Works/runs/img2img_turbo_distinct5_remote_smoke_{stamp}"
    )
    remote_log = f"{smoke_run_root}/{args.target}/remote_launcher.log"
    task_name = f"img2img-turbo-smoke-{args.target}-{stamp}"
    remote_repo_root = (
        f"{args.remote_workspace_root.rstrip('/')}/Related_Works/repos/cyclegan_turbo/img2img-turbo"
    )
    classview_test = "/mnt/i/wikiart_distinct5_samam_512_classview/test"

    smoke_cmd = (
        "set -euo pipefail; "
        f"{args.python_bin} Related_Works/baseline_pipeline/scripts/prepare_distinct5_img2img_turbo_datasets.py "
        f"--train-root {classview_test} --test-root {classview_test} "
        f"--output-root {smoke_dataset_root} "
        f"--train-images-per-style {int(args.smoke_train_images_per_style)} "
        f"--test-images-per-style {int(args.smoke_test_images_per_style)} "
        "--overwrite --copy-mode copy; "
        f"{args.python_bin} Related_Works/baseline_pipeline/scripts/run_img2img_turbo_distinct5_smoke.py "
        f"--target {args.target} "
        f"--repo-root {remote_repo_root} "
        f"--datasets-root {smoke_dataset_root} "
        f"--run-root {smoke_run_root} "
        f"--python {args.python_bin} "
        f"--main-process-port 29531 "
        "--mixed-precision no "
        f"--pretrained-model-name-or-path {args.pretrained_model_name_or_path} "
        "--train-img-prep resize_512 "
        "--val-img-prep resize_512 "
        f"--train-batch-size {int(args.train_batch_size)} "
        f"--max-train-steps {int(args.max_train_steps)} "
        "--max-train-epochs 100 "
        f"--validation-steps {int(args.validation_steps)} "
        "--validation-num-images 8 "
        f"--checkpointing-steps {int(args.checkpointing_steps)} "
        "--dataloader-num-workers 0 "
        "--report-to none "
        "--run"
    )

    launcher_cmd = [
        sys.executable,
        str(GENERIC_LAUNCHER),
        "--task-name",
        task_name,
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
    ]
    if args.dry_run:
        launcher_cmd.append("--dry-run")
    for path in SYNC_PATHS:
        launcher_cmd.extend(["--sync-path", path])
    launcher_cmd.extend(
        [
            "--verify-python-file",
            "Related_Works/baseline_pipeline/scripts/prepare_distinct5_img2img_turbo_datasets.py",
            "--verify-python-file",
            "Related_Works/baseline_pipeline/scripts/run_img2img_turbo_distinct5_smoke.py",
            "--",
            "bash",
            "-lc",
            smoke_cmd,
        ]
    )
    print(f"task_name={task_name}")
    print(f"smoke_dataset_root={smoke_dataset_root}")
    print(f"smoke_run_root={smoke_run_root}")
    print(f"remote_log={remote_log}")
    print("launcher_cmd=" + " ".join(launcher_cmd))
    result = _run(launcher_cmd)
    sys.stdout.write(result.stdout)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
