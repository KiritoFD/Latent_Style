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

LATENT_SYNC_PATHS = [
    "Related_Works/baseline_pipeline/scripts/run_samam_latent_baseline.py",
    "Related_Works/baseline_pipeline/scripts/generate_samam_latent_eval.py",
    "Related_Works/baseline_pipeline/scripts/run_samam_latent_eval_bundle.py",
    "Related_Works/baseline_pipeline/scripts/run_samst_latent_baseline.py",
    "Related_Works/baseline_pipeline/scripts/generate_samst_latent_eval.py",
    "Related_Works/baseline_pipeline/scripts/run_samst_latent_eval_bundle.py",
    "Related_Works/repos/SaMam/ARCHI/Decoder.py",
    "Related_Works/repos/SaMam/ARCHI/StyleEmbedder.py",
    "Related_Works/repos/SaMam/MODEL/SaMam_model.py",
    "Related_Works/repos/SaMam/TRAIN/lightning_module/latent_dataset.py",
    "Related_Works/repos/SaMam/TRAIN/lightning_module/latent_datamodule.py",
    "Related_Works/repos/SaMam/TRAIN/lightning_module/latent_lightningmodel.py",
    "Related_Works/repos/SaMam/TRAIN/train_SaMam_latent.py",
    "Related_Works/repos/SaMST-main/networks/transfer_net.py",
    "Related_Works/repos/SaMST-main/networks/latent_transfer_net.py",
    "Related_Works/repos/SaMST-main/train_model/train2/train_latent.py",
    "SchrodingerBridge/src/utils/inference.py",
    "SchrodingerBridge/src/utils/run_evaluation.py",
    "SchrodingerBridge/src/utils/targetwise_artfid_summary.py",
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


def _default_budget(method: str, lane: str) -> int:
    if lane == "same-cost":
        return 600
    return 1800 if method == "samam" else 2000


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Launch a Distinct5-512 latent baseline lane on the remote 3060 through the generic host-owned WSL launcher."
    )
    parser.add_argument("--method", choices=["samam", "samst"], required=True)
    parser.add_argument("--lane", choices=["same-cost", "convergence"], required=True)
    parser.add_argument("--remote-workspace-root", default="/mnt/i/Github/Latent_Style")
    parser.add_argument("--python-bin", default="/home/xy/venvs/samam312/bin/python")
    parser.add_argument("--host", default="100.115.18.62")
    parser.add_argument("--port", type=int, default=2222)
    parser.add_argument("--user", default="administrator")
    parser.add_argument("--wsl-distro", default="Ubuntu-26.04")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--checkpoint-interval", type=int, default=50)
    parser.add_argument("--precision", default="32-true", help="SaMAM only.")
    parser.add_argument("--accumulate-grad-batches", type=int, default=1, help="SaMAM only.")
    parser.add_argument("--identity-gradient-checkpointing", type=int, default=1, help="SaMAM only. This is the first low-VRAM lever after batch-size=1.")
    parser.add_argument("--vae-gradient-checkpointing", type=int, default=0, help="SaMAM only. Checkpoint VAE decode so gradients recompute activations instead of storing them.")
    parser.add_argument("--iterations", type=int, default=0, help="SaMAM only. 0 uses the lane default.")
    parser.add_argument("--checkpoint", default="", help="SaMAM only. Resume from a latent Lightning checkpoint.")
    parser.add_argument("--max-steps", type=int, default=0, help="SaMST only. 0 uses the lane default.")
    parser.add_argument("--loss-network-half", type=int, default=0, help="SaMST only. Keep the perceptual loss network in FP32 by default for stability.")
    parser.add_argument("--begin-checkpoint", default="", help="SaMST only. Resume from an interval checkpoint.")
    parser.add_argument("--begin-epoch", type=int, default=0, help="SaMST only. Epoch index paired with --begin-checkpoint.")
    parser.add_argument("--max-prelaunch-memory-mib", type=int, default=1500)
    parser.add_argument("--health-wait-seconds", type=int, default=30)
    parser.add_argument("--max-runtime-memory-mib", type=int, default=11500)
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--no-health-check", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.method}_latent_distinct5_512_{args.lane.replace('-', '')}_{stamp}"
    remote_run_root = (
        f"{args.remote_workspace_root.rstrip('/')}/Related_Works/baseline_pipeline/results/{run_name}"
    )
    remote_launcher_log = f"{remote_run_root}/remote_launcher.log"
    task_name = f"latent-{args.method}-distinct5-{args.lane}-{stamp}"

    command = [args.python_bin]
    if args.method == "samam":
        iterations = int(args.iterations) if int(args.iterations) > 0 else _default_budget(args.method, args.lane)
        command.extend(
            [
                "Related_Works/baseline_pipeline/scripts/run_samam_latent_baseline.py",
                "--dataset",
                "distinct5_512",
                "--out-root",
                remote_run_root,
                "--iterations",
                str(iterations),
                "--batch-size",
                str(int(args.batch_size)),
                "--precision",
                str(args.precision),
                "--checkpoint-every-n-steps",
                str(int(args.checkpoint_interval)),
                "--gradient-checkpointing",
                "1",
                "--identity-gradient-checkpointing",
                str(int(args.identity_gradient_checkpointing)),
                "--vae-gradient-checkpointing",
                str(int(args.vae_gradient_checkpointing)),
                "--limit-val-batches",
                "0",
                "--num-sanity-val-steps",
                "0",
                "--accumulate-grad-batches",
                str(int(args.accumulate_grad_batches)),
                "--val-interval",
                "1000",
            ]
        )
        if str(args.checkpoint).strip():
            command.extend(["--checkpoint", str(args.checkpoint)])
    else:
        max_steps = int(args.max_steps) if int(args.max_steps) > 0 else _default_budget(args.method, args.lane)
        command.extend(
            [
                "Related_Works/baseline_pipeline/scripts/run_samst_latent_baseline.py",
                "--dataset",
                "distinct5_512",
                "--out-root",
                remote_run_root,
                "--epochs",
                "30",
                "--max-steps",
                str(max_steps),
                "--batch-size",
                str(int(args.batch_size)),
                "--checkpoint-interval",
                str(int(args.checkpoint_interval)),
                "--loss-network-half",
                str(int(args.loss_network_half)),
            ]
        )
        if str(args.begin_checkpoint).strip():
            command.extend(
                [
                    "--begin-checkpoint",
                    str(args.begin_checkpoint),
                    "--begin-epoch",
                    str(int(args.begin_epoch)),
                ]
            )

    launcher_cmd = [
        sys.executable,
        str(GENERIC_LAUNCHER),
        "--task-name",
        task_name,
        "--remote-log-path",
        remote_launcher_log,
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
    if args.no_verify:
        launcher_cmd.append("--no-verify")
    if args.no_health_check:
        launcher_cmd.append("--no-health-check")
    if args.dry_run:
        launcher_cmd.append("--dry-run")
    for path in LATENT_SYNC_PATHS:
        launcher_cmd.extend(["--sync-path", path, "--verify-python-file", path])
    launcher_cmd.append("--")
    launcher_cmd.extend(command)

    print(f"task_name={task_name}")
    print(f"run_name={run_name}")
    print(f"remote_run_root={remote_run_root}")
    print(f"remote_launcher_log={remote_launcher_log}")
    print("launcher_cmd=" + " ".join(launcher_cmd))
    result = _run(launcher_cmd)
    sys.stdout.write(result.stdout)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
