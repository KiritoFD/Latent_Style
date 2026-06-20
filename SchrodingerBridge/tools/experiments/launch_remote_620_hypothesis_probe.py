from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SB_ROOT = SCRIPT_DIR.parent.parent
WORKSPACE = SB_ROOT.parent

PYTHON_BIN = "/home/xy/venvs/samam312/bin/python"
REMOTE_ROOT = "/mnt/i/Github/Latent_Style"
REMOTE_SB = f"{REMOTE_ROOT}/SchrodingerBridge"

SYNC_PATHS = [
    "SchrodingerBridge/src/config_schema.py",
    "SchrodingerBridge/src/model.py",
    "SchrodingerBridge/src/model620.py",
    "SchrodingerBridge/src/blocks620.py",
    "SchrodingerBridge/src/style_encoder620.py",
    "SchrodingerBridge/src/utils/dataset.py",
    "SchrodingerBridge/src/utils/inference.py",
    "SchrodingerBridge/src/utils/training.py",
    "SchrodingerBridge/tools/probe_620_hypothesis_metrics.py",
    "SchrodingerBridge/tools/experiments/launch_remote_620_hypothesis_probe.py",
    "SchrodingerBridge/configs/620_spatial_bridge_targetlinear.json",
    "SchrodingerBridge/configs/620_spatial_bridge_targetlinear_endpointlowhigh.json",
    "SchrodingerBridge/configs/620_spatial_bridge_targetlinear_endpointstylehead.json",
    "SchrodingerBridge/docs/620/fog/README.md",
]

VERIFY_PYTHON_FILES = [
    "SchrodingerBridge/src/config_schema.py",
    "SchrodingerBridge/src/model.py",
    "SchrodingerBridge/src/model620.py",
    "SchrodingerBridge/src/blocks620.py",
    "SchrodingerBridge/src/style_encoder620.py",
    "SchrodingerBridge/src/utils/dataset.py",
    "SchrodingerBridge/src/utils/inference.py",
    "SchrodingerBridge/src/utils/training.py",
    "SchrodingerBridge/tools/probe_620_hypothesis_metrics.py",
    "SchrodingerBridge/tools/experiments/launch_remote_620_hypothesis_probe.py",
]


def _run(cmd: list[str]) -> int:
    print("[launch_remote_620_hypothesis_probe] " + " ".join(shlex.quote(str(x)) for x in cmd), flush=True)
    return int(subprocess.run(cmd, cwd=str(WORKSPACE), check=False).returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch the 620 whitening hypothesis probe on remote 3060 WSL.")
    parser.add_argument(
        "--checkpoint",
        default=f"{REMOTE_ROOT}/exp/620_spatial_bridge/620_targetlinear_swd8_sigma002_nfe8_b64/epoch_0008.pt",
    )
    parser.add_argument(
        "--config",
        default="SchrodingerBridge/configs/620_spatial_bridge_targetlinear.json",
    )
    parser.add_argument(
        "--output-dir",
        default=f"{REMOTE_SB}/docs/620/fog/remote_probe_targetlinear_hypothesis_epoch8",
    )
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--state-mode", choices=["training_linear", "content_only"], default="training_linear")
    parser.add_argument("--task-name", default="620_targetlinear_hypothesis_probe_epoch8")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    remote_log = f"{REMOTE_SB}/exp/620_spatial_bridge/{args.task_name}.remote.log"
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "launch_remote_wsl_command.py"),
        "--task-name",
        str(args.task_name),
        "--remote-log-path",
        remote_log,
        "--remote-wsl-cwd",
        REMOTE_ROOT,
        "--remote-workspace-root",
        REMOTE_ROOT,
        "--python-bin",
        PYTHON_BIN,
        "--max-prelaunch-memory-mib",
        "1500",
        "--runtime-guard-max-memory-mib",
        "12288",
        "--runtime-guard-poll-seconds",
        "15",
    ]
    for path in SYNC_PATHS:
        cmd.extend(["--sync-path", path])
    for path in VERIFY_PYTHON_FILES:
        cmd.extend(["--verify-python-file", path])
    if args.dry_run:
        cmd.append("--dry-run")

    remote_cmd = [
        PYTHON_BIN,
        "SchrodingerBridge/tools/probe_620_hypothesis_metrics.py",
        "--config",
        str(args.config),
        "--checkpoint",
        str(args.checkpoint),
        "--output-dir",
        str(args.output_dir),
        "--sample-count",
        str(int(args.sample_count)),
        "--state-mode",
        str(args.state_mode),
        "--data-root-override",
        "/mnt/i/wikiart_distinct5_samam_512_latents_ema/train",
        "--latent-cache-dir-override",
        "/mnt/i/wikiart_distinct5_samam_512_latents_ema/train/.latent_cache",
        "--pairing-cache-override",
        "/mnt/i/wikiart_distinct5_samam_512_latents_ema/train/.latent_cache/dino_pairing_top8.pt",
        "--dino-cache-override",
        "/mnt/i/Github/Latent_Style/eval_cache/offline_pairing/dinov2_wikiart_distinct5_samam_512_train_cache.pt",
    ]
    cmd.extend(["--", *remote_cmd])
    return _run(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
