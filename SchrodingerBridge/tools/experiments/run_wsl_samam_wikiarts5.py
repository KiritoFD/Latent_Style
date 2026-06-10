from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SB_ROOT = SCRIPT_DIR.parent.parent
WORKSPACE = SB_ROOT.parent
DEFAULT_WSL_DATA_ROOT = "/mnt/f/wikiarts_5_full_notest/train"
DEFAULT_WSL_PIPELINE_ROOT = "/mnt/g/GitHub/Latent_Style/Related_Works/baseline_pipeline"


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch a local-WSL SaMAM RGB reproduction run on the new wikiarts-5 dataset.")
    parser.add_argument("--wsl-distro", default="f")
    parser.add_argument("--wsl-python", default="python3")
    parser.add_argument("--content-root", default=DEFAULT_WSL_DATA_ROOT)
    parser.add_argument("--style-root", default=DEFAULT_WSL_DATA_ROOT)
    parser.add_argument("--out-root", default="")
    parser.add_argument("--iterations", type=int, default=200000)
    parser.add_argument("--val-interval", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--train-image-size", type=int, default=256)
    parser.add_argument("--train-crop-size", type=int, default=256)
    parser.add_argument("--eval-image-size", type=int, default=256)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--precision", default="32-true")
    parser.add_argument("--mamba-from-trion", type=int, default=1)
    parser.add_argument("--gradient-checkpointing", type=int, default=1)
    parser.add_argument("--identity-gradient-checkpointing", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", type=int, default=0)
    parser.add_argument("--limit-val-batches", type=float, default=0.2)
    parser.add_argument("--num-sanity-val-steps", type=int, default=0)
    parser.add_argument("--accumulate-grad-batches", type=int, default=1)
    parser.add_argument("--checkpoint-every-n-steps", type=int, default=500)
    parser.add_argument("--stdout-log", required=True)
    parser.add_argument("--stderr-log", required=True)
    args = parser.parse_args()

    out_root = str(args.out_root).strip() or f"{DEFAULT_WSL_PIPELINE_ROOT}/results/samam_wikiarts5_wsl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    meta = {
        "content_root": str(args.content_root),
        "style_root": str(args.style_root),
        "out_root": out_root,
        "iterations": int(args.iterations),
        "val_interval": int(args.val_interval),
        "batch_size": int(args.batch_size),
        "train_image_size": int(args.train_image_size),
        "train_crop_size": int(args.train_crop_size),
        "eval_image_size": int(args.eval_image_size),
        "patch_size": int(args.patch_size),
        "precision": str(args.precision),
        "mamba_from_trion": int(args.mamba_from_trion),
        "created_at": datetime.now().isoformat(),
    }
    meta_path = Path(args.stdout_log).resolve().with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    script_path = "/mnt/g/GitHub/Latent_Style/Related_Works/repos/SaMam/TRAIN/train_SaMam.py"
    remote_cmd = (
        "set -euo pipefail; "
        f"{args.wsl_python} {script_path} "
        f"--content {args.content_root} "
        f"--style {args.style_root} "
        f"--gpus 0 "
        f"--iterations {int(args.iterations)} "
        f"--val-interval {int(args.val_interval)} "
        f"--batch-size {int(args.batch_size)} "
        f"--train-image-size {int(args.train_image_size)} "
        f"--train-crop-size {int(args.train_crop_size)} "
        f"--eval-image-size {int(args.eval_image_size)} "
        f"--patch-size {int(args.patch_size)} "
        f"--precision {args.precision} "
        f"--mamba-from-trion {int(args.mamba_from_trion)} "
        f"--gradient-checkpointing {int(args.gradient_checkpointing)} "
        f"--identity-gradient-checkpointing {int(args.identity_gradient_checkpointing)} "
        f"--num-workers {int(args.num_workers)} "
        f"--pin-memory {int(args.pin_memory)} "
        f"--limit-val-batches {float(args.limit_val_batches)} "
        f"--num-sanity-val-steps {int(args.num_sanity_val_steps)} "
        f"--accumulate-grad-batches {int(args.accumulate_grad_batches)} "
        f"--checkpoint-every-n-steps {int(args.checkpoint_every_n_steps)} "
        f"--log-dir {out_root}"
    )
    cmd = [
        "wsl",
        "-d",
        str(args.wsl_distro),
        "bash",
        "-lc",
        remote_cmd,
    ]
    stdout_path = Path(args.stdout_log).resolve()
    stderr_path = Path(args.stderr_log).resolve()
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as stdout_f, stderr_path.open("w", encoding="utf-8") as stderr_f:
        subprocess.Popen(cmd, stdout=stdout_f, stderr=stderr_f, creationflags=subprocess.CREATE_NO_WINDOW)
    print(stdout_path)
    print(stderr_path)
    print(meta_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
