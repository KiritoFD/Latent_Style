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
DEFAULT_WSL_DATA_ROOT = "/mnt/f/wikiarts_5_full_notest"
DEFAULT_WSL_PIPELINE_ROOT = "/mnt/g/GitHub/Latent_Style/Related_Works/baseline_pipeline"


def _run(cmd: list[str]) -> int:
    print("[run_wsl_samst_wikiarts5] " + " ".join(str(x) for x in cmd), flush=True)
    proc = subprocess.run(cmd, check=False)
    return int(proc.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch a local-WSL SaMST reproduction run on the new wikiarts-5 dataset.")
    parser.add_argument("--wsl-distro", default="f")
    parser.add_argument("--wsl-python", default="python3")
    parser.add_argument("--data-root", default=DEFAULT_WSL_DATA_ROOT)
    parser.add_argument("--out-root", default="")
    parser.add_argument("--styles", default="Early_Renaissance,Impressionism,Minimalism,Rococo,Ukiyo_e")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--style-size", type=int, default=512)
    parser.add_argument("--save-interval", type=int, default=5)
    parser.add_argument("--max-train-per-class", type=int, default=0)
    parser.add_argument("--skip-styles-with-epoch-at-least", type=int, default=0)
    parser.add_argument("--stop-after-one-pending-style", action="store_true")
    parser.add_argument("--stdout-log", required=True)
    parser.add_argument("--stderr-log", required=True)
    args = parser.parse_args()

    out_root = str(args.out_root).strip() or f"{DEFAULT_WSL_PIPELINE_ROOT}/results/samst_wikiarts5_wsl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    meta = {
        "dataset": str(args.data_root),
        "out_root": out_root,
        "styles": str(args.styles),
        "epochs": int(args.epochs),
        "max_steps": int(args.max_steps),
        "batch_size": int(args.batch_size),
        "image_size": int(args.image_size),
        "style_size": int(args.style_size),
        "save_interval": int(args.save_interval),
        "max_train_per_class": int(args.max_train_per_class),
        "skip_styles_with_epoch_at_least": int(args.skip_styles_with_epoch_at_least),
        "stop_after_one_pending_style": bool(args.stop_after_one_pending_style),
        "created_at": datetime.now().isoformat(),
    }
    meta_path = Path(args.stdout_log).resolve().with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    script_path = "/mnt/g/GitHub/Latent_Style/Related_Works/baseline_pipeline/scripts/run_samst_distinct5_local.py"
    remote_cmd = (
        "set -euo pipefail; "
        f"{args.wsl_python} {script_path} "
        f"--data-root {args.data_root} "
        f"--out-root {out_root} "
        f"--styles {args.styles} "
        f"--epochs {int(args.epochs)} "
        f"--max-steps {int(args.max_steps)} "
        f"--batch-size {int(args.batch_size)} "
        f"--image-size {int(args.image_size)} "
        f"--style-size {int(args.style_size)} "
        f"--save-interval {int(args.save_interval)} "
        f"--max-train-per-class {int(args.max_train_per_class)} "
        f"--skip-styles-with-epoch-at-least {int(args.skip_styles_with_epoch_at_least)}"
    )
    if bool(args.stop_after_one_pending_style):
        remote_cmd += " --stop-after-one-pending-style"
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
