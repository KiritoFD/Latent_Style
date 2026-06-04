from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = PIPELINE_ROOT.parent.parent
DEFAULT_REPO_ROOT = WORKSPACE_ROOT / "Related_Works" / "repos" / "cyclegan_turbo" / "img2img-turbo"
DEFAULT_DATASETS_ROOT = Path("F:/wikiart_distinct5_img2img_turbo_datasets")
DEFAULT_RUN_ROOT = WORKSPACE_ROOT / "Related_Works" / "runs" / "img2img_turbo_distinct5_smoke"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare or launch a minimal Distinct5 img2img-turbo smoke run."
    )
    parser.add_argument("--target", required=True)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--datasets-root", type=Path, default=DEFAULT_DATASETS_ROOT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--accelerate", type=Path, default=None)
    parser.add_argument("--main-process-port", type=int, default=29531)
    parser.add_argument("--pretrained-model-name-or-path", default="stabilityai/sd-turbo")
    parser.add_argument("--train-img-prep", default="resize_512")
    parser.add_argument("--val-img-prep", default="resize_512")
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-train-steps", type=int, default=20)
    parser.add_argument("--max-train-epochs", type=int, default=100)
    parser.add_argument("--validation-steps", type=int, default=10)
    parser.add_argument("--validation-num-images", type=int, default=8)
    parser.add_argument("--checkpointing-steps", type=int, default=10)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--tracker-project-name", default=None)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--enable-xformers", action="store_true")
    parser.add_argument("--run", action="store_true")
    return parser.parse_args()


def resolve_accelerate(python_path: Path, accelerate_path: Path | None) -> Path:
    if accelerate_path is not None:
        return accelerate_path.resolve()
    scripts_dir = python_path.resolve().parent / "Scripts"
    candidate = scripts_dir / "accelerate.exe"
    if candidate.exists():
        return candidate
    return Path("accelerate")


def build_command(args: argparse.Namespace, dataset_dir: Path, output_dir: Path) -> list[str]:
    tracker_project_name = args.tracker_project_name or f"distinct5_turbo_smoke_{args.target}"
    accelerate_path = resolve_accelerate(args.python, args.accelerate)
    cmd = [
        str(accelerate_path),
        "launch",
        "--main_process_port",
        str(args.main_process_port),
        "src/train_cyclegan_turbo.py",
        "--pretrained_model_name_or_path",
        str(args.pretrained_model_name_or_path),
        "--dataset_folder",
        str(dataset_dir),
        "--output_dir",
        str(output_dir),
        "--train_img_prep",
        str(args.train_img_prep),
        "--val_img_prep",
        str(args.val_img_prep),
        "--train_batch_size",
        str(args.train_batch_size),
        "--gradient_accumulation_steps",
        str(args.gradient_accumulation_steps),
        "--max_train_steps",
        str(args.max_train_steps),
        "--max_train_epochs",
        str(args.max_train_epochs),
        "--validation_steps",
        str(args.validation_steps),
        "--validation_num_images",
        str(args.validation_num_images),
        "--checkpointing_steps",
        str(args.checkpointing_steps),
        "--dataloader_num_workers",
        str(args.dataloader_num_workers),
        "--tracker_project_name",
        str(tracker_project_name),
        "--report_to",
        str(args.report_to),
        "--learning_rate",
        str(args.learning_rate),
    ]
    if args.allow_tf32:
        cmd.append("--allow_tf32")
    if args.gradient_checkpointing:
        cmd.append("--gradient_checkpointing")
    if args.enable_xformers:
        cmd.append("--enable_xformers_memory_efficient_attention")
    return cmd


def main() -> int:
    args = parse_args()
    dataset_dir = (args.datasets_root / f"to_{args.target}").resolve()
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")
    if not (dataset_dir / "train_A").is_dir():
        raise FileNotFoundError(f"Missing train_A under dataset folder: {dataset_dir}")

    repo_root = args.repo_root.resolve()
    output_dir = (args.run_root / args.target).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    command = build_command(args, dataset_dir, output_dir)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "target": args.target,
        "repo_root": str(repo_root),
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "command": command,
        "run_requested": bool(args.run),
    }
    (output_dir / "launch_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    command_txt = " ".join(shlex.quote(part) for part in command)
    (output_dir / "launch_command.txt").write_text(command_txt + "\n", encoding="utf-8")
    print(command_txt)

    if not args.run:
        return 0

    log_path = output_dir / "train.log"
    env = dict(os.environ)
    env["NCCL_P2P_DISABLE"] = "1"
    with log_path.open("a", encoding="utf-8", errors="replace") as log_file:
        log_file.write(f"[launch] {datetime.now().isoformat(timespec='seconds')}\n")
        log_file.write(command_txt + "\n")
        log_file.flush()
        proc = subprocess.Popen(
            command,
            cwd=str(repo_root),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
        )
        return proc.wait()


if __name__ == "__main__":
    raise SystemExit(main())
