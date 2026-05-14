"""Serial CycleGAN training launcher for artist-domain time-to-quality runs.

This script intentionally runs targets one after another so a single GPU is not
oversubscribed by parallel CycleGAN jobs.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = PIPELINE_ROOT.parent.parent
REPO_ROOT = WORKSPACE_ROOT / "Related_Works" / "pytorch-CycleGAN-and-pix2pix"
DEFAULT_DATASETS_ROOT = WORKSPACE_ROOT / "Related_Works" / "runs" / "cut_5x5" / "datasets"
DEFAULT_RUN_ROOT = WORKSPACE_ROOT / "Related_Works" / "runs" / "cyclegan_5x5"
DEFAULT_TARGETS = ["monet", "vangogh", "cezanne", "Hayao"]


def run_one(cmd: list[str], cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", errors="replace") as f:
        f.write("\n\n=== CMD ===\n")
        f.write(" ".join(cmd) + "\n")
        f.flush()
        proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=f, stderr=subprocess.STDOUT)
        return proc.wait()


def main() -> int:
    parser = argparse.ArgumentParser(description="Train CycleGAN targets serially.")
    parser.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
    parser.add_argument("--datasets_root", type=Path, default=DEFAULT_DATASETS_ROOT)
    parser.add_argument("--run_root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--n_epochs_decay", type=int, default=10)
    parser.add_argument("--max_dataset_size", type=int, default=1000)
    parser.add_argument("--load_size", type=int, default=256)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--netG", default="resnet_6blocks")
    parser.add_argument("--ngf", type=int, default=32)
    parser.add_argument("--ndf", type=int, default=32)
    parser.add_argument("--save_epoch_freq", type=int, default=5)
    parser.add_argument("--print_freq", type=int, default=200)
    parser.add_argument("--continue_train", action="store_true")
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    checkpoints_dir = run_root / "checkpoints"
    logs_dir = run_root / "logs"
    timing_csv = run_root / "train_timing.csv"
    summary_json = run_root / "train_summary.json"
    run_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for target in args.targets:
        dataroot = args.datasets_root.resolve() / f"to_{target}"
        if not dataroot.is_dir():
            rows.append({"target": target, "status": "missing_dataset", "dataroot": str(dataroot)})
            continue

        name = f"cyclegan_to_{target}"
        log_path = logs_dir / f"{name}.log"
        cmd = [
            str(args.python.resolve()),
            "train.py",
            "--dataroot",
            str(dataroot),
            "--name",
            name,
            "--model",
            "cycle_gan",
            "--checkpoints_dir",
            str(checkpoints_dir),
            "--no_html",
            "--n_epochs",
            str(args.n_epochs),
            "--n_epochs_decay",
            str(args.n_epochs_decay),
            "--save_epoch_freq",
            str(args.save_epoch_freq),
            "--print_freq",
            str(args.print_freq),
            "--max_dataset_size",
            str(args.max_dataset_size),
            "--load_size",
            str(args.load_size),
            "--crop_size",
            str(args.crop_size),
            "--netG",
            args.netG,
            "--ngf",
            str(args.ngf),
            "--ndf",
            str(args.ndf),
            "--batch_size",
            "1",
            "--num_threads",
            "0",
        ]
        if args.continue_train:
            cmd.append("--continue_train")

        start = time.time()
        status = "ok"
        rc = run_one(cmd, REPO_ROOT, log_path)
        elapsed = round(time.time() - start, 3)
        if rc != 0:
            status = "failed"
        row = {
            "target": target,
            "status": status,
            "returncode": rc,
            "elapsed_sec": elapsed,
            "dataroot": str(dataroot),
            "checkpoint_dir": str(checkpoints_dir / name),
            "log_path": str(log_path),
            "n_epochs": args.n_epochs,
            "n_epochs_decay": args.n_epochs_decay,
            "max_dataset_size": args.max_dataset_size,
            "load_size": args.load_size,
            "crop_size": args.crop_size,
            "netG": args.netG,
            "ngf": args.ngf,
            "ndf": args.ndf,
        }
        rows.append(row)

        with timing_csv.open("w", encoding="utf-8", newline="") as f:
            fieldnames = list(rows[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        summary_json.write_text(json.dumps({"runs": rows}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        if rc != 0:
            break

    return 0 if all(row.get("status") == "ok" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
