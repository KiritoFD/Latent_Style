from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = PIPELINE_ROOT.parent.parent
SAMST_REPO = WORKSPACE_ROOT / "Related_Works" / "repos" / "SaMST-main"
DEFAULT_DATA = Path(r"F:\wikiart_distinct5_samam_512_classview")
STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]


def image_files(path: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in exts)


def prepare_content_view(data_root: Path, work_dir: Path, max_train_per_class: int) -> Path:
    train_root = data_root / "train"
    if max_train_per_class <= 0:
        return train_root
    out_root = work_dir / "content_train_subset"
    shutil.rmtree(out_root, ignore_errors=True)
    for style_dir in sorted(p for p in train_root.iterdir() if p.is_dir()):
        dst_dir = out_root / style_dir.name
        dst_dir.mkdir(parents=True, exist_ok=True)
        for src in image_files(style_dir)[:max_train_per_class]:
            shutil.copy2(src, dst_dir / src.name)
    return out_root


def prepare_style_dir(data_root: Path, style: str, work_dir: Path) -> Path:
    style_dir = work_dir / "style_single" / style
    shutil.rmtree(style_dir, ignore_errors=True)
    style_dir.mkdir(parents=True, exist_ok=True)
    srcs = image_files(data_root / "train" / style)
    if not srcs:
        raise FileNotFoundError(f"No style images under {data_root / 'train' / style}")
    shutil.copy2(srcs[0], style_dir / srcs[0].name)
    return style_dir


def write_train_config(
    train_dir: Path,
    train_dataset: Path,
    style_dir: Path,
    ckpt_dir: Path,
    epochs: int,
    batch_size: int,
    image_size: int,
    style_size: int,
) -> Path:
    cfg = {
        "epochs": epochs,
        "batch_size": batch_size,
        "dataset": str(train_dataset),
        "style_image": str(style_dir) + "\\",
        "save_model_dir": str(ckpt_dir),
        "image_size": image_size,
        "style_size": style_size,
        "cuda": 1,
        "seed": 7,
        "content_weight": 1e5,
        "style_weight": 1e10,
        "ae_weight": 1e3,
        "lr": 0.001,
        "weight_decay": 0.5,
        "step_size": 25,
        "save_interval": epochs,
        "log_interval": 10,
        "checkpoint_interval": 100,
        "checkpoint_model_dir": None,
        "begin_checkpoint": None,
        "begin_epoch": None,
    }
    path = train_dir / "train.yml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def run_target(args: argparse.Namespace, out_root: Path, style: str) -> int:
    train_dir = SAMST_REPO / "train_model" / "train2"
    train_py = train_dir / "train.py"
    if not train_py.exists():
        raise FileNotFoundError(train_py)

    style_dir = prepare_style_dir(args.data_root, style, out_root)
    train_dataset = prepare_content_view(args.data_root, out_root, int(args.max_train_per_class))
    ckpt_dir = out_root / "checkpoints" / style
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    write_train_config(
        train_dir=train_dir,
        train_dataset=train_dataset,
        style_dir=style_dir,
        ckpt_dir=ckpt_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        style_size=args.style_size,
    )

    log_path = out_root / "logs" / f"train_{style}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(train_py)]
    start = time.time()
    with log_path.open("a", encoding="utf-8", errors="replace") as log:
        log.write(f"\n=== START {style} {datetime.now().isoformat()} cmd={cmd} ===\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=str(train_dir), stdout=log, stderr=subprocess.STDOUT)
        elapsed = time.time() - start
        log.write(f"\n=== END {style} rc={proc.returncode} elapsed_sec={elapsed:.2f} ===\n")
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--styles", type=str, default=",".join(STYLES))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--style-size", type=int, default=512)
    parser.add_argument("--max-train-per-class", type=int, default=0, help="Use only the first N train images per class; <=0 uses full train split.")
    args = parser.parse_args()

    if not args.data_root.exists():
        raise FileNotFoundError(args.data_root)
    args.data_root = args.data_root.resolve()
    out_root = args.out_root or (
        PIPELINE_ROOT
        / "results"
        / f"samst_distinct5_512_local_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    out_root = out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "logs").mkdir(exist_ok=True)

    selected = [s.strip() for s in args.styles.split(",") if s.strip()]
    summary = out_root / "run.log"
    with summary.open("a", encoding="utf-8") as f:
        f.write(
            f"started={datetime.now().isoformat()} data_root={args.data_root} "
            f"styles={selected} epochs={args.epochs} batch_size={args.batch_size} "
            f"max_train_per_class={args.max_train_per_class}\n"
        )

    for style in selected:
        rc = run_target(args, out_root, style)
        with summary.open("a", encoding="utf-8") as f:
            f.write(f"target={style} rc={rc} time={datetime.now().isoformat()}\n")
        if rc != 0:
            return rc

    with summary.open("a", encoding="utf-8") as f:
        f.write(f"finished={datetime.now().isoformat()}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
