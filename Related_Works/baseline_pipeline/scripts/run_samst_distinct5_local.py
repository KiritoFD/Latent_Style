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


def _infer_style_from_flat_name(path: Path) -> str | None:
    prefix, sep, _ = path.name.partition("__")
    return prefix if sep and prefix else None


def _is_classview_layout(data_root: Path) -> bool:
    return (data_root / "train").is_dir()


def _is_flat_layout(data_root: Path) -> bool:
    return (data_root / "train_flat" / "content").is_dir() and (data_root / "train_flat" / "style").is_dir()


def prepare_content_view(data_root: Path, work_dir: Path, max_train_per_class: int, styles: list[str]) -> Path:
    train_root = data_root / "train"
    if _is_classview_layout(data_root) and max_train_per_class <= 0:
        return train_root
    out_root = work_dir / "content_train_subset"
    shutil.rmtree(out_root, ignore_errors=True)
    if _is_classview_layout(data_root):
        for style_dir in sorted(p for p in train_root.iterdir() if p.is_dir()):
            dst_dir = out_root / style_dir.name
            dst_dir.mkdir(parents=True, exist_ok=True)
            for src in image_files(style_dir)[:max_train_per_class]:
                shutil.copy2(src, dst_dir / src.name)
        return out_root

    if not _is_flat_layout(data_root):
        raise FileNotFoundError(f"Unsupported data layout under {data_root}")

    content_root = data_root / "train_flat" / "content"
    grouped: dict[str, list[Path]] = {style: [] for style in styles}
    for src in image_files(content_root):
        style = _infer_style_from_flat_name(src)
        if style in grouped:
            grouped[style].append(src)

    for style in styles:
        dst_dir = out_root / style
        dst_dir.mkdir(parents=True, exist_ok=True)
        selected = grouped[style][:max_train_per_class] if max_train_per_class > 0 else grouped[style]
        if not selected:
            raise FileNotFoundError(f"No flat-layout train content found for style={style} under {content_root}")
        for src in selected:
            shutil.copy2(src, dst_dir / src.name)
    return out_root


def prepare_style_dir(data_root: Path, style: str, work_dir: Path) -> Path:
    style_dir = work_dir / "style_single" / style
    shutil.rmtree(style_dir, ignore_errors=True)
    style_dir.mkdir(parents=True, exist_ok=True)
    if _is_classview_layout(data_root):
        srcs = image_files(data_root / "train" / style)
    elif _is_flat_layout(data_root):
        style_root = data_root / "train_flat" / "style"
        srcs = [src for src in image_files(style_root) if _infer_style_from_flat_name(src) == style]
    else:
        raise FileNotFoundError(f"Unsupported data layout under {data_root}")
    if not srcs:
        raise FileNotFoundError(f"No style images for style={style} under {data_root}")
    shutil.copy2(srcs[0], style_dir / srcs[0].name)
    return style_dir


def write_train_config(
    train_dir: Path,
    train_dataset: Path,
    style_dir: Path,
    ckpt_dir: Path,
    epochs: int,
    max_steps: int,
    batch_size: int,
    image_size: int,
    style_size: int,
) -> Path:
    cfg = {
        "epochs": epochs,
        "batch_size": batch_size,
        "dataset": train_dataset.as_posix(),
        "style_image": style_dir.as_posix() + "/",
        "save_model_dir": ckpt_dir.as_posix(),
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
        "max_steps": max_steps,
        "step_model_name_template": "step_{step:06d}.model",
    }
    path = train_dir / "train.yml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def _read_optional_text(path: Path) -> str | None:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def _restore_optional_text(path: Path, content: str | None) -> None:
    if content is None:
        if path.exists():
            path.unlink()
        return
    path.write_text(content, encoding="utf-8")


def run_target(args: argparse.Namespace, out_root: Path, style: str) -> int:
    train_dir = SAMST_REPO / "train_model" / "train2"
    train_py = train_dir / "train.py"
    if not train_py.exists():
        raise FileNotFoundError(train_py)

    style_dir = prepare_style_dir(args.data_root, style, out_root)
    selected_styles = [s.strip() for s in str(args.styles).split(",") if s.strip()]
    train_dataset = prepare_content_view(args.data_root, out_root, int(args.max_train_per_class), selected_styles)
    ckpt_dir = out_root / "checkpoints" / style
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    train_yml = train_dir / "train.yml"
    original_train_yml = _read_optional_text(train_yml)
    write_train_config(
        train_dir=train_dir,
        train_dataset=train_dataset,
        style_dir=style_dir,
        ckpt_dir=ckpt_dir,
        epochs=args.epochs,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        image_size=args.image_size,
        style_size=args.style_size,
    )

    log_path = out_root / "logs" / f"train_{style}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(train_py)]
    start = time.time()
    try:
        with log_path.open("a", encoding="utf-8", errors="replace") as log:
            log.write(f"\n=== START {style} {datetime.now().isoformat()} cmd={cmd} ===\n")
            log.flush()
            proc = subprocess.run(cmd, cwd=str(train_dir), stdout=log, stderr=subprocess.STDOUT)
            elapsed = time.time() - start
            log.write(f"\n=== END {style} rc={proc.returncode} elapsed_sec={elapsed:.2f} ===\n")
        return proc.returncode
    finally:
        _restore_optional_text(train_yml, original_train_yml)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--styles", type=str, default=",".join(STYLES))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=0, help="Stop after this many optimizer steps per target style; 0 keeps epoch-only training.")
    parser.add_argument("--batch-size", type=int, default=1)
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
            f"styles={selected} epochs={args.epochs} max_steps={args.max_steps} batch_size={args.batch_size} "
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
