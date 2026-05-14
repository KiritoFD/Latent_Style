"""Self-contained StyTR-2 train + 750-image inference launcher.

This script lives in run_511 and does not import or reference Related_Works.
It uses:
  - run_511/repos/StyTR-2 for model code
  - style_data/ for local training and evaluation content/style images
  - the Ours 750-image folder only as an optional filename manifest
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
RUN511_ROOT = THIS_DIR.parent
WORKSPACE_ROOT = RUN511_ROOT.parent.parent
STYTR_REPO = RUN511_ROOT / "repos" / "StyTR-2"
PYTHON_EXE = os.environ.get("UV_PYTHON") or sys.executable
STYLE_DATA = WORKSPACE_ROOT / "style_data"
TRAIN_DATA = STYLE_DATA / "train"
OVERFIT50 = STYLE_DATA / "overfit50"
DEFAULT_REFERENCE_IMAGES = (
    WORKSPACE_ROOT
    / "SchrodingerBridge"
    / "exp"
    / "pareto_probe_4"
    / "S-add__K-3_C-2_W-10_Col-15"
    / "full_eval"
    / "epoch_0001"
    / "images"
)
STYLES = ["photo", "monet", "vangogh", "cezanne", "Hayao"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


PROFILES = {
    "4g": {"batch_size": 1, "train_images_per_style": 16, "max_iter": 500},
    "7g": {"batch_size": 1, "train_images_per_style": 32, "max_iter": 1000},
    "11g": {"batch_size": 2, "train_images_per_style": 64, "max_iter": 2000},
}


def run(cmd: list[str], cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", errors="replace") as f:
        f.write("\n\n=== CMD ===\n")
        f.write(" ".join(cmd) + "\n")
        f.flush()
        proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=f, stderr=subprocess.STDOUT)
        return proc.wait()


def copy_images(src: Path, dst: Path, limit: int | None = None, prefix: str | None = None) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    count = 0
    for img in sorted(src.iterdir()):
        if not img.is_file() or img.suffix.lower() not in IMG_EXTS:
            continue
        name = f"{prefix}_{img.name}" if prefix else img.name
        shutil.copy2(img, dst / name)
        count += 1
        if limit is not None and count >= limit:
            break
    return count


def prepare_train_data(work_dir: Path, images_per_style: int) -> tuple[Path, Path]:
    content_dir = work_dir / "train_content"
    style_dir = work_dir / "train_style"
    if content_dir.exists():
        shutil.rmtree(content_dir)
    if style_dir.exists():
        shutil.rmtree(style_dir)
    copy_images(TRAIN_DATA / "photo", content_dir, images_per_style)
    for style in STYLES:
        src = TRAIN_DATA / style
        if src.is_dir():
            copy_images(src, style_dir / style, images_per_style)
    return content_dir, style_dir


def reference_names(reference_images_dir: Path) -> list[str]:
    if not reference_images_dir.is_dir():
        names = []
        for src_style in STYLES:
            src_dir = OVERFIT50 / src_style
            for img in sorted(src_dir.glob("*.jpg"))[:30]:
                for target in STYLES:
                    names.append(f"{src_style}_{img.stem}_to_{target}.jpg")
        return names
    return sorted(p.name for p in reference_images_dir.iterdir() if p.is_file() and "_to_" in p.stem)


def prepare_infer_content(work_dir: Path, target_style: str, reference: list[str], limit: int = 0) -> Path:
    content_dir = work_dir / "infer_content" / target_style
    if content_dir.exists():
        shutil.rmtree(content_dir)
    content_dir.mkdir(parents=True, exist_ok=True)

    selected = [name for name in reference if name.endswith(f"_to_{target_style}.jpg")]
    if limit > 0:
        selected = selected[:limit]
    for out_name in selected:
        prefix = out_name[: -len(f"_to_{target_style}.jpg")]
        src_style, stem = prefix.split("_", 1)
        src = OVERFIT50 / src_style / f"{stem}.jpg"
        if not src.exists():
            continue
        shutil.copy2(src, content_dir / f"{src_style}_{src.name}")
    return content_dir


def prepare_style_ref(work_dir: Path, target_style: str) -> Path:
    style_dir = work_dir / "infer_style" / target_style
    if style_dir.exists():
        shutil.rmtree(style_dir)
    style_dir.mkdir(parents=True, exist_ok=True)
    src_dir = OVERFIT50 / target_style
    first = next(iter(sorted(src_dir.glob("*.jpg"))), None)
    if first is None:
        raise FileNotFoundError(f"No style reference images in {src_dir}")
    shutil.copy2(first, style_dir / f"{target_style}.jpg")
    return style_dir


def train(args: argparse.Namespace, profile: dict[str, int]) -> dict[str, object]:
    work_dir = args.run_root / "work" / "stytr2"
    save_dir = args.run_root / "checkpoints" / "stytr2"
    log_dir = args.run_root / "tb_logs" / "stytr2"
    train_log = args.run_root / "logs" / "stytr2_train.log"
    content_dir, style_dir = prepare_train_data(work_dir, int(args.train_images_per_style or profile["train_images_per_style"]))
    vgg = STYTR_REPO / "experiments" / "vgg_normalised.pth"
    if not vgg.exists():
        raise FileNotFoundError(f"Missing VGG: {vgg}")
    max_iter = int(args.max_iter or profile["max_iter"])
    batch_size = int(args.batch_size or profile["batch_size"])
    cmd = [
        PYTHON_EXE,
        "train.py",
        "--content_dir",
        str(content_dir),
        "--style_dir",
        str(style_dir),
        "--vgg",
        str(vgg),
        "--save_dir",
        str(save_dir),
        "--log_dir",
        str(log_dir),
        "--max_iter",
        str(max_iter),
        "--batch_size",
        str(batch_size),
        "--n_threads",
        "0",
        "--save_model_interval",
        str(max_iter),
    ]
    start = time.time()
    rc = run(cmd, STYTR_REPO, train_log)
    return {
        "stage": "train",
        "status": "ok" if rc == 0 else "failed",
        "returncode": rc,
        "elapsed_sec": round(time.time() - start, 3),
        "checkpoint_dir": str(save_dir),
        "log_path": str(train_log),
        "max_iter": max_iter,
        "batch_size": batch_size,
    }


def infer(args: argparse.Namespace, profile: dict[str, int]) -> dict[str, object]:
    max_iter = int(args.max_iter or profile["max_iter"])
    ckpt_dir = args.run_root / "checkpoints" / "stytr2"
    decoder = ckpt_dir / f"decoder_iter_{max_iter}.pth"
    transformer = ckpt_dir / f"transformer_iter_{max_iter}.pth"
    embedding = ckpt_dir / f"embedding_iter_{max_iter}.pth"
    missing = [str(p) for p in [decoder, transformer, embedding] if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing StyTR-2 checkpoints: " + "; ".join(missing))

    work_dir = args.run_root / "work" / "stytr2"
    output_dir = args.run_root / "infer_750" / "images"
    raw_root = args.run_root / "infer_750" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_root.mkdir(parents=True, exist_ok=True)
    reference = reference_names(args.reference_images_dir)

    rows = []
    start_all = time.time()
    total = 0
    for target in STYLES:
        content_dir = prepare_infer_content(work_dir, target, reference, args.limit_per_target)
        style_dir = prepare_style_ref(work_dir, target)
        raw_dir = raw_root / target
        if raw_dir.exists():
            shutil.rmtree(raw_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)
        log_path = args.run_root / "logs" / f"stytr2_infer_{target}.log"
        cmd = [
            PYTHON_EXE,
            "test.py",
            "--content_dir",
            str(content_dir),
            "--style_dir",
            str(style_dir),
            "--output",
            str(raw_dir),
            "--vgg",
            str(STYTR_REPO / "experiments" / "vgg_normalised.pth"),
            "--decoder_path",
            str(decoder),
            "--Trans_path",
            str(transformer),
            "--embedding_path",
            str(embedding),
        ]
        start = time.time()
        rc = run(cmd, STYTR_REPO, log_path)
        renamed = 0
        for img in sorted(raw_dir.glob("*.jpg")):
            stem = img.stem
            if "_stylized_" not in stem:
                continue
            content_stem, _style_stem = stem.split("_stylized_", 1)
            dst = output_dir / f"{content_stem}_to_{target}.jpg"
            shutil.copy2(img, dst)
            renamed += 1
        total += renamed
        rows.append(
            {
                "target": target,
                "returncode": rc,
                "renamed": renamed,
                "elapsed_sec": round(time.time() - start, 3),
                "log_path": str(log_path),
            }
        )
        if rc != 0:
            break

    status = "ok" if all(row["returncode"] == 0 for row in rows) else "failed"
    if args.limit_per_target == 0 and total != 750:
        status = "failed"
    return {
        "stage": "infer",
        "status": status,
        "elapsed_sec": round(time.time() - start_all, 3),
        "images": total,
        "images_dir": str(output_dir),
        "per_target": rows,
    }


def write_summary(run_root: Path, rows: list[dict[str, object]]) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "summary.json").write_text(json.dumps({"runs": rows}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (run_root / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        keys = sorted({k for row in rows for k in row.keys() if k != "per_target"})
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in keys})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "infer", "all", "smoke"], default="all")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="7g")
    parser.add_argument("--run_root", type=Path, default=RUN511_ROOT / "outputs" / "stytr2_750")
    parser.add_argument("--reference_images_dir", type=Path, default=DEFAULT_REFERENCE_IMAGES)
    parser.add_argument("--max_iter", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--train_images_per_style", type=int, default=0)
    parser.add_argument("--limit_per_target", type=int, default=0, help="0 means full 150 per target / 750 total.")
    args = parser.parse_args()
    args.run_root = args.run_root.resolve()
    args.reference_images_dir = args.reference_images_dir.resolve()
    profile = PROFILES[args.profile]
    if args.mode == "smoke":
        args.max_iter = 1
        args.batch_size = 1
        args.train_images_per_style = 2
        args.limit_per_target = 1
        args.mode = "all"

    rows: list[dict[str, object]] = []
    if args.mode in {"train", "all"}:
        rows.append(train(args, profile))
        write_summary(args.run_root, rows)
        if rows[-1]["status"] != "ok":
            return 1
    if args.mode in {"infer", "all"}:
        rows.append(infer(args, profile))
        write_summary(args.run_root, rows)
        if rows[-1]["status"] != "ok":
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
