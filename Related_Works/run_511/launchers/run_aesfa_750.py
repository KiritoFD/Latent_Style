"""Self-contained AesFA train + 750-image inference launcher.

AesFA (Yoo et al. 2023) uses an encoder-decoder with frequency-aware style
aggregation.  Training uses content/style image pairs; inference transfers
each content image to each target style.

This script lives in run_511 and uses run_511/repos/AesFA for model code.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = THIS_DIR.parent
AESFA_REPO = THIS_DIR / "repos" / "AesFA"
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
    "4g": {"batch_size": 1, "train_images_per_style": 16, "max_iter": 16000},
    "7g": {"batch_size": 2, "train_images_per_style": 32, "max_iter": 48000},
    "11g": {"batch_size": 4, "train_images_per_style": 64, "max_iter": 100000},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def copy_images(src: Path, dst: Path, limit: int | None = None) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    count = 0
    for img in sorted(src.iterdir()):
        if not img.is_file() or img.suffix.lower() not in IMG_EXTS:
            continue
        shutil.copy2(img, dst / img.name)
        count += 1
        if limit is not None and count >= limit:
            break
    return count


def prepare_train_data(work_dir: Path, images_per_style: int) -> tuple[Path, Path]:
    """AesFA expects content_dir/train/ and style_dir/train/ structure."""
    content_root = work_dir / "aesfa_content"
    style_root = work_dir / "aesfa_style"
    if content_root.exists():
        shutil.rmtree(content_root)
    if style_root.exists():
        shutil.rmtree(style_root)
    copy_images(TRAIN_DATA / "photo", content_root / "train", images_per_style)
    for style in STYLES:
        src = TRAIN_DATA / style
        if src.is_dir():
            copy_images(src, style_root / "train", images_per_style)
    return content_root, style_root


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


def run_cmd(cmd: list[str], cwd: Path, log_path: Path) -> int:
    import subprocess
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", errors="replace") as f:
        f.write("\n\n=== CMD ===\n")
        f.write(" ".join(cmd) + "\n")
        f.flush()
        proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=f, stderr=subprocess.STDOUT)
        return proc.wait()


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace, profile: dict[str, int]) -> dict[str, object]:
    vgg = AESFA_REPO / "vgg_normalised.pth"
    start = time.time()
    if not vgg.exists():
        return {"stage": "train", "status": "blocked", "error": f"missing {vgg}", "elapsed_sec": 0}

    images_per_style = int(args.train_images_per_style or profile["train_images_per_style"])
    max_iter = int(args.max_iter or profile["max_iter"])
    batch_size = int(args.batch_size or profile["batch_size"])

    content_root, style_root = prepare_train_data(args.run_root / "datasets", images_per_style)
    ckpt_dir = args.run_root / "checkpoints" / "aesfa"
    log_dir = args.run_root / "tb_logs" / "aesfa"
    log_path = args.run_root / "logs" / "aesfa_train.log"

    # AesFA uses Config class attributes — launch via inline code
    code = (
        "import os, sys; "
        f"repo={str(AESFA_REPO)!r}; os.chdir(repo); sys.path.insert(0, repo); "
        "from Config import Config; "
        f"Config.phase='train'; Config.train_continue='off'; Config.data_num={images_per_style}; "
        f"Config.content_dir={str(content_root)!r}; Config.style_dir={str(style_root)!r}; "
        f"Config.file_n='run511'; Config.log_dir={str(log_dir)!r}; Config.ckpt_dir={str(ckpt_dir)!r}; "
        f"Config.img_dir={str(args.run_root / 'preview' / 'aesfa')!r}; Config.vgg_model={str(vgg)!r}; "
        f"Config.n_iter={max_iter}; Config.save_interval={max_iter}; "
        f"Config.batch_size={batch_size}; Config.num_workers=0; Config.load_size=256; Config.crop_size=256; "
        "import train; train.main()"
    )
    rc = run_cmd([sys.executable, "-c", code], AESFA_REPO, log_path)
    return {
        "stage": "train",
        "status": "ok" if rc == 0 else "failed",
        "returncode": rc,
        "elapsed_sec": round(time.time() - start, 3),
        "checkpoint_dir": str(ckpt_dir),
        "log_path": str(log_path),
        "max_iter": max_iter,
        "batch_size": batch_size,
    }


# ---------------------------------------------------------------------------
# Infer
# ---------------------------------------------------------------------------

def infer(args: argparse.Namespace, profile: dict[str, int]) -> dict[str, object]:
    vgg = AESFA_REPO / "vgg_normalised.pth"
    ckpt_dir = args.run_root / "checkpoints" / "aesfa"
    ckpt = ckpt_dir / "main.pth"
    if not ckpt.exists():
        return {"stage": "infer", "status": "blocked", "error": f"missing checkpoint {ckpt}"}

    output_dir = args.run_root / "infer_750" / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    reference = reference_names(args.reference_images_dir)

    rows = []
    start_all = time.time()
    total = 0

    for target in STYLES:
        target_ref = [n for n in reference if n.endswith(f"_to_{target}.jpg")]
        if args.limit_per_target > 0:
            target_ref = target_ref[:args.limit_per_target]

        # Prepare content/style dirs for this target
        work_dir = args.run_root / "work" / "aesfa" / target
        content_dir = work_dir / "content"
        style_dir = work_dir / "style"
        if content_dir.exists():
            shutil.rmtree(content_dir)
        if style_dir.exists():
            shutil.rmtree(style_dir)
        content_dir.mkdir(parents=True, exist_ok=True)
        style_dir.mkdir(parents=True, exist_ok=True)

        first_style = next(iter(sorted((OVERFIT50 / target).glob("*.jpg"))), None)
        if first_style is None:
            rows.append({"target": target, "returncode": 1, "renamed": 0, "error": "no style image"})
            continue
        shutil.copy2(first_style, style_dir / f"{target}.jpg")

        selected = []
        for out_name in target_ref:
            prefix = out_name[: -len(f"_to_{target}.jpg")]
            src_style, stem = prefix.split("_", 1)
            src = OVERFIT50 / src_style / f"{stem}.jpg"
            if src.exists():
                shutil.copy2(src, content_dir / f"{src_style}_{stem}.jpg")
                selected.append((out_name, src_style, stem))

        infer_img_dir = args.run_root / "preview" / "aesfa" / target
        log_path = args.run_root / "logs" / f"aesfa_infer_{target}.log"

        code = (
            "import os, sys; "
            f"repo={str(AESFA_REPO)!r}; os.chdir(repo); sys.path.insert(0, repo); "
            "from Config import Config; "
            f"Config.phase='test'; Config.multi_to_multi=True; "
            f"Config.test_content_size=256; Config.test_style_size=256; "
            f"Config.content_dir={str(content_dir)!r}; Config.style_dir={str(style_dir)!r}; "
            f"Config.img_dir={str(infer_img_dir)!r}; "
            f"Config.ckpt_dir={str(ckpt_dir)!r}; Config.vgg_model={str(vgg)!r}; "
            "import test; test.main()"
        )
        start = time.time()
        rc = run_cmd([sys.executable, "-c", code], AESFA_REPO, log_path)

        # Rename outputs to match 750-image naming convention
        renamed = 0
        if infer_img_dir.exists():
            for img in sorted(infer_img_dir.glob("*stylized*")):
                # AesFA saves as {content}_stylized_{style}.jpg
                parts = img.stem.split("_stylized_")
                if len(parts) == 2:
                    # Find matching output name
                    for out_name, src_style, stem in selected:
                        dst = output_dir / out_name
                        if not dst.exists():
                            shutil.copy2(img, dst)
                            renamed += 1
                            break

        total += renamed
        rows.append({
            "target": target,
            "returncode": rc,
            "renamed": renamed,
            "elapsed_sec": round(time.time() - start, 3),
            "log_path": str(log_path),
        })
        if rc != 0:
            break

    status = "ok" if all(row["returncode"] == 0 for row in rows) else "failed"
    if args.limit_per_target == 0 and total != 750:
        status = "partial" if total > 0 else "failed"
    return {
        "stage": "infer",
        "status": status,
        "elapsed_sec": round(time.time() - start_all, 3),
        "images": total,
        "images_dir": str(output_dir),
        "per_target": rows,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
    parser.add_argument("--run_root", type=Path, default=THIS_DIR / "outputs" / "aesfa_750")
    parser.add_argument("--reference_images_dir", type=Path, default=DEFAULT_REFERENCE_IMAGES)
    parser.add_argument("--max_iter", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--train_images_per_style", type=int, default=0)
    parser.add_argument("--limit_per_target", type=int, default=0)
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
        if rows[-1]["status"] not in {"ok", "blocked"}:
            return 1
    if args.mode in {"infer", "all"}:
        if rows and rows[-1].get("status") == "blocked":
            return 1
        rows.append(infer(args, profile))
        write_summary(args.run_root, rows)
        if rows[-1]["status"] != "ok":
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
