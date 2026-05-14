"""Self-contained AesPA-Net train + 750-image inference launcher.

AesPA-Net (Kim et al. 2024) uses a style decorator with contextual attention
and a multi-scale discriminator for adversarial training.

This script lives in run_511 and uses run_511/repos/AesPA-Net for model code.
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
AESPA_REPO = RUN511_ROOT / "repos" / "AesPA-Net"
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
    "4g": {"batch_size": 1, "train_images_per_style": 16, "max_iter": 200},
    "7g": {"batch_size": 2, "train_images_per_style": 32, "max_iter": 500},
    "11g": {"batch_size": 4, "train_images_per_style": 64, "max_iter": 1000},
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


def run_cmd(cmd: list[str], cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"
    env["WANDB_DISABLED"] = "true"
    with log_path.open("a", encoding="utf-8", errors="replace") as f:
        f.write("\n\n=== CMD ===\n")
        f.write(" ".join(cmd) + "\n")
        f.flush()
        proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=f, stderr=subprocess.STDOUT, env=env)
        return proc.wait()


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

def check_assets() -> dict[str, object]:
    """Check that required AesPA-Net assets exist."""
    candidates = [
        AESPA_REPO / "baseline_checkpoints" / "vgg_normalised_conv5_1.pth",
        AESPA_REPO / "baseline_checkpoints" / "vgg_normalised_conv5_1.t7",
        AESPA_REPO / "baseline_checkpoints" / "vgg_normalised_conv5_1.pkl",
    ]
    if not any(path.exists() for path in candidates):
        return {
            "status": "blocked",
            "error": "missing baseline_checkpoints/vgg_normalised_conv5_1.[pth|t7|pkl]",
        }
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace, profile: dict[str, int]) -> dict[str, object]:
    assets = check_assets()
    if assets["status"] == "blocked":
        return {"stage": "train", **assets, "elapsed_sec": 0}

    images_per_style = int(args.train_images_per_style or profile["train_images_per_style"])
    max_iter = int(args.max_iter or profile["max_iter"])
    batch_size = int(args.batch_size or profile["batch_size"])

    content_dir, style_dir = prepare_train_data(args.run_root / "datasets", images_per_style)
    train_result_dir = args.run_root / "checkpoints" / "aespa"
    train_result_dir.mkdir(parents=True, exist_ok=True)

    log_path = args.run_root / "logs" / "aespa_train.log"
    cmd = [
        PYTHON_EXE,
        "main.py",
        "--type", "train",
        "--content_dir", str(content_dir),
        "--style_dir", str(style_dir),
        "--max_iter", str(max_iter),
        "--batch_size", str(batch_size),
        "--imsize", "256",
        "--cropsize", "256",
        "--num_workers", "0",
        "--train_result_dir", str(train_result_dir),
        "--comment", "run511",
    ]
    start = time.time()
    rc = run_cmd(cmd, AESPA_REPO, log_path)
    return {
        "stage": "train",
        "status": "ok" if rc == 0 else "failed",
        "returncode": rc,
        "elapsed_sec": round(time.time() - start, 3),
        "checkpoint_dir": str(train_result_dir),
        "log_path": str(log_path),
        "max_iter": max_iter,
        "batch_size": batch_size,
    }


# ---------------------------------------------------------------------------
# Infer
# ---------------------------------------------------------------------------

def infer(args: argparse.Namespace, profile: dict[str, int]) -> dict[str, object]:
    # Check for trained checkpoint — AesPA saves to train_result_dir/comment/log/
    ckpt_dir = args.run_root / "checkpoints" / "aespa" / "run511" / "log"
    if not ckpt_dir.exists():
        # Try to find any checkpoint
        ckpt_parent = args.run_root / "checkpoints" / "aespa"
        candidates = []
        if ckpt_parent.exists():
            for d in ckpt_parent.rglob("dec_model_.pth"):
                candidates.append(d.parent)
        if not candidates:
            return {"stage": "infer", "status": "blocked", "error": f"no checkpoint found in {ckpt_parent}"}
        ckpt_dir = candidates[0]

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

        # Prepare content/style for this target
        work_dir = args.run_root / "work" / "aespa" / target
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
                selected.append(out_name)

        test_result_dir = args.run_root / "test_results" / target
        log_path = args.run_root / "logs" / f"aespa_infer_{target}.log"

        cmd = [
            PYTHON_EXE,
            "main.py",
            "--type", "test",
            "--content_dir", str(content_dir),
            "--style_dir", str(style_dir),
            "--train_result_dir", str(args.run_root / "checkpoints" / "aespa"),
            "--test_result_dir", str(test_result_dir),
            "--comment", "run511",
            "--batch_size", "1",
            "--num_workers", "0",
            "--imsize", "256",
            "--cropsize", "256",
        ]
        start = time.time()
        rc = run_cmd(cmd, AESPA_REPO, log_path)

        # Rename outputs
        renamed = 0
        if test_result_dir.exists():
            candidates = list(test_result_dir.rglob("*.jpg")) + list(test_result_dir.rglob("*.png"))
            for img in sorted(candidates):
                if renamed < len(selected):
                    shutil.copy2(img, output_dir / selected[renamed])
                    renamed += 1

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
    parser.add_argument("--mode", choices=["train", "infer", "all", "smoke", "preflight"], default="all")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="7g")
    parser.add_argument("--run_root", type=Path, default=RUN511_ROOT / "outputs" / "aespa_750")
    parser.add_argument("--reference_images_dir", type=Path, default=DEFAULT_REFERENCE_IMAGES)
    parser.add_argument("--max_iter", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--train_images_per_style", type=int, default=0)
    parser.add_argument("--limit_per_target", type=int, default=0)
    args = parser.parse_args()
    args.run_root = args.run_root.resolve()
    args.reference_images_dir = args.reference_images_dir.resolve()
    profile = PROFILES[args.profile]

    if args.mode == "preflight":
        result = check_assets()
        print(json.dumps(result, indent=2))
        return 0 if result["status"] == "ok" else 1

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
