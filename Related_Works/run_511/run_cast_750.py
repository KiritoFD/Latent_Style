"""Self-contained CAST train + 750-image inference launcher.

CAST (Zhang et al. SIGGRAPH 2022) uses contrastive learning for arbitrary
style transfer with a style encoder + AdaIN + generator architecture.

This script lives in run_511 and references run_511/repos/cast for model code.
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
WORKSPACE_ROOT = THIS_DIR.parent
CAST_REPO = THIS_DIR / "repos" / "cast"
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
    "4g": {"batch_size": 1, "train_images_per_style": 16, "n_epochs": 50, "n_epochs_decay": 50},
    "7g": {"batch_size": 1, "train_images_per_style": 32, "n_epochs": 100, "n_epochs_decay": 100},
    "11g": {"batch_size": 2, "train_images_per_style": 64, "n_epochs": 200, "n_epochs_decay": 200},
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
    with log_path.open("a", encoding="utf-8", errors="replace") as f:
        f.write("\n\n=== CMD ===\n")
        f.write(" ".join(cmd) + "\n")
        f.flush()
        proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=f, stderr=subprocess.STDOUT)
        return proc.wait()


def latest_cast_epoch(ckpt_dir: Path) -> str:
    """Return the newest CAST checkpoint epoch prefix usable by test.py."""
    if (ckpt_dir / "latest_net_AE.pth").exists():
        return "latest"
    epochs: list[int] = []
    for path in ckpt_dir.glob("*_net_AE.pth"):
        prefix = path.name.split("_net_AE.pth", 1)[0]
        if prefix.isdigit():
            epochs.append(int(prefix))
    if not epochs:
        raise FileNotFoundError(f"no *_net_AE.pth checkpoint found in {ckpt_dir}")
    return str(max(epochs))


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

def check_assets() -> dict[str, object]:
    missing = []
    if not (CAST_REPO / "models" / "vgg_normalised.pth").exists():
        missing.append("models/vgg_normalised.pth")
    if not (CAST_REPO / "models" / "style_vgg.pth").exists():
        missing.append("models/style_vgg.pth")
    if missing:
        return {"status": "blocked", "error": f"missing: {', '.join(missing)}"}
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def prepare_train_data(work_dir: Path, images_per_style: int) -> Path:
    """CAST training expects trainA/ (content) and trainB/ (style) under dataroot."""
    dataroot = work_dir / "cast_data"
    trainA = dataroot / "trainA"
    trainB = dataroot / "trainB"
    if dataroot.exists():
        shutil.rmtree(dataroot)
    copy_images(TRAIN_DATA / "photo", trainA, images_per_style)
    for style in STYLES:
        src = TRAIN_DATA / style
        if src.is_dir():
            copy_images(src, trainB, images_per_style)
    return dataroot


def train(args: argparse.Namespace, profile: dict[str, int]) -> dict[str, object]:
    assets = check_assets()
    if assets["status"] == "blocked":
        return {"stage": "train", **assets, "elapsed_sec": 0}

    images_per_style = int(args.train_images_per_style or profile["train_images_per_style"])
    n_epochs = profile["n_epochs"]
    n_epochs_decay = profile["n_epochs_decay"]
    batch_size = int(args.batch_size or profile["batch_size"])

    dataroot = prepare_train_data(args.run_root / "work", images_per_style)

    log_path = args.run_root / "logs" / "cast_train.log"
    cmd = [
        sys.executable, "train.py",
        "--dataroot", str(dataroot),
        "--name", "run511_cast",
        "--model", "cast",
        "--batch_size", str(batch_size),
        "--n_epochs", str(n_epochs),
        "--n_epochs_decay", str(n_epochs_decay),
        "--save_epoch_freq", str(max(1, min(n_epochs + n_epochs_decay, 5))),
        "--save_latest_freq", "100000000",
        "--checkpoints_dir", str(args.run_root / "checkpoints"),
        "--load_size", "286",
        "--crop_size", "256",
        "--no_html",
        "--display_id", "-1",
        "--gpu_ids", "0",
    ]
    start = time.time()
    rc = run_cmd(cmd, CAST_REPO, log_path)
    return {
        "stage": "train",
        "status": "ok" if rc == 0 else "failed",
        "returncode": rc,
        "elapsed_sec": round(time.time() - start, 3),
        "checkpoint_dir": str(args.run_root / "checkpoints" / "run511_cast"),
        "log_path": str(log_path),
        "n_epochs": n_epochs + n_epochs_decay,
        "batch_size": batch_size,
    }


# ---------------------------------------------------------------------------
# Infer
# ---------------------------------------------------------------------------

def infer(args: argparse.Namespace, profile: dict[str, int]) -> dict[str, object]:
    ckpt_dir = args.run_root / "checkpoints" / "run511_cast"
    if not ckpt_dir.exists():
        return {"stage": "infer", "status": "blocked", "error": f"no checkpoint: {ckpt_dir}"}
    try:
        epoch = latest_cast_epoch(ckpt_dir)
    except FileNotFoundError as exc:
        return {"stage": "infer", "status": "blocked", "error": str(exc)}

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

        # Prepare content/style dirs
        work_dir = args.run_root / "work" / "infer" / target
        dataroot = work_dir / "cast_data"
        testA = dataroot / "testA"
        testB = dataroot / "testB"
        if dataroot.exists():
            shutil.rmtree(dataroot)
        testA.mkdir(parents=True, exist_ok=True)
        testB.mkdir(parents=True, exist_ok=True)

        first_style = next(iter(sorted((OVERFIT50 / target).glob("*.jpg"))), None)
        if first_style is None:
            rows.append({"target": target, "returncode": 1, "renamed": 0, "error": "no style image"})
            continue
        shutil.copy2(first_style, testB / f"{target}.jpg")

        selected = []
        for out_name in target_ref:
            prefix = out_name[: -len(f"_to_{target}.jpg")]
            src_style, stem = prefix.split("_", 1)
            src = OVERFIT50 / src_style / f"{stem}.jpg"
            if src.exists():
                shutil.copy2(src, testA / f"{src_style}_{stem}.jpg")
                selected.append(out_name)

        result_dir = args.run_root / "results_cast" / target
        log_path = args.run_root / "logs" / f"cast_infer_{target}.log"

        cmd = [
            sys.executable, "test.py",
            "--dataroot", str(dataroot),
            "--name", "run511_cast",
            "--model", "cast",
            "--checkpoints_dir", str(args.run_root / "checkpoints"),
            "--results_dir", str(result_dir),
            "--epoch", epoch,
            "--eval",
            "--num_test", str(len(selected)),
            "--gpu_ids", "0",
        ]
        start = time.time()
        rc = run_cmd(cmd, CAST_REPO, log_path)

        # Rename generated outputs only. CAST also writes real/input images.
        renamed = 0
        img_dirs = [
            p
            for p in (list(result_dir.rglob("*.png")) + list(result_dir.rglob("*.jpg")))
            if "fake" in p.stem.lower()
        ]
        if not img_dirs:
            img_dirs = list(result_dir.rglob("*.png")) + list(result_dir.rglob("*.jpg"))
        for img in sorted(img_dirs):
            if renamed < len(selected):
                dst = output_dir / selected[renamed]
                if not dst.exists():
                    shutil.copy2(img, dst)
                renamed += 1

        total += renamed
        rows.append({
            "target": target,
            "returncode": rc,
            "renamed": renamed,
            "elapsed_sec": round(time.time() - start, 3),
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
    parser.add_argument("--run_root", type=Path, default=THIS_DIR / "outputs" / "cast_750")
    parser.add_argument("--reference_images_dir", type=Path, default=DEFAULT_REFERENCE_IMAGES)
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
        profile = {"batch_size": 1, "train_images_per_style": 2, "n_epochs": 1, "n_epochs_decay": 0}
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
        if rows[-1]["status"] not in {"ok", "partial"}:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
