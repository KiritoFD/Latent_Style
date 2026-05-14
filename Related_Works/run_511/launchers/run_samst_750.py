"""Self-contained SaMST train + 750-image inference launcher.

SaMST (Liu et al. ACCV 2024) uses a lightweight TransformerNet with dynamic
convolutions conditioned on a compact 32-dim style representation.

This script lives in run_511 and uses run_511/repos/SaMST-main for model code.
It wraps the existing train2/test subprocess pipelines.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml


THIS_DIR = Path(__file__).resolve().parent
RUN511_ROOT = THIS_DIR.parent
WORKSPACE_ROOT = RUN511_ROOT.parent.parent
SAMST_REPO = RUN511_ROOT / "repos" / "SaMST-main"
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
    "4g": {"batch_size": 1, "train_images_per_style": 16, "epochs": 30},
    "7g": {"batch_size": 2, "train_images_per_style": 32, "epochs": 100},
    "11g": {"batch_size": 4, "train_images_per_style": 64, "epochs": 100},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def prepare_dataset(style_name: str, images_per_style: int) -> Path:
    """Prepare dataset in SaMST format."""
    dataset_dir = SAMST_REPO / "train_dataset"
    content_dir = dataset_dir / "content" / "content"
    style_dir = dataset_dir / "style"
    shutil.rmtree(str(content_dir), ignore_errors=True)
    shutil.rmtree(str(style_dir), ignore_errors=True)
    content_dir.mkdir(parents=True, exist_ok=True)
    style_dir.mkdir(parents=True, exist_ok=True)

    photo_src = TRAIN_DATA / "photo"
    if photo_src.exists():
        for img in sorted(photo_src.glob("*.jpg"))[:images_per_style]:
            dst = content_dir / img.name
            if not dst.exists():
                shutil.copy2(str(img), str(dst))

    style_src = TRAIN_DATA / style_name
    if style_src.exists():
        for img in sorted(style_src.glob("*.jpg"))[:1]:
            dst = style_dir / img.name
            if not dst.exists():
                shutil.copy2(str(img), str(dst))

    return dataset_dir


# ---------------------------------------------------------------------------
# Train (per style)
# ---------------------------------------------------------------------------

def train_one_style(style_name: str, epochs: int, batch_size: int, checkpoint_dir: Path) -> int:
    """Train SaMST for one style using train2 pipeline."""
    train_dir = SAMST_REPO / "train_model" / "train2"
    train_script = train_dir / "train.py"
    if not train_script.exists():
        print(f"[ERROR] Training script not found: {train_script}", flush=True)
        return 1

    config = {
        "epochs": epochs,
        "batch_size": batch_size,
        "dataset": "../../train_dataset/content/",
        "style_image": "../../train_dataset/style/",
        "save_model_dir": str(checkpoint_dir),
        "image_size": 256,
        "style_size": 512,
        "cuda": 1,
        "seed": 7,
        "content_weight": 1e5,
        "style_weight": 1e10,
        "ae_weight": 1e3,
        "lr": 0.001,
        "weight_decay": 0.5,
        "step_size": 25,
        "save_interval": min(10, max(1, epochs)),
        "log_interval": 10,
        "checkpoint_interval": 100,
        "checkpoint_model_dir": None,
        "begin_checkpoint": None,
        "begin_epoch": None,
    }

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config_path = train_dir / "train.yml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    result = subprocess.run(
        [sys.executable, str(train_script)],
        cwd=str(train_dir),
    )
    return result.returncode


def selected_styles(args: argparse.Namespace) -> list[str]:
    if not args.styles:
        return STYLES
    names = [s.strip() for s in args.styles.split(",") if s.strip()]
    unknown = [s for s in names if s not in STYLES]
    if unknown:
        raise ValueError(f"unknown styles: {unknown}; valid={STYLES}")
    return names


def train(args: argparse.Namespace, profile: dict[str, int]) -> dict[str, object]:
    if not SAMST_REPO.exists():
        return {"stage": "train", "status": "blocked", "error": f"SaMST repo not found: {SAMST_REPO}"}

    epochs = int(args.epochs_override or profile["epochs"])
    batch_size = int(args.batch_size or profile["batch_size"])
    images_per_style = int(args.train_images_per_style or profile["train_images_per_style"])
    ckpt_root = args.run_root / "checkpoints" / "samst"

    start = time.time()
    rows = []
    for style in selected_styles(args):
        prepare_dataset(style, images_per_style)
        style_ckpt = ckpt_root / style
        print(f"\n[SaMST TRAIN] style={style}, epochs={epochs}, batch={batch_size}", flush=True)
        rc = train_one_style(style, epochs, batch_size, style_ckpt)
        rows.append({"style": style, "returncode": rc})
        if rc != 0:
            return {
                "stage": "train",
                "status": "failed",
                "returncode": rc,
                "elapsed_sec": round(time.time() - start, 3),
                "per_style": rows,
            }

    return {
        "stage": "train",
        "status": "ok",
        "returncode": 0,
        "elapsed_sec": round(time.time() - start, 3),
        "checkpoint_dir": str(ckpt_root),
        "epochs": epochs,
        "batch_size": batch_size,
        "train_images_per_style": images_per_style,
        "per_style": rows,
    }


# ---------------------------------------------------------------------------
# Infer (per style)
# ---------------------------------------------------------------------------

def infer_one_style(target_style: str, reference: list[str], limit: int,
                    ckpt_root: Path, output_dir: Path) -> dict[str, object]:
    """Run SaMST inference for one target style."""
    test_dir = SAMST_REPO / "test_model" / "test"
    test_script = test_dir / "test.py"
    if not test_script.exists():
        return {"target": target_style, "returncode": 1, "error": f"test script not found: {test_script}"}

    # Find trained model
    style_ckpt = ckpt_root / target_style
    model_files = sorted(style_ckpt.glob("epoch_*.model"))
    if not model_files:
        return {"target": target_style, "returncode": 1, "error": f"no model in {style_ckpt}"}
    model_path = model_files[-1]
    state_dict = __import__("torch").load(str(model_path), map_location="cpu")
    style_bank_ids = sorted(
        int(k.split("style_para_list.", 1)[1].split(".", 1)[0])
        for k in state_dict
        if "style_bank.style_para_list." in k and k.endswith(".params")
    )
    if not style_bank_ids:
        return {"target": target_style, "returncode": 1, "error": f"no style bank params in {model_path}"}
    desired_style_id = max(style_bank_ids)
    style_num = desired_style_id

    # Prepare test content images
    test_content = SAMST_REPO / "content"
    shutil.rmtree(str(test_content), ignore_errors=True)
    test_content.mkdir(parents=True, exist_ok=True)

    target_ref = [n for n in reference if n.endswith(f"_to_{target_style}.jpg")]
    if limit > 0:
        target_ref = target_ref[:limit]

    for out_name in target_ref:
        prefix = out_name[: -len(f"_to_{target_style}.jpg")]
        src_style, stem = prefix.split("_", 1)
        src = OVERFIT50 / src_style / f"{stem}.jpg"
        if src.exists():
            shutil.copy2(str(src), test_content / f"{src_style}_{stem}.jpg")

    # Write test config
    raw_output = SAMST_REPO / "outputs"
    config = {
        "content_image_dir": str(test_content),
        "content_scale": None,
        "output_image_dir": str(raw_output) + "/",
        "model": str(model_path),
        "style_num": style_num,
        "cuda": 1,
    }
    config_path = test_dir / "test.yml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Run test
    start = time.time()
    result = subprocess.run(
        [sys.executable, str(test_script)],
        cwd=str(test_dir),
    )
    if result.returncode != 0:
        return {"target": target_style, "returncode": result.returncode, "elapsed_sec": round(time.time() - start, 3)}

    # Rename only the actual target style output. style0 is the AE/identity branch.
    renamed = 0
    prefix = f"style{desired_style_id}_"
    for f in sorted(raw_output.glob(f"{prefix}*.jpg")):
        original = f.name[len(prefix):]
        parts = original.split("_", 1)
        if len(parts) == 2:
            content_style, img_name = parts
            new_name = f"{content_style}_{img_name.replace('.jpg', '')}_to_{target_style}.jpg"
        else:
            new_name = original.replace(".jpg", f"_to_{target_style}.jpg")
        dst = output_dir / new_name
        shutil.copy2(str(f), str(dst))
        renamed += 1

    shutil.rmtree(str(raw_output), ignore_errors=True)
    return {
        "target": target_style,
        "returncode": 0,
        "renamed": renamed,
        "elapsed_sec": round(time.time() - start, 3),
        "style_output_id": desired_style_id,
    }


def infer(args: argparse.Namespace, profile: dict[str, int]) -> dict[str, object]:
    if not SAMST_REPO.exists():
        return {"stage": "infer", "status": "blocked", "error": f"SaMST repo not found: {SAMST_REPO}"}

    ckpt_root = args.run_root / "checkpoints" / "samst"
    output_dir = args.run_root / "infer_750" / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    reference = reference_names(args.reference_images_dir)

    rows = []
    start_all = time.time()
    total = 0

    for target in selected_styles(args):
        print(f"\n[SaMST INFER] target={target}", flush=True)
        row = infer_one_style(target, reference, args.limit_per_target, ckpt_root, output_dir)
        rows.append(row)
        total += row.get("renamed", 0)
        if row.get("returncode", 0) != 0:
            break

    status = "ok"
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
        keys = sorted({k for row in rows for k in row.keys() if k != "per_target" and k != "per_style"})
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in keys})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "infer", "all", "smoke"], default="all")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="7g")
    parser.add_argument("--run_root", type=Path, default=RUN511_ROOT / "outputs" / "samst_750")
    parser.add_argument("--reference_images_dir", type=Path, default=DEFAULT_REFERENCE_IMAGES)
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--train_images_per_style", type=int, default=0)
    parser.add_argument("--epochs_override", type=int, default=0)
    parser.add_argument("--limit_per_target", type=int, default=0, help="0 means full 150 per target / 750 total.")
    parser.add_argument("--styles", default="", help="Comma-separated subset, e.g. photo,Hayao. Empty means all styles.")
    args = parser.parse_args()
    args.run_root = args.run_root.resolve()
    args.reference_images_dir = args.reference_images_dir.resolve()
    profile = PROFILES[args.profile]
    if args.mode == "smoke":
        profile = {"batch_size": 1, "train_images_per_style": 2, "epochs": 1}
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
