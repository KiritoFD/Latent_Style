"""Conservative local-training launcher for new style-transfer baselines.

The launcher is intentionally serial. It prepares tiny local training subsets,
runs only baselines that have the minimum local assets, and records blocked
baselines instead of attempting expensive downloads or heavyweight training.
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


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = PIPELINE_ROOT.parent.parent
RELATED_ROOT = WORKSPACE_ROOT / "Related_Works"
STYLE_DATA = WORKSPACE_ROOT / "style_data" / "train"
RUN_ROOT_DEFAULT = RELATED_ROOT / "runs" / "new_baseline_train"
STYLE_NAMES = ["monet", "vangogh", "cezanne", "Hayao"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def copy_subset(src_dir: Path, dst_dir: Path, limit: int) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for img in sorted(src_dir.iterdir()):
        if not img.is_file() or img.suffix.lower() not in IMG_EXTS:
            continue
        shutil.copy2(img, dst_dir / img.name)
        count += 1
        if count >= limit:
            break
    return count


def prepare_flat_subset(root: Path, images_per_style: int) -> tuple[Path, Path]:
    content_dir = root / "flat" / "content"
    style_dir = root / "flat" / "style"
    if content_dir.exists():
        shutil.rmtree(content_dir)
    if style_dir.exists():
        shutil.rmtree(style_dir)
    copy_subset(STYLE_DATA / "photo", content_dir, images_per_style)
    for style in STYLE_NAMES:
        copy_subset(STYLE_DATA / style, style_dir / style, images_per_style)
    return content_dir, style_dir


def prepare_aesfa_subset(root: Path, images_per_style: int) -> tuple[Path, Path]:
    content_root = root / "aesfa_content"
    style_root = root / "aesfa_style"
    if content_root.exists():
        shutil.rmtree(content_root)
    if style_root.exists():
        shutil.rmtree(style_root)
    copy_subset(STYLE_DATA / "photo", content_root / "train", images_per_style)
    for style in STYLE_NAMES:
        copy_subset(STYLE_DATA / style, style_root / "train", images_per_style)
    return content_root, style_root


def run_cmd(cmd: list[str], cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", errors="replace") as f:
        f.write("\n\n=== CMD ===\n")
        f.write(" ".join(cmd) + "\n")
        f.flush()
        proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=f, stderr=subprocess.STDOUT)
        return proc.wait()


def append_rows(rows: list[dict[str, object]], run_root: Path) -> None:
    csv_path = run_root / "train_status.csv"
    json_path = run_root / "train_status.json"
    if rows:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            fieldnames = sorted({key for row in rows for key in row.keys()})
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    json_path.write_text(json.dumps({"runs": rows}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def train_aesfa(args: argparse.Namespace, rows: list[dict[str, object]]) -> None:
    repo = RELATED_ROOT / "AesFA"
    vgg = repo / "vgg_normalised.pth"
    start = time.time()
    row: dict[str, object] = {"baseline": "aesfa", "status": "started"}
    if not vgg.exists():
        row.update({"status": "blocked", "error": f"missing {vgg}"})
        rows.append(row)
        return

    content_root, style_root = prepare_aesfa_subset(args.run_root / "datasets", args.images_per_style)
    ckpt_dir = args.run_root / "checkpoints" / "aesfa"
    log_dir = args.run_root / "tb_logs" / "aesfa"
    log_path = args.run_root / "logs" / "aesfa_train.log"
    code = (
        "import os, sys; "
        f"repo={str(repo)!r}; os.chdir(repo); sys.path.insert(0, repo); "
        "from Config import Config; "
        f"Config.phase='train'; Config.train_continue='off'; Config.data_num={args.images_per_style}; "
        f"Config.content_dir={str(content_root)!r}; Config.style_dir={str(style_root)!r}; "
        f"Config.file_n='local_smoke'; Config.log_dir={str(log_dir)!r}; Config.ckpt_dir={str(ckpt_dir)!r}; "
        f"Config.img_dir={str(args.run_root / 'preview' / 'aesfa')!r}; Config.vgg_model={str(vgg)!r}; "
        f"Config.n_iter={args.aesfa_iters}; Config.save_interval={args.aesfa_iters}; "
        f"Config.batch_size={args.batch_size}; Config.num_workers=0; Config.load_size={args.load_size}; Config.crop_size={args.crop_size}; "
        "import train; train.main()"
    )
    rc = run_cmd([str(args.python), "-c", code], repo, log_path)
    row.update(
        {
            "status": "ok" if rc == 0 else "failed",
            "returncode": rc,
            "elapsed_sec": round(time.time() - start, 3),
            "checkpoint_dir": str(ckpt_dir),
            "log_path": str(log_path),
            "iters": args.aesfa_iters,
        }
    )
    rows.append(row)


def train_stytr2(args: argparse.Namespace, rows: list[dict[str, object]]) -> None:
    repo = RELATED_ROOT / "StyTR-2"
    exp_dir = repo / "experiments"
    vgg = exp_dir / "vgg_normalised.pth"
    fallback_vgg = RELATED_ROOT / "AesFA" / "vgg_normalised.pth"
    start = time.time()
    row: dict[str, object] = {"baseline": "stytr2", "status": "started"}
    exp_dir.mkdir(parents=True, exist_ok=True)
    if not vgg.exists() and fallback_vgg.exists():
        shutil.copy2(fallback_vgg, vgg)
    if not vgg.exists():
        row.update({"status": "blocked", "error": f"missing {vgg}"})
        rows.append(row)
        return

    content_dir, style_dir = prepare_flat_subset(args.run_root / "datasets", args.images_per_style)
    save_dir = args.run_root / "checkpoints" / "stytr2"
    log_dir = args.run_root / "tb_logs" / "stytr2"
    log_path = args.run_root / "logs" / "stytr2_train.log"
    cmd = [
        str(args.python),
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
        str(args.stytr2_iters),
        "--batch_size",
        str(args.batch_size),
        "--n_threads",
        "0",
        "--save_model_interval",
        str(args.stytr2_iters),
    ]
    rc = run_cmd(cmd, repo, log_path)
    row.update(
        {
            "status": "ok" if rc == 0 else "failed",
            "returncode": rc,
            "elapsed_sec": round(time.time() - start, 3),
            "checkpoint_dir": str(save_dir),
            "log_path": str(log_path),
            "iters": args.stytr2_iters,
        }
    )
    rows.append(row)


def preflight_aespa(args: argparse.Namespace, rows: list[dict[str, object]]) -> None:
    repo = RELATED_ROOT / "AesPA-Net"
    required = repo / "baseline_checkpoints" / "vgg_normalised_conv5_1.t7"
    status = "ready" if required.exists() else "blocked"
    rows.append(
        {
            "baseline": "aespa-net",
            "status": status,
            "error": "" if status == "ready" else f"missing {required}",
            "note": "training wrapper pending after encoder checkpoint is present",
        }
    )


def preflight_artbank(args: argparse.Namespace, rows: list[dict[str, object]]) -> None:
    repo = RELATED_ROOT / "ArtBank"
    sd = repo / "models" / "sd" / "sd-v1-4.ckpt"
    embeddings = list(repo.glob("logs/**/embeddings.pt"))
    mapper = list(repo.glob("logs/**/Mapper.pt"))
    ok = sd.exists() and embeddings and mapper
    rows.append(
        {
            "baseline": "artbank",
            "status": "ready" if ok else "blocked",
            "error": "" if ok else "missing sd-v1-4.ckpt and/or ArtBank embeddings.pt/Mapper.pt",
            "note": "heavy diffusion baseline; do not train until required assets are local",
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Train/preflight new baselines serially.")
    parser.add_argument("--baselines", nargs="+", default=["aesfa", "stytr2", "aespa-net", "artbank"])
    parser.add_argument("--run_root", type=Path, default=RUN_ROOT_DEFAULT)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--images_per_style", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--load_size", type=int, default=128)
    parser.add_argument("--crop_size", type=int, default=128)
    parser.add_argument("--aesfa_iters", type=int, default=20)
    parser.add_argument("--stytr2_iters", type=int, default=20)
    args = parser.parse_args()
    args.run_root = args.run_root.resolve()
    args.python = args.python.resolve()
    args.run_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    handlers = {
        "aesfa": train_aesfa,
        "stytr2": train_stytr2,
        "aespa": preflight_aespa,
        "aespa-net": preflight_aespa,
        "artbank": preflight_artbank,
    }
    for name in args.baselines:
        if name.lower() in {"prepare-data", "data"}:
            prepare_flat_subset(args.run_root / "datasets", args.images_per_style)
            prepare_aesfa_subset(args.run_root / "datasets", args.images_per_style)
            rows.append({"baseline": "prepare-data", "status": "ok", "run_root": str(args.run_root)})
            append_rows(rows, args.run_root)
            continue
        handler = handlers.get(name.lower())
        if handler is None:
            rows.append({"baseline": name, "status": "unknown", "error": "no handler"})
        else:
            handler(args, rows)
        append_rows(rows, args.run_root)

    return 0 if all(row.get("status") in {"ok", "ready", "blocked"} for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
