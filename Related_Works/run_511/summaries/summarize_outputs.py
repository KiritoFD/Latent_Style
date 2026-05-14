"""Summarize existing run_511 outputs into CSV and Markdown."""
from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def all_row(eval_data: dict) -> dict:
    for row in eval_data.get("results", []):
        if str(row.get("target", "")).upper() == "ALL":
            return row
    return {}


def summarize_run(path: Path) -> dict:
    summary = load_json(path / "summary.json")
    eval_data = load_json(path / "infer_750" / "eval_fixed.json") or load_json(path / "infer_750" / "eval.json")
    all_metrics = all_row(eval_data)
    images_dir = path / "infer_750" / "images"
    ckpt_dir = path / "checkpoints"
    image_count = len(list(images_dir.glob("*.jpg"))) if images_dir.exists() else 0
    ckpt_count = 0
    if ckpt_dir.exists():
        for pattern in ("*.pth", "*.pt", "*.ckpt", "*.pkl", "*.model"):
            ckpt_count += len(list(ckpt_dir.rglob(pattern)))

    train_sec = ""
    infer_sec = ""
    train_status = ""
    infer_status = ""
    for run in summary.get("runs", []):
        if run.get("stage") == "train":
            train_sec = run.get("elapsed_sec", "")
            train_status = run.get("status", "")
        if run.get("stage") == "infer":
            infer_sec = run.get("elapsed_sec", "")
            infer_status = run.get("status", "")

    return {
        "run": path.name,
        "images": image_count,
        "checkpoints": ckpt_count,
        "has_summary": bool(summary),
        "has_eval": bool(eval_data),
        "train_status": train_status,
        "train_sec": train_sec,
        "infer_status": infer_status,
        "infer_sec": infer_sec,
        "lpips": all_metrics.get("lpips", ""),
        "clip_style": all_metrics.get("clip_style", ""),
        "clip_content": all_metrics.get("clip_content", ""),
        "updated": path.stat().st_mtime,
    }


def fmt(value: object) -> str:
    if value == "" or value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def main() -> int:
    rows = [summarize_run(p) for p in sorted(OUT.iterdir()) if p.is_dir()]
    docs_dir = ROOT / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    csv_path = docs_dir / "outputs_inventory.csv"
    md_path = docs_dir / "outputs_inventory.md"
    keys = [
        "run",
        "images",
        "checkpoints",
        "has_summary",
        "has_eval",
        "train_status",
        "train_sec",
        "infer_status",
        "infer_sec",
        "lpips",
        "clip_style",
        "clip_content",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})

    lines = [
        "# run_511 Outputs Inventory",
        "",
        "| Run | Images | Ckpts | Train | Train sec | Infer | Infer sec | LPIPS | CLIP-style | CLIP-content |",
        "| --- | ---: | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {run} | {images} | {checkpoints} | {train_status} | {train_sec} | {infer_status} | {infer_sec} | {lpips} | {clip_style} | {clip_content} |".format(
                run=row["run"],
                images=row["images"],
                checkpoints=row["checkpoints"],
                train_status=row["train_status"],
                train_sec=fmt(row["train_sec"]),
                infer_status=row["infer_status"],
                infer_sec=fmt(row["infer_sec"]),
                lpips=fmt(row["lpips"]),
                clip_style=fmt(row["clip_style"]),
                clip_content=fmt(row["clip_content"]),
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
