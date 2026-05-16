from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
ABLATION_ROOT = ROOT / "ablation_destructive_7epoch"
RELATED_METRICS_DIR = ROOT.parent / "Related_Works" / "results" / "metrics_summary"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def read_registry() -> list[dict[str, str]]:
    path = ABLATION_ROOT / "destructive_ablation_7epoch_registry.csv"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_metrics(run_dir: Path) -> dict[str, Any]:
    summary = read_json(run_dir / "full_eval" / "epoch_0007" / "summary.json")
    analysis = summary.get("analysis", {})
    all_pairs = analysis.get("all_pairs_overview", {}) or {}
    transfer = analysis.get("style_transfer_ability", {}) or {}
    photo = analysis.get("photo_to_art_performance", {}) or {}
    return {
        "clip_style": all_pairs.get("clip_style", ""),
        "clip_content": all_pairs.get("clip_content", ""),
        "content_lpips": all_pairs.get("content_lpips", ""),
        "transfer_clip_style": transfer.get("clip_style", ""),
        "transfer_clip_content": transfer.get("clip_content", ""),
        "transfer_content_lpips": transfer.get("content_lpips", ""),
        "photo_to_art_clip_style": photo.get("clip_style", ""),
        "photo_to_art_clip_content": photo.get("clip_content", ""),
        "photo_to_art_content_lpips": photo.get("content_lpips", ""),
        "cmmd_all": all_pairs.get("cmmd", ""),
        "dino_structure_all": all_pairs.get("dino_structure", ""),
        "gram_micro_all": all_pairs.get("gram_micro", ""),
        "gram_macro_all": all_pairs.get("gram_macro", ""),
        "summary_exists": bool(summary),
    }


def train_epoch_time(run_dir: Path) -> dict[str, Any]:
    logs = sorted((run_dir / "logs").glob("training_*.csv"))
    if not logs:
        return {"train_logged_epochs": "", "train_total_epoch_sec": "", "train_avg_epoch_sec": ""}
    with logs[-1].open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    secs = []
    for row in rows:
        try:
            secs.append(float(row.get("epoch_time_sec", "")))
        except Exception:
            pass
    total = sum(secs)
    return {
        "train_logged_epochs": len(secs),
        "train_total_epoch_sec": round(total, 3) if secs else "",
        "train_avg_epoch_sec": round(total / len(secs), 3) if secs else "",
    }


def build_rows() -> list[dict[str, Any]]:
    rows = []
    for reg in read_registry():
        run_dir = Path(reg.get("save_dir", ""))
        row: dict[str, Any] = {
            "id": reg.get("id", ""),
            "label": reg.get("label", ""),
            "purpose": reg.get("purpose", ""),
            "status": reg.get("status", ""),
            "train_sec_wall": reg.get("train_sec", ""),
            "eval_status": reg.get("eval_status", ""),
            "eval_sec_wall": reg.get("eval_sec", ""),
            "run_dir": str(run_dir),
        }
        row.update(train_epoch_time(run_dir))
        row.update(load_metrics(run_dir))
        rows.append(row)
    return rows


def fmt(value: Any, digits: int = 4) -> str:
    if value in ("", None):
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def write_reports(rows: list[dict[str, Any]]) -> None:
    ABLATION_ROOT.mkdir(parents=True, exist_ok=True)
    RELATED_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    csv_paths = [
        ABLATION_ROOT / "destructive_ablation_7epoch_summary.csv",
        RELATED_METRICS_DIR / "destructive_ablation_7epoch_summary.csv",
    ]
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    for path in csv_paths:
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)

    lines = [
        "# Destructive 7-Epoch Ablation Summary",
        "",
        "| ID | Label | Status | Eval | LPIPS down | CLIP-style up | CLIP-content up | Train sec | Eval sec |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| `{row.get('id', '')}` | {row.get('label', '')} | {row.get('status', '')} | "
            f"{row.get('eval_status', '')} | {fmt(row.get('content_lpips'))} | "
            f"{fmt(row.get('clip_style'))} | {fmt(row.get('clip_content'))} | "
            f"{fmt(row.get('train_sec_wall'))} | {fmt(row.get('eval_sec_wall'))} |"
        )
    for path in [
        ABLATION_ROOT / "destructive_ablation_7epoch_summary.md",
        RELATED_METRICS_DIR / "destructive_ablation_7epoch_summary.md",
    ]:
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    rows = build_rows()
    write_reports(rows)
    print(ABLATION_ROOT / "destructive_ablation_7epoch_summary.csv")
    print(RELATED_METRICS_DIR / "destructive_ablation_7epoch_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
