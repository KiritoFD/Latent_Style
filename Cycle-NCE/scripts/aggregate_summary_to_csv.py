#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from datetime import datetime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate Cycle-NCE summary/*.json files into a single CSV."
    )
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "summary",
        help="Directory containing summary JSON files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: <summary-dir>/summary_aggregate.csv).",
    )
    return parser.parse_args()


def collect_rows(summary_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(summary_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        latest = data.get("latest") or {}
        mean = data.get("mean") or {}
        rows.append(
            {
                "experiment": path.stem,
                "num_rounds": data.get("num_rounds"),
                "updated_at": data.get("updated_at"),
                "latest_epoch": latest.get("epoch"),
                "latest_summary_path": latest.get("summary_path"),
                "latest_transfer_clip_style": latest.get("transfer_clip_style"),
                "latest_transfer_content_lpips": latest.get("transfer_content_lpips"),
                "latest_transfer_fid": latest.get("transfer_fid"),
                "latest_transfer_art_fid": latest.get("transfer_art_fid"),
                "latest_transfer_classifier_acc": latest.get("transfer_classifier_acc"),
                "latest_photo_to_art_clip_style": latest.get("photo_to_art_clip_style"),
                "latest_photo_to_art_fid": latest.get("photo_to_art_fid"),
                "latest_photo_to_art_art_fid": latest.get("photo_to_art_art_fid"),
                "latest_photo_to_art_classifier_acc": latest.get("photo_to_art_classifier_acc"),
                "mean_transfer_clip_style": mean.get("transfer_clip_style"),
                "mean_transfer_content_lpips": mean.get("transfer_content_lpips"),
                "mean_transfer_fid": mean.get("transfer_fid"),
                "mean_transfer_art_fid": mean.get("transfer_art_fid"),
                "mean_transfer_classifier_acc": mean.get("transfer_classifier_acc"),
                "mean_photo_to_art_clip_style": mean.get("photo_to_art_clip_style"),
                "mean_photo_to_art_fid": mean.get("photo_to_art_fid"),
                "mean_photo_to_art_art_fid": mean.get("photo_to_art_art_fid"),
                "mean_photo_to_art_classifier_acc": mean.get("photo_to_art_classifier_acc"),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    summary_dir = args.summary_dir.resolve()
    if not summary_dir.is_dir():
        raise FileNotFoundError(f"Summary directory not found: {summary_dir}")

    output = (args.output or (summary_dir / "summary_aggregate.csv")).resolve()
    rows = collect_rows(summary_dir)

    if rows:
        fieldnames = list(rows[0].keys())
    else:
        fieldnames = ["experiment", "num_rounds", "updated_at", "latest_epoch", "latest_summary_path"]

    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    except PermissionError:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback = output.with_name(f"{output.stem}_{stamp}{output.suffix}")
        with fallback.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        output = fallback

    print(f"Wrote {len(rows)} rows -> {output}")


if __name__ == "__main__":
    main()
