from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _summary_curve_row(summary_path: Path) -> dict[str, object]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    analysis = payload.get("analysis") or {}
    full = analysis.get("all_pairs_overview") or {}
    transfer = analysis.get("style_transfer_ability") or {}
    timings = payload.get("timings_sec") or {}
    return {
        "epoch": summary_path.parent.name,
        "full_clip_style": full.get("clip_style"),
        "full_clip_s_delta_idt": full.get("clip_s_delta_idt"),
        "full_clip_t": full.get("clip_t"),
        "full_content_lpips": full.get("content_lpips"),
        "transfer_clip_style": transfer.get("clip_style"),
        "transfer_clip_s_delta_idt": transfer.get("clip_s_delta_idt"),
        "transfer_clip_t": transfer.get("clip_t"),
        "transfer_content_lpips": transfer.get("content_lpips"),
        "wall_total_seconds": timings.get("wall_total"),
        "summary_path": str(summary_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build clip/lpips curve CSV by scanning epoch_*/summary.json under an eval root.")
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    eval_root = Path(args.eval_root).resolve()
    rows = []
    for summary_path in sorted(eval_root.glob("epoch_*/summary.json")):
        rows.append(_summary_curve_row(summary_path))
    if not rows:
        raise RuntimeError(f"No summary.json files found under {eval_root}")

    out_csv = Path(args.output_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
