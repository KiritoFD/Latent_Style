from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDNAMES = [
    "checkpoint",
    "summary_path",
    "wall_total_sec",
    "lancet_generation_sec",
    "lpips_sec",
    "clip_sec",
    "generated_count",
    "transfer_clip_style",
    "transfer_content_lpips",
    "allpairs_clip_style",
    "allpairs_content_lpips",
]


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _read_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a compact full-eval runtime table from summary.json files.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--eval-subdir", default="full_eval")
    parser.add_argument("--csv-out", type=Path, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    output_root = run_dir / str(args.eval_subdir)
    csv_out = args.csv_out.resolve() if args.csv_out else (run_dir / "logs" / "full_eval_runtime_rebuilt.csv")

    rows: list[dict[str, object]] = []
    for summary_path in sorted(output_root.glob("epoch_*/summary.json")):
        summary = _read_summary(summary_path)
        timings = dict((summary.get("timings_sec") or {}))
        analysis = dict((summary.get("analysis") or {}))
        transfer = dict((analysis.get("style_transfer_ability") or {}))
        allpairs = dict((analysis.get("all_pairs_overview") or {}))
        rows.append(
            {
                "checkpoint": summary_path.parent.name,
                "summary_path": str(summary_path),
                "wall_total_sec": _safe_float(timings.get("wall_total")),
                "lancet_generation_sec": _safe_float(timings.get("lancet_generation")),
                "lpips_sec": _safe_float(timings.get("lpips")),
                "clip_sec": _safe_float(timings.get("clip")),
                "generated_count": int(summary.get("generated_count", 0) or 0),
                "transfer_clip_style": _safe_float(transfer.get("clip_style")),
                "transfer_content_lpips": _safe_float(transfer.get("content_lpips")),
                "allpairs_clip_style": _safe_float(allpairs.get("clip_style")),
                "allpairs_content_lpips": _safe_float(allpairs.get("content_lpips")),
            }
        )

    csv_out.parent.mkdir(parents=True, exist_ok=True)
    with csv_out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[build_full_eval_runtime_table] wrote {csv_out} rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
