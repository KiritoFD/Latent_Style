from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _epoch_key(path: Path) -> int:
    stem = path.name
    if stem.startswith("epoch_"):
        try:
            return int(stem.split("_", 1)[1])
        except ValueError:
            return 10**9
    return 10**9


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a manifest from local epoch_* eval directories.")
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--label-prefix", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--require-images", action="store_true")
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    eval_root = Path(args.eval_root)
    for epoch_dir in sorted([p for p in eval_root.iterdir() if p.is_dir() and p.name.startswith("epoch_")], key=_epoch_key):
        images_dir = epoch_dir / "images"
        metrics_csv = epoch_dir / "metrics.csv"
        summary_json = epoch_dir / "summary.json"
        if not metrics_csv.is_file():
            continue
        if args.require_images and not images_dir.is_dir():
            continue
        epoch_num = _epoch_key(epoch_dir)
        label = f"{args.label_prefix} {epoch_num:02d}" if epoch_num < 10**9 else f"{args.label_prefix} {epoch_dir.name}"
        rows.append(
            {
                "method": str(args.method),
                "label": label,
                "run": epoch_dir.name,
                "images_dir": str(images_dir),
                "metrics_csv": str(metrics_csv),
                "introstyle_summary": str(summary_json) if summary_json.is_file() else "",
                "source_root": str(args.source_root),
            }
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["method", "label", "run", "images_dir", "metrics_csv", "introstyle_summary", "source_root"]
    with args.output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(args.output_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
