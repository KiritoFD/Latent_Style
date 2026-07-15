from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an IntroStyle manifest from a best-few handoff CSV.")
    parser.add_argument("--handoff-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--method", default="LBM")
    parser.add_argument("--label-prefix", default="BestFew")
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()

    rows = _read_rows(args.handoff_csv)
    out_rows: list[dict[str, str]] = []
    for row in rows:
        epoch = str(row["epoch"]).strip()
        images_dir = Path(str(row["images_dir"]).strip())
        metrics_csv = Path(str(row["metrics_csv"]).strip())
        summary_json = Path(str(row["summary_json"]).strip())
        # Local best-few handoff tables may include a "latest" row whose local packet
        # has not yet been mirrored. Skip unresolved rows instead of poisoning downstream
        # IntroStyle/DINO review with missing paths.
        if not images_dir.is_dir() or not metrics_csv.is_file() or not summary_json.is_file():
            print(
                f"SKIP {epoch}: missing local bestfew artifact "
                f"(images={images_dir.is_dir()} metrics={metrics_csv.is_file()} summary={summary_json.is_file()})",
                flush=True,
            )
            continue
        out_rows.append(
            {
                "method": str(args.method),
                "label": f"{args.label_prefix} {epoch}",
                "run": epoch,
                "images_dir": str(images_dir),
                "metrics_csv": str(metrics_csv),
                "introstyle_summary": str(summary_json),
                "source_root": str(args.source_root),
            }
        )

    if not out_rows:
        raise RuntimeError("No valid local bestfew rows survived manifest filtering.")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["method", "label", "run", "images_dir", "metrics_csv", "introstyle_summary", "source_root"]
    with args.output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    print(args.output_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
