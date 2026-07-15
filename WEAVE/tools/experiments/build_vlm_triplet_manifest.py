from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _find_by_run(rows: list[dict[str, str]], run: str) -> dict[str, str]:
    for row in rows:
        if str(row.get("run", "")).strip() == str(run).strip():
            return row
    raise KeyError(run)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a 3-run VLM comparison manifest from existing local review tables.")
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--candidate-run", required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--candidate-method", default="LBM")
    parser.add_argument("--candidate-label", required=True)
    parser.add_argument(
        "--baseline-runs",
        nargs="+",
        default=["LBM-Knee_e13", "Seedream_repaired750"],
        help="One or more baseline runs to copy from --baseline-manifest in the requested order.",
    )
    args = parser.parse_args()

    base_rows = _read_rows(args.baseline_manifest)
    cand_rows = _read_rows(args.candidate_manifest)
    cand = _find_by_run(cand_rows, args.candidate_run)

    out_rows = []
    for baseline_run in args.baseline_runs:
        row = _find_by_run(base_rows, baseline_run)
        out_rows.append(
            {
                "method": str(row.get("method", "")).strip(),
                "run": str(row["run"]).strip(),
                "images_dir": row["images_dir"],
                "source_root": row["source_root"],
                "metrics_csv": row["metrics_csv"],
            }
        )
    out_rows.append(
        {
            "method": str(args.candidate_method),
            "run": str(args.candidate_label),
            "images_dir": cand["images_dir"],
            "source_root": cand["source_root"],
            "metrics_csv": cand["metrics_csv"],
        }
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "run", "images_dir", "source_root", "metrics_csv"])
        writer.writeheader()
        writer.writerows(out_rows)
    print(args.output_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
