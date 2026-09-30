from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge local IntroStyle probe rows with DINO manifest rows.")
    parser.add_argument("--introstyle-csv", type=Path, required=True)
    parser.add_argument("--dino-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    intro_rows = _read_rows(args.introstyle_csv)
    dino_rows = _read_rows(args.dino_csv)
    intro_by_run = {str(row.get("run", "")).strip(): row for row in intro_rows}

    merged: list[dict[str, str]] = []
    for dino in dino_rows:
        run = str(dino.get("run", "")).strip()
        intro = intro_by_run.get(run, {})
        merged.append(
            {
                "label": str(dino.get("label", "")),
                "run": run,
                "n_pairs": str(dino.get("n_pairs", "")),
                "transfer_target_style_score": str(intro.get("transfer_target_style_score", "")),
                "transfer_source_style_score": str(intro.get("transfer_source_style_score", "")),
                "transfer_best_non_target_score": str(intro.get("transfer_best_non_target_score", "")),
                "transfer_style_margin": str(intro.get("transfer_style_margin", "")),
                "identity_target_style_score": str(intro.get("identity_target_style_score", "")),
                "dino_structure": str(dino.get("dino_structure", "")),
                "images_dir": str(dino.get("images_dir", "")),
                "metrics_csv": str(dino.get("metrics_csv", "")),
            }
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "label",
        "run",
        "n_pairs",
        "transfer_target_style_score",
        "transfer_source_style_score",
        "transfer_best_non_target_score",
        "transfer_style_margin",
        "identity_target_style_score",
        "dino_structure",
        "images_dir",
        "metrics_csv",
    ]
    with args.output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged)
    print(args.output_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
