from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge best-few handoff CSV with IntroStyle probe CSV.")
    parser.add_argument("--handoff-csv", type=Path, required=True)
    parser.add_argument("--introstyle-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    handoff_rows = _read_rows(args.handoff_csv)
    intro_rows = _read_rows(args.introstyle_csv)
    intro_by_run = {str(r.get("run", "")).strip(): r for r in intro_rows}

    out_rows: list[dict[str, str]] = []
    for row in handoff_rows:
        epoch = str(row.get("epoch", "")).strip()
        intro = intro_by_run.get(epoch, {})
        out_rows.append(
            {
                **row,
                "introstyle_target_style_score": str(intro.get("transfer_target_style_score", "")),
                "introstyle_source_style_score": str(intro.get("transfer_source_style_score", "")),
                "introstyle_best_non_target_score": str(intro.get("transfer_best_non_target_score", "")),
                "introstyle_style_margin": str(intro.get("transfer_style_margin", "")),
                "introstyle_identity_target_score": str(intro.get("identity_target_style_score", "")),
            }
        )

    fieldnames = list(out_rows[0].keys()) if out_rows else []
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    print(args.output_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
