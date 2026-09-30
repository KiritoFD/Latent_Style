from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: str) -> str:
    try:
        return f"{float(value):.3f}"
    except Exception:
        return value


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge multiple VLM method summary CSVs into one external-baseline board.")
    parser.add_argument("--input", nargs="+", required=True, help="comparison_key=path/to/method_summary.csv")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    for item in args.input:
        if "=" not in item:
            raise ValueError(f"expected comparison_key=path, got: {item}")
        comparison, raw_path = item.split("=", 1)
        path = Path(raw_path)
        for row in _read_rows(path):
            merged = {"comparison": comparison}
            merged.update(row)
            rows.append(merged)

    fieldnames = [
        "comparison",
        "method",
        "cases_completed",
        "wins_so_far",
        "win_rate_so_far",
        "style_wins_so_far",
        "structure_wins_so_far",
        "artifact_wins_so_far",
        "mean_style_specificity",
        "mean_structure_preservation",
        "mean_artifact_control",
    ]
    _write_csv(args.output_csv, rows, fieldnames)

    lines = [
        "# VLM External Baseline Board",
        "",
        "| Comparison | Method | Cases | Wins | WinRate | StyleWins | StructWins | ArtifactWins | MeanStyle | MeanStruct | MeanArtifact |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["comparison"],
                    row["method"],
                    row["cases_completed"],
                    row["wins_so_far"],
                    _fmt(row["win_rate_so_far"]),
                    row["style_wins_so_far"],
                    row["structure_wins_so_far"],
                    row["artifact_wins_so_far"],
                    _fmt(row["mean_style_specificity"]),
                    _fmt(row["mean_structure_preservation"]),
                    _fmt(row["mean_artifact_control"]),
                ]
            )
            + " |"
        )

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output_csv)
    print(args.output_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
