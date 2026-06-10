from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _f(row: dict[str, str], key: str) -> float:
    if key in row and str(row[key]).strip() != "":
        return float(row[key])
    aliases = {
        "transfer_lpips": ["transfer_content_lpips"],
        "allpairs_lpips": ["allpairs_content_lpips", "full_content_lpips"],
        "allpairs_clip_style": ["full_clip_style"],
        "wall_total": ["wall_total_seconds"],
    }
    for alt in aliases.get(key, []):
        if alt in row and str(row[alt]).strip() != "":
            return float(row[alt])
    raise KeyError(key)


def _pick_unique(rows: list[dict[str, str]]) -> list[tuple[str, dict[str, str]]]:
    if not rows:
        return []
    latest = sorted(rows, key=lambda r: r["epoch"])[-1]
    best_lpips = min(rows, key=lambda r: _f(r, "transfer_lpips"))
    best_clip = max(rows, key=lambda r: _f(r, "transfer_clip_style"))
    best_allpairs_style = max(rows, key=lambda r: _f(r, "allpairs_clip_style"))
    best_allpairs_lpips = min(rows, key=lambda r: _f(r, "allpairs_lpips"))
    tagged = [
        ("best_transfer_clip_style", best_clip),
        ("best_transfer_lpips", best_lpips),
        ("best_allpairs_clip_style", best_allpairs_style),
        ("best_structure_preserving", best_allpairs_lpips),
        ("latest", latest),
    ]
    merged: dict[str, tuple[list[str], dict[str, str]]] = {}
    for reason, row in tagged:
        epoch = row["epoch"]
        if epoch in merged:
            merged[epoch][0].append(reason)
            continue
        merged[epoch] = ([reason], row)
    out: list[tuple[str, dict[str, str]]] = []
    for reasons, row in merged.values():
        out.append((" | ".join(reasons), row))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a small remote-to-local handoff CSV of best candidate epochs.")
    parser.add_argument("--curve-csv", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    rows = _read_rows(args.curve_csv)
    picks = _pick_unique(rows)
    out_rows: list[dict[str, str]] = []
    for reason, row in picks:
        epoch = row["epoch"]
        out_rows.append(
            {
                "run_name": str(args.run_name),
                "reason": reason,
                "epoch": epoch,
                "transfer_clip_style": row["transfer_clip_style"],
                "transfer_lpips": str(_f(row, "transfer_lpips")),
                "allpairs_clip_style": str(_f(row, "allpairs_clip_style")),
                "allpairs_lpips": str(_f(row, "allpairs_lpips")),
                "wall_total": str(_f(row, "wall_total")),
                "summary_json": str(args.eval_root / epoch / "summary.json"),
                "metrics_csv": str(args.eval_root / epoch / "metrics.csv"),
                "images_dir": str(args.eval_root / epoch / "images"),
            }
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_name",
        "reason",
        "epoch",
        "transfer_clip_style",
        "transfer_lpips",
        "allpairs_clip_style",
        "allpairs_lpips",
        "wall_total",
        "summary_json",
        "metrics_csv",
        "images_dir",
    ]
    with args.output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    print(args.output_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
