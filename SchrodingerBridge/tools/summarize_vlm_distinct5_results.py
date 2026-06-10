from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _sorted_methods(methods: set[str]) -> list[str]:
    return sorted(methods)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize Distinct5 VLM jsonl results into compact CSVs.")
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-method-summary", type=Path, required=True)
    parser.add_argument("--output-interim-summary", type=Path, required=True)
    args = parser.parse_args()

    rows = _read_jsonl(args.input_jsonl)
    methods: set[str] = set()
    for row in rows:
        parsed = row.get("parsed") or {}
        scores = parsed.get("scores") or {}
        methods.update(str(k) for k in scores.keys())

    run_order = _sorted_methods(methods)
    cases_completed = len(rows)

    win_counts = defaultdict(int)
    style_win_counts = defaultdict(int)
    structure_win_counts = defaultdict(int)
    artifact_win_counts = defaultdict(int)
    style_sum = defaultdict(float)
    structure_sum = defaultdict(float)
    artifact_sum = defaultdict(float)

    for row in rows:
        parsed = row.get("parsed") or {}
        scores = parsed.get("scores") or {}
        best_overall = str(parsed.get("best_overall") or "").strip()
        best_style = str(parsed.get("best_style_specificity") or "").strip()
        best_structure = str(parsed.get("best_structure") or "").strip()
        best_artifact = str(parsed.get("best_artifact_control") or "").strip()
        if best_overall:
            win_counts[best_overall] += 1
        if best_style:
            style_win_counts[best_style] += 1
        if best_structure:
            structure_win_counts[best_structure] += 1
        if best_artifact:
            artifact_win_counts[best_artifact] += 1

        for method in run_order:
            block = scores.get(method) or {}
            style_sum[method] += float(block.get("style_specificity") or 0.0)
            structure_sum[method] += float(block.get("structure_preservation") or 0.0)
            artifact_sum[method] += float(block.get("artifact_control") or 0.0)

    interim_rows = [
        {
            "method": method,
            "wins_so_far": win_counts[method],
            "cases_completed": cases_completed,
            "style_wins_so_far": style_win_counts[method],
            "structure_wins_so_far": structure_win_counts[method],
            "artifact_wins_so_far": artifact_win_counts[method],
        }
        for method in run_order
    ]
    _write_csv(
        args.output_interim_summary,
        interim_rows,
        [
            "method",
            "wins_so_far",
            "cases_completed",
            "style_wins_so_far",
            "structure_wins_so_far",
            "artifact_wins_so_far",
        ],
    )

    method_rows = []
    for method in run_order:
        denom = max(1, cases_completed)
        method_rows.append(
            {
                "method": method,
                "cases_completed": cases_completed,
                "wins_so_far": win_counts[method],
                "win_rate_so_far": win_counts[method] / denom,
                "style_wins_so_far": style_win_counts[method],
                "structure_wins_so_far": structure_win_counts[method],
                "artifact_wins_so_far": artifact_win_counts[method],
                "mean_style_specificity": style_sum[method] / denom,
                "mean_structure_preservation": structure_sum[method] / denom,
                "mean_artifact_control": artifact_sum[method] / denom,
            }
        )
    _write_csv(
        args.output_method_summary,
        method_rows,
        [
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
        ],
    )

    print(args.output_interim_summary)
    print(args.output_method_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
