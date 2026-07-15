#!/usr/bin/env python3
"""Collect WFI / CLIP-S / LPIPS results from massive ablation experiments."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


def extract_wfi_report(path: Path) -> dict:
    """Read wfi_eval_report.json or wfi_benchmark.json."""
    for name in ["wfi_eval_report.json", "wfi_benchmark.json"]:
        p = path / name
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                return json.load(f)
    return {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-root", default="./exp/620_massive_ablation")
    parser.add_argument("--matrix", default="matrix.csv")
    parser.add_argument("--out", default="massive_ablation_results.csv")
    args = parser.parse_args()

    exp_root = Path(args.exp_root)
    rows = []
    with open(args.matrix, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        matrix_rows = list(reader)

    for mr in matrix_rows:
        name = mr["name"]
        group = mr["group"]
        overrides = mr["overrides"]
        exp_dir = exp_root / name
        best_epoch = None
        best = {}
        # Search all epoch eval dirs
        for epoch_dir in sorted(exp_dir.glob("full_eval_wfi/epoch_*")):
            epoch = int(epoch_dir.name.split("_")[-1])
            report = extract_wfi_report(epoch_dir)
            if not report:
                continue
            metrics = report.get("metrics", report)
            wfi = metrics.get("wfi_score")
            clip = metrics.get("clip_style")
            lpips = metrics.get("content_lpips")
            if wfi is None:
                continue
            if best_epoch is None or wfi < best.get("wfi_score", 999.0):
                best_epoch = epoch
                best = {
                    "epoch": epoch,
                    "wfi_score": wfi,
                    "clip_style": clip,
                    "content_lpips": lpips,
                    "delta_wfi": metrics.get("delta_wfi"),
                }
        rows.append({
            "name": name,
            "group": group,
            "overrides": overrides,
            **best,
        })

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "name", "group", "overrides", "epoch", "wfi_score",
            "clip_style", "content_lpips", "delta_wfi",
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Collected {len(rows)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
