#!/usr/bin/env python3
"""Aggregate ablation evaluation results from exp_ablation_620/*/full_eval/epoch_0003/summary.json.

Outputs:
- ablation_results.csv : one row per experiment with key metrics
- ablation_results.md  : markdown table for documentation
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

EXP_DIR = Path("/mnt/i/Github/Latent_Style/SchrodingerBridge/exp_ablation_620")
OUT_CSV = Path("/mnt/i/Github/Latent_Style/SchrodingerBridge/docs/ablation_results.csv")
OUT_MD = Path("/mnt/i/Github/Latent_Style/SchrodingerBridge/docs/ablation_results.md")


def _get(d: dict, *path, default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def extract_metrics(summary: dict) -> dict:
    analysis = summary.get("analysis") or {}
    transfer = analysis.get("style_transfer_ability") or {}
    allpairs = analysis.get("all_pairs_overview") or {}
    idt = analysis.get("identity_reconstruction") or {}
    timings = summary.get("timings_sec") or {}
    return {
        "transfer_clip_style": float(transfer.get("clip_style", 0.0) or 0.0),
        "transfer_clip_t": float(transfer.get("clip_t", 0.0) or 0.0),
        "transfer_content_lpips": float(transfer.get("content_lpips", 0.0) or 0.0),
        "allpairs_clip_style": float(allpairs.get("clip_style", 0.0) or 0.0),
        "allpairs_clip_t": float(allpairs.get("clip_t", 0.0) or 0.0),
        "allpairs_content_lpips": float(allpairs.get("content_lpips", 0.0) or 0.0),
        "identity_clip_style": float(idt.get("clip_style", 0.0) or 0.0),
        "identity_content_lpips": float(idt.get("content_lpips", 0.0) or 0.0),
        "wall_total_sec": float(timings.get("wall_total", 0.0) or 0.0),
    }


def categorize(name: str) -> str:
    if name.startswith("DA"):
        return "Architecture"
    if name.startswith("DD"):
        return "Data"
    if name.startswith("DI"):
        return "Infrastructure"
    if name.startswith("DL"):
        return "Loss"
    if name.startswith("DN"):
        return "Inference"
    if name == "infra_I0_baseline":
        return "Baseline"
    return "Other"


def main() -> int:
    rows = []
    for exp_dir in sorted(EXP_DIR.iterdir()):
        if not exp_dir.is_dir():
            continue
        name = exp_dir.name
        summary_path = exp_dir / "full_eval" / "epoch_0003" / "summary.json"
        if not summary_path.is_file():
            print(f"[SKIP] {name}: no summary.json", file=sys.stderr)
            continue
        try:
            with summary_path.open("r", encoding="utf-8") as f:
                summary = json.load(f)
        except Exception as exc:
            print(f"[ERROR] {name}: {exc}", file=sys.stderr)
            continue
        metrics = extract_metrics(summary)
        rows.append({"name": name, "category": categorize(name), **metrics})
        print(f"[OK] {name}: tCLIP-S={metrics['transfer_clip_style']:.4f} "
              f"tLPIPS={metrics['transfer_content_lpips']:.4f} "
              f"apCLIP-S={metrics['allpairs_clip_style']:.4f}")

    rows.sort(key=lambda r: (r["category"], r["name"]))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "name", "category",
        "transfer_clip_style", "transfer_clip_t", "transfer_content_lpips",
        "allpairs_clip_style", "allpairs_clip_t", "allpairs_content_lpips",
        "identity_clip_style", "identity_content_lpips",
        "wall_total_sec",
    ]
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote CSV: {OUT_CSV}")

    # Markdown table grouped by category
    lines = ["# Ablation Results (epoch_0003)", ""]
    lines.append("Evaluated on `/mnt/i/wikiart_distinct5_samam_512_classview/test` "
                 "(5 styles × 30 images = 150 per direction, 25 transfer pairs).")
    lines.append("")
    cur_cat = None
    for row in rows:
        if row["category"] != cur_cat:
            cur_cat = row["category"]
            lines.append(f"\n## {cur_cat}\n")
            lines.append("| Experiment | tCLIP-S↑ | tCLIP-T↑ | tLPIPS↓ | apCLIP-S↑ | apLPIPS↓ | idtCLIP-S↑ | idtLPIPS↓ |")
            lines.append("|---|---|---|---|---|---|---|---|")
        lines.append(
            f"| {row['name']} | {row['transfer_clip_style']:.4f} | {row['transfer_clip_t']:.4f} | "
            f"{row['transfer_content_lpips']:.4f} | {row['allpairs_clip_style']:.4f} | "
            f"{row['allpairs_content_lpips']:.4f} | {row['identity_clip_style']:.4f} | "
            f"{row['identity_content_lpips']:.4f} |"
        )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote MD: {OUT_MD}")
    print(f"\nTotal experiments with results: {len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
