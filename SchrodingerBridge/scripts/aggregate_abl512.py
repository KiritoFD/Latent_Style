"""Aggregate 48 ablation results into a single CSV for scatter plot generation.

Reads each experiment's summary.json under exp/abl512/X*/full_eval/epoch_*/summary.json
and outputs a CSV with: name, axis, transfer_clip_style, transfer_content_lpips,
allpairs_clip_style, allpairs_content_lpips, identity_clip_style, identity_content_lpips

Usage:
    python aggregate_abl512.py --exp_root exp/abl512 --output docs/experiments/abl512_v3_results.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


# 48 experiments with theoretical axis mapping
EXPERIMENTS = [
    ("X01_euler", "solver", "Euler (1st-order)"),
    ("X02_rk4", "solver", "RK4 (4th-order)"),
    ("X03_steps_1", "solver", "1 step"),
    ("X04_steps_32", "solver", "32 steps"),
    ("X05_corrector_4", "solver", "Corrector x4"),
    ("X06_no_spectral_ode", "spectral", "No spectral ODE"),
    ("X07_spectral_levels_4", "spectral", "4 DWT levels"),
    ("X08_spectral_levels_5", "spectral", "5 DWT levels"),
    ("X09_lowpass_avg", "spectral", "Lowpass=avg"),
    ("X10_w_ll_0", "spectral", "w_ll=0"),
    ("X11_w_hh_3x", "spectral", "w_hh=3x"),
    ("X12_adain_0", "adain", "AdaIN=0"),
    ("X13_adain_4x", "adain", "AdaIN=4x"),
    ("X14_adain_every_step", "adain", "AdaIN every step"),
    ("X15_lowpass_1", "adain", "Lowpass=1"),
    ("X16_lowpass_5", "adain", "Lowpass=5"),
    ("X17_velocity_floor_0", "bridge", "v_floor=0"),
    ("X18_velocity_floor_0p3", "bridge", "v_floor=0.3"),
    ("X19_path_linear", "coupling", "Linear path"),
    ("X20_path_slerp", "coupling", "SLERP path"),
    ("X21_sigma_0", "coupling", "sigma=0"),
    ("X22_sigma_0p5", "coupling", "sigma=0.5"),
    ("X23_no_target_proj", "coupling", "No target proj"),
    ("X24_hungarian", "coupling", "Hungarian"),
    ("X25_no_structure_cost", "loss", "No structure cost"),
    ("X26_structure_5x", "loss", "Structure 5x"),
    ("X27_sinkhorn_eps_0p5", "loss", "Sinkhorn eps=0.5"),
    ("X28_sinkhorn_iters_10", "loss", "Sinkhorn iters=10"),
    ("X29_no_content_loss", "loss", "No content loss"),
    ("X30_content_5x", "loss", "Content 5x"),
    ("X31_no_style_loss", "loss", "No style loss"),
    ("X32_style_32x", "loss", "Style 32x"),
    ("X33_style_64x", "loss", "Style 64x"),
    ("X34_no_flow", "loss", "No flow loss"),
    ("X35_no_kinetic", "loss", "No kinetic"),
    ("X36_attn_softmax", "arch", "Attn softmax"),
    ("X37_heads_1", "arch", "Heads=1"),
    ("X38_heads_16", "arch", "Heads=16"),
    ("X39_no_shortcut", "arch", "No shortcut"),
    ("X40_extrap_1", "arch", "Extrap=1"),
    ("X41_dim_32", "arch", "Dim=32"),
    ("X42_dim_128", "arch", "Dim=128"),
    ("X43_res_blocks_2", "arch", "Res blocks=2"),
    ("X44_no_skip", "arch", "No skip"),
    ("X45_epochs_1", "training", "Epochs=1"),
    ("X46_lr_10x", "training", "LR 10x"),
    ("X47_lr_0p1x", "training", "LR 0.1x"),
    ("X48_t_uniform", "training", "t uniform"),
]


def load_summary(exp_dir: Path) -> dict | None:
    """Load summary.json from full_eval/epoch_*/summary.json (try 0005 first, then 0001)."""
    for epoch_name in ["epoch_0005", "epoch_0001"]:
        path = exp_dir / "full_eval" / epoch_name / "summary.json"
        if path.is_file():
            try:
                with path.open("r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"WARN: failed to parse {path}: {e}")
    return None


def extract_metrics(summary: dict) -> dict:
    """Extract key metrics from summary.json."""
    analysis = summary.get("analysis", {}) or {}
    transfer = analysis.get("style_transfer_ability", {}) or {}
    allpairs = analysis.get("all_pairs_overview", {}) or {}
    idt = analysis.get("identity_reconstruction", {}) or {}
    return {
        "transfer_clip_style": float(transfer.get("clip_style", 0.0) or 0.0),
        "transfer_content_lpips": float(transfer.get("content_lpips", 0.0) or 0.0),
        "allpairs_clip_style": float(allpairs.get("clip_style", 0.0) or 0.0),
        "allpairs_content_lpips": float(allpairs.get("content_lpips", 0.0) or 0.0),
        "identity_clip_style": float(idt.get("clip_style", 0.0) or 0.0),
        "identity_content_lpips": float(idt.get("content_lpips", 0.0) or 0.0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_root", default="exp/abl512", help="Root dir of ablation experiments")
    parser.add_argument("--output", default="docs/experiments/abl512_v3_results.csv",
                        help="Output CSV path")
    parser.add_argument("--include_failed", action="store_true",
                        help="Include experiments without summary.json (with empty metrics)")
    args = parser.parse_args()

    exp_root = Path(args.exp_root)
    if not exp_root.is_absolute():
        # Resolve relative to repo root (assume script is in scripts/)
        repo_root = Path(__file__).resolve().parent.parent
        exp_root = repo_root / exp_root

    output_path = Path(args.output)
    if not output_path.is_absolute():
        repo_root = Path(__file__).resolve().parent.parent
        output_path = repo_root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    found = 0
    missing = 0
    for exp_name, axis, label in EXPERIMENTS:
        exp_dir = exp_root / exp_name
        if not exp_dir.is_dir():
            if args.include_failed:
                rows.append({
                    "name": exp_name, "axis": axis, "label": label,
                    "transfer_clip_style": "", "transfer_content_lpips": "",
                    "allpairs_clip_style": "", "allpairs_content_lpips": "",
                    "identity_clip_style": "", "identity_content_lpips": "",
                    "status": "MISSING",
                })
                missing += 1
            continue
        summary = load_summary(exp_dir)
        if summary is None:
            if args.include_failed:
                rows.append({
                    "name": exp_name, "axis": axis, "label": label,
                    "transfer_clip_style": "", "transfer_content_lpips": "",
                    "allpairs_clip_style": "", "allpairs_content_lpips": "",
                    "identity_clip_style": "", "identity_content_lpips": "",
                    "status": "EVAL_FAIL",
                })
                missing += 1
            continue
        metrics = extract_metrics(summary)
        metrics["name"] = exp_name
        metrics["axis"] = axis
        metrics["label"] = label
        metrics["status"] = "OK"
        rows.append(metrics)
        found += 1

    fieldnames = [
        "name", "axis", "label", "status",
        "transfer_clip_style", "transfer_content_lpips",
        "allpairs_clip_style", "allpairs_content_lpips",
        "identity_clip_style", "identity_content_lpips",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Aggregated {found} OK + {missing} missing/failed = {found + missing} total")
    print(f"Output: {output_path}")

    # Print summary table
    print("\n=== Summary (sorted by axis) ===")
    print(f"{'name':<28} {'axis':<10} {'CLIP-S':<8} {'LPIPS':<8} {'1-LPIPS':<8}")
    for row in sorted(rows, key=lambda r: (r["axis"], r["name"])):
        if row["status"] == "OK":
            cs = row["transfer_clip_style"]
            lp = row["transfer_content_lpips"]
            print(f"{row['name']:<28} {row['axis']:<10} {cs:<8.4f} {lp:<8.4f} {1 - lp:<8.4f}")
        else:
            print(f"{row['name']:<28} {row['axis']:<10} {'-':<8} {'-':<8} {'-':<8} [{row['status']}]")


if __name__ == "__main__":
    main()
