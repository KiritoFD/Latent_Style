"""Batch runner for V9-V12 spectral ODE experiments targeting CLIP > 0.74.

V9:  w_ll=1.0, w_hh=1.5, 8ep       (full lowfreq unlock)
V10: w_ll=2.0, w_hh=1.5, 8ep       (emphasize lowfreq)
V11: w_ll=1.0, w_hh=2.0, 8ep       (lowfreq + strong highfreq)
V12: w_ll=1.0, w_hh=1.5, 24ep, lr=1e-4  (lowfreq + long training)

Reports best CLIP after each experiment. Goal: CLIP > 0.74.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJ_ROOT = ROOT
CONFIGS = [
    ("V9_ll10", "configs/620_spectral_v9_ll10.json", "exp/620_spectral_v9_ll10"),
    ("V10_ll20", "configs/620_spectral_v10_ll20.json", "exp/620_spectral_v10_ll20"),
    ("V11_ll10_hh20", "configs/620_spectral_v11_ll10_hh20.json", "exp/620_spectral_v11_ll10_hh20"),
    ("V12_ll10_long", "configs/620_spectral_v12_ll10_long.json", "exp/620_spectral_v12_ll10_long"),
]

CLIP_TARGET = 0.74


def _read_num_epochs(cfg_path: Path) -> int:
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return int(raw.get("training", {}).get("num_epochs", 8))
    except Exception:
        return 8


def _is_done(exp_dir: Path) -> bool:
    convergence = exp_dir / "full_eval" / "round2_convergence.json"
    return convergence.is_file()


def _report_best(name: str, exp_dir: Path) -> dict:
    """Read clip_lpips_curve.csv and report best CLIP. Returns best row dict."""
    curve = exp_dir / "full_eval" / "clip_lpips_curve.csv"
    if not curve.exists():
        print(f"[batch] {name}: no curve csv found", flush=True)
        return {}
    rows = []
    with open(curve, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    if not rows:
        print(f"[batch] {name}: empty curve csv", flush=True)
        return {}
    # Primary: best CLIP (higher = better style transfer)
    best_clip = max(rows, key=lambda r: float(r["transfer_clip_style"]))
    clip_val = float(best_clip["transfer_clip_style"])
    lpips_val = float(best_clip["transfer_content_lpips"])
    target_gap = clip_val - CLIP_TARGET
    status = "TARGET HIT" if clip_val >= CLIP_TARGET else f"gap={target_gap:+.4f}"
    print(f"[batch] --- {name} RESULTS (target CLIP>{CLIP_TARGET}) ---", flush=True)
    print(f"[batch]   best CLIP:  epoch={best_clip['epoch']} clip={clip_val:.4f} lpips={lpips_val:.4f}  [{status}]", flush=True)
    # Also report best LPIPS for reference
    best_lpips = min(rows, key=lambda r: float(r["transfer_content_lpips"]))
    print(f"[batch]   best LPIPS: epoch={best_lpips['epoch']} clip={float(best_lpips['transfer_clip_style']):.4f} lpips={float(best_lpips['transfer_content_lpips']):.4f}", flush=True)
    print(f"[batch]   total epochs evaluated: {len(rows)}", flush=True)
    return {"name": name, "best_clip_epoch": best_clip["epoch"], "clip": clip_val, "lpips": lpips_val, "target_gap": target_gap}


def main() -> None:
    overall_start = time.perf_counter()
    print(f"[batch] project root: {PROJ_ROOT}", flush=True)
    print(f"[batch] TARGET: CLIP > {CLIP_TARGET}", flush=True)
    all_results = []
    for name, cfg_rel, exp_rel in CONFIGS:
        cfg_path = PROJ_ROOT / cfg_rel
        exp_dir = PROJ_ROOT / exp_rel
        num_epochs = _read_num_epochs(cfg_path)
        print(f"\n{'='*70}", flush=True)
        print(f"[batch] === {name} === config={cfg_path.name} epochs={num_epochs}", flush=True)
        print(f"[batch] exp_dir={exp_dir}", flush=True)
        if _is_done(exp_dir):
            print(f"[batch] SKIP {name}: already done", flush=True)
            res = _report_best(name, exp_dir)
            if res:
                all_results.append(res)
            continue
        # Train + auto eval
        log_file = open(exp_dir.with_name(f"_b2_{name}.log"), "w", encoding="utf-8", errors="replace")
        try:
            print(f"[batch] launching training for {name}...", flush=True)
            start = time.perf_counter()
            proc = subprocess.run(
                ["python", "-u", "run.py", "--config", str(cfg_path)],
                cwd=str(PROJ_ROOT),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
            )
            wall = time.perf_counter() - start
            print(f"[batch] {name} training+eval exited code={proc.returncode} in {wall:.1f}s", flush=True)
        finally:
            log_file.close()
        if _is_done(exp_dir):
            print(f"[batch] {name} DONE", flush=True)
            res = _report_best(name, exp_dir)
            if res:
                all_results.append(res)
        else:
            print(f"[batch] {name} WARNING: round2_convergence.json missing", flush=True)
    total = time.perf_counter() - overall_start
    print(f"\n{'='*70}", flush=True)
    print(f"[batch] ALL EXPERIMENTS COMPLETE in {total:.1f}s ({total/60:.2f} min)", flush=True)
    # Final summary sorted by CLIP
    print(f"\n{'='*70}", flush=True)
    print(f"[batch] FINAL SUMMARY (sorted by CLIP, target={CLIP_TARGET}):", flush=True)
    all_results.sort(key=lambda r: r["clip"], reverse=True)
    for i, r in enumerate(all_results, 1):
        status = "TARGET HIT" if r["clip"] >= CLIP_TARGET else f"gap={r['target_gap']:+.4f}"
        print(f"[batch]   #{i} {r['name']}: epoch={r['best_clip_epoch']} CLIP={r['clip']:.4f} LPIPS={r['lpips']:.4f}  [{status}]", flush=True)


if __name__ == "__main__":
    main()
