"""Batch runner for V5-V8 spectral ODE tuning experiments on Windows native Python.

Runs each experiment sequentially: train + auto full_eval (run.py auto-triggers).
Skips experiments already done. Reports best result after each experiment.

V5: w_ll=0.1 (conservative lowfreq relax)
V6: w_ll=0.5 (aggressive lowfreq relax)
V7: V2+V3 combo (w_ll=0.3, w_hh=1.5 + Brownian sigma=0.1)
V8: V2+V4 combo (w_ll=0.3, w_hh=1.5 + 24ep + lr=1e-4)
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
    ("V5_ll01", "configs/620_spectral_v5_ll01.json", "exp/620_spectral_v5_ll01"),
    ("V6_ll05", "configs/620_spectral_v6_ll05.json", "exp/620_spectral_v6_ll05"),
    ("V7_combo_brownian", "configs/620_spectral_v7_combo_brownian.json", "exp/620_spectral_v7_combo_brownian"),
    ("V8_combo_long", "configs/620_spectral_v8_combo_long.json", "exp/620_spectral_v8_combo_long"),
]


def _read_num_epochs(cfg_path: Path) -> int:
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return int(raw.get("training", {}).get("num_epochs", 8))
    except Exception:
        return 8


def _is_done(exp_dir: Path, num_epochs: int) -> bool:
    """Done if round2_convergence.json exists (eval completed)."""
    convergence = exp_dir / "full_eval" / "round2_convergence.json"
    return convergence.is_file()


def _report_best(name: str, exp_dir: Path) -> None:
    """Read clip_lpips_curve.csv and report best epoch."""
    curve = exp_dir / "full_eval" / "clip_lpips_curve.csv"
    if not curve.exists():
        print(f"[batch] {name}: no curve csv found", flush=True)
        return
    rows = []
    with open(curve, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    if not rows:
        print(f"[batch] {name}: empty curve csv", flush=True)
        return
    # Find best by transfer_content_lpips (lower = better content preservation)
    best_lpips = min(rows, key=lambda r: float(r["transfer_content_lpips"]))
    # Find best by transfer_clip_style (higher = better style transfer)
    best_clip = max(rows, key=lambda r: float(r["transfer_clip_style"]))
    print(f"[batch] --- {name} RESULTS ---", flush=True)
    print(f"[batch]   best LPIPS: epoch={best_lpips['epoch']} clip={float(best_lpips['transfer_clip_style']):.4f} lpips={float(best_lpips['transfer_content_lpips']):.4f}", flush=True)
    print(f"[batch]   best CLIP:  epoch={best_clip['epoch']} clip={float(best_clip['transfer_clip_style']):.4f} lpips={float(best_clip['transfer_content_lpips']):.4f}", flush=True)
    print(f"[batch]   total epochs evaluated: {len(rows)}", flush=True)


def main() -> None:
    overall_start = time.perf_counter()
    print(f"[batch] project root: {PROJ_ROOT}", flush=True)
    for name, cfg_rel, exp_rel in CONFIGS:
        cfg_path = PROJ_ROOT / cfg_rel
        exp_dir = PROJ_ROOT / exp_rel
        num_epochs = _read_num_epochs(cfg_path)
        print(f"\n{'='*70}", flush=True)
        print(f"[batch] === {name} === config={cfg_path.name} epochs={num_epochs}", flush=True)
        print(f"[batch] exp_dir={exp_dir}", flush=True)
        if _is_done(exp_dir, num_epochs):
            print(f"[batch] SKIP {name}: already done (round2_convergence.json exists)", flush=True)
            _report_best(name, exp_dir)
            continue
        # Train + auto eval (run.py auto-triggers full_eval_defer_until_training_end)
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
        # Verify and report
        if _is_done(exp_dir, num_epochs):
            print(f"[batch] {name} DONE: round2_convergence.json present", flush=True)
            _report_best(name, exp_dir)
        else:
            print(f"[batch] {name} WARNING: round2_convergence.json missing; check log", flush=True)
    total = time.perf_counter() - overall_start
    print(f"\n{'='*70}", flush=True)
    print(f"[batch] ALL EXPERIMENTS COMPLETE in {total:.1f}s ({total/60:.2f} min)", flush=True)
    # Final summary
    print(f"\n{'='*70}", flush=True)
    print(f"[batch] FINAL SUMMARY (V5-V8):", flush=True)
    for name, _, exp_rel in CONFIGS:
        _report_best(name, PROJ_ROOT / exp_rel)


if __name__ == "__main__":
    main()
