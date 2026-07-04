"""Batch runner for B2 V1-V4 tuning experiments on Windows native Python.

Runs each experiment sequentially:
- V1: scale-up (dim=128, blocks=6, 8 epoch)
- V2: weight rebalance (w_ll=0.3, w_hh=1.5, 8 epoch)
- V3: Brownian bridge (sigma=0.1, 8 epoch)
- V4: long training (24 epoch, lr=1e-4)

Each experiment: train + auto full_eval on last checkpoint.
Skips experiments whose final checkpoint + summary.json already exist.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJ_ROOT = ROOT
CONFIGS = [
    ("V1_scale", "configs/620_spectral_v1_scale.json", "exp/620_spectral_v1_scale"),
    ("V2_weights", "configs/620_spectral_v2_weights.json", "exp/620_spectral_v2_weights"),
    ("V3_brownian", "configs/620_spectral_v3_brownian.json", "exp/620_spectral_v3_brownian"),
    ("V4_long", "configs/620_spectral_v4_long.json", "exp/620_spectral_v4_long"),
]


def _read_num_epochs(cfg_path: Path) -> int:
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return int(raw.get("training", {}).get("num_epochs", 8))
    except Exception:
        return 8


def _is_done(exp_dir: Path, num_epochs: int) -> bool:
    last_ckpt = exp_dir / f"epoch_{num_epochs:04d}.pt"
    summary = exp_dir / "full_eval" / f"epoch_{num_epochs:04d}" / "summary.json"
    return last_ckpt.is_file() and summary.is_file()


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
            print(f"[batch] SKIP {name}: already done (final ckpt + summary exist)", flush=True)
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
        # Verify
        if _is_done(exp_dir, num_epochs):
            print(f"[batch] {name} DONE: checkpoint + summary.json present", flush=True)
        else:
            print(f"[batch] {name} WARNING: final checkpoint or summary missing; check log", flush=True)
    total = time.perf_counter() - overall_start
    print(f"\n{'='*70}", flush=True)
    print(f"[batch] ALL EXPERIMENTS COMPLETE in {total:.1f}s ({total/60:.2f} min)", flush=True)


if __name__ == "__main__":
    main()
