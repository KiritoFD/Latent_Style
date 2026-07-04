"""Batch runner for destructive ablation experiments.
Runs each config sequentially, logs progress.
"""
import os
import sys
import subprocess
import time
import json
from pathlib import Path

ROOT = Path(r"I:\Github\Latent_Style\SchrodingerBridge")
CONFIG_DIR = ROOT / "configs" / "ablations" / "628_destructive"
LOG_DIR = ROOT / "exp" / "628_ablation" / "destructive_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

PYTHON = r"C:\Progra~1\Python312\python.exe"
RUN_SCRIPT = str(ROOT / "src" / "run.py")

batch_log_path = LOG_DIR / "batch_log.txt"

def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(batch_log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def _sort_key(p):
    """Sort X configs first (extreme-weight = priority), then alphabetical."""
    name = p.stem
    if name.startswith("X"):
        return (0, name)  # X configs first
    return (1, name)  # others after


def main():
    configs = sorted(CONFIG_DIR.glob("*.json"), key=_sort_key)
    total = len(configs)
    log(f"Starting batch: {total} experiments (X configs prioritized)")

    success = 0
    fail = 0
    skipped = 0

    for i, cfg_path in enumerate(configs, 1):
        name = cfg_path.stem

        exp_dir = ROOT / "exp" / "628_ablation" / "destructive" / name
        done_marker = exp_dir / "epoch_0010.pt"
        if done_marker.exists():
            log(f"[{i}/{total}] SKIP {name} (already done)")
            skipped += 1
            continue

        log(f"[{i}/{total}] START {name}")
        log_path = LOG_DIR / f"{name}.log"

        t0 = time.time()
        try:
            result = subprocess.run(
                [PYTHON, RUN_SCRIPT, "--config", str(cfg_path)],
                capture_output=False,
                stdout=open(log_path, "w", encoding="utf-8"),
                stderr=subprocess.STDOUT,
                timeout=1800,
            )
            elapsed = time.time() - t0
            if result.returncode == 0:
                success += 1
                log(f"[{i}/{total}] DONE {name} SUCCESS ({elapsed:.0f}s)")
            else:
                fail += 1
                log(f"[{i}/{total}] DONE {name} FAILED rc={result.returncode} ({elapsed:.0f}s)")
        except subprocess.TimeoutExpired:
            fail += 1
            log(f"[{i}/{total}] DONE {name} TIMEOUT (1800s)")
        except Exception as e:
            fail += 1
            log(f"[{i}/{total}] DONE {name} ERROR: {e}")

    log(f"Batch complete: {success} succeeded, {fail} failed, {skipped} skipped out of {total}")


if __name__ == "__main__":
    main()
