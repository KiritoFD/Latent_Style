"""Batch evaluation for ALL 628 experiments (D/L/E/P/X series).
Skips experiments that already have summary.json.
Reads each config, extracts eval params, calls run_evaluation.py.
"""
import os
import sys
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(r"I:\Github\Latent_Style\SchrodingerBridge")
CONFIG_DIR = ROOT / "configs" / "ablations" / "628_destructive"
EXP_DIR = ROOT / "exp" / "628_ablation" / "destructive"
EVAL_SCRIPT = ROOT / "src" / "utils" / "run_evaluation.py"
PYTHON = r"C:\Progra~1\Python312\python.exe"
LOG_DIR = ROOT / "exp" / "628_ablation" / "destructive_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def get_eval_params(config_path: Path) -> dict:
    """Extract eval params from config."""
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    train = cfg.get("training", {})
    return {
        "test_dir": train.get("test_image_dir", r"I:\wikiart_distinct5_samam_512_classview\test"),
        "cache_dir": train.get("full_eval_cache_dir", r"I:\_cache\lpips_clip"),
        "batch_size": str(int(train.get("full_eval_batch_size", 4))),
    }


def already_evaluated(exp_name: str) -> bool:
    """Check if summary.json already exists for this experiment."""
    summary_path = EXP_DIR / exp_name / "full_eval" / "epoch_0010" / "summary.json"
    return summary_path.is_file()


def has_checkpoint(exp_name: str) -> bool:
    """Check if epoch_0010.pt exists."""
    return (EXP_DIR / exp_name / "epoch_0010.pt").is_file()


def main():
    configs = sorted(CONFIG_DIR.glob("*.json"), key=lambda p: p.stem)
    total = len(configs)

    pending = []
    skipped_done = []
    skipped_no_ckpt = []
    for cfg in configs:
        name = cfg.stem
        if not has_checkpoint(name):
            skipped_no_ckpt.append(name)
            continue
        if already_evaluated(name):
            skipped_done.append(name)
            continue
        pending.append(cfg)

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Eval batch: {total} total, "
          f"{len(pending)} pending, {len(skipped_done)} already evaluated, "
          f"{len(skipped_no_ckpt)} no checkpoint", flush=True)

    if not pending:
        print("All experiments already evaluated. Nothing to do.")
        return

    success = 0
    fail = 0
    for i, cfg in enumerate(pending, 1):
        name = cfg.stem
        ckpt_path = EXP_DIR / name / "epoch_0010.pt"
        eval_out = EXP_DIR / name / "full_eval" / "epoch_0010"
        params = get_eval_params(cfg)

        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [{i}/{len(pending)}] EVAL {name}", flush=True)
        t0 = time.time()
        try:
            cmd = [
                PYTHON, str(EVAL_SCRIPT),
                "--checkpoint", str(ckpt_path),
                "--output", str(eval_out),
                "--test_dir", params["test_dir"],
                "--cache_dir", params["cache_dir"],
                "--batch_size", params["batch_size"],
                "--eval_only_lpips_clip_style",
            ]
            result = subprocess.run(cmd, timeout=600)
            elapsed = time.time() - t0
            if result.returncode == 0:
                success += 1
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [{i}/{len(pending)}] DONE {name} SUCCESS ({elapsed:.0f}s)", flush=True)
            else:
                fail += 1
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [{i}/{len(pending)}] DONE {name} FAILED rc={result.returncode} ({elapsed:.0f}s)", flush=True)
        except subprocess.TimeoutExpired:
            fail += 1
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [{i}/{len(pending)}] DONE {name} TIMEOUT (600s)", flush=True)
        except Exception as e:
            fail += 1
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [{i}/{len(pending)}] DONE {name} ERROR: {e}", flush=True)

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Eval batch complete: {success} succeeded, {fail} failed out of {len(pending)}", flush=True)


if __name__ == "__main__":
    main()
