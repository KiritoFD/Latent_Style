"""Batch evaluation for 628 X experiments.
Reads each X config, extracts eval params, calls run_evaluation.py.
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

def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_DIR / "eval_batch_log.txt", "a", encoding="utf-8") as f:
        f.write(line + "\n")

def main():
    x_configs = sorted(CONFIG_DIR.glob("X*.json"), key=lambda p: p.stem)
    total = len(x_configs)
    log(f"Starting eval batch: {total} X experiments")

    success = 0
    fail = 0
    skipped = 0

    for i, cfg_path in enumerate(x_configs, 1):
        name = cfg_path.stem
        ckpt_path = EXP_DIR / name / "epoch_0010.pt"
        if not ckpt_path.exists():
            log(f"[{i}/{total}] SKIP {name} (no checkpoint)")
            skipped += 1
            continue

        # Check if eval already done
        eval_out = ckpt_path.parent / "full_eval" / ckpt_path.stem
        convergence_json = eval_out / "round2_convergence.json"
        if convergence_json.exists():
            log(f"[{i}/{total}] SKIP {name} (eval done)")
            skipped += 1
            continue

        # Read config for eval params
        with open(cfg_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        train_cfg = config.get("training", {})

        test_dir = train_cfg.get("test_image_dir", "I:/wikiart_distinct5_samam_512_classview/test")
        cache_dir = train_cfg.get("full_eval_cache_dir", "I:/eval_cache")
        clip_hf_cache_dir = train_cfg.get("full_eval_clip_hf_cache_dir", "")
        batch_size = int(train_cfg.get("full_eval_batch_size", 24))
        num_steps = train_cfg.get("full_eval_num_steps", 12)
        step_size = train_cfg.get("full_eval_step_size", 1.0)
        style_strength = train_cfg.get("full_eval_style_strength", None)
        max_src = train_cfg.get("full_eval_max_src_samples", 30)

        cmd = [
            PYTHON, str(EVAL_SCRIPT),
            "--checkpoint", str(ckpt_path),
            "--output", str(eval_out),
            "--test_dir", str(test_dir),
            "--cache_dir", str(cache_dir),
            "--batch_size", str(batch_size),
            "--num_steps", str(int(num_steps)) if num_steps is not None else "12",
            "--step_size", str(float(step_size)) if step_size is not None else "1.0",
            "--max_src_samples", str(int(max_src)) if max_src is not None else "30",
            "--eval_only_lpips_clip_style",
        ]
        if clip_hf_cache_dir:
            cmd += ["--clip_hf_cache_dir", str(clip_hf_cache_dir)]
        if style_strength is not None:
            cmd += ["--style_strength", str(float(style_strength))]

        log(f"[{i}/{total}] EVAL {name}")
        log_path = LOG_DIR / f"{name}_eval.log"
        t0 = time.time()
        try:
            with open(log_path, "w", encoding="utf-8") as logf:
                result = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, timeout=600)
            elapsed = time.time() - t0
            if result.returncode == 0:
                success += 1
                log(f"[{i}/{total}] DONE {name} SUCCESS ({elapsed:.0f}s)")
            else:
                fail += 1
                log(f"[{i}/{total}] DONE {name} FAILED rc={result.returncode} ({elapsed:.0f}s)")
        except subprocess.TimeoutExpired:
            fail += 1
            log(f"[{i}/{total}] DONE {name} TIMEOUT (600s)")
        except Exception as e:
            fail += 1
            log(f"[{i}/{total}] DONE {name} ERROR: {e}")

    log(f"Eval batch complete: {success} succeeded, {fail} failed, {skipped} skipped out of {total}")

if __name__ == "__main__":
    main()
