"""Phase 8B: Fine-grained num_steps sweep (5-12) to find other transient peaks.

Phase 7C #11 found steps=8 produces transient peak (0.7307), while 4/16/32
degrade to attractor (~0.701). This script sweeps steps=5,6,7,9,10,11,12 to
verify whether steps=8 is the UNIQUE resonance point or if other peaks exist.

All runs use style_strength=None (default) to avoid the cache_key side effect
that degrades results to 0.7015.

Usage (remote):
    python _628_p8b_steps_fine_sweep.py
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(r"I:\Github\Latent_Style\SchrodingerBridge")
PYTHON = r"C:\Progra~1\Python312\python.exe"
EVAL_SCRIPT = ROOT / "src" / "utils" / "run_evaluation.py"
CKPT = ROOT / "exp" / "p4_fusion_breakout" / "t5_b2v2_d2_d4" / "epoch_0007.pt"
OUTPUT_DIR = ROOT / "exp" / "628_ablation" / "p8b_steps_fine"

TEST_DIR = r"I:\wikiart_distinct5_samam_512_classview\test"
CACHE_DIR = r"I:\Github\Latent_Style\eval_cache"

# Sweep steps=5,6,7,9,10,11,12 (8 already known as peak, 4/16/32 known as attractor)
STEPS_TO_TEST = [5, 6, 7, 9, 10, 11, 12]


def run_one(num_steps: int) -> dict:
    exp_name = f"P8B_steps_{num_steps}"
    out_dir = OUTPUT_DIR / exp_name
    summary_path = out_dir / "summary.json"

    if summary_path.is_file():
        print(f"[P8B] SKIP {exp_name} (already done)", flush=True)
        return _read_summary(out_dir, num_steps)

    out_dir.mkdir(parents=True, exist_ok=True)
    eval_log = out_dir / "eval.log"

    cmd = [
        PYTHON, str(EVAL_SCRIPT),
        "--checkpoint", str(CKPT),
        "--output", str(out_dir),
        "--test_dir", TEST_DIR,
        "--cache_dir", CACHE_DIR,
        "--batch_size", "4",
        "--num_steps", str(num_steps),
        "--eval_only_lpips_clip_style",
    ]
    # NOTE: deliberately NOT passing --style_strength to keep it None (default),
    # avoiding the cache_key side effect that degrades results to 0.7015.

    print(f"[P8B] START {exp_name} (num_steps={num_steps})", flush=True)
    t0 = time.time()
    with open(eval_log, "w", encoding="utf-8") as f:
        result = subprocess.run(
            cmd, cwd=str(ROOT), timeout=900,
            stdout=f, stderr=subprocess.STDOUT,
        )
    elapsed = time.time() - t0
    print(f"[P8B] DONE {exp_name} rc={result.returncode} ({elapsed:.0f}s)", flush=True)
    return _read_summary(out_dir, num_steps)


def _read_summary(out_dir: Path, num_steps: int) -> dict:
    """Read summary.json reporting BOTH clip_style metrics to avoid metric confusion.

    Metric confusion bug (Phase 8B discovery):
    - all_pairs_overview.clip_style (~0.73) includes identity pairs (inflated)
    - style_transfer_ability.clip_style (~0.70) excludes identity pairs (pure transfer)
    - Historical baseline 0.7307 is all_pairs_overview, NOT style_transfer_ability
    """
    summary_path = out_dir / "summary.json"
    if not summary_path.is_file():
        print(f"[P8B] WARNING: no summary.json for steps={num_steps}", flush=True)
        return {"num_steps": num_steps, "clip_allpairs": None, "clip_transfer": None,
                "lpips_allpairs": None, "lpips_transfer": None}
    with summary_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    a = data.get("analysis", {}) or {}
    transfer = a.get("style_transfer_ability", {}) or {}
    allpairs = a.get("all_pairs_overview", {}) or {}
    return {
        "num_steps": num_steps,
        "clip_allpairs": allpairs.get("clip_style"),
        "clip_transfer": transfer.get("clip_style"),
        "lpips_allpairs": allpairs.get("content_lpips"),
        "lpips_transfer": transfer.get("content_lpips"),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for ns in STEPS_TO_TEST:
        r = run_one(ns)
        results.append(r)
        print(f"[P8B] steps={r['num_steps']} | "
              f"clip_allpairs={r['clip_allpairs']} clip_transfer={r['clip_transfer']} "
              f"lpips_allpairs={r['lpips_allpairs']} lpips_transfer={r['lpips_transfer']}",
              flush=True)

    # Save combined results
    combined_path = OUTPUT_DIR / "p8b_combined.json"
    with combined_path.open("w", encoding="utf-8") as f:
        json.dump({
            "phase": "8B",
            "description": "Fine-grained num_steps sweep (5-12) to find transient peaks",
            "metric_note": "clip_allpairs = all_pairs_overview (includes identity, ~0.73); "
                           "clip_transfer = style_transfer_ability (pure transfer, ~0.70); "
                           "baseline 0.7307 is all_pairs_overview",
            "baseline": {"num_steps": 8, "clip_allpairs": 0.7307, "lpips_allpairs": 0.3403,
                         "clip_transfer": 0.7016, "lpips_transfer": 0.3520},
            "results": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n[P8B] Combined results saved to {combined_path}", flush=True)

    # Print summary table with both metrics
    print("\n=== Phase 8B Summary (all_pairs_overview metric) ===", flush=True)
    print(f"{'steps':>5} | {'clip_ap':>8} | {'lpips_ap':>8} | {'delta_clip':>10}", flush=True)
    print("-" * 50, flush=True)
    print(f"{'8':>5} | {'0.7307':>8} | {'0.3403':>8} | {'baseline':>10}", flush=True)
    for r in results:
        ns = r["num_steps"]
        c = f"{r['clip_allpairs']:.4f}" if r["clip_allpairs"] is not None else "N/A"
        l = f"{r['lpips_allpairs']:.4f}" if r["lpips_allpairs"] is not None else "N/A"
        delta = f"{r['clip_allpairs']-0.7307:+.4f}" if r["clip_allpairs"] is not None else "N/A"
        print(f"{ns:>5} | {c:>8} | {l:>8} | {delta:>10}", flush=True)

    print("\n=== Phase 8B Summary (style_transfer_ability metric) ===", flush=True)
    print(f"{'steps':>5} | {'clip_tr':>8} | {'lpips_tr':>8}", flush=True)
    print("-" * 35, flush=True)
    print(f"{'8':>5} | {'0.7016':>8} | {'0.3520':>8} | baseline", flush=True)
    for r in results:
        ns = r["num_steps"]
        c = f"{r['clip_transfer']:.4f}" if r["clip_transfer"] is not None else "N/A"
        l = f"{r['lpips_transfer']:.4f}" if r["lpips_transfer"] is not None else "N/A"
        print(f"{ns:>5} | {c:>8} | {l:>8}", flush=True)


if __name__ == "__main__":
    main()
