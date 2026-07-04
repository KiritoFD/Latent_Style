"""Phase 8F: Orthogonal ablation batch runner (train + eval)."""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(r"I:\Github\Latent_Style\SchrodingerBridge")
PYTHON = r"C:\Progra~1\Python312\python.exe"
CONFIG_DIR = ROOT / "configs" / "ablations" / "628_orthogonal"
EVAL_SCRIPT = ROOT / "src" / "utils" / "run_evaluation.py"
RUN_SCRIPT = ROOT / "src" / "run.py"
LOG_DIR = ROOT / "exp" / "628_ablation" / "orthogonal_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

TEST_DIR = r"I:\wikiart_distinct5_samam_512_classview\test"
CACHE_DIR = r"I:\Github\Latent_Style\eval_cache"

EXPERIMENTS = [
    "O1_only_ll", "O2_only_lhhl0", "O3_only_chvar", "O4_only_color",
    "O5_ll_chvar", "O6_ll_color", "O7_chvar_color",
    "O8_lhhl0_chvar", "O9_lhhl0_color", "O10_ll_lhhl0", "O11_chvar_color_lhhl0",
]


def run_one(name: str) -> dict:
    cfg_path = CONFIG_DIR / f"{name}.json"
    save_dir = ROOT / "exp" / "628_ablation" / "orthogonal" / name
    ckpt = save_dir / "epoch_0010.pt"
    eval_dir = save_dir / "full_eval" / "epoch_0010"
    summary = eval_dir / "summary.json"

    if summary.is_file():
        print(f"[P8F] SKIP {name} (already done)", flush=True)
        return _read_summary(name, summary)

    # Train
    if not ckpt.is_file():
        print(f"[P8F] TRAIN {name}", flush=True)
        t0 = time.time()
        log = LOG_DIR / f"{name}_train.log"
        with open(log, "w", encoding="utf-8") as f:
            result = subprocess.run(
                [PYTHON, str(RUN_SCRIPT), "--config", str(cfg_path)],
                cwd=str(ROOT), timeout=600,
                stdout=f, stderr=subprocess.STDOUT,
            )
        print(f"[P8F] TRAIN DONE {name} rc={result.returncode} ({time.time()-t0:.0f}s)", flush=True)
        if result.returncode != 0 or not ckpt.is_file():
            print(f"[P8F] TRAIN FAILED {name}", flush=True)
            return {"name": name, "error": "train_failed"}

    # Eval
    print(f"[P8F] EVAL {name}", flush=True)
    t0 = time.time()
    eval_log = LOG_DIR / f"{name}_eval.log"
    eval_dir.mkdir(parents=True, exist_ok=True)
    with open(eval_log, "w", encoding="utf-8") as f:
        result = subprocess.run(
            [PYTHON, str(EVAL_SCRIPT),
             "--checkpoint", str(ckpt),
             "--output", str(eval_dir),
             "--test_dir", TEST_DIR,
             "--cache_dir", CACHE_DIR,
             "--batch_size", "16",
             "--num_steps", "8",
             "--eval_only_lpips_clip_style"],
            cwd=str(ROOT), timeout=300,
            stdout=f, stderr=subprocess.STDOUT,
        )
    print(f"[P8F] EVAL DONE {name} rc={result.returncode} ({time.time()-t0:.0f}s)", flush=True)
    return _read_summary(name, summary)


def _read_summary(name: str, summary: Path) -> dict:
    if not summary.is_file():
        return {"name": name, "error": "no_summary"}
    with summary.open("r", encoding="utf-8") as f:
        data = json.load(f)
    a = data.get("analysis", {}) or {}
    ap = a.get("all_pairs_overview", {}) or {}
    tr = a.get("style_transfer_ability", {}) or {}
    return {
        "name": name,
        "clip_allpairs": ap.get("clip_style"),
        "lpips_allpairs": ap.get("content_lpips"),
        "clip_transfer": tr.get("clip_style"),
        "lpips_transfer": tr.get("content_lpips"),
    }


def main():
    print(f"[P8F] === Orthogonal Ablation START ===", flush=True)
    results = []
    for name in EXPERIMENTS:
        r = run_one(name)
        results.append(r)
        ca = r.get("clip_allpairs")
        print(f"[P8F] {name}: clip_allpairs={ca}", flush=True)

    # Save combined
    combined = LOG_DIR / "p8f_combined.json"
    with combined.open("w", encoding="utf-8") as f:
        json.dump({
            "phase": "8F",
            "baseline": {"clip_allpairs": 0.7307, "lpips_allpairs": 0.3403},
            "clean_base": {"clip_allpairs": 0.7073, "note": "negative interaction!"},
            "results": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n[P8F] Combined saved to {combined}", flush=True)

    # Summary table
    print("\n=== Phase 8F Orthogonal Ablation Summary ===", flush=True)
    print(f"{'name':>25} | {'clip_ap':>8} | {'lpips_ap':>8} | {'delta_clip':>10}", flush=True)
    print("-" * 65, flush=True)
    print(f"{'BASELINE (T5 ep7)':>25} | {'0.7307':>8} | {'0.3403':>8} | {'---':>10}", flush=True)
    print(f"{'CLEAN_BASE (5 mods)':>25} | {'0.7073':>8} | {'---':>8} | {'-0.0234':>10}", flush=True)
    for r in results:
        name = r["name"]
        ca = r.get("clip_allpairs")
        la = r.get("lpips_allpairs")
        if ca is None:
            print(f"{name:>25} | {'FAIL':>8} | {'---':>8} | {'---':>10}", flush=True)
        else:
            delta = f"{ca-0.7307:+.4f}"
            print(f"{name:>25} | {ca:>8.4f} | {la:>8.4f} | {delta:>10}", flush=True)


if __name__ == "__main__":
    main()
