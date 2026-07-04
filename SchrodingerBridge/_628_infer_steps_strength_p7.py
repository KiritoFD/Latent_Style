"""Phase 7C supplementary: #11 (num_steps) and #12 (style_strength) inference ablations.

These two ablations modify full_eval-section params (not model/bridge), so they
cannot go through 628_infer_ablation.py. Instead we directly invoke
src/utils/run_evaluation.py with --num_steps / --style_strength overrides.

We re-use the T5 ep7 checkpoint and write outputs into
exp/628_ablation/infer_ablation_p7/<exp>/ to keep parity with #1-#10.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(r"I:\Github\Latent_Style\SchrodingerBridge")
PYTHON = r"C:\Progra~1\Python312\python.exe"
EVAL_SCRIPT = ROOT / "src" / "utils" / "run_evaluation.py"
CKPT = ROOT / "exp" / "p4_fusion_breakout" / "t5_b2v2_d2_d4" / "epoch_0007.pt"
OUTPUT_DIR = ROOT / "exp" / "628_ablation" / "infer_ablation_p7"

TEST_DIR = r"I:\wikiart_distinct5_samam_512_classview\test"
CACHE_DIR = r"I:\Github\Latent_Style\eval_cache"

# #11: num_steps sweep (baseline=8)
# #12: style_strength sweep (baseline=1.0)
ABLATIONS = [
    ("P7I11_steps_4", {"num_steps": 4, "style_strength": 1.0}),
    ("P7I11_steps_16", {"num_steps": 16, "style_strength": 1.0}),
    ("P7I11_steps_32", {"num_steps": 32, "style_strength": 1.0}),
    ("P7I12_ss_05", {"num_steps": 8, "style_strength": 0.5}),
    ("P7I12_ss_15", {"num_steps": 8, "style_strength": 1.5}),
    ("P7I12_ss_20", {"num_steps": 8, "style_strength": 2.0}),
]


def already_done(exp_name: str) -> bool:
    out = OUTPUT_DIR / exp_name / "summary.json"
    return out.is_file()


def run_one(exp_name: str, num_steps: int, style_strength: float) -> bool:
    print(f"\n{'='*60}")
    print(f"[P7C-SS] Running {exp_name} num_steps={num_steps} style_strength={style_strength}")
    print(f"{'='*60}")
    out_dir = OUTPUT_DIR / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        PYTHON, str(EVAL_SCRIPT),
        "--checkpoint", str(CKPT),
        "--output", str(out_dir),
        "--test_dir", TEST_DIR,
        "--cache_dir", CACHE_DIR,
        "--batch_size", "4",
        "--num_steps", str(num_steps),
        "--style_strength", str(style_strength),
        "--eval_only_lpips_clip_style",
    ]
    log_path = out_dir / "eval.log"
    try:
        with log_path.open("w", encoding="utf-8") as logf:
            result = subprocess.run(
                cmd,
                cwd=str(ROOT),
                timeout=900,
                stdout=logf,
                stderr=subprocess.STDOUT,
                text=True,
            )
        if result.returncode != 0:
            print(f"[P7C-SS] FAIL {exp_name} rc={result.returncode}")
            return False
        # Read summary
        summary_path = out_dir / "summary.json"
        if summary_path.is_file():
            with summary_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            a = data.get("analysis", {})
            transfer = a.get("style_transfer_ability", {})
            allpairs = a.get("all_pairs_overview", {})
            c = transfer.get("clip_style") or allpairs.get("clip_style")
            l = transfer.get("content_lpips") or allpairs.get("content_lpips")
            print(f"[P7C-SS] OK {exp_name} | clip={c} lpips={l}")
        else:
            print(f"[P7C-SS] OK {exp_name} (no summary.json)")
        return True
    except subprocess.TimeoutExpired:
        print(f"[P7C-SS] TIMEOUT {exp_name}")
        return False
    except Exception as e:
        print(f"[P7C-SS] EXCEPTION {exp_name}: {e}")
        return False


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pending = [(n, p) for n, p in ABLATIONS if not already_done(n)]
    print(f"[P7C-SS] Total={len(ABLATIONS)} already_done={len(ABLATIONS)-len(pending)} pending={len(pending)}")
    if not pending:
        print("[P7C-SS] All done.")
        return 0
    t0 = time.time()
    succ = fail = 0
    for i, (name, params) in enumerate(pending, 1):
        print(f"\n[P7C-SS] Progress {i}/{len(pending)} elapsed={time.time()-t0:.0f}s")
        ok = run_one(name, params["num_steps"], params["style_strength"])
        if ok: succ += 1
        else: fail += 1
    print(f"\n[P7C-SS] DONE: {succ} ok, {fail} fail")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
