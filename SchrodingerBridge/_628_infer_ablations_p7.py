"""Phase 7C: 12 inference-side ablations on T5 ep7 checkpoint (no training).

Spec tasks 4.1-4.12. Each ablation modifies ONE inference parameter and runs
eval-only. Output goes to exp/628_ablation/infer_ablation_p7/<exp>.json

Skips experiments that already have output JSON.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(r"I:\Github\Latent_Style\SchrodingerBridge")
PYTHON = r"C:\Progra~1\Python312\python.exe"
INFER_SCRIPT = ROOT / "628_infer_ablation.py"
OUTPUT_DIR = ROOT / "exp" / "628_ablation" / "infer_ablation_p7"

# 12 inference ablations from spec Tasks 4.1-4.12
# Each entry: (exp_name, overrides_dict)
# Note: full_eval_num_steps and full_eval_style_strength are full_eval-section
# params; 628_infer_ablation.py only routes to model/bridge. For #11/#12 we
# patch them via env vars / direct call to run_evaluation.py separately.
INFER_ABLATIONS = [
    # #1: transport_prediction_mode (velocity -> endpoint)
    ("P7I1_endpoint_pred", {"transport_prediction_mode": "endpoint"}),
    # #2: style_attn_sharpen_scale (inference-side)
    ("P7I2_sharpen_0", {"style_attn_sharpen_scale": 0}),
    ("P7I2_sharpen_25", {"style_attn_sharpen_scale": 2.5}),
    ("P7I2_sharpen_50", {"style_attn_sharpen_scale": 5.0}),
    # #3: endpoint_high_scale (inference-side)
    ("P7I3_ephigh_0", {"endpoint_high_scale": 0}),
    ("P7I3_ephigh_20", {"endpoint_high_scale": 2.0}),
    # #4: affine_connection_gamma_scale (inference-side)
    ("P7I4_gamma_0", {"affine_connection_gamma_scale": 0}),
    ("P7I4_gamma_05", {"affine_connection_gamma_scale": 0.5}),
    # #5: affine_connection_beta_scale (inference-side)
    ("P7I5_beta_0", {"affine_connection_beta_scale": 0}),
    ("P7I5_beta_20", {"affine_connection_beta_scale": 2.0}),
    # #6: endpoint_film_init_std inference amplification
    ("P7I6_film_01", {"endpoint_film_init_std": 0.1}),
    # #7: style_attn_num_tokens (inference-side)
    ("P7I7_tokens_64", {"style_attn_num_tokens": 64}),
    ("P7I7_tokens_512", {"style_attn_num_tokens": 512}),
    # #8: solver_stochastic_noise_scale
    ("P7I8_noise_0", {"solver_stochastic_noise_scale": 0}),
    ("P7I8_noise_008", {"solver_stochastic_noise_scale": 0.08}),
    # #9: bridge_path_mode (vertical -> tri_band, inference-side)
    ("P7I9_triband", {"bridge_path_mode": "tri_band"}),
    # #10: swd_distance_mode (cdf -> squared, inference-side)
    ("P7I10_swd_squared", {"swd_distance_mode": "squared"}),
]


def already_done(exp_name: str) -> bool:
    out = OUTPUT_DIR / f"{exp_name}.json"
    return out.is_file()


def run_one(exp_name: str, overrides: dict) -> bool:
    print(f"\n{'='*60}")
    print(f"[P7C] Running {exp_name}")
    print(f"     overrides={overrides}")
    print(f"{'='*60}")
    overrides_json = json.dumps(overrides)
    cmd = [PYTHON, str(INFER_SCRIPT), exp_name, overrides_json]
    try:
        result = subprocess.run(
            cmd,
            cwd=str(ROOT),
            timeout=600,
            capture_output=True,
            text=True,
        )
        log_path = OUTPUT_DIR / f"{exp_name}.log"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as f:
            f.write("=== STDOUT ===\n")
            f.write(result.stdout or "")
            f.write("\n=== STDERR ===\n")
            f.write(result.stderr or "")
        if result.returncode != 0:
            print(f"[P7C] FAIL {exp_name} rc={result.returncode}")
            tail = (result.stderr or "")[-500:]
            print(f"[P7C] stderr tail: {tail}")
            return False
        # Try to read result JSON
        out_json = OUTPUT_DIR / f"{exp_name}.json"
        if out_json.is_file():
            with out_json.open("r", encoding="utf-8") as f:
                rec = json.load(f)
            m = rec.get("metrics", {})
            c = m.get("allpairs_clip_style") or m.get("transfer_clip_style")
            l = m.get("allpairs_content_lpips") or m.get("transfer_content_lpips")
            print(f"[P7C] OK {exp_name} | clip={c} lpips={l}")
        else:
            print(f"[P7C] OK {exp_name} (no metrics JSON)")
        return True
    except subprocess.TimeoutExpired:
        print(f"[P7C] TIMEOUT {exp_name}")
        return False
    except Exception as e:
        print(f"[P7C] EXCEPTION {exp_name}: {e}")
        return False


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pending = [(n, o) for n, o in INFER_ABLATIONS if not already_done(n)]
    done_count = len(INFER_ABLATIONS) - len(pending)
    print(f"[P7C] Total={len(INFER_ABLATIONS)} already_done={done_count} pending={len(pending)}")

    if not pending:
        print("[P7C] All inference ablations already completed.")
        return 0

    t0 = time.time()
    successes = 0
    failures = 0
    for i, (name, overrides) in enumerate(pending, 1):
        print(f"\n[P7C] Progress {i}/{len(pending)} elapsed={time.time()-t0:.0f}s")
        ok = run_one(name, overrides)
        if ok:
            successes += 1
        else:
            failures += 1

    print(f"\n{'='*60}")
    print(f"[P7C] DONE: {successes} ok, {failures} fail, total={len(pending)}")
    print(f"[P7C] Elapsed: {time.time()-t0:.0f}s")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
