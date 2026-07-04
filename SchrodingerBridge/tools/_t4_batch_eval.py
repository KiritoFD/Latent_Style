"""T4 batch evaluation: test inference parameter overrides on 4J.1 checkpoint.

Tests 5 parameter combinations to find directions that increase clip_style
without worsening lpips, targeting all_pairs_clip>0.7319 AND all_pairs_lpips<0.3068.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CKPT = REPO / "exp" / "630_phase4j1_dwt_route" / "epoch_0005.pt"
TEST_DIR = "G:/GitHub/Latent_Style/Dataset/distinct5_512/test"
CACHE_DIR = "G:/GitHub/Latent_Style/SchrodingerBridge/exp/eval_cache"
CLIP_HF_CACHE = "G:/GitHub/Latent_Style/SchrodingerBridge/exp/eval_cache/hf"
BATCH_SIZE = 2

# 4J.1 baseline: all_pairs_clip=0.7226, all_pairs_lpips=0.3068
# Target: clip>0.7319 AND lpips<0.3068
EXPERIMENTS = [
    {
        "name": "t3b_adain_ll_010",
        "override": {"model": {"endpoint_adain_scale_ll": 0.10}},
        "rationale": "Larger LL AdaIN scale - test if 0.10 gives more style color injection",
    },
    {
        "name": "t3c_adain_ll_015",
        "override": {"model": {"endpoint_adain_scale_ll": 0.15}},
        "rationale": "Even larger LL AdaIN scale - find the breaking point",
    },
    {
        "name": "t4a_extrap_alpha_05",
        "override": {"model": {"style_extrap_alpha": 0.5}},
        "rationale": "Stronger style extrapolation in high-freq direction (0.4->0.5)",
    },
    {
        "name": "t4b_adain_hh_07",
        "override": {"model": {"endpoint_adain_scale_hh": 0.7}},
        "rationale": "Stronger HH AdaIN (0.5->0.7) for more texture style injection",
    },
    {
        "name": "t4c_adain_lhhl_05",
        "override": {"model": {"endpoint_adain_scale_lh": 0.5, "endpoint_adain_scale_hl": 0.5}},
        "rationale": "Stronger mid-freq LH/HL AdaIN (0.3->0.5) for structure style",
    },
]

OVERRIDE_DIR = REPO / "configs" / "_t4_overrides"
OVERRIDE_DIR.mkdir(parents=True, exist_ok=True)


def run_eval(name: str, override: dict) -> dict:
    """Run evaluation with config override, return all_pairs metrics."""
    override_path = OVERRIDE_DIR / f"{name}.json"
    with open(override_path, "w") as f:
        json.dump(override, f, indent=2)

    out_dir = REPO / "exp" / f"630_local_t4_eval" / name / "full_eval" / "epoch_0005"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(REPO / "src" / "utils" / "run_evaluation.py"),
        "--checkpoint", str(CKPT),
        "--config_override", str(override_path),
        "--test_dir", TEST_DIR,
        "--cache_dir", CACHE_DIR,
        "--clip_hf_cache_dir", CLIP_HF_CACHE,
        "--batch_size", str(BATCH_SIZE),
        "--output", str(out_dir),
    ]

    print(f"\n[{name}] Start: {time.strftime('%H:%M:%S')}")
    print(f"  override: {json.dumps(override)}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True, timeout=600)
    dur = time.time() - t0

    if result.returncode != 0:
        print(f"  FAILED (exit={result.returncode}) in {dur:.0f}s")
        print(f"  stderr: {result.stderr[-500:]}")
        return {"name": name, "status": "failed", "duration_sec": dur}

    # Extract metrics from summary.json
    summary_path = out_dir / "summary.json"
    if not summary_path.exists():
        print(f"  No summary.json in {out_dir}")
        return {"name": name, "status": "no_summary", "duration_sec": dur}

    with open(summary_path) as f:
        summary = json.load(f)

    all_pairs = summary.get("analysis", {}).get("all_pairs_overview", {})
    transfer = summary.get("analysis", {}).get("style_transfer_ability", {})
    clip = all_pairs.get("clip_style", None)
    lpips = all_pairs.get("content_lpips", None)
    t_clip = transfer.get("clip_style", None)
    t_lpips = transfer.get("content_lpips", None)

    print(f"  DONE in {dur:.0f}s")
    print(f"  all_pairs: clip={clip:.4f} lpips={lpips:.4f}")
    print(f"  transfer:  clip={t_clip:.4f} lpips={t_lpips:.4f}")

    return {
        "name": name,
        "status": "ok",
        "duration_sec": dur,
        "all_pairs_clip": clip,
        "all_pairs_lpips": lpips,
        "transfer_clip": t_clip,
        "transfer_lpips": t_lpips,
        "override": override,
    }


def main():
    print("=" * 70)
    print("T4 Batch Evaluation: 4J.1 checkpoint + inference parameter overrides")
    print(f"Baseline 4J.1: all_pairs_clip=0.7226, all_pairs_lpips=0.3068")
    print(f"Target:        all_pairs_clip>0.7319, all_pairs_lpips<0.3068")
    print("=" * 70)

    results = []
    for exp in EXPERIMENTS:
        r = run_eval(exp["name"], exp["override"])
        r["rationale"] = exp["rationale"]
        results.append(r)

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: T4 Batch Evaluation Results")
    print("=" * 70)
    print(f"{'Experiment':<25} {'all_clip':>10} {'all_lpips':>10} {'Δclip':>8} {'Δlpips':>8} {'verdict':>10}")
    print("-" * 70)

    base_clip, base_lpips = 0.7226, 0.3068
    for r in results:
        if r["status"] != "ok":
            print(f"{r['name']:<25} {'FAILED':>10}")
            continue
        c, l = r["all_pairs_clip"], r["all_pairs_lpips"]
        dc, dl = c - base_clip, l - base_lpips
        # Verdict: good if clip up AND lpips down
        if dc > 0.002 and dl < 0.002:
            verdict = "PROMISING"
        elif dc > 0.002 and dl < 0.006:
            verdict = "maybe"
        elif dc > 0 and dl < 0:
            verdict = "trade-off"
        else:
            verdict = "skip"
        print(f"{r['name']:<25} {c:>10.4f} {l:>10.4f} {dc:>+8.4f} {dl:>+8.4f} {verdict:>10}")

    # Save results
    results_path = REPO / "exp" / "630_local_t4_eval" / "batch_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {results_path}")


if __name__ == "__main__":
    main()
