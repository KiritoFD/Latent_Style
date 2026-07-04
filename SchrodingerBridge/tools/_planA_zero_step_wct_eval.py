"""Plan A: Zero-Step WCT Pre-alignment batch evaluation on T11 checkpoint.

Tests 3 alpha values (0.5, 0.7, 1.0) to find the best trade-off between
CLIP-S (style similarity, boosted by LL color transfer) and LPIPS (content
fidelity, potentially damaged by LL modification).

T11 baseline: clip=0.7213, lpips=0.2868
Target: clip > 0.732 AND lpips < 0.30 (DUAL BEAT)
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CKPT = REPO / "exp" / "630_local_t11_stochastic_dwt_p08" / "epoch_0005.pt"
TEST_DIR = "G:/GitHub/Latent_Style/Dataset/distinct5_512/test"
CACHE_DIR = "G:/GitHub/Latent_Style/SchrodingerBridge/exp/eval_cache"
CLIP_HF_CACHE = "G:/GitHub/Latent_Style/SchrodingerBridge/exp/eval_cache/hf"
BATCH_SIZE = 2

EXPERIMENTS = [
    {
        "name": "planA_alpha_05",
        "override": {"model": {"zero_step_wct_enabled": True, "zero_step_wct_alpha": 0.5}},
        "rationale": "Moderate WCT — 50% style color blend on LL",
    },
    {
        "name": "planA_alpha_07",
        "override": {"model": {"zero_step_wct_enabled": True, "zero_step_wct_alpha": 0.7}},
        "rationale": "Strong WCT — 70% style color blend on LL",
    },
    {
        "name": "planA_alpha_10",
        "override": {"model": {"zero_step_wct_enabled": True, "zero_step_wct_alpha": 1.0}},
        "rationale": "Full WCT — 100% style color match on LL (max CLIP boost)",
    },
]

OVERRIDE_DIR = REPO / "configs" / "_planA_overrides"
OVERRIDE_DIR.mkdir(parents=True, exist_ok=True)


def run_eval(name: str, override: dict) -> dict:
    """Run evaluation with config override, return all_pairs metrics."""
    override_path = OVERRIDE_DIR / f"{name}.json"
    with open(override_path, "w") as f:
        json.dump(override, f, indent=2)

    out_dir = REPO / "exp" / "630_planA_zero_step_wct" / name / "full_eval" / "epoch_0005"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(REPO / "src" / "utils" / "run_evaluation.py"),
        "--checkpoint", str(CKPT),
        "--config_override", str(override_path),
        "--test_dir", TEST_DIR,
        "--cache_dir", CACHE_DIR,
        "--clip_hf_cache_dir", CLIP_HF_CACHE,
        "--clip_backend", "hf",
        "--batch_size", str(BATCH_SIZE),
        "--output", str(out_dir),
    ]

    print(f"\n[{name}] Start: {time.strftime('%H:%M:%S')}")
    print(f"  override: {json.dumps(override)}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True, timeout=900)
    dur = time.time() - t0
    print(f"  elapsed: {dur:.1f}s")

    # Parse metrics from output
    metrics = {"name": name, "duration": dur, "override": override}
    stdout = result.stdout
    stderr = result.stderr

    if result.returncode != 0:
        print(f"  FAIL (returncode={result.returncode})")
        print(f"  stderr tail: {stderr[-500:] if stderr else '(empty)'}")
        metrics["status"] = "fail"
        metrics["error"] = stderr[-500:] if stderr else "(empty stderr)"
        return metrics

    # Look for all_pairs metrics in stdout
    for line in stdout.splitlines():
        line_lower = line.lower().strip()
        if "all_pairs_clip" in line_lower or "all_pairs_lpips" in line_lower:
            print(f"  {line.strip()}")
        if "clip_style" in line_lower and "=" in line:
            print(f"  {line.strip()}")

    # Try to read summary.json
    summary_path = out_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        for key in ["all_pairs_clip", "all_pairs_lpips", "clip_style", "lpips"]:
            if key in summary:
                metrics[key] = float(summary[key])

    metrics["status"] = "pass" if "all_pairs_clip" in metrics else "no_metrics"
    print(f"  metrics: clip={metrics.get('all_pairs_clip', '?')}, lpips={metrics.get('all_pairs_lpips', '?')}")
    return metrics


def main():
    print(f"=== Plan A: Zero-Step WCT Pre-alignment ===")
    print(f"Checkpoint: {CKPT}")
    print(f"Test dir: {TEST_DIR}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Experiments: {len(EXPERIMENTS)}")
    print(f"T11 baseline: clip=0.7213, lpips=0.2868")
    print()

    results = []
    for exp in EXPERIMENTS:
        try:
            metrics = run_eval(exp["name"], exp["override"])
            metrics["rationale"] = exp["rationale"]
            results.append(metrics)
        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT (900s)")
            results.append({"name": exp["name"], "status": "timeout", "override": exp["override"]})
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            results.append({"name": exp["name"], "status": "error", "error": str(e), "override": exp["override"]})

    # Summary
    print(f"\n{'='*60}")
    print(f"=== Plan A Results Summary ===")
    print(f"{'Name':<20} {'CLIP':>8} {'LPIPS':>8} {'Status':>10}")
    print(f"{'-'*20} {'-'*8} {'-'*8} {'-'*10}")
    print(f"{'T11 baseline':<20} {'0.7213':>8} {'0.2868':>8} {'ref':>10}")
    for r in results:
        clip = r.get("all_pairs_clip", "?")
        lpips = r.get("all_pairs_lpips", "?")
        status = r.get("status", "?")
        clip_str = f"{clip:.4f}" if isinstance(clip, float) else str(clip)
        lpips_str = f"{lpips:.4f}" if isinstance(lpips, float) else str(lpips)
        print(f"{r['name']:<20} {clip_str:>8} {lpips_str:>8} {status:>10}")

    # Save results
    results_path = REPO / "exp" / "630_planA_zero_step_wct" / "results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
