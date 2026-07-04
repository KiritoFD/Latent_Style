"""T12 batch evaluation: test inference parameter overrides on T11 (p=0.8) checkpoint.

T11 baseline: all_pairs_clip=0.7213, all_pairs_lpips=0.2868
Target: clip>0.7319 AND lpips<0.3068 (margin 0.0200, need clip +0.0106)

Focus: dwt_ll_route_alpha (LL参与cross-attention query, 未测试过的新机制)
"""
import json
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
        "name": "t12a_dwt_ll_route_01",
        "override": {"model": {"cross_attn_dwt_ll_route_alpha": 0.1}},
        "rationale": "LL轻度参与cross-attn query (0.0->0.1), 提升低频风格化",
    },
    {
        "name": "t12b_dwt_ll_route_02",
        "override": {"model": {"cross_attn_dwt_ll_route_alpha": 0.2}},
        "rationale": "LL中度参与cross-attn query (0.0->0.2)",
    },
    {
        "name": "t12c_dwt_ll_route_03",
        "override": {"model": {"cross_attn_dwt_ll_route_alpha": 0.3}},
        "rationale": "LL较强参与cross-attn query (0.0->0.3)",
    },
    {
        "name": "t12d_extrap_06",
        "override": {"model": {"style_extrap_alpha": 0.6}},
        "rationale": "对照: 更强风格外推 (0.4->0.6)",
    },
    {
        "name": "t12e_adain_ll_010",
        "override": {"model": {"endpoint_adain_scale_ll": 0.10}},
        "rationale": "对照: LL endpoint adain (0.0->0.10)",
    },
]

OVERRIDE_DIR = REPO / "configs" / "_t12_overrides"
OVERRIDE_DIR.mkdir(parents=True, exist_ok=True)


def run_eval(name, override):
    override_path = OVERRIDE_DIR / f"{name}.json"
    with open(override_path, "w") as f:
        json.dump(override, f, indent=2)

    out_dir = REPO / "exp" / "630_local_t12_eval" / name / "full_eval" / "epoch_0005"
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

    summary_path = out_dir / "summary.json"
    if not summary_path.exists():
        print(f"  No summary.json in {out_dir}")
        return {"name": name, "status": "no_summary", "duration_sec": dur}

    with open(summary_path) as f:
        summary = json.load(f)

    all_pairs = summary.get("analysis", {}).get("all_pairs_overview", {})
    clip = all_pairs.get("clip_style", None)
    lpips = all_pairs.get("content_lpips", None)

    print(f"  DONE in {dur:.0f}s")
    if clip is not None:
        print(f"  all_pairs: clip={clip:.4f} lpips={lpips:.4f}")

    return {
        "name": name,
        "status": "ok",
        "duration_sec": dur,
        "all_pairs_clip": clip,
        "all_pairs_lpips": lpips,
        "override": override,
    }


def main():
    print("=" * 70)
    print("T12 Batch Eval: T11 (p=0.8) checkpoint + inference parameter overrides")
    print(f"Baseline T11: all_pairs_clip=0.7213, all_pairs_lpips=0.2868")
    print(f"Target:       all_pairs_clip>0.7319, all_pairs_lpips<0.3068")
    print("=" * 70)

    results = []
    for exp in EXPERIMENTS:
        r = run_eval(exp["name"], exp["override"])
        r["rationale"] = exp["rationale"]
        results.append(r)

    print("\n" + "=" * 70)
    print("SUMMARY: T12 Batch Evaluation Results")
    print("=" * 70)
    print(f"{'Experiment':<25} {'all_clip':>10} {'all_lpips':>10} {'d_clip':>8} {'d_lpips':>8} {'verdict':>12}")
    print("-" * 75)

    base_clip, base_lpips = 0.7213, 0.2868
    for r in results:
        if r["status"] != "ok" or r["all_pairs_clip"] is None:
            print(f"{r['name']:<25} {'FAILED':>10}")
            continue
        c, l = r["all_pairs_clip"], r["all_pairs_lpips"]
        dc, dl = c - base_clip, l - base_lpips
        if c > 0.7319 and l < 0.3068:
            verdict = "TARGET HIT!"
        elif dc > 0.003 and dl < 0.005:
            verdict = "PROMISING"
        elif dc > 0.001 and dl < 0.010:
            verdict = "maybe"
        elif dc > 0 and dl < 0:
            verdict = "win-win"
        else:
            verdict = "skip"
        print(f"{r['name']:<25} {c:>10.4f} {l:>10.4f} {dc:>+8.4f} {dl:>+8.4f} {verdict:>12}")

    results_path = REPO / "exp" / "630_local_t12_eval" / "batch_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {results_path}")


if __name__ == "__main__":
    main()
