"""628 Inference ablation batch runner for Windows Python.
Runs all 11 inference ablation experiments sequentially.
"""
import subprocess
import sys
import os
import time
from pathlib import Path

ROOT = Path(r"I:\Github\Latent_Style\SchrodingerBridge")
PYTHON = sys.executable
SCRIPT = ROOT / "628_infer_ablation.py"
LOGDIR = ROOT / "exp" / "628_ablation" / "infer_ablation" / "logs"
LOGDIR.mkdir(parents=True, exist_ok=True)

EXPERIMENTS = [
    # I5: fiber_cfg_scale
    ("I5_cfg1.0", {"fiber_cfg_scale": 1.0}),
    ("I5_cfg2.0", {"fiber_cfg_scale": 2.0}),
    ("I5_cfg3.0", {"fiber_cfg_scale": 3.0}),
    # I6: fiber_velocity_scale
    ("I6_vel0.5", {"fiber_velocity_scale": 0.5}),
    ("I6_vel1.5", {"fiber_velocity_scale": 1.5}),
    ("I6_vel2.0", {"fiber_velocity_scale": 2.0}),
    # I7: fiber_source_repulse_scale
    ("I7_repulse0.5", {"fiber_source_repulse_scale": 0.5}),
    ("I7_repulse1.0", {"fiber_source_repulse_scale": 1.0}),
    # I8: tri_band_inference_lock
    ("I8_triband_a0.3", {"tri_band_inference_lock": True, "tri_band_edge_lock_alpha": 0.3}),
    ("I8_triband_a0.7", {"tri_band_inference_lock": True, "tri_band_edge_lock_alpha": 0.7}),
    # I9: fiber_only_endpoint
    ("I9_fiber_only", {"fiber_only_endpoint": True}),
    # I10: lowpass_mode = avg_pool
    ("I10_avgpool", {"lowpass_mode": "avg_pool"}),
]

import json

def main():
    print(f"=== 628 Inference Ablation Batch ===")
    print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python: {PYTHON}")
    print(f"Root: {ROOT}")
    print(f"Experiments: {len(EXPERIMENTS)}")

    results = {}
    for i, (name, overrides) in enumerate(EXPERIMENTS, 1):
        overrides_json = json.dumps(overrides)
        log_path = LOGDIR / f"{name}.log"
        print(f"\n[{i}/{len(EXPERIMENTS)}] {time.strftime('%H:%M:%S')} Running {name}: {overrides_json}")

        try:
            with open(log_path, "w") as logf:
                proc = subprocess.run(
                    [PYTHON, str(SCRIPT), name, overrides_json],
                    cwd=str(ROOT),
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                    timeout=600,
                )
            # Read result
            result_path = ROOT / "exp" / "628_ablation" / "infer_ablation" / f"{name}.json"
            if result_path.exists():
                r = json.load(open(result_path))
                m = r.get("metrics", {})
                clip = m.get("allpairs_clip_style") or m.get("transfer_clip_style")
                lpips = m.get("allpairs_content_lpips") or m.get("transfer_content_lpips")
                results[name] = {"clip": clip, "lpips": lpips, "status": "OK"}
                print(f"  -> clip={clip}, lpips={lpips}")
            else:
                results[name] = {"status": "NO_RESULT"}
                print(f"  -> NO RESULT FILE")
        except subprocess.TimeoutExpired:
            results[name] = {"status": "TIMEOUT"}
            print(f"  -> TIMEOUT")
        except Exception as e:
            results[name] = {"status": f"ERROR: {e}"}
            print(f"  -> ERROR: {e}")

    print(f"\n=== Summary ===")
    print(f"{'Name':25s} {'clip':>8s} {'lpips':>8s} {'Status':>10s}")
    print("-" * 55)
    for name, r in results.items():
        c = f"{r['clip']:.4f}" if r.get('clip') is not None else "N/A"
        l = f"{r['lpips']:.4f}" if r.get('lpips') is not None else "N/A"
        s = r.get('status', '?')
        print(f"{name:25s} {c:>8s} {l:>8s} {s:>10s}")

    print(f"\nDone: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Save summary
    summary_path = ROOT / "exp" / "628_ablation" / "infer_ablation" / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({"baseline": {"clip": 0.7307, "lpips": 0.3403}, "results": results}, f, indent=2)
    print(f"Summary saved: {summary_path}")

if __name__ == "__main__":
    main()
