"""Exp1c: AdaIN scale inference sensitivity sweep.
Uses existing checkpoint, only varies inference-time endpoint_adain_scale.
No training needed.

Run on remote RTX 3060.
"""
import json, os, sys, subprocess, csv, tempfile
from pathlib import Path

WEAVE_ROOT = Path(r"I:\Github\Latent_Style\WEAVE")
os.chdir(WEAVE_ROOT)

CKPT = "runs/submission/hf_oriented_internal_early_stop/epoch_0004.pt"
TEST_DIR = "data/test"
SCALES = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
OUTPUT = WEAVE_ROOT / "exp" / "rebuttal" / "exp1c_adain_sweep.json"

def run_eval(scale, out_dir):
    """Run evaluation with given AdaIN scale."""
    # Create config override
    override = {"model": {"endpoint_adain_scale": scale}}
    override_path = Path(tempfile.gettempdir()) / f"adain_{scale}.json"
    override_path.write_text(json.dumps(override))

    cmd = [
        sys.executable, "-u", "utils/run_evaluation.py",
        "--checkpoint", CKPT,
        "--config_override", str(override_path),
        "--output", out_dir,
        "--test_dir", TEST_DIR,
        "--batch_size", "8",
        "--vae_decode_batch_size", "8",
        "--num_steps", "8",
        "--max_ref_cache", "16",
        "--max_ref_compare", "16",
    ]
    print(f"  CMD: {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=str(WEAVE_ROOT))
    if r.returncode != 0:
        print(f"  ERROR: eval failed for scale={scale}")
        return None

    # Read metrics
    metrics_path = Path(out_dir) / "metrics.csv"
    if not metrics_path.exists():
        print(f"  ERROR: metrics.csv not found")
        return None
    with open(metrics_path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    clip_s = sum(float(r["clip_style"]) for r in rows) / len(rows)
    lpips = sum(float(r["content_lpips"]) for r in rows) / len(rows)

    # Run DINO
    dino_cmd = [
        sys.executable, "-u", "utils/compute_dino_metrics.py",
        "--eval_dir", out_dir,
        "--test_dir", TEST_DIR,
        "--batch_size", "4",
        "--max_refs_per_style", "30",
        "--cache_dir", "exp/eval_cache/hf",
    ]
    r = subprocess.run(dino_cmd, cwd=str(WEAVE_ROOT))
    dino_s = dino_c = None
    dino_summary = Path(out_dir) / "dino_summary.json"
    if dino_summary.exists():
        d = json.loads(dino_summary.read_text())
        dino_s = d.get("all_dino_s")
        dino_c = d.get("all_dino_c")

    return {"clip_s": clip_s, "lpips": lpips, "dino_s": dino_s, "dino_c": dino_c}

def main():
    print("=" * 60)
    print("Exp1c: AdaIN Scale Inference Sensitivity Sweep")
    print("=" * 60)
    print(f"Scales: {SCALES}")
    print(f"Checkpoint: {CKPT}")
    print()

    results = {}
    for scale in SCALES:
        print(f"\n--- Scale = {scale} ---", flush=True)
        out_dir = f"exp/rebuttal/adain_{scale}"
        r = run_eval(scale, out_dir)
        if r:
            results[str(scale)] = r
            print(f"  DINO-S={r['dino_s']:.6f}, CLIP-S={r['clip_s']:.4f}, LPIPS={r['lpips']:.4f}, DINO-C={r['dino_c']:.6f}")
        else:
            results[str(scale)] = None

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Scale':<8} {'DINO-S':<12} {'CLIP-S':<10} {'LPIPS':<10} {'DINO-C':<12}")
    print("-" * 52)
    for scale in SCALES:
        r = results.get(str(scale))
        if r:
            print(f"{scale:<8} {r['dino_s']:<12.6f} {r['clip_s']:<10.4f} {r['lpips']:<10.4f} {r['dino_c']:<12.6f}")
        else:
            print(f"{scale:<8} FAILED")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to: {OUTPUT}")

if __name__ == "__main__":
    main()
