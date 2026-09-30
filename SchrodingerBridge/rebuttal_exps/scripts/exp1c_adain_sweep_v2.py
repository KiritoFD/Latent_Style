"""Exp1c: AdaIN scale inference sweep.
Runs inference with different endpoint_adain_scale values using the production
checkpoint, then evaluates with paper-canonical DINO metrics.

Missing points: 1.0, 1.25, 1.5 (have 0.0, 0.5, 2.0 from ablation_v2)
Also re-runs 2.0 (production default) for consistency with paper-canonical DINO.

Run on remote RTX 3060.
"""
import json, os, sys, subprocess, time
from pathlib import Path

WEAVE_ROOT = Path(r"I:\Github\Latent_Style\WEAVE")
os.chdir(WEAVE_ROOT)

CKPT = WEAVE_ROOT / "runs" / "submission" / "hf_oriented_internal_early_stop" / "epoch_0004.pt"
TEST_DIR = WEAVE_ROOT / "data" / "test"
OUTPUT_ROOT = WEAVE_ROOT / "exp" / "rebuttal" / "exp1c_adain_sweep"
HF_CACHE = WEAVE_ROOT / "exp" / "eval_cache" / "hf"

# AdaIN scales to run (1.0, 1.25, 1.5 are missing; 2.0 is production default for consistency)
SCALES = [1.0, 1.25, 1.5, 2.0]

def run_cmd(cmd, log_file=None):
    """Run a command, streaming output to log file."""
    print(f"  CMD: {' '.join(str(c) for c in cmd)}", flush=True)
    if log_file:
        with open(log_file, "w") as f:
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
        return result.returncode
    else:
        result = subprocess.run(cmd, text=True)
        return result.returncode

def main():
    print("=" * 60)
    print("Exp1c: AdaIN Scale Inference Sweep")
    print("=" * 60)
    print(f"Checkpoint: {CKPT}")
    print(f"Test dir: {TEST_DIR}")
    print(f"Output: {OUTPUT_ROOT}")
    print(f"Scales: {SCALES}")
    print()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "_logs").mkdir(exist_ok=True)

    results = []

    for scale in SCALES:
        name = f"adain_{str(scale).replace('.', '_')}"
        out_dir = OUTPUT_ROOT / name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Create override JSON (no BOM)
        override = {"model": {"endpoint_adain_scale": scale}}
        override_file = OUTPUT_ROOT / f"_override_{name}.json"
        override_file.write_text(json.dumps(override), encoding="utf-8")

        print(f"\n=== Scale={scale} ({name}) ===", flush=True)
        t0 = time.time()

        # Step 1: Run inference
        infer_log = OUTPUT_ROOT / "_logs" / f"infer_{name}.log"
        cmd = [
            sys.executable, "run_evaluation.py",
            "--checkpoint", str(CKPT),
            "--output", str(out_dir),
            "--batch_size", "2",
            "--ref_feature_batch_size", "2",
            "--vae_decode_batch_size", "16",
            "--test_dir", str(TEST_DIR),
            "--force_regen",
            "--config_override", str(override_file),
        ]
        rc = run_cmd(cmd, infer_log)
        t_infer = time.time() - t0
        print(f"  Inference: rc={rc}, time={t_infer:.1f}s", flush=True)

        if rc != 0:
            print(f"  ERROR: inference failed for scale={scale}", flush=True)
            results.append({"scale": scale, "status": "infer_failed", "rc": rc})
            continue

        # Step 2: Run paper-canonical DINO evaluation
        t1 = time.time()
        dino_log = OUTPUT_ROOT / "_logs" / f"dino_{name}.log"
        cmd = [
            sys.executable, "utils/compute_dino_metrics.py",
            "--eval_dir", str(out_dir),
            "--test_dir", str(TEST_DIR),
            "--cache_dir", str(HF_CACHE),
            "--batch_size", "8",
            "--max_refs_per_style", "30",
        ]
        rc = run_cmd(cmd, dino_log)
        t_dino = time.time() - t1
        print(f"  DINO eval: rc={rc}, time={t_dino:.1f}s", flush=True)

        if rc != 0:
            print(f"  ERROR: DINO eval failed for scale={scale}", flush=True)
            results.append({"scale": scale, "status": "dino_failed", "rc": rc})
            continue

        # Step 3: Read results
        dino_summary = out_dir / "dino_summary.json"
        if dino_summary.exists():
            summary = json.loads(dino_summary.read_text())
            result = {
                "scale": scale,
                "status": "ok",
                "infer_time_s": t_infer,
                "dino_time_s": t_dino,
                "dino_s": summary.get("all_dino_s"),
                "dino_c": summary.get("all_dino_c"),
                "dino_structure": summary.get("all_dino_structure"),
                "clip_s": summary.get("all_clip_s"),
                "lpips": summary.get("all_lpips"),
                "off_dino_s": summary.get("off_dino_s"),
                "off_dino_c": summary.get("off_dino_c"),
            }
            results.append(result)
            print(f"  RESULT: DINO-S={result['dino_s']:.4f}, DINO-C={result['dino_c']:.4f}, CLIP-S={result['clip_s']:.4f}, LPIPS={result['lpips']:.4f}", flush=True)
        else:
            results.append({"scale": scale, "status": "no_summary"})
            print(f"  WARNING: no dino_summary.json found", flush=True)

        # Save intermediate results
        (OUTPUT_ROOT / "_results.json").write_text(json.dumps(results, indent=2))

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"{'Scale':<8} {'DINO-S':<10} {'DINO-C':<10} {'CLIP-S':<10} {'LPIPS':<10} {'Status'}")
    for r in results:
        if r["status"] == "ok":
            print(f"{r['scale']:<8} {r['dino_s']:<10.4f} {r['dino_c']:<10.4f} {r['clip_s']:<10.4f} {r['lpips']:<10.4f} ok")
        else:
            print(f"{r['scale']:<8} {'-':<10} {'-':<10} {'-':<10} {'-':<10} {r['status']}")

    (OUTPUT_ROOT / "_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved to: {OUTPUT_ROOT / '_results.json'}")

if __name__ == "__main__":
    main()
