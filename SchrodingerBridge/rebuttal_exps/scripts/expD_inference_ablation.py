"""Exp D: Inference ablations on current production checkpoint.

D0: Full current WEAVE (reference, already have results)
D1: AdaIN scale = 0 (disable stepwise statistics injection)
D2: Disable oriented target-HF residual route

D1 uses inference.json override (endpoint_adain_scale=0).
D2 uses config override (model.target_latent_hf_subband_fusion_enabled=false).
   inference.py falls back to strict=False when HF route weights are unexpected.

Both use the SAME production checkpoint (epoch_0004.pt).
"""
import hashlib, json, os, sys, subprocess, time
from pathlib import Path

WEAVE_ROOT = Path(r"I:\Github\Latent_Style\WEAVE")
os.chdir(WEAVE_ROOT)

PROD_CKPT = WEAVE_ROOT / "runs" / "submission" / "hf_oriented_internal_early_stop" / "epoch_0004.pt"
TEST_DIR = "data/test"
HF_CACHE = "exp/eval_cache/hf"
OUTPUT_DIR = WEAVE_ROOT / "exp" / "rebuttal" / "expD_inference_ablation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def read_dino(eval_dir):
    p = Path(eval_dir) / "dino_summary.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    return {
        "dino_s": d.get("all_dino_s"),
        "dino_c": d.get("all_dino_c"),
        "clip_s": d.get("all_clip_s"),
        "lpips": d.get("all_lpips"),
    }


def run_eval(ckpt_path, config_override, out_dir, tag):
    out_dir_str = str(out_dir).replace("\\", "/").replace("I:/Github/Latent_Style/WEAVE/", "")

    existing = read_dino(out_dir)
    if existing:
        print(f"  {tag}: already done, DINO-S={existing['dino_s']:.4f}")
        return existing

    print(f"  {tag}: generating images...", flush=True)
    gen_cmd = [
        sys.executable, "-u", "utils/run_evaluation.py",
        "--checkpoint", str(ckpt_path),
        "--config_override", config_override,
        "--output", out_dir_str,
        "--test_dir", TEST_DIR,
        "--batch_size", "2",
        "--ref_feature_batch_size", "2",
        "--vae_decode_batch_size", "16",
        "--force_regen",
    ]
    t0 = time.time()
    proc = subprocess.run(gen_cmd, cwd=str(WEAVE_ROOT))
    if proc.returncode != 0:
        print(f"  {tag}: ERROR generation failed")
        return None
    print(f"  {tag}: generation done in {time.time()-t0:.0f}s", flush=True)

    print(f"  {tag}: computing DINO...", flush=True)
    dino_cmd = [
        sys.executable, "-u", "utils/compute_dino_metrics.py",
        "--eval_dir", out_dir_str,
        "--test_dir", TEST_DIR,
        "--cache_dir", HF_CACHE,
    ]
    subprocess.run(dino_cmd, cwd=str(WEAVE_ROOT))
    return read_dino(out_dir)


def compare_output_hashes(reference_dir, candidate_dir, limit=20):
    """Prove that an inference override changed the generated packet."""
    reference_images = Path(reference_dir) / "images"
    candidate_images = Path(candidate_dir) / "images"
    pairs = []
    for candidate in sorted(candidate_images.glob("*.png")):
        reference = reference_images / candidate.name
        if reference.exists():
            pairs.append((reference, candidate))
        if len(pairs) >= limit:
            break
    compared = []
    for reference, candidate in pairs:
        ref_hash = hashlib.sha256(reference.read_bytes()).hexdigest()
        cand_hash = hashlib.sha256(candidate.read_bytes()).hexdigest()
        compared.append({"image": candidate.name, "same_sha256": ref_hash == cand_hash})
    return {
        "pairs_checked": len(compared),
        "pairs_different": sum(not row["same_sha256"] for row in compared),
        "all_different": bool(compared) and all(not row["same_sha256"] for row in compared),
        "details": compared,
    }


def main():
    print("=" * 60)
    print("Exp D: Inference ablations on production checkpoint")
    print("=" * 60)

    results = {}

    # D0: Full WEAVE (from existing repro_weave_d5)
    print("\n--- D0: Full WEAVE (reference) ---")
    d0 = read_dino(WEAVE_ROOT / "exp" / "repro_weave_d5")
    if d0:
        results["D0_full"] = d0
        print(f"  D0: DINO-S={d0['dino_s']:.4f}, DINO-C={d0['dino_c']:.4f}, CLIP-S={d0['clip_s']:.4f}, LPIPS={d0['lpips']:.4f}")

    # D1: AdaIN scale = 0
    print("\n--- D1: AdaIN scale = 0 ---")
    d1_override = OUTPUT_DIR / "d1_adain0_inference.json"
    base_inf = json.loads((WEAVE_ROOT / "inference.json").read_text(encoding="utf-8"))
    base_inf.setdefault("model", {})["endpoint_adain_scale"] = 0.0
    d1_override.write_text(json.dumps(base_inf, indent=2), encoding="utf-8")
    d1_rel = str(d1_override.relative_to(WEAVE_ROOT)).replace("\\", "/")
    d1_dir = OUTPUT_DIR / "D1_adain0_corrected"
    results["D1_adain0_corrected"] = run_eval(PROD_CKPT, d1_rel, d1_dir, "D1")
    if results["D1_adain0_corrected"]:
        r = results["D1_adain0_corrected"]
        print(f"  D1: DINO-S={r['dino_s']:.4f}, DINO-C={r['dino_c']:.4f}, CLIP-S={r['clip_s']:.4f}, LPIPS={r['lpips']:.4f}")
        results["D1_adain0_corrected_hash_check"] = compare_output_hashes(
            WEAVE_ROOT / "exp" / "repro_weave_d5", d1_dir
        )
        proof = results["D1_adain0_corrected_hash_check"]
        print(f"  D1 hash check: {proof['pairs_different']}/{proof['pairs_checked']} checked images differ")

    # D2: Disable oriented target-HF route
    print("\n--- D2: Disable oriented target-HF route ---")
    d2_override = OUTPUT_DIR / "d2_no_hf_route.json"
    d2_config = {
        "model": {
            "target_latent_hf_subband_fusion_enabled": False,
            "target_latent_hf_subband_head_fusion_enabled": False,
        },
        "inference": json.loads((WEAVE_ROOT / "inference.json").read_text(encoding="utf-8")),
    }
    d2_override.write_text(json.dumps(d2_config, indent=2), encoding="utf-8")
    d2_rel = str(d2_override.relative_to(WEAVE_ROOT)).replace("\\", "/")
    results["D2_no_hf_route"] = run_eval(PROD_CKPT, d2_rel, OUTPUT_DIR / "D2_no_hf_route", "D2")
    if results["D2_no_hf_route"]:
        r = results["D2_no_hf_route"]
        print(f"  D2: DINO-S={r['dino_s']:.4f}, DINO-C={r['dino_c']:.4f}, CLIP-S={r['clip_s']:.4f}, LPIPS={r['lpips']:.4f}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY: Inference Ablation")
    print(f"{'='*60}")
    print(f"  {'Variant':<20} {'DINO-S':<12} {'DINO-C':<12} {'CLIP-S':<10} {'LPIPS':<10}")
    for tag, r in results.items():
        if r and r.get("dino_s"):
            print(f"  {tag:<20} {r['dino_s']:<12.4f} {r['dino_c']:<12.4f} {r['clip_s']:<10.4f} {r['lpips']:<10.4f}")

    out_path = OUTPUT_DIR / "results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")
    print("EXPD_EXIT=0")


if __name__ == "__main__":
    main()
