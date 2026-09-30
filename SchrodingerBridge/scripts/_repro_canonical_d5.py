"""Reproduce D5-512/weave images using the canonical checkpoint.
Checkpoint: hf_oriented_internal_early_stop/epoch_0004.pt
"""
import subprocess, sys, os, json, csv
from pathlib import Path

WEAVE_ROOT = Path(r"I:\Github\Latent_Style\WEAVE")
os.chdir(WEAVE_ROOT)

CKPT = "runs/submission/hf_oriented_internal_early_stop/epoch_0004.pt"
OUT_DIR = "exp/repro_weave_d5"
TEST_DIR = "data/test"

# Step 1: Generate images
print("=" * 60)
print("Step 1: Generate images with canonical checkpoint")
print("=" * 60)
cmd = [
    sys.executable, "-u", "utils/run_evaluation.py",
    "--checkpoint", CKPT,
    "--config_override", "inference.json",
    "--output", OUT_DIR,
    "--test_dir", TEST_DIR,
    "--batch_size", "8",
    "--vae_decode_batch_size", "8",
    "--num_steps", "8",
    "--max_ref_cache", "16",
    "--max_ref_compare", "16",
]
print("CMD:", " ".join(cmd))
result = subprocess.run(cmd, cwd=str(WEAVE_ROOT))
if result.returncode != 0:
    print(f"ERROR: generation failed with code {result.returncode}")
    sys.exit(1)

# Step 2: Compute DINO
print("\n" + "=" * 60)
print("Step 2: DINO metrics")
print("=" * 60)
dino_cmd = [
    sys.executable, "-u", "utils/compute_dino_metrics.py",
    "--eval_dir", OUT_DIR,
    "--test_dir", TEST_DIR,
    "--batch_size", "4",
    "--max_refs_per_style", "30",
    "--allow_network",
]
result = subprocess.run(dino_cmd, cwd=str(WEAVE_ROOT))
if result.returncode != 0:
    print(f"ERROR: DINO failed with code {result.returncode}")
    sys.exit(1)

# Step 3: Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
summary_path = Path(OUT_DIR) / "dino_summary.json"
if summary_path.exists():
    summary = json.loads(summary_path.read_text())
    print(json.dumps(summary, indent=2))
    dino_s = summary.get("all_dino_s", "N/A")
    dino_c = summary.get("all_dino_c", "N/A")
    print(f"\n>>> DINO-S={dino_s}, DINO-C={dino_c}")
    if isinstance(dino_s, (int, float)) and dino_s > 0.49:
        print(">>> DINO-S > 0.49 PASS!")
    else:
        print(">>> DINO-S <= 0.49 FAIL")

# Step 4: CLIP-S/LPIPS
metrics_path = Path(OUT_DIR) / "metrics.csv"
if metrics_path.exists():
    with open(metrics_path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    clip_s = sum(float(r["clip_style"]) for r in rows) / len(rows)
    lpips = sum(float(r["content_lpips"]) for r in rows) / len(rows)
    print(f">>> CLIP-S={clip_s:.4f}, LPIPS={lpips:.4f}")