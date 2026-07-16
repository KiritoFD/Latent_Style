"""Reproduce DINO metrics for repro_brk_a_15ep/epoch_0004 using canonical protocol.
Run this on the remote RTX 3060 server.
"""
import subprocess, sys, os, json
from pathlib import Path

WEAVE_ROOT = Path(r"I:\Github\Latent_Style\WEAVE")
os.chdir(WEAVE_ROOT)

CKPT = "runs/submission/repro_brk_a_15ep/epoch_0004.pt"
OUT_DIR = "exp/repro_dino_epoch4"
CONFIG_OVERRIDE = "inference.json"
TEST_DIR = "data/test"

# Step 1: Generate images with run_evaluation.py
print("=" * 60)
print("Step 1: Generate images")
print("=" * 60)
cmd = [
    sys.executable, "-u", "utils/run_evaluation.py",
    "--checkpoint", CKPT,
    "--config_override", CONFIG_OVERRIDE,
    "--output_dir", OUT_DIR,
    "--test_dir", TEST_DIR,
    "--batch_size", "8",
    "--vae_batch_size", "8",
    "--no_compile_vae",
]
print("CMD:", " ".join(cmd))
result = subprocess.run(cmd, cwd=str(WEAVE_ROOT))
if result.returncode != 0:
    print(f"ERROR: run_evaluation.py failed with code {result.returncode}")
    sys.exit(1)

# Step 2: Compute DINO metrics
print("\n" + "=" * 60)
print("Step 2: Compute DINO metrics")
print("=" * 60)
dino_cmd = [
    sys.executable, "-u", "utils/compute_dino_metrics.py",
    "--eval_dir", OUT_DIR,
    "--test_dir", TEST_DIR,
    "--batch_size", "4",
    "--max_refs_per_style", "30",
    "--allow_network",
]
print("CMD:", " ".join(dino_cmd))
result = subprocess.run(dino_cmd, cwd=str(WEAVE_ROOT))
if result.returncode != 0:
    print(f"ERROR: compute_dino_metrics.py failed with code {result.returncode}")
    sys.exit(1)

# Step 3: Print summary
print("\n" + "=" * 60)
print("Step 3: Summary")
print("=" * 60)
summary_path = Path(OUT_DIR) / "dino_summary.json"
if summary_path.exists():
    summary = json.loads(summary_path.read_text())
    print(json.dumps(summary, indent=2))
    dino_s = summary.get("all_dino_s", "N/A")
    dino_c = summary.get("all_dino_c", "N/A")
    print(f"\n>>> DINO-S={dino_s}, DINO-C={dino_c}")
    if isinstance(dino_s, (int, float)) and dino_s > 0.49:
        print(">>> DINO-S > 0.49 PASS")
    else:
        print(">>> DINO-S <= 0.49 FAIL")
else:
    print("ERROR: dino_summary.json not found")

# Step 4: Also read metrics.csv
metrics_path = Path(OUT_DIR) / "metrics.csv"
if metrics_path.exists():
    import csv
    with open(metrics_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    clip_s = sum(float(r["clip_style"]) for r in rows) / len(rows)
    lpips = sum(float(r["content_lpips"]) for r in rows) / len(rows)
    print(f">>> CLIP-S={clip_s:.4f}, LPIPS={lpips:.4f}")