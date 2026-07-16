"""Quick DINO-S sweep across epochs for repro_brk_a_15ep.
Run on remote RTX 3060. Only generates images, then runs DINO compute.
"""
import subprocess, sys, os, json, shutil
from pathlib import Path

WEAVE_ROOT = Path(r"I:\Github\Latent_Style\WEAVE")
os.chdir(WEAVE_ROOT)

EPOCHS = [3, 5, 6, 7, 8, 9, 10, 12, 15]
BASE_DIR = "exp/repro_dino_epoch4"  # reuse generated images from epoch 4

results = {}

for ep in EPOCHS:
    CKPT = f"runs/submission/repro_brk_a_15ep/epoch_{ep:04d}.pt"
    OUT_DIR = f"exp/dino_sweep/ep{ep:04d}"
    
    # Skip if already done
    summary_path = Path(OUT_DIR) / "dino_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        dino_s = summary.get("all_dino_s", "N/A")
        clip_s = summary.get("all_clip_s", "N/A")
        results[ep] = {"dino_s": dino_s, "clip_s": clip_s}
        print(f"Epoch {ep}: DINO-S={dino_s:.4f}, CLIP-S={clip_s:.4f} (cached)")
        continue
    
    print(f"\n{'='*60}")
    print(f"Evaluating epoch {ep}")
    print(f"{'='*60}")
    
    # Step 1: Generate images
    cmd = [
        sys.executable, "-u", "utils/run_evaluation.py",
        "--checkpoint", CKPT,
        "--config_override", "inference.json",
        "--output", OUT_DIR,
        "--test_dir", "data/test",
        "--batch_size", "8",
        "--vae_decode_batch_size", "8",
        "--num_steps", "8",
        "--max_ref_cache", "16",
        "--max_ref_compare", "16",
    ]
    print("GEN:", " ".join(cmd))
    r = subprocess.run(cmd, cwd=str(WEAVE_ROOT))
    if r.returncode != 0:
        print(f"Generation failed for epoch {ep}")
        continue
    
    # Step 2: DINO
    dino_cmd = [
        sys.executable, "-u", "utils/compute_dino_metrics.py",
        "--eval_dir", OUT_DIR,
        "--test_dir", "data/test",
        "--batch_size", "4",
        "--max_refs_per_style", "30",
        "--allow_network",
    ]
    print("DINO:", " ".join(dino_cmd))
    r = subprocess.run(dino_cmd, cwd=str(WEAVE_ROOT))
    if r.returncode != 0:
        print(f"DINO failed for epoch {ep}")
        continue
    
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        dino_s = summary.get("all_dino_s", "N/A")
        clip_s = summary.get("all_clip_s", "N/A")
        results[ep] = {"dino_s": dino_s, "clip_s": clip_s}
        print(f"Epoch {ep}: DINO-S={dino_s:.4f}, CLIP-S={clip_s:.4f}")

# Print final summary
print("\n" + "=" * 60)
print("FINAL SUMMARY (sorted by DINO-S)")
print("=" * 60)
for ep in sorted(results.keys(), key=lambda e: results[e]["dino_s"] if isinstance(results[e]["dino_s"], (int, float)) else 0, reverse=True):
    print(f"  epoch_{ep:04d}: DINO-S={results[ep]['dino_s']:.4f}, CLIP-S={results[ep]['clip_s']:.4f}")