"""Extract sweep CLIP-S and 1-LPIPS from summary.json files."""
import json, os, glob

ROOT = r"I:\Github\Latent_Style\SchrodingerBridge"
SWEEP_DIR = os.path.join(ROOT, "exp", "hp_simple_swd12_15ep", "full_eval")

# Find all sweep_* dirs
sweep_dirs = sorted(glob.glob(os.path.join(SWEEP_DIR, "sweep_*")))
print(f"Found {len(sweep_dirs)} sweep dirs")
for d in sweep_dirs:
    name = os.path.basename(d)
    summary_path = os.path.join(d, "summary.json")
    if not os.path.exists(summary_path):
        print(f"  {name}: NO summary.json")
        continue
    with open(summary_path) as f:
        data = json.load(f)
    ov = data.get("analysis", {}).get("all_pairs_overview", {})
    clip_s = ov.get("clip_style", "?")
    lpips = ov.get("content_lpips", "?")
    one_minus = 1.0 - lpips if isinstance(lpips, (int, float)) else "?"
    print(f"  {name}: CLIP-S={clip_s}  1-LPIPS={one_minus:.4f}  LPIPS={lpips}")

# Also get DINO results
print("\nDINO results:")
dino_dir = os.path.join(ROOT, "exp", "_dino_results")
for d in sweep_dirs:
    name = os.path.basename(d)
    dino_path = os.path.join(dino_dir, f"{name}.json")
    if not os.path.exists(dino_path):
        print(f"  {name}: NO dino")
        continue
    with open(dino_path) as f:
        dino = json.load(f)
    ds = dino.get("dino_style", "?")
    dc = dino.get("dino_content", "?")
    print(f"  {name}: DINO-sty={ds}  DINO-con={dc}")
