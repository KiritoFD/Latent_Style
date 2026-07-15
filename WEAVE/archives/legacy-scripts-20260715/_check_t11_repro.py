"""Check T11 repro metrics and find historical 0.734/0.29 point."""
import json
from pathlib import Path

# Check T11 repro
t11_path = Path(r"I:\Github\Latent_Style\SchrodingerBridge\exp\630_local_t11_repro_i\full_eval\epoch_0005\summary.json")
if t11_path.exists():
    with open(t11_path, "r", encoding="utf-8") as f:
        s = json.load(f)
    overview = s.get("analysis", {}).get("all_pairs_overview", {})
    clip = overview.get("clip_style", 0)
    lpips_raw = overview.get("content_lpips", 1)
    print(f"T11 repro_i: clip_style={clip:.4f}, content_lpips={lpips_raw:.4f}, 1-LPIPS={1-lpips_raw:.4f}")
else:
    print(f"T11 repro summary not found: {t11_path}")

# Check for DINO results
dino_path = Path(r"I:\Github\Latent_Style\SchrodingerBridge\exp\_dino_results\630_local_t11_repro_i.json")
if dino_path.exists():
    with open(dino_path, "r", encoding="utf-8") as f:
        d = json.load(f)
    print(f"T11 repro DINO: sty={d.get('dino_style',0):.4f}, con={d.get('dino_content',0):.4f}, str={d.get('dino_structure',0):.4f}")
else:
    print(f"T11 repro DINO not found")

# List all exp dirs to find 0.734 clip
import os
exp_root = Path(r"I:\Github\Latent_Style\SchrodingerBridge\exp")
print("\n=== All experiments with summary.json ===")
for exp_dir in sorted(exp_root.iterdir()):
    if not exp_dir.is_dir():
        continue
    # Find any summary.json
    for summary_path in exp_dir.rglob("summary.json"):
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                s = json.load(f)
            overview = s.get("analysis", {}).get("all_pairs_overview", {})
            clip = overview.get("clip_style", 0)
            lpips_raw = overview.get("content_lpips", 1)
            if clip > 0.72:  # Only show high-clip experiments
                rel_path = summary_path.relative_to(exp_root)
                print(f"  {rel_path}: clip={clip:.4f}, 1-LPIPS={1-lpips_raw:.4f}")
        except Exception as e:
            pass
        break  # Only first summary per exp
