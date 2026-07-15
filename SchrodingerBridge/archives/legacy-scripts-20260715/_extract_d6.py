"""Extract D6 complete metrics."""
import json, os
ROOT = r"I:\Github\Latent_Style\SchrodingerBridge"
exp = "d6_style_consist_15ep"

# Summary
summary_path = os.path.join(ROOT, "exp", exp, "full_eval", "epoch_0015", "summary.json")
with open(summary_path) as f:
    d = json.load(f)
ov = d.get("analysis", {}).get("all_pairs_overview", {})
clip_s = ov.get("clip_style", "?")
lpips = ov.get("content_lpips", "?")
one_minus = 1.0 - lpips if isinstance(lpips, (int, float)) else "?"

# DINO
dino_path = os.path.join(ROOT, "exp", "_dino_results", f"{exp}.json")
with open(dino_path) as f:
    dino = json.load(f)
dino_sty = dino.get("dino_style", "?")
dino_con = dino.get("dino_content", "?")
dino_str = dino.get("dino_structure", "?")

print(f"D6 Results: {exp}")
print(f"  CLIP-S:   {clip_s}")
print(f"  1-LPIPS:  {one_minus}")
print(f"  DINO-sty: {dino_sty}")
print(f"  DINO-con: {dino_con}")
print(f"  DINO-str: {dino_str}")

# Compare with baseline
print(f"\nBaseline (hp_simple_swd12_15ep):")
print(f"  CLIP-S:   0.7167")
print(f"  1-LPIPS:  0.7010")
print(f"  DINO-sty: 0.4762")
print(f"  DINO-con: 0.8052")
print(f"  DINO-str: 0.0243")

if isinstance(clip_s, (int, float)):
    print(f"\nDelta vs baseline:")
    print(f"  CLIP-S:   {clip_s - 0.7167:+.4f}")
    print(f"  1-LPIPS:  {one_minus - 0.7010:+.4f}")
    print(f"  DINO-sty: {dino_sty - 0.4762:+.4f}")
    print(f"  DINO-con: {dino_con - 0.8052:+.4f}")
