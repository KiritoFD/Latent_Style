"""Compute SaMam 256 mean CLIP-S and LPIPS (excluding identity pairs)."""
import json
from pathlib import Path

p = Path(r"I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_v2\eval\samam\summary.json")
data = json.loads(p.read_text(encoding="utf-8"))

mb = data["matrix_breakdown"]
styles = list(mb.keys())
print(f"Styles: {styles}")

clip_s_list = []
lpips_list = []
for src in styles:
    for tgt in styles:
        if src == tgt:
            continue
        m = mb[src][tgt]
        clip_s_list.append(m["clip_style"])
        lpips_list.append(m["content_lpips"])

print(f"\nSaMam 256 (excluding identity, N={len(clip_s_list)}):")
print(f"  CLIP-S = {sum(clip_s_list)/len(clip_s_list):.4f}")
print(f"  LPIPS  = {sum(lpips_list)/len(lpips_list):.4f}")

# Also with identity
all_clip = []
all_lpips = []
for src in styles:
    for tgt in styles:
        m = mb[src][tgt]
        all_clip.append(m["clip_style"])
        all_lpips.append(m["content_lpips"])
print(f"\nWith identity (N={len(all_clip)}):")
print(f"  CLIP-S = {sum(all_clip)/len(all_clip):.4f}")
print(f"  LPIPS  = {sum(all_lpips)/len(all_lpips):.4f}")
