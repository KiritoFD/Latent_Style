"""Compute CUT 256 mean CLIP-S and LPIPS from final_works/CUT/summary.json.

The summary.json contains 5x5 pairwise metrics: per (src_style, tgt_style)
- clip_style: cos(CLIP(gen), CLIP(target prototype))
- content_lpips: LPIPS(gen, source content)

We compute the mean over all 5x5=25 pairs (excluding identity pairs since
they correspond to IDT, not real transfer). This matches the table's protocol.
"""
import json
from pathlib import Path

summary_path = Path(r"I:\Github\Latent_Style\final_works\CUT\summary.json")
data = json.loads(summary_path.read_text(encoding="utf-8"))

pair_metrics = data["matrix_breakdown"]
styles = ["Hayao", "cezanne", "monet", "photo", "vangogh"]

clip_s_list = []
lpips_list = []
for src in styles:
    for tgt in styles:
        if src == tgt:
            continue  # skip identity pairs (IDT)
        m = pair_metrics.get(src, {}).get(tgt)
        if m is None:
            continue
        clip_s_list.append(m["clip_style"])
        lpips_list.append(m["content_lpips"])

mean_clip_s = sum(clip_s_list) / len(clip_s_list)
mean_lpips = sum(lpips_list) / len(lpips_list)

print(f"CUT 256 (excluding identity pairs):")
print(f"  N_pairs = {len(clip_s_list)}")
print(f"  CLIP-S (mean) = {mean_clip_s:.4f}")
print(f"  LPIPS  (mean) = {mean_lpips:.4f}")

# Also include identity for reference
all_clip = []
all_lpips = []
for src in styles:
    for tgt in styles:
        m = pair_metrics.get(src, {}).get(tgt)
        if m is None:
            continue
        all_clip.append(m["clip_style"])
        all_lpips.append(m["content_lpips"])
print(f"\nWith identity pairs (5x5=25):")
print(f"  CLIP-S = {sum(all_clip)/len(all_clip):.4f}")
print(f"  LPIPS  = {sum(all_lpips)/len(all_lpips):.4f}")
