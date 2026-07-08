"""Collect ablation results - extract CLIP-S and LPIPS from summary.json files."""
import json
import glob
import os

REPO = r"I:\Github\Latent_Style\SchrodingerBridge"
# Include baseline (clean_base_v2) and ablation experiments
patterns = [
    os.path.join(REPO, "exp", "abl_*", "full_eval", "epoch_*", "summary.json"),
    os.path.join(REPO, "exp", "clean_base_v2", "full_eval", "epoch_*", "summary.json"),
    os.path.join(REPO, "exp", "sem_r8_*", "full_eval", "epoch_*", "summary.json"),
]
pattern = patterns[0]  # keep for compatibility
all_paths = []
for pat in patterns:
    all_paths.extend(sorted(glob.glob(pat)))

results = []
for p in all_paths:
    parts = p.replace("\\", "/").split("/")
    exp_name = parts[-4]
    epoch = parts[-2]

    try:
        with open(p) as f:
            data = json.load(f)
    except Exception as e:
        results.append((exp_name, epoch, None, None, None, None, None))
        continue

    # Metrics are in analysis.all_pairs_overview (all 745 pairs) and
    # analysis.style_transfer_ability (cross-style only)
    apo = data.get("analysis", {}).get("all_pairs_overview", {})
    sta = data.get("analysis", {}).get("style_transfer_ability", {})

    # all_pairs metrics (full D5 = 745 pairs)
    clip_s_all = apo.get("clip_style")
    lpips_all = apo.get("content_lpips")
    clip_t_all = apo.get("clip_t")
    artfid_lpips_all = apo.get("art_fid_content_lpips")

    # style_transfer_ability metrics (cross-style only)
    clip_s_xfer = sta.get("clip_style")
    lpips_xfer = sta.get("content_lpips")

    results.append((exp_name, epoch, clip_s_all, lpips_all, clip_s_xfer, lpips_xfer, clip_t_all))

print("=" * 95)
print(f"{'Experiment':<22} {'CLIP-S(all)':<12} {'LPIPS(all)':<12} {'CLIP-S(xfer)':<13} {'LPIPS(xfer)':<12} {'CLIP-T':<10}")
print("=" * 95)
for name, epoch, cs_all, lp_all, cs_xfer, lp_xfer, ct_all in results:
    def fmt(v):
        return f"{v:.4f}" if isinstance(v, (int, float)) else "N/A"
    print(f"{name:<22} {fmt(cs_all):<12} {fmt(lp_all):<12} {fmt(cs_xfer):<13} {fmt(lp_xfer):<12} {fmt(ct_all):<10}")
print("=" * 95)
print(f"Total: {len(results)} experiments")
print("\nNote: 'all' = all 745 pairs (full D5), 'xfer' = cross-style only (style_transfer_ability)")

# Show available keys for first experiment
if all_paths:
    with open(all_paths[0]) as f:
        data = json.load(f)
    apo = data.get("analysis", {}).get("all_pairs_overview", {})
    print(f"\nanalysis.all_pairs_overview keys: {list(apo.keys()) if isinstance(apo, dict) else type(apo)}")
