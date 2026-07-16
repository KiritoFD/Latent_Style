"""Update SaMST scatter to epoch 15 curve point, label below.

Run: python _patch_plot_curves9.py
Target: G:/GitHub/Latent_Style/WEAVE/aaai2027_v4/plot_page1_summary.py
"""
from pathlib import Path

TARGET = Path(r"G:\GitHub\Latent_Style\WEAVE\aaai2027_v4\plot_page1_summary.py")
src = TARGET.read_text(encoding="utf-8")

# 1. Update SaMST point to epoch 15 values: DINO-S=0.4404, CLIP-S=0.7247, LPIPS=0.6255
OLD_SAMST = '    point("SaMST", 0.2710, 0.6183, 0.7490, "trained", label=True, train_min=39.5),'
NEW_SAMST = '    point("SaMST", 0.4404, 0.7247, 0.6255, "trained", label=True, train_min=39.5),'

assert OLD_SAMST in src, "SaMST point not found"
src = src.replace(OLD_SAMST, NEW_SAMST)

# 2. Update SaMST label position to below the point
OLD_SAMST_POS = '    "SaMST": {"xytext": (12, 10), "ha": "left", "va": "bottom", "arrow": False},'
NEW_SAMST_POS = '    "SaMST": {"xytext": (0, -10), "ha": "center", "va": "top", "arrow": False},'

assert OLD_SAMST_POS in src, "SaMST label pos not found"
src = src.replace(OLD_SAMST_POS, NEW_SAMST_POS)

TARGET.write_text(src, encoding="utf-8")
print(f"Patched: {TARGET}")
print("  - SaMST scatter updated to epoch 15: DINO-S=0.4404, CLIP-S=0.7247, LPIPS=0.6255")
print("  - SaMST label moved to below point (center, top)")
