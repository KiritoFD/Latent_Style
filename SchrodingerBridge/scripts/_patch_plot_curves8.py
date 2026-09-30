"""Remove Latent-WCT point.

Run: python _patch_plot_curves8.py
Target: G:/GitHub/Latent_Style/WEAVE/aaai2027_v4/plot_page1_summary.py
"""
from pathlib import Path

TARGET = Path(r"G:\GitHub\Latent_Style\WEAVE\aaai2027_v4\plot_page1_summary.py")
src = TARGET.read_text(encoding="utf-8")

# Remove Latent-WCT from BASELINES
OLD_LATENT_WCT = '    point("Latent-WCT", 0.3620, 0.6730, 0.4410, "classical"),\n'
assert OLD_LATENT_WCT in src, "Latent-WCT line not found"
src = src.replace(OLD_LATENT_WCT, '')

# Remove Latent-WCT from LABEL_POS (if it exists)
OLD_LWCT_LABEL = '    "Latent-WCT": {"xytext": (12, 0), "ha": "left", "va": "center", "arrow": False},\n'
if OLD_LWCT_LABEL in src:
    src = src.replace(OLD_LWCT_LABEL, '')

TARGET.write_text(src, encoding="utf-8")
print(f"Patched: {TARGET}")
print("  - Removed Latent-WCT point")
