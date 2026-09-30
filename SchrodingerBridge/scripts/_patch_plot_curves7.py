"""Add Z-STAR and StyleID points, add AdaIN label.

Run: python _patch_plot_curves7.py
Target: G:/GitHub/Latent_Style/WEAVE/aaai2027_v4/plot_page1_summary.py
"""
from pathlib import Path

TARGET = Path(r"G:\GitHub\Latent_Style\WEAVE\aaai2027_v4\plot_page1_summary.py")
src = TARGET.read_text(encoding="utf-8")

# ---------------------------------------------------------------------------
# 1. Add Z-STAR and StyleID to BASELINES (D5-512 data from Table 1).
#    Insert after Latent-WCT line.
# ---------------------------------------------------------------------------
OLD_LATENT_WCT = '    point("Latent-WCT", 0.3620, 0.6730, 0.4410, "classical"),'
NEW_LINES = '''    point("Latent-WCT", 0.3620, 0.6730, 0.4410, "classical"),
    point("Z-STAR", 0.4490, 0.7840, 0.3470, "training_free", label=True),
    point("StyleID", 0.5480, 0.8220, 0.5520, "training_free", label=True),'''

assert OLD_LATENT_WCT in src, "Latent-WCT line not found"
src = src.replace(OLD_LATENT_WCT, NEW_LINES)

# ---------------------------------------------------------------------------
# 2. Add LABEL_POS for Z-STAR and StyleID.
# ---------------------------------------------------------------------------
OLD_LABEL_POS_TAIL = '''    "WEAVE-q": {"xytext": (11, -8), "ha": "left", "va": "top", "arrow": False},
    "WEAVE-m": {"xytext": (14, 0), "ha": "left", "va": "center", "arrow": False},
}'''

NEW_LABEL_POS_TAIL = '''    "WEAVE-q": {"xytext": (11, -8), "ha": "left", "va": "top", "arrow": False},
    "WEAVE-m": {"xytext": (14, 0), "ha": "left", "va": "center", "arrow": False},
    "Z-STAR": {"xytext": (-6, 10), "ha": "right", "va": "bottom", "arrow": False},
    "StyleID": {"xytext": (12, 0), "ha": "left", "va": "center", "arrow": False},
}'''

assert OLD_LABEL_POS_TAIL in src, "LABEL_POS tail not found"
src = src.replace(OLD_LABEL_POS_TAIL, NEW_LABEL_POS_TAIL)

# ---------------------------------------------------------------------------
# 3. Add AdaIN label (it should have one — classical method near IDT floor).
# ---------------------------------------------------------------------------
OLD_ADAIN = 'point("AdaIN", 0.3362, 0.6679, 0.7425, "classical"),'
NEW_ADAIN = 'point("AdaIN", 0.3362, 0.6679, 0.7425, "classical", label=True),'

assert OLD_ADAIN in src, "AdaIN point not found"
src = src.replace(OLD_ADAIN, NEW_ADAIN)

# Add AdaIN label position
OLD_ADAIN_LABEL = '''    "Identity": {"xytext": (0, -8), "ha": "center", "va": "top", "arrow": False},
    "WCT": {"xytext": (14, 0), "ha": "left", "va": "center", "arrow": False},'''

NEW_ADAIN_LABEL = '''    "Identity": {"xytext": (0, -8), "ha": "center", "va": "top", "arrow": False},
    "AdaIN": {"xytext": (12, 0), "ha": "left", "va": "center", "arrow": False},
    "WCT": {"xytext": (14, 0), "ha": "left", "va": "center", "arrow": False},'''

assert OLD_ADAIN_LABEL in src, "AdaIN label pos not found"
src = src.replace(OLD_ADAIN_LABEL, NEW_ADAIN_LABEL)

TARGET.write_text(src, encoding="utf-8")
print(f"Patched: {TARGET}")
print("  - Added Z-STAR (0.449, 0.784, LPIPS=0.347) with label")
print("  - Added StyleID (0.548, 0.822, LPIPS=0.552) with label")
print("  - Added AdaIN label")
