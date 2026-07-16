"""Remove CUT point, update SaMam red label to 'Samam CVPR2025'.

Run: python _patch_plot_curves6.py
"""
from pathlib import Path

TARGET = Path(r"G:\GitHub\Latent_Style\WEAVE\aaai2027_v4\plot_page1_summary.py")
src = TARGET.read_text(encoding="utf-8")

# 1. Remove CUT point from BASELINES
OLD_CUT = '    point("CUT", 0.4709, 0.7137, 0.3743, "trained", label=True, train_min=322.6),\n'
assert OLD_CUT in src, "CUT point not found"
src = src.replace(OLD_CUT, '')

# 2. Update SaMam red label text
OLD_TEXT = '        "SaMam\\nbelow IDT",'
NEW_TEXT = '        "SaMam\\nCVPR 2025",'
assert OLD_TEXT in src, "SaMam label text not found"
src = src.replace(OLD_TEXT, NEW_TEXT)

# 3. Remove CUT from LABEL_POS
OLD_CUT_LABEL = '    "CUT": {"xytext": (-6, 10), "ha": "right", "va": "bottom", "arrow": False},\n'
assert OLD_CUT_LABEL in src, "CUT LABEL_POS not found"
src = src.replace(OLD_CUT_LABEL, '')

TARGET.write_text(src, encoding="utf-8")
print("Done: CUT removed, SaMam label -> 'SaMam CVPR 2025'")
