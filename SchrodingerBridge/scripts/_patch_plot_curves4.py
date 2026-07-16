"""Revert SaMam below-IDT removal, remove SaMam black label instead.

Run: python _patch_plot_curves4.py
Target: G:/GitHub/Latent_Style/WEAVE/aaai2027_v4/plot_page1_summary.py
"""
from pathlib import Path

TARGET = Path(r"G:\GitHub\Latent_Style\WEAVE\aaai2027_v4\plot_page1_summary.py")
src = TARGET.read_text(encoding="utf-8")

# ---------------------------------------------------------------------------
# 1. Restore "SaMam below IDT" red annotation (was wrongly deleted).
#    Insert it back before the WEAVE label block.
# ---------------------------------------------------------------------------
OLD_WEAVE_LABEL = '''    t11 = next(p for p in ALL_POINTS if p["name"] == "WEAVE-m")
    ax.annotate(
        "1.4 min, RTX 3060",'''

NEW_WEAVE_LABEL = '''    samam = next(p for p in ALL_POINTS if p["name"] == "SaMam")
    ax.annotate(
        "SaMam\\nbelow IDT",
        xy=(samam["x"], samam["avg"]),
        xytext=(34, -2),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=8.9,
        color="#7A1E14",
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "white",
            "edgecolor": "#D9B7B1",
            "linewidth": 0.7,
            "alpha": 0.96,
        },
        arrowprops={"arrowstyle": "-", "lw": 0.95, "color": "#7A1E14"},
        zorder=7,
    )

    t11 = next(p for p in ALL_POINTS if p["name"] == "WEAVE-m")
    ax.annotate(
        "1.4 min, RTX 3060",'''

assert OLD_WEAVE_LABEL in src, "WEAVE label block not found"
src = src.replace(OLD_WEAVE_LABEL, NEW_WEAVE_LABEL)

# ---------------------------------------------------------------------------
# 2. Remove SaMam black label: set label=False in BASELINES.
#    This prevents annotate_point from drawing the black "SaMam" text.
# ---------------------------------------------------------------------------
OLD_SAMAM_BASELINE = 'point("SaMam", 0.475826, 0.590472, 0.320912, "trained", label=True, train_min=436.0),  # step 7000 (DINO-S peak)'
NEW_SAMAM_BASELINE = 'point("SaMam", 0.475826, 0.590472, 0.320912, "trained", label=False, train_min=436.0),  # step 7000 (DINO-S peak), black label removed'

assert OLD_SAMAM_BASELINE in src, "SaMam baseline line not found"
src = src.replace(OLD_SAMAM_BASELINE, NEW_SAMAM_BASELINE)

TARGET.write_text(src, encoding="utf-8")
print(f"Patched: {TARGET}")
print("  - Restored 'SaMam below IDT' red annotation")
print("  - Removed SaMam black label (label=False)")
