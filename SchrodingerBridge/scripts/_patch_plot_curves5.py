"""Fix WEAVE labels: red back to original position, black to right side.
Also fix WEAVE trajectory dot size to match SaMam style.

Run: python _patch_plot_curves5.py
Target: G:/GitHub/Latent_Style/WEAVE/aaai2027_v4/plot_page1_summary.py
"""
from pathlib import Path

TARGET = Path(r"G:\GitHub\Latent_Style\WEAVE\aaai2027_v4\plot_page1_summary.py")
src = TARGET.read_text(encoding="utf-8")

# ---------------------------------------------------------------------------
# 1. Move WEAVE black label to right of point (was at top: (0,7) center/bottom).
# ---------------------------------------------------------------------------
OLD_LABEL_POS = '    "WEAVE-m": {"xytext": (0, 7), "ha": "center", "va": "bottom", "arrow": False},'
NEW_LABEL_POS = '    "WEAVE-m": {"xytext": (14, 0), "ha": "left", "va": "center", "arrow": False},'

assert OLD_LABEL_POS in src, "WEAVE-m LABEL_POS not found"
src = src.replace(OLD_LABEL_POS, NEW_LABEL_POS)

# ---------------------------------------------------------------------------
# 2. Move red label back to original position (above, with arrow).
# ---------------------------------------------------------------------------
OLD_RED_LABEL = '''    t11 = next(p for p in ALL_POINTS if p["name"] == "WEAVE-m")
    ax.annotate(
        "1.4 min, RTX 3060",
        xy=(t11["x"], t11["avg"]),
        xytext=(14, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=13.5,
        color="#7F1F10",
        bbox={
            "boxstyle": "round,pad=0.30",
            "facecolor": "white",
            "edgecolor": "#E1C2BC",
            "linewidth": 0.8,
            "alpha": 0.96,
        },
        arrowprops=None,
        zorder=7,
    )'''

NEW_RED_LABEL = '''    t11 = next(p for p in ALL_POINTS if p["name"] == "WEAVE-m")
    ax.annotate(
        "1.4 min, RTX 3060",
        xy=(t11["x"], t11["avg"]),
        xytext=(58, 38),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=14.5,
        color="#7F1F10",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#E1C2BC",
            "linewidth": 0.7,
            "alpha": 0.96,
        },
        arrowprops={"arrowstyle": "-", "lw": 0.95, "color": "#7F1F10"},
        zorder=7,
    )'''

assert OLD_RED_LABEL in src, "Red label block not found"
src = src.replace(OLD_RED_LABEL, NEW_RED_LABEL)

# ---------------------------------------------------------------------------
# 3. Fix WEAVE trajectory dot size to match SaMam style (s=5, alpha=0.30).
# ---------------------------------------------------------------------------
OLD_WEAVE_TRAJ = '''        plot_curve_trajectory(ax, weave_curve,
                              {"dino_s": weave_pt["dino_s"], "clip_s": weave_pt["clip_s"], "lpips": weave_pt["lpips"]},
                              color="#D6452F",
                              lw=0.9,
                              alpha_line=0.35,
                              s=7,
                              alpha_dots=0.40,
                              zorder=2.5,
                              glow=True)'''

NEW_WEAVE_TRAJ = '''        plot_curve_trajectory(ax, weave_curve,
                              {"dino_s": weave_pt["dino_s"], "clip_s": weave_pt["clip_s"], "lpips": weave_pt["lpips"]},
                              color="#D6452F",
                              lw=0.7,
                              alpha_line=0.28,
                              s=5,
                              alpha_dots=0.30,
                              zorder=2.5,
                              glow=True)'''

assert OLD_WEAVE_TRAJ in src, "WEAVE trajectory block not found"
src = src.replace(OLD_WEAVE_TRAJ, NEW_WEAVE_TRAJ)

TARGET.write_text(src, encoding="utf-8")
print(f"Patched: {TARGET}")
print("  - WEAVE black label moved to right of point (14, 0)")
print("  - WEAVE red label Back to original position (58, 38) above")
print("  - WEAVE trajectory dots: s=5, alpha=0.30 (same as SaMam)")
