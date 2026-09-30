"""Patch plot_page1_summary.py: extend WEAVE trajectory to scatter, fix labels.

Run: python _patch_plot_curves3.py
Target: G:/GitHub/Latent_Style/WEAVE/aaai2027_v4/plot_page1_summary.py
"""
from pathlib import Path

TARGET = Path(r"G:\GitHub\Latent_Style\WEAVE\aaai2027_v4\plot_page1_summary.py")
src = TARGET.read_text(encoding="utf-8")

# ---------------------------------------------------------------------------
# 1. Extend WEAVE trajectory: append scatter point as final point so the
#    trajectory line ends exactly at the highlighted WEAVE bubble.
# ---------------------------------------------------------------------------
OLD_WEAVE_TRAJ = '''    # --- WEAVE convergence trajectory (red, faithful, with glow) ---
    weave_pt = OURS_FRONTIER[0]
    weave_curve = load_ours_curve(max_epoch=4)
    if weave_curve:
        plot_curve_trajectory(ax, weave_curve,
                              {"dino_s": weave_pt["dino_s"], "clip_s": weave_pt["clip_s"], "lpips": weave_pt["lpips"]},
                              color="#D6452F",
                              lw=0.9,
                              alpha_line=0.35,
                              s=7,
                              alpha_dots=0.40,
                              zorder=2.5,
                              glow=True)'''

NEW_WEAVE_TRAJ = '''    # --- WEAVE convergence trajectory (red, faithful, with glow) ---
    # Trajectory = per-epoch curve + scatter point as final endpoint, so the
    # line lands exactly on the highlighted WEAVE bubble.
    weave_pt = OURS_FRONTIER[0]
    weave_curve = load_ours_curve(max_epoch=4)
    if weave_curve:
        weave_curve = weave_curve + [
            (999, weave_pt["dino_s"], weave_pt["clip_s"], weave_pt["lpips"])
        ]
        plot_curve_trajectory(ax, weave_curve,
                              {"dino_s": weave_pt["dino_s"], "clip_s": weave_pt["clip_s"], "lpips": weave_pt["lpips"]},
                              color="#D6452F",
                              lw=0.9,
                              alpha_line=0.35,
                              s=7,
                              alpha_dots=0.40,
                              zorder=2.5,
                              glow=True)'''

assert OLD_WEAVE_TRAJ in src, "WEAVE trajectory block not found"
src = src.replace(OLD_WEAVE_TRAJ, NEW_WEAVE_TRAJ)

# ---------------------------------------------------------------------------
# 2. Remove the "SaMam below IDT" red annotation entirely.
# ---------------------------------------------------------------------------
OLD_SAMAM_ANNOT = '''    samam = next(p for p in ALL_POINTS if p["name"] == "SaMam")
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

    t11 = next(p for p in ALL_POINTS if p["name"] == "WEAVE-m")'''

NEW_SAMAM_ANNOT = '''    t11 = next(p for p in ALL_POINTS if p["name"] == "WEAVE-m")'''

assert OLD_SAMAM_ANNOT in src, "SaMam annotation block not found"
src = src.replace(OLD_SAMAM_ANNOT, NEW_SAMAM_ANNOT)

# ---------------------------------------------------------------------------
# 3. WEAVE label: "2.07 min" -> "1.4 min", move to right of point.
# ---------------------------------------------------------------------------
OLD_WEAVE_LABEL = '''    t11 = next(p for p in ALL_POINTS if p["name"] == "WEAVE-m")
    ax.annotate(
        "2.07 min, RTX 3060",
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

NEW_WEAVE_LABEL = '''    t11 = next(p for p in ALL_POINTS if p["name"] == "WEAVE-m")
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

assert OLD_WEAVE_LABEL in src, "WEAVE label block not found"
src = src.replace(OLD_WEAVE_LABEL, NEW_WEAVE_LABEL)

TARGET.write_text(src, encoding="utf-8")
print(f"Patched: {TARGET}")
print("  - WEAVE trajectory extended to scatter point (final endpoint)")
print("  - Removed 'SaMam below IDT' annotation")
print("  - WEAVE label: '1.4 min, RTX 3060', moved to right of point (ha=left, va=center)")
