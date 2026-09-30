"""Patch plot_page1_summary.py: move SaMam scatter to step 7000, add WEAVE trajectory.

Run: python _patch_plot_curves2.py
Target: G:/GitHub/Latent_Style/WEAVE/aaai2027_v4/plot_page1_summary.py
"""
from pathlib import Path

TARGET = Path(r"G:\GitHub\Latent_Style\WEAVE\aaai2027_v4\plot_page1_summary.py")
src = TARGET.read_text(encoding="utf-8")

# ---------------------------------------------------------------------------
# 1. Move SaMam scatter to step 7000 (DINO-S peak, faithful to curve).
#    Main table DINO-S=0.4771 matches step 7000 (0.4758) — not step 20000 (0.4154).
#    Using step 7000's full triple makes the scatter land exactly on the curve.
# ---------------------------------------------------------------------------
OLD_SAMAM_POINT = 'point("SaMam", 0.4771, 0.5816, 0.2434, "trained", label=True, train_min=436.0),'
NEW_SAMAM_POINT = 'point("SaMam", 0.475826, 0.590472, 0.320912, "trained", label=True, train_min=436.0),  # step 7000 (DINO-S peak)'

assert OLD_SAMAM_POINT in src, "SaMam point line not found"
src = src.replace(OLD_SAMAM_POINT, NEW_SAMAM_POINT)

# ---------------------------------------------------------------------------
# 2. Add WEAVE convergence trajectory (red, faithful, with glow).
#    Insert after SaMST trajectory block, before the label-annotation loop.
# ---------------------------------------------------------------------------
OLD_TRAJ_TAIL = '''    samst_pt = next(p for p in ALL_POINTS if p["name"] == "SaMST")
    samst_curve = load_samst_curve()
    if samst_curve:
        plot_curve_trajectory(ax, samst_curve,
                              {"dino_s": samst_pt["dino_s"], "clip_s": samst_pt["clip_s"], "lpips": samst_pt["lpips"]},
                              color="#3B82C4",
                              lw=0.7,
                              alpha_line=0.28,
                              s=5,
                              alpha_dots=0.30,
                              zorder=2.0)

    for p in ALL_POINTS:
        if p["label"]:
            annotate_point(ax, p)'''

NEW_TRAJ_TAIL = '''    samst_pt = next(p for p in ALL_POINTS if p["name"] == "SaMST")
    samst_curve = load_samst_curve()
    if samst_curve:
        plot_curve_trajectory(ax, samst_curve,
                              {"dino_s": samst_pt["dino_s"], "clip_s": samst_pt["clip_s"], "lpips": samst_pt["lpips"]},
                              color="#3B82C4",
                              lw=0.7,
                              alpha_line=0.28,
                              s=5,
                              alpha_dots=0.30,
                              zorder=2.0)

    # --- WEAVE convergence trajectory (red, faithful, with glow) ---
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
                              glow=True)

    for p in ALL_POINTS:
        if p["label"]:
            annotate_point(ax, p)'''

assert OLD_TRAJ_TAIL in src, "SaMST trajectory tail not found"
src = src.replace(OLD_TRAJ_TAIL, NEW_TRAJ_TAIL)

TARGET.write_text(src, encoding="utf-8")
print(f"Patched: {TARGET}")
print("  - SaMam scatter moved to step 7000 (DINO-S=0.4758, CLIP-S=0.5905, LPIPS=0.3209)")
print("  - WEAVE trajectory added (4 epochs, red #D6452F, glow=True)")
