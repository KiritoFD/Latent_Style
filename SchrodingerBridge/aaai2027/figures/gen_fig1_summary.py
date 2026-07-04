"""
Generate Figure 1: IDT calibration scatter + convergence curves + efficiency comparison.

Panel (a): Transfer plane with SaMAM convergence curve (below IDT) and LBM Pareto frontier (above IDT).
Panel (b): Training time comparison (log scale).

Data sourced from phase616_live_dashboard.html.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ============================================================================
# Data from dashboard
# ============================================================================
IDT_CLIP_S = 0.6399208252628644

# SaMAM convergence curve: (1-LPIPS, CLIP-S - IDT) — all below IDT
SAMAM_CURVE = [
    (0.4072, -0.0587), (0.4273, -0.0410), (0.4884, -0.0540),
    (0.5107, -0.0380), (0.5299, -0.0407), (0.5596, -0.0476),
    (0.5382, -0.0338), (0.5785, -0.0297), (0.6043, -0.0346),
    (0.5998, -0.0306), (0.6130, -0.0246), (0.6361, -0.0416),
    (0.6627, -0.0342), (0.6442, -0.0404), (0.6647, -0.0381),
    (0.6733, -0.0343), (0.6705, -0.0285), (0.6531, -0.0319),
    (0.6699, -0.0292), (0.6863, -0.0289), (0.6598, -0.0281),
    (0.6651, -0.0232), (0.6496, -0.0226), (0.6570, -0.0274),
    (0.6745, -0.0273), (0.6739, -0.0264), (0.6820, -0.0271),
    (0.6562, -0.0255),
]

# LBM Pareto frontier: (1-LPIPS, CLIP-S - IDT) — above IDT
LBM_PARETO = [
    (0.627, 0.0321),
    (0.502, 0.0651),
    (0.494, 0.0751),
    (0.456, 0.0771),
    (0.419, 0.0871),
    (0.398, 0.0531),
    (0.441, 0.0641),
]

# LBM labeled operating points
LBM_POINTS = [
    {"label": "LBM (kinetic)",    "x": 0.6277, "y": 0.6712 - IDT_CLIP_S, "color": "#2CA02C"},
    {"label": "LBM (balanced)",   "x": 0.5397, "y": 0.7102 - IDT_CLIP_S, "color": "#1F77B4"},
    {"label": "LBM (enhanced)",   "x": 0.3967, "y": 0.7274 - IDT_CLIP_S, "color": "#17BECF"},
    {"label": "LBM (max style)",  "x": 0.3817, "y": 0.7307 - IDT_CLIP_S, "color": "#000080"},
]

# Baseline points
BASELINE_POINTS = [
    {"label": "SaMST e15",   "x": 0.3681, "y": 0.6957 - IDT_CLIP_S, "color": "#FF7F0E", "marker": "s"},
    {"label": "Seedream-4.5", "x": 0.5077, "y": 0.6920 - IDT_CLIP_S, "color": "#9467BD", "marker": "D"},
]

# Training efficiency data
EFFICIENCY_DATA = [
    {"name": "LBM (kinetic)",    "time": 1.2,  "color": "#2CA02C"},
    {"name": "LBM (balanced)",   "time": 10,   "color": "#1F77B4"},
    {"name": "LBM (enhanced)",   "time": 10,   "color": "#17BECF"},
    {"name": "LBM (max style)",  "time": 10,   "color": "#000080"},
    {"name": "Seedream-4.5",     "time": 60,   "color": "#9467BD"},
    {"name": "SaMST",            "time": 348,  "color": "#FF7F0E"},
    {"name": "SaMAM",            "time": 456,  "color": "#D62728"},
]

# ============================================================================
# Figure layout
# ============================================================================
FIG_W = 15.0
FIG_H = 5.5
COLOR_BG = "#ffffff"
COLOR_TEXT = "#1a1a1a"
COLOR_MUTED = "#666666"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 10.5,
    "axes.titlesize": 11.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.20,
    "grid.linewidth": 0.5,
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_W, FIG_H),
                                gridspec_kw={"width_ratios": [1.15, 1.0]})

# ============================================================================
# Panel (a): Transfer plane with convergence curves
# ============================================================================
ax1.set_facecolor("#FAFAFA")

# IDT floor line at y=0 — placed lower in the plot to create visual separation
ax1.axhline(y=0.0, color="#8E63C0", lw=2.0, ls=(0, (8, 4)), zorder=2)
ax1.text(0.73, 0.008, "IDT floor", color="#8E63C0", fontsize=9.5, ha="left",
         fontweight="bold", fontstyle="italic")

# Region shading — make the below-IDT zone more prominent
ax1.axhspan(-0.10, 0.0, color="#FDE8E8", alpha=0.25, zorder=0)
ax1.axhspan(0.0, 0.12, color="#E8FDE8", alpha=0.12, zorder=0)

# SaMAM convergence curve: points connected by line, all below IDT
samam_sorted = sorted(SAMAM_CURVE, key=lambda p: p[0])
samam_x = [p[0] for p in samam_sorted]
samam_y = [p[1] for p in samam_sorted]
ax1.plot(samam_x, samam_y, color="#5D8FBF", lw=2.2, alpha=0.85, zorder=3,
         label="SaMAM convergence")
ax1.scatter(samam_x, samam_y, s=22, c="#5D8FBF", edgecolors="#2c3e50",
            linewidths=0.4, zorder=4, alpha=0.8)

# LBM Pareto frontier: points connected by line, above IDT
lbm_sorted = sorted(LBM_PARETO, key=lambda p: p[0])
lbm_x = [p[0] for p in lbm_sorted]
lbm_y = [p[1] for p in lbm_sorted]
ax1.plot(lbm_x, lbm_y, color="#14B8A6", lw=2.5, alpha=0.85, zorder=3,
         label="LBM Pareto frontier")
ax1.scatter(lbm_x, lbm_y, s=40, c="#14B8A6", edgecolors="#2c3e50",
            linewidths=0.5, zorder=5, alpha=0.9)

# LBM labeled operating points connected as a curve (conservative → aggressive)
lbm_op_sorted = sorted(LBM_POINTS, key=lambda p: p["x"])
lbm_op_x = [p["x"] for p in lbm_op_sorted]
lbm_op_y = [p["y"] for p in lbm_op_sorted]
ax1.plot(lbm_op_x, lbm_op_y, color="#1F77B4", lw=2.8, alpha=0.70, zorder=5,
         ls=(0, (6, 3)), label="LBM operating points")
for pt in LBM_POINTS:
    ax1.scatter([pt["x"]], [pt["y"]], s=80, c=pt["color"],
                edgecolors="white", linewidths=1.2, zorder=6)

# Baseline points
for pt in BASELINE_POINTS:
    ax1.scatter([pt["x"]], [pt["y"]], s=60, c=pt["color"],
                edgecolors="#2c3e50", linewidths=0.6, zorder=5,
                marker=pt["marker"])

# --- Annotations: spread out to avoid overlap ---
# LBM (kinetic) — rightmost, label to the upper-right
ax1.annotate("LBM\n(kinetic)", (0.6277, 0.6712 - IDT_CLIP_S),
             xytext=(14, 12), textcoords="offset points",
             fontsize=8, ha="left", fontweight="bold",
             color="#1a1a1a",
             bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                       edgecolor="#2CA02C", alpha=0.9, linewidth=0.8))

# LBM (balanced) — label above the point
ax1.annotate("LBM\n(balanced)", (0.5397, 0.7102 - IDT_CLIP_S),
             xytext=(8, 16), textcoords="offset points",
             fontsize=8, ha="left", fontweight="bold",
             color="#1a1a1a",
             bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                       edgecolor="#1F77B4", alpha=0.9, linewidth=0.8))

# LBM (enhanced) — label to the left
ax1.annotate("LBM\n(enhanced)", (0.3967, 0.7274 - IDT_CLIP_S),
             xytext=(-60, 8), textcoords="offset points",
             fontsize=8, ha="right", fontweight="bold",
             color="#1a1a1a",
             bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                       edgecolor="#17BECF", alpha=0.9, linewidth=0.8))

# LBM (max style) — label to the left and below
ax1.annotate("LBM\n(max style)", (0.3817, 0.7307 - IDT_CLIP_S),
             xytext=(-60, -18), textcoords="offset points",
             fontsize=8, ha="right", fontweight="bold",
             color="#1a1a1a",
             bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                       edgecolor="#000080", alpha=0.9, linewidth=0.8))

# SaMAM endpoint annotation
ax1.annotate("SaMAM\n(below IDT)", (0.6395, 0.5523 - IDT_CLIP_S),
             xytext=(14, -18), textcoords="offset points",
             fontsize=8, ha="left", fontweight="bold",
             color="#5D8FBF",
             bbox=dict(boxstyle="round,pad=0.15", facecolor="#F0F4FF",
                       edgecolor="#5D8FBF", alpha=0.9, linewidth=0.8))

# SaMST annotation
ax1.annotate("SaMST e15", (0.3681, 0.6957 - IDT_CLIP_S),
             xytext=(12, 12), textcoords="offset points",
             fontsize=8, ha="left", fontweight="medium",
             color="#FF7F0E", alpha=0.9)

# Seedream annotation
ax1.annotate("Seedream-4.5", (0.5077, 0.6920 - IDT_CLIP_S),
             xytext=(-12, -18), textcoords="offset points",
             fontsize=8, ha="right", fontweight="medium",
             color="#9467BD", alpha=0.9)

ax1.set_xlabel("Content Preservation (1 − LPIPS) ↑", fontsize=10.5, fontweight="bold")
ax1.set_ylabel("Style Gain over IDT (Δ CLIP-S) ↑", fontsize=10.5, fontweight="bold")
ax1.set_xlim(0.30, 0.78)
ax1.set_ylim(-0.10, 0.12)
ax1.set_title("(a) IDT Calibration: Convergence Curves", fontsize=11.5, fontweight="bold", pad=8)

# Legend
ax1.legend(loc="lower right", framealpha=0.9, edgecolor="#cccccc",
           frameon=True, fancybox=True)

# ============================================================================
# Panel (b): Training efficiency comparison
# ============================================================================
ax2.set_facecolor("#FAFAFA")

method_names = [d["name"] for d in EFFICIENCY_DATA]
times = [d["time"] for d in EFFICIENCY_DATA]
colors = [d["color"] for d in EFFICIENCY_DATA]

y_pos = np.arange(len(method_names))
bars = ax2.barh(y_pos, times, color=colors, edgecolor="#333333",
                linewidth=0.8, alpha=0.85, height=0.65)

# Time labels
for i, (bar, time) in enumerate(zip(bars, times)):
    if time < 10:
        label = f"{time:.1f} min"
    elif time < 60:
        label = f"{time:.0f} min"
    else:
        hours = time / 60
        label = f"{hours:.1f} h"
    ax2.text(bar.get_width() + max(times) * 0.03, bar.get_y() + bar.get_height() / 2,
             label, va="center", fontsize=8.5, fontweight="bold", color=COLOR_TEXT)

ax2.set_yticks(y_pos)
ax2.set_yticklabels(method_names, fontsize=9)
ax2.set_xlabel("Training Time (minutes, log scale)", fontsize=10.5, fontweight="bold")
ax2.set_xscale("log")
ax2.set_xlim(0.5, max(times) * 2.5)
ax2.set_title("(b) Training Efficiency Comparison", fontsize=11.5, fontweight="bold", pad=8)

# 1-hour reference line
ax2.axvline(x=60, color="#D62728", linestyle="--", linewidth=1.5, alpha=0.6, zorder=2)
ax2.text(60, len(method_names) - 0.3, "1 hour", fontsize=8, color="#D62728",
         ha="center", va="bottom", fontweight="bold")

# ============================================================================
# Save
# ============================================================================
out_dir = r"g:\GitHub\Latent_Style\SchrodingerBridge\aaai2027\figures"
pdf_path = f"{out_dir}\\fig_distinct5_page1_summary.pdf"
png_path = f"{out_dir}\\fig_distinct5_page1_summary.png"

plt.tight_layout(w_pad=2.5)
plt.savefig(pdf_path, format="pdf", dpi=300, bbox_inches="tight", facecolor=COLOR_BG)
plt.savefig(png_path, format="png", dpi=300, bbox_inches="tight", facecolor=COLOR_BG)
print(f"Figure 1 saved: {pdf_path}")
print(f"Figure 1 saved: {png_path}")
plt.close()
