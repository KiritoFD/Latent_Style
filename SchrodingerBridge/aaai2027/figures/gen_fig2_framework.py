"""
Generate Figure 2: Three-Stage Framework for Latent Style Transfer

Stage 1: Style-ID Encoding - Encode target style identifier into compact control code
Stage 2: Latent Transport - Inference-time path from source latent to stylized output
Stage 3: Training Objectives - Kinetic regularization and terminal distribution matching (training only)

Output: fig2_framework.pdf and fig2_framework.png at 300 DPI
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ---------------------------------------------------------------------------
# Style constants - Academic conference color palette
# ---------------------------------------------------------------------------
COLOR_BG = "#ffffff"
COLOR_STAGE1_BG = "#fef5e7"  # Warm beige
COLOR_STAGE1_BORDER = "#c87f00"  # Deep gold
COLOR_STAGE2_BG = "#e8f4f8"  # Light blue
COLOR_STAGE2_BORDER = "#2e6da4"  # Professional blue
COLOR_STAGE3_BG = "#eaf5ea"  # Soft green
COLOR_STAGE3_BORDER = "#4a7c4a"  # Muted green

COLOR_BOX_STAGE1 = "#fff4e0"
COLOR_BOX_STAGE2 = "#d6eaf8"
COLOR_BOX_STAGE3 = "#d5f0d5"

COLOR_ARROW_INFERENCE = "#2c3e50"  # Dark blue-gray for inference
COLOR_ARROW_TRAINING = "#7f8c8d"  # Muted gray for training
COLOR_TEXT = "#2c3e50"
COLOR_MUTED = "#5a6c7d"

FONT_SIZE_TITLE = 22
FONT_SIZE_STAGE = 15
FONT_SIZE_LABEL = 11
FONT_SIZE_SMALL = 9
FONT_SIZE_TINY = 8

# ---------------------------------------------------------------------------
# Figure layout
# ---------------------------------------------------------------------------
FIG_W = 16.0
FIG_H = 10.5
PANEL_LEFT = 0.3
PANEL_RIGHT = FIG_W - 0.3
PANEL_W = PANEL_RIGHT - PANEL_LEFT

fig, ax = plt.subplots(1, 1, figsize=(FIG_W, FIG_H), facecolor=COLOR_BG)
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.set_aspect("equal")
ax.axis("off")

# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------
ax.text(FIG_W / 2, FIG_H - 0.25, "Three-Stage Framework for Latent Style Transfer",
        fontsize=FONT_SIZE_TITLE, fontweight="bold", ha="center", va="top",
        color=COLOR_TEXT)

# ---------------------------------------------------------------------------
# Helper: draw a rounded box
# ---------------------------------------------------------------------------
def draw_box(ax, x, y, w, h, fill_color, edge_color, lw=1.8, alpha=1.0,
             label=None, label_fontsize=FONT_SIZE_LABEL, label_color=COLOR_TEXT,
             label_weight="normal", sublabel=None, sublabel_fontsize=FONT_SIZE_SMALL):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.1",
                         facecolor=fill_color, edgecolor=edge_color,
                         linewidth=lw, alpha=alpha)
    ax.add_patch(box)
    if label is not None:
        ax.text(x + w / 2, y + h / 2 + (0.08 if sublabel else 0),
                label, fontsize=label_fontsize, ha="center", va="center",
                color=label_color, fontweight=label_weight)
    if sublabel is not None:
        ax.text(x + w / 2, y + h / 2 - 0.2,
                sublabel, fontsize=sublabel_fontsize, ha="center", va="center",
                color=COLOR_MUTED, style="italic")


# ---------------------------------------------------------------------------
# Helper: draw an arrow
# ---------------------------------------------------------------------------
def draw_arrow(ax, x1, y1, x2, y2, color=COLOR_ARROW_INFERENCE, lw=2.0,
               dashed=False, arrowstyle="->", mutation_scale=20):
    linestyle = "dashed" if dashed else "solid"
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle=arrowstyle, mutation_scale=mutation_scale,
                            color=color, linewidth=lw, linestyle=linestyle,
                            connectionstyle="arc3,rad=0")
    ax.add_patch(arrow)


# ---------------------------------------------------------------------------
# Stage background panels
# ---------------------------------------------------------------------------
# Stage 1: Style-ID Encoding (top)
stage1_y = 7.2
stage1_h = 2.0
ax.add_patch(FancyBboxPatch((PANEL_LEFT, stage1_y), PANEL_W, stage1_h,
                             boxstyle="round,pad=0.15",
                             facecolor=COLOR_STAGE1_BG, edgecolor=COLOR_STAGE1_BORDER,
                             linewidth=2.5))
ax.text(PANEL_LEFT + 0.25, stage1_y + stage1_h - 0.25, "Stage 1: Style-ID Encoding",
        fontsize=FONT_SIZE_STAGE, fontweight="bold", color=COLOR_STAGE1_BORDER,
        va="top")

# Stage 2: Latent Transport (middle)
stage2_y = 4.0
stage2_h = 2.8
ax.add_patch(FancyBboxPatch((PANEL_LEFT, stage2_y), PANEL_W, stage2_h,
                             boxstyle="round,pad=0.15",
                             facecolor=COLOR_STAGE2_BG, edgecolor=COLOR_STAGE2_BORDER,
                             linewidth=2.5))
ax.text(PANEL_LEFT + 0.25, stage2_y + stage2_h - 0.25, "Stage 2: Latent Transport (Inference)",
        fontsize=FONT_SIZE_STAGE, fontweight="bold", color=COLOR_STAGE2_BORDER,
        va="top")

# Stage 3: Training Objectives (bottom)
stage3_y = 0.5
stage3_h = 3.2
ax.add_patch(FancyBboxPatch((PANEL_LEFT, stage3_y), PANEL_W, stage3_h,
                             boxstyle="round,pad=0.15",
                             facecolor=COLOR_STAGE3_BG, edgecolor=COLOR_STAGE3_BORDER,
                             linewidth=2.5))
ax.text(PANEL_LEFT + 0.25, stage3_y + stage3_h - 0.25, "Stage 3: Training Objectives (Training Only)",
        fontsize=FONT_SIZE_STAGE, fontweight="bold", color=COLOR_STAGE3_BORDER,
        va="top")

# ---------------------------------------------------------------------------
# Stage 1: Style-ID Encoding - Simplified conceptual flow
# ---------------------------------------------------------------------------
s1_y = stage1_y + 0.35
s1_h = 1.2

# Style Identifier
draw_box(ax, 1.5, s1_y, 2.8, s1_h, COLOR_BOX_STAGE1, COLOR_STAGE1_BORDER,
         label="Style Identifier", sublabel="Target style specification",
         label_fontsize=FONT_SIZE_LABEL, label_weight="bold")

# Arrow
draw_arrow(ax, 4.3, s1_y + s1_h / 2, 5.5, s1_y + s1_h / 2,
           color=COLOR_STAGE1_BORDER, lw=2.5)

# Encoding Process
draw_box(ax, 5.5, s1_y, 3.5, s1_h, COLOR_BOX_STAGE1, COLOR_STAGE1_BORDER,
         label="Style Encoding", sublabel="Compact representation learning",
         label_fontsize=FONT_SIZE_LABEL, label_weight="bold")

# Arrow
draw_arrow(ax, 9.0, s1_y + s1_h / 2, 10.2, s1_y + s1_h / 2,
           color=COLOR_STAGE1_BORDER, lw=2.5)

# Compact Control Code
draw_box(ax, 10.2, s1_y, 3.0, s1_h, COLOR_BOX_STAGE1, COLOR_STAGE1_BORDER,
         label="Control Code", sublabel="Conditioning signal",
         label_fontsize=FONT_SIZE_LABEL, label_weight="bold")

# Annotation
ax.text(14.0, s1_y + s1_h / 2, "Encodes style identity\ninto compact form",
        fontsize=FONT_SIZE_SMALL, ha="center", va="center",
        color=COLOR_STAGE1_BORDER, style="italic")

# Arrow from Stage 1 to Stage 2
draw_arrow(ax, 11.7, s1_y, 11.7, stage2_y + stage2_h,
           color=COLOR_STAGE1_BORDER, lw=2.5, dashed=True)

# ---------------------------------------------------------------------------
# Stage 2: Latent Transport - Inference path
# ---------------------------------------------------------------------------
s2_y = stage2_y + 0.4
s2_h = 1.8

# Source Latent
draw_box(ax, 0.8, s2_y, 2.2, s2_h, COLOR_BOX_STAGE2, COLOR_STAGE2_BORDER,
         label="Source Latent", sublabel="Content representation",
         label_fontsize=FONT_SIZE_LABEL, label_weight="bold")

# Arrow
draw_arrow(ax, 3.0, s2_y + s2_h / 2, 4.0, s2_y + s2_h / 2, lw=2.5)

# Latent Space Transport
draw_box(ax, 4.0, s2_y, 3.8, s2_h, COLOR_BOX_STAGE2, COLOR_STAGE2_BORDER,
         label="Latent Transport", sublabel="Style-guided transformation",
         label_fontsize=FONT_SIZE_LABEL, label_weight="bold")

# Arrow
draw_arrow(ax, 7.8, s2_y + s2_h / 2, 8.8, s2_y + s2_h / 2, lw=2.5)

# Numerical Integration
draw_box(ax, 8.8, s2_y, 2.8, s2_h, COLOR_BOX_STAGE2, COLOR_STAGE2_BORDER,
         label="Integration", sublabel="Path discretization",
         label_fontsize=FONT_SIZE_LABEL, label_weight="bold")

# Arrow
draw_arrow(ax, 11.6, s2_y + s2_h / 2, 12.6, s2_y + s2_h / 2, lw=2.5)

# Stylized Latent
draw_box(ax, 12.6, s2_y, 2.2, s2_h, COLOR_BOX_STAGE2, COLOR_STAGE2_BORDER,
         label="Stylized Latent", sublabel="Transformed representation",
         label_fontsize=FONT_SIZE_LABEL, label_weight="bold")

# Dashed arrows from Stage 2 to Stage 3 (training supervision)
draw_arrow(ax, 5.9, s2_y, 5.9, stage3_y + stage3_h,
           color=COLOR_ARROW_TRAINING, lw=2.0, dashed=True)
draw_arrow(ax, 10.2, s2_y, 10.2, stage3_y + stage3_h,
           color=COLOR_ARROW_TRAINING, lw=2.0, dashed=True)

# ---------------------------------------------------------------------------
# Stage 3: Training Objectives - Two main components
# ---------------------------------------------------------------------------
s3_y = stage3_y + 0.4
s3_h = 2.2

# Left branch: Kinetic Regularization
draw_box(ax, 1.2, s3_y + 0.6, 3.5, 1.2, COLOR_BOX_STAGE3, COLOR_STAGE3_BORDER,
         label="Kinetic Regularization", sublabel="Smooth trajectory constraint",
         label_fontsize=FONT_SIZE_LABEL, label_weight="bold")

# Right branch: Terminal Distribution Matching
draw_box(ax, 6.5, s3_y + 0.6, 4.5, 1.2, COLOR_BOX_STAGE3, COLOR_STAGE3_BORDER,
         label="Terminal Distribution Matching", sublabel="Style distribution alignment",
         label_fontsize=FONT_SIZE_LABEL, label_weight="bold")

# Combined objective annotation
ax.text(13.0, s3_y + 1.2, "Training objectives\nensure quality and\nstyle fidelity",
        fontsize=FONT_SIZE_SMALL, ha="center", va="center",
        color=COLOR_STAGE3_BORDER, style="italic")

# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------
legend_y = stage3_y + 0.15
legend_x = 1.5

# Solid arrow example
draw_arrow(ax, legend_x, legend_y + 0.35, legend_x + 0.8, legend_y + 0.35,
           color=COLOR_ARROW_INFERENCE, lw=2.5)
ax.text(legend_x + 1.1, legend_y + 0.35, "Inference path",
        fontsize=FONT_SIZE_SMALL, ha="left", va="center", color=COLOR_MUTED, fontweight="bold")

# Dashed arrow example
draw_arrow(ax, legend_x + 4.5, legend_y + 0.35, legend_x + 5.3, legend_y + 0.35,
           color=COLOR_ARROW_TRAINING, lw=2.0, dashed=True)
ax.text(legend_x + 5.6, legend_y + 0.35, "Training supervision",
        fontsize=FONT_SIZE_SMALL, ha="left", va="center", color=COLOR_MUTED, fontweight="bold")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_dir = r"g:\GitHub\Latent_Style\SchrodingerBridge\aaai2027\figures"
pdf_path = f"{out_dir}\\fig2_framework.pdf"
png_path = f"{out_dir}\\fig2_framework.png"

plt.tight_layout()
plt.savefig(pdf_path, format="pdf", dpi=300, bbox_inches="tight", facecolor=COLOR_BG)
plt.savefig(png_path, format="png", dpi=300, bbox_inches="tight", facecolor=COLOR_BG)
print(f"Figure 2 saved: {pdf_path}")
print(f"Figure 2 saved: {png_path}")
plt.close()
