"""Publication-quality AAAI architecture diagram using Matplotlib.

Renders the Spectral ODE Bridge with low-saturation colors, LaTeX math,
embedded thumbnails, and orthogonal arrows. Outputs PNG (300 dpi) and SVG.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter
from matplotlib.patches import FancyBboxPatch, Polygon, FancyArrowPatch, Arc
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties

OUT_DIR = Path("g:/GitHub/Latent_Style/SchrodingerBridge/docs/630")
EMOJI_FP = FontProperties(family="Segoe UI Emoji", size=10)
THUMB_DIR = OUT_DIR / "thumbs"

# -----------------------------------------------------------------------------
# Palette (Morandi / low saturation)
# -----------------------------------------------------------------------------
C = {
    "ll": "#DAE8FC",
    "ll_dark": "#6C8EBF",
    "mid": "#D5E8D4",
    "mid_dark": "#82B366",
    "net": "#E1D5E7",
    "net_dark": "#9673A6",
    "style": "#FFF2E6",
    "style_dark": "#C65911",
    "style_line": "#7B1FA2",
    "dead": "#F5F5F5",
    "dead_dark": "#666666",
    "red": "#C62828",
    "train": "#FFEBEE",
    "train_dark": "#B71C1C",
    "bypass": "#FBE9E7",
    "bypass_dark": "#C62828",
    "lane_style": "#FFF9E6",
    "lane_infer": "#F4F9FF",
    "lane_train": "#FFF5F5",
    "text": "#1F2937",
    "white": "#FFFFFF",
}

FIG_W, FIG_H = 17, 9
DPI = 300


def load_thumb(name, kind=None):
    img = Image.open(THUMB_DIR / name).convert("RGB")
    if kind == "ll":
        img = img.filter(ImageFilter.GaussianBlur(radius=2.5))
    elif kind == "edge":
        img = img.convert("L").filter(ImageFilter.FIND_EDGES).convert("RGB")
    return img


def place_image(ax, img, x, y, w, h, zorder=5):
    arr = np.array(img)
    ax.imshow(arr, extent=[x, x + w, y, y + h], origin="lower", aspect="auto", zorder=zorder)


# -----------------------------------------------------------------------------
# Drawing helpers
# -----------------------------------------------------------------------------
def add_box(ax, x, y, w, h, label, fill, edge, fs=10, bold=False,
            zorder=3, lw=1, dashed=False, align="center", math=True):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.005,rounding_size=0.04",
        facecolor=fill, edgecolor=edge, linewidth=lw,
        linestyle="--" if dashed else "-",
        zorder=zorder,
    )
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    ax.text(x + w / 2, y + h / 2, label, fontsize=fs, color=C["text"],
            ha=align, va="center", fontweight=weight, zorder=zorder + 1)
    return (x, y, w, h)


def add_text(ax, x, y, label, fs=10, color=None, bold=False, align="left", va="center",
             fp=None):
    color = color or C["text"]
    weight = "bold" if bold else "normal"
    if fp is not None:
        fp = fp.copy()
        fp.set_size(fs)
        ax.text(x, y, label, fontproperties=fp, color=color, ha=align, va=va,
                fontweight=weight, zorder=6)
    else:
        ax.text(x, y, label, fontsize=fs, color=color, ha=align, va=va,
                fontweight=weight, zorder=6)


def point(node, side):
    x, y, w, h = node
    if side == "left":
        return (x, y + h / 2)
    if side == "right":
        return (x + w, y + h / 2)
    if side == "top":
        return (x + w / 2, y + h)
    if side == "bottom":
        return (x + w / 2, y)
    if side == "center":
        return (x + w / 2, y + h / 2)
    raise ValueError(side)


def arrow(ax, pts, color, lw=1.5, dashed=False, zorder=2):
    """Draw polyline arrow; pts are data coords, arrow head at last point."""
    ls = "--" if dashed else "-"
    ax.add_line(Line2D([p[0] for p in pts], [p[1] for p in pts],
                       color=color, linewidth=lw, linestyle=ls, zorder=zorder))
    # arrowhead
    if len(pts) < 2:
        return
    x1, y1 = pts[-2]
    x2, y2 = pts[-1]
    dx, dy = x2 - x1, y2 - y1
    ang = np.arctan2(dy, dx)
    hl, hw = 0.12, 0.08
    a1 = (x2 - hl * np.cos(ang) + hw * np.sin(ang),
          y2 - hl * np.sin(ang) - hw * np.cos(ang))
    a2 = (x2 - hl * np.cos(ang) - hw * np.sin(ang),
          y2 - hl * np.sin(ang) + hw * np.cos(ang))
    head = Polygon([a1, (x2, y2), a2], closed=True, facecolor=color,
                   edgecolor=color, linewidth=0, zorder=zorder + 1)
    ax.add_patch(head)


def connect(ax, n1, side1, n2, side2, color, lw=1.5, dashed=False, waypoints=None, zorder=2):
    p1 = point(n1, side1)
    p2 = point(n2, side2)
    pts = [p1]
    if waypoints:
        pts.extend(waypoints)
    pts.append(p2)
    arrow(ax, pts, color, lw, dashed, zorder)


def draw_lock(ax, x, y, size, color):
    """Simple padlock icon using matplotlib primitives."""
    shackle_w, shackle_h = size * 0.55, size * 0.35
    body_w, body_h = size * 0.75, size * 0.55
    # shackle arc
    arc = Arc((x, y + body_h * 0.45), shackle_w, shackle_h,
              angle=0, theta1=0, theta2=180, color=color, lw=1.5, zorder=6)
    ax.add_patch(arc)
    # body
    body = FancyBboxPatch((x - body_w / 2, y - body_h / 2), body_w, body_h,
                          boxstyle="round,pad=0.005,rounding_size=0.02",
                          facecolor=color, edgecolor=color, linewidth=0, zorder=6)
    ax.add_patch(body)
    # keyhole
    ax.add_patch(plt.Circle((x, y - body_h * 0.05), size * 0.08,
                            facecolor="white", edgecolor="none", zorder=7))


def draw_cross(ax, x, y, size, color, lw=2):
    ax.add_line(Line2D([x - size / 2, x + size / 2], [y - size / 2, y + size / 2],
                       color=color, linewidth=lw, zorder=6))
    ax.add_line(Line2D([x - size / 2, x + size / 2], [y + size / 2, y - size / 2],
                       color=color, linewidth=lw, zorder=6))


# -----------------------------------------------------------------------------
# Figure setup
# -----------------------------------------------------------------------------
fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.set_aspect("equal")
ax.axis("off")
ax.set_facecolor(C["white"])

# Lanes
for x, y, w, h, fc in [
    (0.2, 7.85, 16.6, 1.0, C["lane_style"]),
    (0.2, 3.55, 16.6, 4.3, C["lane_infer"]),
    (0.2, 0.55, 16.6, 3.0, C["lane_train"]),
]:
    lane = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.05",
                          facecolor=fc, edgecolor="none", zorder=0)
    ax.add_patch(lane)

# Section labels
add_text(ax, 0.35, 8.55, "Style pathway", fs=13, color=C["style_dark"], bold=True)
add_text(ax, 0.35, 7.55, "Main inference path", fs=13, color=C["ll_dark"], bold=True)
add_text(ax, 0.35, 3.30, "Training objective", fs=13, color=C["train_dark"], bold=True)

# -----------------------------------------------------------------------------
# Style pathway
# -----------------------------------------------------------------------------
place_image(ax, load_thumb("style_thumb.png"), 0.6, 8.0, 0.8, 0.8)
add_text(ax, 1.0, 7.90, "$I_s$", fs=12, align="center")
style_mem = add_box(ax, 1.6, 8.15, 1.3, 0.55, "Style Memory\n(learnable)\n5 styles × 256",
                    C["style"], C["style_dark"], fs=10)
style_tok = add_box(ax, 3.1, 8.15, 1.5, 0.55, "$S\\in\\mathbb{R}^{256\\times64}$",
                    C["style"], C["style_dark"], fs=12)
connect(ax, (0.6, 8.0, 0.8, 0.8), "right", style_mem, "left", C["style_dark"])
connect(ax, style_mem, "right", style_tok, "left", C["style_dark"])

# -----------------------------------------------------------------------------
# Content input
# -----------------------------------------------------------------------------
place_image(ax, load_thumb("content_thumb.png"), 0.6, 6.15, 0.9, 0.9)
add_text(ax, 1.05, 6.05, "$I_c$", fs=12, align="center")
enc = add_box(ax, 1.7, 6.35, 1.0, 0.55, "VAE\nEncoder", C["net"], C["net_dark"], fs=11)
z0 = add_box(ax, 2.95, 6.40, 0.65, 0.45, "$z_0$", C["ll"], C["ll_dark"], fs=16, bold=True)
connect(ax, (0.6, 6.15, 0.9, 0.9), "right", enc, "left", C["ll_dark"])
connect(ax, enc, "right", z0, "left", C["ll_dark"])

# -----------------------------------------------------------------------------
# DWT split
# -----------------------------------------------------------------------------
dwt = add_box(ax, 4.15, 7.25, 0.95, 0.32, "Haar DWT", C["net"], C["net_dark"], fs=10, bold=True)
connect(ax, z0, "top", dwt, "bottom", C["ll_dark"], waypoints=[(3.275, 7.0), (4.625, 7.0)])

# LL locked
place_image(ax, load_thumb("content_thumb.png", kind="ll"), 4.25, 7.70, 0.55, 0.40, zorder=4)
ll = add_box(ax, 4.2, 6.70, 0.7, 0.48, "LL", C["ll"], C["ll_dark"], fs=16, bold=True)
lock = add_box(ax, 5.1, 6.75, 1.4, 0.42, "$v_{LL} \\equiv 0$\n(Base Locked)",
               C["ll"], C["ll_dark"], fs=11, bold=True, align="left")
draw_lock(ax, 5.32, 6.93, 0.18, C["ll_dark"])
connect(ax, dwt, "bottom", ll, "top", C["ll_dark"])

# Mid-frequency
place_image(ax, load_thumb("content_thumb.png", kind="edge"), 4.25, 6.05, 0.50, 0.35, zorder=4)
lh = add_box(ax, 4.2, 5.78, 0.6, 0.42, "LH", C["mid"], C["mid_dark"], fs=14, bold=True)
hl = add_box(ax, 4.2, 5.05, 0.6, 0.42, "HL", C["mid"], C["mid_dark"], fs=14, bold=True)
connect(ax, dwt, "bottom", lh, "top", C["mid_dark"])
# small vertical between LH and HL not needed, they share origin

# HH discarded
hh = add_box(ax, 4.2, 4.30, 0.6, 0.36, "HH", C["dead"], C["dead_dark"], fs=13)
draw_cross(ax, 4.85, 4.48, 0.35, C["red"], lw=3)
add_text(ax, 4.85, 4.30, "Discarded", fs=10, color=C["red"], va="center")
connect(ax, dwt, "bottom", hh, "top", C["dead_dark"])

# -----------------------------------------------------------------------------
# Spectral ODE Integrator
# -----------------------------------------------------------------------------
ode_ode = add_box(ax, 6.0, 4.45, 5.4, 2.75, "", C["white"], C["mid_dark"],
                   fs=1, zorder=1, lw=2, dashed=True)
add_text(ax, 6.2, 6.95, "Spectral ODE Integrator  ($t: 0 \\rightarrow 1$)",
         fs=13, color=C["mid_dark"], bold=True)

h_t = add_box(ax, 6.2, 5.35, 0.8, 0.55, "$H_t$", C["mid"], C["mid_dark"], fs=15, bold=True)
t_box = add_box(ax, 7.35, 6.25, 0.45, 0.35, "$t$", C["net"], C["net_dark"], fs=15)
bb = add_box(ax, 7.15, 5.25, 1.6, 0.95, "Shared\nBackbone\n(×4 blocks)",
             C["net"], C["net_dark"], fs=11, bold=True)
v_lh = add_box(ax, 9.2, 5.65, 0.75, 0.42, "$v_{LH}$", C["mid"], C["mid_dark"], fs=13, bold=True)
v_hl = add_box(ax, 9.2, 5.00, 0.75, 0.42, "$v_{HL}$", C["mid"], C["mid_dark"], fs=13, bold=True)
update = add_box(ax, 7.25, 4.65, 2.7, 0.5,
                 "$H_{t+\\Delta t} = H_t + v_H \\cdot \\Delta t$",
                 C["mid"], C["mid_dark"], fs=12)

# inputs into H_t
connect(ax, lh, "right", h_t, "left", C["mid_dark"], waypoints=[(5.2, 5.99), (5.2, 5.62), (6.2, 5.62)])
connect(ax, hl, "right", h_t, "left", C["mid_dark"], waypoints=[(5.2, 5.26), (5.2, 5.62), (6.2, 5.62)])
connect(ax, h_t, "right", bb, "left", C["net_dark"])
connect(ax, t_box, "bottom", bb, "top", C["net_dark"])
connect(ax, bb, "right", v_lh, "left", C["mid_dark"])
connect(ax, bb, "right", v_hl, "left", C["mid_dark"])
connect(ax, v_lh, "bottom", update, "top", C["mid_dark"], waypoints=[(9.575, 5.65), (9.575, 5.32), (9.35, 5.32)])
connect(ax, v_hl, "top", update, "top", C["mid_dark"], waypoints=[(9.575, 5.21), (9.575, 5.32), (9.35, 5.32)])

# ODE loop back to H_t
arrow(ax, [(8.6, 4.9), (8.6, 4.35), (5.9, 4.35), (5.9, 5.62), (6.2, 5.62)],
      C["mid_dark"], lw=2, dashed=True, zorder=2)
add_text(ax, 6.0, 4.22, "$K$ steps", fs=10, color=C["mid_dark"], bold=True)

# One-block zoom inset
inset = add_box(ax, 11.7, 5.7, 1.6, 1.1, "", "#FFF0EE", "#A67C7A", fs=1, zorder=3)
add_text(ax, 11.8, 6.60, "(a) One Block", fs=11, bold=True)
add_text(ax, 11.8, 6.15, "AdaLN($t$) → Self-Attn\nDWT-Route X-Attn\nReLU$^2$ + tanh gate\n→ FFN",
         fs=10, va="center")

# -----------------------------------------------------------------------------
# Reconstruction
# -----------------------------------------------------------------------------
h1 = add_box(ax, 11.3, 5.35, 0.85, 0.55, "$\\hat{H}_1$", C["mid"], C["mid_dark"], fs=15, bold=True)
connect(ax, update, "right", h1, "left", C["mid_dark"],
        waypoints=[(9.95, 4.9), (9.95, 5.62), (11.3, 5.62)])

idwt = add_box(ax, 12.5, 5.35, 0.85, 0.55, "iDWT", C["net"], C["net_dark"], fs=16, bold=True)
connect(ax, h1, "right", idwt, "left", C["mid_dark"])

# LL bypass line across top
arrow(ax, [(4.9, 6.94), (12.92, 6.94)], C["ll_dark"], lw=3, zorder=2)
arrow(ax, [(12.92, 6.94), (12.92, 5.90)], C["ll_dark"], lw=3, zorder=2)

# Endpoint AdaIN as diamond
ada_x, ada_y, ada_w, ada_h = 13.6, 5.20, 1.3, 0.85
diamond = Polygon([
    (ada_x + ada_w / 2, ada_y + ada_h),
    (ada_x + ada_w, ada_y + ada_h / 2),
    (ada_x + ada_w / 2, ada_y),
    (ada_x, ada_y + ada_h / 2),
], closed=True, facecolor=C["style"], edgecolor=C["style_line"], linewidth=2.5, zorder=3)
ax.add_patch(diamond)
add_text(ax, ada_x + ada_w / 2, ada_y + ada_h / 2, "Endpoint\nAdaIN/WCT",
         fs=11, bold=True, align="center", va="center")

connect(ax, idwt, "right", (ada_x, ada_y, ada_w, ada_h), "left", C["ll_dark"])

dec = add_box(ax, 15.2, 5.35, 0.95, 0.55, "VAE\nDecoder", C["net"], C["net_dark"], fs=11)
connect(ax, (ada_x, ada_y, ada_w, ada_h), "right", dec, "left", C["ll_dark"])

place_image(ax, load_thumb("output_thumb.png"), 16.35, 5.10, 0.9, 0.9)
add_text(ax, 16.80, 5.00, "$\\hat{x}_{out}$", fs=12, align="center")
connect(ax, dec, "right", (16.35, 5.10, 0.9, 0.9), "left", C["ll_dark"])

# -----------------------------------------------------------------------------
# Style condition lines
# -----------------------------------------------------------------------------
# S_tokens down to backbone cross-attention
arrow(ax, [(3.85, 8.15), (3.85, 6.9), (7.95, 6.9), (7.95, 6.20)],
      C["style_line"], lw=2.5, zorder=2)
add_text(ax, 4.0, 6.95, "$S_{tokens}$", fs=10, color=C["style_line"], bold=True)

# S_global across top to AdaIN
arrow(ax, [(4.6, 8.70), (14.25, 8.70), (14.25, 6.05)],
      C["style_line"], lw=3.5, zorder=2)
add_text(ax, 8.5, 8.78, "$S_{global}$", fs=11, color=C["style_line"], bold=True)

# -----------------------------------------------------------------------------
# Training objective
# -----------------------------------------------------------------------------
train_box = add_box(ax, 0.5, 0.70, 10.2, 2.20, "", C["train"], C["train_dark"],
                    fs=1, zorder=1, lw=1)
add_text(ax, 0.65, 2.55, "$x_t = (1-t)z_0 + t z_{target}$", fs=11)

xt = add_box(ax, 0.75, 1.70, 0.6, 0.42, "$x_t$", C["train"], C["train_dark"], fs=14, bold=True)
dwt_t = add_box(ax, 1.6, 1.70, 0.6, 0.42, "DWT", C["train"], C["train_dark"], fs=13)
pred = add_box(ax, 2.45, 1.60, 1.4, 0.55, "Predict\n$v_{LH}, v_{HL}$", C["train"], C["train_dark"], fs=11)
tgt = add_box(ax, 4.15, 1.60, 2.0, 0.55,
              "Target\n$\\Delta_i = \\mathrm{DWT}(z_t-z_0)_i$", C["train"], C["train_dark"], fs=11)
loss = add_box(ax, 6.45, 1.55, 3.8, 0.65,
               "$\\mathcal{L} = \\omega_{LH}\\|v_{LH}-\\Delta_{LH}\\|_2^2 + "
               "\\omega_{HL}\\|v_{HL}-\\Delta_{HL}\\|_2^2$\n($\\omega_{LL}=0$)",
               C["train"], C["train_dark"], fs=12)

connect(ax, xt, "right", dwt_t, "left", C["train_dark"])
connect(ax, dwt_t, "right", pred, "left", C["train_dark"])
connect(ax, pred, "right", loss, "left", C["train_dark"])
connect(ax, tgt, "right", loss, "left", C["train_dark"])

# Training dashed feedback
connect(ax, xt, "top", z0, "bottom", C["train_dark"], dashed=True,
        waypoints=[(1.05, 2.12), (1.05, 4.0), (3.275, 4.0), (3.275, 6.40)])
connect(ax, loss, "top", bb, "bottom", C["train_dark"], dashed=True,
        waypoints=[(8.35, 2.20), (8.35, 4.4), (7.95, 4.4), (7.95, 5.25)])

# -----------------------------------------------------------------------------
# Bypassed mechanisms
# -----------------------------------------------------------------------------
bypass = add_box(ax, 11.0, 0.85, 5.6, 1.90, "", C["bypass"], C["bypass_dark"],
                 fs=1, zorder=1, lw=1)
add_text(ax, 11.2, 2.45, "Bypassed Mechanisms (Ablated)", fs=13,
         color=C["bypass_dark"], bold=True)
add_text(ax, 11.2, 1.85, "×  Euclidean OT Matching\n"
         "×  GroupNorm / Whitening\n"
         "×  Multi-step Style Guidance",
         fs=12, color=C["text"], va="center")

# -----------------------------------------------------------------------------
# Legend
# -----------------------------------------------------------------------------
leg_y = 0.25
add_text(ax, 0.5, leg_y, "Legend:", fs=11, bold=True)
# small color swatches
swatches = [
    (C["ll"], C["ll_dark"], "LL / content"),
    (C["mid"], C["mid_dark"], "LH / HL / spectral"),
    (C["net"], C["net_dark"], "network"),
    (C["style"], C["style_line"], "style condition"),
]
x_off = 1.2
for fc, ec, lab in swatches:
    rect = FancyBboxPatch((x_off, leg_y - 0.08), 0.25, 0.16,
                          boxstyle="round,pad=0.01,rounding_size=0.02",
                          facecolor=fc, edgecolor=ec, linewidth=1, zorder=3)
    ax.add_patch(rect)
    add_text(ax, x_off + 0.35, leg_y, lab, fs=10)
    x_off += 1.4
add_text(ax, x_off + 0.2, leg_y, "— inference   · · · training   — style", fs=10)

# -----------------------------------------------------------------------------
# Caption
# -----------------------------------------------------------------------------
caption = (
    "Figure 2. Overview of Spectral ODE Bridge. The content latent is decomposed by Haar DWT; "
    "LL is locked ($v_{LL}\\equiv0$), LH/HL form the spectral state $H_t$ driven by a shared backbone "
    "with per-subband velocity heads, and HH is discarded. The ODE is integrated for $K$ steps; "
    "the locked LL bypasses the ODE and reunites with the final high-frequency estimate at iDWT. "
    "Style is injected only at the endpoint via AdaIN/WCT ($S_{global}$), while $S_{tokens}$ conditions "
    "the backbone cross-attention. Training supervises only LH/HL velocities with $\\omega_{LL}=0$."
)
add_text(ax, 0.5, 0.05, caption, fs=10, va="bottom")

# -----------------------------------------------------------------------------
# Save
# -----------------------------------------------------------------------------
out_png = OUT_DIR / "aaai_arch_diagram_v6.png"
out_svg = OUT_DIR / "aaai_arch_diagram_v6.svg"
fig.savefig(out_png, dpi=DPI, bbox_inches="tight", pad_inches=0.03, facecolor="white")
fig.savefig(out_svg, bbox_inches="tight", pad_inches=0.03, facecolor="white")
print(f"Saved {out_png} and {out_svg}")
