#!/usr/bin/env python3
"""Generate the AAAI-style WD-VF architecture figure."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR                                 # aaai2027_v4/
PDF_PATH = OUT_DIR / "framework_sfm_main.pdf"
PNG_PATH = OUT_DIR / "framework_sfm_main.png"


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9.2,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
)


BLACK = "#131313"
EDGE = "#2E2E2E"
STYLE_BG = "#FBF4E8"
MAIN_BG = "#EAF3FD"
DASH = "#545454"
BLUE = "#DCE8FB"
BLUE_DARK = "#2F55C7"
GREEN = "#B7D6A8"
GREEN_DARK = "#4D6E45"
FIBER = "#DDF3E6"
FIBER_DARK = "#1E7A57"
GRAY = "#E7EBF0"
GRAY_DARK = "#67748A"
PURPLE = "#E6DDF6"
PURPLE_DARK = "#6E42C1"
AMBER = "#FFF1C9"
AMBER_DARK = "#BE8400"
RED = "#C53030"
MAGENTA = "#A72E93"


def band(ax, x, y, w, h, fc, title):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.28,rounding_size=1.05",
            linewidth=1.0,
            linestyle=(0, (4.2, 3.2)),
            edgecolor=DASH,
            facecolor=fc,
            zorder=0,
        )
    )
    ax.text(x + 0.8, y + h - 1.1, title, ha="left", va="center", fontsize=12.2, fontweight="semibold", color=BLACK)


def rounded(ax, x, y, w, h, text="", fc="#FFFFFF", ec=EDGE, lw=1.0, fs=9.2, weight="normal", radius=0.22, color=BLACK, z=2):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.12,rounding_size={radius}",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        zorder=z,
    )
    ax.add_patch(patch)
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight=weight, color=color, zorder=z + 1)
    return patch


def trapezoid(ax, x, y, w, h, text, flip=False, fc="#F7F9FB", ec=EDGE):
    slant = 0.8
    if not flip:
        pts = [(x + slant, y), (x + w, y), (x + w - slant, y + h), (x, y + h)]
    else:
        pts = [(x, y), (x + w - slant, y), (x + w, y + h), (x + slant, y + h)]
    patch = Polygon(pts, closed=True, facecolor=fc, edgecolor=ec, linewidth=1.0, zorder=2)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.2, color=BLACK, zorder=3)
    return patch


def prism(ax, x, y, w, h, text, inverse=False):
    if not inverse:
        pts = [(x, y), (x + w * 0.64, y), (x + w, y + h / 2), (x + w * 0.64, y + h), (x, y + h)]
    else:
        pts = [(x + w, y), (x + w * 0.36, y), (x, y + h / 2), (x + w * 0.36, y + h), (x + w, y + h)]
    patch = Polygon(pts, closed=True, facecolor=GREEN, edgecolor=GREEN_DARK, linewidth=1.2, zorder=2)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.0, color=BLACK, zorder=3)
    return patch


def diamond(ax, cx, cy, w, h, text="", fc=AMBER, ec=AMBER_DARK, glow=False):
    if glow:
        for scale, alpha in [(1.35, 0.10), (1.18, 0.16)]:
            pts = [(cx, cy + h * scale / 2), (cx + w * scale / 2, cy), (cx, cy - h * scale / 2), (cx - w * scale / 2, cy)]
            ax.add_patch(Polygon(pts, closed=True, facecolor="#FFD56A", edgecolor="none", alpha=alpha, zorder=1))
    pts = [(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)]
    patch = Polygon(pts, closed=True, facecolor=fc, edgecolor=ec, linewidth=1.2, zorder=3)
    ax.add_patch(patch)
    if text:
        ax.text(cx, cy, text, ha="center", va="center", fontsize=9.2, color=BLACK, zorder=4)
    return patch


def arrow(ax, start, end, color=BLACK, lw=1.3, dashed=False, rad=0.0, z=4):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=lw,
            linestyle=(0, (4, 2.8)) if dashed else "solid",
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            zorder=z,
        )
    )


def line(ax, points, color=BLACK, lw=1.25, dashed=False, z=3):
    xs, ys = zip(*points)
    ax.plot(xs, ys, color=color, lw=lw, linestyle=(0, (4, 2.8)) if dashed else "solid", zorder=z)


def content_icon(ax, x, y, w, h, tinted=False):
    back = FancyBboxPatch((x - 0.45, y + 0.35), w, h, boxstyle="round,pad=0.06,rounding_size=0.2", linewidth=1.0, edgecolor=EDGE, facecolor="#FFFFFF", zorder=1)
    front = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.2", linewidth=1.0, edgecolor=EDGE, facecolor="#F8FBFD", zorder=2)
    ax.add_patch(back)
    ax.add_patch(front)
    tint = "#EADCCB" if tinted else "#DCEBD8"
    ax.add_patch(Rectangle((x + 0.35, y + 0.42), w - 0.7, h - 0.84, facecolor=tint, edgecolor=EDGE, linewidth=0.7, zorder=3))
    ax.plot([x + 0.6, x + 1.2, x + 1.8, x + 2.35], [y + 0.75, y + 1.5, y + 1.05, y + 2.05], color=GREEN_DARK if not tinted else "#9C6B35", lw=1.0, zorder=4)
    ax.add_patch(Circle((x + 1.0, y + 2.4), 0.18, facecolor="#FFFFFF", edgecolor=EDGE, linewidth=0.7, zorder=4))


def subband_grid(ax, x, y, w, h):
    rounded(ax, x, y, w, h, fc="#F7FBF7", ec=GREEN_DARK, radius=0.16, lw=1.0, z=2)
    gap = 0.22
    cw = (w - 3 * gap) / 2
    ch = (h - 3 * gap) / 2
    cells = {
        "LL": (x + gap, y + h - gap - ch, BLUE, BLUE_DARK),
        "LH": (x + 2 * gap + cw, y + h - gap - ch, FIBER, FIBER_DARK),
        "HL": (x + gap, y + gap, FIBER, FIBER_DARK),
        "HH": (x + 2 * gap + cw, y + gap, GRAY, GRAY_DARK),
    }
    pos = {}
    for label, (cx, cy, fc, tc) in cells.items():
        ax.add_patch(Rectangle((cx, cy), cw, ch, facecolor=fc, edgecolor="#97A3B0", linewidth=0.8, zorder=3))
        ax.text(cx + cw / 2, cy + ch / 2, label, ha="center", va="center", fontsize=10.0, fontweight="semibold", color=tc, zorder=4)
        pos[label] = (cx, cy, cw, ch)
    hhx, hhy, hhw, hhh = pos["HH"]
    ax.plot([hhx + 0.22, hhx + hhw - 0.22], [hhy + 0.22, hhy + hhh - 0.22], color=RED, lw=1.5, zorder=5)
    ax.plot([hhx + 0.22, hhx + hhw - 0.22], [hhy + hhh - 0.22, hhy + 0.22], color=RED, lw=1.5, zorder=5)
    ax.text(hhx + hhw / 2, y - 0.45, "inactive head", ha="center", va="top", fontsize=7.4, color=GRAY_DARK)
    return pos


def cycle_icon(ax, cx, cy, r=0.92):
    ax.add_patch(Arc((cx, cy), 2 * r, 2 * r, theta1=20, theta2=200, lw=1.2, color=BLACK, zorder=3))
    ax.add_patch(Arc((cx, cy), 2 * r, 2 * r, theta1=205, theta2=385, lw=1.2, color=BLACK, zorder=3))
    arrow(ax, (cx + 0.06, cy + r - 0.04), (cx + 0.42, cy + 0.45), color=BLACK, lw=0.9, z=4)
    arrow(ax, (cx - 0.06, cy - r + 0.04), (cx - 0.42, cy - 0.45), color=BLACK, lw=0.9, z=4)


def lock_badge(ax, x, y):
    rounded(ax, x, y, 1.4, 0.92, fc="#FFCF49", ec=AMBER_DARK, radius=0.13, z=5)
    ax.add_patch(Arc((x + 0.7, y + 0.84), 0.78, 0.78, theta1=15, theta2=165, lw=0.8, color=AMBER_DARK, zorder=6))
    ax.add_patch(Circle((x + 0.7, y + 0.43), 0.065, facecolor=EDGE, edgecolor=EDGE, zorder=6))
    ax.add_patch(Rectangle((x + 0.675, y + 0.28), 0.05, 0.12, facecolor=EDGE, edgecolor=EDGE, zorder=6))


def zoom_box(ax, x, y, w, h):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.2,rounding_size=0.32",
            linewidth=1.0,
            linestyle=(0, (3.6, 2.4)),
            edgecolor="#6B7280",
            facecolor="#FFFFFF",
            zorder=1,
        )
    )
    ax.text(x + 0.8, y + h - 0.65, "Micro-Architecture (One Block)", ha="left", va="center", fontsize=8.8, fontweight="semibold", color=BLACK)
    base_y = y + 1.55
    bx = x + 0.8
    widths = [3.1, 3.5, 4.2, 3.0, 2.4]
    labels = [r"AdaLN($t$)", "Self-Attn", r"ReLU$^2$ X-Attn", r"$\tanh$(gate)", "FFN"]
    for idx, (ww, lab) in enumerate(zip(widths, labels)):
        rounded(ax, bx, base_y, ww, 1.6, lab, fc="#F9FAFB", ec="#6B7280", lw=0.8, fs=7.2, radius=0.12, z=2)
        if idx < len(widths) - 1:
            arrow(ax, (bx + ww, base_y + 0.8), (bx + ww + 0.7, base_y + 0.8), lw=1.0, z=3)
        bx += ww + 0.95
    t_cx = x + 2.2
    t_cy = y + h - 1.8
    ax.add_patch(Circle((t_cx, t_cy), 0.42, facecolor="#EEF2FF", edgecolor="#64748B", linewidth=0.8, zorder=2))
    ax.text(t_cx, t_cy, r"$t$", ha="center", va="center", fontsize=7.4, color=BLACK, zorder=3)
    arrow(ax, (t_cx + 0.45, t_cy - 0.05), (x + 2.25, base_y + 1.6), lw=0.9, z=3)
    return {
        "cross_attn_anchor": (x + 13.1, base_y + 1.55),
        "left_anchor": (x, y + h * 0.42),
        "lower_anchor": (x + 2.0, y),
    }


def main():
    fig, ax = plt.subplots(figsize=(7.0, 3.15))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 42)
    ax.axis("off")

    band(ax, 1.4, 29.9, 97.2, 10.2, STYLE_BG, "Style Conditioning")
    band(ax, 1.4, 3.0, 97.2, 24.6, MAIN_BG, "Main Inference Path")

    rounded(ax, 30.0, 33.8, 8.7, 3.2, "Style ID", fc="#FFF8EF")
    rounded(ax, 42.8, 33.5, 9.7, 3.8, "Style\nMemory", fc="#FFF8EF")
    rounded(ax, 56.9, 33.5, 9.7, 3.8, "Style\nTokens", fc="#FFF8EF")
    arrow(ax, (38.7, 35.4), (42.8, 35.4))
    arrow(ax, (52.5, 35.4), (56.9, 35.4))

    zoom = zoom_box(ax, 69.5, 31.4, 27.1, 7.4)
    line(ax, [(59.8, 14.9), (66.2, 18.0), zoom["left_anchor"]], color="#8C8C8C", lw=0.8)
    line(ax, [(59.8, 11.7), (67.2, 12.5), zoom["lower_anchor"]], color="#8C8C8C", lw=0.8)
    line(ax, [(61.8, 35.4), (61.8, 32.2), zoom["cross_attn_anchor"]], color=BLACK, lw=1.0, dashed=True)
    arrow(ax, zoom["cross_attn_anchor"], (zoom["cross_attn_anchor"][0], zoom["cross_attn_anchor"][1] - 1.0), lw=1.0, dashed=True)

    content_icon(ax, 4.1, 13.2, 3.6, 3.2, tinted=False)
    trapezoid(ax, 10.1, 12.9, 6.0, 3.9, "VAE\nEncoder")
    rounded(ax, 18.1, 13.35, 4.6, 3.0, "Latent\n$z_0$")
    prism(ax, 24.7, 12.85, 3.9, 4.0, "Haar\nDWT")
    cells = subband_grid(ax, 30.9, 11.5, 6.4, 5.9)
    rounded(ax, 40.0, 12.95, 5.2, 3.8, r"spectral $H_t$", fc=FIBER, ec=FIBER_DARK, fs=8.8)
    rounded(ax, 49.0, 12.25, 10.3, 5.2, "Shared Backbone", fc=PURPLE, ec=PURPLE_DARK, fs=10.0, weight="semibold", radius=0.34)
    rounded(ax, 61.7, 14.6, 3.8, 2.0, r"$v_{LH}$", fc=FIBER, ec=FIBER_DARK, fs=8.8, weight="semibold")
    rounded(ax, 61.7, 11.3, 3.8, 2.0, r"$v_{HL}$", fc=FIBER, ec=FIBER_DARK, fs=8.8, weight="semibold")
    rounded(ax, 68.0, 10.6, 11.8, 6.9, "", fc="#FFFFFF", ec=FIBER_DARK, lw=1.0, radius=0.2, z=2)
    ax.text(73.9, 15.9, "Spectral ODE Integrator", ha="center", va="center", fontsize=9.6, color=BLACK)
    cycle_icon(ax, 70.9, 13.3, r=0.85)
    ax.text(75.0, 12.8, r"$H_{t+\Delta t}=H_t+\mathbf{v}_H\Delta t$", ha="center", va="center", fontsize=8.1, color="#475467")
    rounded(ax, 81.9, 12.95, 4.6, 3.8, r"$\hat{H}_1$", fc=FIBER, ec=FIBER_DARK, fs=9.2, weight="semibold")
    prism(ax, 88.6, 12.85, 3.9, 4.0, "iDWT", inverse=True)
    diamond(ax, 94.7, 14.85, 4.3, 4.9, glow=True)
    ax.text(94.7, 11.55, "Terminal\nWCT", ha="center", va="top", fontsize=9.6, color=BLACK)
    trapezoid(ax, 96.8, 12.9, 2.4, 3.9, "", flip=True)
    ax.text(98.0, 14.85, "VAE\nDec.", ha="center", va="center", fontsize=7.8, color=BLACK)
    content_icon(ax, 100.4, 13.2, 3.6, 3.2, tinted=True)

    arrow(ax, (7.7, 14.8), (10.1, 14.8))
    arrow(ax, (16.1, 14.8), (18.1, 14.8))
    arrow(ax, (22.7, 14.8), (24.7, 14.8))
    arrow(ax, (28.6, 14.8), (30.9, 14.8))
    arrow(ax, (37.3, 14.8), (40.0, 14.8))
    arrow(ax, (45.2, 14.8), (49.0, 14.8))
    arrow(ax, (59.3, 15.55), (61.7, 15.55))
    arrow(ax, (65.5, 15.55), (68.0, 14.95))
    arrow(ax, (65.5, 12.3), (68.0, 13.0))
    arrow(ax, (79.8, 14.8), (81.9, 14.8))
    arrow(ax, (86.5, 14.8), (88.6, 14.8))
    arrow(ax, (92.5, 14.8), (92.7, 14.8))
    arrow(ax, (96.85, 14.8), (100.25, 14.8))

    ll_mid_y = 20.0
    line(ax, [(32.4, 17.4), (32.4, ll_mid_y), (90.0, ll_mid_y)], lw=1.5)
    arrow(ax, (90.0, ll_mid_y), (90.0, 16.85), lw=1.5)
    ax.text(34.6, 20.45, r"$LL$", ha="center", va="bottom", fontsize=10.0, fontweight="semibold", color=BLACK)
    ax.text(49.8, 21.55, "Base Locked", ha="center", va="center", fontsize=10.0, color=BLACK)
    lock_badge(ax, 49.3, 19.3)
    ax.text(51.3, 19.82, r"$\mathbf{v}_{LL}\equiv 0$", ha="left", va="center", fontsize=8.8, color=BLACK)

    ax.text(34.1, 18.1, "2x2 Haar subbands", ha="center", va="bottom", fontsize=8.2, color="#475467")

    target_y = 8.5
    line(ax, [(61.2, target_y), (79.2, target_y)], color="#7A7A7A", lw=1.1, dashed=True)
    ax.text(70.2, 9.15, r"target $\Delta$", ha="center", va="bottom", fontsize=8.1, color="#667085")
    rounded(ax, 63.0, 7.35, 3.6, 1.65, r"$\Delta_{LH}$", fc="#FFFFFF", ec="#7A7A7A", fs=7.8, radius=0.12)
    rounded(ax, 70.4, 7.35, 3.6, 1.65, r"$\Delta_{HL}$", fc="#FFFFFF", ec="#7A7A7A", fs=7.8, radius=0.12)
    ax.add_patch(Circle((67.5, 8.55), 0.78, facecolor="#FFD4D4", edgecolor=RED, linewidth=1.1, zorder=5))
    ax.text(67.5, 8.55, r"$L_2$", ha="center", va="center", fontsize=8.4, color=RED, fontweight="semibold", zorder=6)
    arrow(ax, (63.6, 14.6), (66.9, 9.25), color=RED, lw=1.05, dashed=True)
    arrow(ax, (63.6, 11.3), (68.0, 9.25), color=RED, lw=1.05, dashed=True)
    ax.text(76.1, 7.9, r"$\mathcal{L}_{FM}=\Vert v_{LH}-\Delta_{LH}\Vert_2^2+\Vert v_{HL}-\Delta_{HL}\Vert_2^2$", ha="left", va="center", fontsize=7.4, color=RED)
    line(ax, [(54.3, ll_mid_y), (54.3, 12.0)], color=RED, lw=1.0, dashed=True)
    line(ax, [(54.3, 10.9), (54.3, 9.6)], color=RED, lw=1.0, dashed=True)
    ax.text(54.85, 11.25, "x", ha="left", va="center", fontsize=9.0, color=RED, fontweight="semibold")
    ax.text(55.9, 10.15, r"$\omega_{LL}\equiv 0$", ha="left", va="center", fontsize=7.8, color=RED)
    ax.text(55.9, 9.15, "No supervision", ha="left", va="center", fontsize=7.4, color="#7A1C1C")

    line(ax, [(61.8, 33.5), (61.8, 22.8), (94.7, 22.8)], color=AMBER_DARK, lw=1.2, dashed=True)
    arrow(ax, (94.7, 22.8), (94.7, 17.4), color=AMBER_DARK, lw=1.2, dashed=True)
    ax.text(83.0, 23.45, "endpoint style injection", ha="center", va="bottom", fontsize=8.0, color="#8F5A00")

    fig.savefig(PDF_PATH)
    fig.savefig(PNG_PATH, dpi=300)
    print(f"Saved {PDF_PATH}")
    print(f"Saved {PNG_PATH}")


if __name__ == "__main__":
    main()
