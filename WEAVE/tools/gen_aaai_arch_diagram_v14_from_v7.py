"""
AAAI main architecture diagram v7 — redrawn in a clean, publication-ready style
inspired by the Latent Bridge Matching reference figure.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Polygon, FancyArrowPatch, Arc, Circle, Rectangle
from matplotlib.lines import Line2D
from PIL import Image
import numpy as np
from pathlib import Path

OUT_DIR = Path("g:/GitHub/Latent_Style/SchrodingerBridge/docs/630")

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["mathtext.fontset"] = "stix"

C = {
    "bg": "#FFFFFF",
    "text": "#1F2937",
    "title": "#111827",
    "section1_bg": "#FFFBEB",   # warm yellow
    "section1_stroke": "#D97706",
    "section2_bg": "#EFF6FF",   # cool blue
    "section2_stroke": "#2563EB",
    "section3_bg": "#FEF2F2",   # red
    "section3_stroke": "#DC2626",
    "style": "#F59E0B",         # orange
    "style_dark": "#92400E",
    "content": "#3B82F6",       # blue
    "content_dark": "#1E40AF",
    "spectral": "#10B981",      # green
    "spectral_dark": "#047857",
    "network": "#8B5CF6",       # purple
    "network_dark": "#5B21B6",
    "train": "#EF4444",         # red
    "train_dark": "#991B1B",
    "gray": "#9CA3AF",
    "gray_dark": "#4B5563",
    "white": "#FFFFFF",
    "red": "#DC2626",
}


def save_im():
    fig, ax = plt.subplots(figsize=(20.8, 10), facecolor=C["bg"])
    ax.set_xlim(0, 21)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_aspect("equal")

    # --- Section backgrounds -------------------------------------------------
    sec_h = [1.5, 4.6, 3.1]
    sec_y = [8.2, 3.2, -0.1]
    sec_labels = [
        ("1. STYLE CONTROL", C["section1_bg"], C["section1_stroke"]),
        ("2. MAIN INFERENCE PATH", C["section2_bg"], C["section2_stroke"]),
        ("3. TRAINING (SUPERVISION & ENDPOINT)", C["section3_bg"], C["section3_stroke"]),
    ]
    for (label, fill, stroke), y, h in zip(sec_labels, sec_y, sec_h):
        bg = FancyBboxPatch((0.25, y), 20.25, h,
                            boxstyle="round,pad=0.02,rounding_size=0.2",
                            facecolor=fill, edgecolor=stroke, linewidth=2.5, zorder=0)
        ax.add_patch(bg)
        ax.text(0.45, y + h - 0.28, label, fontsize=15, weight="bold",
                color=stroke, va="top", ha="left")

    # --- Title ---------------------------------------------------------------
    ax.text(11.0, 9.82, "Spectral ODE Bridge", fontsize=24, weight="bold",
            color=C["title"], ha="center", va="top")

    # --- Helpers --------------------------------------------------------------
    def block(x, y, w, h, label, fill=C["white"], stroke=C["text"],
              fontsize=11, bold=True, label_y=None, radius=0.12, lw=2.2):
        r = FancyBboxPatch((x, y), w, h,
                           boxstyle=f"round,pad=0.01,rounding_size={radius}",
                           facecolor=fill, edgecolor=stroke, linewidth=lw, zorder=2)
        ax.add_patch(r)
        ypos = y + h / 2 if label_y is None else label_y
        ax.text(x + w / 2, ypos, label, fontsize=fontsize, weight="bold" if bold else "normal",
                color=C["text"], ha="center", va="center", zorder=3)
        return r

    def icon_block(x, y, w, h, label, icon_func, fill=C["white"], stroke=C["text"],
                   fontsize=10, icon_args=()):
        block(x, y, w, h, "", fill=fill, stroke=stroke, lw=2.2)
        icon_func(ax, x + w / 2, y + h * 0.55, *icon_args)
        ax.text(x + w / 2, y + h * 0.18, label, fontsize=fontsize, weight="bold",
                color=C["text"], ha="center", va="center")

    def arrow(pts, color="black", lw=2, dashed=False, zorder=1, style="-|>", head_width=0.16):
        ls = "--" if dashed else "-"
        for i in range(len(pts) - 1):
            ax.annotate("", xy=pts[i + 1], xytext=pts[i],
                        arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                        ls=ls, connectionstyle="arc3,rad=0"),
                        zorder=zorder)

    def point(rect, side):
        x, y, w, h = rect.get_x(), rect.get_y(), rect.get_width(), rect.get_height()
        if side == "top":    return (x + w / 2, y + h)
        if side == "bottom": return (x + w / 2, y)
        if side == "left":   return (x, y + h / 2)
        if side == "right":  return (x + w, y + h / 2)

    def connect(n1, s1, n2, s2, color="black", lw=2, dashed=False, waypoints=None):
        p1 = point(n1, s1)
        p2 = point(n2, s2)
        pts = [p1]
        if waypoints:
            pts.extend(waypoints)
        pts.append(p2)
        arrow(pts, color, lw, dashed)

    # --- Icons ----------------------------------------------------------------
    def icon_vae(ax, x, y, size, color):
        # lens / funnel trapezoid
        w, h = size * 1.0, size * 0.7
        pts = [(x - w * 0.5, y - h * 0.5),
               (x + w * 0.15, y - h * 0.35),
               (x + w * 0.15, y + h * 0.35),
               (x - w * 0.5, y + h * 0.5)]
        ax.add_patch(Polygon(pts, closed=True, facecolor=color, edgecolor="none", zorder=3))
        # right triangle
        pts2 = [(x + w * 0.15, y - h * 0.35),
                (x + w * 0.5, y),
                (x + w * 0.15, y + h * 0.35)]
        ax.add_patch(Polygon(pts2, closed=True, facecolor=color, edgecolor="none", alpha=0.7, zorder=3))

    def icon_grid(ax, x, y, size, color, rows=4, cols=4):
        n = rows * cols
        shades = [0.25, 0.45, 0.65, 0.85]
        np.random.seed(7)
        for i in range(rows):
            for j in range(cols):
                s = size / rows * 0.82
                xx = x - size / 2 + j * size / cols + 0.02
                yy = y - size / 2 + i * size / rows + 0.02
                a = shades[(i + j) % 4]
                ax.add_patch(Rectangle((xx, yy), s, s, facecolor=color, alpha=a, edgecolor="white", lw=0.5, zorder=3))

    def icon_clock(ax, x, y, size, color):
        ax.add_patch(Circle((x, y), size / 2, facecolor="white", edgecolor=color, linewidth=2, zorder=3))
        ax.plot([x, x], [y, y + size * 0.25], color=color, lw=2, zorder=4)
        ax.plot([x, x + size * 0.18], [y, y - size * 0.05], color=color, lw=2, zorder=4)

    def icon_graph(ax, x, y, size, color):
        pts = [(x - size * 0.3, y + size * 0.1),
               (x - size * 0.05, y + size * 0.25),
               (x + size * 0.25, y + size * 0.05),
               (x + size * 0.05, y - size * 0.25)]
        for p in pts:
            ax.add_patch(Circle(p, size * 0.08, facecolor=color, edgecolor="white", zorder=3))
        for i in range(len(pts) - 1):
            ax.plot([pts[i][0], pts[i + 1][0]], [pts[i][1], pts[i + 1][1]], color=color, lw=1.5, zorder=2)

    def icon_unet(ax, x, y, size, color):
        # encoder-decoder U shape
        w, h = size * 0.9, size * 0.55
        enc = FancyBboxPatch((x - w / 2, y - h / 2), w * 0.35, h,
                             boxstyle="round,pad=0.01,rounding_size=0.04",
                             facecolor=color, edgecolor=color, zorder=3)
        dec = FancyBboxPatch((x + w * 0.03, y - h / 2), w * 0.35, h,
                             boxstyle="round,pad=0.01,rounding_size=0.04",
                             facecolor=color, edgecolor=color, alpha=0.7, zorder=3)
        ax.add_patch(enc)
        ax.add_patch(dec)
        # bridge
        ax.plot([x - w * 0.13, x + w * 0.13], [y + h * 0.05, y + h * 0.05], color=color, lw=2, zorder=4)

    def icon_euler(ax, x, y, size, color):
        pts = [(x - size * 0.25, y - size * 0.1),
               (x, y + size * 0.2),
               (x + size * 0.25, y - size * 0.1),
               (x + size * 0.35, y + size * 0.15),
               (x, y - size * 0.2),
               (x - size * 0.35, y + size * 0.15)]
        for p in pts:
            ax.add_patch(Circle(p, size * 0.07, facecolor=color, edgecolor="white", zorder=3))
        for i in range(len(pts) - 1):
            ax.plot([pts[i][0], pts[i + 1][0]], [pts[i][1], pts[i + 1][1]], color=color, lw=1.2, zorder=2)

    def icon_style_bars(ax, x, y, w, h, color):
        n = 5
        bw = w / (2 * n + 1)
        np.random.seed(3)
        for i in range(n):
            hh = h * (0.4 + np.random.rand() * 0.5)
            ax.add_patch(Rectangle((x - w / 2 + (2 * i + 1) * bw, y - hh / 2),
                                    bw, hh, facecolor=color, edgecolor="white", lw=0.5, zorder=3))

    def icon_style_tokens(ax, x, y, size, color):
        rows, cols = 4, 4
        s = size / cols * 0.85
        for i in range(rows):
            for j in range(cols):
                a = 0.3 + ((i * cols + j) % 4) * 0.2
                ax.add_patch(Rectangle((x - size / 2 + j * size / cols + 0.02,
                                        y - size / 2 + i * size / rows + 0.02),
                                       s, s, facecolor=color, alpha=a, edgecolor="white", lw=0.5, zorder=3))

    def icon_dwt(ax, x, y, size, color):
        # 2x2 quadrants
        half = size / 2
        labels = [("LL", C["content"]), ("LH", C["spectral"]), ("HL", C["spectral"]), ("HH", C["gray"])]
        positions = [(0, 1), (1, 1), (0, 0), (1, 0)]
        for (lab, col), (ix, iy) in zip(labels, positions):
            xx = x - half + ix * half
            yy = y - half + iy * half
            ax.add_patch(Rectangle((xx + 0.02, yy + 0.02), half - 0.04, half - 0.04,
                                   facecolor=col, alpha=0.2, edgecolor=col, lw=1, zorder=3))
            ax.text(xx + half / 2, yy + half / 2, lab, fontsize=8, weight="bold",
                    color=col, ha="center", va="center", zorder=4)
        # cross over HH
        draw_cross(ax, x + half / 2, y - half / 2, half * 0.5, C["red"])

    def draw_cross(ax, x, y, size, color):
        ax.add_line(Line2D([x - size / 2, x + size / 2], [y - size / 2, y + size / 2],
                           color=color, linewidth=2.5, zorder=5))
        ax.add_line(Line2D([x - size / 2, x + size / 2], [y + size / 2, y - size / 2],
                           color=color, linewidth=2.5, zorder=5))

    def draw_lock(ax, x, y, size, color):
        shackle_w, shackle_h = size * 0.55, size * 0.35
        body_w, body_h = size * 0.75, size * 0.55
        ax.add_patch(Arc((x, y + body_h * 0.45), shackle_w, shackle_h,
                         angle=0, theta1=0, theta2=180, color=color, lw=1.5, zorder=5))
        ax.add_patch(FancyBboxPatch((x - body_w / 2, y - body_h / 2), body_w, body_h,
                                    boxstyle="round,pad=0.005,rounding_size=0.02",
                                    facecolor=color, edgecolor=color, linewidth=0, zorder=5))
        ax.add_patch(Circle((x, y - body_h * 0.05), size * 0.08,
                            facecolor="white", edgecolor="none", zorder=6))

    def embed_img(ax, x, y, w, h, path):
        try:
            img = Image.open(path)
            ax.imshow(img, extent=[x, x + w, y, y + h], aspect="auto", zorder=3)
        except Exception:
            ax.add_patch(Rectangle((x, y), w, h, facecolor="#E5E7EB", edgecolor="#9CA3AF"))

    # --- 1. STYLE CONTROL -----------------------------------------------------
    y1 = 8.55
    style_id = block(0.7, y1, 1.5, 0.95, "Style ID\n$s$", fill="#FEF3C7", stroke=C["style_dark"])
    style_mem = FancyBboxPatch((2.5, y1), 2.0, 0.95,
                               boxstyle="round,pad=0.01,rounding_size=0.12",
                               facecolor=C["white"], edgecolor=C["style_dark"], linewidth=2.2, zorder=2)
    ax.add_patch(style_mem)
    icon_style_bars(ax, 3.5, y1 + 0.55, 1.4, 0.45, C["style"])
    ax.text(3.5, y1 + 0.22, "Style Memory", fontsize=10, weight="bold", ha="center", va="center")

    style_tok = FancyBboxPatch((4.9, y1), 1.8, 0.95,
                               boxstyle="round,pad=0.01,rounding_size=0.12",
                               facecolor=C["white"], edgecolor=C["style_dark"], linewidth=2.2, zorder=2)
    ax.add_patch(style_tok)
    icon_style_tokens(ax, 5.8, y1 + 0.55, 0.55, C["style"])
    ax.text(5.8, y1 + 0.22, "Style Tokens", fontsize=10, weight="bold", ha="center", va="center")

    style_code = block(7.2, y1, 1.6, 0.95, "style code\n$c_s$", fill="#FEF3C7", stroke=C["style_dark"])

    connect(style_id, "right", style_mem, "left", C["style_dark"])
    connect(style_mem, "right", style_tok, "left", C["style_dark"])
    connect(style_tok, "right", style_code, "left", C["style_dark"])

    # style lines down
    ax.plot([5.8, 5.8], [y1, 6.85], color=C["style_dark"], lw=2, ls="--", zorder=1)
    ax.plot([7.4, 7.4], [y1, 5.0], color=C["style_dark"], lw=2, ls="--", zorder=1)
    # global at endpoint
    ax.annotate("", xy=(15.9, 5.0), xytext=(7.4, 5.0),
                arrowprops=dict(arrowstyle="-|>", color=C["style_dark"], lw=2, ls="--"), zorder=1)

    # --- 2. MAIN INFERENCE PATH ----------------------------------------------
    y2 = 3.6

    # Content image
    img_w, img_h = 1.1, 1.1
    content = FancyBboxPatch((0.6, y2 + 1.0), img_w, img_h,
                             boxstyle="round,pad=0.01,rounding_size=0.1",
                             facecolor=C["white"], edgecolor=C["content_dark"], linewidth=2.2, zorder=2)
    ax.add_patch(content)
    embed_img(ax, 0.65, y2 + 1.05, img_w - 0.1, img_h - 0.1,
              OUT_DIR / "thumbs/content_thumb.png")
    ax.text(1.15, y2 + 0.88, "Content $x$", fontsize=10, weight="bold", ha="center", va="top")

    # VAE Encode
    vae_enc = FancyBboxPatch((2.1, y2 + 1.05), 1.3, 1.0,
                             boxstyle="round,pad=0.01,rounding_size=0.12",
                             facecolor=C["white"], edgecolor=C["content_dark"], linewidth=2.2, zorder=2)
    ax.add_patch(vae_enc)
    icon_vae(ax, 2.75, y2 + 1.55, 0.7, C["content"])
    ax.text(2.75, y2 + 1.12, "VAE Encode", fontsize=10, weight="bold", ha="center", va="center")

    # z0
    z0 = FancyBboxPatch((3.8, y2 + 1.05), 1.2, 1.0,
                        boxstyle="round,pad=0.01,rounding_size=0.12",
                        facecolor=C["white"], edgecolor=C["content_dark"], linewidth=2.2, zorder=2)
    ax.add_patch(z0)
    icon_grid(ax, 4.4, y2 + 1.55, 0.55, C["content"])
    ax.text(4.4, y2 + 1.12, "latent $z_0$", fontsize=10, weight="bold", ha="center", va="center")

    # Haar DWT
    dwt = FancyBboxPatch((5.4, y2 + 1.05), 1.4, 1.0,
                         boxstyle="round,pad=0.01,rounding_size=0.12",
                         facecolor=C["white"], edgecolor=C["content_dark"], linewidth=2.2, zorder=2)
    ax.add_patch(dwt)
    icon_dwt(ax, 6.1, y2 + 1.55, 0.55, C["content"])
    ax.text(6.1, y2 + 1.12, "Haar DWT", fontsize=10, weight="bold", ha="center", va="center")

    # LL locked
    ll = FancyBboxPatch((5.3, y2 + 2.35), 1.0, 0.55,
                        boxstyle="round,pad=0.01,rounding_size=0.1",
                        facecolor="#DBEAFE", edgecolor=C["content_dark"], linewidth=2, zorder=2)
    ax.add_patch(ll)
    ax.text(5.8, y2 + 2.62, "$v_{LL} \equiv 0$", fontsize=10, weight="bold",
            color=C["content_dark"], ha="center", va="center")
    draw_lock(ax, 5.42, y2 + 2.62, 0.14, C["content_dark"])

    # spectral H_t
    ht = FancyBboxPatch((7.1, y2 + 1.05), 1.3, 1.0,
                        boxstyle="round,pad=0.01,rounding_size=0.12",
                        facecolor="#ECFDF5", edgecolor=C["spectral_dark"], linewidth=2.2, zorder=2)
    ax.add_patch(ht)
    icon_grid(ax, 7.75, y2 + 1.55, 0.55, C["spectral"])
    ax.text(7.75, y2 + 1.12, "spectral $H_t$", fontsize=10, weight="bold",
            color=C["spectral_dark"], ha="center", va="center")

    # Shared Backbone
    backbone = FancyBboxPatch((8.8, y2 + 0.85), 2.2, 1.4,
                              boxstyle="round,pad=0.01,rounding_size=0.15",
                              facecolor="#F3E8FF", edgecolor=C["network_dark"], linewidth=2.5, zorder=2)
    ax.add_patch(backbone)
    icon_unet(ax, 9.9, y2 + 1.55, 0.9, C["network"])
    ax.text(9.9, y2 + 1.05, "Shared Backbone (x4)", fontsize=11, weight="bold",
            color=C["network_dark"], ha="center", va="center")

    # time input
    tbox = FancyBboxPatch((9.1, y2 + 2.45), 0.9, 0.5,
                          boxstyle="round,pad=0.01,rounding_size=0.1",
                          facecolor=C["white"], edgecolor=C["gray_dark"], linewidth=1.8, zorder=2)
    ax.add_patch(tbox)
    icon_clock(ax, 9.55, y2 + 2.7, 0.28, C["gray_dark"])
    ax.text(9.55, y2 + 2.9, "time $t$", fontsize=9, weight="bold", ha="center", va="bottom")

    # velocity heads
    v_lh = FancyBboxPatch((11.4, y2 + 1.55), 0.95, 0.55,
                          boxstyle="round,pad=0.01,rounding_size=0.1",
                          facecolor="#ECFDF5", edgecolor=C["spectral_dark"], linewidth=2, zorder=2)
    ax.add_patch(v_lh)
    ax.text(11.875, y2 + 1.82, "$v_{LH}$", fontsize=11, weight="bold",
            color=C["spectral_dark"], ha="center", va="center")

    v_hl = FancyBboxPatch((11.4, y2 + 0.85), 0.95, 0.55,
                          boxstyle="round,pad=0.01,rounding_size=0.1",
                          facecolor="#ECFDF5", edgecolor=C["spectral_dark"], linewidth=2, zorder=2)
    ax.add_patch(v_hl)
    ax.text(11.875, y2 + 1.12, "$v_{HL}$", fontsize=11, weight="bold",
            color=C["spectral_dark"], ha="center", va="center")

    # ODE integrator box
    ode = FancyBboxPatch((12.7, y2 + 0.6), 2.4, 1.9,
                         boxstyle="round,pad=0.01,rounding_size=0.15",
                         facecolor=C["white"], edgecolor=C["spectral_dark"], linewidth=2.2,
                         linestyle="--", zorder=2)
    ax.add_patch(ode)
    ax.text(13.9, y2 + 2.25, "Spectral ODE Integrator", fontsize=11, weight="bold",
            color=C["spectral_dark"], ha="center", va="center")
    ax.text(13.9, y2 + 1.95, "$t: 0 \\rightarrow 1$", fontsize=9,
            color=C["spectral_dark"], ha="center", va="center")
    ax.annotate("", xy=(13.4, y2 + 1.55), xytext=(14.4, y2 + 1.55),
                arrowprops=dict(arrowstyle="-|>", color=C["spectral_dark"], lw=1.8,
                                connectionstyle="arc3,rad=0.25"), zorder=1)
    ax.text(13.9, y2 + 1.2, "$H_{t+\\Delta t} = H_t + v_H \\Delta t$", fontsize=9,
            color=C["spectral_dark"], ha="center", va="center")

    # H_1
    h1 = FancyBboxPatch((15.15, y2 + 1.05), 1.0, 1.0,
                        boxstyle="round,pad=0.01,rounding_size=0.12",
                        facecolor="#ECFDF5", edgecolor=C["spectral_dark"], linewidth=2.2, zorder=2)
    ax.add_patch(h1)
    icon_grid(ax, 15.65, y2 + 1.55, 0.55, C["spectral"])
    ax.text(15.65, y2 + 1.12, "$\\hat{H}_1$", fontsize=11, weight="bold",
            color=C["spectral_dark"], ha="center", va="center")

    # iDWT
    idwt = FancyBboxPatch((16.45, y2 + 1.05), 1.0, 1.0,
                          boxstyle="round,pad=0.01,rounding_size=0.12",
                          facecolor=C["white"], edgecolor=C["content_dark"], linewidth=2.2, zorder=2)
    ax.add_patch(idwt)
    icon_dwt(ax, 16.95, y2 + 1.55, 0.5, C["content"])
    ax.text(16.95, y2 + 1.12, "iDWT", fontsize=10, weight="bold", ha="center", va="center")

    # Endpoint AdaIN/WCT
    ep = FancyBboxPatch((17.75, y2 + 1.05), 1.25, 1.0,
                        boxstyle="round,pad=0.01,rounding_size=0.12",
                        facecolor="#FEF3C7", edgecolor=C["style_dark"], linewidth=2.2, zorder=2)
    ax.add_patch(ep)
    ax.add_patch(Circle((18.375, y2 + 1.55), 0.54, facecolor="#FDE68A", edgecolor="none", alpha=0.26, zorder=1))
    ax.text(18.375, y2 + 1.55, "Endpoint", fontsize=10, weight="bold",
            color=C["style_dark"], ha="center", va="center")
    ax.text(18.375, y2 + 1.25, "AdaIN / WCT", fontsize=9,
            color=C["style_dark"], ha="center", va="center")

    # VAE Decode
    vae_dec = FancyBboxPatch((17.75, y2 - 0.1), 1.25, 0.9,
                             boxstyle="round,pad=0.01,rounding_size=0.12",
                             facecolor=C["white"], edgecolor=C["content_dark"], linewidth=2.2, zorder=2)
    ax.add_patch(vae_dec)
    icon_vae(ax, 18.375, y2 + 0.35, 0.54, C["content"])
    ax.text(18.375, y2 - 0.02, "VAE Decode", fontsize=9, weight="bold", ha="center", va="center")

    # Output image on the right, not wrapped back to the left
    out = FancyBboxPatch((19.25, y2 + 0.95), 1.05, 1.05,
                         boxstyle="round,pad=0.01,rounding_size=0.1",
                         facecolor=C["white"], edgecolor=C["style_dark"], linewidth=2.2, zorder=2)
    ax.add_patch(out)
    embed_img(ax, 19.3, y2 + 1.0, 0.95, 0.95, OUT_DIR / "thumbs/output_thumb.png")
    ax.text(19.775, y2 + 0.78, "output $\\hat{x}$", fontsize=10, weight="bold", ha="center", va="top")

    # Arrows main flow
    arrow([(1.75, y2 + 1.55), (2.1, y2 + 1.55)], color="black", lw=2)
    arrow([(3.4, y2 + 1.55), (3.8, y2 + 1.55)], color="black", lw=2)
    arrow([(5.0, y2 + 1.55), (5.4, y2 + 1.55)], color="black", lw=2)
    ax.plot([6.1, 6.1], [y2 + 2.05, y2 + 2.35], color=C["content_dark"], lw=2, zorder=1)
    arrow([(6.8, y2 + 1.55), (7.1, y2 + 1.55)], color="black", lw=2)
    arrow([(8.4, y2 + 1.55), (8.8, y2 + 1.55)], color="black", lw=2)
    ax.plot([9.55, 9.55], [y2 + 2.45, y2 + 2.25], color=C["gray_dark"], lw=1.5, zorder=1)
    arrow([(11.0, y2 + 1.82), (11.4, y2 + 1.82)], color=C["spectral_dark"], lw=2)
    arrow([(11.0, y2 + 1.12), (11.4, y2 + 1.12)], color=C["spectral_dark"], lw=2)
    arrow([(12.35, y2 + 1.82), (12.7, y2 + 1.82)], color=C["spectral_dark"], lw=2)
    arrow([(12.35, y2 + 1.12), (12.7, y2 + 1.12)], color=C["spectral_dark"], lw=2)
    arrow([(15.1, y2 + 1.55), (15.15, y2 + 1.55)], color="black", lw=2)
    arrow([(16.15, y2 + 1.55), (16.45, y2 + 1.55)], color="black", lw=2)
    arrow([(17.45, y2 + 1.55), (17.75, y2 + 1.55)], color="black", lw=2)
    ax.plot([18.375, 18.375], [y2 + 1.05, y2 + 0.8], color="black", lw=2, zorder=1)
    arrow([(19.0, y2 + 0.35), (19.25, y2 + 1.47)], color="black", lw=2)

    # LL bypass rejoins at iDWT
    arrow([(6.3, y2 + 2.35), (6.3, y2 + 2.65), (16.95, y2 + 2.65), (16.95, y2 + 2.05)],
          color=C["content_dark"], lw=1.8, dashed=True)

    # Micro-architecture inset to fill the upper-right whitespace
    micro = FancyBboxPatch((12.0, y2 + 2.02), 4.9, 0.95,
                           boxstyle="round,pad=0.01,rounding_size=0.1",
                           facecolor=C["white"], edgecolor=C["gray_dark"], linewidth=1.8,
                           linestyle="--", zorder=2)
    ax.add_patch(micro)
    ax.text(12.15, y2 + 2.84, "Micro-Architecture (One Block)", fontsize=9.5, weight="bold",
            color=C["gray_dark"], ha="left", va="center")
    xs = [12.25, 13.28, 14.38, 15.83, 16.72]
    ws = [0.88, 0.92, 1.22, 0.86, 0.42]
    labels = ["AdaLN($t$)", "Self-Attn", "ReLU$^2$ X-Attn", "tanh(g)", "FFN"]
    for idx, (x0, w0, lab) in enumerate(zip(xs, ws, labels)):
        cell = FancyBboxPatch((x0, y2 + 2.24), w0, 0.36,
                              boxstyle="round,pad=0.01,rounding_size=0.05",
                              facecolor="#F9FAFB", edgecolor=C["gray_dark"], linewidth=1.1, zorder=2)
        ax.add_patch(cell)
        ax.text(x0 + w0 / 2, y2 + 2.42, lab, fontsize=7.0, color=C["text"], ha="center", va="center")
        if idx < len(xs) - 1:
            ax.annotate("", xy=(x0 + w0 + 0.12, y2 + 2.42), xytext=(x0 + w0 + 0.02, y2 + 2.42),
                        arrowprops=dict(arrowstyle="-|>", color=C["gray_dark"], lw=1.0), zorder=1)
    ax.plot([11.0, 12.0], [y2 + 1.82, y2 + 2.18], color=C["gray"], lw=1.3, zorder=1)
    ax.plot([11.0, 12.0], [y2 + 1.18, y2 + 2.66], color=C["gray"], lw=1.3, zorder=1)
    ax.annotate("", xy=(15.0, y2 + 2.24), xytext=(5.8, 6.85),
                arrowprops=dict(arrowstyle="-|>", color=C["style_dark"], lw=1.4, ls="--"), zorder=1)

    # --- 3. TRAINING ----------------------------------------------------------
    y3 = 0.35
    xt = block(5.15, y3, 1.45, 0.9, "Mix $x_t$\n$(1-t)z_0 + tz_{target}$", fill=C["white"],
               stroke=C["train_dark"], fontsize=8.8)
    dwt_t = FancyBboxPatch((6.95, y3), 1.2, 0.9,
                           boxstyle="round,pad=0.01,rounding_size=0.12",
                           facecolor=C["white"], edgecolor=C["train_dark"], linewidth=2.2, zorder=2)
    ax.add_patch(dwt_t)
    icon_dwt(ax, 7.55, y3 + 0.55, 0.45, C["train"])
    ax.text(7.55, y3 + 0.18, "DWT", fontsize=10, weight="bold", ha="center", va="center")

    pred = block(8.55, y3, 1.6, 0.9, "Predict\n$v_{LH}, v_{HL}$", fill=C["white"],
                 stroke=C["train_dark"], fontsize=9)
    tgt = block(10.5, y3, 1.85, 0.9, "Target\n$\\Delta = \\mathrm{DWT}(z_{target}-z_0)$",
                fill=C["white"], stroke=C["train_dark"], fontsize=9)
    loss = block(12.7, y3, 3.3, 0.9,
                 "$\\mathcal{L} = \\omega_{LH}\\|v_{LH}-\\Delta_{LH}\\|_2^2 + \\omega_{HL}\\|v_{HL}-\\Delta_{HL}\\|_2^2$\n($\\omega_{LL}=0$)",
                 fill="#FEE2E2", stroke=C["train_dark"], fontsize=8.8)

    arrow([(6.6, y3 + 0.45), (6.95, y3 + 0.45)], color="black", lw=2)
    arrow([(8.15, y3 + 0.45), (8.55, y3 + 0.45)], color="black", lw=2)
    arrow([(10.15, y3 + 0.45), (10.5, y3 + 0.45)], color="black", lw=2)
    arrow([(12.35, y3 + 0.45), (12.7, y3 + 0.45)], color="black", lw=2)

    # feedback to backbone, aligned under the main trunk
    ax.plot([14.35, 14.35], [y3 + 0.9, y3 + 1.35], color=C["train_dark"], lw=2, ls="--", zorder=1)
    ax.plot([14.35, 9.9], [y3 + 1.35, y3 + 1.35], color=C["train_dark"], lw=2, ls="--", zorder=1)
    ax.annotate("", xy=(9.9, y2 + 0.85), xytext=(9.9, y3 + 1.35),
                arrowprops=dict(arrowstyle="-|>", color=C["train_dark"], lw=2, ls="--"), zorder=1)

    # Legend
    leg_y = -0.05
    ax.add_patch(Rectangle((0.6, leg_y - 0.2), 0.35, 0.12, facecolor=C["content"], edgecolor="none"))
    ax.text(1.05, leg_y - 0.14, "content", fontsize=10, color=C["text"], va="center")
    ax.add_patch(Rectangle((2.0, leg_y - 0.2), 0.35, 0.12, facecolor=C["spectral"], edgecolor="none"))
    ax.text(2.45, leg_y - 0.14, "spectral", fontsize=10, color=C["text"], va="center")
    ax.add_patch(Rectangle((3.4, leg_y - 0.2), 0.35, 0.12, facecolor=C["network"], edgecolor="none"))
    ax.text(3.85, leg_y - 0.14, "network", fontsize=10, color=C["text"], va="center")
    ax.add_patch(Rectangle((4.8, leg_y - 0.2), 0.35, 0.12, facecolor=C["style"], edgecolor="none"))
    ax.text(5.25, leg_y - 0.14, "style", fontsize=10, color=C["text"], va="center")
    ax.plot([6.5, 7.0], [leg_y - 0.14, leg_y - 0.14], color="black", lw=2)
    ax.text(7.1, leg_y - 0.14, "inference", fontsize=10, color=C["text"], va="center")
    ax.plot([8.0, 8.5], [leg_y - 0.14, leg_y - 0.14], color=C["train_dark"], lw=2, ls="--")
    ax.text(8.6, leg_y - 0.14, "training", fontsize=10, color=C["text"], va="center")

    # Caption
    ax.text(10.5, -0.42,
            "Figure 2. Overview of Spectral ODE Bridge. "
            "Content latent is decomposed by Haar DWT; LL is locked ($v_{LL}\\equiv0$) and "
            "LH/HL form the spectral state $H_t$ driven by a shared backbone with per-subband velocity heads. "
            "HH is discarded. The ODE is integrated for $K$ steps; the locked LL bypasses the ODE and "
            "rejoins at iDWT. Style is injected only at the endpoint via AdaIN/WCT ($c_s$), while style tokens "
            "condition the backbone cross-attention.",
            fontsize=10, color=C["text"], ha="center", va="top", wrap=True)

    plt.tight_layout(pad=0.2)
    png = OUT_DIR / "aaai_arch_diagram_v14_from_v7.png"
    svg = OUT_DIR / "aaai_arch_diagram_v14_from_v7.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor=C["bg"], pad_inches=0.05)
    fig.savefig(svg, format="svg", bbox_inches="tight", facecolor=C["bg"], pad_inches=0.05)
    print(f"saved {png} and {svg}")
    plt.close(fig)


if __name__ == "__main__":
    save_im()
