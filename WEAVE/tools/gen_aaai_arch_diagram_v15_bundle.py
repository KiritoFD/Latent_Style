"""AAAI architecture figure v15: fiber-bundle layout aligned to WD-VF method."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, FancyBboxPatch, Polygon, Rectangle
from matplotlib.lines import Line2D
from PIL import Image


OUT_DIR = Path("g:/GitHub/Latent_Style/SchrodingerBridge/docs/630")

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["mathtext.fontset"] = "stix"

C = {
    "bg": "#FFFFFF",
    "text": "#1F2937",
    "title": "#111827",
    "section1_bg": "#FFFBEB",
    "section1_stroke": "#D97706",
    "section2_bg": "#EFF6FF",
    "section2_stroke": "#2563EB",
    "section3_bg": "#FEF2F2",
    "section3_stroke": "#DC2626",
    "style": "#F59E0B",
    "style_dark": "#92400E",
    "content": "#3B82F6",
    "content_dark": "#1E40AF",
    "spectral": "#10B981",
    "spectral_dark": "#047857",
    "network": "#8B5CF6",
    "network_dark": "#5B21B6",
    "train_dark": "#991B1B",
    "gray": "#9CA3AF",
    "gray_dark": "#4B5563",
    "white": "#FFFFFF",
    "red": "#DC2626",
}


def block(ax, x, y, w, h, label="", *, fill=None, stroke=None, fontsize=11, bold=True, radius=0.12, lw=2.2, color=None, z=2):
    fill = fill or C["white"]
    stroke = stroke or C["text"]
    color = color or C["text"]
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.01,rounding_size={radius}",
        facecolor=fill,
        edgecolor=stroke,
        linewidth=lw,
        zorder=z,
    )
    ax.add_patch(patch)
    if label:
        ax.text(x + w / 2, y + h / 2, label, fontsize=fontsize, weight="bold" if bold else "normal", color=color, ha="center", va="center", zorder=z + 1)
    return patch


def arrow(ax, pts, *, color="black", lw=2, dashed=False, z=1):
    ls = "--" if dashed else "-"
    for i in range(len(pts) - 1):
        ax.annotate(
            "",
            xy=pts[i + 1],
            xytext=pts[i],
            arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls, connectionstyle="arc3,rad=0"),
            zorder=z,
        )


def embed_img(ax, x, y, w, h, path):
    try:
        img = Image.open(path)
        ax.imshow(img, extent=[x, x + w, y, y + h], aspect="auto", zorder=3)
    except Exception:
        ax.add_patch(Rectangle((x, y), w, h, facecolor="#E5E7EB", edgecolor="#9CA3AF", zorder=3))


def icon_vae(ax, x, y, size, color):
    w, h = size * 1.0, size * 0.7
    pts = [(x - w * 0.5, y - h * 0.5), (x + w * 0.15, y - h * 0.35), (x + w * 0.15, y + h * 0.35), (x - w * 0.5, y + h * 0.5)]
    ax.add_patch(Polygon(pts, closed=True, facecolor=color, edgecolor="none", zorder=3))
    pts2 = [(x + w * 0.15, y - h * 0.35), (x + w * 0.5, y), (x + w * 0.15, y + h * 0.35)]
    ax.add_patch(Polygon(pts2, closed=True, facecolor=color, edgecolor="none", alpha=0.7, zorder=3))


def icon_grid(ax, x, y, size, color, rows=4, cols=4):
    s = size / rows * 0.82
    for i in range(rows):
        for j in range(cols):
            xx = x - size / 2 + j * size / cols + 0.02
            yy = y - size / 2 + i * size / rows + 0.02
            alpha = 0.25 + 0.15 * ((i + j) % 4)
            ax.add_patch(Rectangle((xx, yy), s, s, facecolor=color, alpha=alpha, edgecolor="white", lw=0.5, zorder=3))


def icon_style_tokens(ax, x, y, size, color):
    s = size / 4 * 0.82
    for i in range(4):
        for j in range(4):
            xx = x - size / 2 + j * size / 4 + 0.02
            yy = y - size / 2 + i * size / 4 + 0.02
            alpha = 0.3 + 0.15 * ((i + j) % 4)
            ax.add_patch(Rectangle((xx, yy), s, s, facecolor=color, alpha=alpha, edgecolor="white", lw=0.5, zorder=3))


def icon_style_bars(ax, x, y, w, h, color):
    bw = w / 11
    heights = [0.35, 0.65, 0.52, 0.72, 0.48]
    for i, hh in enumerate(heights):
        ax.add_patch(Rectangle((x - w / 2 + (2 * i + 1) * bw, y - h * hh / 2), bw, h * hh, facecolor=color, edgecolor="white", lw=0.5, zorder=3))


def icon_clock(ax, x, y, size, color):
    ax.add_patch(Circle((x, y), size / 2, facecolor="white", edgecolor=color, linewidth=2, zorder=3))
    ax.plot([x, x], [y, y + size * 0.25], color=color, lw=2, zorder=4)
    ax.plot([x, x + size * 0.18], [y, y - size * 0.05], color=color, lw=2, zorder=4)


def icon_unet(ax, x, y, size, color):
    w, h = size * 0.9, size * 0.55
    enc = FancyBboxPatch((x - w / 2, y - h / 2), w * 0.35, h, boxstyle="round,pad=0.01,rounding_size=0.04", facecolor=color, edgecolor=color, zorder=3)
    dec = FancyBboxPatch((x + w * 0.03, y - h / 2), w * 0.35, h, boxstyle="round,pad=0.01,rounding_size=0.04", facecolor=color, edgecolor=color, alpha=0.7, zorder=3)
    ax.add_patch(enc)
    ax.add_patch(dec)
    ax.plot([x - w * 0.13, x + w * 0.13], [y + h * 0.05, y + h * 0.05], color=color, lw=2, zorder=4)


def icon_dwt(ax, x, y, size):
    half = size / 2
    labels = [("LL", C["content"]), ("LH", C["spectral"]), ("HL", C["spectral"]), ("HH", C["gray"])]
    positions = [(0, 1), (1, 1), (0, 0), (1, 0)]
    for (lab, col), (ix, iy) in zip(labels, positions):
        xx = x - half + ix * half
        yy = y - half + iy * half
        ax.add_patch(Rectangle((xx + 0.02, yy + 0.02), half - 0.04, half - 0.04, facecolor=col, alpha=0.2, edgecolor=col, lw=1, zorder=3))
        ax.text(xx + half / 2, yy + half / 2, lab, fontsize=8, weight="bold", color=col, ha="center", va="center", zorder=4)
    hx = x + half / 2
    hy = y - half / 2
    s = half * 0.45
    ax.add_line(Line2D([hx - s, hx + s], [hy - s, hy + s], color=C["red"], linewidth=2.4, zorder=5))
    ax.add_line(Line2D([hx - s, hx + s], [hy + s, hy - s], color=C["red"], linewidth=2.4, zorder=5))


def draw_lock(ax, x, y, size, color):
    shackle_w, shackle_h = size * 0.55, size * 0.35
    body_w, body_h = size * 0.75, size * 0.55
    ax.add_patch(Arc((x, y + body_h * 0.45), shackle_w, shackle_h, angle=0, theta1=0, theta2=180, color=color, lw=1.5, zorder=5))
    ax.add_patch(FancyBboxPatch((x - body_w / 2, y - body_h / 2), body_w, body_h, boxstyle="round,pad=0.005,rounding_size=0.02", facecolor=color, edgecolor=color, linewidth=0, zorder=5))
    ax.add_patch(Circle((x, y - body_h * 0.05), size * 0.08, facecolor="white", edgecolor="none", zorder=6))


def fiber_stack(ax, x, y, w, h, title, entries, color_fill, color_edge, text_color):
    outer = block(ax, x, y, w, h, "", fill=color_fill, stroke=color_edge, radius=0.15, lw=2.0)
    ax.text(x + w / 2, y + h - 0.18, title, fontsize=10, weight="bold", color=text_color, ha="center", va="top", zorder=4)
    row_h = (h - 0.48) / len(entries)
    for idx, entry in enumerate(entries):
        yy = y + h - 0.3 - (idx + 1) * row_h
        inner = Rectangle((x + 0.16, yy + 0.04), w - 0.32, row_h - 0.08, facecolor=C["white"], edgecolor=color_edge, lw=1.0, alpha=0.55, zorder=3)
        ax.add_patch(inner)
        ax.text(x + w / 2, yy + row_h / 2, entry, fontsize=9, weight="bold", color=text_color, ha="center", va="center", zorder=4)
    return outer


def save_im():
    fig, ax = plt.subplots(figsize=(20.8, 10), facecolor=C["bg"])
    ax.set_xlim(0, 21.5)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_aspect("equal")

    sec_h = [1.5, 4.6, 3.1]
    sec_y = [8.2, 3.2, -0.1]
    sec_labels = [
        ("1. STYLE CONTROL", C["section1_bg"], C["section1_stroke"]),
        ("2. MAIN INFERENCE PATH", C["section2_bg"], C["section2_stroke"]),
        ("3. TRAINING (SUPERVISION & ENDPOINT)", C["section3_bg"], C["section3_stroke"]),
    ]
    for (label, fill, stroke), y, h in zip(sec_labels, sec_y, sec_h):
        bg = FancyBboxPatch((0.25, y), 20.75, h, boxstyle="round,pad=0.02,rounding_size=0.2", facecolor=fill, edgecolor=stroke, linewidth=2.5, zorder=0)
        ax.add_patch(bg)
        ax.text(0.45, y + h - 0.28, label, fontsize=15, weight="bold", color=stroke, va="top", ha="left")

    ax.text(11.0, 9.82, "Spectral ODE Bridge", fontsize=24, weight="bold", color=C["title"], ha="center", va="top")

    # Style control
    y1 = 8.55
    style_id = block(ax, 0.72, y1, 1.5, 0.95, "Style ID\n$s$", fill="#FEF3C7", stroke=C["style_dark"], fontsize=11)
    style_mem = block(ax, 2.55, y1, 2.0, 0.95, "", fill=C["white"], stroke=C["style_dark"], fontsize=10)
    icon_style_bars(ax, 3.55, y1 + 0.56, 1.35, 0.45, C["style"])
    ax.text(3.55, y1 + 0.21, "Style Memory", fontsize=10, weight="bold", ha="center", va="center")
    style_tok = block(ax, 4.98, y1, 1.8, 0.95, "", fill=C["white"], stroke=C["style_dark"], fontsize=10)
    icon_style_tokens(ax, 5.88, y1 + 0.56, 0.56, C["style"])
    ax.text(5.88, y1 + 0.21, "Style Tokens", fontsize=10, weight="bold", ha="center", va="center")
    style_code = block(ax, 7.25, y1, 1.6, 0.95, "style code\n$c_s$", fill="#FEF3C7", stroke=C["style_dark"], fontsize=11)
    arrow(ax, [(2.22, y1 + 0.48), (2.55, y1 + 0.48)], color=C["style_dark"], lw=2)
    arrow(ax, [(4.55, y1 + 0.48), (4.98, y1 + 0.48)], color=C["style_dark"], lw=2)
    arrow(ax, [(6.78, y1 + 0.48), (7.25, y1 + 0.48)], color=C["style_dark"], lw=2)

    # Style conditioning lines
    ax.plot([5.88, 5.88], [y1, 7.0], color=C["style_dark"], lw=2, ls="--", zorder=1)
    ax.plot([7.45, 7.45], [y1, 4.95], color=C["style_dark"], lw=2, ls="--", zorder=1)

    # Main path: left to right
    y2 = 3.6
    img_w, img_h = 1.08, 1.08
    content = block(ax, 0.6, y2 + 1.0, img_w, img_h, "", fill=C["white"], stroke=C["content_dark"], radius=0.1)
    embed_img(ax, 0.65, y2 + 1.05, img_w - 0.1, img_h - 0.1, OUT_DIR / "thumbs/content_thumb.png")
    ax.text(1.14, y2 + 0.86, "Content $x$", fontsize=10, weight="bold", ha="center", va="top")

    vae_enc = block(ax, 2.08, y2 + 1.05, 1.35, 1.0, "", fill=C["white"], stroke=C["content_dark"])
    icon_vae(ax, 2.76, y2 + 1.57, 0.68, C["content"])
    ax.text(2.76, y2 + 1.13, "VAE Encode", fontsize=10, weight="bold", ha="center", va="center")

    z0 = block(ax, 3.75, y2 + 1.05, 1.25, 1.0, "", fill=C["white"], stroke=C["content_dark"])
    icon_grid(ax, 4.38, y2 + 1.57, 0.55, C["content"])
    ax.text(4.38, y2 + 1.13, "latent $z_0$", fontsize=10, weight="bold", ha="center", va="center")

    dwt = block(ax, 5.35, y2 + 1.0, 1.45, 1.1, "", fill=C["white"], stroke=C["content_dark"])
    icon_dwt(ax, 6.08, y2 + 1.56, 0.58)
    ax.text(6.08, y2 + 1.08, "Haar DWT", fontsize=10, weight="bold", ha="center", va="center")

    arrow(ax, [(1.68, y2 + 1.54), (2.08, y2 + 1.54)], color="black", lw=2)
    arrow(ax, [(3.43, y2 + 1.54), (3.75, y2 + 1.54)], color="black", lw=2)
    arrow(ax, [(5.0, y2 + 1.54), (5.35, y2 + 1.54)], color="black", lw=2)

    # Fiber bundle decomposition
    ll_lane = block(ax, 6.95, y2 + 2.18, 1.45, 0.55, r"$\ell_t$ (LL)", fill="#DBEAFE", stroke=C["content_dark"], fontsize=10, color=C["content_dark"])
    draw_lock(ax, 7.12, y2 + 2.45, 0.14, C["content_dark"])
    ax.text(7.95, y2 + 2.98, "base manifold", fontsize=9.5, color=C["content_dark"], weight="bold", ha="center")
    ax.text(7.95, y2 + 2.83, r"$w_\ell=0,\ \pi(z_t)=\ell_t$", fontsize=8.7, color=C["content_dark"], ha="center")

    fibers = fiber_stack(
        ax,
        6.78,
        y2 + 0.55,
        1.7,
        1.95,
        r"Fiber Bundle $H_t$",
        [r"$h_{1,t}$", r"$h_{2,t}$", r"$h_{3,t}$"],
        "#ECFDF5",
        C["spectral_dark"],
        C["spectral_dark"],
    )

    # DWT fan-out
    ax.plot([6.08, 6.08], [y2 + 2.1, y2 + 2.18], color=C["content_dark"], lw=2, zorder=1)
    arrow(ax, [(6.78, y2 + 1.54), (6.78, y2 + 1.54), (6.78, y2 + 1.54)], color="black", lw=0)  # no-op for stable locals
    arrow(ax, [(6.8, y2 + 1.54), (6.78, y2 + 1.54)], color="black", lw=0)
    arrow(ax, [(6.8, y2 + 1.54), (6.78, y2 + 1.5)], color="black", lw=0)
    arrow(ax, [(6.8, y2 + 1.54), (6.78, y2 + 1.52)], color="black", lw=0)
    arrow(ax, [(6.8, y2 + 1.54), (6.78, y2 + 1.5)], color="black", lw=0)
    arrow(ax, [(6.8, y2 + 1.54), (6.78, y2 + 1.5)], color="black", lw=0)
    ax.plot([6.8, 6.95], [y2 + 1.82, y2 + 2.18], color=C["content_dark"], lw=2, zorder=1)
    ax.plot([6.8, 6.78], [y2 + 1.36, y2 + 1.53], color=C["spectral_dark"], lw=2, zorder=1)

    # Expanded backbone
    backbone = block(ax, 9.0, y2 + 0.72, 3.05, 2.22, "", fill="#F3E8FF", stroke=C["network_dark"], radius=0.16, lw=2.5)
    ax.text(10.53, y2 + 2.72, "Shared Backbone", fontsize=11, weight="bold", color=C["network_dark"], ha="center", va="center")
    ax.text(10.53, y2 + 2.5, r"routed on $(h_1,h_2,h_3)$", fontsize=8.7, color=C["network_dark"], ha="center")
    inner_specs = [
        (9.22, y2 + 1.93, 1.24, 0.42, r"AdaLN($t$)"),
        (10.63, y2 + 1.93, 1.18, 0.42, "Self-Attn"),
        (9.22, y2 + 1.28, 1.82, 0.5, r"Routed X-Attn"),
        (11.18, y2 + 1.28, 0.62, 0.5, r"gate"),
        (9.62, y2 + 0.66, 1.95, 0.44, "FFN + Res"),
    ]
    for x, y, w, h, label in inner_specs:
        block(ax, x, y, w, h, label, fill=C["white"], stroke=C["network_dark"], fontsize=8.3, radius=0.06, lw=1.3, color=C["network_dark"])
    icon_unet(ax, 10.5, y2 + 1.58, 0.86, C["network"])
    arrow(ax, [(8.48, y2 + 1.55), (9.0, y2 + 1.55)], color="black", lw=2)
    ax.plot([9.55, 9.55], [y2 + 2.45, y2 + 2.37], color=C["gray_dark"], lw=1.5, zorder=1)
    tbox = block(ax, 9.1, y2 + 2.53, 0.92, 0.5, "", fill=C["white"], stroke=C["gray_dark"], fontsize=9, radius=0.1, lw=1.8)
    icon_clock(ax, 9.56, y2 + 2.78, 0.28, C["gray_dark"])
    ax.text(9.56, y2 + 2.98, "time $t$", fontsize=9, weight="bold", ha="center", va="bottom")

    # Routed style query
    ax.annotate("", xy=(10.1, y2 + 1.58), xytext=(5.88, 7.0), arrowprops=dict(arrowstyle="-|>", color=C["style_dark"], lw=1.5, ls="--"), zorder=1)
    ax.text(11.05, y2 + 1.85, r"$q=Q[h_1;h_2;h_3]$", fontsize=8.7, color=C["style_dark"], ha="left")

    # Head bank
    heads = fiber_stack(
        ax,
        12.28,
        y2 + 0.78,
        1.2,
        2.15,
        "Fiber Heads",
        [r"$v_1$", r"$v_2$", r"$h_3$: endpoint"],
        "#ECFDF5",
        C["spectral_dark"],
        C["spectral_dark"],
    )
    # lighten last row meaning no learned head
    ax.add_patch(Rectangle((12.44, y2 + 0.98), 0.88, 0.46, facecolor="#F3F4F6", edgecolor=C["gray_dark"], lw=1.0, alpha=0.9, zorder=4))
    ax.text(12.88, y2 + 1.21, r"$h_3$ endpoint", fontsize=8.0, color=C["gray_dark"], ha="center", va="center", zorder=5)
    arrow(ax, [(12.05, y2 + 1.92), (12.28, y2 + 1.92)], color=C["spectral_dark"], lw=2)
    arrow(ax, [(12.05, y2 + 1.26), (12.28, y2 + 1.26)], color=C["spectral_dark"], lw=2)

    # ODE on active fibers only
    ode = block(ax, 13.88, y2 + 0.92, 2.5, 1.95, "", fill=C["white"], stroke=C["spectral_dark"], radius=0.15, lw=2.2)
    ax.text(15.13, y2 + 2.65, "Spectral ODE Integrator", fontsize=11, weight="bold", color=C["spectral_dark"], ha="center")
    ax.text(15.13, y2 + 2.37, r"$t:0\rightarrow1$", fontsize=9, color=C["spectral_dark"], ha="center")
    ax.annotate("", xy=(14.58, y2 + 1.62), xytext=(15.55, y2 + 1.62), arrowprops=dict(arrowstyle="-|>", color=C["spectral_dark"], lw=1.8, connectionstyle="arc3,rad=0.25"), zorder=1)
    ax.text(15.13, y2 + 1.15, r"$H_{t+\Delta t}=H_t+v_H\Delta t$", fontsize=9, color=C["spectral_dark"], ha="center")
    arrow(ax, [(13.48, y2 + 1.92), (13.88, y2 + 1.92)], color=C["spectral_dark"], lw=2)
    arrow(ax, [(13.48, y2 + 1.26), (13.88, y2 + 1.26)], color=C["spectral_dark"], lw=2)

    # Endpoint fibers and fiber-preserving WCT
    hhat = fiber_stack(
        ax,
        16.65,
        y2 + 0.88,
        1.25,
        2.02,
        r"$\hat H_1$",
        [r"$\hat h_1$", r"$\hat h_2$", r"$\hat h_3$"],
        "#ECFDF5",
        C["spectral_dark"],
        C["spectral_dark"],
    )
    arrow(ax, [(16.38, y2 + 1.92), (16.65, y2 + 1.92)], color="black", lw=2)
    arrow(ax, [(16.38, y2 + 1.26), (16.65, y2 + 1.26)], color="black", lw=2)
    # h3 bypass to endpoint stack
    arrow(ax, [(8.48, y2 + 0.92), (16.65, y2 + 0.92)], color=C["gray_dark"], lw=1.5, dashed=True)
    ax.text(12.55, y2 + 0.78, r"$h_3$: endpoint-only fiber", fontsize=8.4, color=C["gray_dark"], ha="center")

    wct = block(ax, 18.0, y2 + 1.03, 1.22, 1.72, "", fill="#FEF3C7", stroke=C["style_dark"], radius=0.12, lw=2.2)
    ax.add_patch(Circle((18.61, y2 + 1.89), 0.55, facecolor="#FDE68A", edgecolor="none", alpha=0.28, zorder=1))
    ax.text(18.61, y2 + 2.05, "Fiber WCT", fontsize=10, weight="bold", color=C["style_dark"], ha="center")
    ax.text(18.61, y2 + 1.72, r"$T_1,T_2,T_3$", fontsize=9, color=C["style_dark"], ha="center")
    ax.text(18.61, y2 + 1.39, r"$\ell$ fixed", fontsize=8.7, color=C["content_dark"], ha="center")
    # style code to endpoint
    arrow(ax, [(7.45, 4.95), (18.61, 4.95)], color=C["style_dark"], lw=2, dashed=True)
    ax.text(16.95, 5.08, "endpoint style injection", fontsize=8.8, color=C["style_dark"], ha="center")
    # base lane through endpoint then to iDWT
    arrow(ax, [(8.4, y2 + 2.46), (18.0, y2 + 2.46), (19.55, y2 + 2.46)], color=C["content_dark"], lw=1.8, dashed=True)
    # fibers to endpoint
    arrow(ax, [(17.9, y2 + 1.86), (18.0, y2 + 1.86)], color="black", lw=2)
    arrow(ax, [(17.9, y2 + 1.24), (18.0, y2 + 1.24)], color="black", lw=2)

    idwt = block(ax, 19.42, y2 + 1.05, 0.92, 1.0, "", fill=C["white"], stroke=C["content_dark"], fontsize=10)
    icon_dwt(ax, 19.88, y2 + 1.55, 0.46)
    ax.text(19.88, y2 + 1.12, "iDWT", fontsize=10, weight="bold", ha="center", va="center")

    vae_dec = block(ax, 19.15, y2 + 0.0, 1.18, 0.9, "", fill=C["white"], stroke=C["content_dark"], fontsize=9)
    icon_vae(ax, 19.74, y2 + 0.46, 0.5, C["content"])
    ax.text(19.74, y2 + 0.08, "VAE Decode", fontsize=9, weight="bold", ha="center", va="center")

    out = block(ax, 20.5, y2 + 0.98, 0.92, 1.05, "", fill=C["white"], stroke=C["style_dark"], radius=0.1)
    embed_img(ax, 20.54, y2 + 1.03, 0.84, 0.95, OUT_DIR / "thumbs/output_thumb.png")
    ax.text(20.96, y2 + 0.77, "output $\\hat{x}$", fontsize=10, weight="bold", ha="center", va="top")
    arrow(ax, [(19.22, y2 + 1.86), (19.42, y2 + 1.86)], color="black", lw=2)
    ax.plot([19.74, 19.74], [y2 + 1.03, y2 + 0.9], color="black", lw=2, zorder=1)
    arrow(ax, [(20.33, y2 + 0.45), (20.5, y2 + 1.51)], color="black", lw=2)

    # Training panel centered under method trunk
    y3 = 0.35
    xt = block(ax, 5.45, y3, 1.5, 0.9, "Mix $x_t$\n$(1-t)z_0+tz_{target}$", fill=C["white"], stroke=C["train_dark"], fontsize=8.8)
    dwt_t = block(ax, 7.3, y3, 1.2, 0.9, "", fill=C["white"], stroke=C["train_dark"], fontsize=10)
    icon_dwt(ax, 7.9, y3 + 0.55, 0.45)
    ax.text(7.9, y3 + 0.18, "DWT", fontsize=10, weight="bold", ha="center", va="center")
    pred = block(ax, 8.92, y3, 1.65, 0.9, "Predict\n$v_1, v_2$", fill=C["white"], stroke=C["train_dark"], fontsize=9)
    tgt = block(ax, 10.95, y3, 1.95, 0.9, "Target\n$u_i=\\mathrm{DWT}(z_{target}-z_0)$", fill=C["white"], stroke=C["train_dark"], fontsize=8.8)
    loss = block(ax, 13.28, y3, 3.55, 0.9, "$\\mathcal{L}_{\\mathrm{WD-VF}}$\n$w_\\ell=0$, no $h_3$ head", fill="#FEE2E2", stroke=C["train_dark"], fontsize=9)
    arrow(ax, [(6.95, y3 + 0.45), (7.3, y3 + 0.45)], color="black", lw=2)
    arrow(ax, [(8.5, y3 + 0.45), (8.92, y3 + 0.45)], color="black", lw=2)
    arrow(ax, [(10.57, y3 + 0.45), (10.95, y3 + 0.45)], color="black", lw=2)
    arrow(ax, [(12.9, y3 + 0.45), (13.28, y3 + 0.45)], color="black", lw=2)
    ax.plot([15.05, 15.05], [y3 + 0.9, y3 + 1.32], color=C["train_dark"], lw=2, ls="--", zorder=1)
    ax.plot([15.05, 10.48], [y3 + 1.32, y3 + 1.32], color=C["train_dark"], lw=2, ls="--", zorder=1)
    ax.annotate("", xy=(10.48, y2 + 0.78), xytext=(10.48, y3 + 1.32), arrowprops=dict(arrowstyle="-|>", color=C["train_dark"], lw=2, ls="--"), zorder=1)

    # Caption
    ax.text(
        10.5,
        -0.42,
        "Figure 2. Fiber-bundle view of WD-VF. Haar DWT decomposes each latent into a base manifold "
        "$\\ell_t$ and fiber coordinates $(h_{1,t},h_{2,t},h_{3,t})$. The base lane is locked "
        "($w_\\ell=0$), style queries are routed through the fiber coordinates only, and the shared "
        "backbone predicts active velocities only for $h_1$ and $h_2$. The $h_3$ fiber bypasses transport "
        "and is stylized only at the endpoint. Fiber WCT acts on $(h_1,h_2,h_3)$ while preserving $\\ell_t$, "
        "after which iDWT reconstructs the latent for decoding.",
        fontsize=10,
        color=C["text"],
        ha="center",
        va="top",
        wrap=True,
    )

    plt.tight_layout(pad=0.2)
    png = OUT_DIR / "aaai_arch_diagram_v15_bundle.png"
    svg = OUT_DIR / "aaai_arch_diagram_v15_bundle.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor=C["bg"], pad_inches=0.05)
    fig.savefig(svg, format="svg", bbox_inches="tight", facecolor=C["bg"], pad_inches=0.05)
    print(f"saved {png} and {svg}")
    plt.close(fig)


if __name__ == "__main__":
    save_im()
