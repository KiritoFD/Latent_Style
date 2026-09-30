"""AAAI architecture figure v16: staggered pathway layout for WD-VF."""

from matplotlib.patches import Circle, FancyBboxPatch
import matplotlib.pyplot as plt

from gen_aaai_arch_diagram_v15_bundle import (
    C,
    OUT_DIR,
    arrow,
    block,
    draw_lock,
    embed_img,
    fiber_stack,
    icon_clock,
    icon_dwt,
    icon_grid,
    icon_style_bars,
    icon_style_tokens,
    icon_unet,
    icon_vae,
)


def save_im():
    fig, ax = plt.subplots(figsize=(24.0, 9.2), facecolor=C["bg"])
    ax.set_xlim(0, 28.2)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_aspect("equal")

    # Three bands without titles.
    band_specs = [
        (8.75, 0.82, C["section1_bg"], C["section1_stroke"]),
        (2.15, 5.95, C["section2_bg"], C["section2_stroke"]),
        (0.72, 0.86, C["section3_bg"], C["section3_stroke"]),
    ]
    for y, h, fill, stroke in band_specs:
        ax.add_patch(
            FancyBboxPatch(
                (0.25, y),
                27.65,
                h,
                boxstyle="round,pad=0.02,rounding_size=0.18",
                facecolor=fill,
                edgecolor=stroke,
                linewidth=2.5,
                zorder=0,
            )
        )

    ax.text(14.1, 9.83, "Spectral ODE Bridge", fontsize=22, weight="bold", color=C["title"], ha="center", va="top")

    # Style row.
    y1 = 8.79
    block(ax, 2.85, y1, 1.55, 0.70, "Style ID\n$s$", fill="#FEF3C7", stroke=C["style_dark"], fontsize=10.2, lw=2.0)
    block(ax, 4.85, y1, 2.18, 0.70, "", fill=C["white"], stroke=C["style_dark"], lw=2.0)
    icon_style_bars(ax, 5.94, y1 + 0.42, 1.40, 0.34, C["style"])
    ax.text(5.94, y1 + 0.14, "Style Memory", fontsize=9.8, weight="bold", ha="center", va="center")
    block(ax, 7.55, y1, 1.95, 0.70, "", fill=C["white"], stroke=C["style_dark"], lw=2.0)
    icon_style_tokens(ax, 8.52, y1 + 0.42, 0.48, C["style"])
    ax.text(8.52, y1 + 0.14, "Style Tokens", fontsize=9.8, weight="bold", ha="center", va="center")
    block(ax, 10.05, y1, 1.78, 0.70, "style code\n$c_s$", fill="#FEF3C7", stroke=C["style_dark"], fontsize=10.2, lw=2.0)
    arrow(ax, [(4.40, y1 + 0.35), (4.85, y1 + 0.35)], color=C["style_dark"], lw=2)
    arrow(ax, [(7.03, y1 + 0.35), (7.55, y1 + 0.35)], color=C["style_dark"], lw=2)
    arrow(ax, [(9.50, y1 + 0.35), (10.05, y1 + 0.35)], color=C["style_dark"], lw=2)
    ax.plot([8.52, 8.52], [y1, 7.25], color=C["style_dark"], lw=2, ls="--", zorder=1)
    ax.plot([10.94, 10.94], [y1, 4.95], color=C["style_dark"], lw=2, ls="--", zorder=1)

    # Left trunk.
    row_y = 4.65
    img_w = 1.12
    block(ax, 0.85, row_y, img_w, img_w, "", fill=C["white"], stroke=C["content_dark"], radius=0.1)
    embed_img(ax, 0.90, row_y + 0.05, img_w - 0.10, img_w - 0.10, OUT_DIR / "thumbs/content_thumb.png")
    ax.text(1.41, row_y - 0.14, "Content $x$", fontsize=10.4, weight="bold", ha="center", va="top")

    block(ax, 2.55, row_y + 0.05, 1.55, 1.00, "", fill=C["white"], stroke=C["content_dark"])
    icon_vae(ax, 3.33, row_y + 0.58, 0.72, C["content"])
    ax.text(3.33, row_y + 0.13, "VAE Encode", fontsize=10.1, weight="bold", ha="center", va="center")

    block(ax, 4.55, row_y + 0.05, 1.40, 1.00, "", fill=C["white"], stroke=C["content_dark"])
    icon_grid(ax, 5.25, row_y + 0.58, 0.58, C["content"])
    ax.text(5.25, row_y + 0.13, "latent $z_0$", fontsize=10.1, weight="bold", ha="center", va="center")

    block(ax, 6.50, row_y, 1.55, 1.10, "", fill=C["white"], stroke=C["content_dark"])
    icon_dwt(ax, 7.28, row_y + 0.57, 0.60)
    ax.text(7.28, row_y + 0.10, "Haar DWT", fontsize=10.1, weight="bold", ha="center", va="center")

    arrow(ax, [(1.97, row_y + 0.55), (2.55, row_y + 0.55)], color="black", lw=2)
    arrow(ax, [(4.10, row_y + 0.55), (4.55, row_y + 0.55)], color="black", lw=2)
    arrow(ax, [(5.95, row_y + 0.55), (6.50, row_y + 0.55)], color="black", lw=2)

    # Top pathway: base manifold.
    block(ax, 8.65, 6.45, 1.95, 0.62, r"$\ell_t$  (LL)", fill="#DBEAFE", stroke=C["content_dark"], fontsize=10.3, color=C["content_dark"])
    draw_lock(ax, 8.90, 6.75, 0.15, C["content_dark"])
    block(ax, 11.05, 6.46, 1.55, 0.60, r"$w_\ell = 0$", fill="#DBEAFE", stroke=C["content_dark"], fontsize=9.6, color=C["content_dark"], lw=2.0)
    ax.text(9.62, 7.38, "base manifold", fontsize=10.0, color=C["content_dark"], weight="bold", ha="center")
    ax.text(9.62, 7.18, r"$\pi(z_t)=\ell_t$", fontsize=8.8, color=C["content_dark"], ha="center")
    ax.plot([7.28, 7.28], [row_y + 1.10, 6.45], color=C["content_dark"], lw=2, zorder=1)
    arrow(ax, [(10.60, 6.76), (11.05, 6.76)], color=C["content_dark"], lw=2)

    # Middle pathway: active fibers.
    fibers = fiber_stack(
        ax,
        8.25,
        4.15,
        2.15,
        1.90,
        r"Fiber Bundle $H_t$",
        [r"$h_{1,t}$", r"$h_{2,t}$"],
        "#ECFDF5",
        C["spectral_dark"],
        C["spectral_dark"],
    )
    ax.plot([7.85, 8.25], [row_y + 0.55, 5.10], color=C["spectral_dark"], lw=2, zorder=1)

    backbone = block(ax, 11.10, 3.95, 4.55, 2.70, "", fill="#F3E8FF", stroke=C["network_dark"], radius=0.18, lw=2.5)
    ax.text(13.38, 6.36, "Shared Backbone", fontsize=11.5, weight="bold", color=C["network_dark"], ha="center")
    ax.text(13.38, 6.14, r"routed on $(h_1,h_2,h_3)$", fontsize=8.9, color=C["network_dark"], ha="center")
    inner_specs = [
        (11.35, 5.45, 1.55, 0.50, r"AdaLN($t$)"),
        (13.15, 5.45, 1.55, 0.50, "Self-Attn"),
        (11.35, 4.55, 2.40, 0.66, r"Routed X-Attn"),
        (14.05, 4.55, 0.92, 0.66, r"gate"),
        (11.75, 4.00, 2.90, 0.42, "FFN + Res"),
    ]
    for x, y, w, h, label in inner_specs:
        block(ax, x, y, w, h, label, fill=C["white"], stroke=C["network_dark"], fontsize=8.8, radius=0.06, lw=1.3, color=C["network_dark"])
    icon_unet(ax, 13.35, 5.12, 0.98, C["network"])
    arrow(ax, [(10.40, 5.08), (11.10, 5.08)], color="black", lw=2)
    block(ax, 11.38, 6.18, 0.98, 0.52, "", fill=C["white"], stroke=C["gray_dark"], fontsize=9, radius=0.1, lw=1.8)
    icon_clock(ax, 11.87, 6.44, 0.28, C["gray_dark"])
    ax.text(11.87, 6.66, "time $t$", fontsize=9, weight="bold", ha="center")

    heads = fiber_stack(
        ax,
        16.25,
        4.20,
        1.60,
        1.80,
        "Fiber Heads",
        [r"$v_1$", r"$v_2$"],
        "#ECFDF5",
        C["spectral_dark"],
        C["spectral_dark"],
    )
    arrow(ax, [(15.65, 5.38), (16.25, 5.38)], color=C["spectral_dark"], lw=2)
    arrow(ax, [(15.65, 4.82), (16.25, 4.82)], color=C["spectral_dark"], lw=2)

    ode = block(ax, 18.35, 3.95, 3.55, 2.80, "", fill=C["white"], stroke=C["spectral_dark"], radius=0.16, lw=2.2)
    ax.text(20.12, 6.38, "Spectral ODE Integrator", fontsize=11.5, weight="bold", color=C["spectral_dark"], ha="center")
    ax.text(20.12, 6.05, r"active fibers: $(h_1,h_2)$", fontsize=9.0, color=C["spectral_dark"], ha="center")
    block(ax, 18.75, 5.35, 1.15, 0.42, r"$k_1=v(H_t,t)$", fill="#F9FFFC", stroke=C["spectral_dark"], fontsize=8.2, lw=1.2, color=C["spectral_dark"])
    block(ax, 20.30, 5.35, 1.35, 0.42, r"$k_2=v(H_t+\Delta t\,k_1)$", fill="#F9FFFC", stroke=C["spectral_dark"], fontsize=8.0, lw=1.2, color=C["spectral_dark"])
    ax.annotate("", xy=(19.55, 4.95), xytext=(20.65, 4.95), arrowprops=dict(arrowstyle="-|>", color=C["spectral_dark"], lw=1.8, connectionstyle="arc3,rad=0.25"), zorder=1)
    ax.text(20.12, 4.22, r"$H_{t+\Delta t}=H_t+\frac{\Delta t}{2}(k_1+k_2)$", fontsize=8.9, color=C["spectral_dark"], ha="center")
    arrow(ax, [(17.85, 5.38), (18.35, 5.38)], color=C["spectral_dark"], lw=2)
    arrow(ax, [(17.85, 4.82), (18.35, 4.82)], color=C["spectral_dark"], lw=2)

    hhat = fiber_stack(
        ax,
        22.35,
        4.20,
        1.45,
        1.80,
        r"$\hat H_1$",
        [r"$\hat h_1$", r"$\hat h_2$"],
        "#ECFDF5",
        C["spectral_dark"],
        C["spectral_dark"],
    )
    arrow(ax, [(21.90, 5.38), (22.35, 5.38)], color="black", lw=2)
    arrow(ax, [(21.90, 4.82), (22.35, 4.82)], color="black", lw=2)

    # Bottom pathway: h3 endpoint-only fiber.
    block(ax, 8.45, 3.20, 1.75, 0.55, r"$h_{3,t}$", fill="#F3F4F6", stroke=C["gray_dark"], fontsize=9.8, color=C["gray_dark"], lw=1.8)
    ax.text(11.35, 3.36, r"$h_3$: endpoint-only fiber", fontsize=8.8, color=C["gray_dark"], ha="left", va="center")
    arrow(ax, [(7.78, 4.05), (8.45, 3.48)], color=C["gray_dark"], lw=1.5, dashed=True)
    block(ax, 16.35, 3.20, 1.55, 0.55, r"$h_3$ skip", fill="#F3F4F6", stroke=C["gray_dark"], fontsize=9.2, color=C["gray_dark"], lw=1.8)
    arrow(ax, [(10.20, 3.48), (16.35, 3.48)], color=C["gray_dark"], lw=1.5, dashed=True)
    arrow(ax, [(17.90, 3.48), (23.85, 3.48)], color=C["gray_dark"], lw=1.5, dashed=True)

    # Endpoint styling and reconstruction.
    wct = block(ax, 24.35, 4.15, 1.45, 1.95, "", fill="#FEF3C7", stroke=C["style_dark"], radius=0.12, lw=2.2)
    ax.add_patch(Circle((25.08, 5.13), 0.58, facecolor="#FDE68A", edgecolor="none", alpha=0.28, zorder=1))
    ax.text(25.08, 5.35, "Fiber WCT", fontsize=10.4, weight="bold", color=C["style_dark"], ha="center")
    ax.text(25.08, 5.00, r"$T_1,T_2,T_3$", fontsize=9.2, color=C["style_dark"], ha="center")
    ax.text(25.08, 4.62, r"$\ell$ fixed", fontsize=8.9, color=C["content_dark"], ha="center")
    arrow(ax, [(23.80, 5.38), (24.35, 5.38)], color="black", lw=2)
    arrow(ax, [(23.80, 4.82), (24.35, 4.82)], color="black", lw=2)
    arrow(ax, [(10.94, 4.95), (25.08, 4.95)], color=C["style_dark"], lw=2, dashed=True)
    ax.text(22.60, 5.13, "endpoint style injection", fontsize=8.9, color=C["style_dark"], ha="center")

    idwt = block(ax, 26.15, 4.65, 0.95, 0.95, "", fill=C["white"], stroke=C["content_dark"], fontsize=10)
    icon_dwt(ax, 26.62, 5.12, 0.46)
    ax.text(26.62, 4.72, "iDWT", fontsize=10.0, weight="bold", ha="center", va="center")
    arrow(ax, [(25.80, 5.12), (26.15, 5.12)], color="black", lw=2)
    arrow(ax, [(12.60, 6.76), (26.15, 6.76), (26.15, 5.60)], color=C["content_dark"], lw=1.8, dashed=True)

    block(ax, 25.95, 3.15, 1.10, 0.88, "", fill=C["white"], stroke=C["content_dark"], fontsize=9)
    icon_vae(ax, 26.50, 3.59, 0.48, C["content"])
    ax.text(26.50, 3.22, "VAE Decode", fontsize=8.8, weight="bold", ha="center", va="center")
    ax.plot([26.50, 26.50], [4.65, 4.03], color="black", lw=2, zorder=1)

    block(ax, 27.15, 4.55, 0.80, 1.00, "", fill=C["white"], stroke=C["style_dark"], radius=0.1)
    embed_img(ax, 27.19, 4.59, 0.72, 0.92, OUT_DIR / "thumbs/output_thumb.png")
    ax.text(27.55, 4.36, "output $\\hat{x}$", fontsize=9.8, weight="bold", ha="center", va="top")
    ax.plot([26.90, 27.15], [5.12, 5.12], color="black", lw=2, zorder=1)
    ax.plot([26.50, 27.15], [4.03, 5.05], color="black", lw=2, zorder=1)

    # Single-row training strip.
    y3 = 0.88
    block(ax, 9.10, y3, 1.35, 0.50, r"Mix $x_t$", fill=C["white"], stroke=C["train_dark"], fontsize=9.0, lw=2.0)
    block(ax, 10.85, y3, 0.98, 0.50, "", fill=C["white"], stroke=C["train_dark"], lw=2.0)
    icon_dwt(ax, 11.34, y3 + 0.29, 0.30)
    block(ax, 12.25, y3, 1.95, 0.50, r"Predict $(v_1,v_2)$", fill=C["white"], stroke=C["train_dark"], fontsize=8.8, lw=2.0)
    block(ax, 14.65, y3, 2.25, 0.50, r"Target $(u_1,u_2)$", fill=C["white"], stroke=C["train_dark"], fontsize=8.8, lw=2.0)
    block(ax, 17.35, y3, 4.05, 0.50, r"$\mathcal{L}_{WD\!-\!VF}\ (w_\ell=0,\ \mathrm{no}\ h_3\ \mathrm{head})$", fill="#FEE2E2", stroke=C["train_dark"], fontsize=8.6, lw=2.0)
    arrow(ax, [(10.45, y3 + 0.25), (10.85, y3 + 0.25)], color="black", lw=2)
    arrow(ax, [(11.83, y3 + 0.25), (12.25, y3 + 0.25)], color="black", lw=2)
    arrow(ax, [(14.20, y3 + 0.25), (14.65, y3 + 0.25)], color="black", lw=2)
    arrow(ax, [(16.90, y3 + 0.25), (17.35, y3 + 0.25)], color="black", lw=2)
    ax.plot([19.35, 19.35], [y3 + 0.50, 1.72], color=C["train_dark"], lw=2, ls="--", zorder=1)
    ax.plot([19.35, 13.38], [1.72, 1.72], color=C["train_dark"], lw=2, ls="--", zorder=1)
    ax.annotate("", xy=(13.38, 3.95), xytext=(13.38, 1.72), arrowprops=dict(arrowstyle="-|>", color=C["train_dark"], lw=2, ls="--"), zorder=1)

    # Caption.
    ax.text(
        14.1,
        -0.38,
        "Figure 2. Staggered fiber-bundle view of WD-VF. The middle panel separates three pathways: "
        "the top base-manifold lane preserves $\\ell_t$, the middle active-fiber lane transports $(h_1,h_2)$ "
        "through a routed backbone and spectral ODE integrator, and the bottom $h_3$ lane bypasses transport "
        "to the endpoint. Fiber WCT acts only on fiber coordinates while keeping $\\ell_t$ fixed before iDWT reconstruction.",
        fontsize=9.8,
        color=C["text"],
        ha="center",
        va="top",
        wrap=True,
    )

    plt.tight_layout(pad=0.2)
    png = OUT_DIR / "aaai_arch_diagram_v16_staggered_bundle.png"
    svg = OUT_DIR / "aaai_arch_diagram_v16_staggered_bundle.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor=C["bg"], pad_inches=0.05)
    fig.savefig(svg, format="svg", bbox_inches="tight", facecolor=C["bg"], pad_inches=0.05)
    print(f"saved {png} and {svg}")
    plt.close(fig)


if __name__ == "__main__":
    save_im()
