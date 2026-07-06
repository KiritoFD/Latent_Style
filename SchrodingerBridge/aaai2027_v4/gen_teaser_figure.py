"""Generate teaser figure: Content vs SaMam vs StyleID vs WD-VF vs Seedream 4.5.

Each column shows an image + CLIP-S / LPIPS metrics below.
Source image: photo_2013-11-17 16_40_10 (cherry blossom path).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

OUT_DIR = Path(__file__).resolve().parent
OUT_PDF = OUT_DIR / "fig_teaser_comparison.pdf"

# (label, filepath, clip_s, lpips) — use None for N/A
IMAGES = [
    ("Content",       str(OUT_DIR / "teaser_content_photo_vangogh.jpg"),     None,   None),
    ("SaMam",         str(OUT_DIR / "teaser_samam_photo_vangogh.jpg"),      0.677,  None),
    ("StyleID\n(target)", str(OUT_DIR / "teaser_styleid_photo_vangogh.jpg"),  0.724,  0.771),
    ("WEAVE (ours)",  str(OUT_DIR / "teaser_ours_photo_vangogh.png"),       0.736,  0.444),
    ("Seedream 4.5",  str(OUT_DIR / "teaser_seedream_photo_vangogh.jpg"),    0.714,  0.526),
]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
})


def main() -> None:
    n = len(IMAGES)
    # Taller fig to make room for metrics text below images
    fig, axes = plt.subplots(1, n, figsize=(3.05 * n, 4.2))

    for ax, (label, path, cs_val, lp_val) in zip(axes, IMAGES):
        img = Image.open(path).convert("RGB").resize((512, 512), Image.LANCZOS)
        ax.imshow(img)
        ax.set_title(label, fontsize=11, pad=6)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.2)
            spine.set_color("#AAAAAA")

        # Metric annotation below image (bottom-aligned so columns share a baseline)
        if cs_val is not None or lp_val is not None:
            parts = []
            if cs_val is not None:
                parts.append(f"CLIP-S {cs_val:.3f}")
            if lp_val is not None:
                parts.append(f"LPIPS {lp_val:.3f}")
            metric_text = "\n".join(parts)
            ax.text(
                0.5, -0.12, metric_text,
                transform=ax.transAxes,
                fontsize=8.5,
                ha="center", va="bottom",
                color="#333333",
                fontfamily="monospace",
            )
        # Content column intentionally shows no metric dash.

    fig.suptitle(
        r"Photo $\Rightarrow$ Van Gogh",
        fontsize=13,
        y=0.97,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.06, 1, 0.94])
    fig.savefig(OUT_PDF, dpi=300, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(OUT_DIR / "fig_teaser_comparison.png", dpi=200, bbox_inches="tight", pad_inches=0.05)
    print(f"Saved {OUT_PDF}")


if __name__ == "__main__":
    main()
