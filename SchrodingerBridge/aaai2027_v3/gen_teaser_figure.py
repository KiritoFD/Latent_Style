"""Generate teaser figure: Original vs SaMam vs Diffusion-damage vs WD-VF."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

OUT_DIR = Path(__file__).resolve().parent
OUT_FILE = OUT_DIR / "fig_teaser_comparison.pdf"

# Source image and three method outputs for Early Renaissance -> Ukiyo-e
IMAGES = {
    "Content": "g:/GitHub/Latent_Style/SchrodingerBridge/exp/baseline/images/identity/Early_Renaissance__Early_Renaissance__andrea-mantegna_maria-with-the-sleeping-child-1455__to__Ukiyo_e.png",
    "SaMam (No-op)": "g:/GitHub/Latent_Style/SchrodingerBridge/exp/baseline/images/samam_diag_3000/Early_Renaissance__andrea-mantegna_maria-with-the-sleeping-child-1455__to__Ukiyo_e.png",
    "SDEdit $s$=0.40 (structure damage)": "g:/GitHub/Latent_Style/SchrodingerBridge/exp/baseline/images/sdedit_str0.40/Early_Renaissance__andrea-mantegna_maria-with-the-sleeping-child-1455__to__Ukiyo_e.png",
    "WD-VF (ours)": "g:/GitHub/Latent_Style/SchrodingerBridge/aaai2027_v3/teaser_wdvf/wdvf_er_to_ukiyo_e.png",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
})


def main() -> None:
    fig, axes = plt.subplots(1, 4, figsize=(14.5, 4.2))
    for ax, (label, path) in zip(axes, IMAGES.items()):
        img = Image.open(path).convert("RGB")
        ax.imshow(img)
        ax.set_title(label, fontsize=12, pad=8)
        ax.axis("off")
        # Add a subtle border
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.2)
            spine.set_color("#AAAAAA")

    fig.suptitle(
        "Early Renaissance $\Rightarrow$ Ukiyo-e",
        fontsize=13,
        y=0.98,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT_FILE, dpi=300, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(OUT_DIR / "fig_teaser_comparison.png", dpi=200, bbox_inches="tight", pad_inches=0.05)
    print(f"Saved {OUT_FILE}")


if __name__ == "__main__":
    main()
