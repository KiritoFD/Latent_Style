"""Crop content/style/output thumbnails from qual grid for AAAI arch diagram."""
from pathlib import Path
from PIL import Image

src = Path("g:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/figures/qual_case_preview.png")
out_dir = Path("g:/GitHub/Latent_Style/SchrodingerBridge/docs/630/thumbs")
out_dir.mkdir(parents=True, exist_ok=True)

img = Image.open(src)
W, H = img.size
print(f"Source size: {W}x{H}")

# Conservative crop boxes focusing on image centers, avoiding text labels and borders.
crops = {
    "content_thumb": (200, 90, 290, 160),   # Source image center
    "output_thumb":  (1020, 90, 1110, 160), # LBM-PS-v2 image center
    "style_thumb":   (1150, 90, 1240, 160), # Target ref image center
}

for name, box in crops.items():
    thumb = img.crop(box)
    thumb = thumb.resize((70, 70), Image.LANCZOS)
    path = out_dir / f"{name}.png"
    thumb.save(path)
    print(f"Saved {path}: {thumb.size}")
