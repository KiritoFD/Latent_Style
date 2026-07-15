"""Create IDT (identity) baseline images: content image copied as output.

Naming convention: {src_style}_{src_stem}_to_{tgt_style}.png
"""
import shutil
from pathlib import Path

TEST_DIR = Path("G:/GitHub/Latent_Style/Dataset/distinct5_512/test")
OUT_DIR = Path("G:/GitHub/Latent_Style/SchrodingerBridge/exp/idt_baseline/images")

STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]

OUT_DIR.mkdir(parents=True, exist_ok=True)

count = 0
for src_style in STYLES:
    src_dir = TEST_DIR / src_style
    src_images = sorted([p for p in src_dir.iterdir() if p.suffix.lower() in ['.jpg', '.png', '.jpeg', '.webp']])
    
    for tgt_style in STYLES:
        if tgt_style == src_style:
            continue  # Skip identity pairs
        
        for src_img in src_images:
            src_stem = src_img.stem
            out_name = f"{src_style}_{src_stem}_to_{tgt_style}.png"
            out_path = OUT_DIR / out_name
            
            if not out_path.exists():
                shutil.copy2(src_img, out_path)
                count += 1

print(f"Created {count} IDT baseline images in {OUT_DIR}")
print(f"Expected: 5 styles x 30 imgs x 4 targets = 600 pairs")