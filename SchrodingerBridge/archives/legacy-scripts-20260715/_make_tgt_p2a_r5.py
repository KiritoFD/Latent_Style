"""Create TGT baseline images for P2A-256 and R5-WikiArt datasets.
Run on remote server (I: drive accessible).
"""
import shutil
import os
from pathlib import Path

# ===== P2A-256 =====
P2A_TEST = Path("I:/datasets/legacy256_overfit50/test")
P2A_OUT = Path("I:/Github/Latent_Style/SchrodingerBridge/exp/target_style_baseline_p2a/images")
P2A_STYLES = ["cezanne", "Hayao", "monet", "photo", "vangogh"]

P2A_OUT.mkdir(parents=True, exist_ok=True)

count = 0
for src_style in P2A_STYLES:
    src_dir = P2A_TEST / src_style
    src_images = sorted([p for p in src_dir.iterdir() if p.suffix.lower() in ['.jpg', '.png', '.jpeg', '.webp']])
    
    for tgt_style in P2A_STYLES:
        if tgt_style == src_style:
            continue
        tgt_dir = P2A_TEST / tgt_style
        tgt_images = sorted([p for p in tgt_dir.iterdir() if p.suffix.lower() in ['.jpg', '.png', '.jpeg', '.webp']])
        if not tgt_images:
            print(f"WARNING: No images in {tgt_dir}")
            continue
        tgt_ref = tgt_images[0]
        for src_img in src_images:
            out_name = f"{src_style}_{src_img.stem}_to_{tgt_style}.png"
            out_path = P2A_OUT / out_name
            if not out_path.exists():
                shutil.copy2(tgt_ref, out_path)
                count += 1

print(f"P2A-256: Created {count} TGT baseline images in {P2A_OUT}")
exp = len(P2A_STYLES) * (len(P2A_STYLES) - 1) * len(src_images)
print(f"P2A-256: Expected {exp} pairs")

# ===== R5-WikiArt =====
R5_TEST = Path("I:/datasets/wikiarts20_512_test")
R5_OUT = Path("I:/Github/Latent_Style/SchrodingerBridge/exp/target_style_baseline_r5/images")
R5_STYLES = ["Cubism", "Expressionism", "Pop_Art", "Romanticism", "Symbolism"]

R5_OUT.mkdir(parents=True, exist_ok=True)

count = 0
for src_style in R5_STYLES:
    src_dir = R5_TEST / src_style
    src_images = sorted([p for p in src_dir.iterdir() if p.suffix.lower() in ['.jpg', '.png', '.jpeg', '.webp']])
    
    for tgt_style in R5_STYLES:
        if tgt_style == src_style:
            continue
        tgt_dir = R5_TEST / tgt_style
        tgt_images = sorted([p for p in tgt_dir.iterdir() if p.suffix.lower() in ['.jpg', '.png', '.jpeg', '.webp']])
        if not tgt_images:
            print(f"WARNING: No images in {tgt_dir}")
            continue
        tgt_ref = tgt_images[0]
        for src_img in src_images:
            out_name = f"{src_style}_{src_img.stem}_to_{tgt_style}.png"
            out_path = R5_OUT / out_name
            if not out_path.exists():
                shutil.copy2(tgt_ref, out_path)
                count += 1

print(f"R5-WikiArt: Created {count} TGT baseline images in {R5_OUT}")
exp = len(R5_STYLES) * (len(R5_STYLES) - 1) * len(src_images)
print(f"R5-WikiArt: Expected {exp} pairs")