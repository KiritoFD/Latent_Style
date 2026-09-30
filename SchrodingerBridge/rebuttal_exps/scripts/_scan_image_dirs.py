"""Scan candidate raw image dataset directories to find the WEAVE training source."""
from pathlib import Path

CANDIDATES = [
    r"F:\wikiarts_5_full_notest\train",
    r"F:\wikiart_distinct5_512_pixel256\train",
    r"F:\wikiart_distinct5_samam_512_pixel256\train",
    r"F:\wikiart_distinct5_samam_512_pixel128\train",
    r"F:\wikiart_distinct5_samam_512_classview_real\train",
    r"F:\wikiart_distinct5_512_classview\train",
    r"F:\wikiart_distinct5_vavae_256\train",
    r"F:\wikiart27_sd15_512_classview\train",
    r"F:\wikiart27_sd15_512_latents_ema\train",
    r"F:\wikiart_distinct5_samam_512_latents_ema\train",
    r"F:\wikiart_distinct5_samam_512_vavae_f16d32\train",
    r"F:\wikiart_distinct5_512_latents_sdxl_fix\train",
    r"G:\wikiart27_latents_compact\train",
    r"F:\wikiart\wikiart",
]

STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
EXTS = {".jpg", ".jpeg", ".png", ".webp"}

print("=" * 80)
print("Scanning candidate raw image dataset directories")
print("=" * 80)
for root in CANDIDATES:
    p = Path(root)
    if not p.exists():
        print(f"\n[MISS] {root}  (does not exist)")
        continue
    print(f"\n[OK]   {root}")
    total = 0
    has_all_styles = True
    for s in STYLES:
        sp = p / s
        if not sp.exists():
            print(f"    - {s}: MISSING")
            has_all_styles = False
            continue
        files = [f for f in sp.iterdir() if f.is_file() and f.suffix.lower() in EXTS]
        n = len(files)
        total += n
        if n > 0:
            sample = files[0].name
            print(f"    - {s}: {n} images  (e.g. {sample})")
        else:
            # may be latents instead
            all_files = list(sp.iterdir())
            print(f"    - {s}: 0 images, {len(all_files)} total files (non-image?)")
            if all_files:
                print(f"      sample: {all_files[0].name}")
    print(f"    TOTAL images: {total}  all_styles_present={has_all_styles}")
