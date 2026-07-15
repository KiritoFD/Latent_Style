"""Generate identity baseline images on remote server."""
import os, shutil
from pathlib import Path

STYLES = ['Early_Renaissance', 'Impressionism', 'Minimalism', 'Rococo', 'Ukiyo_e']
TEST_DIR = Path(r'I:\wikiart_distinct5_samam_512_classview\test')
OUT_DIR = Path(r'I:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_v2\images\identity')

OUT_DIR.mkdir(parents=True, exist_ok=True)
count = 0
for src_style in STYLES:
    src_dir = TEST_DIR / src_style
    for f in sorted(src_dir.iterdir()):
        if not f.suffix.lower() in ('.jpg', '.png', '.jpeg'):
            continue
        src_stem = f.stem
        for tgt_style in STYLES:
            out_name = f'{src_style}__{src_stem}__to__{tgt_style}.png'
            out_path = OUT_DIR / out_name
            if not out_path.exists():
                shutil.copy2(str(f), str(out_path))
            count += 1
print(f'Created {count} identity images')
