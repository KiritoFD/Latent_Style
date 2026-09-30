"""Filter latent_wct images to 750 pairs matching test set, then run DINO."""
import os
import shutil
import sys
from pathlib import Path

base = Path(r"I:\Github\Latent_Style\SchrodingerBridge\exp\latent_wct_baseline")

# ---- P2A-256 ----
p2a_test = Path(r"I:\datasets\legacy256_overfit50\test")
p2a_domains = ["cezanne", "Hayao", "monet", "photo", "vangogh"]
# Get test image IDs (stem without domain prefix) for each domain
p2a_test_ids = {}
for d in p2a_domains:
    folder = p2a_test / d
    if folder.is_dir():
        ids = set()
        for f in folder.iterdir():
            if f.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                # filename might be "{domain}_{id}" or just "{id}"
                stem = f.stem
                if stem.startswith(d + "_"):
                    ids.add(stem[len(d) + 1:])
                else:
                    ids.add(stem)
        p2a_test_ids[d] = ids
        print(f"  p2a test {d}: {len(ids)} ids")

# Filter p2a generated images: keep only those whose source is in test set
p2a_img_dir = base / "p2a_256" / "images"
p2a_filtered = base / "p2a_256" / "images_750"
p2a_filtered.mkdir(exist_ok=True)

# p2a filename format: {src}_{id}_to_{tgt}.png
# We need to match src and id to test set
import re
p2a_re = re.compile(r"^(?P<src>[a-zA-Z]+)_(?P<id>.+?)_to_(?P<tgt>[a-zA-Z]+)$")

kept = 0
for f in sorted(p2a_img_dir.iterdir()):
    if f.suffix.lower() not in [".png"]:
        continue
    m = p2a_re.match(f.stem)
    if not m:
        continue
    src = m.group("src")
    id_ = m.group("id")
    tgt = m.group("tgt")
    # Check if this source-id is in test set
    if src in p2a_test_ids and id_ in p2a_test_ids[src]:
        # Only keep 30 per src-tgt combo
        # We'll track counts
        pass

# Actually, let's just count per src-tgt and keep first 30
from collections import defaultdict
counts = defaultdict(int)
max_per_pair = 30

for f in sorted(p2a_img_dir.iterdir()):
    if f.suffix.lower() not in [".png"]:
        continue
    m = p2a_re.match(f.stem)
    if not m:
        continue
    src = m.group("src")
    id_ = m.group("id")
    tgt = m.group("tgt")
    if src not in p2a_test_ids or id_ not in p2a_test_ids[src]:
        continue
    key = (src, tgt)
    if counts[key] < max_per_pair:
        dst = p2a_filtered / f.name
        if not dst.exists():
            shutil.copy2(f, dst)
        counts[key] += 1
        kept += 1

print(f"\nP2A: kept {kept} images (target: 5*5*30=750)")
print(f"  counts: {dict(counts)}")

# ---- R5-WikiArt ----
r5_test = Path(r"I:\datasets\wikiarts20_512_test")
r5_styles = ["Cubism", "Expressionism", "Pop_Art", "Romanticism", "Symbolism"]
# Get test image stems for each style
r5_test_ids = {}
for s in r5_styles:
    folder = r5_test / s
    if folder.is_dir():
        ids = set()
        for f in folder.iterdir():
            if f.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                ids.add(f.stem)
        r5_test_ids[s] = ids
        print(f"  r5 test {s}: {len(ids)} ids")

# r5 filename format: {SrcStyle}__{SrcStyle}__{artist}__to__{TgtStyle}.png
# or {SrcStyle}__{artist}__to__{TgtStyle}.png
r5_img_dir = base / "r5_wikiart" / "images"
r5_filtered = base / "r5_wikiart" / "images_750"
r5_filtered.mkdir(exist_ok=True)

counts = defaultdict(int)
kept = 0
for f in sorted(r5_img_dir.iterdir()):
    if f.suffix.lower() not in [".png"]:
        continue
    stem = f.stem
    if "__to__" not in stem:
        continue
    left, tgt = stem.rsplit("__to__", 1)
    # Find target style
    tgt_style = None
    for s in r5_styles:
        if tgt == s or tgt.startswith(s):
            tgt_style = s
            break
    if tgt_style is None:
        continue
    # Parse source: {SrcStyle}__{SrcStyle}__{artist} or {SrcStyle}__{artist}
    src_style = None
    artist = left
    for s in sorted(r5_styles, key=len, reverse=True):
        if artist.startswith(s + "__"):
            src_style = s
            artist = artist[len(s) + 2:]
            # Check for repeated prefix
            if artist.startswith(s + "__"):
                artist = artist[len(s) + 2:]
            break
        elif artist.startswith(s + "_"):
            src_style = s
            artist = artist[len(s) + 1:]
            break
    if src_style is None:
        continue
    # Check if artist is in test set
    # test set filenames: {style}__{artist}.{ext}
    test_stem = f"{src_style}__{artist}"
    if src_style in r5_test_ids and test_stem in r5_test_ids[src_style]:
        key = (src_style, tgt_style)
        if counts[key] < max_per_pair:
            dst = r5_filtered / f.name
            if not dst.exists():
                shutil.copy2(f, dst)
            counts[key] += 1
            kept += 1

print(f"\nR5: kept {kept} images (target: 5*5*30=750)")
print(f"  counts: {dict(counts)}")
