import os, json
from PIL import Image, ImageDraw, ImageFont

ROOT = r"g:/GitHub/Latent_Style/SchrodingerBridge"
IMG_DIR = os.path.join(ROOT, "exp/swd_cm_sem_r8/eval_r5_ht008/images")
OUT_DIR = os.path.join(ROOT, "aaai2027_v4/teaser_pairs")
os.makedirs(OUT_DIR, exist_ok=True)

recs = json.load(open(os.path.join(ROOT, "exp/swd_cm_sem_r8/eval_r5_ht008/_musiq_per_image.json")))

def parse(fname):
    base = fname[:-4]
    prefix, tgt = base.split("_to_")
    dup = prefix.split("__")[0]
    src = dup[:(len(dup) - 1) // 2]
    artist_work = prefix.split("__")[1]      # artist_work (may contain _)
    return src, tgt, artist_work

# group by (src,tgt)
groups = {}
for r in recs:
    s, t, aw = parse(r["file"])
    groups.setdefault((s, t), []).append((r["musiq"], r["file"], aw))

# the 8 candidate style pairs from before (diverse, cross-style)
pairs = [
    ("Early_Renaissance", "Rococo"),
    ("Early_Renaissance", "Impressionism"),
    ("Ukiyo_e", "Rococo"),
    ("Rococo", "Impressionism"),
    ("Early_Renaissance", "Ukiyo_e"),
    ("Rococo", "Ukiyo_e"),
    ("Minimalism", "Rococo"),
    ("Rococo", "Early_Renaissance"),
]

CELL = 256; GAP = 8; LAB = 44
N = 7
W = N * CELL + (N + 1) * GAP
H = LAB + CELL
try:
    font = ImageFont.truetype("arial.ttf", 14)
    fontb = ImageFont.truetype("arial.ttf", 14)
except Exception:
    font = fontb = ImageFont.load_default()

def wrap(draw, text, fnt, maxw):
    words = text.split("_")
    lines, cur = [], ""
    for w in words:
        test = (cur + "_" + w) if cur else w
        if draw.textlength(test, font=fnt) <= maxw:
            cur = test
        else:
            if cur: lines.append(cur)
            cur = w
    if cur: lines.append(cur)
    return lines

print(f"{len(pairs)} style-pair teaser strips (7 works each) -> {OUT_DIR}\n")
for idx, (s, t) in enumerate(pairs, 1):
    items = sorted(groups[(s, t)], key=lambda x: -x[0])[:N]   # top-7 by MUSIQ
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    # pair header
    d.text((GAP, 2), f"{s} -> {t}", fill=(0, 0, 0), font=fontb)
    for j, (m, fname, aw) in enumerate(items):
        img = Image.open(os.path.join(IMG_DIR, fname)).convert("RGB").resize((CELL, CELL), Image.LANCZOS)
        x = GAP + j * (CELL + GAP)
        canvas.paste(img, (x, LAB))
        # label: work name (wrapped) + musiq
        lines = wrap(d, aw, font, CELL - 6)
        yy = 20
        for ln in lines[:2]:
            d.text((x + 2, yy), ln, fill=(60, 60, 60), font=font)
            yy += 16
        d.text((x + 2, LAB - 16), f"M{m:.1f}", fill=(200, 0, 0), font=font)
    outpath = os.path.join(OUT_DIR, f"p{idx:02d}_{s}__{t}.png")
    canvas.save(outpath)
    print(f"  [{idx}] {s} -> {t}  ({len(items)} works): {os.path.basename(outpath)}")
    for m, fname, aw in items:
        print(f"        M{m:6.2f}  {aw}")

print("\nReview order (one file per style pair):")
for idx, (s, t) in enumerate(pairs, 1):
    print(f"  p{idx:02d}  {s} -> {t}")
