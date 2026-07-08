import os, json, glob
from PIL import Image, ImageDraw, ImageFont

ROOT = r"g:/GitHub/Latent_Style/SchrodingerBridge"
IMG_DIR = os.path.join(ROOT, "exp/swd_cm_sem_r8/eval_r5_ht008/images")
DATASET = r"G:/GitHub/Latent_Style/Dataset/wikiart_random20_512/wikiart_random20_512/images/test"
OUT_PNG = os.path.join(ROOT, "aaai2027_v4/teaser_candidates.png")

# ---- load per-image MUSIQ ranking ----
recs = json.load(open(os.path.join(ROOT, "exp/swd_cm_sem_r8/eval_r5_ht008/_musiq_per_image.json")))

def parse(fname):
    base = fname[:-4]  # strip .png
    prefix, tgt = base.split("_to_")
    # prefix = "{src}_{src}__{artist}_{work}"  -> src duplicated, separated by "__"
    dup = prefix.split("__")[0]            # "{src}_{src}"
    src = dup[:(len(dup) - 1) // 2]        # first half = src style
    content_stem = prefix[len(src) + 1:]   # drop leading "{src}_"
    return src, tgt, content_stem + ".jpg"

# cross-style only, ranked by MUSIQ desc
cross = [r for r in recs if parse(r["file"])[0] != parse(r["file"])[1]]
cross.sort(key=lambda r: -r["musiq"])
# enforce (src,tgt) diversity -> varied teaser
seen = set()
sel = []
for r in cross:
    src, tgt, _ = parse(r["file"])
    if (src, tgt) in seen:
        continue
    seen.add((src, tgt))
    sel.append(r)
    if len(sel) == 8:
        break

# ---- layout ----
CELL = 256          # resized cell
GAP = 12
LABEL_H = 34       # top transfer label
SCORE_H = 22        # bottom musiq label
NAME_H = 22         # artwork name
COLS = len(sel)
W = COLS * CELL + (COLS + 1) * GAP
H = LABEL_H + CELL + SCORE_H + NAME_H + 2 * GAP
canvas = Image.new("RGB", (W, H), (255, 255, 255))
draw = ImageDraw.Draw(canvas)
try:
    font = ImageFont.truetype("arial.ttf", 16)
    fontb = ImageFont.truetype("arial.ttf", 16)
except Exception:
    font = fontb = ImageFont.load_default()

for i, r in enumerate(sel):
    fname = r["file"]
    src, tgt, content_file = parse(fname)
    x0 = GAP + i * (CELL + GAP)
    # content source
    cpath = os.path.join(DATASET, src, content_file)
    opath = os.path.join(IMG_DIR, fname)
    if not os.path.exists(cpath):
        print("MISSING content", cpath); continue
    if not os.path.exists(opath):
        print("MISSING output", opath); continue
    cimg = Image.open(cpath).convert("RGB").resize((CELL, CELL))
    oimg = Image.open(opath).convert("RGB").resize((CELL, CELL))
    # content on top
    canvas.paste(cimg, (x0, LABEL_H))
    # output below
    y_out = LABEL_H + CELL + GAP
    canvas.paste(oimg, (x0, y_out))
    # labels
    draw.text((x0, 6), f"{src} -> {tgt}", fill=(0,0,0), font=fontb)
    draw.text((x0, LABEL_H + CELL + 2), f"MUSIQ {r['musiq']:.2f}", fill=(180,0,0), font=font)
    name = content_file.replace(src + "__", "").replace(".jpg", "")
    if len(name) > 30: name = name[:28] + ".."
    draw.text((x0, LABEL_H + CELL + GAP + SCORE_H), name, fill=(80,80,80), font=font)

canvas.save(OUT_PNG)
print("saved", OUT_PNG, "size", canvas.size)
for r in sel:
    src, tgt, cf = parse(r["file"])
    print(f"  {r['musiq']:6.2f}  {src}->{tgt}  {cf}")
