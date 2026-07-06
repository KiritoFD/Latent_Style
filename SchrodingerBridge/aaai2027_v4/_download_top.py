"""Download top-N ours photo->vangogh candidates and build a preview grid."""
import subprocess, os, sys

TOP_CANDIDATES = [
    "photo_2013-11-25 10_46_18",
    "photo_2013-11-17 16_40_10",
    "photo_2013-11-12 16_58_40",
    "photo_2013-11-21 17_44_44",
    "photo_2013-11-18 06_58_36",
    "photo_2013-11-08 16_45_24",   # current teaser
]

REMOTE_BASE = "I:/Github/Latent_Style/SchrodingerBridge/exp/latent256_photo2art/latent256_b16_e10/full_eval/epoch_0010/images"
LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))

def scp(remote_rel, local_name):
    remote_full = f'administrator@100.115.18.62:{REMOTE_BASE}/{remote_rel}'
    local_path = os.path.join(LOCAL_DIR, local_name)
    cmd = ['scp', '-P', '2222', '-o', 'LogLevel=ERROR', remote_full, local_path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    ok = r.returncode == 0 and os.path.exists(local_path)
    sz = os.path.getsize(local_path) if ok else 0
    print(f"  {'OK' if ok else 'FAIL'} {local_name} ({sz} B)")
    return ok if ok else None

print("=== Downloading top candidates (ours) ===")
files = []
for c in TOP_CANDIDATES:
    png_name = f"{c}_to_vangogh.png"
    local_name = f"_pv_{c.replace(' ','_').replace(':','')}.png"
    ok = scp(png_name, local_name)
    if ok:
        files.append((c, local_name))

# Also grab identity/content for #1
ident_src = TOP_CANDIDATES[0] + "_to_photo.jpg"
scp_ident = subprocess.run(
    ['scp','-P','2222','-o','LogLevel=ERROR',
     f'administrator@100.115.18.62:I:/exp_256_photo2art/identity_256/images/{ident_src}',
     os.path.join(LOCAL_DIR, '_pv_identity.jpg')],
    capture_output=True, text=True)
print(f"  Identity: {'OK' if scp_ident.returncode==0 else 'FAIL'}")

# Build preview grid
from PIL import Image, ImageDraw, ImageFont

THUMB = 200
cols = min(len(files), 3)
rows = (len(files) + cols - 1) // cols
grid_w = cols * (THUMB + 60) + 40
grid_h = rows * (THUMB + 50) + 60
grid = Image.new('RGB', (grid_w, grid_h), (255,255,255))
draw = ImageDraw.Draw(grid)

try:
    font = ImageFont.truetype("arial.ttf", 14)
except:
    font = ImageFont.load_default()

for idx, (label, fname) in enumerate(files):
    r, c = divmod(idx, cols)
    x = 20 + c * (THUMB + 60)
    y = 30 + r * (THUMB + 50)
    img = Image.open(fname).convert('RGB').resize((THUMB, THUMB), Image.LANCZOS)
    grid.paste(img, (x, y+18))
    rank = idx + 1
    short = label.replace("photo_", "")[:22]
    draw.text((x, y), f"#{rank}  {short}", fill=(0,0,0), font=font)

out = os.path.join(LOCAL_DIR, "_preview_top_candidates.png")
grid.save(out, quality=95)
print(f"\nSaved preview: {out}")
