import os, shutil, json
from collections import defaultdict

ZSTAR = r"g:\GitHub\Latent_Style\SchrodingerBridge\results\D5-512\zstar"
OUT = r"G:\GitHub\Latent_Style\Dataset\distinct5_512_train_zstar_src"
STYLES = ["Early_Renaissance","Impressionism","Minimalism","Rococo","Ukiyo_e"]

# candidate roots that may hold the original source jpgs (Style__artist_work.jpg)
ROOTS = [
    r"G:\GitHub\Latent_Style\Dataset\distinct5_512\train",
    r"G:\GitHub\Latent_Style\Dataset\distinct5_512\test",
    r"F:\wikiart_distinct5_512_images\test",
    r"F:\wikiart_distinct5_samam_512_classview_real\test",
    r"F:\wikiart_distinct5_samam_512_classview\test",
]

seen = defaultdict(set)
for fn in os.listdir(ZSTAR):
    if not fn.endswith(".png"): continue
    base = fn[:-4]
    if "__to__" not in base: continue
    left, tgt = base.rsplit("__to__",1)
    parts = left.split("__",2)
    if len(parts) < 3: continue
    style = parts[0]
    src_stem = parts[0] + "__" + parts[2]
    seen[style].add(src_stem)

print("parsed source counts per style:")
for s in STYLES:
    print(f"  {s}: {len(seen[s])}")

os.makedirs(OUT, exist_ok=True)
missing = []
copied = 0
origin = defaultdict(list)
for s in STYLES:
    sd = os.path.join(OUT, s)
    os.makedirs(sd, exist_ok=True)
    for stem in sorted(seen[s]):
        found = None
        for root in ROOTS:
            cand = os.path.join(root, s, stem + ".jpg")
            if os.path.exists(cand):
                found = cand; break
            cand2 = os.path.join(root, s, stem + ".png")
            if os.path.exists(cand2):
                found = cand2; break
        if found is None:
            missing.append((s, stem)); continue
        dst = os.path.join(sd, stem + os.path.splitext(found)[1])
        if not os.path.exists(dst):
            shutil.copy(found, dst)
        copied += 1
        origin[s].append(os.path.relpath(found, "G:\\GitHub\\Latent_Style\\Dataset\\distinct5_512"))

print(f"copied={copied} missing={len(missing)}")
for s, stem in missing:
    print("  MISSING:", s, stem)
json.dump({"manifest": {s: sorted(seen[s]) for s in STYLES}, "origin": origin},
          open(os.path.join(OUT,"_src_manifest.json"),"w"), indent=1)
print("manifest saved")
