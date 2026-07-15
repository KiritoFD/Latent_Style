"""Rename d5_512 Latent-WCT images from double-style prefix to single-style prefix.

{style}__{style}__{name}__to__{tgt}.png  ->  {style}__{name}__to__{tgt}.png
"""
import os
import re
import glob

d = r"I:\Github\Latent_Style\SchrodingerBridge\exp\latent_wct_baseline\d5_512\images"
pat = re.compile(r"^(.+?)__\1__(.+__to__.+)$")
c = 0
for f in os.listdir(d):
    if f.endswith(".png") and pat.match(f):
        new = pat.sub(r"\1__\2", f)
        os.rename(os.path.join(d, f), os.path.join(d, new))
        c += 1
print("Renamed:", c)
print("Total:", len(glob.glob(os.path.join(d, "*.png"))))
# Show first 3 for verification
for f in sorted(os.listdir(d))[:3]:
    print("  ", f)
