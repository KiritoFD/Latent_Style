"""Count images in latent_wct baseline directories."""
import os

base = r"I:\Github\Latent_Style\SchrodingerBridge\exp\latent_wct_baseline"
for sub in ["d5_512", "p2a_256", "r5_wikiart"]:
    img_dir = os.path.join(base, sub, "images")
    if os.path.isdir(img_dir):
        n = len([f for f in os.listdir(img_dir) if f.endswith(".png")])
        print(f"{sub}: {n} images")
    else:
        print(f"{sub}: NO IMAGES DIR at {img_dir}")

# Also check summary.json to get CLIP-S/LPIPS already recorded
import json
for sub in ["d5_512", "p2a_256", "r5_wikiart"]:
    sp = os.path.join(base, sub, "summary.json")
    if os.path.isfile(sp):
        with open(sp) as f:
            s = json.load(f)
        a = s.get("analysis", {})
        st = a.get("style_transfer_ability", {})
        ap = a.get("all_pairs_overview", {})
        print(f"  {sub} transfer: clip_s={st.get('clip_style'):.4f}, lpips={st.get('content_lpips'):.4f}")
        print(f"  {sub} allpairs: clip_s={ap.get('clip_style'):.4f}, lpips={ap.get('content_lpips'):.4f}")
