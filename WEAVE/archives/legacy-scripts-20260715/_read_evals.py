import json
import os

base = "I:/Github/Latent_Style/SchrodingerBridge/exp/dino_s_break/brk_a_ll03_15ep/full_eval"
print(f"{'Ep':>3} | {'CLIP-S':>7} | {'CLIP-T':>7} | {'LPIPS':>7} | {'dCLIP-S':>7}")
print("-" * 50)
for e in range(1, 16):
    p = os.path.join(base, f"epoch_{e:04d}", "summary.json")
    if os.path.exists(p):
        with open(p) as f:
            d = json.load(f)
        ov = d.get("analysis", {}).get("all_pairs_overview", {})
        st = d.get("analysis", {}).get("style_transfer_ability", {})
        clip_s = ov.get("clip_style", 0)
        clip_t = ov.get("clip_t", 0)
        lpips = ov.get("content_lpips", 0)
        dclip = ov.get("clip_s_delta_idt", 0)
        # Check for DINO fields
        dino_keys = [k for k in ov.keys() if "dino" in k.lower()]
        if dino_keys:
            print(f"{e:>3} | {clip_s:>7.4f} | {clip_t:>7.4f} | {lpips:>7.4f} | {dclip:>7.4f} | DINO: {', '.join(dino_keys)}")
        else:
            print(f"{e:>3} | {clip_s:>7.4f} | {clip_t:>7.4f} | {lpips:>7.4f} | {dclip:>7.4f}")
    else:
        print(f"{e:>3} | (pending)")
