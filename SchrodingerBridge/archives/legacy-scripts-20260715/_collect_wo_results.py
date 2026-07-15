"""Collect wo_* ablation results."""
import json, os

BASE = r"I:\Github\Latent_Style\SchrodingerBridge\exp"
EXPS = ["wo_endpoint_adain", "wo_flow", "wo_spectral_ode", "wo_wavelet", "wo_asg", "wo_cross_attn"]

for exp in EXPS:
    dino_path = os.path.join(BASE, exp, "eval", "dino_summary.json")
    if os.path.exists(dino_path):
        with open(dino_path) as f:
            d = json.load(f)
        clip_s = d.get("all_clip_s", "—")
        lpips = d.get("all_lpips", "—")
        dino_s = d.get("all_dino_s", "—")
        dino_c = d.get("all_dino_c", "—")
        print(f"{exp}: CLIP-S={clip_s}, LPIPS={lpips}, DINO-S={dino_s}, DINO-C={dino_c}")
    else:
        # Try state/dino
        alt_path = os.path.join(r"I:\Github\Latent_Style\SchrodingerBridge\state\dino", f"D5-512__{exp}.json")
        if os.path.exists(alt_path):
            with open(alt_path) as f:
                d = json.load(f)
            dino_s = d.get("dino_style", d.get("all_dino_s", "—"))
            dino_c = d.get("dino_content", d.get("all_dino_c", "—"))
            print(f"{exp}: DINO-S={dino_s}, DINO-C={dino_c} (no CLIP-S/LPIPS)")
        else:
            print(f"{exp}: NO RESULTS FOUND")
