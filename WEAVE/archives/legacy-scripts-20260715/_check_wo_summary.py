"""Check summary.json for wo_* experiments."""
import json, os

BASE = r"I:\Github\Latent_Style\SchrodingerBridge\exp"
EXPS = ["wo_endpoint_adain", "wo_flow", "wo_spectral_ode", "wo_wavelet", "wo_asg"]

for exp in EXPS:
    for path_pattern in ["eval/summary.json", "full_eval/epoch_0005/summary.json", "summary.json"]:
        path = os.path.join(BASE, exp, path_pattern)
        if os.path.exists(path):
            with open(path) as f:
                d = json.load(f)
            agg = d.get("aggregate", d.get("summary", d))
            clip_s = agg.get("clip_s", {})
            lpips = agg.get("lpips", {})
            if isinstance(clip_s, dict):
                clip_s = clip_s.get("mean", "—")
            if isinstance(lpips, dict):
                lpips = lpips.get("mean", "—")
            print(f"{exp} ({path_pattern}): CLIP-S={clip_s}, LPIPS={lpips}")
            break
    else:
        # List what files exist
        eval_dir = os.path.join(BASE, exp, "eval")
        if os.path.exists(eval_dir):
            files = os.listdir(eval_dir)
            print(f"{exp}: no summary.json found. eval/ contains: {files}")
        else:
            print(f"{exp}: no eval/ directory")
