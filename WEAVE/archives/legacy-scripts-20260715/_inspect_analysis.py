"""Extract the actual CLIP-S and LPIPS values from analysis.all_pairs_overview."""
import json
import os

base = r"I:\Github\Latent_Style\SchrodingerBridge\exp"
exps = ["d1_gram_hf1_15ep", "d1_gram_hf5_15ep", "hp_simple_swd12_15ep"]

for name in exps:
    path = os.path.join(base, name, "full_eval", "epoch_0015", "summary.json")
    if not os.path.exists(path):
        print(f"{name}: MISSING")
        continue
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    print(f"\n=== {name} ===")
    ov = d.get("analysis", {}).get("all_pairs_overview", {})
    print("all_pairs_overview keys:", list(ov.keys()))
    for k, v in ov.items():
        if isinstance(v, (int, float)):
            print(f"  {k} = {v:.6f}")
        elif isinstance(v, str):
            print(f"  {k} = {v}")
    # Also check matrix_breakdown for one style
    mb = d.get("matrix_breakdown", {})
    if mb:
        first_style = list(mb.keys())[0]
        print(f"matrix_breakdown[{first_style}] keys:", list(mb[first_style].keys()))
        for k, v in mb[first_style].items():
            if isinstance(v, (int, float)):
                print(f"  {k} = {v:.6f}")
