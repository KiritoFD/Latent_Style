"""Extract CLIP-S and LPIPS from Latent-WCT summary.json files."""
import json
import os

datasets = {
    "d5_512": r"I:\Github\Latent_Style\SchrodingerBridge\exp\latent_wct_baseline\d5_512\summary.json",
    "p2a_256": r"I:\Github\Latent_Style\SchrodingerBridge\exp\latent_wct_baseline\p2a_256\summary.json",
    "r5_wikiart": r"I:\Github\Latent_Style\SchrodingerBridge\exp\latent_wct_baseline\r5_wikiart\summary.json",
}

for name, path in datasets.items():
    if not os.path.exists(path):
        print(f"{name}: NOT FOUND")
        continue
    with open(path) as f:
        data = json.load(f)
    analysis = data.get("analysis", {})
    overview = analysis.get("all_pairs_overview", {})
    print(f"\n=== {name} ===")
    if isinstance(overview, dict):
        for k, v in overview.items():
            if isinstance(v, (int, float)):
                print(f"  {k} = {v}")
            elif isinstance(v, dict):
                for sk, sv in v.items():
                    if isinstance(sv, (int, float)):
                        print(f"  {k}.{sk} = {sv}")
    else:
        print(f"  overview = {overview}")
