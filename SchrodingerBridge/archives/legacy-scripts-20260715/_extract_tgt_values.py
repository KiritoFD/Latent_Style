import json

for name, path in [
    ("P2A-256", "I:/Github/Latent_Style/SchrodingerBridge/exp/target_style_baseline_p2a/summary.json"),
    ("R5-WikiArt", "I:/Github/Latent_Style/SchrodingerBridge/exp/target_style_baseline_r5/summary.json"),
]:
    d = json.load(open(path))
    print(f"=== {name} TGT ===")
    # Check all_pairs_overview
    apo = d.get("analysis", {}).get("all_pairs_overview", {})
    for k, v in apo.items():
        if isinstance(v, (int, float)):
            print(f"  all_pairs_overview.{k}: {v:.4f}")
    print()