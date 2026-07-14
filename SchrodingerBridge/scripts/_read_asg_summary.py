import json
with open(r"I:\Github\Latent_Style\SchrodingerBridge\exp\asg_restore_verify\t1_asg_5ep\summary.json") as f:
    data = json.load(f)
overview = data.get("analysis", {}).get("all_pairs_overview", {})
print(f"CLIP-S (clip_style): {overview.get('clip_style', 'N/A')}")
print(f"LPIPS (content_lpips): {overview.get('content_lpips', 'N/A')}")
