import json

DIRS = {
    "target_hf_subband_ft6": "I:/Github/Latent_Style/SchrodingerBridge/exp/model_probe/target_hf_subband_ft6/full_eval/adain15",
    "target_hf_subband_scratch_5ep": "I:/Github/Latent_Style/SchrodingerBridge/exp/model_probe/target_hf_subband_scratch_5ep/full_eval/adain15",
    "target_hf_subband_scratch_10ep": "I:/Github/Latent_Style/SchrodingerBridge/exp/model_probe/target_hf_subband_scratch_10ep/full_eval/adain15",
}

print(f"{'Name':32s}  {'DINO-S':>8s}  {'CLIP-S':>8s}  {'LPIPS':>8s}  {'DINO-C':>8s}")
print("-" * 81)
for name, d in DIRS.items():
    try:
        s = json.load(open(d + "/summary.json"))
        apo = s["analysis"]["all_pairs_overview"]
        clip_s = apo["clip_style"]
        lpips = apo["content_lpips"]
        dino = json.load(open(d + "/dino.json"))
        dino_c = dino["dino_content"]
        dino_s = dino["dino_style"]
        print(f"{name:32s}  {dino_s:8.4f}  {clip_s:8.4f}  {lpips:8.4f}  {dino_c:8.4f}")
    except Exception as e:
        print(f"{name:32s}: ERROR - {e}")
