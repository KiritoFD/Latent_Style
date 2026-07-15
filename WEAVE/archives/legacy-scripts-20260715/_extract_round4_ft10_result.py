import json

RUN = "target_hf_subband_ft10"
ROOT = "I:/Github/Latent_Style/SchrodingerBridge"

dino = json.load(open(f"{ROOT}/exp/model_probe/{RUN}/full_eval/adain15/dino.json"))
summary = json.load(open(f"{ROOT}/exp/model_probe/{RUN}/full_eval/adain15/summary.json"))
apo = summary["analysis"]["all_pairs_overview"]

print(f"{RUN:25s}  DINO-S={dino['dino_style']:.4f}  CLIP-S={apo['clip_style']:.4f}  LPIPS={apo['content_lpips']:.4f}  DINO-C={dino['dino_content']:.4f}")
