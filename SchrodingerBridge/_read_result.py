import json, sys
p = sys.argv[1] if len(sys.argv) > 1 else r"I:\Github\Latent_Style\SchrodingerBridge\exp\630_local_t11_long30ep\full_eval\epoch_0001\summary.json"
d = json.load(open(p))
a = d["analysis"]["all_pairs_overview"]
s = d["analysis"]["style_transfer_ability"]
i = d["analysis"]["identity_reconstruction"]
print(f"all_pairs: clip={a['clip_style']:.4f} lpips={a['content_lpips']:.4f}")
print(f"style_only: clip={s['clip_style']:.4f} lpips={s['content_lpips']:.4f}")
print(f"identity: clip={i['clip_style']:.4f} lpips={i['content_lpips']:.4f}")
