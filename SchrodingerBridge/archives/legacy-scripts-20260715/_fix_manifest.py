import json

remote_path = r"G:\GitHub\Latent_Style\Dataset\wikiart_distinct5_samam_512_latents_ema\train\.latent_cache\packed\manifest_remote.json"
local_path = r"G:\GitHub\Latent_Style\Dataset\wikiart_distinct5_samam_512_latents_ema\train\.latent_cache\packed\manifest.json"
local_data_root = r"G:\GitHub\Latent_Style\Dataset\wikiart_distinct5_samam_512_latents_ema\train"

with open(remote_path, "r", encoding="utf-8") as f:
    m = json.load(f)

m["data_root"] = local_data_root

with open(local_path, "w", encoding="utf-8") as f:
    json.dump(m, f, indent=2, ensure_ascii=False)

print("OK")
print("data_root=", m["data_root"])
for s in m["styles"]:
    print(f"  {s}: count={m['styles'][s]['count']}")
