import os, json
p = r"I:/GitHub/Latent_Style/SchrodingerBridge/tools/zstar"
out = {"zstar_pkg": os.path.isdir(p)}
if os.path.isdir(p):
    out["files"] = sorted(os.listdir(p))
# also confirm SD1.5 hub cache
sd = r"C:/Users/Administrator/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5"
out["sd15_hub"] = os.path.isdir(sd)
print(json.dumps(out, indent=2))
