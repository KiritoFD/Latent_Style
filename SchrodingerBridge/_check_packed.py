"""Check packed latent cache files."""
import torch
import os

packed_dir = r"I:\wikiart_distinct5_samam_512_latents_ema\train\.latent_cache\packed\packed"
for fname in sorted(os.listdir(packed_dir)):
    if not fname.endswith(".pt"):
        continue
    fpath = os.path.join(packed_dir, fname)
    try:
        payload = torch.load(fpath, map_location="cpu", weights_only=False)
        if isinstance(payload, dict):
            keys = list(payload.keys())
            schema = payload.get("schema")
            subdir = payload.get("subdir")
            count = payload.get("count")
            latents_shape = payload["latents"].shape if "latents" in payload else None
            print(f"{fname}: dict keys={keys} schema={schema} subdir={subdir} count={count} latents={latents_shape}")
        else:
            print(f"{fname}: type={type(payload).__name__} shape={payload.shape if hasattr(payload, 'shape') else 'N/A'}")
    except Exception as e:
        print(f"{fname}: ERROR {e}")
