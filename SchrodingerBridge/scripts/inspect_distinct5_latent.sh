#!/usr/bin/env bash
echo "=== /mnt/i/wikiart_distinct5_samam_512_latent256/train ==="
ls -la /mnt/i/wikiart_distinct5_samam_512_latent256/train/ 2>/dev/null
echo "=== /mnt/i/wikiart_distinct5_samam_512_latent256/train/.latent_cache/packed ==="
ls -la /mnt/i/wikiart_distinct5_samam_512_latent256/train/.latent_cache/packed/ 2>/dev/null
echo "=== Early_Renaissance first 3 ==="
ls /mnt/i/wikiart_distinct5_samam_512_latent256/train/Early_Renaissance/ 2>/dev/null | head -3
echo "=== manifest.json ==="
cat /mnt/i/wikiart_distinct5_samam_512_latent256/train/.latent_cache/packed/manifest.json 2>/dev/null | head -30
echo "=== check latent shape (one file) ==="
PYTHON=/home/xy/venvs/samam312/bin/python
"$PYTHON" - <<'PYEOF'
import torch, glob, json
files = sorted(glob.glob("/mnt/i/wikiart_distinct5_samam_512_latent256/train/Early_Renaissance/*.pt"))[:1]
print("file:", files)
if files:
    obj = torch.load(files[0], map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        print("keys:", list(obj.keys())[:10])
        for k in ("latent","latents","z","tensor","data"):
            if k in obj:
                print(f"  {k} shape:", obj[k].shape)
                break
    else:
        print("shape:", obj.shape, "dtype:", obj.dtype)

packed = sorted(glob.glob("/mnt/i/wikiart_distinct5_samam_512_latent256/train/.latent_cache/packed/*.pt"))[:1]
print("packed:", packed)
if packed:
    obj = torch.load(packed[0], map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        print("packed keys:", list(obj.keys()))
        if "latents" in obj:
            print("packed latents shape:", obj["latents"].shape)
    else:
        print("packed shape:", obj.shape)
PYEOF
