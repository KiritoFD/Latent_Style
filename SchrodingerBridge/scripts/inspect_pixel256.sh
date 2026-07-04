#!/usr/bin/env bash
echo "=== pixel256 distinct5 train/Early_Renaissance ==="
ls /mnt/i/wikiart_distinct5_samam_512_pixel256/train/Early_Renaissance/ 2>/dev/null | sed -n '1,3p'
echo ""
echo "=== check one .pt shape ==="
PYTHON=/home/xy/venvs/samam312/bin/python
"$PYTHON" - <<'PYEOF'
import torch, glob
files = sorted(glob.glob("/mnt/i/wikiart_distinct5_samam_512_pixel256/train/Early_Renaissance/*.pt"))[:1]
print("file:", files)
if files:
    obj = torch.load(files[0], map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        print("keys:", list(obj.keys()))
        for k in ("latent","latents","z","tensor","data"):
            if k in obj:
                print(f"  {k} shape:", obj[k].shape)
                break
    else:
        print("shape:", obj.shape, "dtype:", obj.dtype)
PYEOF
