#!/usr/bin/env bash
# Only run Step 2 (pack) for legacy256 latent cache (Step 1 already done).
set -uo pipefail
PYTHON=/home/xy/venvs/samam312/bin/python
REPO=/mnt/i/Github/Latent_Style/SchrodingerBridge
PACK_SCRIPT=$REPO/tools/build_latent_packed_cache.py
OUTPUT_ROOT=/mnt/i/legacy256_overfit50_latent256/train
STYLES_CSV="cezanne,Hayao,monet,photo,vangogh"
CACHE_DIR=$OUTPUT_ROOT/.latent_cache/packed

echo "[INFO] Pack legacy256 latent cache"
echo "START=$(date '+%Y-%m-%dT%H:%M:%S')"
timeout 600 "$PYTHON" -u "$PACK_SCRIPT" \
    --data-root "$OUTPUT_ROOT" \
    --styles "$STYLES_CSV" \
    --cache-dir "$CACHE_DIR"
RC=$?
echo "PACK_RC=$RC"
echo "=== Verify ==="
ls -la "$CACHE_DIR"/packed/ 2>/dev/null
"$PYTHON" - <<'PYEOF'
import torch, glob
files = sorted(glob.glob("/mnt/i/legacy256_overfit50_latent256/train/.latent_cache/packed/packed/*.pt"))
print("packed files:", len(files))
for f in files:
    obj = torch.load(f, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        print(f"  {f.split('/')[-1]}: subdir={obj.get('subdir')} count={obj.get('count')} shape={obj.get('latents').shape if 'latents' in obj else 'N/A'}")
PYEOF
echo "END=$(date '+%Y-%m-%dT%H:%M:%S')"
exit $RC
