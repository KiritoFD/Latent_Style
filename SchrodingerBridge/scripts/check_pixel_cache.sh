#!/bin/bash
echo "===PIXEL CACHE STRUCTURE==="
find /mnt/i/legacy256_overfit50_pixel256/train/.latent_cache/ -maxdepth 3 -name "*.pt" 2>/dev/null | head -10
echo ""
echo "===PIXEL CACHE DIRS==="
ls -la /mnt/i/legacy256_overfit50_pixel256/train/.latent_cache/ 2>/dev/null
echo "---"
ls -la /mnt/i/legacy256_overfit50_pixel256/train/.latent_cache/packed/ 2>/dev/null
echo "---"
ls -la /mnt/i/legacy256_overfit50_pixel256/train/.latent_cache/packed/packed/ 2>/dev/null
echo ""
echo "===LATENT CACHE STRUCTURE (for comparison)==="
ls -la /mnt/i/legacy256_overfit50_latent256/train/.latent_cache/packed/ 2>/dev/null
echo "---"
ls -la /mnt/i/legacy256_overfit50_latent256/train/.latent_cache/packed/packed/ 2>/dev/null
echo ""
echo "===MANIFEST==="
cat /mnt/i/legacy256_overfit50_pixel256/train/.latent_cache/packed/manifest.json 2>/dev/null | head -5 || echo "NO manifest at packed/"
cat /mnt/i/legacy256_overfit50_pixel256/train/.latent_cache/manifest.json 2>/dev/null | head -5 || echo "NO manifest at .latent_cache/"
