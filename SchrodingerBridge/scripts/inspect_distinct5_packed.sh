#!/usr/bin/env bash
echo "=== .latent_cache 全部内容 ==="
find /mnt/i/wikiart_distinct5_samam_512_latent256/train/.latent_cache -maxdepth 3 -type f 2>/dev/null | head -20
echo ""
echo "=== .latent_cache 全部目录 ==="
find /mnt/i/wikiart_distinct5_samam_512_latent256/train/.latent_cache -maxdepth 3 -type d 2>/dev/null
echo ""
echo "=== 找所有 .pt 文件 in .latent_cache ==="
find /mnt/i/wikiart_distinct5_samam_512_latent256/train/.latent_cache -name "*.pt" 2>/dev/null | head -10
echo ""
echo "=== manifest 在哪 ==="
find /mnt/i/wikiart_distinct5_samam_512_latent256/train/.latent_cache -name "manifest.json" 2>/dev/null
echo ""
echo "=== manifest content (前 50 行) ==="
find /mnt/i/wikiart_distinct5_samam_512_latent256/train/.latent_cache -name "manifest.json" -exec cat {} \; 2>/dev/null | head -50
