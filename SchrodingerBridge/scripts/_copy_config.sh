#!/usr/bin/env bash
set -uo pipefail
cp /mnt/c/Users/Administrator/630_pixel_256_photo2art.json /mnt/i/Github/Latent_Style/SchrodingerBridge/configs/
echo COPIED
grep -E "num_workers|persistent_workers|latent_cache_dir" /mnt/i/Github/Latent_Style/SchrodingerBridge/configs/630_pixel_256_photo2art.json
