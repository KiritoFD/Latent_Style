#!/usr/bin/env bash
find /mnt/i/Github/Latent_Style/SchrodingerBridge/docs -name "compare*" -type f 2>/dev/null
echo "==="
find /mnt/i/Github/Latent_Style/SchrodingerBridge/docs -name "*256*photo2art*" -type f 2>/dev/null
echo "==="
find /mnt/i/Github/Latent_Style/SchrodingerBridge/docs -name "*baseline_256*" -type d 2>/dev/null
echo "==="
ls /mnt/i/Github/Latent_Style/SchrodingerBridge/docs/baseline_256/ 2>/dev/null
