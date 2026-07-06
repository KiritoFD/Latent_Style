#!/usr/bin/env bash
echo "===search for compare_256==="
find /mnt/i/Github/Latent_Style -name "compare_256*" -type f 2>/dev/null
echo "===search for any 256 comparison docs==="
find /mnt/i/Github/Latent_Style -name "*256*vs*" -type f 2>/dev/null
find /mnt/i/Github/Latent_Style -name "*photo2art*" -type f 2>/dev/null
echo "===check SchrodingerBridge docs root==="
ls /mnt/i/Github/Latent_Style/SchrodingerBridge/docs/*.md 2>/dev/null | head -20
echo "===search for latent migration docs==="
find /mnt/i/Github/Latent_Style/SchrodingerBridge/docs -name "*migration*" -type d 2>/dev/null
find /mnt/i/Github/Latent_Style/SchrodingerBridge/docs -name "*latent_migration*" -type d 2>/dev/null
echo "===search for SAMST/SaMam eval results==="
find /mnt/i/Github/Latent_Style -name "summary.json" -path "*samst*" 2>/dev/null | head -5
find /mnt/i/Github/Latent_Style -name "summary.json" -path "*samam*" 2>/dev/null | head -5
find /mnt/i/Github/Latent_Style -name "summary.json" -path "*adain*" 2>/dev/null | head -5
find /mnt/i/Github/Latent_Style -name "summary.json" -path "*wct*" 2>/dev/null | head -5
