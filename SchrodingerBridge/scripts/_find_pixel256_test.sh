#!/usr/bin/env bash
set -uo pipefail
echo "===wikiart 256 dirs==="
ls -d /mnt/i/wikiart*256* 2>/dev/null
ls -d /mnt/i/wikiart_distinct5*256* 2>/dev/null
echo "===pixel256 config==="
cat /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/pixel256_photo2art/pixel256_b1_e5_softmax/epoch_0003.pt.config.json 2>/dev/null | head -40
echo "===config json files==="
find /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/pixel256_photo2art -name "*.json" 2>/dev/null | head -5
find /mnt/i/Github/Latent_Style/SchrodingerBridge/configs -name "*pixel*" 2>/dev/null
