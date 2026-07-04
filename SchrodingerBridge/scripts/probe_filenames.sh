#!/usr/bin/env bash
echo "=== protocol_a_800 image filenames ==="
ls /mnt/i/Github/Latent_Style/seedream45_api/protocol_a_800/images/ 2>/dev/null | head -5
echo "..."
ls /mnt/i/Github/Latent_Style/seedream45_api/protocol_a_800/images/ 2>/dev/null | tail -3
echo ""
total=$(ls /mnt/i/Github/Latent_Style/seedream45_api/protocol_a_800/images/ 2>/dev/null | wc -l)
echo "Total: $total"
echo ""
echo "=== Check subdirs in protocol_a_800 ==="
ls -d /mnt/i/Github/Latent_Style/seedream45_api/protocol_a_800/*/ 2>/dev/null
