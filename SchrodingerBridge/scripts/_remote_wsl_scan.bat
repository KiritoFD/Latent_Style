@echo off
wsl -e bash -c "ls /mnt/i/ 2>/dev/null | head -20; echo '---'; ls /mnt/i/Github/Latent_Style/ 2>/dev/null | head -10; echo '---'; ls /mnt/i/wikiart_distinct5_samam_512_classview/train/ 2>/dev/null | head -10; echo '---'; ls /mnt/i/wikiart_distinct5_samam_512_latents_ema/train/ 2>/dev/null | head -10"
