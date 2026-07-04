#!/usr/bin/env bash
echo "=== latent256_e10 image filenames ==="
ls /mnt/c/Users/Administrator/exp/latent256_sfm/latent256_b16_e10/full_eval/epoch_0010/images/ 2>/dev/null | head -5
echo ""
echo "=== latent256_e10 config data section ==="
grep -A 10 '"data"' /mnt/c/Users/Administrator/exp/latent256_sfm/latent256_b16_e10/config.json 2>/dev/null
echo ""
echo "=== latent256_e10 config full_eval test_image_dir ==="
grep -E 'test_image_dir|data_root|style_subdirs' /mnt/c/Users/Administrator/exp/latent256_sfm/latent256_b16_e10/config.json 2>/dev/null
echo ""
echo "=== Search for legacy256 overfit50 anywhere ==="
find /mnt -maxdepth 6 -type d -iname "legacy256*" 2>/dev/null | head -10
echo ""
echo "=== Find test dataset with photo2art 5 styles ==="
find /mnt -maxdepth 5 -type d -name "cezanne" 2>/dev/null | grep -v "exp_" | head -10
