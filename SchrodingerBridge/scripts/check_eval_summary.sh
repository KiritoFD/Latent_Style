#!/bin/bash
echo "===EPOCH_0001 SUMMARY==="
cat /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/latent256_photo2art/latent256_b16_e10/full_eval/epoch_0001/summary.json 2>/dev/null | python3 -m json.tool 2>/dev/null || cat /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/latent256_photo2art/latent256_b16_e10/full_eval/epoch_0001/summary.json 2>/dev/null
echo ""
echo "===CURVE SUMMARY==="
cat /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/latent256_photo2art/latent256_b16_e10/full_eval/curve_summary.json 2>/dev/null
echo ""
echo "===CLIP_LPIPS_CURVE==="
cat /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/latent256_photo2art/latent256_b16_e10/full_eval/clip_lpips_curve.csv 2>/dev/null
echo ""
echo "===EVAL DIRS==="
ls /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/latent256_photo2art/latent256_b16_e10/full_eval/
