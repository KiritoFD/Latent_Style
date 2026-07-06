#!/bin/bash
echo "===EPOCH_0001 EVAL FILES==="
ls -la /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/latent256_photo2art/latent256_b16_e10/full_eval/epoch_0001/ 2>/dev/null
echo ""
echo "===METRICS CSV==="
cat /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/latent256_photo2art/latent256_b16_e10/full_eval/epoch_0001/metrics.csv 2>/dev/null || echo "NO metrics.csv"
echo ""
echo "===IMAGES DIR==="
ls /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/latent256_photo2art/latent256_b16_e10/full_eval/epoch_0001/images/ 2>/dev/null | head -10
echo ""
echo "===TRAIN LOG LAST 10 LINES==="
tail -10 /mnt/i/exp_256_photo2art/_train_latent256_photo2art.log | tr '\r' '\n' | tail -10
echo ""
echo "===GPU==="
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
