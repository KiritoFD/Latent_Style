#!/bin/bash
echo "===CKPT DIR==="
ls -la /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/latent256_photo2art/latent256_b16_e10/
echo ""
echo "===FULL_EVAL DIR==="
ls -la /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/latent256_photo2art/latent256_b16_e10/full_eval/ 2>/dev/null || echo "NO full_eval DIR"
echo ""
echo "===EPOCH_0001 EVAL==="
ls /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/latent256_photo2art/latent256_b16_e10/full_eval/epoch_0001/ 2>/dev/null | head -20
echo ""
echo "===TRAIN LOG TAIL==="
tail -5 /mnt/i/exp_256_photo2art/_train_latent256_photo2art.log | tr '\r' '\n' | tail -5
echo ""
echo "===GPU==="
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
echo "===PROCESS==="
ps -ef | grep -E "python|run.py" | grep -v grep || echo "NO PYTHON RUNNING"
