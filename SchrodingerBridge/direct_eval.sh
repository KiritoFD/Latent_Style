#!/bin/bash
echo "=== Running eval at $(date) ==="
cd /home/xy/Latent_Style/SchrodingerBridge

CKPT="/mnt/c/Users/Administrator/fc_sb_sigma04/checkpoints/epoch_0010.pt"
CONFIG="/mnt/c/Users/Administrator/fc_sb_sigma04/checkpoints/config.json"
STYLES="Hayao,cezanne,monet,photo,vangogh"

echo "Checking files..."
ls -la "$CKPT" 2>&1 || echo "No checkpoint"
ls -la "$CONFIG" 2>&1 || echo "No config"

echo ""
echo "Starting evaluation..."
python run.py \
    --config "$CONFIG" \
    --eval_only \
    --checkpoint_path "$CKPT" \
    --style_subdirs "$STYLES" \
    2>&1 | tee "/mnt/c/Users/Administrator/fc_sb_sigma04/eval_log.txt"

echo "=== Done at $(date) ==="
