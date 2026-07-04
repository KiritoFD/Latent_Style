#!/bin/bash
cd /home/xy/Latent_Style/SchrodingerBridge

echo "=== Starting fc_sb_sigma04 evaluation ==="
echo "Time: $(date)"

CKPT="/mnt/c/Users/Administrator/fc_sb_sigma04/checkpoints/epoch_0010.pt"
STYLES="Hayao,cezanne,monet,photo,vangogh"

if [ ! -f "$CKPT" ]; then
    echo "ERROR: No checkpoint found at $CKPT"
    exit 1
fi

if [ ! -f "/mnt/c/Users/Administrator/fc_sb_sigma04/checkpoints/config.json" ]; then
    echo "ERROR: No config found"
    exit 1
fi

echo "Using checkpoint: $CKPT"
echo "Using config: /mnt/c/Users/Administrator/fc_sb_sigma04/checkpoints/config.json"

# Run evaluation
python run.py \
    --config "/mnt/c/Users/Administrator/fc_sb_sigma04/checkpoints/config.json" \
    --eval_only \
    --checkpoint_path "$CKPT" \
    --style_subdirs "$STYLES" \
    2>&1 | tee "/mnt/c/Users/Administrator/fc_sb_sigma04/eval_log.txt"

echo "=== Evaluation complete at $(date) ==="
