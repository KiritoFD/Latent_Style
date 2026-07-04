#!/bin/bash
cd /home/xy/Latent_Style/SchrodingerBridge

echo "=== Evaluating fc_sb_sigma04 ==="
CKPT="/mnt/c/Users/Administrator/fc_sb_sigma04/checkpoints/epoch_0010.pt"
STYLES="Hayao,cezanne,monet,photo,vangogh"

if [ ! -f "$CKPT" ]; then
    echo "No checkpoint found"
    exit 1
fi

echo "Checkpoint: $CKPT"

# Run evaluation
python run.py \
    --config "/mnt/c/Users/Administrator/fc_sb_sigma04/checkpoints/config.json" \
    --eval_only \
    --checkpoint_path "$CKPT" \
    --style_subdirs "$STYLES" \
    2>&1 | tee "/mnt/c/Users/Administrator/fc_sb_sigma04/eval_log.txt"

echo "=== fc_sb_sigma04 evaluation complete ==="
