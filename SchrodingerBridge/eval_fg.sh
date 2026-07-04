#!/bin/bash
cd /home/xy/Latent_Style/SchrodingerBridge

echo "=== Starting fc_sb_sigma04 evaluation ==="
echo "Time: $(date)"

CKPT="/mnt/c/Users/Administrator/fc_sb_sigma04/checkpoints/epoch_0010.pt"
OUTPUT="/mnt/c/Users/Administrator/fc_sb_sigma04/full_eval"
STYLES="Hayao,cezanne,monet,photo,vangogh"

mkdir -p "$OUTPUT"

if [ ! -f "$CKPT" ]; then
    echo "ERROR: No checkpoint found at $CKPT"
    exit 1
fi

echo "Using checkpoint: $CKPT"
echo "Output dir: $OUTPUT"

# Run evaluation
python run_evaluation.py \
    --checkpoint "$CKPT" \
    --output "$OUTPUT" \
    --style_subdirs "$STYLES" \
    --batch_size 4 \
    --num_steps 12

echo "=== Evaluation complete at $(date) ==="
