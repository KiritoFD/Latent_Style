#!/bin/bash
cd /home/xy/Latent_Style/SchrodingerBridge

echo "=== Starting fc_sb_sigma04 evaluation ==="
echo "Time: $(date)" > /tmp/eval_fc_sb_v3_log.txt

CKPT="/mnt/c/Users/Administrator/fc_sb_sigma04/checkpoints/epoch_0010.pt"
OUTPUT="/mnt/c/Users/Administrator/fc_sb_sigma04/full_eval"
STYLES="Hayao,cezanne,monet,photo,vangogh"

mkdir -p "$OUTPUT"

if [ ! -f "$CKPT" ]; then
    echo "ERROR: No checkpoint found at $CKPT" >> /tmp/eval_fc_sb_v3_log.txt
    exit 1
fi

echo "Using checkpoint: $CKPT" >> /tmp/eval_fc_sb_v3_log.txt
echo "Output dir: $OUTPUT" >> /tmp/eval_fc_sb_v3_log.txt

# Run evaluation with nohup
nohup python run_evaluation.py \
    --checkpoint "$CKPT" \
    --output "$OUTPUT" \
    --style_subdirs "$STYLES" \
    --batch_size 4 \
    --num_steps 12 \
    > "$OUTPUT/eval_log.txt" 2>&1 &

echo "Started with PID: $!" >> /tmp/eval_fc_sb_v3_log.txt
echo "=== Started ===" >> /tmp/eval_fc_sb_v3_log.txt
