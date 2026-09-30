#!/bin/bash
# Run eval with explicit image saving on intrinsic_v2 epoch 8
cd /mnt/i/Github/Latent_Style/SchrodingerBridge

export PYTHONPATH=/mnt/i/Github/Latent_Style/SchrodingerBridge/src:$PYTHONPATH

python3 src/utils/run_evaluation.py \
    --checkpoint /mnt/i/Github/Latent_Style/exp/620_spatial_bridge/620_intrinsic_v2/epoch_0008.pt \
    --output /mnt/i/Github/Latent_Style/exp/620_spatial_bridge/620_intrinsic_v2/full_eval_wfi \
    --save-generated-images \
    --num-steps 8 \
    --batch-size 4 \
    --target_dino_cache /mnt/i/Github/Latent_Style/eval_cache/offline_pairing/dinov2_small_wikiart_distinct5_train_cache.pt \
    2>&1 | tee /mnt/i/Github/Latent_Style/exp/620_spatial_bridge/wfi_eval_intrinsic_v2.log

echo "=== Eval done, running WFI ==="
python3 /mnt/i/Github/Latent_Style/wfi_tools/run_wfi_benchmark.py \
    --checkpoint-dir /mnt/i/Github/Latent_Style/exp/620_spatial_bridge/620_intrinsic_v2 \
    --output-dir /mnt/i/Github/Latent_Style/exp/620_spatial_bridge/wfi_benchmark \
    2>&1 | tee -a /mnt/i/Github/Latent_Style/exp/620_spatial_bridge/wfi_eval_intrinsic_v2.log
