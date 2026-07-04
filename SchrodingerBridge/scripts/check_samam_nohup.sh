#!/bin/bash
echo "=== nohup log ==="
cat /mnt/i/exp_samam_latent_train_nohup.log 2>/dev/null | tail -50
echo ""
echo "=== dmesg | grep -i oom ==="
dmesg 2>/dev/null | grep -iE "oom|kill" | tail -10
echo ""
echo "=== Check if vae_gradient_checkpointing helps ==="
echo "Plan: re-launch with batch=2, iter=2000, all gradient_checkpointing=1"
