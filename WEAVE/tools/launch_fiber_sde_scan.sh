#!/bin/bash
# Launch Fiber-SDE scan on topogate e2 checkpoint
# Theory: noise × TopoGate breaks ODE mean collapse without destroying structure

cd /mnt/i/Github/Latent_Style/SchrodingerBridge

CKPT="/mnt/i/Github/Latent_Style/exp/aaai2027_phase2_vel_tok32_safe_semantic_topogate_k085_appalign_seed42_b12a1/epoch_0002.pt"

echo "=== Fiber-SDE Scan: Testing fiber-aligned vs isotropic noise ==="
echo "Baseline: topogate e2 (0.671/0.314)"
echo ""

# Fiber-aligned sigma scan (priority: breaks mean collapse)
for sigma in 0.005 0.01 0.02 0.03 0.05; do
    echo "Launching fiber-aligned sigma=$sigma"
    python src/run.py \
        --config configs/aaai2027/phase2_fiber_sde_fiber_sigma0p${sigma/.}.json \
        --override training.resume_checkpoint=$CKPT \
        > /tmp/fiber_sde_fiber_${sigma}.log 2>&1
    
    # Extract result
    tail -50 /tmp/fiber_sde_fiber_${sigma}.log | grep -E "transfer_clip_style|transfer_content_lpips|all_pairs"
    echo ""
done

# Isotropic control (should degrade LPIPS)
for sigma in 0.005 0.01 0.02; do
    echo "Launching isotropic sigma=$sigma (control)"
    python src/run.py \
        --config configs/aaai2027/phase2_fiber_sde_iso_sigma0p${sigma/.}.json \
        --override training.resume_checkpoint=$CKPT \
        > /tmp/fiber_sde_iso_${sigma}.log 2>&1
    
    tail -50 /tmp/fiber_sde_iso_${sigma}.log | grep -E "transfer_clip_style|transfer_content_lpips"
    echo ""
done

echo "=== Scan complete. Check logs in /tmp/fiber_sde_*.log ==="
