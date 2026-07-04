#!/bin/bash
exec > /home/xy/checkpoint_check.txt 2>&1
for d in fc_sb_kernel7 fc_sb_floor0 fc_sb_curriculum fc_sb_fiber_ep fc_sb_wavelet; do
  echo "=== $d ==="
  ls -la /home/xy/Latent_Style/SchrodingerBridge/exp/p3_remote_10h/$d/checkpoints/ 2>/dev/null || echo "NO_CHECKPOINT_DIR"
  echo ""
done
