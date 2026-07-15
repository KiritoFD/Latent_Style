#!/usr/bin/env bash
echo "=== Search samam_wsl_mamba_512_scratch_clean_silent_b1_20k ==="
find /mnt/i/Github/Latent_Style -maxdepth 5 -type d -name 'samam_wsl_mamba_512_scratch_clean_silent_b1_20k*' 2>/dev/null
echo "---"
echo "=== Check recovered json ==="
cat /mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_mamba_b6_seg250_remote_wsl_20260601_2130_diag/eval_curve/curve_metrics_recovered.json 2>/dev/null | head -100
echo "---"
echo "=== Check curve_metrics.json (diag) ==="
cat /mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_mamba_b6_seg250_remote_wsl_20260601_2130_diag/eval_curve/curve_metrics.json 2>/dev/null | head -50
echo "=== DONE ==="
