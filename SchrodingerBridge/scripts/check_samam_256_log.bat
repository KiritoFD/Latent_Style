@echo off
echo === SaMam 256 log ===
wsl -- bash -lc "cat /mnt/i/Github/Latent_Style/exp_samam/eval_256/samam_final_20k_256.log 2>&1"
echo === Image count so far ===
wsl -- bash -lc "find /mnt/i/Github/Latent_Style/exp_samam/eval_256/samam_final_20k_256/ -name '*.png' 2>/dev/null | wc -l"
echo === GPU status ===
wsl -- bash -lc "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>&1"
echo === DONE ===
