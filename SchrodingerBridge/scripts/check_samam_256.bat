@echo off
echo === SaMam 256 progress ===
wsl -- bash -lc "tail -30 /mnt/i/Github/Latent_Style/exp_samam/eval_256/samam_final_20k_256.log 2>/dev/null"
echo === Image count ===
wsl -- bash -lc "ls /mnt/i/Github/Latent_Style/exp_samam/eval_256/samam_final_20k_256/step_020000/images/ 2>/dev/null | wc -l"
echo === GPU ===
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
echo === DONE ===
