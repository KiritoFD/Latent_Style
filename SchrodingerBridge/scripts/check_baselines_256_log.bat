@echo off
echo === Baselines 256 log ===
wsl -- bash -lc "cat /mnt/i/Github/Latent_Style/exp_baseline_256/baseline_256.log 2>&1"
echo === Image counts ===
wsl -- bash -lc "find /mnt/i/Github/Latent_Style/exp_baseline_256 -name '*.png' 2>/dev/null | wc -l"
echo === Per-method counts ===
wsl -- bash -lc "for m in adain wct samst; do echo -n '$m: '; find /mnt/i/Github/Latent_Style/exp_baseline_256/$m -name '*.png' 2>/dev/null | wc -l; done"
echo === GPU ===
wsl -- bash -lc "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1"
echo === DONE ===
