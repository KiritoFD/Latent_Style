@echo off
echo === KEEPALIVE HISTORY ===
type C:\Users\Administrator\keepalive_history.log 2>nul
echo.
echo === WSL LIST ===
wsl -l -v
echo.
echo === FULL TRAIN.LOG (last 80 lines) ===
wsl -d Ubuntu-22.04 -e bash -c "tail -80 /mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote/train.log 2>&1"
echo.
echo === DMESG (last 20 lines, may show OOM) ===
wsl -d Ubuntu-22.04 -e bash -c "dmesg 2>/dev/null | tail -20"
echo.
echo === FREE MEMORY ===
wsl -d Ubuntu-22.04 -e bash -c "free -h"
echo.
echo === DMESG GREP OOM ===
wsl -d Ubuntu-22.04 -e bash -c "dmesg 2>/dev/null | grep -i 'oom\|killed\|out of memory' | tail -10"
exit /b 0
