@echo off
echo === KEEPALIVE TASK STATUS ===
schtasks /Query /TN WSL_KeepAlive /FO LIST /V
echo.
echo === KEEPALIVE LOG (persistent) ===
wsl -d Ubuntu-22.04 -e bash -c "cat /mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote/keepalive.log 2>/dev/null | tail -20"
echo.
echo === WSL LIST ===
wsl -l -v
echo.
echo === RECENT WSL EVENTS ===
wevtutil qe System /q:"*[System[Provider[@Name='WSL']]]" /c:10 /rd:true /f:text 2>nul
echo.
echo === KEEPALIVE BAT CONTENT ===
type C:\Users\Administrator\keep_wsl_alive.bat
exit /b 0
