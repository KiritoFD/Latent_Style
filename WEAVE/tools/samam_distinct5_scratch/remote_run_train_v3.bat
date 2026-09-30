@echo off
REM Launch SaMam training via PowerShell Start-Process with redirect
REM This creates a truly detached process that survives SSH disconnect
REM Runs as current user (administrator) to preserve WSL GPU access

set LOG_DIR=I:\Github\Latent_Style\Related_Works\baseline_pipeline\results\samam_distinct5_512_scratch_7k_250eval_remote
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

REM Clear old log
del "%LOG_DIR%\train.log" 2>nul

REM Use PowerShell Start-Process to create detached process with redirection
powershell -Command "Start-Process -FilePath 'C:\Windows\System32\wsl.exe' -ArgumentList 'bash','-c','exec bash /mnt/i/Github/Latent_Style/SchrodingerBridge/tools/samam_distinct5_scratch/remote_run_train.sh > /mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote/train.log 2>&1' -WindowStyle Hidden -RedirectStandardOutput '%LOG_DIR%\stdout.log' -RedirectStandardError '%LOG_DIR%\stderr.log' -PassThru | Select-Object Id | Out-File -FilePath '%LOG_DIR%\pid.txt' -Encoding ASCII"

echo Launched. Waiting 30s...
timeout /t 30 /nobreak >nul
echo === PID ===
type "%LOG_DIR%\pid.txt" 2>nul
echo === LOG TAIL ===
type "%LOG_DIR%\train.log" 2>nul
echo === STDERR ===
type "%LOG_DIR%\stderr.log" 2>nul
