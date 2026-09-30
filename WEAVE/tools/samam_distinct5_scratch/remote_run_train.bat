@echo off
REM Launch SaMam training in WSL background, independent of SSH session
REM Uses `start` to create a detached window that survives SSH disconnect

set WSL_BASH=C:\Windows\System32\wsl.exe
set TRAIN_SCRIPT=/mnt/i/Github/Latent_Style/SchrodingerBridge/tools/samam_distinct5_scratch/remote_run_train.sh
set LOG_DIR=I:\Github\Latent_Style\Related_Works\baseline_pipeline\results\samam_distinct5_512_scratch_7k_250eval_remote
set LOG_FILE=%LOG_DIR%\train.log

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

REM Start WSL training in a detached new console window that stays open
start "SaMam_Train_7k" /MIN %WSL_BASH% bash -c "exec bash %TRAIN_SCRIPT% > /mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote/train.log 2>&1"

echo Training launched in detached window: SaMam_Train_7k
echo Log file: %LOG_FILE%
timeout /t 10 /nobreak >nul
echo === First 20 lines of log ===
type "%LOG_FILE%" 2>nul | more +0
