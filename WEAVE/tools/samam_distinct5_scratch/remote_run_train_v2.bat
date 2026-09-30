@echo off
REM Create a scheduled task that runs immediately and survives SSH disconnect
REM Task runs as SYSTEM so it's independent of user session

set TASK_NAME=SaMam_Train_7k
set WSL_BASH=C:\Windows\System32\wsl.exe
set TRAIN_SCRIPT=/mnt/i/Github/Latent_Style/SchrodingerBridge/tools/samam_distinct5_scratch/remote_run_train.sh

REM Delete existing task if any
schtasks /Delete /TN "%TASK_NAME%" /F 2>nul

REM Create and run task as SYSTEM (independent of session)
REM Using wsl.exe with bash -c to run the training script
schtasks /Create /TN "%TASK_NAME%" /TR "%WSL_BASH% bash -c 'bash %TRAIN_SCRIPT% > /mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote/train.log 2>&1'" /SC ONCE /ST 00:00 /RU SYSTEM /F

REM Run the task immediately
schtasks /Run /TN "%TASK_NAME%"

echo Task created and started.
echo Waiting 30s for initialization...
timeout /t 30 /nobreak >nul

REM Check task status
schtasks /Query /TN "%TASK_NAME%" /V /FO LIST | findstr "Status\|Last\|Next"
