@echo off
REM Chain runner: waits for exp1c to finish, then runs exp1ab training sweep
REM Polls for the EXP1C_EXIT marker in exp1c_adain_sweep.log every 30s
cd /d I:\Github\Latent_Style\WEAVE
if not exist C:\Users\Administrator\logs mkdir C:\Users\Administrator\logs

echo === WAITING FOR EXP1C TO FINISH === > C:\Users\Administrator\logs\exp1ab_train_sweep.log
echo Started: %DATE% %TIME% >> C:\Users\Administrator\logs\exp1ab_train_sweep.log

:WAIT_LOOP
findstr /C:"EXP1C_EXIT=" C:\Users\Administrator\logs\exp1c_adain_sweep.log >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Still waiting for exp1c to finish... %DATE% %TIME% >> C:\Users\Administrator\logs\exp1ab_train_sweep.log
    timeout /t 30 /nobreak >nul
    goto WAIT_LOOP
)

echo === EXP1C FINISHED, STARTING EXP1AB === >> C:\Users\Administrator\logs\exp1ab_train_sweep.log
echo Started exp1ab: %DATE% %TIME% >> C:\Users\Administrator\logs\exp1ab_train_sweep.log

python -u scripts\exp1ab_train_sweep.py >> C:\Users\Administrator\logs\exp1ab_train_sweep.log 2>&1
echo EXP1AB_EXIT=%ERRORLEVEL% >> C:\Users\Administrator\logs\exp1ab_train_sweep.log
echo Finished: %DATE% %TIME% >> C:\Users\Administrator\logs\exp1ab_train_sweep.log
