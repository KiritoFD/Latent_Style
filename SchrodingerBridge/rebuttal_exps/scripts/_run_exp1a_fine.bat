@echo off
cd /d I:\Github\Latent_Style\WEAVE
if not exist C:\Users\Administrator\logs mkdir C:\Users\Administrator\logs
echo === STARTING EXP1A FINE SWEEP === > C:\Users\Administrator\logs\exp1a_fine_sweep.log
echo Started: %DATE% %TIME% >> C:\Users\Administrator\logs\exp1a_fine_sweep.log
python -u scripts\exp1a_fine_sweep.py >> C:\Users\Administrator\logs\exp1a_fine_sweep.log 2>&1
echo EXP1A_FINE_EXIT=%ERRORLEVEL% >> C:\Users\Administrator\logs\exp1a_fine_sweep.log
echo Finished: %DATE% %TIME% >> C:\Users\Administrator\logs\exp1a_fine_sweep.log
