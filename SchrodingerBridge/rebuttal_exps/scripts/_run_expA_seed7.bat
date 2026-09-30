@echo off
cd /d I:\Github\Latent_Style\WEAVE
if not exist C:\Users\Administrator\logs mkdir C:\Users\Administrator\logs
echo === STARTING EXP A SEED7 PER-EPOCH EVAL === > C:\Users\Administrator\logs\expA_seed7.log
echo Started: %DATE% %TIME% >> C:\Users\Administrator\logs\expA_seed7.log
python -u scripts\expA_per_epoch_eval.py --run_dir runs\submission\robustness\early_stop_seed7 --seed 7 >> C:\Users\Administrator\logs\expA_seed7.log 2>&1
echo EXPA_SEED7_EXIT=%ERRORLEVEL% >> C:\Users\Administrator\logs\expA_seed7.log
echo Finished: %DATE% %TIME% >> C:\Users\Administrator\logs\expA_seed7.log
