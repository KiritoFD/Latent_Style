@echo off
cd /d I:\Github\Latent_Style\WEAVE
if not exist C:\Users\Administrator\logs mkdir C:\Users\Administrator\logs
echo === CHAIN: D3/D4/D5 matched ablations (waiting for batch1) === > C:\Users\Administrator\logs\chain_D3_D4_D5.log
echo Start wait: %DATE% %TIME% >> C:\Users\Administrator\logs\chain_D3_D4_D5.log

REM === Wait for batch 1 chain runner to complete ===
:WAIT_BATCH1
findstr /C:"CHAIN_RUNNER_EXIT=0" C:\Users\Administrator\logs\chain_after_seed7.log >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Waiting for batch1... %DATE% %TIME% >> C:\Users\Administrator\logs\chain_D3_D4_D5.log
    timeout /t 120 /nobreak >nul
    goto WAIT_BATCH1
)
echo Batch1 done. Starting D3/D4/D5... >> C:\Users\Administrator\logs\chain_D3_D4_D5.log
echo Batch1 done: %DATE% %TIME% >> C:\Users\Administrator\logs\chain_D3_D4_D5.log

REM === D3: lambda_LL=1.0 ===
echo === D3: Training lambda_LL=1.0 === >> C:\Users\Administrator\logs\chain_D3_D4_D5.log
echo D3 train start: %DATE% %TIME% >> C:\Users\Administrator\logs\chain_D3_D4_D5.log
python -u run.py --config configs\rebuttal_D3_wll_1p0.json >> C:\Users\Administrator\logs\rebuttal_D3_train.log 2>&1
echo D3 train done: %DATE% %TIME% >> C:\Users\Administrator\logs\chain_D3_D4_D5.log
echo D3_TRAIN_EXIT=%ERRORLEVEL% >> C:\Users\Administrator\logs\chain_D3_D4_D5.log

echo === D3: Per-epoch evaluation === >> C:\Users\Administrator\logs\chain_D3_D4_D5.log
python -u scripts\expA_per_epoch_eval.py --run_dir runs\submission\rebuttal_D3_wll_1p0 --seed 42 --tag D3 >> C:\Users\Administrator\logs\rebuttal_D3_eval.log 2>&1
echo D3_EVAL_EXIT=%ERRORLEVEL% >> C:\Users\Administrator\logs\chain_D3_D4_D5.log

REM === D4: Direct target endpoint ===
echo === D4: Training direct target endpoint === >> C:\Users\Administrator\logs\chain_D3_D4_D5.log
echo D4 train start: %DATE% %TIME% >> C:\Users\Administrator\logs\chain_D3_D4_D5.log
python -u run.py --config configs\rebuttal_D4_direct_target.json >> C:\Users\Administrator\logs\rebuttal_D4_train.log 2>&1
echo D4 train done: %DATE% %TIME% >> C:\Users\Administrator\logs\chain_D3_D4_D5.log
echo D4_TRAIN_EXIT=%ERRORLEVEL% >> C:\Users\Administrator\logs\chain_D3_D4_D5.log

echo === D4: Per-epoch evaluation === >> C:\Users\Administrator\logs\chain_D3_D4_D5.log
python -u scripts\expA_per_epoch_eval.py --run_dir runs\submission\rebuttal_D4_direct_target --seed 42 --tag D4 >> C:\Users\Administrator\logs\rebuttal_D4_eval.log 2>&1
echo D4_EVAL_EXIT=%ERRORLEVEL% >> C:\Users\Administrator\logs\chain_D3_D4_D5.log

REM === D5: Learned HH head ===
echo === D5: Training learned HH head === >> C:\Users\Administrator\logs\chain_D3_D4_D5.log
echo D5 train start: %DATE% %TIME% >> C:\Users\Administrator\logs\chain_D3_D4_D5.log
python -u run.py --config configs\rebuttal_D5_hh_head.json >> C:\Users\Administrator\logs\rebuttal_D5_train.log 2>&1
echo D5 train done: %DATE% %TIME% >> C:\Users\Administrator\logs\chain_D3_D4_D5.log
echo D5_TRAIN_EXIT=%ERRORLEVEL% >> C:\Users\Administrator\logs\chain_D3_D4_D5.log

echo === D5: Per-epoch evaluation === >> C:\Users\Administrator\logs\chain_D3_D4_D5.log
python -u scripts\expA_per_epoch_eval.py --run_dir runs\submission\rebuttal_D5_hh_head --seed 42 --tag D5 >> C:\Users\Administrator\logs\rebuttal_D5_eval.log 2>&1
echo D5_EVAL_EXIT=%ERRORLEVEL% >> C:\Users\Administrator\logs\chain_D3_D4_D5.log

echo CHAIN_D3_D4_D5_EXIT=0 >> C:\Users\Administrator\logs\chain_D3_D4_D5.log
echo All done: %DATE% %TIME% >> C:\Users\Administrator\logs\chain_D3_D4_D5.log
