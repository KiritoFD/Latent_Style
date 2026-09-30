@echo off
cd /d I:\Github\Latent_Style\WEAVE
if not exist C:\Users\Administrator\logs mkdir C:\Users\Administrator\logs
echo === CHAIN RUNNER: waiting for seed7, then seed42/123/B1/D === > C:\Users\Administrator\logs\chain_after_seed7.log

:WAIT_SEED7
findstr /C:"EXPA_SEED7_EXIT" C:\Users\Administrator\logs\expA_seed7.log >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Waiting for seed7... %DATE% %TIME% >> C:\Users\Administrator\logs\chain_after_seed7.log
    timeout /t 60 /nobreak >nul
    goto WAIT_SEED7
)
echo Seed7 done. Starting seed42... >> C:\Users\Administrator\logs\chain_after_seed7.log

REM === Exp A: seed42 (4 epochs) ===
python -u scripts\expA_per_epoch_eval.py --run_dir runs\submission\hf_oriented_internal_early_stop --seed 42 >> C:\Users\Administrator\logs\expA_seed42.log 2>&1
echo EXPA_SEED42_DONE >> C:\Users\Administrator\logs\chain_after_seed7.log

REM === Exp A: seed123 (3 epochs) ===
python -u scripts\expA_per_epoch_eval.py --run_dir runs\submission\robustness\early_stop_seed123 --seed 123 >> C:\Users\Administrator\logs\expA_seed123.log 2>&1
echo EXPA_SEED123_DONE >> C:\Users\Administrator\logs\chain_after_seed7.log

REM === Exp B1: reference-pool paired margin ===
python -u scripts\expB_reference_margin.py --n_iters 1000 >> C:\Users\Administrator\logs\expB1.log 2>&1
echo EXPB1_DONE >> C:\Users\Administrator\logs\chain_after_seed7.log

REM === Exp D: inference ablation (D1 + D2) ===
python -u scripts\expD_inference_ablation.py >> C:\Users\Administrator\logs\expD.log 2>&1
echo EXPD_DONE >> C:\Users\Administrator\logs\chain_after_seed7.log

echo CHAIN_RUNNER_EXIT=0 >> C:\Users\Administrator\logs\chain_after_seed7.log
echo All done: %DATE% %TIME% >> C:\Users\Administrator\logs\chain_after_seed7.log
