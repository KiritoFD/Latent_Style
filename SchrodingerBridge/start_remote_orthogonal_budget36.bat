@echo off
cd /d I:\Github\Latent_Style\SchrodingerBridge
set LANCET_BATCH_SIZE=160
set LANCET_EVAL_BATCH_SIZE=8
set PYTHONPATH=I:\Github\Latent_Style\SchrodingerBridge\src
if not exist logs mkdir logs
"C:\Program Files\Python312\python.exe" tools\experiments\run_orthogonal_budget36.py --train-epochs 6 --eval-epochs 4,6 --max-total 36 > logs\orthogonal_budget36_remote.log 2>&1
