@echo off
cd /d I:\Github\Latent_Style\SchrodingerBridge
set LANCET_BATCH_SIZE=192
set LANCET_EVAL_BATCH_SIZE=20
set PYTHONPATH=I:\Github\Latent_Style\SchrodingerBridge\src
if not exist logs mkdir logs
"C:\Program Files\Python312\python.exe" tools\experiments\run_dynamic_metric_probe.py --max-total 4 --train-epochs 8 --eval-epochs 6,7,8 --output-root exp/dynamic_metric_probe --config-root configs/dynamic_metric_probe --force-eval > logs\dynamic_metric_probe_remote.log 2>&1
