@echo off
setlocal
cd /d I:\Github\Latent_Style\SchrodingerBridge

"C:\Program Files\Python312\python.exe" tools\experiments\run_dynamic_metric_probe.py --max-total 4 --train-epochs 8 --eval-epochs 6,7,8 --output-root exp/dynamic_metric_probe --config-root configs/dynamic_metric_probe --force-eval 1> logs\dynamic_metric_probe_resume_e8.log 2> logs\dynamic_metric_probe_resume_e8.err.log

endlocal
