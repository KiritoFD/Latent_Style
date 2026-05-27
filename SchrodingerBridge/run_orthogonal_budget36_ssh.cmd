@echo off
setlocal
cd /d I:\Github\Latent_Style\SchrodingerBridge

"C:\Program Files\Python312\python.exe" tools\experiments\run_orthogonal_budget36.py --train-epochs 6 --eval-epochs 4,6 --max-total 36 1> logs\orthogonal_budget36_remote.log 2> logs\orthogonal_budget36_remote.err.log

endlocal
