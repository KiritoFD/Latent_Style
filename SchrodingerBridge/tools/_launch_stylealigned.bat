@echo off
cd /d I:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_stylealigned
python I:\GitHub\Latent_Style\SchrodingerBridge\tools\_run_stylealigned_remote.py > stylealigned_remote.log 2>&1
echo DONE >> stylealigned_remote.log
