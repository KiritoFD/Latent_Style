@echo off
cd /d I:\Github\Latent_Style\SchrodingerBridge
set PYTHONPATH=
set PYTHON="C:\Program Files\Python312\python.exe"
%PYTHON% run.py --config configs/p4_n11_n16_gate03_whh25.json > exp\p4_fusion_breakout\n11_n16_train.log 2>&1
echo TRAIN_EXIT_CODE=%ERRORLEVEL%
