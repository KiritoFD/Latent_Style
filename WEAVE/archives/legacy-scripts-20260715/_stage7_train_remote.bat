@echo off
cd /d I:\Github\Latent_Style\SchrodingerBridge
set PYTHONIOENCODING=utf-8
python -u src\run.py --config configs\exp_sty_stage7_delta.json > C:\Users\Administrator\logs\sty_inject\stage7_delta_train.out 2>&1
echo TRAIN_EXIT_CODE=%ERRORLEVEL% >> C:\Users\Administrator\logs\sty_inject\stage7_delta_train.out
