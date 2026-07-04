@echo off
cd /d C:\Users\Administrator
set PYTHON=C:\Users\Administrator\AppData\Local\Programs\Python\Python312\python.exe
if not exist "%PYTHON%" set PYTHON=C:\Program Files\Python312\python.exe
set PYTHONUNBUFFERED=1
echo TRAIN_START=%date% %time% > C:\Users\Administrator\logs\latent256_train.log
"%PYTHON%" -u run.py --config configs\630_latent_256.json >> C:\Users\Administrator\logs\latent256_train.log 2>&1
echo TRAIN_EXIT_CODE=%ERRORLEVEL% >> C:\Users\Administrator\logs\latent256_train.log
echo TRAIN_END=%date% %time% >> C:\Users\Administrator\logs\latent256_train.log
exit 0
