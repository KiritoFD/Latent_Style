@echo off
REM Test Windows python with one ablation config (smoke test)
REM This tests: config loading, dataset path resolution, model build (no training)
set REPO=I:\Github\Latent_Style\SchrodingerBridge
set PYTHON=C:\Program Files\Python312\python.exe
set CONFIG=%REPO%\configs\abl512_X06_no_spectral_ode.json
set PYTHONPATH=%REPO%\src
set CUDA_VISIBLE_DEVICES=0
set HF_HOME=%REPO%\exp\eval_cache\hf

cd /d %REPO%
echo === Windows Python smoke test ===
echo Python: %PYTHON%
echo Config: %CONFIG%
echo Repo:   %REPO%
echo.
echo === Python version and torch check ===
"%PYTHON%" -c "import sys; print('Python:', sys.version); import torch; print('Torch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
echo.
echo === Config load test ===
"%PYTHON%" -c "import json; c=json.load(open(r'%CONFIG%')); print('Config loaded OK'); print('data_root:', c['data']['data_root']); print('test_image_dir:', c['training']['test_image_dir'])"
echo.
echo === Dataset path existence check ===
"%PYTHON%" -c "import os; p=r'I:\datasets\wikiart_distinct5_samam_512_latents_ema\train'; print('Train dir exists:', os.path.isdir(p)); p2=r'I:\datasets\wikiart_distinct5_samam_512_classview\test'; print('Test dir exists:', os.path.isdir(p2))"
echo.
echo === DONE ===
