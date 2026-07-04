@echo off
cd /d C:\Users\Administrator
set PYTHON=C:\Program Files\Python312\python.exe
if not exist "%PYTHON%" set PYTHON=C:\Users\Administrator\AppData\Local\Programs\Python\Python312\python.exe
set PYTHONUNBUFFERED=1
del /q C:\Users\Administrator\exp\pixel256_sfm\pixel256_b2_e10\full_eval\epoch_0003\*.png 2>nul
del /q C:\Users\Administrator\exp\pixel256_sfm\pixel256_b2_e10\full_eval\epoch_0003\summary.json 2>nul
echo EVAL_START=%date% %time% > C:\Users\Administrator\logs\pixel256_eval.log
"%PYTHON%" -u scripts\eval_pixel128.py --checkpoint C:\Users\Administrator\exp\pixel256_sfm\pixel256_b2_e10\epoch_0003.pt --config C:\Users\Administrator\configs\630_pixel_256.json --test_dir I:\wikiart_distinct5_samam_512_classview\test --output C:\Users\Administrator\exp\pixel256_sfm\pixel256_b2_e10\full_eval\epoch_0003 --clip_cache_dir I:\Github\Latent_Style\eval_cache\hf --pixel_size 256 --max_per_style 10 >> C:\Users\Administrator\logs\pixel256_eval.log 2>&1
echo EVAL_EXIT_CODE=%ERRORLEVEL% >> C:\Users\Administrator\logs\pixel256_eval.log
echo EVAL_END=%date% %time% >> C:\Users\Administrator\logs\pixel256_eval.log
