@echo off
cd /d C:\Users\Administrator
set PYTHON=C:\Program Files\Python312\python.exe
if not exist "%PYTHON%" set PYTHON=C:\Users\Administrator\AppData\Local\Programs\Python\Python312\python.exe
"%PYTHON%" src\utils\run_evaluation.py --checkpoint C:\Users\Administrator\exp\latent256_sfm\latent256_b16_e10\epoch_0010.pt --output C:\Users\Administrator\exp\latent256_sfm\latent256_b16_e10\full_eval\epoch_0010 --test_dir I:\wikiart_distinct5_samam_512_classview\test --cache_dir C:\Users\Administrator\exp\eval_cache --clip_hf_cache_dir I:\Github\Latent_Style\eval_cache\hf --batch_size 2 --num_steps 8 --max_ref_compare 8 --max_ref_cache 8 --ref_feature_batch_size 2 --target_chunk_size 1 --force_regen > C:\Users\Administrator\logs\latent256_eval.log 2>&1
echo EVAL_EXIT_CODE=%ERRORLEVEL% >> C:\Users\Administrator\logs\latent256_eval.log
