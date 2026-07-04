@echo off
REM Phase 8E: Validate clean_base.json - train + eval
set PYTHON=C:\Progra~1\Python312\python.exe
set ROOT=I:\Github\Latent_Style\SchrodingerBridge
set LOGDIR=%ROOT%\exp\clean_base
mkdir "%LOGDIR%" 2>nul

echo [%date% %time%] === Clean Base Validation START === > "%LOGDIR%\validate.log"

echo [%date% %time%] Step 1: Train clean_base from T5 ep7 >> "%LOGDIR%\validate.log"
cd /d "%ROOT%"
"%PYTHON%" "%ROOT%\src\run.py" --config "%ROOT%\configs\clean_base.json" >> "%LOGDIR%\validate.log" 2>&1

echo [%date% %time%] Step 2: Evaluate epoch_0010 >> "%LOGDIR%\validate.log"
"%PYTHON%" "%ROOT%\src\utils\run_evaluation.py" ^
    --checkpoint "%ROOT%\exp\clean_base\epoch_0010.pt" ^
    --output "%ROOT%\exp\clean_base\full_eval\epoch_0010" ^
    --test_dir "I:\wikiart_distinct5_samam_512_classview\test" ^
    --cache_dir "I:\Github\Latent_Style\eval_cache" ^
    --batch_size 16 ^
    --num_steps 8 ^
    --eval_only_lpips_clip_style >> "%LOGDIR%\validate.log" 2>&1

echo [%date% %time%] === Clean Base Validation END === >> "%LOGDIR%\validate.log"
