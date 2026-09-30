@echo off
echo === WCT alpha=0.6 Evaluation ===
set PYTHON=C:\Program Files\Python312\python.exe
set EVAL_SCRIPT=I:\GitHub\Latent_Style\SchrodingerBridge\src\utils\run_evaluation.py
set EVAL_DIR=I:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_v2\eval\wct_a06
set TEST_DIR=I:\wikiart_distinct5_samam_512_classview\test
set IMAGES_SRC=I:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_v2\images\wct_v32k_a0.60

echo Creating eval directory...
if not exist "%EVAL_DIR%\images" mkdir "%EVAL_DIR%\images"

echo Copying WCT images to eval dir...
xcopy /Y /Q "%IMAGES_SRC%\*.png" "%EVAL_DIR%\images\"

echo Running evaluation...
"%PYTHON%" "%EVAL_SCRIPT%" "%EVAL_DIR%" --reuse_generated --save_generated_images --style_subdirs Early_Renaissance,Impressionism,Minimalism,Rococo,Ukiyo_e --test_dir "%TEST_DIR%" --eval_only_lpips_clip_style --clip_style_idt_baseline 0.6399

echo ==WCT_A06_EVAL_DONE==
