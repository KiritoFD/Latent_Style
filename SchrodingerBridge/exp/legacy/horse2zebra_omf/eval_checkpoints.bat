@echo off
setlocal
cd /d "%~dp0"

set "CLASSIFIER_CKPT=%~dp0eval_cache\horse2zebra_image_classifier.pt"
set "TEST_DIR=%~dp0..\datasets\horse2zebra\test_images"
set "OUT_DIR=%~dp0full_eval"

python ..\run_evaluation.py artifacts --test_dir "%TEST_DIR%" --output "%OUT_DIR%" --reuse_generated --image_classifier_path "%CLASSIFIER_CKPT%"

exit /b %ERRORLEVEL%
