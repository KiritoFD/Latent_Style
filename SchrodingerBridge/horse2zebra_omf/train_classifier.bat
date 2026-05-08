@echo off
setlocal
cd /d "%~dp0"

python ..\src\utils\classify.py ^
  --config config.json ^
  --train_root "..\datasets\horse2zebra\train_images" ^
  --val_root "..\datasets\horse2zebra\test_images" ^
  --epochs 40 ^
  --batch_size 48 ^
  --num_workers 2 ^
  --min_epochs 8 ^
  --patience 10 ^
  --target_acc 0.98 ^
  --target_recall 0.98 ^
  --target_confidence_min 0.80 ^
  --target_confidence_max 0.995 ^
  --out_ckpt ".\eval_cache\horse2zebra_image_classifier.pt" ^
  --out_report ".\eval_cache\horse2zebra_image_classifier_report.json"

exit /b %ERRORLEVEL%
