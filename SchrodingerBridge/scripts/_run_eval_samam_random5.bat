@echo off
REM Run SaMam Random5 evaluation on Windows
set HF_HOME=C:\Users\Administrator\.cache\huggingface
set TRANSFORMERS_OFFLINE=1
set TORCH_HOME=C:\Users\Administrator\.cache\torch
set CUDA_VISIBLE_DEVICES=0

echo === START SaMam Random5 eval ===
echo %DATE% %TIME%

python C:\Users\Administrator\_eval_unified.py ^
    --image-dir "I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_wikiarts20\samam\images" ^
    --dataset wiki20distinct5 ^
    --output "I:\Github\Latent_Style\SchrodingerBridge\exp\_eval_samam_random5_w20.json" ^
    --max-images 750

echo === END ===
echo %DATE% %TIME%
type "I:\Github\Latent_Style\SchrodingerBridge\exp\_eval_samam_random5_w20.json"
