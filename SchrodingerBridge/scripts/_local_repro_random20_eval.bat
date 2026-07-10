@echo off
REM Local reproduction of 630_random20_heun_5ep on D5 test (clip=0.7434, lpips=0.2910)
REM Local GPU: RTX 4070 Laptop 8GB, batch_size=2 to keep VRAM < 7GB
setlocal
cd /d G:\GitHub\Latent_Style\SchrodingerBridge
set PYTHONPATH=.
set PYTHONIOENCODING=utf-8

set CKPT=exp\630_random20_heun_5ep\epoch_0005.pt
set OUTDIR=exp\630_random20_heun_5ep\repro_d5_local\epoch_0005
set TESTDIR=G:\GitHub\Latent_Style\Dataset\distinct5_512\test
set CACHEDIR=exp\eval_cache
set HFCACHE=%USERPROFILE%\.cache\huggingface\hub

echo === LOCAL REPRO EVAL START %date% %time% ===

if exist %OUTDIR% rmdir /s /q %OUTDIR%

python -u src\utils\run_evaluation.py ^
    --checkpoint %CKPT% ^
    --output %OUTDIR% ^
    --test_dir %TESTDIR% ^
    --cache_dir %CACHEDIR% ^
    --clip_hf_cache_dir %HFCACHE% ^
    --batch_size 2 --generation_batch_size 2 --metric_batch_size 2 ^
    --target_chunk_size 1 --vae_decode_batch_size 16 ^
    --eval_only_lpips_clip_style --eval_lpips_chunk_size 4

echo === LOCAL REPRO EVAL DONE exit=%errorlevel% %date% %time% ===
if exist %OUTDIR%\summary.json (
    echo === SUMMARY ===
    type %OUTDIR%\summary.json
)
echo === ALL DONE ===
