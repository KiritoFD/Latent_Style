# Evaluate d5_512 Latent-WCT with correct test_dir (no fallback)
$ErrorActionPreference = "Continue"
Set-Location "I:\Github\Latent_Style\SchrodingerBridge"
$env:PYTHONIOENCODING = "utf-8"

$evalDir = "I:\Github\Latent_Style\SchrodingerBridge\exp\latent_wct_baseline\d5_512"
$testDir = "I:\datasets\wikiart_distinct5_512_images\test"
$cacheDir = "I:\Github\Latent_Style\SchrodingerBridge\exp\eval_cache"
$hfCache = "C:\Users\Administrator\.cache\huggingface\hub"
$evalLog = "C:\Users\Administrator\logs\latent_wct_d5_512_eval3.log"

# Clean stale summary
if (Test-Path "$evalDir\summary.json") { Remove-Item "$evalDir\summary.json" -Force }
if (Test-Path "$evalDir\images\summary.json") { Remove-Item "$evalDir\images\summary.json" -Force }

$imgCount = (Get-ChildItem "$evalDir\images\*_to_*.png" -ErrorAction SilentlyContinue).Count
Write-Output "Found $imgCount reusable images in $evalDir\images\"

python -u src\utils\run_evaluation.py `
    --output $evalDir `
    --test_dir $testDir `
    --cache_dir $cacheDir `
    --clip_hf_cache_dir $hfCache `
    --eval_only_lpips_clip_style `
    --eval_lpips_chunk_size 4 `
    --reuse_generated `
    --style_subdirs "Early_Renaissance,Impressionism,Minimalism,Rococo,Ukiyo_e" `
    --batch_size 16 --metric_batch_size 16 `
    2>&1 | Tee-Object -FilePath $evalLog

Write-Output "=== D5_512 EVAL DONE exit=$LASTEXITCODE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
