# Latent-WCT baseline: CLIP-S + LPIPS eval only (images renamed to _to_ format)
$ErrorActionPreference = "Continue"
Set-Location "I:\Github\Latent_Style\SchrodingerBridge"
$env:PYTHONIOENCODING = "utf-8"

$logOut = "C:\Users\Administrator\logs\latent_wct_eval2.out"
$evalDir = "I:\Github\Latent_Style\SchrodingerBridge\exp\latent_wct_baseline\full_eval\epoch_0000"
$testDir = "I:\datasets\wikiart_distinct5_samam_512_classview\test"
$cacheDir = "I:\Github\Latent_Style\SchrodingerBridge\exp\eval_cache"
$hfCache = "C:\Users\Administrator\.cache\huggingface\hub"

Write-Output "=== CLIP-S + LPIPS eval START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
if (Test-Path "$evalDir\summary.json") { Remove-Item "$evalDir\summary.json" -Force }
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
    2>&1 | Tee-Object -FilePath $logOut -Append
Write-Output "=== DONE exit=$LASTEXITCODE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
