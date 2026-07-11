# Latent-WCT baseline: evaluate on D5 (images already generated)
$ErrorActionPreference = "Continue"
Set-Location "I:\Github\Latent_Style\SchrodingerBridge"
$env:PYTHONIOENCODING = "utf-8"

$logOut = "C:\Users\Administrator\logs\latent_wct_eval.out"
$evalDir = "I:\Github\Latent_Style\SchrodingerBridge\exp\latent_wct_baseline\full_eval\epoch_0000"
$testDir = "I:\datasets\wikiart_distinct5_samam_512_classview\test"
$cacheDir = "I:\Github\Latent_Style\SchrodingerBridge\exp\eval_cache"
$hfCache = "C:\Users\Administrator\.cache\huggingface\hub"
$imagesDir = "I:\Github\Latent_Style\SchrodingerBridge\exp\latent_wct_baseline\full_eval\epoch_0000\images"

# Step 2: CLIP-S + LPIPS
Write-Output "=== STEP 2: CLIP-S + LPIPS eval START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
if (Test-Path "$evalDir\summary.json") { Remove-Item "$evalDir\summary.json" -Force }
python -u src\utils\run_evaluation.py `
    --output $evalDir `
    --test_dir $testDir `
    --cache_dir $cacheDir `
    --clip_hf_cache_dir $hfCache `
    --eval_only_lpips_clip_style `
    --eval_lpips_chunk_size 4 `
    --reuse_generated `
    --batch_size 16 --metric_batch_size 16 `
    2>&1 | Tee-Object -FilePath $logOut -Append
Write-Output "=== STEP 2 DONE exit=$LASTEXITCODE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="

# Step 3: DINO eval
Write-Output "=== STEP 3: DINO eval START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
python _compute_dino.py `
    --images_dir $imagesDir `
    --test_dir $testDir `
    --dataset wikiart `
    --output "I:\Github\Latent_Style\SchrodingerBridge\exp\_dino_results\latent_wct.json" `
    --max_refs 30 `
    2>&1 | Tee-Object -FilePath $logOut -Append
Write-Output "=== STEP 3 DONE exit=$LASTEXITCODE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="

Write-Output "=== ALL DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
