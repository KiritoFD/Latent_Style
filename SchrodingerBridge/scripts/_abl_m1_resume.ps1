# Resume eval for abl_no_ll_fm only (other 3 already done).
# Launched via Start-Process to survive ssh disconnects.
$ErrorActionPreference = "Continue"
Set-Location "I:\Github\Latent_Style\SchrodingerBridge"
$logDir = "C:\Users\Administrator\logs"
$testDir = "I:\datasets\wikiart_distinct5_samam_512_classview\test"
$cacheDir = "I:\Github\Latent_Style\SchrodingerBridge\exp\eval_cache"
$hfCache = "C:\Users\Administrator\.cache\huggingface\hub"
$dinoOut = "exp\_dino_results"
$env:PYTHONPATH = "."

$exp = "abl_no_ll_fm"
$ckpt = "exp\$exp\epoch_0015.pt"
$evalDir = "exp\$exp\full_eval\epoch_0015"
$summary = Join-Path $evalDir "summary.json"

if (-not (Test-Path $summary)) {
    Write-Output "=== full_eval $exp $(Get-Date -Format 'HH:mm:ss') ==="
    python -u src\utils\run_evaluation.py `
        --checkpoint $ckpt `
        --output $evalDir `
        --test_dir $testDir `
        --cache_dir $cacheDir `
        --clip_hf_cache_dir $hfCache `
        --batch_size 2 --generation_batch_size 2 --metric_batch_size 2 `
        --target_chunk_size 1 --vae_decode_batch_size 16 `
        --eval_only_lpips_clip_style --eval_lpips_chunk_size 4
    Write-Output "full_eval exit=$LASTEXITCODE $(Get-Date -Format 'HH:mm:ss')"
} else {
    Write-Output "SKIP full_eval $exp : summary exists"
}

$imgDir = Join-Path $evalDir "images"
$dinoJson = Join-Path $dinoOut "$exp.json"
if ((Test-Path $imgDir) -and -not (Test-Path $dinoJson)) {
    Write-Output "=== DINO $exp $(Get-Date -Format 'HH:mm:ss') ==="
    python _compute_dino.py --images_dir $imgDir --test_dir $testDir --dataset wikiart --output $dinoJson --hf_cache $hfCache --max_refs 30
    Write-Output "DINO exit=$LASTEXITCODE $(Get-Date -Format 'HH:mm:ss')"
} else {
    Write-Output "SKIP DINO $exp : imgDir=$([bool](Test-Path $imgDir)) json=$([bool](Test-Path $dinoJson))"
}
Write-Output "=== RESUME COMPLETE $(Get-Date -Format 'HH:mm:ss') ==="
