$ErrorActionPreference = "Continue"
Set-Location "G:\GitHub\Latent_Style\SchrodingerBridge"
$env:PYTHONPATH = "."
$env:PYTHONIOENCODING = "utf-8"

$ckpt = "exp\630_random20_heun_5ep\epoch_0005.pt"
$outDir = "exp\630_random20_heun_5ep\repro_d5_local\epoch_0005"
$testDir = "G:\GitHub\Latent_Style\Dataset\distinct5_512\test"
$cacheDir = "exp\eval_cache"
$hfCache = "$env:USERPROFILE\.cache\huggingface\hub"

Write-Output "=== LOCAL REPRO EVAL START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="

if (Test-Path $outDir) { Remove-Item $outDir -Recurse -Force }

python -u src\utils\run_evaluation.py `
    --checkpoint $ckpt `
    --output $outDir `
    --test_dir $testDir `
    --cache_dir $cacheDir `
    --clip_hf_cache_dir $hfCache `
    --batch_size 2 --generation_batch_size 2 --metric_batch_size 2 `
    --target_chunk_size 1 --vae_decode_batch_size 16 `
    --eval_only_lpips_clip_style --eval_lpips_chunk_size 4 2>&1 | Tee-Object -FilePath "G:\GitHub\Latent_Style\SchrodingerBridge\repro_eval_local.log"

$evalEc = $LASTEXITCODE
Write-Output "=== LOCAL REPRO EVAL DONE exit=$evalEc $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="

$summary = Join-Path $outDir "summary.json"
if (Test-Path $summary) {
    Write-Output "=== SUMMARY ==="
    Get-Content $summary
}
Write-Output "=== ALL DONE ==="
