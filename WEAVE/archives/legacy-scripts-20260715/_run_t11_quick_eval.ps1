$ErrorActionPreference = "Continue"
Set-Location "I:\Github\Latent_Style\SchrodingerBridge"
$env:PYTHONPATH = "."
$env:PYTHONIOENCODING = "utf-8"

$testDir = "I:\datasets\wikiart_distinct5_samam_512_classview\test"
$cacheDir = "I:\Github\Latent_Style\SchrodingerBridge\exp\eval_cache"
$hfCache = "C:\Users\Administrator\.cache\huggingface\hub"

# --- t11_repro ep5 quick eval (max_src_samples=5 for fast LPIPS verification) ---
$exp = "t11_repro_15ep"
$ckpt = "exp\$exp\epoch_0005.pt"
$evalDir = "exp\$exp\quick_eval\epoch_0005"
$logOut = "C:\Users\Administrator\logs\t11_repro_quick_eval.out"

Write-Output "=== T11_REPRO QUICK EVAL START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="

# Remove old quick_eval if exists
if (Test-Path $evalDir) { Remove-Item $evalDir -Recurse -Force }

python -u src\utils\run_evaluation.py `
    --checkpoint $ckpt `
    --output $evalDir `
    --test_dir $testDir `
    --cache_dir $cacheDir `
    --clip_hf_cache_dir $hfCache `
    --batch_size 2 --generation_batch_size 2 --metric_batch_size 2 `
    --target_chunk_size 1 --vae_decode_batch_size 16 `
    --max_src_samples 5 `
    --eval_only_lpips_clip_style --eval_lpips_chunk_size 4 2>&1 | Tee-Object -FilePath $logOut

$evalEc = $LASTEXITCODE
Write-Output "=== T11_REPRO QUICK EVAL DONE exit=$evalEc $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="

# Print summary if exists
$summary = Join-Path $evalDir "summary.json"
if (Test-Path $summary) {
    Write-Output "=== SUMMARY ==="
    Get-Content $summary
}
Write-Output "=== ALL DONE ==="
