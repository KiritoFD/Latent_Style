# Run full_eval for t11_repro epoch_0005 and t11e2 epoch_0015
# Uses fast WCT (GPU eigh) - spectral_bridge620.py already updated
$ErrorActionPreference = "Continue"
Set-Location "I:\Github\Latent_Style\SchrodingerBridge"
$env:PYTHONPATH = "."

$testDir = "I:\datasets\wikiart_distinct5_samam_512_classview\test"
$cacheDir = "I:\Github\Latent_Style\SchrodingerBridge\exp\eval_cache"
$hfCache = "C:\Users\Administrator\.cache\huggingface\hub"

# Eval 1: t11_repro epoch_0005
Write-Output "=== EVAL 1: t11_repro_15ep epoch_0005 START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
python -u src\utils\run_evaluation.py `
    --checkpoint exp\t11_repro_15ep\epoch_0005.pt `
    --output exp\t11_repro_15ep\full_eval\epoch_0005 `
    --test_dir $testDir `
    --cache_dir $cacheDir `
    --clip_hf_cache_dir $hfCache `
    --batch_size 2 --generation_batch_size 2 --metric_batch_size 2 `
    --target_chunk_size 1 --vae_decode_batch_size 16 `
    --eval_only_lpips_clip_style --eval_lpips_chunk_size 4 2>&1 | Tee-Object -FilePath "C:\Users\Administrator\logs\t11_repro_fulleval.out"
$ec1 = $LASTEXITCODE
Write-Output "=== EVAL 1 DONE exit=$ec1 $(Get-Date -Format 'HH:mm:ss') ==="

# Eval 2: t11e2 epoch_0015
Write-Output "=== EVAL 2: t11e2_extrap05_15ep epoch_0015 START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
python -u src\utils\run_evaluation.py `
    --checkpoint exp\t11e2_extrap05_15ep\epoch_0015.pt `
    --output exp\t11e2_extrap05_15ep\full_eval\epoch_0015 `
    --test_dir $testDir `
    --cache_dir $cacheDir `
    --clip_hf_cache_dir $hfCache `
    --batch_size 2 --generation_batch_size 2 --metric_batch_size 2 `
    --target_chunk_size 1 --vae_decode_batch_size 16 `
    --eval_only_lpips_clip_style --eval_lpips_chunk_size 4 2>&1 | Tee-Object -FilePath "C:\Users\Administrator\logs\t11e2_fulleval.out"
$ec2 = $LASTEXITCODE
Write-Output "=== EVAL 2 DONE exit=$ec2 $(Get-Date -Format 'HH:mm:ss') ==="

Write-Output "=== ALL EVALS COMPLETE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
