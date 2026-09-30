# M1 Ablation eval batch: full_eval + DINO for all 4 experiments
# Run AFTER all training is complete.
$ErrorActionPreference = "Continue"
Set-Location "I:\Github\Latent_Style\SchrodingerBridge"
$logDir = "C:\Users\Administrator\logs"
$testDir = "I:\datasets\wikiart_distinct5_samam_512_classview\test"
$cacheDir = "I:\Github\Latent_Style\SchrodingerBridge\exp\eval_cache"
$hfCache = "C:\Users\Administrator\.cache\huggingface\hub"
$dinoOut = "exp\_dino_results"
if (-not (Test-Path $dinoOut)) { New-Item -ItemType Directory -Force -Path $dinoOut | Out-Null }
$env:PYTHONPATH = "."

$exps = @("abl_no_edge", "abl_no_terminal", "abl_no_ep_content", "abl_no_ll_fm")
foreach ($exp in $exps) {
    $ckpt = "exp\$exp\epoch_0015.pt"
    if (-not (Test-Path $ckpt)) {
        Write-Host "SKIP $exp : no epoch_0015.pt"
        continue
    }
    $evalDir = "exp\$exp\full_eval\epoch_0015"
    $summary = Join-Path $evalDir "summary.json"
    if (Test-Path $summary) {
        Write-Host "SKIP $exp : full_eval already done"
    } else {
        Write-Host "=== full_eval $exp $(Get-Date -Format 'HH:mm:ss') ==="
        python -u src\utils\run_evaluation.py `
            --checkpoint $ckpt `
            --output $evalDir `
            --test_dir $testDir `
            --cache_dir $cacheDir `
            --clip_hf_cache_dir $hfCache `
            --batch_size 2 --generation_batch_size 2 --metric_batch_size 2 `
            --target_chunk_size 1 --vae_decode_batch_size 16 `
            --eval_only_lpips_clip_style --eval_lpips_chunk_size 4 2>&1 | Tee-Object -FilePath (Join-Path $logDir "abl_${exp}_eval.log")
        if ($LASTEXITCODE -ne 0) {
            Write-Host "FAILED $exp eval (exit $LASTEXITCODE), continuing..."
            continue
        }
    }
    # DINO evaluation
    $imgDir = Join-Path $evalDir "images"
    $dinoJson = Join-Path $dinoOut "$exp.json"
    if (-not (Test-Path $imgDir)) {
        Write-Host "SKIP $exp DINO : no images dir"
        continue
    }
    if (Test-Path $dinoJson) {
        Write-Host "SKIP $exp DINO : already done"
    } else {
        Write-Host "=== DINO $exp $(Get-Date -Format 'HH:mm:ss') ==="
        python _compute_dino.py --images_dir $imgDir --test_dir $testDir --dataset wikiart --output $dinoJson --hf_cache $hfCache --max_refs 30 2>&1 | Tee-Object -FilePath (Join-Path $logDir "abl_${exp}_dino.log")
    }
    Write-Host "=== DONE $exp $(Get-Date -Format 'HH:mm:ss') ==="
}
Write-Host "=== M1 EVAL BATCH COMPLETE $(Get-Date -Format 'HH:mm:ss') ==="
