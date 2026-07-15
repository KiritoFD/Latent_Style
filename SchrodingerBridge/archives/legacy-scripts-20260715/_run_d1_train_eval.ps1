# D1 Gram matrix: train + full_eval + DINO, all in sequence.
# Run via: ssh ... "powershell -File _run_d1_train_eval.ps1"
$ErrorActionPreference = "Continue"
Set-Location "I:\Github\Latent_Style\SchrodingerBridge"
$env:PYTHONPATH = "."

$exp = "d1_gram_hf1_15ep"
$config = "configs\$exp.json"
$ckptDir = "exp\$exp"
$ckpt = "exp\$exp\epoch_0015.pt"
$evalDir = "exp\$exp\full_eval\epoch_0015"
$testDir = "I:\datasets\wikiart_distinct5_samam_512_classview\test"
$cacheDir = "I:\Github\Latent_Style\SchrodingerBridge\exp\eval_cache"
$hfCache = "C:\Users\Administrator\.cache\huggingface\hub"
$dinoOut = "exp\_dino_results\$exp.json"
$logOut = "C:\Users\Administrator\logs\d1_train_eval.out"
$logErr = "C:\Users\Administrator\logs\d1_train_eval.err"

# ===== Phase 1: Training =====
Write-Output "=== TRAIN START $(Get-Date -Format 'HH:mm:ss') ==="
python -u src\run.py --config $config 2>&1 | Tee-Object -FilePath $logOut
$trainEc = $LASTEXITCODE
Write-Output "=== TRAIN DONE exit=$trainEc $(Get-Date -Format 'HH:mm:ss') ==="
if ($trainEc -ne 0 -or -not (Test-Path $ckpt)) {
    Write-Output "FATAL: training failed or no checkpoint. Aborting."
    exit 1
}

# ===== Phase 2: full_eval =====
Write-Output "=== EVAL START $(Get-Date -Format 'HH:mm:ss') ==="
python -u src\utils\run_evaluation.py `
    --checkpoint $ckpt `
    --output $evalDir `
    --test_dir $testDir `
    --cache_dir $cacheDir `
    --clip_hf_cache_dir $hfCache `
    --batch_size 2 --generation_batch_size 2 --metric_batch_size 2 `
    --target_chunk_size 1 --vae_decode_batch_size 16 `
    --eval_only_lpips_clip_style --eval_lpips_chunk_size 4 2>&1 | Tee-Object -FilePath $logOut -Append
$evalEc = $LASTEXITCODE
Write-Output "=== EVAL DONE exit=$evalEc $(Get-Date -Format 'HH:mm:ss') ==="

# ===== Phase 3: DINO =====
if ($evalEc -eq 0) {
    Write-Output "=== DINO START $(Get-Date -Format 'HH:mm:ss') ==="
    $imgDir = Join-Path $evalDir "images"
    python _compute_dino.py `
        --images_dir $imgDir `
        --test_dir $testDir `
        --dataset wikiart `
        --output $dinoOut `
        --hf_cache $hfCache `
        --max_refs 30 2>&1 | Tee-Object -FilePath $logOut -Append
    Write-Output "=== DINO DONE exit=$LASTEXITCODE $(Get-Date -Format 'HH:mm:ss') ==="
}

Write-Output "=== ALL COMPLETE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
