# 712 Phase StyleInject Aggressive: train + eval on D5-512
# structure_aligned_target + HF非零init + 统计矩损失 + DWT route + HH head
$ErrorActionPreference = "Continue"
Set-Location "I:\Github\Latent_Style\SchrodingerBridge"
$env:PYTHONIOENCODING = "utf-8"

$logDir = "C:\Users\Administrator\logs\sty_inject"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }

$cacheDir = "I:\Github\Latent_Style\SchrodingerBridge\exp\eval_cache"
$hfCache = "C:\Users\Administrator\.cache\huggingface\hub"

$testDir = "I:\datasets\wikiart_distinct5_samam_512_classview\test"
$styles = "Early_Renaissance,Impressionism,Minimalism,Rococo,Ukiyo_e"

$cfg = @{name="sty_aggressive"; config="configs\exp_sty_aggressive.json"; ckpt="exp\sty_inject\aggressive\epoch_0005.pt"}

# ===== PHASE 1: TRAINING =====
$trainLog = "$logDir\$($cfg.name)_train.out"
Write-Output ""
Write-Output "=== TRAIN $($cfg.name) START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
& python -u src\run.py --config $cfg.config 2>&1 | Tee-Object -FilePath $trainLog
Write-Output "=== TRAIN $($cfg.name) DONE exit=$LASTEXITCODE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="

if (-not (Test-Path $cfg.ckpt)) {
    Write-Output "ERROR: checkpoint $($cfg.ckpt) not found after training."
    exit 1
}

# ===== PHASE 2: EVALUATION (D5-512) =====
$evalDir = "exp\sty_inject\$($cfg.name)_d5_eval"
$evalLog = "$logDir\$($cfg.name)_d5_eval.out"

Write-Output ""
Write-Output "=== EVAL $($cfg.name) on d5 START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="

# Step 1: inference + CLIP/LPIPS
$summaryPath = "$evalDir\full_eval\epoch_0005\summary.json"
if (-not (Test-Path $summaryPath)) {
    $evalArgs = @(
        "-u", "src\utils\run_evaluation.py",
        "--checkpoint", $cfg.ckpt,
        "--output", "$evalDir\full_eval\epoch_0005",
        "--test_dir", $testDir,
        "--style_subdirs", $styles,
        "--cache_dir", $cacheDir,
        "--clip_hf_cache_dir", $hfCache,
        "--batch_size", "2", "--generation_batch_size", "2", "--metric_batch_size", "2",
        "--target_chunk_size", "1", "--vae_decode_batch_size", "16",
        "--eval_only_lpips_clip_style", "--eval_lpips_chunk_size", "4"
    )
    & python @evalArgs 2>&1 | Tee-Object -FilePath $evalLog
} else {
    Write-Output "  summary.json exists, skipping inference."
}

# Step 2: DINO eval
$dinoOut = "exp\sty_inject\_dino\$($cfg.name)_d5.json"
if (-not (Test-Path $dinoOut)) {
    $dinoArgs = @(
        "_compute_dino.py",
        "--images_dir", "$evalDir\full_eval\epoch_0005\images",
        "--test_dir", $testDir,
        "--dataset", "wikiart",
        "--output", $dinoOut,
        "--hf_cache", $hfCache,
        "--max_refs", "30"
    )
    & python @dinoArgs 2>&1 | Tee-Object -FilePath $evalLog -Append
} else {
    Write-Output "  DINO result exists, skipping."
}

Write-Output "=== EVAL $($cfg.name) on d5 DONE exit=$LASTEXITCODE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
Write-Output ""
Write-Output "=== AGGRESSIVE EXPERIMENT COMPLETE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
