# 712 Phase Stage6: 强化训练时风格通路信号
# stage6_strong: gate=0.5, AdaLN std=0.1, FiLM std=0.1
# stage6_xstrong: gate=1.0, AdaLN std=0.2, FiLM std=0.2
$ErrorActionPreference = "Continue"
Set-Location "I:\Github\Latent_Style\SchrodingerBridge"
$env:PYTHONIOENCODING = "utf-8"

$logDir = "C:\Users\Administrator\logs\sty_inject"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }

$cacheDir = "I:\Github\Latent_Style\SchrodingerBridge\exp\eval_cache"
$hfCache = "C:\Users\Administrator\.cache\huggingface\hub"

$testDir = "I:\datasets\wikiart_distinct5_samam_512_classview\test"
$styles = "Early_Renaissance,Impressionism,Minimalism,Rococo,Ukiyo_e"

$cfgs = @(
    @{name="stage6_strong";   config="configs\exp_sty_stage6_strong.json";   ckpt="exp\sty_inject\stage6_strong\epoch_0005.pt"},
    @{name="stage6_xstrong";  config="configs\exp_sty_stage6_xstrong.json";  ckpt="exp\sty_inject\stage6_xstrong\epoch_0005.pt"}
)

foreach ($cfg in $cfgs) {
    Write-Output ""
    Write-Output "############################################################"
    Write-Output "# EXP $($cfg.name) START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') #"
    Write-Output "############################################################"

    # ===== PHASE 1: TRAINING =====
    $trainLog = "$logDir\$($cfg.name)_train.out"
    Write-Output "=== TRAIN $($cfg.name) START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
    & python -u src\run.py --config $cfg.config 2>&1 | Tee-Object -FilePath $trainLog
    Write-Output "=== TRAIN $($cfg.name) DONE exit=$LASTEXITCODE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="

    if (-not (Test-Path $cfg.ckpt)) {
        Write-Output "ERROR: checkpoint $($cfg.ckpt) not found after training. SKIP eval."
        continue
    }

    # ===== PHASE 2: EVALUATION (D5-512) =====
    $evalDir = "exp\sty_inject\$($cfg.name)_d5_eval"
    $evalLog = "$logDir\$($cfg.name)_d5_eval.out"

    Write-Output "=== EVAL $($cfg.name) on d5 START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="

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
    Write-Output "### EXP $($cfg.name) COMPLETE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ###"
}

Write-Output ""
Write-Output "============================================================"
Write-Output "ALL STAGE6 EXPERIMENTS COMPLETE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Output "============================================================"
