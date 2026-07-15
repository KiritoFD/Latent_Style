# 712 Phase StyleInject Stage 5: 3 mechanism variants with multi-point eval
# 变体A: stage5_p08 (DWT p=0.8, 历史最优)
# 变体B: stage5_no_sat (无SAT, 测试SAT贡献)
# 变体C: stage5_whh4 (spectral_w_hh=4.0, 强化精细笔触)
# 每个变体: save_interval=2 -> epoch 2,4,6,8,10 共5个checkpoint, 多eval点
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
    @{name="stage5_p08";    config="configs\exp_sty_stage5_p08.json";    ckptDir="exp\sty_inject\stage5_p08"},
    @{name="stage5_no_sat"; config="configs\exp_sty_stage5_no_sat.json"; ckptDir="exp\sty_inject\stage5_no_sat"},
    @{name="stage5_whh4";   config="configs\exp_sty_stage5_whh4.json";   ckptDir="exp\sty_inject\stage5_whh4"}
)

# Eval epoch points (save_interval=2 -> epoch 2,4,6,8,10)
$evalEpochs = @(2, 4, 6, 8, 10)

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

    # ===== PHASE 2: MULTI-POINT EVALUATION =====
    foreach ($epoch in $evalEpochs) {
        $ckpt = "$($cfg.ckptDir)\epoch_$($epoch.ToString('0000')).pt"
        if (-not (Test-Path $ckpt)) {
            Write-Output "  SKIP epoch=${epoch} - checkpoint not found ($ckpt)"
            continue
        }

        $evalDir = "exp\sty_inject\$($cfg.name)_d5_eval_ep$($epoch.ToString('0000'))"
        $evalLog = "$logDir\$($cfg.name)_d5_eval_ep$($epoch.ToString('0000')).out"

        Write-Output "--- EVAL $($cfg.name) epoch=${epoch} START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ---"

        $summaryPath = "$evalDir\full_eval\epoch_$($epoch.ToString('0000'))\summary.json"
        if (-not (Test-Path $summaryPath)) {
            $evalArgs = @(
                "-u", "src\utils\run_evaluation.py",
                "--checkpoint", $ckpt,
                "--output", "$evalDir\full_eval\epoch_$($epoch.ToString('0000'))",
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

        $dinoOut = "exp\sty_inject\_dino\$($cfg.name)_d5_ep$($epoch.ToString('0000')).json"
        if (-not (Test-Path $dinoOut)) {
            $dinoArgs = @(
                "_compute_dino.py",
                "--images_dir", "$evalDir\full_eval\epoch_$($epoch.ToString('0000'))\images",
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

        Write-Output "--- EVAL $($cfg.name) epoch=${epoch} DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ---"
    }

    Write-Output "### EXP $($cfg.name) COMPLETE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ###"
}

Write-Output ""
Write-Output "============================================================"
Write-Output "ALL 3 STAGE5 VARIANTS COMPLETE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Output "============================================================"
