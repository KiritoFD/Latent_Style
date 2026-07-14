# 712 Phase: 推理时 AdaIN 消融 — 用 stage4 checkpoint 测试不同 adain_scale
# adain_scale=1.0 (baseline, 已有结果), 0.5 (减弱), 0.0 (完全关闭)
# 目标: 测试网络自身是否学到了风格注入
$ErrorActionPreference = "Continue"
Set-Location "I:\Github\Latent_Style\SchrodingerBridge"
$env:PYTHONIOENCODING = "utf-8"

$logDir = "C:\Users\Administrator\logs\sty_inject"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }

$cacheDir = "I:\Github\Latent_Style\SchrodingerBridge\exp\eval_cache"
$hfCache = "C:\Users\Administrator\.cache\huggingface\hub"

$testDir = "I:\datasets\wikiart_distinct5_samam_512_classview\test"
$styles = "Early_Renaissance,Impressionism,Minimalism,Rococo,Ukiyo_e"

$ckpt = "exp\sty_inject\stage4_long_big\epoch_0010.pt"

$cfgs = @(
    @{name="stage4_adain05"; config="configs\eval_stage4_adain05.json"},
    @{name="stage4_adain0";  config="configs\eval_stage4_adain0.json"}
)

foreach ($cfg in $cfgs) {
    Write-Output ""
    Write-Output "############################################################"
    Write-Output "# EVAL $($cfg.name) START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') #"
    Write-Output "############################################################"

    $evalDir = "exp\sty_inject\$($cfg.name)_d5_eval"
    $evalLog = "$logDir\$($cfg.name)_d5_eval.out"

    $summaryPath = "$evalDir\full_eval\epoch_0010\summary.json"
    if (-not (Test-Path $summaryPath)) {
        $evalArgs = @(
            "-u", "src\utils\run_evaluation.py",
            "--checkpoint", $ckpt,
            "--config_override", $cfg.config,
            "--output", "$evalDir\full_eval\epoch_0010",
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
            "--images_dir", "$evalDir\full_eval\epoch_0010\images",
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

    Write-Output "### EVAL $($cfg.name) COMPLETE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ###"
}

Write-Output ""
Write-Output "============================================================"
Write-Output "ADAIN ABLATION COMPLETE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Output "============================================================"
