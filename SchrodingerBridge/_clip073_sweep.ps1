# 精细单点实验：LPIPS<0.35 约束下推 CLIP>0.73
# 从 T11 baseline (8-step: clip=0.7213, lpips=0.2868) 出发
# 策略：组合 lpips 降低手段（4-step, lock_ll）+ zero_step_wct 推 CLIP

$ErrorActionPreference = "Continue"

$ckpt = "I:/Github/Latent_Style/SchrodingerBridge/exp/630_local_t11_stochastic_dwt_p08/epoch_0005.pt"
$testDir = "I:/wikiart_distinct5_samam_512_classview/test"
$cacheDir = "I:/Github/Latent_Style/SchrodingerBridge/exp/eval_cache"
$hfCacheDir = "I:/Github/Latent_Style/SchrodingerBridge/exp/eval_cache/hf"
$resultDir = "I:/Github/Latent_Style/SchrodingerBridge/exp/clip073_results"
$expRoot = "I:/Github/Latent_Style/SchrodingerBridge/exp/clip073_sweep"
$configsDir = "I:/Github/Latent_Style/SchrodingerBridge/configs/_clip073_overrides"

# 单点实验配置 — 每个都是从 T11 baseline 出发的单点/双点调整
# 格式: @{name=...; steps=...; alpha=...; lock_ll=...; adain=...}
$experiments = @(
    # 组1: 8-step + 低 alpha zero_step_wct (确认 lpips<0.35 边界)
    @{name="zsw_a03_8s";    steps=8; alpha=0.3; lock_ll=$false; adain=0.5},
    @{name="zsw_a04_8s";    steps=8; alpha=0.4; lock_ll=$false; adain=0.5},
    # 组2: 4-step + 中 alpha zero_step_wct (4-step 降 lpips 给预算)
    @{name="zsw_a04_4s";    steps=4; alpha=0.4; lock_ll=$false; adain=0.5},
    @{name="zsw_a05_4s";    steps=4; alpha=0.5; lock_ll=$false; adain=0.5},
    @{name="zsw_a06_4s";    steps=4; alpha=0.6; lock_ll=$false; adain=0.5},
    # 组3: 8-step + lock_ll + zero_step_wct (lock_ll 降 lpips 给预算)
    @{name="lock_zsw_a05_8s"; steps=8; alpha=0.5; lock_ll=$true;  adain=0.5},
    @{name="lock_zsw_a07_8s"; steps=8; alpha=0.7; lock_ll=$true;  adain=0.5},
    # 组4: 8-step + 高 adain + 低 alpha (adain 推 clip, alpha 补充)
    @{name="adain08_zsw_a03"; steps=8; alpha=0.3; lock_ll=$false; adain=0.8}
)

$results = @()

Write-Host "========== CLIP>0.73 LPIPS<0.35 SINGLE-POINT SWEEP =========="
Write-Host "Baseline: T11 8-step clip=0.7213 lpips=0.2868"
Write-Host "Target: clip>0.73 AND lpips<0.35"
Write-Host ""

if (-not (Test-Path $resultDir)) { New-Item -ItemType Directory -Path $resultDir -Force | Out-Null }
if (-not (Test-Path $expRoot)) { New-Item -ItemType Directory -Path $expRoot -Force | Out-Null }
if (-not (Test-Path $configsDir)) { New-Item -ItemType Directory -Path $configsDir -Force | Out-Null }

if (-not (Test-Path $ckpt)) { Write-Host "ERROR: Checkpoint not found: $ckpt"; exit 1 }
Write-Host "Checkpoint verified."
Write-Host ""

foreach ($exp in $experiments) {
    $name = $exp.name
    $steps = $exp.steps
    $alpha = $exp.alpha
    $lockLl = $exp.lock_ll
    $adain = $exp.adain
    $outDir = "$expRoot/$name"

    Write-Host "========== EXPERIMENT: $name (steps=$steps, alpha=$alpha, lock_ll=$lockLl, adain=$adain) =========="

    if (Test-Path $outDir) { Remove-Item $outDir -Recurse -Force }

    $modelSection = @{
        zero_step_wct_enabled = $true
        zero_step_wct_alpha = $alpha
        endpoint_adain_scale = $adain
        endpoint_adain_scale_lh = 0.5
        endpoint_adain_scale_hl = 0.5
        endpoint_adain_scale_hh = $adain
        endpoint_adain_scale_ll = 0.0
    }
    if ($lockLl) {
        $modelSection["endpoint_lock_ll"] = $true
    }

    $config = @{
        "_base" = "../630_remote_t11_long30ep.json"
        "model" = $modelSection
        "checkpoint" = @{
            save_dir = $outDir
            resume_checkpoint = $ckpt
        }
        "training" = @{
            num_epochs = 1
            patience = 1
            full_eval_each_epoch = $true
            resume_checkpoint = $ckpt
            test_image_dir = $testDir
            full_eval_cache_dir = $cacheDir
            full_eval_clip_hf_cache_dir = $hfCacheDir
            full_eval_num_steps = $steps
            full_eval_batch_size = 2
            full_eval_ref_feature_batch_size = 2
            full_eval_save_generated_images = $true
            save_dir = $outDir
        }
        "ablation" = @{
            name = $name
            notes = "steps=$steps alpha=$alpha lock_ll=$lockLl adain=$adain"
        }
    }

    $configPath = "$configsDir/${name}.json"
    $jsonStr = $config | ConvertTo-Json -Depth 10
    [System.IO.File]::WriteAllText($configPath, $jsonStr, [System.Text.UTF8Encoding]::new($false))

    Write-Host "Running evaluation..."
    $logFile = "I:/Github/Latent_Style/SchrodingerBridge/${name}_eval.log"
    if (Test-Path $logFile) { Remove-Item $logFile -Force }

    & python I:/Github/Latent_Style/SchrodingerBridge/src/run.py --config $configPath `
        > $logFile 2>&1

    if ($LASTEXITCODE -ne 0) {
        Write-Host "WARNING: Eval exited with code $LASTEXITCODE. Checking for results anyway..."
    }

    $summaryPath = Get-ChildItem $outDir -Recurse -Filter "summary.json" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($summaryPath) {
        $summary = Get-Content $summaryPath.FullName -Raw | ConvertFrom-Json
        $clip = $summary.analysis.all_pairs_overview.clip_style
        $lpips = $summary.analysis.all_pairs_overview.content_lpips
        Write-Host "RESULT ${name} : clip=$clip lpips=$lpips"

        $uniqueSummary = "$resultDir/summary_${name}.json"
        Copy-Item $summaryPath.FullName $uniqueSummary -Force
        $results += @{ name=$name; steps=$steps; alpha=$alpha; lock_ll=$lockLl; adain=$adain; clip=$clip; lpips=$lpips }
    } else {
        Write-Host "ERROR: No summary.json found"
        $results += @{ name=$name; steps=$steps; alpha=$alpha; lock_ll=$lockLl; adain=$adain; clip="N/A"; lpips="N/A" }
    }
    Write-Host ""
}

Write-Host ""
Write-Host "========== SINGLE-POINT SWEEP SUMMARY =========="
Write-Host "Target: clip>0.73 AND lpips<0.35"
Write-Host "{0,-22} {1,6} {2,7} {3,8} {4,7} {5,12} {6,12} {7,10}" -f "Experiment", "Steps", "Alpha", "LockLL", "Adain", "CLIP-S", "LPIPS", "Status"
foreach ($r in $results) {
    $status = "?"
    if ($r.clip -ne "N/A" -and $r.lpips -ne "N/A") {
        $cv = [double]$r.clip
        $lv = [double]$r.lpips
        if ($cv -gt 0.73 -and $lv -lt 0.35) { $status = "DOUBLE WIN!" }
        elseif ($cv -gt 0.73) { $status = "clip OK" }
        elseif ($lv -lt 0.35) { $status = "lpips OK" }
        else { $status = "neither" }
    }
    Write-Host "{0,-22} {1,6} {2,7} {3,8} {4,7} {5,12} {6,12} {7,10}" -f $r.name, $r.steps, $r.alpha, $r.lock_ll, $r.adain, $r.clip, $r.lpips, $status
}

$csvPath = "$resultDir/clip073_summary.csv"
$results | Export-Csv $csvPath -NoTypeInformation
Write-Host ""
Write-Host "Saved CSV: $csvPath"
