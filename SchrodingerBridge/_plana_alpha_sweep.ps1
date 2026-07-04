# Plan A Zero-Step WCT Alpha Sweep — 推 CLIP-S > 0.73
# 已知: alpha=0.7 -> clip=0.7329, alpha=1.0 -> clip=0.7333
# 目标: 找到 CLIP 最高的 alpha 值，尝试更高 alpha + HF WCT

$ErrorActionPreference = "Continue"

# T11 SOTA checkpoint
$ckpt = "I:/Github/Latent_Style/SchrodingerBridge/exp/630_local_t11_stochastic_dwt_p08/epoch_0005.pt"
$testDir = "I:/wikiart_distinct5_samam_512_classview/test"
$cacheDir = "I:/Github/Latent_Style/SchrodingerBridge/exp/eval_cache"
$hfCacheDir = "I:/Github/Latent_Style/SchrodingerBridge/exp/eval_cache/hf"
$resultDir = "I:/Github/Latent_Style/SchrodingerBridge/exp/plana_sweep_results"
$expRoot = "I:/Github/Latent_Style/SchrodingerBridge/exp/plana_sweep"
$configsDir = "I:/Github/Latent_Style/SchrodingerBridge/configs/_plana_overrides"

# Alpha values to sweep (including HF WCT variants)
$experiments = @(
    @{name="plana_a07";    alpha=0.7;  hf_wct=$false},
    @{name="plana_a10";    alpha=1.0;  hf_wct=$false},
    @{name="plana_a13";    alpha=1.3;  hf_wct=$false},
    @{name="plana_a15";    alpha=1.5;  hf_wct=$false},
    @{name="plana_a20";    alpha=2.0;  hf_wct=$false},
    @{name="plana_a10_hf"; alpha=1.0;  hf_wct=$true},
    @{name="plana_a07_hf"; alpha=0.7;  hf_wct=$true}
)

$results = @()

Write-Host "========== PLAN A ZERO-STEP WCT SWEEP =========="
Write-Host "Checkpoint: $ckpt"
Write-Host ""

if (-not (Test-Path $resultDir)) { New-Item -ItemType Directory -Path $resultDir -Force | Out-Null }
if (-not (Test-Path $expRoot)) { New-Item -ItemType Directory -Path $expRoot -Force | Out-Null }
if (-not (Test-Path $configsDir)) { New-Item -ItemType Directory -Path $configsDir -Force | Out-Null }

# Verify checkpoint
if (-not (Test-Path $ckpt)) {
    Write-Host "ERROR: Checkpoint not found: $ckpt"
    exit 1
}
Write-Host "Checkpoint verified."
Write-Host ""

foreach ($exp in $experiments) {
    $name = $exp.name
    $alpha = $exp.alpha
    $hfWct = $exp.hf_wct
    $outDir = "$expRoot/$name"

    Write-Host "========== EXPERIMENT: $name (alpha=$alpha, hf_wct=$hfWct) =========="

    # Clean output directory
    if (Test-Path $outDir) { Remove-Item $outDir -Recurse -Force }

    # Build config JSON with _base inheritance
    $modelSection = @{
        zero_step_wct_enabled = $true
        zero_step_wct_alpha = $alpha
    }
    if ($hfWct) {
        $modelSection["zero_step_wct_hf_enabled"] = $true
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
            full_eval_num_steps = 8
            full_eval_batch_size = 2
            full_eval_ref_feature_batch_size = 2
            full_eval_save_generated_images = $true
            save_dir = $outDir
        }
        "ablation" = @{
            name = $name
            notes = "Plan A zero-step WCT alpha=$alpha hf_wct=$hfWct"
        }
    }

    $configPath = "$configsDir/${name}.json"
    $jsonStr = $config | ConvertTo-Json -Depth 10
    [System.IO.File]::WriteAllText($configPath, $jsonStr, [System.Text.UTF8Encoding]::new($false))

    # Run evaluation via run.py
    Write-Host "Running evaluation..."
    $logFile = "I:/Github/Latent_Style/SchrodingerBridge/${name}_eval.log"
    if (Test-Path $logFile) { Remove-Item $logFile -Force }

    & python I:/Github/Latent_Style/SchrodingerBridge/src/run.py --config $configPath `
        > $logFile 2>&1

    if ($LASTEXITCODE -ne 0) {
        Write-Host "WARNING: Eval exited with code $LASTEXITCODE. Checking for results anyway..."
    }

    # Find summary.json
    $summaryPath = Get-ChildItem $outDir -Recurse -Filter "summary.json" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($summaryPath) {
        $summary = Get-Content $summaryPath.FullName -Raw | ConvertFrom-Json
        $clip = $summary.analysis.all_pairs_overview.clip_style
        $lpips = $summary.analysis.all_pairs_overview.content_lpips
        Write-Host "RESULT ${name} : clip=$clip lpips=$lpips"

        $uniqueSummary = "$resultDir/summary_${name}.json"
        Copy-Item $summaryPath.FullName $uniqueSummary -Force
        $results += @{ name=$name; alpha=$alpha; hf_wct=$hfWct; clip=$clip; lpips=$lpips }
    } else {
        Write-Host "ERROR: No summary.json found"
        # Try to find clip/lpips in log
        $logContent = Get-Content $logFile -Raw -ErrorAction SilentlyContinue
        if ($logContent -match "all_pairs.*?clip_style.*?([0-9.]+)") {
            $clip = $matches[1]
            Write-Host "  Found clip in log: $clip"
        } else { $clip = "N/A" }
        if ($logContent -match "content_lpips.*?([0-9.]+)") {
            $lpips = $matches[1]
            Write-Host "  Found lpips in log: $lpips"
        } else { $lpips = "N/A" }
        $results += @{ name=$name; alpha=$alpha; hf_wct=$hfWct; clip=$clip; lpips=$lpips }
    }
    Write-Host ""
}

# Print summary
Write-Host ""
Write-Host "========== PLAN A SWEEP SUMMARY =========="
Write-Host "{0,-20} {1,8} {2,8} {3,12} {4,12}" -f "Experiment", "Alpha", "HF_WCT", "CLIP-S", "LPIPS"
foreach ($r in $results) {
    Write-Host "{0,-20} {1,8} {2,8} {3,12} {4,12}" -f $r.name, $r.alpha, $r.hf_wct, $r.clip, $r.lpips
}

# CSV
$csvPath = "$resultDir/plana_summary.csv"
$results | Export-Csv $csvPath -NoTypeInformation
Write-Host ""
Write-Host "Saved CSV: $csvPath"

# CLIP > 0.73 check
Write-Host ""
Write-Host "========== CLIP > 0.73 CHECK =========="
foreach ($r in $results) {
    if ($r.clip -eq "N/A") { continue }
    $clipVal = [double]$r.clip
    if ($clipVal -gt 0.73) {
        Write-Host "$($r.name) (alpha=$($r.alpha), hf=$($r.hf_wct)): clip=$clipVal lpips=$($r.lpips) -> CLIP > 0.73!"
    }
}
