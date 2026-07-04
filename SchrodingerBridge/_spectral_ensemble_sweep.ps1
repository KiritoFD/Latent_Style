# Spectral Ensemble Sweep: SaMam LL + T11 HF blend at multiple alpha values
# alpha=0.0: pure T11 (control)
# alpha=0.3, 0.5, 0.7, 0.9, 1.0: increasing SaMam LL contribution
# beta=0.0 for all (pure T11 HF)

$ErrorActionPreference = "Continue"

# Directories
$samamDir = "I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_v2\eval\samam\images"
$t11Dir = "I:\Github\Latent_Style\SchrodingerBridge\exp\630_local_t11_long30ep\full_eval\epoch_0001\images"
$ensembleRoot = "I:\Github\Latent_Style\SchrodingerBridge\exp\spectral_ensemble"
$testDir = "I:\wikiart_distinct5_samam_512_classview\test"
$cacheDir = "I:\Github\Latent_Style\SchrodingerBridge\exp\eval_cache"
$hfCacheDir = "I:\Github\Latent_Style\SchrodingerBridge\exp\eval_cache\hf"
$evalScript = "I:\Github\Latent_Style\SchrodingerBridge\src\utils\run_evaluation.py"
$resultDir = "I:\Github\Latent_Style\SchrodingerBridge\exp\accel_sweep_results"

# Alpha values to sweep
$alphas = @(0.0, 0.3, 0.5, 0.7, 0.9, 1.0)

# Results array
$results = @()

Write-Host "========== SPECTRAL ENSEMBLE SWEEP =========="
Write-Host "SaMam dir: $samamDir"
Write-Host "T11 dir: $t11Dir"
Write-Host "Alpha values: $($alphas -join ', ')"
Write-Host ""

# Create result dir
if (-not (Test-Path $resultDir)) {
    New-Item -ItemType Directory -Path $resultDir -Force | Out-Null
}

# Verify directories exist
if (-not (Test-Path $samamDir)) {
    Write-Host "ERROR: SaMam images directory not found: $samamDir"
    exit 1
}
if (-not (Test-Path $t11Dir)) {
    Write-Host "ERROR: T11 images directory not found: $t11Dir"
    Write-Host "Waiting for T11 eval to complete..."
    Start-Sleep -Seconds 30
    if (-not (Test-Path $t11Dir)) {
        Write-Host "ERROR: T11 images still not found after waiting"
        exit 1
    }
}

$samamCount = (Get-ChildItem $samamDir -Filter *.png).Count
$t11Count = (Get-ChildItem $t11Dir -Filter *.png).Count
Write-Host "SaMam images: $samamCount"
Write-Host "T11 images: $t11Count"
Write-Host ""

foreach ($alpha in $alphas) {
    $name = "ensemble_a{0:D2}" -f [int]($alpha * 10)
    $outDir = "$ensembleRoot\$name"
    $imagesDir = "$outDir\images"

    Write-Host "========== EXPERIMENT: $name (alpha=$alpha) =========="

    # Clean output directory
    if (Test-Path $outDir) {
        Remove-Item $outDir -Recurse -Force
    }

    # Run blend
    Write-Host "Blending images (alpha=$alpha, beta=0.0)..."
    python "I:\Github\Latent_Style\SchrodingerBridge\spectral_ensemble.py" `
        --samam_dir $samamDir `
        --t11_dir $t11Dir `
        --output_dir $outDir `
        --alpha $alpha `
        --beta 0.0

    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Blend failed for alpha=$alpha"
        $results += @{ name=$name; alpha=$alpha; clip="N/A"; lpips="N/A" }
        continue
    }

    # Verify images were created
    $blendCount = (Get-ChildItem $imagesDir -Filter *.png -ErrorAction SilentlyContinue).Count
    Write-Host "Blended images: $blendCount"
    if ($blendCount -eq 0) {
        Write-Host "ERROR: No blended images created"
        $results += @{ name=$name; alpha=$alpha; clip="N/A"; lpips="N/A" }
        continue
    }

    # Run evaluation (one-shot mode with --reuse_generated)
    Write-Host "Running evaluation..."
    $logFile = "I:\Github\Latent_Style\SchrodingerBridge\${name}_eval.log"
    if (Test-Path $logFile) { Remove-Item $logFile -Force }

    & python $evalScript $outDir `
        --test_dir $testDir `
        --cache_dir $cacheDir `
        --clip_hf_cache_dir $hfCacheDir `
        --batch_size 2 `
        --reuse_generated `
        --eval_only_lpips_clip_style `
        > $logFile 2>&1

    if ($LASTEXITCODE -ne 0) {
        Write-Host "WARNING: Eval exited with code $LASTEXITCODE. Checking for results anyway..."
    }

    # Read summary
    $summaryPath = "$outDir\summary.json"
    if (Test-Path $summaryPath) {
        $summary = Get-Content $summaryPath -Raw | ConvertFrom-Json
        $clip = $summary.analysis.all_pairs_overview.clip_style
        $lpips = $summary.analysis.all_pairs_overview.content_lpips
        Write-Host "RESULT ${name} : clip=$clip lpips=$lpips"

        # Copy summary to unique name
        $uniqueSummary = "$resultDir\summary_${name}.json"
        Copy-Item $summaryPath $uniqueSummary -Force

        $results += @{ name=$name; alpha=$alpha; clip=$clip; lpips=$lpips }
    } else {
        Write-Host "ERROR: No summary.json found at $summaryPath"
        # Try to find clip/lpips in log
        $logContent = Get-Content $logFile -Raw -ErrorAction SilentlyContinue
        if ($logContent -match "clip_style.*?([0-9.]+)") {
            $clip = $matches[1]
            Write-Host "  Found clip in log: $clip"
        } else {
            $clip = "N/A"
        }
        if ($logContent -match "lpips.*?([0-9.]+)") {
            $lpips = $matches[1]
            Write-Host "  Found lpips in log: $lpips"
        } else {
            $lpips = "N/A"
        }
        $results += @{ name=$name; alpha=$alpha; clip=$clip; lpips=$lpips }
    }
    Write-Host ""
}

# Print summary
Write-Host ""
Write-Host "========== SPECTRAL ENSEMBLE SWEEP SUMMARY =========="
Write-Host "{0,-25} {1,10} {2,12} {3,12}" -f "Experiment", "Alpha", "CLIP-S", "LPIPS"
foreach ($r in $results) {
    Write-Host "{0,-25} {1,10} {2,12} {3,12}" -f $r.name, $r.alpha, $r.clip, $r.lpips
}

# Save CSV
$csvPath = "$resultDir\spectral_ensemble_summary.csv"
$results | Export-Csv $csvPath -NoTypeInformation
Write-Host ""
Write-Host "Saved CSV: $csvPath"
