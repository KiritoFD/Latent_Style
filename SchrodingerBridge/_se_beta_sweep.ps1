# Reverse SE: T11 LL + SaMam HF (beta sweep at alpha=0.0)
# Tests if SaMam's HF can improve LPIPS while T11's LL maintains CLIP

$ErrorActionPreference = "Continue"

$samamDir = "I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_v2\eval\samam\images"
$t11Dir = "I:\Github\Latent_Style\SchrodingerBridge\exp\630_local_t11_long30ep\full_eval\epoch_0001\images"
$ensembleRoot = "I:\Github\Latent_Style\SchrodingerBridge\exp\spectral_ensemble"
$testDir = "I:\wikiart_distinct5_samam_512_classview\test"
$cacheDir = "I:\Github\Latent_Style\SchrodingerBridge\exp\eval_cache"
$hfCacheDir = "I:\Github\Latent_Style\SchrodingerBridge\exp\eval_cache\hf"
$evalScript = "I:\Github\Latent_Style\SchrodingerBridge\src\utils\run_evaluation.py"
$resultDir = "I:\Github\Latent_Style\SchrodingerBridge\exp\accel_sweep_results"

# beta sweep at alpha=0.0 (T11 LL + varying SaMam HF)
$betas = @(0.3, 0.5, 1.0)

Write-Host "========== REVERSE SE BETA SWEEP (alpha=0.0) =========="
Write-Host "T11 LL + varying SaMam HF"
Write-Host "Betas: $($betas -join ', ')"
Write-Host ""

foreach ($beta in $betas) {
    $name = "ensemble_rev_b{0:D2}" -f [int]($beta * 10)
    $outDir = "$ensembleRoot\$name"

    Write-Host "========== EXPERIMENT: $name (alpha=0.0, beta=$beta) =========="

    if (Test-Path $outDir) { Remove-Item $outDir -Recurse -Force }

    python "I:\Github\Latent_Style\SchrodingerBridge\spectral_ensemble.py" `
        --samam_dir $samamDir `
        --t11_dir $t11Dir `
        --output_dir $outDir `
        --alpha 0.0 `
        --beta $beta

    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Blend failed for beta=$beta"
        continue
    }

    $blendCount = (Get-ChildItem "$outDir\images" -Filter *.png -ErrorAction SilentlyContinue).Count
    Write-Host "Blended images: $blendCount"
    if ($blendCount -eq 0) { continue }

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

    $summaryPath = "$outDir\summary.json"
    if (Test-Path $summaryPath) {
        $summary = Get-Content $summaryPath -Raw | ConvertFrom-Json
        $clip = $summary.analysis.all_pairs_overview.clip_style
        $lpips = $summary.analysis.all_pairs_overview.content_lpips
        Write-Host "RESULT ${name} : clip=$clip lpips=$lpips"
    } else {
        Write-Host "ERROR: No summary.json"
    }
    Write-Host ""
}

Write-Host "========== DONE =========="
