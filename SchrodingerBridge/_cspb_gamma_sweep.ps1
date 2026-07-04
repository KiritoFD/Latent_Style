# CSPB (Content-Structure Preserving Blend) Gamma Sweep
# gamma=0.0: pure T11 (control, should match T11 baseline clip=0.7172 lpips=0.2726)
# gamma=0.1, 0.2, 0.3, 0.5, 0.7, 1.0: increasing content LL injection
#
# Theory: content image LL is the LPIPS ground truth. Injecting content LL should
# reduce LPIPS while keeping T11 HF (style signal) intact -> dual win.

$ErrorActionPreference = "Continue"

# Directories
$t11Dir = "I:\Github\Latent_Style\SchrodingerBridge\exp\630_local_t11_long30ep\full_eval\epoch_0001\images"
$contentDir = "I:\wikiart_distinct5_samam_512_classview\test"
$cspbRoot = "I:\Github\Latent_Style\SchrodingerBridge\exp\cspb"
$testDir = "I:\wikiart_distinct5_samam_512_classview\test"
$cacheDir = "I:\Github\Latent_Style\SchrodingerBridge\exp\eval_cache"
$hfCacheDir = "I:\Github\Latent_Style\SchrodingerBridge\exp\eval_cache\hf"
$evalScript = "I:\Github\Latent_Style\SchrodingerBridge\src\utils\run_evaluation.py"
$resultDir = "I:\Github\Latent_Style\SchrodingerBridge\exp\cspb_results"

# Gamma values to sweep
$gammas = @(0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0)

# Results array
$results = @()

Write-Host "========== CSPB GAMMA SWEEP =========="
Write-Host "T11 dir: $t11Dir"
Write-Host "Content dir: $contentDir"
Write-Host "Gamma values: $($gammas -join ', ')"
Write-Host ""

# Create result dir
if (-not (Test-Path $resultDir)) {
    New-Item -ItemType Directory -Path $resultDir -Force | Out-Null
}

# Verify directories exist
if (-not (Test-Path $t11Dir)) {
    Write-Host "ERROR: T11 images directory not found: $t11Dir"
    exit 1
}
if (-not (Test-Path $contentDir)) {
    Write-Host "ERROR: Content directory not found: $contentDir"
    exit 1
}

$t11Count = (Get-ChildItem $t11Dir -Filter *.png).Count
Write-Host "T11 images: $t11Count"
Write-Host ""

foreach ($gamma in $gammas) {
    $name = "cspb_g{0:D2}" -f [int]($gamma * 10)
    $outDir = "$cspbRoot\$name"
    $imagesDir = "$outDir\images"

    Write-Host "========== EXPERIMENT: $name (gamma=$gamma) =========="

    # Clean output directory
    if (Test-Path $outDir) {
        Remove-Item $outDir -Recurse -Force
    }

    # Run blend
    Write-Host "Blending images (gamma=$gamma)..."
    python "I:\Github\Latent_Style\SchrodingerBridge\content_structure_blend.py" `
        --t11_dir $t11Dir `
        --content_dir $contentDir `
        --output_dir $outDir `
        --gamma $gamma

    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Blend failed for gamma=$gamma"
        $results += @{ name=$name; gamma=$gamma; clip="N/A"; lpips="N/A" }
        continue
    }

    # Verify images were created
    $blendCount = (Get-ChildItem $imagesDir -Filter *.png -ErrorAction SilentlyContinue).Count
    Write-Host "Blended images: $blendCount"
    if ($blendCount -eq 0) {
        Write-Host "ERROR: No blended images created"
        $results += @{ name=$name; gamma=$gamma; clip="N/A"; lpips="N/A" }
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

        $results += @{ name=$name; gamma=$gamma; clip=$clip; lpips=$lpips }
    } else {
        Write-Host "ERROR: No summary.json found at $summaryPath"
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
        $results += @{ name=$name; gamma=$gamma; clip=$clip; lpips=$lpips }
    }
    Write-Host ""
}

# Print summary
Write-Host ""
Write-Host "========== CSPB GAMMA SWEEP SUMMARY =========="
Write-Host "{0,-25} {1,10} {2,12} {3,12}" -f "Experiment", "Gamma", "CLIP-S", "LPIPS"
foreach ($r in $results) {
    Write-Host "{0,-25} {1,10} {2,12} {3,12}" -f $r.name, $r.gamma, $r.clip, $r.lpips
}

# Save CSV
$csvPath = "$resultDir\cspb_summary.csv"
$results | Export-Csv $csvPath -NoTypeInformation
Write-Host ""
Write-Host "Saved CSV: $csvPath"

# Pareto analysis
Write-Host ""
Write-Host "========== PARETO ANALYSIS =========="
Write-Host "Baselines:"
Write-Host "  SaMam      : clip=0.7175 lpips=0.2423"
Write-Host "  T11 (8step): clip=0.7213 lpips=0.2868"
Write-Host "  T11 (eval) : clip=0.7172 lpips=0.2726"
Write-Host ""
Write-Host "Dual-win threshold: clip > 0.7213 AND lpips < 0.2423"
Write-Host ""
foreach ($r in $results) {
    if ($r.clip -eq "N/A" -or $r.lpips -eq "N/A") {
        Write-Host "$($r.name): N/A"
        continue
    }
    $clipVal = [double]$r.clip
    $lpipsVal = [double]$r.lpips
    $beatClip = $clipVal -gt 0.7213
    $beatLpips = $lpipsVal -lt 0.2423
    $beatClipT11 = $clipVal -gt 0.7172
    $beatLpipsT11 = $lpipsVal -lt 0.2726
    if ($beatClip -and $beatLpips) {
        Write-Host "$($r.name) (gamma=$($r.gamma)): clip=$clipVal lpips=$lpipsVal -> DOUBLE WIN vs both baselines!"
    } elseif ($beatClip -and $beatLpipsT11) {
        Write-Host "$($r.name) (gamma=$($r.gamma)): clip=$clipVal lpips=$lpipsVal -> beats T11 both, lpips>SaMam"
    } elseif ($beatClipT11 -and $beatLpips) {
        Write-Host "$($r.name) (gamma=$($r.gamma)): clip=$clipVal lpips=$lpipsVal -> beats SaMam lpips, clip<T11"
    } else {
        Write-Host "$($r.name) (gamma=$($r.gamma)): clip=$clipVal lpips=$lpipsVal"
    }
}
