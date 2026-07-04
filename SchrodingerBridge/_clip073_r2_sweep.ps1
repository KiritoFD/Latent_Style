# Round 2: HF WCT focused sweep from zsw_a03_8s
# Base: zsw_a03_8s (alpha=0.3, 8-step, adain=0.5) clip=0.7260 lpips=0.3419
# HF WCT CLIP efficiency (0.228) >> alpha efficiency (0.092)
# Target: clip>0.73 AND lpips<0.35

$ErrorActionPreference = "Continue"

$ckpt = "I:/Github/Latent_Style/SchrodingerBridge/exp/630_local_t11_stochastic_dwt_p08/epoch_0005.pt"
$testDir = "I:/wikiart_distinct5_samam_512_classview/test"
$cacheDir = "I:/Github/Latent_Style/SchrodingerBridge/exp/eval_cache"
$hfCacheDir = "I:/Github/Latent_Style/SchrodingerBridge/exp/eval_cache/hf"
$resultDir = "I:/Github/Latent_Style/SchrodingerBridge/exp/clip073_r2_results"
$expRoot = "I:/Github/Latent_Style/SchrodingerBridge/exp/clip073_r2_sweep"
$configsDir = "I:/Github/Latent_Style/SchrodingerBridge/configs/_clip073_r2_overrides"

$experiments = @()
$experiments += @{name="a03_hf_8s";     steps=8; alpha=0.30; hf=$true;  adain=0.5}
$experiments += @{name="a025_hf_8s";    steps=8; alpha=0.25; hf=$true;  adain=0.5}
$experiments += @{name="a02_hf_8s";     steps=8; alpha=0.20; hf=$true;  adain=0.5}
$experiments += @{name="a015_hf_8s";    steps=8; alpha=0.15; hf=$true;  adain=0.5}
$experiments += @{name="a03_nohf_8s";   steps=8; alpha=0.30; hf=$false; adain=0.5}

$results = @()

Write-Host "========== ROUND 2: HF WCT FOCUSED SWEEP =========="
Write-Host "Base: zsw_a03_8s clip=0.7260 lpips=0.3419"
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
    $hf = $exp.hf
    $adain = $exp.adain
    $outDir = "$expRoot/$name"

    Write-Host "========== EXPERIMENT: $name (steps=$steps, alpha=$alpha, hf=$hf, adain=$adain) =========="

    if (Test-Path $outDir) { Remove-Item $outDir -Recurse -Force }

    $modelSection = @{
        zero_step_wct_enabled = $true
        zero_step_wct_alpha = $alpha
        zero_step_wct_hf_enabled = $hf
        endpoint_adain_scale = $adain
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
            notes = "steps=$steps alpha=$alpha hf=$hf adain=$adain"
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
        $results += @{ name=$name; steps=$steps; alpha=$alpha; hf=$hf; adain=$adain; clip=$clip; lpips=$lpips }
    } else {
        Write-Host "ERROR: No summary.json found"
        $results += @{ name=$name; steps=$steps; alpha=$alpha; hf=$hf; adain=$adain; clip="N/A"; lpips="N/A" }
    }
    Write-Host ""
}

Write-Host ""
Write-Host "========== ROUND 2 SUMMARY =========="
Write-Host "Base: zsw_a03_8s clip=0.7260 lpips=0.3419 | Target: clip>0.73 AND lpips<0.35"
Write-Host "{0,-18} {1,6} {2,7} {3,5} {4,6} {5,10} {6,10} {7,14}" -f "Experiment", "Steps", "Alpha", "HF", "Adain", "CLIP-S", "LPIPS", "Status"
foreach ($r in $results) {
    $status = "?"
    if ($r.clip -ne "N/A" -and $r.lpips -ne "N/A") {
        $cv = [double]$r.clip
        $lv = [double]$r.lpips
        if ($cv -gt 0.73 -and $lv -lt 0.35) { $status = "*** DOUBLE WIN ***" }
        elseif ($cv -gt 0.73) { $status = "clip OK" }
        elseif ($lv -lt 0.35) { $status = "lpips OK" }
        else { $status = "neither" }
    }
    $hfStr = if ($r.hf) { "Y" } else { "N" }
    Write-Host "{0,-18} {1,6} {2,7} {3,5} {4,6} {5,10} {6,10} {7,14}" -f $r.name, $r.steps, $r.alpha, $hfStr, $r.adain, $r.clip, $r.lpips, $status
}

$jsonResults = $results | ConvertTo-Json -Depth 5
$jsonPath = "$resultDir/r2_results.json"
[System.IO.File]::WriteAllText($jsonPath, $jsonResults, [System.Text.UTF8Encoding]::new($false))
Write-Host ""
Write-Host "Saved JSON: $jsonPath"
