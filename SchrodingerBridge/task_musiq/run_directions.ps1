# Batch runner: train + eval + MUSIQ for 4 MUSIQ architecture directions
# Each direction: ~3.5min train + ~2min eval + ~1min MUSIQ = ~7min total
# Total: ~28min for all 4 directions

$ErrorActionPreference = "Continue"
$ROOT = "G:\GitHub\Latent_Style\SchrodingerBridge"
$PYTHON = "python"

$directions = @(
    @{name="s1_sem_region";  config="configs\musiq_s1_sem_region.json"},
    @{name="s2_sem_patch";   config="configs\musiq_s2_sem_patch.json"},
    @{name="s3_sem_band";    config="configs\musiq_s3_sem_band.json"},
    @{name="s4_sem_xattn";   config="configs\musiq_s4_sem_xattn.json"}
)

$results = @{}

foreach ($d in $directions) {
    $name = $d.name
    $config = $d.config
    $expDir = Join-Path $ROOT "exp\musiq_$name"
    $imageDir = Join-Path $expDir "full_eval\epoch_0010\images"
    $summaryPath = Join-Path $expDir "full_eval\epoch_0010\summary.json"
    $musiqOut = Join-Path $expDir "musiq_result.json"

    Write-Host "`n========== [$name] START ==========" -ForegroundColor Cyan
    Write-Host "Config: $config"
    Write-Host "ExpDir: $expDir"

    # Step 1: Train (includes auto eval at end since full_eval_defer_until_training_end=true)
    Write-Host "[$name] Training + Eval..."
    $trainCmd = "cd '$ROOT'; $PYTHON run.py --config '$config'"
    Write-Host "  CMD: $trainCmd"
    $trainLog = Join-Path $expDir "train_stdout.log"
    New-Item -ItemType Directory -Force -Path $expDir | Out-Null

    $start = Get-Date
    Invoke-Expression $trainCmd 2>&1 | Tee-Object -FilePath $trainLog
    $elapsed = (Get-Date) - $start
    Write-Host "[$name] Train+Eval done in $($elapsed.TotalMinutes.ToString('F1'))min"

    # Step 2: Extract CLIP-S and LPIPS from summary.json
    $clipS = "N/A"
    $lpips = "N/A"
    if (Test-Path $summaryPath) {
        $summary = Get-Content $summaryPath -Raw | ConvertFrom-Json
        # Find aggregate clip_style and content_lpips
        $pool = $summary.pool
        if ($pool) {
            if ($pool.PSObject.Properties.Name -contains "clip_style") {
                $clipS = [math]::Round($pool.clip_style, 4)
            }
            if ($pool.PSObject.Properties.Name -contains "content_lpips") {
                $lpips = [math]::Round($pool.content_lpips, 4)
            }
        }
        # Fallback: search in results
        if ($clipS -eq "N/A" -and $summary.results) {
            $firstResult = $summary.results[0]
            if ($firstResult.PSObject.Properties.Name -contains "clip_style") {
                $clipS = [math]::Round($firstResult.clip_style, 4)
            }
        }
    }
    Write-Host "[$name] CLIP-S=$clipS  LPIPS=$lpips"

    # Step 3: Compute MUSIQ on generated images
    $musiq = "N/A"
    if (Test-Path $imageDir) {
        Write-Host "[$name] Computing MUSIQ on generated images..."
        $musiqCmd = "cd '$ROOT'; $PYTHON scripts\_compute_musiq_batch.py --methods `"$name=$imageDir`" --output $musiqOut"
        Write-Host "  CMD: $musiqCmd"
        Invoke-Expression $musiqCmd 2>&1 | Tee-Object -FilePath (Join-Path $expDir "musiq_stdout.log")
        if (Test-Path $musiqOut) {
            $musiqData = Get-Content $musiqOut -Raw | ConvertFrom-Json
            if ($musiqData.$name) {
                $musiq = [math]::Round($musiqData.$name.mean, 2)
            }
        }
    } else {
        Write-Host "[$name] WARNING: No generated images found at $imageDir" -ForegroundColor Yellow
    }
    Write-Host "[$name] MUSIQ=$musiq"

    # Step 4: Record results
    $results[$name] = @{
        clip_s = $clipS
        lpips = $lpips
        musiq = $musiq
        config = $config
    }

    Write-Host "========== [$name] DONE: CLIP-S=$clipS LPIPS=$lpips MUSIQ=$musiq ==========" -ForegroundColor Green
}

# Final summary
Write-Host "`n`n========== FINAL SUMMARY ==========" -ForegroundColor Magenta
$resultsJson = $results | ConvertTo-Json -Depth 3
Write-Host $resultsJson
$resultsPath = Join-Path $ROOT "task_musiq\state\directions_results.json"
$results | ConvertTo-Json -Depth 3 | Out-File -FilePath $resultsPath -Encoding UTF8
Write-Host "Results saved to: $resultsPath"
