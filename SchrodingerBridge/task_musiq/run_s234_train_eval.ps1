# Orchestrator: Train S2-S4 (no auto-eval) + eval epoch_0010 + MUSIQ
# S1 is handled separately (already trained, eval_only.py running)
$ErrorActionPreference = "Continue"
$ROOT = "G:\GitHub\Latent_Style\SchrodingerBridge"
$PYTHON = "python"

$directions = @(
    @{name="s2_sem_patch";  config="configs\musiq_s2_sem_patch.json"},
    @{name="s3_sem_band";   config="configs\musiq_s3_sem_band.json"},
    @{name="s4_sem_xattn";  config="configs\musiq_s4_sem_xattn.json"}
)

$results = @()

foreach ($d in $directions) {
    $name = $d.name
    $config = $d.config
    $expDir = Join-Path $ROOT "exp\musiq_$name"
    $ckptPath = Join-Path $expDir "epoch_0010.pt"
    $summaryPath = Join-Path $expDir "full_eval\epoch_0010\summary.json"
    $imageDir = Join-Path $expDir "full_eval\epoch_0010\images"
    $musiqOut = Join-Path $expDir "musiq_result.json"

    Write-Host "`n========== [$name] START ==========" -ForegroundColor Cyan

    # Step 1: Train (deferred eval disabled, so training only)
    Write-Host "[$name] Training (no auto-eval)..."
    $trainStart = Get-Date
    $trainCmd = "cd '$ROOT'; $PYTHON run.py --config '$config'"
    Invoke-Expression $trainCmd 2>&1 | Tee-Object -FilePath (Join-Path $expDir "train_stdout.log")
    $trainElapsed = (Get-Date) - $trainStart
    Write-Host "[$name] Training done in $($trainElapsed.TotalMinutes.ToString('F1'))min"

    if (-not (Test-Path $ckptPath)) {
        Write-Host "[$name] ERROR: checkpoint not found: $ckptPath" -ForegroundColor Red
        continue
    }

    # Step 2: Eval epoch_0010 only (saves images)
    Write-Host "[$name] Evaluating epoch_0010..."
    $evalStart = Get-Date
    $evalCmd = "cd '$ROOT'; $PYTHON task_musiq\eval_only.py --config '$config' --checkpoint '$ckptPath'"
    Invoke-Expression $evalCmd 2>&1 | Tee-Object -FilePath (Join-Path $expDir "eval_stdout.log")
    $evalElapsed = (Get-Date) - $evalStart
    Write-Host "[$name] Eval done in $($evalElapsed.TotalMinutes.ToString('F1'))min"

    # Step 3: Extract CLIP-S and LPIPS from summary.json
    $clipS = "N/A"
    $lpips = "N/A"
    if (Test-Path $summaryPath) {
        $summary = Get-Content $summaryPath -Raw | ConvertFrom-Json
        $pool = $summary.analysis.style_transfer_ability
        if ($pool) {
            $clipS = [math]::Round($pool.clip_style, 4)
            $lpips = [math]::Round($pool.content_lpips, 4)
        }
    }
    Write-Host "[$name] CLIP-S=$clipS  LPIPS=$lpips"

    # Step 4: Compute MUSIQ on generated images
    $musiq = "N/A"
    if (Test-Path $imageDir) {
        Write-Host "[$name] Computing MUSIQ..."
        $musiqCmd = "cd '$ROOT'; $PYTHON scripts\_compute_musiq_batch.py --methods `"$name=$imageDir`" --output $musiqOut"
        Invoke-Expression $musiqCmd 2>&1 | Tee-Object -FilePath (Join-Path $expDir "musiq_stdout.log")
        if (Test-Path $musiqOut) {
            $musiqData = Get-Content $musiqOut -Raw | ConvertFrom-Json
            if ($musiqData.$name) {
                $musiq = [math]::Round($musiqData.$name.mean, 2)
            }
        }
    } else {
        Write-Host "[$name] WARNING: No images at $imageDir" -ForegroundColor Yellow
    }
    Write-Host "[$name] MUSIQ=$musiq"

    $results += @{
        name = $name
        clip_s = $clipS
        lpips = $lpips
        musiq = $musiq
    }

    Write-Host "========== [$name] DONE: CLIP-S=$clipS LPIPS=$lpips MUSIQ=$musiq ==========" -ForegroundColor Green
}

# Final summary
Write-Host "`n`n========== S2-S4 SUMMARY ==========" -ForegroundColor Magenta
$results | ForEach-Object {
    Write-Host ("{0}: CLIP-S={1} LPIPS={2} MUSIQ={3}" -f $_.name, $_.clip_s, $_.lpips, $_.musiq)
}
$resultsPath = Join-Path $ROOT "task_musiq\state\s234_results.json"
$results | ConvertTo-Json -Depth 3 | Out-File -FilePath $resultsPath -Encoding UTF8
Write-Host "Results saved to: $resultsPath"
