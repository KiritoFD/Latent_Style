# S5-S8 batch train + eval + MUSIQ orchestrator
# Runs sequentially: S5, S6, S7, S8 (train 10ep, eval epoch_0010, compute MUSIQ)
# Results saved to task_musiq/state/s5678_results.json
$ErrorActionPreference = "Continue"
$root = "G:\GitHub\Latent_Style\SchrodingerBridge"
$python = "python"
$dirs = @(
    @{name="s5_band_xattn";   config="configs\musiq_s5_band_xattn.json"},
    @{name="s6_patch_xattn";  config="configs\musiq_s6_patch_xattn.json"},
    @{name="s7_dwt_energy";   config="configs\musiq_s7_dwt_energy.json"},
    @{name="s8_combined";     config="configs\musiq_s8_combined.json"}
)

$results = @{}
$resultsPath = "$root\task_musiq\state\s5678_results.json"

foreach ($d in $dirs) {
    $name = $d.name
    $config = $d.config
    $expDir = "$root\exp\$name"
    $ckpt = "$expDir\epoch_0010.pt"

    Write-Output ""
    Write-Output "========== [$name] START =========="
    Write-Output "[$name] Config: $config"

    # Step 1: Train (no auto-eval)
    Write-Output "[$name] Training (no auto-eval)..."
    $trainStart = Get-Date
    & $python "$root\src\run.py" --config "$root\$config" 2>&1 | Select-Object -Last 5
    if ($LASTEXITCODE -ne 0) {
        Write-Output "[$name] TRAIN FAILED (exit=$LASTEXITCODE), skipping eval"
        $results[$name] = @{status="train_failed"; musiq=$null; clip_s=$null; lpips=$null}
        $results | ConvertTo-Json -Depth 3 | Out-File -FilePath $resultsPath -Encoding utf8
        continue
    }
    $trainMin = [math]::Round(((Get-Date) - $trainStart).TotalMinutes, 1)
    Write-Output "[$name] Training done in $trainMin min"

    # Step 2: Eval epoch_0010
    Write-Output "[$name] Evaluating epoch_0010..."
    $evalStart = Get-Date
    & $python "$root\task_musiq\eval_only.py" --config "$root\$config" --checkpoint $ckpt 2>&1 | Select-Object -Last 3
    if ($LASTEXITCODE -ne 0) {
        Write-Output "[$name] EVAL FAILED (exit=$LASTEXITCODE), skipping MUSIQ"
        $results[$name] = @{status="eval_failed"; musiq=$null; clip_s=$null; lpips=$null}
        $results | ConvertTo-Json -Depth 3 | Out-File -FilePath $resultsPath -Encoding utf8
        continue
    }
    $evalMin = [math]::Round(((Get-Date) - $evalStart).TotalMinutes, 1)
    Write-Output "[$name] Eval done in $evalMin min"

    # Extract CLIP-S and LPIPS from summary
    $summaryPath = "$expDir\full_eval\epoch_0010\summary.json"
    $clipS = $null
    $lpips = $null
    if (Test-Path $summaryPath) {
        $summary = Get-Content $summaryPath -Raw | ConvertFrom-Json
        $allpairs = $summary.analysis.all_pairs_overview
        $clipS = [math]::Round($allpairs.clip_style, 4)
        $lpips = [math]::Round($allpairs.content_lpips, 4)
        Write-Output "[$name] CLIP-S=$clipS  LPIPS=$lpips"
    } else {
        Write-Output "[$name] summary.json not found, MUSIQ may fail"
    }

    # Step 3: Compute MUSIQ
    Write-Output "[$name] Computing MUSIQ..."
    $imgDir = "$expDir\full_eval\epoch_0010\images"
    $musiqOut = "$expDir\musiq_result.json"
    & $python "$root\scripts\_compute_musiq_batch.py" --methods "$name=$imgDir" --output $musiqOut 2>&1 | Select-Object -Last 3

    $musiq = $null
    if (Test-Path $musiqOut) {
        $musiqData = Get-Content $musiqOut -Raw | ConvertFrom-Json
        # FIX: use .musiq not .mean
        if ($musiqData.$name.musiq) {
            $musiq = [math]::Round($musiqData.$name.musiq, 4)
            Write-Output "[$name] MUSIQ=$musiq"
        } else {
            Write-Output "[$name] MUSIQ not found in $musiqOut"
        }
    }

    $results[$name] = @{
        status="done"
        clip_s=$clipS
        lpips=$lpips
        musiq=$musiq
        train_min=$trainMin
        eval_min=$evalMin
    }
    Write-Output "========== [$name] DONE: CLIP-S=$clipS LPIPS=$lpips MUSIQ=$musiq =========="

    # Save incremental
    $results | ConvertTo-Json -Depth 3 | Out-File -FilePath $resultsPath -Encoding utf8
}

Write-Output ""
Write-Output "========== S5-S8 SUMMARY =========="
foreach ($name in $results.Keys) {
    $r = $results[$name]
    Write-Output "${name}: CLIP-S=$($r.clip_s) LPIPS=$($r.lpips) MUSIQ=$($r.musiq)"
}
Write-Output "Results saved to: $resultsPath"
