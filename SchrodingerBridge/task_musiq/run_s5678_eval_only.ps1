# Eval-only script for S5-S8 (fixes path bug in run_s5678)
# Run after all training is complete
$ErrorActionPreference = "Continue"
$root = "G:\GitHub\Latent_Style\SchrodingerBridge"
$python = "python"
$dirs = @(
    @{name="musiq_s5_band_xattn";  config="configs\musiq_s5_band_xattn.json"},
    @{name="musiq_s6_patch_xattn"; config="configs\musiq_s6_patch_xattn.json"},
    @{name="musiq_s7_dwt_energy";  config="configs\musiq_s7_dwt_energy.json"},
    @{name="musiq_s8_combined";    config="configs\musiq_s8_combined.json"}
)

$results = @{}
$resultsPath = "$root\task_musiq\state\s5678_results.json"

foreach ($d in $dirs) {
    $name = $d.name
    $config = $d.config
    $expDir = "$root\exp\$name"
    $ckpt = "$expDir\epoch_0010.pt"

    Write-Output ""
    Write-Output "========== [$name] EVAL START =========="

    if (-not (Test-Path $ckpt)) {
        Write-Output "[$name] Checkpoint not found: $ckpt -- skipping"
        $results[$name] = @{status="no_checkpoint"}
        continue
    }

    # Eval
    Write-Output "[$name] Evaluating epoch_0010..."
    $evalStart = Get-Date
    & $python "$root\task_musiq\eval_only.py" --config "$root\$config" --checkpoint $ckpt 2>&1 | Select-Object -Last 3
    if ($LASTEXITCODE -ne 0) {
        Write-Output "[$name] EVAL FAILED (exit=$LASTEXITCODE)"
        $results[$name] = @{status="eval_failed"}
        $results | ConvertTo-Json -Depth 3 | Out-File -FilePath $resultsPath -Encoding utf8
        continue
    }
    $evalMin = [math]::Round(((Get-Date) - $evalStart).TotalMinutes, 1)

    # Extract CLIP-S and LPIPS
    $summaryPath = "$expDir\full_eval\epoch_0010\summary.json"
    $clipS = $null; $lpips = $null
    if (Test-Path $summaryPath) {
        $summary = Get-Content $summaryPath -Raw | ConvertFrom-Json
        $allpairs = $summary.analysis.all_pairs_overview
        $clipS = [math]::Round($allpairs.clip_style, 4)
        $lpips = [math]::Round($allpairs.content_lpips, 4)
    }

    # Compute MUSIQ
    $imgDir = "$expDir\full_eval\epoch_0010\images"
    $musiqOut = "$expDir\musiq_result.json"
    & $python "$root\scripts\_compute_musiq_batch.py" --methods "$name=$imgDir" --output $musiqOut 2>&1 | Select-Object -Last 3

    $musiq = $null
    if (Test-Path $musiqOut) {
        $musiqData = Get-Content $musiqOut -Raw | ConvertFrom-Json
        if ($musiqData.$name.musiq) {
            $musiq = [math]::Round($musiqData.$name.musiq, 4)
        }
    }

    $results[$name] = @{
        status="done"; clip_s=$clipS; lpips=$lpips; musiq=$musiq; eval_min=$evalMin
    }
    Write-Output "[$name] CLIP-S=$clipS  LPIPS=$lpips  MUSIQ=$musiq  (${evalMin}min)"

    $results | ConvertTo-Json -Depth 3 | Out-File -FilePath $resultsPath -Encoding utf8
}

Write-Output ""
Write-Output "========== S5-S8 SUMMARY =========="
Write-Output "Baseline:  MUSIQ=41.11  CLIP-S=0.7275  LPIPS=0.4347"
Write-Output "S4(best):  MUSIQ=42.01  CLIP-S=0.7095  LPIPS=0.5320"
Write-Output "Seedream:  MUSIQ=69.51  CLIP-S=0.7198  LPIPS=0.4767"
foreach ($name in $results.Keys) {
    $r = $results[$name]
    Write-Output "${name}: CLIP-S=$($r.clip_s) LPIPS=$($r.lpips) MUSIQ=$($r.musiq)"
}
