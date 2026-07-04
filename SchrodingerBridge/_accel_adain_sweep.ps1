# FC-SB T11 Acceleration + AdaIN Sweep (v2 - fixed save_dir issue)
# Strategy: All experiments save to same 630_local_t11_long30ep dir (save_dir override broken).
# After each experiment, copy summary.json to unique name before next experiment overwrites.
$env:PYTHONPATH = "I:\Github\Latent_Style\SchrodingerBridge\src"
$env:CUDA_VISIBLE_DEVICES = "0"
Set-Location I:\Github\Latent_Style\SchrodingerBridge

# Common paths (all experiments save here due to save_dir override bug)
$commonSaveDir = "I:\Github\Latent_Style\SchrodingerBridge\exp\630_local_t11_long30ep"
$commonSummary = "$commonSaveDir\full_eval\epoch_0001\summary.json"
$resultDir = "I:\Github\Latent_Style\SchrodingerBridge\exp\accel_sweep_results"
New-Item -ItemType Directory -Force -Path $resultDir | Out-Null

$experiments = @(
    @{ name="2step";        config="630_remote_t11_accel2_only.json" },
    @{ name="4step_adain06"; config="630_remote_t11_4step_adain06.json" },
    @{ name="4step_adain07"; config="630_remote_t11_4step_adain07.json" },
    @{ name="4step_adain08"; config="630_remote_t11_4step_adain08.json" },
    @{ name="8step_adain07"; config="630_remote_t11_8step_adain07.json" }
)

$results = @()
foreach ($exp in $experiments) {
    $name = $exp.name
    $config = $exp.config
    Write-Host "`n========== RUN: $name (config: $config) ==========" -ForegroundColor Cyan
    $logFile = "I:\Github\Latent_Style\SchrodingerBridge\exp\accel_sweep_${name}.log"
    # Delete old summary to detect if new one is created
    if (Test-Path $commonSummary) { Remove-Item $commonSummary -Force }
    try {
        python src\run.py --config configs\$config 2>&1 | Tee-Object -FilePath $logFile
        $exitCode = $LASTEXITCODE
        Write-Host "EXIT CODE: $exitCode" -ForegroundColor $(if($exitCode -eq 0){'Green'}else{'Yellow'})
    } catch {
        Write-Host "ERROR: $_" -ForegroundColor Red
    }
    # Copy summary to unique name
    if (Test-Path $commonSummary) {
        $uniqueSummary = "$resultDir\summary_${name}.json"
        Copy-Item $commonSummary $uniqueSummary -Force
        $d = Get-Content $uniqueSummary | ConvertFrom-Json
        $a = $d.analysis.all_pairs_overview
        $clip = $a.clip_style
        $lpips = $a.content_lpips
        Write-Host "RESULT $name : clip=$clip lpips=$lpips" -ForegroundColor Green
        $results += @{ name=$name; clip=$clip; lpips=$lpips }
    } else {
        Write-Host "WARNING: summary not found at $commonSummary" -ForegroundColor Yellow
        $results += @{ name=$name; clip="N/A"; lpips="N/A" }
    }
}

Write-Host "`n========== SWEEP SUMMARY ==========" -ForegroundColor Cyan
Write-Host ("{0,-20} {1,-10} {2,-10}" -f "Experiment", "CLIP-S", "LPIPS")
foreach ($r in $results) {
    Write-Host ("{0,-20} {1,-10} {2}" -f $r.name, $r.clip, $r.lpips)
}
# Also save summary CSV
$csvPath = "$resultDir\sweep_summary.csv"
"experiment,clip_style,content_lpips" | Out-File $csvPath
foreach ($r in $results) {
    "$($r.name),$($r.clip),$($r.lpips)" | Out-File $csvPath -Append
}
Write-Host "`nSaved CSV: $csvPath" -ForegroundColor Green
