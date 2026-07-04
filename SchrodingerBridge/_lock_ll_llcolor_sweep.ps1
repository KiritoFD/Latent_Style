# FC-SB T11 Lock-LL + LL Color WCT Sweep — THE BREAKTHROUGH EXPERIMENT
# Theory: lock_ll=true preserves LL structure (no ODE drift) → great LPIPS
#         adain_scale_ll>0 injects color statistics via WCT at endpoint → recovers CLIP
#         WCT only changes mean+covariance (color), not spatial structure (edges) → LPIPS neutral
# This decouples content preservation (ODE lock) from color style transfer (endpoint WCT).
$env:PYTHONPATH = "I:\Github\Latent_Style\SchrodingerBridge\src"
$env:CUDA_VISIBLE_DEVICES = "0"
Set-Location I:\Github\Latent_Style\SchrodingerBridge

$commonSummary = "I:\Github\Latent_Style\SchrodingerBridge\exp\630_local_t11_long30ep\full_eval\epoch_0001\summary.json"
$resultDir = "I:\Github\Latent_Style\SchrodingerBridge\exp\accel_sweep_results"
New-Item -ItemType Directory -Force -Path $resultDir | Out-Null

$experiments = @(
    @{ name="lock_ll_llcolor03"; config="630_remote_t11_lock_ll_llcolor03.json" },
    @{ name="lock_ll_llcolor05"; config="630_remote_t11_lock_ll_llcolor05.json" },
    @{ name="lock_ll_llcolor08"; config="630_remote_t11_lock_ll_llcolor08.json" },
    @{ name="lock_ll_llcolor10"; config="630_remote_t11_lock_ll_llcolor10.json" }
)

$results = @()
foreach ($exp in $experiments) {
    $name = $exp.name
    $config = $exp.config
    Write-Host "`n========== RUN: $name (config: $config) ==========" -ForegroundColor Cyan
    $logFile = "I:\Github\Latent_Style\SchrodingerBridge\exp\llcolor_sweep_${name}.log"
    if (Test-Path $commonSummary) { Remove-Item $commonSummary -Force }
    if (Test-Path $logFile) { Remove-Item $logFile -Force }
    try {
        python src\run.py --config configs\$config 2>&1 | Tee-Object -FilePath $logFile
        $exitCode = $LASTEXITCODE
        Write-Host "EXIT CODE: $exitCode" -ForegroundColor $(if($exitCode -eq 0){'Green'}else{'Yellow'})
    } catch {
        Write-Host "ERROR: $_" -ForegroundColor Red
    }
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
        Write-Host "WARNING: summary not found" -ForegroundColor Yellow
        $results += @{ name=$name; clip="N/A"; lpips="N/A" }
    }
}

Write-Host "`n========== LL-COLOR SWEEP SUMMARY ==========" -ForegroundColor Cyan
Write-Host ("{0,-25} {1,-10} {2,-10}" -f "Experiment", "CLIP-S", "LPIPS")
foreach ($r in $results) {
    Write-Host ("{0,-25} {1,-10} {2}" -f $r.name, $r.clip, $r.lpips)
}
$csvPath = "$resultDir\llcolor_sweep_summary.csv"
"experiment,clip_style,content_lpips" | Out-File $csvPath
foreach ($r in $results) {
    "$($r.name),$($r.clip),$($r.lpips)" | Out-File $csvPath -Append
}
Write-Host "`nSaved CSV: $csvPath" -ForegroundColor Green
