# Phase 4J Batch Run: 4J.2 (WCT-Aligned) + 4J.4 (Progressive Alpha) + 4J.5 (Combined)
# Error handling: continue after individual experiment failures
# Memory constraints: batch_size=24, 5 epochs, patience=2, VRAM 9-11G
$ErrorActionPreference = "Continue"
Set-Location "g:\GitHub\Latent_Style\SchrodingerBridge"

$experiments = @(
    @{name="4J.2_wct_aligned"; config="configs/630_phase4j2_wct_aligned.json"},
    @{name="4J.4_progressive_alpha"; config="configs/630_phase4j4_progressive_alpha.json"},
    @{name="4J.5_combined"; config="configs/630_phase4j5_wct_aligned_progressive.json"}
)

$logDir = "exp\phase4j_batch_logs"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }

foreach ($exp in $experiments) {
    $name = $exp.name
    $config = $exp.config
    $logFile = Join-Path $logDir "$name.log"
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "START: $name ($config)" -ForegroundColor Green
    Write-Host "Log: $logFile" -ForegroundColor Yellow
    Write-Host "Start: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Gray
    Write-Host "========================================" -ForegroundColor Cyan

    try {
        python run.py --config $config *>&1 | Tee-Object -FilePath $logFile
        $exitCode = $LASTEXITCODE
        if ($exitCode -eq 0) {
            Write-Host "SUCCESS: $name completed (exit=$exitCode)" -ForegroundColor Green
        } else {
            Write-Host "FAIL: $name exited with code $exitCode, continuing to next" -ForegroundColor Red
        }
    } catch {
        Write-Host "EXCEPTION: $name - $_ , continuing to next" -ForegroundColor Red
    }

    Write-Host "End: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Gray
    Write-Host ""
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "BATCH COMPLETE: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
