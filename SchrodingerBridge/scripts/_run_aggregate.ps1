# Run aggregation script
$REPO = "I:\Github\Latent_Style\SchrodingerBridge"
$PYTHON = "C:\Program Files\Python312\python.exe"

Set-Location $REPO
& $PYTHON scripts\aggregate_abl512.py --exp_root exp/abl512 --output docs/experiments/abl512_v3_results.csv --include_failed

Write-Host ""
Write-Host "=== CSV content (first 5 lines) ==="
if (Test-Path "docs\experiments\abl512_v3_results.csv") {
    Get-Content "docs\experiments\abl512_v3_results.csv" -Head 5
    Write-Host "..."
    Write-Host "Total lines: $((Get-Content 'docs\experiments\abl512_v3_results.csv').Count)"
}
