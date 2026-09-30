$logFile = "C:\Users\Administrator\logs\refactor_verify_eval.log"
if (Test-Path $logFile) {
    $content = Get-Content $logFile -Tail 30
    $content | Out-String | Write-Output
} else {
    Write-Output "Log file not found: $logFile"
}
# Also check if summary.json exists
$summary = "I:\Github\Latent_Style\SchrodingerBridge\exp\refactor_verify\t1_asg_5ep\summary.json"
if (Test-Path $summary) {
    Write-Output "=== SUMMARY EXISTS ==="
} else {
    Write-Output "=== SUMMARY NOT YET ==="
}
# Check python process
$py = Get-Process python -ErrorAction SilentlyContinue
if ($py) {
    Write-Output "Python running: PID=$($py.Id) Mem=$([math]::Round($py.WorkingSet64/1MB))MB"
} else {
    Write-Output "Python NOT running"
}
