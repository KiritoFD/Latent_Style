$logFile = "C:\Users\Administrator\logs\refactor_verify_eval.log"
$summary = "I:\Github\Latent_Style\SchrodingerBridge\exp\refactor_verify\t1_asg_5ep\summary.json"

# Wait up to 5 minutes for summary
$timeout = 300
$elapsed = 0
while (-not (Test-Path $summary) -and $elapsed -lt $timeout) {
    Start-Sleep -Seconds 15
    $elapsed += 15
    $py = Get-Process python -ErrorAction SilentlyContinue
    if ($py) {
        Write-Output "[$elapsed s] Python running, waiting..."
    } else {
        Write-Output "[$elapsed s] Python finished"
        break
    }
}

if (Test-Path $summary) {
    Write-Output "=== SUMMARY FOUND ==="
    # Show last 10 lines of log
    Get-Content $logFile -Tail 10
} else {
    Write-Output "=== TIMEOUT: summary not found ==="
    Get-Content $logFile -Tail 20
}
